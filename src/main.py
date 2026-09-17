import logging
import time
from datetime import datetime
from functools import partial
from pathlib import Path

from ai.llm_client import LLMConfig, analyze_sentiment, summarize_video
from ai.vlm_client import VLMConfig
from config import Settings
from messaging.consumer import KafkaRequestConsumer
from messaging.producer import KafkaResultProducer
from pipeline.analyze import AnalyzeDependencies, analyze
from schemas import AnalysisResult, Status
from video.analyzer import analyze_video, load_models
from video.downloader import download_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_dependencies(settings: Settings) -> AnalyzeDependencies:
    """
    Loads the local faster-whisper model once (CPU-only, this machine has no
    GPU) -- not exercised in unit tests (real model weights, slow to
    download/load; see plan Global Constraints). The VLM is a remote HTTP
    call (ai/vlm_client.py) again, same LM Studio instance as the LLM but a
    separate config entry (spec revision-1 section 2).
    """
    vlm_config = VLMConfig(
        base_url=settings.vlm_base_url,
        model=settings.vlm_model,
        timeout_seconds=settings.http_timeout_seconds,
        retry_attempts=settings.retry_attempts,
        retry_backoff_seconds=settings.retry_backoff_seconds,
    )
    analyzer_models = load_models(
        vlm_config,
        settings.whisper_model_size,
        settings.whisper_vad_filter,
        vlm_batch_enabled=settings.vlm_batch_enabled,
        vlm_batch_size=settings.vlm_batch_size,
        keyframe_diff_threshold=settings.keyframe_diff_threshold,
        ffmpeg_hwaccel_cuda=settings.ffmpeg_hwaccel_cuda,
    )
    llm_config = LLMConfig(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout_seconds=settings.http_timeout_seconds,
        retry_attempts=settings.retry_attempts,
        retry_backoff_seconds=settings.retry_backoff_seconds,
    )
    return AnalyzeDependencies(
        download_video=download_video,
        analyze_video=partial(analyze_video, analyzer_models),
        summarize_video=partial(summarize_video, llm_config),
        analyze_sentiment=partial(analyze_sentiment, llm_config),
        max_frames=settings.max_frames,
        max_tokens=settings.max_tokens,
    )


def _log_result(request_id: str, status: Status, error: str | None) -> None:
    if status == Status.OK:
        logger.info("Processed request %s: status=%s", request_id, status)
    elif status == Status.PARTIAL:
        logger.warning("Processed request %s: status=%s error=%s", request_id, status, error)
    else:
        logger.error("Processed request %s: status=%s error=%s", request_id, status, error)


def _dump_result_for_dev(result: AnalysisResult, output_dir: str) -> None:
    """
    Dev-only, best-effort local copy of a processed item's final
    AnalysisResult (spec revision-1 section 6) -- additional to, never
    instead of, the Kafka publish. Any failure here (disk full,
    permissions, etc.) is logged and swallowed: it must never fail the
    item's processing or block the publish that follows.
    """
    try:
        dump_dir = Path(output_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        dump_path = dump_dir / f"output-{result.id}-{timestamp}.json"
        dump_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to write dev dump for request %s: %s", result.id, e)


def run_once(
    consumer: KafkaRequestConsumer,
    producer: KafkaResultProducer,
    deps: AnalyzeDependencies,
    poll_timeout: float = 1.0,
    dev_dump_output: bool = False,
    output_dir: str = "output",
) -> bool:
    """
    One Kafka message can now yield several requests (the post itself plus
    one per nested comment, see messaging/scrapper_dto.py) -- all of them
    are analyzed and published before the single underlying offset is
    committed, so a failure partway through leaves the offset uncommitted
    and the whole message (including already-published items) is retried
    next time, rather than silently losing the rest of the batch.
    """
    polled = consumer.poll_requests(poll_timeout)
    if polled is None:
        return False
    requests, msg = polled
    current_id = None
    try:
        for request in requests:
            current_id = request.id
            result = analyze(request, deps)
            _log_result(request.id, result.status, result.error)
            if dev_dump_output:
                _dump_result_for_dev(result, output_dir)
            producer.publish(result)
        consumer.commit(msg)
    except Exception as e:
        logger.error("Failed to fully process request %s: %s", current_id, e)
        raise
    return True


HEARTBEAT_INTERVAL_SECONDS = 30.0


def run_forever(
    consumer: KafkaRequestConsumer,
    producer: KafkaResultProducer,
    deps: AnalyzeDependencies,
    dev_dump_output: bool = False,
    output_dir: str = "output",
) -> None:
    """
    Runs run_once() forever. A KeyboardInterrupt (Ctrl-C) stops the loop
    cleanly instead of dumping a raw traceback -- any other exception is
    logged and the loop continues (see run_once's own error handling).

    Logs a heartbeat during idle polling (no new message) so a long silence
    can be told apart from a hang -- without this, a real run once looked
    completely silent for the whole time it was up, with no way to tell
    whether it was actually still polling.
    """
    last_heartbeat = time.monotonic()
    try:
        while True:
            try:
                processed = run_once(
                    consumer,
                    producer,
                    deps,
                    dev_dump_output=dev_dump_output,
                    output_dir=output_dir,
                )
                now = time.monotonic()
                if processed:
                    last_heartbeat = now
                elif now - last_heartbeat >= HEARTBEAT_INTERVAL_SECONDS:
                    logger.info(
                        "Still polling, no new messages in the last ~%.0fs.", now - last_heartbeat
                    )
                    last_heartbeat = now
            except Exception as e:
                logger.error(
                    "Unhandled exception while processing message; offset not committed, "
                    "continuing to next message: %s",
                    e,
                )
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down gracefully.")
    finally:
        consumer.close()


def _log_startup_banner(settings: Settings) -> None:
    logger.info(
        "Analytics Backend starting -- kafka_bootstrap_servers=%s scrapper_topic=%s "
        "result_topic=%s",
        settings.kafka_bootstrap_servers,
        settings.kafka_scrapper_topic,
        settings.kafka_result_topic,
    )
    logger.info(
        "Models in use -- vlm_base_url=%s vlm_model=%s whisper_model_size=%s "
        "whisper_vad_filter=%s llm_model=%s llm_base_url=%s max_frames=%s max_tokens=%s",
        settings.vlm_base_url,
        settings.vlm_model,
        settings.whisper_model_size,
        settings.whisper_vad_filter,
        settings.llm_model,
        settings.llm_base_url,
        settings.max_frames,
        settings.max_tokens,
    )


def main() -> None:
    settings = Settings()  # type: ignore[call-arg]
    _log_startup_banner(settings)
    deps = build_dependencies(settings)
    logger.info("Models ready.")
    consumer = KafkaRequestConsumer(
        settings.kafka_bootstrap_servers,
        settings.kafka_scrapper_topic,
        settings.kafka_consumer_group,
    )
    producer = KafkaResultProducer(
        settings.kafka_bootstrap_servers, settings.kafka_result_topic
    )
    logger.info(
        "Kafka consumer/producer ready (consumer_group=%s); entering poll loop.",
        settings.kafka_consumer_group,
    )
    run_forever(consumer, producer, deps, dev_dump_output=settings.dev_dump_output)


if __name__ == "__main__":
    main()
