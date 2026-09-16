import logging
import time
from functools import partial

from ai.llm_client import LLMConfig, analyze_sentiment, summarize_video
from config import Settings
from messaging.consumer import KafkaRequestConsumer
from messaging.producer import KafkaResultProducer
from pipeline.analyze import AnalyzeDependencies, analyze
from schemas import Status
from video.analyzer import analyze_video, load_models
from video.downloader import download_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_dependencies(settings: Settings) -> AnalyzeDependencies:
    """
    Loads the local faster-whisper and VLM models once. Both run in-process,
    CPU-only, on this same machine -- neither is exercised in unit tests
    (real model weights, slow to download/load; see plan Global Constraints).
    """
    analyzer_models = load_models(
        settings.vlm_model_id, settings.whisper_model_size, settings.whisper_vad_filter
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


def run_once(
    consumer: KafkaRequestConsumer,
    producer: KafkaResultProducer,
    deps: AnalyzeDependencies,
    poll_timeout: float = 1.0,
) -> bool:
    polled = consumer.poll_request(poll_timeout)
    if polled is None:
        return False
    request, msg = polled
    try:
        result = analyze(request, deps)
        _log_result(request.id, result.status, result.error)
        producer.publish(result)
        consumer.commit(msg)
    except Exception as e:
        logger.error("Failed to fully process request %s: %s", request.id, e)
        raise
    return True


HEARTBEAT_INTERVAL_SECONDS = 30.0


def run_forever(
    consumer: KafkaRequestConsumer,
    producer: KafkaResultProducer,
    deps: AnalyzeDependencies,
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
                processed = run_once(consumer, producer, deps)
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
        "Models in use -- vlm_model_id=%s whisper_model_size=%s whisper_vad_filter=%s "
        "llm_model=%s llm_base_url=%s max_frames=%s max_tokens=%s",
        settings.vlm_model_id,
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
    run_forever(consumer, producer, deps)


if __name__ == "__main__":
    main()
