import logging
from functools import partial

from ai.llm_client import LLMConfig, analyze_sentiment, summarize_video
from ai.vlm_client import VLMConfig
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
    Loads the local faster-whisper model once. Requires no special hardware
    (CPU-only) but is not exercised in unit tests (real model weights, slow to
    load; see plan Global Constraints).
    """
    vlm_config = VLMConfig(
        base_url=settings.vlm_base_url or "",
        model=settings.vlm_model or "",
        timeout_seconds=settings.http_timeout_seconds,
        retry_attempts=settings.retry_attempts,
        retry_backoff_seconds=settings.retry_backoff_seconds,
    )
    analyzer_models = load_models(vlm_config, settings.whisper_model_size)
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


def main() -> None:
    settings = Settings()  # type: ignore[call-arg]
    deps = build_dependencies(settings)
    consumer = KafkaRequestConsumer(
        settings.kafka_bootstrap_servers,
        settings.kafka_post_topic,
        settings.kafka_comment_topic,
        settings.kafka_consumer_group,
    )
    producer = KafkaResultProducer(
        settings.kafka_bootstrap_servers, settings.kafka_result_topic
    )
    while True:
        try:
            run_once(consumer, producer, deps)
        except Exception as e:
            logger.error(
                "Unhandled exception while processing message; offset not committed, "
                "continuing to next message: %s",
                e,
            )


if __name__ == "__main__":
    main()
