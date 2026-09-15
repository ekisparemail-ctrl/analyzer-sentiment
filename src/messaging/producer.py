import logging

from confluent_kafka import KafkaException, Producer

from schemas import AnalysisResult

logger = logging.getLogger(__name__)


class PublishError(Exception):
    pass


class KafkaResultProducer:
    def __init__(self, bootstrap_servers: str, topic: str) -> None:
        self._topic = topic
        self._producer = Producer({"bootstrap.servers": bootstrap_servers})

    def publish(self, result: AnalysisResult) -> None:
        errors: list[Exception] = []

        def _on_delivery(err: object, _msg: object) -> None:
            if err is not None:
                errors.append(Exception(str(err)))

        try:
            self._producer.produce(
                self._topic,
                key=result.id.encode("utf-8"),
                value=result.model_dump_json().encode("utf-8"),
                callback=_on_delivery,
            )
        except (BufferError, KafkaException) as e:
            logger.error("Failed to enqueue message %s for publish: %s", result.id, e)
            raise PublishError(str(e)) from e
        pending = self._producer.flush(10)

        if pending > 0:
            logger.error(
                "Failed to publish result %s: %s message(s) not delivered before flush timeout",
                result.id,
                pending,
            )
            raise PublishError(f"{pending} message(s) not delivered before flush timeout")
        if errors:
            logger.error("Failed to publish result %s: %s", result.id, errors[0])
            raise PublishError(str(errors[0]))
