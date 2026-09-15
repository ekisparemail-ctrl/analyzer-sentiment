import json
import logging

from confluent_kafka import Consumer, KafkaException
from pydantic import ValidationError

from schemas import AnalysisRequest

logger = logging.getLogger(__name__)


class CommitError(Exception):
    pass


class KafkaRequestConsumer:
    def __init__(self, bootstrap_servers: str, topic: str, group_id: str) -> None:
        self._consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )
        self._consumer.subscribe([topic])

    def poll_request(self, timeout: float) -> tuple[AnalysisRequest, object] | None:
        msg = self._consumer.poll(timeout)
        if msg is None:
            return None
        if msg.error():
            logger.warning("Kafka poll returned a message-level error: %s", msg.error())
            return None
        try:
            msg_value = msg.value()
            if msg_value is None:
                self.commit(msg)
                return None
            data = json.loads(msg_value.decode("utf-8"))
            request = AnalysisRequest(**data)
        except (json.JSONDecodeError, ValidationError, UnicodeDecodeError, TypeError) as e:
            logger.warning("Skipping malformed message: %s", e)
            self.commit(msg)
            return None
        return request, msg

    def commit(self, msg: object) -> None:
        try:
            self._consumer.commit(message=msg, asynchronous=False)  # type: ignore[call-overload]
        except KafkaException as e:
            logger.error("Failed to commit Kafka offset: %s", e)
            raise CommitError(str(e)) from e
