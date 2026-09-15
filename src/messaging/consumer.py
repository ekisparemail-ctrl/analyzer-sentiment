import json
import logging

from confluent_kafka import Consumer, KafkaException
from pydantic import ValidationError

from messaging.scrapper_dto import (
    NormalizedComment,
    NormalizedPost,
    request_from_comment,
    request_from_post,
)
from schemas import AnalysisRequest

logger = logging.getLogger(__name__)


class CommitError(Exception):
    pass


class KafkaRequestConsumer:
    def __init__(
        self,
        bootstrap_servers: str,
        post_topic: str,
        comment_topic: str,
        group_id: str,
    ) -> None:
        self._post_topic = post_topic
        self._comment_topic = comment_topic
        self._consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )
        self._consumer.subscribe([post_topic, comment_topic])

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
            topic = msg.topic()
            if topic == self._post_topic:
                request = request_from_post(NormalizedPost(**data))
            elif topic == self._comment_topic:
                request = request_from_comment(NormalizedComment(**data))
            else:
                logger.warning("Skipping message from unexpected topic: %s", topic)
                self.commit(msg)
                return None
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

    def close(self) -> None:
        self._consumer.close()
