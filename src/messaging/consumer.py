import json
import logging

from confluent_kafka import Consumer, KafkaException
from pydantic import ValidationError

from messaging.scrapper_dto import NormalizedData, requests_from_normalized_data
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

    def poll_requests(self, timeout: float) -> tuple[list[AnalysisRequest], object] | None:
        """
        Returns every analyzable item from one Kafka message -- the post
        itself plus one item per nested comment (see
        messaging/scrapper_dto.py) -- and the raw message, so the caller can
        process all of them and commit the single underlying offset once,
        after all have been published.
        """
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
            requests = requests_from_normalized_data(NormalizedData(**data))
        except (json.JSONDecodeError, ValidationError, UnicodeDecodeError, TypeError) as e:
            # partition/offset logged so a skipped-and-committed message can
            # be found and, if the underlying bug gets fixed later, targeted
            # for a manual consumer-group offset reset to reprocess it --
            # otherwise it's silently gone (offset already advanced past it).
            logger.warning(
                "Skipping malformed message (partition=%s offset=%s): %s",
                msg.partition(),
                msg.offset(),
                e,
            )
            self.commit(msg)
            return None
        # TEMPORARY (dev-only, remove once Kafka integration is verified in
        # staging): confirms messages are actually being consumed/parsed.
        # Deliberately does not log request.text/metadata content -- those
        # are user-generated post/comment data (Acme security standard:
        # never log PII / user data bodies), so only shape/size is logged.
        for request in requests:
            logger.info(
                "Consumed request id=%s partition=%s offset=%s platform=%s type=%s "
                "has_video=%s text_len=%s",
                request.id,
                msg.partition(),
                msg.offset(),
                request.platform,
                request.metadata.get("type"),
                request.video_url is not None,
                len(request.text) if request.text else 0,
            )
        return requests, msg

    def commit(self, msg: object) -> None:
        try:
            self._consumer.commit(message=msg, asynchronous=False)  # type: ignore[call-overload]
        except KafkaException as e:
            logger.error("Failed to commit Kafka offset: %s", e)
            raise CommitError(str(e)) from e

    def close(self) -> None:
        self._consumer.close()
