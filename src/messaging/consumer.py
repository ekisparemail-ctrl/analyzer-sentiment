import json

from confluent_kafka import Consumer
from pydantic import ValidationError

from schemas import AnalysisRequest


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
            return None
        try:
            msg_value = msg.value()
            if msg_value is None:
                self._consumer.commit(msg)  # type: ignore[call-overload]
                return None
            data = json.loads(msg_value.decode("utf-8"))
            request = AnalysisRequest(**data)
        except (json.JSONDecodeError, ValidationError, UnicodeDecodeError, TypeError):
            self._consumer.commit(msg)  # type: ignore[call-overload]
            return None
        return request, msg

    def commit(self, msg: object) -> None:
        self._consumer.commit(msg)  # type: ignore[call-overload]
