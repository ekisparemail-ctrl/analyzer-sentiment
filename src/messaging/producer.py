from confluent_kafka import Producer

from schemas import AnalysisResult


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

        self._producer.produce(
            self._topic,
            key=result.id.encode("utf-8"),
            value=result.model_dump_json().encode("utf-8"),
            callback=_on_delivery,
        )
        self._producer.flush(10)

        if errors:
            raise PublishError(str(errors[0]))
