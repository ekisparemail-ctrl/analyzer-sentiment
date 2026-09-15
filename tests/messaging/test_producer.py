from unittest.mock import MagicMock

import pytest

from messaging.producer import KafkaResultProducer, PublishError
from schemas import AnalysisResult, Status


def _result() -> AnalysisResult:
    return AnalysisResult(id="abc123", status=Status.OK, video_summary="a summary")


def test_publish_produces_message_and_flushes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_producer_instance = MagicMock()

    def fake_produce(topic, key, value, callback):
        callback(None, MagicMock())

    fake_producer_instance.produce.side_effect = fake_produce
    fake_producer_instance.flush.return_value = 0
    fake_producer_class = MagicMock(return_value=fake_producer_instance)
    monkeypatch.setattr("messaging.producer.Producer", fake_producer_class)

    producer = KafkaResultProducer("localhost:9092", "analytics.results")
    producer.publish(_result())

    fake_producer_class.assert_called_once_with({"bootstrap.servers": "localhost:9092"})
    args, kwargs = fake_producer_instance.produce.call_args
    assert args[0] == "analytics.results" or kwargs.get("topic") == "analytics.results"
    assert kwargs["key"] == b"abc123"
    assert b"abc123" in kwargs["value"]
    fake_producer_instance.flush.assert_called_once()


def test_publish_raises_publish_error_on_delivery_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_producer_instance = MagicMock()

    def fake_produce(topic, key, value, callback):
        callback(Exception("broker unavailable"), None)

    fake_producer_instance.produce.side_effect = fake_produce
    fake_producer_instance.flush.return_value = 0
    fake_producer_class = MagicMock(return_value=fake_producer_instance)
    monkeypatch.setattr("messaging.producer.Producer", fake_producer_class)

    producer = KafkaResultProducer("localhost:9092", "analytics.results")

    with pytest.raises(PublishError, match="broker unavailable"):
        producer.publish(_result())


def test_publish_raises_publish_error_when_flush_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_producer_instance = MagicMock()

    def fake_produce(topic, key, value, callback):
        callback(None, MagicMock())

    fake_producer_instance.produce.side_effect = fake_produce
    fake_producer_instance.flush.return_value = 1
    fake_producer_class = MagicMock(return_value=fake_producer_instance)
    monkeypatch.setattr("messaging.producer.Producer", fake_producer_class)

    producer = KafkaResultProducer("localhost:9092", "analytics.results")

    with pytest.raises(PublishError, match="not delivered"):
        producer.publish(_result())
