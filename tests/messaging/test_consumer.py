import json
from unittest.mock import MagicMock

import pytest
from confluent_kafka import KafkaException

from messaging.consumer import CommitError, KafkaRequestConsumer

TOPIC = "scrapper-to-analysis"


def _make_consumer(monkeypatch: pytest.MonkeyPatch) -> tuple[KafkaRequestConsumer, MagicMock]:
    fake_consumer_instance = MagicMock()
    fake_consumer_class = MagicMock(return_value=fake_consumer_instance)
    monkeypatch.setattr("messaging.consumer.Consumer", fake_consumer_class)

    consumer = KafkaRequestConsumer("localhost:9092", TOPIC, "analytics-backend")
    return consumer, fake_consumer_instance


def test_subscribes_to_the_scrapper_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    _, fake_consumer_instance = _make_consumer(monkeypatch)

    fake_consumer_instance.subscribe.assert_called_once_with([TOPIC])


def test_poll_request_returns_none_when_nothing_polled(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, fake_consumer_instance = _make_consumer(monkeypatch)
    fake_consumer_instance.poll.return_value = None

    result = consumer.poll_request(1.0)

    assert result is None


def test_poll_request_parses_valid_message(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, fake_consumer_instance = _make_consumer(monkeypatch)

    fake_msg = MagicMock()
    fake_msg.error.return_value = None
    fake_msg.value.return_value = json.dumps(
        {
            "id": "item-1",
            "platform": "twitter",
            "type": "POST",
            "message": "Judul post",
            "videoUrl": "https://cdn.example.com/v.mp4",
        }
    ).encode("utf-8")
    fake_consumer_instance.poll.return_value = fake_msg

    result = consumer.poll_request(1.0)

    assert result is not None
    request, msg = result
    assert request.id == "item-1"
    assert request.text == "Judul post"
    assert request.video_url == "https://cdn.example.com/v.mp4"
    assert msg is fake_msg
    fake_consumer_instance.commit.assert_not_called()


def test_poll_request_skips_and_commits_malformed_message(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, fake_consumer_instance = _make_consumer(monkeypatch)

    fake_msg = MagicMock()
    fake_msg.error.return_value = None
    fake_msg.value.return_value = b"not json"
    fake_consumer_instance.poll.return_value = fake_msg

    result = consumer.poll_request(1.0)

    assert result is None
    fake_consumer_instance.commit.assert_called_once_with(message=fake_msg, asynchronous=False)


def test_poll_request_skips_and_commits_message_missing_required_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, fake_consumer_instance = _make_consumer(monkeypatch)

    fake_msg = MagicMock()
    fake_msg.error.return_value = None
    fake_msg.value.return_value = json.dumps({"message": "no id field"}).encode("utf-8")
    fake_consumer_instance.poll.return_value = fake_msg

    result = consumer.poll_request(1.0)

    assert result is None
    fake_consumer_instance.commit.assert_called_once_with(message=fake_msg, asynchronous=False)


def test_poll_request_returns_none_on_kafka_level_error(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, fake_consumer_instance = _make_consumer(monkeypatch)

    fake_msg = MagicMock()
    fake_msg.error.return_value = "partition EOF"
    fake_consumer_instance.poll.return_value = fake_msg

    result = consumer.poll_request(1.0)

    assert result is None
    fake_consumer_instance.commit.assert_not_called()


def test_poll_request_commits_when_msg_value_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, fake_consumer_instance = _make_consumer(monkeypatch)

    fake_msg = MagicMock()
    fake_msg.error.return_value = None
    fake_msg.value.return_value = None
    fake_consumer_instance.poll.return_value = fake_msg

    result = consumer.poll_request(1.0)

    assert result is None
    fake_consumer_instance.commit.assert_called_once_with(message=fake_msg, asynchronous=False)


def test_commit_delegates_to_underlying_consumer(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, fake_consumer_instance = _make_consumer(monkeypatch)
    fake_msg = MagicMock()

    consumer.commit(fake_msg)

    fake_consumer_instance.commit.assert_called_once_with(message=fake_msg, asynchronous=False)


def test_commit_raises_commit_error_and_does_not_swallow_kafka_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, fake_consumer_instance = _make_consumer(monkeypatch)
    fake_msg = MagicMock()
    fake_consumer_instance.commit.side_effect = KafkaException("commit failed")

    with pytest.raises(CommitError):
        consumer.commit(fake_msg)


def test_close_delegates_to_underlying_consumer(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, fake_consumer_instance = _make_consumer(monkeypatch)

    consumer.close()

    fake_consumer_instance.close.assert_called_once()
