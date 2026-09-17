import logging
from unittest.mock import MagicMock, patch

import pytest

from ai.llm_client import SentimentAnalysis
from config import Settings
from main import _log_startup_banner, run_forever, run_once
from messaging.producer import PublishError
from pipeline.analyze import AnalyzeDependencies
from schemas import (
    AnalysisContext,
    AnalysisRequest,
    EmotionResult,
    MotivationResult,
    Platform,
    SentimentResult,
)
from video.analyzer import VideoAnalysis


def _deps() -> AnalyzeDependencies:
    def analyze_video_mock(
        path, prompt, max_frames, max_tokens, include_transcript
    ):
        return VideoAnalysis(summary="s", transcript="t", transcript_segments=[])

    return AnalyzeDependencies(
        download_video=lambda url, dest_dir: "/tmp/video.mp4",
        analyze_video=analyze_video_mock,
        summarize_video=lambda vlm_summary, transcript: "video summary",
        analyze_sentiment=lambda content: SentimentAnalysis(
            context=AnalysisContext(topic="x", key_themes=[]),
            sentiment=SentimentResult(label="neutral", score=0.0, indicators=[]),
            emotion=EmotionResult(
                primary="calm", secondary="calm", intensity=0.1, indicators=[]
            ),
            motivation=MotivationResult(
                type="informative", confidence_score=0.5, indicators=[]
            ),
        ),
    )


def test_log_startup_banner_reports_the_models_in_use(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        kafka_bootstrap_servers="172.16.16.100:21000",
        llm_base_url="http://localhost:1234/v1",
        llm_model="test-llm-model",
        vlm_base_url="http://localhost:5678/v1",
        vlm_model="test-vlm-model",
        whisper_model_size="base",
    )

    with caplog.at_level(logging.INFO):
        _log_startup_banner(settings)

    assert "http://localhost:5678/v1" in caplog.text
    assert "test-vlm-model" in caplog.text
    assert "base" in caplog.text
    assert "test-llm-model" in caplog.text
    assert "scrapper-to-analysis" in caplog.text


def test_run_once_returns_false_and_does_nothing_when_no_message_polled() -> None:
    consumer = MagicMock()
    consumer.poll_requests.return_value = None
    producer = MagicMock()

    processed = run_once(consumer, producer, _deps())

    assert processed is False
    producer.publish.assert_not_called()
    consumer.commit.assert_not_called()


def test_run_once_analyzes_publishes_and_commits_when_message_polled() -> None:
    request = AnalysisRequest(id="1", platform=Platform.TWITTER, text="teks asli")
    fake_msg = object()
    consumer = MagicMock()
    consumer.poll_requests.return_value = ([request], fake_msg)
    producer = MagicMock()

    processed = run_once(consumer, producer, _deps())

    assert processed is True
    producer.publish.assert_called_once()
    published_result = producer.publish.call_args[0][0]
    assert published_result.id == "1"
    consumer.commit.assert_called_once_with(fake_msg)


def test_run_once_processes_every_request_from_one_message_and_commits_once() -> None:
    # Regression test: the Scrapper Backend stopped publishing each comment
    # as its own top-level message -- one Kafka message (e.g. a post plus
    # its nested comments) can now yield several requests, all of which
    # must be analyzed and published before the single underlying offset
    # is committed.
    post = AnalysisRequest(id="post-1", platform=Platform.TIKTOK, text="post asli")
    comment_1 = AnalysisRequest(id="comment-1", platform=Platform.TIKTOK, text="komentar 1")
    comment_2 = AnalysisRequest(id="comment-2", platform=Platform.TIKTOK, text="komentar 2")
    fake_msg = object()
    consumer = MagicMock()
    consumer.poll_requests.return_value = ([post, comment_1, comment_2], fake_msg)
    producer = MagicMock()

    processed = run_once(consumer, producer, _deps())

    assert processed is True
    assert producer.publish.call_count == 3
    published_ids = [call.args[0].id for call in producer.publish.call_args_list]
    assert published_ids == ["post-1", "comment-1", "comment-2"]
    consumer.commit.assert_called_once_with(fake_msg)


def test_run_once_does_not_commit_when_publish_raises() -> None:
    request = AnalysisRequest(id="1", platform=Platform.TWITTER, text="teks asli")
    fake_msg = object()
    consumer = MagicMock()
    consumer.poll_requests.return_value = ([request], fake_msg)
    producer = MagicMock()
    producer.publish.side_effect = PublishError("boom")

    with pytest.raises(PublishError):
        run_once(consumer, producer, _deps())

    producer.publish.assert_called_once()
    consumer.commit.assert_not_called()


def test_run_once_stops_processing_further_requests_when_one_publish_raises() -> None:
    post = AnalysisRequest(id="post-1", platform=Platform.TIKTOK, text="post asli")
    comment_1 = AnalysisRequest(id="comment-1", platform=Platform.TIKTOK, text="komentar 1")
    fake_msg = object()
    consumer = MagicMock()
    consumer.poll_requests.return_value = ([post, comment_1], fake_msg)
    producer = MagicMock()
    producer.publish.side_effect = [None, PublishError("boom")]

    with pytest.raises(PublishError):
        run_once(consumer, producer, _deps())

    assert producer.publish.call_count == 2
    consumer.commit.assert_not_called()


def test_run_forever_stops_gracefully_on_keyboard_interrupt_and_closes_consumer() -> None:
    consumer = MagicMock()
    producer = MagicMock()
    call_count = 0

    def fake_run_once(*args: object, **kwargs: object) -> bool:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise KeyboardInterrupt
        return False

    with patch("main.run_once", side_effect=fake_run_once):
        run_forever(consumer, producer, _deps())  # must not raise

    assert call_count == 2
    consumer.close.assert_called_once()


def test_run_forever_logs_a_heartbeat_during_idle_polling(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Regression test: a real run (docs/issue-findings.md) looked completely
    # silent for its whole lifetime, with no way to tell whether it was
    # actually still polling or stuck -- a periodic heartbeat during idle
    # polling closes that gap.
    consumer = MagicMock()
    producer = MagicMock()
    fake_now = 0.0

    def fake_monotonic() -> float:
        return fake_now

    monkeypatch.setattr("main.time.monotonic", fake_monotonic)

    call_count = 0

    def fake_run_once(*args: object, **kwargs: object) -> bool:
        nonlocal call_count, fake_now
        call_count += 1
        fake_now += 31.0  # exceeds HEARTBEAT_INTERVAL_SECONDS every call
        if call_count >= 2:
            raise KeyboardInterrupt
        return False

    with patch("main.run_once", side_effect=fake_run_once), caplog.at_level(logging.INFO):
        run_forever(consumer, producer, _deps())

    assert "Still polling" in caplog.text


def test_run_forever_does_not_log_a_heartbeat_while_messages_keep_arriving(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    consumer = MagicMock()
    producer = MagicMock()
    fake_now = 0.0

    def fake_monotonic() -> float:
        return fake_now

    monkeypatch.setattr("main.time.monotonic", fake_monotonic)

    call_count = 0

    def fake_run_once(*args: object, **kwargs: object) -> bool:
        nonlocal call_count, fake_now
        call_count += 1
        fake_now += 31.0
        if call_count >= 2:
            raise KeyboardInterrupt
        return True  # a message was processed each time -> no idle heartbeat

    with patch("main.run_once", side_effect=fake_run_once), caplog.at_level(logging.INFO):
        run_forever(consumer, producer, _deps())

    assert "Still polling" not in caplog.text


def test_run_forever_continues_after_a_non_interrupt_exception() -> None:
    consumer = MagicMock()
    producer = MagicMock()
    call_count = 0

    def fake_run_once(*args: object, **kwargs: object) -> bool:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient failure")
        if call_count >= 2:
            raise KeyboardInterrupt
        return False

    with patch("main.run_once", side_effect=fake_run_once):
        run_forever(consumer, producer, _deps())  # must not raise

    assert call_count == 2
    consumer.close.assert_called_once()
