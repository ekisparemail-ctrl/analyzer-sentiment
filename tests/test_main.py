from unittest.mock import MagicMock

import pytest

from ai.llm_client import SentimentAnalysis
from main import run_once
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


def test_run_once_returns_false_and_does_nothing_when_no_message_polled() -> None:
    consumer = MagicMock()
    consumer.poll_request.return_value = None
    producer = MagicMock()

    processed = run_once(consumer, producer, _deps())

    assert processed is False
    producer.publish.assert_not_called()
    consumer.commit.assert_not_called()


def test_run_once_analyzes_publishes_and_commits_when_message_polled() -> None:
    request = AnalysisRequest(id="1", platform=Platform.TWITTER, text="teks asli")
    fake_msg = object()
    consumer = MagicMock()
    consumer.poll_request.return_value = (request, fake_msg)
    producer = MagicMock()

    processed = run_once(consumer, producer, _deps())

    assert processed is True
    producer.publish.assert_called_once()
    published_result = producer.publish.call_args[0][0]
    assert published_result.id == "1"
    consumer.commit.assert_called_once_with(fake_msg)


def test_run_once_does_not_commit_when_publish_raises() -> None:
    request = AnalysisRequest(id="1", platform=Platform.TWITTER, text="teks asli")
    fake_msg = object()
    consumer = MagicMock()
    consumer.poll_request.return_value = (request, fake_msg)
    producer = MagicMock()
    producer.publish.side_effect = PublishError("boom")

    with pytest.raises(PublishError):
        run_once(consumer, producer, _deps())

    producer.publish.assert_called_once()
    consumer.commit.assert_not_called()
