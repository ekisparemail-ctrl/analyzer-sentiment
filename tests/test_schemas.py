import pytest
from pydantic import ValidationError

from schemas import (
    AnalysisContext,
    AnalysisRequest,
    AnalysisResult,
    EmotionResult,
    MotivationResult,
    Platform,
    SentimentResult,
    Status,
)


def test_analysis_request_requires_id_but_platform_text_and_video_are_optional() -> None:
    request = AnalysisRequest(id="abc123", platform=Platform.TWITTER)

    assert request.id == "abc123"
    assert request.platform == Platform.TWITTER
    assert request.text is None
    assert request.video_url is None
    assert request.metadata == {}


def test_analysis_request_allows_omitting_platform() -> None:
    # Real Kafka messages from the Scrapper Backend never carry a platform field
    # (see messaging/scrapper_dto.py) -- platform must remain optional.
    request = AnalysisRequest(id="abc123")

    assert request.platform is None


def test_analysis_request_accepts_text_and_video_url() -> None:
    request = AnalysisRequest(
        id="abc123",
        platform=Platform.TIKTOK,
        text="hello world",
        video_url="https://tiktok.com/video/123",
        metadata={"source": "scrapper"},
    )

    assert request.text == "hello world"
    assert request.video_url == "https://tiktok.com/video/123"
    assert request.metadata == {"source": "scrapper"}


def test_analysis_request_rejects_missing_id() -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest(platform=Platform.TWITTER)  # type: ignore[call-arg]


def test_analysis_request_rejects_unknown_platform() -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest(id="abc123", platform="myspace")  # type: ignore[arg-type]


def test_analysis_result_ok_status_carries_full_analysis() -> None:
    result = AnalysisResult(
        id="abc123",
        status=Status.OK,
        video_summary="a person explains something",
        context=AnalysisContext(topic="politics", key_themes=["corruption"]),
        sentiment=SentimentResult(label="negative", score=-0.8, indicators=["marah"]),
        emotion=EmotionResult(
            primary="anger",
            secondary="disgust",
            intensity=0.9,
            indicators=["MAMPUS"],
        ),
        motivation=MotivationResult(
            type="criticizing",
            confidence_score=0.7,
            indicators=["skandal"],
        ),
    )

    assert result.status == Status.OK
    assert result.error is None


def test_analysis_result_failed_status_only_requires_id_and_error() -> None:
    result = AnalysisResult(id="abc123", status=Status.FAILED, error="insufficient input")

    assert result.status == Status.FAILED
    assert result.video_summary is None
    assert result.context is None
    assert result.sentiment is None
    assert result.error == "insufficient input"
