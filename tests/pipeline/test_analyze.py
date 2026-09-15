from ai.llm_client import LLMClientError, SentimentAnalysis
from pipeline.analyze import AnalyzeDependencies, analyze
from schemas import (
    AnalysisContext,
    AnalysisRequest,
    EmotionResult,
    MotivationResult,
    Platform,
    SentimentResult,
    Status,
)
from video.analyzer import VideoAnalysis, VideoAnalysisError
from video.downloader import VideoDownloadError


def _sentiment_analysis() -> SentimentAnalysis:
    return SentimentAnalysis(
        context=AnalysisContext(topic="politics", key_themes=["korupsi"]),
        sentiment=SentimentResult(label="negative", score=-0.8, indicators=["marah"]),
        emotion=EmotionResult(
            primary="anger", secondary="disgust", intensity=0.9, indicators=["MAMPUS"]
        ),
        motivation=MotivationResult(
            type="criticizing", confidence_score=0.7, indicators=["skandal"]
        ),
    )


def _deps(**overrides: object) -> AnalyzeDependencies:
    defaults: dict = dict(
        download_video=lambda url, dest_dir: "/tmp/video.mp4",
        analyze_video=lambda path, prompt, max_frames, max_tokens, include_transcript: (
            VideoAnalysis(
                summary="a person talking",
                transcript="hello everyone",
                transcript_segments=[],
            )
        ),
        summarize_video=lambda vlm_summary, transcript: "combined video summary",
        analyze_sentiment=lambda content: _sentiment_analysis(),
    )
    defaults.update(overrides)
    return AnalyzeDependencies(**defaults)  # type: ignore[arg-type]


def test_analyze_text_and_video_produces_ok_status() -> None:
    request = AnalysisRequest(id="1", platform=Platform.TWITTER, text="teks asli", video_url="https://x.com/1")

    result = analyze(request, _deps())

    assert result.id == "1"
    assert result.status == Status.OK
    assert result.video_summary == "combined video summary"
    assert result.sentiment is not None and result.sentiment.label == "negative"
    assert result.error is None


def test_analyze_video_only_produces_ok_status() -> None:
    request = AnalysisRequest(id="2", platform=Platform.TIKTOK, video_url="https://tiktok.com/2")

    result = analyze(request, _deps())

    assert result.status == Status.OK
    assert result.video_summary == "combined video summary"


def test_analyze_text_only_skips_video_steps() -> None:
    calls: list[str] = []
    request = AnalysisRequest(id="3", platform=Platform.FACEBOOK, text="teks asli")

    result = analyze(
        request,
        _deps(download_video=lambda url, dest_dir: calls.append("download") or "/tmp/video.mp4"),
    )

    assert result.status == Status.OK
    assert result.video_summary is None
    assert calls == []


def test_analyze_both_missing_returns_failed_status() -> None:
    request = AnalysisRequest(id="4", platform=Platform.TWITTER)

    result = analyze(request, _deps())

    assert result.status == Status.FAILED
    assert result.error is not None and "insufficient input" in result.error


def test_analyze_video_download_failure_degrades_to_text_only() -> None:
    def broken_download(url: str, dest_dir: str) -> str:
        raise VideoDownloadError("private video")

    request = AnalysisRequest(id="5", platform=Platform.INSTAGRAM, text="teks asli", video_url="https://instagram.com/5")

    result = analyze(request, _deps(download_video=broken_download))

    assert result.status == Status.PARTIAL
    assert result.video_summary is None
    assert result.error is not None and "video processing failed" in result.error
    assert result.sentiment is not None


def test_analyze_video_analyzer_failure_degrades_to_text_only() -> None:
    def broken_analyze_video(path, prompt, max_frames, max_tokens, include_transcript):
        raise VideoAnalysisError("out of memory")

    request = AnalysisRequest(id="6", platform=Platform.INSTAGRAM, text="teks asli", video_url="https://instagram.com/6")

    result = analyze(request, _deps(analyze_video=broken_analyze_video))

    assert result.status == Status.PARTIAL
    assert result.video_summary is None
    assert result.sentiment is not None


def test_analyze_video_only_download_failure_with_no_text_returns_failed() -> None:
    def broken_download(url: str, dest_dir: str) -> str:
        raise VideoDownloadError("private video")

    request = AnalysisRequest(id="7", platform=Platform.INSTAGRAM, video_url="https://instagram.com/7")

    result = analyze(request, _deps(download_video=broken_download))

    assert result.status == Status.FAILED
    assert result.error is not None and "insufficient input" in result.error


def test_analyze_sentiment_retry_exhausted_returns_failed_status() -> None:
    def broken_analyze_sentiment(content: str) -> SentimentAnalysis:
        raise LLMClientError("LLM unreachable")

    request = AnalysisRequest(id="8", platform=Platform.TWITTER, text="teks asli")

    result = analyze(request, _deps(analyze_sentiment=broken_analyze_sentiment))

    assert result.status == Status.FAILED
    assert result.error is not None and "sentiment analysis failed" in result.error


def test_analyze_video_failure_and_sentiment_failure_preserves_both_errors() -> None:
    def broken_download(url: str, dest_dir: str) -> str:
        raise VideoDownloadError("private video")

    def broken_analyze_sentiment(content: str) -> SentimentAnalysis:
        raise LLMClientError("LLM unreachable")

    request = AnalysisRequest(
        id="9", platform=Platform.TWITTER, text="teks asli", video_url="https://x.com/9"
    )

    result = analyze(
        request,
        _deps(download_video=broken_download, analyze_sentiment=broken_analyze_sentiment),
    )

    assert result.status == Status.FAILED
    assert result.error is not None
    assert "video processing failed" in result.error
    assert "sentiment analysis failed" in result.error
