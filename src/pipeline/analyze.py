import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass

from ai.llm_client import LLMClientError, SentimentAnalysis
from schemas import AnalysisRequest, AnalysisResult, Status
from video.analyzer import VideoAnalysis, VideoAnalysisError
from video.downloader import VideoDownloadError

logger = logging.getLogger(__name__)


@dataclass
class AnalyzeDependencies:
    download_video: Callable[[str, str], str]
    analyze_video: Callable[[str, str, int, int, bool], VideoAnalysis]
    summarize_video: Callable[[str, str | None], str]
    analyze_sentiment: Callable[[str], SentimentAnalysis]
    video_prompt: str = "Describe what is happening in this video."
    max_frames: int = 32
    max_tokens: int = 500


def analyze(request: AnalysisRequest, deps: AnalyzeDependencies) -> AnalysisResult:
    video_summary: str | None = None
    error_parts: list[str] = []

    if request.video_url:
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                video_path = deps.download_video(request.video_url, tmp_dir)
                analysis = deps.analyze_video(
                    video_path, deps.video_prompt, deps.max_frames, deps.max_tokens, True
                )
            video_summary = deps.summarize_video(analysis.summary, analysis.transcript)
        except (VideoDownloadError, VideoAnalysisError, LLMClientError) as e:
            logger.warning("Video processing failed for request %s: %s", request.id, e)
            error_parts.append(f"video processing failed: {e}")

    context_parts = []
    if request.text:
        context_parts.append(request.text)
    if video_summary:
        context_parts.append(video_summary)

    if not context_parts:
        return AnalysisResult(
            id=request.id,
            status=Status.FAILED,
            video_summary=video_summary,
            error="insufficient input: no text or video content available"
            + ("; " + "; ".join(error_parts) if error_parts else ""),
        )

    combined_context = "\n\n".join(context_parts)
    try:
        sentiment_analysis = deps.analyze_sentiment(combined_context)
    except LLMClientError as e:
        logger.warning("Sentiment analysis failed for request %s: %s", request.id, e)
        return AnalysisResult(
            id=request.id,
            status=Status.FAILED,
            video_summary=video_summary,
            error=f"sentiment analysis failed: {e}"
            + ("; " + "; ".join(error_parts) if error_parts else ""),
        )

    status = Status.PARTIAL if error_parts else Status.OK
    return AnalysisResult(
        id=request.id,
        status=status,
        video_summary=video_summary,
        context=sentiment_analysis.context,
        sentiment=sentiment_analysis.sentiment,
        emotion=sentiment_analysis.emotion,
        motivation=sentiment_analysis.motivation,
        error="; ".join(error_parts) if error_parts else None,
    )
