import logging
from collections.abc import Callable
from dataclasses import dataclass

from ai.vlm_client import VLMConfig, describe_images
from video.frames import extract_frames

logger = logging.getLogger(__name__)


class VideoAnalysisError(Exception):
    pass


@dataclass
class VideoAnalysis:
    summary: str
    transcript: str | None
    transcript_segments: list[dict] | None


@dataclass
class AnalyzerModels:
    # (video_path, prompt, max_frames, max_tokens) -> summary text
    generate_summary: Callable[[str, str, int, int], str]
    # (video_path) -> (transcript text, transcript segments)
    transcribe: Callable[[str], tuple[str, list[dict]]]


def load_models(
    vlm_config: VLMConfig,
    whisper_model_size: str,
    whisper_vad_filter: bool,
    # vlm_batch_enabled/vlm_batch_size and the keyframe-selection settings
    # they travel alongside are received as extra load_models() parameters
    # (rather than, say, a closure captured from module-global Settings) so
    # every knob generate_summary's closure depends on is visible in one
    # place and stays unit-testable via plain arguments -- same reasoning
    # already applied to vlm_config/whisper_model_size/whisper_vad_filter.
    vlm_batch_enabled: bool = True,
    vlm_batch_size: int = 4,
    keyframe_diff_threshold: float = 10.0,
    ffmpeg_hwaccel_cuda: bool = False,
) -> AnalyzerModels:
    """
    Loads the local faster-whisper model once (CPU-only, this host has no
    GPU) -- not exercised in unit tests (real model weights, slow to
    download/load; see plan Global Constraints). The VLM is a remote HTTP
    call (ai/vlm_client.py) again, not an in-process model, so there's
    nothing to load for it here.
    """
    from faster_whisper import WhisperModel  # type: ignore[import-untyped]

    logger.info("Loading local faster-whisper model (size=%s)...", whisper_model_size)
    whisper_model = WhisperModel(whisper_model_size, device="cpu")
    logger.info("faster-whisper model loaded.")

    def generate_summary(video_path: str, prompt: str, max_frames: int, max_tokens: int) -> str:
        logger.info(
            "Extracting up to %s keyframes from %s (diff_threshold=%s)...",
            max_frames,
            video_path,
            keyframe_diff_threshold,
        )
        frames = extract_frames(
            video_path, max_frames, keyframe_diff_threshold, ffmpeg_hwaccel_cuda
        )
        logger.info("Extracted %s keyframes (target was %s).", len(frames), max_frames)
        return _summarize_frames(frames, prompt, vlm_config, vlm_batch_enabled, vlm_batch_size)

    def transcribe(video_path: str) -> tuple[str, list[dict]]:
        logger.info("Transcribing audio from %s...", video_path)
        segments_iter, _info = whisper_model.transcribe(
            video_path, word_timestamps=False, vad_filter=whisper_vad_filter
        )
        texts = []
        segments = []
        for s in segments_iter:
            text = s.text.strip()
            texts.append(text)
            segments.append({"start": s.start, "end": s.end, "text": text})
        logger.info("Transcription complete (%s segments).", len(segments))
        return " ".join(texts).strip(), segments

    return AnalyzerModels(generate_summary=generate_summary, transcribe=transcribe)


def _summarize_frames(
    frames: list[str],
    prompt: str,
    vlm_config: VLMConfig,
    vlm_batch_enabled: bool,
    vlm_batch_size: int,
) -> str:
    """
    Splits keyframes into chunks of vlm_batch_size (or one chunk containing
    every frame, when vlm_batch_enabled is False), calls describe_images()
    once per chunk, and joins the per-batch descriptions -- each prefixed
    with its frame-index range so multiple batches read back in
    chronological order (spec revision-1 section 3.2, mirroring
    video-analyzer's "Frame N (timestamp): description" convention;
    extract_frames() doesn't carry real video timestamps through its
    base64 data: URLs, so the range is expressed in frame position rather
    than elapsed time).

    Zero keyframes (video/frames.py's documented degrade case) returns an
    empty string without calling the VLM at all -- video/analyzer.py's
    caller then has only the transcript to fall back on.
    """
    if not frames:
        logger.info("No keyframes to describe; skipping VLM call(s).")
        return ""

    chunk_size = vlm_batch_size if vlm_batch_enabled else len(frames)
    batches = [frames[i : i + chunk_size] for i in range(0, len(frames), chunk_size)]

    descriptions = []
    start = 0
    for batch_num, batch in enumerate(batches, start=1):
        end = start + len(batch)
        logger.info(
            "Sending VLM batch %s/%s (frames %s-%s, %s images)...",
            batch_num,
            len(batches),
            start + 1,
            end,
            len(batch),
        )
        description = describe_images(vlm_config, prompt, batch)
        logger.info("VLM batch %s/%s complete.", batch_num, len(batches))
        descriptions.append(f"[Frames {start + 1}-{end}] {description}")
        start = end

    return "\n\n".join(descriptions)


def analyze_video(
    models: AnalyzerModels,
    video_path: str,
    prompt: str,
    max_frames: int,
    max_tokens: int,
    include_transcript: bool,
) -> VideoAnalysis:
    try:
        summary = models.generate_summary(video_path, prompt, max_frames, max_tokens)
    except Exception as e:
        raise VideoAnalysisError(f"VLM analysis failed: {e}") from e

    transcript: str | None = None
    transcript_segments: list[dict] | None = None
    if include_transcript:
        try:
            transcript, transcript_segments = models.transcribe(video_path)
        except Exception:
            # Transcription is supplementary to the VLM summary; don't fail the
            # whole video analysis over it (spec section 6).
            transcript, transcript_segments = None, None

    return VideoAnalysis(
        summary=summary, transcript=transcript, transcript_segments=transcript_segments
    )
