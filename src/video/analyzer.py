from collections.abc import Callable
from dataclasses import dataclass


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


def load_models(vlm_model_id: str, whisper_model_size: str) -> AnalyzerModels:
    """
    Loads the local faster-whisper and VLM models once, both CPU-only (this
    host has no GPU) -- neither load is exercised in unit tests (real model
    weights, slow to download/load; see plan Global Constraints).
    """
    from faster_whisper import WhisperModel  # type: ignore[import-untyped]

    from ai.vlm_local import describe_images, load_local_vlm
    from video.frames import extract_frames

    whisper_model = WhisperModel(whisper_model_size, device="cpu")
    vlm = load_local_vlm(vlm_model_id)

    def generate_summary(video_path: str, prompt: str, max_frames: int, max_tokens: int) -> str:
        frames = extract_frames(video_path, max_frames)
        return describe_images(vlm, prompt, frames, max_tokens)

    def transcribe(video_path: str) -> tuple[str, list[dict]]:
        segments_iter, _info = whisper_model.transcribe(video_path, word_timestamps=False)
        texts = []
        segments = []
        for s in segments_iter:
            text = s.text.strip()
            texts.append(text)
            segments.append({"start": s.start, "end": s.end, "text": text})
        return " ".join(texts).strip(), segments

    return AnalyzerModels(generate_summary=generate_summary, transcribe=transcribe)


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
