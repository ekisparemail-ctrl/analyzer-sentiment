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


def load_models(vlm_model_id: str, whisper_model_id: str) -> AnalyzerModels:
    """
    Loads the mlx-vlm and mlx-whisper models once and returns them wrapped as plain
    callables. Requires macOS on Apple Silicon with mlx-vlm/mlx-whisper installed —
    not exercised in unit tests (see plan Global Constraints / spec section 7).
    """
    import mlx_vlm  # type: ignore[import-not-found]
    import mlx_whisper  # type: ignore[import-not-found]
    from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore[import-not-found]
    from mlx_vlm.utils import load_config  # type: ignore[import-not-found]

    model, processor = mlx_vlm.load(vlm_model_id)
    config = load_config(vlm_model_id)

    def generate_summary(video_path: str, prompt: str, max_frames: int, max_tokens: int) -> str:
        formatted_prompt = apply_chat_template(processor, config, prompt, num_images=0)
        output = mlx_vlm.generate(
            model,
            processor,
            formatted_prompt,
            video=video_path,
            max_tokens=max_tokens,
            max_frames=max_frames,
            temperature=0.0,
        )
        text = getattr(output, "text", output)
        return str(text)

    def transcribe(video_path: str) -> tuple[str, list[dict]]:
        result = mlx_whisper.transcribe(
            video_path, path_or_hf_repo=whisper_model_id, word_timestamps=False
        )
        segments = [
            {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in result.get("segments", [])
        ]
        return result["text"].strip(), segments

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
