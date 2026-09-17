import math

import pytest

from ai.vlm_client import VLMConfig
from video.analyzer import AnalyzerModels, VideoAnalysisError, _summarize_frames, analyze_video


def _vlm_config(**overrides: object) -> VLMConfig:
    defaults = dict(base_url="http://localhost:1234/v1", model="test-vlm-model")
    defaults.update(overrides)
    return VLMConfig(**defaults)  # type: ignore[arg-type]


def _models(
    generate_summary=None,
    transcribe=None,
) -> AnalyzerModels:
    return AnalyzerModels(
        generate_summary=(
            generate_summary
            or (lambda path, prompt, max_frames, max_tokens: "a summary")
        ),
        transcribe=(
            transcribe
            or (
                lambda path: (
                    "a transcript",
                    [{"start": 0.0, "end": 1.0, "text": "hi"}],
                )
            )
        ),
    )


def test_analyze_video_returns_summary_and_transcript_when_requested() -> None:
    models = _models()

    result = analyze_video(
        models, "/tmp/video.mp4", "Describe what is happening.", max_frames=32, max_tokens=500,
        include_transcript=True,
    )

    assert result.summary == "a summary"
    assert result.transcript == "a transcript"
    assert result.transcript_segments == [{"start": 0.0, "end": 1.0, "text": "hi"}]


def test_analyze_video_skips_transcript_when_not_requested() -> None:
    transcribe_calls = []

    def transcribe(path: str) -> tuple[str, list[dict]]:
        transcribe_calls.append(path)
        return "a transcript", []

    models = _models(transcribe=transcribe)

    result = analyze_video(
        models, "/tmp/video.mp4", "Describe what is happening.", max_frames=32, max_tokens=500,
        include_transcript=False,
    )

    assert result.transcript is None
    assert result.transcript_segments is None
    assert transcribe_calls == []


def test_analyze_video_raises_video_analysis_error_when_vlm_call_fails() -> None:
    def broken_generate_summary(path: str, prompt: str, max_frames: int, max_tokens: int) -> str:
        raise RuntimeError("out of memory")

    models = _models(generate_summary=broken_generate_summary)

    with pytest.raises(VideoAnalysisError, match="out of memory"):
        analyze_video(
            models, "/tmp/video.mp4", "Describe what is happening.", max_frames=32, max_tokens=500,
            include_transcript=True,
        )


def test_analyze_video_degrades_gracefully_when_only_transcription_fails() -> None:
    def broken_transcribe(path: str) -> tuple[str, list[dict]]:
        raise RuntimeError("whisper crashed")

    models = _models(transcribe=broken_transcribe)

    result = analyze_video(
        models, "/tmp/video.mp4", "Describe what is happening.", max_frames=32, max_tokens=500,
        include_transcript=True,
    )

    assert result.summary == "a summary"
    assert result.transcript is None
    assert result.transcript_segments is None


# --- batched VLM calls (spec revision-1 section 3.2) ---


def test_summarize_frames_calls_describe_images_once_per_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_describe_images(config: VLMConfig, prompt: str, images: list[str]) -> str:
        calls.append(images)
        return f"batch of {len(images)}"

    monkeypatch.setattr("video.analyzer.describe_images", fake_describe_images)
    frames = [f"frame{i}" for i in range(10)]

    _summarize_frames(frames, "Describe.", _vlm_config(), vlm_batch_enabled=True, vlm_batch_size=4)

    assert len(calls) == math.ceil(10 / 4)


def test_summarize_frames_uses_a_single_call_when_batching_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_describe_images(config: VLMConfig, prompt: str, images: list[str]) -> str:
        calls.append(images)
        return "one big batch"

    monkeypatch.setattr("video.analyzer.describe_images", fake_describe_images)
    frames = [f"frame{i}" for i in range(7)]

    _summarize_frames(
        frames, "Describe.", _vlm_config(), vlm_batch_enabled=False, vlm_batch_size=4
    )

    assert len(calls) == 1
    assert len(calls[0]) == 7


def test_summarize_frames_joins_batch_descriptions_chronologically_with_frame_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_describe_images(config: VLMConfig, prompt: str, images: list[str]) -> str:
        return f"desc-{images[0]}"

    monkeypatch.setattr("video.analyzer.describe_images", fake_describe_images)
    frames = ["f0", "f1", "f2", "f3", "f4"]

    result = _summarize_frames(
        frames, "Describe.", _vlm_config(), vlm_batch_enabled=True, vlm_batch_size=2
    )

    assert result == (
        "[Frames 1-2] desc-f0\n\n[Frames 3-4] desc-f2\n\n[Frames 5-5] desc-f4"
    )


def test_summarize_frames_returns_empty_string_and_skips_the_vlm_for_zero_keyframes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_describe_images(config: VLMConfig, prompt: str, images: list[str]) -> str:
        nonlocal called
        called = True
        return "should not happen"

    monkeypatch.setattr("video.analyzer.describe_images", fake_describe_images)

    result = _summarize_frames(
        [], "Describe.", _vlm_config(), vlm_batch_enabled=True, vlm_batch_size=4
    )

    assert result == ""
    assert called is False
