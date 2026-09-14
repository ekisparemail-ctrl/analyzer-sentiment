import pytest

from video.analyzer import AnalyzerModels, VideoAnalysisError, analyze_video


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
