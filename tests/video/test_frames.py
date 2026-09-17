import io
import json
import os
import subprocess
from unittest.mock import MagicMock

import pytest
from PIL import Image

from video.frames import FrameExtractionError, _grayscale_diff_score, extract_frames


def _frames_dir_of(pattern: str) -> str:
    return pattern.rsplit("\\", 1)[0] if "\\" in pattern else pattern.rsplit("/", 1)[0]


def _solid_jpeg_bytes(gray_value: int, size: tuple[int, int] = (4, 4)) -> bytes:
    buf = io.BytesIO()
    Image.new("L", size, color=gray_value).convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()


def _fake_run_factory(gray_values: list[int], duration: str = "10.0"):
    def _fake_run(cmd, check, capture_output, text=False):
        if cmd[0] == "ffprobe":
            return MagicMock(stdout=json.dumps({"format": {"duration": duration}}), returncode=0)
        if cmd[0] == "ffmpeg":
            frames_dir = _frames_dir_of(cmd[-1])
            for i, gray in enumerate(gray_values):
                with open(os.path.join(frames_dir, f"frame-{i:03d}.jpg"), "wb") as f:
                    f.write(_solid_jpeg_bytes(gray))
            return MagicMock(returncode=0)
        raise AssertionError(f"unexpected command: {cmd}")

    return _fake_run


# --- grayscale-difference scoring helper (spec revision-1 section 8: direct,
# synthetic, no real video needed) ---


def test_grayscale_diff_score_is_near_zero_for_identical_frames() -> None:
    frame = _solid_jpeg_bytes(128)

    score = _grayscale_diff_score(frame, frame)

    assert score < 1.0


def test_grayscale_diff_score_is_near_255_for_black_vs_white_frames() -> None:
    black = _solid_jpeg_bytes(0)
    white = _solid_jpeg_bytes(255)

    score = _grayscale_diff_score(black, white)

    assert score > 250.0


# --- extract_frames: keyframe selection ---


def test_extract_frames_returns_base64_data_urls_for_kept_keyframes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("video.frames.subprocess.run", _fake_run_factory([0, 255, 0]))

    result = extract_frames("/tmp/video.mp4", max_frames=5, diff_threshold=50.0)

    assert len(result) == 2  # first candidate is the seed reference, not itself kept
    for frame in result:
        assert frame.startswith("data:image/jpeg;base64,")


def test_extract_frames_does_not_keep_near_identical_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("video.frames.subprocess.run", _fake_run_factory([128, 128, 128]))

    result = extract_frames("/tmp/video.mp4", max_frames=5, diff_threshold=10.0)

    assert result == []


def test_extract_frames_returns_empty_list_when_no_candidate_clears_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Known edge case (spec revision-1 section 3.1 step 5): a visually static
    # clip can produce zero keyframes -- this must be a valid result, not an
    # error, so video/analyzer.py can degrade to transcript-only.
    monkeypatch.setattr("video.frames.subprocess.run", _fake_run_factory([100, 105, 102, 101]))

    result = extract_frames("/tmp/video.mp4", max_frames=5, diff_threshold=50.0)

    assert result == []


def test_extract_frames_keeps_candidate_when_score_exceeds_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("video.frames.subprocess.run", _fake_run_factory([100, 150]))

    result = extract_frames("/tmp/video.mp4", max_frames=5, diff_threshold=40.0)

    assert len(result) == 1


def test_extract_frames_skips_candidate_when_score_is_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("video.frames.subprocess.run", _fake_run_factory([100, 150]))

    result = extract_frames("/tmp/video.mp4", max_frames=5, diff_threshold=60.0)

    assert result == []


def test_extract_frames_caps_kept_frames_at_max_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    alternating = [0, 255] * 5  # 10 candidates, every neighbor pair clears any real threshold
    monkeypatch.setattr("video.frames.subprocess.run", _fake_run_factory(alternating))

    result = extract_frames("/tmp/video.mp4", max_frames=3, diff_threshold=50.0)

    assert len(result) == 3


def test_extract_frames_raises_on_ffprobe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_run(cmd, check, capture_output, text=False):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr("video.frames.subprocess.run", broken_run)

    with pytest.raises(FrameExtractionError):
        extract_frames("/tmp/video.mp4", max_frames=5, diff_threshold=10.0)


def test_extract_frames_raises_when_ffmpeg_produces_no_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("video.frames.subprocess.run", _fake_run_factory([]))

    with pytest.raises(FrameExtractionError, match="no frames"):
        extract_frames("/tmp/video.mp4", max_frames=5, diff_threshold=10.0)


def test_extract_frames_raises_on_ffmpeg_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd, check, capture_output, text=False):
        if cmd[0] == "ffprobe":
            return MagicMock(stdout=json.dumps({"format": {"duration": "10.0"}}), returncode=0)
        if cmd[0] == "ffmpeg":
            raise subprocess.CalledProcessError(1, cmd)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("video.frames.subprocess.run", fake_run)

    with pytest.raises(FrameExtractionError, match="ffmpeg frame extraction failed"):
        extract_frames("/tmp/video.mp4", max_frames=5, diff_threshold=10.0)


def test_extract_frames_computes_fps_from_duration_and_oversampled_candidate_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_cmds: list[list[str]] = []

    def fake_run(cmd, check, capture_output, text=False):
        captured_cmds.append(cmd)
        if cmd[0] == "ffprobe":
            return MagicMock(stdout=json.dumps({"format": {"duration": "20.0"}}), returncode=0)
        if cmd[0] == "ffmpeg":
            frames_dir = _frames_dir_of(cmd[-1])
            with open(os.path.join(frames_dir, "frame-000.jpg"), "wb") as f:
                f.write(_solid_jpeg_bytes(0))
            return MagicMock(returncode=0)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("video.frames.subprocess.run", fake_run)

    extract_frames("/tmp/video.mp4", max_frames=4, diff_threshold=10.0)

    # candidate_count = max_frames(4) * CANDIDATE_OVERSAMPLE_FACTOR(3) = 12
    ffmpeg_cmd = next(cmd for cmd in captured_cmds if cmd[0] == "ffmpeg")
    fps_arg = ffmpeg_cmd[ffmpeg_cmd.index("-vf") + 1]
    assert fps_arg == "fps=0.6"


def test_extract_frames_falls_back_to_1fps_when_duration_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_cmds: list[list[str]] = []

    def fake_run(cmd, check, capture_output, text=False):
        captured_cmds.append(cmd)
        if cmd[0] == "ffprobe":
            return MagicMock(stdout=json.dumps({"format": {"duration": "0"}}), returncode=0)
        if cmd[0] == "ffmpeg":
            frames_dir = _frames_dir_of(cmd[-1])
            with open(os.path.join(frames_dir, "frame-000.jpg"), "wb") as f:
                f.write(_solid_jpeg_bytes(0))
            return MagicMock(returncode=0)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("video.frames.subprocess.run", fake_run)

    extract_frames("/tmp/video.mp4", max_frames=5, diff_threshold=10.0)

    ffmpeg_cmd = next(cmd for cmd in captured_cmds if cmd[0] == "ffmpeg")
    fps_arg = ffmpeg_cmd[ffmpeg_cmd.index("-vf") + 1]
    assert fps_arg == "fps=1.0"


# --- CUDA hardware-acceleration flag (spec revision-1 section 4) ---


def test_extract_frames_adds_cuda_hwaccel_flags_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_cmds: list[list[str]] = []

    def fake_run(cmd, check, capture_output, text=False):
        captured_cmds.append(cmd)
        return _fake_run_factory([0])(cmd, check, capture_output, text)

    monkeypatch.setattr("video.frames.subprocess.run", fake_run)

    extract_frames("/tmp/video.mp4", max_frames=5, diff_threshold=10.0, hwaccel_cuda=True)

    ffmpeg_cmd = next(cmd for cmd in captured_cmds if cmd[0] == "ffmpeg")
    assert "-hwaccel" in ffmpeg_cmd
    assert ffmpeg_cmd[ffmpeg_cmd.index("-hwaccel") + 1] == "cuda"
    assert "-hwaccel_output_format" in ffmpeg_cmd
    assert ffmpeg_cmd[ffmpeg_cmd.index("-hwaccel_output_format") + 1] == "cuda"
    # must come before the input, per ffmpeg's own argument-ordering rules
    assert ffmpeg_cmd.index("-hwaccel") < ffmpeg_cmd.index("-i")


def test_extract_frames_omits_cuda_hwaccel_flags_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_cmds: list[list[str]] = []

    def fake_run(cmd, check, capture_output, text=False):
        captured_cmds.append(cmd)
        return _fake_run_factory([0])(cmd, check, capture_output, text)

    monkeypatch.setattr("video.frames.subprocess.run", fake_run)

    extract_frames("/tmp/video.mp4", max_frames=5, diff_threshold=10.0)

    ffmpeg_cmd = next(cmd for cmd in captured_cmds if cmd[0] == "ffmpeg")
    assert "-hwaccel" not in ffmpeg_cmd
    assert "-hwaccel_output_format" not in ffmpeg_cmd
