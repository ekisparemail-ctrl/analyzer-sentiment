import json
import os
import subprocess
from unittest.mock import MagicMock

import pytest

from video.frames import FrameExtractionError, extract_frames


def _frames_dir_of(pattern: str) -> str:
    return pattern.rsplit("\\", 1)[0] if "\\" in pattern else pattern.rsplit("/", 1)[0]


def _fake_run_factory(frame_count: int):
    def _fake_run(cmd, check, capture_output, text=False):
        if cmd[0] == "ffprobe":
            return MagicMock(stdout=json.dumps({"format": {"duration": "10.0"}}), returncode=0)
        if cmd[0] == "ffmpeg":
            frames_dir = _frames_dir_of(cmd[-1])
            for i in range(frame_count):
                with open(os.path.join(frames_dir, f"frame-{i:03d}.jpg"), "wb") as f:
                    f.write(b"\xff\xd8\xff\xe0fakejpeg")
            return MagicMock(returncode=0)
        raise AssertionError(f"unexpected command: {cmd}")

    return _fake_run


def test_extract_frames_returns_base64_data_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("video.frames.subprocess.run", _fake_run_factory(3))

    result = extract_frames("/tmp/video.mp4", max_frames=5)

    assert len(result) == 3
    for url in result:
        assert url.startswith("data:image/jpeg;base64,")


def test_extract_frames_caps_at_max_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("video.frames.subprocess.run", _fake_run_factory(10))

    result = extract_frames("/tmp/video.mp4", max_frames=4)

    assert len(result) == 4


def test_extract_frames_raises_on_ffprobe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_run(cmd, check, capture_output, text=False):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr("video.frames.subprocess.run", broken_run)

    with pytest.raises(FrameExtractionError):
        extract_frames("/tmp/video.mp4", max_frames=5)


def test_extract_frames_raises_when_ffmpeg_produces_no_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("video.frames.subprocess.run", _fake_run_factory(0))

    with pytest.raises(FrameExtractionError, match="no frames"):
        extract_frames("/tmp/video.mp4", max_frames=5)


def test_extract_frames_raises_on_ffmpeg_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd, check, capture_output, text=False):
        if cmd[0] == "ffprobe":
            return MagicMock(stdout=json.dumps({"format": {"duration": "10.0"}}), returncode=0)
        if cmd[0] == "ffmpeg":
            raise subprocess.CalledProcessError(1, cmd)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("video.frames.subprocess.run", fake_run)

    with pytest.raises(FrameExtractionError, match="ffmpeg frame extraction failed"):
        extract_frames("/tmp/video.mp4", max_frames=5)


def test_extract_frames_computes_fps_from_duration_and_max_frames(
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
                f.write(b"\xff\xd8\xff\xe0fakejpeg")
            return MagicMock(returncode=0)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("video.frames.subprocess.run", fake_run)

    extract_frames("/tmp/video.mp4", max_frames=4)

    ffmpeg_cmd = next(cmd for cmd in captured_cmds if cmd[0] == "ffmpeg")
    fps_arg = ffmpeg_cmd[ffmpeg_cmd.index("-vf") + 1]
    assert fps_arg == "fps=0.2"


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
                f.write(b"\xff\xd8\xff\xe0fakejpeg")
            return MagicMock(returncode=0)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("video.frames.subprocess.run", fake_run)

    extract_frames("/tmp/video.mp4", max_frames=5)

    ffmpeg_cmd = next(cmd for cmd in captured_cmds if cmd[0] == "ffmpeg")
    fps_arg = ffmpeg_cmd[ffmpeg_cmd.index("-vf") + 1]
    assert fps_arg == "fps=1.0"
