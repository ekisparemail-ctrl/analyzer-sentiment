import json
import subprocess
from unittest.mock import MagicMock

import pytest

from video.frames import FrameExtractionError, extract_frames


def _fake_run_factory(frame_count: int):
    def _fake_run(cmd, check, capture_output, text=False):
        if cmd[0] == "ffprobe":
            return MagicMock(stdout=json.dumps({"format": {"duration": "10.0"}}), returncode=0)
        if cmd[0] == "ffmpeg":
            pattern = cmd[-1]
            if "\\" in pattern:
                frames_dir = pattern.rsplit("\\", 1)[0]
            else:
                frames_dir = pattern.rsplit("/", 1)[0]
            import os

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
