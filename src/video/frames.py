import json
import os
import subprocess
import tempfile


class FrameExtractionError(Exception):
    pass


def extract_frames(video_path: str, max_frames: int) -> list[bytes]:
    """
    Extracts up to `max_frames` JPEG frames, evenly spaced across the video's
    duration, using ffmpeg/ffprobe, and returns their raw JPEG bytes (fed
    directly to the in-process VLM, see ai/vlm_local.py).
    """
    duration = _probe_duration_seconds(video_path)
    fps = max_frames / duration if duration > 0 else 1.0

    with tempfile.TemporaryDirectory() as frames_dir:
        pattern = os.path.join(frames_dir, "frame-%03d.jpg")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path,
                    "-vf",
                    f"fps={fps}",
                    "-frames:v",
                    str(max_frames),
                    pattern,
                ],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, OSError) as e:
            raise FrameExtractionError(f"ffmpeg frame extraction failed: {e}") from e

        frame_files = sorted(os.listdir(frames_dir))[:max_frames]
        if not frame_files:
            raise FrameExtractionError("ffmpeg produced no frames")

        frames = []
        for filename in frame_files:
            with open(os.path.join(frames_dir, filename), "rb") as f:
                frames.append(f.read())
        return frames


def _probe_duration_seconds(video_path: str) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                video_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (
        subprocess.CalledProcessError,
        OSError,
        KeyError,
        ValueError,
        json.JSONDecodeError,
    ) as e:
        raise FrameExtractionError(f"ffprobe duration probe failed: {e}") from e
