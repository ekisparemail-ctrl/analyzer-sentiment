import base64
import json
import os
import subprocess
import tempfile


class FrameExtractionError(Exception):
    pass


def extract_frames(video_path: str, max_frames: int) -> list[str]:
    """
    Extracts up to `max_frames` JPEG frames, evenly spaced across the video's
    duration, using ffmpeg/ffprobe, and returns them as base64-encoded
    data: URLs suitable for an OpenAI-compatible vision chat completion request.
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

        data_urls = []
        for filename in frame_files:
            with open(os.path.join(frames_dir, filename), "rb") as f:
                encoded = base64.b64encode(f.read()).decode("ascii")
            data_urls.append(f"data:image/jpeg;base64,{encoded}")
        return data_urls


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
