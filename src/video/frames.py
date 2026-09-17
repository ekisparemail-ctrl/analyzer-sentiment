import base64
import io
import json
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image


class FrameExtractionError(Exception):
    pass


# ffmpeg extracts candidates at this multiple of max_frames -- enough
# temporal resolution to have real choices when scoring for keyframes,
# without decoding/scoring far more frames than needed. Mirrors
# video-analyzer's two-stage "oversample, then pick the best subset"
# approach (spec revision-1 section 3.1) while ffmpeg remains the only
# thing doing video decode (no OpenCV VideoCapture).
CANDIDATE_OVERSAMPLE_FACTOR = 3


def extract_frames(
    video_path: str,
    max_frames: int,
    diff_threshold: float,
    hwaccel_cuda: bool = False,
) -> list[str]:
    """
    Extracts keyframes from a video using grayscale-difference scoring
    (video-analyzer's algorithm, spec revision-1 section 3.1, reproduced
    with Pillow + numpy instead of OpenCV) and returns them as base64
    data:image/jpeg;base64,... URLs for an OpenAI-compatible vision chat
    completion request.

    ffmpeg extracts up to `max_frames * CANDIDATE_OVERSAMPLE_FACTOR`
    candidate frames, evenly spaced. Each candidate is scored against the
    immediately preceding candidate (matching video-analyzer's own
    frame.py exactly -- its `prev_frame` is reassigned unconditionally
    after every comparison, whether or not the candidate cleared the
    threshold) and kept only if its score exceeds `diff_threshold`, capped
    at `max_frames`. A visually static video can legitimately produce zero
    keyframes -- this returns an empty list in that case rather than
    raising; callers must treat that as a valid, non-error outcome (spec
    revision-1 section 3.1 step 5).
    """
    duration = _probe_duration_seconds(video_path)
    candidate_count = max_frames * CANDIDATE_OVERSAMPLE_FACTOR
    fps = candidate_count / duration if duration > 0 else 1.0

    with tempfile.TemporaryDirectory() as frames_dir:
        pattern = os.path.join(frames_dir, "frame-%03d.jpg")
        cmd = ["ffmpeg", "-y"]
        if hwaccel_cuda:
            # Must precede -i, per ffmpeg's own argument-ordering rules
            # (spec revision-1 section 4).
            cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        cmd += [
            "-i",
            video_path,
            "-vf",
            f"fps={fps}",
            "-frames:v",
            str(candidate_count),
            pattern,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except (subprocess.CalledProcessError, OSError) as e:
            raise FrameExtractionError(f"ffmpeg frame extraction failed: {e}") from e

        frame_files = sorted(os.listdir(frames_dir))[:candidate_count]
        if not frame_files:
            raise FrameExtractionError("ffmpeg produced no frames")

        candidates = []
        for filename in frame_files:
            with open(os.path.join(frames_dir, filename), "rb") as f:
                candidates.append(f.read())

        kept = _select_keyframes(candidates, max_frames, diff_threshold)
        return [
            f"data:image/jpeg;base64,{base64.b64encode(frame).decode('ascii')}" for frame in kept
        ]


def _select_keyframes(
    candidates: list[bytes], max_frames: int, diff_threshold: float
) -> list[bytes]:
    """
    The first candidate seeds the comparison reference but is never itself
    counted as a keyframe (there is nothing earlier to diff it against).
    Every later candidate is scored against the *immediately preceding
    candidate* -- the reference always advances to the current candidate
    after scoring, regardless of whether it was kept -- matching
    video-analyzer's own frame.py exactly (its `prev_frame = frame.copy()`
    runs unconditionally after every comparison, not only when the
    candidate clears the threshold).
    """
    if not candidates:
        return []

    reference = candidates[0]
    kept: list[bytes] = []
    for candidate in candidates[1:]:
        if len(kept) >= max_frames:
            break
        if _grayscale_diff_score(reference, candidate) > diff_threshold:
            kept.append(candidate)
        reference = candidate
    return kept


def _grayscale_diff_score(frame_a: bytes, frame_b: bytes) -> float:
    """
    Grayscale mean-absolute-difference between two JPEG frames -- the same
    metric video-analyzer uses (cv2.absdiff + numpy.mean), reproduced with
    Pillow + numpy instead of OpenCV so this project doesn't need to add
    opencv-python as a dependency (spec revision-1 sections 3.1 and 7).
    """
    array_a = np.asarray(Image.open(io.BytesIO(frame_a)).convert("L"))
    array_b = np.asarray(Image.open(io.BytesIO(frame_b)).convert("L"))
    return float(np.abs(array_a.astype(int) - array_b.astype(int)).mean())


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
