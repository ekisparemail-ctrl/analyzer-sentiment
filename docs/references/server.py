"""
Video analysis API server for macOS (Apple Silicon) using mlx-vlm.

Unlike a frame-by-frame loop with FastVLM, mlx-vlm passes multiple frames from
the video into a SINGLE model call. The model (Qwen2.5-VL / Qwen3-VL) reasons
across the whole clip natively, so "describe what is happening" produces one
coherent, temporally-aware summary instead of disconnected per-frame captions.

Install:
    pip install mlx-vlm fastapi uvicorn python-multipart mlx-whisper

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000

Requires ffmpeg on PATH (brew install ffmpeg) - used internally by mlx-vlm
for video frame sampling and by mlx-whisper for audio extraction.
"""

import os
import tempfile
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Swap for "mlx-community/Qwen3-VL-4B-Instruct-4bit" or a larger checkpoint
# if you have the RAM and want higher quality.
VLM_MODEL_ID = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"

# Only loaded if a request asks for a transcript.
WHISPER_MODEL_ID = "mlx-community/whisper-large-v3-turbo"

DEFAULT_PROMPT = "Describe what is happening in this video."
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB safety cap

# ---------------------------------------------------------------------------
# Model loading (once, at startup)
# ---------------------------------------------------------------------------

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading VLM: {VLM_MODEL_ID} ...")
    model, processor = load(VLM_MODEL_ID)
    config = load_config(VLM_MODEL_ID)
    state["model"] = model
    state["processor"] = processor
    state["config"] = config
    print("VLM ready.")
    yield
    state.clear()


app = FastAPI(title="Local Video VLM API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class AnalyzeResponse(BaseModel):
    summary: str
    transcript: str | None = None
    transcript_segments: list[TranscriptSegment] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_upload(video: UploadFile, dst_dir: str) -> str:
    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    video_path = os.path.join(dst_dir, f"input{suffix}")
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)
    size = os.path.getsize(video_path)
    if size == 0:
        raise HTTPException(400, "Uploaded video is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Video exceeds the size limit.")
    return video_path


def _run_vlm_on_video(video_path: str, prompt: str, max_frames: int, max_tokens: int) -> str:
    """
    Runs a single mlx-vlm call over sampled frames from the whole video, so
    the model reasons across the clip as one sequence rather than per-frame.
    """
    model = state["model"]
    processor = state["processor"]
    config = state["config"]

    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=0)

    output = generate(
        model,
        processor,
        formatted_prompt,
        video=video_path,
        max_tokens=max_tokens,
        # mlx-vlm samples frames internally; this caps how many it pulls from
        # the clip. Lower = faster / less VRAM, higher = finer temporal detail.
        max_frames=max_frames,
        temperature=0.0,
    )
    # mlx-vlm's generate() may return a plain string or a result object
    # depending on version; normalize to text.
    return getattr(output, "text", output)


def _run_transcription(video_path: str) -> dict:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        video_path,
        path_or_hf_repo=WHISPER_MODEL_ID,
        word_timestamps=False,
    )
    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in result.get("segments", [])
    ]
    return {"text": result["text"].strip(), "segments": segments}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model": VLM_MODEL_ID}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    video: UploadFile = File(...),
    prompt: str = Form(DEFAULT_PROMPT),
    max_frames: int = Form(32),
    max_tokens: int = Form(500),
    include_transcript: bool = Form(False),
):
    """
    Upload a video and get back a single, whole-video summary from mlx-vlm.
    Set include_transcript=true to also run local speech-to-text and return it.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = _save_upload(video, tmpdir)

        try:
            summary = _run_vlm_on_video(video_path, prompt, max_frames, max_tokens)
        except Exception as e:
            raise HTTPException(500, f"VLM analysis failed: {e}")

        transcript_text = None
        transcript_segments = None
        if include_transcript:
            try:
                t = _run_transcription(video_path)
                transcript_text = t["text"]
                transcript_segments = t["segments"]
            except Exception as e:
                raise HTTPException(500, f"Transcription failed: {e}")

        return AnalyzeResponse(
            summary=summary,
            transcript=transcript_text,
            transcript_segments=transcript_segments,
        )
