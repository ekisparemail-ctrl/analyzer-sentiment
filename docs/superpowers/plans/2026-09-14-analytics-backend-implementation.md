# Analytics Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Analytics Backend: a Python service that consumes text/video items from Kafka, produces a video summary (via a video analyzer that extracts frames locally and calls a remote VLM, plus local CPU speech transcription, then an LLM pass), runs sentiment/emotion/motivation analysis via an LLM, and publishes the result back to Kafka — one item at a time.

**Architecture:** A single Python process wired from `main.py`: a Kafka consumer reads one `AnalysisRequest` at a time, `pipeline.analyze()` orchestrates optional video download + video analysis (local `ffmpeg` frame extraction + a remote VLM HTTP call + local `faster-whisper` transcription) + LLM summarization + LLM sentiment analysis (all through dependency-injected callables for testability), and a Kafka producer publishes the resulting `AnalysisResult` before the offset is committed. The service runs on a local Windows machine — separate from the Mac Studio that hosts the LLM (LM Studio) and VLM — so it has no in-process MLX dependency, and is packaged as a Docker image.

**Tech Stack:** Python 3.11+, pydantic v2 + pydantic-settings, httpx (LLM/VLM HTTP clients), confluent-kafka, yt-dlp, ffmpeg/ffprobe (system binaries, frame extraction), faster-whisper (local CPU transcription), pytest + respx + ruff + mypy.

**Spec:** `docs/superpowers/specs/2026-09-14-analytics-backend-design.md`

## Global Constraints

- Python 3.11+ (spec §4, §7).
- **Revised mid-implementation (after Task 5):** this service runs on a local Windows machine, not the Mac Studio. `video/analyzer.py` has no in-process MLX dependency — it extracts video frames locally via `ffmpeg`, calls a remote VLM over HTTP (via `ai/vlm_client.py`) for the summary, and runs `faster-whisper` locally on CPU for the transcript. Docker is a viable deployment target again (spec §8); `ffmpeg` must be present on PATH / in the container image.
- Kafka bootstrap servers come from env var `KAFKA_BOOTSTRAP_SERVERS` (current environment value: `172.16.16.100:21000`) — never hardcoded in source (spec §9).
- LLM is accessed via an OpenAI-compatible HTTP endpoint (LM Studio, on the Mac Studio) — config-driven base URL + model name, not hardcoded (spec §4).
- `ai/vlm_client.py` is built per spec §4's component list. **Revised:** it is now actively used by `video/analyzer.py`'s summary step (see Task 4) — it is not wired directly into `pipeline.analyze()` itself, but it is no longer dead/unused code. Its target VLM endpoint mechanism on the Mac Studio is still undecided by devops (spec §9) — only the configured base URL should need to change once confirmed.
- All functions carry type hints; data models use pydantic — no implicit `any`, no unchecked casts (Acme Types standard).
- `pytest`, `ruff`, and `mypy` must all pass before any task is considered done (Acme Testing standard).
- New logic requires tests in the same PR; bug fixes require a regression test that fails before the fix and passes after (Acme Testing standard).
- Never commit directly to main/master (Acme Review standard) — this plan assumes commits happen on a feature branch.
- `video/analyzer.py`'s `load_models()` loads a real local `faster-whisper` model — not exercised in unit tests (real model weights, slow to load; spec §7), but it requires no special hardware (CPU-only, runs on this Windows machine). Everything else in `video/analyzer.py`, and all of `video/frames.py`, is fully unit tested via dependency injection / mocked subprocess calls.

---

## Task 1: Project Scaffolding & Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `.env.example`
- Create: `src/config.py`
- Create: `src/messaging/__init__.py`
- Create: `src/pipeline/__init__.py`
- Create: `src/video/__init__.py`
- Create: `src/ai/__init__.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings` (pydantic-settings `BaseSettings` subclass) in `src/config.py`, with fields: `kafka_bootstrap_servers: str`, `kafka_request_topic: str = "analytics.requests"`, `kafka_response_topic: str = "analytics.results"`, `kafka_consumer_group: str = "analytics-backend"`, `llm_base_url: str`, `llm_model: str`, `vlm_base_url: str | None = None`, `vlm_model: str | None = None`, `vlm_model_id: str = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"`, `whisper_model_id: str = "mlx-community/whisper-large-v3-turbo"`, `max_frames: int = 32`, `max_tokens: int = 500`, `http_timeout_seconds: float = 30.0`, `retry_attempts: int = 3`, `retry_backoff_seconds: float = 1.0`.

**Superseded by a later architecture correction (see Task 4):** `vlm_model_id` and `whisper_model_id` above referred to local MLX model identifiers, which no longer apply now that the service runs on a Windows host with no in-process MLX dependency. Task 4's rework amends `Settings` to drop `vlm_model_id` (the VLM is now reached only via the existing `vlm_base_url`/`vlm_model` fields, through `ai/vlm_client.py`) and replaces `whisper_model_id` with `whisper_model_size: str = "base"` (a `faster-whisper` model size/name, not an MLX Hugging Face repo id — chosen small by default since this host is CPU-only). `max_frames`/`max_tokens` are unaffected. This note documents the amendment for anyone reading Task 1 in isolation; the authoritative field list is in Task 4.

- [ ] **Step 1: Create the dependency and tool-config files**

`requirements.txt`:
```
pydantic>=2.7,<3
pydantic-settings>=2.3,<3
httpx>=0.27,<1
confluent-kafka>=2.4,<3
yt-dlp>=2024.8.6
mlx-vlm>=0.1.0; sys_platform == "darwin"
mlx-whisper>=0.4.0; sys_platform == "darwin"
```

`requirements-dev.txt`:
```
-r requirements.txt
pytest>=8.2,<9
pytest-mock>=3.14,<4
respx>=0.21,<1
ruff>=0.5,<1
mypy>=1.10,<2
```

`pyproject.toml`:
```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]

[tool.mypy]
python_version = "3.11"
mypy_path = "src"
warn_unused_ignores = true
warn_return_any = true
disallow_untyped_defs = true
```

`.env.example`:
```
# Kafka
KAFKA_BOOTSTRAP_SERVERS=172.16.16.100:21000
KAFKA_REQUEST_TOPIC=analytics.requests
KAFKA_RESPONSE_TOPIC=analytics.results
KAFKA_CONSUMER_GROUP=analytics-backend

# LLM (LM Studio, OpenAI-compatible)
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL=replace-with-lm-studio-model-name

# VLM (external, OpenAI-compatible) - not wired into the pipeline yet, see spec section 9
VLM_BASE_URL=
VLM_MODEL=

# In-process video analyzer (mlx-vlm / mlx-whisper model ids)
VLM_MODEL_ID=mlx-community/Qwen2.5-VL-7B-Instruct-4bit
WHISPER_MODEL_ID=mlx-community/whisper-large-v3-turbo
MAX_FRAMES=32
MAX_TOKENS=500

# HTTP client behavior
HTTP_TIMEOUT_SECONDS=30
RETRY_ATTEMPTS=3
RETRY_BACKOFF_SECONDS=1
```

Create empty package marker files (each containing nothing but a trailing newline):
`src/messaging/__init__.py`, `src/pipeline/__init__.py`, `src/video/__init__.py`, `src/ai/__init__.py`.

- [ ] **Step 2: Write the failing test for `Settings`**

`tests/test_config.py`:
```python
import pytest
from pydantic import ValidationError

from config import Settings


def test_settings_loads_required_and_applies_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "172.16.16.100:21000")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    settings = Settings()

    assert settings.kafka_bootstrap_servers == "172.16.16.100:21000"
    assert settings.llm_base_url == "http://localhost:1234/v1"
    assert settings.llm_model == "test-model"
    assert settings.kafka_request_topic == "analytics.requests"
    assert settings.kafka_response_topic == "analytics.results"
    assert settings.kafka_consumer_group == "analytics-backend"
    assert settings.vlm_base_url is None
    assert settings.vlm_model_id == "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    assert settings.whisper_model_id == "mlx-community/whisper-large-v3-turbo"
    assert settings.max_frames == 32
    assert settings.max_tokens == 500
    assert settings.http_timeout_seconds == 30.0
    assert settings.retry_attempts == 3
    assert settings.retry_backoff_seconds == 1.0


def test_settings_missing_required_field_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    with pytest.raises(ValidationError):
        Settings()
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `python -m venv .venv && .venv/Scripts/pip install -r requirements-dev.txt` (or `.venv/bin/pip` on macOS/Linux), then `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'config'` (the file doesn't exist yet).

- [ ] **Step 4: Implement `src/config.py`**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    kafka_bootstrap_servers: str
    # Topic name defaults below are placeholders -- the exact contract with the
    # Scrapper Backend is not yet confirmed (spec section 9). Override via env
    # once confirmed; no other code changes should be needed.
    kafka_request_topic: str = "analytics.requests"
    kafka_response_topic: str = "analytics.results"
    kafka_consumer_group: str = "analytics-backend"

    llm_base_url: str
    llm_model: str
    vlm_base_url: str | None = None
    vlm_model: str | None = None

    vlm_model_id: str = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    whisper_model_id: str = "mlx-community/whisper-large-v3-turbo"
    max_frames: int = 32
    max_tokens: int = 500

    http_timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_backoff_seconds: float = 1.0
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Run ruff and mypy**

Run: `ruff check . && mypy src`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml requirements.txt requirements-dev.txt .env.example src/config.py src/messaging/__init__.py src/pipeline/__init__.py src/video/__init__.py src/ai/__init__.py tests/test_config.py
git commit -m "chore: scaffold project and add env-based Settings"
```

---

## Task 2: Schemas

**Files:**
- Create: `src/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Consumes: nothing (pure data models).
- Produces: `Platform` (str enum: `TWITTER="twitter"`, `TIKTOK="tiktok"`, `INSTAGRAM="instagram"`, `FACEBOOK="facebook"`), `Status` (str enum: `OK="ok"`, `PARTIAL="partial"`, `FAILED="failed"`), `AnalysisRequest(id: str, platform: Platform, text: str | None, video_url: str | None, metadata: dict)`, `AnalysisContext(topic: str, key_themes: list[str])`, `SentimentResult(label: str, score: float, indicators: list[str])`, `EmotionResult(primary: str, secondary: str, intensity: float, indicators: list[str])`, `MotivationResult(type: str, confidence_score: float, indicators: list[str])`, `AnalysisResult(id: str, status: Status, video_summary: str | None, context: AnalysisContext | None, sentiment: SentimentResult | None, emotion: EmotionResult | None, motivation: MotivationResult | None, error: str | None)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_schemas.py`:
```python
import pytest
from pydantic import ValidationError

from schemas import (
    AnalysisContext,
    AnalysisRequest,
    AnalysisResult,
    EmotionResult,
    MotivationResult,
    Platform,
    SentimentResult,
    Status,
)


def test_analysis_request_requires_id_and_platform_but_text_and_video_are_optional() -> None:
    request = AnalysisRequest(id="abc123", platform=Platform.TWITTER)

    assert request.id == "abc123"
    assert request.platform == Platform.TWITTER
    assert request.text is None
    assert request.video_url is None
    assert request.metadata == {}


def test_analysis_request_accepts_text_and_video_url() -> None:
    request = AnalysisRequest(
        id="abc123",
        platform=Platform.TIKTOK,
        text="hello world",
        video_url="https://tiktok.com/video/123",
        metadata={"source": "scrapper"},
    )

    assert request.text == "hello world"
    assert request.video_url == "https://tiktok.com/video/123"
    assert request.metadata == {"source": "scrapper"}


def test_analysis_request_rejects_missing_id() -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest(platform=Platform.TWITTER)  # type: ignore[call-arg]


def test_analysis_request_rejects_unknown_platform() -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest(id="abc123", platform="myspace")  # type: ignore[arg-type]


def test_analysis_result_ok_status_carries_full_analysis() -> None:
    result = AnalysisResult(
        id="abc123",
        status=Status.OK,
        video_summary="a person explains something",
        context=AnalysisContext(topic="politics", key_themes=["corruption"]),
        sentiment=SentimentResult(label="negative", score=-0.8, indicators=["marah"]),
        emotion=EmotionResult(primary="anger", secondary="disgust", intensity=0.9, indicators=["MAMPUS"]),
        motivation=MotivationResult(type="criticizing", confidence_score=0.7, indicators=["skandal"]),
    )

    assert result.status == Status.OK
    assert result.error is None


def test_analysis_result_failed_status_only_requires_id_and_error() -> None:
    result = AnalysisResult(id="abc123", status=Status.FAILED, error="insufficient input")

    assert result.status == Status.FAILED
    assert result.video_summary is None
    assert result.context is None
    assert result.sentiment is None
    assert result.error == "insufficient input"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'schemas'`.

- [ ] **Step 3: Implement `src/schemas.py`**

```python
from enum import StrEnum

from pydantic import BaseModel, Field


class Platform(StrEnum):
    TWITTER = "twitter"
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"


class Status(StrEnum):
    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"


class AnalysisRequest(BaseModel):
    id: str
    platform: Platform
    text: str | None = None
    video_url: str | None = None
    metadata: dict = Field(default_factory=dict)


class AnalysisContext(BaseModel):
    topic: str
    key_themes: list[str] = Field(default_factory=list)


class SentimentResult(BaseModel):
    label: str
    score: float
    indicators: list[str] = Field(default_factory=list)


class EmotionResult(BaseModel):
    primary: str
    secondary: str
    intensity: float
    indicators: list[str] = Field(default_factory=list)


class MotivationResult(BaseModel):
    type: str
    confidence_score: float
    indicators: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    id: str
    status: Status
    video_summary: str | None = None
    context: AnalysisContext | None = None
    sentiment: SentimentResult | None = None
    emotion: EmotionResult | None = None
    motivation: MotivationResult | None = None
    error: str | None = None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_schemas.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Run ruff and mypy**

Run: `ruff check . && mypy src`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/schemas.py tests/test_schemas.py
git commit -m "feat: add AnalysisRequest/AnalysisResult pydantic schemas"
```

---

## Task 3: Video Downloader

**Files:**
- Create: `src/video/downloader.py`
- Test: `tests/video/test_downloader.py`

**Interfaces:**
- Consumes: nothing project-specific (wraps the third-party `yt_dlp` package).
- Produces: `VideoDownloadError(Exception)`, `download_video(url: str, dest_dir: str) -> str` (returns the local file path of the downloaded video).

- [ ] **Step 1: Write the failing tests**

`tests/video/test_downloader.py`:
```python
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from video.downloader import VideoDownloadError, download_video


def test_download_video_returns_path_of_downloaded_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    written_file = tmp_path / "abc123.mp4"

    def fake_download(self: object, urls: list[str]) -> None:
        written_file.write_bytes(b"fake video bytes")

    def fake_prepare_filename(self: object, info: dict) -> str:
        return str(written_file)

    fake_ydl_instance = MagicMock()
    fake_ydl_instance.__enter__.return_value = fake_ydl_instance
    fake_ydl_instance.__exit__.return_value = False
    fake_ydl_instance.extract_info.return_value = {"id": "abc123", "ext": "mp4"}
    fake_ydl_instance.prepare_filename.return_value = str(written_file)

    fake_ydl_class = MagicMock(return_value=fake_ydl_instance)
    monkeypatch.setattr("video.downloader.YoutubeDL", fake_ydl_class)

    result_path = download_video("https://twitter.com/x/status/1", str(tmp_path))

    assert result_path == str(written_file)
    fake_ydl_instance.extract_info.assert_called_once_with(
        "https://twitter.com/x/status/1", download=True
    )


def test_download_video_wraps_extraction_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from yt_dlp.utils import DownloadError

    fake_ydl_instance = MagicMock()
    fake_ydl_instance.__enter__.return_value = fake_ydl_instance
    fake_ydl_instance.__exit__.return_value = False
    fake_ydl_instance.extract_info.side_effect = DownloadError("video unavailable")

    fake_ydl_class = MagicMock(return_value=fake_ydl_instance)
    monkeypatch.setattr("video.downloader.YoutubeDL", fake_ydl_class)

    with pytest.raises(VideoDownloadError, match="video unavailable"):
        download_video("https://twitter.com/x/status/1", str(tmp_path))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/video/test_downloader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'video.downloader'`.

- [ ] **Step 3: Implement `src/video/downloader.py`**

```python
import os

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError


class VideoDownloadError(Exception):
    pass


def download_video(url: str, dest_dir: str) -> str:
    """Downloads `url` into `dest_dir` using yt-dlp and returns the local file path."""
    options = {
        "outtmpl": os.path.join(dest_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
    }
    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)
    except DownloadError as e:
        raise VideoDownloadError(str(e)) from e
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/video/test_downloader.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run ruff and mypy**

Run: `ruff check . && mypy src`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/video/downloader.py tests/video/test_downloader.py
git commit -m "feat: add yt-dlp video downloader"
```

---

## Task 4: Video Analyzer (adapted from `docs/references/server.py`) — REWORKED

> **This task supersedes an earlier version already implemented and committed** (commit range `016452b..4f67d28` on this branch), which loaded `mlx-vlm`/`mlx-whisper` in-process. That version was built on the incorrect assumption that this service runs on the Mac Studio. It was clarified afterward that this service runs on a **local Windows machine**, and MLX cannot run there under any circumstance (Apple-Silicon-only, no Windows build, Docker cannot change the host's real CPU architecture). This reworked task keeps `VideoAnalysisError`, `VideoAnalysis`, `AnalyzerModels`, and `analyze_video()` **exactly as already implemented and approved** (do not change their code or tests) and replaces only `load_models()`'s implementation, plus adds a new `video/frames.py` module and amends `Settings`/`requirements.txt`.

**Files:**
- Modify: `requirements.txt` (remove the `mlx-vlm`/`mlx-whisper` lines, add `faster-whisper`)
- Modify: `src/config.py` (remove `vlm_model_id`, replace `whisper_model_id` with `whisper_model_size`)
- Modify: `tests/test_config.py` (update the two assertions/env references for the renamed/removed fields)
- Modify: `.env.example` (update the corresponding env var lines — see Step 8b)
- Create: `src/video/frames.py`
- Test: `tests/video/test_frames.py`
- Modify: `src/video/analyzer.py` (replace `load_models()`'s body only — leave `VideoAnalysisError`, `VideoAnalysis`, `AnalyzerModels`, `analyze_video()` untouched)
- Do NOT modify: `tests/video/test_analyzer.py` — its 4 existing tests are re-run (unchanged) in Step 12 to confirm `analyze_video()`'s behavior is unaffected by this rework.

**Interfaces:**
- Consumes: `VLMConfig`, `describe_images` from `ai/vlm_client.py` (Task 6, already built: `describe_images(config: VLMConfig, prompt: str, image_data_urls: list[str]) -> str`).
- Produces: `FrameExtractionError(Exception)`, `extract_frames(video_path: str, max_frames: int) -> list[str]` (both in `src/video/frames.py`); a new `load_models(vlm_config: VLMConfig, whisper_model_size: str) -> AnalyzerModels` in `src/video/analyzer.py`, replacing the old signature — `AnalyzerModels`, `analyze_video()`, `VideoAnalysis`, `VideoAnalysisError` keep their existing shapes unchanged from the already-approved implementation.

### Part A: `src/video/frames.py`

Extracts evenly-spaced JPEG frames from a video via `ffmpeg`/`ffprobe` (system binaries — must be on PATH; see Global Constraints), returned as base64 `data:` URLs ready for `ai/vlm_client.py`'s `describe_images()`.

- [ ] **Step 1: Write the failing tests**

`tests/video/test_frames.py`:
```python
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
            frames_dir = pattern.rsplit("\\", 1)[0] if "\\" in pattern else pattern.rsplit("/", 1)[0]
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


def test_extract_frames_raises_when_ffmpeg_produces_no_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("video.frames.subprocess.run", _fake_run_factory(0))

    with pytest.raises(FrameExtractionError, match="no frames"):
        extract_frames("/tmp/video.mp4", max_frames=5)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/video/test_frames.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'video.frames'`.

- [ ] **Step 3: Implement `src/video/frames.py`**

```python
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
    except (subprocess.CalledProcessError, OSError, KeyError, ValueError, json.JSONDecodeError) as e:
        raise FrameExtractionError(f"ffprobe duration probe failed: {e}") from e
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/video/test_frames.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run ruff and mypy**

Run: `ruff check . && mypy src`
Expected: no errors.

### Part B: amend `Settings` and `requirements.txt`

- [ ] **Step 6: Update `requirements.txt`**

Remove these two lines:
```
mlx-vlm>=0.1.0; sys_platform == "darwin"
mlx-whisper>=0.4.0; sys_platform == "darwin"
```
Add this line in their place:
```
faster-whisper>=1.0,<2
```

- [ ] **Step 7: Install the new dependency**

Run: `pip install -r requirements.txt` (using the existing `.venv`)

- [ ] **Step 8: Update `src/config.py`**

In the `Settings` class, remove the `vlm_model_id` field entirely, and replace:
```python
    whisper_model_id: str = "mlx-community/whisper-large-v3-turbo"
```
with:
```python
    whisper_model_size: str = "base"
```
Leave every other field unchanged (`vlm_base_url`, `vlm_model`, `max_frames`, `max_tokens`, etc. all stay).

- [ ] **Step 8b: Update `.env.example`**

Replace:
```
# In-process video analyzer (mlx-vlm / mlx-whisper model ids)
VLM_MODEL_ID=mlx-community/Qwen2.5-VL-7B-Instruct-4bit
WHISPER_MODEL_ID=mlx-community/whisper-large-v3-turbo
MAX_FRAMES=32
MAX_TOKENS=500
```
with:
```
# Video analyzer (local ffmpeg frame extraction + faster-whisper transcription;
# VLM calls for the summary go through VLM_BASE_URL/VLM_MODEL above)
WHISPER_MODEL_SIZE=base
MAX_FRAMES=32
MAX_TOKENS=500
```
Also update the comment above `VLM_BASE_URL`/`VLM_MODEL` (currently "not wired into the pipeline yet, see spec section 9") to: `# VLM (external, OpenAI-compatible) - used by video/analyzer.py for video summaries; endpoint mechanism on the Mac Studio still TBD by devops, see spec section 9`.

- [ ] **Step 9: Update `tests/test_config.py`**

In `test_settings_loads_required_and_applies_defaults`, replace:
```python
    assert settings.vlm_model_id == "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    assert settings.whisper_model_id == "mlx-community/whisper-large-v3-turbo"
```
with:
```python
    assert settings.whisper_model_size == "base"
```

- [ ] **Step 10: Run the config tests to verify they still pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 tests).

### Part C: replace `load_models()` in `src/video/analyzer.py`

Leave the existing `VideoAnalysisError`, `VideoAnalysis`, `AnalyzerModels` dataclasses, and `analyze_video()` function exactly as they are (already implemented, already tested, already approved). Replace only the `load_models()` function and its imports.

- [ ] **Step 11: Replace `load_models()`**

Remove the old `load_models()` function (the one that does `import mlx_vlm` / `import mlx_whisper`) and its now-unused imports at the top of the file, and replace with:

```python
def load_models(vlm_config: VLMConfig, whisper_model_size: str) -> AnalyzerModels:
    """
    Loads the local faster-whisper model once (CPU — this host has no GPU) and
    builds a generate_summary callable that extracts frames locally and sends
    them to a remote VLM. The whisper load is real and requires no special
    hardware, but is still not exercised in unit tests (real model weights,
    slow to load; see plan Global Constraints).
    """
    from faster_whisper import WhisperModel

    from ai.vlm_client import describe_images
    from video.frames import extract_frames

    whisper_model = WhisperModel(whisper_model_size, device="cpu")

    def generate_summary(video_path: str, prompt: str, max_frames: int, max_tokens: int) -> str:
        # max_tokens is part of AnalyzerModels.generate_summary's shared shape;
        # describe_images has no token-cap parameter, so it's unused here.
        frames = extract_frames(video_path, max_frames)
        return describe_images(vlm_config, prompt, frames)

    def transcribe(video_path: str) -> tuple[str, list[dict]]:
        segments_iter, _info = whisper_model.transcribe(video_path, word_timestamps=False)
        texts = []
        segments = []
        for s in segments_iter:
            text = s.text.strip()
            texts.append(text)
            segments.append({"start": s.start, "end": s.end, "text": text})
        return " ".join(texts).strip(), segments

    return AnalyzerModels(generate_summary=generate_summary, transcribe=transcribe)
```

Add this import near the top of the file, alongside the existing `from collections.abc import Callable` / `from dataclasses import dataclass` lines:
```python
from ai.vlm_client import VLMConfig
```
(This one is safe as a top-level import — `ai.vlm_client` has no platform-specific dependency, unlike the old `mlx_vlm`/`mlx_whisper` imports it replaces. Only `faster_whisper`, `ai.vlm_client.describe_images`, and `video.frames.extract_frames` need to stay as lazy imports inside `load_models()`, matching the file's existing pattern of keeping heavy/optional dependencies out of module-level imports.)

- [ ] **Step 12: Run the full test suite to verify nothing broke**

Run: `pytest -v`
Expected: PASS, same count as before this task plus the 4 new `test_frames.py` tests (the 4 existing `test_analyzer.py` tests must still pass unchanged, since `analyze_video()` itself was not touched).

- [ ] **Step 13: Run ruff and mypy**

Run: `ruff check . && mypy src`
Expected: no errors. (If mypy flags the `faster_whisper` import inside `load_models()` for missing stubs, add `# type: ignore[import-untyped]` to that line — `faster-whisper` is a real installed dependency, unlike the old `mlx_vlm`/`mlx_whisper` case, so confirm the exact mypy error code before picking the ignore code, the same way Task 3 and Task 4's original version each had to.)

- [ ] **Step 14: Commit**

```bash
git add requirements.txt src/config.py tests/test_config.py .env.example src/video/frames.py tests/video/test_frames.py src/video/analyzer.py
git commit -m "$(cat <<'EOF'
fix: rework video analyzer for a Windows host with a remote VLM

This service runs on a local Windows machine, not the Mac Studio, so
video/analyzer.py cannot load mlx-vlm/mlx-whisper in-process (MLX is
Apple-Silicon-only). load_models() now loads a local faster-whisper
model (CPU) for transcription, and its generate_summary callable
extracts frames locally via the new video/frames.py (ffmpeg) and sends
them to the VLM hosted on the Mac Studio over HTTP via the existing
ai/vlm_client.py. analyze_video()'s public behavior is unchanged.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: LLM Client

**Files:**
- Create: `src/ai/llm_client.py`
- Test: `tests/ai/test_llm_client.py`

**Interfaces:**
- Consumes: `AnalysisContext`, `SentimentResult`, `EmotionResult`, `MotivationResult` from `schemas.py` (Task 2).
- Produces: `LLMClientError(Exception)`, `LLMConfig` (dataclass: `base_url: str`, `model: str`, `timeout_seconds: float = 30.0`, `retry_attempts: int = 3`, `retry_backoff_seconds: float = 1.0`), `SentimentAnalysis` (dataclass: `context: AnalysisContext`, `sentiment: SentimentResult`, `emotion: EmotionResult`, `motivation: MotivationResult`), `summarize_video(config: LLMConfig, vlm_summary: str, transcript: str | None) -> str`, `analyze_sentiment(config: LLMConfig, content: str) -> SentimentAnalysis`.

- [ ] **Step 1: Write the failing tests**

`tests/ai/test_llm_client.py`:
```python
import httpx
import pytest
import respx

from ai.llm_client import LLMClientError, LLMConfig, analyze_sentiment, summarize_video


def _config(**overrides: object) -> LLMConfig:
    defaults = dict(
        base_url="http://localhost:1234/v1",
        model="test-model",
        timeout_seconds=5.0,
        retry_attempts=3,
        retry_backoff_seconds=0.0,
    )
    defaults.update(overrides)
    return LLMConfig(**defaults)  # type: ignore[arg-type]


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


@respx.mock
def test_summarize_video_sends_prompt_and_returns_text() -> None:
    route = respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response("combined summary"))
    )

    result = summarize_video(_config(), "a person is talking", "hello everyone")

    assert result == "combined summary"
    assert route.called
    sent_body = route.calls[0].request.content
    assert b"a person is talking" in sent_body
    assert b"hello everyone" in sent_body


@respx.mock
def test_summarize_video_retries_then_succeeds() -> None:
    route = respx.post("http://localhost:1234/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(200, json=_chat_response("combined summary")),
        ]
    )

    result = summarize_video(_config(), "a person is talking", None)

    assert result == "combined summary"
    assert route.call_count == 2


@respx.mock
def test_summarize_video_raises_after_exhausting_retries() -> None:
    route = respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(500)
    )

    with pytest.raises(LLMClientError):
        summarize_video(_config(retry_attempts=2), "a person is talking", None)

    assert route.call_count == 2


@respx.mock
def test_analyze_sentiment_parses_structured_json_response() -> None:
    payload = {
        "context": {"topic": "politics", "key_themes": ["korupsi"]},
        "sentiment": {"label": "negative", "score": -0.8, "indicators": ["marah"]},
        "emotion": {"primary": "anger", "secondary": "disgust", "intensity": 0.9, "indicators": ["MAMPUS"]},
        "motivation": {"type": "criticizing", "confidence_score": 0.7, "indicators": ["skandal"]},
    }
    import json

    respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response(json.dumps(payload)))
    )

    result = analyze_sentiment(_config(), "some social media content")

    assert result.context.topic == "politics"
    assert result.sentiment.label == "negative"
    assert result.emotion.primary == "anger"
    assert result.motivation.type == "criticizing"


@respx.mock
def test_analyze_sentiment_raises_llm_client_error_on_malformed_json() -> None:
    respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response("not json"))
    )

    with pytest.raises(LLMClientError):
        analyze_sentiment(_config(), "some social media content")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/ai/test_llm_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ai.llm_client'`.

- [ ] **Step 3: Implement `src/ai/llm_client.py`**

```python
import json
import time
from dataclasses import dataclass

import httpx
from pydantic import ValidationError

from schemas import AnalysisContext, EmotionResult, MotivationResult, SentimentResult


class LLMClientError(Exception):
    pass


@dataclass
class LLMConfig:
    base_url: str
    model: str
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_backoff_seconds: float = 1.0


@dataclass
class SentimentAnalysis:
    context: AnalysisContext
    sentiment: SentimentResult
    emotion: EmotionResult
    motivation: MotivationResult


SENTIMENT_SYSTEM_PROMPT = """You are an expert AI agent specialized in contextual sentiment \
analysis for Indonesian social media content. Given the content below, respond with ONLY a \
JSON object of this exact shape, no other text:
{
  "context": {"topic": "string", "key_themes": ["string"]},
  "sentiment": {"label": "positive|negative|neutral", "score": number, "indicators": ["string"]},
  "emotion": {"primary": "string", "secondary": "string", "intensity": number, "indicators": ["string"]},
  "motivation": {"type": "string", "confidence_score": number, "indicators": ["string"]}
}"""


def _chat_completion(config: LLMConfig, messages: list[dict]) -> str:
    last_error: Exception | None = None
    for attempt in range(1, config.retry_attempts + 1):
        try:
            response = httpx.post(
                f"{config.base_url}/chat/completions",
                json={"model": config.model, "messages": messages, "temperature": 0.0},
                timeout=config.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            return str(data["choices"][0]["message"]["content"])
        except (httpx.HTTPError, KeyError, IndexError) as e:
            last_error = e
            if attempt < config.retry_attempts:
                time.sleep(config.retry_backoff_seconds * attempt)
    raise LLMClientError(f"LLM request failed after {config.retry_attempts} attempts: {last_error}")


def summarize_video(config: LLMConfig, vlm_summary: str, transcript: str | None) -> str:
    parts = [f"Visual description: {vlm_summary}"]
    if transcript:
        parts.append(f"Spoken transcript: {transcript}")
    prompt = (
        "Combine the following visual description and spoken transcript of a short "
        "social media video into one coherent summary of what the video contains:\n\n"
        + "\n\n".join(parts)
    )
    return _chat_completion(config, [{"role": "user", "content": prompt}])


def analyze_sentiment(config: LLMConfig, content: str) -> SentimentAnalysis:
    raw = _chat_completion(
        config,
        [
            {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    try:
        data = json.loads(raw)
        return SentimentAnalysis(
            context=AnalysisContext(**data["context"]),
            sentiment=SentimentResult(**data["sentiment"]),
            emotion=EmotionResult(**data["emotion"]),
            motivation=MotivationResult(**data["motivation"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as e:
        raise LLMClientError(f"Failed to parse sentiment analysis response: {e}") from e
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/ai/test_llm_client.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Run ruff and mypy**

Run: `ruff check . && mypy src`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/ai/llm_client.py tests/ai/test_llm_client.py
git commit -m "feat: add LLM client for video summarization and sentiment analysis"
```

---

## Task 6: VLM Client

**Files:**
- Create: `src/ai/vlm_client.py`
- Test: `tests/ai/test_vlm_client.py`

**Interfaces:**
- Consumes: nothing project-specific.
- Produces: `VLMClientError(Exception)`, `VLMConfig` (dataclass: `base_url: str`, `model: str`, `timeout_seconds: float = 30.0`, `retry_attempts: int = 3`, `retry_backoff_seconds: float = 1.0`), `describe_images(config: VLMConfig, prompt: str, image_data_urls: list[str]) -> str`.

Per spec §4/§9, this client exists as a component (same OpenAI-compatible HTTP shape as `ai/llm_client.py`) but is **not** called from `pipeline/analyze.py` in v1 — its endpoint and exact role in sentiment analysis are still an open decision. Do not wire it into Task 7's pipeline.

- [ ] **Step 1: Write the failing tests**

`tests/ai/test_vlm_client.py`:
```python
import httpx
import pytest
import respx

from ai.vlm_client import VLMClientError, VLMConfig, describe_images


def _config(**overrides: object) -> VLMConfig:
    defaults = dict(
        base_url="http://localhost:8001/v1",
        model="test-vlm-model",
        timeout_seconds=5.0,
        retry_attempts=2,
        retry_backoff_seconds=0.0,
    )
    defaults.update(overrides)
    return VLMConfig(**defaults)  # type: ignore[arg-type]


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


@respx.mock
def test_describe_images_sends_prompt_and_images_and_returns_text() -> None:
    route = respx.post("http://localhost:8001/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response("a meme about politics"))
    )

    result = describe_images(_config(), "Describe these images.", ["data:image/png;base64,AAA"])

    assert result == "a meme about politics"
    sent_body = route.calls[0].request.content
    assert b"data:image/png;base64,AAA" in sent_body


@respx.mock
def test_describe_images_raises_after_exhausting_retries() -> None:
    respx.post("http://localhost:8001/v1/chat/completions").mock(return_value=httpx.Response(500))

    with pytest.raises(VLMClientError):
        describe_images(_config(), "Describe these images.", ["data:image/png;base64,AAA"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/ai/test_vlm_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ai.vlm_client'`.

- [ ] **Step 3: Implement `src/ai/vlm_client.py`**

```python
import time
from dataclasses import dataclass

import httpx


class VLMClientError(Exception):
    pass


@dataclass
class VLMConfig:
    base_url: str
    model: str
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_backoff_seconds: float = 1.0


def describe_images(config: VLMConfig, prompt: str, image_data_urls: list[str]) -> str:
    """
    Calls an OpenAI-compatible vision chat completion endpoint with one or more
    images (as data: URLs) and a text prompt, returning the model's text response.
    Not currently wired into pipeline.analyze() -- see spec section 9.
    """
    content: list[dict] = [{"type": "text", "text": prompt}]
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    messages = [{"role": "user", "content": content}]

    last_error: Exception | None = None
    for attempt in range(1, config.retry_attempts + 1):
        try:
            response = httpx.post(
                f"{config.base_url}/chat/completions",
                json={"model": config.model, "messages": messages, "temperature": 0.0},
                timeout=config.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            return str(data["choices"][0]["message"]["content"])
        except (httpx.HTTPError, KeyError, IndexError) as e:
            last_error = e
            if attempt < config.retry_attempts:
                time.sleep(config.retry_backoff_seconds * attempt)
    raise VLMClientError(f"VLM request failed after {config.retry_attempts} attempts: {last_error}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/ai/test_vlm_client.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run ruff and mypy**

Run: `ruff check . && mypy src`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/ai/vlm_client.py tests/ai/test_vlm_client.py
git commit -m "feat: add standalone VLM client (not yet wired into the pipeline)"
```

---

## Task 7: Pipeline Orchestrator

**Files:**
- Create: `src/pipeline/analyze.py`
- Test: `tests/pipeline/test_analyze.py`

**Interfaces:**
- Consumes: `AnalysisRequest`, `AnalysisResult`, `Status`, `AnalysisContext`, `SentimentResult`, `EmotionResult`, `MotivationResult` from `schemas.py` (Task 2); `VideoAnalysis`, `VideoAnalysisError` type from `video/analyzer.py` (Task 4, referenced only via the injected callable's return type); `VideoDownloadError` from `video/downloader.py` (Task 3); `SentimentAnalysis`, `LLMClientError` from `ai/llm_client.py` (Task 5).
- Produces: `AnalyzeDependencies` (dataclass: `download_video: Callable[[str, str], str]`, `analyze_video: Callable[[str, str, int, int, bool], VideoAnalysis]`, `summarize_video: Callable[[str, str | None], str]`, `analyze_sentiment: Callable[[str], SentimentAnalysis]`, `video_prompt: str = "Describe what is happening in this video."`, `max_frames: int = 32`, `max_tokens: int = 500`), `analyze(request: AnalysisRequest, deps: AnalyzeDependencies) -> AnalysisResult`.

- [ ] **Step 1: Write the failing tests**

`tests/pipeline/test_analyze.py`:
```python
import tempfile

import pytest

from ai.llm_client import LLMClientError, SentimentAnalysis
from pipeline.analyze import AnalyzeDependencies, analyze
from schemas import (
    AnalysisContext,
    AnalysisRequest,
    EmotionResult,
    MotivationResult,
    Platform,
    SentimentResult,
    Status,
)
from video.analyzer import VideoAnalysis, VideoAnalysisError
from video.downloader import VideoDownloadError


def _sentiment_analysis() -> SentimentAnalysis:
    return SentimentAnalysis(
        context=AnalysisContext(topic="politics", key_themes=["korupsi"]),
        sentiment=SentimentResult(label="negative", score=-0.8, indicators=["marah"]),
        emotion=EmotionResult(primary="anger", secondary="disgust", intensity=0.9, indicators=["MAMPUS"]),
        motivation=MotivationResult(type="criticizing", confidence_score=0.7, indicators=["skandal"]),
    )


def _deps(**overrides: object) -> AnalyzeDependencies:
    defaults: dict = dict(
        download_video=lambda url, dest_dir: "/tmp/video.mp4",
        analyze_video=lambda path, prompt, max_frames, max_tokens, include_transcript: VideoAnalysis(
            summary="a person talking", transcript="hello everyone", transcript_segments=[]
        ),
        summarize_video=lambda vlm_summary, transcript: "combined video summary",
        analyze_sentiment=lambda content: _sentiment_analysis(),
    )
    defaults.update(overrides)
    return AnalyzeDependencies(**defaults)  # type: ignore[arg-type]


def test_analyze_text_and_video_produces_ok_status() -> None:
    request = AnalysisRequest(id="1", platform=Platform.TWITTER, text="teks asli", video_url="https://x.com/1")

    result = analyze(request, _deps())

    assert result.id == "1"
    assert result.status == Status.OK
    assert result.video_summary == "combined video summary"
    assert result.sentiment is not None and result.sentiment.label == "negative"
    assert result.error is None


def test_analyze_video_only_produces_ok_status() -> None:
    request = AnalysisRequest(id="2", platform=Platform.TIKTOK, video_url="https://tiktok.com/2")

    result = analyze(request, _deps())

    assert result.status == Status.OK
    assert result.video_summary == "combined video summary"


def test_analyze_text_only_skips_video_steps() -> None:
    calls: list[str] = []
    request = AnalysisRequest(id="3", platform=Platform.FACEBOOK, text="teks asli")

    result = analyze(
        request,
        _deps(download_video=lambda url, dest_dir: calls.append("download") or "/tmp/video.mp4"),
    )

    assert result.status == Status.OK
    assert result.video_summary is None
    assert calls == []


def test_analyze_both_missing_returns_failed_status() -> None:
    request = AnalysisRequest(id="4", platform=Platform.TWITTER)

    result = analyze(request, _deps())

    assert result.status == Status.FAILED
    assert result.error is not None and "insufficient input" in result.error


def test_analyze_video_download_failure_degrades_to_text_only() -> None:
    def broken_download(url: str, dest_dir: str) -> str:
        raise VideoDownloadError("private video")

    request = AnalysisRequest(id="5", platform=Platform.INSTAGRAM, text="teks asli", video_url="https://instagram.com/5")

    result = analyze(request, _deps(download_video=broken_download))

    assert result.status == Status.PARTIAL
    assert result.video_summary is None
    assert result.error is not None and "video processing failed" in result.error
    assert result.sentiment is not None


def test_analyze_video_analyzer_failure_degrades_to_text_only() -> None:
    def broken_analyze_video(path, prompt, max_frames, max_tokens, include_transcript):
        raise VideoAnalysisError("out of memory")

    request = AnalysisRequest(id="6", platform=Platform.INSTAGRAM, text="teks asli", video_url="https://instagram.com/6")

    result = analyze(request, _deps(analyze_video=broken_analyze_video))

    assert result.status == Status.PARTIAL
    assert result.video_summary is None
    assert result.sentiment is not None


def test_analyze_video_only_download_failure_with_no_text_returns_failed() -> None:
    def broken_download(url: str, dest_dir: str) -> str:
        raise VideoDownloadError("private video")

    request = AnalysisRequest(id="7", platform=Platform.INSTAGRAM, video_url="https://instagram.com/7")

    result = analyze(request, _deps(download_video=broken_download))

    assert result.status == Status.FAILED
    assert result.error is not None and "insufficient input" in result.error


def test_analyze_sentiment_retry_exhausted_returns_failed_status() -> None:
    def broken_analyze_sentiment(content: str) -> SentimentAnalysis:
        raise LLMClientError("LLM unreachable")

    request = AnalysisRequest(id="8", platform=Platform.TWITTER, text="teks asli")

    result = analyze(request, _deps(analyze_sentiment=broken_analyze_sentiment))

    assert result.status == Status.FAILED
    assert result.error is not None and "sentiment analysis failed" in result.error
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/pipeline/test_analyze.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.analyze'`.

- [ ] **Step 3: Implement `src/pipeline/analyze.py`**

```python
import tempfile
from collections.abc import Callable
from dataclasses import dataclass

from ai.llm_client import LLMClientError, SentimentAnalysis
from schemas import AnalysisRequest, AnalysisResult, Status
from video.analyzer import VideoAnalysis, VideoAnalysisError
from video.downloader import VideoDownloadError


@dataclass
class AnalyzeDependencies:
    download_video: Callable[[str, str], str]
    analyze_video: Callable[[str, str, int, int, bool], VideoAnalysis]
    summarize_video: Callable[[str, str | None], str]
    analyze_sentiment: Callable[[str], SentimentAnalysis]
    video_prompt: str = "Describe what is happening in this video."
    max_frames: int = 32
    max_tokens: int = 500


def analyze(request: AnalysisRequest, deps: AnalyzeDependencies) -> AnalysisResult:
    video_summary: str | None = None
    error_parts: list[str] = []

    if request.video_url:
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                video_path = deps.download_video(request.video_url, tmp_dir)
                analysis = deps.analyze_video(
                    video_path, deps.video_prompt, deps.max_frames, deps.max_tokens, True
                )
            video_summary = deps.summarize_video(analysis.summary, analysis.transcript)
        except (VideoDownloadError, VideoAnalysisError, LLMClientError) as e:
            error_parts.append(f"video processing failed: {e}")

    context_parts = []
    if request.text:
        context_parts.append(request.text)
    if video_summary:
        context_parts.append(video_summary)

    if not context_parts:
        return AnalysisResult(
            id=request.id,
            status=Status.FAILED,
            video_summary=video_summary,
            error="insufficient input: no text or video content available"
            + ("; " + "; ".join(error_parts) if error_parts else ""),
        )

    combined_context = "\n\n".join(context_parts)
    try:
        sentiment_analysis = deps.analyze_sentiment(combined_context)
    except LLMClientError as e:
        return AnalysisResult(
            id=request.id,
            status=Status.FAILED,
            video_summary=video_summary,
            error=f"sentiment analysis failed: {e}"
            + ("; " + "; ".join(error_parts) if error_parts else ""),
        )

    status = Status.PARTIAL if error_parts else Status.OK
    return AnalysisResult(
        id=request.id,
        status=status,
        video_summary=video_summary,
        context=sentiment_analysis.context,
        sentiment=sentiment_analysis.sentiment,
        emotion=sentiment_analysis.emotion,
        motivation=sentiment_analysis.motivation,
        error="; ".join(error_parts) if error_parts else None,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/pipeline/test_analyze.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Run ruff and mypy**

Run: `ruff check . && mypy src`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/analyze.py tests/pipeline/test_analyze.py
git commit -m "feat: add pipeline orchestrator with graceful video degrade"
```

---

## Task 8: Kafka Producer

**Files:**
- Create: `src/messaging/producer.py`
- Test: `tests/messaging/test_producer.py`

**Interfaces:**
- Consumes: `AnalysisResult` from `schemas.py` (Task 2).
- Produces: `PublishError(Exception)`, `KafkaResultProducer` (class: `__init__(self, bootstrap_servers: str, topic: str)`, `publish(self, result: AnalysisResult) -> None`).

- [ ] **Step 1: Write the failing tests**

`tests/messaging/test_producer.py`:
```python
from unittest.mock import MagicMock

import pytest

from messaging.producer import KafkaResultProducer, PublishError
from schemas import AnalysisResult, Status


def _result() -> AnalysisResult:
    return AnalysisResult(id="abc123", status=Status.OK, video_summary="a summary")


def test_publish_produces_message_and_flushes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_producer_instance = MagicMock()

    def fake_produce(topic, key, value, callback):
        callback(None, MagicMock())

    fake_producer_instance.produce.side_effect = fake_produce
    fake_producer_class = MagicMock(return_value=fake_producer_instance)
    monkeypatch.setattr("messaging.producer.Producer", fake_producer_class)

    producer = KafkaResultProducer("localhost:9092", "analytics.results")
    producer.publish(_result())

    fake_producer_class.assert_called_once_with({"bootstrap.servers": "localhost:9092"})
    args, kwargs = fake_producer_instance.produce.call_args
    assert args[0] == "analytics.results" or kwargs.get("topic") == "analytics.results"
    assert kwargs["key"] == b"abc123"
    assert b"abc123" in kwargs["value"]
    fake_producer_instance.flush.assert_called_once()


def test_publish_raises_publish_error_on_delivery_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_producer_instance = MagicMock()

    def fake_produce(topic, key, value, callback):
        callback(Exception("broker unavailable"), None)

    fake_producer_instance.produce.side_effect = fake_produce
    fake_producer_class = MagicMock(return_value=fake_producer_instance)
    monkeypatch.setattr("messaging.producer.Producer", fake_producer_class)

    producer = KafkaResultProducer("localhost:9092", "analytics.results")

    with pytest.raises(PublishError, match="broker unavailable"):
        producer.publish(_result())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/messaging/test_producer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'messaging.producer'`.

- [ ] **Step 3: Implement `src/messaging/producer.py`**

```python
from confluent_kafka import Producer

from schemas import AnalysisResult


class PublishError(Exception):
    pass


class KafkaResultProducer:
    def __init__(self, bootstrap_servers: str, topic: str) -> None:
        self._topic = topic
        self._producer = Producer({"bootstrap.servers": bootstrap_servers})

    def publish(self, result: AnalysisResult) -> None:
        errors: list[Exception] = []

        def _on_delivery(err: object, _msg: object) -> None:
            if err is not None:
                errors.append(Exception(str(err)))

        self._producer.produce(
            self._topic,
            key=result.id.encode("utf-8"),
            value=result.model_dump_json().encode("utf-8"),
            callback=_on_delivery,
        )
        self._producer.flush(10)

        if errors:
            raise PublishError(str(errors[0]))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/messaging/test_producer.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run ruff and mypy**

Run: `ruff check . && mypy src`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/messaging/producer.py tests/messaging/test_producer.py
git commit -m "feat: add Kafka result producer"
```

---

## Task 9: Kafka Consumer

**Files:**
- Create: `src/messaging/consumer.py`
- Test: `tests/messaging/test_consumer.py`

**Interfaces:**
- Consumes: `AnalysisRequest` from `schemas.py` (Task 2).
- Produces: `KafkaRequestConsumer` (class: `__init__(self, bootstrap_servers: str, topic: str, group_id: str)`, `poll_request(self, timeout: float) -> tuple[AnalysisRequest, object] | None`, `commit(self, msg: object) -> None`).

- [ ] **Step 1: Write the failing tests**

`tests/messaging/test_consumer.py`:
```python
import json
from unittest.mock import MagicMock

import pytest

from messaging.consumer import KafkaRequestConsumer


def _make_consumer(monkeypatch: pytest.MonkeyPatch) -> tuple[KafkaRequestConsumer, MagicMock]:
    fake_consumer_instance = MagicMock()
    fake_consumer_class = MagicMock(return_value=fake_consumer_instance)
    monkeypatch.setattr("messaging.consumer.Consumer", fake_consumer_class)

    consumer = KafkaRequestConsumer("localhost:9092", "analytics.requests", "analytics-backend")
    return consumer, fake_consumer_instance


def test_poll_request_returns_none_when_nothing_polled(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, fake_consumer_instance = _make_consumer(monkeypatch)
    fake_consumer_instance.poll.return_value = None

    result = consumer.poll_request(1.0)

    assert result is None


def test_poll_request_parses_valid_message(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, fake_consumer_instance = _make_consumer(monkeypatch)

    fake_msg = MagicMock()
    fake_msg.error.return_value = None
    fake_msg.value.return_value = json.dumps({"id": "1", "platform": "twitter"}).encode("utf-8")
    fake_consumer_instance.poll.return_value = fake_msg

    result = consumer.poll_request(1.0)

    assert result is not None
    request, msg = result
    assert request.id == "1"
    assert msg is fake_msg
    fake_consumer_instance.commit.assert_not_called()


def test_poll_request_skips_and_commits_malformed_message(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, fake_consumer_instance = _make_consumer(monkeypatch)

    fake_msg = MagicMock()
    fake_msg.error.return_value = None
    fake_msg.value.return_value = b"not json"
    fake_consumer_instance.poll.return_value = fake_msg

    result = consumer.poll_request(1.0)

    assert result is None
    fake_consumer_instance.commit.assert_called_once_with(fake_msg)


def test_poll_request_returns_none_on_kafka_level_error(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, fake_consumer_instance = _make_consumer(monkeypatch)

    fake_msg = MagicMock()
    fake_msg.error.return_value = "partition EOF"
    fake_consumer_instance.poll.return_value = fake_msg

    result = consumer.poll_request(1.0)

    assert result is None
    fake_consumer_instance.commit.assert_not_called()


def test_commit_delegates_to_underlying_consumer(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, fake_consumer_instance = _make_consumer(monkeypatch)
    fake_msg = MagicMock()

    consumer.commit(fake_msg)

    fake_consumer_instance.commit.assert_called_once_with(fake_msg)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/messaging/test_consumer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'messaging.consumer'`.

- [ ] **Step 3: Implement `src/messaging/consumer.py`**

```python
import json

from confluent_kafka import Consumer
from pydantic import ValidationError

from schemas import AnalysisRequest


class KafkaRequestConsumer:
    def __init__(self, bootstrap_servers: str, topic: str, group_id: str) -> None:
        self._consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )
        self._consumer.subscribe([topic])

    def poll_request(self, timeout: float) -> tuple[AnalysisRequest, object] | None:
        msg = self._consumer.poll(timeout)
        if msg is None:
            return None
        if msg.error():
            return None
        try:
            data = json.loads(msg.value().decode("utf-8"))
            request = AnalysisRequest(**data)
        except (json.JSONDecodeError, ValidationError, UnicodeDecodeError, TypeError):
            self._consumer.commit(msg)
            return None
        return request, msg

    def commit(self, msg: object) -> None:
        self._consumer.commit(msg)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/messaging/test_consumer.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Run ruff and mypy**

Run: `ruff check . && mypy src`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/messaging/consumer.py tests/messaging/test_consumer.py
git commit -m "feat: add Kafka request consumer"
```

---

## Task 10: Main Entrypoint

**Files:**
- Create: `src/main.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `Settings` from `config.py` (Task 1); `AnalyzeDependencies`, `analyze` from `pipeline/analyze.py` (Task 7); `download_video` from `video/downloader.py` (Task 3); `load_models`, `analyze_video` from `video/analyzer.py` (Task 4, reworked); `VLMConfig` from `ai/vlm_client.py` (Task 6); `LLMConfig`, `summarize_video`, `analyze_sentiment` from `ai/llm_client.py` (Task 5); `KafkaRequestConsumer` from `messaging/consumer.py` (Task 9); `KafkaResultProducer` from `messaging/producer.py` (Task 8).
- Produces: `build_dependencies(settings: Settings) -> AnalyzeDependencies`, `run_once(consumer: KafkaRequestConsumer, producer: KafkaResultProducer, deps: AnalyzeDependencies, poll_timeout: float = 1.0) -> bool`, `main() -> None`.

`build_dependencies()` calls `load_models()`, which loads a real local `faster-whisper` model — not exercised in unit tests (real model weights, slow; Global Constraints), but requires no special hardware. `run_once()` contains all the testable orchestration logic and is fully covered here with fake consumer/producer/deps.

- [ ] **Step 1: Write the failing tests**

`tests/test_main.py`:
```python
from unittest.mock import MagicMock

from ai.llm_client import SentimentAnalysis
from main import run_once
from pipeline.analyze import AnalyzeDependencies
from schemas import AnalysisContext, AnalysisRequest, EmotionResult, MotivationResult, Platform, SentimentResult
from video.analyzer import VideoAnalysis


def _deps() -> AnalyzeDependencies:
    return AnalyzeDependencies(
        download_video=lambda url, dest_dir: "/tmp/video.mp4",
        analyze_video=lambda path, prompt, max_frames, max_tokens, include_transcript: VideoAnalysis(
            summary="s", transcript="t", transcript_segments=[]
        ),
        summarize_video=lambda vlm_summary, transcript: "video summary",
        analyze_sentiment=lambda content: SentimentAnalysis(
            context=AnalysisContext(topic="x", key_themes=[]),
            sentiment=SentimentResult(label="neutral", score=0.0, indicators=[]),
            emotion=EmotionResult(primary="calm", secondary="calm", intensity=0.1, indicators=[]),
            motivation=MotivationResult(type="informative", confidence_score=0.5, indicators=[]),
        ),
    )


def test_run_once_returns_false_and_does_nothing_when_no_message_polled() -> None:
    consumer = MagicMock()
    consumer.poll_request.return_value = None
    producer = MagicMock()

    processed = run_once(consumer, producer, _deps())

    assert processed is False
    producer.publish.assert_not_called()
    consumer.commit.assert_not_called()


def test_run_once_analyzes_publishes_and_commits_when_message_polled() -> None:
    request = AnalysisRequest(id="1", platform=Platform.TWITTER, text="teks asli")
    fake_msg = object()
    consumer = MagicMock()
    consumer.poll_request.return_value = (request, fake_msg)
    producer = MagicMock()

    processed = run_once(consumer, producer, _deps())

    assert processed is True
    producer.publish.assert_called_once()
    published_result = producer.publish.call_args[0][0]
    assert published_result.id == "1"
    consumer.commit.assert_called_once_with(fake_msg)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_main.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'main'`.

- [ ] **Step 3: Implement `src/main.py`**

```python
from functools import partial

from ai.llm_client import LLMConfig, analyze_sentiment, summarize_video
from ai.vlm_client import VLMConfig
from config import Settings
from messaging.consumer import KafkaRequestConsumer
from messaging.producer import KafkaResultProducer
from pipeline.analyze import AnalyzeDependencies, analyze
from video.analyzer import analyze_video, load_models
from video.downloader import download_video


def build_dependencies(settings: Settings) -> AnalyzeDependencies:
    """
    Loads the local faster-whisper model once. Requires no special hardware
    (CPU-only) but is not exercised in unit tests (real model weights, slow to
    load; see plan Global Constraints).
    """
    vlm_config = VLMConfig(base_url=settings.vlm_base_url or "", model=settings.vlm_model or "")
    analyzer_models = load_models(vlm_config, settings.whisper_model_size)
    llm_config = LLMConfig(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout_seconds=settings.http_timeout_seconds,
        retry_attempts=settings.retry_attempts,
        retry_backoff_seconds=settings.retry_backoff_seconds,
    )
    return AnalyzeDependencies(
        download_video=download_video,
        analyze_video=partial(analyze_video, analyzer_models),
        summarize_video=partial(summarize_video, llm_config),
        analyze_sentiment=partial(analyze_sentiment, llm_config),
        max_frames=settings.max_frames,
        max_tokens=settings.max_tokens,
    )


def run_once(
    consumer: KafkaRequestConsumer,
    producer: KafkaResultProducer,
    deps: AnalyzeDependencies,
    poll_timeout: float = 1.0,
) -> bool:
    polled = consumer.poll_request(poll_timeout)
    if polled is None:
        return False
    request, msg = polled
    result = analyze(request, deps)
    producer.publish(result)
    consumer.commit(msg)
    return True


def main() -> None:
    settings = Settings()
    deps = build_dependencies(settings)
    consumer = KafkaRequestConsumer(
        settings.kafka_bootstrap_servers, settings.kafka_request_topic, settings.kafka_consumer_group
    )
    producer = KafkaResultProducer(settings.kafka_bootstrap_servers, settings.kafka_response_topic)
    while True:
        run_once(consumer, producer, deps)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_main.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full test suite, ruff, and mypy**

Run: `pytest -v && ruff check . && mypy src`
Expected: all tests pass (33 tests across all tasks), no lint or type errors.

- [ ] **Step 6: Commit**

```bash
git add src/main.py tests/test_main.py
git commit -m "feat: wire Kafka consumer/producer to the pipeline in main entrypoint"
```

---

## Post-Plan Notes (not implementation tasks)

- **This service runs on the local Windows machine, not the Mac Studio** (revised after Task 5 — see Global Constraints and Task 4's rework note). The Mac Studio hosts LM Studio (LLM) and, once devops decides how, the VLM — both reached over the network from this service.
- Running locally for development: `python -m venv .venv && .venv\Scripts\pip install -r requirements.txt && .venv\Scripts\python src\main.py` (Windows), with a real `.env` (copy from `.env.example`) loaded into the process environment — `Settings` reads from `os.environ`, it does not load `.env` itself. `ffmpeg`/`ffprobe` must be installed and on PATH (not a pip package).
- **Docker packaging** was deliberately left out of this plan's task list (no Task creates a Dockerfile) even though the spec now calls for it (spec §8, revised) — this plan focused on getting the service correct and tested first. Writing the Dockerfile (Python base image + `apt-get install ffmpeg`, copy `src/`, install `requirements.txt`, `CMD ["python", "src/main.py"]`) is a small, independent follow-up once Task 10 is complete and verified; it doesn't require its own spec/plan cycle.
- The remaining open items flagged in spec §9 (the VLM's exact hosting mechanism on the Mac Studio, and the exact Kafka topic contract) are intentionally not resolved by this plan — they need a follow-up decision once devops and the Scrapper Backend developer confirm their sides. If the VLM turns out not to be OpenAI-compatible, `ai/vlm_client.py` (and therefore `video/analyzer.py`'s `load_models()`) will need rework, isolated to those two files.
