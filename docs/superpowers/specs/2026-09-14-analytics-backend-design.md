# Analytics Backend — Design Spec

Date: 2026-09-14
Status: Approved by user (Kacang), ready for implementation planning

## 1. Purpose

`analyzer_sentiment` implements the **Analytics Backend** in the broader
scraping/sentiment system (see `docs/references/system-flow.mmd` and
`docs/to-do.md`). It is a standalone Python service, independent from the
Scrapper Backend (built by another developer) and the Sentiment Backend.

Per item, it must:

1. Receive a normalized item (plain text and/or a video URL) from the
   Scrapper Backend via Kafka.
2. If a video URL is present, download the video (max ~10 minutes long)
   and run it through a video analyzer (adapted from
   `docs/references/server.py`'s approach, handed off by the supervisor —
   see §4) to get both a VLM-generated summary and a raw speech
   transcript.
3. Combine that video output with our own LLM into a final
   `video_summary` text.
4. Combine available text and `video_summary` into one context.
5. Run sentiment/emotion/motivation analysis on that context using an
   LLM (and the VLM if visual cues are analyzed directly as part of this
   step).
6. Publish the result back to the Scrapper Backend via Kafka, one item
   at a time — mirroring how items arrive.

`docs/references/Analyst_Batches.json` (an n8n workflow) is **reference
only** — useful for understanding a prior text-only sentiment prompt/schema
attempt, not part of the target architecture.

## 2. Scope / Non-goals

**In scope (v1):**
- Kafka consumer/producer for the Analytics Backend's own request/response
  topics.
- Video download (TikTok, Instagram, Facebook, Twitter/X — max ~10
  minutes per video, confirmed by the user).
- A video analyzer that adapts `docs/references/server.py`'s *approach*
  (sample frames, get a whole-clip VLM description, optionally
  transcribe audio) using components that run on this service's actual
  host — a local Windows machine, not the Mac Studio (see §3, §4):
  local `ffmpeg` frame extraction, a network call to a VLM hosted on the
  Mac Studio, and local CPU speech transcription via `faster-whisper`.
- Turning that video analyzer's output (VLM summary + raw transcript)
  into a final `video_summary` via our own LLM.
- Sentiment/emotion/motivation analysis via an LLM (and VLM where
  relevant), called over an OpenAI-compatible HTTP API (model-agnostic —
  see §4).
- Graceful handling of items missing text or missing video.

**Out of scope (v1):**
- Building or training a video understanding/summarization model, or
  hosting/choosing the VLM itself — that lives on the Mac Studio, owned
  by devops. We only adapt `server.py`'s frame-sampling-then-describe
  approach into components that run where this service actually runs.
- Kafka topic names/message contract finalization with the Scrapper
  Backend developer (not yet coordinated — see §9 Open Assumptions).
- Dead-letter topics / advanced retry orchestration beyond bounded
  in-process retries.
- End-to-end integration tests against a real Kafka cluster or the real
  remote VLM endpoint.

## 3. Architecture

Single Python service, single Kafka consumer group, **synchronous
per-message processing**: each message is fully processed (video
download → video analysis → video summarization → text+video sentiment
analysis → publish result) before the next message is picked up, and the
offset is committed only after a successful publish.

This was chosen over two alternatives:
- **Async worker pool in-process** — rejected for v1: added concurrency
  complexity, but the real bottleneck is shared model inference (the
  remote VLM, and the LLM endpoint), not the orchestration code, so
  in-process concurrency wouldn't reliably raise throughput.
- **Separate gateway + task-queue workers (e.g. Celery/RQ + Redis)** —
  rejected for v1: introduces a second broker alongside Kafka with no
  demonstrated need yet.

If throughput becomes a bottleneck later, scale via standard Kafka
mechanisms: more partitions + more consumer replicas. Each replica loads
its own local `faster-whisper` model at startup (modest CPU/RAM cost,
unlike the earlier in-process-VLM design this spec previously
considered) and makes outbound HTTP calls for the VLM and LLM — the
service itself is otherwise stateless, so replica scaling is
straightforward.

**Revision history note:** this service's video-analysis approach has
changed twice during design. It was originally going to host/choose its
own open-source video summarization model; then it was going to run the
supervisor's `mlx-vlm`/`mlx-whisper` code in-process, assuming this
service would run on the same Mac Studio as the models. It was
clarified that this service actually runs on a **local Windows
machine**, separate from the Mac Studio — and `mlx-vlm`/`mlx-whisper`
depend on MLX, which is Apple-Silicon-only with no Windows build
(Docker cannot change this — a container still runs on the host's real
CPU architecture). §4 below reflects the current, final approach.

## 4. Components

```
src/
  main.py              # entrypoint: load the local whisper model once, then start Kafka consumer loop
  config.py            # env-based settings (Kafka, LLM base url+model, VLM base url+model, whisper model size, frame/timeout/retry settings)
  schemas.py           # pydantic models: AnalysisRequest, AnalysisResult, internal DTOs
  messaging/
    consumer.py        # Kafka consume wrapper (infra only, no business logic)
    producer.py         # Kafka publish wrapper
  pipeline/
    analyze.py          # use-case orchestrator: optional text/video -> summary -> sentiment result
  video/
    downloader.py       # yt-dlp wrapper (TikTok/IG/FB/X)
    frames.py            # ffmpeg wrapper: extract N evenly-spaced frames from a video file as base64-encoded images
    analyzer.py          # adapted from docs/references/server.py's approach: frame extraction + remote VLM call (via ai/vlm_client.py) for the summary, local faster-whisper for the transcript
  ai/
    vlm_client.py        # thin HTTP client, OpenAI-compatible endpoint, used by video/analyzer.py for whole-video summarization AND available for direct visual sentiment reasoning in step 5 if needed
    llm_client.py         # thin HTTP client, OpenAI-compatible endpoint (LM Studio), video summarization + sentiment/emotion/motivation
```

Business logic (`pipeline/`, `video/`, `ai/`) is kept independent of the
Kafka transport (`messaging/`) — `main.py` wires them together. Each
module has one clear responsibility, consistent with Acme's structure
standards (business logic out of handlers, one responsibility per file).

### `video/analyzer.py` and `video/frames.py` (adapted from `docs/references/server.py`)

`docs/references/server.py` is a FastAPI server, meant for macOS/Apple
Silicon, that loads `mlx-vlm` and `mlx-whisper` once at startup and, per
request, runs a **single whole-video mlx-vlm call** (multi-frame,
temporally-aware — not a frame-by-frame loop) to produce `summary`, plus
optional `mlx-whisper` transcription to produce `transcript` +
`transcript_segments`. Since this service runs on a Windows machine, we
adapt the *approach*, not the code, using pieces that actually run
there:

- `video/frames.py` extracts a fixed number of evenly-spaced frames from
  the downloaded video via `ffmpeg` (a system binary, cross-platform —
  must be present on PATH; see §9), returning them as base64-encoded
  image data URLs. This replaces `mlx-vlm`'s internal frame sampling.
- `video/analyzer.py`'s summary step sends those frames plus a prompt to
  `ai/vlm_client.py`'s `describe_images()` (already built for exactly
  this OpenAI-compatible vision-chat shape) — a network call to whatever
  VLM devops ends up hosting on the Mac Studio (see §9). This replaces
  `mlx_vlm.generate()`.
- `video/analyzer.py`'s transcript step runs `faster-whisper` **locally,
  on this Windows machine's CPU** (no GPU available here — see §9). This
  replaces `mlx_whisper.transcribe()`. The whisper model is loaded once
  at `main.py` startup, mirroring `server.py`'s `lifespan` pattern.
- The public shape of this module — `VideoAnalysisError`,
  `VideoAnalysis`, `AnalyzerModels`, `analyze_video()` — is unchanged
  from the original in-process design: `analyze_video()` still takes a
  video file path and returns a summary + optional transcript, so
  `pipeline/analyze.py` (§5) does not need to know or care that the
  summary now comes from a network call and the transcript from a local
  CPU model rather than both being in-process MLX calls.
- We call it with `include_transcript=True` always (see §5) — the raw
  transcript is needed downstream for exact wording, not just the VLM's
  descriptive summary.

### Model access (LLM, and VLM)

The LLM used for video summarization (step 3) and sentiment/emotion/
motivation analysis (step 5) is accessed as an **OpenAI-compatible HTTP
API** — already decided: **LM Studio**, running on the Mac Studio.

The VLM (used by `video/analyzer.py` for the whole-video summary, and
optionally by step 5 for direct visual sentiment reasoning) is also
assumed OpenAI-compatible over HTTP, but its hosting mechanism on the
Mac Studio is **not yet decided by devops** — LM Studio itself does not
support vision models, so devops is still evaluating alternatives (see
§9). `ai/vlm_client.py` is written against the OpenAI-compatible
vision-chat shape as the working assumption; only its configured base
URL needs to change once devops confirms the real endpoint.

`ai/vlm_client.py` is not redundant under the current design: it is the
only way `video/analyzer.py` reaches the VLM at all (there is no
in-process VLM anymore), and it remains available for step 5 to use
directly if sentiment analysis ever needs visual reasoning beyond the
summary/transcript already produced in step 3.

## 5. Data Flow

1. Consumer reads a message from the request topic, decodes it into
   `AnalysisRequest { id, text: str | None, video_url: str | None,
   platform: str, metadata: dict }`. Malformed messages are logged and
   skipped (not retried).
2. `pipeline.analyze(request)`:
   - If `video_url` is present:
     1. Download it (`video/downloader.py`).
     2. Run it through `video/analyzer.py`:
        a. `video/frames.py` extracts evenly-spaced frames via `ffmpeg`.
        b. The frames + a prompt go to `ai/vlm_client.py` (network call
           to the Mac-Studio-hosted VLM) to produce `summary`.
        c. If `include_transcript=True` (always, per §4): the local
           `faster-whisper` model transcribes the audio to produce
           `transcript` and `transcript_segments`.
     3. Send `summary` + `transcript` to `ai/llm_client.py` to produce
        the final `video_summary: str` (our LLM synthesizes/refines the
        VLM description and the verbatim transcript into one coherent
        summary for downstream sentiment analysis).
   - If `video_url` is absent: `video_summary = None`.
   - Combine `text` (if present) and `video_summary` (if present) into
     one analysis context.
   - If both are absent/empty: skip analysis, result is marked as failed
     with reason "insufficient input" (no text or video content).
   - Otherwise, call `ai/llm_client.py` with a structured prompt (adapted
     from the n8n reference prompt) to produce sentiment, emotion, and
     motivation analysis.
3. Results are assembled into an `AnalysisResult`, correlated to the
   original item via `id`, and published to the response topic.
4. The consumer commits the offset only after a successful publish
   (at-least-once delivery).

### Message schemas (draft — pending Kafka contract confirmation, see §9)

Request:
```json
{
  "id": "string",
  "platform": "twitter | tiktok | instagram | facebook",
  "text": "string | null",
  "video_url": "string | null",
  "metadata": {}
}
```

Result:
```json
{
  "id": "string",
  "status": "ok | partial | failed",
  "video_summary": "string | null",
  "context": {
    "topic": "string",
    "key_themes": ["string"]
  },
  "sentiment": {
    "label": "positive | negative | neutral",
    "score": 0.0,
    "indicators": ["string"]
  },
  "emotion": {
    "primary": "string",
    "secondary": "string",
    "intensity": 0.0,
    "indicators": ["string"]
  },
  "motivation": {
    "type": "string",
    "confidence_score": 0.0,
    "indicators": ["string"]
  },
  "error": "string | null"
}
```

`status: "partial"` covers graceful-degrade cases (e.g. video failed to
download but text analysis still succeeded). `status: "failed"` covers
insufficient input or exhausted retries; `error` carries the reason.

## 6. Error Handling

- **Video download failure** (private, deleted, unsupported format):
  graceful degrade — continue with text-only analysis if text is
  available, set `video_summary: null`, note the reason in `error`, mark
  `status: "partial"`. Does not fail the whole item.
- **Video analyzer failure** (`video/analyzer.py` / `video/frames.py` —
  `ffmpeg` errors, e.g. corrupt file or unsupported codec; local
  `faster-whisper` errors; or the remote VLM call failing after its own
  retries are exhausted — see below): same graceful degrade as a
  download failure — fall back to text-only with `status: "partial"` if
  text is available, or `status: "failed"` if it isn't.
- **Transient errors calling the LLM/VLM HTTP endpoints** (timeout, 5xx)
  — this now includes the VLM call inside `video/analyzer.py`'s summary
  step, since it is a real network call to the Mac Studio, not an
  in-process call: bounded retry with backoff (default 3 attempts,
  configurable) via `ai/vlm_client.py`/`ai/llm_client.py`'s own
  retry logic. If retries are exhausted, the failure surfaces as a video
  analyzer failure (previous bullet) or, for the sentiment-analysis LLM
  call, a `status: "failed"` result with the error reason (so the
  Scrapper Backend is not left waiting indefinitely) — offset is
  committed either way, avoiding a poison-pill message blocking the
  consumer forever.
- No dead-letter topic in v1; failed items are visible via the
  `status`/`error` fields in the result and via service logs.

## 7. Testing Strategy

- Unit tests for `pipeline.analyze()` with mocked LLM/VLM clients,
  downloader, and video analyzer, covering: text+video, video-only,
  text-only, both-missing, video-download-failure (degrade path),
  video-analyzer-failure (degrade path), LLM-retry-exhausted (failed
  path).
- Unit tests for `schemas.py`: malformed input rejected, optional fields
  behave as optional.
- Unit tests for `ai/vlm_client.py` and `ai/llm_client.py`: request
  construction, response parsing, retry/backoff behavior — HTTP calls
  mocked, no real endpoint hit in tests.
- Unit tests for `video/downloader.py`: invalid/private/unsupported URL
  handling — yt-dlp calls mocked, no real downloads in tests.
- Unit tests for `video/frames.py`: `ffmpeg` invocation mocked, no real
  frame extraction in tests.
- Unit tests for `video/analyzer.py`: `video/frames.py`, `ai/vlm_client.py`,
  and the local whisper model are all injected as fakes/mocks — verify
  our wrapper passes the right arguments, combines their outputs
  correctly, and handles their failure modes (halt on VLM failure,
  degrade on transcription-only failure). Loading the real local whisper
  model at startup is not exercised in these tests (real model weights,
  slow); everything downstream of that load is.
- All functions carry type hints; data models use pydantic. `pytest` +
  a linter/type-checker (e.g. `ruff` + `mypy`) must pass before any task
  is considered done, per Acme's testing standard.
- No end-to-end Kafka integration test in v1; boundaries are covered by
  mocked unit tests instead.

## 8. Deployment

**Revised again in this update:** Docker is viable once more. The
earlier "no Docker" decision was specifically because `mlx-vlm`/
`mlx-whisper` need direct Apple Silicon GPU access — that constraint no
longer applies, because this service (a) runs on a Windows machine, not
the Mac Studio, and (b) has no in-process MLX dependency anymore: the
VLM is called remotely over HTTP, and `faster-whisper` (CPU) runs fine
inside an ordinary Linux container.

The service is packaged as a Docker image. Its container image must
include `ffmpeg` (a system package, e.g. `apt-get install ffmpeg` on a
Debian-based Python base image) alongside the Python dependencies, since
`video/frames.py` and `faster-whisper` both need it. Horizontal scaling
(§3) means running more container replicas in the same Kafka consumer
group; each replica pays the local whisper model's load cost at
startup, which is modest compared to the previously-considered
in-process VLM.

## 9. Open Assumptions (to confirm before/while implementing)

- **Kafka broker address (devops-provided):** `172.16.16.100:21000` —
  this is the bootstrap server for the sentiment analyzer's Kafka
  cluster. It must be supplied to the service via config/env (e.g.
  `KAFKA_BOOTSTRAP_SERVERS`), never hardcoded in source.
- **Kafka topic names and exact message contract** are still not
  coordinated with the Scrapper Backend developer (only the broker
  address is known so far). This spec assumes the request/result shapes
  in §5; adjust once confirmed.
- **VLM hosting mechanism on the Mac Studio is still undecided by
  devops** — LM Studio does not support vision models, and devops is
  still evaluating alternatives. This spec assumes the eventual endpoint
  is OpenAI-compatible over HTTP (matching `ai/vlm_client.py`'s existing
  design) — only its configured base URL/model name should need to
  change once devops confirms; if the real mechanism turns out not to be
  OpenAI-compatible, `ai/vlm_client.py` will need rework.
- **Video duration**: confirmed max ~10 minutes per video (not
  long-form/hour-scale) — keeps the video pipeline lightweight (one
  download + one frame-extraction-and-analysis pass per item, no
  chunking needed).
- **This service's host has no GPU** — confirmed CPU-only for the local
  Windows machine. `faster-whisper`'s model size (default TBD — see
  `config.py`) should be chosen with CPU-only latency in mind, not
  assumed to be the largest/most accurate variant; this is a tunable
  default, not a hard requirement.
- **`ffmpeg` must be present on PATH** (or in the Docker image) — a
  system-level dependency of `video/frames.py` and `faster-whisper`,
  not a pip package. Document this as a prerequisite / Dockerfile step.
- Supported video platforms assumed: TikTok, Instagram, Facebook,
  Twitter/X, matching the platforms referenced in
  `docs/references/Analyst_Batches.json`.
