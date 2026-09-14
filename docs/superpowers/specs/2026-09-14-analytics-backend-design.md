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
   and run it through an in-process video analyzer (adapted from
   `docs/references/server.py`, handed off by the supervisor — see §4)
   to get both a whole-video VLM summary and a raw speech transcript.
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
- Adapting the supervisor-provided `docs/references/server.py` logic
  (mlx-vlm whole-video summary + mlx-whisper transcript) into an
  in-process module — no separate HTTP service, no FastAPI layer; the
  core `load`/`generate`/`transcribe` calls are imported directly.
- Turning that video analyzer's output (VLM summary + raw transcript)
  into a final `video_summary` via our own LLM.
- Sentiment/emotion/motivation analysis via an LLM (and VLM where
  relevant), called over an OpenAI-compatible HTTP API (model-agnostic —
  see §4).
- Graceful handling of items missing text or missing video.

**Out of scope (v1):**
- Building or training a video understanding/summarization model — we
  adapt the supervisor's already-working `server.py` logic, we don't
  build one from scratch.
- Kafka topic names/message contract finalization with the Scrapper
  Backend developer (not yet coordinated — see §9 Open Assumptions).
- Dead-letter topics / advanced retry orchestration beyond bounded
  in-process retries.
- Docker packaging (see §8 — dropped due to an MLX/Apple-Silicon
  constraint).
- End-to-end integration tests against a real Kafka cluster.

## 3. Architecture

Single Python service, single Kafka consumer group, **synchronous
per-message processing**: each message is fully processed (video
download → video analysis → video summarization → text+video sentiment
analysis → publish result) before the next message is picked up, and the
offset is committed only after a successful publish.

This was chosen over two alternatives:
- **Async worker pool in-process** — rejected for v1: added concurrency
  complexity, but the real bottleneck is shared model inference (the
  in-process VLM/Whisper, and the LLM endpoint), not the orchestration
  code, so in-process concurrency wouldn't reliably raise throughput.
- **Separate gateway + task-queue workers (e.g. Celery/RQ + Redis)** —
  rejected for v1: introduces a second broker alongside Kafka with no
  demonstrated need yet.

If throughput becomes a bottleneck later, scale via standard Kafka
mechanisms: more partitions + more consumer replicas. **Caveat introduced
in this revision:** because the video analyzer (mlx-vlm + mlx-whisper) is
now loaded in-process (see §4) rather than called over HTTP, each
consumer replica loads its own full copy of both models into the Mac
Studio's unified memory at startup. This makes replica scaling
meaningfully more expensive (memory + cold-start time) than a stateless
HTTP-calling service would be — acceptable for a small number of
replicas on a Mac Studio's large unified memory, but worth re-evaluating
before scaling out aggressively.

## 4. Components

```
src/
  main.py              # entrypoint: load VLM+Whisper models once, then start Kafka consumer loop
  config.py            # env-based settings (Kafka, LLM base url+model, VLM/Whisper model ids, timeouts, retries)
  schemas.py           # pydantic models: AnalysisRequest, AnalysisResult, internal DTOs
  messaging/
    consumer.py        # Kafka consume wrapper (infra only, no business logic)
    producer.py         # Kafka publish wrapper
  pipeline/
    analyze.py          # use-case orchestrator: optional text/video -> summary -> sentiment result
  video/
    downloader.py       # yt-dlp wrapper (TikTok/IG/FB/X)
    analyzer.py          # adapted from docs/references/server.py: mlx-vlm whole-video summary + mlx-whisper transcript, loaded once at startup, called in-process (no HTTP)
  ai/
    vlm_client.py        # thin HTTP client, OpenAI-compatible endpoint, used in sentiment analysis if visual cues are analyzed directly (separate from video/analyzer.py's local VLM — see note below)
    llm_client.py         # thin HTTP client, OpenAI-compatible endpoint (LM Studio), video summarization + sentiment/emotion/motivation
```

Business logic (`pipeline/`, `video/`, `ai/`) is kept independent of the
Kafka transport (`messaging/`) — `main.py` wires them together. Each
module has one clear responsibility, consistent with Acme's structure
standards (business logic out of handlers, one responsibility per file).

### `video/analyzer.py` (adapted from `docs/references/server.py`)

The supervisor handed off a working FastAPI server
(`docs/references/server.py`) that loads `mlx-vlm` (Qwen2.5-VL / Qwen3-VL,
swappable via `VLM_MODEL_ID`) and `mlx-whisper` once at startup, then on
each request runs a **single whole-video mlx-vlm call** (multi-frame,
temporally-aware — not a frame-by-frame loop) to produce `summary`, and
optionally `mlx-whisper` transcription to produce `transcript` +
`transcript_segments`.

Per the agreed integration approach, we do **not** run this as a
separate FastAPI/HTTP service. Instead:
- `video/analyzer.py` imports the same core functions directly
  (`mlx_vlm.load`/`generate`, `mlx_whisper.transcribe`), stripped of the
  FastAPI/pydantic/upload-handling layer, which is replaced by our own
  `video/downloader.py` output (a local file path) as input.
- Models are loaded **once**, at `main.py` startup, before the Kafka
  consumer loop begins — mirroring `server.py`'s `lifespan` pattern, just
  without FastAPI.
- We call it with `include_transcript=True` always (see §5) — the raw
  transcript is needed downstream for exact wording, not just the VLM's
  descriptive summary.
- `VLM_MODEL_ID` and `WHISPER_MODEL_ID` become config values in
  `config.py`, defaulting to `server.py`'s choices
  (`mlx-community/Qwen2.5-VL-7B-Instruct-4bit`,
  `mlx-community/whisper-large-v3-turbo`) unless devops specifies
  otherwise.

### Model access (LLM, and VLM for sentiment analysis)

The LLM used for video summarization (step 3) and sentiment/emotion/
motivation analysis (step 5) is accessed as an **OpenAI-compatible HTTP
API** — already decided: **LM Studio**. `ai/vlm_client.py` remains listed
as a separate, still-undecided OpenAI-compatible VLM endpoint for the
case where step 5's sentiment analysis needs direct visual reasoning
beyond what `video/analyzer.py`'s summary+transcript already capture.

**Open question flagged, not decided here:** since `video/analyzer.py`
already loads a VLM in-process, `ai/vlm_client.py` may turn out to be
redundant (the in-process VLM could be reused for both video summarization
and any visual sentiment reasoning) rather than a second, separately
hosted VLM. This is called out in §9 rather than assumed either way.

## 5. Data Flow

1. Consumer reads a message from the request topic, decodes it into
   `AnalysisRequest { id, text: str | None, video_url: str | None,
   platform: str, metadata: dict }`. Malformed messages are logged and
   skipped (not retried).
2. `pipeline.analyze(request)`:
   - If `video_url` is present:
     1. Download it (`video/downloader.py`).
     2. Run it through `video/analyzer.py` (in-process mlx-vlm +
        mlx-whisper, `include_transcript=True`) to get `summary`,
        `transcript`, and `transcript_segments`.
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
- **Video analyzer failure** (`video/analyzer.py` — mlx-vlm or
  mlx-whisper call errors, e.g. corrupt file, out-of-memory, unsupported
  codec): same graceful degrade as a download failure — fall back to
  text-only with `status: "partial"` if text is available, or
  `status: "failed"` if it isn't. No retry for this step by default (a
  local in-process model failure is unlikely to be transient in the way
  a network call is); revisit if evidence suggests otherwise.
- **Transient errors calling the LLM/VLM HTTP endpoints** (timeout, 5xx):
  bounded retry with backoff (default 3 attempts, configurable). If
  retries are exhausted, publish a `status: "failed"` result with the
  error reason (so the Scrapper Backend is not left waiting indefinitely)
  and commit the offset — avoids a poison-pill message blocking the
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
- Unit tests for `video/analyzer.py`: mock `mlx_vlm`/`mlx_whisper` calls
  (these require real Apple Silicon hardware and multi-GB model weights,
  so they are not exercised for real in unit tests) — verify our wrapper
  passes the right arguments and handles their failure modes.
- All functions carry type hints; data models use pydantic. `pytest` +
  a linter/type-checker (e.g. `ruff` + `mypy`) must pass before any task
  is considered done, per Acme's testing standard.
- No end-to-end Kafka integration test in v1; boundaries are covered by
  mocked unit tests instead.

## 8. Deployment

**Revised in this update:** no longer Docker. `mlx-vlm`/`mlx-whisper`
depend on MLX, which needs direct access to Apple Silicon's GPU/unified
memory (Metal) — Docker Desktop on macOS runs Linux containers inside a
VM without Metal passthrough, so containerizing this service would very
likely break (or silently run without GPU acceleration, which is not
viable for a 7B-parameter VLM).

The whole service — Kafka consumer, pipeline, and the in-process video
analyzer — runs as a **native Python process directly on the Mac
Studio** (venv + a process supervisor appropriate for macOS, e.g.
`launchd`), not containerized. Horizontal scaling (§3) means running
multiple native processes on the same (or another) Mac Studio, each in
the same Kafka consumer group, mindful of the per-replica memory cost
noted in §3.

## 9. Open Assumptions (to confirm before/while implementing)

- **Kafka broker address (devops-provided):** `172.16.16.100:21000` —
  this is the bootstrap server for the sentiment analyzer's Kafka
  cluster. It must be supplied to the service via config/env (e.g.
  `KAFKA_BOOTSTRAP_SERVERS`), never hardcoded in source.
- **Kafka topic names and exact message contract** are still not
  coordinated with the Scrapper Backend developer (only the broker
  address is known so far). This spec assumes the request/result shapes
  in §5; adjust once confirmed.
- **`ai/vlm_client.py` may be redundant** now that `video/analyzer.py`
  loads a VLM in-process (see §4) — needs a decision once we know
  exactly what step 5's "VLM" role is meant to add beyond the video
  summary already produced in step 3. Not resolved in this spec; flagged
  so it isn't silently assumed either way.
- **Video duration**: confirmed max ~10 minutes per video (not
  long-form/hour-scale) — keeps the video pipeline lightweight (one
  download + one in-process analyzer call per item, no chunking needed).
- **Hardware assumption**: the service (and therefore the whole
  deployment) requires an Apple Silicon Mac (Mac Studio) to run at all,
  since `video/analyzer.py` hard-depends on MLX. This is a real
  portability constraint, not just a deployment preference — worth
  surfacing to devops explicitly if it wasn't already clear from handing
  off `server.py`.
- Supported video platforms assumed: TikTok, Instagram, Facebook,
  Twitter/X, matching the platforms referenced in
  `docs/references/Analyst_Batches.json`.
