# Analytics Backend — Design Spec (Revision 1)

Date: 2026-09-17
Status: Draft, pending review before implementation
Supersedes (for video processing only): `docs/superpowers/specs/2026-09-14-analytics-backend-design.md`

## 0. Why this revision exists

New supervisor directive (`docs/to-do.md`, 2026-09-17), ten numbered points,
all scoped to **video processing only** (point 10 is explicit: consume,
produce, and sentiment analysis are unchanged). This is the **fourth**
revision of how this service gets a description out of a video:

1. Own open-source model, never built.
2. In-process `mlx-vlm`/`mlx-whisper` (Mac Studio assumption) — wrong host,
   MLX doesn't run on Windows.
3. Remote HTTP VLM (Mac Studio, mechanism undecided).
4. In-process `transformers`/`llava-onevision` (this machine, CPU) — the
   supervisor asked for no HTTP serving.
5. **This revision:** remote HTTP VLM again, but now the *same* LM Studio
   endpoint/model already used for sentiment analysis
   (`qwen3-vl-8b-instruct-mlx`) — explicitly the supervisor's decision,
   stated directly in `docs/to-do.md` point 6 ("kita tidak lagi
   menggunakan local model. ini sudah keputusan supervisor").

This is not a mistake being corrected — each revision was a reasonable
decision given what was known/decided at the time. Documenting the history
so nobody re-litigates revision 3 or 4's reasoning against revision 5's
different constraints.

Three points needed clarification before this could be written; answers
below are locked in for implementation, not open questions anymore:

- **VLM/LLM config stays separate** (`VLM_BASE_URL`/`VLM_MODEL` alongside
  `LLM_BASE_URL`/`LLM_MODEL`), even though both currently point at the same
  server and model. If they diverge later, no code change is needed — only
  `.env`.
- **"Batch" means grouping keyframes into one multi-image call per batch**,
  size configurable, not one HTTP call per single frame (video-analyzer's
  own approach, rejected as a literal copy — see §3).
- **The in-process VLM code is deleted outright** (`ai/vlm_local.py`, the
  `torch`/`transformers` dependency, the CVE-driven pins that came with
  them), not kept dormant. Recoverable from git history if ever needed
  again.

## 1. Scope

**In scope (this revision):**
- Keyframe selection: replace uniform time-based sampling with
  grayscale-difference scoring, adopting `video-analyzer`'s algorithm
  (`C:\Users\kacang\IdeaProjects\video-analyzer\video_analyzer\frame.py`)
  without adopting its code or its OpenCV dependency (§3).
- The frame-difference threshold becomes a configurable setting.
- `ffmpeg` decode gets an optional, configurable CUDA hardware-acceleration
  path (§4) — implemented but **not testable on this machine** (confirmed
  CPU-only, integrated AMD GPU, no NVIDIA hardware; see spec history in the
  original design doc §9 for why DirectML was already ruled out for the
  now-deleted in-process VLM).
- Console logging for the video-processing path becomes as granular as
  `video-analyzer`'s (§5): a visible line per meaningful step, not just
  start/end of the whole pipeline.
- Keyframes are sent to the VLM in **batches** of a configurable size,
  batching itself toggleable on/off (§3).
- The VLM becomes a **remote HTTP call again** to
  `http://172.16.16.101:1234/v1`, model `qwen3-vl-8b-instruct-mlx` — the
  same LM Studio instance already serving `LLM_BASE_URL`/`LLM_MODEL` for
  sentiment analysis, but a separate config entry (§2).
- A dev-only local dump of each processed item's final `AnalysisResult` to
  `output/output-<id>-<timestamp>.json` (§6).
- `whisper_vad_filter` stays `true` (no change; already the default) and
  transcription stays language-auto-detect (no change; already the
  default — `video/analyzer.py`'s `transcribe()` closure has never passed
  a `language` argument, so faster-whisper has always auto-detected).
  Both are called out explicitly here only because the directive asked for
  them by name, not because anything needs to change.

**Out of scope (unchanged, per directive point 10):**
- `messaging/consumer.py`, `messaging/producer.py`, `messaging/scrapper_dto.py`
  — Kafka consume/produce and the nested-comments translation are untouched.
- `ai/llm_client.py`'s `analyze_sentiment()` and the overall
  `pipeline/analyze.py` orchestration shape (download → analyze video →
  summarize → combine with text → sentiment) — untouched, except that
  `summarize_video()`'s visual-description input now comes from possibly
  multiple batch descriptions joined together instead of one whole-video
  description (§3) — the function signature and its role in the pipeline
  don't change.

## 2. Configuration changes

All additions to `src/config.py` / `.env.example`. Everything keeps a
sensible default so `.env` files that predate this revision don't break
(the directive doesn't ask for a breaking config change, only new
capability).

| Setting | Type | Default | Purpose |
|---|---|---|---|
| `VLM_BASE_URL` | `str` | *(required)* | Reintroduced. `http://172.16.16.101:1234/v1` in this deployment. |
| `VLM_MODEL` | `str` | *(required)* | Reintroduced. `qwen3-vl-8b-instruct-mlx` in this deployment. |
| `KEYFRAME_DIFF_THRESHOLD` | `float` | `10.0` | Grayscale mean-absolute-difference a candidate frame must exceed vs. the previous *kept* candidate to be considered a keyframe. `video-analyzer` hardcodes this at 10.0 (see its `frame.py`, `FRAME_DIFFERENCE_THRESHOLD` — a real bug we found: its own `analysis_threshold`/`min_difference` config keys are dead, never read by that constant). Point 2 of the directive is specifically about not repeating that mistake. |
| `FFMPEG_HWACCEL_CUDA` | `bool` | `false` | When true, adds `-hwaccel cuda -hwaccel_output_format cuda` to the `ffmpeg` decode invocation in `video/frames.py`. Off by default because this deployment has no NVIDIA GPU to test it against (§4) — exists for whichever machine runs this with one. |
| `VLM_BATCH_ENABLED` | `bool` | `true` | On: keyframes are grouped into batches of `VLM_BATCH_SIZE` and sent as separate multi-image calls. Off: every keyframe goes into a single call, same shape as the revision-4 in-process design (just over HTTP now). |
| `VLM_BATCH_SIZE` | `int` | `4` | Keyframes per batch when `VLM_BATCH_ENABLED=true`. Ignored otherwise. Starting default chosen for a reasonable balance of per-call payload size vs. number of round trips to the Mac Studio; not measured yet (§9). |
| `DEV_DUMP_OUTPUT` | `bool` | `true` | Dev-only (directive point 7): also write each processed item's final `AnalysisResult` to a local JSON file, in addition to publishing it to Kafka as before. Defaults on per the directive's explicit ask; flagged in §9 as something to reconsider before a real production rollout (unbounded local disk growth, one file per item forever). |

Removed: `VLM_MODEL_ID` (was the in-process HuggingFace model id — no
longer meaningful), `whisper_vad_filter`'s status quo is kept as-is
(default `true`, still overridable — no removal, no change).

## 3. Keyframe selection and batched VLM calls

### 3.1 Keyframe selection (`video/frames.py`)

Current behavior (revision 4, being replaced): `fps = max_frames /
duration`, uniformly spaced — no awareness of what's actually different
between frames. This is simple but wasteful on visually static footage
(evenly spaced samples of a static talking-head shot are all nearly
identical) and was never the bottleneck we were tuning; frame *count* and
resolution were (see the original design doc's real-latency notes).

New behavior, adopting `video-analyzer`'s algorithm (its `frame.py`) but
**not its library** (see below):

1. `ffmpeg` extracts *candidate* frames at a higher sampling rate than the
   final target — enough temporal resolution to have real choices, not so
   many that scoring them is expensive. Mirrors `video-analyzer`'s
   two-stage approach (oversample, then pick the best subset) without
   needing OpenCV's frame-by-frame video decode loop — `ffmpeg` remains
   the one thing doing video decode, per directive point 3.
2. Each candidate is converted to grayscale and compared against the
   **immediately preceding candidate that was considered** (not the last
   one that was *kept*) using mean absolute pixel difference — the same
   metric `video-analyzer` uses (`cv2.absdiff` + `numpy.mean`), and the
   same reference-update behavior: `video-analyzer`'s own `frame.py`
   reassigns its comparison reference (`prev_frame = frame.copy()`)
   unconditionally after every scored candidate, whether or not that
   candidate cleared the threshold — this spec originally mischaracterized
   this as "compare against the previously kept candidate" (an error
   caught during implementation by directly re-reading `video-analyzer`'s
   source rather than trusting this document; corrected here, see the
   implementation plan's revision-1 ledger for the ruling). Implemented
   with **Pillow + numpy instead of OpenCV**: `Image.convert("L")` for
   grayscale, `numpy.abs(a.astype(int) - b.astype(int)).mean()` for the
   score. Numerically equivalent; avoids adding `opencv-python` as a new
   dependency when Pillow (already a dependency) and numpy (already
   present transitively, made direct once `torch`/`transformers` are
   removed — see §7) cover it. This is a deliberate, documented
   dependency decision, not an oversight.
3. A candidate is kept as a keyframe only if its score exceeds
   `KEYFRAME_DIFF_THRESHOLD` (configurable — directive point 2; fixing the
   exact bug we found in `video-analyzer`'s own hardcoded, unconfigurable
   version of this same threshold).
4. Kept keyframes are capped at `max_frames` (existing setting, reused —
   its meaning shifts from "how many to evenly sample" to "how many
   difference-selected keyframes to keep at most", same role
   `video-analyzer`'s `frames.max_count` plays).
5. **Known edge case, inherited deliberately:** a visually static clip can
   still produce zero or very few keyframes (this is exactly what we
   diagnosed happening to `video-analyzer` on a 10s static news-anchor
   shot). `video/analyzer.py`'s downstream handling must treat "zero
   keyframes" as a valid, non-error outcome — degrade to transcript-only
   visual context for that item rather than failing the whole video step,
   consistent with the graceful-degrade philosophy already in place for
   download/analysis failures (original spec §6).

### 3.2 Batched VLM calls (`ai/vlm_client.py`, recreated)

The in-process `ai/vlm_local.py` (LLaVA-OneVision, `transformers`, local
CPU inference) is deleted. `ai/vlm_client.py` — deleted during revision 4,
recreated here — goes back to being an HTTP client, but its calling
contract changes from "one call with every frame" to "one call per batch
of frames":

- `VLMConfig` (base_url, model, timeout_seconds, retry_attempts,
  retry_backoff_seconds) — same shape as the pre-revision-4 version.
- `describe_images(config, prompt, image_data_urls: list[str]) -> str` —
  same signature as before (one call, N images, one text result) — batching
  is a *calling convention* on top of this, not a change to the function
  itself. `video/frames.py` goes back to returning base64 `data:` URLs
  (not raw bytes, which only made sense for the in-process PIL path) since
  we're back to sending images over HTTP.
- New orchestration in `video/analyzer.py`'s `generate_summary`-equivalent:
  split the keyframe list into chunks of `VLM_BATCH_SIZE` (or one chunk
  containing everything, if `VLM_BATCH_ENABLED=false`), call
  `describe_images()` once per chunk, and join the per-batch descriptions
  — each prefixed with its timestamp range, mirroring `video-analyzer`'s
  "Frame N (timestamp): description" convention (directive point 4,
  logging/output style parity) — into one combined visual-description
  string.
- **No new synthesis call is added.** The combined, multi-batch visual
  description is passed into the *existing* `ai/llm_client.py`'s
  `summarize_video(vlm_summary, transcript)` exactly as today's
  single-batch description is — that function already exists specifically
  to synthesize a visual description with the audio transcript into one
  coherent `video_summary`, and doesn't care whether the visual
  description came from one VLM call or several stitched together.
  `pipeline/analyze.py` does not change.
- Per-batch progress is logged (directive point 4): one line per batch
  sent, one per batch completed, matching the granularity
  `video-analyzer` logs per-frame (`"Successfully analyzed frame N"`).

### 3.3 Frame resolution

Revision 4 downscaled frames to 448px on the longest side specifically to
avoid LLaVA-OneVision's context-window overflow (see the original design
doc's real-latency section — 8 full-resolution frames blew a 32768-token
budget). The new model (`qwen3-vl-8b-instruct-mlx`) is a different model
with unknown token-per-image behavior. The downscale step is **kept as a
precaution** (cheap, and this project has already paid once for not doing
it), but the exact dimension is flagged in §9 as something to re-measure
against the new model rather than assumed correct.

## 4. `ffmpeg` and CUDA

Directive point 3: "tetap menggunakan ffmpeg dan configurable untuk
menggunakan cuda." `ffmpeg` remains the only video decoder (no OpenCV
`VideoCapture`, per §3.1). When `FFMPEG_HWACCEL_CUDA=true`,
`video/frames.py`'s `ffmpeg` invocation adds `-hwaccel cuda
-hwaccel_output_format cuda` ahead of the input, enabling GPU-accelerated
decode on hardware that has an NVIDIA GPU and an `ffmpeg` build with CUDA
support.

**This cannot be exercised on the current development machine** (AMD
integrated GPU only — already established when DirectML was investigated
and rejected for the now-deleted in-process VLM, see the original design
doc §9). The flag is implemented and defaults to `false`; verifying it
actually works requires either a machine with an NVIDIA GPU or accepting
it ships untested until one is available. Flagged in §9 as an open item,
not something this revision can close out.

## 5. Console logging parity with `video-analyzer`

Directive point 4. `video-analyzer`'s logs (`--log-level DEBUG`) show a
line per: model load, audio extraction start, transcription start, each
VAD-kept segment being processed, frame-extraction result count vs.
target, each frame analyzed, reconstruction start/end. This project
already added comparable per-step logging incrementally (checkpoint logs
for model loading, per-step logs in `pipeline/analyze.py` and
`video/analyzer.py`, a startup banner, an idle-polling heartbeat — see the
original design doc's revision history). This revision extends the same
pattern to the two new steps that don't exist yet:

- Keyframe selection: log the candidate count considered, the count kept,
  and the threshold used (mirrors `video-analyzer`'s "Extracted N frames
  from video (target was M)").
- Batched VLM calls: log each batch's frame count and timestamp range
  before the call, and confirmation after (mirrors "Successfully analyzed
  frame N").

No new log *level* configuration is introduced — this project's logging
is `INFO`-level by default already (`main.py`'s `logging.basicConfig`);
matching `video-analyzer`'s console verbosity is about adding lines at the
same points, not adopting its `--log-level` flag.

## 6. Dev-only local output dump

Directive point 7. After a processed item's `AnalysisResult` is built
(same point in `pipeline/analyze.py`'s flow as today, right before
`producer.publish()` in `main.py`'s `run_once()`), and gated by
`DEV_DUMP_OUTPUT`, also write it to
`output/output-<id>-<timestamp>.json` (timestamp format
`YYYYMMDDTHHMMSS`, id included specifically because a post and its nested
comments — see the original design doc's nested-comments revision — can
all be processed within the same second, and a bare timestamp alone could
collide). This does **not** replace or delay the Kafka publish; it's an
additional, best-effort side write for local inspection, matching how
`video-analyzer` always writes `output/analysis.json` for itself. A
failure writing this file is logged as a warning and must never fail the
item's processing or block the Kafka publish.

## 7. Dependency changes

Removed (§0, confirmed): `torch`, `transformers`, and their CVE-driven
version pins (`ai/vlm_local.py`'s reason for existing). `pillow` is
**kept** — still needed for the grayscale-difference scoring (§3.1) and
for decoding/downscaling frames before the HTTP call (§3.3).

Added: `numpy`, as a **direct** dependency (per Acme's dependency-is-a-
decision standard). It was already present transitively (via `torch` and
others) so this isn't a new supply-chain surface in practice — removing
`torch` just means it needs to be declared explicitly now that nothing
else pulls it in. Pin `numpy>=1.26,<3` (avoids the 1.x/2.x ABI break for
any transitive consumer still expecting 1.x, while allowing 2.x once the
rest of the dependency tree is confirmed compatible — no known CVEs
against either major line at the pin floor).

`opencv-python` is **deliberately not added** (§3.1) — Pillow + numpy
reproduce the one algorithm this revision needs from it, at a much
smaller dependency footprint (`opencv-python` wheels are large and pull
in their own native binary dependencies).

## 8. Testing strategy (additions only)

Following this project's existing conventions (`docs/testing-guidelines.md`):

- `video/frames.py`: unit tests for the grayscale-difference scoring
  (deterministic given synthetic Pillow images — a solid-color pair
  scores near zero, a black/white pair scores near 255), the threshold
  cutoff, the `max_frames` cap, and the CUDA flag's effect on the
  constructed `ffmpeg` command (mocked subprocess, per existing
  convention — no real GPU needed to test that the flag is *passed*).
- `ai/vlm_client.py`: recreated with the same `respx`-mocked-HTTP test
  style the pre-revision-4 version had (git history has the exact file).
- `video/analyzer.py`: batching orchestration tested via injected fakes
  (existing `AnalyzerModels`-style dependency injection) — assert the
  right number of `describe_images()`-equivalent calls happen for a given
  keyframe count and `VLM_BATCH_SIZE`, and that batch descriptions get
  joined with timestamp prefixes in chronological order.
- `main.py`/`pipeline/analyze.py`: a test that `DEV_DUMP_OUTPUT=true`
  writes the expected file and that a write failure there doesn't prevent
  `producer.publish()` from being called.
- Real-model verification (per this project's established practice of not
  trusting mocked tests alone for this specific class of change): a real
  run via `scripts/replay_sample_payload.py` against the real
  `qwen3-vl-8b-instruct-mlx` endpoint, checked for the same class of
  issue already found once with a different remote model (empty/truncated
  `content` from a reasoning-style model eating its token budget — see
  the `video-analyzer` investigation in this same conversation). Not
  assumed safe just because it's the same model already used for text
  sentiment analysis.

## 9. Open assumptions / follow-ups (not blocking, but not verified yet)

- `VLM_BATCH_SIZE=4` default is a starting guess, not a measurement —
  revisit once real latency with the new remote model is observed (same
  spirit as the original design doc's `MAX_FRAMES` tuning history).
- `FFMPEG_HWACCEL_CUDA` is implemented but unverified — no NVIDIA hardware
  available in this environment.
- The 448px frame-downscale carried over from revision 4 is a precaution,
  not a re-validated number for `qwen3-vl-8b-instruct-mlx` specifically.
- `DEV_DUMP_OUTPUT` defaulting to `true` writes one file per processed
  item forever with no cleanup — fine for the current development phase
  (directive explicitly asks for it "untuk development"), but should be
  revisited (default off, or add rotation/retention) before any real
  production rollout.
- Whether `qwen3-vl-8b-instruct-mlx` has the same "reasoning model empties
  its `content` field" failure mode observed with `deepseek-v4-flash-
  vision-exp` during the `video-analyzer` evaluation is unknown until
  tested for real (see §8) — this revision does not assume it's safe.
