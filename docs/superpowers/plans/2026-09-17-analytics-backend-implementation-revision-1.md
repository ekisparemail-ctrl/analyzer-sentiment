# Analytics Backend Implementation Plan (Revision 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework only the video-processing path of the Analytics Backend
per the supervisor's 2026-09-17 directive (`docs/to-do.md`): grayscale-
difference keyframe selection (replacing uniform time sampling),
configurable keyframe threshold, configurable `ffmpeg` CUDA
hardware-accel, logging parity with `video-analyzer`, batched VLM calls
(configurable on/off and batch size), the VLM moving back to a remote
HTTP endpoint (the same LM Studio instance already used for sentiment
analysis — `qwen3-vl-8b-instruct-mlx`, but a separate config entry), and a
dev-only local JSON dump of each processed item's result. Consume,
produce, and sentiment analysis are explicitly unchanged.

**Spec:** `docs/superpowers/specs/2026-09-17-analytics-backend-design-revision-1.md`
(read this first — it has the full rationale, the three clarified
decisions this plan assumes, and what's deliberately out of scope)

## Global Constraints

- This is a **revision of an existing, working, merged codebase** — not a
  from-scratch build. Every task below touches existing files; check the
  current content before editing rather than assuming Task-1-era shapes.
- `pytest`, `ruff`, and `mypy` must all pass before any task is considered
  done (Acme Testing standard) — same as every prior task in this project.
- New logic requires tests in the same PR (Acme Testing standard).
- Never commit directly to main/master — branch, commit, merge, push, per
  this project's established workflow (see recent git log for the exact
  pattern: one focused branch per logical change, merged with `--no-edit`,
  deleted after).
- `opencv-python` is deliberately **not** added as a dependency — the
  grayscale-difference algorithm is reimplemented with Pillow + numpy
  (spec revision-1 §3.1, §7). Do not reach for `cv2` during implementation.
- The real endpoint (`http://172.16.16.101:1234/v1`,
  `qwen3-vl-8b-instruct-mlx`) is a live, shared Mac-Studio-hosted service —
  real-model verification steps in this plan hit it for real; budget for
  that when estimating task time, and never hardcode the API path assuming
  it will always be reachable during automated test runs (unit tests stay
  fully mocked, per this project's standing convention).

---

## Task 1: Configuration additions

**Files:**
- Modify: `src/config.py`
- Modify: `.env.example`
- Modify: `requirements.txt`
- Modify: `tests/test_config.py`

**Interfaces:**
- `Settings` gains: `vlm_base_url: str`, `vlm_model: str`,
  `keyframe_diff_threshold: float = 10.0`, `ffmpeg_hwaccel_cuda: bool = False`,
  `vlm_batch_enabled: bool = True`, `vlm_batch_size: int = 4`,
  `dev_dump_output: bool = True`.
- `Settings` loses: `vlm_model_id` (the in-process HuggingFace model id —
  meaningless once `ai/vlm_local.py` is deleted in Task 2).

- [ ] **Step 1: Add the new settings fields**
  Add the six new fields above to `Settings` with the defaults from spec
  revision-1 §2 and doc-comments explaining each (matching this project's
  existing style — see `whisper_vad_filter`'s comment for the tone/length
  to match). Remove `vlm_model_id`.

- [ ] **Step 2: Update `requirements.txt`**
  Remove `torch`, `transformers` (and their CVE-driven pin comments —
  the reason for the pins no longer exists once `ai/vlm_local.py` is
  gone). Add `numpy>=1.26,<3` as a direct dependency (spec revision-1 §7 —
  it's already present transitively today; this just makes it explicit
  now that nothing else in the dependency tree will pull it in once
  `torch`/`transformers` are removed). Keep `pillow` (still needed).

- [ ] **Step 3: Update `.env.example`**
  Reintroduce `VLM_BASE_URL`/`VLM_MODEL` (real values for this
  deployment: `http://172.16.16.101:1234/v1` /
  `qwen3-vl-8b-instruct-mlx`), remove the old `VLM_MODEL_ID` line and its
  in-process-specific comment, add `KEYFRAME_DIFF_THRESHOLD`,
  `FFMPEG_HWACCEL_CUDA`, `VLM_BATCH_ENABLED`, `VLM_BATCH_SIZE`,
  `DEV_DUMP_OUTPUT` with the same explanatory-comment style as existing
  entries (e.g. `WHISPER_VAD_FILTER`'s comment).

- [ ] **Step 4: Update `tests/test_config.py`**
  Update the defaults-assertion test for the new fields; remove the
  `vlm_model_id` assertion. `VLM_BASE_URL`/`VLM_MODEL` are required
  (no default, like `llm_base_url`/`llm_model`) — add them to the test's
  `monkeypatch.setenv(...)` calls alongside the other required fields, and
  add a dedicated test asserting `Settings(_env_file=None)` raises
  `ValidationError` when they're missing (mirroring the existing
  `llm_base_url` coverage).

**Verify:** `pytest tests/test_config.py -q`, `ruff check src/config.py`,
`mypy src/config.py`.

---

## Task 2: Delete the in-process VLM

**Files:**
- Delete: `src/ai/vlm_local.py`
- Delete: `tests/ai/test_vlm_local.py`

**Interfaces:** none produced — this is pure removal. `video/analyzer.py`
and `main.py` still reference `ai.vlm_local` after this task; they are
fixed in Tasks 4/5, not here — **expect `pytest`/`mypy` to fail after this
task alone** (broken imports elsewhere). Land Tasks 2–5 together in one
branch/commit sequence rather than trying to keep the tree green after
every single task in isolation, since they're one tightly-coupled change.

- [ ] **Step 1: Delete the files**
  `git rm src/ai/vlm_local.py tests/ai/test_vlm_local.py`. Do not delete
  `src/ai/llm_client.py` or its tests — unrelated, still used for
  sentiment analysis and video summarization synthesis (spec revision-1
  §3.2's "no new synthesis call" point).

**Verify:** nothing yet — this task is intentionally left red until
Task 3 recreates `ai/vlm_client.py` and Tasks 4/5 rewire the callers.

---

## Task 3: Recreate `ai/vlm_client.py` (HTTP-based)

**Files:**
- Create: `src/ai/vlm_client.py`
- Create: `tests/ai/test_vlm_client.py`

**Interfaces:**
- `VLMClientError(Exception)`
- `VLMConfig` dataclass: `base_url: str`, `model: str`,
  `timeout_seconds: float = 30.0`, `retry_attempts: int = 3`,
  `retry_backoff_seconds: float = 1.0` — identical shape to
  `ai/llm_client.py`'s `LLMConfig`.
- `describe_images(config: VLMConfig, prompt: str, image_data_urls: list[str]) -> str`
  — one HTTP call, N images, one text result. **This is the exact
  pre-revision-4 file** (deleted in commit history when the in-process
  VLM was introduced) — check it out from git history
  (`git log --diff-filter=D -- src/ai/vlm_client.py` to find the deleting
  commit, then `git show <parent-commit>:src/ai/vlm_client.py`) as a
  starting point rather than writing it from scratch, since it already
  existed, was reviewed, and was fully tested. Re-verify it still matches
  this project's current conventions (logging, error wrapping) before
  reusing verbatim.

- [ ] **Step 1: Recreate the client from git history**
  Retrieve the pre-revision-4 version (see Interfaces above), place it at
  `src/ai/vlm_client.py`, adjust only what's needed to match current
  conventions (e.g. this project now has module-level `logger =
  logging.getLogger(__name__)` in most `src/` files — add one here if the
  historical version predates that pattern).

- [ ] **Step 2: Recreate its tests from git history**
  Same approach for `tests/ai/test_vlm_client.py` (respx-mocked HTTP,
  covering: request/image construction, retry-then-succeed,
  retries-exhausted). Re-run and confirm they still pass unmodified
  against the recreated client.

**Verify:** `pytest tests/ai/test_vlm_client.py -q`,
`ruff check src/ai/vlm_client.py`, `mypy src/ai/vlm_client.py`.

---

## Task 4: Grayscale-difference keyframe selection in `video/frames.py`

**Files:**
- Modify: `src/video/frames.py`
- Modify: `tests/video/test_frames.py`

**Interfaces:**
- `extract_frames(video_path: str, max_frames: int, diff_threshold: float, hwaccel_cuda: bool = False) -> list[str]`
  — signature grows two parameters; return type reverts from `list[bytes]`
  (revision-4's raw-bytes design, for the in-process PIL path) back to
  `list[str]` of base64 `data:image/jpeg;base64,...` URLs (needed again
  now that images travel over HTTP — spec revision-1 §3.2).
- New internal helper (name your choice, not part of the public
  interface): grayscale-difference scoring — given two JPEG byte strings
  (or two `PIL.Image` objects), return a float score via
  `numpy.abs(array_a.astype(int) - array_b.astype(int)).mean()` on
  `Image.convert("L")` data. Keep this as a small, separately-testable
  function — spec revision-1 §8 calls for direct unit tests against
  synthetic solid-color image pairs (deterministic, no real video needed).

- [ ] **Step 1: Write the failing tests first (TDD)**
  Extend `tests/video/test_frames.py` (existing tests for the ffmpeg
  subprocess mocking, `fps`/duration computation, error handling stay —
  only the *selection* logic changes): a test that two identical
  synthetic frames score ~0 and are not both kept; a test that two very
  different synthetic frames (e.g. solid black vs. solid white) score
  near 255 and are kept; a test that `diff_threshold` is respected (score
  just above vs. just below); a test that the `max_frames` cap still
  applies when many candidates clear the threshold; a test that
  `hwaccel_cuda=True` adds `-hwaccel cuda -hwaccel_output_format cuda` to
  the mocked `ffmpeg` subprocess call args, and `False` (default) doesn't.
  Confirm RED before implementing (this project's standing TDD practice).

- [ ] **Step 2: Implement candidate oversampling + scoring + selection**
  `ffmpeg` extracts candidates at a higher rate than `max_frames` (spec
  revision-1 §3.1 step 1 — pick a concrete oversampling factor, e.g. 3x,
  and document why in a comment). Score each candidate against the
  *immediately preceding candidate that was considered* — not the last
  one that was *kept* — matching `video-analyzer`'s own logic exactly:
  its `frame.py` reassigns the comparison reference unconditionally after
  every scored candidate, whether or not that candidate cleared the
  threshold. (**Corrected 2026-09-17, mid-implementation:** an earlier
  version of this step said the opposite — "compare against the
  previously kept candidate" — which was wrong; caught by re-reading
  `video-analyzer`'s actual `frame.py` rather than trusting this
  document. See the implementation ledger's ruling if you need the full
  story.) Keep the candidate if its score exceeds `diff_threshold`; cap
  total kept at `max_frames`; encode kept frames as base64 `data:` URLs
  (same encoding revision-4 removed — git history has it too, in the
  same deleted-code search as Task 3).

- [ ] **Step 3: Handle the zero-keyframes edge case explicitly**
  Confirm (with a test) that a video producing zero keyframes returns an
  empty list rather than raising — spec revision-1 §3.1 step 5 requires
  `video/analyzer.py` (Task 5) to treat this as a valid degrade-to-
  transcript-only case, not an error.

**Verify:** `pytest tests/video/test_frames.py -q`,
`ruff check src/video/frames.py`, `mypy src/video/frames.py`.

---

## Task 5: Batched VLM calls in `video/analyzer.py`

**Files:**
- Modify: `src/video/analyzer.py`
- Modify: `tests/video/test_analyzer.py`
- Modify: `src/main.py` (dependency wiring only — see Step 3)
- Modify: `tests/test_main.py` (wiring assertions only, if any reference
  the removed `vlm_model_id`/in-process load path)

**Interfaces:**
- `load_models(vlm_config: VLMConfig, whisper_model_size: str, whisper_vad_filter: bool) -> AnalyzerModels`
  — `vlm_model_id: str` parameter is replaced by `vlm_config: VLMConfig`
  (base_url/model now come from the HTTP client config, not a local model
  id). No more in-process model loading for the VLM — only
  `faster-whisper` still loads a real local model here.
- `generate_summary`'s closure gains the batching orchestration: given the
  keyframe list from `video/frames.py`, split into chunks of
  `vlm_batch_size` (or a single chunk if `vlm_batch_enabled=False`), call
  `ai.vlm_client.describe_images()` once per chunk, prefix each batch's
  result with its frame-timestamp range, join in chronological order.
  Needs `vlm_batch_enabled`/`vlm_batch_size` threaded through from
  `Settings` — decide during implementation whether they become
  additional `load_models()` parameters or additional fields the closure
  captures; either is fine, pick whichever keeps the function's arity
  reasonable and document the choice in a comment if it's not obvious.
- `analyze_video()`'s public signature and `VideoAnalysis` shape are
  **unchanged** — batching is entirely internal to how `summary` gets
  produced.

- [ ] **Step 1: Write the failing tests first (TDD)**
  Using this project's existing dependency-injection test style (fake
  `describe_images`-equivalent callables, no real HTTP) — a test that N
  keyframes with `vlm_batch_size=4` produce exactly `ceil(N/4)` calls; a
  test that `vlm_batch_enabled=False` produces exactly 1 call regardless
  of keyframe count; a test that batch results are joined with visible
  timestamp markers in chronological order; a test that **zero keyframes**
  (Task 4 Step 3's edge case) degrades to a transcript-only result rather
  than raising or calling the VLM at all. Confirm RED first.

- [ ] **Step 2: Implement the batching orchestration and rewire `load_models()`**
  Replace the `ai.vlm_local` import/call with `ai.vlm_client`. Add the
  per-batch logging called for in spec revision-1 §5 (frame count +
  timestamp range before each batch call, confirmation after — same
  granularity as the existing "Extracted N frames..." /
  "VLM inference complete." lines this project already has, just at
  batch granularity instead of whole-video granularity).

- [ ] **Step 3: Rewire `main.py`'s `build_dependencies()`**
  Construct a `VLMConfig` from `settings.vlm_base_url`/`settings.vlm_model`
  (plus the existing `http_timeout_seconds`/`retry_attempts`/
  `retry_backoff_seconds`, same pattern `LLMConfig` already uses) and pass
  it into `load_models()` instead of the removed `settings.vlm_model_id`.
  Thread `settings.vlm_batch_enabled`/`settings.vlm_batch_size` through
  however Step 2 of this task decided to receive them. Update the startup
  banner (`_log_startup_banner()`) to log the VLM's `base_url`/`model`
  again (it currently logs `vlm_model_id`, which no longer exists).

**Verify:** `pytest tests/video/test_analyzer.py tests/test_main.py -q`,
`ruff check src/video/analyzer.py src/main.py`,
`mypy src/video/analyzer.py src/main.py`.

---

## Task 6: Dev-only local output dump

**Files:**
- Modify: `src/main.py` (or `src/pipeline/analyze.py` — decide during
  implementation which is the more natural seam; `main.py`'s `run_once()`
  already has the finished `AnalysisResult` right before
  `producer.publish()`, which is likely the simpler insertion point since
  it needs no new data `pipeline.analyze()` doesn't already return)
- Modify: `tests/test_main.py`

**Interfaces:**
- A small helper (e.g. `_dump_result_for_dev(result: AnalysisResult, output_dir: str) -> None`)
  called from `run_once()` when `settings.dev_dump_output` is true, after
  a successful `analyze()` call and before/alongside `producer.publish()`.
  Writes `output/output-<id>-<timestamp>.json` (spec revision-1 §6 —
  timestamp format `YYYYMMDDTHHMMSS`, `id` included to avoid collisions
  between a post and its comments processed in the same second).

- [ ] **Step 1: Write the failing tests first (TDD)**
  A test that `dev_dump_output=True` writes a file matching the expected
  naming pattern with the expected JSON content (use `tmp_path`, don't
  write into the real `output/` dir from tests); a test that
  `dev_dump_output=False` writes nothing; a test that a write failure
  (e.g. mock `open()`/`Path.write_text` to raise) is logged as a warning
  and does **not** prevent `producer.publish()` from being called
  afterward — this is the same "must not break the real pipeline" concern
  already established for other best-effort side effects in this
  codebase (spec revision-1 §6 is explicit about this).

- [ ] **Step 2: Implement it**
  Create the `output/` directory if missing (`Path.mkdir(parents=True,
  exist_ok=True)`), write via `result.model_dump_json(indent=2)` (matches
  the pretty-printing style `scripts/replay_sample_payload.py` already
  uses for local inspection), wrap the write in a try/except that logs and
  swallows any failure.

**Verify:** `pytest tests/test_main.py -q`, `ruff check src/main.py`,
`mypy src/main.py`.

---

## Task 7: Documentation and real-model verification

**Files:**
- Modify: `docs/testing-guidelines.md`
- Modify: `.env.example` (if Task 1 left anything doc-only out)
- No spec changes expected — revision-1 spec is written before this plan
  and should not need amending unless implementation surfaces a real
  surprise (if it does, amend the spec and note why, per this project's
  standing practice of keeping docs and code in sync)

- [ ] **Step 1: Update the manual testing guide**
  `docs/testing-guidelines.md`'s manual end-to-end section currently
  describes the in-process VLM step ("in-process, CPU, tanpa jaringan").
  Update it to describe the batched remote HTTP calls instead, and add
  `VLM_BASE_URL`/`VLM_MODEL`/`KEYFRAME_DIFF_THRESHOLD`/
  `FFMPEG_HWACCEL_CUDA`/`VLM_BATCH_ENABLED`/`VLM_BATCH_SIZE`/
  `DEV_DUMP_OUTPUT` to the prerequisites list where relevant.

- [ ] **Step 2: Real-model verification**
  Run `scripts/replay_sample_payload.py` against the real
  `qwen3-vl-8b-instruct-mlx` endpoint (already configured in the real,
  gitignored `.env` — do not commit real credentials into
  `.env.example`). Confirm: keyframe count is sensibly smaller than the
  old uniform-sampling count for the same video (evidence the
  difference-scoring is actually selective, not just passing everything
  through); batching produces the configured number of HTTP calls;
  `content` comes back non-empty for every batch (the specific failure
  mode found with a different remote reasoning-model during the
  `video-analyzer` evaluation — spec revision-1 §8/§9 — must be checked
  for here, not assumed absent just because it's a different model);
  `output/output-<id>-<timestamp>.json` is written; the published
  `AnalysisResult` still reaches Kafka unaffected.

- [ ] **Step 3: Full verification gate**
  `pytest -q`, `ruff check .`, `mypy src` all green — the same gate this
  project has run before every merge throughout its history.

**Verify:** all of the above, plus a final read-through confirming
`git grep -n "vlm_local\|vlm_model_id"` returns nothing under `src/`,
`tests/`, or `docs/` (no stale references left over from the deleted
in-process design).
