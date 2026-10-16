# Canonical video pipeline audit

**Audit date:** 2026-09-16
**Scope:** `ghostcaddie-tour-legacy` (branch `pga-research-analyzer`) perception modules,
the `local-tracker-restart` reference working tree at `~/ghostcaddie-tour`,
the upload/review service, and the local research media corpus.
**Status:** audit only. No canonical execution path was moved, deleted, rewritten, or
rewired. Stale `/tmp` worktrees and the untracked `uv.lock` were left in place.

This document answers one question: *if we declared a canonical body / ball / clubhead
pipeline today, what would we be declaring, and would it be honest?*

---

## 1. Summary of recommendations

| Target | Recommended canonical path | Confidence | Locality |
|---|---|---|---|
| **Body** | `BodyMoveNetAdapter` (`ghostcaddie/upload/adapters.py:103`) over MoveNet SinglePose Lightning TFLite, executed through `RuntimeRegistry` in the `.venv` interpreter | Adopt, with a named crop limitation | Fully local / offline |
| **Ball** | **UNAVAILABLE.** No path is safe enough to declare canonical | — | — |
| **Clubhead** | **UNAVAILABLE.** No path is safe enough to declare canonical | — | — |

The honest shape of the system today is **one canonical target, not three**. Anything
that presents body, ball and clubhead as peers of equal standing is misleading, because
only the body route has an unseeded, locally reproducible, non-falsified implementation.

---

## 2. Body path

### 2.1 Recommendation

Canonical: **`BodyMoveNetAdapter`**, MoveNet SinglePose Lightning v3 TFLite via
`ai_edge_litert`, dispatched out-of-process by `ghostcaddie/upload/runtimes.py` into the
`.venv` interpreter (Python 3.11, `ai_edge_litert` + `cv2` + `numpy`, **no torch**).

### 2.2 Evidence

- The adapter runs **automatically**: `"initialization": "automatic"`,
  `"assistance": "none; single-person model, no human seed"`. It is the only one of the
  four registered targets where `requires_reviewed_seed` is false
  (`runtimes.py:126`), i.e. the only one that can produce a result unattended.
- **No pickle surface.** MoveNet is a TFLite flatbuffer; the `weights_only=False`
  Ultralytics loader concern recorded in `docs/current-research-handoff.md` does not
  apply. `UltralyticsBodyAdapter` (`adapters.py:172`) is retained *specifically* as a
  permanently-blocked route so the reason stays visible — that is the right treatment
  and should not be "fixed".
- **Geometry is invertible and recorded.** The letterbox transform
  (`letterbox_scale`, `offset_xy`, `model_input_wh`) is serialized per frame, so
  native-resolution review is reconstructible rather than assumed.
- **Corpus coverage is real.** `out/demo_improvement_campaign/pexels_regression_corpus.json`
  records pose coverage `1.0` on every one of its ten Pexels rows (29–48 frames each),
  `ball_rejected: 0`, `false_ball_markers: 0`, `decode: true`.
- The `local-tracker-restart` reference tree independently converged on the same model
  (`body.py`, MoveNet SinglePose TFLite, `MIN_SCORE = 0.3`, keypoints below threshold
  dropped rather than drawn at a guessed position). Two independent efforts picking the
  same safe local model is the strongest signal in this audit.

### 2.3 Failure modes

1. **MoveNet is single-pose and the legacy adapter feeds it the whole letterboxed
   frame.** `_run()` letterboxes the full frame and invokes once. On a broadcast wide
   shot with spectators, a caddie, or a camera operator, MoveNet will return *one*
   skeleton with no guarantee it is the player. The restart tree solves this correctly
   (`body.py:square_crop` around a per-frame person box) — but it gets that box from
   `vision.py`, which is a **network** call. So the crop discipline is currently
   inseparable from the network dependency.
2. **Pose jitter is not bounded.** The regression corpus shows `jitter_median` from
   8.92 px (`38053415`) up to **140.23 px** (`6573534`) and 53.82 px (`37995798`). Two
   of ten clips are visibly unstable; there is no gate that rejects them.
3. **Coverage 1.0 is not identity.** Every row is `ground_truth: false`,
   `status: research_only`. 46/46 pose observations means MoveNet returned a skeleton on
   every frame, not that the skeleton was on the player.
4. **The model file lives in `/tmp`** — see §6.

### 2.4 Verdict

Adopt as canonical **with the whole-frame limitation stated in the output**, not
silently. The correct next step is a *local* person-box stage, not the network one.

---

## 3. Ball path — UNAVAILABLE

### 3.1 Why no path qualifies

Four candidate routes exist. Every one is disqualified:

| Route | Module | Disqualifier |
|---|---|---|
| Heuristic circle/contrast proposals | `research_ball.py:ResearchBallTracker` | Falsified on the accepted artifact: the track followed the flagstick base, not a ball |
| Human-seeded appearance track | `research_ball.py:SeededBallTracker`, driven by `research_ball_track.py` | Requires a human seed per clip; never unattended |
| BootsTAPIR learned point tracker | `BallTapirAdapter` (`adapters.py:487`) | Falsified *and* weights are gone (§6) |
| SAM2.1 video segmentation | `BallSam2Adapter` (`adapters.py:451`) | Promptable segmenter, needs a seed box; checkpoint is gone (§6) |

### 3.2 Evidence

- **The heuristic route was actively withdrawn.** `STATUS.md` records that on the
  accepted Pexels `6573485` artifact, ball observations are "explicitly unavailable on
  all 121 frames after the 2026-09-01 ball-plausibility correction … the earlier ball
  track followed background texture (the flagstick base), not a golf ball". The rejected
  run is retained as evidence under
  `out/research_training_gauntlet/fairwayos_unified_pexels_6573485_pre_ball_plausibility/`.
  That is a correct scientific outcome and must not be quietly reversed.
- **The learned route was falsified under a source-binding bug.**
  `docs/current-research-handoff.md`: Morikawa and Gotterup BootsTAPIR runs were named
  as separate clips but consumed the same Lipsky bytes (`61e852fb…c1ef6`). All
  cross-clip/transfer claims from those runs are **withdrawn**. The corrected reruns
  gave Lipsky genuine, **Morikawa 1/51, Gotterup 0/71** — the point tracker does not
  transfer under frozen settings.
- The durable fix, `ghostcaddie/video/source_binding.py`, is sound and should be a
  mandatory precondition of any future canonical path: it fails closed before model
  import unless media bytes match the caller's expected SHA-256.
- `yolo11n.pt` COCO `sports ball` output is explicitly *not* golf-ball evidence
  (`STATUS.md`, `docs/clubhead-tracking-local-research-probe.md`): 124 `sports ball`
  hits on MMU alongside 25 `clock` and 2 `wine glass`.

### 3.3 Failure modes to carry forward

- Background texture lock (flagstick base, shoes, logos, signage).
- Byte-level source confusion producing fake transfer wins — mitigated by
  `bind_source()`, which must be called *before* model import.
- Seed-frame support being counted as predictive coverage.

### 3.4 Verdict

**UNAVAILABLE.** Declaring a canonical ball path today would re-promote a track that
has already been shown to follow scenery.

---

## 4. Clubhead path — UNAVAILABLE

### 4.1 Evidence

- **MIL region tracking does not transfer.** `out/mil_run/mil_summary.json`, frozen
  defaults, no config changes between dev and transfer:
  - dev clip `pexels_33511561` (f145–200): 16 emitted frames, **all 16 independently
    QA'd on-clubhead**. A genuine segment.
  - dev clip `pexels_6541855` (f128–156): MIL emitted **only the seed frame**. Zero
    predictive coverage on a near-side 30 fps downswing.
  - transfer clip `pexels_6541842` (f25–51): `"transfer_status": "FAILED"` — 7 frames
    on-clubhead, then **16 consecutive frames locked to the wrong object on the body**.
  - Per-frame records confirm it: `pexels_*_mil_perframe.json` shows **0 observed**
    across all three clips after honest state accounting, against baseline
    `reacquire_color` at 32/56, 38/46 and 15/29 — but that baseline is the drifting one.
- **The baseline's extra frames are wrong frames.** `out/mil_run/baseline_qa_verdicts.json`
  on `pexels_6541855`: f140 `on_shaft`, f143 `on_ground`, f146 `on_ground`,
  f149 `on_other`, f152 `on_other`, f155 `on_ground`. Only 2 sampled verdicts across the
  whole set are `on_clubhead`, both on `33511561`. Higher observed-count is
  anti-correlated with correctness here.
- **Appearance post-filters were decisively killed.** `docs/current-research-handoff.md`:
  "HOG was decisively killed by a second false-lock object; HSV did not establish a
  positive generalizable case. **Do not retune thresholds to revive this.**"
- **SAM2 head segmentation is partial/emerging only.** f310–f343 full, f344–f360
  partial/emerging. "Partial mask centroids are not head centers, material points, or
  trajectories."
- **No detector exists to train against.** `docs/clubhead-tracking-local-research-probe.md`:
  `clubhead_available: false`, `impact_available: false`. `yolo11n-pose.pt` produced
  **0/165** frames of output on MMU. ClubheadDB is CC-BY-NC-4.0 with third-party
  YouTube/Reddit source media and no checkpoint; GolfDB/SwingNet gives swing events, not
  coordinates; GolfPose is authorization-gated.
- **Source adequacy is a hard bound**, not a tuning problem: clubhead-through-impact at
  the available frame rates exceeds the tracker design point — required inter-frame
  displacement is larger than the search radius.

### 4.2 Verdict

**UNAVAILABLE**, and unlike the ball this is blocked at the *data* layer, not just the
implementation layer. `tests/test_detector_training_feasibility.py` already encodes the
right framing: the blocker is the pixels and the dataset rights, not a missing
checkpoint.

---

## 5. Local/offline versus network-dependent

| Path | Locality | Notes |
|---|---|---|
| `BodyMoveNetAdapter` (MoveNet TFLite) | **Local, offline** | Local flatbuffer; `RuntimeRegistry` downloads nothing |
| `research_ball.py` heuristics, `clubhead_methods.py`, `hybrid_tracking.py`, `seeded_tracking.py`, `region_tracker_mil.py`, `temporal_candidate_gate.py`, `stationarity_gate.py`, `window_selection.py` | **Local, offline** | Pure OpenCV/NumPy; no network at any point |
| `BallSam2Adapter`, `ClubheadSam2Adapter`, `BallTapirAdapter` | Local **inference**, but weights must be fetched once | Currently un-runnable (§6) |
| `ghostcaddie/video/perception.py:OllamaPerceptionAdapter` | **Network — loopback only** | `http://127.0.0.1:11434`; opt-in (`enabled=False` default), rejects `:cloud` model names. Local-machine network, not internet |
| `ghostcaddie/video/fresh_ai_vision.py` | **Network — external paid API** | Spawns bounded `hermes` child processes; consumes account quota |
| `ghostcaddie/video/ai_demo.py` (1456 lines) | Mixed | Orchestration around the above |
| `local-tracker-restart` `vision.py` | **Network — external paid API** | One `hermes`/`gpt-5.5` subprocess **per frame**, 120 s timeout |
| `local-tracker-restart` `body.py`, `camera.py`, `launch.py`, `analytics.py`, `approx_analytics.py` | **Local, offline** | |
| `ghostcaddie/video/youtube.py`, `youtube_auto_try.py`, `pga_source_gate.py` | **Network — acquisition** | Must never run in CI |

The restart tree's README states the trap plainly and correctly: *"Local video input does
not mean offline inference."* That sentence should survive into any canonical
documentation.

---

## 6. Architectural blockers

These are the findings that would make a canonical pipeline declaration actively
misleading today.

### B1 — Runtime and model paths are hardcoded to `/tmp`, and `/tmp` has already been purged

`ghostcaddie/upload/adapters.py`:

```
adapters.py:107  body      → /tmp/fairway-safe-body-runtime/movenet-singlepose-lightning-v3.tflite
adapters.py:176  body_legacy → ~/ghostcaddie-tour/yolo11n-pose.pt
adapters.py:189  clubhead  → /tmp/fairway-sam-head/checkpoints/sam2.1_hiera_tiny.pt
adapters.py:461  ball_sam2 → /tmp/fairway-sam-head/checkpoints/sam2.1_hiera_tiny.pt
adapters.py:490  ball      → /tmp/fairway-learned/tracker/weights/bootstapir_checkpoint_v2.pt
runtimes.py:92   ball interpreter → /tmp/fairway-learned/tracker/.venv/bin/python
```

Observed on disk at audit time:

- `/tmp/fairway-sam-head/checkpoints/` — **empty** (directory mtime `Sep 16 00:00`)
- `/tmp/fairway-learned/tracker/weights/` — **empty** (same mtime)
- `/tmp/fairway-sam-head/{masks,overlays_full,frames_310_360}` — **empty**
- `/tmp/fairway-safe-body-runtime/movenet-singlepose-lightning-v3.tflite` — **present**,
  9,372,828 bytes, `Sep 11 15:15`

The macOS periodic `/tmp` cleaner ran at `Sep 16 00:00` and removed the large
checkpoints, leaving the directory skeletons behind. The body model survived by luck.
Three of four registered targets are un-runnable right now, and **this will happen
again**. `body_legacy_ultralytics` additionally hardcodes an absolute path into the
*other* repo's working tree.

Any canonical pipeline whose model resolution depends on `/tmp` is not a pipeline, it is
a cache that happens to still be warm. Note that `local-tracker-restart` already solved
this: `body.py:model_path()` resolves `--body-model` → `FAIRWAY_BODY_MODEL` → default,
and `load_model()` checks file existence *before* the runtime import so a missing file is
reported as a missing file.

### B2 — Four interpreters, none of which can run the whole thing

| Interpreter | Python | Has |
|---|---|---|
| `.venv` | 3.11.16 | `ai_edge_litert`, `cv2` 5.0.0, `numpy` — **no torch** |
| `.venv-video-ai` | 3.11.16 | `torch`, `cv2`, `ultralytics` — **no `sam2`, no `tapnet`, no `ai_edge_litert`** |
| `.venv-video-modern` | 3.11.16 | no `cv2` |
| system `python3` | 3.9.6 | no `cv2` |

The out-of-process `RuntimeRegistry` design is the right call and should be kept — it is
what keeps torch and pickle out of the serving process. But `.venv-video-ai` is missing
both `sam2` and `tapnet`, so even if the checkpoints were restored, the clubhead and ball
interpreters would still fail their probes.

### B3 — `/ready` reports capability the system cannot deliver

`runtimes.py:readiness()` is careful and honest (`interpreter_ready`,
`model_file_present`, `requires_reviewed_seed`, `can_execute_now` reported separately).
But the practical result today is `can_execute_now: true` for exactly one target. A UI
that lists four targets implies four capabilities.

### B4 — The upload service holds jobs in `awaiting_seeds` for targets that can never run

`server.py:_run_job` correctly stops at `awaiting_seeds` and keeps media readable for
frame review. But with no SAM2/TAPIR checkpoint, every seeded target resolves to
`UNAVAILABLE` before the human is ever asked. The flow is sound; its preconditions are
not met.

### B5 — Two repositories, both live, with no declared relationship

`ghostcaddie-tour-legacy` (97 Python modules, 18,244 lines under `video/` + `upload/`)
and `local-tracker-restart` (7 modules, 2,564 lines) both implement body pose from
MoveNet. Neither imports the other. Declaring "the canonical pipeline" without first
declaring which repo owns it will produce two canonical pipelines.

---

## 7. Determinism and portable testing

### 7.1 Per-path determinism

| Path | Deterministic? |
|---|---|
| `FixturePerception` / `load_fixture_observations` | **Yes** — checked-in JSON, no inference. This is the existing deterministic backbone |
| `automatic_perception.py` contracts (`PixelTrackStore`, `continuity_metrics`, `GateDecision`, `precision_recall`, `SwingPhaseStateMachine`) | **Yes** — no model or runtime imports by design |
| `clubhead_proposal.py`, `temporal_candidate_gate.py`, `source_binding.py`, `team_layer_integration.py`, `siwoo_layers.py` | **Yes** — pure functions over supplied geometry |
| MoveNet TFLite inference | **Yes, given the weights** — fixed flatbuffer, fixed letterbox, no sampling |
| OpenCV heuristics (`research_ball`, `clubhead_methods`, `region_tracker_mil`) | **Mostly** — but `TrackerMIL` adapts online and `region_tracker_mil.py` is bounded by a **wall-clock budget**, which makes its stopping point machine-dependent |
| `fresh_ai_vision.py`, restart `vision.py` | **No** — remote LLM, wall-clock budgets, `ThreadPoolExecutor`, per-frame subprocess timeouts. `detect_repeated_decisions()` exists precisely because the model sometimes repeats itself |
| `OllamaPerceptionAdapter` | **No** — `temperature: 0` helps but does not guarantee reproducibility across model versions |

Portable testing is viable for everything above the LLM line. The module authors were
disciplined about this: `automatic_perception.py` opens with *"This module intentionally
contains no model/runtime imports."*

### 7.2 The portable-test-suite problem

Measured at audit time:

| Interpreter | Result |
|---|---|
| system `python3` 3.9.6 | **890 tests, 5 failures, 97 errors**, 19 skipped |
| `.venv-video-ai` 3.11.16 | **988 tests, 1 failure**, 1 skipped, 211 s |
| `local-tracker-restart` `.venv-tracker` | **198 passed, 1 skipped, 10.4 s** |

Three separate problems are tangled together here, and they should not be:

1. **The 98 fewer tests and 97 errors on 3.9 are an environment artifact, not a
   regression.** `pyproject.toml` declares `requires-python = ">=3.9"`, but the suite
   needs `cv2`, and ~100 tests fail collection or error on `ModuleNotFoundError: cv2`
   (`tests/test_research_ball` alone raises 15 × *"SeededBallTracker requires OpenCV for
   its HSV appearance model … install opencv-python-headless instead of silently
   treating RGB values as pseudo-HSV"*). The `requires-python` floor is wrong, and there
   is no environment marker telling a fresh checkout which interpreter to use.

2. **One test is hard-bound to the deleted SAM2 checkpoint.**
   `tests/test_upload_product_gaps.py:459`
   (`test_readiness_tells_the_operator_how_to_make_a_seeded_target_run`) asserts that the
   readiness reason contains `/jobs/<id>/seeds`. It now reads
   `model file not found: /tmp/fairway-sam-head/checkpoints/sam2.1_hiera_tiny.pt`. This
   is the **only** failure in the good interpreter. The test is not wrong about the
   product behaviour it wants; it is wrong to assume a `/tmp` checkpoint exists. It
   should either inject a fake model path or assert the seed guidance *conditional on*
   `model_file_present`.

3. **`tests/test_ball_sam2_records.py` gets this right and is the model to copy.** Its
   record-shape tests run everywhere; only the execution test calls `self.skipTest(f"SAM2
   not runnable here: {cap.reason}")`. Its docstring is explicit that it deliberately
   does *not* assert SAM2 finds the ball. That separation — portable contract tests
   always run, weight-dependent execution tests skip with a stated reason — is exactly
   the pattern the rest of the suite needs.

The consequence: **there is currently no single command that a fresh machine can run to
get a truthful pass/fail.** The suite's real state is "988 tests, 1 environment-caused
failure", but a newcomer running `python3 -m unittest` gets 97 errors and would
reasonably conclude the repo is broken.

---

## 8. Media corpus: are the Pexels clips sufficient?

**Correction to the framing:** the local regression corpus holds **ten** Pexels clips,
not eight. `out/demo_improvement_campaign/pexels_regression_corpus.json` lists ten rows
and all ten MP4s are present in
`out/demo_improvement_campaign/development/pexels_round4/`
(`34883764`, `37030686`, `37080178`, `37995798`, `38053415`, `6541674`, `6541682`,
`6541964`, `6573534`, `6573621`). Eight is the count that survives excluding the two
high-jitter clips (`6573534` at 140.23 px, `37995798` at 53.82 px) — if that was the
intended set, the exclusion is defensible but it is *selection on the metric being
evaluated*, and a validation set chosen that way cannot then be used to report that
metric.

Separately, twenty-three distinct `pexels_*` IDs appear across `out/`, so the round-4
corpus is not the whole local Pexels footprint.

**Sufficient for:** body-pose stability regression. Ten clips × 29–48 frames, all with
recorded SHA match, license, decode status and research flags, is a reasonable
smoke/regression corpus for detecting *changes* in pose behaviour.

**Not sufficient for the next validation phase**, for four reasons:

1. **No ground truth.** `"ground_truth": false` at the corpus root and
   `"luna_score": null` on every row. Every metric is self-reported coverage. You cannot
   compute precision or recall against it, so it cannot validate anything — only detect
   drift.
2. **No held-out split.** Every row lives under `development/`. `frozen_holdout/`
   contains one subject (`bobby_demo`, `bobby_demo_v2`) and `evaluation_input/` contains
   one Tommy Fleetwood clip. A corpus that has already driven iteration is
   development-consumed; the handoff document already flags exactly this failure mode
   for the repaired TAPIR reruns.
3. **Wrong content for ball and clubhead.** These are amateur/stock swing clips. Every
   row has `ball_rejected: 0` and `false_ball_markers: 0` because the ball path is
   disabled, not because it succeeded. Clubhead-through-impact needs frame rates these
   clips do not have — that is already recorded as a hard source-adequacy bound.
4. **Rights asymmetry.** Pexels clips are retained locally and usable for research;
   PGA TOUR material under `out/pga_official_acquisition/` is gated by
   `pga_source_gate.py` and cannot be published. Any validation set mixing the two has
   two different publication rules inside one number.

**What the next phase actually needs:** a frozen, never-iterated-on holdout of ≥20 clips
with per-frame human body-keypoint labels on a sampled subset; and for any ball or
clubhead claim, rights-cleared paired annotations that do not exist yet in any form.
`research_split.py` already validates frozen split manifests — the contract is built, the
data is not.

---

## 9. What should remain experimental

Keep these as research modules; do not let them near a canonical path.

| Module | Reason |
|---|---|
| `luna_roi_experiment.py` | Labels come from a vision model; explicitly not production observations |
| `clubhead_pseudo_labels.py` | Pseudo by construction |
| `tracer_weak_supervision.py` | Broadcast tracer graphics are hints, not ball positions |
| `region_tracker_mil.py`, `mil_overlay.py`, `mil_corrected_run.py` | Transfer FAILED on the third clip with frozen settings; retain as evidence |
| `clubhead_methods.py` (`_region_template_track`, `_reacquire_color_track`, `_metal_relock_candidate`) | Baseline QA shows on-ground/on-shaft/on-other verdicts dominating |
| `hybrid_tracking.py`, `seeded_tracking.py` | Require human seeds; never unattended |
| `research_ball.py`, `research_ball_track.py`, `research_ball_model.py`, `pga_sliced_ball.py` | Falsified on the accepted artifact |
| `fresh_ai_vision.py`, `ai_demo.py` | Network, non-deterministic, quota-consuming |
| `OllamaPerceptionAdapter` | Opt-in by design; `enabled=False` is correct and should stay |
| `pga_research_analyzer.py`, `pga_analyze.py`, `pga_fallback.py`, `siwoo_layers.py` | Rights-gated; publication blocked |
| `stationarity_gate.py`, `temporal_candidate_gate.py` | Correct and useful, but they are *filters on proposals*, not detectors. Promote only alongside a detector worth filtering |
| `training/feasibility.py` | Encodes why training is blocked; keep as the reason, not as a step |

Promote-worthy *contracts* (deterministic, dependency-free, already tested) that a
canonical pipeline should sit on top of: `source_binding.py`, `automatic_perception.py`,
`clubhead_proposal.py`, `team_layer_integration.py`, `observations.py`, `contracts.py`,
`paths.py`, `errors.py`.

---

## 10. Upload / UI pieces that should wrap the future canonical workflow

Keep and build on:

- **`ghostcaddie/upload/runtimes.py`** — out-of-process interpreter probing. The single
  best architectural decision in the codebase: the serving process never imports torch,
  and "blocked" means unsafe, never "this interpreter lacks the module"
  (`server.py:_plan_from_isolated_interpreters`). Keep verbatim; only fix the paths.
- **`ghostcaddie/upload/targets.py`** — the four-state `TargetOutcome`
  (`OBSERVED` / `UNAVAILABLE` / `BLOCKED` / `NOT_RUN`). Collapsing these into a boolean
  is how a canonical pipeline starts lying.
- **`ghostcaddie/upload/server.py` phase-1 hold** — running unseeded targets, then
  stopping at `awaiting_seeds` with media still readable, is the correct shape for any
  human-in-the-loop step.
- **`ghostcaddie/upload/ui.py`** — *"There are no placeholder numbers: a target with no
  result renders its reason, never a plausible-looking figure."* This is the UI contract
  the canonical pipeline needs.
- **`ghostcaddie/upload/validation.py`**, `jobs.py` retention/cancel, `observations.py`,
  `seeds.py` (seed bundles bound to the uploaded source SHA), and the
  `/ready` + `/runtime` endpoints.
- **`tools/compare_video.py`** (restart tree) — decoded-pixel comparison sheets that do
  not reconstruct prediction markers. This is the review tool; port it.

Do **not** wrap the canonical workflow in: `ai_demo.py` orchestration, the YouTube
acquisition CLIs, or any `pga-research-demo` path.

---

## 11. Proposed canonical pipeline

```
                      ┌─────────────────────────────┐
   local video file   │  bind_source()              │  fail closed if bytes
   ──────────────────►│  source_binding.py          │  ≠ expected SHA-256
                      │  BEFORE any model import    │
                      └──────────────┬──────────────┘
                                     │ verified bytes
                      ┌──────────────▼──────────────┐
                      │  extraction.py              │  ffmpeg decode,
                      │  native frames + timestamps │  source-frame indices kept
                      └──────────────┬──────────────┘
                                     │
                      ┌──────────────▼──────────────┐
                      │  RuntimeRegistry.readiness()│  probe interpreters
                      │  → TargetPlan per target    │  + model files, per target
                      └──────────────┬──────────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                          │
 ┌────────▼────────┐        ┌────────▼────────┐        ┌────────▼────────┐
 │ BODY            │        │ BALL            │        │ CLUBHEAD        │
 │ MoveNet TFLite  │        │                 │        │                 │
 │ .venv, no torch │        │  UNAVAILABLE    │        │  UNAVAILABLE    │
 │ automatic       │        │  (falsified)    │        │  (no detector,  │
 │ subprocess      │        │                 │        │   source-bound) │
 └────────┬────────┘        └────────┬────────┘        └────────┬────────┘
          │  keypoints + invertible   │  reason string           │  reason string
          │  letterbox transform      │                          │
          └──────────────────────────┬┴──────────────────────────┘
                                     │
                      ┌──────────────▼──────────────┐
                      │  automatic_perception.py    │  continuity, gates,
                      │  + temporal_candidate_gate  │  no promotion
                      └──────────────┬──────────────┘
                                     │
                      ┌──────────────▼──────────────┐
                      │  team_layer_integration.py  │  independent layers;
                      │  layer handoff validation   │  unavailable stays unavailable
                      └──────────────┬──────────────┘
                                     │
                      ┌──────────────▼──────────────┐
                      │  renderer                   │  clean-frame copy in,
                      │  overlay onto SEPARATE copy │  overlays onto a second copy
                      └──────────────┬──────────────┘
                                     │
                      ┌──────────────▼──────────────┐
                      │  upload/ui.py               │  every absent target shows
                      │  reason, never a placeholder│  WHY, with no number
                      └─────────────────────────────┘

   research_only=true · pseudo_label=true · ground_truth=false ·
   production_eligible=false   ── carried on every record, end to end
```

Two properties this diagram enforces that the current code does not: the ball and
clubhead branches terminate in a **reason string**, not in an optional result; and model
resolution happens through `RuntimeRegistry`, never through a literal `/tmp` path.

---

## 12. Implementation plan (not implemented)

Ordered so that each step is independently reviewable and nothing is rewired until the
foundations hold.

**Phase 0 — make the truth measurable** *(no behaviour change)*
1. Raise `requires-python` to `>=3.11` in `pyproject.toml` and document the
   `.venv` / `.venv-video-ai` split in `README.md`. Record the measured baselines: 988
   tests / 1 failure on `.venv-video-ai`; 198 / 1 skip on the restart tree.
2. Fix `tests/test_upload_product_gaps.py:459` to inject a model path or condition its
   assertion on `model_file_present`, so the suite stops depending on a `/tmp`
   checkpoint. Target: **0 failures** on `.venv-video-ai`.
3. Apply the `test_ball_sam2_records.py` pattern to every weight-dependent test:
   portable contract assertions always run; execution assertions `skipTest` with the
   capability reason. Target: a meaningful pass/fail on a bare 3.11 + cv2 checkout.

**Phase 1 — kill the `/tmp` dependency** *(blocker B1)*
4. Introduce explicit model resolution — CLI flag → environment variable → configured
   path, existence checked before the runtime import — copying `body.py:model_path()` /
   `load_model()` from the restart tree. Applies to all five `default_model` constants
   and the `ball` interpreter path in `runtimes.py:92`.
5. Move surviving weights out of `/tmp` into a gitignored, documented model root. Record
   SHA-256 for each. Document the restore procedure for the two lost checkpoints —
   including whether restoring them is even wanted, given §3 and §4.
6. Remove the absolute cross-repo path at `adapters.py:176`.

**Phase 2 — declare the body path canonical** *(one target, honestly)*
7. Add a local person-box stage ahead of MoveNet so `square_crop` discipline is
   available without the network. This is a real research task, not plumbing: no safe
   local person detector is currently cleared (Ultralytics is blocked on loader
   grounds). Until it lands, emit an explicit `whole_frame_single_pose` limitation on
   every body record.
8. Add a jitter gate using the corpus numbers as the reference distribution — the
   140.23 px and 53.82 px clips must be *rejected*, not silently rendered.
9. Wire body through the Phase-1 diagram end to end, with `bind_source()` first.

**Phase 3 — make the absences first-class**
10. Ensure the ball and clubhead branches always emit `TargetOutcome.UNAVAILABLE` plus a
    stable reason code, and that `ui.py` renders them. No configuration should be able
    to turn a heuristic back on as a default.
11. Fold `temporal_candidate_gate` and `stationarity_gate` in as filters that can only
    ever *remove* candidates.

**Phase 4 — decide the repo question** *(blocker B5)*
12. Pick one home. Recommendation: the restart tree's **shape** (7 plain modules, 198
    tests in 10 s, configurable model paths) with the legacy tree's
    **contracts** (`source_binding`, `RuntimeRegistry`, `TargetOutcome`,
    `team_layer_integration`) ported in. Do not merge the 18k lines of research modules —
    leave them where they are, as evidence.

**Not in scope until a validated detector exists:** calibration, course coordinates,
`ShotEvent`, carry/apex/landing, production analytics, recommendations. These stay null.

---

## 13. Open questions for review

1. Was the intended Pexels validation set eight clips or ten? If eight, which two were
   dropped and on what criterion (§8)?
2. Should the SAM2 and BootsTAPIR checkpoints be restored at all, given both routes are
   already falsified — or should the adapters be demoted to `BLOCKED` with the
   falsification as the reason?
3. Which repository is the canonical home (§12 step 12)?
4. Is a local, license-clear person detector acquirable? Phase 2 step 7 is the only item
   on the critical path that is not plumbing.
