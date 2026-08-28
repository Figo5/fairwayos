# Automatic Golf Perception MVP Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a local-only, fixed-camera, single-golfer Automatic Golf Perception MVP that emits validated pixel observations and only reconstructs a `ShotEvent` when continuity, anchor, impact, ball/landing, and calibration gates pass.

**Architecture:** Keep the existing standard-library GhostCaddie analytics core unchanged. Add an optional AI adapter in the isolated `.venv-video-ai/` environment that produces the existing `video-observations.v1` pixel contract, diagnostics, and annotated media. The adapter must separate observed, inferred, and unavailable quantities; a gatekeeper must reject incomplete evidence before exactly-once calibration mapping and before the unchanged `run_pipeline()` call.

**Tech Stack:** Python 3.11 optional AI environment; OpenCV; PyTorch/MPS; Ultralytics pose/detection/tracking as an implementation baseline; classical optical flow and temporal filters; existing `VideoObservations`, `VideoCalibration`, reconstruction, human-import, annotation, and pipeline boundaries; `python3 -m unittest` and `compileall`.

---

## 1. Current state and non-goals

The accepted prior milestone is **environment-ready, not perception-validated**:

- `.venv-video-ai/` contains OpenCV, PyTorch, torchvision, Ultralytics, and tracking support.
- Official Ultralytics `yolo11n.pt` and `yolo11n-pose.pt` loaded on Apple M2 Pro MPS.
- The 480p YouTube stress clip produced generic person/pose results, but only 1/100 sports-ball detections and 24 person track IDs across 100 frames.
- The 480p artifact is a negative/stress case only and cannot be used as an acceptance sample.
- Current deterministic fixture and explicit human-annotation workflows remain authoritative.
- Current baseline must remain **233 tests passing**.

This milestone does **not** claim arbitrary broadcast-video support, multi-golfer support, autonomous calibration, reliable ball-flight reconstruction, or general-purpose club classification.

## 2. Required evaluation input before implementation acceptance

Obtain a separate consenting, high-resolution, single-shot evaluation clip. It must be locally available before the perception release gate can be evaluated.

Minimum clip requirements:

- 1080p preferred; 720p is the minimum acceptable fallback only if the ball and clubhead are visibly resolvable.
- Native frame rate at least 60 FPS preferred; 30 FPS is allowed only with an explicit impact-timing limitation.
- One golfer, one shot, one mostly fixed camera, no cuts, no zoom, no broadcast graphics over the golfer or ball path.
- Golfer visible from address through early ball flight; feet and lower body are not persistently occluded.
- Ball and clubhead visible for enough frames to evaluate them; if not visible, the test must record them as unavailable rather than fail open.
- Supplied four-point calibration for the exact video dimensions, or a separate calibration task must block analytics.
- A human-reviewed ground-truth annotation file for golfer box/track, feet anchor, swing phases, impact interval, ball positions when visible, clubhead positions when visible, and landing point when visible.
- Ground truth must include visibility/occlusion flags so unavailable evidence is not scored as a detector miss.

The existing `out/youtube_smoke_480p/` artifact remains a negative/stress fixture. It must be included in regression tests as a case that cannot produce an automatic `ShotEvent`.

## 3. Candidate comparison

### Candidate A: Generic YOLO detection + pose + classical tracking

**Strengths:** Already runnable locally; official weights are available; pose supplies body keypoints; Ultralytics tracking offers configurable ByteTrack/BoT-SORT; useful for a fast instrumentation baseline.

**Weaknesses:** COCO-style generic classes do not provide golf-specific clubhead semantics; sports-ball class is too coarse and tiny-object recall is inadequate; pose confidence and missed frames do not establish a stable feet anchor; generic tracking can create ID switches in broadcast scenes.

**Role:** Required baseline and fallback observation source, never sufficient by itself for automatic golf acceptance.

### Candidate B: Golf-specific pretrained weights

**Strengths:** Potentially better semantics for golfer, club, ball, clubhead, or swing phases without training from zero.

**Weaknesses:** No trusted, documented, locally available golf-specific weight set is currently established in this project. A public weight file is not acceptable merely because its filename suggests golf; provenance, license, architecture compatibility, training/evaluation data, and reproducible metrics must be verified.

**Role:** Investigate only if a documented source satisfies provenance and license requirements. It can replace or augment Candidate A after offline validation, but it is not an assumed dependency.

### Candidate C: Custom/fine-tuned golf detector

**Strengths:** Directly targets the required classes: golfer, club/clubhead, ball, and optionally feet/anchor landmarks. It is the only candidate that can plausibly satisfy the MVP semantics on the chosen capture setup.

**Weaknesses:** Requires labeled high-resolution clips, annotation effort, train/validation/test separation, reproducible training, model provenance, and evaluation beyond one clip. It will not generalize to arbitrary broadcast footage without broader data.

**Role:** Preferred minimum semantic detector for the MVP, starting with the narrow fixed-camera capture domain.

### Candidate D: Optical-flow and motion-based ball/club tracking

**Strengths:** Useful for sub-frame temporal refinement, detecting motion bursts, propagating a visible club/ball point between detector frames, and estimating camera motion from background features. It does not require a class model for every frame.

**Weaknesses:** Cannot distinguish golfer motion from club/ball reliably under blur, occlusion, shadows, grass texture, or camera motion; it accumulates drift and is especially fragile for tiny objects.

**Role:** Supporting temporal module only. It may refine or reject detector tracks but may not create a semantic ball, club, impact, or landing observation without detector evidence.

### Candidate E: Hybrid detector plus explicit human fallback

**Strengths:** Provides a truthful product boundary: automation is used when measurable gates pass, while difficult clips remain usable through an explicitly requested annotation workflow. It preserves current fixture and human workflows and avoids silent substitution.

**Weaknesses:** Requires clear provenance and UI/CLI status separation; human fallback is not automatic perception and cannot be used to claim detector performance.

**Role:** Required product architecture and release behavior.

### Minimum viable choice

Implement a hybrid of **Candidate C + Candidate A + Candidate D**, with Candidate E as the mandatory safety boundary:

1. A narrow custom/fine-tuned detector for golfer, ball, and clubhead where training data supports those labels.
2. Official pose only as a body-keypoint/anchor auxiliary signal, not as proof of golf semantics.
3. BoT-SORT or ByteTrack for golfer association, with camera-motion compensation enabled only after evaluation; retain a deterministic single-golfer track-selection rule.
4. Optical flow for short-gap propagation and background camera-motion estimation, with bounded gap length and confidence decay.
5. A deterministic temporal state machine for swing phase and impact candidate generation.
6. Explicit `unavailable` values and a hard gate before reconstruction.
7. Human annotation only when the caller explicitly requests fallback.

A generic-only implementation is not an acceptable MVP.

## 4. Proposed observation and provenance contract

Extend the existing `video-observations.v1` contract only through backward-compatible optional metadata or a versioned `video-observations.v2`; do not silently reinterpret existing fixture files.

Each frame-level quantity must carry:

- value, when observed or sufficiently supported;
- confidence in `[0, 1]`;
- provenance category: `detected`, `tracked`, `pose`, `flow_refined`, `inferred`, or `unavailable`;
- visibility/occlusion status;
- warning codes for blur, camera motion, low confidence, missing ball, missing clubhead, and track break.

Required sequence-level diagnostics:

- source dimensions and frame rate, without source URL or absolute local paths;
- selected golfer track ID and track continuity;
- detection/pose/ball/clubhead coverage;
- anchor stability and error metrics when ground truth is available;
- swing-phase confidence and transition indices;
- impact candidate frame and uncertainty interval;
- camera-motion and cut flags;
- false-positive counts;
- per-stage latency and memory;
- gate decisions and exact blocking reasons.

Unknown values must be serialized as `null` with the required warning/provenance, never as zero, guessed coordinates, or fabricated labels.

## 5. Release gates

These are proposed initial hard gates for the narrow fixed-camera MVP. They must be measured on a held-out high-resolution single-shot evaluation set, not tuned against the acceptance clip.

### Input/camera gates

- Exactly one golfer track candidate.
- No detected cut across the shot interval.
- Background camera motion median displacement below 2% of image diagonal per frame, or the sequence is marked `camera_motion` and blocked.
- No persistent golfer occlusion exceeding 0.25 s during address-to-impact.

### Golfer track continuity

Pass only if all are true:

- golfer detection/track coverage at least **95%** of frames from address through impact;
- longest uninterrupted gap no more than **3 frames**;
- ID-switch count **0** for a one-golfer clip;
- track-box center jitter after smoothing no more than **3% of image diagonal** during static address.

### Feet/body anchor

- Anchor availability at least **95%** of address-to-impact frames.
- Median feet-anchor error no greater than **2% of image diagonal** against held-out ground truth.
- 95th-percentile error no greater than **5% of image diagonal**.
- Anchor must be marked unavailable during occlusion or missed pose frames.

### Swing phase and impact candidate

- Address and finish/early follow-through phases must be identified when visible.
- Impact candidate must be an interval, not an unjustified exact frame.
- Median impact-frame absolute error no greater than **2 frames** at 60 FPS, with 95th percentile no greater than **4 frames**.
- If frame rate is 30 FPS, report the doubled temporal uncertainty and do not claim 60-FPS-level impact precision.
- No `contact` observation passes reconstruction unless its confidence and uncertainty gate pass.

### Ball detection/tracking

- Evaluate only frames marked ball-visible by ground truth.
- Ball detection precision at least **0.90** and recall at least **0.80** over visible frames.
- At least **90%** of accepted ball tracks must have no gap longer than 2 frames before landing/visibility loss.
- If the ball is not genuinely visible, ball values remain unavailable and reconstruction is blocked.

### Club/clubhead detection/tracking

- Evaluate only clubhead-visible frames.
- Precision at least **0.90** and recall at least **0.80** over visible frames.
- Clubhead track must cover at least 80% of the visible address-to-impact interval.
- Club name/classification remains unavailable unless it is supplied by context or a separately validated classifier; clubhead geometry must not be relabeled as club identity.

### Landing and reconstruction

- Landing point is required for analytics that use actual landing position.
- Median landing error no greater than **5% of image diagonal** or a domain-defined course-space threshold after calibration, whichever is stricter.
- No automatic `ShotEvent` if landing is not observed/reliably inferred, calibration is absent/invalid, or any required confidence gate fails.
- Exactly one calibration mapping operation per accepted source pixel quantity; test by injecting a counting mapper.

### False positives and runtime

- False-positive golfer/ball/clubhead detections below **5%** of evaluated visible-frame opportunities.
- Mean end-to-end inference latency no greater than **100 ms/frame** on the target M2 Pro/MPS environment at the chosen processing resolution; report p95 latency separately.
- Offline processing may exceed real time, but p95 latency must be recorded and bounded; no unbounded memory growth.
- Peak RSS must be recorded and remain below a documented project limit, initially **4 GB** for the isolated process.

Failure of any gate yields `automatic_status: blocked` and explicit unavailable fields. It must not invoke `run_pipeline()`.

## 6. Implementation milestones

Each milestone is vertical TDD: write one focused failing test, run it and confirm the expected failure, implement the minimum behavior, rerun the focused test, then run the full suite and compileall. Do not batch an unverified pile of tests.

### M0: Evaluation package and data contract

**Files likely to change:**

- Create: `docs/automatic-golf-perception-mvp.md`
- Create: `docs/automatic-golf-evaluation-protocol.md`
- Create: a project-owned evaluation manifest/annotation schema under `tests/fixtures/` or an approved local evaluation directory.
- Test: new contract tests alongside `tests/test_video_observations.py` or a dedicated `tests/test_automatic_perception_contract.py`.

**Work:** Define annotation schema, visibility rules, normalized metrics, provenance fields, and the high-resolution acceptance clip requirements. Do not add detector code yet.

**Validation:** Schema round-trip tests; reject fabricated fields; verify 480p stress artifact is marked non-acceptance.

### M1: AI adapter boundary and capability report

**Files likely to change:**

- Create: `ghostcaddie/video/automatic_perception.py`
- Create: `tests/test_automatic_perception_adapter.py`
- Modify only if needed: `ghostcaddie/video/__init__.py`, diagnostics exports.

**Work:** Add an optional adapter that checks the isolated runtime/model manifest, never changes the standard-library import path, and returns explicit unavailable status when weights or runtime are missing. Add deterministic configuration validation and sanitized diagnostics.

**Validation:** Tests run in the standard environment without importing optional dependencies; AI-environment smoke test runs separately; full 233-test baseline remains green.

### M2: Golfer detector, pose anchor, and single-track selector

**Files likely to change:**

- Modify: `ghostcaddie/video/automatic_perception.py`
- Create: `tests/test_automatic_golfer_tracking.py`
- Optional model-training/evaluation scripts outside core package under `tmp/` or a documented tooling directory.

**Work:** Integrate the selected detector and pose model; add deterministic one-golfer selection; calculate feet/body anchor from keypoints and/or detector geometry; add temporal smoothing and gap limits. Generic YOLO output must remain labeled generic unless the custom golf model is active.

**Validation:** Synthetic fixture tests for continuity, missing frames, ID switch, anchor bounds, and explicit unavailable values; real evaluation report against the high-resolution clip.

### M3: Ball and clubhead evidence modules

**Files likely to change:**

- Modify: `ghostcaddie/video/automatic_perception.py`
- Create: `ghostcaddie/video/temporal_tracking.py`
- Create: `tests/test_automatic_object_tracking.py`.

**Work:** Add custom/fine-tuned class outputs if approved weights/data exist. Add bounded optical-flow refinement only between trusted detections, with camera-motion rejection and confidence decay. Never promote flow-only points to accepted ball/clubhead evidence without the configured evidence policy.

**Validation:** Visible-vs-occluded fixture tests; ball/clubhead precision/recall report; stress-case report must contain unavailable ball/clubhead rather than guesses.

### M4: Swing phase and impact candidate

**Files likely to change:**

- Create: `ghostcaddie/video/swing_phase.py`
- Create: `tests/test_swing_phase.py`.

**Work:** Implement a deterministic state machine using validated golfer pose/anchor motion and clubhead evidence where available. Emit an impact candidate interval with confidence and uncertainty. Do not emit a definitive contact observation when only a generic motion spike exists.

**Validation:** Ground-truth frame-error metrics; tests for blur, missed pose, short gaps, and ambiguous impact; no contact value when gates fail.

### M5: Camera stability, cut detection, and sequence gate

**Files likely to change:**

- Create: `ghostcaddie/video/sequence_gates.py`
- Create: `tests/test_automatic_sequence_gates.py`.

**Work:** Compute background motion, cut signals, coverage, ID switches, false positives, and latency. Return a structured gate decision with per-gate reasons.

**Validation:** 480p broadcast stress case is blocked; fixed-camera acceptance clip can pass only if all thresholds pass; diagnostics contain no URL or absolute path.

### M6: Exactly-once calibration and guarded reconstruction

**Files likely to change:**

- Modify: `ghostcaddie/video/reconstruction.py` only through a narrowly tested automatic evidence entry point, preserving fixture/human semantics.
- Create: `tests/test_automatic_reconstruction_gate.py`.

**Work:** Convert accepted automatic pixel observations into the existing reconstruction boundary. Use a counting calibration stub to prove each source pixel is mapped exactly once. Do not alter `run_pipeline()`; invoke it only after reconstruction passes.

**Validation:** Required-evidence failure tests; calibration mismatch tests; exactly-once mapping test; successful path uses the unchanged pipeline and preserves existing provenance boundaries.

### M7: Annotated output and CLI integration

**Files likely to change:**

- Modify: `ghostcaddie/video/cli.py` and/or `ghostcaddie/cli.py` only after the gate API is stable.
- Create: `ghostcaddie/video/automatic_render.py`
- Create: `tests/test_automatic_video_cli.py`.

**Work:** Render annotated frames/video showing accepted detections, tracks, phase, impact interval, missing evidence, and gate failures. Keep `youtube-analyze` honest and preserve explicit `--fallback-human`. Never emit recommendation/normalized shot artifacts for blocked automatic analysis.

**Validation:** Visual inspection of the high-resolution acceptance output and blocked 480p output; CLI regression scenarios; no path/URL leakage.

### M8: Full evaluation and release decision

**Files likely to change:**

- Create: `docs/automatic-golf-evaluation-report.md`
- Create: sanitized machine-readable evaluation report under the approved output directory.

**Work:** Run the held-out evaluation protocol, report every metric and failure mode, compare against the gates, and decide pass/blocked. Do not broaden claims beyond the tested capture domain.

**Validation:** `python3 -m unittest discover -s tests`; `python3 -m compileall -q ghostcaddie tests`; all existing fixture, human-annotation, provider, session, and single-shot CLI scenarios; visual review of overlays/video; independent artifact/security inspection.

## 7. Verification commands

Use the project’s required test runner:

```bash
cd /Users/giofiore/ghostcaddie-tour
python3 -m unittest discover -s tests
python3 -m compileall -q ghostcaddie tests
```

Optional AI-environment checks must remain isolated:

```bash
.venv-video-ai/bin/python -c "import cv2, torch, torchvision, ultralytics; print(cv2.__version__, torch.__version__, torchvision.__version__, ultralytics.__version__, torch.backends.mps.is_available())"
```

Acceptance evaluation commands must operate on the separate high-resolution local clip and its supplied calibration/ground truth. The 480p YouTube artifact is a negative case only.

## 8. Risks and mitigations

- **No trusted golf-specific weights:** stop at M1 or M2 with explicit unavailable status; do not relabel generic detections.
- **Insufficient labeled data:** use the human annotation workflow to create a versioned local evaluation/training set only with user approval; never claim generalization from one clip.
- **Tiny ball/clubhead:** preserve null values and visibility flags; do not infer from language models or visual guesses.
- **Camera movement/cuts:** detect and block rather than compensating silently.
- **Calibration mismatch:** validate exact dimensions and four-point mapping before reconstruction.
- **Core regression:** keep optional imports lazy and preserve all existing test fixtures and human paths.
- **Runtime/memory instability on MPS:** record CPU fallback explicitly, bound frame count/resolution, and report the actual device used.
- **Overclaiming:** release language is restricted to the tested fixed-camera, single-golfer domain and only after every hard gate passes.

## 9. Open decisions before implementation

1. Provide or approve the separate consenting high-resolution single-shot clip and its four-point calibration.
2. Approve the annotation schema and whether the user will supply ground truth or authorize creation through the existing human annotation workspace.
3. Confirm whether custom fine-tuning is permitted locally and where the labeled dataset may be stored.
4. Approve the proposed numerical release gates or provide project-specific thresholds, especially landing error in course units and the maximum acceptable runtime/RSS.
5. Confirm whether automatic output should support only one fixed camera initially or also a tightly defined moving-camera extension after the fixed-camera gate passes.

No production implementation should begin until the evaluation clip/data decision and the minimum detector/tracker choice are approved.
