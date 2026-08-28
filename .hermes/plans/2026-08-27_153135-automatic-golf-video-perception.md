# Automatic Golf Video Perception Milestone Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add an optional, local-only automatic perception path for one fixed-camera golf shot that detects and tracks golfer/club/ball evidence, reconstructs a shot event with explicit uncertainty, runs the unchanged GhostCaddie analytics core, and produces a viewable annotated result without weakening the accepted human fallback.

**Architecture:** Keep the existing standard-library analytics core, fixture mode, human annotation path, `run_pipeline()`, and exactly-once coordinate-mapping boundary unchanged. Add a dependency-isolated computer-vision adapter that converts frame pixels into a versioned observation sidecar, a temporal event/tracking layer that consumes only validated observations, and a renderer that keeps overlays in original pixel space while course-space analytics remain stable. Automatic results are accepted only when evidence and confidence gates pass; otherwise the command emits explicit unavailable fields and a human-review handoff rather than fabricated positions.

**Tech Stack:** FFmpeg/ffprobe for media I/O; OpenCV for frame access, image operations, optical flow, homography support, and encoding integration where useful; a local detector runtime (initially Ultralytics YOLO with a project-owned/custom golf-ball and club model, with ONNX Runtime as a later portable backend); ByteTrack or BoT-SORT-style association; optional pose/keypoint model for golfer feet/anchor; NumPy/OpenCV numerical operations isolated from the core package. No cloud calls, generic chat interpretation, remote assets, or identity recognition.

---

## 1. Findings from GhostBall-Engine and golf-specific adaptation

GhostBall-Engine is a small prototype, not a validated production reference. Its README and source show the useful architectural concepts: FFmpeg/OpenCV video handling, YOLO player/ball detections, ByteTrack-style temporal IDs, feet/contact-with-ground anchors, a homography from camera pixels to stable pitch coordinates, a separate kinematic engine, raw-pixel rendering, and re-projection of tactical output back to video.[1][2][3]

The golf adaptation should preserve the same separation but replace football entities and objectives:

```text
local video
  -> decoded frames / timestamps
  -> golfer, club, clubhead, ball detections
  -> temporal tracks + camera-motion estimate
  -> swing phases / contact candidate / flight candidate
  -> pixel-space observations with confidence and provenance
  -> one validated pixel-to-course calibration
  -> existing ShotEvent adapter
  -> unchanged run_pipeline()
  -> recommendation in course space
  -> inverse re-projection for pixel-space overlay
  -> annotated frames / sampled annotated video
```

The most important difference is observability: a golf ball can be only a few pixels wide, the clubhead can disappear under motion blur, impact may occur between frames, and landing/roll may be outside the camera view. The system must model these as missing evidence, not infer them silently.

## 2. Scope and non-goals

### MVP scope

- One trimmed golf shot per input video.
- Fixed, mostly stationary camera; tripod or stable phone placement.
- One visible golfer; no identity recognition.
- Automatic golfer bounding box and feet/ground anchor.
- Automatic club/shaft/clubhead evidence where visible.
- Automatic ball-before-impact and ball-after-impact evidence where visible.
- Temporal tracking and swing phase/contact-frame estimation.
- A required one-time four-point image-to-course calibration resource for the first MVP; automatic course-line calibration is a later milestone.
- Conversion to the existing `ShotEvent` only after required evidence and confidence gates pass.
- Existing expected-strokes, dispersion, wind, hazard, and decision analysis via unchanged `run_pipeline()`.
- Annotated sampled frames and a viewable sampled annotated video.

### Explicit non-goals for MVP

- Reliable arbitrary-video analysis.
- Broadcast-camera or heavy pan/zoom support.
- Ball spin, launch monitor-grade speed, carry, apex, or 3D trajectory claims.
- Identity, pose coaching, club-brand recognition, or player re-identification.
- Cloud inference or uploading user footage.
- Automatic calibration from arbitrary course imagery.
- Silent fallback to fabricated coordinates or human-like certainty.

## 3. Compare implementation options

| Option | Proposed use | Benefits | Risks / decision |
|---|---|---|---|
| Local YOLO/OpenCV detector + tracker | **MVP baseline** | Fits FFmpeg/OpenCV workflow; local and inspectable; YOLO detects objects while ByteTrack/BoT-SORT associates detections over time.[1][2] | Generic weights will be weak for tiny balls and clubheads; requires a golf-specific labeled dataset and dependency installation. Use custom classes/models, not COCO-only claims. |
| Available pretrained vision models | Phase 2 experiments | Faster bootstrap for golfer/pose/swing phases; GolfDB/SwingNet provides a golf-swing event baseline, but its published setup uses trimmed clips and reports PCE rather than guaranteeing arbitrary-video performance.[4] | Domain mismatch, license/weight provenance, temporal resolution, and unsupported ball/club classes. Treat as optional evidence providers behind a common adapter. |
| Hybrid CV + vision-model verification | Phase 3 | Detector/tracker supplies geometry and temporal continuity; a pose/phase model verifies golfer anchor and swing timing; disagreement becomes uncertainty. | More dependencies and calibration complexity. Use only after deterministic tracker contracts and evaluation fixtures exist. |
| Human annotation fallback | **Required safety path** | Handles occlusion, unseen landing, blur, poor FPS, and failed calibration; already implemented and accepted. | Not automatic; must be presented as review/fallback, never as automatic success. |

Recommended architecture: local detector/tracker as the primary provider, optional phase/pose model as a second provider, and human annotation as an explicit fallback. The chat model must not be in the perception loop.

## 4. Proposed contracts and data flow

### 4.1 Automatic observation contract

Create a separate versioned contract, for example `video-auto-observations.v1`, rather than pretending automatic output is `video-human-annotations.v1`.

Each frame record should include:

- zero-based source frame index;
- timestamp in seconds;
- source image width/height;
- detector/tracker records for golfer, feet anchor, club/clubhead, ball;
- optional bounding boxes, keypoints, track IDs, and trajectories;
- confidence per detection and per derived event;
- provenance such as `detected`, `tracked`, `model_verified`, `interpolated`, or `unavailable`;
- camera-motion quality and calibration status.

Derived event records should include:

- `address`, `backswing`, `top`, `downswing`, `contact`, `follow_through`, `ball_flight`, `landing`, and `rolling` when supported;
- candidate frame index plus timestamp interval, not false sub-frame precision;
- confidence and evidence references;
- explicit `null` for unavailable events;
- warnings explaining blur, occlusion, frame-rate limits, out-of-frame flight, or camera motion.

Do not serialize absolute source paths, model secrets, or user identity data.

### 4.2 Calibration contract

For MVP, require a project-bound calibration file containing:

- source image width/height;
- exactly four finite source points;
- exactly four paired finite course points;
- source and engine units;
- calibration provenance and optional quality/error metadata.

A calibration is invalid if dimensions differ from the video, points are out of bounds/non-finite, the homography is degenerate, or the source path escapes the project boundary. Automatic calibration detection is not a prerequisite for MVP; it is a later provider.

### 4.3 Exactly-once coordinate boundary

- Detection, tracking, contact estimation, and visual overlays remain in pixel space.
- Convert only the validated address anchor, intended target, and supported landing point through the existing calibration/reconstruction seam.
- Never map a point to course space, map it again, then feed it back as though it were a pixel.
- Use inverse calibration only to re-project a course-space recommendation for rendering.
- If a required mapping is unavailable or low confidence, do not construct an analytics-ready `ShotEvent`; emit unavailable diagnostics and offer human fallback.

## 5. Exact proposed CLI contract

Add a distinct command so existing fixture and human commands remain behaviorally unchanged:

```bash
python3 -m ghostcaddie video-auto-analyze \
  --video /path/to/clip.mp4 \
  --calibration calibration.json \
  --course sample_hole.json \
  --player sample_player.json \
  --project-root /path/to/project \
  --out /path/to/project/out/clip \
  [--detector-model models/golf_detector.onnx] \
  [--pose-model models/golfer_pose.onnx] \
  [--tracker bytetrack] \
  [--sample-fps 30] \
  [--max-frames 900] \
  [--render-video]
```

Required MVP inputs: `--video`, `--calibration`, `--course`, `--player`, and `--out`. `--project-root` governs project-relative resources; absolute video input is permitted internally but never serialized.

Optional model/runtime flags must be explicit and local:

- `--detector-model`: project-relative model file, required unless a documented installed default is selected;
- `--pose-model`: optional;
- `--tracker`: `bytetrack` initially, with `botsort` only when supported;
- `--device`: `cpu` or explicitly available GPU backend;
- `--sample-fps` and `--max-frames` for bounded processing;
- `--render-video` for sampled annotated output.

Output directory:

```text
out/clip/
├── frames/frame_*.jpg
├── frames/frame_manifest.json
├── contact_sheet.jpg
├── observations.json
├── tracks.json
├── event_timeline.json
├── normalized_shot.json
├── recommendation.json          # only when analytics gate passes
├── overlay.svg                  # only when analytics gate passes
├── diagnostics.json
├── annotated_frames/frame_*.jpg
├── annotated_video.mp4          # only with --render-video
└── human_review.html             # generated when automatic gates fail
```

`diagnostics.json` must always state:

- `status`: `complete`, `partial`, or `failed`;
- `analytics_status`: `complete`, `unavailable`, or `not_run`;
- detector/model/tracker provenance without absolute paths;
- confidence and failure reasons;
- relative artifact references;
- whether calibration was supplied, valid, and used exactly once;
- explicit statement that no model result is accepted when required evidence is missing.

If automatic reconstruction fails, the command must still produce inspectable pixel-space tracking/diagnostics and a human-review package, but must not produce a misleading recommendation.

## 6. Realistic MVP implementation milestones

### A0 — Environment and dependency boundary

- Add an optional perception dependency group or documented virtual environment; do not add OpenCV/NumPy/PyTorch to the standard-library analytics installation by default.
- Record supported Python versions, FFmpeg/ffprobe versions, CPU/GPU requirements, model-weight locations, licenses, and offline installation instructions.
- Add a runtime capability command or diagnostics section showing whether detector, tracker, pose model, and encoder are available.

Acceptance: base project and all existing commands run without perception dependencies; automatic command fails clearly with actionable setup instructions when optional dependencies/models are absent.

### A1 — Detector/tracker interfaces and pixel-space records

- Define provider-neutral interfaces for frame batches, detections, tracking, and model provenance.
- Implement an initial local YOLO/OpenCV provider behind the interface.
- Implement temporal track records and association; do not yet construct `ShotEvent`.
- Preserve missing detections as `null` and attach confidence/warnings.

Acceptance: deterministic synthetic/hand-labeled clips produce stable track IDs and serialized pixel-space records; no analytics code is changed.

### A2 — Fixed-camera golfer and club tracking

- Detect one golfer and estimate feet/ground anchor from bounding box or pose keypoints.
- Detect club/shaft/clubhead with a custom golf model or line/keypoint fallback.
- Add camera-motion score using background features or frame registration; classify the clip as fixed, mildly moving, or unsupported.
- Add occlusion and motion-blur flags; never bridge long gaps without a confidence downgrade.

Acceptance: on fixed-camera clips, golfer anchor continuity and clubhead track continuity meet predefined thresholds; unsupported camera motion becomes explicit unavailable output.

### A3 — Ball detection and flight/impact timeline

- Use a golf-specific ball detector at full resolution or high-resolution regions of interest; run dense detection around address/impact and adaptive search after impact.
- Combine detector hits with a Kalman/constant-velocity tracker or equivalent bounded filter; predictions are not labeled observed.
- Estimate address, downswing, contact candidate, post-impact ball, flight, and landing only when evidence supports each.
- Represent impact as a frame interval when frame rate cannot identify a single frame.

Acceptance: ball-before/after-impact precision and contact-frame tolerance meet the evaluation thresholds below; tiny-ball misses produce unavailable landing/flight rather than fabricated trajectories.

### A4 — Calibration and automatic-to-analytics adapter

- Consume the existing four-point calibration contract for MVP.
- Validate dimensions and homography quality.
- Convert only supported required pixel points once through the existing reconstruction boundary.
- Feed a validated `ShotEvent` to unchanged `run_pipeline()` exactly once.
- Keep `video-human-analyze` and fixture `video-analyze` paths unchanged.

Acceptance: adapter tests assert one pipeline call, one forward mapping per engine-needed point, no absolute-path leakage, and rejection of missing/low-confidence evidence.

### A5 — Pixel-space rendering and re-projection

- Render detector boxes, IDs, trails, golfer anchor, clubhead, ball trajectory, phase labels, confidence, and warnings in original image pixels.
- Render course-space recommendation only after analytics succeeds.
- Re-project recommendation target/path into original pixels using inverse calibration solely for visualization.
- Encode sampled annotated video with FFmpeg and preserve a clear sampled-output disclaimer.

Acceptance: annotated frames align with source pixels; course-space analytics remain stable under a controlled synthetic camera transform; output video is readable by ffprobe and viewable.

### A6 — Human fallback and operator workflow

- If required automatic evidence fails, generate `human_review.html` and/or reuse `video-prepare` output with a clear reason.
- Allow explicit human annotation export and route it through the existing accepted `video-human-analyze` path.
- Never merge inferred automatic fields into human-submitted fields without provenance and explicit user confirmation.

Acceptance: every automatic failure has a deterministic, actionable fallback; no failed automatic run claims a recommendation.

### A7 — Real-footage evaluation and release gate

- Run the protocol in Section 8 on user-provided clips.
- Report per-condition metrics, failure categories, runtime, and hardware.
- Keep the feature experimental until the release gate passes on held-out real footage.

## 7. Model, dependency, and runtime choices

### Recommended initial stack

- FFmpeg/ffprobe: decode, metadata, frame extraction, and final encoding.
- OpenCV: frame access, resize/crop, optical flow/camera-motion checks, drawing, and homography utilities.
- NumPy: isolated perception-side arrays and tracking math.
- Ultralytics YOLO or exported ONNX detector: golfer plus golf-specific ball/clubhead classes.
- ByteTrack first; BoT-SORT only after evaluating camera motion and identity stability.
- Optional pose model for feet/ankle keypoints and golfer anchor.
- Optional SwingNet/GolfDB-derived phase model as a phase-evidence provider only; do not treat its trimmed-clip benchmark as arbitrary-video validation.[4]

### Weight and installation policy

- Do not silently download weights.
- Model files must be explicitly installed, versioned, checksummed, license-reviewed, and referenced by a project-relative identifier.
- Keep model weights outside reports; reports contain model name/version/checksum prefix only.
- Provide CPU mode for correctness tests and a GPU path for practical processing. Document expected memory, input resolution, and approximate throughput only after measurement.
- Use an isolated optional environment because the accepted analytics package remains Python 3.9 standard-library-only.

## 8. Evaluation protocol and metrics

### Required clip set

Build a consented, locally stored, versioned evaluation set containing held-out real user golf footage, not generated synthetic footage alone. Label:

- golfer box/feet anchor;
- clubhead/shaft where visible;
- ball before impact and after impact;
- address/top/contact/follow-through frame intervals;
- landing/roll only when visibly supported;
- four calibration points and course coordinates;
- camera type, resolution, FPS, lighting, occlusion, blur, and whether ball leaves frame.

Split by golfer, camera, location, and clip—not random frames—to avoid leakage.

### Metrics

- Golfer detection: precision, recall, IoU, and anchor pixel error.
- Tracking: IDF1/HOTA or equivalent track continuity, ID switches, and maximum tolerated gap.
- Ball: precision/recall at pixel tolerance, center error in pixels, false trajectory rate, pre/post-impact detection recall.
- Clubhead: detection recall, centerline/endpoint error, track continuity, and blur-conditioned recall.
- Swing events: contact-frame absolute error in frames and percentage within ±1/±2 frames; phase F1.
- Flight/landing: availability precision, trajectory pixel RMSE while visible, landing error only for labeled visible landings, and false-positive landing rate.
- Calibration: reprojection error in pixels and course-coordinate error in yards/meters on held-out points.
- End-to-end: percentage of clips producing a valid `ShotEvent`, percentage correctly withheld, recommendation artifact validity, and annotation overlay alignment.
- Operational: wall-clock time per video second, peak RAM/VRAM, dropped/decoded frames, and failure reason distribution.

### Initial release-gate targets (to be confirmed after baseline)

Targets are proposed gates, not claims about current performance:

- >=95% of unsupported/missing-evidence clips correctly withheld from analytics;
- >=90% golfer-anchor availability on fixed-camera MVP clips with <=15 px median anchor error;
- >=80% contact event within ±2 frames on high-FPS fixed-camera clips;
- >=70% ball pre/post-impact recall when the ball is visibly resolvable;
- <=5% false landing claims when landing is not visible;
- calibration reprojection median error <=10 px on the evaluated image region;
- 100% of accepted analytics outputs pass schema, path, provenance, and exactly-once mapping tests;
- zero regressions in the existing 219+ test suite and all existing CLI commands.

No target authorizes a claim of reliable automatic perception before real-footage evaluation passes and results are reported.

## 9. Test strategy and acceptance criteria

### TDD and regression rules

For every new provider, contract, adapter, or CLI behavior:

1. write a focused failing `unittest` first;
2. run the focused test and record the expected RED failure;
3. implement the smallest behavior;
4. run focused GREEN tests;
5. run the complete suite with `python3 -m unittest discover -s tests`;
6. run `python3 -m compileall -q ghostcaddie tests`;
7. run existing `run`, `session`, `provider-session`, `video-analyze`, `video-import`, `video-human-analyze`, and `video-prepare` scenarios.

### Required test groups

- Contract tests for automatic observations, confidence, timestamps, frame indices, null/unavailable values, provenance, model metadata, and path safety.
- Detector-provider tests with deterministic fixtures and mocked model outputs only at the provider boundary.
- Tracker tests for association, missed detections, occlusion gaps, camera-motion flags, and no fabricated predictions.
- Ball/club tests for tiny detections, blur, false positives, and out-of-frame behavior.
- Phase/contact tests for frame intervals and confidence gates.
- Calibration tests for homography validity, dimension mismatch, reprojection error, and exactly-once mapping.
- CLI tests for missing dependencies/models, malformed inputs, unsafe paths, unsupported camera motion, and failed analytics gates.
- Artifact tests for relative references, no source-path leakage, parseable JSON, offline HTML, valid FFmpeg output, and annotations aligned to source dimensions.
- Property tests for deterministic repeated runs given fixed model/runtime versions and seeds where applicable.

### Acceptance criteria for the milestone

The design is ready for implementation only when:

- the optional dependency boundary does not alter the standard-library analytics core;
- fixture mode and all existing commands remain unchanged;
- a fixed-camera real clip can produce pixel-space detections/tracks and a viewable annotated result;
- automatic evidence is versioned and auditable;
- low confidence, tiny-ball failure, blur, occlusion, camera movement, and unseen landing produce explicit unavailable fields;
- a valid fixed-camera clip can pass exactly once through existing `run_pipeline()`;
- no fabricated coordinates, landing points, or confidence values are emitted;
- human fallback is reachable from every automatic failure class;
- the real-footage evaluation report exists and is honest about unsupported conditions;
- all existing tests plus new perception tests pass.

## 10. Risks and mitigations

- **Tiny ball:** full-resolution ROI search, high-FPS impact window, golf-specific training, temporal confirmation; otherwise unavailable.
- **Clubhead blur:** shaft/line/keypoint evidence, phase-window search, interval estimates, no sub-frame precision claim.
- **Camera movement:** background registration and motion score; reject or recalibrate when homography assumptions break.
- **Occlusion:** bounded track gaps and confidence decay; never convert predictions into observations.
- **Low FPS:** report frame-level uncertainty; require ±frame tolerance and do not claim exact impact timing.
- **Fixed vs moving cameras:** MVP gates out moving cameras; later add camera-motion compensation and per-frame homography validation.
- **Calibration drift:** monitor reprojection residuals and expose calibration warnings; never reuse stale mapping silently.
- **Dependency/weight burden:** optional environment, explicit install, local weight inventory, CPU smoke tests, and documented GPU acceleration.
- **Privacy:** local-only processing, no uploads, no identity recognition, no absolute paths in reports, and user-controlled output cleanup.
- **Model bias/domain shift:** evaluate by golfer/camera/location split and retain human fallback; do not generalize from one demo.

## Sources

[1] [GhostBall-Engine repository README](https://github.com/footballanalystrohan-glitch/GhostBall-Engine)

[2] [GhostBall-Engine main.py](https://raw.githubusercontent.com/footballanalystrohan-glitch/GhostBall-Engine/main/main.py)

[3] [GhostBall-Engine pipeline_integration.py](https://raw.githubusercontent.com/footballanalystrohan-glitch/GhostBall-Engine/main/pipeline_integration.py)

[4] [GolfDB / SwingNet repository](https://github.com/wmcnally/golfdb)
