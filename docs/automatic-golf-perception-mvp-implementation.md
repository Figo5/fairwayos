# Automatic Golf Perception MVP — Infrastructure Implementation

Status: **implemented as non-data-dependent infrastructure; real-perception gate remains blocked**.

## Implemented

- `automatic-perception.v1` contracts with explicit unavailable values and provenance categories.
- Detector and tracker protocols.
- Deterministic single-golfer track selection.
- Pose-derived body-anchor validation.
- Bounded optical-flow refinement policy; flow cannot promote itself to semantic detection.
- Deterministic continuity, confidence, precision, and recall metrics.
- Provisional swing-phase state machine and impact candidate interval.
- Camera-motion/cut/coverage/anchor sequence gates.
- Guarded automatic reconstruction that requires ball, clubhead, contact, landing, and calibration evidence.
- Exactly-once calibration mapping through the existing reconstruction boundary.
- Safe deterministic automatic reports and evaluation reports.
- Annotated automatic frame output through the existing renderer.
- New `video-automatic-analyze` CLI path for validated automatic-observation documents.
- Explicit `--fallback-human` preparation path; it never silently substitutes human labels.
- Public exports from `ghostcaddie.video`.

## Safety behavior

The automatic CLI accepts a project-bound `video-observations.v1` document produced by an approved automatic adapter. It does not relabel generic YOLO output as golf observations and does not run model inference itself.

When required evidence or provisional gates fail:

- diagnostics and evaluation reports are written;
- annotated frames may be written;
- missing quantities remain unavailable;
- `recommendation.json` and `normalized_shot.json` are not written;
- `run_pipeline()` is not invoked;
- explicit `--fallback-human` may prepare the existing blank annotation workspace.

When a fully populated, validated observation document passes all gates, the CLI invokes the existing reconstruction boundary and unchanged `run_pipeline()` exactly once. The current success-shaped tests use fixture data only and are not a real-perception validation.

## Not yet validated

The following require separate consenting data and must not be inferred from the 480p YouTube stress artifact:

- high-resolution single-shot clip;
- exact four-point calibration;
- ground-truth golfer, ball, clubhead, and impact annotations;
- trusted golf-specific pretrained weights or an approved custom-weight package;
- held-out evaluation metrics;
- production thresholds;
- reliability on real footage.

The 480p YouTube artifact remains a negative/stress case only. No automatic golf-analysis claim, recommendation, `ShotEvent`, landing, trajectory, contact, or club result is made from it.

## Verification

Focused and full verification completed during implementation:

- automatic contract/temporal/gate tests passed;
- automatic renderer/evaluation tests passed;
- automatic CLI tests passed;
- full suite: **383 tests passed, 6 skipped**;
- `python3 -m compileall -q ghostcaddie tests`: passed.

Provisional thresholds are documented in the implementation plan and must be tuned only against actual annotated footage, then evaluated on held-out footage. They are not validated production thresholds.
