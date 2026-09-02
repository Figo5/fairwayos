# Research-training gauntlet status

Date: 2026-09-01

Status: **research-only / production blocked**.

## Temporal spread diagnostic

`aggregate_temporal_uncertainty` reports `spread_status` as `unavailable` when
no finite points are present, `bounded` when the centroid radius is at most
50.0 px, and `wide` when the radius exceeds 50.0 px (a potentially unstable
spread). This fixed threshold is a research diagnostic only; the aggregate
never asserts object identity, and its `identity` field remains `unavailable`.

## Batch acquisition

A bounded batch of three public-platform golf clips was attempted with the existing `yt-dlp` adapter, Node/EJS runtime, no playlist expansion, and 20-second segments. Two clips downloaded locally and one was blocked by the configured estimated-size limit. Source URLs and hashes are retained only in the ignored local manifest:

```text
out/research_training_gauntlet/manifest.json
```

Downloaded media, extracted frames, annotations, reports, and model outputs remain under the ignored local directory `out/research_training_gauntlet/`.

The YouTube ingestion boundary rejects provider metadata with boolean, non-finite, or non-positive duration and rejects a returned video ID that differs from the requested ID before invoking the download subprocess. Estimated `filesize` metadata now has precedence over `filesize_approx`; when present it must be a finite, non-negative integer and is rejected as malformed metadata otherwise. Missing duration or size remains an explicit provider limitation rather than an invented value.

The automatic YouTube boundary preserves distinct blocked categories for `duration_limit` and `segment_limit`; these are not collapsed into generic video-unavailable failures.

- Contact/landing candidate diagnostics accept ball points only when provenance is explicitly `observed`, `native`, or `user_confirmed`; missing, inferred, automatic, and unknown provenance remain unavailable and cannot be promoted.
- Submitted human annotation imports reject `contact.source` and `landing.source` values of `"inferred"`; only explicit/user-reviewed sources (`user_supplied`, `user_confirmed`, or `observed`) may cross the import boundary.

Research ball sidecars reject booleans, strings, fractional values, and negative values for `frame_index` and `longest_gap`; these fields remain strict non-negative integers and are never silently coerced.

- ResearchBallTracker rejects non-finite/boolean `max_step_pixels` and coerced fractional/boolean `max_gap_frames` or `min_pixels`; bounds remain explicit constructor contracts.
- The automatic YouTube rerun removes stale `annotated_video.mp4` before processing when the current run does not request video rendering, so diagnostics cannot point to an unreferenced prior render.
- Generated extraction outputs reject symlinked output directories and contact-sheet output files instead of resolving through them; stale `frame_manifest.json` files are removed before a rerun so a failed extraction cannot leave a manifest describing regenerated or missing frames.
- The optional YOLO pose adapter leaves the golfer anchor unavailable when neither ankle keypoint is confident; it no longer substitutes a bounding-box bottom, and emits `anchor_missing` explicitly.
- VideoDiagnostics rejects malformed top-level container shapes with VideoContractError before iteration, membership checks, or serialization.
- YouTube acquisition requires probe metadata to contain a canonical returned video ID matching the requested source before download; missing or mismatched identity is malformed metadata.
- Research impact brackets reject negative, boolean, or fractional frame indices, non-finite/boolean confidence, and single-frame brackets; `min/max` ordering no longer masks malformed input.
- FFprobe metadata parsing accepts normal numeric strings but rejects boolean, fractional, non-finite, and malformed dimensions, frame counts, and durations.

## Triage and automatic overlays
The local generic YOLO pose checkpoint was run on the two downloaded clips:

```text
model: yolo11n-pose.pt
sha256: 869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0
sampling: 2 FPS / 40 frames per clip
```

Results:

- one clip classified `usable`, with person-frame coverage `0.75`;
- one clip classified `partial`, with person-frame coverage `0.70`;
- the blocked acquisition is recorded as `unsuitable` with `size_limit_exceeded`;
- per-clip triage includes resolution, FPS, blur proxy, motion/cut proxy, person count, anchors, warnings, and null unsupported fields;
- a batch montage and annotated H.264/yuv420p MP4s were generated.

The model is generic person/pose only. It is not a golf-ball, clubhead, impact, trajectory, landing, calibration, or analytics model.

## SwingNet research smoke

The existing GolfDB SwingNet checkpoint was run on both downloaded clips on MPS. Original event labels were preserved:

```text
Address, Toe-up, Mid-backswing, Top,
Mid-downswing, Impact, Mid-follow-through, Finish
```

Predictions and separate annotated videos are recorded in:

```text
out/research_training_gauntlet/swingnet_predictions.json
out/research_training_gauntlet/*/swingnet_annotated_video.mp4
```

Checkpoint provenance and SHA-256 are recorded in that report. Its license remains uncleared for production. Predictions are model outputs only; no ground truth, event accuracy, frame-tolerance accuracy, or PCE is claimed.

## Training and evaluation decision

Experimental training is externally blocked for this batch:

- valid paired ball/clubhead/impact/landing annotations: `0`;
- frozen video- or golfer-level held-out split: unavailable;
- pseudo-labels created: `false`;
- training/evaluation on public footage would not establish real accuracy.

Therefore no new trained model was produced, no pseudo-label was promoted to ground truth, and no evaluation result was fabricated.

## Production gates

All remain closed:

```text
ball: null
clubhead: null
impact: null
trajectory: null
landing: null
calibration: null
ShotEvent: null
recommendation: null
```

`run_pipeline()` was not invoked. Core analytics, calibration, wind, dispersion, hazards, session behavior, CLI commands, YouTube ingestion, CI, and human fallback were not changed by this gauntlet run.

## Verification

- full unittest suite: 396 tests passed, 6 skipped;
- `compileall`: passed;
- `git diff --check`: passed;
- accepted Pexels 6573485 MP4: H.264, 1920x1080, 15 FPS, 121 frames, `yuv420p`;
- FFmpeg decode and first-frame extraction: passed;
- visual QA: person/pose/anchor overlays and honest unavailable/research-only labels visible;
- artifact assertions: passed.

## Ball-plausibility correction (2026-09-01)

Native-resolution visual QA found the accepted clip's "ball" track following
background texture (the flagstick base and grass), not any golf ball. Root
cause: the local ball model emits frame-filling boxes on some clips; their box
centers (frame center) seeded a phantom track. `normalize_box` now rejects
boxes wider/taller than a quarter of a frame dimension or covering more than
5% of frame area, and the demo skips such boxes individually so a real
detection in the same frame survives. After the fix the accepted clip renders
121/121 pose observations with ball observations explicitly unavailable on all
121 frames; the prior ball track is retained as rejected evidence in
`fairwayos_unified_pexels_6573485_pre_ball_plausibility/`. Close-up candidate
Pexels 6573486 passed source triage but failed output ball-alignment QA
(marker off-ball, tracker unavailable when the ball was largest); candidate
Pexels 6573612 passed source triage and, after the fix, keeps 2 honest
near-ball observations at the clubface pass-through and marks the rest
unavailable. No ground truth exists, so no ball precision/recall is claimed.

The next genuine milestone requires legally usable paired annotations and a documented golf-ball/clubhead checkpoint or a consented FairwayOS-owned dataset. Until then, the project remains technically ready but data/model blocked.

## Cycle 9 verification (2026-09-01, HEAD b99db9e)

Baseline re-verified: accepted source `pexels_6573485/source.mp4` sha256
`a6e48474045365d1de2d4af76f65da558531684d67da87172cdd15a6dc45e1d6`; accepted
output 121/121 pose observed, ball explicitly unavailable on all 121 frames;
H.264/yuv420p 1920x1080, 15 fps, 121 frames; full decode clean; provenance
`source.video_id=6573485` matches local naming.

Deterministic reruns: two fresh renders of the accepted clip at the accepted
parameters (sample-fps 15, max-duration 8) produce byte-identical annotated
frames and the identical annotated-video sha256 `eb01781a9a8b36768e5c403fe2a5b
00996a5d85818f0bdd3461ac5f4fdd59950` as the accepted artifact. Diagnostics and
provenance are semantically identical across reruns. Evidence:
`run_c9_deterministic_rerun.py`, `c9_deterministic_rerun_check.json`. A
`python3 -m ghostcaddie.cli` invocation silently did nothing before this cycle
because `ghostcaddie/cli.py` had no `__main__` guard; the guard now runs `main()`.

Renderer integrity: pose, ball, clubhead-candidate, and SwingNet overlays all
consume one clean `clean_frame_for_components` copy and compose once; the
accepted artifact contains zero ball-marker/tracer/inset structures on all 121
frames, matching diagnostics (0 ball observations). Evidence: `c9_renderer_review/`.

Pose stability: the accepted clip contains a second background person in all 39
sampled frames (conf 0.56-0.84); the single `golfer-0` track stays viable only
through top-confidence selection. `_pose_observation` now records honest
`person_count`, `second_person_count`, and `multi_person_frame` metadata on
every pose observation (no behavior change; accepted clip reports person_count=2
on 121/121). Evidence: `c9_pose_eval/`.

Ball evidence per clip (current code, separate rows, never merged):
6573485 unavailable 104/104; 6573612 observed 15, predicted 2, unavailable 28;
6573486 observed 5, predicted 1, unavailable 55; impact_candidate observed 147;
flight_candidate observed 1; mmu_candidate observed 8. Implausible-box
rejections: 125/123/134/99/131/220. Native QA of the 6573612 frame-74
observation shows the marker/label rendered on the clubhead with no visible
ball (single-frame low-confidence detection at clubface pass-through) — it
stays recorded as non-ground-truth evidence, not promoted. Evidence:
`c9_ball_eval/`.

Trajectory/landing: no local clip reaches 8 defensible consecutive observed
ball points; best is 2. Trajectory proposal and landing remain unavailable.
Evidence: `c9_trajectory_eval/trajectory_proposal_verdict.json`.

Clubhead/contact: every local candidate was re-triaged at native resolution
(6573618 rejected: single-frame contact only; 5200687 cuts away before impact;
6573486 no strike on screen; others framing-limited). No locally runnable,
license-clear clubhead/contact checkpoint exists (SwingNet CC BY-NC code with
unstated checkpoint license; ClubheadDB package ships no checkpoint; CADDIE/GolfClub
has no public checkpoint). Evidence: `c9_clubhead_recon/recon_summary.json`.

SwingNet (research-only): predictions for all three pexels clips reproduce
exactly (rerun match <1e-6); same-frame event clustering and convention
sensitivity (chunked vs whole-clip LSTM) recorded as honesty findings. Evidence:
`c9_swingnet_eval/`.

Video QA: all three annotated MP4s are H.264/yuv420p with monotonic
decoded-order timestamps, clean full decodes, and provenance sha256 matching
their sources. Evidence: `c9_video_qa/`.

## Cycle 10 renderer portability probe (2026-09-01)

A dependency-free research renderer was added to `research_overlay.py`. It
consumes an existing candidate sidecar and clean source frames, drawing a
current candidate box, an uncertainty envelope, a persistent dotted candidate
trail, and portable boundary bars. It never changes production observations,
ball identity, calibration, trajectory, landing, or analytics fields.

The bounded MMU source window `[53, 164]` was rendered locally to:

```text
out/research_training_gauntlet/mmu_candidate/analysis/automatic_ball_tracer_research_overlay_v2.mp4
```

The output is H.264/yuv420p, 600x480, 25 FPS, 112 frames, 4.48 seconds, and
fully decodes with FFmpeg. Native-resolution spot QA found the geometric
overlay itself rendered correctly, but the inherited heuristic candidate was
not aligned with the visible bright object in all inspected frames: early and
late markers were on/near club or ground texture. The visual gate therefore
failed; this is a renderer portability improvement and a rejected perception
candidate, not a ball-tracking success. The candidate remains
`ground_truth=false`, `research_only=true`, and `production_eligible=false`.

The exact local QA sheets and sidecar are kept under the ignored MMU analysis
directory. No generated media is published.

## Cycle 11 visual-alignment gate (2026-09-01)

The research renderer now accepts an explicit `visually_aligned` decision. When
that decision is false, it suppresses the candidate marker, uncertainty
envelope, and candidate trail instead of presenting a visually rejected track.
This is a review gate, not an automatic detector or ground-truth decision.

Using the clean MMU source and the same bounded window `[53, 164]`, the rejected
render is:

```text
out/research_training_gauntlet/mmu_candidate/analysis/automatic_ball_tracer_research_overlay_v3_rejected.mp4
```

The output is H.264/yuv420p, 600x480, 25 FPS, 112 frames, 4.48 seconds, and
fully decodes. Exhaustive contact-sheet QA confirmed that no candidate marker,
uncertainty box, or dotted trail remains; only the rejection boundary bars are
visible. The bright source object is therefore not falsely marked. The sidecar
records `ground_truth=false`, `research_only=true`, and
`production_eligible=false`. Ball identity and all downstream production gates
remain unavailable.

## Cycle 12 temporal sanity gate (2026-09-01)

A research-only `evaluate_candidate_quality` gate now checks candidate bounds,
marker/point consistency, and observed-frame step size. It reports rejection
reasons and per-frame marker/trail decisions while explicitly retaining
`ground_truth_available=false`, `research_only=true`, and
`production_eligible=false`.

Applied to the MMU candidate, the gate passed geometric/temporal sanity for
112 points with a maximum step of 80.399005 pixels. A fresh clean-source render
was produced at:

```text
out/research_training_gauntlet/mmu_candidate/analysis/automatic_ball_tracer_quality_gate_v4.mp4
```

The output is H.264/yuv420p, 600x480, 25 FPS, 112 frames, 4.48 seconds, and
fully decodes. Exhaustive visual QA still rejected the candidate: the marker
and trail follow club/ground texture rather than the visible bright object.
This establishes that temporal smoothness and image bounds are insufficient to
establish target identity. The explicit visual-alignment suppression path from
Cycle 11 remains required, and no ball capability or production gate was
promoted.

## Cycle 13 object-consistency gate (2026-09-01)

The research-only quality gate now supports an explicit
`require_object_consistency` mode. When enabled, every candidate point must carry
upstream object-match evidence with `matched=true` and a finite center offset
within the configured bound. Missing evidence is rejected as
`object_consistency_unavailable`; contradictory evidence is rejected as
`object_consistency_mismatch`. This is a conservative evidence gate, not a
ball detector or ground-truth evaluator.

The MMU diagnostics contain no object-match evidence, so the required mode
correctly rejected the inherited candidate. A fresh clean-source render is:

```text
out/research_training_gauntlet/mmu_candidate/analysis/automatic_ball_tracer_object_consistency_rejected_v5.mp4
```

The output is H.264/yuv420p, 600x480, 25 FPS, 112 frames, 4.48 seconds, and
fully decodes. Exhaustive contact-sheet QA confirmed that no candidate marker,
uncertainty box, or dotted trail is visible; only the yellow/red rejection
boundary bars remain. The bright source object is not asserted to be a golf
ball. Ball identity and downstream production gates remain unavailable.

## Cycle 14 semantic rejection legend (2026-09-01)

The portable research renderer now accepts an explicit rejection reason. For
`object_consistency_*` rejection, the top boundary is blue and the bottom
boundary remains red; generic visual-alignment rejection retains the yellow/red
legend. Candidate geometry remains suppressed whenever visual alignment is
false. This adds visual distinction between geometric sanity and semantic
object-match failure without claiming detection or ground truth.

A fresh clean-source render is:

```text
out/research_training_gauntlet/mmu_candidate/analysis/automatic_ball_tracer_object_consistency_rejected_v6.mp4
```

The output is H.264/yuv420p, 600x480, 25 FPS, 112 frames, 4.48 seconds, and
fully decodes. Exhaustive contact-sheet QA confirmed blue semantic-rejection
and red rejection bars with no candidate marker, uncertainty box, or dotted
trail. Ball identity and downstream production gates remain unavailable.

## Cycle 15 exclusive image-boundary validation (2026-09-01)

The research candidate quality gate now treats image bounds as half-open:
`0 <= x < width` and `0 <= y < height`. A point exactly at `(width, height)`
is rejected as out of bounds instead of being accepted at the exclusive image
edge. This is a geometry-contract fix only; it does not identify a golf ball or
open any production gate.

A fresh MMU render was generated from clean source frames:

```text
out/research_training_gauntlet/mmu_candidate/analysis/automatic_ball_tracer_object_consistency_rejected_v7.mp4
```

The output is H.264/yuv420p, 600x480, 25 FPS, 112 frames, 4.48 seconds, and
fully decodes. Exhaustive contact-sheet QA confirmed the blue semantic-rejection
and red unavailable bars remain visible with all candidate geometry suppressed.

## Cycle 16 detached-render provenance attempt (2026-09-01)

The research legend API now accepts a relative `source_label` and an explicit
`diagnostic`. It validates and emits both fields while retaining the validated
ball-identity disclaimer. This preserves provenance when a text-capable renderer
is used.

The installed dependency-free FFmpeg path does not provide `drawtext`; adding
these strings to the actual MP4 would therefore require a different renderer.
The available OpenCV demo path still draws the rejected heuristic marker, so it
was not used. The safe clean-source artifact for this attempt remains a
candidate-suppressed render:

```text
out/research_training_gauntlet/mmu_candidate/analysis/automatic_ball_tracer_object_consistency_rejected_v8.mp4
```

It is H.264/yuv420p, 600x480, 25 FPS, 112 frames, 4.48 seconds, and fully
decodes. Frame QA confirms the blue semantic-rejection and red unavailable bars
with no candidate marker, uncertainty box, or dotted trail. The visible
provenance improvement is blocked by the local FFmpeg text-filter limitation;
red corner endcaps were added to the unavailable bar as a portable visual
indicator. No ball identity or production capability was promoted.

## Cycle 18 research-only impact bracket (2026-09-01)

The research overlay now supports an optional temporal impact bracket. When the
state is `candidate_bracket_only`, it renders only a short orange frame-gated
bar; it never draws a spatial impact point or promotes contact timing. When the
state is `unavailable`, the temporal bar is suppressed. Rejected ball geometry
remains suppressed, and the output remains research-only with
`ground_truth=false` and `production_eligible=false`.

The MMU bounded window was rendered with the existing object-consistency
rejection state and a five-frame local bracket. The new artifact is:

```text
out/research_training_gauntlet/mmu_candidate/analysis/automatic_ball_tracer_object_consistency_rejected_v9.mp4
```

It is H.264/yuv420p, 600x480, 25 FPS, 112 frames, and 4.48 seconds. Full-frame
and enlarged bottom-band QA confirmed the blue semantic-rejection bar, red
unavailable U/endcaps, and orange bar only on local frames 15-19. No candidate
marker, uncertainty box, dotted trail, or spatial impact geometry appears.

A local model comparison also found the PT checkpoint useful only as a separate
research candidate: misses and oversized/ambiguous boxes remain. The ONNX
adapter collapses detections to a zero-height boundary, so its output is
unavailable pending coordinate-decoding repair. Generic sports-ball detections
were not relabeled as golf-ball evidence.

## Cycle 19 research-only model comparison (2026-09-01)

The local ball adapter now rejects degenerate or out-of-frame boxes before they
reach research tracking. Candidate observations emitted by the local model are
explicitly serialized with `research_only=true`, `ground_truth=false`, and
`production_eligible=false`.

A separate model-comparison renderer was added for local experimentation. It
keeps PT and ONNX unavailable in this comparison, labels the generic output as
`GENERIC CANDIDATE`, and displays `NOT GOLF-BALL IDENTITY | IDENTITY UNAVAILABLE`.
It does not convert generic sports-ball output into golf-ball evidence.

The rendered local artifact is:

```text
out/research_model_comparison/comparison_overlay_h264_yuv420p.mp4
```

It is H.264/yuv420p, 600x480, 25 FPS, 223 frames, and 8.92 seconds. Full
contact-sheet QA confirmed persistent research-only/unavailable diagnostics and
no unlabeled ball identity or production analytics. The artifact hash is:

```text
e2f14cf99fd08f7a79acf33f8bfccdb0e6b3e71655e1c956fb45e9c1efda930f
```

## Cycle 20 temporal arbitration and availability timeline (2026-09-01)

The multi-hypothesis research tracker now returns `unavailable` with
`ambiguous_candidates` when its leading continuity/quality hypotheses are within
the configured ambiguity margin. This prevents a near-tie from being silently
reported as an observed ball track; it does not establish identity or ground
truth.

A separate temporal comparison render shows per-frame backend availability lanes
for PT, ONNX, and GENERIC without drawing any spatial candidate markers. The
artifact is:

```text
out/research_model_comparison_temporal/temporal_comparison_h264_yuv420p.mp4
```

It is H.264/yuv420p, 600x480, 25 FPS, 165 frames, and 6.60 seconds. PT output
is present on 146/165 frames, ONNX on 139/165, and GENERIC on 152/165. The
artifact hash is:

```text
7c03a42855de99d2b0b424734392a203851ab21f6bfe7b2d18ea94f84ddb9c10
```

The local temporal comparison remains diagnostic only: `research_only=true`,
`ground_truth=false`, and `production_eligible=false`. No calibration, golf-ball
identity, impact point, trajectory, landing, analytics, or recommendation is
asserted.

## Cycle 21 explicit observed/predicted research states (2026-09-01)

The reusable model-comparison overlay contract now accepts `observed` and
`predicted` states in addition to `candidate` and `unavailable`. Rendered labels
use the state name and confidence, while identity remains unavailable and
`production_eligible` remains false. This is a state-provenance improvement, not
an accuracy claim.

The Cycle 20 temporal MP4 remains the inspected availability artifact:

```text
out/research_model_comparison_temporal/temporal_comparison_h264_yuv420p.mp4
```

It uses backend `OUTPUT/NO OUTPUT` lanes and deliberately contains no spatial
markers. It remains H.264/yuv420p, 600x480, 25 FPS, 165 frames, 6.60 seconds,
with SHA-256 `7c03a42855de99d2b0b424734392a203851ab21f6bfe7b2d18ea94f84ddb9c10`.
No observed/predicted identity is inferred from the availability lanes.

## Cycle 22 confidence and uncertainty state diagnostics (2026-09-01)

The research tracker and model-comparison contract now identify confidence as
`detection_quality_not_identity`. This makes the scalar explicit: it describes
model output quality only and cannot be interpreted as ball identity or ground
truth.

A new bounded state-diagnostics render shows observed, predicted, and unavailable
states with confidence and uncertainty scalar bars. It intentionally draws no
point, box, trajectory, or location marker:

```text
out/research_state_diagnostics/state_diagnostics_h264_yuv420p.mp4
```

The artifact is H.264/yuv420p, 600x480, 25 FPS, 67 frames, and 2.68 seconds.
It contains 36 observed, 2 predicted, and 29 unavailable diagnostic states. Its
SHA-256 is:

```text
37ebb5a8e6a8d1c59bcfa6bef3717f7c684153f00eff57cbe4985de7ff1782d7
```

## Cycle 23 temporal confidence history (2026-09-01)

A bounded confidence-history render now shows temporal detection-quality history
for the rejected MMU research sequence. The source image area is retained, while
an information band plots confidence over time without drawing a ball marker,
box, trail, impact point, trajectory, or location claim:

```text
out/research_training_gauntlet/mmu_candidate/analysis/temporal_confidence_history_h264_yuv420p.mp4
```

The output is H.264/yuv420p, 600x480, 25 FPS, 112 frames, and 4.48 seconds.
Its verified SHA-256 is:

```text
ef34ca9bff0ae16537d547d54049f51bc2387c4691872304efd30724b0ce95a9
```

The associated summary aggregates only a bounded recent window of finite
research observations. No-point input remains unavailable; confidence remains
`detection_quality_not_identity`, with identity unavailable. All output states
remain `research_only=true`, `ground_truth=false`, and `production_eligible=false`.

## Cycle 24 bounded spread metadata (2026-09-01)

The temporal uncertainty summary now reports `window_limit`, `window_size`,
`window_bounded`, and `spread_status`. A usable finite recent window reports
`spread_status=bounded`; empty or no-point input remains `spread_status=unavailable`.
These fields describe research diagnostics only and do not establish object
identity, location evidence, or ground truth.

## Cycle 25 temporal spread transition render (2026-09-01)

The bounded MMU diagnostic now classifies centroid-radius spread as `bounded`
when it is at most 50.0 px and `wide` above that threshold. This is a
research-only stability diagnostic, not a tracking or identity decision.

The new temporal transition render is:

```text
out/research_state_diagnostics/mmu_temporal_spread/mmu_temporal_spread_h264_yuv420p.mp4
```

It is H.264/yuv420p, 600x480, 25 FPS, 199 frames, and 7.96 seconds. Its
verified SHA-256 is:

```text
448d5b7b7c453e12e48243bb8fd6fe15397e2563b58a7278ac46a7c3ce2cabad
```

The render visibly communicates observed, predicted, and unavailable temporal
transitions through diagnostic bands only. No spatial marker, identity claim,
impact, trajectory, landing, calibration, analytics, or recommendation is
present. It remains `research_only=true`, `ground_truth=false`, and
`production_eligible=false`.

## Cycle 26 explicit spread-status semantics (2026-09-01)

The temporal uncertainty summary now includes a fixed `spread_threshold_px` and
human-readable `spread_status_description` alongside `spread_status`. The
statuses are `bounded` at or below 50.0 px, `wide` above 50.0 px, and
`unavailable` when no finite points are present. These remain research stability
diagnostics; identity remains unavailable.

A separate transition-focused local render is available at:

```text
out/research_state_diagnostics/mmu_temporal_spread_transitions/mmu_temporal_spread_transitions_h264_yuv420p.mp4
```

It is H.264/yuv420p, 600x480, 25 FPS, 199 frames, and 7.96 seconds. Its
SHA-256 is:

```text
8470132253bd85217708c7ee919605e524fc6e73093893858de52540b2063e95
```

The 199-frame decoded contact sheet shows bounded, wide, and unavailable
transitions through temporal diagnostic bands only. No spatial marker, ball
identity, impact, trajectory, landing, calibration, analytics, or recommendation
is rendered. The generated media remains local and ignored; no deterministic
rerender claim is made because no generator script is tracked for this artifact.
The output remains `research_only=true`, `ground_truth=false`, and
`production_eligible=false`.

The Cycle 24 render is the verified temporal confidence-history MP4 above:
H.264/yuv420p, 600x480, 25 FPS, 112 frames, 4.48 seconds, SHA-256
`ef34ca9bff0ae16537d547d54049f51bc2387c4691872304efd30724b0ce95a9`. The
source imagery is retained only as bounded local research material; no media is
published. Output remains `research_only=true`, `ground_truth=false`, and
`production_eligible=false`.

## Cycle 27 spread availability and threshold diagnostic (2026-09-01)

Temporal spread summaries now expose `spread_available`, allowing consumers to
distinguish a usable finite spread measurement from an explicitly unavailable
result without inferring state from a missing field. Identity remains unavailable
and confidence remains `detection_quality_not_identity`.

The new threshold-labeled render is:

```text
out/research_state_diagnostics/mmu_bounded_spread_diagnostic/mmu_bounded_spread_diagnostic_h264_yuv420p.mp4
```

It is H.264/yuv420p, 600x480, 25 FPS, 199 frames, and 7.96 seconds. Its
verified SHA-256 is:

```text
f9d04df0c813d26ec5e4266c69c0ccfec188ac6969075b8d30780b0be33f941d
```

The diagnostic visibly labels `50 PX`, `BOUNDED`, `WIDE`, and `UNAVAILABLE`
transitions through temporal bands only. No spatial marker, ball identity,
impact, trajectory, landing, calibration, analytics, or recommendation is
rendered. The artifact remains local and ignored with
`research_only=true`, `ground_truth=false`, and `production_eligible=false`.

## Cycle 28 machine-readable temporal spread diagnostics (2026-09-01)

Temporal uncertainty summaries now expose `schema_version=1` and integer
`spread_status_code` values: `0` for unavailable, `1` for bounded, and `2` for
wide. Existing text descriptions, threshold values, availability, and identity
unavailable semantics remain unchanged.

The new local threshold/spread diagnostic render is:

```text
out/research_training_gauntlet/mmu_candidate/analysis/temporal_threshold_spread_diagnostics_h264_yuv420p.mp4
```

It is H.264/yuv420p, 600x480, 25 FPS, 112 frames, and 4.48 seconds. Its
verified SHA-256 is:

```text
75074879d145e4d53fb73c55a9fb8e2f66128e044390c0db16ec8a7e8cdabdfd
```

The decoded 112-frame contact sheet shows temporal threshold and spread labels
in diagnostic bands only. Domain ball, contact, landing, and calibration values
remain unavailable. No spatial marker, ball identity, impact, trajectory,
analytics, recommendation, or production claim is rendered. The artifact remains
local and ignored with `research_only=true`, `ground_truth=false`, and
`production_eligible=false`.

## Cycle 29 numeric spread-status consumer contract (2026-09-01)

Temporal summaries now expose `spread_status_code_values=[0, 1, 2]` alongside
`schema_version=1` and `spread_status_code`. This gives machine consumers a
stable enumeration without changing the underlying research-only semantics:
`0=unavailable`, `1=bounded`, and `2=wide`.

The corresponding local numeric-status render is:

```text
out/research_training_gauntlet/mmu_candidate/analysis/temporal_spread_status_numeric_h264_yuv420p.mp4
```

It is H.264/yuv420p, 600x480, 25 FPS, 112 frames, and 4.48 seconds. Its
verified SHA-256 is:

```text
312e75e9e1d365f16849fdbda0703af1917491a271fd5a87574c561fa6d8a6be
```

All numeric labels remain temporal diagnostics. The source area has no added
spatial marker; domain ball, contact, landing, and calibration remain
unavailable. The output remains `research_only=true`, `ground_truth=false`, and
`production_eligible=false`.

## Cycle 30 spread-status consistency validation (2026-09-01)

Temporal summaries now include `spread_status_consistent`, and the reusable
validator accepts only the exact mapping `unavailable=0`, `bounded=1`, and
`wide=2`. Invalid types or mismatched pairs are rejected by the helper; this is
schema validation only and does not promote any observation.

The consistency render is:

```text
out/research_training_gauntlet/mmu_candidate/analysis/temporal_spread_status_consistency_h264_yuv420p.mp4
```

It is H.264/yuv420p, 600x480, 25 FPS, 112 frames, and 4.48 seconds, with
SHA-256 `312e75e9e1d365f16849fdbda0703af1917491a271fd5a87574c561fa6d8a6be`.
It is byte-identical to the Cycle 29 numeric-status render: the consistency
check changes diagnostics, not pixels. The decoded contact sheet contains only
temporal bands and no spatial marker, identity, impact, trajectory, landing,
calibration, analytics, recommendation, or production claim. Output remains
`research_only=true`, `ground_truth=false`, and `production_eligible=false`.

## Cycle 31 render-coverage documentation and multi-specialist re-verification (2026-09-01)

The unified AI-demo diagnostics contract now declares render coverage and audio
state explicitly. `build_demo_report` accepts an additive, optional `render`
block (omitted when not provided, so legacy reports are unchanged), and
`run_local_demo` populates it with `rendered_frames` (observation count),
`sample_fps`, `duration_seconds` (rendered frames / sample fps), and
`audio="unavailable_dropped_by_reencode"` with the reason that the annotated
re-render covers sampled frames only and the source audio is not carried into
the re-encode. The block carries `research_only=true`, `ground_truth=false`,
and `production_eligible=false`. This documents what was already true: the
accepted annotated video is 121 frames at 15 fps (8.067 s) while the source is
310 frames at 30 fps (10.347 s) with AAC audio that the re-encode drops.

The accepted artifact `out/research_training_gauntlet/fairwayos_unified_pexels_6573485`
was re-rendered in place with the change. The annotated video sha256 is
unchanged at `eb01781a9a8b36768e5c403fe2a5b00996a5d85818f0bdd3461ac5f4fdd59950`,
all 121 annotated frames are byte-identical, provenance.json is semantically
identical, and diagnostics.json differs only by the additive render block
(verification: `out/research_training_gauntlet/c23_deterministic_rerun/promotion_check.json`
and `out/research_training_gauntlet/c23_deterministic_rerun_check.json`, both
`all_checks_pass=true`; a second fresh render reproduced the accepted video
sha256, frames, provenance, and diagnostics exactly).

Eight bounded read-only specialists re-verified the accepted artifact and local
clips (evidence under `out/research_training_gauntlet/c23_*/`):

- Video QA (`c23_video_qa/qa_report.json`): annotated_video.mp4 passes H.264
  High / yuv420p / 1920x1080 / strict CFR 15 fps with 0 negative or
  non-monotonic PTS, 0.0 s VFR drift, clean full decode, and a valid 5280x2970
  contact sheet; the audio drop and 78% source coverage were undocumented
  before this cycle.
- Renderer review (`c23_renderer_review/verdict.json`): one decode per sampled
  frame, one composition canvas, overlays drawn exactly once
  (`ghostcaddie/video/ai_demo.py` clean-frame path), no ghosting, double-draw,
  or misalignment in native-resolution crops; annotated_frames, video frames,
  and contact sheet are mutually consistent; no change proposed.
- Tests/visual QA (`c23_tests_visualqa/qa_report.json`): 436 tests OK
  (skipped=6), compileall clean, 121 annotated frames == 121 observations with
  pose observed in all and ball honestly `unavailable` in all (the ball is
  visible in-scene but never resolved), provenance sha256 matches the
  recomputed source sha256, and zero overlay pixels fall outside the banner and
  golfer bbox in 8 frame pairs.
- Pose stability (`c23_pose_eval/verdict.json`): stable, correctly localized
  pose across 19 sampled frames; camera is a slow handheld pan (~166 px) that
  the pipeline does not compensate (per-frame `camera_motion:not_assessed`
  warning is honest); golfer-0 selection never jumped. Honesty caveat: the
  `person_count=2` / `second_person_count=1` fields are faithful to model
  output, but native-resolution inspection found no second human; the
  humanoid-shaped golf bag at the left edge is the plausible source.
- SwingNet research-only (`c23_swingnet_eval/c23_swingnet_eval.json`): fresh
  full-clip predictions are bitwise-identical to stored c21/c22 values (max
  abs confidence delta 0.0), original 8 event labels and checkpoint sha256
  unchanged, same-frame clustering honesty finding preserved. Bracket sanity:
  frames 170-195 show a stationary standing pose with no swing, and visual
  contact evidence sits near frames 50-52 (ball airborne by ~55), so the
  predicted bracket [180,184] is not a defensible visual contact bracket; it
  remains a faithfully recorded `candidate_bracket_only` SwingNet prediction
  and is preserved unchanged with exact contact unavailable.
- Ball tracking (`c23_ball_eval/c23_ball_eval_report.json`): the accepted clip
  stays honestly unavailable end-to-end (all 52 raw boxes rejected, no false
  markers rendered, tracer correctly absent). On secondary clips the plausibility
  gate accepted non-ball objects: 9 of 10 point-checked detections on
  pexels_6573486 were trouser/shoe false positives (confidence up to 0.97), and
  pexels_6573612 mixed a verified static-ball track with a degenerate y=540
  center-line false track. Geometry-only gating cannot separate these safely,
  so no gate change was made; these remain per-clip evidence, not cross-clip
  truth.
- Trajectory/landing (`c23_trajectory_eval/c23_trajectory_eval_verdict.json`):
  a c23 HSV-heuristic candidate on pexels_6573612 produced 22 consecutive
  visually-verified pixel-space flight points (frames 155-176), exceeding the
  8-point gate and the prior best of 2 - but the repo's gated ball model never
  accepted these points, the tail is faint, and the prior "best 2" points on
  6573486 were visually rejected (shoe tread). NOT promoted: trajectory and
  landing remain unavailable in accepted diagnostics, and no course-space claim
  exists. Promotion would require the gated model itself to produce and verify
  the points.
- Clubhead/contact recon (`c23_clubhead_recon/recon_summary.json`): still no
  local, runnable, license-clear clubhead/contact checkpoint (only ball-only
  YOLOv8n weights, AGPL-3.0). Exactly one local clip passes the content gate -
  highfps_candidate (V1 Sports 240fps close-up, clubhead+ball sharp at
  ~25.6-26.4 s) - but it is license-unclear proprietary footage, so the
  unavailable state is preserved. pexels_6573618's clubhead is a single-frame
  motion-blur streak at 30 fps; contact exists near 1.83 s but the clubhead is
  never resolvable.

All outputs remain `research_only=true`, `ground_truth=false`, and
`production_eligible=false`. Analytics, shot event, landing, calibration, and
recommendation stay unavailable in the accepted diagnostics; production gates
remain closed and human fallback unchanged.
