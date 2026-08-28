# FairwayOS-Owned Evaluation Dataset and Public-Source Acquisition Plan

> **For Hermes:** Keep GolfDB acquisition planning unchanged. Keep SwingNet research-only. Do not enable production perception, train custom weights, or alter core analytics/workflows until the licensing, annotation schema, and evaluation gates are separately approved.

**Goal:** Design a small, reproducible GhostCaddie-owned evaluation dataset while separately acquiring permitted public GolfDB/PGA research material for phase/impact evaluation and qualitative ingestion stress testing.

**Architecture:** Maintain three strictly separate evidence domains: (1) GolfDB labeled material for SwingNet event evaluation, (2) public PGA TOUR material for bounded qualitative stress testing only, and (3) GhostCaddie-owned consented clips for end-to-end perception evaluation. Public clips must never be represented as consented, owned, production-cleared, or ground-truth data. All course-space evidence remains separate from swing-only evidence.

**Tech Stack:** Local-only processing, `.venv-video-ai` for optional AI, `.venv-video-modern/bin/yt-dlp` with Node/EJS runtime for permitted public-video acquisition, OpenCV/FFmpeg for media inspection, standard-library SHA-256 manifests, versioned JSON annotations, `python3 -m unittest`, and `compileall`.

---

## Non-goals and hard constraints

- Do not modify the existing GolfDB acquisition plan at `.hermes/plans/2026-08-28_134500-golfdb-swingnet-data-acquisition-evaluation.md`.
- Keep SwingNet research-only until licensed labeled evaluation passes.
- Do not use generic YOLO person/pose output for golf phases or impact.
- Do not treat SwingNet predictions as validated GhostCaddie phase or impact evidence before labeled-data scoring.
- Do not train custom weights during this design/acquisition phase.
- Do not change core analytics, `run_pipeline()`, current CLI behavior, human fallback, or the 265-test baseline.
- Do not claim production accuracy from public PGA clips.
- Do not call public PGA footage consented, owned, or production-cleared.
- Do not redistribute downloaded footage.
- Do not upload video, annotations, or models to cloud services.
- Never use cookies, credentials, browser sessions, proxies, DRM bypass, authentication bypass, age-restriction bypass, rate-limit bypass, scraping, or platform-protection bypasses.
- Keep source URLs and local absolute paths out of serialized runtime reports unless an approved acquisition manifest explicitly requires a source URL; runtime diagnostics must use sanitized identifiers.

---

## Dataset domains and allowed claims

### Domain A: GolfDB labeled evaluation

Purpose: evaluate official SwingNet event sequencing against compatible GolfDB labels.

Allowed outputs:

- Address, Toe-up, Mid-backswing, Top, Mid-downswing, Impact, Mid-follow-through, Finish;
- PCE;
- exact-frame accuracy;
- frame-tolerance accuracy;
- per-event metrics;
- latency and failure reports.

Not allowed:

- ball/clubhead/landing claims;
- course calibration;
- GhostCaddie expected-strokes or recommendations;
- promotion to production GhostCaddie phase/impact evidence without separate approval.

### Domain B: Public PGA TOUR research stress set

Purpose: test URL validation, bounded downloading, media inspection, frame extraction, generic detector behavior, failure classification, and annotation/report rendering.

Allowed outputs:

- ingestion success/failure;
- media metadata;
- frame/contact-sheet artifacts;
- qualitative person/pose observations;
- camera-cut, multiple-person, occlusion, resolution, and overlay diagnostics;
- explicit unavailable values.

Not allowed:

- ground-truth accuracy metrics unless matching labels exist;
- consent or ownership claims;
- production model claims;
- course-space recommendations or analytics;
- public-video redistribution.

### Domain C: GhostCaddie-owned evaluation set

Purpose: validate the complete automatic perception stack after lawful ownership/permission, annotation, and evaluation approval.

Only this domain may eventually support end-to-end product evaluation, and only after every gate passes.

---

## Target dataset design

### Size and composition

Create 3–10 consented single-shot clips, with the preferred composition:

- 3 clips minimum for an initial smoke evaluation;
- 6–10 clips for a meaningful small-set evaluation;
- at least one face-on clip and one down-the-line clip;
- both right-handed and left-handed golfers when available;
- mixed club contexts only if each club is explicitly labeled;
- one golfer and one shot per clip for the first release;
- no cuts, edits, overlays, or other golfers in the target track;
- pre-address and post-finish coverage;
- landing only when actually visible and annotatable.

Preferred media:

- 1080p or higher;
- 60 FPS or higher where available;
- original frame rate and dimensions retained;
- fixed or nearly fixed camera;
- sufficient shutter speed and lighting to expose the ball/clubhead where possible.

### Dataset split

Reserve the split before tuning:

- **Train/tuning:** 1–6 clips, used only for adapter development and threshold selection;
- **Validation:** 1–2 different clips, used for gate tuning and error analysis;
- **Held-out:** at least 1 clip never used for tuning, final acceptance only.

For a three-clip minimum, use one tuning, one validation, and one held-out clip, while labeling the result as low-power evidence. Do not pool frames from one clip across splits.

Record the split assignment in a manifest and prohibit reassignment after evaluation begins without a new manifest version.

---

## Ownership and permission records

For every GhostCaddie-owned clip, retain an external permission record that is not embedded in public/runtime artifacts:

- contributor/owner identity and contact location;
- explicit permission for local storage, annotation, model evaluation, and internal report generation;
- permission scope and expiration, if any;
- whether the golfer is identifiable;
- withdrawal/deletion procedure;
- recording date and capture device, where permitted;
- whether derivative annotated frames/video may be retained;
- reviewer and approval date.

The runtime dataset manifest should contain only a non-identifying clip ID, permission status, permission-record reference, and retention classification. A public PGA or GolfDB download must never be put in the GhostCaddie-owned domain.

---

## Annotation schema: `ghostcaddie-ground-truth.v1`

Create a strict, versioned JSON schema with explicit unavailable/occluded states. No placeholder coordinates are valid.

Top-level fields:

```json
{
  "schema_version": "ghostcaddie-ground-truth.v1",
  "clip_id": "non-identifying-id",
  "video": {
    "width": 1920,
    "height": 1080,
    "fps": 60.0,
    "frame_count": 0,
    "view": "face_on"
  },
  "split": "held_out",
  "frames": [],
  "events": {},
  "calibration": null,
  "provenance": {
    "annotation_version": "v1",
    "annotator_id": "redacted-or-internal-id",
    "review_status": "pending"
  }
}
```

Each frame annotation must support:

- golfer: bounding box, stable track ID, visibility, occlusion, confidence/review state;
- anchor: explicit pixel point, anchor definition, visibility, occlusion, and unavailable reason;
- ball: explicit pixel point or unavailable/occluded state;
- clubhead: explicit pixel point or unavailable/occluded state;
- club: label only when observed or separately documented;
- phase: `address`, `backswing`, `top`, `downswing`, `contact`, `follow_through`, `finish`, or `unknown`;
- impact: exact frame or bounded interval, ambiguity flag, and unavailable reason;
- landing: pixel position only when visible and reviewed, otherwise null with reason;
- frame-level occlusion, blur, lighting, camera-motion, and annotation-exclusion flags.

The event section should preserve GolfDB names without replacing them:

```json
{
  "source_label": "Impact",
  "ghostcaddie_phase": "contact",
  "mapping_provenance": "explicit evaluation mapping; not runtime validation"
}
```

The mapping must be one-way and provenance-preserving. GolfDB `Toe-up`, `Mid-backswing`, `Mid-downswing`, and `Mid-follow-through` do not need to be collapsed into GhostCaddie phases for scoring; retain the original labels for model evaluation.

### Annotation review

Use two-pass annotation:

1. primary annotation of all required fields;
2. independent review of golfer track, anchor, phase, impact, ball, clubhead, and landing fields.

Resolve disagreements explicitly. If unresolved, mark the field unavailable or ambiguous rather than choosing a convenient value.

---

## Four-point calibration design

Only clips with a stable camera and suitable visible landmarks may receive course-space calibration.

For each eligible clip:

- exactly four image pixel points;
- four corresponding engine/course coordinates;
- source dimensions matching the video exactly;
- landmark descriptions and reviewer status;
- calibration residual/error report;
- calibration version and SHA-256 hash.

Calibration is not required for swing-only phase/impact evaluation. It is prohibited from entering a report merely because course-space analysis is desired. Calibration must be applied once at the existing reconstruction boundary and never during pixel-space comparison.

---

## Acquisition workflow

### Step A: GolfDB first

1. Preserve the existing GolfDB acquisition plan unchanged.
2. Verify repository commit, implementation files, dataset metadata, preprocessed-video source, checkpoint source, and applicable licenses.
3. Acquire only with direct HTTPS and no credentials, cookies, browser sessions, or bypasses.
4. Store data only under the isolated evaluation directory.
5. Hash every downloaded file with SHA-256.
6. Generate and verify `val_split_1.pkl` through `val_split_4.pkl` according to the official split generator.
7. Match each annotation row to `<id>.mp4` in `videos_160`.
8. Reject missing, duplicate, or ambiguous matches.
9. Run official SwingNet evaluation and report PCE, exact-frame, frame-tolerance, per-event, latency, and failures.
10. Keep checkpoint/dataset license status explicit. If unresolved, retain research-only smoke artifacts and do not promote.

### Step B: Public PGA TOUR qualitative stress set

Use only publicly accessible official PGA TOUR sources identified at acquisition time. Candidate sources must be checked individually for platform availability and terms; search results are not license grants.

For each selected source:

1. record sanitized platform/video ID, channel/source identity, retrieval time, and license status;
2. use the configured executable:

   ```text
   /Users/giofiore/ghostcaddie-tour/.venv-video-modern/bin/yt-dlp
   ```

3. pass the configured Node/EJS runtime:

   ```text
   --js-runtimes node:/usr/local/bin/node
   ```

4. use `--no-playlist`;
5. request a bounded segment no longer than 20 seconds;
6. prefer 720p or higher where available, while retaining download-size, disk, timeout, and media-validation limits;
7. select short single-swing segments only when the source visibly supports that choice;
8. retain downloaded media locally only and do not redistribute it;
9. extract frames/contact sheets and generate qualitative diagnostics;
10. record whether the source is broadcast, edited, moving-camera, multi-golfer, occluded, overlaid, or otherwise unsuitable;
11. never create ground truth from model output or visual assumption;
12. keep course-space analytics and recommendations blocked.

If a source is unavailable, age-restricted, authentication-gated, or protected, record a sanitized failure category and move to another public candidate. Do not bypass the restriction.

---

## Reproducible manifests and reports

Create separate manifests:

```text
out/ghostcaddie_evaluation/dataset_manifest.json
out/ghostcaddie_evaluation/permission_manifest.json
out/ghostcaddie_evaluation/hash_manifest.json
out/ghostcaddie_evaluation/evaluation.json
```

Every report must include:

- schema version;
- dataset domain (`golfdb`, `public_pga_stress`, or `ghostcaddie_owned`);
- split and clip IDs;
- source commit/version;
- model name/version and weight hash;
- dataset/media hashes;
- package/runtime/device versions;
- evaluation configuration and frame tolerance;
- per-clip and aggregate metrics;
- unavailable metrics with explicit reasons;
- failure counts and sanitized error categories;
- artifact references using safe relative paths;
- license status and promotion status.

Do not serialize raw consent records, credentials, cookies, absolute paths, or hidden user data.

### Required metrics for owned clips

At minimum:

- golfer detection precision/recall or reviewed detection coverage;
- track continuity and longest gap;
- anchor pixel error and coverage;
- phase accuracy and confusion matrix;
- impact frame absolute error and within-tolerance accuracy;
- ball precision/recall when labeled visible;
- clubhead precision/recall when labeled visible;
- landing error only where labeled visible;
- false positives;
- runtime, mean/p95 latency, and peak memory;
- per-view and per-clip breakdown;
- held-out results separated from tuning results.

For GolfDB, additionally report:

- PCE;
- exact-frame accuracy;
- frame-tolerance accuracy;
- per-event metrics for all eight official labels;
- skipped/mismatched clip counts;
- official split identifiers.

For public PGA stress clips, report only qualitative/operational metrics and unavailable values for ground-truth accuracy.

---

## Annotation and evaluation gates

### Dataset readiness gate

Pass only when:

- 3–10 clips are available for the owned-domain design target;
- ownership/permission records are complete;
- all media and annotation hashes are recorded;
- schema validation passes;
- split assignments are frozen;
- at least one held-out clip is reserved;
- view labels include face-on and down-the-line examples where available.

### Perception evaluation gate

Pass only when:

- labels have independent review;
- golfer/anchor/phase/impact metrics are computed on held-out data;
- thresholds are frozen before held-out scoring;
- unavailable and occluded fields remain explicit;
- annotated comparison artifacts are visually inspected;
- model and dataset provenance is complete.

### Production promotion gate

This plan does not open the production gate. A later approval is required after measured held-out results demonstrate the documented thresholds. SwingNet alone cannot open ball, clubhead, landing, calibration, hazard, expected-strokes, or recommendation gates.

---

## TDD and verification plan for later implementation

No production code is changed by this design. If an evaluator or schema validator is later implemented:

1. add failing `unittest` cases first;
2. verify RED for schema, hash, license, split-matching, and tolerance behavior;
3. implement the smallest isolated evaluator;
4. run focused tests GREEN;
5. run the full suite;
6. run compileall;
7. run existing CLI scenarios;
8. inspect generated JSON and annotated artifacts;
9. verify no recommendation or course-space artifact is produced from swing-only/public stress data;
10. confirm the 265-test baseline remains green.

Required later commands:

```bash
python3 -m unittest tests.test_swingnet_evaluation
python3 -m unittest tests.test_owned_dataset_schema
python3 -m unittest discover -s tests
python3 -m compileall -q ghostcaddie tests scripts
```

## Expected blocked outcome

If public-source terms, GolfDB data, checkpoint permissions, or owned-clip permissions cannot be cleared:

- retain the acquisition plan and research-only artifacts;
- report hashes for files actually obtained;
- set accuracy metrics to `null` when labels are absent;
- preserve qualitative stress diagnostics where lawful;
- do not train custom weights;
- do not enable SwingNet in production;
- do not claim consent, ownership, production clearance, or model accuracy;
- keep the human fallback and all current GhostCaddie gates unchanged.
