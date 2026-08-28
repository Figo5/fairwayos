# Human-in-the-Loop Real-Video Analysis Plan

> **For Hermes:** Use the milestone gauntlet and strict TDD after explicit user approval. This document is design-only; do not edit source or test files until approval.

**Goal:** Let a user inspect a real local golf video, explicitly supply or confirm pixel observations and calibration, then run the unchanged GhostCaddie analytics pipeline with transparent provenance and annotated outputs.

**Architecture:** Add a local human-annotation boundary above the existing deterministic video pipeline. The browser UI edits only a versioned annotation document; it never performs analytics or silently invents detections. On explicit save and submit, the CLI validates the document, converts confirmed pixel observations through the existing `VideoCalibration` exactly once, reconstructs the existing `ShotEvent`, calls unchanged `run_pipeline()`, and reprojects recommendations into annotated frames.

**Tech Stack:** Existing Python standard library, ffmpeg/ffprobe, standalone HTML/CSS/JavaScript or SVG, existing `ghostcaddie.video` contracts, existing `CoordinateMapper`, existing analytics engine, and unittest.

---

## 1. Current context and protected boundaries

The accepted foundation currently includes:

- deterministic frame extraction and contact sheets;
- versioned video metadata and diagnostics contracts;
- explicit project-bound paths;
- four-point pixel-to-course calibration and inverse mapping;
- pixel observation contracts with confidence and warnings;
- fixture perception;
- reconstruction into the existing `ShotEvent`;
- unchanged `run_pipeline()` integration;
- deterministic annotations and sampled annotated-video export;
- fixture-mode `video-analyze` CLI;
- 189 passing unittest tests and compileall passing at the latest gate.

Milestone 9 status is fixed:

- adapter safety and validation complete;
- real `gemma4:e2b` image test failed schema compliance;
- model-backed perception remains unavailable;
- no arbitrary-video AI capability is claimed.

The following must remain unchanged:

```text
ghostcaddie/models.py
ghostcaddie/geometry.py
ghostcaddie/pipeline.py
ghostcaddie/session.py
ghostcaddie/dispersion.py
ghostcaddie/simulation.py
ghostcaddie/expected_strokes.py
ghostcaddie/hazards.py
ghostcaddie/decision.py
ghostcaddie/adapters/provider*.py
ghostcaddie/adapters/shotlink.py
ghostcaddie/adapters/trackman.py
```

Fixture mode and the existing `run`, `session`, `provider-session`, and fixture `video-analyze` behavior must remain available.

---

## 2. Option comparison

### Option A — local HTML/SVG annotation tool: recommended

Generate a standalone local HTML page containing:

- sampled frame/contact-sheet thumbnails;
- a selected-frame canvas or SVG overlay;
- coordinate readout;
- controls for frame selection;
- click-to-place points;
- calibration-point labels;
- shot fields and club selection;
- confidence, source, and warning controls;
- explicit Save Draft and Submit Annotations actions.

The page writes no files silently. It can:

1. download a JSON annotation document through a browser download;
2. display a copyable JSON payload;
3. optionally use a narrowly scoped local helper command in a later iteration.

The CLI accepts the saved JSON explicitly. This keeps the annotation UI stateless with respect to analytics and makes the saved document reviewable, testable, and replayable.

**Advantages:** usable visual point selection, no Python UI dependency, easy to inspect, local/offline, preserves the existing CLI composition root, and can run from a generated file or preview pane.

**Risks:** browser download handling is user-mediated; the first version should not rely on arbitrary browser filesystem writes or a local HTTP server.

### Option B — CLI-only workflow

Provide commands such as:

```text
video-annotate --video ... --out annotation_workspace/
video-submit --workspace ... --annotations annotations.json ...
```

The CLI prints frame paths and dimensions, then accepts coordinates and frame indices interactively or through flags.

**Advantages:** smallest dependency surface and easiest security model.

**Risks:** poor usability for real video; users must manually determine pixel coordinates; difficult to confirm visual alignment; likely to cause coordinate-entry mistakes.

This is a useful fallback for headless environments but not the recommended primary workflow.

### Option C — minimal desktop/UI dependency

Add a Tkinter or similar desktop picker.

**Advantages:** direct file saving and native coordinate events.

**Risks:** platform-specific behavior, display dependency, additional UI testing burden, packaging complexity, and a less portable development path. It is unnecessary for the first reliable slice.

### Recommendation

Use **Option A**, with a CLI-generated standalone HTML/SVG annotator and an explicit JSON save/submit boundary. Keep Option B as a headless fallback. Do not introduce a desktop/UI dependency in this slice.

The first version should use a standalone file rather than a server. If browser restrictions later make the workflow too cumbersome, add a loopback-only server as a separately reviewed feature; do not make it a hidden default.

---

## 3. Proposed workflow

```text
REAL LOCAL MP4/MOV
        |
        v
validate readable source path
        |
        v
ffprobe metadata + deterministic sampled frames
        |
        v
contact sheet + annotation_workspace/index.html
        |
        v
user selects frames and supplies/confirms points
        |
        v
explicit downloaded annotation.json
        |
        v
CLI validates schema, paths, dimensions, frames, coordinates
        |
        v
VideoCalibration + confirmed pixel observations
        |
        v
CoordinateMapper.to_engine() exactly once per accepted position
        |
        v
existing ShotEvent reconstruction
        |
        v
unchanged run_pipeline()
        |
        v
recommendation.json + overlay.svg + diagnostics.json
        |
        v
reprojection with CoordinateMapper.from_engine()
        |
        v
annotated frames + optional sampled annotated_video.mp4
```

The UI is an evidence-collection tool only. It must not calculate expected strokes, select clubs, classify hazards, infer identity, or call a model.

---

## 4. Human annotation input contract

Add a new versioned document, proposed as:

```text
video-human-annotations.v1
```

Example shape:

```json
{
  "schema_version": "video-human-annotations.v1",
  "source": {
    "video_identifier": "user-supplied-label",
    "width": 1920,
    "height": 1080,
    "frame_count": 120,
    "frame_rate": 30.0,
    "duration_seconds": 4.0,
    "sampling": {
      "method": "fixed_interval",
      "sample_fps": 2.0,
      "frame_indices": [0, 15, 30, 45],
      "timestamps_seconds": [0.0, 0.5, 1.0, 1.5]
    }
  },
  "calibration": {
    "source_units": "pixels",
    "engine_units": "yards",
    "source_points": [
      {"x": 320, "y": 700},
      {"x": 1580, "y": 700},
      {"x": 1380, "y": 260},
      {"x": 520, "y": 260}
    ],
    "engine_points": [
      {"x": 0, "y": 0},
      {"x": 220, "y": 0},
      {"x": 220, "y": 160},
      {"x": 0, "y": 160}
    ],
    "point_order": ["near_left", "near_right", "far_right", "far_left"]
  },
  "shot": {
    "event_id": "VIDEO-HUMAN-0001",
    "hole_number": 1,
    "shot_number": 1,
    "lie": "fairway",
    "club": {
      "value": "7i",
      "source": "user_supplied",
      "confidence": 1.0
    },
    "distance_to_pin": 150.0,
    "wind": {"speed_mph": 0.0, "direction_deg": 0.0},
    "timestamp": "user-supplied-label",
    "target_pixel": {
      "x": 1100,
      "y": 340,
      "source": "user_supplied",
      "confidence": 1.0
    }
  },
  "observations": {
    "address": {
      "frame_index": 15,
      "timestamp_seconds": 0.5,
      "golfer_anchor": {
        "x": 940,
        "y": 780,
        "source": "user_supplied",
        "confidence": 1.0
      },
      "ball": {
        "x": 965,
        "y": 782,
        "source": "user_confirmed",
        "confidence": 0.9
      },
      "clubhead": null,
      "warnings": ["club_not_visible"]
    },
    "contact": {
      "frame_index": 30,
      "timestamp_seconds": 1.0,
      "clubhead": {
        "x": 980,
        "y": 775,
        "source": "user_supplied",
        "confidence": 0.85
      },
      "ball": null,
      "warnings": ["ball_missing"]
    },
    "landing": {
      "frame_index": 75,
      "timestamp_seconds": 2.5,
      "position": {
        "x": 1250,
        "y": 500,
        "source": "user_confirmed",
        "confidence": 0.8
      },
      "status": "observed"
    }
  },
  "submission": {
    "state": "draft|submitted",
    "submitted_explicitly": true,
    "annotation_version": 1
  }
}
```

### Required source labels

Each non-null value must declare one of:

```text
user_supplied
user_confirmed
observed
inferred
unavailable
```

The first implementation should allow `unavailable` only as an explicit null-compatible state and should not represent it as a fake coordinate.

The source label is separate from confidence:

- `user_supplied` means the user entered the value directly;
- `user_confirmed` means the UI displayed evidence and the user accepted it;
- `observed` means a future detector supplied it and the user confirmed it;
- `inferred` means a derived value, which must include a method and warning;
- `unavailable` means no value is present.

For this slice, automatic model perception remains separate and is not invoked by the human workflow.

### Required user-confirmable fields

- four ordered calibration points and matching engine points;
- golfer anchor at address;
- optional golfer box/keypoints if desired for annotation only;
- ball position at address or another selected frame;
- clubhead position where visible;
- contact frame;
- intended-direction vector or target pixel;
- landing position when visible;
- club selection if not observable;
- lie, distance, wind, and other required shot context.

### Validation rules

Reject:

- missing schema version;
- draft documents submitted without an explicit submit action;
- missing required shot context;
- non-finite values;
- negative or out-of-range frame indices;
- timestamps inconsistent with source metadata;
- coordinates outside image dimensions;
- invalid calibration point count or ordering;
- duplicate calibration points;
- degenerate or near-singular calibration;
- ambiguous source labels;
- unknown phase names;
- user-entered landing with no source/confidence metadata;
- extra fabricated fields in strict mode;
- absolute or traversal paths in annotation documents;
- source-video path serialization.

The annotation document should contain an opaque user-supplied video identifier or a safe basename label, not an absolute path.

---

## 5. UI behavior and explicit save/submit boundary

### Workspace generation

Proposed command:

```bash
python3 -m ghostcaddie video-annotate \
  --video /absolute/path/shot.mp4 \
  --project-root . \
  --course data/sample_hole.json \
  --player data/sample_player.json \
  --out out/human-shot/
```

This command:

1. validates the local video;
2. runs `ffprobe`;
3. extracts deterministic frames;
4. creates a contact sheet;
5. writes a standalone `annotation_workspace/index.html`;
6. writes a safe frame manifest;
7. prints the exact output files and next command.

It must not run analytics or write a recommendation.

### Browser interaction

The page should provide:

- frame thumbnails and a selected-frame view;
- image dimensions and current frame/timestamp;
- click-to-place coordinate mode;
- named point modes: calibration 1–4, golfer anchor, ball, clubhead, contact, target, landing;
- frame selector for temporal fields;
- club dropdown or free-form validated club field;
- source/provenance selector constrained to the allowed enum;
- confidence entry constrained to `[0, 1]`;
- warning checklist;
- reset point and clear draft controls;
- a visible validation summary;
- `Save Draft` and `Submit Annotations` controls.

`Save Draft` may download a draft but cannot invoke analysis. `Submit Annotations` must:

- require all mandatory fields;
- set `submission.state` to `submitted`;
- set `submitted_explicitly` to `true`;
- serialize a deterministic JSON document;
- instruct the user to pass that document to `video-submit`.

The browser page must not use network requests, external scripts, CDNs, analytics, cookies, or identity services.

### Submission command

Proposed command:

```bash
python3 -m ghostcaddie video-submit \
  --video /absolute/path/shot.mp4 \
  --annotations annotation_workspace/annotations.json \
  --project-root . \
  --course data/sample_hole.json \
  --player data/sample_player.json \
  --out out/human-shot/
```

`--video` remains the only input allowed to be absolute. The annotations, course, player, and any other project resources must resolve through `ProjectBoundary`.

A combined convenience command may be considered later, but separate workspace generation and submit commands provide the clearest explicit-save boundary and safest failure behavior.

---

## 6. Coordinate and analytics flow

### Pixel-space boundary

All user-selected points remain in image pixel coordinates in the annotation document and visual artifacts.

At submission:

```text
user pixel point
  -> validate against source dimensions
  -> VideoCalibration.to_engine()
  -> engine/course coordinate
  -> ShotEvent / unchanged analytics
```

The mapping must occur exactly once per accepted event position:

- origin/golfer anchor;
- target/intended endpoint;
- landing, when available.

The engine must never receive pixel coordinates.

### Reverse projection

For visual output:

```text
recommendation engine point
  -> VideoCalibration.from_engine()
  -> source pixel point
  -> annotated frame
```

`from_engine()` is presentation-only and must never feed back into analytics.

### ShotEvent handling

Use the existing reconstruction seam. Do not add confidence fields to `ShotEvent` and do not alter its validation. Store source labels, confidence, frame references, and warnings in reconstruction metadata and diagnostics.

If required evidence remains unavailable:

- preserve `null` in the annotation document;
- produce a structured unavailable/partial result where allowed;
- do not invent a landing point or club;
- skip analytics when the existing `ShotEvent` contract cannot be safely satisfied;
- still preserve the extracted frames and human annotation document for correction.

---

## 7. Output contract

A successful submission should produce:

```text
out/human-shot/
├── annotation_workspace/
│   ├── index.html
│   └── frame_manifest.json
├── annotations.json
├── diagnostics.json
├── normalized_shot.json
├── recommendation.json
├── overlay.svg
├── contact_sheet.jpg
├── frames/
├── annotated_frames/
└── annotated_video.mp4       # optional sampled sequence
```

`diagnostics.json` remains `video-diagnostics.v1` and must include:

- source metadata without absolute source paths;
- sampling metadata;
- calibration and bounds status;
- frame observations;
- contact and landing timing;
- normalized shot and analytics result when available;
- confidence values;
- warnings/errors;
- model/provider provenance indicating `human_annotation` and no model use;
- artifact references relative to the output package;
- privacy/network status.

The human annotation document should be retained as an auditable input artifact. Its `submission` section should record explicit submission, not a timestamp that would harm deterministic replay unless timestamps are deliberately separated as volatile metadata.

### Status semantics

```text
complete
partial
rejected
```

- `complete`: required fields validated and analytics ran;
- `partial`: frames and/or annotations were valid but analytics was gated by unavailable evidence;
- `rejected`: document or calibration invalid.

---

## 8. Security and privacy model

### Local-only behavior

- local MP4/MOV files only;
- no live APIs, scraping, credentials, or cloud inference;
- no external resources in generated HTML;
- no automatic model invocation;
- no identity recognition;
- no hidden network calls;
- no raw video copies outside the requested output directory.

### Path rules

- absolute local video path is accepted only if it is a regular readable file;
- calibration, course, player, and annotation resources are project-bound;
- reject absolute project-resource paths, `..` traversal, missing files, unreadable files, and symlink escapes;
- output artifact references are relative names only;
- reject annotation documents containing absolute paths or unsafe artifact references;
- never include source paths, environment variables, prompts, or secrets in diagnostics.

### HTML safety

- escape all user-controlled labels before insertion into HTML/SVG;
- do not place user strings into JavaScript source without safe JSON encoding;
- use a strict allowlist for point names, source labels, warning codes, phases, and artifact names;
- do not permit submitted JSON to choose an arbitrary output path;
- do not execute annotation-document values as code;
- reject files larger than configured bounds;
- retain only bounded frame images and JSON data.

### Explicit action boundary

No analytics may run on a draft. The submitter must validate:

```text
submission.state == "submitted"
submission.submitted_explicitly == true
```

This is a safety and audit requirement, not merely a UI label.

---

## 9. Test and fixture strategy

No source or test files should be changed until approval.

### Unit tests

Proposed files:

```text
tests/test_video_human_contracts.py
tests/test_video_human_workspace.py
tests/test_video_human_submit.py
tests/test_video_human_security.py
```

Test first, then implementation, one vertical behavior at a time.

Cover:

- version and required fields;
- explicit draft-vs-submitted gating;
- canonical phase and source-label validation;
- coordinate bounds and finite values;
- frame index/timestamp validation;
- four-point calibration and exact point order;
- duplicate/degenerate calibration;
- unsafe paths and symlink escapes;
- no absolute source path serialization;
- deterministic JSON serialization;
- HTML escaping and no external resource references;
- deterministic workspace generation;
- valid user-selected address/contact/landing fields;
- unavailable ball/landing/club behavior;
- explicit warning preservation;
- exact-once `to_engine()` calls;
- unchanged `run_pipeline()` invocation count;
- recommendation and SVG generation;
- rejection of fabricated fields and malformed JSON.

### Integration tests

Use a tiny ffmpeg-generated synthetic clip, not copyrighted golf footage. Generate it at test setup with:

- fixed dimensions;
- fixed frame rate;
- deterministic frame count;
- visible geometric markers or a test pattern;
- no real person identity.

Tests should:

1. create a workspace from the synthetic clip;
2. inspect the generated HTML and manifest;
3. write a submitted annotation JSON fixture;
4. run `video-submit`;
5. verify normalized `ShotEvent`, diagnostics, recommendation, overlay, and annotations;
6. rerun and compare deterministic fields;
7. verify draft submission is rejected;
8. verify incomplete evidence produces partial/unavailable output rather than fabricated values.

### Visual QA

Visually inspect:

- contact sheet with distinct test-pattern frames;
- HTML workspace frame display;
- calibration point markers;
- golfer anchor, ball, clubhead, contact, target, and landing markers;
- frame/timestamp labels;
- confidence/source labels;
- warnings and unavailable states;
- annotated recommendation and reprojected target;
- optional sampled annotated video representative frame.

Automated checks should verify image dimensions, artifact existence, relative references, and deterministic manifests. Visual inspection is required for coordinate-picker usability and annotation legibility.

---

## 10. Exact files likely to change

No changes before approval.

### New modules

```text
ghostcaddie/video/human_contracts.py
ghostcaddie/video/human_workspace.py
ghostcaddie/video/human_submit.py
ghostcaddie/video/human_html.py
```

Possible responsibilities:

- `human_contracts.py` — versioned annotation dataclasses and strict parser;
- `human_workspace.py` — deterministic workspace manifest and frame references;
- `human_html.py` — standalone escaped HTML/SVG generation with no network assets;
- `human_submit.py` — explicit-submission validation and handoff to existing calibration/reconstruction/orchestration.

Prefer reusing existing `contracts.py`, `paths.py`, `calibration.py`, `observations.py`, `reconstruction.py`, `orchestration.py`, `annotations.py`, and `diagnostics.py` rather than duplicating validation.

### Existing files likely to modify

```text
ghostcaddie/cli.py
ghostcaddie/video/__init__.py
README.md
```

Keep CLI changes limited to new commands and help text. Do not alter existing command semantics.

### New tests

```text
tests/test_video_human_contracts.py
tests/test_video_human_workspace.py
tests/test_video_human_submit.py
tests/test_video_human_security.py
tests/test_video_human_cli.py
```

### Fixture documentation

```text
data/video_fixtures/human_annotation/
├── README.md
├── submitted_annotations.json
└── generated by tests
```

Do not commit large or copyrighted real-video binaries.

---

## 11. Milestone gauntlet after approval

### Milestone H0 — baseline and UI feasibility

- Record the current full unittest count and compileall result.
- Verify ffmpeg/ffprobe.
- Confirm standalone HTML opens locally or in the in-app preview.
- No source/test edits until the baseline is recorded.

**Acceptance:** baseline remains green; no network resources are required; the UI approach is feasible on the local environment.

### Milestone H1 — annotation contract and source semantics

- Add failing tests first.
- Implement `video-human-annotations.v1` parser.
- Add canonical source labels and explicit submission gating.
- Validate coordinates, frames, confidence, calibration, and safe JSON.

**Acceptance:** malformed, draft, unsafe, non-finite, out-of-bounds, and ambiguous documents are rejected; valid submitted documents round-trip deterministically.

### Milestone H2 — workspace and standalone HTML/SVG picker

- Generate deterministic frames/contact sheet.
- Build the offline annotation workspace.
- Add frame selection, point modes, confidence/source controls, and download/save behavior.

**Acceptance:** workspace contains no external resources, displays correct dimensions/timestamps, emits a valid draft, and emits a valid explicitly submitted JSON document. No analytics runs.

### Milestone H3 — human annotations to pixel observations

- Convert submitted human fields into validated `video-observations.v1` or a clearly defined handoff object.
- Preserve user source labels and warnings.
- Keep unavailable values null.

**Acceptance:** pixel observations are validated, canonical phases are used, and no point is fabricated or silently inferred.

### Milestone H4 — submission to existing reconstruction and analytics

- Reuse `VideoCalibration`, reconstruction, and orchestration.
- Assert exact-once mapping.
- Assert exactly one unchanged `run_pipeline()` call.

**Acceptance:** a valid submitted annotation produces the existing `ShotEvent`, recommendation, and SVG with equivalent analytics to an equivalent JSON shot fixture.

### Milestone H5 — diagnostics and reprojected visual output

- Assemble complete diagnostics with human provenance.
- Render annotated frames and optional sampled annotated video.
- Add unavailable and partial states.

**Acceptance:** source, confidence, warning, frame, and provenance fields are complete; artifact references are relative; visual QA passes.

### Milestone H6 — CLI integration and security regression

- Add `video-annotate` and `video-submit` help and commands.
- Test absolute-video/project-resource path distinction.
- Test draft rejection, malformed JSON, traversal, symlink, no-network HTML, and path-leak prevention.
- Run all existing CLI scenarios.

**Acceptance:** human-in-the-loop workflow works end to end on a synthetic clip and real local MP4/MOV input, while all existing behavior remains unchanged.

### Per-milestone quality gates

After every milestone:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q ghostcaddie tests
```

Also run:

- focused milestone tests;
- existing single-shot scenarios;
- session and provider-session fixtures;
- CLI smoke tests;
- deterministic repeated-run comparisons;
- visual inspection for generated media;
- independent inspection of every changed file.

At most two targeted revision loops are allowed per milestone. If a gate fails, stop and report the blocker rather than weakening validation.

---

## 12. Acceptance criteria for the complete next slice

The human-in-the-loop slice is complete only when:

1. A real local MP4 works, and a supported MOV works when ffmpeg supports its codec.
2. Deterministic frames and a contact sheet are generated without loading the entire video into memory.
3. The standalone local HTML/SVG workspace opens without external resources.
4. The user can explicitly select or confirm calibration, golfer anchor, ball, club/contact frame, intended direction, landing, and club.
5. Submitted JSON is versioned, deterministic, strictly validated, and explicitly marked submitted.
6. Draft annotations cannot run analytics.
7. Pixel coordinates are mapped to engine coordinates exactly once.
8. The existing `ShotEvent` contract is reused without adding hidden confidence fields.
9. Existing `run_pipeline()` is called unchanged.
10. Human source labels, confidence, warnings, and frame/timestamp provenance appear in diagnostics.
11. User-supplied, observed, inferred, and unavailable values are visibly distinct.
12. Recommendations and ghost targets are reprojected onto annotated frames.
13. No source paths, secrets, prompts, environment values, or unsafe artifact names appear in reports.
14. No network calls or identity recognition occur.
15. Fixture mode remains available.
16. Existing tests and CLI commands remain green.
17. The workflow does not claim arbitrary-video AI perception or production accuracy.

---

## 13. Rollback strategy

- Keep all new human-in-the-loop code isolated under `ghostcaddie/video/`.
- Revert only the human workspace/submit modules, tests, CLI subcommands, and README section if a milestone fails.
- Do not roll back or modify accepted core analytics, calibration mathematics, provider, session, or fixture-mode behavior.
- If browser interaction proves unreliable, retain the validated JSON contract and provide CLI-only submission as a fallback rather than weakening the contract.
- If real-video perception remains unavailable, the human workflow remains the supported path; no automatic model capability is implied.

---

**Status:** Design complete and saved. No source or test files were edited. Implementation must wait for explicit approval.
