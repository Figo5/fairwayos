# Automatic Golf Perception: Data-Dependent Evaluation Workflow

Status: **prepared, not run**.

This document defines the next evaluation run. It does not validate automatic golf perception and does not change the current contracts, gates, CLI, human fallback, YouTube ingestion, `run_pipeline()`, or core analytics.

## Current data status

No suitable real acceptance clip is currently available in this project. The existing media is not an acceptance dataset:

- `out/youtube_smoke_480p/source.mp4` is a broadcast stress/negative case;
- `out/youtube_smoke_format160/source.mp4` is ingestion-only;
- `out/hitl_demo/demo_golf.mp4` is a human-in-the-loop demo artifact, not a consenting, annotated automatic-perception evaluation clip.

Therefore no automatic evaluation has been run, no thresholds have been tuned, and no automatic golf-analysis reliability claim is permitted.

## Required inputs

The evaluator must receive all of the following before a run starts. Do not substitute synthetic fixtures, the YouTube artifact, unannotated footage, or guessed values.

### 1. High-resolution single-shot clip

Provide a local file, preferably:

- 1080p or higher;
- 60 FPS or higher when available;
- one consenting golfer and one shot;
- continuous view with no cuts, edits, or other golfers entering the target track;
- enough pre-shot and post-impact frames to cover address through landing when visible;
- original frame dimensions and frame rate retained;
- no upload to cloud services.

Record a consent/ownership note outside the serialized report. The evaluator must never serialize the absolute source path.

### 2. Four-point image-to-engine calibration

Provide exactly four paired points for the same video dimensions. Each pair contains:

- `image`: source pixel coordinate `{ "x": number, "y": number }`;
- `engine`: corresponding engine/course coordinate `{ "x": number, "y": number }`.

Example shape (placeholders only; these values are **not** valid evaluation data):

```json
{
  "schema_version": "video-calibration.v1",
  "image_width": 1920,
  "image_height": 1080,
  "source_units": "pixels",
  "engine_units": "yards",
  "points": [
    {"image": {"x": 0, "y": 0}, "engine": {"x": 0, "y": 0}},
    {"image": {"x": 0, "y": 0}, "engine": {"x": 0, "y": 0}},
    {"image": {"x": 0, "y": 0}, "engine": {"x": 0, "y": 0}},
    {"image": {"x": 0, "y": 0}, "engine": {"x": 0, "y": 0}}
  ]
}
```

Replace every placeholder with measured points. The four image points must be valid, paired, within the source image, and sufficient for the existing four-point mapper. Calibration is supplied once and must not be applied a second time during comparison.

### 3. Ground-truth annotation document

Provide one versioned JSON document for the exact clip. It must identify the video dimensions and contain frame-indexed annotations for:

- golfer bounding box and stable golfer track identity;
- feet/body anchor, with visibility/occlusion state;
- golf ball position, visibility, and occlusion state;
- clubhead position, visibility, and occlusion state;
- swing phase (`address`, `backswing`, `top`, `downswing`, `contact`, `follow_through`, or explicit `unknown`);
- impact frame or a bounded impact interval, with ambiguity represented explicitly;
- landing position when actually visible, otherwise unavailable;
- annotation provenance, annotator/version, and any excluded frames.

A minimal illustrative shape is shown below. It is a schema template, not ground truth and must not be executed as-is:

```json
{
  "schema_version": "automatic-ground-truth.v1",
  "video": {"width": 1920, "height": 1080, "fps": 60.0},
  "frames": [
    {
      "frame_index": 0,
      "golfer": {
        "track_id": "replace-me",
        "bbox": {"x": 0, "y": 0, "width": 1, "height": 1},
        "anchor": {"x": 0, "y": 0},
        "visible": true
      },
      "ball": {"x": 0, "y": 0, "visible": false},
      "clubhead": {"x": 0, "y": 0, "visible": false},
      "phase": "unknown",
      "impact": {"is_candidate": false, "ambiguous": true},
      "landing": null
    }
  ],
  "provenance": {
    "annotator": "replace-me",
    "annotation_version": "replace-me"
  }
}
```

Coordinates and visibility must be observations, not inferred placeholders. If an object cannot be annotated reliably, use the document's explicit unavailable representation; never fill it with a predicted or guessed coordinate. A second held-out clip or held-out segment must be reserved before tuning thresholds.

### 4. Approved detector/model weights

Provide a model manifest before inference. Approval must identify:

- detector/tracker/pose model name and version;
- local weight path or approved package reference, kept outside the standard analytics core;
- source URL or publisher, dataset/source, and license;
- training/fine-tuning method if custom;
- SHA-256 model hash;
- permitted use and approval owner/date;
- expected input resolution and class/keypoint semantics.

Do not add custom weights until this provenance and approval record exists. Generic COCO person, sports-ball, or pose weights are not automatically golf detectors and cannot be described as validated golf models.

## Next-run command template

The current `video-automatic-analyze` command is the guarded execution boundary for an already-produced, approved `video-observations.v1` document. It does **not** accept ground truth or model weights and must not be presented as the complete evaluation command.

For best-effort arbitrary public YouTube ingestion, use the implemented command below. It is not an accuracy or reliability claim:

```bash
python3 -m ghostcaddie youtube-auto-try \\
  --url "https://youtu.be/VIDEO_ID" \\
  --out out/youtube_auto_try \\
  --segment-start 0 \\
  --segment-duration 20 \\
  --yt-dlp /Users/giofiore/ghostcaddie-tour/.venv-video-modern/bin/yt-dlp \\
  --render-video
```

The command uses the modern configured yt-dlp executable by default and passes `--js-runtimes node:/usr/local/bin/node`. `--yt-dlp` is optional and is validated as an executable regular file. The bounded low-resolution selector, 20-second maximum segment, size/disk/timeout limits, `--no-playlist`, `shell=False`, and no-credential/no-cookie/no-proxy policy remain enforced.

Its output is useful even when blocked: `diagnostics.json`, extracted frames, a contact sheet, and copied annotated-frame artifacts are produced after successful ingestion. Missing detector, camera cuts, multiple golfers, low confidence, missing ball/club/contact/landing, and missing calibration remain explicit blocking reasons. Without calibration, output remains pixel-space and no recommendation is emitted. `--fallback-human` only prepares the explicit human annotation workspace.

After an approved evaluator adapter is supplied, the planned data-dependent comparison command should have this shape:

```bash
# TEMPLATE ONLY — not executable until a real clip, annotations,
# calibration, approved adapter, and approved model manifest exist.
python3 scripts/run_automatic_evaluation.py \
  --video /absolute/path/to/consented_single_shot.mp4 \
  --calibration calibration.json \
  --ground-truth ground_truth.json \
  --model-manifest approved_model_manifest.json \
  --adapter-module approved_adapter_module \
  --project-root /path/to/project \
  --out out/automatic_evaluation_run \
  --held-out-split held-out \
  --sample-fps 60
```

The evaluator command must:

1. validate the clip, dimensions, calibration, annotation schema, model manifest, and project boundaries;
2. run the approved local detector/tracker adapter to produce automatic `video-observations.v1` evidence;
3. compare automatic observations against the ground-truth document without silently converting human annotations into automatic evidence;
4. calculate track continuity, anchor error, impact-frame error, ball precision/recall, clubhead precision/recall, landing error, false-positive rate, runtime, and peak memory;
5. write deterministic `evaluation.json` with unavailable metrics represented as `null` plus reasons;
6. write annotated comparison frames showing automatic evidence, ground truth, disagreements, occlusions, and gate failures;
7. report the exact provisional thresholds used and whether each gate passed;
8. keep source URLs, credentials, cookies, absolute paths, and consent records out of serialized reports;
9. refuse to emit a recommendation or invoke unchanged `run_pipeline()` unless all existing automatic sequence, confidence, evidence, and calibration gates pass.

The command must not silently fall back to human annotations. Human fallback remains an explicit separate workflow.

## Acceptance rule

Automatic perception may only be described as passing after:

- the command runs on real, consenting, high-resolution golf footage;
- thresholds are tuned only using the designated tuning data;
- the final metrics are computed on held-out real footage;
- every provisional gate passes, including continuity, anchor, impact, ball, clubhead, landing, false-positive, runtime, and memory criteria;
- the resulting `evaluation.json` and annotated comparison frames are inspected;
- model provenance and approval are recorded.

Until then, the valid status is **infrastructure ready; automatic golf perception not validated**.

## Existing command for the current guarded boundary

For a validated observation document produced by an approved adapter, the existing command remains:

```bash
python3 -m ghostcaddie video-automatic-analyze \
  --video /absolute/path/to/local_video.mp4 \
  --observations observations.json \
  --calibration calibration.json \
  --course course.json \
  --player player.json \
  --project-root /path/to/project \
  --out out/automatic_boundary_run
```

This command is not a substitute for the data-dependent comparison workflow above. It is a guarded integration path and must continue to block incomplete or unvalidated evidence.
