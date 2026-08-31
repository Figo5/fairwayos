# FairwayOS AI Demo Mode

`ghostcaddie ai-demo` is a bounded, local-first, research-only visual demo. It is
separate from `run_pipeline()` and cannot produce validated golf analytics.

## Usage

```text
ghostcaddie ai-demo --url https://www.youtube.com/watch?v=<11-char-id> --out <dir>
ghostcaddie ai-demo --video <local-file> --out <dir>
```

YouTube input uses the existing allowlisted, no-playlist downloader boundary.
The downloaded source remains local and ignored. Use only clips whose local
research use is permitted by the source terms; a successful download is not a
rights determination and is not ground truth.

Important bounds are explicit in the CLI:

- HTTPS canonical YouTube URLs only;
- one video, never a playlist;
- bounded segment duration (default 20 seconds at the ingestion boundary);
- bounded sample rate and frame count;
- local OpenCV/Ultralytics inference only;
- no cookies, credentials, cloud upload, DRM bypass, scraping, or proxies.

## Outputs

The output directory contains local ignored artifacts:

- `annotated_video.mp4`: H.264/yuv420p viewable demo video;
- `annotated_frames/`: sampled annotated frames for visual QA;
- `diagnostics.json`: `fairwayos-ai-demo.v1` report;
- `provenance.json`: source/acquisition provenance when acquired through YouTube.

Diagnostics record source identity, media metadata, selected motion window,
methods compared, candidate counts, states, confidence, uncertainty, warnings,
and safe relative artifact references. Absolute local paths and credentials are
not serialized.

## Evidence states and labels

Every visual field uses one of:

- `observed`: direct current-frame detector/model evidence;
- `interpolated`: reconstructed only between bounded observed frames;
- `predicted`: extrapolated and visibly labeled as such;
- `unavailable`: no defensible evidence; coordinates remain null.

The demo can show:

- golfer box and local pose keypoints;
- local golf-ball candidates and a guarded pixel-space tracer; the renderer records
  `rendered_overlay.marker`, `rendered_overlay.tracer_points`, and
  `rendered_overlay.zoom_inset` beside every rendered ball observation;
- classical frame-difference motion used for swing-window selection;
- rejected research clubhead candidates when no validated clubhead checkpoint
  exists;
- unavailable exact-impact/contact evidence;
- confidence, uncertainty, warnings, and research-only labels.

Pseudo-labels, if produced by any research adapter, must retain:

```json
{
  "pseudo_label": true,
  "ground_truth": false,
  "research_only": true,
  "production_eligible": false
}
```

The current demo never promotes clubhead or impact proposals. Obvious false
positives are rejected using bounds, golfer support, confidence, and temporal
support checks. A generic `sports ball` detector is not relabeled as a golf-ball
truth source.

The tracer is persistent only within one guarded continuity segment. It is
cleared on unavailable, terminated, or reacquisition-gap states so unrelated
candidate positions are never joined by a misleading line.

## Closed gates

AI Demo Mode always emits `null`/unavailable for calibration, course
coordinates, trajectory/landing in physical space, `ShotEvent`, expected
strokes, hazards, analytics, and recommendations. The annotated tracer is
pixel-space research visualization only. These fields require the separate
validated analytics mode and its existing evidence gates.

## Visual QA

Do not accept `diagnostics.json` alone. Inspect the actual MP4 and representative
frames. Confirm that golfer/pose overlays align with the golfer and that ball
markers are visibly plausible; reject outputs where markers follow background,
club/ground geometry, or other obvious false positives. Warnings must remain
readable in the rendered video.

All media, frames, models, weights, caches, and generated outputs stay outside
Git. Only source, tests, and this documentation are publishable.
