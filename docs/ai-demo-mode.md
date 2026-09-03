# FairwayOS video demo

`ghostcaddie fairwayos-demo` is the user-facing name for the bounded,
local-first visual demo. `ghostcaddie ai-demo` remains an equivalent
compatibility alias. Both commands are separate from `run_pipeline()` and
cannot produce validated golf analytics.

## Usage

Local video:

```text
ghostcaddie fairwayos-demo \
  --video /path/to/local.mp4 \
  --out out/my-demo
```

Bounded YouTube segment:

```text
ghostcaddie fairwayos-demo \
  --url https://www.youtube.com/watch?v=<11-char-id> \
  --segment-start 12 \
  --segment-duration 8 \
  --out out/my-demo
```

For the existing Pexels regression baseline:

```text
ghostcaddie fairwayos-demo \
  --video out/research_training_gauntlet/pexels_6573644/source.mp4 \
  --out out/demo_artifact_bundle \
  --source-platform pexels \
  --source-video-id 6573644 \
  --source-url https://www.pexels.com/video/6573644/
```

YouTube input uses the existing allowlisted, no-playlist downloader boundary.
The downloaded source remains local and ignored. Use only clips whose local
research use is permitted by the source terms; a successful download is not a
rights determination and is not ground truth.

Important bounds are explicit in the CLI:

- HTTPS canonical YouTube URLs only;
- one video, never a playlist;
- bounded segment duration (maximum 20 seconds at the ingestion boundary);
- bounded sample rate, source-time read, and optional frame count;
- local OpenCV/Ultralytics/SwingNet inference only when installed and loaded;
- no cookies, credentials, cloud upload, DRM bypass, scraping, or proxies.

## Outputs

The output directory contains local ignored artifacts:

- `annotated_video.mp4`: H.264/yuv420p viewable demo video;
- `contact_sheet.jpg`: deterministic JPEG sheet containing every rendered frame;
- `annotated_frames/`: sampled annotated frames for visual QA;
- `diagnostics.json`: `fairwayos-ai-demo.v1` report;
- `provenance.json`: source/acquisition provenance and the same relative artifact references.

Diagnostics record source identity, media metadata, selected motion window,
methods compared, candidate counts, states, confidence, uncertainty, warnings,
and safe relative artifact references. Absolute local paths and credentials are
not serialized. For a local copy of externally hosted media, pass
`--source-platform`, `--source-video-id`, and `--source-url` so the provenance
identifies the external asset rather than the local filename. The accepted
research-only visual-demo example uses Pexels video `6573644`; it is not paired
ground-truth evidence. Its local output directory and `source.video_id` must
both remain `6573644`.

## Bounded coarse-to-fine processing

When a local ball model is available, the demo first probes at most four sampled
full frames to locate a research candidate region. Those probes are submitted as one
bounded detector batch. It then rereads only that
motion window at native FPS and runs ball inference on one padded, clipped ROI.
The native ROI pass is hard-bounded to eight frames, 120 seconds, 64 MiB of
frame memory, and 64 candidate records. Pose runs only on the coarse frames and
SwingNet is skipped on the native ROI path to keep optional model work bounded.
Per-inference timeouts and budget termination are recorded in
`render.processing`; partial frames are encoded cleanly when a limit is reached.
Per-inference and model-load timeouts are recorded as distinct warnings; a timeout
leaves the affected model unavailable rather than promoting partial output.

ROI candidates are still research candidates, not labels. Rejected candidates
are rendered only as rejected diagnostics; they never become `observed`, and all
outputs remain `research_only=true`, `ground_truth=false`, and
`production_eligible=false`.


Every visual field uses one of:

- `observed`: direct current-frame detector/model evidence;
- `interpolated`: reconstructed only between bounded observed frames;
- `predicted`: extrapolated and visibly labeled as such;
- `unavailable`: no defensible evidence; coordinates remain null.

The demo can show:

- golfer box and local pose keypoints;
- local golf-ball candidates and a guarded pixel-space tracer; the renderer records
  `rendered_overlay.marker`, `rendered_overlay.tracer_points`, and
  `rendered_overlay.zoom_inset` beside every rendered ball observation. Model
  boxes wider/taller than a quarter of a frame dimension or covering more than
  5% of the frame area are rejected as implausible ball geometry (a golf ball
  cannot fill the frame), and such boxes are skipped individually so a genuine
  detection in the same frame survives. Malformed candidate records and
  non-finite coordinates/confidences are skipped at the tracker boundary rather
  than raising or seeding a non-finite track;
- classical frame-difference motion used for swing-window selection;
- sampled-frame ingestion is bounded by the requested duration in source-frame
  time (using the source FPS, plus one boundary frame), and further constrained
  by `max_frames` when supplied;
- SwingNet event predictions remain research-only; when an `Impact` prediction is
  present, the displayed candidate bracket is derived from the neighboring
  sampled source frames around that prediction. It never validates exact contact
  and remains `ground_truth=false` and `production_eligible=false`;
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

The automatic ball demo now applies strict fail-closed gates before tracking: candidates
inside or overlapping the golfer box, head/hat/torso/hand/foot pose regions, or
wrist-to-ground club/shaft corridors are rejected. Implausible size, aspect ratio,
area, bounds, and confidence are rejected as well. A model candidate must agree
with a `ResearchBallTracker` candidate on consecutive clean source frames before
it can become `ball=observed`; otherwise confidence decays through the tracker and
false tracks terminate quickly. Rejected boxes are rendered with a crossed-out
`BALL REJECTED` label, never as observed markers or zooms. This remains a
false-positive guard only and does not establish golf-ball identity.

Tracer state is cleared on unavailable, terminated, or reacquisition-gap states so
unrelated candidate positions are never joined by a misleading line. Before each
render, only stale generated `annotated_frames/frame_*.jpg` files are removed, so
a shorter rerun cannot encode frames left by an earlier run.

The unified renderer consumes one clean source frame per iteration for every
component, then composes all accepted evidence exactly once. The ball zoom is
cropped from that clean source frame before marker/tracer pixels are added.

MMU is the first ball/tracer acceptance fixture because its moving-ball close-up
is visually inspectable. It does not contain a golfer in the selected view, so
`golfer`, pose, and body anchor remain unavailable there. A golfer-framed clip
must pass a separate visual ball-alignment review before both components can be
claimed together; the demo never substitutes a false positive to satisfy that
condition.


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
