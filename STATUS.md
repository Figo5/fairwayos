# FairwayOS Research Status

Updated 2026-08-31.

## Verified research artifact

The MMU biomechanics clip is the strongest current local demonstration:

- Source: https://www.youtube.com/watch?v=6dG9hb3_blo
- Local source: `out/research_training_gauntlet/mmu_candidate/source.mp4`
- Media: 600x480, encoded 25 FPS, 71.8 seconds
- Automatically selected window for the original parity run: frames 0-164, 6.6 seconds
- Final shared-adapter render: frames 0-159, 160 rendered frames, after omitting low-confidence terminal candidates
- Automatic method: shared `ResearchBallTracker` with Hough-circle proposals, white compactness scoring, automatic logo-strip ROI exclusion, bounded continuity linking, multi-scale fallback, and temporal comparison
- Final net pixel displacement: 363.25 pixels
- Visual QA: final contact sheet inspected; marker remained on the visible ball and the terminal offset candidate was not rendered as tracked
- Renderer QA: frame/time labels are visible; the inset is cropped from pristine source pixels; terminal top-boundary clipping is explicitly labeled without moving the marker
- States: rendered observations are `observed`/`tracked`; interpolation and prediction were not used in this accepted run

Open the best artifact:

```bash
open out/research_training_gauntlet/mmu_candidate/analysis/automatic_ball_tracer_h264.mp4
open out/research_training_gauntlet/mmu_candidate/analysis/automatic_ball_tracer_contact.jpg
open out/research_training_gauntlet/mmu_candidate/analysis/automatic_ball_tracer_diagnostics.json
```

The H.264 artifact is verified as `h264`, `yuv420p`, 600x480, 25 FPS, 160 frames, and decodes fully with FFmpeg.

The research-only FairwayOS sidecar can be generated without entering the
production observation or analytics contracts:

```bash
python3 -m ghostcaddie fairwayos-ball-sidecar \
  --input out/research_training_gauntlet/mmu_candidate/analysis/automatic_ball_tracer_diagnostics.json \
  --out out/research_training_gauntlet/mmu_candidate/analysis/fairwayos_ball_research_sidecar.json \
  --source mmu_candidate/source.mp4
```

It preserves pixel-space candidates and human fallback while setting
`production_eligible: false`.

The multi-clip evaluation and provenance record is:

```bash
open out/research_training_gauntlet/multi_clip_evaluation.json
```

It records nine bounded candidate outcomes. The MMU result has 160 rendered frames after terminal low-confidence rejection, zero accepted gaps in the rendered run, and no ground-truth precision/recall. The shared-adapter probe record is `shared_adapter_evaluation.json`.

Local model hashes recorded there include:

- `yolo11n.pt`: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` — generic COCO; `sports ball` is not golf-ball evidence.
- `yolo11n-pose.pt`: `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0` — generic COCO pose; not clubhead or ball evidence.
- `out/golfdb_evaluation/swingnet_1800.pth.tar`: `6331e303a9e86d0c19f183899f958bf2a71cf5a7070d46899e25e1ac877b23d4` — GolfDB/SwingNet event model, research-only and not a ball detector.

## Parallel gauntlet review

The accepted MP4 is now generated through the shared `ResearchBallTracker` adapter using its circle-proposal path, bottom-scene exclusion, and continuity ranking. The generic component fallback remains dependency-light but is not used when circle proposals are available.

Model discovery found the MIT-licensed `rucv/golf_ball` implementation and a Hugging Face YOLOv8-nano golf-ball tracker lead, but their dataset/weight provenance and evaluation splits were not sufficiently clear for safe download or promotion. No external model or dataset was downloaded.

SwingNet is locally runnable for its eight GolfDB swing events plus background, but it does not localize a ball or clubhead. Clubhead and impact timing remain unavailable on MMU because the clubhead is blurred/overlapping at the available resolution and frame rate. The shared adapter was also run as bounded five-frame probes on `shk`, `q4k`, `impact`, and `bunker`. These are research measurements only: `shk` observed 0/5 frames; `q4k`, `impact`, and `bunker` observed 5/5. No ground-truth labels exist, so precision and recall remain null.

## Evidence boundary

This is an automatically aligned heuristic research demonstration, not a validated golf-ball detector. The source is public YouTube footage with unverified reuse rights. The encoded FPS is not proof of the camera capture FPS. No pseudo-labels or human coordinates were used for the automatic track.

The following remain unavailable and must remain null until rights-cleared paired annotations, validated models, calibration, and held-out evaluation exist:

- clubhead and validated impact/contact
- validated trajectory and landing
- calibration and course coordinates
- `ShotEvent`
- production analytics and recommendation

`run_pipeline()` and production analytics were not invoked.

## Verification

```bash
python3 -m unittest -q
python3 -m compileall -q ghostcaddie tests
python3 -m py_compile out/research_training_gauntlet/run_mmu_auto_demo.py
ffmpeg -v error -i out/research_training_gauntlet/mmu_candidate/analysis/automatic_ball_tracer_h264.mp4 -f null -
git diff --check
```

The current full suite passes 313 tests with one environment-dependent test skipped. Research media and generated artifacts remain local under ignored `out/` paths. Diagnostic source labels are relative to the research artifact root, so reruns do not serialize local filesystem paths. YouTube provenance metadata is strict JSON; research sidecars reject traversal, URL, and home-relative source identifiers, non-finite values, and promoted production flags.
