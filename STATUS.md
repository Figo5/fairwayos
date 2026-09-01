# FairwayOS Research Status

Updated 2026-08-31.

## Verified accepted artifact

The current accepted continuous demo is the local Pexels clip. It is a
research-only visual artifact, not a validated golf-perception result:

- Source: Pexels video `6573485`, https://www.pexels.com/video/boy-hitting-a-golf-ball-6573485/
- Local source: `out/research_training_gauntlet/pexels_6573485/source.mp4`
- Source SHA-256: `a6e48474045365d1de2d4af76f65da558531684d67da87172cdd15a6dc45e1d6`
- Accepted output: `out/research_training_gauntlet/fairwayos_unified_pexels_6573485/`
- Media: source 1920x1080 H.264/yuv420p at 30 FPS with 310 frames; encoded demo is 1920x1080 H.264/yuv420p at 15 FPS with 121 frames
- Observations: 121/121 pose and ball observations; ball states include 117 observed and 4 predicted states
- Provenance: `research_only: true`, `production_eligible: false`, `ground_truth: false`

Open the accepted artifact and its QA contact sheet:

```bash
open out/research_training_gauntlet/fairwayos_unified_pexels_6573485/annotated_video.mp4
open out/research_training_gauntlet/fairwayos_unified_pexels_6573485/contact_sheet_current.jpg
open out/research_training_gauntlet/fairwayos_unified_pexels_6573485/provenance.json
open out/research_training_gauntlet/fairwayos_unified_pexels_6573485/diagnostics.json
```

Native-resolution frame inspection found consistently rendered pose, golfer
box, feet anchor, ball marker, tracer, and zoom inset. The renderer now creates
an explicit independent clean-frame copy before pose and ball inference, then
composes overlays onto a separate output copy. The marker remains a
research candidate: no ground-truth labels establish golf-ball identity,
reacquisition, precision, recall, or false-positive rate. SwingNet event
hypotheses are also research-only; exact impact, landing, calibration,
course-space trajectory, `ShotEvent`, analytics, and recommendations remain
unavailable. Do not use Pexels `6573474` for this artifact.

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

This is an automatically aligned heuristic research demonstration, not a validated golf-ball detector. The accepted source is Pexels video `6573485`, marked free to use on its source page and retained locally only; this does not establish ground truth or production eligibility. The encoded FPS is not proof of the camera capture FPS. No pseudo-labels or human coordinates were used for the automatic track.

The following remain unavailable and must remain null until rights-cleared paired annotations, validated models, calibration, and held-out evaluation exist:

- clubhead and validated impact/contact
- validated trajectory and landing
- calibration and course coordinates
- `ShotEvent`
- production analytics and recommendation

`run_pipeline()` and production analytics were not invoked.

## Verification

```bash
python3 -m unittest discover -s tests -q
python3 -m compileall -q ghostcaddie tests
python3 -m py_compile out/research_training_gauntlet/run_mmu_auto_demo.py
ffmpeg -v error -i out/research_training_gauntlet/mmu_candidate/analysis/automatic_ball_tracer_h264.mp4 -f null -
git diff --check
```

The current full suite passes 372 tests with 6 optional-dependency tests skipped. Research media and generated artifacts remain local under ignored `out/` paths. Diagnostic source labels are relative to the research artifact root, so reruns do not serialize local filesystem paths. YouTube provenance metadata is strict JSON; research sidecars reject traversal, URL, and home-relative source identifiers, boolean or non-finite numeric values, and promoted production flags. YouTube duration metadata is rejected when boolean, non-finite, non-positive, or inconsistent with the requested video ID; filesize metadata is rejected when malformed, non-finite, negative, fractional, or boolean, and an explicit zero `filesize` is not replaced by `filesize_approx`.
