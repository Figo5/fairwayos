# Research-training gauntlet status

Date: 2026-08-31

Status: **research-only / production blocked**.

## Batch acquisition

A bounded batch of three public-platform golf clips was attempted with the existing `yt-dlp` adapter, Node/EJS runtime, no playlist expansion, and 20-second segments. Two clips downloaded locally and one was blocked by the configured estimated-size limit. Source URLs and hashes are retained only in the ignored local manifest:

```text
out/research_training_gauntlet/manifest.json
```

Downloaded media, extracted frames, annotations, reports, and model outputs remain under the ignored local directory `out/research_training_gauntlet/`.

The YouTube ingestion boundary rejects provider metadata with boolean, non-finite, or non-positive duration and rejects a returned video ID that differs from the requested ID before invoking the download subprocess. Estimated `filesize` metadata now has precedence over `filesize_approx`; when present it must be a finite, non-negative integer and is rejected as malformed metadata otherwise. Missing duration or size remains an explicit provider limitation rather than an invented value.

The automatic YouTube boundary preserves distinct blocked categories for `duration_limit` and `segment_limit`; these are not collapsed into generic video-unavailable failures.

Contact/landing candidate diagnostics accept ball points only when provenance is explicitly `observed`, `native`, or `user_confirmed`; missing, inferred, automatic, and unknown provenance remain unavailable and cannot be promoted.

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

`run_pipeline()` was not invoked. Core analytics, calibration, wind, dispersion, hazards, session behavior, CLI commands, YouTube ingestion, CI, and human fallback were not changed.

## Verification

- full unittest suite: 367 tests passed, 6 skipped;
- `compileall`: passed;
- `git diff --check`: passed;
- accepted Pexels 6573485 MP4: H.264, 1920x1080, 15 FPS, 121 frames, `yuv420p`;
- FFmpeg decode and first-frame extraction: passed;
- visual QA: person/pose/anchor overlays and honest unavailable/research-only labels visible;
- artifact assertions: passed.

The next genuine milestone requires legally usable paired annotations and a documented golf-ball/clubhead checkpoint or a consented FairwayOS-owned dataset. Until then, the project remains technically ready but data/model blocked.
