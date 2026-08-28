# Automatic Perception Capability Smoke Test

Date: 2026-08-27

This is a technical capability smoke test, not an acceptance evaluation.

## Environments

Existing environments were preserved:

- `.venv-video/`: Python 3.9.6, yt-dlp 2025.10.14
- `.venv-video-modern/`: Python 3.11.16, yt-dlp 2026.08.19, yt-dlp[default], EJS

Isolated AI environment:

- `.venv-video-ai/`
- Python 3.11.16
- OpenCV 5.0.0
- PyTorch 2.13.0
- torchvision 0.28.0
- Ultralytics 8.4.131
- MPS built and available on Apple M2 Pro
- Ultralytics installed `lap` for tracking inside this environment only

Official weights loaded:

- `yolo11n.pt`
- `yolo11n-pose.pt`

## Clip

Source provenance is recorded only as platform/video ID:

- platform: YouTube
- video ID: `PsJvcITOVRc`
- bounded segment: first 20 seconds
- format: YouTube format 135, 854x480 H.264 video
- duration: 20.053 seconds
- FPS: 29.97

Artifact:

```text
out/youtube_smoke_480p/source.mp4
```

Prepared frames and contact sheets are under:

```text
out/youtube_smoke_480p/prepared/
```

## Measured inference

Generic YOLO detection:

- model: `yolo11n.pt`
- device: MPS
- frames: 100 at 5 FPS
- elapsed: 7.789 seconds
- mean latency: 75.6 ms/frame
- person detections: 89/100 frames
- sports-ball detections: 1/100 frames
- unique person IDs in the sequence pass: 24

Generic pose:

- model: `yolo11n-pose.pt`
- device: MPS
- frames: 100
- elapsed: 3.630 seconds
- mean latency: 33.9 ms/frame
- frames with pose: 72/100

Tracking and overlay reports:

- `out/youtube_smoke_480p/detector_smoke_report.json`
- `out/youtube_smoke_480p/tracking_smoke_report.json`
- `out/youtube_smoke_480p/pose_smoke_report.json`
- `out/youtube_smoke_480p/detector_contact/contact_sheet.jpg`
- `out/youtube_smoke_480p/pose_contact/contact_sheet.jpg`

## Interpretation and gate

Visual inspection shows one apparent golfer and a stable-looking sampled angle,
but this is a broadcast highlights clip and is not a fixed-camera acceptance
clip. The generic detector does not provide reliable golf-ball coverage, and
there is no clubhead/contact estimator. Track identity continuity is unstable.

Therefore the automatic-perception capability gate remains **closed**:

- no automatic `ShotEvent` was created;
- no calibration was applied;
- `run_pipeline()` was not invoked;
- no recommendation was generated;
- no landing, trajectory, club, or contact values were fabricated;
- human annotation fallback remains the only supported path for this clip.
