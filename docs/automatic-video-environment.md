# Automatic Video Environment

Status: capability gate **blocked** as of 2026-08-27.

The standard GhostCaddie analytics environment remains unchanged. An isolated
video environment was created at `.venv-video/` with Python 3.9.6. No packages,
model weights, credentials, cookies, or network downloads were installed.

## Verified host

- Python: 3.9.6 (`/usr/bin/python3`)
- `venv`: available
- FFmpeg: 8.1.2
- ffprobe: 8.1.2
- CPU: 10 cores
- Memory: 16 GB
- GPU: Apple M2 Pro, Metal 3
- yt-dlp: unavailable
- OpenCV: unavailable
- PyTorch: unavailable
- torchvision: unavailable
- Ultralytics: unavailable
- MediaPipe: unavailable
- ONNX Runtime: unavailable
- Project model weights (`.pt`, `.onnx`, `.gguf`): none found

## Isolated installation plan

These commands are documentation only and were **not executed**. Run them only
with an approved, network-enabled environment and pinned versions/ hashes:

```bash
cd /Users/giofiore/ghostcaddie-tour
.venv-video/bin/python -m pip install --upgrade pip
.venv-video/bin/python -m pip install yt-dlp
.venv-video/bin/python -m pip install opencv-python-headless torch torchvision ultralytics
```

The downloader must then be passed explicitly, without credentials or cookies:

```bash
.venv-video/bin/yt-dlp --version
python3 -m ghostcaddie youtube-analyze \
  --yt-dlp /Users/giofiore/ghostcaddie-tour/.venv-video/bin/yt-dlp \
  ...
```

Model weights are a separate controlled input. They must be obtained from an
approved upstream source, pinned by exact version and checksum, stored outside
the analytics core, and registered in diagnostics before enabling automatic
analysis. No weight is currently approved or installed.

## Capability gate

Automatic analysis must remain unavailable until all of the following are
verified on consenting real golf footage:

1. The detector runtime imports successfully in `.venv-video`.
2. The configured model weights are present and checksum-verified.
3. Actual frame inference produces validated golfer observations.
4. Golfer anchor, ball/club visibility, contact timing, and camera stability are
   measured and recorded.
5. Missing observations remain explicitly unavailable.
6. Calibration is supplied and validated.
7. Existing `run_pipeline()` runs exactly once through the established adapter.
8. Annotated frames/video are independently inspected.
9. The held-out real-footage evaluation passes the declared release threshold.

Until then, `youtube-analyze` must either fail with its diagnostic hard gate or
require the explicit `--fallback-human` workflow. The synthetic HITL video is
suitable for plumbing tests only and is not evidence of automatic perception
reliability.
