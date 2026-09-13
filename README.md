# Fairway

Local golf video → AI body/ball/clubhead overlays and an analytics text report.
Research prototype: short development intervals reviewed, not reliable whole-video tracking or calibrated launch-monitor measurements.

## Setup (tested macOS ARM64, Python 3.11)

```sh
uv venv --python 3.11 .venv-tracker
uv pip install --python .venv-tracker/bin/python -r requirements-dev.txt
```

This installs independent pinned direct dependencies, not imports from the archived project's environments. Cached installs can add `--offline`. Requires `ffmpeg`, `ffprobe`, authenticated `hermes` and `codex` CLIs with GPT-5.5 access. Local video input does **not** mean offline inference: frame images are sent to the configured AI provider, and inference consumes account quota.

MoveNet TFLite weights must be available locally. Pass `--body-model /absolute/path/to/movenet.tflite` or set `FAIRWAY_BODY_MODEL`. The CLI downloads nothing. Weights and footage are not distributed with source.

## Run

```sh
.venv-tracker/bin/python fairway.py VIDEO --body-model /absolute/path/to/movenet.tflite --out-dir OUTPUT --start-frame 240 --frames 48 --workers 2
```

Choose an interval within your own input; the example frame numbers are not detections or seed coordinates. Processing is per-frame AI inference, not real-time. Output: `annotated.mp4`, `analytics.txt`; `vision_raw.json` and decoded frames are retained for internal review, not required user-facing coordinate exports.

The report includes supported 2D body angles, not definitive 3D biomechanics. Ball speed, clubhead speed and carry distance are currently unavailable; no calibrated physical measurement implementation has been accepted. Missing markers can mean uncertain or failed recognition, not necessarily an offscreen object.

## Verify

```sh
.venv-tracker/bin/python -m pytest -q
ffmpeg -v error -i OUTPUT/annotated.mp4 -f null -
```

Tests and successful encoding do not prove object identity. Review actual overlays against source pixels. No website or upload server.

Generate native-resolution clean/encoded comparison sheets (use a new output directory):

```sh
.venv-tracker/bin/python tools/compare_video.py VIDEO OUTPUT/annotated.mp4 --start-frame 240 --out-dir REVIEW
```

This tool compares decoded pixels without reconstructing prediction markers. The start frame is caller-declared; frame-count agreement alone does not verify source alignment.

## Approximate analytics development

See [the measurement assumptions](docs/approximate-analytics.md) and [the Furyk local-scale experiment](docs/furyk-scale-experiment.md). Add a separate analytics panel to the paired outputs from one tracker run:

```sh
.venv-tracker/bin/python approx_analytics.py --input OUTPUT/annotated.mp4 --vision OUTPUT/vision_raw.json --output OUTPUT/with-analytics.mp4
```

Currently supports 1280x720 input at 30000/1001 FPS. Supply video and JSON from the same run: count/FPS checks do not independently establish their identity. By default it reports pixel displacement and pixel speed per display second. Optional `--m-per-px` (or `--ball-diameter-px`, assuming 42.67 mm) plus `--action-time-scale` enable conditional projected m/s. The time multiplier is capture cadence divided by playback cadence for simple slow motion; it is not inferred. These inputs are assumptions, not validated calibration. Pixel motion includes camera motion and detection jitter. Physical estimates require explicit scale and action-time assumptions; a broadcast's playback FPS is not necessarily its capture rate.
