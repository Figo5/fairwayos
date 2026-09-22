# Fairway OS

Fairway OS is a finished local golf-video analysis project with two modes:
**canonical body-focused analysis** and an **operator-assisted demo** for selected
clips. The internal Python package remains `ghostcaddie` for compatibility.

| | Canonical mode | Demo mode |
|---|---|---|
| Command | `analyze-video` | `demo-video` |
| Processing | Local/offline | Local/offline, operator-assisted |
| Body | Conditional golfer identification and 2D image-space pose/angles | Body analysis within an optional operator-selected region |
| Ball / clubhead | Truthful `UNAVAILABLE` states | Experimental seeded/corrected image-space tracking |
| Output | Annotated video, analysis, report, provenance, diagnostics | Polished annotated video, keyframe cards, report, reproducible config/provenance |
| Physical metrics | Unavailable | Unavailable |

There is **no validated physical ball speed**, and no carry, spin, or launch-monitor
claim. Pixel displacement and playback time do not establish physical speed.
Body angles are 2D projections, not validated 3D biomechanics. Demo ball/club
tracking is experimental and operator-assisted; seeds, corrections, interpolation,
and reviewed anchors are disclosed rather than presented as automatic detections.

## Setup

Use Python 3.11 and install `ffmpeg` / `ffprobe` on your PATH. From the repository:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The requirements provide the tested local OpenCV, NumPy and LiteRT runtime. Installation may use the network; neither finished analysis mode downloads
models or calls remote inference services. For the optional graphical picker,
install `opencv-python==5.0.0.93` **instead of** `opencv-python-headless` in a separate
GUI environment (do not install both OpenCV distributions together). Editing JSON
requires no GUI.

Supply your own local MoveNet SinglePose Lightning v3 TFLite file:

```sh
export FAIRWAY_BODY_MODEL=/absolute/path/to/movenet-singlepose-lightning-v3.tflite
export FAIRWAY_BODY_PYTHON="$PWD/.venv/bin/python"
.venv/bin/python -m ghostcaddie models
```

Weights and source footage are not distributed. Missing model/runtime evidence is
reported as unavailable; it must not be read as successful body detection.

## Run canonical mode

```sh
.venv/bin/python -m ghostcaddie analyze-video /absolute/path/to/clip.mp4 \
  --out out/canonical --body-python "$FAIRWAY_BODY_PYTHON"
```

Outputs: `annotated.mp4`, `report.md`, `analysis.json`, `provenance.json`, and
`diagnostics.json`. Source hashing, bounded processing, golfer identity checks and
per-metric sufficiency checks precede any supported claim. Ambiguous subjects,
occlusion and insufficient observations can yield unavailable body results.
Ball, clubhead and physical metrics remain unavailable. See the
[canonical operator guide](docs/video-analysis-user-guide.md) for interval, sampling,
ROI and failure-state details. An explicitly supplied ROI is recorded as user input.

## Run demo mode

Copy [the demo config](examples/demo_config.template.json) and replace the
illustrative interval, region, seeds, phase marks and corrections with inputs for
your selected clip. Coordinates are native source pixels; frame indices refer to
the source. Alternatively, use `demo-setup CLIP.mp4 --out demo_config.json` in the
optional GUI environment.

```sh
.venv/bin/python -m ghostcaddie demo-video /absolute/path/to/clip.mp4 \
  --config demo_config.json --out out/demo --cv-python "$PWD/.venv/bin/python" \
  --body-model "$FAIRWAY_BODY_MODEL"

# Polished showcase presentation, with the same assistance disclosures:
.venv/bin/python -m ghostcaddie demo-video /absolute/path/to/clip.mp4 \
  --config demo_config.json --out out/showcase --showcase \
  --cv-python "$PWD/.venv/bin/python" --body-model "$FAIRWAY_BODY_MODEL"
```

The standard set uses `demo_*` filenames; showcase uses `showcase_*`. Both retain
source/config hashes and resolved operator inputs in provenance. Keep the source,
config and model versions together locally for reproducibility. The
[demo guide](docs/demo-mode.md) documents tracking, presentation, cadence caveats,
manual anchors, holds and output contracts. The
[showcase template](examples/showcase_config.template.json) contains illustrative
values, not measurements that can be reused across clips.

## Architecture and boundaries

- `ghostcaddie/cli.py` dispatches the two finished modes.
- `ghostcaddie/video/canonical.py` binds the source, runs the bounded body worker,
  aggregates supported 2D metrics and writes the canonical contract.
- `ghostcaddie/video/canonical_worker.py` performs local MoveNet inference,
  golfer identity checks and rendering in a separate interpreter.
- `ghostcaddie/demo/` validates operator config, tracks assisted targets, computes
  image-space analytics, and renders standard or showcase artifacts.
- `ghostcaddie/upload/` retains a local UI wrapper and tested historical modes.
  Legacy commands, provider fixtures and research tools remain for compatibility;
  they are not alternate entry points for the finished video product. Some opt-in
  historical modes can use network services; the offline guarantee above applies
  to `analyze-video` and `demo-video`.

Research code remains outside the canonical tracking path. Failed experiments are
retained where tests or decision evidence depend on them. They are not automatic
fallbacks and do not expand the capability boundary. See
[research decisions](docs/research-decisions.md) and the
[original pipeline audit](docs/canonical-video-pipeline-audit.md).
The older synthetic shot-analytics engine and fictional provider samples under
`data/` are separate from video analysis and are not official PGA data.

## Testing

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
ffmpeg -v error -i out/canonical/annotated.mp4 -f null -
ffmpeg -v error -i out/demo/demo_annotated.mp4 -f null -
```

Development requirements also install PyTorch for retained ranker tests; it is not
a serving dependency. Tests cover source binding, unavailable states, canonical/research isolation,
assisted provenance, tracking loss, rendering, CLI contracts and historical
regressions. Optional research dependencies and external corpora can cause explicit
skips. Synthetic CLI smoke tests check execution and artifact contracts, not tracking
accuracy. The [final status](FINAL_STATUS.md) records closeout validation.

## Repository policy

Keep copyrighted PGA source footage, all generated demo/showcase media, downloaded
corpora, large models, caches, `out/` and `uv.lock` out of Git. The only committed
raster fixture is a tiny synthetic test image generated by a checked-in script.
Local evidence artifacts referenced by historical reports are not bundled; do not
mistake their paths for portable inputs. No physical launch-monitor validation or
new research result is implied by project completion.

[License](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)
