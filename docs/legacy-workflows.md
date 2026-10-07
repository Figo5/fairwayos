# Archived workflow guide

Earlier assisted and canonical workflows. For the current paused state, see [the README](../README.md) and [resume guide](resume-fairwayos.md). Historical numbers describe their original development collections.

# Fairway OS

Fairway OS is a local golf-video analysis and showcase project with
**canonical body-focused analysis**, **operator-assisted demos**, and source-bound
**saved tracking review and export** for selected clips. The internal Python package remains `ghostcaddie` for compatibility.

| | Canonical mode | Demo mode | Broadcast showcase |
|---|---|---|---|
| Command | `analyze-video` | `demo-video` | `broadcast-showcase` |
| Scope | the clip, no operator input | one operator-chosen interval | the whole shot, overlays only in declared windows |
| Processing | Local/offline | Local/offline, operator-assisted | Local/offline, operator-assisted |
| Body | Conditional golfer identification and 2D image-space pose/angles | Body analysis within an optional operator-selected region | Same body pipeline, per scene window |
| Ball / clubhead | Truthful `UNAVAILABLE` states | Experimental seeded/corrected image-space tracking | Same, per scene window, plus opt-in saved research observations; absent stays absent |
| Output | Annotated video, analysis, report, provenance, diagnostics | Polished annotated video, keyframe cards, report, reproducible config/provenance | Whole-shot video (**source cadence, source audio copied**), per-scene report, config, provenance |
| Physical metrics | Unavailable | Unavailable | Unavailable |

There is **no validated physical ball speed**, and no carry, spin, or launch-monitor
claim. Pixel displacement and playback time do not establish physical speed.
Body angles are 2D projections, not validated 3D biomechanics. Demo ball/club
tracking is experimental and operator-assisted; seeds, corrections, interpolation,
and reviewed anchors are disclosed rather than presented as automatic detections.

## Review and export saved tracking

The broadcast review UI can preview source-bound automatic research positions,
step through tracking gaps, and export them with the showcase presentation. Load a
matching config, then choose **Reviewed showcase (assisted)** or **Saved automatic
targets (research)**. Each export has its own immutable config snapshot and output
folder, with an inline video preview after verification.

```sh
.venv/bin/python -m ghostcaddie broadcast-showcase ui \
  --config-dir /absolute/path/to/configs --out-root /absolute/path/to/exports \
  --extra-root /absolute/path/to/clips \
  --tracking-predictions /absolute/path/to/results.predictions.json \
  --cv-python "$FAIRWAY_BODY_PYTHON" --body-model "$FAIRWAY_BODY_MODEL"
```

Saved mode uses the original saved seed and automatic observations; current seed
edits, corrections and assisted paths stay in the document but do not alter this
export. Missing, lost and interpolated frames have no automatic marker or trail.
It imports prior results without running a new tracking model. Body/phase inputs
remain operator-assisted; physical metrics remain unavailable. See the
[saved tracking guide](../docs/saved-tracking-user-guide.md).

An isolated [ball evidence research ablation](../docs/ball-evidence-research.md)
checks classic proposals against compact current-frame pixels from the original
seed. On the twelve inspected Tiger development clips it removes four labeled
false positives while retaining 108 of 109 previously correct ball samples.
It remains opt-in research; the canonical detector and tracking defaults are unchanged.

The opt-in [full-swing recovery policy](../docs/full-swing-research.md) adds
current-frame wrist and image-contrast checks to mask measurements. On the same
development collection, selected full-swing head samples improve from 21 to 23
correct out of 38, with no incorrect labeled position. Sparse scores alone missed
shaft/grip failures in rejected versions, so visual checks are also recorded.

The [proposal support experiment](../docs/proposal-supported-head-research.md) uses
the frozen learned detector to restrict current pixel/shaft measurements. It
adds one visually reviewed missing head point to the selected twelve clips.
Four newer shots remain body-only: their unseeded candidate confused two balls
with clubheads and still missed the fast swing. This is opt-in development
research; tracking defaults are unchanged.

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
[canonical operator guide](../docs/video-analysis-user-guide.md) for interval, sampling,
ROI and failure-state details. An explicitly supplied ROI is recorded as user input.

## Run demo mode

Copy [the demo config](../examples/demo_config.template.json) and replace the
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
[demo guide](../docs/demo-mode.md) documents tracking, presentation, cadence caveats,
manual anchors, holds and output contracts. The
[showcase template](../examples/showcase_config.template.json) contains illustrative
values, not measurements that can be reused across clips.

## Run the broadcast showcase

A whole shot, rendered end to end at the source's own cadence with the source's own
audio, where overlays and metrics exist **only** inside the scene windows declared in
the config. Outside every window the frame is the source picture; at a window boundary
the body, trails, cards, phase badge and `hand_rise` reference are all cleared, and no
scene inherits another's tracks. A putt is marked
`address / backstroke / forward_stroke / contact / finish` — never a full-swing TOP.

```sh
# cut/dissolve candidates for a clip (suggestions with confidence; nothing applied)
.venv/bin/python -m ghostcaddie broadcast-showcase scenes CLIP.mp4 --out cuts.json

# check a config against a clip without rendering anything
.venv/bin/python -m ghostcaddie broadcast-showcase validate CLIP.mp4 --config C.json

# render it
.venv/bin/python -m ghostcaddie broadcast-showcase run CLIP.mp4 --config C.json \
  --out showcases/my_shot --cv-python "$PWD/.venv/bin/python" \
  --body-model "$FAIRWAY_BODY_MODEL"

# the loopback browser review UI: drag the golfer box, mark phases, save, render
.venv/bin/python -m ghostcaddie broadcast-showcase ui --port 8899
```

The [broadcast guide](../docs/broadcast-showcase.md) documents the config, the four
non-negotiables, the UI and what the render verification proves.

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
[research decisions](../docs/research-decisions.md) and the
[original pipeline audit](../docs/canonical-video-pipeline-audit.md).
The older synthetic shot-analytics engine and fictional provider samples under
`data/` are separate from video analysis and are not official PGA data.

The [Masters compilation and motion review](../docs/masters-compilation-motion-review.md)
records the complete local 16-clip compilation, source-bound image-motion exports,
tempo/coverage limits and validation. Motion rates use source video time and native
pixels; physical club/ball speed remains unavailable without calibrated evidence.

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
accuracy. The [final status](../FINAL_STATUS.md) records closeout validation.

## Repository policy

Keep copyrighted PGA source footage, all generated demo/showcase media, downloaded
corpora, large models, caches, `out/` and `uv.lock` out of Git. The only committed
raster fixture is a tiny synthetic test image generated by a checked-in script.
Local evidence artifacts referenced by historical reports are not bundled; do not
mistake their paths for portable inputs. No physical launch-monitor validation or
new research result is implied by project completion.

[License](../LICENSE) · [Third-party notices](../THIRD_PARTY_NOTICES.md)
