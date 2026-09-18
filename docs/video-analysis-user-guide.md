# Video analysis: what works, how to run it, and what it refuses to do

This is the operator guide for the **canonical local video workflow**. It covers one
command and one output contract. Everything else under `ghostcaddie/video/` is
research code and is not part of this path.

The audit behind these boundaries is
[`canonical-video-pipeline-audit.md`](canonical-video-pipeline-audit.md), which is
preserved unchanged. The measured behaviour of this workflow on the local clip
corpus is in [`canonical-validation-sweep.md`](canonical-validation-sweep.md).

---

## 1. WHAT WORKS TODAY

| Capability | Status |
|---|---|
| Local video ingest, validation, SHA-256 source binding | works |
| Bounded, deterministic frame sampling | works |
| Golfer identification, locally and offline | works, **conditionally** — see §5 |
| 2D image-space body pose (MoveNet SinglePose Lightning) | works where the golfer is identified |
| Conservative 2D body-angle metrics with per-metric sufficiency checks | works |
| Clean annotated video | works |
| Report, provenance, diagnostics, machine-readable analysis | works |
| Upload UI wrapping the same backend | works |

## 2. WHAT DOES NOT WORK TODAY

| Capability | Status |
|---|---|
| Automatic ball tracking | **UNAVAILABLE** — every route is withdrawn or falsified |
| Clubhead tracking | **UNAVAILABLE** — blocked at the data layer, not the code layer |
| Ball speed, clubhead speed, launch angle, carry, apex | **UNAVAILABLE** — no validated calibration |
| Any 3D quantity | **UNAVAILABLE** — single uncalibrated camera |
| Swing phase, club plane, kinematic sequence | not implemented, not claimed |

These are not "coming soon" placeholders. They are states the workflow reports on
every run, with the reason attached. No flag turns them on.

## 3. WHAT IS EXPERIMENTAL

Research-only. Never reached from the canonical path, never a fallback:

- `fresh_ai` upload mode — network, external paid AI service, non-deterministic.
- `research_ball.py`, `research_ball_track.py`, `pga_sliced_ball.py` — falsified on
  the accepted artifact; the track followed the flagstick base.
- `ball_pipeline.py`, `ball_evaluation.py`, `ball_annotations.py`,
  `tools/ball_research.py`, `tools/ball_annotate.py` — the 2026-09-18 ball-tracking
  research pass. Classified **FAILED_TO_GENERALIZE**: it tracks one clip of ten
  correctly and abstains on the rest, and on the held-out split it locked onto a
  golfer's shirt for fifteen frames. Reachable only through `tools/ball_research.py`;
  the canonical ball layer is untouched. Evidence:
  [`ball-tracking-results.md`](ball-tracking-results.md).
- `region_tracker_mil.py`, `mil_overlay.py`, `clubhead_methods.py` — transfer FAILED
  with frozen settings; the baseline's extra frames are wrong frames.
- `BallTapirAdapter` (BootsTAPIR) — falsified for transfer under a hash-bound rerun.
- `luna_roi_experiment.py`, `clubhead_pseudo_labels.py`, `tracer_weak_supervision.py`
  — pseudo-labels by construction; never presented as observations.
- `OllamaPerceptionAdapter` — opt-in, `enabled=False`, loopback only.

Retained as evidence. Do not retune thresholds to revive any of them.

---

## 4. Setup

### Interpreter

The canonical workflow uses **two** interpreters on purpose:

- the **orchestrator** — any Python ≥ 3.11. It imports no cv2, no numpy and no model
  runtime, and opens no socket.
- the **body worker** — must have `ai_edge_litert`, `cv2` and `numpy`. In this repo
  that is `.venv/bin/python3` (Python 3.11.16, **no torch**).

Keeping them apart is what lets the whole output contract be tested on a bare
interpreter, and keeps torch and pickle out of the serving process.

Override the worker interpreter with `--body-python` or `$FAIRWAY_BODY_PYTHON`.

Run the suite with an interpreter that has cv2:

```bash
.venv-video-ai/bin/python3 -m pytest tests -q
```

On Python 3.9 roughly a hundred tests error at collection on `ModuleNotFoundError:
cv2`. That is an environment artifact, not a regression; `requires-python` is now
`>=3.11`.

### Model files

Nothing is ever downloaded. You place model files; the tool tells you where it
looked.

```bash
python3 -m ghostcaddie models
```

Resolution order, highest priority first:

1. an explicit CLI argument (`--body-model`)
2. the per-model environment variable (`$FAIRWAY_BODY_MODEL`, …)
3. `$FAIRWAY_MODEL_ROOT/<filename>`
4. `<repo>/models/<filename>`, then `~/.fairwayos/models/<filename>`
5. otherwise **unavailable**, stated as such

| Model | Expected filename | Env var | Needed for |
|---|---|---|---|
| MoveNet SinglePose Lightning v3 | `movenet-singlepose-lightning-v3.tflite` | `FAIRWAY_BODY_MODEL` | **canonical** body layer |
| SAM2.1 tiny | `sam2.1_hiera_tiny.pt` | `FAIRWAY_SAM2_MODEL` | research only |
| BootsTAPIR v2 | `bootstapir_checkpoint_v2.pt` | `FAIRWAY_TAPIR_MODEL` | research only |

`models/` is gitignored. Model files are never committed.

> An explicit path that does not exist is a **configuration error**, reported as
> such. It never falls through to some other file.

**`/tmp` is not a model directory.** The previous defaults pointed there; macOS
purged `/tmp` on 2026-09-16 and took the SAM2 and BootsTAPIR checkpoints with it.
The canonical path no longer depends on any `/tmp` location.

---

## 5. The canonical command

```bash
.venv/bin/python3 -m ghostcaddie analyze-video CLIP.mp4 --out OUTPUT_DIR
```

Useful options:

| Option | Meaning |
|---|---|
| `--sample-fps N` | frames analysed per second of source (default 6) |
| `--max-frames N` | hard bound on analysed frames (default 180) |
| `--start S` / `--end S` | restrict the processed interval, in seconds |
| `--roi X,Y,W,H` | operator-supplied golfer region — see below |
| `--body-model PATH` | explicit model file |
| `--body-python PATH` | explicit worker interpreter |
| `--expected-sha256 HEX` | fail closed unless the media bytes hash to this |
| `--no-render` | skip the annotated video |
| `--redact-path` | record the file name only, not the full path |
| `--json` | print `analysis.json` instead of the summary |

Guarantees:

- local input only; **no network calls, no external AI service, no seeded target
  fallback, no experimental tracker cascade**;
- the source is hashed and `bind_source()` runs before any model is touched;
- processing is bounded in frames and in wall time;
- the same five outputs every time.

### Outputs

```
OUTPUT_DIR/
  annotated.mp4      clean overlay; only supported layers are drawn
  report.md          the concise human-facing report
  analysis.json      THE SOURCE OF TRUTH (schema fairwayos-analysis.v1)
  provenance.json    methods, model paths, hashes, environment
  diagnostics.json   per-frame records, thresholds, rejection reasons
  debug/             identity overlay and other research art
```

If the annotated video cannot be encoded, that is reported — `annotated_video_written:
false` with a reason, and `NOT PRODUCED` in the report. It is never an empty success.

---

## 6. Body: conditional, never assumed

MoveNet is a **single-pose** model. Run on a whole wide frame it returns one skeleton
with no guarantee it belongs to the player — a spectator, a caddie or a camera
operator will do. The workflow therefore never runs whole-frame MoveNet and calls the
result the golfer. It establishes identity first:

1. **Propose.** Whole-frame MoveNet on every sampled frame. A candidate exists only
   if a full-torso pose is found on at least 60% of them; the median box is the
   subject.
2. **Challenge.** Blank the subject out of five probe frames and run MoveNet again.
   A second person that survives on at least three probes, away from the subject and
   consistent with itself, means the frame holds more than one body.
3. **Tie-break — only if challenged.** Compare camera-compensated motion density
   inside each body. A golfer swings; a bystander stands. Without a 3× margin the
   answer is AMBIGUOUS. If the camera is also moving hard, the tie-break is not
   trusted at all and the answer is AMBIGUOUS.
4. **Measure.** Re-run MoveNet on the padded subject crop, so the pose that reaches
   the metrics is not the whole-frame downscale.

Then, per frame, a pose is attributed to the golfer only if it has ≥ 8 confident
keypoints, a complete torso (both shoulders, both hips), covers ≥ 25% of the subject
region, and is centred inside it.

### Resulting layer state

| Attributed frames | Body state |
|---|---|
| ≥ 80% of sampled frames | `supported` |
| 40–80% | `partial` — metrics use only the attributed frames |
| < 40% | `ambiguous` — **no metrics at all** |
| no stable proposal, or no model | `unavailable` |
| the model or an operator ROI was invalid | `failed` |

**This is a verification gate built on a single-pose model, not a person detector.**
A body overlapping the subject, or a person too small or too occluded for MoveNet to
resolve, cannot be challenged this way and would not be found. That limitation is
printed on every run.

### Operator ROI

`--roi X,Y,W,H` skips steps 1–3. It is:

- clearly user-supplied, and labelled `identity_method: operator_roi`;
- **not** described as automatic perception;
- persisted in `provenance.json`;
- **not** a seed — it never initialises a tracker and never propagates.

---

## 7. Body metrics

Only conservative 2D image-space angles, each with its own landmark sufficiency
check: `shoulder_tilt`, `hip_tilt`, `left/right_knee_flexion`,
`left/right_elbow_angle`.

A metric is reported only when its own landmarks are confidently visible on at least
60% of the attributed frames **and** on at least 5 frames. Otherwise it is listed as
unavailable with the count. Each reports median, p10 and p90 across frames — a single
number would hide the spread.

Every supported metric carries this label:

> 2D image-space estimate from a single uncalibrated camera. It measures the angle
> between projected pixel coordinates, not an anatomical joint angle: it changes with
> camera position and it is not a clinical or coaching measurement.

No 3D is inferred. No medical or coaching certainty language is produced.

---

## 8. Ball, clubhead and physical metrics

`ball` and `clubhead` are `unavailable` on every run, with the reason attached. The
renderer draws nothing for them: no stale placeholders, no retained trails, no
"last known" points.

Physical metrics (`ball_speed`, `clubhead_speed`, `launch_angle`, `carry_distance`,
`apex_height`) are unavailable, each with its own reason — no validated physical
scale, no validated action-time calibration, or the ball layer being unavailable.

Nominal container FPS is **not** accepted as action time. A speed derived from it
would inherit any capture/container rate mismatch as a silent scale error.

---

## 9. Upload UI

```bash
.venv/bin/python3 -m ghostcaddie.upload.server --root /tmp/fairwayos-uploads
```

Binds `127.0.0.1` only. **Analyse (canonical)** posts `analysis=canonical`, which
calls `ghostcaddie.video.canonical.analyze_video` — the same function the CLI calls.
The service is a wrapper, not a second implementation.

The research modes (`fresh_ai`) are separately labelled. Canonical mode never falls
back to them; an unknown mode is refused rather than defaulted.

---

## 10. Validating a change

```bash
.venv/bin/python3 tools/canonical_validation.py CLIP_DIR --out out/canonical_validation
```

Runs every clip in the directory and writes `validation_report.md` +
`validation_summary.json`.

**This is drift detection, not a benchmark.** The local corpus has no ground truth,
no held-out split, and has already been consumed during development. Clips are never
excluded for being hard — dropping the ones that hurt a number is selection on the
metric being reported.
