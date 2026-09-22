# FairwayOS

FairwayOS is a local golf video product workbench: two strictly separate modes over
one codebase — a **strict canonical analysis** pipeline and an **operator-assisted
Demo Mode** that turns one chosen clip into a presentable artifact set. Nothing is
uploaded, no remote inference is required, and the two answer different questions.

**Strict canonical mode (`analyze-video`).** What can this system find *on its own*?
Body pose only, and only where the golfer can be identified; `ball`, `clubhead` and
every physical metric are reported `UNAVAILABLE` on every run, with the reason
attached. The source is hashed and bound before any model loads, and the pipeline
declines rather than guessing.

**Demo Mode (`demo-video`).** Can an operator *show* one swing clearly? An operator
draws a region, marks the ball, club and phases; the demo tracks those seeded points,
measures 2D image-space body quantities inside that region, and renders an annotated
video, keyframe cards and a report. Seeded and corrected points are never called
detections, and the mode still produces no physical metric.

```bash
# Demo Mode: operator-assisted, selected clip (local only)
.venv/bin/python3 -m ghostcaddie demo-setup CLIP.mp4 --out demo_config.json  # optional picker
.venv/bin/python3 -m ghostcaddie demo-video CLIP.mp4 --config demo_config.json --out demo_out/

# Strict canonical mode: what the system can find on its own
.venv/bin/python3 -m ghostcaddie analyze-video CLIP.mp4 --out OUTPUT_DIR
```

| | Canonical (`analyze-video`) | Demo (`demo-video`) |
|---|---|---|
| Question | what can the system detect **on its own**? | can an operator show one chosen swing clearly? |
| Body | conditional 2D pose; `ambiguous`/`unavailable` when the golfer cannot be identified | same body pipeline; an operator ROI may say *which* golfer |
| Ball / clubhead | `UNAVAILABLE` — every route is withdrawn or falsified | operator-seeded, operator-corrected 2D image-space tracks, labelled as such everywhere |
| Physical metrics | `UNAVAILABLE` | still `UNAVAILABLE` — pixel/image-space only |
| Seeds, corrections, interpolation | never | expected, bounded, and recorded in `demo_provenance.json` |
| Failure mode | declines | declines *and* says what the operator would have to supply |

Demo Mode is a portfolio and presentation tool, not a perception claim. Interpolated
points are drawn dashed and counted separately; every rendered frame says
**OPERATOR-ASSISTED / EXPERIMENTAL / 2D IMAGE-SPACE**. The v3 ball research stays closed
here: Demo Mode does not train, benchmark, evaluate a held-out split or promote
anything.

Guides: [`docs/demo-mode.md`](docs/demo-mode.md) (Demo Mode) and
[`docs/video-analysis-user-guide.md`](docs/video-analysis-user-guide.md) (canonical).
Template config: [`examples/demo_config.template.json`](examples/demo_config.template.json).

## Project boundary and data disclaimer

FairwayOS is an analytics-engine prototype, not a consumer app and not a replacement
for ShotLink / TrackMan / TOURCAST. The **shot-analytics fixtures and provider samples
are synthetic/fictional** — no real player names, no real tournament data, no scraped
Tour data. That synthetic-data statement covers the analytic fixtures
(`data/sample_*`, `data/providers/*`, `data/scenarios/*`) only. The **video** side is
different and is stated separately: the clips used for local video work are **local
stock clips, kept out of Git and referenced by source SHA-256**; no media is committed
to this repository.

**Data disclaimer:** FairwayOS output is not sourced from ShotLink, TrackMan, TOURCAST,
or any official PGA TOUR system, and is not for competitive or broadcast use.

**Compatibility:** FairwayOS is the public project name. The internal
`ghostcaddie` Python package, imports, CLI commands, and schema identifiers are
preserved for compatibility. See [`docs/fairwayos-compatibility.md`](docs/fairwayos-compatibility.md).

## Canonical local video analysis

One command, one output contract, entirely on this machine:

```bash
.venv/bin/python3 -m ghostcaddie analyze-video CLIP.mp4 --out OUTPUT_DIR
```

```
OUTPUT_DIR/
  annotated.mp4   report.md   analysis.json   provenance.json   diagnostics.json
```

No network calls, no remote inference, no seeded fallback, no experimental
tracker cascade. The source is hashed and bound before any model is loaded, and
processing is bounded in frames and wall time.

**What it reports**

| Layer | Today |
|---|---|
| **body** | 2D image-space pose, **only where the golfer can be identified** — otherwise `ambiguous` or `unavailable` |
| **ball** | `UNAVAILABLE` — every route is withdrawn or falsified |
| **clubhead** | `UNAVAILABLE` — blocked at the data layer, not the code layer |
| **physical metrics** (ball speed, clubhead speed, launch, carry, apex, spin) | `UNAVAILABLE` — no validated calibration; each is listed explicitly in the output in its own unit with its own reason |

MoveNet is a single-pose model, so a whole-frame result is not evidence about *who*
was detected. The workflow proposes a body, challenges it by re-running with that
body blanked out, and declines rather than guess. An operator may supply `--roi`;
that is recorded as user input, never as automatic perception.

Model files are never downloaded. `python3 -m ghostcaddie models` prints where they
are looked for and which are present.

Full operator guide: [`docs/video-analysis-user-guide.md`](docs/video-analysis-user-guide.md).
Evidence: [`docs/canonical-video-pipeline-audit.md`](docs/canonical-video-pipeline-audit.md).
Measured behaviour on the local corpus:
[`docs/canonical-validation-sweep.md`](docs/canonical-validation-sweep.md).

## Legacy shot analytics (engine vertical slice)

The original FairwayOS slice: raw shot data → stable top-down coordinate system →
dispersion modeling → expected-strokes calculation → prescriptive "what should have
happened" output → annotated SVG overlay. It is still the whole shot-analytics engine,
and it is unrelated to the video modes above.

```bash
python3 -m ghostcaddie run --shot data/sample_shot.json --course data/sample_hole.json --player data/sample_player.json --out out/
```

Writes `out/recommendation.json` and `out/overlay.svg`, and prints a terminal summary.

## Ball-tracking research (2026-09-18)

**Automatic ball tracking: `UNAVAILABLE` in canonical mode.** The 2026-09-18 research
pass did not change that, and no configuration turns it on.

A bounded research pass over 10 rights-cleared Pexels clips produced the first ball
ground truth in this repository:

| | |
|---|---|
| clips | 10 (720p–4K, 24–60 fps) |
| human-reviewed frames | 325 — 189 ball-visible, 136 ball-absent, 19 ambiguous |
| split | frozen and group-disjoint (`data/ball_split.json`) |
| labels | source-hash-bound (`data/ball_annotations/*.json`) |

**Result: `FAILED_TO_GENERALIZE`.** Held-out recall 0.000. The held-out false-positive
rate on ball-absent frames was 0.333 against a budget of ≤0.05 written down before any
measurement, and the one held-out failure was a fifteen-frame lock onto the golfer's
shirt. Body and clothing distractors remain unsolved, and on held-out the simpler
baseline — this pipeline's own detector with the temporal stages deleted — beat the
full method. Ball tracking was **not** promoted to the canonical path.

This corpus is development-consumed. It is **not a benchmark**, and no untouched
held-out portion remains, so it cannot support further threshold tuning or any claim
of improvement on itself. The next step in this line needs new rights-cleared data,
not a retune of these ten clips.

The annotations and the split **are** committed. The underlying video is **not**:
those clips are rights-cleared local media that stays out of Git, so this repository
alone does not reproduce the evaluation. Re-running it requires the exact media bytes
matching the recorded SHA-256 values.

Evidence, per-clip metrics and reproduction commands:
[`docs/ball-tracking-results.md`](docs/ball-tracking-results.md). Acceptance bar,
written before implementation:
[`docs/ball-tracking-research-plan.md`](docs/ball-tracking-research-plan.md).

### Dataset v2 (2026-09-18)

A second, larger corpus now exists for the next attempt: **56 rights-cleared
stock clips** (Pexels License) across 26 shoot families, 720p to 4K, 23.98 to 60
fps, with a provenance manifest, `data/ball_dataset_v2/sources.json`, and a
frozen group-disjoint split, `data/ball_dataset_v2/split.json` — 34 train, 11
validation, 11 final test, with no shoot family in two splits and no family
shared with the consumed v1 corpus in the final test.

The labels are **not** done. This corpus ships with **0 reviewed frames** and
machine candidates recorded as pending *proposals* instead. That is deliberate:
the pass that built the corpus declined to promote its own machine reads to
labels after a control experiment showed ball-scale review was unreliable in both
directions (documented in
[`docs/ball-annotation-guide.md`](docs/ball-annotation-guide.md#5-verify-every-label-by-rendering-it-back-onto-its-own-pixels)).
Labelling is the blocking task, and the tool for it is
`tools/ball_annotate_web.py`. **Final test has not been evaluated: NO — and, after
the v3 pass below, it stays unevaluated by decision, not by omission.**

Media is not committed and the corpus is not a benchmark until it has labels and
an untouched final test. Canonical ball detection remains `UNAVAILABLE`.

Tooling and statistics: [`docs/ball-dataset-v2-status.md`](docs/ball-dataset-v2-status.md).

## Ball-tracker v3 — closed, not promoted (2026-09-21)

**Status: `FAILED_TO_GENERALIZE`, on development evidence. No promotion. No
canonical ball integration. The v3 final test remains SEALED / NOT RUN.**

The v3 architecture (blob proposals → context features → listwise tracklet ranker)
recovered **0 of 185** human-labelled ball-visible frames on `legacy` in all six
validation variants, including the full model. The only real-labelled corpus in the
repository is development-consumed, so that is a statement about the development
evidence and **not a held-out accuracy claim** — no clean held-out accuracy figure
exists for v3 and none is claimed.

Why the sealed final test was not spent: the tracker never cleared review
readiness, so evaluating it on the untouched 10-clip final test would consume
irreplaceable evidence to confirm an already-negative result. It stays sealed.
The archived research snapshot is **archive provenance, not a promotion freeze**.

Canonical states are unchanged and are enforced by tests: `ball` is `UNAVAILABLE`
with `method: none`, `CLUBHEAD` is `UNAVAILABLE`, and **every physical metric —
ball speed, clubhead speed, launch, carry, apex and spin — is `UNAVAILABLE`**
(no calibration, no validated action time). No further ball research loop is
planned on this corpus.

Authoritative evidence: [`docs/ball-tracker-v3-results.md`](docs/ball-tracker-v3-results.md)
(matrix, corrected AUC, timing provenance, seal audit, correction ledger).
Independent review provenance: `reviewer-1/2/3-report.md` in the orchestration
evidence root listed in [`docs/fairway-os-completion.md`](docs/fairway-os-completion.md).
Superseded worker narrative: [`docs/ball-tracker-v3-worker-pass-1.md`](docs/ball-tracker-v3-worker-pass-1.md).

## Test

The analytics core is standard library only, but the full suite needs `cv2`. Use an
interpreter that has it:

```bash
.venv-video-ai/bin/python3 -m pytest tests -q
```

`requires-python` is `>=3.11`. On 3.9 roughly a hundred tests error at collection on
`ModuleNotFoundError: cv2`; that is an environment artifact, not a regression.

Optional research checkpoints (SAM2, BootsTAPIR) are **not** required. Their absence
is an expected, reported state (`availability: model_absent`), never a test failure.

## Architecture

Two layers, strictly separated:

- **Engine** (`ghostcaddie/geometry.py`, `hazards.py`, `dispersion.py`,
  `expected_strokes.py`, `simulation.py`, `decision.py`, `explanation.py`):
  coordinate math, hazard classification, Monte Carlo dispersion sampling,
  expected strokes, decision logic, and plain-text explanation. Pure math —
  it never imports or depends on rendering code, and never exposes raw
  per-sample Monte Carlo data to the renderer.
- **Renderer** (`ghostcaddie/overlay.py`): the only module that draws. It
  consumes finished `CandidateResult` / `Recommendation` / `CourseModel` /
  `ShotEvent` objects and emits an SVG string with zero image libraries. It
  does not import the engine modules that produced the numbers.

`pipeline.py` wires them in strict order: load → simulate → decide → explain
& render (rendering is always called after the decision is built).

**Adapters:** `ghostcaddie/adapters/base.py` defines three structural
`typing.Protocol`s (`ShotDataSource`, `CourseDataSource`,
`PlayerProfileSource`). `adapters/json_file.py` is the concrete JSON-file
implementation. The composition root (CLI) loads the course first and passes
its declared `coordinate_system` into the shot adapter, so "one declared
coordinate system, reused by ingestion" is real and testable. A future
ShotLink/TrackMan/TOURCAST integration only needs a new adapter implementing
the same protocols — the engine never knows where data came from.

The current synthetic provider slice adds `adapters/shotlink.py` and
`adapters/trackman.py`. Both require an explicit provider and schema version,
strictly validate required fields, record unknown fields in provenance (or
reject them with `strict=True`), and emit the existing `ShotEvent` through a
`load_shot()` source. ShotLink converts GPS degrees in an explicit `+x east,
+y north` local frame to engine yards. TrackMan reconstructs landing from
carry yards and signed side-offset yards (`+` right of the aim line) using
separate course context. A supplied `CoordinateMapper` is called once per
normalized position; course and player sources remain separate pipeline
inputs. Fixtures are synthetic only: `data/providers/shotlink.json` and
`data/providers/trackman.json`.

## Validation Scenarios

Four additive end-to-end scenarios under `data/scenarios/`, each with its own
course/player/shot fixtures and a `tests/test_scenario_*.py` covering a
specific engine behavior. All four run through the full pipeline (`run_pipeline`).

- **`layup_vs_attack/`** — Blackwater Links, Hole 11 (Par 5): a water hazard
  guards the green frontage. The player's aggressive Hybrid line (distance to
  pin 225, carry 200) flies into the water, while 7i/PW lay up well short of
  it. Validates that hazard classification flags the aggressive line as
  water-risky and that the decision layer prefers the safe layup over the
  attacking line.
- **`ob_risk/`** — Ridgeline National, Hole 4 (Par 4): an OB strip hugs the
  right of the fairway and the player's Driver carries a rightward miss bias
  toward it, while a tighter 2i is OB-safe. Validates that the OB
  stroke-and-distance penalty (replaying from the shot's *original*
  tee-to-pin distance, `ob_penalty_strokes + strokes_from_lie(FAIRWAY, original)`)
  manifests correctly in a full pipeline run — not just in the isolated unit
  test — and that the OB-risky line costs meaningfully more strokes.
- **`lie_dispersion/`** — Sandhaven, Hole 15 (Par 4): the identical PW shot
  evaluated three times, changing only the lie (`fairway` / `rough` /
  `bunker`). Validates that lie-based dispersion modifiers degrade accuracy
  and expected strokes in the correct fairway < rough < bunker order, with the
  green-hit probability falling as the lie gets worse.
- **`wind_adjusted_dispersion/`** — Crosswind Pines, Hole 6 (Par 5): a water
  hazard hugs the positive-y side of the 5i layup corridor and a 10 mph `90°`
  crosswind (toward `+y`) blows the straight line into it. Validates that the
  wind-adjusted dispersion changes the recommended target (aim toward `-y`,
  into the wind) with a measurably safer cross-track outcome, robust across
  multiple seeds. The `ob_risk` shot fixture deliberately declares `0` mph
  wind so that scenario keeps testing pure OB stroke-and-distance behavior;
  the engine still consumes its `ShotEvent.wind` (every sample receives it).

Each test sets its random seed explicitly and asserts only comparative /
threshold properties (never golden floats), verified against the observed
output at that seed.

Run each end-to-end (or with `--samples`/`--seed` overrides):

```bash
python3 -m ghostcaddie run --shot data/scenarios/layup_vs_attack/shot.json --course data/scenarios/layup_vs_attack/hole.json --player data/scenarios/layup_vs_attack/player.json --out out/scenario_a/
python3 -m ghostcaddie run --shot data/scenarios/ob_risk/shot.json --course data/scenarios/ob_risk/hole.json --player data/scenarios/ob_risk/player.json --out out/scenario_b/
python3 -m ghostcaddie run --shot data/scenarios/lie_dispersion/shot_fairway.json --course data/scenarios/lie_dispersion/hole.json --player data/scenarios/lie_dispersion/player.json --out out/scenario_c_fairway/
python3 -m ghostcaddie run --shot data/scenarios/lie_dispersion/shot_rough.json --course data/scenarios/lie_dispersion/hole.json --player data/scenarios/lie_dispersion/player.json --out out/scenario_c_rough/
python3 -m ghostcaddie run --shot data/scenarios/lie_dispersion/shot_bunker.json --course data/scenarios/lie_dispersion/hole.json --player data/scenarios/lie_dispersion/player.json --out out/scenario_c_bunker/
python3 -m ghostcaddie run --shot data/scenarios/wind_adjusted_dispersion/shot.json --course data/scenarios/wind_adjusted_dispersion/hole.json --player data/scenarios/wind_adjusted_dispersion/player.json --out out/scenario_d/
```

## Wind-adjusted dispersion

`ShotEvent.wind` drives a configurable, deterministic wind shift in the
dispersion engine.

**Wind contract** (documented trust boundary, validated in `ShotEvent`):

- `speed_mph` is the wind speed in miles per hour (non-negative, finite).
- `direction_deg` is the direction the wind vector **travels toward**, not the
  direction it comes from, in the engine frame: top-down yards, `0°` along
  `+x` (tee-to-pin), `90°` along `+y`, angles increasing counterclockwise.
  So for a shot aimed straight along `+x`: `0°` is a tailwind, `180°` a
  headwind, `90°` a crosswind toward `+y`, and `270°` a crosswind toward `-y`.
- Internal math uses the direct vector `(speed_mph * cos(theta), speed_mph *
  sin(theta))`. A future weather adapter converting "wind from" readings would
  use `(from_direction + 180) % 360`; none is implemented yet.

**Coefficient model.** The wind vector is projected onto the shot's strike
frame (along the aim, and left-lateral perpendicular to it) and applied as a
mean shift to the two Gaussian means:

- `along_wind_carry_yd_per_mph` (default `1.5`) scales the along-aim
  component's carry shift.
- `crosswind_lateral_drift_yd_per_mph` (default `1.0`) scales the lateral
  component's drift.

These live in `SimulationConfig` and are illustrative linear sensitivities in
yards per mph — not physical flight constants. The lateral Gaussian mean
becomes `club.miss_bias_yd + lateral_component * crosswind_coefficient`, and
the carry mean gains `along_component * along_coefficient`.

**Deterministic zero-wind guarantee.** Wind shifts only the Gaussian means;
it never adds random draws. With `speed_mph == 0` (or `wind=None`), the
sampled landing sequence is bit-for-bit identical to the pre-wind
implementation for the same seed.

**Limitations.** Wind is applied as a mean shift only. It does not model
launch angle, trajectory, spin, loft, elevation, gusts, or time-varying wind;
dispersion spread is unchanged by wind; and the along-wind shift is the same
for every club regardless of ball flight. Full ball-flight physics is out of
scope.

**Scenario.** `data/scenarios/wind_adjusted_dispersion/` is a synthetic
crosswind course: a water hazard hugs the positive-y side of the 5i layup
corridor, and the 10 mph `90°` crosswind blows the straight line into it. The
engine recommends aiming 15 yd toward `-y` (into the wind). Its test
(`tests/test_scenario_wind_adjusted_dispersion.py`) asserts comparative
properties across five explicit seeds — never golden floats.

```bash
python3 -m ghostcaddie run --shot data/scenarios/wind_adjusted_dispersion/shot.json --course data/scenarios/wind_adjusted_dispersion/hole.json --player data/scenarios/wind_adjusted_dispersion/player.json --out out/scenario_d/
```

## Four-point calibration

`CoordinateSystem` supports two modes. Manual mode is unchanged and remains
the default.

- **`mode: "manual"`** — raw shot points are already engine-yard coordinates,
  relative to `origin`. `to_engine` subtracts the origin; `from_engine` adds
  it back. Behavior is exactly as before this feature.
- **`mode: "four_point"`** — raw shot points are source-image coordinates
  (typically pixels) mapped to engine coordinates by a planar homography fit
  to exactly four ordered correspondences.

**JSON shape.** A four-point course declares:

```json
"coordinate_system": {
  "mode": "four_point",
  "units": "yards",
  "source_units": "pixels",
  "source_points": [
    {"x": 100, "y": 80}, {"x": 900, "y": 80},
    {"x": 900, "y": 620}, {"x": 100, "y": 620}
  ],
  "engine_points": [
    {"x": 0, "y": 0}, {"x": 300, "y": 0},
    {"x": 300, "y": 200}, {"x": 0, "y": 200}
  ]
}
```

**Coordinate contract.**

- `source_points` are raw 2D image points in `source_units` (default
  `"pixels"`), with the conventional image axes: x right, y down.
- `engine_points` are top-down course coordinates in `units` (default
  `"yards"`), in the course's declared frame.
- Point `i` in `source_points` corresponds to point `i` in `engine_points`.
  List both quadrilaterals in the same perimeter order (for example
  top-left, top-right, bottom-right, bottom-left) and do not reorder them.
- `to_engine` is applied once at ingestion, so analytics, hazards, and the
  renderer always receive engine coordinates. `from_engine` is an explicit,
  separate reverse re-projection for tooling/rendering callers only; it never
  feeds source coordinates back into engine calculations.

**Validation.** Four-point construction requires exactly four paired points
and rejects duplicate, near-collinear, or otherwise near-singular point sets
with a `ValueError` (scale-aware, so pixel- and yard-sized coordinates are
both checked).

**Limitations.** The homography is a planar four-point calibration. It does
not perform lens-distortion correction, camera-pose estimation, video
processing, or automatic point detection. Points outside the calibrated
quadrilateral are extrapolated and may be unreliable.

### Fixture video analysis

Milestone 8 provides a deterministic, fixture-only command:

```bash
python -m ghostcaddie video-analyze --video /absolute/path/shot.mp4 \
  --project-root . --calibration fixtures/calibration.json \
  --course data/sample_hole.json --player data/sample_player.json \
  --observations fixtures/observations.json --out out/video \
  --event-id E1 --tournament-id T1 --hole 7 --shot-number 2 \
  --lie fairway --club 7i --distance-to-pin 150 --target-x 700 --target-y 500
```

The versioned pixel observation contract uses this canonical golf phase enum:

```text
unknown, address, backswing, top, downswing, contact,
follow_through, ball_flight, landing, rolling, finish
```

Only these documented aliases are normalized, case-insensitively and with
surrounding whitespace removed: `setup` and `setup/address` → `address`,
`impact` → `contact`, `follow-through` and `follow through` → `follow_through`,
and `flight` and `ball flight` → `ball_flight`. Unknown or ambiguous phase
terms are rejected; model output is never silently coerced beyond this list.

Model-backed perception is isolated and opt-in. The default adapter target is the
local `gemma4:e2b` Ollama model; cloud model testing is a separate explicit mode
because it uploads selected frames. At present, `gemma4:e2b` is reachable but
has not produced a valid `video-observations.v1` response in smoke testing, so
fixture mode remains the supported path and no model-derived analytics are
claimed.

`--video` is a local regular readable file and may be absolute; its path is
used only as input and is never serialized. `--calibration`, `--course`,
`--player`, and `--observations` are project resources and must be relative to
`--project-root`; absolute paths, `..` traversal, and symlink escapes are
rejected. The command runs `ffprobe`, extracts deterministic sampled frames,
creates a contact sheet, loads fixture observations, calls the unchanged
`run_pipeline`, and writes `diagnostics.json`, `recommendation.json`,
`overlay.svg`, `normalized_shot.json`, and annotated frames. Add
`--render-video` to encode those sampled annotated frames as
`annotated_video.mp4`; this is honestly a sampled sequence export, not a
promise of original-frame preservation.

## Multi-shot session orchestration

`ghostcaddie session` runs a whole round slice: one versioned inline envelope
is parsed once, every normalized shot is run through the unchanged
`run_pipeline()`, and a deterministic per-shot / per-hole / session report is
written.

```bash
python3 -m ghostcaddie session --input data/sample_session.json --out out/session/
```

Writes `out/session/session_report.json` and prints a concise session summary.
The existing `run` command is unchanged and still writes `recommendation.json`
and `overlay.svg`.

**Envelope schema (`schema_version: "0.1"`).** The input is a self-contained
JSON envelope with four required top-level sections:

- `session` — mandatory non-empty `session_id`, `tournament_id`, `player_id`,
  `course_id`; positive finite integral `round_number`; optional integral
  `seed` (default 42).
- `player_profile` — the existing player record (shared by every shot).
- `course` — mandatory `course_id` plus a non-empty `holes` list; each hole is
  an existing course record with a `hole_number`. Hole numbers must be unique.
- `shots` — a non-empty, strictly ordered list of shot records. Each shot has a
  mandatory non-empty `shot_id`, positive finite integral `hole_number` and
  `shot_number`, and the existing shot fields. Optional per-shot `player_id`,
  `tournament_id`, and `course_id` must match the session when present and are
  inherited from the session when absent.

**Validation.** The parser is the sole inline-ingestion boundary. It rejects
malformed, inconsistent, duplicate, out-of-order, non-finite, missing-reference,
and unknown-reference data before any analytics call: wrong `schema_version`,
missing sections, empty IDs, non-positive or non-integral ordinals, duplicate
hole numbers, duplicate `shot_id`s, duplicate `(hole_number, shot_number)`
pairs, shots not strictly ordered by `(hole_number, shot_number)`, shots
referencing undeclared holes, and identity mismatches all raise `ValueError`.

**Coordinate mapping.** Each hole's `CoordinateMapper` is applied exactly once
at parse time: `start_position`, `target_position`, and
`actual_landing_position` are normalized to engine coordinates and never mapped
again. `run_session()` feeds protocol-compatible in-memory sources (never
temporary files) to the unchanged `run_pipeline()` once per ordered shot.

**Seeds.** Per-shot seeds are derived with SHA-256 over
`f"{session_seed}:{ordinal}:{shot_id}"` (zero-based ordinal), so a fixed
session seed reproduces identical analytics for every shot. The built-in
`hash()` is never used.

**Report.** `session_report.json` has top-level `schema_version`, `session`,
`summary`, `holes`, `shot_results`, and `provenance`. `shot_results` are in
input order with `shot_id`, `hole_number`, `shot_number`, `recommendation`, and
per-shot `provenance` (inline sources are identified as `session:inline:...`,
never fabricated filesystem paths). Hole entries group their ordered shots with
`shot_count`, `shot_ids`, `sum_local_decision_cost`, `hazard_risk_summary`,
and `recommendations`. The session summary carries `shot_count`, `hole_count`,
`sum_local_decision_cost`, `decision_cost_semantics`, `highest_cost_decisions`,
and `hazard_risk_summary`.

**Aggregation semantics.** Only local recommendation metadata is aggregated:
`decision_cost` is summed per hole and per session, and existing rounded hazard
probabilities are aggregated as per-region `max`, `mean`, and
`nonzero_shot_count` (omitted hazards treated as zero). `sum_local_decision_cost`
is explicitly labeled a **local, non-additive diagnostic** — it is NOT official
Strokes Gained and NOT an official round stroke total. `expected_strokes` and
`actual_expected_strokes` remain only inside per-shot recommendations and are
never summed or emitted as session/hole aggregate metrics. Output is validated
recursively for finiteness and serialized with `json.dumps(..., allow_nan=False)`.

**Limitations.** The session layer is synthetic-data and standard-library only
(`dataclasses`, `json`, `hashlib`, `math`, `argparse`, `pathlib`, `unittest`).
It adds no future integrations or dependencies.

## Provider-aware session envelopes

Provider-shaped multi-shot inputs use `schema_version: "provider-session.v1"` and keep the concrete vendor payload under each ordered shot wrapper's `provider_record`. The `session` section declares `provider` (`shotlink` or `trackman`) and its matching provider schema version, while `course_source.path` and `player_source.path` point to separate JSON files relative to the envelope file. Paths must be relative, existing regular files inside the project root, and symlinks escaping that root are rejected. No source paths are emitted in reports.

```bash
python3 -m ghostcaddie provider-session \
  --input data/providers/sessions/shotlink_session.json \
  --out out/provider-session-shotlink/
```

Validation is strict by default, including unknown nested fields. Pass `--permissive` to retain unknown-field paths in diagnostics/provenance. Provider coordinates are normalized exactly once at ingestion and then passed to the existing session pipeline. Course context remains separate and may be supplied by the course source or explicitly per shot. The included provider-session fixtures are synthetic/provider-shaped only; there are no live APIs, credentials, scraping, or production integrations.


The whole simulation is reproducible for a fixed seed: every random draw goes
through a single seeded `random.Random` consumed sequentially; the global
`random` module is never used.

## Current research status

A bounded local research demo remains available for the MMU biomechanics sequence, but it is a separate ball/tracer research probe: the source is 600x480 and encoded at 25 FPS, with capture FPS and reuse rights unverified. The current accepted continuous golfer/pose visual baseline is the local Pexels 6573485 artifact; ball observations are unavailable after the plausibility correction documented in [`STATUS.md`](STATUS.md). The MMU marker was inspected across all 160 rendered frames (source indices 0–159) and remained on the visible ball from pre-impact through upward/rightward translation; it must not be combined with the Pexels golfer track or treated as ground truth.

Best local artifacts:

```bash
open out/research_training_gauntlet/mmu_candidate/analysis/automatic_ball_tracer_h264.mp4
open out/research_training_gauntlet/mmu_candidate/analysis/automatic_ball_tracer_contact_full.jpg
open out/research_training_gauntlet/mmu_candidate/analysis/automatic_ball_tracer_diagnostics.json
python3 -m ghostcaddie fairwayos-ball-sidecar \
  --input out/research_training_gauntlet/mmu_candidate/analysis/automatic_ball_tracer_diagnostics.json \
  --out out/research_training_gauntlet/mmu_candidate/analysis/fairwayos_ball_research_sidecar.json \
  --source mmu_candidate/source.mp4
```

The sidecar is a diagnostics handoff only: it preserves pixel-space track items, frame provenance, warnings, and human fallback while explicitly setting `production_eligible: false`. It does not construct `VideoObservations`, `ShotEvent`, analytics, or recommendations.

This remains heuristic pixel-space research evidence. Clubhead, validated impact, trajectory, landing, calibration, `ShotEvent`, analytics, and recommendation remain unavailable. Production gates are closed and `run_pipeline()` is not invoked. See [`STATUS.md`](STATUS.md) for provenance, QA artifacts, and verification commands.

## Dependencies

Pure Python standard library only for the analytics core, the default workflow, and
the canonical video orchestrator (which imports no cv2, no numpy, no model runtime
and opens no socket — decode, pose and rendering run in a separate worker process) (dataclasses, json, math, random,
statistics, argparse, unittest, pathlib, enum, typing). Optional video/research environments may use third-party
packages such as numpy, OpenCV, PyTorch, and Ultralytics; those dependencies and downloaded weights remain outside
this public source tree. They are documented as future/optional upgrades in `requirements.txt` (commented out).
