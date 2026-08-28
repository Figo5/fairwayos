# Four-Point Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a validated pure-stdlib four-point projective mapper for source-image coordinates while preserving manual coordinate mapping and the accepted wind-adjusted dispersion behavior.

**Architecture:** Keep calibration entirely inside the existing geometry/ingestion boundary. `CoordinateSystem(mode="manual")` continues to subtract its origin exactly as before; `CoordinateSystem(mode="four_point")` stores four ordered source-image ↔ engine-coordinate correspondences, and `CoordinateMapper` solves one source→engine homography plus its inverse once at construction. Adapters continue converting raw shot points to engine coordinates before analytics, while `from_engine` is available only for explicit re-projection and rendering/tooling callers; no analytics or renderer changes are needed.

**Tech Stack:** Pure Python 3.9 standard library (`dataclasses`, `math`, `json`, `typing`, `unittest`); no external packages or image/video processing.

## Global Constraints

- Preserve `CoordinateSystem(mode="manual")` and its current origin subtraction behavior exactly.
- Implement a four-point source-image to engine-coordinate projective mapping and the reverse engine-coordinate to source-image mapping.
- Use a pure-stdlib homography solver; do not install packages.
- Reject duplicate, near-collinear, and near-singular point sets with clear `ValueError`s.
- Use tolerance-based assertions for floating-point projective mappings; do not require exact equality except for unchanged manual behavior where appropriate.
- Source-image coordinates are raw 2D image points in the declared `source_units` (typically pixels, x right and y down); engine coordinates are top-down course coordinates in the declared `engine_units` (currently yards, x/y as the course frame declares).
- Four correspondences are ordered consistently: source point `i` corresponds to engine point `i`; ordering should follow the same perimeter order (for example top-left, top-right, bottom-right, bottom-left) and must not be silently reordered.
- Keep analytics in engine coordinates; reverse mapping is a separate explicit mapper operation and must not feed rendered/source coordinates back into engine calculations.
- Do not modify wind code or add video-processing, camera-pose, lens-distortion, FastAPI, or third-party dependencies.
- Git is unavailable; do not add Git steps or retry Git writes.

## File Map

- Modify `ghostcaddie/geometry.py`: extend `CoordinateSystem` schema and implement the pure-stdlib homography/inverse plus mapper dispatch and validation.
- Modify `ghostcaddie/adapters/json_file.py`: parse optional four-point calibration fields without changing manual JSON behavior.
- Modify `tests/test_geometry.py`: manual regression, perspective mapping, reverse mapping, round trips, and degenerate validation.
- Modify `tests/test_pipeline_end_to_end.py`: prove adapter-fed points remain engine coordinates under four-point calibration while the pipeline stays unchanged.
- Modify `README.md`: document calibration JSON shape, source/engine coordinate conventions, point ordering, units, reverse mapping, and limitations.
- Do not modify wind implementation files; only the existing wind tests should continue to pass.

---

### Task 1: Define calibration schema and manual-mode compatibility

**Files:**
- Modify: `ghostcaddie/geometry.py:18-50`
- Modify: `ghostcaddie/adapters/json_file.py:33-53`
- Test: `tests/test_geometry.py`

**Interfaces:**
- Consumes: existing `CoordinateSystem(mode="manual", origin=Point2D(...), units="yards")` and JSON `coordinate_system` objects.
- Produces: `CoordinateSystem` fields for four-point mode, using explicit paired sequences such as `source_points: Tuple[Point2D, ...]` and `engine_points: Tuple[Point2D, ...]`, plus `source_units` (default `"pixels"`) and existing `units` as engine units. `JsonCourseDataSource` must pass these fields when present and preserve current manual defaults.

- [ ] **Step 1: Add failing schema/compatibility tests.** In `tests/test_geometry.py`, add a manual-mode regression that constructs the existing nonzero-origin mapper and asserts the current `to_engine`/`from_engine` results. Add a four-point construction using four simple corner pairs and assert the `CoordinateSystem` stores mode, units, source units, and point tuples without altering manual semantics. Add a JSON adapter fixture in the test using a temporary course JSON or a direct `CoordinateSystem` parse probe.

- [ ] **Step 2: Run focused geometry tests and confirm the new schema assertions fail.**

Run:
```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest tests.test_geometry -v
```

Expected: current `CoordinateSystem` rejects the new calibration fields or the adapter ignores them.

- [ ] **Step 3: Implement schema parsing with no wind changes.** Add optional calibration fields with immutable tuple defaults, validate mode values (`manual` and `four_point`), require exactly four paired points for `four_point`, and have the JSON course adapter parse a structure such as:

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

Keep `origin` accepted for manual mode and leave `CoordinateMapper` behavior unchanged until the homography task. Reject mismatched source/engine lengths and invalid modes with `ValueError`.

- [ ] **Step 4: Run focused and baseline tests.**

Run:
```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest tests.test_geometry -v
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest discover -s tests -v
```

Expected: all current tests and the new schema/manual tests pass.

---

### Task 2: Implement homography and inverse mapping

**Files:**
- Modify: `ghostcaddie/geometry.py:34-81`
- Test: `tests/test_geometry.py`

**Interfaces:**
- Consumes: four finite paired `Point2D` tuples from `CoordinateSystem`.
- Produces: `CoordinateMapper.to_engine(raw: dict) -> Point2D` and `from_engine(point: Point2D) -> dict` supporting both modes; a private or module-level homography implementation may expose `Homography` only if useful, but the public mapper API must remain these two methods.

- [ ] **Step 1: Write failing perspective and reverse tests.** Use a known projective mapping represented by:

```text
X = (2u + 0.5v + 10) / (0.001u + 0.002v + 1)
Y = (-0.25u + 3v + 20) / (0.001u + 0.002v + 1)
```

Generate four source corner points, compute their expected engine points from that formula, construct `CoordinateSystem(mode="four_point", ...)`, and assert each corner plus at least two interior source points map within `1e-8` or an appropriately justified tolerance. Assert `from_engine(to_engine(source))` returns the source coordinates within tolerance and `to_engine(from_engine(engine))` returns the engine point within tolerance.

- [ ] **Step 2: Run focused tests and confirm they fail.**

Run:
```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest tests.test_geometry -v
```

Expected: four-point mapping currently still performs manual origin subtraction or lacks implementation.

- [ ] **Step 3: Implement an 8-unknown homography solver with inverse.** Solve the standard eight-equation linear system for coefficients `[a, b, c, d, e, f, g, h]` with `H = [[a,b,c],[d,e,f],[g,h,1]]` and mapping `X=(au+bv+c)/(gu+hv+1)`, `Y=(du+ev+f)/(gu+hv+1)`. Implement partial-pivot Gaussian elimination using only `math` and lists. Reject pivots whose absolute value is below a scale-aware tolerance; reject non-finite inputs/results and denominators near zero. Construct the inverse by solving the reverse correspondence with the same solver (rather than relying on a fragile hand-coded matrix inverse), retaining independent validation. In `CoordinateMapper.__init__`, dispatch to manual origin subtraction or the two homographies based on `coordinate_system.mode`. `to_engine` accepts raw `{"x", "y"}` and `from_engine` returns the same shape.

- [ ] **Step 4: Add near-degenerate validation tests.** Add cases with duplicate source/destination points, source points almost collinear, destination points almost collinear, and a system with a near-zero solve pivot; assert each raises `ValueError` during mapper construction. Use coordinates scaled like image pixels and yards so the test checks scale-aware thresholds rather than only exact zero. Also assert a valid perspective quadrilateral with ordinary image/yard scales is accepted.

- [ ] **Step 5: Run focused geometry tests and the full regression suite.**

Run:
```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest tests.test_geometry -v
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest discover -s tests -v
```

Expected: all projective, inverse, round-trip, validation, manual, and pre-existing wind tests pass.

---

### Task 3: Verify adapter and engine-coordinate boundaries

**Files:**
- Modify: `tests/test_pipeline_end_to_end.py`
- Modify: `ghostcaddie/adapters/json_file.py` only if Task 1 parsing needs a minimal correction

**Interfaces:**
- Consumes: JSON `CoordinateSystem(mode="four_point")`, `JsonCourseDataSource`, `JsonShotDataSource`, and the unchanged pipeline.
- Produces: proof that shot ingestion maps source-image coordinates into engine coordinates once, analytics receives engine points, and explicit reverse mapping is available without affecting rendering or wind dispersion.

- [ ] **Step 1: Add a failing adapter-boundary integration test.** Create a temporary or in-memory course declaration with four-point calibration, load it through `JsonCourseDataSource`, load a shot whose source positions are image coordinates, and assert `JsonShotDataSource.load_shot()` returns the expected engine-coordinate `start_position`, `target_position`, and `actual_landing_position` within tolerance. Assert the loaded `course.coordinate_system.mode` is `four_point`; assert the reverse mapper returns the original source point within tolerance.

- [ ] **Step 2: Run the integration test and confirm it fails if parsing/wiring is incomplete.**

Run:
```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest tests.test_pipeline_end_to_end -v
```

Expected: the calibration course schema or adapter path is not yet fully integrated if Task 1 did not cover it.

- [ ] **Step 3: Make the minimal adapter correction.** Ensure `JsonCourseDataSource` parses `source_points` and `engine_points` through the same `_point` helper and that `JsonShotDataSource` continues to call `CoordinateMapper.to_engine` for all three shot positions. Do not pass source-image coordinates into simulation, dispersion, hazards, or renderer code.

- [ ] **Step 4: Run the focused integration and full suite.**

Run:
```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest tests.test_pipeline_end_to_end -v
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest discover -s tests -v
```

Expected: integration and all 55 wind-baseline tests pass.

---

### Task 4: Document and verify the calibration slice

**Files:**
- Modify: `README.md:30-55`
- Test: `tests/test_geometry.py` and/or `tests/test_pipeline_end_to_end.py` only for final assertions

**Interfaces:**
- Consumes: the implemented schema, mapper, adapter boundary, and existing CLI fixtures.
- Produces: complete user-facing calibration documentation and fresh evidence for focused calibration tests, full regression, and all existing CLI scenarios.

- [ ] **Step 1: Document the JSON shape and coordinate contract.** Add a “Four-point calibration” section explaining:
  - manual mode remains `mode: "manual"`, raw points are engine-yard coordinates relative to `origin`, and behavior is unchanged;
  - four-point mode uses `mode: "four_point"`, `source_units` (usually pixels), existing `units` for engine units (yards), and paired `source_points`/`engine_points` arrays;
  - point `i` in each array corresponds to point `i` in the other, and authors should list both quadrilaterals in matching perimeter order (top-left, top-right, bottom-right, bottom-left) without reordering;
  - source-image axes convention is x right/y down unless the source declares otherwise, while engine axes are the course’s top-down frame;
  - `to_engine` is used at ingestion, `from_engine` is explicit reverse reprojection, and analytics remain engine-coordinate only;
  - the homography is planar four-point calibration and does not perform lens-distortion correction, camera-pose estimation, video processing, or automatic point detection; near-collinear/near-singular correspondences are rejected.

- [ ] **Step 2: Run focused calibration tests.**

Run:
```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest tests.test_geometry -v
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest tests.test_pipeline_end_to_end -v
```

Expected: all focused tests pass, including tolerance-based perspective, reverse, round-trip, degenerate validation, manual compatibility, and adapter-boundary checks.

- [ ] **Step 3: Run the complete regression suite.**

Run:
```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest discover -s tests -v
```

Expected: **at least 55 tests**, zero failures/errors; wind tests remain unchanged and passing.

- [ ] **Step 4: Run every existing CLI scenario.**

Run:
```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m ghostcaddie run --shot data/sample_shot.json --course data/sample_hole.json --player data/sample_player.json --out "$TMPDIR/ghostcaddie-calibration-cli/sample"
cd /Users/giofiore/ghostcaddie-tour && python3 -m ghostcaddie run --shot data/scenarios/layup_vs_attack/shot.json --course data/scenarios/layup_vs_attack/hole.json --player data/scenarios/layup_vs_attack/player.json --out "$TMPDIR/ghostcaddie-calibration-cli/layup_vs_attack"
cd /Users/giofiore/ghostcaddie-tour && python3 -m ghostcaddie run --shot data/scenarios/ob_risk/shot.json --course data/scenarios/ob_risk/hole.json --player data/scenarios/ob_risk/player.json --out "$TMPDIR/ghostcaddie-calibration-cli/ob_risk"
cd /Users/giofiore/ghostcaddie-tour && python3 -m ghostcaddie run --shot data/scenarios/lie_dispersion/shot_fairway.json --course data/scenarios/lie_dispersion/hole.json --player data/scenarios/lie_dispersion/player.json --out "$TMPDIR/ghostcaddie-calibration-cli/lie_fairway"
cd /Users/giofiore/ghostcaddie-tour && python3 -m ghostcaddie run --shot data/scenarios/lie_dispersion/shot_rough.json --course data/scenarios/lie_dispersion/hole.json --player data/scenarios/lie_dispersion/player.json --out "$TMPDIR/ghostcaddie-calibration-cli/lie_rough"
cd /Users/giofiore/ghostcaddie-tour && python3 -m ghostcaddie run --shot data/scenarios/lie_dispersion/shot_bunker.json --course data/scenarios/lie_dispersion/hole.json --player data/scenarios/lie_dispersion/player.json --out "$TMPDIR/ghostcaddie-calibration-cli/lie_bunker"
cd /Users/giofiore/ghostcaddie-tour && python3 -m ghostcaddie run --shot data/scenarios/wind_adjusted_dispersion/shot.json --course data/scenarios/wind_adjusted_dispersion/hole.json --player data/scenarios/wind_adjusted_dispersion/player.json --out "$TMPDIR/ghostcaddie-calibration-cli/wind_adjusted_dispersion"
```

Expected: every command exits 0 and writes both `recommendation.json` and `overlay.svg`.

- [ ] **Step 5: Independently inspect before reporting completion.** Read the final geometry solver, mapper dispatch, validation, adapter parsing, tests, and README. Confirm no wind source files were modified, all projective assertions use tolerances, manual behavior remains unchanged, and no external package was installed. Report changed files, focused/full test results, CLI results, and known planar-homography limitations. Do not claim calibration complete until these checks are all evidenced.
