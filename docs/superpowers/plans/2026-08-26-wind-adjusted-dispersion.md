# Wind-Adjusted Dispersion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic, configurable wind-induced carry and lateral drift to the existing Monte Carlo dispersion engine while preserving exact zero-wind behavior.

**Architecture:** Keep wind interpretation at the dispersion boundary. `ShotEvent.wind` remains the source of `speed_mph` and `direction_deg`; `ShotSimulator` passes it to `GaussianDispersionModel`, which projects the wind vector onto the shot's along and left-lateral strike-frame axes and shifts the two Gaussian means. Coefficients live in `SimulationConfig`, while the model retains defaults so existing direct callers remain valid.

**Tech Stack:** Pure Python 3.9 standard library (`dataclasses`, `math`, `random`, `statistics`, `unittest`, JSON fixtures); no new packages.

## Global Constraints

- `direction_deg` is the direction the wind vector travels toward, not the direction it comes from.
- The engine frame is top-down yards with `0°` along `+x` (tee-to-pin), `90°` along `+y`, and angles increasing counterclockwise.
- Wind speed is measured in mph.
- For a shot along `+x`: `0°` is tailwind, `180°` is headwind, `90°` is crosswind toward `+y`, and `270°` is crosswind toward `-y`.
- Internal math uses the direct vector `(speed_mph * cos(theta), speed_mph * sin(theta))`.
- A future adapter converting weather-style “wind from” data uses `(from_direction + 180) % 360`; no weather adapter is implemented in this slice.
- Zero wind must produce the exact pre-wind landing sequence for the same seed, including the same random draws and floating-point landing coordinates.
- The simulation must remain deterministic for a fixed seed; never use the global random module.
- Do not implement full ball-flight physics, video processing, FastAPI, or third-party dependencies.
- Git is unavailable in this environment; do not add Git commit steps or retry `.git` writes.

## File Map

- Modify `ghostcaddie/models.py`: document the `ShotEvent.wind` schema and validate the small trust-boundary contract.
- Modify `ghostcaddie/config.py`: add named, documented along-wind carry and crosswind lateral-drift coefficients to `SimulationConfig`.
- Modify `ghostcaddie/dispersion.py`: project wind into the strike frame, apply configurable mean shifts, and preserve the no-wind path.
- Modify `ghostcaddie/simulation.py`: pass the current `ShotEvent.wind` to every dispersion sample.
- Modify `ghostcaddie/pipeline.py`: construct the dispersion model with the active simulation configuration.
- Modify `tests/test_dispersion.py`: test vector signs, zero-wind exact compatibility, cardinal wind effects, and seeded determinism.
- Create `data/scenarios/wind_adjusted_dispersion/hole.json`: synthetic crosswind course with a hazard boundary that makes wind-aware aiming meaningful.
- Create `data/scenarios/wind_adjusted_dispersion/player.json`: at least two clubs with distinct carry/spread trade-offs.
- Create `data/scenarios/wind_adjusted_dispersion/shot.json`: explicit toward-direction wind metadata and a reproducible shot setup.
- Create `tests/test_scenario_wind_adjusted_dispersion.py`: pipeline fixture assertions, preferred target/club change, and multi-seed robustness.
- Modify `README.md`: describe the wind contract, coefficient model, deterministic zero-wind guarantee, fixture command, and limitations.

---

### Task 1: Establish the wind data and configuration contract

**Files:**
- Modify: `ghostcaddie/models.py:54-77`
- Modify: `ghostcaddie/config.py:19-24`
- Test: `tests/test_dispersion.py`

**Interfaces:**
- Consumes: existing `ShotEvent.wind: Dict[str, float]` and `SimulationConfig`.
- Produces: documented `speed_mph` / `direction_deg` semantics and two named configuration fields used by `GaussianDispersionModel`:
  - `along_wind_carry_yd_per_mph: float = 1.5`
  - `crosswind_lateral_drift_yd_per_mph: float = 1.0`

- [ ] **Step 1: Add contract tests before implementation.** Extend `tests/test_dispersion.py` with a `TestWindContract` class that constructs a zero-wind `ShotEvent` and asserts it is accepted, rejects negative `speed_mph`, and rejects missing `speed_mph` or `direction_deg` with `ValueError`. Keep the test independent of JSON adapters.

- [ ] **Step 2: Run the focused tests and confirm the new validation fails.**

Run:
```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest tests.test_dispersion.TestWindContract -v
```

Expected: the new validation assertions fail because the current model only checks distance/hole/shot and treats wind as provenance.

- [ ] **Step 3: Implement the smallest contract change.** In `ShotEvent.__post_init__`, require the two documented numeric keys, coerce/check their numeric values without mutating the frozen-free dataclass, reject non-finite values and negative speed, and leave direction unbounded so equivalent angles such as `-90` and `270` remain usable by `sin`/`cos`. Replace the old “provenance only” comment with the exact toward-direction coordinate convention. Add the two coefficient fields to `SimulationConfig` with comments explaining that they are illustrative linear sensitivity coefficients in yards per mph, not physical flight constants.

- [ ] **Step 4: Run the focused tests and the full existing suite.**

Run:
```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest tests.test_dispersion -v
```

Then:
```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest discover -s tests -v
```

Expected: the new contract tests and all pre-existing tests pass.

---

### Task 2: Add direct-vector wind shifts to dispersion

**Files:**
- Modify: `ghostcaddie/dispersion.py:15-69`
- Modify: `tests/test_dispersion.py`

**Interfaces:**
- Consumes: `wind: Dict[str, float]`, `SimulationConfig` coefficients, existing `sample_landing` inputs, and the passed seeded `random.Random`.
- Produces: backward-compatible `GaussianDispersionModel(config: Optional[SimulationConfig] = None)`, plus `sample_landing(..., wind: Optional[Dict[str, float]] = None)` and `sample_many(..., wind: Optional[Dict[str, float]] = None)`. `None` means the pre-wind path for existing direct callers.

- [ ] **Step 1: Write failing mathematical tests.** Add tests that use a zero-spread substitute only through valid positive standard deviations and a sufficiently large sample, or an injected deterministic RNG if the implementation permits, to assert the signs with a straight `+x` aim:
  - `0°` wind raises mean `x` by approximately `speed * along_coefficient` and leaves mean `y` unchanged.
  - `180°` wind lowers mean `x` by the same amount.
  - `90°` wind raises mean `y` by approximately `speed * crosswind_coefficient` and leaves mean `x` unchanged.
  - `270°` wind lowers mean `y` by the same amount.
  Also add an exact equality test comparing `sample_many(..., wind=None)` with `sample_many(..., wind={"speed_mph": 0, "direction_deg": 90})` under identical seeds, and retain the existing same-seed/different-seed test.

- [ ] **Step 2: Run the focused tests and confirm the wind assertions fail.**

Run:
```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest tests.test_dispersion -v
```

Expected: zero-wind compatibility already passes for `None` versus the current ignored wind only if the implementation accepts the argument; cardinal shift tests fail because no wind mean shift exists.

- [ ] **Step 3: Implement the direct-vector projection.** Store the supplied `SimulationConfig` or its two coefficients in `GaussianDispersionModel`. Before sampling, preserve the existing effective carry/spread calculations. If `wind is None` or `wind["speed_mph"] == 0`, set shifts to exactly zero and execute the existing two `rng.gauss` calls unchanged. Otherwise compute:

```python
wind_radians = math.radians(wind["direction_deg"])
wind_x = wind["speed_mph"] * math.cos(wind_radians)
wind_y = wind["speed_mph"] * math.sin(wind_radians)
shot_radians = math.radians(bearing_deg(start, aim))
along_component_mph = wind_x * math.cos(shot_radians) + wind_y * math.sin(shot_radians)
lateral_component_mph = -wind_x * math.sin(shot_radians) + wind_y * math.cos(shot_radians)
```

Use `effective_carry_mean + along_component_mph * along_wind_carry_yd_per_mph` as the along Gaussian mean and `club.miss_bias_yd + lateral_component_mph * crosswind_lateral_drift_yd_per_mph` as the lateral Gaussian mean. Keep the existing rotation and random draw order exactly unchanged. Pass the optional wind through `sample_many` without adding any random calls.

- [ ] **Step 4: Run the focused and complete tests.**

Run:
```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest tests.test_dispersion -v
```

Then:
```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest discover -s tests -v
```

Expected: all dispersion tests pass and no existing scenario regresses beyond its comparative assertions.

---

### Task 3: Thread shot wind through the production simulation

**Files:**
- Modify: `ghostcaddie/simulation.py:78-113`
- Modify: `ghostcaddie/pipeline.py:37-43`
- Test: `tests/test_pipeline_end_to_end.py`

**Interfaces:**
- Consumes: `ShotEvent.wind`, `SimulationConfig`, and the new optional wind argument on `DispersionModel.sample_landing`.
- Produces: every candidate and actual-decision Monte Carlo sample uses the same shot wind and the active configured coefficients; fixed-seed ordering remains unchanged.

- [ ] **Step 1: Add a production wiring probe.** In `tests/test_pipeline_end_to_end.py`, add a test using a recording dispersion implementation or a recording subclass that captures the wind argument and assert every evaluation receives the loaded `ShotEvent.wind`. Also assert that `run_pipeline` passes the test config's custom coefficient values into the constructed Gaussian model indirectly through a measurable sample shift or directly through the model's public configuration attributes, whichever matches the implementation's minimal seam.

- [ ] **Step 2: Run the new wiring test and observe failure.**

Run:
```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest tests.test_pipeline_end_to_end -v
```

Expected: the recording probe sees no wind because `ShotSimulator` currently calls `sample_landing` without it, and the pipeline currently constructs `GaussianDispersionModel()` without the active simulation config.

- [ ] **Step 3: Thread the values without changing candidate order.** Pass `wind=shot.wind` from `ShotSimulator.evaluate_candidate` to `sample_landing`; pass `wind=shot.wind` from `sample_many` only if any production caller uses that helper. In `run_pipeline`, construct `GaussianDispersionModel(config.simulation)`. Do not alter RNG initialization, candidate generation, candidate evaluation order, or actual-decision sequencing.

- [ ] **Step 4: Run the complete suite and the sample CLI.**

Run:
```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest discover -s tests -v
```

Then:
```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m ghostcaddie run --shot data/sample_shot.json --course data/sample_hole.json --player data/sample_player.json --out /tmp/ghostcaddie-wind-sample
```

Expected: all tests pass, the command exits 0, and it writes `recommendation.json` and `overlay.svg`.

---

### Task 4: Add an end-to-end wind scenario and robustness checks

**Files:**
- Create: `data/scenarios/wind_adjusted_dispersion/hole.json`
- Create: `data/scenarios/wind_adjusted_dispersion/player.json`
- Create: `data/scenarios/wind_adjusted_dispersion/shot.json`
- Create: `tests/test_scenario_wind_adjusted_dispersion.py`

**Interfaces:**
- Consumes: the JSON adapters, `run_pipeline`, `Config`, `replace`, and the production wind path.
- Produces: a documented synthetic fixture where wind changes the preferred target or club, with assertions robust across multiple seeds rather than golden floating-point values.

- [ ] **Step 1: Add the fixture and failing scenario assertions.** Author a straight tee-to-pin hole in the declared `manual`/yards frame. Put a lateral hazard or out-of-bounds strip along one side of the target corridor, use at least two clubs with different carry/lateral trade-offs, and set `shot.json` to an explicit nonzero toward-direction crosswind (`speed_mph` and `direction_deg`) plus the same fields as existing shots. The test must run the same setup twice—once with `speed_mph: 0` and once with the fixture wind—and assert the recommended target or club differs, the wind-aware line has a measurably different cross-track/hazard outcome, and the wind metadata uses the documented toward convention.

- [ ] **Step 2: Add the multi-seed robustness check.** For a fixed list of at least five explicit seeds (for example `7, 42, 1234, 4242, 99999`), run the zero-wind and windy versions with a moderate sample count. Assert the zero-wind recommendation is stable relative to the pre-wind baseline and that the windy recommendation remains within the intended set of safe candidates, rather than asserting one exact Monte Carlo float for every seed. Include a same-seed repeated-run equality assertion for recommendation values.

- [ ] **Step 3: Run the new scenario test and tune only fixture geometry if needed.**

Run:
```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest tests.test_scenario_wind_adjusted_dispersion -v
```

Expected: the fixture demonstrates a wind-caused preferred target or club change and passes across the selected seeds. If a property is seed-fragile, widen the authored hazard slack or use comparative thresholds; do not weaken the wind math or hard-code a recommendation.

- [ ] **Step 4: Run all tests again.**

Run:
```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest discover -s tests -v
```

Expected: the complete suite passes, including all prior 33 tests plus the new wind coverage.

---

### Task 5: Document assumptions and perform the required verification pass

**Files:**
- Modify: `README.md:96-108`
- Modify: `tests/test_dispersion.py` or `tests/test_scenario_wind_adjusted_dispersion.py` only if the verification probe needs a small assertion

**Interfaces:**
- Consumes: implemented wind contract, fixture paths, and CLI behavior.
- Produces: user-facing assumptions/limitations and evidence that old zero-wind behavior is unchanged.

- [ ] **Step 1: Add README documentation.** Add a “Wind-adjusted dispersion” section that states, verbatim in substance: speed is mph; direction is where wind travels toward; engine coordinates are yards with `0°=+x`, `90°=+y`, counterclockwise angles; straight `+x` interpretation for tail/head and ± crosswind; the linear coefficients are configurable illustrative sensitivities in yards per mph; weather “from” values require `(from_direction + 180) % 360`; wind is applied as a mean shift only and does not model launch angle, trajectory, spin, loft, elevation, gusts, or time-varying wind. Document the new scenario command and the multi-seed nature of its test.

- [ ] **Step 2: Compare zero-wind output with the pre-wind implementation.** Use a small inline Python probe or a test that implements the old two-`gauss`/rotation formula locally, then compare a fixed-seed sequence from `GaussianDispersionModel` with `wind={"speed_mph": 0, "direction_deg": 90}` point-for-point against the old formula. Also run the existing sample/scenario CLI commands with `--seed` and `--samples` overrides to ensure all commands still exit successfully. Do not compare generated timestamps or require golden recommendation floats.

- [ ] **Step 3: Run the complete final verification commands.**

Run:
```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest discover -s tests -v
```

Then run every existing and new CLI scenario:
```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m ghostcaddie run --shot data/sample_shot.json --course data/sample_hole.json --player data/sample_player.json --out /tmp/ghostcaddie-cli/sample
cd /Users/giofiore/ghostcaddie-tour && python3 -m ghostcaddie run --shot data/scenarios/layup_vs_attack/shot.json --course data/scenarios/layup_vs_attack/hole.json --player data/scenarios/layup_vs_attack/player.json --out /tmp/ghostcaddie-cli/layup_vs_attack
cd /Users/giofiore/ghostcaddie-tour && python3 -m ghostcaddie run --shot data/scenarios/ob_risk/shot.json --course data/scenarios/ob_risk/hole.json --player data/scenarios/ob_risk/player.json --out /tmp/ghostcaddie-cli/ob_risk
cd /Users/giofiore/ghostcaddie-tour && python3 -m ghostcaddie run --shot data/scenarios/lie_dispersion/shot_fairway.json --course data/scenarios/lie_dispersion/hole.json --player data/scenarios/lie_dispersion/player.json --out /tmp/ghostcaddie-cli/lie_fairway
cd /Users/giofiore/ghostcaddie-tour && python3 -m ghostcaddie run --shot data/scenarios/lie_dispersion/shot_rough.json --course data/scenarios/lie_dispersion/hole.json --player data/scenarios/lie_dispersion/player.json --out /tmp/ghostcaddie-cli/lie_rough
cd /Users/giofiore/ghostcaddie-tour && python3 -m ghostcaddie run --shot data/scenarios/lie_dispersion/shot_bunker.json --course data/scenarios/lie_dispersion/hole.json --player data/scenarios/lie_dispersion/player.json --out /tmp/ghostcaddie-cli/lie_bunker
cd /Users/giofiore/ghostcaddie-tour && python3 -m ghostcaddie run --shot data/scenarios/wind_adjusted_dispersion/shot.json --course data/scenarios/wind_adjusted_dispersion/hole.json --player data/scenarios/wind_adjusted_dispersion/player.json --out /tmp/ghostcaddie-cli/wind_adjusted_dispersion
```

Expected: every command exits 0 and writes both output files.

- [ ] **Step 4: Report the final implementation facts.** The implementation report must list changed files, test count/result, all CLI commands run, the zero-wind comparison result, wind assumptions, coefficient values, and remaining risks/limitations. Explicitly state that no Git commit was made because `.git` writes are blocked in this environment.
