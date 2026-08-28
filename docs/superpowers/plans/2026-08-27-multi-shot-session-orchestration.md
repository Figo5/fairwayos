# Multi-Shot / Multi-Hole Session Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a thin, validated session layer that parses one versioned inline envelope, runs every normalized shot through the unchanged `run_pipeline()`, and emits deterministic per-shot, per-hole, and session summaries.

**Architecture:** Add reusable pure JSON-to-domain helpers at the adapter boundary, then build a normalized in-memory session object containing one shared player, hole-number-indexed courses, and ordered engine-coordinate shots. `run_session()` will create protocol-compatible in-memory sources for each shot, derive a stable SHA-256 seed per shot, invoke the existing pipeline without duplicating analytics, and aggregate only local recommendation metadata. The CLI will serialize a finite report while the existing `run` path remains unchanged.

**Tech Stack:** Python 3.9 standard library only (`dataclasses`, `json`, `hashlib`, `math`, `argparse`, `pathlib`, `unittest`).

## Global Constraints

- Keep `run_pipeline(shot_source, course_source, player_source, config: Config)` and all single-shot behavior unchanged.
- The public input is a mandatory `schema_version: "0.1"` self-contained JSON envelope with top-level `session`, `player_profile`, `course`, and ordered inline `shots`.
- IDs are non-empty strings; `round_number`, `hole_number`, and `shot_number` are positive finite numeric values and normalize to integers only when integral.
- Reject malformed, inconsistent, duplicate, out-of-order, non-finite, missing-reference, and unknown-reference data before any analytics call.
- Use the exact validation wording `duplicate (hole_number, shot_number) pairs` for duplicate shot ordinals.
- The session parser is the sole inline-ingestion boundary and applies each hole `CoordinateMapper` exactly once; normalized shot positions passed to `run_pipeline()` are engine coordinates.
- Use protocol-compatible in-memory sources, never temporary files or public filesystem paths.
- Derive per-shot seeds with `hashlib.sha256` from session seed, stable zero-based ordinal, and `shot_id`; never use built-in `hash()`; preserve single-shot seed behavior.
- Aggregate only `Recommendation.decision_cost` and existing rounded recommendation hazard probabilities; do not add aggregate expected-strokes metrics.
- Label `sum_local_decision_cost` as local/non-additive and not official Strokes Gained or official round strokes.
- Validate finite input and output recursively and serialize with `json.dumps(..., allow_nan=False)`.
- Do not modify completed wind/calibration logic or install dependencies.

---

### Task 1: Factor reusable JSON parsing and strict scalar validation

**Files:**
- Modify: `ghostcaddie/adapters/json_file.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Produce pure helpers for parsing existing domain records from dictionaries, such as `_parse_point`, `_parse_course`, `_parse_shot`, and `_parse_player` (names may follow the existing module idiom).
- Preserve `JsonCourseDataSource`, `JsonShotDataSource`, and `JsonPlayerProfileSource` public behavior and path-based coordinate mapping.
- Helpers must accept an explicit `CoordinateMapper`/`CoordinateSystem` boundary so session parsing can map each inline position once, while existing `JsonShotDataSource` still maps its raw file positions once.

- [ ] **Step 1: Add failing tests** for helper equivalence with current sample adapters, missing required fields, wrong container types, non-empty string IDs, positive integral ordinals, and recursive finite-value rejection.
- [ ] **Step 2: Run the focused tests and confirm the new tests fail before implementation.**
- [ ] **Step 3: Extract the existing player/course/shot construction into pure helpers.** Validate dictionaries and required fields before `float`/`int` conversion; reject booleans, non-finite numbers, non-integral ordinal values, and malformed nested polygons/wind rather than allowing incidental `KeyError`/`TypeError` leaks.
- [ ] **Step 4: Keep the existing adapter classes as thin file-loading wrappers around those helpers and retain their existing one-time `CoordinateMapper.to_engine()` calls.**
- [ ] **Step 5: Run the focused adapter/regression tests and confirm the existing sample records still normalize identically.**

### Task 2: Add normalized session input model and in-memory protocol sources

**Files:**
- Create: `ghostcaddie/session.py` (model/source definitions and parser boundary, or split only if the implementation keeps each responsibility clear)
- Modify: `ghostcaddie/adapters/base.py` only if a type import/documentation seam is needed
- Test: `tests/test_session.py`

**Interfaces:**
- `SESSION_SCHEMA_VERSION = "0.1"`.
- `SessionInput` stores the validated session metadata, shared `PlayerProfile`, hole-number-indexed `CourseModel` values, ordered normalized shot records, and any optional envelope metadata needed for provenance.
- A normalized shot record stores `shot_id`, `hole_number`, `shot_number`, and the parsed `ShotEvent` whose three positions are already engine coordinates.
- `parse_session(raw: dict) -> SessionInput` parses/validates the complete envelope exactly once and performs no analytics.
- In-memory sources implement `load_shot() -> ShotEvent`, `load_course() -> CourseModel`, and `load_player() -> PlayerProfile`, returning the supplied normalized objects and exposing descriptive non-path identifiers for provenance.

- [ ] **Step 1: Add failing tests** for the approved envelope shape, shared top-level player/course parsing, hole lookup, normalized shot identity, protocol compatibility, and exact-once mapping on a four-point course.
- [ ] **Step 2: Run those tests to verify failure.**
- [ ] **Step 3: Implement strict envelope validation before constructing any session analytics object:** require exact schema version and sections; require non-empty `session_id`, `tournament_id`, `player_id`, and `course_id`; require positive finite `round_number`; require a non-empty holes collection and inline shots; require unique hole numbers and shot IDs; reject duplicate `(hole_number, shot_number) pairs`; reject any input not already strictly ordered by `(hole_number, shot_number)`; reject missing course holes, unknown shot holes, and identity mismatches. Validate optional per-shot `player_id`, `tournament_id`, and `course_id` when present, without requiring fields absent from `ShotEvent`.
- [ ] **Step 4: Parse each hole’s existing `CourseModel` fields once, construct a mapper for that hole, and pass it explicitly to the reusable shot parser.** Map `start_position`, `target_position`, and `actual_landing_position` exactly once; never map normalized `Point2D` values again in an in-memory source.
- [ ] **Step 5: Run tests that inspect normalized coordinates and assert the in-memory sources satisfy the existing Protocols.**

### Task 3: Implement `run_session()` and report aggregation

**Files:**
- Modify: `ghostcaddie/session.py`
- Test: `tests/test_session.py`

**Interfaces:**
- `derive_shot_seed(session_seed: int, ordinal: int, shot_id: str) -> int` hashes a UTF-8 canonical string with SHA-256 and converts digest bytes to a deterministic integer.
- `run_session(session: SessionInput, config: Config) -> SessionReport` invokes unchanged `run_pipeline()` once per ordered shot with a per-shot config whose `simulation.random_seed` is the derived seed.
- `SessionReport` is a JSON-ready dictionary or dataclass converted to the exact report shape: `schema_version`, `session`, `summary`, `holes`, `shot_results`, `provenance`.
- `serialize_session_report(report: object) -> str` recursively rejects non-finite numeric values and calls `json.dumps(report, allow_nan=False, indent=2)`.

- [ ] **Step 1: Add failing tests** proving one pipeline invocation per shot, stable per-shot seeds, fixed-seed repeated-run equivalence for analytics fields, input-order preservation, no aggregate expected-strokes field, and local decision-cost aggregation.
- [ ] **Step 2: Run the tests and verify failure.**
- [ ] **Step 3: Implement seed derivation with `hashlib.sha256(f"{session_seed}:{ordinal}:{shot_id}".encode("utf-8")).digest()` and a stable integer conversion; use `dataclasses.replace` to change only the per-shot simulation seed.**
- [ ] **Step 4: For each shot, select its already-parsed hole, instantiate in-memory shot/course/player sources, call unchanged `run_pipeline()`, and retain the recommendation plus safe provenance.** Do not expose fake paths; identify inline sources as session data and preserve envelope/session identifiers and pipeline provenance.
- [ ] **Step 5: Build exact report fields.** Keep ordered `shot_results` entries with `shot_id`, `hole_number`, `shot_number`, `recommendation`, and `provenance`; group ordered hole entries with `hole_number`, `shot_count`, `shot_ids`, `sum_local_decision_cost`, `hazard_risk_summary`, and `recommendations`; build the session summary with `shot_count`, `hole_count`, `sum_local_decision_cost`, `decision_cost_semantics`, `highest_cost_decisions`, and `hazard_risk_summary`. Sum only local `decision_cost`; aggregate only existing rounded hazard probabilities, treating omitted hazards as zero and reporting maximum/mean/nonzero-shot counts (or an equivalent explicitly documented shape).
- [ ] **Step 6: Add a recursive finite-value walker covering dict keys/values, sequences, dataclasses converted to values, and numeric leaves; invoke it before `allow_nan=False` serialization.** Ensure `expected_strokes` and `actual_expected_strokes` remain only inside per-shot recommendations and are never summed or emitted as session/hole aggregate metrics.
- [ ] **Step 7: Run focused aggregation, provenance, serialization, and determinism tests.** Account for `run_pipeline()`’s `generated_at` timestamp by comparing deterministic report fields separately from timestamps.

### Task 4: Add fixture and session CLI without changing `run`

**Files:**
- Create: `data/sample_session.json`
- Modify: `ghostcaddie/cli.py`
- Test: `tests/test_session.py`
- Test: `tests/test_pipeline_end_to_end.py` only if a small non-regression assertion is needed

**Interfaces:**
- Add `session --input PATH --out PATH [--seed INT] [--samples INT]` to the existing parser.
- The command loads JSON once, calls `parse_session`, applies optional config overrides, calls `run_session`, writes `session_report.json`, and prints a concise session summary. The existing `run` command continues to write `recommendation.json` and `overlay.svg` through its current path.

- [ ] **Step 1: Add failing subprocess tests** for `data/sample_session.json`, session output files, exact report keys, at least two holes and multiple shots per hole, and successful existing `run` behavior.
- [ ] **Step 2: Run the CLI tests and verify failure.**
- [ ] **Step 3: Create a self-contained synthetic fixture using existing player/course/shot field semantics, with at least two hole declarations and multiple strictly ordered shots per hole; include no filesystem paths.**
- [ ] **Step 4: Add only the `session` branch and output writer to `cli.py`; use `json.loads` once and `serialize_session_report`, preserving the existing `run` branch unchanged in behavior and filenames.**
- [ ] **Step 5: Run the session subprocess test and inspect parsed output.**

### Task 5: Document the public contract and limitations

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the exact versioned envelope schema and a `session` CLI example.**
- [ ] **Step 2: Document mandatory IDs, optional identity fields, positive finite numeric fields, strict pre-analytics ordering, duplicate terminology, hole references, and malformed/non-finite rejection.**
- [ ] **Step 3: Document that top-level player/course data is reused, inline positions are mapped exactly once into engine coordinates, protocol-compatible in-memory sources call unchanged `run_pipeline()` per shot, and seeds use SHA-256 over session seed/order/shot ID.**
- [ ] **Step 4: Document report field names, per-shot/per-hole provenance, rounded reported hazard aggregation, `sum_local_decision_cost` as local/non-additive and not official Strokes Gained or round strokes, and the absence of aggregate expected-strokes estimates.**
- [ ] **Step 5: State synthetic-data and standard-library limitations without adding future integrations or dependencies.**

### Task 6: Verification gate and regression checks

**Files:**
- Modify: `tests/test_session.py` if any verification regression test is needed

- [ ] **Step 1: Run focused session tests from the project directory:**

```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest tests.test_session -v
```

- [ ] **Step 2: Run the full suite and require every existing test plus the new tests to pass:**

```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m unittest discover -s tests -v
```

- [ ] **Step 3: Run all seven existing single-shot CLI scenarios and verify each output directory contains both `recommendation.json` and `overlay.svg`; do not alter their fixtures or output contract.**
- [ ] **Step 4: Run the new command and inspect the raw report independently:**

```bash
cd /Users/giofiore/ghostcaddie-tour && python3 -m ghostcaddie session --input data/sample_session.json --out "$TMPDIR/ghostcaddie-session"
```

  Check exact top-level/report keys, counts, ordered IDs, local-cost sums, hazard summary values, no aggregate expected-strokes fields, provenance without fabricated filesystem paths, and absence of `NaN`/`Infinity`.
- [ ] **Step 5: Run two fixed-seed sessions and compare deterministic analytics/aggregation/report fields while excluding or explicitly allowing `generated_at` timestamp differences.**
- [ ] **Step 6: Run an independent four-point session probe that checks source positions → normalized engine positions → in-memory source → `run_pipeline()` and proves no second mapper is applied.**
- [ ] **Step 7: Run `python3 -m compileall ghostcaddie tests`, inspect the final changed-file list, and confirm no wind or calibration implementation/fixture files were modified.**
