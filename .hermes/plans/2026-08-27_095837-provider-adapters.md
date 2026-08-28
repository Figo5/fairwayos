# Provider-Neutral Tour Data Adapter Layer Implementation Plan

> **For Hermes:** Implement task-by-task with strict TDD and preserve the accepted 119-test baseline.

**Goal:** Add synthetic, provider-neutral ShotLink-shaped and TrackMan-shaped shot adapters that normalize into the existing `ShotEvent` boundary and work with both `run` and `session` pipelines.

**Architecture:** Keep course geometry and player profiles as separate existing sources. Each provider adapter validates an explicit provider/schema envelope, preserves source record IDs and diagnostics in provenance, accepts unknown fields by default with an opt-in strict mode, and maps provider-native coordinates exactly once before producing existing domain records. No live APIs or dependencies.

**Tech Stack:** Python 3.9 standard library, `unittest`, existing `ghostcaddie` dataclasses/adapters/pipelines.

---

### Task 1: Establish provider adapter contracts with failing tests

- Add `tests/test_provider_adapters.py` covering provider/schema metadata, required fields/types, permissive and strict unknown fields, units/conventions, and source record IDs.
- Run focused tests with `python3 -m unittest tests.test_provider_adapters -v`; confirm expected failures because adapters do not exist.

### Task 2: Implement normalized provider envelope and diagnostics

- Create `ghostcaddie/adapters/provider.py` with shared strict validation, explicit provider/schema version, unknown-field diagnostics, provenance, and normalized-shot result types.
- Keep validation standard-library-only and reuse existing parsing/domain constructors where possible.
- Run focused tests, then the full existing suite.

### Task 3: Implement ShotLink-shaped adapter

- Add `ghostcaddie/adapters/shotlink.py` for absolute GPS-style latitude/longitude payloads plus explicit local geospatial frame/origin.
- Normalize start, aim, and landing coordinates exactly once to engine coordinates; preserve units, coordinate convention, provider, schema, and source record ID.
- Add synthetic fixture under `data/providers/shotlink/` and tests for GPS normalization and exact-once behavior.

### Task 4: Implement TrackMan-shaped adapter

- Add `ghostcaddie/adapters/trackman.py` for carry and signed side-offset metrics, reconstructing landing from separate course-context start and aim positions.
- Validate units and signed side-offset convention; preserve provider provenance and source record ID.
- Add synthetic fixture under `data/providers/trackman/` and tests for reconstruction and missing course/player context.

### Task 5: Wire both adapters to run and session pipelines

- Add adapter source classes/functions implementing existing `ShotDataSource` protocol and session-envelope conversion without changing analytics semantics.
- Add tests proving single-shot `run_pipeline` compatibility, multi-shot `run_session` compatibility, identity/provenance preservation, and no double coordinate mapping.
- Keep course and player loading separate.

### Task 6: CLI scenarios and documentation

- Add minimal CLI options or provider fixture entry points only if needed to exercise both adapters without disturbing existing `run`/`session` commands.
- Update README with provider adapter contracts and synthetic-only limitations.
- Run every existing single-shot and session CLI scenario and capture exit codes/output.

### Final quality gate

- Independently inspect every raw changed file and `git diff` if repository metadata becomes available; do not retry blocked Git operations.
- Run `python3 -m unittest discover -s tests -v` and verify no regressions from 119 baseline tests plus new tests.
- Run all existing README `run` commands and `python3 -m ghostcaddie session --input data/sample_session.json --out <temporary-output>`.
- Confirm no packages were installed, no secrets were printed, and no live integrations were added.

**Assumptions:** The project directory is not currently a Git worktree; filesystem diff inspection will be used if Git remains unavailable. The existing JSON/domain parsing helpers are the canonical normalization boundary. Provider payload schemas will be concise synthetic contracts rather than claims of official vendor schemas.

**Risks:** GPS-to-local conversion must be explicit and deterministic; TrackMan side-offset sign mistakes could silently mirror targets; accidental second mapping would corrupt four-point/manual coordinates; provider adapters must not make course/player context implicit.
