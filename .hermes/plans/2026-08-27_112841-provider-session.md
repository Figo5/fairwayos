# Provider-Aware Session Envelope and CLI Implementation Plan

> **For Hermes:** Implement task-by-task with strict TDD after user approval. Preserve the accepted 130-test baseline.

**Goal:** Add a secure provider-aware session envelope and CLI that adapts ordered concrete ShotLink or TrackMan `provider_record` payloads into the unchanged session and single-shot analytics pipelines.

**Architecture:** Add a thin ingestion boundary above the existing provider adapters and `parse_session()`/`run_session()` flow. The envelope resolves separate course/player JSON sources relative to the envelope file, permits only safe relative paths inside the project/data boundary, and passes normalized `ShotEvent` records into the existing session pipeline without changing wind, calibration, simulation, hazard, expected-strokes, decision, or rendering code.

**Tech Stack:** Python 3.9 standard library only; `unittest`, `argparse`, `json`, `pathlib`, existing provider adapters and session pipeline.

---

### Task 1: Freeze the secure envelope contract in failing tests

**Files:**
- Create: `tests/test_provider_session.py`
- Inspect only: `ghostcaddie/adapters/shotlink.py`, `ghostcaddie/adapters/trackman.py`, `ghostcaddie/session.py`, `ghostcaddie/cli.py`

**Tests:** Define the concrete envelope shape:
- `schema_version: "provider-session.v1"`;
- `session.provider` is `shotlink` or `trackman`;
- `session.provider_schema_version` matches the adapter;
- each ordered shot contains exactly a concrete `provider_record` payload;
- course/player sources are explicit relative JSON paths.

Verify malformed envelopes, unsupported providers, mismatched provider schema versions, missing sections, missing `provider_record`, and non-dict records fail before analytics.

Run:

```bash
python3 -m unittest tests.test_provider_session -v
```

Expected initial result: focused failures because the provider-session boundary does not exist.

### Task 2: Implement safe source-path resolution

**Files:**
- Create or modify: `ghostcaddie/adapters/provider_session.py`
- Test: `tests/test_provider_session.py`

Implement one path resolver that:
- resolves paths relative to the envelope file directory;
- rejects absolute paths;
- rejects `..` traversal that resolves outside the permitted project/data boundary;
- rejects symlink-resolved paths outside that boundary;
- accepts only existing regular files with the expected JSON source role;
- never stores or emits the resolved filesystem path in report data.

Define the permitted boundary explicitly as the project root containing the envelope, with the normal fixture location under `data/` accepted. Keep the boundary decision centralized and testable.

Run focused path-safety tests, then the full suite.

### Task 3: Implement ordered provider-record dispatch

**Files:**
- Modify: `ghostcaddie/adapters/provider_session.py`
- Test: `tests/test_provider_session.py`

Implement provider dispatch:
- `shotlink` records call `adapt_shotlink()`;
- `trackman` records call `adapt_trackman()`;
- course context is derived from the separately loaded course source and explicit per-shot context;
- player context is loaded from the separately loaded player source;
- no second adapter or coordinate-mapper pass occurs after normalization;
- source record IDs and adapter provenance are retained.

Preserve exact input shot order, validate duplicate IDs and duplicate/non-increasing ordinals, and reject provider identity mismatches.

Run focused dispatch, ordering, provenance, and exact-once tests.

### Task 4: Add strict-default and explicit permissive modes

**Files:**
- Modify: `ghostcaddie/adapters/provider_session.py`
- Modify: `ghostcaddie/cli.py`
- Test: `tests/test_provider_session.py`

Implement strict validation as the programmatic and CLI default. Add one explicit CLI option such as:

```text
--permissive
```

Strict mode must reject unknown nested fields in the envelope, session, source descriptors, shot wrapper, and concrete provider record. Permissive mode must preserve unknown-field paths in diagnostics/provenance without changing normalized analytics input.

Do not expose source paths in the report; use source roles, provider names, and source record IDs instead.

Run strict/permissive focused tests and CLI help validation.

### Task 5: Reuse the existing session pipeline

**Files:**
- Modify minimally: `ghostcaddie/adapters/provider_session.py`
- Test: `tests/test_provider_session.py`

Build an in-memory `SessionInput` from normalized provider shots and invoke the unchanged `run_session()` exactly once for the session. Confirm each shot still uses the existing `run_pipeline()` path and that normalized provider coordinates are not remapped.

Do not modify `ghostcaddie/session.py` or any wind, calibration, simulation, hazard, expected-strokes, decision, or rendering modules unless a narrowly scoped compatibility type is unavoidable and proven by a failing test.

Add tests for:
- ShotLink multi-shot session;
- TrackMan multi-shot session;
- source/player/course separation;
- missing course source;
- missing player source;
- missing provider course context;
- provider provenance in each shot and the session report;
- deterministic repeated session runs for a fixed seed;
- exact input-order preservation.

### Task 6: Add the provider-session CLI

**Files:**
- Modify: `ghostcaddie/cli.py`
- Create: synthetic fixtures under `data/providers/sessions/`
- Test: `tests/test_provider_session.py`

Add a new command without changing existing `run` or `session` behavior, for example:

```bash
python3 -m ghostcaddie provider-session \
  --input data/providers/sessions/shotlink_session.json \
  --out out/provider-session-shotlink/
```

Add `--permissive` as the only mode override. Keep course/player paths inside the envelope and resolve them relative to the envelope file. Write only the normalized session report; never include absolute or relative filesystem source paths in it.

Run both synthetic provider-session CLI fixtures and verify exit code, report schema, shot order, provider diagnostics, and absence of path strings.

### Task 7: Documentation and regression verification

**Files:**
- Modify: `README.md`
- Test-only inspection: all existing suites

Document the provider-session envelope, concrete `provider_record` definition, safe relative path rules, strict default, `--permissive`, source separation, and synthetic-only boundary.

Run:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q ghostcaddie tests
```

Verify:
- all original 119 tests remain present and passing;
- final test count equals 130 plus the new provider-session tests;
- every existing single-shot and existing session CLI scenario still exits 0;
- no changes to wind, calibration, or core analytics files;
- no report contains filesystem paths;
- no packages, APIs, credentials, scraping, or production integrations were added.

### Final quality gate

Independently inspect every changed file and the raw command output. If a test or report discrepancy appears, make only the smallest correction, rerun the affected test, then rerun the complete suite. Do not declare success from an implementation-agent summary alone.

**Open design decision for approval:** The permitted path boundary should be the project root containing the envelope, with source files allowed anywhere beneath it (including `data/`), while absolute paths, traversal, and symlink escapes are rejected. This is the narrowest interpretation that supports envelope-relative fixtures without introducing a new global configuration setting.
