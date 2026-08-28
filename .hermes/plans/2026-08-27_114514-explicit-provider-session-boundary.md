# Explicit Provider Session Boundary Plan

> **For Hermes:** Implement task-by-task with strict TDD after user approval. Preserve the accepted 136-test baseline.

**Goal:** Replace provider-session’s inferred package-ancestor boundary with an explicit caller-supplied project boundary, making source resolution auditable and portable without adding live integrations or changing analytics.

**User value:** A provider-session envelope cannot silently widen its trust boundary based on directory layout. Operators and tests can state exactly which project root is permitted, while reports remain free of filesystem paths and existing fixtures continue to run unchanged when invoked from the project root.

**Architecture:** Keep `provider-session.v1` and the concrete `provider_record` contract unchanged. Refactor only the provider-session ingestion boundary so path validation receives an explicit `boundary_root`; the CLI exposes an explicit `--project-root` option with a documented default of the current working directory, and the programmatic API requires or clearly resolves the same boundary once at its public entry point. `run_provider_session()` continues to call unchanged `run_session()`, which continues to call unchanged `run_pipeline()`.

**Tech Stack:** Python 3.9 standard library only; `pathlib`, `json`, `argparse`, `unittest`.

---

### Task 1: Define the explicit boundary API with failing tests

**Files:**
- Modify: `tests/test_provider_session.py`
- Inspect: `ghostcaddie/adapters/provider_session.py`, `ghostcaddie/cli.py`, `README.md`

Add tests establishing:
- source resolution accepts a path beneath an explicitly supplied project root;
- source resolution rejects a path outside that root even when the package ancestor would have accepted it;
- the programmatic loader/runner receives the boundary explicitly;
- the CLI exposes `--project-root` and documents its current-working-directory default;
- reports still contain no boundary or source filesystem path.

Run:

```bash
python3 -m unittest tests.test_provider_session -v
```

Expected: new API/CLI assertions fail before implementation.

### Task 2: Replace package-ancestor inference with explicit boundary validation

**Files:**
- Modify: `ghostcaddie/adapters/provider_session.py`
- Test: `tests/test_provider_session.py`

Implement a single path-validation helper with signature equivalent to:

```python
_safe_source(envelope_path, source_descriptor, role, boundary_root)
```

Requirements:
- normalize and resolve `boundary_root` once;
- reject a relative path whose resolved target escapes `boundary_root`;
- reject absolute source paths;
- reject traversal and symlink escapes;
- require an existing regular file;
- do not infer the boundary by searching for a `ghostcaddie` package ancestor;
- never return or serialize the path into a report—use only to load the separate course/player source.

Keep the provider envelope schema unchanged unless a schema-versioned boundary field is required; prefer a CLI/API argument to avoid putting machine-local paths into portable envelopes.

Run focused path tests and confirm the existing fixtures still resolve under the explicit project root.

### Task 3: Wire the CLI and public loader entry points

**Files:**
- Modify: `ghostcaddie/cli.py`
- Modify: `ghostcaddie/adapters/provider_session.py`
- Test: `tests/test_provider_session.py`

Add:

```text
--project-root PATH
```

for `provider-session`, defaulting to the process current working directory. Pass it explicitly to the loader/runner. The default must be documented as a CLI convenience, not package-ancestor discovery. Preserve `--permissive`, `--seed`, `--samples`, output shape, and existing `run`/`session` commands.

Add tests invoking the CLI from both the project root and a different working directory with `--project-root` supplied. Confirm relative paths remain relative to the envelope file, while trust validation is against the explicit boundary.

### Task 4: Regression, security, and report checks

**Files:**
- Modify: `README.md`
- Test: `tests/test_provider_session.py`

Document:
- envelope-relative source paths;
- explicit project boundary argument;
- CLI default of current working directory;
- rejection of absolute paths, `..` traversal, and symlink escapes;
- no source paths in reports;
- JSON-only course/player descriptors remain the current limitation.

Add/retain tests for:
- absolute path rejection;
- traversal rejection;
- symlink escape rejection;
- valid nested source path acceptance;
- strict and permissive nested unknown-field behavior;
- provider provenance and source record IDs;
- input ordering;
- exactly-once normalization/mapping;
- deterministic repeated reports;
- both ShotLink and TrackMan fixtures.

Run:

```bash
python3 -m unittest tests.test_provider_session -v
python3 -m unittest discover -s tests -v
python3 -m compileall -q ghostcaddie tests
```

Then run both provider-session fixtures and every existing single-shot/session CLI scenario. Verify the existing 136 tests remain intact and the final count is 136 plus only the new boundary tests.

### Acceptance criteria

- No package-ancestor boundary inference remains.
- The boundary is explicit in the programmatic and CLI path.
- Existing provider-session fixtures pass from the project root.
- A provider-session run from another working directory passes when given `--project-root`.
- Absolute paths, traversal, and symlink escapes fail before analytics.
- No session report contains filesystem paths.
- Existing `run_session()` and `run_pipeline()` code and analytics outputs remain unchanged.
- No wind, calibration, provider schema, live API, credential, scraping, or production integration changes.
- Full tests, compileall, all existing CLI scenarios, both provider fixtures, and independent raw-report checks pass.

**Why this slice first:** It closes the only security boundary that currently depends on repository layout. Multi-provider envelopes and alternate course/player source types can then build on an explicit, testable trust boundary rather than carrying forward inferred-path behavior.

**Deferred slices:** supporting more than one provider per envelope, non-JSON course/player descriptors, and any real external data integration remain separate follow-up work and are not included here.
