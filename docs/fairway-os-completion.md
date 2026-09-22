# FairwayOS — completion pass (2026-09-21)

This is the summary of the pass that **completed and audited the truthful local
product** and closed the ball-tracker research line. It is deliberately short and
evidence-first: every number here is reproducible from the commands and paths it
names, and every capability it lists was exercised, not asserted.

**Final status (PRIMARY, 2026-09-21).** Claude's independent product audit returned
`APPROVE_WITH_FIXES`, with the fixes applied and **1397 tests passing**. PRIMARY then
verified canonical import, result display, downloads becoming available, and actual
annotated-video playback in Opera GX. Safari's player still loaded indefinitely in
this environment; Safari playback is **not** cleared (§9.8). The finished capability
boundary is conditional 2D BODY support, with BALL, CLUBHEAD and physical metrics
unavailable. No code was committed, pushed or merged during this orchestration.

Related documents:

- [`ball-tracker-v3-results.md`](ball-tracker-v3-results.md) — decisive research
  evidence, correction ledger, seal audit.
- [`current-research-handoff.md`](current-research-handoff.md) — research index and
  safety boundary.
- [`video-analysis-user-guide.md`](video-analysis-user-guide.md) — operator guide.
- [`canonical-video-pipeline-audit.md`](canonical-video-pipeline-audit.md) — the
  audit behind the capability boundary (preserved unchanged).

---

## 1. Classification, settled

| | |
|---|---|
| Ball tracker (v1, v2, v3) | **`FAILED_TO_GENERALIZE`**, on **development evidence** |
| Held-out accuracy claim for v3 | **none exists** — the only real-labelled corpus is development-consumed |
| v3 final test | **SEALED / NOT RUN**, by decision |
| Promotion / canonical ball integration | **none** |
| Research snapshot | **archive provenance, NOT a promotion freeze; no future-promotion promise** |

The v3 architecture recovered **0 of 185** human-labelled ball-visible frames on
`legacy` in **all six** validation variants, including the full model. The final test
was not spent: the tracker never cleared review readiness, so consuming untouched
evidence to confirm an already-negative result is not justified. **No further ball
research loop is planned on this corpus.**

## 2. What the product does

```bash
# canonical local video analysis — one command, one contract, no network
.venv/bin/python3 -m ghostcaddie analyze-video CLIP.mp4 --out OUTPUT_DIR

# where model files are looked for, and which are present (never downloads)
.venv/bin/python3 -m ghostcaddie models

# the shot-analytics engine slice
.venv/bin/python3 -m ghostcaddie run --shot data/sample_shot.json \
    --course data/sample_hole.json --player data/sample_player.json --out out/

# local upload + review service (loopback only)
.venv/bin/python3 -m ghostcaddie.upload.server --root /tmp/fairwayos-uploads --port 8790
```

`analyze-video` writes exactly five outputs —
`annotated.mp4`, `report.md`, `analysis.json` (schema `fairwayos-analysis.v1`, the
source of truth), `provenance.json`, `diagnostics.json` — and a `debug/` directory.

## 3. Capability boundary

| Layer | State | Basis |
|---|---|---|
| **body** | `supported` / `partial` / `ambiguous` / `unavailable`, per clip | MoveNet SinglePose Lightning on an identity-established subject or an explicit operator ROI |
| **ball** | `unavailable`, `method: none` | every route withdrawn or falsified |
| **clubhead** | `unavailable`, `method: none` | blocked at the data layer, not the code layer |
| **physical metrics** | `supported` list **empty by construction** | no validated physical scale, no validated action time |

Physical metrics are declared individually, each in its own unit with its own
reason: `ball_speed` (m/s), `clubhead_speed` (m/s), `launch_angle` (degrees),
`carry_distance` (m), `apex_height` (m) and **`spin_rate` (rpm)**. Spin is listed
explicitly rather than omitted so that its absence cannot read as an oversight or as
something a flag unlocks. Nothing on this path observes impact, and no validated
rotation measurement or spin calibration exists.

**Body is conditional, and that is a limitation, not a hedge.** MoveNet returns one
skeleton with no identity; the workflow proposes a body, challenges it by blanking it
out and re-running, and declines (`ambiguous` / `unavailable`) rather than guess.
Bodies that overlap the subject, or people too small/occluded for MoveNet to resolve,
cannot be challenged this way and would not be found. That sentence is printed on
every run.

## 4. Verified in this pass

Everything below was run against real artifacts, not described.

| Check | Result |
|---|---|
| Canonical CLI on a rights-cleared real clip (`pexels_37080178`, sha256 `69e6763c…8c9ade`, 1080x1920, 30 fps, 15.233 s) | five outputs written; body `SUPPORTED` 92/92 sampled frames; ball `UNAVAILABLE`/`none`; clubhead `UNAVAILABLE`; physical `supported` empty; 6 of 6 2D metrics supported; ~7.1 s wall |
| Source binding in the output | `source.sha256` equals the annotation manifest's `source_sha256`; `resolution 1080x1920`; `frame_range [0, 455]`; `sampled_frame_count 92` at `sample_fps 6` |
| Annotated MP4 | ffprobe: 1080x1920, 6/1 fps, **92** frames, 15.333 s; byte-identical across two independent runs (`5bff558e…c0c1`); renderer draws keypoints and a label only — **no ball, no trail, no stale placeholder** |
| Declining cleanly | missing source → exit 1; wrong `--expected-sha256` → exit 1, fail closed; `--max-frames 0` → exit 1; malformed `--roi` → exit 1; explicit bad `--body-model` → body `UNAVAILABLE` (0/19), **not** a forced identity |
| Model discovery | `python -m ghostcaddie models`: body present; SAM2 / BootsTAPIR absent and reported as an expected `model_absent` state. **Nothing downloads.** |
| Upload API | CSRF-less POST → `403`; canonical upload → `202` then `done`; unknown analysis mode → `400` refused; `fresh_ai` without an interval → `400` refused |
| Upload artifacts | `/result-video` → `200 video/mp4` **10 963 070 B**, byte-identical to the CLI render; `/video` → `200`; `/frame?n=55` → `200` with `X-Source-Sha256` matching the bound source |
| Retention / delete | media retained **3600 s** with an explicit `expires_at`; `media/delete` → `retained: false`, `bytes_deleted: true`; afterwards `/video`, `/frame` and `/result-video` → **`410`** while the job result document and the on-disk artifacts remain — **no silent loss on refresh** |
| Offline posture | the canonical orchestrator is stdlib-only, opens no socket **under test instrumentation of the orchestrator process**, imports no network client, strips proxy variables before launching the worker and launches it with `FAIRWAYOS_OFFLINE=1`; no canonical code path reaches `fresh_ai` or a research tracker. This is code-level and test-level evidence, not a kernel-enforced sandbox — the worker runs as a separate process and is not network-isolated by the OS |
| Media serving (added in the final audit) | `/video` and `/result-video` answer a single HTTP byte range: `206` with `Content-Range` and the exact slice, `200` for a plain or unparsable request, `416 Content-Range: bytes */N` for an unsatisfiable one, and retention/job checks still run **before** any byte is served |

### UI mapping defect found and fixed (narrow, demonstrated)

The served review page read the research result shape (`targets`) unconditionally, so
on a **successful canonical job** it threw
`TypeError: Cannot convert undefined or null to object` into the error line, and
`poll()` aborted before it populated the raw result, enabled *Download results*, or
pointed the rendered playback at `/result-video`. The result video was also only
looked for at `outputs.video`, while canonical results expose
`outputs.annotated_video`. Both were fixed in `ghostcaddie/upload/ui.py` and pinned
by `tests/test_upload_ui_canonical_result_mapping.py`, which runs the **shipped**
page JavaScript under node. Before the fix the check reports the TypeError and an
empty raw panel; after it, no error, the raw panel populated, the download enabled
and the playback pointed at `/result-video?job=…`.

### Two served-copy defects found by the primary's browser QA, and fixed

The primary's independent Safari run (own isolated server on port 8791, own tab)
confirmed the mapping fix above in a real browser, and found two statements that were
**false or misleading for a mode the page serves**:

1. The page header said, for the whole page, *"Video stays on this machine. Nothing
   is uploaded anywhere… Results are pseudo-labels, not ground truth."* That is false
   for `fresh_ai`, which sends selected frames to an external AI service, and the
   pseudo-label wording misdescribes a canonical body report, which is a
   **measurement**, not a pseudo-label. The page title also framed the whole tool as
   research.
2. The readiness panel was headed *"runtime readiness"* and listed the research
   targets (ball, clubhead, SAM2) alongside the body worker, which invited the
   reading that placing a model file or a reviewed seed could enable canonical ball
   or clubhead.

Both were corrected in the served copy: canonical and research claims are now stated
separately, the fresh-AI mode explicitly says it sends selected frames externally,
the title no longer calls the tool research, and the panel is headed *"research
dependency readiness"* with the sentence that a present model or seed does **not**
enable ball or clubhead in canonical mode. Two tests in
`tests/test_upload_ui_endpoints.py` pin the corrected copy against the served page.

## 5. Retention, exactly

- Uploaded media is retained for **3600 s** from upload
  (`ghostcaddie/upload/jobs.py::DEFAULT_MEDIA_RETENTION_SECONDS`). The deadline is
  re-checked on **every** access, so waiting for something cannot keep media alive
  past it, and `media_retention` is reported as a record (`retained`,
  `retention_seconds`, `expires_at`, `reason`) — never a bare boolean.
- Deleting media deletes the **uploaded copy** inside the job's own directory. A file
  the operator pointed at in the import directory is theirs: access is revoked, the
  file is never deleted.
- The job result document is **not** media and survives; nothing is silently dropped
  and no error is swallowed.

## 6. Offline and security

- Canonical mode makes **no network calls by construction** and downloads **no
  models**: the orchestrator imports no network client, opens no socket under test,
  strips proxy variables and sets `FAIRWAYOS_OFFLINE=1` for the worker. That is a
  design-and-test guarantee, not an OS-enforced one. Model files are placed by the
  operator; the search path and the presence of each file are printed by
  `python -m ghostcaddie models`.
- The serving process never imports torch: body runs in a separate interpreter with
  `ai_edge_litert`, `cv2` and `numpy`. A TFLite flatbuffer has no pickle surface.
- The upload service binds `127.0.0.1` only, requires a per-launch CSRF token on
  state-changing requests, and confines uploads to its own root.
- `fresh_ai` is the only mode that can touch the network. It is explicitly opt-in,
  separately labelled in the UI, requires a native frame interval, and the canonical
  path never falls back to it. Its existing unit tests are mocked-only and no
  external request was made in this pass.

## 7. Test results

Targeted suites run while fixing, then one genuine full run:

```
.venv-video-ai/bin/python3 -m pytest tests/test_canonical_pipeline.py \
    tests/test_ball_research_isolation.py \
    tests/test_upload_ui_canonical_result_mapping.py \
    tests/test_upload_ui_click_mapping.py -q
  → 74 passed

.venv-video-ai/bin/python3 -m pytest tests/test_upload_ui_canonical_result_mapping.py -q
  → 7 passed

.venv-video-ai/bin/python3 -m pytest tests -q          # full suite, frozen tree
  → 1383 passed, 2 skipped, 2 warnings, 195 subtests passed in 254.80s (0:04:14)
    PYTEST_EXIT=0
```

After the primary's visual QA, two more served-copy defects were corrected (see §4)
and two tests were added for them, so the suite was run a second time on the frozen
tree; the run before those two tests reported 1381 passed.

Before this pass the same command reported 1373 passed, 2 skipped, 195 subtests.
The delta is exactly the 10 tests added here (7 canonical-result UI mapping tests,
1 explicit `spin_rate` contract test and 2 served-copy tests); nothing was removed or
skipped. The two skips are pre-existing and unrelated. The two warnings are the `cgi`
deprecation in `ghostcaddie/upload/server.py` and a test returning a value; neither is
introduced here and neither blocks anything.

### Final audit run (2026-09-21, after the byte-range fix)

The independent final audit added byte-range serving (§9.8) and re-ran the whole
suite once on the resulting tree:

```
.venv-video-ai/bin/python3 -m pytest tests -q
  → 1397 passed, 2 skipped, 2 warnings, 240 subtests passed in 269.99s (0:04:29)
    PYTEST_EXIT=0
```

Delta over the completion pass: **+14 tests and +45 subtests**, all in the new
`tests/test_upload_media_range.py`. Nothing was removed, nothing newly skipped, and
the two skips and two warnings are the same pre-existing ones. Focused re-runs
during the audit: 147 passed across `test_canonical_pipeline`,
`test_upload_ui_endpoints`, `test_upload_ui_canonical_result_mapping`,
`test_upload_ui_click_mapping`, `test_upload_server`, `test_upload_security`,
`test_upload_product_gaps` and `test_ball_research_isolation`. Logs:
`review-4/full-suite.log`, `review-4/focused-suites.log`, `review-4/range-suite.log`.

## 8. QA evidence (this pass)

The CLI/UI QA stores live **outside Git**:

```
/private/tmp/fairway-v3-orchestration.jQ7Tk6/artifacts/product-qa/
  cli-run-final/                 five CLI outputs + debug/identity_roi.png
  cli-run-pexels_37080178/       the pre-spin-contract run, kept for comparison
  frames/                        annotated_start_f000.png, annotated_mid_f045.png,
                                 annotated_end_f091.png
  smoke-badmodel/ smoke-missing/ smoke-badsha/ smoke-unbounded/ smoke-badroi/
  job-A.json  job-post-A.json    the kept canonical upload job (media retained)
  job-B*.json  jobB-media-delete.json   the retention/delete exercise
  jobA-result-video.mp4  jobA-source-frame55.png  jobA-frame-headers.txt
  upload-smoke*.log  ui-index*.html  ui-index-served-final.html
  ui-mapping-before-fix.txt  ui-mapping-after-fix.txt
  playback-diagnostic.json       read-only MP4 atom order + endpoint headers
  scripts/                       the exact smoke, mapping-check, playback-diagnostic
                                 and archive scripts used
```

The durable copy of the research archive and of this QA evidence is under
`/Users/giofiore/fairway-os-research/v3-2026-09-21/` — a **copy-only** preservation of
the orchestration evidence: 294 files, ~53 MB, with `MANIFEST.json` +
`MANIFEST.sha256` (every copied file, source and archived SHA-256), `EXCLUSIONS.json`
(the 8 documented exclusions, the repo-side stores left in place, and the 2 documented
redactions) and `PATH-INDEX.md` explaining the original `/tmp` paths that older
documents cite. Verified independently by `verify_archive.py`: 294/294 hashes match,
every excluded path is still present in the original tree, and the repo-side ranker
weights, training manifest and split manifest still hash to their recorded values.
Nothing was moved or deleted anywhere.

**Re-verified by the final audit**, re-hashing all 294 files from scratch: 0 missing,
0 mismatched, every excluded path still present in the original tree
(`review-4/archive-verify.json`). Two findings, recorded in
`review-4/archive-findings.md` and **not** silently remediated:

1. The manifest's `secret_scan_hits: []` is **overstated**. A regex sweep is a
   bounded check, never proof of secret-freeness, and this one missed a match: the
   archived `artifacts/product-qa/ui-index-served-final.html` still carries a
   per-launch loopback CSRF token value, while the two other served-page snapshots
   *were* redacted. Risk is low — the token is ephemeral, scoped to one loopback
   process, and dies when that QA server is stopped — but the redaction policy was
   applied inconsistently and the manifest claim is wrong.
2. The excluded 493 MB `artifacts/tracklets/shards` store is **generated training
   data, not a runtime cache**: the retained derived `tracklets.json` does not
   reconstruct the exact candidate lists the ranker consumed.

Finding 2 is addressed additively at
`/Users/giofiore/fairway-os-research/v3-2026-09-21-final-audit-supplement/` — the
shard store at its original relative path, the final-audit repo files and the audit's
test evidence, each with its own SHA-256 in that supplement's `MANIFEST.json`. The
prior snapshot and both of its manifests are **unmodified**.

PRIMARY closed finding 1's live-token exposure by stopping its owning QA server
(PID 29202), along with the other owned QA servers. Historical archived bytes and
hashes remain intact, with the finding attached; the archive is local and unpublished.
The user's unrelated server and original browser tabs were preserved.

The final handoff at
`/Users/giofiore/fairway-os-research/v3-2026-09-21-primary-handoff/` adds the final
review report, PRIMARY's native-browser observations, this final document, and a
hash-bound snapshot of the complete changed/untracked project files. Its manifest
links the two earlier archives without rewriting them. This is preservation, not a
promotion freeze.

## 9. Known remaining limitations

1. **Body identity is not a person detector.** The challenge step is built on a
   single-pose model; overlapping bodies and unresolvable people are not detected.
   State is `ambiguous`/`unavailable` when identity cannot be established.
2. **All physical metrics are unavailable**, including spin — no calibration exists,
   and none is planned on this path. Container FPS is never accepted as action time.
3. **No ball and no clubhead**, and no configuration turns either on.
4. **Body metrics are 2D image-space angles** from one uncalibrated camera. They
   change with camera position and are not clinical or coaching measurements.
5. **The v3 research line is closed without a held-out number.** Anyone reading
   `ball-tracker-v3-results.md` must keep its evidence classes apart: `legacy` is
   development-consumed, `v3-validation` is unlabelled (behaviour only), and the
   separability numbers come from synthetic candidate lists with known ranker
   optimism (72.8 % of family-split rows).
6. **The upload service is a single-machine review tool**, not a multi-user service,
   and it is not part of the canonical CLI path.
7. **The sealed final test stays sealed.** It is the one piece of untouched evidence
   in the repository; consuming it requires a model that has already cleared review
   readiness, which v3 did not.
8. **Browser playback is verified in Opera GX, not Safari.** The audit added standard
   single-byte-range serving on both media endpoints, with retention enforced before
   serving. PRIMARY restarted an isolated server with that code and independently
   confirmed `Range: bytes=0-1` returns `206`, exactly two bytes and the correct
   `Content-Range`. This fixes the HTTP contract; it did not establish the cause of
   Safari's black/loading player.

   Native-browser retests used only the rights-cleared development clip
   `pexels_37080178`. Safari canonical import job `3683aad4cc4c480f` completed with
   correct states and download availability, but both embedded and direct video
   playback still showed indefinite loading. In Opera GX the same served annotated
   video visibly advanced through changing frames to **0:15**, with the body overlay
   and unavailable BALL/CLUBHEAD banner. A second canonical import through Opera,
   job `fce7bf31e7664920`, completed and its **embedded** source and annotated videos
   rendered; annotated playback advanced after Play. Thus the upload workflow and
   browser rendering have a real positive check, while Safari remains an explicit
   compatibility limitation. No additional speculative encoding changes were made.

   The complete observations are in the final handoff's
   `evidence/primary-product-visual-qa.md`. The three representative decoded annotated
   frames were also visually inspected. No research mode or external AI was invoked
   during these UI checks.
