# GolfDB Labeled Split and SwingNet Evaluation Plan

> **For Hermes:** Execute this plan only after dataset and checkpoint licensing are explicitly cleared; keep SwingNet research-only otherwise.

**Goal:** Obtain a reproducible, legally cleared GolfDB labeled validation split and measure the official SwingNet model against its event annotations without opening GhostCaddie’s automatic-perception, phase, impact, analytics, or recommendation gates.

**Architecture:** Keep the official GolfDB/SwingNet implementation and weights isolated under `.venv-video-ai` and the evaluation artifacts under `out/golfdb_evaluation/`. Treat GolfDB event labels as an evaluation vocabulary only. Produce a separate evaluation report and a provenance-preserving adapter output; do not modify `run_pipeline()`, core analytics, human fallback, or existing CLI behavior.

**Tech Stack:** Python 3.11 in `.venv-video-ai`, official GolfDB PyTorch implementation, PyTorch/TorchVision, OpenCV, standard-library hashing/JSON, `python3 -m unittest`, `compileall`.

---

## Current state and non-goals

- The official SwingNet architecture loads locally on MPS.
- The official checkpoint was downloaded from the link in the GolfDB README.
- Checkpoint SHA-256 currently recorded as:

  ```text
  6331e303a9e86d0c19f183899f958bf2a71cf5a7070d46899e25e1ac877b23d4
  ```

- The repository README declares CC BY-NC 4.0 for repository code, but no separate license for the checkpoint has been found.
- The standalone `test_video.mp4` has no paired GolfDB annotation record. Its SwingNet predictions are smoke evidence only.
- The required preprocessed labeled validation videos and `val_split_*.pkl` files are not present locally.
- Do not use generic YOLO outputs for swing phases or impact.
- Do not use SwingNet output as validated GhostCaddie phase or impact evidence before labeled-data evaluation passes.
- Do not evaluate or enable ball tracking, clubhead tracking, landing, calibration, hazard analysis, expected strokes, recommendations, or course-space analytics in this plan.

## Required stop conditions

Stop the acquisition/evaluation path and leave SwingNet evaluation-only if any of the following is true:

1. The checkpoint’s applicable license cannot be established for the intended non-commercial research use.
2. The dataset/preprocessed video license terms cannot be established.
3. Download provenance cannot be recorded reproducibly.
4. The downloaded files do not match their published hashes or expected sizes.
5. Videos cannot be matched unambiguously to the official annotation records.
6. The split data lacks the event labels required for the official evaluator.
7. The official model or dataset requires credentials, cookies, browser sessions, DRM bypass, or platform-protection bypass.

A public download is not by itself a license clearance. Record `license_status: unresolved` and do not promote the model when terms are ambiguous.

---

## Task 1: Freeze source and license manifest

**Objective:** Create an auditable manifest before acquiring labeled data.

**Files:**
- Create: `out/golfdb_evaluation/swingnet_acquisition_manifest.json`
- Preserve: `out/golfdb_evaluation/swingnet_1800.pth.tar`
- Preserve: `out/golfdb_evaluation/source_metadata.json`

Record:

- repository: `wmcnally/golfdb`;
- immutable implementation commit;
- README URL and exact retrieved README hash;
- implementation files and hashes (`model.py`, `MobileNetV2.py`, `eval.py`, `dataloader.py`, `util.py`);
- checkpoint source URL and redirect target if applicable;
- checkpoint filename, byte size, SHA-256;
- dataset source URL(s), archive/file names, byte sizes, and SHA-256;
- stated license text and URL;
- whether the license explicitly covers code, weights, annotations, and preprocessed videos;
- intended use: isolated, non-commercial research evaluation only;
- approval status and unresolved-license reasons.

Do not put absolute local paths, credentials, cookies, or consent records into shareable reports.

**Verification:** Recompute all hashes with `shasum -a 256`; compare byte sizes; fail closed on mismatch.

---

## Task 2: Acquire official labeled metadata without bypasses

**Objective:** Obtain the official metadata and generate the same split files used by GolfDB.

**Files:**
- Download only into: `out/golfdb_evaluation/golfdb_labeled/`
- Create: `out/golfdb_evaluation/golfdb_labeled/acquisition_log.json`

Acquire the official metadata source identified by the repository, preferably `data/golfDB.mat` or the published `golfDB.pkl`, using direct HTTPS retrieval only. Do not use cookies, credentials, browser sessions, proxies, scraping, DRM bypass, or platform circumvention.

Run the official split-generation logic or a byte-for-byte documented equivalent to produce:

```text
val_split_1.pkl
val_split_2.pkl
val_split_3.pkl
val_split_4.pkl
train_split_1.pkl
train_split_2.pkl
train_split_3.pkl
train_split_4.pkl
```

Record for every row:

- GolfDB clip ID;
- source YouTube ID as sanitized metadata only;
- player/sex/club/view/slow fields;
- eight event frame labels;
- bounding box;
- assigned split.

Keep raw metadata and generated split files inside the isolated evaluation directory. Do not copy dataset material into the standard analytics environment.

**Verification:** Confirm four validation splits exist, event arrays contain eight event positions, event ordering is strictly increasing, and no row is duplicated across a validation split. Confirm the standalone `test_video.mp4` is not falsely associated with a labeled row.

---

## Task 3: Acquire and match preprocessed validation videos

**Objective:** Obtain the official `videos_160` material required by the GolfDB dataloader and prove annotation/video matching.

**Files:**
- Download only into: `out/golfdb_evaluation/golfdb_labeled/videos_160/`
- Create: `out/golfdb_evaluation/golfdb_labeled/video_match_manifest.json`

Use only the official dataset distribution link documented by GolfDB. Before download, verify applicable terms for the videos separately from code and model weights. If the terms are absent or incompatible, stop here.

For each labeled row in `val_split_1.pkl` through `val_split_4.pkl`:

1. locate `<id>.mp4` under `videos_160/`;
2. verify it is a regular local file;
3. inspect width, height, FPS, duration, frame count, and codec;
4. confirm the frame range can address all eight event labels;
5. compute SHA-256;
6. record the relation `annotation.id -> video filename -> video hash`.

Do not infer or repair missing IDs. Missing or ambiguous matches are explicit failures.

**Verification:** Require a complete match for the selected evaluation split before scoring. Retain a deterministic manifest so a later run can prove that the same annotation/video pairs were evaluated.

---

## Task 4: Add focused evaluator-contract tests first

**Objective:** Define the evaluation behavior before changing any evaluator code.

**Files:**
- Test: `tests/test_swingnet_evaluation.py`
- Likely implementation: `scripts/run_swingnet_evaluation.py` or `ghostcaddie/video/swingnet_evaluation.py`

Write failing `unittest` tests for:

- exact eight-label ordering and GhostCaddie phase-label mapping;
- rejection of missing/duplicate/ambiguous video IDs;
- rejection of mismatched video dimensions or frame counts;
- deterministic checkpoint/dataset hash recording;
- frame-tolerance scoring;
- PCE calculation matching GolfDB’s `correct_preds` semantics;
- explicit `null` metrics when labels or model outputs are unavailable;
- no generation of `ShotEvent`, recommendation, calibration, ball, clubhead, or landing fields;
- provenance preservation when mapping GolfDB labels to GhostCaddie phases;
- no modification of existing command behavior.

Run each new test and observe the expected RED failure before implementing it.

---

## Task 5: Implement isolated official evaluation runner

**Objective:** Run the official SwingNet procedure on a selected labeled validation split without changing production workflows.

**Files:**
- Create or modify only the isolated evaluator identified in Task 4.
- Do not modify: `ghostcaddie/pipeline.py`, `ghostcaddie/video/orchestration.py`, existing analytics, human fallback, or existing CLI commands.

The runner must:

1. validate the acquisition manifest and license status;
2. refuse to run if model or dataset license status is unresolved unless explicitly invoked in a non-scoring smoke mode;
3. load the official SwingNet architecture and checkpoint in `.venv-video-ai`;
4. use the official 160×160 preprocessing and normalization;
5. use the official validation split and full-clip sequence procedure;
6. preserve all eight original GolfDB label names;
7. map labels to GhostCaddie’s phase vocabulary only in a separate field, for example:

   ```json
   {
     "source_label": "Impact",
     "ghostcaddie_phase": "contact",
     "mapping_provenance": "GolfDB/SwingNet-compatible evaluation mapping"
   }
   ```

8. never use the mapping to create validated runtime phase evidence;
9. write deterministic predictions, confidence values, source frame indices, and timing;
10. record runtime, mean latency, peak memory where measurable, device, package versions, and hashes;
11. keep ball, clubhead, landing, calibration, course position, and analytics unavailable;
12. emit `evaluation.json` only after all input validation succeeds.

Use device-aware hidden-state initialization for MPS/CPU, but preserve the official model architecture and checkpoint semantics.

---

## Task 6: Implement PCE and frame-tolerance reporting

**Objective:** Match the official GolfDB event-scoring semantics and expose tolerance metrics clearly.

For each clip and split, compute:

- predicted frame for each of the eight event channels;
- absolute frame error per event;
- tolerance in frames using the official definition: `max(round((impact - address) / 30), 1)` unless the official evaluator specifies otherwise;
- per-event accuracy within tolerance;
- clip-level PCE: proportion of events within tolerance;
- split-level mean PCE;
- exact-count and weighted summaries;
- failure counts for missing/low-confidence predictions.

Report both:

```text
strict_frame_accuracy: exact frame match
within_tolerance_accuracy: official tolerance match
PCE: official GolfDB-style percentage of correctly localized events
```

Do not report PCE for the standalone `test_video.mp4`. Its metrics must remain `null` with reason `ground_truth_unavailable`.

---

## Task 7: Generate annotated comparison artifacts

**Objective:** Make model predictions and errors visually reviewable.

**Files:**
- `out/golfdb_evaluation/swingnet_labeled/annotated_frames/`
- `out/golfdb_evaluation/swingnet_labeled/event_montages/`
- `out/golfdb_evaluation/swingnet_labeled/evaluation.json`

For every evaluated clip, render:

- source frame index and timestamp;
- predicted event label and confidence;
- ground-truth event label and frame;
- signed and absolute frame error;
- tolerance window;
- disagreement/failure status;
- model and dataset provenance reference.

Do not render ball, clubhead, landing, calibration, trajectory, hazard, expected-strokes, or recommendation overlays.

Visually inspect at least one montage per split and record the result in `evaluation.json`.

---

## Task 8: Evaluate split-by-split and held-out behavior

**Objective:** Prevent a single aggregate score from hiding split failures.

Run:

```bash
/Users/giofiore/ghostcaddie-tour/.venv-video-ai/bin/python \
  scripts/run_swingnet_evaluation.py \
  --dataset-root out/golfdb_evaluation/golfdb_labeled \
  --split 1 \
  --weights out/golfdb_evaluation/swingnet_1800.pth.tar \
  --out out/golfdb_evaluation/swingnet_labeled/split_1
```

Repeat for splits 2, 3, and 4, or use one deterministic multi-split invocation. Record:

- split ID;
- number of clips and events;
- PCE;
- per-event tolerance accuracy;
- strict frame accuracy;
- confidence distribution;
- runtime and memory;
- skipped clips and exact reasons;
- dataset/model hashes.

If a supplied held-out set is available, reserve it before tuning thresholds and report it separately from validation splits. Do not tune and score on the same held-out material.

---

## Task 9: Promotion decision and gate preservation

**Objective:** Keep production behavior unchanged unless measured evidence and licensing justify a later, separately approved integration.

SwingNet remains research-only unless all of the following are true:

- model and dataset licenses are explicitly cleared;
- all evaluated files have reproducible hashes;
- official labeled split matching is complete;
- PCE and frame-tolerance metrics are produced for the selected validation protocol;
- failures and confidence distributions are reviewed;
- any held-out evaluation passes documented provisional thresholds;
- a separate approval authorizes promotion;
- a production adapter emits only validated `video-observations.v1` evidence;
- phase and impact gates are updated only from measured results, never from the standalone smoke.

Even after a phase/impact promotion, keep independent gates closed for:

- ball detection/tracking;
- clubhead detection/tracking;
- landing;
- trajectory;
- calibration;
- hazard analysis;
- expected-strokes analytics;
- recommendations.

The human annotation fallback remains available and explicit.

---

## Verification checklist

After implementation and evaluation:

```bash
python3 -m unittest tests.test_swingnet_evaluation
python3 -m unittest discover -s tests
python3 -m compileall -q ghostcaddie tests scripts
```

Also verify:

- `evaluation.json` parses with `allow_nan=False`;
- model, implementation, metadata, split, and video hashes match manifests;
- no absolute local paths or credentials are serialized;
- no URL/cookie/browser-session data is retained in reports;
- no `recommendation.json`, normalized course-space `ShotEvent`, or analytics artifact is created;
- existing CLI help and existing workflows remain unchanged;
- the original 265-test baseline remains green;
- annotated outputs are visually inspectable;
- standalone `test_video.mp4` remains clearly labeled as smoke-only with no accuracy claim.

## Expected outcome if blocked

If licensing or labeled data cannot be cleared, produce a blocker report containing:

- exact missing license or dataset artifact;
- source URL and retrieval status;
- hashes for any downloaded research-only files;
- the last successful architecture/checkpoint smoke;
- `event_accuracy: null`, `PCE: null`, and `frame_tolerance: null` with reasons;
- explicit confirmation that SwingNet is not enabled in production mode and all GhostCaddie gates remain closed.
