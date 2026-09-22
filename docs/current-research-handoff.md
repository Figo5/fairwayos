# FairwayOS current research handoff

Status date: 2026-09-21 (previously 2026-09-16). This document is an index and safety boundary for local golf-perception research artifacts. It is not an acceptance report and does not make production claims.

**Primary classification, settled 2026-09-21: `FAILED_TO_GENERALIZE`** for the ball-tracker line (v1, v2 and v3), on **development evidence** — the only real-labelled corpus in this repository (`data/ball_annotations/`, 10 clips, 185 ball-visible frames) was consumed during development, so **no clean held-out accuracy figure exists for v3 and none is claimed**. The v3 final test remains **SEALED / NOT RUN** by decision. **No promotion. No canonical ball integration.** The research snapshot is archive provenance, not a promotion freeze. Canonical `ball` stays `UNAVAILABLE` (`method: none`), `CLUBHEAD` stays `UNAVAILABLE`, and every physical metric — ball speed, clubhead speed, launch, carry, apex and **spin** — stays `UNAVAILABLE`. The finished product pass is summarised in [`fairway-os-completion.md`](fairway-os-completion.md); decisive evidence is [`ball-tracker-v3-results.md`](ball-tracker-v3-results.md). **No further ball research loop is planned on this corpus.**

## There is now a canonical path, and this is not it

Everything below this section is **research**. It is retained as evidence and as a record of what was falsified. None of it is reachable from the canonical workflow.

The canonical local video workflow is:

```bash
.venv/bin/python3 -m ghostcaddie analyze-video CLIP.mp4 --out OUTPUT_DIR
```

- **Operator guide:** [`video-analysis-user-guide.md`](video-analysis-user-guide.md)
- **Evidence:** [`canonical-video-pipeline-audit.md`](canonical-video-pipeline-audit.md) (preserved unchanged)
- **Measured behaviour on the local corpus:** [`canonical-validation-sweep.md`](canonical-validation-sweep.md)

What changed on 2026-09-16:

1. **Model paths no longer touch `/tmp`.** Resolution is explicit argument → env var → `$FAIRWAY_MODEL_ROOT` → `<repo>/models/` → `~/.fairwayos/models/` → *unavailable*. Nothing is downloaded. `python3 -m ghostcaddie models` prints the search path and what is present. The surviving MoveNet flatbuffer was moved out of `/tmp` into the gitignored `models/` directory (sha256 `0491b130b2dd622d71a180936e957212341daca78e9f6a04e0d9c2045eff2ecb`). The `ball` interpreter no longer defaults into a `/tmp` venv; absent `$FAIRWAY_TAPIR_PYTHON` the probe reports that it is not configured.
2. **`body_legacy_ultralytics` no longer hardcodes a path into the other repository's working tree.** It resolves nothing; it remains permanently blocked on loader-safety grounds, which is the point of keeping it.
3. **`/ready` gained an `availability` tri-state** — `ready` / `model_absent` / `misconfigured` / `interpreter_missing`. An absent optional checkpoint is an expected state, distinguishable from a misconfiguration. Readiness honesty was not weakened: `can_execute_now` is still false for everything that cannot run.
4. **The portable suite passes with no SAM2 or BootsTAPIR checkpoint present.** The one test that was hard-bound to a `/tmp` checkpoint now conditions its assertion on the target actually being ready, and asserts the absent-asset case as well.
5. **Body is conditional.** Whole-frame single-pose output is never called "the golfer". See the guide, §6.
6. **Ball and clubhead are `unavailable` by construction on the canonical path**, with the falsification as the reason. No configuration turns them back on.

### On restoring the falsified checkpoints

Not done, and not recommended yet. SAM2 and BootsTAPIR would revive code whose claims are already withdrawn. Restore an optional asset only when the canonical path actually needs it, the audit evidence supports the method, the environment can load it safely, and there is a concrete validation plan. Until then they stay `model_absent`, documented, and out of the canonical path.

## Ball-tracking research pass, 2026-09-18

Outcome: **FAILED_TO_GENERALIZE**. Canonical `ball` stays `UNAVAILABLE`; nothing on the
canonical path changed and `tests/test_ball_research_isolation.py` fails if that stops
being true.

What it added that is worth keeping regardless of the outcome:

- **The first ball ground truth in this repository.** `data/ball_annotations/` — 325
  human-read frames across 10 rights-cleared Pexels clips (189 ball-visible, 136
  ball-absent, 19 ambiguous), in 20 consecutive runs, every label bound to its clip's
  SHA-256 and every visible label rendered back onto its own frame and checked by eye.
  Four clips were mislabelled on the first pass and corrected — one of them had the
  centre on the *tee* rather than the ball.
- **A frozen clip-level split**, `data/ball_split.json`, validated by the existing
  `research_split` contract, split by shoot group rather than by clip so a shared
  golfer cannot leak across splits. Held-out was scored once, after freezing.
- **An evaluator that scores the right thing** (`ball_evaluation.py`): correct silence
  on a ball-absent frame is a success, a prediction on the wrong object is a false
  positive rather than a near miss, and only `observed` states can earn credit.
- **Two measured baselines and one method**, per clip, in `docs/ball-tracking-results.md`.

Findings to carry forward:

- The flagstick/tee/shoe failure family that got the earlier tracker withdrawn does not
  occur anywhere in this corpus under the new method. The tee case is explicit: on
  `pexels_6573621` the bare tee stands in the ball's former position and the method
  correctly says nothing.
- The unsolved distractor is the **golfer's own body**. A head, shoulder or cap has a
  closed circular boundary at ball scale and a strong colour separation from grass, and
  no bottom-up appearance feature tried here separates it from a ball.
- `stationarity_gate.py` must never be applied to the ball layer: a tee'd ball is
  definitionally sustained-stationary, and that gate would delete every correct at-rest
  observation.
- `temporal_candidate_gate.accept_candidate_run` requires `residual_motion > 0` and so
  also discards every at-rest ball. Concept reusable, implementation not.

## Ball-tracking research pass v3, 2026-09-21 — closed

Outcome: **`FAILED_TO_GENERALIZE`**, on development evidence. Canonical `ball` stays
`UNAVAILABLE`; nothing on the canonical path changed, and
`tests/test_ball_v3_isolation.py` fails if that stops being true.

What it measured, plainly:

- The v3 architecture (blob proposals → context features → listwise tracklet ranker)
  recovered **0 of 185** human-labelled ball-visible frames on `legacy` in **all six**
  validation variants, including the full model. That corpus is development-consumed,
  so this is the shape of the failure, **not** a held-out accuracy claim.
- On unlabelled real video the tracker mostly abstains (coverage 0.17–0.48). No
  identity recall, false-track rate or lock rate is reported there: the corpus has no
  labels and inventing them is forbidden. `switches` and `body_lock_rate` in the
  results doc are **geometric proxies**, not label-derived measurements.
- Temporal separability rests on synthetic candidate lists, and 72.8 % of the
  family-split validation rows sit in clips the ranker was trained on. No split is
  both family-disjoint and ranker-optimism-free, and no genuinely frame-independent
  ablation was run. All of that is disclosed in
  [`ball-tracker-v3-results.md`](ball-tracker-v3-results.md), whose §8.2 correction
  ledger records the reporting errors an independent review caught and fixed.

**Sealed.** The 10 final-test clip ids were never decoded, viewed, inferred on,
labelled or cached in any pass, and were independently re-verified as absent from
all 113 files of the closeout artifact tree. The seal is intact and stays closed:
spending untouched evidence on an already-negative model is not justified. **No
promotion, freeze or canonical integration is authorized.**

**Archive, not freeze.** The research snapshot is preserved as provenance. It is
explicitly **not** a promotion freeze and carries **no future-promotion promise**.
Reopening this line requires new rights-cleared labelled data, not a retune.

## Hard research flags

All referenced perception records remain:

- `research_only=true`
- `pseudo_label=true`
- `ground_truth=false`
- `production_eligible=false`

Do not call production analytics, calibration, carry/apex/landing, recommendations, or ShotEvent logic from these records. Do not publish or upload derived PGA TOUR renders without express rights clearance.

## Baseline preserved

Baseline and historical repo commits are preserved on branch `pga-research-analyzer`. Existing local `out/` media, model weights, and generated artifacts were not moved into source control by this handoff. The committed PGA analyzer/demo history remains the baseline record; rejected experiments are retained as evidence, not rewritten into wins.

## Source-binding failure and durable fix

Completed work in `/tmp/fairway-tapir-source-repair` found a source-binding defect wider than first reported: Morikawa and Gotterup BootsTAPIR runs were named as separate clips but consumed the same Lipsky video bytes (`61e852fb...c1ef6`). All cross-clip/transfer claims from those runs are invalid and withdrawn.

Durable repo fix added here: `ghostcaddie/video/source_binding.py` plus `tests/test_source_binding.py`. It fails closed before model import unless media bytes match the caller's expected SHA-256 and, when provided, the seed JSON's own `source_sha256`. Clip names/player names are never accepted as byte provenance.

## Reusable repo code retained

- `ghostcaddie/video/team_layer_integration.py`: validates independent body/clubhead/ball layer handoffs, source SHA, research flags, finite bounded geometry, blocked unavailable-only layers, and renderer timeline clearing.
- `ghostcaddie/video/siwoo_layers.py`: local Si Woo helper contracts for independent layer states, cut resets, caddie/body selection risk, exact source-to-display mapping, visibility intervals, body continuation budget, and bbox clamping.
- `tools/demo/build_team_layer_demo.py`: local-only renderer that consumes strict team layer handoffs and writes a checkpoint instead of fabricating missing layers.
- `tools/demo/build_pga_approach_overlay.py`: preexisting local multilayer Si Woo approach overlay work was preserved for parent review rather than overwritten.

## Local work roots audited and classification

Do not delete these `/tmp` roots; they contain useful evidence and rejected history.

- `/tmp/fairway-single-source-build`: first-party single-source Tommy prototype. Useful design: source-bound timeline, independent object states, exact 30000/1001 timing, no missing-ball default, partial occlusion state, no centroid trails. Not copied wholesale because `build_single_source.py` is a local prototype tied to acquired local media and frozen worker paths. Its durable lesson is captured in repo layer contracts/tests and this handoff.
- `/tmp/fairway-parallel-acceptance`: first-party structural acceptance harness. Useful design: source hash checks, finite bounded geometry, seed-vs-predictive counts, demo eligibility defaults false, cross-identity collision detection. Not copied wholesale because it snapshots mutable `/tmp` roots and local artifact paths; selected semantics already overlap with `team_layer_integration.py` and `source_binding.py`.
- `/tmp/fairway-frame-hash-audit`: first-party canonical decoded-frame digest helper. Useful for audits, but not copied as production source because it hardcodes local source paths, a local venv, and OpenCV decoder policy. Keep as an offline audit prototype until generalized.
- `/tmp/fairway-tapir-source-repair`: first-party source-binding repair. Safe reusable subset was ported to `ghostcaddie/video/source_binding.py` with portable tests.
- `/tmp/fairway-learned/tracker`: learned point-tracker prototype, local weights, runs, sheets, and media-derived outputs. Do not copy weights, rendered media, seed files, or run outputs. Loader safety concern remains for body/Ultralytics usage where unrestricted pickle loading was observed in a separate loader trace.
- `/tmp/fairway-sam-head`: SAM2 clubhead worker with third-party SAM2 source checkout, checkpoint-derived outputs, masks, overlays, and local media. Do not vendor `sam2-src`, weights, masks, overlays, or media. The useful result is classified as partial/emerging SAM correction evidence only; it is not a production clubhead tracker.

## Explicit rejected/limited findings

- Appearance post-filters do not salvage the MIL clubhead path. HOG was decisively killed by a second false-lock object; HSV did not establish a positive generalizable case. Do not retune thresholds to revive this.
- Source adequacy remains a bound: clubhead-through-impact at current available frame rates is out of scope for the clips where required displacement exceeds the tracker design point.
- The corrected TAPIR reruns showed Lipsky genuine, Morikawa 1/51, Gotterup 0/71. The learned point tracker does not transfer under the frozen crop/model settings.
- The source-binding failure invalidates previous cross-clip/transfer claims; repaired reruns are development-consumed, not held-out proof.
- Body pose via Ultralytics has an unresolved unrestricted loader concern unless the effective safe-loading path is actually verified. Do not claim safe loading from wrapper intent alone.
- SAM2 head segmentation is partial/emerging correction evidence: f310-f343 full, f344-f360 partial/emerging after independent correction. Partial mask centroids are not head centers, material points, or trajectories.
- The latest single-source Tommy demo is an experimental local artifact, not accepted production: honest two-layer (body + clubhead) only, ball unavailable, no transfer/held-out claim, no safe body-loader claim.

## Reproduction and local asset requirements

Reproduction of media-dependent artifacts requires local ignored assets that are not committed:

- Official PGA TOUR and user-supplied local clips under `out/pga_official_acquisition/` or worker roots.
- Local model weights such as `yolo11n-pose.pt`, SAM2 checkpoints, and learned tracker checkpoints.
- Optional local CV/AI environment (`.venv-video-ai`) with OpenCV/ffmpeg/ultralytics/SAM dependencies as applicable.

Without those local assets, run only portable unit tests and compile checks. CI must not download live media, model weights, or scrape PGA TOUR.

## Evidence hashes to preserve

- Tommy source bytes: `2358a7c9b912b728f1d748a1f7e8aea6e5d7bc9219472ec4e724d503875a0a44`.
- Si Woo source bytes used by team-layer renderer: `cefbdf25400f5821893747e2b4a60ca5a11f990920ab3a9c8bba32a8c2d3deae`.
- Scheffler putt source bytes: `21c8bb54cd550dec2da680351cfd48f9cc61c24ca2ece43e07e30622b26e28f1`.
- Rejected/invalid Lipsky byte reuse in source-binding audit: `61e852fb...c1ef6` (see `/tmp/fairway-tapir-source-repair/source_audit.json` for full value).
- Tommy single-source local video from `/tmp/fairway-single-source-build`: `0356d3ba11456fca2cc2f2f486592feaa9303234e37f976d2dc8edeb7cfe002f`.
- Approved preexisting Si Woo team-layer demo hash recorded in loop checkpoint: `79ace086bb1d1db84aac2007a2c41afc53680ec36c60ad4032b10577f5aaa3bd`.

## Known blockers / next safe work

1. Generalize decoded-frame hash audit into a portable repo tool only after replacing hardcoded paths/venv assumptions with CLI arguments and synthetic tests.
2. If using Ultralytics body pose again, verify the effective safe-loader branch (`ULTRALYTICS_SAFE_LOAD=1` or equivalent) by execution before inference; do not retry unrestricted loading.
3. If integrating SAM masks, carry `observed_full`, `observed_partial_occluded`, and `observed_emerging` separately; never promote partial/emerging masks to full object centers.
4. Any new learned-tracker run must call `bind_source()` before model import/decoder use and must treat seed frame support separately from predictive frames.
