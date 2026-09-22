# Ball tracker v3 — current results (worker pass 3, closeout)

**This is the authoritative evidence document for the v3 ball tracker.** The
worker-pass-1 document (`docs/ball-tracker-v3-worker-pass-1.md`) is superseded and
kept only as a history of how these numbers were arrived at.

**Status: `FAILED_TO_GENERALIZE` — classified by the primary on 2026-09-21, on
development evidence.** The architecture is **archived as non-promoted**: the
snapshot preserved for provenance is an **archive, not a promotion freeze**, and it
carries **no future-promotion promise**. All six variants recovered **0 of 185**
human-labelled ball-visible frames on `legacy`, and the only real-labelled corpus in
this repository is development-consumed — so **no clean held-out accuracy figure
exists for v3 and none is claimed here**. The v3 final test is **SEALED / NOT RUN**
(see §10): the tracker never cleared review readiness, so spending untouched
evidence on a known-negative model is not justified. Canonical `ball` remains
`UNAVAILABLE` / `method: none`; canonical `CLUBHEAD` is `UNAVAILABLE`; every
physical metric including spin is `UNAVAILABLE`; no canonical code imports any of
this. Nothing here is a product-completion claim.

See [`fairway-os-completion.md`](fairway-os-completion.md) for the finished-product
summary, and [`current-research-handoff.md`](current-research-handoff.md) for the
research index.

## 1. Evidence classes, kept separate

| Class | Corpus | What it can support | What it cannot |
|---|---|---|---|
| Real-labelled, **development-consumed** | `legacy` — `data/ball_annotations/` (10 clips, 185 visible labelled frames) | tracking behaviour on real video; abstention; the shape of the failure | accuracy on unseen data: this corpus scored an earlier pass and chose this pass's proposal parameters |
| **Unlabelled** real video | `v3-validation` — 11 validation clips of `data/ball_dataset_v3_eval` | behaviour only: abstention, coverage, lock-type diagnostics, wall time | identity recall, false-track rate, lock rate — there are no labels, so those are reported `null` with a reason |
| Synthetic candidate lists | `out/ball_tracker_v3` tracklet store | separability of feature vectors under a saved checkpoint | real-ball accuracy: the supervision is synthetic insertion |
| **Sealed** | 10 final-test clips | nothing yet | anything, until deliberately consumed |

Every metric carries its definition and denominator; `null` means *not measurable
here*, never zero.

## 2. Corrections in this pass

Five review-2 must-fixes, implemented and verified; no retraining, no threshold
change, no new experiment.

1. **`as_deployed_no_mask` withdrawn.** `cmd_recompute_auc` computed it with
   `mask=None`, but the ablation mask is intrinsic to
   `TrackletDiscriminator.forward`, so all 10/10 rows were bit-for-bit duplicates
   of `as_trained` — a reader would conclude masking changed nothing. The branch is
   removed; the field is now `{"measurement": null, "reason": ...}`. The evidence
   that the defect was real is review 1's independent measurement: feeding a
   dropped group live values collapses validation AUC **0.9713 → 0.5612**. Prior
   artifact retained as
   `artifacts/closeout/superseded/temporal_auc_recomputed.superseded-2026-09-21.json`.
2. **Corrected timings recomputed by exact nested key**
   (`components.components_seconds_per_frame`, `components.frames`,
   `components.decode_cold_seconds`). The previous version walked top-level keys,
   matched nothing and wrote empty shells (`component_key: null`); it now extracts
   22 corrected keys per run. No re-measurement.
3. **Extraction cache keyed by fingerprint.** `ball_v3_tracker.extraction_fingerprint`
   binds the source media SHA256, the **ordered** frame indices, the ranker weights
   SHA256, the proposal **and** tracker parameters (so `top_k` is inside the key),
   the digests of the preprocessing code, the body-input mode **and** the selected
   poses' own digest, plus `EXTRACTION_VERSION`. A stored entry whose recorded
   fingerprint disagrees is **refused**, not reused. `params=tracker_params` now
   reaches `extract_frames` (it previously fell back to defaults, leaving `top_k`
   uncoupled from the tracker). Cache-hit timings are reported as inherited build
   time, never as the variant's runtime.
4. **Seal regression at the entry point.** A new test drives the real `main` with a
   mocked run list containing a sealed clip id (metadata only) while the decoder,
   the body cache and the model loader are recording stand-ins: `SystemExit`,
   nothing touched, no artifact written. A control run proves the stand-ins are
   wired. The older test only proved that the guard function itself decodes
   nothing — not that the entry point calls it first.
5. **Accurate labels.** `--no-body-upstream` is now *refused* unless the loaded
   checkpoint drops the body group, so a "full no-body" run cannot be a
   body-present-but-zero run. The framewise reference is labelled
   *no temporal model or association; ranker motion-residual context retained* —
   its input is **not** frame-independent, because the ranker reads
   camera-motion-residual context channels. A partial motion-group ablation is
   labelled partial, never "all temporal information removed".

Also added: a non-blank FrameViews regression fixture
(`tests/fixtures/frameviews_turf.png`, 320x240, synthetic, 94 KB — no real media in
Git) that pins the crop-path structure *without* claiming equivalence. Review 1's
real-crop numbers stand as the honest statement: mean |Δpixel| **0.905/255** and
**1.758/255**, with the ranker's top-1 flipping on **3 of 15** frames on one clip
and **0 of 15** on the other. That is an approximation and a preprocess shift.

## 3. Timing, corrected

Sources: `artifacts/corrected/corrected_timings.json` (recomputed from the raw
per-clip samples in `artifacts/profile/components_{baseline,after}.json`, 43 frames,
`repeats=2`; nothing re-measured). The earlier report divided by
`frames_total × repeats`, understating every value by 2×.

| component (cold, s/frame) | baseline | after optimisation |
|---|---|---|
| six generators total | 4.2428 | 1.1204 |
| `generator:local_contrast` | 1.9916 | 0.1100 |
| `generator:brightness_cue` | 0.8392 | 0.2104 |
| `generator:radial_symmetry` | 0.6578 | 0.1340 |
| `generator:local_distinctness` | 0.3894 | 0.3228 |
| `generator:flow_residual` | 0.3206 | 0.3078 |
| `build_contexts` | 0.0700 | 0.0708 |
| decode (cold, per frame) | 0.0472 | 0.0500 |

Decode is timed once on the cold pass, so it is **not** multiplied by `repeats`;
that is stated in the artifact. Filesystem cache state is uncontrolled between
runs, so these are matched-condition comparisons, not absolute constants.

**End-to-end tracker runtime** comes from this pass's matrix: the extraction build
cost is measured on the first variant of each corpus, per frame, and reported in
§7. A cache hit's motion/proposal/ranking seconds are the build run's — real
measurements of the extraction, but not that variant's runtime — and the artifacts
keep the two apart (`timings_measured_this_run` vs
`timings_inherited_extraction_build`).

## 4. Proposal stage — A/B/C measurements (reused, not re-swept)

`artifacts/validation/sets_legacy.json` (before) and `sets_after.json` (after), 7
legacy clips, 84 labelled frames, `repeats=1`. Recall = a proposal within one ball
radius of the human label. These are the numbers that chose the proposal
configuration; they are development evidence on a development-consumed corpus.

| generator set | before recall | after recall | before proposals/frame | after | before s/frame | after |
|---|---|---|---|---|---|---|
| A `local_distinctness` only | 0.7381 | 0.7500 | 95.53 | 95.48 | 0.3810 | 0.3057 |
| B A + `brightness_cue` | 0.7500 | 0.7500 | 112.50 | 113.04 | 1.2382 | 0.5314 |
| C legacy five (v2 stack) | 0.3214 | 0.3929 | 61.99 | 105.10 | 3.6792 | 0.8225 |
| C2 all six | 0.7381 | 0.7500 | 138.99 | 179.57 | 4.1093 | 1.1161 |

The trade is explicit: more proposals for more recall, at a downstream cost the
shared-pyramid crop reduced. Per-clip regressions after the change are single
frames. The share of candidate positions agreeing with the pre-optimisation path
is in `artifacts/profile/equivalence_v2.json`; the two paths are an approximation,
not an identity (review 1's real-crop deltas above).

## 5. Temporal model — tie-correct AUC on fixed checkpoints

`artifacts/corrected/temporal_auc_recomputed.json`, `estimator: mid-rank
Mann-Whitney (ties contribute 0.5)`, masks applied from each checkpoint, migrated
weights (`artifacts/corrected/temporal/`). **These are fixed-checkpoint scores, not
a valid model selection**: `best_epoch` was chosen with the faulty estimator.
Synthetic-candidate-list measures, not real-ball accuracy. `ties` is the number of
observations sharing a score with another, reported so a high AUC cannot be
mistaken for resolution.

| checkpoint | split | val AUC | tied scores | note |
|---|---|---|---|---|
| full (D) | family | 0.9761 | 75 | |
| full (D) | clip | 0.9835 | 11 | |
| E body-dropped | family | 0.9838 | 0 | |
| E body-dropped | clip | 0.9773 | 100 | |
| F motion-dropped | family | 0.9641 | 437 | dominated by ties |
| F motion-dropped | clip | 0.9980 | 3 | |
| ranking-dropped probe | family | 0.8678 | 66 | shortcut probe |
| appearance-dropped probe | family | 0.9713 | 2 | |
| chroma-dropped probe | family | 0.9683 | 65 | |
| scale-dropped probe | family | 0.5090 | 567 | **collapsed constant, non-converged: not interpreted** |

The old §7.2 headline — "dropping the temporal features does not hurt" with F
0.9820 vs D 0.9778 — is **withdrawn**: it came from the tie-blind estimator that
rewarded collapse. On the corrected estimator the family split reads D 0.9761 vs F
0.9641, i.e. **D > F**, and D's own margin over its splits is not evidence of a
usable temporal model either, because the dominant feature is the ranker's own
synthetic-trained score and 72.8 % of the validation rows sit in clips the ranker
was trained on (§9).

`as_deployed_no_mask` is `null` for every checkpoint with the reason recorded;
there is no valid unmasked measurement to report.

## 6. Real-video validation matrix

Six variants per corpus, run once each, no selection and no tuning from the
outputs: `artifacts/closeout/validation/*.json`, driver
`artifacts/closeout/run_closeout_matrix.sh`, per-row status and artifact SHA256 in
`artifacts/closeout/matrix_status.jsonl`. Extraction is reused across the variants
that share ranker inputs through the fingerprint-keyed cache, so only the D row per
corpus (and the E-full row, which changes the ranker's inputs) pays for a full
extraction.

`legacy` — 22 runs / 325 frames (10 clips, human-labelled, development-consumed).
`v3-validation` — 11 runs / 165 frames (11 clips, **unlabelled**). Every number
below is what the run measured; `null` is "not measurable here".

Variant labels are load-bearing:

| variant | what it is | what it is NOT |
|---|---|---|
| **D** | full tracklet model, mask applied from its checkpoint, upstream poses supplied | |
| **heuristic** | no learned temporal model: association, tracklet linking and the tracker's stated heuristic still run | not frame-independent |
| **E partial** | body-group ablation inside the tracklet model, ranker still fed poses | not "no body anywhere" |
| **F partial** | motion-group ablation only | **not** "all temporal information removed": association, track-level rules and every other feature group stay |
| **E full** | ranker fed no poses at all **and** the body-masked checkpoint (the validator refuses the flag without one) | |
| **F reference** | framewise: no temporal model, no association; the ranker's camera-motion-residual context channels are retained | not frame-independent either |

Two columns below are **geometric proxies, not label-derived measurements**, and
must not be read as identity or rejection evidence:

- **`switches`** counts consecutive OBSERVED frames whose centre moved further than
  `LINK_RADII × radius` (`tools/ball_v3_validate.py:223-240`). It is a **jump proxy**.
  No label is consulted, so it is **not** a labelled identity-switch count — on
  `v3-validation` there are no labels at all, yet the column still prints numbers.
- **`body-lock rate`** is the fraction of OBSERVED frames carrying a `BODY_LOCK`
  diagnostic, i.e. a **geometric torso-polygon overlap proxy**
  (`ball_v3_validate.py:242`). `body_lock_rate = 0` therefore means *no accepted
  track overlapped the torso polygon*; it is **not** a measurement that body
  rejection worked, and nothing here measures body rejection.

*Generated by `artifacts/closeout/report_tables.py` from `artifacts/closeout/validation/*.json` — no cell typed by hand.*

#### legacy — human-labelled, development-consumed

| variant | label | runs | frames | accepted | observed | coverage | abstention | switches | body-lock rate | recall | false-track on visible | runtime s/frame (measured) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **D** | D corrected full tracklet model (mask applied from checkpoint, upstream poses) | 22 | 325 | 11 | 58 | 0.1785 | 0.8215 | 0 | 0.0000 | 0.0000 | 0.1892 | 0.67990 |
| heuristic | reference: no learned temporal model -- association and tracklet heuristic retained, not frame-independent | 22 | 325 | 7 | 55 | 0.1692 | 0.8308 | 0 | 0.0000 | 0.0000 | 0.1730 | 0.00010 |
| **E** partial | E partial: body-group ablation in the tracklet model, upstream poses still fed to the ranker | 22 | 325 | 9 | 58 | 0.1785 | 0.8215 | 0 | 0.0000 | 0.0000 | 0.1892 | 0.00020 |
| **F** partial | F partial: motion-group ablation only -- association and every other temporal feature retained | 22 | 325 | 12 | 56 | 0.1723 | 0.8277 | 0 | 0.0000 | 0.0000 | 0.1784 | 0.00020 |
| **E** full | E full: no body anywhere -- ranker fed no poses, plus the body-masked checkpoint | 22 | 325 | 9 | 60 | 0.1846 | 0.8154 | 0 | 0.0000 | 0.0000 | 0.2162 | 0.72980 |
| **F** reference (framewise) | F reference: no temporal model or association; ranker motion-residual context retained | 22 | 325 | 16 | 139 | 0.4277 | 0.5723 | 6 | 0.0000 | 0.0000 | 0.4270 | 0.00000 |

#### v3-validation — unlabelled: behaviour only

| variant | label | runs | frames | accepted | observed | coverage | abstention | switches | body-lock rate | recall | false-track on visible | runtime s/frame (measured) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **D** | D corrected full tracklet model (mask applied from checkpoint, upstream poses) | 11 | 165 | 5 | 33 | 0.2000 | 0.8000 | 0 | 0.0000 | null | null | 0.84150 |
| heuristic | reference: no learned temporal model -- association and tracklet heuristic retained, not frame-independent | 11 | 165 | 4 | 37 | 0.2242 | 0.7758 | 0 | 0.0000 | null | null | 0.00010 |
| **E** partial | E partial: body-group ablation in the tracklet model, upstream poses still fed to the ranker | 11 | 165 | 5 | 45 | 0.2727 | 0.7273 | 0 | 0.0000 | null | null | 0.00020 |
| **F** partial | F partial: motion-group ablation only -- association and every other temporal feature retained | 11 | 165 | 5 | 45 | 0.2727 | 0.7273 | 0 | 0.0000 | null | null | 0.00020 |
| **E** full | E full: no body anywhere -- ranker fed no poses, plus the body-masked checkpoint | 11 | 165 | 5 | 52 | 0.3152 | 0.6848 | 0 | 0.0000 | null | null | 0.82900 |
| **F** reference (framewise) | F reference: no temporal model or association; ranker motion-residual context retained | 11 | 165 | 9 | 80 | 0.4848 | 0.5152 | 8 | 0.0000 | null | null | 0.00000 |

### Detailed counters

| variant | frames | absolute/structure | counters | nulls with reason |
|---|---|---|---|---|
| **D** | 325 | labelled_visible=185, in_regime=127, labelled_absent=136 | observed_on_absent=23 (rate 0.1691), wrong_object=35, longest_run=15 | none |
| heuristic | 325 | labelled_visible=185, in_regime=127, labelled_absent=136 | observed_on_absent=23 (rate 0.1691), wrong_object=32, longest_run=15 | none |
| **E** partial | 325 | labelled_visible=185, in_regime=127, labelled_absent=136 | observed_on_absent=23 (rate 0.1691), wrong_object=35, longest_run=15 | none |
| **F** partial | 325 | labelled_visible=185, in_regime=127, labelled_absent=136 | observed_on_absent=23 (rate 0.1691), wrong_object=33, longest_run=15 | none |
| **E** full | 325 | labelled_visible=185, in_regime=127, labelled_absent=136 | observed_on_absent=20 (rate 0.1471), wrong_object=40, longest_run=14 | none |
| **F** reference (framewise) | 325 | labelled_visible=185, in_regime=127, labelled_absent=136 | observed_on_absent=59 (rate 0.4338), wrong_object=79, longest_run=15 | none |
| **D** | 165 | labelled_visible=0, in_regime=0, labelled_absent=0 | observed_on_absent=0 (rate null), wrong_object=0, longest_run=14 | identity recall, false-track rate, lock rate (no labels) |
| heuristic | 165 | labelled_visible=0, in_regime=0, labelled_absent=0 | observed_on_absent=0 (rate null), wrong_object=0, longest_run=14 | identity recall, false-track rate, lock rate (no labels) |
| **E** partial | 165 | labelled_visible=0, in_regime=0, labelled_absent=0 | observed_on_absent=0 (rate null), wrong_object=0, longest_run=14 | identity recall, false-track rate, lock rate (no labels) |
| **F** partial | 165 | labelled_visible=0, in_regime=0, labelled_absent=0 | observed_on_absent=0 (rate null), wrong_object=0, longest_run=14 | identity recall, false-track rate, lock rate (no labels) |
| **E** full | 165 | labelled_visible=0, in_regime=0, labelled_absent=0 | observed_on_absent=0 (rate null), wrong_object=0, longest_run=15 | identity recall, false-track rate, lock rate (no labels) |
| **F** reference (framewise) | 165 | labelled_visible=0, in_regime=0, labelled_absent=0 | observed_on_absent=0 (rate null), wrong_object=0, longest_run=14 | identity recall, false-track rate, lock rate (no labels) |

*`measured s/frame` is association time only on a cache-hit row; see §7 for what the D and E-full rows measure. The framewise reference returns no tracker timings by construction, so its cost is its wall time (7.17 s / 325 frames = 0.022 s/frame on legacy).*


## 6.1 What the matrix says, honestly

- **No variant recovers a real labelled ball.** Identity recall is **0 of 185** on
  `legacy` in every row, including the full model (D). This is the same negative the
  previous pass reported and review 1 independently reproduced; the matrix does not
  change it, and nothing in this pass was allowed to be tuned towards changing it.
- **The ablations barely move the decision counts.** Accepted runs: D 11, heuristic 7,
  E partial 9, F partial 12, E full 9 (of 22). Coverage stays 0.17–0.18 for every
  tracklet variant. Differences of a few runs on a corpus where nothing is recovered
  are not evidence that one variant is better; they are the shape of a tracker that
  is mostly abstaining.
- **The framewise reference is the one clear behavioural difference**: coverage
  0.4277 vs 0.1785 and false-track-on-visible 0.4270 vs 0.1892, with 6 **jump-proxy**
  switches (every tracklet variant has 0) — a geometric count, not labelled identity
  switches. That is what the track-level rules buy: far fewer observations, and far
  fewer wrong ones. It is a trade, not an improvement — recall stays 0 either way.
  Note also what this row is **not**: it removes the temporal model and association,
  but the ranker still reads camera-motion-residual context channels, so **no
  genuinely frame-independent ablation was run in this pass** and none of these rows
  isolates frame-independent behaviour.
- **`v3-validation` is behaviour only**: coverage 0.20–0.48, accepted 4–9 of 11 runs,
  no accuracy claim, recall/false-track/lock rates `null` because the corpus has no
  labels.
- **Body-lock rate is 0.0000 everywhere**, and that is a **geometric overlap proxy**:
  no accepted track's centre fell inside the torso polygon. It does **not** show body
  rejection working — nothing in this pass measures body rejection. The body feature
  is only ever a feature, never an exclusion region, and `club_lock_rate` is `null`
  (no clubhead tracker exists).
- **No claim below rests on ground truth for `v3-validation`.** That corpus is
  unlabelled; its coverage, abstention, switch and body-lock columns are behavioural
  and geometric only.

**Conclusion: not promotable.** The matrix confirms the negative on the only
real-labelled corpus available, on unlabelled real video shows a conservative tracker
that mostly abstains, and the temporal model's separability on synthetic lists is
optimistic (72.8 % ranker-exposed rows, §9). Classification remains the primary's
decision.

## 7. Runtime, and what "runtime" means here

*Generated by `artifacts/closeout/report_tables.py`.*

| corpus | variant | wall s (this invocation) | runs | measured tracker s | measured s/frame | inherited extraction s | runs built | runs reused | artifact sha256 |
|---|---|---|---|---|---|---|---|---|---|
| legacy | D | 228.77 | 22 | 220.97 | 0.67990 | 28.22 | 20 | 2 | `5628a834071377db` |
| legacy | heuristic | 6.78 | 22 | 0.04 | 0.00010 | 249.10 | 0 | 22 | `fa89f747b47ef276` |
| legacy | E_partial | 7.11 | 22 | 0.05 | 0.00020 | 249.10 | 0 | 22 | `937896d80acf5f37` |
| legacy | F_partial | 7.13 | 22 | 0.06 | 0.00020 | 249.10 | 0 | 22 | `c83cff37353a0425` |
| legacy | E_full | 244.79 | 22 | 237.19 | 0.72980 | 0.00 | 22 | 0 | `5481b3c2c32f26d1` |
| legacy | F_reference | 7.17 | 22 | 0.00 | 0.00000 | 0.00 | 0 | 22 | `ad486ffddfbfc68f` |
| v3-validation | D | 146.74 | 11 | 138.85 | 0.84150 | 0.00 | 11 | 0 | `8cf894c4fb8520e4` |
| v3-validation | heuristic | 7.39 | 11 | 0.02 | 0.00010 | 138.80 | 0 | 11 | `22bafd9fd1313f74` |
| v3-validation | E_partial | 6.78 | 11 | 0.04 | 0.00020 | 138.80 | 0 | 11 | `24b1ba271c13eba9` |
| v3-validation | F_partial | 6.59 | 11 | 0.03 | 0.00020 | 138.80 | 0 | 11 | `28eccf7d660fbed2` |
| v3-validation | E_full | 144.06 | 11 | 136.79 | 0.82900 | 0.00 | 11 | 0 | `2c887c7591bda7ba` |
| v3-validation | F_reference | 6.59 | 11 | 0.00 | 0.00000 | 0.00 | 0 | 11 | `e1c123fb06cac2bd` |


Three things must be read off this table carefully:

- **Only `v3-validation` D is a wholly uncached end-to-end measurement.** All 11 of
  its runs built their own extraction, so its wall clock is the defensible
  end-to-end figure: **146.74 s / 165 frames = 0.889 s/frame**, of which 0.8415
  s/frame is tracker time. `legacy` E-full (22/22 built) and `v3-validation` E-full
  (11/11 built) are likewise uncached.
- **`legacy` D is a mix, not a clean end-to-end run.** 20 of its 22 runs built their
  extraction; **2 were cache hits**, whose 28.22 s of inherited build time is
  excluded from the 220.97 s measured. Its 0.6799 s/frame is therefore a measured
  average over a mixed set and must not be quoted as uncached end-to-end cost.
- **Every remaining row reuses the cached extraction.** Its inherited seconds are the
  *build* run's, kept in `timings_inherited_extraction_build` and never added to
  `measured_tracker_seconds`. Reporting a cache-hit row as end-to-end speed would be
  wrong, and the artifacts make that impossible to do by accident. Inherited seconds
  are never summed into this run's cost.

Machine state during the matrix, as observed: load average 9.1 on 10 cores, with two
unrelated long-running `node` processes at 262 % and 167 % CPU. **No slowdown is
demonstrated by the recorded timings.** The previous pass's `legacy` D artifact
(`artifacts/validation/legacy_D.json`) records 241.0 tracker seconds, 0.7416
s/frame, 248.3 s wall; this pass records 220.97 tracker seconds, 0.6799 s/frame,
228.77 s wall — slightly *faster*, not slower. An earlier draft asserted this run
was "about 3× slower"; that ratio is not supported by any recorded artifact timing
and is withdrawn. Contention may well have been present, but it is not measured
here, and no benchmark was re-run to chase it.

## 8. Tests, reproduction, hashes

- **Targeted (this pass's changes):**
  `.venv-video-ai/bin/python3 -m pytest tests/test_ball_v3_worker_pass.py tests/test_ball_v3_isolation.py -q`
  → **68 passed, 18 subtests passed, exit 0**.
- **Genuine full suite:** `.venv-video-ai/bin/python3 -m pytest tests -q` over all
  121 files in `tests/` — result in §8.1 below.
- `artifacts/logs/run_tests.sh` named 8 of those 121 files and called itself the full
  suite. It is renamed `run_v3_subset_tests.sh` and its header now says what it is.
- Repro commands:
  - matrix: `bash artifacts/closeout/run_closeout_matrix.sh`
  - corrected AUC: `.venv-video-ai/bin/python3 tools/ball_v3_train.py recompute-auc --spec artifacts/corrected/auc_spec.json --data artifacts/tracklets/tracklets.json --ranker out/ball_tracker_v3/ranker/ranker.pt --out artifacts/corrected/temporal_auc_recomputed.json`
  - corrected timings: `.venv-video-ai/bin/python3 artifacts/logs/corrected_timings.py`
  - manifest: `.venv-video-ai/bin/python3 artifacts/closeout/build_manifest.py`
  - tables: `.venv-video-ai/bin/python3 artifacts/closeout/report_tables.py`
- Hashes of code, weights, configs, source media and the retained-but-invalid
  artifacts: `artifacts/closeout/artifact_manifest.json`. Its 15 code digests were
  re-verified against the working tree by review 3: **15/15 match**, and its 12
  validation-artifact digests match the files on disk.
- Extraction cache: **66 entry files across 21 distinct clips** (one file per
  `(clip, fingerprint)` pair; up to 6 per clip). An earlier draft said "21 entries",
  conflating files with clips. `declared_sources` in the manifest is keyed by clip
  and its `entries` field counts *distinct source digests* (1 per clip), not files.

### 8.2 Correction ledger (review 3, reporting/provenance only)

No prediction, weight, threshold or metric was recomputed; nothing was re-run except
the focused test files. Each item is a reporting or provenance fix.

| # | What was wrong | Correction | Prior copy |
|---|---|---|---|
| 1 | All **12** `artifact_sha256` values in `matrix_status.jsonl` (and in §7 of this document, and in the worker-3 report table) were recorded *before* `reclassify_runtime.py` rewrote each artifact's runtime block, so every one was stale — e.g. `legacy_D` was filed as `0abf22de…`, actual `5628a834…` | refreshed from the files on disk in both places; each status row keeps the old value as `artifact_sha256_superseded_pre_runtime_repair` with a note | `artifacts/closeout/superseded/matrix_status.superseded-stale-hashes-2026-09-21.jsonl` |
| 2 | `legacy` D described as the clean end-to-end extraction measurement | it is 20 built + 2 cache-hit runs with 28.22 s inherited excluded; `v3-validation` D (11/11 built, 0.889 s/frame wall) is the defensible end-to-end figure — §7 | — |
| 3 | "this run's wall clock was about 3× [the previous pass]" | withdrawn: recorded artifacts show 0.6799 vs 0.7416 s/frame and 228.77 vs 248.3 s wall, i.e. marginally faster. Observed machine load is reported without a fabricated ratio — §7 | — |
| 4 | `switches` and `body_lock_rate` read as identity/rejection measurements | relabelled as geometric jump and torso-overlap proxies, with source lines — §6, §6.1 | — |
| 5 | "21 cache entries" | 66 entry files across 21 clips — §8 and the manifest's new `extraction_cache` block | — |
| 6 | Framewise caveat stated once in the variant table only | restated at the point of use in §6.1: no genuinely frame-independent ablation was run | — |

Counts that are correct and are easy to conflate, kept straight here: tracklet store
**204 shards / 189 nonempty / 119 distinct sequences** (§9); matrix **12 rows**;
sealed **10** clip ids; `v3-validation` **11** clips.

### 8.1 Full-suite result

`.venv-video-ai/bin/python3 -m pytest tests -q` over all of `tests/`, run once,
**after** the targeted run and after the matrix:

```
1373 passed, 2 skipped, 2 warnings, 195 subtests passed in 251.60s (0:04:11)
PYTEST_EXIT=0
```

The two skips are pre-existing and unrelated to this pass; there were **no
failures**, and the exit status was captured from pytest itself (no pipe). The two
warnings are pre-existing deprecations (`cgi` in `ghostcaddie/upload/server.py`, and
a test returning a value).

## 9. Data, splits and known optimism

- Tracklet store: 204 shards, 189 nonempty, **119 distinct `sequence_id`s**; 776
  tracklets; **241 of 776** have an all-zero body vector and **9 of 34** clips are
  bodyless — reported as a **proxy for missing pose, not proof**: a candidate far
  from every landmark is legitimately zero, and this store predates the availability
  flag, so missingness is **unknown vs zeros**, not measured.
- Ranker exposure, measured (`artifacts/corrected/ranker_exposure.json`): on the
  family split **18 of 24** validation clips are ranker-training clips, carrying
  **415 of 570 rows (72.8 %)**; on the clip split 0 of 155. Neither rule is a clean
  end-to-end family-disjoint evaluation, and every family-split AUC is
  **ranker-optimistic**. Unfixable without retraining the ranker, which this pass is
  not authorised to do.
- `cycle_agreement` is exactly 1.0 over all 776 tracklets (one distinct value): a
  constant, kept because removing it would change an already-saved model's input
  width, and named here so it is not mistaken for evidence.

## 9.1 What is deliberately not measured

| not measured | why |
|---|---|
| identity recall, false-track rate, lock rate on `v3-validation` | the corpus has no labels; inventing them is forbidden |
| body-missingness rate as a fact | unknown-vs-zeros, see above |
| canonical BALL / CLUBHEAD behaviour | BALL is `UNAVAILABLE` / `method: none`, CLUBHEAD is `UNAVAILABLE`; the research clubhead path was falsified |
| physical metrics (speed, spin, launch) | no calibration exists; nothing here is a physical claim |
| real-ball accuracy of any AUC above | the candidate lists are synthetic; the real-labelled corpus is development-consumed |


## 10. Final test

**V3 FINAL TEST: SEALED / NOT RUN.** No decode, view, inference, label, derivative
or cache read on any of the 10 sealed clips in any pass, including this one; the
only read of that corpus is `split.json`, for the sealed-id exclusion list, and the
validator refuses a sealed id that arrives from anywhere else in the plan (§2.4).
`artifacts/validation/final_test_audit.json` records the audit.

Re-verified independently by review 3: the 10 sealed ids from
`data/ball_dataset_v3_eval/split.json` were searched across **all 113 files** of
`artifacts/closeout/` — the 12 validation artifacts, the status file, all **66**
extraction-cache entries, the manifest, every driver script and every per-run log.
**Zero occurrences.** **No promotion, freeze or canonical integration is authorized
by this document.**
