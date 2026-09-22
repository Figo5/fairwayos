# Ball tracker v3 — worker pass 1

> **SUPERSEDED — history, not current evidence.** Several headline claims below
> were wrong and are struck where they appear; three sections were never written.
> **`docs/ball-tracker-v3-results.md` is the current authoritative evidence
> document.** Read that one for every number; read this one only to see how the
> work was done, what was invalidated, and why. Worker pass 3 (closeout) completed
> the validation matrix, regenerated the corrected AUC artifact with the invalid
> `as_deployed_no_mask` column nulled, and replaced the remaining `<!-- FILL -->`
> sections with that document's content.

Branch `research/ball-tracker-v3-context`, baseline HEAD
`a6b08281062cb591b652a9e477c9c862baea79fe`. This is the implementation worker's
bounded pass: runtime and candidate-stack simplification, completion of the
tracklet data, one lightweight temporal discriminator, and validation only.

**Nothing here is a promotion, a freeze, or a final test.** The v3 final test is
sealed and was not consumed — §9 has the execution inventory that backs that
claim. Canonical `ball` remains `UNAVAILABLE` / `none` and canonical `CLUBHEAD`
remains `UNAVAILABLE`; no canonical code imports any of this.

Artifacts for this pass live under
`/private/tmp/fairway-v3-orchestration.jQ7Tk6/artifacts/`; every SHA-256 quoted
below is of a file there or of a pre-existing file under `out/`, named in full.

---

## 1. What was inspected, and the four defects it turned up

Read before changing anything: `docs/ball-tracker-v3-plan.md`,
`docs/current-research-handoff.md`, the eight `ball_v3_*` modules, the four
`tools/ball_v3_*` tools, the seven v3 test files, and the existing
`out/ball_tracker_v3/` evidence. There is no `AGENTS.md`, `CLAUDE.md` or
`.cursorrules` in the repository.

Existing evidence, hashed:

| artifact | bytes | SHA-256 (first 16) |
|---|---|---|
| `out/ball_tracker_v3/ranker/ranker.pt` | 725,769 | `b2c4e1be8d1ccd3b` |
| `out/ball_tracker_v3/ranker/ranker_report.json` | 6,419 | `0e0c6809cc9b69ab` |
| `out/ball_tracker_v3/tracklets.json` | 93,895 | `a161a5726fef0947` |
| `out/ball_tracker_v3/listwise_train/stacks.u8` | 114,917,376 | `2b2e6fc39a088d08` |
| `out/ball_tracker_v3/proposal_methods.json` | 2,825 | `87affa592ccecc09` |

Split isolation was verified and is clean: the legacy human-read annotations share
no clip with the v2 train split, the v3 validation split or the v3 final-test
split; the listwise store's clips share none with v3 validation or final test.

Four defects in the incoming state, all of which the measurements below depend on:

**D1 — ranker-to-temporal leakage.** All six clips the ranker held out
(`pexels_37995797`, `pexels_37995798`, `pexels_37995802`, `pexels_6541448`,
`pexels_6541451`, `pexels_6573378`) appear in `out/ball_tracker_v3/tracklets.json`,
and both `gen-tracklets` and `train-tracklet` re-split by their own RNG. The
ranker's probability is a *feature* of the temporal model, so a temporal
validation clip the ranker trained on makes that feature optimistic and a
temporal training clip the ranker held out makes it pessimistic. The temporal
split is now a shoot-family split aligned to the ranker holdout (§6).

**D2 — dummy features in the training data.** `cmd_gen_tracklets` never set
`Node.torso_centre`, so `moves_with_torso` was 0.0 for every training tracklet;
and `Node.cycle_ok` was never set, so `cycle_consistency` was the constant 1.0.
Both were zero-by-absence. Torso centres now come from the pose cache,
`cycle_agreement` is measured from the track's own geometry, and `node_gaps()`
makes an unset field detectable — the tracklet builder refuses to write a
tracklet that has one.

**D3 — tracklets grouped by frame adjacency.** `cmd_gen_tracklets` grouped stored
lists into "runs of consecutive frames". Two synthetic insertions on one clip
whose windows touch on the frame axis were therefore linked into a single track
whose truth was the union of two different balls. The store recorded no sequence
id, so this could not be repaired from the existing store: it needed a
regeneration, which is part of what §5 does.

**D4 — body-feature coverage and a cached final-test derivative.**
`pexels_5428844` has 0 of 18 needed pose-cache frames and `pexels_32742697` 4 of
13, so body features were absent for part of the old store.
`out/ball_tracker_v3/body_cache/` also holds pose entries for all 21 v3
evaluation clips, including the 10 final-test ones, written by an earlier pass.
None of those was opened: the seal guard runs before any clip name reaches the
cache.

---

## 2. Runtime conditions

Recorded by `tools/ball_v3_bench.py env` and embedded in every artifact this pass
wrote:

```
macOS 15.7.9, arm64, 10 CPUs, Python 3.11.16
OpenCV 5.0.0, 10 threads (cv2.getNumThreads), useOptimized True
numpy 2.4.6, torch 2.13.0, torch threads 6, MPS built and available
```

Interpreter: `.venv-video-ai/bin/python3` (torch/CV). The MoveNet pose cache is
consumed read-only from the earlier `.venv` pass; no pose inference was run here.

Every timing below states whether it is **cold** (first pass, decode included,
allocator and page cache cold) or **warm** (repeat pass). Cold and warm differ by
about a factor of two on a single pass and are never mixed in one column.

The dev corpus for the proposal measurements is the legacy human-read corpus,
restricted to the v2 detector's trained regime (labelled ball radius ≤ 40 px) and
to the first 12 labelled frames of each of the 7 clips that have any: 84 labelled
frames, the same set `tools/ball_v3_proposals_eval.py` uses. This corpus is
**development-consumed** — an earlier pass scored it and this pass used it to
choose parameters. It is never reported as held-out accuracy.

---

## 3. Baseline, measured fresh

`artifacts/profile/components_baseline.json` — 7 clips, 43 frames, repeats = 2,
`work_max_side` 2880, all six generators, warm and cold separated. Seconds per
frame, warm:

| stage | s/frame |
|---|---|
| `build_contexts` | 0.0325 |
| — of which `phaseCorrelate` | 0.0280 |
| — resize | 0.0011 |
| — grey + float cast | 0.0017 |
| `radial_symmetry` | 0.3303 |
| `local_contrast` | 0.9975 |
| `frame_difference` | 0.0211 |
| `flow_residual` | 0.1461 |
| `brightness_cue` | 0.4285 |
| `local_distinctness` | 0.1926 |
| dedupe + caps | 0.0041 |
| decode (separate, cold) | 0.047 |

All six generators plus contexts: **4.305 s/frame**. The current default pair
(`local_distinctness` + `brightness_cue`) plus contexts: **0.654 s/frame**.

### 3.1 The incoming timings, independently reproduced

The handoff carried `all six 4.93 s/frame, the v2 five 4.56, chroma+brightness
1.38, chroma-only 0.50`, with recall `0.738/0.321/0.738/0.738`. Those figures did
not record thread counts, resolution or machine contention, and one of them is
smaller than the sum of its own per-generator costs, so they were re-measured
rather than compared against.

`artifacts/validation/sets_legacy.json` (the pre-optimisation path,
`--legacy-proposals`), same 84 frames:

| generator set | recall (found/N) | proposals/frame | s/frame (prop) |
|---|---|---|---|
| all six | **0.7381** (62/84) | **139.0** | 4.109 |
| the v2 five | **0.3214** (27/84) | **62.0** | 3.679 |

The recalls and the proposal counts reproduce the recorded values exactly
(`0.7381`, `0.3214`, `139.0`, `62.0`). The *timings* do not: a fresh measurement
on an idle machine is 17–23% faster than the recorded 4.93 and 4.56, which is
consistent with the recorded run sharing the machine with another workload — the
hazard the brief warned about. **The old timing figures are not used anywhere in
this document.**

---

## 4. Optimisation

### 4.1 What was changed, and why only these

Measured bottlenecks, in order: `local_contrast` (0.998 s/frame),
`brightness_cue` (0.429), `radial_symmetry` (0.330), `local_distinctness` (0.193),
`flow_residual` (0.146), the camera-motion FFT (0.028 — more than the whole
`local_distinctness` generator, and read by neither of the two colour generators),
and downstream the per-candidate context crop (10.4 ms per candidate, §4.4).

Two classes of change:

**Exact-equivalent** — same values, fewer operations.
`_peaks` uses `np.maximum` instead of copy-then-mask. `radial_symmetry` hoists the
radius-independent vote field (the strong-gradient mask, the gathers, the unit
vectors, the normalisation) out of the eight-iteration per-scale loop where it was
recomputed identically every time. `build_contexts` gained `motion="eager"|"lazy"`
and `ball_v3_proposals.propose` requests `lazy` whenever no requested generator
reads `FrameContext.motion`.

**Measured approximations, opt-out** — a shared `ImagePyramid` (successive
halving) now backs `local_contrast`, `brightness_cue`, `radial_symmetry` and
`local_distinctness`. The argument is the textbook scale-space one: a Gaussian
with sigma 119 on a 2880-px frame is the same filter as sigma 4 on a 1/30-scale
level, only slower. Each generator keeps its pre-optimisation full-resolution path
behind `pyramid=False`, and `ProposalParams.pyramid` switches the whole proposal
stage, so the "before" column below is the old behaviour re-measured, not a
recollection of it.

`flow_residual` and `frame_difference` were left alone: they are not scale-space
filters, they are not in the default set, and their cost is what it is.

### 4.2 Matched before/after, component by component

`artifacts/profile/components_baseline.json` (before) and
`artifacts/profile/components_after.json` (after). Same 7 clips, same 43 frames,
same `repeats=2`, same threads, warm:

| stage | before | after | speed-up |
|---|---|---|---|
| `radial_symmetry` | 0.3303 | 0.0648 | **5.10×** |
| `local_contrast` | 0.9975 | 0.0601 | **16.60×** |
| `brightness_cue` | 0.4285 | 0.1139 | **3.76×** |
| `local_distinctness` | 0.1926 | 0.1642 | 1.17× |
| `flow_residual` | 0.1461 | 0.1505 | 0.97× |
| `frame_difference` | 0.0211 | 0.0189 | 1.12× |
| `build_contexts` | 0.0325 | 0.0379 | 0.86× |
| dedupe + caps | 0.0041 | 0.0050 | 0.82× |
| **all six + contexts** | **4.3050** | **1.2059** | **3.57×** |
| **legacy five + contexts** | **3.883** | **0.845** | **4.60×** |
| **default pair + contexts** | **0.654** | **0.316** | **2.07×** |
| **distinctness only + contexts** | **0.225** | **0.202** | **1.11×** |

The honest reading: the **full candidate stack** and the **legacy five** got a
3.6–4.6× improvement; the **already-reduced default pair** got 2.1×, and the
single-generator `local_distinctness` only 1.17×. A 5× claim on the default pair
is not supported by these numbers and is not made. The brief anticipated exactly
this case: the reduced baseline was already fast, so the ratio is reported as
achieved rather than manufactured.

### 4.3 Pipeline-level before/after, with recall

84 dev frames, matched, `repeats=1` (so these are cold single-pass numbers
including decode; the component table above is the warm view).
`artifacts/validation/sets_legacy.json` vs `artifacts/validation/sets_after.json`:

| set | recall before | recall after | proposals/f before → after | s/f prop before → after | speed-up | e2e s/f after |
|---|---|---|---|---|---|---|
| **A** distinctness only | 0.7381 (62/84) | 0.7500 (63/84) | 95.5 → 95.5 | 0.381 → 0.306 | 1.25× | 0.330 |
| **B** distinctness + brightness | 0.7500 (63/84) | 0.7500 (63/84) | 112.5 → 113.0 | 1.238 → 0.531 | **2.33×** | 0.555 |
| **C** legacy five (the v2 stack) | 0.3214 (27/84) | 0.3929 (33/84) | 62.0 → 105.1 | 3.679 → 0.823 | **4.47×** | 0.846 |
| **C2** all six | 0.7381 (62/84) | 0.7500 (63/84) | 139.0 → 179.6 | 4.109 → 1.116 | **3.68×** | 1.139 |

Recall is preserved or slightly improved in every set. A and B were also measured
in legacy mode on the same frames (`artifacts/validation/sets_legacy_ab.json`:
A 0.738/95.5/0.437, B 0.750/112.5/1.265), which is where the before column for
those two rows comes from; the historical record lists distinctness-only at 0.7381
and distinctness+brightness at 0.7381, so A reproduces it exactly and B is one
frame better on this machine.

**The recall figures are proposal-stage ceilings on a development-consumed
corpus. They are not accuracy and they are not held-out.**

### 4.4 Downstream: the crop was the bottleneck, not the network

`artifacts/profile/downstream.json` (before) and
`artifacts/profile/downstream_after.json` (after), 60 stored lists, 704
candidates, ranker `b2c4e1be…` on MPS:

| stage | before | after |
|---|---|---|
| context stack, per candidate | 10.45 ms | **4.27 ms** (2.45×) |
| rank inference, per candidate | 0.95 ms | 0.74 ms |
| rank inference, per frame | 11.4 ms | 8.8 ms |
| tracklet linking | 3.5 ms / 8 tracks | 0.8 ms / 56 tracks |
| tracklet features | 22.3 ms total | 20 ms total |

Building a candidate's 7-channel two-scale stack cost **eleven times** what the
ranker cost. `candidate_stack` rebuilt each crop from the full-resolution frame:
the context window is 32 ball radii, so a 40 px ball asked for a 1280 px crop that
INTER_AREA then averaged down to 64. `FrameViews` now takes each crop from a
shared pyramid level about twice the output grid wide. Crops at or below 128 px —
every ball up to radius 4, and both crops for any small ball — are bit-identical
to the old path; above that the context crop is resampled from a level, measured
mean per-channel difference < 2/255 and the tight crop unchanged
(`tests/test_ball_v3_worker_pass.py::FrameViews`).

### 4.5 How much the pyramid changed the candidates

`artifacts/profile/equivalence.json`, per generator, pyramid on vs off, 28
labelled frames, cold single pass:

| generator | candidates/frame on/off | recall on/off | positions matched within 1 px | max score delta |
|---|---|---|---|---|
| `radial_symmetry` | 96.0 / 96.0 | 0.500 / 0.536 | 954 / 2688 | 0.0 |
| `local_contrast` | 65.2 / 60.3 | 0.393 / 0.393 | 720 / 1827 | 0.0 |
| `brightness_cue` | 71.8 / 65.9 | 0.357 / 0.393 | 438 / 2010 | 0.129 |
| `local_distinctness` | 96.0 / 96.0 | 0.750 / 0.714 | 2084 / 2688 | 0.097 |

`local_distinctness` was separately compared against the *original module loaded
from git* (`artifacts/logs/compare_original.py`): **bit-identical**, 0 position
and 0 score differences over 13 frames of `pexels_6541842`. The pyramid changes
its output only through the halving chain's ±1 px level rounding, and the net
effect on the 84-frame dev set is one frame gained.

The pyramid is not a pure speed-up for the others, and the reason is worth
stating: it also changes the peak-suppression neighbourhood. At full resolution
the coarsest `radial_symmetry` scale suppressed peaks within 119 px of each other;
on a level it suppresses within about 8. That admits more candidates, which is why
C's proposals per frame rise from 62 to 105 and C2's from 139 to 180 — and it is
where C's +6 frames of recall on `pexels_37030686` come from. The trade is
explicit: **more proposals for more recall, at a downstream cost that §4.4
reduces.** Per-clip regressions are single frames (B loses 1 frame on
`pexels_37080178` and gains 1 on `pexels_6541842`; net zero).

Two behavioural differences were found and fixed rather than reported as noise:
`brightness_cue` used `int(round(radius))` for its suppression radius where the
pre-optimisation code truncated (`int(radius)`), which moved the legacy candidate
count from 104.7 to 112.5 per frame; and `ImagePyramid.at` in `direct` mode
resized twice. Both are corrected, so the "before" column is the old behaviour
exactly.

---

## 5. Tracklet data

The old store was 334 lists over 32 clips with no sequence id and no round, and
`out/ball_tracker_v3/tracklets.json` was 227 tracklets over 27 clips. Defect D3
made it unusable for training a temporal model without silently mixing two
different synthetic balls into one track, so the data was regenerated rather than
patched.

**Shape of the store now.** One shard per `(round, clip)`, each shard one
synthetic sequence, written by `ball_v3_dataset.ExampleWriter` with a
`manifest.json` written last. A reader skips a directory with no manifest, so an
interrupted round loses the shard it was in and nothing else, and the grouping key
for tracklets is the shard — never frame adjacency. Each record carries
`sequence_id`, `round_index`, `source_sha256`, `split`, `source_group`,
`synthetic_kind`, `synthetic_seed`, `first_frame`, `frame_spacing`, the exact
synthetic truth and the frame's full proposal-pool size.

**Cost and disk, measured before committing.** A pilot (`--pilot`, 3 clips, one
round) took 74.1 s and 4.7 MB for 3 shards — about 25 s and 1.6 MB per shard on
4K backgrounds. The bounded full run was then six rounds over the 34 v2 train
clips that have media on disk:

| quantity | value |
|---|---|
| shards written | 204 |
| shards that produced any list | 189 |
| stored lists | 1,422 |
| of which null (no ball on the frame) | 367 |
| proposal misses on synthetic ball frames | 189 |
| clips | 34 |
| rounds | 6 |
| wall time | 2,886 s (48.1 min) |
| disk | 517 MB |
| `run_manifest.json` code digest | `2c6cdb30da9672d0` |

Pilot to full run scaled as predicted, which is the point of measuring it first.

**Tracklets built from it** (`artifacts/tracklets/tracklets.json`, 680 KB):

| quantity | value |
|---|---|
| tracklets | 776 |
| positives (on the inserted ball ≥ 60% of scored frames) | 136 |
| negatives | 640 |
| sequences | 189 |
| clips | 34 |
| tracklets dropped for unset node fields | **0** |
| length distribution | 3: 204, 4: 109, 5: 68, 6: 51, 7: 54, 8: 23, 9: 267 |
| weights digest of the ranker used | `b2c4e1be8d1ccd3b` |
| `code_sha256` | `f2aa0c480714a260` |

Zero dropped for unset node fields is the check that D2 is actually closed: the
builder refuses to write a tracklet whose caller-supplied fields were never
filled, and none was refused.

**Split.** `family_split()` groups clips by `source_group` and puts every family
containing a ranker-held-out clip into temporal validation. This is what closes
D1: the temporal validation clips are exactly the families the ranker never saw.

It is also **coarse**, and that has to be said plainly: the v2 `source_group`
prefixes are four digits, so six held-out clips pull **24 of the 34 clips** into
validation. The strict split is 206 training tracklets (36 positives) against 570
validation tracklets (100 positives). A `clip`-rule split is provided as well
(621 train / 155 validation, 109 / 27 positives) which is balanced and still
ranker-optimism-free, but lets families straddle the split. Both are reported in
§7; neither is presented as the right one.

**Supervision audit, stated rather than implied.**

- Positives are exact synthetic geometry: a tracklet is positive when it sits
  within one ball radius of the inserted ball on at least 60% of its scored
  frames. This is exact truth about the *inserted* ball.
- Negatives are "not the inserted ball". A real ball elsewhere in the frame would
  be labelled negative. The v2 train clips carry **no human ball annotations**
  (verified: the legacy annotation set is disjoint from the v2 train split), so
  this is **not** exhaustive ground truth about balls, and the store records that
  in `negative_semantics`.
- Null tracklets (frames with no ball before entry, after exit, under full
  occlusion) are kept and counted; they are not skipped.
- No agent-read or model-generated label is called human ground truth anywhere,
  and no v3 evaluation clip enters training.

---

## 6. Feature inventory: computed, or explicitly unavailable

`btrk.FEATURE_NAMES` is now 27 features in six named ablation groups plus
`support_fraction`. Everything in the list is computed from the tracklet's own
nodes; nothing is a constant.

| group | features | source |
|---|---|---|
| — | `support_fraction` | frames held / frames in the window |
| `ranking` | `mean_rank_probability`, `max_rank_probability`, `mean_rank_margin`, `top1_fraction`, `mean_rank_position`, `competitor_pressure` | the listwise ranker's own softmax, and the strongest same-frame candidate that is *not* this track |
| `motion` | `mean_step_radii`, `std_step_radii`, `max_step_radii`, `mean_abs_accel_radii`, `direction_consistency`, `static_fraction`, `cycle_consistency` | camera-compensated geometry; `cycle_consistency` is now the measured forward/reverse link agreement (§6.1) |
| `body` | `body_inside_fraction`, `moves_with_torso`, `mean_distance_to_hands`, `torso_relative_speed` | cached MoveNet pose, consumed as features, never as an exclusion region |
| `appearance` | `tight_mean_intensity`, `tight_std_intensity`, `context_contrast` | the candidate's own stored 7-channel stack |
| `chroma` | `chroma_score_mean`, `chroma_score_max`, `brightness_score_mean`, `brightness_score_std` | the `local_distinctness` and `brightness_cue` generator responses |
| `scale` | `radius_stability`, `radius_over_short_side` | geometry |

**Unavailable, with the reason, and deliberately absent from the vector rather
than present as a zero** (`btrk.UNAVAILABLE_FEATURES`):

- `club_lock`, `club_relative_velocity`, `club_candidate_agreement` — no clubhead
  tracker exists on any path. Canonical `CLUBHEAD` is `UNAVAILABLE` and the
  research clubhead path was falsified, so there is no club geometry to lock a
  candidate to. A club-lock *rate* is therefore reported as `null` with a reason
  everywhere in this pass, never as `0.0`.
- `crop_embedding` — the tight and context crops are summarised by three scalar
  statistics; a learned embedding is not carried here because the listwise ranker
  already reads the same crops through a CNN.

### 6.1 `cycle_consistency` was a constant; now it is measured

It was 1.0 for every tracklet, because nothing set it. It is now
`Tracklet.cycle_agreement`: for each consecutive pair `(a, b)` that has a
following node `c`, `a` is predicted backwards from `b` using the velocity of the
*next* step (`b` to `c`), and the prediction is compared with where `a` actually
is. Using the next step's velocity rather than the pair's own is what makes it a
check rather than a tautology.

It is still reported and never gated on, and the measurement says why: with a link
tolerance of eight radii, almost any track that survives linking is smooth enough
to round-trip, so the feature is near-constant in practice. That is the same
finding the v2 pass made when every final-test clip scored
`cycle_consistency_rate = 1.0` — including three that tracked pure clutter. It is
kept, measured, and documented as weak rather than deleted or trusted.

**Measured, and it is constant.** Over all 776 real tracklets,
`cycle_agreement` is exactly 1.0 — one distinct value, standard deviation 0.0.
That is not a bug in the measurement; it is the measurement. With a link
tolerance of eight radii, a track only survives linking if its geometry is smooth
enough to round-trip, so the reverse check passes by construction. The feature
carries **no information** on this data and is reported as such. It is kept in
the vector because removing it would silently change the input width of an
already-saved model, and it is listed here so no reader mistakes it for
evidence. §7 shows what that costs: it is one of several features that contribute
nothing.

The same table for the other near-constant features: `moves_with_torso` reaches a
maximum of 0.485 and averages 0.038, because the torso-velocity floor
(`torso_motion < 0.5`) skips almost every frame — on most frames the golfer is
barely moving, so there is no torso motion to agree or disagree with. Its
positive/negative means are 0.046 and 0.036. It is a real measurement of a real
quantity and it is nearly uninformative on this corpus; that is stated rather than
papered over by rescaling it.

---

## 7. Temporal discriminator

The architecture is unchanged from the existing lightweight MLP
(`TrackletDiscriminator`: 27 → 64 → 64 → 1, 6,017 parameters) — the brief prefers
it unless measurements justify a bigger model, and §7.3 is the reason they do not.

Every run records its data, config, code and weight digests
(`artifacts/temporal/tracklet_report*.json`). All ten runs share
`data_sha256 c6d3e8803312630a` and `code_sha256 a87728c9ce71f15e`, so they are
comparable.

### 7.1 Model selection

> **WITHDRAWN.** Every `best val AUC` in this table came from the tie-blind
> estimator (it ranked with `argsort` over positives-then-negatives and broke every
> tie in the positive class's favour), so the column, the `best epoch` it selected
> and the F-vs-D comparison below are all invalid. The tie-correct, mask-applied
> scores on these same fixed checkpoints are in §13.3 and in
> `docs/ball-tracker-v3-results.md` §5. The table is kept only to show what was
> reported at the time.

| run | split | dropped group | train / val | train / val positives | best val AUC (tie-blind, invalid) | best epoch |
|---|---|---|---|---|---|---|
| **D** full | family | — | 206 / 570 | 36 / 100 | ~~0.9778~~ | 94 |
| **D** full | clip | — | 621 / 155 | 109 / 27 | ~~0.9835~~ | 299 |
| **E** retrained | family | body | 206 / 570 | 36 / 100 | ~~0.9838~~ | 66 |
| **E** retrained | clip | body | 621 / 155 | 109 / 27 | ~~0.9910~~ | 316 |
| **F** retrained | family | motion | 206 / 570 | 36 / 100 | ~~0.9820~~ | 186 |
| **F** retrained | clip | motion | 621 / 155 | 109 / 27 | ~~0.9980~~ | 163 |
| probe | family | ranking | 206 / 570 | 36 / 100 | ~~0.8697~~ | 67 |
| probe | family | appearance | 206 / 570 | 36 / 100 | ~~0.9713~~ | 13 |
| probe | family | chroma | 206 / 570 | 36 / 100 | ~~0.9683~~ | 67 |
| probe | family | scale | 206 / 570 | 36 / 100 | ~~0.6742~~ | 20 |

### 7.2 ~~The result that matters: dropping the temporal features does not hurt~~

> **WITHDRAWN — the claim inverts on a correct estimator.** Struck in place rather
> than corrected elsewhere. The tie-correct family split reads **D 0.9761 vs
> F 0.9641**, i.e. removing the motion group *costs* 0.0120, and F's own 437 of 570
> tied scores mean much of that split is unresolved. The no-scale probe's 0.6742
> (~~and 0.5090 tie-correct~~) is a **collapsed constant model**, not evidence about
> scale. See §13.3 and `docs/ball-tracker-v3-results.md` §5.

~~**F (no motion features) scores 0.9820 against D's 0.9778 on the family split,
and 0.9980 against 0.9835 on the clip split.**~~ Removing every camera-compensated
motion, velocity, acceleration, direction, static and cycle feature does not
reduce validation AUC. ~~On this data the temporal model is not using temporal
evidence.~~ The comparison that sustains any version of this reading is in §13.3,
and it does not sustain the original wording.

The ablation that does move the number is `ranking` (0.8697) and `scale`
(~~0.6742~~ 0.5090, collapsed). The model is mostly reading the ranker's own
per-frame opinion and the candidate's radius.

Single-feature AUCs on the same validation split quantify it
(`artifacts/logs/feature_auc.py`):

| feature or group | AUC, family split | AUC, clip split |
|---|---|---|
| `mean_rank_margin` alone | 0.9616 | 0.9881 |
| `top1_fraction` alone | 0.9584 | 0.9899 |
| `mean_rank_probability` alone | 0.9434 | 0.9893 |
| the four ranker features averaged | 0.9501 | 0.9942 |
| the temporal (motion) features averaged | 0.6895 | 0.7078 |
| every non-ranker feature averaged | 0.4363 | 0.5122 |
| the full MLP (D) | 0.9778 | 0.9835 |

The ranker's own score, with no temporal model at all, already separates the
validation set at 0.95. The MLP adds about 0.03 over that on the family split.
That is the honest size of the temporal contribution on this data, and it is
within the run-to-run noise of the training curve (loss peaks at 21–45 across
runs at `lr = 0.05`, and `best_epoch` ranges from 13 to 316).

### 7.3 Ranker-to-temporal leakage and shortcut risk

D1 was a split-alignment defect and it is closed: the temporal validation
families are exactly the families the ranker never trained on, so the ranker
feature is not *optimistically* biased by the split. But closing that leak
exposed a bigger one that no split can fix:

- the ranker's probability is not a weak side-feature, it is the dominant signal
  (0.95 AUC on its own);
- the ranker was trained on **synthetic insertions on real backgrounds**, and its
  candidate lists are built the same way;
- the temporal model is therefore trained and evaluated on the same synthetic
  insertion process, and mostly learns to agree with the ranker about it.

So this pass has **not** demonstrated that temporal discrimination adds anything.
It has measured that it does not, on this data, and identified why. A temporal
model can only be shown to add value against supervision that the ranker cannot
already answer — which means real labelled tracks, and the repository has none
beyond the 325 development-consumed legacy frames.

`radius_over_short_side` is worth flagging separately as a suspected
synthetic-insertion shortcut: the synthetic generator draws ball radii from a
distribution, so "small relative to the frame" partly encodes *inserted*, not
*ball*. It is in the `scale` group whose removal costs the most AUC (0.6742),
which is a reason for suspicion, not proof.

---

## 8. Validation

Three evidence classes, kept apart everywhere:

1. **synthetic insertion accuracy** — the ranker's 86.6% top-1 and the temporal
   AUCs above. This is candidate-list accuracy on synthetic balls on real
   backgrounds. It is **not** real detection accuracy and no real accuracy may be
   inferred from it.
2. **real-labelled development evidence** — the 325 human-read frames in
   `data/ball_annotations/`, development-consumed. §8.1.
3. **untouched real-video behaviour** — the 11 v3 *validation* clips, which carry
   no labels, so only behaviour is measurable. §8.2.

The v3 final test is a fourth class and is not measured here at all (§9).

### 8.1 Real-labelled development evidence (development-consumed)

`tools/ball_v3_validate.py legacy`. 22 consecutive-frame runs over the 10 legacy
clips, 325 frames, 185 labelled-visible and 136 labelled-absent. Definitions are
printed in each artifact:

- **identity recovery** — OBSERVED and within `max(6 px, 1.0 × labelled radius)`
  of a visible, unambiguous human label.
- **false track** — OBSERVED on a frame the label marks absent, or OBSERVED
  further than one ball radius from a visible label.
- **correct abstention** — the fraction of labelled-absent frames not called
  OBSERVED.
- **switch** — a consecutive OBSERVED pair more than `LINK_RADII × radius` apart
  after camera compensation: a jump to a different object, not a ball that moved.

| run | accepted runs | OBSERVED frames | coverage | identity recovery | in-regime recovery | observed on absent | on wrong object | correct abstention | switches | body-lock rate | club-lock rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **D** full temporal model | 11 / 22 | 58 / 325 | 0.178 | **0 / 185** | **0 / 127** | 23 / 136 | 35 / 185 | 0.8309 | 0 | 0.0 | null |
| no temporal model | 7 / 22 | 55 / 325 | 0.169 | **0 / 185** | **0 / 127** | 23 / 136 | 31 / 185 | 0.8309 | 0 | 0.0 | null |

**The tracker recovers the real ball on none of 185 labelled-visible frames,
in-regime or out.** It nonetheless accepts a track in 11 of 22 runs and emits 58
observed coordinates, 35 of which sit on a different object than the labelled
ball and 23 of which sit on frames the label marks absent. This is the v2 failure
mode repeating: confident clutter tracks accepted on real footage, and it is
consistent with the incoming handoff's own finding that the golfer's own body is
the unsolved distractor.

The proposal ceiling on the same corpus is 0.75 (§4.3) — the ball *is* proposed on
63 of 84 in-regime frames — so the loss is in the ranking and decision stages, not
in proposal generation. That is a materially different diagnosis from the v2
handoff's "31% proposal recall", and it is the more useful one.

`body_lock_rate` is 0.0 on 58 observed frames. It is a real measurement now (the
pose cache is populated for these clips) rather than v2's zero-by-absence, and
its being zero says the accepted clutter tracks were not inside the torso polygon
— not that body rejection works. `club_lock_rate` is `null` with the stated
reason: no clubhead tracker exists.

### 8.2 Untouched real-video behaviour (no labels, no accuracy claim)

> **Written in worker pass 3 — see `docs/ball-tracker-v3-results.md` §6**, which
> carries the `v3-validation` matrix (11 runs, 165 frames): coverage, abstention,
> body-lock diagnostics and wall time, with identity recall, false-track rate and
> lock rate `null` because the corpus has no labels. `legacy` (human-labelled,
> development-consumed) runs sit in the same document with their denominators.
<!-- Worker pass 3: the FILL placeholder above was removed; the content lives in the results document. -->

### 8.3 Inference masking versus retrained ablations

Two different questions, and they are answered by two different mechanisms:

- **inference masking** (`--mask body`, `--mask motion`) zeroes named feature
  groups at inference on one already-trained model. It asks *does this model's
  output depend on these inputs?*
- **retrained ablations** (`--drop body`, `--drop motion`) train a model without
  those features at all. It asks *can the remaining features carry the signal?* A
  masked checkpoint's mask is now stored with the weights and applied inside
  `forward`, so an ablation cannot silently run as the full model (§13.2).

> **Written in worker pass 3 — see `docs/ball-tracker-v3-results.md` §6.** The
> inference-masked runs are labelled partial (one feature group zeroed; association
> and every other temporal feature retained) and the framewise reference is labelled
> *no temporal model or association; ranker motion-residual context retained*.
<!-- Worker pass 3: the FILL placeholder above was removed. -->

## 13. Worker pass 2 — corrections and invalidation manifest

Independent review 1 (`reviewer-1-report.md`) returned **REJECT** for freeze
readiness. Three headline conclusions in this document were wrong rather than
unpolished, and §8.2/§8.3/§10 were unwritten. This section states what was
corrected, what is now invalid, and what is still missing. **The invalid results
are preserved, not erased**: every original artifact and every original section
above still exists, and the supersedes records are in
`artifacts/corrected/`.

### 13.1 Claims in this document that are now withdrawn

| Claim | Where | Status | Correction |
|---|---|---|---|
| "F (no motion features) scores 0.9820 against D's 0.9778 … the temporal model is not using temporal evidence" | §7.2 | **REVERSED** | tie-corrected: D **0.9761** > F **0.9641**. The original estimator ranked ties in the positive class's favour; 437 of F's 570 validation scores were tied. The weaker statement (the ranker dominates) survives; this one does not |
| "the no-scale model dropped to 0.674, so scale carries the most" | §7.1/§7.2 | **WITHDRAWN** | tie-correct 0.5090 with **567 of 570 scores tied** — the model collapsed to a constant. That is a training failure, not a feature-importance measurement |
| "the temporal validation families are exactly the families the ranker never trained on" (D1 closed) | §5 | **WRONG** | 18 of the 24 temporal-validation clips **are** ranker training clips, carrying **415 of 570 validation rows (72.8 %)**. Every family-split AUC is ranker-optimistic on those rows. Only ranker retraining could fix it, which is not authorised |
| "every feature computed, nothing constant" / cycle agreement as evidence | §6.1 | corrected | `cycle_agreement` is exactly 1.0 for all 776 rows; it is dead weight, and it is now described as dead weight |
| "the loss is in the ranking and decision stages, not in proposal generation" | §8.1 | **OVERSTATED** | the proposal "hit" criterion is one ball radius, and the nearest proposal was 17.9–23.5 px from a radius-25 ball — up to 0.94 radii off. Proposal *localisation* is part of the loss |
| FrameViews crop is equivalent | §4.4 | **WRONG** | on real crops the mean absolute channel delta is 0.905/255 and 1.758/255, and the **ranker's top-1 changed on 3 of 15 frames** on one clip (0/15 on the other). A 0–20 % top-1 flip rate is a behavioural change, not a rounding note. The 2.45× speed claim stands; the equivalence claim does not |
| component timings in §3/§4.2 | §3, §4.2 | **UNDERSTATED 2×** | `suite_components` divided each blend key by `frames_total × repeats` while each key held only `frames_total` samples. Corrected values are exactly 2× the reported ones, recomputed from the raw per-clip samples in the same files; no re-measurement was needed |
| "0 dropped for unset node fields" as evidence D2 is closed | §5 | **INSUFFICIENT** | `node_gaps` did not check pose. 241 of 776 tracklets have an all-zero body vector and 9 of 34 clips are bodyless throughout, which were invisible to that audit |
| `E retrained without body features` | run labels | **MISLABELLED** | `--drop body` removes the *temporal* body group only; the ranker keeps its own body inputs. Likewise `--drop motion` removes 7 named motion features and leaves `support_fraction`, `top1_fraction`, `radius_stability`, `mean_rank_*`, `chroma_*` — all cross-frame aggregates |
| `masked_groups` in the legacy validation artifacts | §8.1 artifacts | **GARBAGE FIELD** | it was populated when `mask is None` and null otherwise — reversed. Fixed; the field is now `masked_groups` + `masked_features` + `mask_source` |
| "189 sequences" | §5 | corrected | 189 is the nonempty-**shard** count; the rows contain **119** distinct `sequence_id`s |
| `validation/legacy_E_retrained.json` | §8.1 | **INVALID** | produced with a mask-less ablation checkpoint. Superseded; see 13.2 |

### 13.2 Code corrections

- **Ablation mask is part of the checkpoint.** `TrackletDiscriminator` carries the
  mask as a registered buffer and applies it inside `forward`. `save_discriminator`
  writes `{state, mask, feature_names, format}`; `load_discriminator` restores it
  and still accepts legacy bare `state_dict` payloads (treated as full models);
  `migrate_ablation_checkpoint` rewrites a legacy ablation with its mask recorded,
  weights untouched. The trainer no longer multiplies features outside the model.
  All **10** existing checkpoints were migrated into
  `artifacts/corrected/temporal/` with digests recorded in
  `artifacts/corrected/mask_migration.json`.
- **Tie-correct AUC.** `_auc` is a mid-rank Mann-Whitney estimator; ties contribute
  0.5. `_auc_ties` reports the tie count beside every AUC. A new `recompute-auc`
  subcommand scores saved checkpoints on their exact prior splits with no
  retraining.
- **Timing denominators.** `suite_components` divides by `frames_total ×
  passes_with_that_blend` and records `blend_passes`; `suite_sets` decode
  accounting corrected (decode is measured once per clip and charged to the cold
  pass) and `proposal_seconds_per_frame_warm` split into `_all_passes` and
  `_warm_only` because it averaged cold and warm together.
- **Shared generator defaults.** `ball_candidates` and `ball_v3_blobs` generators
  default `pyramid=False`; v3 opts in explicitly through `ProposalParams.pyramid`.
  This restores v1/v2 reproduction, which the previous default silently broke
  (`radial_symmetry` shared 0 of 16 candidate positions with the original on one
  real frame).
- **Body missingness.** `Node.body_available` records pose availability per frame.
  `node_gaps(track, include_body=…)` separates caller-field gaps (the builder's
  drop rule) from pose gaps (a corpus coverage fact), and `body_coverage()` returns
  `None`, never 0.0, when nobody recorded it. The tracklet store now writes
  `body_coverage` per row plus a summary block, and
  `sequences_with_tracklets` beside `sequences`.
- **Ablation scope and reference paths.** `masked_groups` fixed; the validator
  records `mask_source` and refuses a `--mask` that conflicts with a checkpoint's
  stored mask. Added `--no-body-upstream` (a genuinely full no-body path: the
  ranker receives no poses either) and `--framewise` (no association, no tracklet,
  no temporal model).
- **Reuse instead of recompute.** `ball_v3_tracker.extract_frames` separates
  model-independent per-frame work from the decision stage;
  `track_ball_v3(..., extraction=…)` consumes a cached extraction, and the
  validator caches it on disk keyed by corpus, window, body-input mode and
  `EXTRACTION_VERSION`. It refuses to reuse a cache built with different body
  inputs rather than scoring stale probabilities.
- **Seal tests are behavioural.** The source-substring isolation check (which
  inspected only the first occurrence) is replaced by one that installs recording
  stand-ins for video decode, the body cache and the feature store and asserts the
  guard fires before any of them is reached, for every sealed clip id.

### 13.3 Tie-corrected AUC, with masks applied

`artifacts/corrected/temporal_auc_recomputed.json`. **These are fixed-checkpoint
scores, not a valid model selection**: `best_epoch` in every run was chosen with
the faulty estimator, so the weights that exist are not the optimum of a correct
metric.

| run | split | as-trained val AUC | tied scores |
|---|---|---|---|
| full | family | 0.9761 | 75 |
| full | clip | 0.9835 | 11 |
| no appearance | family | 0.9713 | 2 |
| no body | family | 0.9838 | 0 |
| no body | clip | 0.9773 | 100 |
| no chroma | family | 0.9683 | 65 |
| no motion | family | **0.9641** | **437** |
| no motion | clip | 0.9980 | 3 |
| no ranking | family | 0.8678 | 66 |
| no scale | family | **0.5090** | **567 of 570** |

Two readings, both honest:

- **D > F.** Removing the motion group costs 0.0120 on the family split. The
  original report's central claim inverts.
- **no-scale is a collapsed model, not a finding.** 567 of 570 scores tied means
  a constant output. It is reported as non-converged and is *not* interpreted as
  scale importance. It was not retrained.

A caveat, corrected in worker pass 3: the `as_deployed_no_mask` column is now
**`null` with a recorded reason** in the regenerated artifact
(`artifacts/corrected/temporal_auc_recomputed.json`; the previous version is kept
as `artifacts/closeout/superseded/temporal_auc_recomputed.superseded-2026-09-21.json`).
It could never have been a valid "deployed without a mask" measurement, because the
mask is intrinsic to the module and no code path bypasses it — every one of its 10
values was a bit-for-bit duplicate of `as_trained`, which read as "masking changed
nothing", the opposite of the truth. The reviewer's independently measured deltas
(body 0.9838→0.9827, motion 0.9636→0.9570, **appearance 0.9713→0.5612**) remain the
evidence that the defect was real and severe; what the migration establishes is
that it can no longer recur.

### 13.4 Ranker exposure, measured

`artifacts/corrected/ranker_exposure.json`:

| split rule | validation clips | validation rows | clips that are ranker-training clips | rows ranker-optimistic | row fraction |
|---|---|---|---|---|---|
| family | 24 | 570 | **18** | **415** | **72.8 %** |
| clip | 6 | 155 | 0 | 0 | 0.0 % |

The clip rule is ranker-optimism-free but lets shoot families straddle the split;
the family rule is family-disjoint but ranker-optimistic on 72.8 % of its rows.
**Neither is a clean end-to-end family-disjoint evaluation**, and the family-split
AUCs above must be labelled ranker-optimistic wherever they are quoted.

### 13.5 Body coverage, measured

`artifacts/corrected/body_audit.json`: **241 of 776** tracklets have an all-zero
body vector and **9 of 34** clips are bodyless throughout
(`pexels_32742697, 6541448, 6541451, 6541467, 6541666, 6541671, 6573443, 9362307,
9362374`). This is a **proxy for a missing pose, not proof**: a candidate far from
every landmark and outside the torso is legitimately zero, and this store predates
the availability flag. The two cannot be separated without regenerating the store,
which was not authorised. Sequence counting: 204 shards on disk, 189 with a
nonempty index, **119** distinct `sequence_id`s in the rows.

### 13.6 Corrected-pass evidence and invalidation manifest

| Artifact | Status | Supersedes / notes |
|---|---|---|
| `artifacts/tracklets/shards/**`, `run_manifest.json`, `tracklets.json` | **valid, retained** | counts independently verified by review 1 |
| `artifacts/temporal/tracklet*.pt`, `tracklet_report*.json` | **retained as-is, results invalid** | weights valid; reported AUCs invalid (tie bug), masks absent. Superseded by `corrected/temporal/` and `corrected/temporal_auc_recomputed.json` |
| `artifacts/corrected/temporal/*.pt` | **new** | 10 migrated checkpoints, weights unchanged, masks recorded (`mask_migration.json`) |
| `artifacts/corrected/temporal_auc_recomputed.json` | **new** | tie-correct AUC on exact prior splits, with masks applied |
| `artifacts/corrected/ranker_exposure.json` | **new** | 18/24 clips, 415/570 rows ranker-optimistic |
| `artifacts/corrected/body_audit.json` | **new** | 241/776 proxy, 9/34 clips, 119 sequences |
| `artifacts/corrected/corrected_timings.json` | **partial** | the ×repeats rule is implemented in `ball_v3_bench.py`; the recomputation from the saved raw samples is written but key discovery was left unfinished — the corrected values are exactly 2× the §3/§4.2 numbers, and that statement rests on the denominator fix, not on this file |
| `artifacts/profile/components_*.json` | **valid raw data, invalid as reported** | need ×2; no rerun required |
| `artifacts/profile/downstream_*.json` | **timing valid, equivalence invalid** | built frames with `np.zeros`, so the zero channel delta was vacuous |
| `artifacts/validation/legacy_D.json`, `legacy_no_temporal.json` | **valid** | unaffected by the mask defect; `masked_groups` field is garbage |
| `artifacts/validation/legacy_E_retrained.json` | **INVALID** | mask-less ablation checkpoint |
| `artifacts/validation/final_test_audit.json` | **valid** | independently confirmed by review 1 |
| `artifacts/validation/sets_*.json` | valid at `repeats=1` | needs `repeats≥3` + dispersion for a quotable ratio |

### 13.7 What is still missing

> **Status in worker pass 3 (closeout)**: items 1–3 are resolved; item 4 stands.

1. ~~**Corrected validation runs are incomplete.**~~ **Done in worker pass 3.** The
   matrix was run on both corpora against the migrated checkpoints
   (`artifacts/closeout/validation/*.json`, 12 runs: D, heuristic, E partial, F
   partial, E full, F reference × `legacy` and `v3-validation`) and the figures are
   in `docs/ball-tracker-v3-results.md` §6/S6–S7. §8.2 and §8.3 above now point at
   that document instead of standing empty.
2. ~~**§10 has not been written** as a results section.~~ **Done in worker pass 3**:
   §10 now names the genuine full suite and records that `run_tests.sh` was a
   8-file subset, renamed accordingly.
3. ~~**FrameViews equivalence needs a real-crop test**, not a blank fixture.~~ **Partly
   done, honestly.** A non-blank synthetic fixture regression is committed
   (`tests/fixtures/frameviews_turf.png` + `make_frameviews_fixture.py`) and pins the
   structure (small crops identical, large crops bounded). No *real-crop* test is
   added, because committing uncleared media is not allowed: review 1's real-crop
   measurement (mean |Δpixel| 0.905/255 and 1.758/255; ranker top-1 flipping 3/15 and
   0/15) is **reported** as the decision-shift evidence, and no equivalence is claimed.
4. **No non-zero real-labelled recovery number exists.** 0 of 185 on the
   development corpus, independently reproduced by review 1, with the evaluation
   itself verified (exact frame indexing, native-rescaled coordinates, pose cache
   resolving, device routing fixed). Per review 1 this supervision is structurally
   poor for real balls: the synthetic insertion process makes "small" and "moving"
   near-sufficient statistics for "ball" (radius 0.0056 vs 0.0097; static 0.14 vs
   0.63) while the real labelled corpus is dominated by large, stationary balls.

### 13.8 Final test

Unchanged and unconsumed. No decode, view, annotation, embedding, feature
generation, inference or scoring on any final-test clip or its cached derivatives
in either pass; no `--final-test` invocation. Split metadata was read only to
build exclusion lists. `artifacts/validation/final_test_audit.json` records the
three mechanical checks, and review 1 independently confirmed them.

## 9. Final test: not consumed

`artifacts/validation/final_test_audit.json`. Three mechanical checks:

1. The frozen final-test clip list (10 clips, metadata read from
   `data/ball_dataset_v3_eval/split.json`, SHA-256 recorded).
2. All 74 artifacts this pass wrote were scanned for those clip names: **zero**
   occurrences.
3. Every decode site in the three tools was listed with the nearest preceding
   guard: `tools/ball_v3_bench.py` (guard at line 157, decode at 176),
   `tools/ball_v3_train.py` (guard at 92, decode at 155),
   `tools/ball_v3_validate.py` (guard at 130, decodes at 135 and 150).

No v3 final-test clip was decoded, viewed, annotated, embedded, featurised,
trained on, inferred on, scored or tuned. No `--final-test` evaluation was run.
The cached pose entries for final-test clips under
`out/ball_tracker_v3/body_cache/` were not opened. The split was not altered.

V2/legacy held-out footage is development-consumed and is never represented as
fresh held-out evidence anywhere in this document.

---

## 10. Tests

> **Written in worker pass 3.** The suite that matters is the repository's own:
> `.venv-video-ai/bin/python3 -m pytest tests -q` over all of `tests/` (121 test
> files). `artifacts/logs/run_tests.sh` named 8 files of those 121 and called itself
> the full suite; it is renamed to `run_v3_subset_tests.sh`, which is what it is — a
> fast v3-only subset with pytest's status propagating (no pipes). The counts,
> exit status and any failures from the genuine full-suite run are recorded in
> `docs/ball-tracker-v3-results.md` §8 and in the worker-3 report.
<!-- Worker pass 3: the FILL placeholder above was removed. -->

---

## 12. The `docs/ball-tracker-v3-results.md` gap

> **Closed in worker pass 3**: `docs/ball-tracker-v3-results.md` now exists and is
> the current authoritative evidence document. The paragraph below is kept as the
> record of the gap that existed at the time.

`ghostcaddie/video/ball_v3_proposals.py` cited `docs/ball-tracker-v3-results.md`
for the recall sweep its defaults were chosen from. **That file does not exist**
and never did on this branch. The reference is repaired in the source to name the
artifacts that actually hold the numbers, and this section records the gap rather
than inventing a document to fill it.

What the sweep actually recorded (`out/ball_tracker_v3/proposal_methods.json`,
`origin: HUMAN_READ_LEGACY_ANNOTATIONS`, `status: development-consumed; parameter
selection only, never accuracy`, 84 frames, radius ≤ 40 px):

| generator set | recall | proposals/frame |
|---|---|---|
| all six | 0.7381 | 139.0 |
| the v2 five | 0.3214 | 62.0 |
| all six minus `local_contrast` | 0.7381 | 129.8 |
| `local_distinctness` + `brightness_cue` | 0.7381 | 104.7 |
| `local_distinctness` only | 0.7381 | 95.5 |

§3.1 reproduces the all-six and v2-five rows exactly, and §4.3 records where the
optimisation moved them. The other three rows were not re-measured in this pass
because the generator sets they describe are not the ones under comparison here.

Also referenced and also absent: `docs/ball-tracker-v3-plan.md` §1 cites
`out/ball_tracker_v2/final_test_result.json`, which does exist and is the v2
record; nothing in this pass needed to reopen it.


