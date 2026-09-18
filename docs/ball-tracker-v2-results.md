# Ball tracker v2 — autonomous research pass

Branch `research/ball-tracker-v2`, base `e61d21a` (dataset v2 frozen). This document
reports what was built, what was measured, and what failed. Every number here comes
from a command in this repository; none of it is an estimate.

**Classification: FAILED_TO_GENERALIZE** (see §11 for the promotion decision).

The tracker does not work end to end. The detector — one component of it — is
measurably good and is worth keeping. Neither statement is a summary of the other,
and the sections below keep them apart on purpose.

---

## 1. The supervision problem, and how it was solved without a human

There are no human labels in this corpus and this pass did not invent any. Three
independent sources were built, and no one of them is trusted alone:

| source | origin tag | what it is | size |
|---|---|---|---|
| synthetic insertions on real backgrounds | `SYNTHETIC_GROUND_TRUTH` | a ball inserted into real train footage, so its position is known because the generator put it there | 9,870 crops (3,366 positive, 6,504 negative) |
| multi-method consensus | `MODEL_CONSENSUS` | five independent candidate generators agreeing, with temporal and plausibility checks | **0 HIGH, 0 MEDIUM** (see §5) |
| hard-negative mining | `MODEL_CONSENSUS` | strong candidates that consensus refused | 1,440 crops from 24 train clips |

401 insertions (11%) were **discarded rather than used**: the label said the ball was
there and the pixels did not measurably change, which means the ball was invisible at
that scale. A label whose object cannot be measured is not supervision.

`docs/ball-annotation-guide.md` and `docs/ball-dataset-v2-status.md` (from the
previous pass) describe a human labelling workflow. **This pass did not use it, and
no human labelled anything.** Nothing in this document is human ground truth.

---

## 2. What was built

| module | role |
|---|---|
| `ghostcaddie/video/ball_synthetic.py` | inserts a ball into real frames with known sub-pixel position, radius, visibility, blur, occlusion, colour and trajectory; degrades noise/compression locally; verifies each insertion against the changed pixels |
| `ghostcaddie/video/ball_candidates.py` | five independent generators — radial symmetry (FRST), local contrast, frame difference, flow residual, brightness/chroma cue — on a bounded working scale, coordinates mapped back to native pixels |
| `ghostcaddie/video/ball_consensus.py` | cross-method clustering, confidence tiers, forward/backward cycle consistency, motion plausibility, body/clothing regions, static-scene lock, hard-negative mining |
| `ghostcaddie/video/ball_detector.py` | 311,395-parameter heatmap CNN: presence + radius + localisation, trained on synthetic supervision |
| `ghostcaddie/video/ball_tracker_v2.py` | the pipeline: compensation → detector → ensemble → fusion → association → cycle check → rejection → track-level acceptance, with per-frame states |
| `ghostcaddie/video/ball_eval_v2.py` | metrics, kept pure and testable |
| `tools/ball_track_v2.py` | research CLI: `gen-synth`, `train`, `eval-synth`, `consensus`, `track-synth`, `negatives`, `diagnostics`, `track`, `final-test` |

Per-frame states are `OBSERVED`, `AMBIGUOUS`, `OCCLUDED`, `LOST`, `UNAVAILABLE`.
Missing frames are never filled with invented observations; an optional interpolation
is flagged per frame and can never be reported as `OBSERVED`.

---

## 3. Synthetic supervision (source A)

9,870 crops from 34 train clips: 3,366 positive, 6,504 negative, 401 insertions
discarded as unmeasurable. Ball radius 2.0–39.4 px, median 6.2 px — the tiny-object
regime the previous pass failed in. Insertion fidelity was verified by comparing the
label with the centroid of the pixels that actually changed: **median error 1.0 px**,
with the residual being the contact shadow the renderer draws.

Trajectories are expressed in ball radii per frame (`stationary`, `slow`, `linear`,
`fast`, `accelerating`, `entrance`, `exit`) and every kind is exercised by the
evaluation. Insertion coordinates are known by construction; this is legitimate
synthetic ground truth on real backgrounds, not human ground truth.

## 4. Candidate ensemble and consensus (source B)

Five methods propose candidates independently; none may influence another.
Calibration on real train frames established the honest starting position: **on
cluttered grass a single method's peak response does not separate the ball from
clutter**, and the ensemble is therefore recall-oriented by design, with the
decisions made downstream by agreement, temporal verification and the detector.

## 5. Real-video pseudo-labels: the consensus channel came back empty

Running consensus over real train footage produced **0 HIGH and 0 MEDIUM labels**
across the clips measured. The funnel, measured rather than assumed:

- 45–50 candidates per frame across five methods on cluttered footage;
- clusters mostly with 1–3 methods agreeing;
- dominant refusals: "temporal verification failed", "implausible motion",
  "N competing candidates with comparable support".

Two defects found while measuring this were fixed (native-vs-working coordinate
spaces in the temporal check; a per-frame tie rule that made every frame ambiguous),
and the channel was still empty afterwards. **The honest reading: classical
candidates alone cannot produce high-confidence ball labels on this footage.** That
is precisely the gap the detector exists to fill, so the design is not wrong — but
as a standalone supervision source this channel contributes nothing, and it must not
be described as if it did.

## 6. Hard-negative mining (source C)

1,440 crops mined from 24 train clips: strong real candidates that consensus refused,
each keeping its method, score and rejection reason. These are the objects that
fooled v1 — bright clutter, clothing, glare, range balls.

Measured effect on the detector (this is the clearest positive result of the pass):
**94.5% of these mined candidates are rejected at the 0.7 threshold** (89.5% at 0.5,
97.5% at 0.9). Before mining, the detector gave essentially every strong candidate,
ball or clutter, a presence probability near 1.0.

## 7. The detector

311,395 parameters, 64×64 crops, heatmap + presence + radius heads, trained on
synthetic supervision in two rounds (round 1 synthetic only, round 2 with the mined
hard negatives), best epoch selected by validation F1 and named in the report.

Held-out synthetic crops (disjoint sample, 500 positives / 900 negatives):

| metric | round 1 | round 2 (with hard negatives) |
|---|---|---|
| AUC | 0.963 | 0.917 |
| precision @0.5 / recall @0.5 | 0.882 / 0.856 | 0.940 / 0.624 |
| precision @0.7 / recall @0.7 | 0.941 / 0.804 | 0.984 / 0.506 |
| centre error (median) | 2.5 px | 2.5 px |

Round 2 traded synthetic recall for real-world false-positive rejection. On the
synthetic crops it is easier to score well than on real footage; the trade is
reported rather than resolved by picking the flattering number.

By ball size (round 2): 4–8 px recall 0.98, 8–16 px 1.00, 16–64 px 0.98, **1–4 px
0.49**. Balls below ~4 px are not reliably detected, and no claim is made for them.

### The failure that decides this pass: the detector saturates

On a validation clip holding a known 19 px ball, the detector scored **the true ball
at 0.98 on every frame — and scored clutter at ~0.98 as well**. The ball outscored
the clutter in only 7 of 12 frames, and by margins of 0.003–0.09. A 64×64 crop
classifier trained on synthetic insertions has no way to separate the ball from a
ball-like bright object at that scale: both crops look alike. Presence probability is
therefore not a ranking signal on wide shots, and the tracker's hypothesis ordering
inherits that.

## 8. Tracker performance, synthetic ground truth (validation split)

Command: `tools/ball_track_v2.py track-synth --split validation --clips 8
--sequence-length 12 --trajectory mix` (a mix of stationary, slow, linear, fast,
accelerating, entrance and exit insertions; final test untouched).

| metric | value |
|---|---|
| frames with a ball | 92 |
| frames without a ball | 4 |
| true positives | **0** |
| false positives | **7** |
| false negatives | 85 |
| correct silence | 4 |
| precision / recall / F1 | 0.000 / 0.000 / 0.000 |

Per clip: seven of eight clips produced **no observations at all** (honest silence);
one clip (`pexels_14980435`) produced 7 false positives on clutter **and was
accepted** — a hallucinated track, the same class of error as v1, on a smaller scale.

This is the result that determines the classification. The tracker is, at the level
of a whole clip, either silent or wrong.

## 9. Real-video diagnostics (validation split)

Not accuracy — these clips have no human labels, and pseudo-label agreement is not
real-world accuracy. They are behavioural diagnostics: how often the tracker abstains,
and whether it does the two things v1 did wrong.

### Negative sequences (validation)

Command: `negatives --split validation --clips 6` — real footage with **no ball
inserted**; nothing should be accepted here.

| metric | value |
|---|---|
| sequences | 6 |
| sequences with an accepted track | **1** |
| fabricated-track rate | **0.167** |
| observed frames on empty background | 15 |

Claiming a ball on a sixth of ball-free sequences is undeserved confidence, and it is
the same failure seen in §8.

### Behavioural diagnostics (validation, 4 clips × 12 frames)

Command: `diagnostics --split validation --clips 4`.

| metric | value |
|---|---|
| consensus tiers | 0 HIGH, 0 MEDIUM, 3 LOW, 44 AMBIGUOUS, 1 UNAVAILABLE |
| consensus rate | 0.00 |
| observed frames / tracked spans | 12 / 48 |
| coverage | 0.25 |
| mean cycle consistency | 0.75 |
| abstention rate | 0.75 |
| tracks accepted | 1 of 4 |
| diagnostics raised | `CLUB_LOCK` 10, `STATIC_SCENE_LOCK` 1, `BODY_LOCK` 0, `CLOTHING_LOCK` 0, `FLAG_LOCK` 0 |
| body-lock / static-lock rate | 0.0 / 0.0 — **body locks are 0 because no pose model was run, not because they were tested** |

`CLUB_LOCK` here is the umbrella diagnostic for a candidate that fails temporal
witness support; it is not a claim that the object was a club.

### Final test (scored once, frozen)

Procedure, fixed before the split was touched: synthetic insertions with exact ground
truth on **final-test backgrounds** using the same trajectory mix, scored once, then
real-footage behavioural diagnostics on the same split. Detector threshold 0.7,
competitor ratio 0.9, competitor detector margin 0.05, cycle span and radii, and the
trajectory mix were all frozen from train/validation evidence only. **Nothing was
tuned after these numbers were seen**, and they cannot be used to tune: the split is
consumed.

| metric (11 clips × 12 frames = 132) | value |
|---|---|
| frames with a ball | 130 |
| true positives | 7 |
| false positives | 22 |
| false negatives | 101 |
| correct silence | 2 |
| precision | 0.241 |
| recall | **0.054** |
| F1 | 0.088 |
| median centre error on hits | 4.34 px |
| false-positive rate on absent frames | — (2 absent frames; 22 FPs include off-target hits on frames where the ball was present) |

Real-footage diagnostics on the sealed split (4 clips × 12 frames): consensus tiers 0
HIGH / 0 MEDIUM / 4 LOW / 43 AMBIGUOUS / 1 UNAVAILABLE, coverage 0.19, mean cycle
consistency 0.50, abstention rate 0.75, 1 track accepted, `CLUB_LOCK` 6, no body or
static locks raised.

The final split says what validation said: the tracker occasionally locks onto the
right object with a good centre (7 hits, 4.3 px) and is otherwise silent or wrong. It
is not close to a promotable bar, and the pass does not go back to tune on it.

Per clip on the sealed split — the shape of the failure, not just its average:

| clip | TP | FP | FN | silence | median error |
|---|---|---|---|---|---|
| `pexels_13671370` | 7 | 0 | 5 | 0 | 4.34 px |
| `pexels_4784177` | 0 | 0 | 11 | 1 | — |
| `pexels_9361937` | 0 | 0 | 11 | 1 | — |
| `pexels_5200680`, `5200681`, `5200689`, `6256583`, `6256764`, `6256766`, `35546329`, `4128811` | 0 | 4–9 on three of them | 3–12 | 0–1 | — |

One clip is tracked correctly (7/12 frames, 4.3 px, no false positives). Three clips
are confidently wrong on clutter. Seven are silent. The tracker is not uniformly
mediocre: it works on some footage and fails on other footage, and at this stage of
the work it cannot tell the two apart.



## 10. What failed, precisely

1. **Presence saturation** (§7). The detector is a good ball detector and a poor
   ball-vs-ball-like-clutter discriminator at 64×64. Ranking hypotheses by presence
   probability cannot work while many candidates sit at 0.95–1.00.
2. **The consensus channel is empty** (§5). No real-footage pseudo-labels were
   produced, so round 2's real-data supervision came only from mined negatives.
3. **Track-level tie-breaks are under-powered.** Temporal verification (cycle
   consistency, plausibility) correctly refuses to accept clutter, but it also
   refuses the true ball: on these clips the ball is inserted and then found at 0.98,
   yet the track is never assembled because rivals are equally strong.
4. **Body/clothing rejection was not exercised on real footage.** The regions,
   the elevated threshold inside them and the `BODY_LOCK`/`CLOTHING_LOCK`
   diagnostics are implemented and unit-tested, but this pass did not run the pose
   model, so `body_lock_rate` in the diagnostics is **0 by absence of data, not 0 by
   measurement**. Nothing about body rejection on real footage is claimed.

## 11. Promotion decision

Bar (§16) and outcome:

| condition | result |
|---|---|
| 1. synthetic test performance strong | **no** — 0 TP, 0 recall end to end |
| 2. validation false-positive behaviour low | no — 7 FPs, one accepted wrong track |
| 3. body/clothing locks rare | **not measured** |
| 4. static bright-object locks rare | not measured on real tracks |
| 5. tracks temporally consistent | no tracks to assess |
| 6. forward/backward checks pass | the check works; nothing passes it |
| 7. difficult clips abstain honestly | yes on 7/8 clips, no on 1 |
| 8. final-test evaluation holds up | see §9 |
| 9. no per-clip seed required | yes |
| 10. no network dependency | yes (tested) |
| 11. canonical tests stay green | yes |
| 12. visual QA plausible | no — the accepted track is on clutter |

**Classification: FAILED_TO_GENERALIZE.**

Canonical ball remains **UNAVAILABLE**. `canonical.ball_layer()` still returns
`state=unavailable, method=none, frames_with_observation=0`, and the canonical modules
import none of this code (enforced by test). No physical metric — ball speed, carry,
launch angle, spin, apex — is unlocked by tracking identity, and none was touched.

## 12. What is preserved

- the synthetic generator, with per-insertion verification;
- the candidate ensemble, consensus, temporal verification and hard-negative mining;
- the trained detector weights (local, gitignored) plus the report and weight hash;
- the tracker with explicit states, diagnostics and acceptance rules;
- the evaluation tooling, the negative-sequence test and the split guards;
- this document, including the negative result.

Reproduce:

```
.venv-video-ai/bin/python3 tools/ball_track_v2.py gen-synth --sequences 340
.venv-video-ai/bin/python3 tools/ball_track_v2.py consensus --clips 24
.venv-video-ai/bin/python3 tools/ball_track_v2.py train --epochs 12 --hard-negatives
.venv-video-ai/bin/python3 tools/ball_track_v2.py track-synth --split validation
.venv-video-ai/bin/python3 tools/ball_track_v2.py final-test --final-test
```

The next attempt should attack presence saturation directly — hard negatives that are
*ball-like* rather than merely bright, and a discriminator that uses motion and
appearance over a track instead of a single crop — and must not tune on the final
split that this pass already scored.
