# Ball tracker v3 — plan, and why it is not the plan that was handed over

Branch `research/ball-tracker-v3-context`, base `f46a6e2` (the v2 pass, classified
`FAILED_TO_GENERALIZE`). This document records the Phase 0 takeover measurement and
the v3 design that measurement forced. Every number here comes from a command in
this repository.

The short version: **the v2 handoff's diagnosis is real but it is the smallest of
three losses, and the pipeline stage it blames is the one that works best.** The v3
design in the task brief targets that stage. Building only that would have raised a
recall ceiling of 8.3% to a recall ceiling of 8.3%.

---

## 1. Phase 0 — what the handoff claims, and what is actually true

The v2 results document (`docs/ball-tracker-v2-results.md` §7, §10) names one
failure and builds the whole "what to do next" paragraph on it:

> On a validation clip holding a known 19 px ball, the detector scored **the true
> ball at 0.98 on every frame — and scored clutter at ~0.98 as well.** […] A 64×64
> crop classifier trained on synthetic insertions has no way to separate the ball
> from a ball-like bright object at that scale.

That was measured on **one synthetic insertion on one clip**. It is also the only
real-footage evidence the pass offers for its central claim, because — and this is
the load-bearing fact the handoff states but does not carry into its conclusions —
**the v2 final test scored synthetic balls, not real ones.** The artifact says so
itself (`out/ball_tracker_v2/final_test_result.json`):

```
"procedure": "synthetic insertions on final-test backgrounds, plus temporal/cycle
              diagnostics; no human labels exist, so no real-footage accuracy
              number is claimed"
```

Meanwhile 325 human-read frames across 10 rights-cleared clips have been sitting in
`data/ball_annotations/` since the previous pass. **The v2 tracker was never scored
against a real golf ball.** The v3 takeover started by doing exactly that.

### 1.1 The measurement

`tools/ball_v3_diagnose.py` runs the real v2 pipeline stage by stage against those
labels and attributes every lost frame to the stage that lost it. A candidate
"finds" the ball when it lands within one ball radius of the label — a prediction
further away than the ball is wide is a prediction on a different object.

Restricted to the 84 labelled frames whose ball radius is inside the v2 detector's
training regime (≤ 40 px):

| stage | frames surviving | lost here |
|---|---|---|
| a labelled, unambiguous, visible ball is present | 84 | — |
| some generator proposes it | 26 (**31.0%**) | **58 (69%)** |
| the proposal clears `ConsensusParams.min_score = 0.25` | 7 (**8.3%**) | **19 (73% of survivors)** |
| greedy clustering keeps a cluster on the ball | 7 (8.3%) | **0** |
| the detector then ranks it first | ~5 (**6.0%**) | 2 (29%) |

Reproduce:

```
.venv-video-ai/bin/python3 tools/ball_v3_diagnose.py
```

### 1.2 What that says, point by point against the brief's Phase 0 questions

**1. Is the true-ball / clutter score saturation real?** *Partly, and not where it
was blamed.* Given an **oracle crop placed exactly on the labelled ball**, the
detector beats the best same-frame distractor on **62.5%** of in-regime frames,
median margin **+0.023**. That is a weak, noisy signal — nowhere near a usable
ranking margin — but it is not the coin flip the handoff describes, and it is
*better* than chance rather than worse.

Where saturation is total is **outside the detector's trained scale**: on the 24
labelled frames with radius > 40 px the oracle beats the best distractor **8.3%**
of the time, median margin **−0.369**. That is a **scale-coverage failure, not a
discrimination failure**. The v2 detector was trained on radii 2.0–39.4 px (median
6.2) and simply does not represent a large ball. Calling this "saturation" points
the next pass at the wrong fix.

**2. Is crop-only appearance the dominant problem?** **No.** Crop appearance is the
*least* broken stage. When a cluster actually reaches the detector, it ranks the
true ball first **71.4%** of the time. The dominant problem is that a cluster
reaches the detector on only 8.3% of frames.

**3. Is temporal association failing because of bad candidate ranking, rather than
one simple threshold bug?** **It is substantially one simple threshold bug.**
`ConsensusParams.min_score = 0.25` deletes the true ball's own proposal on 19 of
the 26 frames where a generator did find it — a **3× recall cost from one
constant**. The measured scores of the deleted true-ball candidates are 0.086–0.243,
i.e. they sit in a band immediately below the floor. The code comment at that
constant reads "Kept low on purpose: the ensemble is recall-oriented". It is
measurably not low enough. Notably, **greedy cluster absorption costs nothing** —
zero frames were lost that way, so the clustering design itself is sound.

**4. Was body/clothing rejection actually exercised?** **No, confirmed.**
`track_ball_v2(..., body_regions=...)` is supplied by exactly one unit test and by
no tool on any path (`grep -rn "track_ball_v2(" tools/ ghostcaddie/`). Every
`body_lock_rate` and `static_lock_rate` in every v2 result file is `0.0` by absence
of data. The handoff states this honestly and it holds up.

**5. Is the final-test report internally consistent?** **Yes, with two caveats
worth naming.** Per-clip counts sum exactly to the pooled counts
(132 frames / 130 with ball / 7 TP / 22 FP / 101 FN / 2 correct silence), and
precision recomputes to 0.2414. But:

- **recall uses a different denominator from precision.** Reported recall 0.0538 is
  `TP / frames_with_ball` = 7/130. The conventional `TP / (TP + FN)` is 7/108 =
  **0.0648**. F1 therefore mixes two denominators. Both inputs are printed in the
  document, so this is traceable rather than hidden, but the headline number is the
  more flattering of the two by one convention and the less flattering by another.
- **`false_positive_rate_on_absent = 11.0`** is 22 false positives over 2 absent
  frames. It is not a rate. The document marks it with an em-dash and explains;
  the JSON field name does not.

### 1.3 Two defects found that the handoff does not mention

**A. Model selection ran on a scrambled validation set.** In
`ball_detector.train_detector`:

```python
val = evaluate_detector(model, images, [labels[i] for i in val_idx], ...)
```

`evaluate_detector` indexes `images[...]` with positions into the *labels list it
was passed* (0…len(val_idx)−1), but `images` is the **full** array. So validation
pairs `images[0..n]` with `labels[val_idx[0..n]]` — image *i* scored against a
different crop's label. Demonstrated directly:

```
as called in train_detector (images NOT sliced): precision 0.2727 recall 0.15  f1 0.1935
with images sliced to val_idx                 : precision 0.8947 recall 0.85  f1 0.8718
```

This is why `detector_report.json` shows `best_val_f1 = 0.2893` while the standalone
`eval-synth` path (which passes matched arrays, and is correct) reproduces at
precision 0.94 / recall 0.62. **`best_epoch = 10` was chosen from noise**, and the
shipped weights are not known to be the best weights. The two CLI call sites are
fine; only the training loop is affected.

**B. The reported parameter count is wrong.** `docs/ball-tracker-v2-results.md` §2
and §7 say 311,395 parameters. The artifact written by the same pass
(`detector_report.json`) says **377,187**, and loading the shipped weights and
counting gives **377,187**. Cosmetic, but it is the number a reader would quote.

Everything else in §6 and §7 reproduces closely on an independent re-run
(hard-negative rejection 88.3/93.5/98.1% at thresholds 0.5/0.7/0.9 against the
reported 89.5/94.5/97.5; recall 0.642/0.515 against the reported 0.624/0.506).
**The v2 pass's own measurements are honest.** Its interpretation of them is what
this pass disagrees with.

### 1.4 Corrected root cause

The v2 ball tracker fails on real footage because, in order of damage:

1. **Candidate generation does not propose the ball (69% of the loss).** Five
   generators, ~50 proposals per frame, and the actual golf ball is among them on
   31% of frames.
2. **A single constant deletes most of what survives (73% of the remainder).**
   `min_score = 0.25`, against true-ball candidate scores of 0.086–0.243.
3. **The detector has no scale coverage above ~40 px radius.** Complete inversion
   there, and it is a training-distribution gap, not a discrimination limit.
4. **Ranking by crop appearance is weak but positive (71% top-1, oracle margin
   +0.023).** This is the stage the brief asks v3 to rebuild, and it is the stage
   that is working.

---

## 2. What v3 does about it

The brief's Phases 3–6 (context, ranking loss, tracklets) are the right *structure*
and are kept. What changes is that they are placed downstream of a proposal stage
that is rebuilt for recall first, because a ranker cannot rank what it never sees.

**The v2 pipeline was conservative at the proposal stage and asked the
discriminator to be permissive. v3 inverts that: propose generously, discriminate
hard.**

### Stage 1 — high-recall proposals (new; fixes losses 1 and 2)

- Drop the `min_score` floor from the candidate path entirely; the ranker is the
  thing that says no.
- Raise per-method proposal limits and keep several hundred candidates per frame.
- Run the scale-sensitive generators at native resolution inside a bounded region
  rather than on the 1920-px working copy.
- **Gate:** ball-proposal rate on the legacy human-labelled corpus must exceed 80%
  before any ranking model is trained. This is a development metric on
  development-consumed labels, and is never reported as accuracy.

### Stage 2 — context representation (brief Phase 4)

Per candidate, three streams rather than one 64×64 crop: a native-resolution
candidate crop, a larger context crop (128–256 px, scaled to the clip), and a
camera-compensated motion residual. Scale coverage is extended so a 70 px ball is
in distribution, which loss 3 requires.

### Stage 3 — listwise ranking over same-frame candidates (brief Phase 5)

Trained with a softmax-over-candidates objective on the true ball versus the
strongest distractors **from the same frame**, not as an independent ball/not-ball
classifier. This is the brief's central instruction and it is kept unchanged — it
is the right fix for the weak +0.023 margin, once candidates exist to rank.

### Stage 4 — tracklet discrimination (brief Phase 6)

Tracklets over 5–11 frames, scored by a small temporal model on camera-compensated
motion, score history, competition history and body-relative position.

### Stage 5 — body context, actually run (brief Phase 8)

MoveNet is present (`models/movenet-singlepose-lightning-v3.tflite`) and runs in
`.venv` only; torch runs in `.venv-video-ai` only. Body keypoints are therefore
extracted in a separate cached pass and consumed as **features**, never as an
exclusion region. `BODY_LOCK` and `CLOTHING_LOCK` become measured numbers.

### Stage 6 — abstention (brief Phase 12)

Rank margin, tracklet probability and cycle consistency together. One note from
Phase 0: **cycle consistency is not discriminative on its own** — every v2
final-test clip scored `cycle_consistency_rate = 1.0`, including the three that
tracked pure clutter with 9 false positives each. Static clutter passes a
forward/backward check trivially. It may stay as one input; it may not be a gate.

---

## 3. Evaluation

The v2 final test is consumed and is not touched. A new corpus was acquired:
`data/ball_dataset_v3_eval/` — 21 clips from 19 Pexels shoot families, every one
disjoint from both the v2 dataset and the legacy human-labelled corpus.
`pexels_38025678` was found in the candidate pool and **excluded because it is
already a v2 train clip**; reusing it would have been training leakage.

The split is frozen before model selection, by family, and the final test is
scored once.
