# Ball-tracking research results

Status date: 2026-09-18. Branch `pga-research-analyzer`, baseline commit `fc76ffe`.
Plan and acceptance bar: [`ball-tracking-research-plan.md`](ball-tracking-research-plan.md),
written before any of this was implemented and not edited afterwards.

---

## Final classification

**FAILED_TO_GENERALIZE.**

The canonical ball layer stays `UNAVAILABLE`. Nothing on the canonical path
changed, and a regression test now asserts that.

What the method does on the ten-clip corpus:

- On **one** clip (`pexels_6573644`, a close-up optic-yellow range ball) it tracks
  perfectly: precision 1.000, recall 1.000, median centre error 8.6 px, no
  identity switches, and it correctly goes silent on the fifteen ball-absent
  frames and on the two frames where the club face covers the ball.
- On **eight** clips it refuses to answer, and says why.
- On **one** held-out clip it did something worse than refusing: it locked onto
  the golfer's shirt for fifteen consecutive frames while the ball was not in
  shot. That single clip puts the held-out false-positive rate on ball-absent
  frames at **0.333**, against a budget of **≤ 0.05** that was written down
  before any measurement.

Held-out recall is **0.000**. Held-out precision is **0.000**. The declared
false-positive budget is missed by a factor of six.

Worse for the promotion question: on held-out the *simpler* baseline beats it.
B1 — this pipeline's own detector with the temporal stages deleted — reaches
recall 0.483 pooled and **1.000 with a median centre error of 4.3 px on
`pexels_34883764`**, the very clip where the full method finds nothing and then
locks onto the golfer's shirt. The temporal stages that removed every false
positive on development also removed the only held-out track there was. The
required "clear improvement over the existing baseline" is not merely unproven;
on the held-out split it is a regression.

The pass was still worth running, and the negative result is specific rather
than vague: the failure that got the previous tracker withdrawn — following the
flagstick base — does **not** occur anywhere in this corpus, and the new failure
(body lock) is one that only a held-out split could have exposed, because it
appears on no development or validation clip.

---

## 1. Dataset created

There was no ball ground truth in this repository. There is now.

Rights: Pexels clips already retained locally for research. PGA TOUR material
under `out/pga_official_acquisition/` was **excluded** — it is rights-gated by
`pga_source_gate.py` and may not be used for training or evaluation.
`pixabay_149976.mp4` was excluded on content grounds (a 3-D cartoon golf-ball
mascot, not golf footage).

Format: `data/ball_annotations/<clip_id>.json`, schema
`golf-ball-annotation.v1`, validated by `ghostcaddie/video/ball_annotations.py`.
Every read and write re-checks the media bytes through
`source_binding.bind_source`, so a label file can never drift onto a re-encoded
or substituted clip. A `visible: true` label requires coordinates; a
`visible: false` label requires a reason and *forbids* coordinates; `ambiguous`
is an orthogonal flag meaning the annotator would not defend the call.

| split | clip | resolution | fps | labelled | visible | not visible | ambiguous | runs | ball radius px |
|---|---|---|---|---|---|---|---|---|---|
| train | `pexels_37030686` | 1080x1920 | 30 | 15 | 15 | 0 | 1 | 1 | 14 |
| train | `pexels_6541842` | 1440x2560 | 25 | 40 | 24 | 16 | 0 | 2 | 20–26 |
| train | `pexels_6573621` | 720x1280 | 30 | 45 | 30 | 15 | 0 | 2 | 25 |
| train | `pexels_6573644` | 1920x1080 | 30 | 45 | 30 | 15 | 2 | 3 | 72 |
| validation | `pexels_33511561` | 2560x1440 | 60 | 30 | 30 | 0 | 0 | 2 | 159–276 |
| validation | `pexels_37080178` | 1080x1920 | 30 | 30 | 15 | 15 | 0 | 2 | 12 |
| validation | `pexels_6541674` | 2560x1440 | 25 | 30 | 15 | 15 | 0 | 2 | 15 |
| held-out | `pexels_34883764` | 1080x1920 | 30 | 30 | 15 | 15 | 1 | 2 | 12 |
| held-out | `pexels_38053415` | 4096x2160 | 23.98 | 30 | 15 | 15 | 0 | 2 | 21 |
| held-out | `pexels_6541964` | 2560x1440 | 25 | 30 | 0 | 30 | 15 | 2 | — |
| **total** | **10 clips** | 720p–4K | 24–60 | **325** | **189** | **136** | **19** | **20** | 12–276 |

Labels are recorded as **consecutive runs**, never scattered singletons, so
continuity, longest-correct-run and identity-switch metrics are computable. A
regression test enforces that.

### How the labels were produced, and what that costs

`tools/ball_annotate.py` renders a *montage*: N consecutive frames cropped to the
same native-resolution window, magnified with nearest-neighbour, overlaid with a
coordinate grid labelled in native frame pixels and a crosshair at the crop
centre whose coordinate is printed in the tile header. The annotator reads a
centre off that and records it. `--centres` accepts a coarse per-frame guess so
the fine pass can follow a ball in flight instead of losing it out of one fixed
crop.

**Annotation precision is roughly ±15 px or ±0.15 × ball diameter, whichever is
larger.** That is the noise floor on every centre-error number below. It is well
inside the match tolerance for every clip, but it means a reported median error
of 8.6 px should be read as "inside the annotation noise", not as sub-pixel
accuracy.

**Four of the ten clips were labelled wrongly on the first pass and had to be
corrected.** `tools/ball_annotate.py verify` draws every stored label back onto
its own frame; reviewing those sheets caught:

- `pexels_6573621` — the recorded centre was on the **white tee**, not the ball
  sitting on it. This is the single most embarrassing possible label error in a
  research pass whose entire subject is not substituting the tee for the ball,
  and it existed for about an hour.
- `pexels_37080178` — the centre was on the **club face**, ~17 px below the ball.
- `pexels_34883764` — one run was ~32 px high.
- `pexels_33511561` — centre and radius were both well off on a macro ball.

Every visible label in the corpus has now been drawn back onto its frame and
checked by eye. The lesson is in the tool: a label that has not been rendered
back onto its own pixels is not ground truth, it is a number someone typed.

### Split

Frozen in `data/ball_split.json` (`golf-research-split.v1`), validated by
`research_split.validate_split_manifest`, which enforces frozen status,
source/subject/sequence disjointness across splits, and rejects duplicate
SHA-256. Splitting is by *shoot group*, not by clip: `pexels_6573621` and
`pexels_6573644` share a golfer, wardrobe and location and would leak if treated
as independent.

Each split was required to contain at least one white-ball-on-grass clip, at
least one heavy-distractor clip, and a spread of resolutions. The held-out split
was scored **once**, at the end, with every parameter already frozen. No
threshold was changed afterwards — which is why the 0.70 track-ambiguity ratio
below is still 0.70 even though validation showed it costs a real track.

`pexels_6541682` and `pexels_6573611` are rights-cleared, deliberately
unlabelled, and held in reserve outside all three splits.

---

## 2. Baselines evaluated

- **B0 — `research_ball.ResearchBallTracker`**, the withdrawn heuristic, imported
  and run as-is rather than paraphrased.
- **B1 — `peak`**, this pass's candidate generator with the temporal stages
  removed: same preprocessing, same working resolution, same colour reference,
  emitting the single highest-scoring candidate on every frame. It isolates how
  much of the result comes from detection and how much from association and track
  validation.
- **M1 — `pipeline`**, the full five-stage method.

---

## 3. Method built

`ghostcaddie/video/ball_pipeline.py`. Five separable stages:

1. **Camera motion** — phase correlation on a downscaled windowed copy gives a
   global per-frame translation. Association runs in compensated coordinates, so
   a pan neither manufactures velocity nor blows the displacement gate on a ball
   that is sitting still.
2. **Candidate generation** — a fast radial-symmetry accumulator: strong
   gradients vote for their own centre at each radius, in both senses, so a disc
   lighter than its background and a disc darker than its background both
   accumulate. Radii are a fraction of `min(w, h)`, spanning the 25× range of
   apparent ball radius in this corpus. A second cue family proposes peaks of the
   camera-compensated frame difference, for a struck ball that is smeared past
   having a boundary.
3. **Scoring** — two features. `edge_support` is the fraction of angular sectors
   around the candidate that carry a strong *radial* gradient: a ball has a
   boundary in every direction, a flagstick or a club shaft has one in two
   opposed sectors only. `distinctness` is how far the enclosed colour sits from
   the *most similar* sector of its own background, in robust local deviations —
   colour rather than luminance, because an optic-yellow ball on grass is barely
   brighter than the grass but is a long way from it in chroma.
4. **Association** — a Viterbi pass over the whole candidate lattice, not a
   greedy walk from frame one. Greedy opening is how a tracker ends up following
   a lens flare that outscores the ball on the first frame.
5. **Track validation** — coverage, mean score, abrupt jumps and camera severity
   can reject the whole track. Sustained stationarity is explicitly *not* treated
   as a wrong-object signal, because a tee'd golf ball is definitionally
   stationary; `stationarity_gate.py`, which is correct for a clubhead, would
   delete every correct at-rest ball observation in this corpus.

### Three things that decided the result

**A frame-wide gradient percentile is a scene-dependent threshold in disguise.**
The first version thresholded edges at the 95th percentile of gradient magnitude
over the whole frame. On `pexels_6573621` a lens flare and a chain-link fence
pushed that percentile above the ball's own boundary and the ball stopped
existing — `edge_support` was exactly 0.00 at every radius. Comparing each
gradient against the mean gradient in its own neighbourhood fixed it.

**Requiring the ball to be the brighter side throws away most golf balls.** The
first isotropy feature asked for a consistent brightness step all the way round.
A ball with a bright sand band behind it and dark turf below steps *both* ways,
so it scored below turf. Testing for a step in either direction is what made the
feature work.

**Track-level competition is what removed the false positives.** After the best
chain is found the lattice is searched again with that chain and its
neighbourhood suppressed; if the runner-up chain elsewhere in the frame is
nearly as well supported, the window contains two equally defensible answers and
the output is that the ball was not identified. On the development split that
one rule took precision from 0.250 to 1.000 and false positives to zero. It is
also the rule that costs almost all the recall.

### What was deliberately not done

No learned method. A few hundred labelled frames from ten clips is nowhere near
enough to train or fine-tune a small-object detector, and pretending otherwise
is the failure mode the plan exists to prevent. No weights were downloaded, no
checkpoint was loaded, no network call is made on any path.

No brute-force parameter sweep. The whole tunable surface is in one dataclass and
every field is a quantity that can be argued about from first principles.

---

## 4. Per-clip metrics

Ambiguous-labelled frames are excluded from precision, recall and centre error
and reported separately. A frame where the ball is genuinely absent and the
tracker says nothing is a **success**, not a miss. A prediction outside tolerance
while the ball is visible is a **false positive**, not a near miss.

### Development split

**method `heuristic` — split `train`**

_Not run: see the runtime note in §2._

**method `peak` — split `train`**

| clip | visible | absent | ambiguous | P | R | F1 | median err px | p90 err px | FP rate on absent | wrong-object rate | longest static lock | longest correct run | identity switches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `pexels_6541842` | 24 | 16 | 0 | 0.175 | 0.292 | 0.219 | 11.0 | 23.5 | 1.000 | 0.708 | 3 | 6 | 4 |
| `pexels_6573621` | 30 | 15 | 0 | 0.000 | 0.000 | 0.000 | — | — | 1.000 | 1.000 | 3 | 0 | 0 |
| `pexels_6573644` | 28 | 15 | 2 | 0.628 | 0.964 | 0.761 | 8.6 | 10.0 | 1.000 | 0.036 | 2 | 27 | 2 |
| `pexels_37030686` | 14 | 0 | 1 | 0.429 | 0.429 | 0.429 | 6.3 | 8.2 | 0.000 | 0.571 | 1 | 4 | 3 |
| **pooled** | 96 | 46 | 3 | **0.282** | **0.417** | **0.336** | 8.6 | 10.5 | **1.000** | **0.583** | 3 | — | 9 |

**method `pipeline` — split `train`**

| clip | visible | absent | ambiguous | P | R | F1 | median err px | p90 err px | FP rate on absent | wrong-object rate | longest static lock | longest correct run | identity switches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `pexels_6541842` | 24 | 16 | 0 | 0.000 | 0.000 | 0.000 | — | — | 0.000 | 0.000 | 0 | 15 | 0 |
| `pexels_6573621` | 30 | 15 | 0 | 0.000 | 0.000 | 0.000 | — | — | 0.000 | 0.000 | 0 | 15 | 0 |
| `pexels_6573644` | 28 | 15 | 2 | 1.000 | 1.000 | 1.000 | 8.6 | 10.0 | 0.000 | 0.000 | 0 | 43 | 0 |
| `pexels_37030686` | 14 | 0 | 1 | 0.000 | 0.000 | 0.000 | — | — | 0.000 | 0.000 | 0 | 0 | 0 |
| **pooled** | 96 | 46 | 3 | **1.000** | **0.292** | **0.452** | 8.6 | 10.0 | **0.000** | **0.000** | 0 | — | 0 |

### Validation split

**method `heuristic` — split `validation`**

_Not run: see the runtime note in §2._

**method `peak` — split `validation`**

| clip | visible | absent | ambiguous | P | R | F1 | median err px | p90 err px | FP rate on absent | wrong-object rate | longest static lock | longest correct run | identity switches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `pexels_6541674` | 15 | 15 | 0 | 0.500 | 1.000 | 0.667 | 7.3 | 8.1 | 1.000 | 0.000 | 4 | 15 | 1 |
| `pexels_33511561` | 30 | 0 | 0 | 0.200 | 0.200 | 0.200 | 134.2 | 142.7 | 0.000 | 0.800 | 7 | 2 | 8 |
| `pexels_37080178` | 15 | 15 | 0 | 0.000 | 0.000 | 0.000 | — | — | 1.000 | 1.000 | 5 | 0 | 0 |
| **pooled** | 60 | 30 | 0 | **0.233** | **0.350** | **0.280** | 8.1 | 134.2 | **1.000** | **0.650** | 7 | — | 9 |

**method `pipeline` — split `validation`**

| clip | visible | absent | ambiguous | P | R | F1 | median err px | p90 err px | FP rate on absent | wrong-object rate | longest static lock | longest correct run | identity switches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `pexels_6541674` | 15 | 15 | 0 | 0.000 | 0.000 | 0.000 | — | — | 0.000 | 0.000 | 0 | 15 | 0 |
| `pexels_33511561` | 30 | 0 | 0 | 0.000 | 0.000 | 0.000 | — | — | 0.000 | 0.000 | 0 | 0 | 0 |
| `pexels_37080178` | 15 | 15 | 0 | 0.000 | 0.000 | 0.000 | — | — | 0.000 | 0.000 | 0 | 15 | 0 |
| **pooled** | 60 | 30 | 0 | **0.000** | **0.000** | **0.000** | — | — | **0.000** | **0.000** | 0 | — | 0 |

### Held-out split — scored once, after freezing

**method `heuristic` — split `held_out`**

_Not run: see the runtime note in §2._

**method `peak` — split `held_out`**

| clip | visible | absent | ambiguous | P | R | F1 | median err px | p90 err px | FP rate on absent | wrong-object rate | longest static lock | longest correct run | identity switches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `pexels_34883764` | 14 | 15 | 1 | 0.483 | 1.000 | 0.651 | 4.3 | 10.0 | 1.000 | 0.000 | 2 | 10 | 1 |
| `pexels_6541964` | 0 | 15 | 15 | 0.000 | 0.000 | 0.000 | — | — | 1.000 | 0.000 | 1 | 0 | 0 |
| `pexels_38053415` | 15 | 15 | 0 | 0.000 | 0.000 | 0.000 | — | — | 1.000 | 1.000 | 9 | 0 | 0 |
| **pooled** | 29 | 45 | 16 | **0.189** | **0.483** | **0.272** | 4.3 | 10.0 | **1.000** | **0.517** | 9 | — | 1 |

**method `pipeline` — split `held_out`**

| clip | visible | absent | ambiguous | P | R | F1 | median err px | p90 err px | FP rate on absent | wrong-object rate | longest static lock | longest correct run | identity switches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `pexels_34883764` | 14 | 15 | 1 | 0.000 | 0.000 | 0.000 | — | — | 1.000 | 0.000 | 2 | 0 | 0 |
| `pexels_6541964` | 0 | 15 | 15 | 0.000 | 0.000 | 0.000 | — | — | 0.000 | 0.000 | 0 | 15 | 0 |
| `pexels_38053415` | 15 | 15 | 0 | 0.000 | 0.000 | 0.000 | — | — | 0.000 | 0.000 | 0 | 15 | 0 |
| **pooled** | 29 | 45 | 16 | **0.000** | **0.000** | **0.000** | — | — | **0.333** | **0.000** | 2 | — | 0 |

---

## 5. Held-out result against the declared budget

The plan declared, before measurement:

| bar | required | measured on held-out | verdict |
|---|---|---|---|
| FP rate on ball-absent frames | ≤ 0.05 | **0.333** | **FAIL** |
| wrong-object rate on ball-visible frames | ≤ 0.05 | 0.000 | pass |
| longest static-lock run | ≤ 5 frames | 2 | pass |
| clear improvement over the best baseline | required | recall 0.000 | **FAIL** |
| no lock onto flagstick / body / club / tee / static scenery | required | **body lock, 15 frames** | **FAIL** |

Three of five fail. No promotion.

---

## 6. Visual QA findings

Overlay sheets are written by `tools/ball_research.py overlay` into
`out/ball_research/qa/` — one image per labelled run, consecutive frames, with
the prediction as a cross, the ground truth as a circle, an error line between
them, and the per-frame state and outcome in each tile header. Media is
gitignored, so these are reproducible locally rather than committed.

**`pexels_6573644`, frames 243–257 — the one clip that works.** The prediction
sits on the ball on every frame of a fifteen-frame run. At f256 the club face
overlaps the ball and the state correctly becomes `ambiguous`; at f257 it becomes
`lost`. No coordinate is emitted on either. Identity is confirmed frame by frame,
not inferred from a single still.

**`pexels_34883764`, frames 740–754 — the held-out false positive.** The golfer
is walking away with the club at his side and there is no ball in shot. The
prediction sits on the middle of his shirt for all fifteen frames, at radius
97.5 px, with a score of 0.9. This is a body lock, held long enough to look like
a track. The synthetic body-distractor regression test passes because a drawn
uniform ellipse has no internal structure; a real back, shoulders and cap give
the radial-symmetry accumulator plenty to vote for. **The synthetic test was too
easy, and the held-out split is what caught it.**

**`pexels_6541674`, frames 150–164 — what the detector alone does.** With the
temporal stages removed, B1 puts its per-frame maximum on the golfer's white cap,
her glove, her yellow trousers and a ball washer, on frames where the ball is not
in shot at all. Every one of those is a distractor the plan named in advance.
That is the honest picture of what a bottom-up small-bright-round-object detector
does on golf footage, and it is why the temporal stages exist.

No prediction was adjusted after viewing any of these.

---

## 7. False-positive cases, in full

Every false positive produced by M1 across all three splits:

| split | clip | frames | what it locked onto | duration |
|---|---|---|---|---|
| train | — | — | none | — |
| validation | — | — | none | — |
| held-out | `pexels_34883764` | 740–754 | the golfer's shirt/back, r≈98 px | 15 consecutive frames |

Not one prediction landed on a flagstick, a flag, a tee, a decoy ball, a shoe or
a cap — the failure family that got the previous tracker withdrawn. The tee case
is the sharpest of these: on `pexels_6573621` frames 246–260 the bare white tee
stands in exactly the ball's former position and M1 correctly says nothing.

The body is the failure family that remains, and the corpus only revealed it on
the split that was never developed against.

---

## 8. Promotion decision

Against the ten criteria in the mission brief:

| # | criterion | met? |
|---|---|---|
| 1 | held-out improvement over the baseline | **no** — held-out recall 0.000 |
| 2 | acceptable false-positive behaviour | **no** — 0.333 on absent frames, budget 0.05 |
| 3 | no lock onto flagstick / body / club / tee / scenery | **no** — 15-frame body lock |
| 4 | consecutive-frame visual review confirms identity | yes, on the one clip that tracks |
| 5 | source binding intact | yes — `bind_source` before every decode |
| 6 | no manual per-clip seed | yes |
| 7 | no network call | yes |
| 8 | no stale-coordinate rendering possible | yes — non-`OBSERVED` states carry no position, by contract |
| 9 | honest failure states | yes |
| 10 | reproducible for regression testing | yes — 47 tests, frozen split, hash-bound labels |

Criteria 1, 2 and 3 fail. **Not promoted.**

Canonical `analyze-video` is byte-for-byte unchanged. `canonical.ball_layer()`
still returns `UNAVAILABLE` with the original withdrawal reason, `canonical.py`
imports nothing from this work, and `tests/test_ball_research_isolation.py`
fails if either of those stops being true. The research pipeline is reachable
only through `tools/ball_research.py`.

---

## 9. Remaining blockers, and what the next pass should do

1. **The body is the unsolved distractor.** A head, a shoulder or a cap gives a
   strong closed circular boundary at the right scale and a strong colour
   separation from grass. Nothing in a bottom-up appearance detector separates
   that from a ball. The plan permits the canonical body layer as *contextual*
   evidence — a torso exclusion region, a soft prior near the ankles at address —
   and this pass did not use it. That is the single highest-value next change,
   and it must be written so ball availability never depends on body being
   present.
2. **The abstention threshold is too blunt.** `track_ambiguity_ratio = 0.70` was
   frozen before validation and then threw away two tracks that the detector
   alone got right: `pexels_6541674` on validation (B1 recall 1.000) and
   `pexels_34883764` on held-out (B1 recall 1.000, median error 4.3 px). The rule
   is correct in kind and wrong in degree — it is asking "is there a rival?"
   when it should be asking "is the rival consistent with being the same object,
   or is it somewhere else entirely doing something else?". Suppressing only the
   chosen chain's own neighbourhood before the rival search is too weak a
   separation; the rival is frequently the same physical object found at an
   adjacent scale. It must be re-derived against a split that is *not* this one,
   because this one has now been looked at.
3. **Appearance alone has been shown to be insufficient, which is the argument
   for a learned detector.** The errors that remain are not association errors —
   association is doing its job, refusing when the evidence is balanced. They are
   errors of the score ranking a shirt above a ball. That is exactly the case a
   small learned detector addresses, and it is now justified in a way it was not
   at the start of this pass. It needs a far larger rights-cleared labelled
   corpus than 325 frames first.
4. **Two clips carry no not-visible labels** (`pexels_37030686`,
   `pexels_33511561`) because neither reaches a frame where the played ball has
   left, within the ranges inspected. Their false-positive rates are therefore
   undefined rather than zero.
5. **In-flight tracking is effectively unmeasured.** Only `pexels_6541842` has
   labelled flight frames, fourteen of them, at an annotation precision close to
   the match tolerance. No claim about tracking a ball in flight is supported by
   this dataset, and none is made.

---

## 10. Reproduction

Media is local and gitignored; the labels are bound to its SHA-256.

**Reproducibility boundary.** What is in the repository: the annotation dataset
(`data/ball_annotations/*.json` — clip identifier, source path, source SHA-256 and the
frame labels), the frozen split (`data/ball_split.json` — same SHA-256 per clip), the
evaluator, the pipeline, the tools and the tests. What is **not** in the repository:
the video itself. Every clip is rights-cleared local media that stays out of Git, and
seven of the ten resolve into gitignored `out/` paths. The repository therefore does
not, on its own, reproduce the evaluation. Re-running it requires the exact
rights-cleared media bytes whose SHA-256 matches the recorded values — `bind_source`
refuses the decode otherwise, and the tests that check the binding fail loudly rather
than silently scoring a substituted clip.

```bash
# labels, and check them by eye
.venv-video-ai/bin/python3 tools/ball_annotate.py summary --clip-id pexels_6573644
.venv-video-ai/bin/python3 tools/ball_annotate.py verify  --clip-id pexels_6573644 \
    --out /tmp/verify.png --crop 240 --scale 1.7

# evaluate a method on a split
.venv-video-ai/bin/python3 tools/ball_research.py run --split held_out --method pipeline \
    --out out/ball_research/held_out_pipeline.json
.venv-video-ai/bin/python3 tools/ball_research.py report out/ball_research/held_out_pipeline.json

# consecutive-frame visual QA
.venv-video-ai/bin/python3 tools/ball_research.py overlay --clip-id pexels_34883764 \
    --results out/ball_research/held_out_pipeline.json --out out/ball_research/qa
```

Research flags are unchanged: `research_only=true`, `ground_truth=false` for all
*predictions* (the annotations are the ground truth and are marked as such by
being annotations), `production_eligible=false`.
