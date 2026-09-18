# Ball dataset v2 — plan

Status date: 2026-09-18. Branch `research/ball-dataset-v2`, base `cba1fdb`.
**Written before any clip is acquired or labelled.** Targets, the state model and the
split policy below are fixed here so they cannot be moved after the numbers are known.

This is a dataset and evaluation-infrastructure plan. It is not a tracker plan, and
nothing in it promotes ball tracking to the canonical path.

---

## 1. Why a second dataset exists

The first pass (`docs/ball-tracking-results.md`) produced ground truth for 10 clips and
ended **FAILED_TO_GENERALIZE**:

- held-out recall 0.000, held-out precision 0.000;
- false-positive rate on ball-absent frames 0.333 against a budget of ≤0.05 declared
  before measurement;
- a fifteen-frame lock onto the golfer's shirt;
- the simpler baseline B1 (same detector, temporal stages deleted) beat the full
  pipeline on held-out.

That corpus is now **development-consumed**. Its split was frozen, then scored, then
read. It cannot support another threshold decision, another prior, or any claim of
improvement on itself, and it is too small and too narrow to train anything: one shoot
group supplies a quarter of the frames, and there is essentially no in-flight coverage.

So: keep the ten clips as a legacy regression corpus, and build a second, larger,
deliberately harder corpus. The point of v2 is to give the *next* research pass enough
independent data that a legitimate attempt at automatic ball tracking is possible at
all — including on data that was never looked at while the method was being built.

## 2. What the dataset is aimed at

The known failure mode, stated as a target: a golf ball is a small, roughly circular,
locally distinct object that competes against a scene full of objects that share two of
those three properties — the golfer's own body and clothing, the tee under the ball, the
clubhead, the flagstick, sun glare, bright grass highlights, white shoes, signage, other
people, and other golf balls. v2 must therefore be dominated by *hard* clips, not by
clips where the ball is a large yellow sphere against flat green — one such clip already
exists and is the only one the previous method tracked.

Deliberately included, not excluded: difficult clips. A corpus of easy clips produces a
tracker that fails on real footage and a metric that hides it.

## 3. Target sizes

Quality targets, not quotas to be met with near-duplicate frames.

| quantity | target | note |
|---|---|---|
| new rights-cleared clips | 40–60 | in addition to the 10 legacy development clips |
| reviewed labelled frames | ≥1,500 | ≥2,500–4,000 preferred if quality holds |
| ball-visible frames | ≥500 | across many clips, not concentrated in one |
| ball-absent frames | substantial | the previous failure was false positives, so negative evidence is first-class |
| occluded frames | meaningful | club face, body, turf in front of the ball |
| ambiguous frames | meaningful | cases the reviewer will not defend |
| hard-negative sequences | every clip where a plausible distractor exists | see §7 |

Prefer fewer, honest, consecutive labels over thousands of nearly identical frames.
Twenty consecutive frames of a static ball teach a tracker far less than five frames
across address, strike and disappearance.

## 4. Required visual diversity

The corpus is built to cover the axes below. `sources.json` records the intended tags
per clip; `tools/ball_dataset.py stats` reports the achieved distribution, including
axes that came out thin, because a diversity claim that is not measured is a wish.

**Camera** — static tripod; handheld; moderate pan; high jitter; zoom; wide; medium;
close.

**Shot type** — driver; iron; wedge; fairway; tee; range; putting; short game where
rights-cleared footage permits.

**Ball appearance** — white; yellow/coloured; strongly lit; shadowed; low contrast;
partially motion-blurred; very small/distant.

**Background** — green fairway; rough; sky; trees; range mats; bunkers; bright clothing;
white shoes; tees; signs; flagsticks; other small bright distractors.

**Player** — different clothing colours; light shirts; dark shirts; patterned shirts;
multiple people in frame where possible.

**Video** — 720p; 1080p; higher where available; multiple frame rates; varied encoding.

## 5. Rights and provenance (non-negotiable)

Permitted: footage that is clearly usable for research under an explicit license, or
user-owned/user-provided. For this pass: **Pexels stock video under the Pexels
License**, plus existing rights-cleared local footage, plus U.S. government (DVIDS)
public-domain footage if used.

Excluded, without exception: PGA TOUR broadcast footage; television broadcasts;
YouTube footage merely because it is publicly viewable; social-media reposts; scraped
copyrighted footage; anything whose license is ambiguous. **If license status is
ambiguous, the clip is excluded. It is never guessed.**

Every acquired source records, in `data/ball_dataset_v2/sources.json`:

`clip_id`, `provider`, `source_url`, `source_identifier`, `title`, `acquisition_date`,
`license`, `license_url`, `license_notes`, `original_filename`, `sha256`, `duration_s`,
`fps`, `width`, `height`, `bytes`, `frames`, `rights_class`, `source_group`,
`intended_tags`, `media_path`, `notes`.

`source_group` is the shoot group: clips sharing a photographer, golfer, wardrobe or
location must share a group, because the split is group-disjoint (§8).

Media is **not committed**. It lives under `out/ball_dataset_v2/media/` (already
gitignored by the repository's `out/` rule) and is addressed by SHA-256 only. A file
whose hash does not match is not a substitute; it is a different clip.

## 6. Label states

Every reviewed frame carries exactly one state:

| state | requires | forbids |
|---|---|---|
| `VISIBLE` | `x`, `y`; optional `radius`/`bbox` | — |
| `NOT_VISIBLE` | `reason` | coordinates |
| `OCCLUDED` | optional `reason` | coordinates unless it is a partial-visibility confidence elsewhere |
| `AMBIGUOUS` | optional candidate region + explanation | coordinates as a defended claim |

Rules: never assign coordinates to `NOT_VISIBLE`; never silently convert an ambiguous
example into a visible one; ambiguous frames are excluded from strict precision/recall
and reported separately rather than dropped.

## 7. Sequences and hard negatives

Labels are **consecutive runs**, never scattered singletons, so continuity, longest
correct run and identity-switch metrics remain computable. A run worth labelling
covers, where the footage supports it: pre-appearance frames, address/setup, the ball at
rest, early motion, launch/in-flight frames when resolvable, temporary occlusion, true
loss, and post-loss frames that still contain distractors.

Negative temporal context is a first-class requirement: a tracker must learn when to
stop rather than jump to another bright object. Every reviewed clip therefore
contributes `NOT_VISIBLE` frames after the ball has left, and a hard-negative tag set
naming what is actually in the frame:

`body`, `clothing`, `shoe`, `club`, `tee`, `flagstick`, `glare`, `background_ball`,
`unknown_bright_object`.

The actual ball state is labelled honestly regardless of the tag. A frame containing a
torrent of distractors *and* the ball is a `VISIBLE` frame with tags, not a negative.

## 8. Split policy — pre-registered

`data/ball_dataset_v2/split.json`, schema `golf-ball-dataset-split.v1`, `status:
frozen`, written **before** any tracker or baseline is run on v2.

- **Group-disjoint**: no `source_group` appears in more than one split. Never split
  frames of one clip across splits, and never split clips of one shoot group across
  splits.
- **SHA-unique**: no `sha256` appears in more than one split.
- Proportions, approximate: train/development ~60%, validation ~20%, final test ~20%.
- The ten legacy clips stay a separate legacy regression corpus and are **never** mixed
  into the v2 final test.
- After freezing: clips do not move between splits because results look bad; thresholds
  are not tuned on final-test performance; final-test predictions are not repeatedly
  inspected.

Selection of which clip goes where is deterministic (`tools/ball_dataset.py split`
assigns by group hash within each axis strata and records the seed), so the assignment
is reproducible rather than chosen by hand after seeing the data.

## 9. Annotation provenance — what may be called ground truth

This pass is annotation **infrastructure plus a reviewed starter set**, and it says so.

- `HUMAN_CONFIRMED` — a person (the user/operator) looked at the frame and confirmed the
  label. Only this may be described as human ground truth.
- `AGENT_REVIEWED` — an AI agent read the magnified pixels and recorded the label. This
  **must not** be called human ground truth. It is machine-reviewed, and it is a weaker
  claim than `HUMAN_CONFIRMED`.
- `AGENT_PROPOSED` — a detector, heuristic, carry-over or optical-flow guess. A proposal
  is not a label. It persists as a proposal until a reviewer accepts, adjusts or rejects
  it, and rejected proposals are kept because they are useful negatives.

No proposal may ever be exported as verified truth automatically. The existing v1
annotations record `annotator: "agent-visual"`; that provenance is preserved as it is
and is **not** upgraded in v2.

## 10. Quality control

`tools/ball_dataset.py validate` must fail on: coordinates outside the frame; a missing
state; `VISIBLE` without coordinates; a non-visible state carrying coordinates; an
impossible radius (≤0 or larger than the frame); duplicate frame entries; a source hash
mismatch between annotation and manifest; a frame index beyond the clip's frame count;
split overlap by clip, by group or by hash; a legacy clip inside the v2 final test;
`HUMAN_CONFIRMED` without a reviewer; a proposal promoted to a reviewed state.

`tools/ball_dataset.py review` flags, without correcting: large frame-to-frame label
jumps, isolated one-frame visibility inside a run, ambiguous clusters, labels sitting
near the match tolerance, and states whose neighbours disagree. Flags are for a human to
adjudicate. **The tooling never rewrites a label.**

## 11. Statistics and QA

`tools/ball_dataset.py stats` reports, per split and overall: clip count, labelled
frames, visible/absent/occluded/ambiguous counts, consecutive run count and longest run,
resolution/fps/provider/group distributions, approximate ball pixel-size distribution,
a measurable camera-motion summary where decoding is available, and the hard-negative
tag distribution. Class imbalance is reported, not hidden.

`tools/ball_dataset.py qa` writes overlay sheets and review strips into gitignored
`out/ball_dataset_v2/qa/`: the label drawn on its own frame, a magnified crop, the state,
the clip and frame id. The purpose is to catch a wrong object, coordinate drift, stale
labels and visibility mistakes by eye — the same discipline that caught four mislabelled
clips in the previous pass.

## 12. Leakage guards

Automated tests assert: no source SHA and no clip id in two splits; no group in two
splits; no legacy clip in the v2 final test; dataset tools default to
`train`/`validation` and require an explicit `--final-test` to touch final test; the
final-test loader refuses without that flag. Accidental leakage should be hard, not
merely discouraged.

## 13. Media status

`tools/ball_dataset.py status` prints, per split, how many clips are present locally and
which are missing, with expected SHA-256 and source reference, so another machine knows
exactly what is absent. Media is never committed, and a differently-hashed file is never
accepted under the same name.

## 14. Acceptance bar for this pass

This pass succeeds if: a materially larger rights-cleared corpus exists locally with
recorded provenance; labels are source-bound; a new group-disjoint split is frozen;
final test is untouched by model development; annotation and review tooling work;
hard negatives are represented; validation passes; statistics are documented; and media
reproducibility boundaries are explicit.

It does **not** require a successful tracker, and it does not license one:
`canonical.ball_layer()` remains `UNAVAILABLE`.

## 15. Non-goals

Not in this pass: promoting ball tracking; changing canonical ball availability;
retuning the failed tracker on the old held-out set; a full tracker build; training a
model; inspecting final-test model performance; claiming the dataset guarantees success;
committing media, generated QA output, `out/`, models or `uv.lock`.
