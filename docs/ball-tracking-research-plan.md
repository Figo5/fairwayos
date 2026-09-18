# Ball-tracking research plan

Status date: 2026-09-16. Branch `pga-research-analyzer`, baseline commit `fc76ffe`
(full suite: 1044 passed, 1 skipped, 0 failed — re-verified before this pass began).

This is a **plan**, written before implementation, so that the acceptance bar cannot be
moved after the numbers are known. It is not a results document. Results go in
[`ball-tracking-results.md`](ball-tracking-results.md).

Scope: **ball identity and tracking only**. Not clubhead, not swing speed, not carry,
not 3D, not physical metrics. Even a successful track infers no ball speed, launch
angle, apex, carry or spin — those need validated physical calibration, which does not
exist here.

---

## 1. What makes golf-ball tracking hard in this corpus

1. **The object is small and scale-varying across the corpus by ~30×.** In
   `pexels_6573611` the ball spans roughly a quarter of the frame height; in
   `pexels_38053415` (4096×2160, wide tee shot) it is a handful of pixels. Any absolute
   pixel radius, absolute area, or absolute blur threshold is wrong for most of the
   corpus by construction. Thresholds must be expressed relative to frame dimensions
   and to *local* image statistics.
2. **The ball is not always white.** Four of the rights-cleared clips
   (`6573611`, `6573621`, `6573644`, `6541674`) use optic-yellow range balls. A
   white-only appearance model — which is exactly what the withdrawn
   `ResearchBallTracker` uses — cannot see them at all.
3. **The scene is full of small bright round things that are not the ball.** Measured,
   per clip, from the corpus itself:
   - `34883764`: white flagsticks *and* white flags *and* a scatter of other white balls
     lying on the fairway behind the golfer. This is the exact configuration that
     produced the original falsification (the withdrawn track followed the flagstick
     base).
   - `33511561`, `37080178`, `37030686`: white golf shoes and white caps, similar size
     and far higher contrast than the ball.
   - `6573611`, `6573621`, `6573644`: a **white tee**, directly beneath the ball,
     persisting in exactly the ball's position after the ball leaves. This is the single
     most dangerous distractor in the corpus: it is co-located with the true target and
     becomes visible precisely when the correct answer changes to "gone".
   - `6541674`: the golfer wears **yellow trousers**, a large region matching the
     yellow-ball colour model.
   - `6541682`: strong backlight; the ball is near-silhouetted and often genuinely
     ambiguous.
4. **The correct answer changes state mid-clip and the transition is the thing that
   matters.** At address the ball is stationary for many seconds; after impact it is
   gone within one or two frames. A tracker that is good at stationary objects will
   hold the tee forever; a tracker that requires motion cannot see the ball at address
   at all. The existing `temporal_candidate_gate.accept_candidate_run` rejects any
   candidate with `residual_motion <= 0` — i.e. it discards every correct at-rest ball
   observation in the corpus.
5. **Camera motion is present and varies by two orders of magnitude.** The canonical
   sweep measured median inter-sample global translation from 0.61 px (`38053415`) to
   44.92 px (`6573534`). Global pan trivially manufactures "motion" on every static
   scene element, including the flagstick, and manufactures an apparent displacement
   for a genuinely stationary ball.
6. **Motion blur at impact.** At 24–30 fps a struck ball smears across tens of pixels in
   a single frame and is frequently unrecoverable. The honest label for many of those
   frames is `not visible` or `ambiguous`, not a coordinate.
7. **Occlusion by the club and by the golfer's body** is routine and brief.

---

## 2. Existing methods, and why each failed

Audited: `research_ball.py`, `research_ball_track.py`, `research_ball_model.py`,
`research_ball_evaluation.py`, `hybrid_tracking.py`, `region_tracker_mil.py`,
`fresh_ai_vision.py`, `automatic_perception.py`, `perception.py`, `observations.py`,
`trajectory.py`, `calibration.py`, `contracts.py`, `temporal_candidate_gate.py`,
`stationarity_gate.py`, `source_binding.py`, `research_split.py`, and `tools/demo/*`.

| Route | Where | Why it is disqualified |
|---|---|---|
| Heuristic bright-blob + Hough | `research_ball.ResearchBallTracker` | Falsified on the accepted artifact: followed the flagstick base. Also structurally wrong — absolute luminance floor (`>= 150`), fixed Hough `minRadius=8, maxRadius=45`, HSV white-only appearance (`S<75, V>145`), and a hardcoded bottom-`18%` exclusion band. None of these survive a 720p→4K corpus or a yellow ball. |
| Human-seeded appearance track | `research_ball.SeededBallTracker`, `research_ball_track.py`, `hybrid_tracking.py` | Requires a per-clip human seed. Automatic operation is a hard requirement (promotion criterion 6). |
| BootsTAPIR learned point tracker | `BallTapirAdapter` | Falsified after the source-binding repair (Lipsky genuine, Morikawa 1/51, Gotterup 0/71). Weights absent. Also seed-dependent. |
| SAM2.1 video segmentation | `BallSam2Adapter` | Promptable — needs a seed box. Checkpoint absent. Partial/emerging masks were explicitly ruled not to be object centres. |
| YOLO COCO `sports ball` | `yolo11n.pt` | Explicitly not golf-ball evidence: 124 `sports ball` hits on MMU alongside 25 `clock` and 2 `wine glass`. |
| Fresh AI vision | `fresh_ai_vision.py` | Network-dependent. Promotion criterion 7 forbids it on the canonical path. |
| MIL region tracker + appearance post-filters | `region_tracker_mil.py` | Appearance post-filters (HOG, HSV) were decisively falsified for the clubhead path and are recorded as "do not retune to revive". |

**A specific warning carried forward:** `stationarity_gate.py` rejects any target whose
box stays within 8 px for ≥0.2 s as a "wrong-object false lock". That heuristic is
correct for a clubhead and catastrophically wrong for a ball — a tee'd golf ball is
*definitionally* sustained-stationary. It must never be applied to the ball layer.

### What is still useful

- `source_binding.bind_source()` — fail-closed byte provenance before any decode or
  model import. **Mandatory precondition**, reused unchanged.
- `research_split.validate_split_manifest()` — already enforces frozen status and
  source/subject/sequence disjointness across splits, and rejects duplicate SHA-256.
  Reused unchanged as the split contract.
- The *concept* in `find_static_ball`: contour circularity plus spatial persistence
  across consecutive frames. Reused as a concept; the implementation is not, because it
  takes a manual ROI and a manual `ball_color` argument.
- The *concept* in `temporal_candidate_gate`: candidate competition on a frame is
  itself evidence of ambiguity. Reused as a concept; the implementation is not, for the
  `residual_motion > 0` reason above and its fixed 2–30 px radius window.

---

## 3. Candidate approaches worth testing

All must run locally, offline, with no downloaded weights and no network call.

- **B0 — existing heuristic** (`ResearchBallTracker`), measured as-is. Baseline floor.
- **B1 — scale-normalised bright/contrast blob detector.** Same family as B0 but with
  every threshold expressed relative to frame size and local statistics. Isolates "was
  the old method wrong, or just wrongly parameterised?"
- **B2 — temporal-difference candidate detector** with global camera-motion
  compensation, to separate true local object motion from pan.
- **M1 — the proposed method:** a five-stage pipeline (§ *Candidate pipeline* below)
  combining a scale-normalised, colour-agnostic **local radial-contrast** candidate
  generator, a small explainable scoring function, camera-motion-compensated temporal
  association with bounded acceleration, and explicit track validation that can reject
  the whole track.

Learned methods are *not* pursued in this pass. A labelled set of a few hundred frames
from ten clips is nowhere near enough to train or fine-tune a robust small-object
detector, and pretending otherwise is the failure mode this plan exists to prevent. If
M1's held-out evidence shows the remaining errors are appearance-limited rather than
association-limited, a learned detector becomes the justified next pass — with a
documented source, licence, checksum and safe loader.

---

## 4. Ground truth

The corpus has none, so it is created here.

**Rights:** Pexels and Pixabay clips already retained locally for research. PGA TOUR
material under `out/pga_official_acquisition/` is **excluded** — it is rights-gated by
`pga_source_gate.py` and may not be used for training or evaluation.
`pixabay_149976.mp4` is excluded on content grounds: it is a 3-D cartoon golf-ball
mascot, not golf footage.

**Format** — one JSON file per clip, `data/ball_annotations/<clip_id>.json`:

```json
{
  "schema_version": "golf-ball-annotation.v1",
  "clip_id": "pexels_6573644",
  "video": "research_stock/pexels_6573644.mp4",
  "source_sha256": "af4e...",
  "width": 1920, "height": 1080, "fps": 30.0,
  "annotator": "...", "notes": "...",
  "frames": {
    "120": {"visible": true,  "x": 1412.0, "y": 511.5, "radius": 21.0, "ambiguous": false},
    "121": {"visible": true,  "x": 1409.5, "y": 511.0, "radius": 21.0, "ambiguous": true,
            "reason": "motion blur, centre uncertain"},
    "122": {"visible": false, "reason": "left frame"}
  }
}
```

Rules: `visible:true` requires `x`/`y`; `visible:false` requires `reason` and forbids
coordinates; `ambiguous` may be set in either state and marks a frame as excluded from
strict precision/recall and from centre-error statistics, but retained and reported.
Every write re-verifies `source_sha256` against the media bytes. Frames are labelled in
**consecutive runs**, never as scattered singletons, so continuity and identity-switch
metrics are computable.

**Not fabricated.** No detector output is written into an annotation file, and no
annotation is produced by running any tracker. Labels are read off magnified native-
resolution crops by a human-or-agent annotator through `tools/ball_annotate.py`.

**Annotation utility** (`tools/ball_annotate.py`) supports: frame stepping, a magnified
pixel-grid montage of consecutive frames for efficient reading, recording a labelled
centre, marking not-visible with a reason, marking ambiguous, save/resume, and refusal
to write against mismatched source bytes.

---

## 5. Dataset split

Clip-level, frozen, and **written down before any evaluation is run**. Splitting is by
*shoot group*, not by clip: several Pexels clips share a golfer, wardrobe and location
and would leak across splits if treated as independent. Group membership is recorded in
the manifest as `source_id`/`subject_id`, and `research_split.validate_split_manifest`
rejects any group that straddles two splits.

Each split is required to contain at least one white-ball-on-grass clip, at least one
heavy-distractor clip, and a spread of resolutions, so that no split is a distribution
of its own.

| split | clips | why |
|---|---|---|
| train / dev | `pexels_6541842`, `pexels_6573621`, `pexels_6573644`, `pexels_37030686` | white ball + visible flight; yellow ball + white tee + flare; yellow close-up; white ball with camera pan |
| validation | `pexels_6541674`, `pexels_33511561`, `pexels_37080178` | yellow ball against yellow trousers; white ball on tee against white shoes; white ball with white shoes, hilly course |
| held-out test | `pexels_34883764`, `pexels_6541964`, `pexels_38053415` | flagsticks + flags + scattered decoy balls; dirt/bunker lie where the ball may not be resolvable at all; 4K wide tee shot, ball a few pixels |

`pexels_6541682` and `pexels_6573611` are labelled-in-reserve and excluded from all
three splits; they exist to re-test later without touching held-out.

**Held-out rule.** The held-out split is evaluated **once**, at the end, after all
parameters are frozen. No threshold, no weight and no rule is changed after seeing it.
If held-out results are poor, the outcome is `FAILED_TO_GENERALIZE` — not another
iteration.

---

## 6. What counts as a successful track

Per frame, the prediction is scored against the label as:

- **true positive** — label `visible:true`, prediction `OBSERVED`, and centre error
  ≤ `0.02 × min(width, height)` (a resolution-relative tolerance: 14 px at 720p,
  21 px at 1080p, 43 px at 4K) **or** ≤ 1.5 × the labelled ball radius where a radius
  is recorded, whichever is larger.
- **false positive** — prediction `OBSERVED` while the label is `visible:false`, or
  `OBSERVED` beyond the tolerance while the label is `visible:true` (a wrong-object
  lock counts as a false positive, not merely a miss).
- **false negative** — label `visible:true`, prediction not `OBSERVED`.
- **true negative** — label `visible:false`, prediction not `OBSERVED`. *This is a
  success, not a failure.* Frames where the ball is genuinely not visible are not
  counted as tracking failures.
- Frames labelled `ambiguous` are excluded from precision/recall/error and reported
  separately as an ambiguity rate.

A **track** is successful on a clip when, on the labelled runs of that clip: recall on
visible frames is meaningful, precision is high, the longest continuous correct run
covers a substantial share of the visible sequence, and there are no identity switches
onto a persistent wrong object.

## 7. What counts as failure

- Any frame where the predicted centre sits on the tee, the flagstick, a shoe, a cap, a
  clubhead, a decoy ball or static scenery while the label says the ball is elsewhere or
  absent.
- A track that survives only because the same coordinate was re-emitted after the
  evidence stopped (stale coordinate reuse).
- A state stream that reports `OBSERVED` on a frame where no candidate was actually
  measured.
- A method that only works with a per-clip manual seed, a per-clip colour choice, a
  per-clip ROI, or a network call.

## 8. How false positives are measured

Three separate numbers, reported per clip and never averaged into one score:

1. **FP rate on not-visible frames** — `OBSERVED` predictions on frames labelled
   `visible:false`, over all such frames. This is the tee-after-departure number.
2. **Wrong-object rate on visible frames** — `OBSERVED` predictions outside tolerance
   while the ball is visible, over all visible frames.
3. **Static-lock frames** — the longest run of `OBSERVED` frames whose predicted centre
   stays within 2 px while the labelled ball is absent or has moved. Reported as an
   absolute frame count, because a single long lock is a qualitatively worse failure
   than the same number of scattered errors.

A `FALSE_POSITIVE_BUDGET` is declared now, before measurement, as the promotion bar:
on the held-out split, FP rate on not-visible frames must be **≤ 0.05**, wrong-object
rate on visible frames **≤ 0.05**, and static-lock runs must be **0 frames longer than
5**. Missing the budget means no promotion regardless of how good recall looks.

## 9. Evidence required before promotion

All ten of the criteria in the mission brief, plus: the method must beat the best
baseline on the **held-out** split, not merely on dev; per-clip numbers must be
published including the clips it does badly on; and consecutive-frame overlay review
must confirm the predicted point is on the ball and not on a co-located distractor.

If any of that is missing, the canonical ball layer stays `UNAVAILABLE` and the result
is labelled `PROMISING_BUT_EXPERIMENTAL` or `FAILED_TO_GENERALIZE`.

---

## 10. Candidate pipeline (design)

Five separable stages, each independently testable:

1. **Candidate generation** — scale-normalised, colour-agnostic. Local radial contrast
   (centre-surround) at a small bank of radii derived from frame size, plus
   camera-motion-compensated temporal difference. Emits every plausible small compact
   region; recall-oriented, deliberately over-generating.
2. **Candidate scoring** — a small, explainable feature set: local contrast against an
   annulus, radial symmetry, compactness/aspect, size plausibility relative to frame,
   edge sharpness, and motion residual after camera compensation. No per-clip colour
   prior, no per-clip radius.
3. **Temporal association** — bounded-velocity, bounded-acceleration linking in
   camera-compensated coordinates, with appearance similarity and explicit ambiguity
   when two candidates compete for the same link. Association is allowed to *refuse*.
4. **Track validation** — track-level signals (coverage, mean score, continuity, jump
   count, stationary duration relative to the scene's own motion, candidate
   competition) that can reject the entire track rather than emit a weak one.
5. **State output** — per frame, exactly one of `OBSERVED`, `AMBIGUOUS`, `OCCLUDED`,
   `LOST`, `UNAVAILABLE`, or `INTERPOLATED`. `INTERPOLATED` is never produced by
   default and is never rendered as an observation.

**Camera motion** is estimated globally per frame pair and used in stages 1–4. Where
camera motion is too severe to compensate reliably, confidence is reduced and the
frame's state degrades to `AMBIGUOUS`, not `OBSERVED`.

**Body context** is contextual evidence only — a probable search region and a torso
exclusion. Ball availability never depends on the body layer being present, and body
joints are never used as ball seeds.

**A two-phase (at-rest / in-flight) strategy is permitted** but any phase transition
must be detected from evidence, never assumed from nominal FPS. If impact cannot be
detected, the two states stay separate and no transition is fabricated.

---

## 11. Cost discipline

A small, explainable parameter set — roughly ten numbers, all scale-relative — chosen on
dev, confirmed once on validation, frozen before held-out. No brute-force sweep over
thousands of threshold combinations, because a parameter set found that way on ten clips
is an overfit, not a method.
