# Soccer → golf transfer / gap matrix

Written 2026-09-11 for the PGA-ONLY reset. Purpose: decide what may be
transferred from the two reference repositories, what must not be, and what has
no golf analogue at all — before any code is reused.

## Licence gate (checked first, because it decides what is even permissible)

| Repo | Licence | Consequence |
|---|---|---|
| `Ahmed-El-Zainy/soccer` | **No licence file found** on the repository page | Default copyright: **no reuse rights**. Architecture *ideas* may inform our design; **no code may be copied**. Any resemblance must be independently written. |
| `footballanalystrohan-glitch/GhostBall-Engine` | **MIT** | Reuse permitted with attribution and licence text. Still subject to the correctness gates below. |

Neither repo's weights or claimed accuracy transfer. Soccer weights are trained
on soccer pitches, soccer balls and soccer players.

## Transfer matrix

| Soccer capability (as described in the references) | Golf analogue | Verdict | Why |
|---|---|---|---|
| Dedicated player detection weights | Golfer + caddie detection | **Transfer architecture, retrain/replace weights** | Generic COCO person detection already works on golfers. Caddie/player distinction stays **unverified** unless sourced. |
| Dedicated ball weights + **sliced 640px inference** | Golf-ball detection | **Transfer the technique; this is the highest-value idea** | A golf ball is far smaller in frame than a soccer ball. Slicing full-res tiles instead of downscaling the whole frame is exactly the right response to a tiny object. Our current failure mode is that the ball vanishes at downscale. |
| Temporal `BallTracker` | Ball track with gaps | **Transfer, with gaps preserved** | Must keep explicit `unavailable`; never interpolate a ball through occlusion. |
| ByteTrack multi-object tracking | Golfer identity across cuts | **Transfer, but cut-aware** | Golf broadcast cuts constantly. Identity must **reset on camera cut**, not persist across it. |
| Learned pitch landmarks → `ViewTransformer` homography | Course landmarks → ground mapping | **Partial, heavily constrained** | A soccer pitch is a known-size plane with painted lines. A golf hole is **nonplanar** — elevation change, slope, no standard dimensions, few reliable landmarks. |
| Radar / top-down view | Top-down hole view | **Only where a mapping is actually supported** | Requires landmarks we can genuinely resolve. Absent those, no radar. |
| Ball position in **metre space** | Ball in yards/metres | **DO NOT TRANSFER as-is** | See hard bans below. |
| `remontada_engine.py` xT / EV / dominance | Strokes-gained / shot EV | **DO NOT TRANSFER** | The reference is heuristic, not calibrated. Golf equivalents (strokes-gained) are defined against ShotLink baselines we do not have. |

## Hard bans (these are correctness gates, not style preferences)

1. **Never apply a ground homography to an airborne ball.** Soccer's ball is
   overwhelmingly on the plane; a golf ball's entire interesting life is 3D
   flight. A ground homography applied to a flying ball yields a number that
   looks like yards and is meaningless. No yardage, no ball speed, no true
   trajectory from a ground plane.
2. **Never copy GhostBall's fallbacks.** `pipeline_integration.py` correctly
   separates raw bboxes from mapped positions, then falls back to **pixel
   positions in metre space** and a **default ball at `[52, 34]`** when mapping
   fails. That converts a failure into a confident wrong coordinate. Our
   equivalent of a mapping failure is `unavailable`.
3. **Never present heuristic scores as calibrated probability.** A qualitative
   scenario overlay may ship *labelled uncalibrated*; it is not a
   recommendation, an EV, or a strokes-gained figure.
4. **Manual seeds stay labelled.** Allowed only when explicitly marked, with
   native-frame evidence and provenance. Never marketed as automatic detection,
   never used as independent ground truth for evaluating the thing that produced
   them.

## Gaps with no soccer analogue

| Gap | Status |
|---|---|
| Clubhead tracking | Falsified locally at available resolution/frame rate. **Not to be retuned.** |
| Impact/contact instant | Unavailable. SwingNet gives a candidate bracket only. |
| Ball flight in 3D | No monocular method available to us that is honest about depth. |
| Course geometry (hole length, elevation, pin position) | No source. Not inferable from broadcast pixels. |
| Player/caddie identity | Only from on-screen captions or a sourced record; never inferred from appearance. |

## What this implies for the build order

1. **Sliced high-resolution ball inference** is the single highest-value transfer
   and directly addresses the measured failure (ball lost at downscale).
2. **Cut-aware tracking with explicit loss** is second: golf broadcast cuts are
   the dominant failure mode, and the current tracker has no cut reset.
3. **Course mapping is deferred** until landmarks are actually resolvable, and
   even then it is ground-only, never applied to a ball in flight.
