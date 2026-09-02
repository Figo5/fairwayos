# Shot-tracer weak supervision (research-only)

`ghostcaddie.video.tracer_weak_supervision` treats shot-tracer graphics as
**graphics**, never as golf-ball observations or ground truth.

## Separation contract

For each clean/source frame and optional tracer-render frame:

1. Detect conservative high-chroma tracer, logo, and UI masks.
2. Record a `TracerHint` from graphic pixels only. Its provenance is
   `tracer_pseudo_hint` and it is not a ball coordinate.
3. Mask graphic pixels in a separate clean-frame copy.
4. Search the clean source independently for compact, neutral, temporally
   supported ball candidates. Tracer coordinates and tracer-derived ROIs are
   never passed to this search.
5. Keep `TracerHint` and `BallObservation` in separate schemas and render layers.
6. Reject missing-ball, ambiguous, static, misleading, or overlay-only clips.

The current implementation is a bounded pixel-space research primitive. It does
not establish object identity, precision/recall, impact, landing, trajectory,
calibration, `ShotEvent`, analytics, or recommendations.

## Render semantics

`build_tracer_render_filter()` uses **magenta** for tracer pseudo-hints and
**lime** for clean-frame ball candidates. Portable legend bars identify the
layers; the JSON sidecar carries their text labels. These colors are diagnostic
conventions, not confidence or truth.

## Bounded local validation

The MMU clean source paired with its separately generated tracer render was
processed at native 600x480/25 FPS for 112 frames. The tracer was detected as a
separate graphic hint, but visual review rejected the lime ball proposals:
they scattered over background, club, and ball-adjacent pixels rather than
following the ball. A 30-frame native 1920x1080/30 FPS Pexels segment with no
tracer and no visible ball was also rejected. These outcomes remain explicit
negative research evidence; no track was promoted.

All outputs must preserve:

```text
pseudo_label=true
ground_truth=false
research_only=true
production_eligible=false
```

## Local evaluation gate

Use a clean source and a separately generated/annotated tracer view only as a
qualitative weak-supervision stress test. Verify native dimensions, FPS, frame
count, and full decode. Inspect the actual rendered MP4 and exhaustive native
contact sheets. A tracer/marker match is not evidence of ball correctness. If
the ball is not independently visible across consecutive source frames, retain
the tracer hint as a rejected pseudo-label and mark the ball unavailable.

No production perception or analytics path imports this module.
