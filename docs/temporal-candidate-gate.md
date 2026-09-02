# Temporal candidate gate (research-only)

`ghostcaddie.video.temporal_candidate_gate` is a detector-agnostic guard for
pixel-space research proposals. It does not identify a golf ball or clubhead
and must not be used as ground truth, calibration, `ShotEvent`, analytics, or
recommendation input.

The gate rejects proposals that are:

- non-finite or outside the native frame;
- outside the bounded compact-object size range;
- unsupported by measured residual motion;
- inside a supplied person box; or
- part of an ambiguous duplicate frame.

A candidate run is accepted for rendering only when it is unambiguous,
consecutive, and within a bounded inter-frame step. A returned run is still a
research proposal and requires visual inspection of the actual rendered MP4.
Track termination and guarded reacquisition belong to the calling temporal
strategy; this module intentionally does not invent missing observations.

## Current evidence

The gate was tested against the Pexels `6573644` native-FPS experiments, but it
does not turn those experiments into a valid track. The bright-blob and
motion-difference renders produced false positives on background highlights,
edge motion, and non-ball regions. Their markers remain rejected.

All research outputs retain:

```text
pseudo_label=true
ground_truth=false
research_only=true
production_eligible=false
```

Production gates remain closed.
