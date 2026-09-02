# Luna-guided ROI experiment

This is a research-only experiment for Pexels clip `6573644`. Luna/native-frame
coordinates are approximate guidance, not ground truth or production evidence.

Every record must retain:

```json
{"pseudo_label":true,"ground_truth":false,"research_only":true,"production_eligible":false}
```

The experimental pass is allowed to use the reviewed coordinates to bound a
local ROI. Candidate selection must still enforce ball-size bounds, ROI
containment, golfer/club exclusion, measured camera translation compensation,
consecutive-frame support, step limits, and fail-closed rejection. It must not
rewrite annotations, call `run_pipeline()`, create `ShotEvent`, or produce
calibration, impact, trajectory, landing, analytics, or recommendations.

Luna labels with an unavailable or occluded object use `null`; no hidden ball,
golfer box, clubhead, or shaft coordinate is inferred. Luna/native-frame proposals use the following strict per-frame shape (with numeric zero values when an object is unavailable):

```json
{
  "ball": {"x": 0, "y": 0, "radius": 0, "visible": false, "confidence": 0.0},
  "clubhead": {"x": 0, "y": 0, "visible": false, "confidence": 0.0},
  "shaft": {"x1": 0, "y1": 0, "x2": 0, "y2": 0, "confidence": 0.0},
  "pseudo_label": true,
  "ground_truth": false,
  "research_only": true,
  "production_eligible": false
}
```

The framewise Pexels `6573644` experiment (native frames 60–120) was rejected:
the six-frame proposal run centered near `(549, 493)` while the visible ball was
elsewhere, and no clubhead track was available. This is a visual false-positive
finding, not a threshold-tuning target.
