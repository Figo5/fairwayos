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
golfer box, clubhead, or shaft coordinate is inferred. A marker is accepted only
when the experimental candidate survives the temporal gates and is visually
checked against the decoded MP4. Otherwise the render shows `UNAVAILABLE`.
