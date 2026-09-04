# FairwayOS video architecture comparison

## Scope and license audit

Reviewed on 2026-09-03:

- [Ahmed-El-Zainy/soccer](https://github.com/Ahmed-El-Zainy/soccer)
- [footballanalystrohan-glitch/GhostBall-Engine](https://github.com/footballanalystrohan-glitch/GhostBall-Engine)

GhostBall-Engine contains an MIT license with copyright notice `Copyright (c) 2026 Remontada-Analyst`. FairwayOS does not copy its code; its conceptual separation is documented as inspiration only.

The soccer repository page does not expose a LICENSE file at the reviewed `main` revision, and its raw `main/LICENSE` path returned not found. Its README cites Roboflow resources and its requirements pin Ultralytics, Torch, supervision, NumPy, scikit-learn, tqdm, transformers, and related packages. Because repository-level licensing for soccer source was not verifiable, no soccer code, assets, weights, or media are copied or translated. Only high-level publicly described patterns are adapted.

## Comparison

| Concern | soccer pattern | GhostBall pattern | FairwayOS adaptation |
|---|---|---|---|
| Frame iteration | OpenCV/supervision generator, one frame at a time | OpenCV capture and bounded loop | Keep bounded local capture and explicit frame/timestamp provenance |
| Detection | YOLO player/pitch/ball models; ball slicing at 640 px | YOLO + ByteTrack-style tracks | Use existing optional local adapters; keep domain promotion fail-closed |
| IDs/history | supervision tracker IDs and ball buffer | track dictionaries plus velocity history | Add a shared pixel-space observation boundary; no ID implies no promoted track |
| Camera motion | Pitch keypoints and view transform, not a general compensation contract | explicit conceptual separation and homography | retain pixel tracks; make camera-motion warning/calibration separate from observations |
| Coordinates | pitch keypoint correspondence to top-down pitch | `PitchViewTransformer`, forward/inverse homography | reuse FairwayOS four-point coordinate mapper; map only validated observations |
| Rendering | clean annotated frame, optional radar inset | analysis overlay reprojected to original video | render source pixels first; optional course-space/radar layer only when calibration passes |
| Kinematics | implicit tracking/positions | velocity, orientation, spatial relationships | guarded pixel kinematics; no impact/trajectory claims without evidence |
| CLI/dependencies | mode enum and heavy CV stack | single video/output command and dependency preflight | preserve FairwayOS commands; optional video environment remains isolated |
| Failure boundary | mostly demo-oriented detection output | prototype prescriptive analytics | FairwayOS keeps research-only, ground-truth, and production gates explicit |

## Golf mapping

Adaptable patterns:

- player detection/tracker IDs -> one golfer association with stable temporal identity;
- ball buffer/history -> observed ball candidates plus explicit rejection and loss states;
- pitch keypoints/view transformer -> tee/green/course landmarks and existing four-point homography;
- clean-frame annotation -> source-pixel overlay with no mutation of inference input;
- radar inset -> optional top-down course visualization after calibration validation;
- velocity/history -> research-only kinematics with finite-value and jump gates.

Football-specific patterns not adopted as golf claims:

- team color clustering, goalkeeper/referee classes, pitch vertices, pass targets,
  interception rays, dominance heatmaps, and tactical EV;
- generic sports-ball detection as golf-ball evidence;
- arbitrary default homographies for a golf course;
- prescriptive recommendations before validated golf observations.

## Implementation boundary

The strongest safe adaptation is a decoupled pipeline:

```text
local media -> probe/decode -> clean pixel frames
           -> optional detector adapters
           -> validated pixel observations and histories
           -> optional four-point course mapping
           -> guarded kinematics
           -> source-pixel overlay / optional radar
           -> diagnostics + provenance + encoded MP4
```

Existing FairwayOS analytics remains unchanged. Video observations must carry state (`observed`, `interpolated`, `predicted`, `unavailable`), confidence, uncertainty, frame index, timestamp, method, and warnings. Generic proposals remain proposals and cannot be relabeled as golfer, golf ball, clubhead, impact, trajectory, or landing.

## Provenance decision

No third-party source code or assets were copied. Inspiration is attributed here. Any future code reuse requires a verified license, preserved copyright notice, and a separate review of dependency and asset licenses.
