# Clubhead tracking: local research probe

**Review date:** 2026-08-31
**Scope:** local research-only methods on the existing MMU clip. No production code or production gates changed. Clubhead and impact outputs remain unavailable.

## Result

- `clubhead_available: false`
- `impact_available: false`
- No method below emitted a defensible clubhead identity, validated track, or impact label.
- Generic sports-ball output was not relabeled as clubhead or golf-ball evidence.

## Exact local evidence

Source clip:

- Path: `out/research_training_gauntlet/mmu_candidate/source.mp4`
- SHA-256: `b8f56a4868c1d8324d8949c0bdf670ec34471ec5bb1cb27756bc05b37d6feba6`
- Decoded window: frames `0..164` inclusive (165 frames), `600x480`, encoded `25.0 FPS`, `0.0..6.56 s`
- Existing MMU ball-demo record: `out/research_training_gauntlet/mmu_candidate/analysis/automatic_ball_tracer_diagnostics.json`; it is explicitly a heuristic ball proposal, not clubhead evidence.

### Generic local models

| Model | SHA-256 | Local task/output | Runtime on 165 MMU frames | Finding |
|---|---|---|---:|---|
| `yolo11n.pt` | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` | COCO detection; class names include `sports ball`, no `clubhead` | 17.9874 s CPU | 152/165 frames had outputs; counts: sports ball 124, clock 25, frisbee 1, wine glass 2. Not clubhead evidence. |
| `yolo11n-pose.pt` | `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0` | generic person pose; class names only `person`, no `clubhead` | 20.3483 s CPU | 0/165 frames had outputs. No clubhead evidence. |

Model license/provenance disposition: local files were loaded and exercised successfully, but their production licensing was not independently re-cleared in this probe; research-only. Do not promote either output to GhostCaddie evidence.

### Classical methods

- Farneback dense optical flow: 5.1694 s for the combined classical pass; mean fraction of pixels with magnitude >2 was `0.0452535`, maximum `0.0765764`. This is a motion field, not an object identity or track.
- MOG2 foreground subtraction: same combined-pass runtime, mean foreground fraction `0.0386260`, maximum `1.0`. Foreground is entangled with person/club/background and initialization; it does not identify the clubhead.

The machine-readable full result is `out/research_training_gauntlet/mmu_candidate/analysis/clubhead_tracking_local_probe.json`.

## Safe research candidates and rights

- ClubheadDB package `clubhead_db-1.0.1.tar.gz`: SHA-256 `15a1494a2c28855b84f76e9802f018c3387af976e32c19324c7ab2b62521d1d5`; declared `CC-BY-NC-4.0`; local recon status is `blocked_before_source_video_acquisition`, with no source videos downloaded and no checkpoint found. Its metadata points to third-party YouTube/Reddit media, so it cannot open a production gate.
- GolfDB/SwingNet is research/non-commercial and provides swing-event predictions only; it does not provide clubhead coordinates. Existing local evaluation records no held-out labels and keep clubhead/impact gates false.
- GolfPose is a conditional research lead with an authorization-gated dataset and derived-model terms, but no local acquisition or validation was performed here. It is not an available clubhead model.
- Existing public MMU/YouTube footage is qualitative stress-test material only; no rights-cleared benchmark or matching clubhead annotations are present.

## Gate decision

Keep clubhead and impact unavailable. Do not infer clubhead from ball, person pose, optical flow, foreground masks, or SwingNet event predictions. A future candidate must provide rights-cleared paired frames/annotations, frozen evaluation splits, reproducible hashes, and a validated clubhead/impact evaluator before any gate changes.

## Higher-FPS follow-up

The bounded local follow-up did not produce a qualifying high-FPS clubhead clip:

- `highfps_candidate/source.mp4`: `1920x1080`, encoded `25 FPS`, `1122` frames. The inspected contact sheet is a V1 Sports title-card/logo sequence, not a continuous golf-clubhead shot.
- `impact320_candidate/source.mp4`: `320x240`, encoded approximately `29.97 FPS`, `190` frames. The inspected contact sheet shows a fixed apparatus and ball with `www.photron.com` overlay; no separable golf-clubhead or exact golf impact is visible.

The existing `local_impact` probe is a separate `1080x1920`, `60 FPS` apparatus sequence, not a golf-club swing. Its motion peaks are therefore not impact evidence. These clips remain rejected qualitative research material and were not added to the acceptance set or published.

## Broader high-FPS acquisition pass

A bounded follow-up searched for real golf-swing footage with an explicit local-use path and inspected native frames before any clubhead or impact processing. No candidate passed the complete acquisition-and-visual gate.

### Pixabay candidate: genuine high-FPS metadata, unusable view

- Source page: <https://pixabay.com/videos/golf-sport-golfer-hole-213192/>
- License page: <https://pixabay.com/service/license-summary/>
- Local file: `out/research_training_gauntlet/pixabay_213192/source.mp4`
- Local SHA-256: `93b26cff1cd5e667dd5df00d64d70c3eb1ba25d9db914dfe3a7d583a344385fc`
- Measured stream: `3840x2160`, H.264, `60000/1001` FPS, `354` frames, `5.9059 s`
- Visual decision: rejected. The native contact sheet shows a wide course/aerial view with the golfer too small for separable clubhead inspection; no qualifying close-range continuous swing was established.
- Clubhead and impact: unavailable; no coordinates or event labels were inferred.

The local acquisition record is `out/research_training_gauntlet/pixabay_213192/triage/acquisition_record.json`. The file remains ignored and local; no media or derived frame artifact is publishable.

### YouTube ultra-slow-motion lead: title rate not preserved in stream

- Source page: <https://www.youtube.com/watch?v=b8LEAMlqE0E>
- Local file: `out/research_training_gauntlet/youtube_b8LEAMlqE0E/source.mp4`
- Acquisition: one bounded 10-second segment through the existing allowlisted HTTPS YouTube adapter, with no cookies, credentials, playlist access, or protection bypass.
- Measured stream: `1920x1080`, VP9, `30000/1001` FPS, `300` frames, `10.021 s`
- Source metadata probe: `502 s` total duration, `30 FPS` source metadata; the title's `800 FPS` wording did not correspond to the downloaded stream.
- Visual decision: rejected as a higher-FPS acquisition. The inspected segment shows a real golfer at address, but the downloaded native stream is standard 29.97 FPS and does not satisfy the temporal-resolution gate.
- Clubhead and impact: unavailable; no coordinates or event labels were inferred.

### Wikimedia Commons leads: licensed but not high-FPS

Two CC BY 4.0 Wikimedia Commons files were downloaded only for local inspection. Their file pages provide direct media and explicit attribution terms, but neither stream meets the higher-FPS gate:

- `Golf swing practice - Kanagawa - slow motion - 2023 June 13.webm`: <https://commons.wikimedia.org/wiki/File:Golf_swing_practice_-_Kanagawa_-_slow_motion_-_2023_June_13.webm>. The direct 1,920x1,080 VP9 stream measured `30000/1001` FPS with `414` frames over `13.813 s`. Native inspection shows a real practice swing, but it is standard frame rate rather than higher-FPS capture; it remains a licensed qualitative reference, not an accepted high-FPS candidate.
- `Manpracticinggolfswing-slowmotion-2021-3-24.webm`: <https://commons.wikimedia.org/wiki/File:Manpracticinggolfswing-slowmotion-2021-3-24.webm>. The direct 464x538 VP8 stream measured `30 FPS` with `513` frames over `17.1 s`. Native inspection shows a real practice scene, but the golfer is small and partly confounded by a truck/background; it is not suitable for separable clubhead research.

The “slow motion” labels were not treated as frame-rate evidence. Both files remain local and ignored; no clubhead or impact labels were inferred.

### Existing local 60-FPS candidates

The audit also rechecked the remaining ignored candidates:

- `flight_candidate/source.mp4`: `1080x1920`, `60 FPS`, `428` frames. It shows continuous golfer footage, but the foreground ball is static and tracer/measurement graphics dominate the relevant region; no qualifying separable evidence was established.
- `impact_candidate/source.mp4`: `1080x1920`, `60 FPS`, `732` frames. It is a controlled close-up of ball deformation with title graphics, not a continuous full swing or translating-ball sequence.

Both are rejected for the requested gate. Neither supplies clubhead or impact evidence.

### Research-only clubhead proposal experiment

A dependency-free proposal contract and ignored local visualizer were exercised on the current Wikimedia and MMU files:

- Source contract: `ghostcaddie/video/clubhead_proposal.py`
- Regression tests: `tests/test_clubhead_proposal.py`
- Local runner: `out/research_training_gauntlet/run_clubhead_proposal.py`
- Inputs fused: pose keypoints when available, ROI/exclusion bounds, Canny/Hough line endpoints, contour centers, and frame-to-frame motion contours.
- Outputs: pixel-space proposal point, confidence, disagreement-derived uncertainty, evidence families, warnings, and explicit `research_candidate` provenance. JSON serialization rejects non-finite values and requires `production_eligible: false`.
- Rendered artifacts: `out/research_training_gauntlet/wikimedia_kanagawa/analysis/clubhead_proposals.mp4` (`414` frames) and `out/research_training_gauntlet/mmu_candidate/analysis/clubhead_proposal/clubhead_proposals.mp4` (`1795` frames). Both passed complete FFmpeg decode and FFprobe source/output count checks.
- Visual QA result: rejected as a clubhead track. Remaining circles were not consistently attached to a visibly separable clubhead; MMU proposals frequently followed the ball/background. The report therefore records `clubhead_available: false` and `visual_gate: blocked_visual_false_positive`.
- Exact impact: unavailable. The MMU research bracket remains frames `68–72`; no proposal point was used to narrow it.
- Landing, calibration, ShotEvent, analytics, and recommendation remain `null` and no production pipeline was invoked.

The proposal output is useful as a false-positive diagnostic and visualization scaffold, not as validated clubhead evidence.

### Current acquisition gate

The broader pass therefore remains a verified blocker for higher-FPS clubhead work. A future accepted candidate must have both (1) measured native frame rate sufficient for the intended impact neighborhood and (2) native-frame visual evidence of a continuous, close-enough golf swing with the club separable from the golfer, shaft, ball, overlays, and background. Metadata, titles, slow-motion playback, generic ball motion, and third-party annotations do not satisfy that gate. Production analytics and `run_pipeline()` remain closed.
