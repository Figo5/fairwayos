# Ball and target scene prototype

The initial demonstrator follows the visible ball across a green camera view and displays an operator-selected cup target. Skeleton graphics are omitted. The run and comparison gallery are saved at `~/fairway-os-runs/fairwayos-shot-intelligence-2026-10-06/`; its `REPORT.md` contains findings, source bindings, media checks and limitations.

`ghostcaddie.video.green_scene_tracker` exposes experimental white-component proposals, masked-ground LK/RANSAC camera similarity registration and quality-gated tracking of a selected ground anchor. It does not claim automatic initialization, full airborne tracking, course calibration, physical speed or general detector accuracy. `coordination/analyze_scene.py` in the run folder binds the exact source SHA256, initializes the ball once and records current-frame detections. `track_target.py` initializes the cup once. Neither fills absent observations.

`ObservedGroundTrail` in `ghostcaddie.video.scene_intelligence` reprojects prior accepted ground observations into each current camera frame. Missing detections, failed registrations, skipped source frames and scene changes clear the trail. A registration matrix is a pixel-to-pixel mapping, not a yard calibration.

`build_strategy_layer(shot, course, player, calibration, binding, config)` reuses the existing strategy pipeline for explicitly supplied golf inputs. `StrategyBinding` declares source SHA256, scene, reference frame and the calibration basis. The resulting layer's `target_in_frame(source_sha256, scene_id, reference_to_current, reference_frame=...)` returns a projected model aim only when that binding matches the camera view. Synthetic calibration is labelled `SCENARIO TARGET`; other inputs remain illustrative model estimates. Yard units, matching player identity and a supported non-putting shot are required. Missing inputs raise an error instead of silently using a sample player.

The current Tiger run uses a selected cup target because reliable hole geometry and Tiger dispersion inputs were not supplied. The optional engine bridge was exercised with explicit synthetic fixtures in tests; its output is not overlaid as a Tiger recommendation.

Focused validation:

```sh
python -m pytest tests/test_scene_intelligence.py tests/test_video_calibration.py tests/test_video_orchestration.py tests/test_pipeline_end_to_end.py tests/test_video_human_orchestration_m5.py -q
```

The subsequent [automatic workflow](automatic-shot-workflow.md) replaces operator initialization with local neural proposals and scene evidence. Its product page starts processing when a user chooses a video. Broader detector evaluation and defensible course/profile inputs remain necessary before adding model aim and landing spread to real-video exports.
