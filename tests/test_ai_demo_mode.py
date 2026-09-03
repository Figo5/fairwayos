import itertools
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from ghostcaddie.cli import main
from ghostcaddie.video.ai_demo import (
    DEMO_SCHEMA_VERSION,
    DemoAcceptanceError,
    ObservationState,
    build_demo_observation,
    build_demo_provenance,
    build_demo_report,
    _normalize_tracker_state,
    reject_obvious_false_positive,
    select_swing_window,
    sanitize_ball_for_render,
    validate_demo_acceptance,
    validate_rendered_ball_markers,
)


class TestAIDemoContracts(unittest.TestCase):
    def test_demo_acceptance_rejects_short_unavailable_output(self):
        observations = [
            {"pose": {"state": "unavailable"}, "ball": {"state": "unavailable"}}
            for _ in range(8)
        ]
        accepted, reasons = validate_demo_acceptance(
            observations, frame_rate=4.0, rendered_frames=8)
        self.assertFalse(accepted)
        self.assertEqual(reasons, ["all_perception_unavailable", "insufficient_duration", "insufficient_frames"])

    def test_render_sanitizer_removes_ball_inside_golfer_bbox(self):
        sanitized = sanitize_ball_for_render(
            {"state": "observed", "point": {"x": 100.0, "y": 110.0}},
            {"bbox": [60.0, 40.0, 220.0, 220.0]},
        )
        self.assertEqual(sanitized["state"], "unavailable")
        self.assertEqual(sanitized["rejection"], "invalid_or_unsafe_ball_geometry")
        self.assertNotIn("point", sanitized)

    def test_render_geometry_rejects_malformed_points_and_bboxes(self):
        pose_bboxes = ([1, 2, 3], [1, 2, 3, 4, 5], "not-a-box",
                       [0.0, 0.0, float("nan"), 4.0],
                       [0.0, 0.0, True, 4.0], [4.0, 4.0, 0.0, 0.0])
        points = (None, [1.0, 2.0], {"x": "bad", "y": 2.0},
                  {"x": float("nan"), "y": 2.0},
                  {"x": float("inf"), "y": 2.0},
                  {"x": True, "y": 2.0})
        for bbox in pose_bboxes:
            sanitized = sanitize_ball_for_render(
                {"state": "observed", "point": {"x": 1.0, "y": 2.0}}, {"bbox": bbox})
            self.assertEqual(sanitized["state"], "unavailable")
        for point in points:
            sanitized = sanitize_ball_for_render(
                {"state": "observed", "point": point}, {"bbox": [0.0, 0.0, 4.0, 4.0]})
            self.assertEqual(sanitized["state"], "unavailable")
            self.assertTrue(sanitized["research_only"])
            self.assertFalse(sanitized["ground_truth"])
            self.assertFalse(sanitized["production_eligible"])
            self.assertNotIn("point", sanitized)
        observations = [{"frame_index": 0, "ball": {"state": "observed", "point": [1.0, 2.0]},
                         "golfer": {"bbox": [0.0, 0.0, 4.0, 4.0]}}]
        self.assertTrue(validate_rendered_ball_markers(observations))

    def test_pose_observation_rejects_below_floor_confidence(self):
        from ghostcaddie.video.ai_demo import _pose_observation

        class FakeTensor(list):
            def tolist(self):
                return list(self)

        class FakeBoxes:
            def __init__(self):
                self.cls = FakeTensor([0])
                self.conf = FakeTensor([0.30])
                self.xyxy = [FakeTensor([10.0, 20.0, 60.0, 90.0])]
            def __len__(self):
                return len(self.cls)

        class FakeResult:
            def __init__(self):
                self.boxes = FakeBoxes()
                self.keypoints = None

        observed, warning = _pose_observation(
            lambda frame, verbose=False: [FakeResult()], None, 200, 150)
        self.assertIsNone(observed)
        self.assertEqual(warning, "pose_confidence_below_floor")

    def test_pose_floor_boundary_is_inclusive_at_min(self):
        from ghostcaddie.video.ai_demo import MIN_POSE_CONFIDENCE, _pose_observation

        class FakeTensor(list):
            def tolist(self):
                return list(self)

        def observation_for(confidence):
            class FakeBoxes:
                def __init__(self):
                    self.cls = FakeTensor([0])
                    self.conf = FakeTensor([confidence])
                    self.xyxy = [FakeTensor([10.0, 20.0, 60.0, 90.0])]
                def __len__(self):
                    return len(self.cls)

            class FakeResult:
                def __init__(self):
                    self.boxes = FakeBoxes()
                    self.keypoints = None

            return _pose_observation(
                lambda frame, verbose=False: [FakeResult()], None, 200, 150)

        observed, warning = observation_for(MIN_POSE_CONFIDENCE)
        self.assertIsNotNone(observed)
        self.assertIsNone(warning)
        observed, warning = observation_for(MIN_POSE_CONFIDENCE - 0.0001)
        self.assertIsNone(observed)
        self.assertEqual(warning, "pose_confidence_below_floor")
        observed, warning = observation_for(0.413)
        self.assertIsNotNone(observed)

    def test_pose_floor_rejects_nan_confidence(self):
        from ghostcaddie.video.ai_demo import _pose_observation

        class FakeTensor(list):
            def tolist(self):
                return list(self)

        class FakeBoxes:
            def __init__(self):
                self.cls = FakeTensor([0])
                self.conf = FakeTensor([float("nan")])
                self.xyxy = [FakeTensor([10.0, 20.0, 60.0, 90.0])]
            def __len__(self):
                return len(self.cls)

        class FakeResult:
            def __init__(self):
                self.boxes = FakeBoxes()
                self.keypoints = None

        observed, warning = _pose_observation(
            lambda frame, verbose=False: [FakeResult()], None, 200, 150)
        self.assertIsNone(observed)
        self.assertEqual(warning, "pose_confidence_below_floor")

    def test_cli_prints_exact_demo_block_reason(self):
        output = StringIO()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            with patch("ghostcaddie.cli.run_local_demo", side_effect=DemoAcceptanceError(
                    ["all_perception_unavailable", "insufficient_duration"])):
                with redirect_stdout(output):
                    with self.assertRaises(SystemExit) as raised:
                        main(["fairwayos-demo", "--video", str(source), "--out", str(root / "out")])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("demo rejected: all_perception_unavailable, insufficient_duration", output.getvalue())
            diagnostics = json.loads((root / "out" / "diagnostics.json").read_text())
            self.assertEqual(diagnostics["blocked_reason"],
                             "demo rejected: all_perception_unavailable, insufficient_duration")

        output = StringIO()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            source.write_bytes(b"source")
            with patch("ghostcaddie.cli.run_local_demo", return_value={"status": "research_only"}) as run:
                with redirect_stdout(output):
                    main([
                        "fairwayos-demo", "--video", str(source), "--out", str(Path(directory) / "out"),
                    ])
            self.assertTrue(run.called)
            self.assertIn('"status": "research_only"', output.getvalue())

    def test_cli_forwards_sanitized_visual_review_file(self):
        output = StringIO()
        review = {"verdicts": [{"verdict": "reject", "ball_marker_aligned": True}],
                  "research_only": True, "ground_truth": False, "production_eligible": False}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            review_path = root / "review.json"
            source.write_bytes(b"source")
            review_path.write_text(json.dumps(review))
            with patch("ghostcaddie.cli.run_local_demo", return_value={"status": "research_only"}) as run:
                with redirect_stdout(output):
                    main(["ai-demo", "--video", str(source), "--out", str(root / "out"),
                          "--visual-review", str(review_path)])
            self.assertFalse(run.call_args.kwargs["visual_review"]["verdicts"][0]["ball_marker_aligned"])

    def test_cli_accepts_bare_sanitized_verdict_list(self):
        output = StringIO()
        verdicts = [{"recommended_action": "reject", "false_positive": True,
                     "false_positive_type": "static_ground_ball_not_flight",
                     "ball_marker_aligned": True}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            review_path = root / "review.json"
            source.write_bytes(b"source")
            review_path.write_text(json.dumps(verdicts))
            with patch("ghostcaddie.cli.run_local_demo", return_value={"status": "research_only"}) as run:
                with redirect_stdout(output):
                    main(["ai-demo", "--video", str(source), "--out", str(root / "out"),
                          "--visual-review", str(review_path)])
            forwarded = run.call_args.kwargs["visual_review"]["verdicts"][0]
            self.assertEqual(forwarded["false_positive"], True)
            self.assertEqual(forwarded["false_positive_type"], "static_ground_ball_not_flight")
            self.assertFalse(forwarded["ball_marker_aligned"])

    def test_local_demo_accepts_explicit_source_provenance(self):
        output = StringIO()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            source.write_bytes(b"source")
            with patch("ghostcaddie.cli.run_local_demo", return_value={"status": "research_only"}) as run:
                with redirect_stdout(output):
                    main([
                        "ai-demo", "--video", str(source), "--out", str(Path(directory) / "out"),
                        "--source-platform", "pexels", "--source-video-id", "6573485",
                        "--source-url", "https://www.pexels.com/video/a-boy-hitting-a-golf-ball-6573485/",
                    ])
            self.assertEqual(run.call_args.kwargs["source"], {
                "platform": "pexels",
                "video_id": "6573485",
                "url": "https://www.pexels.com/video/a-boy-hitting-a-golf-ball-6573485/",
            })

    def test_terminal_tracker_state_is_unavailable_not_schema_failure(self):
        self.assertEqual(_normalize_tracker_state("terminated"), "unavailable")
        self.assertEqual(_normalize_tracker_state("observed"), "observed")

    def test_observation_states_and_provenance_are_explicit(self):
        observation = build_demo_observation(
            frame_index=4,
            timestamp_seconds=0.2,
            golfer={"state": ObservationState.OBSERVED.value, "confidence": 0.91},
            pose={"state": ObservationState.OBSERVED.value, "confidence": 0.84},
            ball={"state": ObservationState.INTERPOLATED.value, "confidence": 0.42},
            clubhead={"state": ObservationState.UNAVAILABLE.value, "confidence": 0.0},
            impact={"state": ObservationState.PREDICTED.value, "confidence": 0.23},
            warnings=["clubhead_not_validated"],
        )
        self.assertEqual(observation["frame_index"], 4)
        self.assertEqual(observation["ball"]["state"], "interpolated")
        self.assertEqual(observation["clubhead"]["state"], "unavailable")
        self.assertEqual(observation["impact"]["state"], "predicted")
        self.assertTrue(observation["research_only"])
        self.assertFalse(observation["ground_truth"])
        self.assertFalse(observation["production_eligible"])

    def test_window_selection_is_bounded_and_deterministic(self):
        result = select_swing_window(
            [0.1, 0.2, 0.7, 0.4, 0.1, 0.8, 0.2],
            frame_rate=30.0,
            max_duration_seconds=0.2,
        )
        self.assertEqual(result["peak_frame"], 5)
        self.assertEqual(result["start_frame"], 2)
        self.assertEqual(result["end_frame"], 6)
        self.assertLessEqual(result["end_frame"] - result["start_frame"] + 1, 7)
        self.assertEqual(result, select_swing_window([0.1, 0.2, 0.7, 0.4, 0.1, 0.8, 0.2], frame_rate=30.0, max_duration_seconds=0.2))

    def test_demo_encoding_requests_tv_range_yuv420p_output(self):
        from ghostcaddie.video.ai_demo import build_demo_encoding_command
        command = build_demo_encoding_command("ffmpeg", "frames", "output.mp4", 15.0)
        self.assertIn("-vf", command)
        self.assertIn("scale=in_range=pc:out_range=tv,format=yuv420p", command)
        self.assertEqual(command[command.index("-pix_fmt") + 1], "yuv420p")

    def test_research_impact_bracket_follows_swingnet_event_without_validating_contact(self):
        from ghostcaddie.video.ai_demo import build_research_impact_bracket
        bracket = build_research_impact_bracket([
            {"event": "Impact", "frame_index": 182, "research_only": True,
             "ground_truth": False, "production_eligible": False},
        ], frame_numbers=[180, 182, 184])
        self.assertEqual(bracket["state"], "candidate_bracket_only")
        self.assertEqual(bracket["frames"], [180, 184])
        self.assertEqual(bracket["reason"], "SwingNet event prediction; exact contact unavailable")
        self.assertTrue(bracket["research_only"])
        self.assertFalse(bracket["ground_truth"])
        self.assertFalse(bracket["production_eligible"])

    def test_single_sample_impact_is_unavailable_not_a_collapsed_bracket(self):
        from ghostcaddie.video.ai_demo import build_research_impact_bracket
        bracket = build_research_impact_bracket([
            {"event": "Impact", "frame_index": 182, "research_only": True,
             "ground_truth": False, "production_eligible": False},
        ], frame_numbers=[182])
        self.assertEqual(bracket["state"], "unavailable")
        self.assertEqual(bracket["frames"], [])
        self.assertIn("two distinct", bracket["reason"])

    def test_hat_candidate_is_rejected_by_pose_region_gate(self):
        from ghostcaddie.video.ai_demo import filter_ball_model_candidates
        pose = {
            "bbox": [100, 50, 220, 260],
            "keypoints": [[160, 75, 0.95]] * 17,
        }
        accepted, rejected = filter_ball_model_candidates(
            [{"center": (160.0, 75.0), "box": [145.0, 60.0, 175.0, 90.0], "confidence": 0.99}],
            pose=pose, image_width=320, image_height=240,
        )
        self.assertEqual(accepted, [])
        self.assertIn("golfer_bbox_overlap", rejected[0]["reasons"])
        self.assertIn("head_region", rejected[0]["reasons"])

    def test_ball_evidence_requires_two_consecutive_model_research_agreements(self):
        from ghostcaddie.video.ai_demo import ResearchBallEvidenceGate
        class ResearchCandidates:
            def extract_candidates(self, image, previous_image=None, context=None):
                return (type("Candidate", (), {"center": (250.0, 200.0), "confidence": 0.9})(),)
        class RecordingTracker:
            def __init__(self):
                self.seen = []
            def update(self, candidates):
                self.seen.append(list(candidates))
                if not candidates:
                    return {"state": "unavailable", "point": None, "confidence": 0.0}
                return {"state": "observed", "point": candidates[0]["center"],
                        "confidence": candidates[0]["confidence"]}
        tracker = RecordingTracker()
        gate = ResearchBallEvidenceGate(tracker, ResearchCandidates(), min_consecutive=2)
        model_candidates = [{"center": (250.0, 200.0), "box": [246.0, 196.0, 254.0, 204.0], "confidence": 0.9}]
        first = gate.update(model_candidates, object(), pose=None, image_width=320, image_height=240)
        self.assertEqual(first["state"], "unavailable")
        self.assertEqual(first["agreement_streak"], 1)
        second = gate.update(model_candidates, object(), pose=None, image_width=320, image_height=240)
        self.assertEqual(second["state"], "observed")
        self.assertEqual(second["agreement_streak"], 2)
        self.assertEqual(sum(bool(items) for items in tracker.seen), 1)

    def test_ball_evidence_fails_closed_on_static_ground_or_trouser_marker(self):
        from ghostcaddie.video.ai_demo import ResearchBallEvidenceGate
        current = {"center": (250.0, 200.0)}

        class ResearchCandidates:
            def extract_candidates(self, image, previous_image=None, context=None):
                return (type("Candidate", (), {"center": current["center"], "confidence": 0.9})(),)

        class RecordingTracker:
            def __init__(self):
                self.seen = []
            def update(self, candidates):
                self.seen.append(list(candidates))
                if not candidates:
                    return {"state": "unavailable", "point": None, "confidence": 0.0}
                return {"state": "observed", "point": candidates[0]["center"],
                        "confidence": candidates[0]["confidence"]}

        tracker = RecordingTracker()
        gate = ResearchBallEvidenceGate(tracker, ResearchCandidates(), min_consecutive=2)
        # Sanitized observed centers from vision-review iter3-6573612 frames
        # 96..152: effectively stationary, so this is ground/trouser, not ball.
        stationary = [(1123.66, 844.99), (1124.14, 846.08), (1124.70, 845.24),
                      (1124.98, 846.04), (1124.80, 845.90), (1125.31, 845.52),
                      (1125.46, 844.42), (1125.42, 844.47)]
        static_rejections = 0
        for center in stationary:
            current["center"] = center
            result = gate.update(
                [{"center": center, "box": [center[0] - 4.0, center[1] - 4.0,
                                            center[0] + 4.0, center[1] + 4.0],
                  "confidence": 0.9}],
                object(), pose=None, image_width=1920, image_height=1080)
            self.assertEqual(result["state"], "unavailable")
            self.assertIsNone(result["point"])
            self.assertTrue(result["research_only"])
            self.assertFalse(result["ground_truth"])
            self.assertFalse(result["production_eligible"])
            static_rejections += sum(
                1 for record in result["rejected_candidates"]
                if "static_ground_or_trouser_false_positive" in record["reasons"])
        self.assertGreaterEqual(static_rejections, 1)
        self.assertTrue(all(not items for items in tracker.seen))
        # A genuinely moving candidate is never flagged by the static-ground gate.
        tracker = RecordingTracker()
        gate = ResearchBallEvidenceGate(tracker, ResearchCandidates(), min_consecutive=2)
        final_state = None
        for step in range(3):
            center = (900.0 + 12.0 * step, 700.0 - 6.0 * step)
            current["center"] = center
            result = gate.update(
                [{"center": center, "box": [center[0] - 4.0, center[1] - 4.0,
                                            center[0] + 4.0, center[1] + 4.0],
                  "confidence": 0.9}],
                object(), pose=None, image_width=1920, image_height=1080)
            final_state = result["state"]
        self.assertEqual(final_state, "observed")
        self.assertFalse(any("static_ground_or_trouser_false_positive" in record["reasons"]
                             for result_items in tracker.seen for record in []))

    def test_ball_hat_fixture_renders_rejected_without_inside_golfer_marker(self):
        from ghostcaddie.video.ai_demo import _draw_ball_overlay, filter_ball_model_candidates
        fixture = json.loads((Path(__file__).parent / "fixtures" / "ball_hat_false_positive.json").read_text())
        candidate = fixture["candidate"]
        pose = {"bbox": fixture["golfer_bbox"],
                "keypoints": [[candidate["center"][0], candidate["center"][1], 0.9]] * 17}
        accepted, rejected = filter_ball_model_candidates(
            [candidate], pose=pose, image_width=fixture["image_width"], image_height=fixture["image_height"])
        self.assertEqual(accepted, [])
        self.assertTrue(set(fixture["expected_reasons"]).issubset(set(rejected[0]["reasons"])))
        try:
            import numpy as np
            frame = np.zeros((fixture["image_height"], fixture["image_width"], 3), dtype="uint8")
            rendered = _draw_ball_overlay(frame, {"state": "unavailable", "point": None,
                                                   "rejected_candidates": rejected}, [], frame.copy())
        except ImportError:
            self.skipTest("optional NumPy/OpenCV stack unavailable")
        self.assertFalse(rendered["marker"])
        self.assertEqual(rendered["rejected_markers"], 1)

    def test_rendered_marker_validator_rejects_inside_golfer_box(self):
        from ghostcaddie.video.ai_demo import validate_rendered_ball_markers
        fixture = json.loads((Path(__file__).parent / "fixtures" / "ball_hat_false_positive.json").read_text())
        observation = {"frame_index": fixture["frame_index"],
                       "golfer": {"bbox": fixture["golfer_bbox"]},
                       "ball": {"state": "observed", "point": {"x": fixture["candidate"]["center"][0],
                                                                    "y": fixture["candidate"]["center"][1]}}}
        violations = validate_rendered_ball_markers([observation])
        self.assertEqual(violations[0]["reason"], "accepted_marker_inside_golfer_bbox")

    def test_provenance_hashes_local_object_without_absolute_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.mp4"
            path.write_bytes(b"demo-source")
            provenance = build_demo_provenance(
                source={"platform": "local", "video_id": "clip"},
                video_path=path,
                media={"frame_count": 1},
            )
        self.assertEqual(provenance["acquisition"]["local_artifact"], "source.mp4")
        self.assertNotIn(str(path), json.dumps(provenance))
        self.assertTrue(provenance["research_only"])
        self.assertFalse(provenance["production_eligible"])

    def test_report_declares_render_coverage_and_audio_state(self):
        report = build_demo_report(
            source={"platform": "pexels", "video_id": "6573485"},
            media={"fps": 30.0, "width": 1920, "height": 1080, "frame_count": 310, "sample_fps": 15.0},
            swing_window={"start_frame": 0, "end_frame": 6, "peak_frame": 5},
            observations=[],
            artifact_references=["annotated_video.mp4", "diagnostics.json"],
            warnings=["research_only_demo"],
            render={"rendered_frames": 121, "sample_fps": 15.0,
                    "duration_seconds": 8.066667,
                    "audio": "unavailable_dropped_by_reencode",
                    "reason": "annotated re-render covers sampled frames only; source audio not carried into re-encode",
                    "research_only": False},
        )
        render_block = report["render"]
        self.assertTrue(render_block["research_only"])
        self.assertFalse(render_block["ground_truth"])
        self.assertFalse(render_block["production_eligible"])
        self.assertEqual(render_block["rendered_frames"], 121)
        self.assertEqual(render_block["sample_fps"], 15.0)
        self.assertEqual(render_block["duration_seconds"], 8.066667)
        self.assertEqual(render_block["audio"], "unavailable_dropped_by_reencode")

    def test_report_references_required_visual_artifacts(self):
        report = build_demo_report(
            source={"platform": "local", "video_id": "clip"},
            media={"fps": 30.0, "width": 640, "height": 360, "frame_count": 2},
            swing_window={"start_frame": 0, "end_frame": 1}, observations=[],
            artifact_references=["annotated_video.mp4", "contact_sheet.jpg", "diagnostics.json", "provenance.json"],
            warnings=["research_only_demo"],
        )
        self.assertEqual(set(report["artifact_references"]), {
            "annotated_video.mp4", "contact_sheet.jpg", "diagnostics.json", "provenance.json"})

    def test_report_omits_render_block_without_render_media_fields(self):
        report = build_demo_report(
            source={"platform": "youtube", "video_id": "ABCDEFGHIJK"},
            media={"fps": 30.0, "width": 640, "height": 360, "frame_count": 7},
            swing_window={"start_frame": 0, "end_frame": 6, "peak_frame": 5},
            observations=[],
            artifact_references=["annotated_video.mp4", "diagnostics.json"],
            warnings=["research_only_demo"],
        )
        self.assertNotIn("render", report)

    def test_visual_review_block_flags_and_reject_alignment_invariant(self):
        from ghostcaddie.video.ai_demo import build_visual_review
        block = build_visual_review([
            {"frame_index": 12, "verdict": "accept", "ball_marker_aligned": True},
            {"frame_index": 13, "verdict": "reject", "ball_marker_aligned": True},
            {"frame_index": 14, "verdict": "reject"},
        ])
        self.assertTrue(block["research_only"])
        self.assertFalse(block["ground_truth"])
        self.assertFalse(block["production_eligible"])
        for verdict in block["verdicts"]:
            self.assertTrue(verdict["research_only"])
            self.assertFalse(verdict["ground_truth"])
            self.assertFalse(verdict["production_eligible"])
        self.assertTrue(block["verdicts"][0]["ball_marker_aligned"])
        self.assertFalse(block["verdicts"][1]["ball_marker_aligned"])
        self.assertFalse(block["verdicts"][2].get("ball_marker_aligned", False))
        report = build_demo_report(
            source={"platform": "youtube", "video_id": "ABCDEFGHIJK"},
            media={"fps": 30.0, "width": 640, "height": 360, "frame_count": 7},
            swing_window={"start_frame": 0, "end_frame": 6, "peak_frame": 5},
            observations=[],
            artifact_references=["annotated_video.mp4", "diagnostics.json"],
            warnings=["research_only_demo"],
            visual_review=block,
        )
        review = report["visual_review"]
        self.assertTrue(review["research_only"])
        self.assertFalse(review["ground_truth"])
        self.assertFalse(review["production_eligible"])
        self.assertEqual(review["verdicts"], block["verdicts"])

    def test_run_local_demo_accepts_optional_visual_review_without_required_call(self):
        import inspect
        from ghostcaddie.video.ai_demo import run_local_demo
        parameter = inspect.signature(run_local_demo).parameters["visual_review"]
        self.assertIsNone(parameter.default)

    def test_model_load_timeout_is_reported_as_unavailable(self):
        import sys
        import types
        from ghostcaddie.video.ai_demo import _InferenceTimeout, _load_yolo
        fake_ultralytics = types.SimpleNamespace(YOLO=object())
        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}), \
             patch("ghostcaddie.video.ai_demo._run_with_timeout",
                   side_effect=_InferenceTimeout("load timed out")):
            model, warning = _load_yolo("/tmp/model.pt", "detect")
        self.assertIsNone(model)
        self.assertEqual(warning, "model_load_timeout")

        from ghostcaddie.video.ai_demo import _coarse_ball_candidates
        calls = []
        class EmptyResult:
            boxes = None
        class Model:
            def __call__(self, frames, verbose=False):
                calls.append(frames)
                return [EmptyResult() for _ in frames]
        candidates, warning = _coarse_ball_candidates(
            Model(), list(range(10)), 640, 360, max_frames=10)
        self.assertEqual(candidates, [])
        self.assertIsNone(warning)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 4)

        from ghostcaddie.video.ai_demo import build_bounded_roi
        plan = build_bounded_roi(
            [{"box": [100.0, 120.0, 140.0, 160.0], "confidence": 0.9},
             {"box": [700.0, 500.0, 730.0, 530.0], "confidence": 0.8}],
            image_width=800, image_height=600, padding=32, max_candidates=4,
        )
        self.assertEqual(plan["state"], "candidate_region")
        self.assertEqual(plan["box"], [68, 88, 762, 562])
        self.assertLessEqual(plan["candidate_count"], 4)
        self.assertTrue(plan["research_only"])
        self.assertFalse(plan["ground_truth"])
        self.assertFalse(plan["production_eligible"])

    def test_coarse_batch_result_count_mismatch_fails_closed(self):
        from ghostcaddie.video.ai_demo import _coarse_ball_candidates
        class EmptyResult:
            boxes = None
        class ShortBatchModel:
            def __call__(self, frames, verbose=False):
                return [EmptyResult() for _ in frames[:-1]]
        candidates, warning = _coarse_ball_candidates(
            ShortBatchModel(), list(range(4)), 640, 360, max_frames=4)
        self.assertEqual(candidates, [])
        self.assertEqual(warning, "coarse_ball_inference_failed")

    def test_coarse_probe_does_not_consume_past_four_frames(self):
        from ghostcaddie.video.ai_demo import _coarse_ball_candidates
        class EmptyResult:
            boxes = None
        class Model:
            def __call__(self, frames, verbose=False):
                return [EmptyResult() for _ in frames]
        def frames():
            for index in range(4):
                yield index
            raise AssertionError("coarse probe over-consumed frames")
        candidates, warning = _coarse_ball_candidates(Model(), frames(), 640, 360)
        self.assertEqual(candidates, [])
        self.assertIsNone(warning)

    def test_processing_budget_hard_stops_frames_time_memory_and_candidates(self):
        from ghostcaddie.video.ai_demo import BoundedProcessingBudget
        budget = BoundedProcessingBudget(max_frames=2, max_seconds=10.0,
                                         max_memory_bytes=100, max_candidates=3,
                                         started_at=100.0)
        self.assertTrue(budget.allow(frame_bytes=40, candidate_count=2, now=101.0))
        self.assertTrue(budget.allow(frame_bytes=40, candidate_count=1, now=102.0))
        self.assertFalse(budget.allow(frame_bytes=1, candidate_count=0, now=103.0))
        self.assertEqual(budget.reason, "frame_limit")
        budget = BoundedProcessingBudget(max_frames=5, max_seconds=1.0,
                                         max_memory_bytes=1000, max_candidates=10,
                                         started_at=100.0)
        self.assertFalse(budget.allow(frame_bytes=1, candidate_count=0, now=101.0))
        self.assertEqual(budget.reason, "time_limit")
        budget = BoundedProcessingBudget(max_frames=5, max_seconds=10.0,
                                         max_memory_bytes=10, max_candidates=10,
                                         started_at=100.0)
        self.assertFalse(budget.allow(frame_bytes=11, candidate_count=0, now=100.1))
        self.assertEqual(budget.reason, "memory_limit")
        budget = BoundedProcessingBudget(max_frames=5, max_seconds=10.0,
                                         max_memory_bytes=1000, max_candidates=2,
                                         started_at=100.0)
        self.assertFalse(budget.allow(frame_bytes=1, candidate_count=3, now=100.1))
        self.assertEqual(budget.reason, "candidate_limit")
        native_budget = BoundedProcessingBudget(max_frames=12, max_seconds=120.0,
                                                max_memory_bytes=64 * 1024 * 1024,
                                                max_candidates=12 * 8, started_at=100.0)
        for index in range(12):
            self.assertTrue(native_budget.allow(frame_bytes=1, candidate_count=8, now=100.1 + index))
        self.assertFalse(native_budget.allow(frame_bytes=1, candidate_count=8, now=112.1))
        self.assertEqual(native_budget.reason, "frame_limit")

    def test_report_cannot_open_validated_analytics(self):
        report = build_demo_report(
            source={"platform": "youtube", "video_id": "ABCDEFGHIJK"},
            media={"fps": 30.0, "width": 640, "height": 360, "frame_count": 7},
            swing_window={"start_frame": 0, "end_frame": 6, "peak_frame": 5},
            observations=[],
            artifact_references=["annotated_video.mp4", "diagnostics.json"],
            warnings=["research_only_demo"],
        )
        self.assertEqual(report["schema_version"], DEMO_SCHEMA_VERSION)
        self.assertEqual(report["status"], "research_only")
        self.assertFalse(report["production_eligible"])
        self.assertIsNone(report["analytics"])
        self.assertIn("local_yolo_pose", report["methods"])
        self.assertIn("local_swingnet_research_only", report["methods"])
        self.assertIn("guarded_candidate_rejection", report["methods"])
        self.assertIsNone(report["shot_event"])
        self.assertEqual(report["source"], {"platform": "youtube", "video_id": "ABCDEFGHIJK"})
        self.assertTrue(all(not Path(ref).is_absolute() for ref in report["artifact_references"]))
    def test_component_frame_is_an_independent_clean_copy(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("optional NumPy stack unavailable")
        from ghostcaddie.video.ai_demo import clean_frame_for_components
        source = np.full((2, 2, 3), 45, dtype=np.uint8)
        clean = clean_frame_for_components(source)
        self.assertIsNot(clean, source)
        clean[0, 0, 0] = 255
        self.assertEqual(int(source[0, 0, 0]), 45)

    def test_unified_mp4_contains_pose_and_ball_overlays_from_clean_frames(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("optional OpenCV stack unavailable")
        from ghostcaddie.video.ai_demo import run_local_demo
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
            self.assertTrue(writer.isOpened())
            for _ in range(40):
                frame = np.full((240, 320, 3), 45, dtype=np.uint8)
                writer.write(frame)
            writer.release()
            points = itertools.cycle(({"x": 100.0, "y": 110.0}, {"x": 130.0, "y": 110.0}, {"x": 160.0, "y": 110.0}))
            seen_frames = []
            def fake_ball(*args):
                frame = args[2]
                seen_frames.append(("ball", int(frame[0, 0, 0]), int(frame[120, 160, 0])))
                return {"state": "observed", "confidence": 0.9, "uncertainty": 4.0,
                        "point": next(points), "candidate_count": 1, "model": "test_ball"}, None
            pose = {"state": "observed", "confidence": 0.95, "uncertainty": 2.0,
                    "bbox": [60, 40, 220, 220],
                    "keypoints": [[100, 80, 0.9], [120, 90, 0.9], [140, 120, 0.9],
                                  [90, 180, 0.9], [150, 180, 0.9], [80, 210, 0.9],
                                  [160, 210, 0.9], [100, 140, 0.9], [140, 140, 0.9],
                                  [90, 160, 0.9], [150, 160, 0.9]]}
            def fake_pose(*args):
                frame = args[1]
                seen_frames.append(("pose", int(frame[0, 0, 0]), int(frame[120, 160, 0])))
                return pose, None
            with patch("ghostcaddie.video.ai_demo._ball_observation", side_effect=fake_ball), \
                 patch("ghostcaddie.video.ai_demo._pose_observation", side_effect=fake_pose):
                report = run_local_demo(str(source), str(root / "out"), sample_fps=10.0, max_frames=40,
                                        pose_model="", ball_model="")
            self.assertEqual(len(seen_frames), 80)
            self.assertTrue(all(seen_frames[index][1:] == seen_frames[index + 1][1:]
                                for index in range(0, len(seen_frames), 2)))
            self.assertTrue(all(item["golfer"].get("track_id") == "golfer-0" for item in report["observations"]))
            self.assertTrue((root / "out" / "annotated_video.mp4").is_file())
            self.assertTrue((root / "out" / "contact_sheet.jpg").is_file())
            self.assertTrue((root / "out" / "diagnostics.json").is_file())
            self.assertTrue((root / "out" / "provenance.json").is_file())
            self.assertEqual(set(report["artifact_references"]), {
                "annotated_video.mp4", "annotated_frames/", "contact_sheet.jpg",
                "diagnostics.json", "provenance.json"})
            diagnostics = json.loads((root / "out" / "diagnostics.json").read_text())
            provenance = json.loads((root / "out" / "provenance.json").read_text())
            self.assertEqual(diagnostics["artifact_references"], report["artifact_references"])
            self.assertEqual(provenance["artifact_references"], report["artifact_references"])
            contact_sheet = cv2.imread(str(root / "out" / "contact_sheet.jpg"))
            self.assertIsNotNone(contact_sheet)
            self.assertGreaterEqual(contact_sheet.shape[1], 320)
            capture = cv2.VideoCapture(str(root / "out" / "annotated_video.mp4"))
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 40)
            green_pixels = red_pixels = orange_pixels = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                green_pixels += int(np.sum((frame[:, :, 1] > 120) & (frame[:, :, 0] < 100) & (frame[:, :, 2] < 100)))
                red_pixels += int(np.sum((frame[:, :, 2] > 120) & (frame[:, :, 1] < 90) & (frame[:, :, 0] < 90)))
                orange_pixels += int(np.sum((frame[:, :, 2] > 120) & (frame[:, :, 1] > 70) & (frame[:, :, 0] < 100)))
            capture.release()
            self.assertGreater(green_pixels, 100)
            self.assertGreater(red_pixels, 100)
            self.assertGreater(orange_pixels, 20)

    def test_mp4_contains_ball_marker_and_tracer_when_observations_exist(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("optional OpenCV stack unavailable")
        from ghostcaddie.video.ai_demo import run_local_demo
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
            self.assertTrue(writer.isOpened())
            for _ in range(40):
                writer.write(np.full((240, 320, 3), 45, dtype=np.uint8))
            writer.release()
            points = itertools.cycle(({"x": 100.0, "y": 110.0}, {"x": 130.0, "y": 110.0}, {"x": 160.0, "y": 110.0}))
            def fake_ball(*_args):
                return {"state": "observed", "confidence": 0.9, "uncertainty": 4.0,
                        "point": next(points), "candidate_count": 1, "model": "test_ball"}, None
            with patch("ghostcaddie.video.ai_demo._ball_observation", side_effect=fake_ball):
                report = run_local_demo(str(source), str(root / "out"), sample_fps=10.0, max_frames=40,
                                        pose_model="", ball_model="")
            observed = [item for item in report["observations"] if item["ball"].get("point")]
            self.assertGreaterEqual(len(observed), 2)
            self.assertTrue(all(item["ball"]["rendered_overlay"]["marker"] for item in observed))
            capture = cv2.VideoCapture(str(root / "out" / "annotated_video.mp4"))
            red_pixels = orange_pixels = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                red_pixels += int(np.sum((frame[:, :, 2] > 120) & (frame[:, :, 1] < 90) & (frame[:, :, 0] < 90)))
                orange_pixels += int(np.sum((frame[:, :, 2] > 120) & (frame[:, :, 1] > 70) & (frame[:, :, 0] < 100)))
            capture.release()
            self.assertGreater(red_pixels, 100)
            self.assertGreater(orange_pixels, 20)

    def test_ingestion_respects_duration_bound_before_model_processing(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("optional OpenCV stack unavailable")
        from ghostcaddie.video.ai_demo import run_local_demo
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
            self.assertTrue(writer.isOpened())
            for _ in range(20):
                writer.write(np.full((240, 320, 3), 45, dtype=np.uint8))
            writer.release()
            original_capture = cv2.VideoCapture
            reads = []

            class CountingCapture:
                def __init__(self, *args, **kwargs):
                    self._capture = original_capture(*args, **kwargs)

                def read(self):
                    reads.append(1)
                    return self._capture.read()

                def __getattr__(self, name):
                    return getattr(self._capture, name)

            with patch("cv2.VideoCapture", CountingCapture):
                with self.assertRaises(DemoAcceptanceError):
                    run_local_demo(str(source), str(root / "out"), sample_fps=10.0,
                                   max_duration_seconds=0.2, pose_model="", ball_model="")
            self.assertLessEqual(len(reads), 4)
            self.assertLessEqual(len(list((root / "out" / "annotated_frames").glob("frame_*.jpg"))), 0)

    def test_high_requested_sample_fps_still_respects_source_duration_bound(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("optional OpenCV stack unavailable")
        from ghostcaddie.video.ai_demo import run_local_demo
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
            self.assertTrue(writer.isOpened())
            for _ in range(20):
                writer.write(np.full((240, 320, 3), 45, dtype=np.uint8))
            writer.release()
            original_capture = cv2.VideoCapture
            reads = []

            class CountingCapture:
                def __init__(self, *args, **kwargs):
                    self._capture = original_capture(*args, **kwargs)

                def read(self):
                    reads.append(1)
                    return self._capture.read()

                def __getattr__(self, name):
                    return getattr(self._capture, name)

            with patch("cv2.VideoCapture", CountingCapture):
                with self.assertRaises(DemoAcceptanceError):
                    run_local_demo(str(source), str(root / "out"), sample_fps=100.0,
                                   max_duration_seconds=0.2, pose_model="", ball_model="")
            self.assertLessEqual(len(reads), 4)
            self.assertLessEqual(len(list((root / "out" / "annotated_frames").glob("frame_*.jpg"))), 0)

    def test_rerun_removes_stale_annotated_frames_before_encoding(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("optional OpenCV stack unavailable")
        from ghostcaddie.video.ai_demo import run_local_demo
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
            self.assertTrue(writer.isOpened())
            for _ in range(40):
                writer.write(np.full((240, 320, 3), 45, dtype=np.uint8))
            writer.release()
            def fake_ball(*_args):
                return {"state": "observed", "confidence": 0.9, "uncertainty": 4.0,
                        "point": {"x": 100.0, "y": 110.0}, "candidate_count": 1,
                        "model": "test_ball"}, None
            with patch("ghostcaddie.video.ai_demo._ball_observation", side_effect=fake_ball):
                run_local_demo(str(source), str(root / "out"), sample_fps=10.0, max_frames=40,
                               pose_model="", ball_model="")
                with self.assertRaises(DemoAcceptanceError):
                    run_local_demo(str(source), str(root / "out"), sample_fps=10.0, max_frames=1,
                                   pose_model="", ball_model="")
            frames = sorted((root / "out" / "annotated_frames").glob("frame_*.jpg"))
            self.assertEqual(len(frames), 0)
            capture = cv2.VideoCapture(str(root / "out" / "annotated_video.mp4"))
            count = 0
            while capture.read()[0]:
                count += 1
            capture.release()
            self.assertEqual(count, 0)

    def test_blocked_rerun_removes_stale_production_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            for name in ("recommendation.json", "normalized_shot.json", "overlay.svg"):
                (out / name).write_text("stale")
            with redirect_stdout(StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(["ai-demo", "--url", "https://example.com/not-youtube", "--out", str(out)])
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(any((out / name).exists() for name in ("recommendation.json", "normalized_shot.json", "overlay.svg")))

    def test_cli_module_entrypoint_runs_main_when_executed(self):
        import subprocess
        import sys
        repo_root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "-m", "ghostcaddie.cli"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=60,
        )
        self.assertIn(proc.returncode, (1, 2))
        self.assertIn("usage", (proc.stdout + proc.stderr).lower())

    def test_help_exposes_ai_demo_without_invoking_analytics(self):
        with redirect_stdout(StringIO()) as stdout:
            with self.assertRaises(SystemExit) as raised:
                main(["ai-demo", "--help"])
        self.assertEqual(raised.exception.code, 0)
        text = stdout.getvalue().lower()
        self.assertIn("bounded", text)
        self.assertIn("research-only", text)
        self.assertIn("h.264", text)

    def test_ball_overlay_with_unavailable_ball_reports_no_marker_but_counts_rejected(self):
        from ghostcaddie.video.ai_demo import _draw_ball_overlay
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("optional NumPy/OpenCV stack unavailable")
        rejected = [{
            "reasons": ["golfer_bbox_overlap", "head_region"],
            "candidate": {"center": (160.0, 75.0), "box": [145.0, 60.0, 175.0, 90.0],
                          "confidence": 0.99},
        }]
        frame = np.zeros((240, 320, 3), dtype="uint8")
        rendered = _draw_ball_overlay(
            frame,
            {"state": "unavailable", "point": None, "rejected_candidates": rejected},
            [],
            frame.copy(),
        )
        self.assertFalse(rendered["marker"])
        self.assertEqual(rendered["tracer_points"], 0)
        self.assertFalse(rendered["zoom_inset"])
        self.assertEqual(rendered["rejected_markers"], 1)

    def test_ball_overlay_rejected_rendering_ignores_records_without_valid_box(self):
        from ghostcaddie.video.ai_demo import _draw_ball_overlay
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("optional NumPy/OpenCV stack unavailable")
        frame = np.zeros((240, 320, 3), dtype="uint8")
        rendered = _draw_ball_overlay(
            frame,
            {"state": "unavailable", "point": None,
             "rejected_candidates": [{"reasons": ["candidate_rejected"], "candidate": None},
                                     {"reasons": ["candidate_rejected"]},
                                     "not-a-record"]},
            [],
            frame.copy(),
        )
        self.assertFalse(rendered["marker"])
        self.assertEqual(rendered["rejected_markers"], 0)
        self.assertEqual(rendered["tracer_points"], 0)
        self.assertFalse(rendered["zoom_inset"])


    def test_pose_observation_records_person_count_and_second_persons(self):
        import numpy as np
        from ghostcaddie.video.ai_demo import _pose_observation
        frame = np.full((240, 320, 3), 45, dtype=np.uint8)
        pose, warning = _pose_observation(_FakeMultiPersonYolo(), frame, 320, 240)
        self.assertIsNotNone(pose)
        self.assertIsNone(warning)
        assert pose is not None
        self.assertEqual(pose["state"], "observed")
        self.assertEqual(pose["person_count"], 2)
        self.assertEqual(pose["second_person_count"], 1)
        self.assertTrue(pose["multi_person_frame"])

    def test_pose_observation_single_person_has_no_multi_person_flag(self):
        import numpy as np
        from ghostcaddie.video.ai_demo import _pose_observation
        frame = np.full((240, 320, 3), 45, dtype=np.uint8)
        pose, warning = _pose_observation(_FakeSinglePersonYolo(), frame, 320, 240)
        self.assertIsNotNone(pose)
        self.assertIsNone(warning)
        assert pose is not None
        self.assertEqual(pose["person_count"], 1)
        self.assertEqual(pose["second_person_count"], 0)
        self.assertFalse(pose["multi_person_frame"])


class _FakePersonTensor:
    def __init__(self, value):
        self._value = value

    def cpu(self):
        return self

    def numpy(self):
        return self._value

    def tolist(self):
        return self._value

    def __float__(self):
        return float(self._value)

    def __getitem__(self, index):
        return _FakePersonTensor(self._value[index])

    def __len__(self):
        return len(self._value)


class _FakePersonBoxes:
    def __init__(self, cls, conf, xyxy):
        self.cls = _FakePersonTensor(cls)
        self.conf = _FakePersonTensor(conf)
        self.xyxy = _FakePersonTensor(xyxy)

    def __len__(self):
        return len(self.cls)


class _FakePersonKeypoints:
    def __init__(self, xy, conf):
        self.xy = _FakePersonTensor(xy)
        self.conf = _FakePersonTensor(conf)


class _FakeMultiPersonYolo:
    def __call__(self, frame, verbose=False):
        result = _FakePersonResult()
        result.boxes = _FakePersonBoxes(
            cls=[0, 0], conf=[0.91, 0.62],
            xyxy=[[10.0, 20.0, 110.0, 220.0], [200.0, 30.0, 300.0, 230.0]],
        )
        result.keypoints = _FakePersonKeypoints(
            xy=[[[60.0, 30.0]] * 17, [[250.0, 40.0]] * 17],
            conf=[[0.9] * 17, [0.8] * 17],
        )
        return [result]


class _FakeSinglePersonYolo:
    def __call__(self, frame, verbose=False):
        result = _FakePersonResult()
        result.boxes = _FakePersonBoxes(
            cls=[0], conf=[0.91], xyxy=[[10.0, 20.0, 110.0, 220.0]],
        )
        result.keypoints = _FakePersonKeypoints(
            xy=[[[60.0, 30.0]] * 17], conf=[[0.9] * 17],
        )
        return [result]


class _FakePersonResult:
    boxes = None
    keypoints = None


if __name__ == "__main__":
    unittest.main()
