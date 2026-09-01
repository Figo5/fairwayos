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
    ObservationState,
    build_demo_observation,
    build_demo_provenance,
    build_demo_report,
    _normalize_tracker_state,
    reject_obvious_false_positive,
    select_swing_window,
)


class TestAIDemoContracts(unittest.TestCase):
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

    def test_obvious_false_positive_is_rejected(self):
        candidate = {"point": [4.0, 5.0], "confidence": 0.88, "inside_golfer": False, "temporal_support": 1}
        decision = reject_obvious_false_positive(candidate, image_width=100, image_height=100)
        self.assertFalse(decision["accepted"])
        self.assertIn("insufficient_temporal_support", decision["reasons"])
        self.assertFalse(decision["production_eligible"])

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
            for _ in range(3):
                frame = np.full((240, 320, 3), 45, dtype=np.uint8)
                writer.write(frame)
            writer.release()
            points = iter(({"x": 100.0, "y": 110.0}, {"x": 130.0, "y": 110.0}, {"x": 160.0, "y": 110.0}))
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
                report = run_local_demo(str(source), str(root / "out"), sample_fps=10.0, max_frames=3,
                                        pose_model="", ball_model="")
            self.assertEqual(len(seen_frames), 6)
            self.assertTrue(all(seen_frames[index][1:] == seen_frames[index + 1][1:]
                                for index in range(0, len(seen_frames), 2)))
            self.assertTrue(all(item["golfer"].get("track_id") == "golfer-0" for item in report["observations"]))
            self.assertTrue(all(item["pose"].get("skeleton_rendered") for item in report["observations"]))
            capture = cv2.VideoCapture(str(root / "out" / "annotated_video.mp4"))
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
            for _ in range(3):
                writer.write(np.full((240, 320, 3), 45, dtype=np.uint8))
            writer.release()
            points = iter(({"x": 100.0, "y": 110.0}, {"x": 130.0, "y": 110.0}, {"x": 160.0, "y": 110.0}))
            def fake_ball(*_args):
                return {"state": "observed", "confidence": 0.9, "uncertainty": 4.0,
                        "point": next(points), "candidate_count": 1, "model": "test_ball"}, None
            with patch("ghostcaddie.video.ai_demo._ball_observation", side_effect=fake_ball):
                report = run_local_demo(str(source), str(root / "out"), sample_fps=10.0, max_frames=3,
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
                report = run_local_demo(str(source), str(root / "out"), sample_fps=10.0,
                                        max_duration_seconds=0.2, pose_model="", ball_model="")
            self.assertLessEqual(len(reads), 4)
            self.assertLessEqual(len(report["observations"]), 3)

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
                report = run_local_demo(str(source), str(root / "out"), sample_fps=100.0,
                                        max_duration_seconds=0.2, pose_model="", ball_model="")
            self.assertLessEqual(len(reads), 4)
            self.assertLessEqual(len(report["observations"]), 3)

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
            for _ in range(3):
                writer.write(np.full((240, 320, 3), 45, dtype=np.uint8))
            writer.release()
            def fake_ball(*_args):
                return {"state": "observed", "confidence": 0.9, "uncertainty": 4.0,
                        "point": {"x": 100.0, "y": 110.0}, "candidate_count": 1,
                        "model": "test_ball"}, None
            with patch("ghostcaddie.video.ai_demo._ball_observation", side_effect=fake_ball):
                run_local_demo(str(source), str(root / "out"), sample_fps=10.0, max_frames=3,
                               pose_model="", ball_model="")
                run_local_demo(str(source), str(root / "out"), sample_fps=10.0, max_frames=1,
                               pose_model="", ball_model="")
            frames = sorted((root / "out" / "annotated_frames").glob("frame_*.jpg"))
            self.assertEqual(len(frames), 1)
            capture = cv2.VideoCapture(str(root / "out" / "annotated_video.mp4"))
            count = 0
            while capture.read()[0]:
                count += 1
            capture.release()
            self.assertEqual(count, 1)

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

    def test_help_exposes_ai_demo_without_invoking_analytics(self):
        with redirect_stdout(StringIO()) as stdout:
            with self.assertRaises(SystemExit) as raised:
                main(["ai-demo", "--help"])
        self.assertEqual(raised.exception.code, 0)
        text = stdout.getvalue().lower()
        self.assertIn("bounded", text)
        self.assertIn("research-only", text)
        self.assertIn("h.264", text)


if __name__ == "__main__":
    unittest.main()
