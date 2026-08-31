import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from ghostcaddie.cli import main
from ghostcaddie.video.ai_demo import (
    DEMO_SCHEMA_VERSION,
    ObservationState,
    build_demo_observation,
    build_demo_provenance,
    build_demo_report,
    reject_obvious_false_positive,
    select_swing_window,
)


class TestAIDemoContracts(unittest.TestCase):
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
        self.assertIn("guarded_candidate_rejection", report["methods"])
        self.assertIsNone(report["shot_event"])
        self.assertEqual(report["source"], {"platform": "youtube", "video_id": "ABCDEFGHIJK"})
        self.assertTrue(all(not Path(ref).is_absolute() for ref in report["artifact_references"]))


class TestAIDemoCLIExposure(unittest.TestCase):
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
