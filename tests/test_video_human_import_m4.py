import copy
import json
import tempfile
import unittest
from pathlib import Path

from ghostcaddie.cli import main
from ghostcaddie.video.errors import VideoContractError
from ghostcaddie.video.human_import import import_human_annotations
from ghostcaddie.video.reconstruction import ReconstructionResult
from tests.test_video_human_contracts import TestHumanAnnotationContract


class TestHumanImportM4(unittest.TestCase):
    def setUp(self):
        source = TestHumanAnnotationContract()
        self.payload = source.valid_payload()
        self.payload.update(status="submitted", explicit_submit=True)
        self.payload["landing"] = {"value": {"x": 1500.0, "y": 500.0, "frame_index": 40,
                                              "timestamp_seconds": 1.345, "confidence": 0.9},
                                   "source": "observed"}
        self.payload["contact"] = {"value": {"x": 900.0, "y": 600.0, "frame_index": 20, "timestamp_seconds": 0.672,
                                              "confidence": 0.9, "phase": "contact"},
                                   "source": "observed"}
        self.payload["context"] = {"value": {"lie": "fairway"}, "source": "user_supplied"}

    def test_contact_coordinate_is_preserved_without_anchor_fabrication(self):
        from ghostcaddie.video.human_import import _observations
        observations = _observations(__import__('ghostcaddie.video.human_contracts', fromlist=['HumanAnnotationDocument']).HumanAnnotationDocument.from_dict(self.payload))
        self.assertEqual(observations.items[1].contact["x"], 900.0)
        self.assertEqual(observations.items[1].contact["y"], 600.0)

    def test_import_reconstructs_one_shot_and_maps_once(self):
        class SpyCalibration:
            width, height = 1920, 1080
            def __init__(self): self.calls = []
            def to_engine(self, point):
                self.calls.append(point)
                return type(point)(point.x / 10, point.y / 10)

        calibration = SpyCalibration()
        result = import_human_annotations(self.payload, calibration, event_id="E1",
                                           player_id="P1", tournament_id="T1", hole_number=7,
                                           shot_number=2, distance_to_pin=150,
                                           wind={"speed_mph": 8, "direction_deg": 90},
                                           timestamp="2026-08-27T12:00:00Z",
                                           target_pixel={"x": 700, "y": 500})
        self.assertIsInstance(result, ReconstructionResult)
        self.assertEqual(result.shot_event.club, "7-iron")
        self.assertEqual(result.shot_event.start_position.x, 50)
        self.assertEqual(result.shot_event.actual_landing_position.x, 150)
        self.assertEqual(len(calibration.calls), 3)
        self.assertEqual(result.metadata["source"], "video-human-annotations.v1")
        self.assertEqual(result.metadata["contact_frame_index"], 20)
        self.assertEqual(result.metadata["landing_timestamp_seconds"], 1.345)

    def test_import_rejects_inferred_contact_at_human_submit_boundary(self):
        inferred = copy.deepcopy(self.payload)
        inferred["contact"]["source"] = "inferred"
        class Calibration:
            width, height = 1920, 1080
            def to_engine(self, point):
                return point
        with self.assertRaises(VideoContractError):
            import_human_annotations(inferred, Calibration(), event_id="E1", player_id="P1", tournament_id="T1",
                                      hole_number=1, shot_number=1, distance_to_pin=1,
                                      wind={"speed_mph": 0, "direction_deg": 0}, timestamp="t",
                                      target_pixel={"x": 1, "y": 1})

    def test_import_rejects_inferred_landing_at_human_submit_boundary(self):
        inferred = copy.deepcopy(self.payload)
        inferred["landing"]["source"] = "inferred"
        class Calibration:
            width, height = 1920, 1080
            def to_engine(self, point):
                return point
        with self.assertRaises(VideoContractError):
            import_human_annotations(inferred, Calibration(), event_id="E1", player_id="P1", tournament_id="T1",
                                      hole_number=1, shot_number=1, distance_to_pin=1,
                                      wind={"speed_mph": 0, "direction_deg": 0}, timestamp="t",
                                      target_pixel={"x": 1, "y": 1})

    def test_import_rejects_unsubmitted_and_missing_evidence(self):
        with self.assertRaises(VideoContractError):
            import_human_annotations(dict(self.payload, status="draft", explicit_submit=False), None,
                                      event_id="E1", player_id="P1", tournament_id="T1",
                                      hole_number=1, shot_number=1, distance_to_pin=1,
                                      wind={"speed_mph": 0, "direction_deg": 0}, timestamp="t",
                                      target_pixel={"x": 1, "y": 1})
        missing = copy.deepcopy(self.payload)
        missing["landing"] = {"value": None, "source": "unavailable"}
        with self.assertRaises(VideoContractError):
            import_human_annotations(missing, None, event_id="E1", player_id="P1", tournament_id="T1",
                                      hole_number=1, shot_number=1, distance_to_pin=1,
                                      wind={"speed_mph": 0, "direction_deg": 0}, timestamp="t",
                                      target_pixel={"x": 1, "y": 1})

    def test_cli_import_writes_normalized_shot_without_absolute_video_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "annotations.json").write_text(json.dumps(self.payload))
            (root / "calibration.json").write_text(json.dumps({
                "image_width": 1920, "image_height": 1080, "source_units": "pixels",
                "engine_units": "yards", "source_points": [{"x": 100, "y": 200}, {"x": 1700, "y": 200},
                {"x": 1700, "y": 900}, {"x": 100, "y": 900}],
                "engine_points": [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 100, "y": 100}, {"x": 0, "y": 100}]}))
            (root / "course.json").write_text(json.dumps({"name": "x", "par": 4,
                "coordinate_system": {"mode": "manual", "units": "yards", "origin": {"x": 0, "y": 0}, "x_axis": {"x": 1, "y": 0}, "y_axis": {"x": 0, "y": 1}},
                "fairway": [[{"x":0,"y":0},{"x":1,"y":0},{"x":1,"y":1}]], "green": [[{"x":0,"y":0},{"x":1,"y":0},{"x":1,"y":1}]],
                "pin_position": {"x":70,"y":50}}))
            (root / "player.json").write_text(json.dumps({"player_id": "P1", "clubs": {
                "7-iron": {"carry_mean_yd": 150, "carry_stddev_yd": 10, "lateral_stddev_yd": 5}}}))
            out = root / "out"
            main(["video-import", "--annotations", "annotations.json", "--calibration", "calibration.json",
                  "--course", "course.json", "--player", "player.json", "--project-root", str(root),
                  "--out", str(out), "--event-id", "E1", "--target-x", "700", "--target-y", "500"])
            data = json.loads((out / "normalized_shot.json").read_text())
            self.assertEqual(data["metadata"]["source"], "video-human-annotations.v1")
            self.assertNotIn(str(root), (out / "normalized_shot.json").read_text())


if __name__ == "__main__":
    unittest.main()
