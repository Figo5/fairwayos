import json
import tempfile
import unittest
from pathlib import Path

from ghostcaddie.video.team_layer_integration import (
    LayerContractError,
    load_completed_layers,
    render_timeline_states,
)

SOURCE_SHA = "cefbdf25400f5821893747e2b4a60ca5a11f990920ab3a9c8bba32a8c2d3deae"


def write_layer(root: Path, name: str, payload: dict, ready: bool = True) -> None:
    layer_dir = root / name
    layer_dir.mkdir(parents=True)
    (layer_dir / "layer.json").write_text(json.dumps(payload), encoding="utf-8")
    if ready:
        (layer_dir / "READY.json").write_text(json.dumps({"ready": True}), encoding="utf-8")


def valid_layer(name: str, observations=None, source_sha: str = SOURCE_SHA) -> dict:
    return {
        "source_sha256": source_sha,
        "layer": name,
        "pseudo_label": True,
        "ground_truth": False,
        "research_only": True,
        "production_eligible": False,
        "observations": observations if observations is not None else [
            {
                "frame_index": 40,
                "state": "visible",
                "geometry": {"type": "point", "x": 486.0, "y": 578.0},
                "evidence_method": "clean_native_frame_manual_review",
                "confidence": 0.7,
                "manual_assistance": True,
                "evidence_paths": ["/tmp/evidence/f0040.jpg"],
            }
        ],
    }


def unavailable_layer(name: str) -> dict:
    return valid_layer(name, observations=[{
        "frame_index": 60,
        "state": "unresolvable",
        "geometry": None,
        "evidence_method": "clean native crop review; unresolved identity",
        "confidence": None,
        "manual_assistance": "manual_visual_review",
        "evidence_paths": ["evidence/address_crop_sheet.jpg"],
    }])


class TeamLayerIntegrationContractTests(unittest.TestCase):
    def test_missing_completed_layer_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_layer(Path(tmp), "ball", valid_layer("ball"))
            with self.assertRaisesRegex(LayerContractError, "missing completed layer"):
                load_completed_layers(Path(tmp), expected_layers=("body", "clubhead", "ball"), source_sha256=SOURCE_SHA)

    def test_mismatched_source_sha_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_layer(Path(tmp), "body", valid_layer("body", source_sha="bad"))
            write_layer(Path(tmp), "clubhead", valid_layer("clubhead"))
            write_layer(Path(tmp), "ball", valid_layer("ball"))
            with self.assertRaisesRegex(LayerContractError, "source_sha256"):
                load_completed_layers(Path(tmp), expected_layers=("body", "clubhead", "ball"), source_sha256=SOURCE_SHA)

    def test_malformed_geometry_and_flags_are_rejected(self):
        bad = valid_layer("ball", observations=[{
            "frame_index": 41,
            "state": "visible",
            "geometry": {"type": "point", "x": float("nan"), "y": 10},
            "evidence_method": "bad",
            "confidence": 0.5,
            "manual_assistance": False,
            "evidence_paths": [],
        }])
        bad["production_eligible"] = True
        with tempfile.TemporaryDirectory() as tmp:
            write_layer(Path(tmp), "ball", bad)
            with self.assertRaises(LayerContractError):
                load_completed_layers(Path(tmp), expected_layers=("ball",), source_sha256=SOURCE_SHA)

    def test_blocked_unavailable_only_layer_is_accepted_but_not_observed(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_layer(Path(tmp), "clubhead", unavailable_layer("clubhead"), ready=False)
            (Path(tmp) / "clubhead" / "READY.json").write_text(
                json.dumps({"readiness": "blocked", "ready_written_last": True}), encoding="utf-8")
            layers = load_completed_layers(Path(tmp), expected_layers=("clubhead",), source_sha256=SOURCE_SHA)
            self.assertEqual(layers["clubhead"]["readiness"], "blocked")
            self.assertEqual(layers["clubhead"]["observed_count"], 0)
            self.assertFalse(layers["clubhead"]["observations"][0]["visible"])

    def test_blocked_layer_with_observed_geometry_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_layer(Path(tmp), "clubhead", valid_layer("clubhead"), ready=False)
            (Path(tmp) / "clubhead" / "READY.json").write_text(
                json.dumps({"readiness": "blocked", "ready_written_last": True}), encoding="utf-8")
            with self.assertRaisesRegex(LayerContractError, "blocked layer"):
                load_completed_layers(Path(tmp), expected_layers=("clubhead",), source_sha256=SOURCE_SHA)

    def test_body_box_keypoints_geometry_is_adapted_to_box_with_keypoints(self):
        obs = {
            "frame_index": 1,
            "state": "observed",
            "geometry": {
                "type": "box_keypoints",
                "box": {"x1": 10, "y1": 20, "x2": 30, "y2": 60},
                "keypoints": [{"name": "left_ankle", "x": 15, "y": 55, "confidence": 0.8}],
            },
            "evidence_method": "pose review",
            "confidence": 0.9,
            "manual_assistance": "reviewed",
            "evidence_paths": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            write_layer(Path(tmp), "body", valid_layer("body", observations=[obs]))
            layers = load_completed_layers(Path(tmp), expected_layers=("body",), source_sha256=SOURCE_SHA, frame_size=(1280, 720))
            self.assertTrue(layers["body"]["observations"][0]["visible"])
            self.assertEqual(layers["body"]["observations"][0]["geometry"]["type"], "box_keypoints")

    def test_layer_name_alias_is_accepted_when_layer_key_is_absent(self):
        payload = valid_layer("body")
        payload.pop("layer")
        payload["layer_name"] = "fairwayos_pga_body_yolo11n_pose_reviewed"
        with tempfile.TemporaryDirectory() as tmp:
            write_layer(Path(tmp), "body", payload)
            layers = load_completed_layers(Path(tmp), expected_layers=("body",), source_sha256=SOURCE_SHA)
            self.assertEqual(layers["body"]["layer"], "body")

    def test_cut_resets_each_layer_history_without_coupling_visibility(self):
        observations = {
            "body": [
                {"frame_index": 214, "state": "visible", "geometry": {"type": "box", "x1": 1, "y1": 2, "x2": 20, "y2": 40}, "evidence_method": "pose", "confidence": 0.8, "manual_assistance": False, "evidence_paths": []},
                {"frame_index": 215, "state": "visible", "geometry": {"type": "box", "x1": 1, "y1": 2, "x2": 20, "y2": 40}, "evidence_method": "pose", "confidence": 0.8, "manual_assistance": False, "evidence_paths": []},
                {"frame_index": 217, "state": "visible", "geometry": {"type": "box", "x1": 2, "y1": 2, "x2": 21, "y2": 40}, "evidence_method": "pose", "confidence": 0.8, "manual_assistance": False, "evidence_paths": []},
            ],
            "ball": [
                {"frame_index": 214, "state": "visible", "geometry": {"type": "point", "x": 10, "y": 10}, "evidence_method": "manual", "confidence": 0.7, "manual_assistance": True, "evidence_paths": []},
                {"frame_index": 215, "state": "unavailable", "geometry": None, "evidence_method": "clean_source", "confidence": None, "manual_assistance": True, "evidence_paths": []},
                {"frame_index": 217, "state": "visible", "geometry": {"type": "point", "x": 30, "y": 10}, "evidence_method": "manual", "confidence": 0.7, "manual_assistance": True, "evidence_paths": []},
            ],
        }
        states = render_timeline_states(observations, frame_range=range(214, 218), cuts=[216])
        self.assertEqual(states[214]["body"]["history_len"], 1)
        self.assertEqual(states[214]["ball"]["history_len"], 1)
        self.assertEqual(states[215]["body"]["history_len"], 2)
        self.assertEqual(states[215]["ball"]["history_len"], 0)
        self.assertFalse(states[216]["body"]["visible"])
        self.assertEqual(states[216]["body"]["state"], "cut")
        self.assertEqual(states[216]["ball"]["state"], "cut")
        self.assertEqual(states[217]["body"]["history_len"], 1)
        self.assertEqual(states[217]["ball"]["history_len"], 1)


if __name__ == "__main__":
    unittest.main()
