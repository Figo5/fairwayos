import json
import tempfile
import unittest
from pathlib import Path

from ghostcaddie.video.errors import VideoContractError, VideoPathError
from ghostcaddie.video.observations import (
    OBSERVATIONS_SCHEMA_VERSION,
    CANONICAL_PHASES,
    VideoObservations,
    load_fixture_observations,
)
from ghostcaddie.video.paths import ProjectBoundary
from ghostcaddie.video.perception import FixturePerception


VALID = {
    "schema_version": "video-observations.v1",
    "image": {"width": 1920, "height": 1080},
    "observations": [{
        "frame_index": 0, "timestamp_seconds": 0.0,
        "golfer": {"bbox": {"x": 100, "y": 100, "width": 200, "height": 400},
                   "anchor": {"x": 200, "y": 500}, "confidence": 0.97},
        "club": None, "clubhead": None, "ball": None,
        "phase": "address", "contact": None, "intended_direction": None,
        "landing": None, "warnings": ["ball_missing"],
    }, {
        "frame_index": 1, "timestamp_seconds": 0.033,
        "golfer": {"bbox": {"x": 100, "y": 100, "width": 200, "height": 400},
                   "anchor": {"x": 200, "y": 500}, "confidence": 0.98},
        "club": {"name": "7i", "confidence": 0.8},
        "clubhead": {"x": 230, "y": 480, "confidence": 0.7},
        "ball": {"x": 250, "y": 520, "confidence": 0.9},
        "phase": "contact", "contact": {"x": 250, "y": 520, "confidence": 0.7, "method": "fixture"},
        "intended_direction": {"x": 1, "y": 0, "confidence": 0.6},
        "landing": {"x": 900, "y": 500, "confidence": 0.4, "method": "fixture"},
        "warnings": ["low_confidence"],
    }]
}


class TestVideoObservations(unittest.TestCase):
    def test_valid_versioned_contract_round_trips_with_explicit_unknowns(self):
        observations = VideoObservations.from_dict(VALID)
        self.assertEqual(observations.schema_version, OBSERVATIONS_SCHEMA_VERSION)
        self.assertIsNone(observations.items[0].ball)
        self.assertEqual(observations.to_dict(), VALID)

    def test_canonical_phase_enum_and_unambiguous_aliases_are_normalized(self):
        expected = {"unknown", "address", "backswing", "top", "downswing", "contact", "follow_through", "ball_flight", "landing", "rolling", "finish"}
        self.assertEqual(CANONICAL_PHASES, expected)
        aliases = {
            "setup": "address", "setup/address": "address", "impact": "contact", "contact": "contact",
            "follow-through": "follow_through", "follow through": "follow_through", "flight": "ball_flight",
            "ball flight": "ball_flight",
        }
        for alias, canonical in aliases.items():
            payload = json.loads(json.dumps(VALID))
            payload["observations"][0]["phase"] = alias
            parsed = VideoObservations.from_dict(payload)
            self.assertEqual(parsed.items[0].phase, canonical)

    def test_rejects_unknown_or_ambiguous_phase_aliases(self):
        for phase in ("swing", "motion", "setup-ish", "maybe_impact", "followup"):
            payload = json.loads(json.dumps(VALID))
            payload["observations"][0]["phase"] = phase
            with self.subTest(phase=phase), self.assertRaises(VideoContractError):
                VideoObservations.from_dict(payload)

        cases = [
            {"frame_index": -1},
            {"timestamp_seconds": -1},
            {"golfer": {"bbox": {"x": 0, "y": 0, "width": 2, "height": 2}, "anchor": {"x": 0, "y": 0}, "confidence": 2}},
            {"phase": "flying"},
            {"ball": {"x": 1, "y": 1, "confidence": 0.5}, "image": {"width": 2, "height": 2}},
        ]
        for override in cases:
            payload = json.loads(json.dumps(VALID))
            payload["observations"][0].update(override)
            with self.subTest(override=override), self.assertRaises(VideoContractError):
                VideoObservations.from_dict(payload)
        payload = json.loads(json.dumps(VALID))
        payload["observations"][0]["unknown"] = 1
        with self.assertRaises(VideoContractError):
            VideoObservations.from_dict(payload)

    def test_rejects_non_monotonic_or_duplicate_frames_and_timestamps(self):
        for second in (0.0, -0.1):
            payload = json.loads(json.dumps(VALID))
            payload["observations"][1]["timestamp_seconds"] = second
            with self.assertRaises(VideoContractError):
                VideoObservations.from_dict(payload)
        payload = json.loads(json.dumps(VALID))
        payload["observations"][1]["frame_index"] = 0
        with self.assertRaises(VideoContractError):
            VideoObservations.from_dict(payload)

    def test_fixture_loader_is_project_bound_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            root = Path(tmp)
            (root / "observations.json").write_text(json.dumps(VALID))
            (Path(outside) / "x.json").write_text(json.dumps(VALID))
            (root / "escape.json").symlink_to(Path(outside) / "x.json")
            boundary = ProjectBoundary(root)
            first = load_fixture_observations("observations.json", boundary)
            second = FixturePerception(boundary, "observations.json").perceive()
            self.assertEqual(first.to_dict(), second.to_dict())
            with self.assertRaises(VideoPathError):
                load_fixture_observations("escape.json", boundary)
            with self.assertRaises(VideoPathError):
                load_fixture_observations(str(root / "observations.json"), boundary)


if __name__ == "__main__":
    unittest.main()
