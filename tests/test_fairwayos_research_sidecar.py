import json
import tempfile
import unittest
from pathlib import Path

from ghostcaddie.cli import main
from ghostcaddie.video.research_ball import BallTrackItem, BallTrackResult
from ghostcaddie.video.fairwayos_research import (
    build_fairwayos_sidecar,
    sidecar_from_mapping,
    write_fairwayos_sidecar,
)


class FairwayOSResearchSidecarTests(unittest.TestCase):
    def test_sidecar_serializes_shared_track_without_promoting_it(self):
        result = BallTrackResult(
            track_id="ball-0",
            items=(
                BallTrackItem(0, (10.0, 20.0), 0.91, "candidate"),
                BallTrackItem(1, None, 0.0, "unavailable", ("gap",)),
            ),
            longest_gap=1,
        )

        sidecar = build_fairwayos_sidecar(result, source="research.mp4")

        self.assertEqual(sidecar["schema_version"], "fairwayos-ball-research.v1")
        self.assertEqual(sidecar["source"], "research.mp4")
        self.assertFalse(sidecar["production_eligible"])
        self.assertEqual(sidecar["human_fallback"]["status"], "available")
        self.assertEqual(sidecar["track"]["observed_frames"], 1)
        self.assertIsNone(sidecar["track"]["items"][1]["center"])
        self.assertIn("gap", sidecar["track"]["items"][1]["warnings"])

    def test_sidecar_rejects_absolute_source_path_to_preserve_provenance_boundary(self):
        result = BallTrackResult("ball-0", (), 0)
        with self.assertRaises(ValueError):
            build_fairwayos_sidecar(result, source="/Users/private/recording.mp4")

    def test_sidecar_rejects_traversal_source_identifier(self):
        result = BallTrackResult("ball-0", (), 0)
        with self.assertRaises(ValueError):
            build_fairwayos_sidecar(result, source="../private/recording.mp4")

    def test_sidecar_rejects_url_and_home_relative_source_identifiers(self):
        result = BallTrackResult("ball-0", (), 0)
        for source in ("https://example.test/recording.mp4", "~/recording.mp4"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                build_fairwayos_sidecar(result, source=source)

    def test_sidecar_accepts_runner_item_shape_without_promoting_it(self):
        sidecar = sidecar_from_mapping({
            "track_id": "ball-0",
            "items": [{"frame": 4, "x": 12.5, "y": 8.5, "confidence": 0.7,
                       "state": "tracked", "warnings": []}],
        }, source="runner.mp4")

        item = sidecar["track"]["items"][0]
        self.assertEqual(item["frame_index"], 4)
        self.assertEqual(item["center"], [12.5, 8.5])
        self.assertFalse(sidecar["production_eligible"])

        payload = {
            "track_id": "ball-0",
            "longest_gap": 0,
            "items": [
                {"frame_index": 3, "center": [4, 5], "confidence": 0.8,
                 "provenance": "candidate", "warnings": []}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "track.json"
            output_path = root / "sidecar.json"
            input_path.write_text(json.dumps(payload))

            main(["fairwayos-ball-sidecar", "--input", str(input_path),
                  "--out", str(output_path), "--source", "clip.mp4"])

            written = json.loads(output_path.read_text())
            self.assertEqual(written["track"]["items"][0]["frame_index"], 3)
            self.assertFalse(written["production_eligible"])

    def test_writer_rejects_promoted_sidecar_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "sidecar.json"
            with self.assertRaises(ValueError):
                write_fairwayos_sidecar(output_path, {"production_eligible": True})


if __name__ == "__main__":
    unittest.main()
