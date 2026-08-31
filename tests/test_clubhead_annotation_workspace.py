import unittest

from ghostcaddie.video.clubhead_annotation_workspace import build_clubhead_annotation_workspace


class TestClubheadAnnotationWorkspace(unittest.TestCase):
    def frames(self):
        return [
            {"frame_index": 0, "source_frame_index": 40, "timestamp_seconds": 1.6, "filename": "frames/frame_000001.jpg"},
            {"frame_index": 1, "source_frame_index": 41, "timestamp_seconds": 1.64, "filename": "frames/frame_000002.jpg"},
        ]

    def test_is_deterministic_and_contains_sequence_label_controls(self):
        kwargs = {"frames": self.frames(), "video": {"width": 600, "height": 480, "frame_count": 2, "frame_rate": 25.0}, "clip_id": "mmu_seed"}
        first = build_clubhead_annotation_workspace(**kwargs)
        second = build_clubhead_annotation_workspace(**kwargs)
        self.assertEqual(first, second)
        for marker in ("clubhead", "shaft_grip", "shaft_neck", "visible", "occluded", "unavailable", "pseudo_label", "ground_truth", "source_frame_index", "Export dataset"):
            self.assertIn(marker, first)
        self.assertNotRegex(first.lower(), r'(?:src|href)=["\'](?:https?:|//)')
        self.assertNotIn("fetch(", first)

    def test_rejects_nonconsecutive_frames_and_remote_assets(self):
        with self.assertRaises(ValueError):
            build_clubhead_annotation_workspace(
                [{"frame_index": 1, "source_frame_index": 40, "timestamp_seconds": 1.6, "filename": "x.jpg"}],
                video={"width": 600, "height": 480, "frame_count": 1, "frame_rate": 25.0}, clip_id="x")
        bad = [dict(self.frames()[0], filename="https://bad.test/frame.jpg"), self.frames()[1]]
        with self.assertRaises(ValueError):
            build_clubhead_annotation_workspace(bad, video={"width": 600, "height": 480, "frame_count": 2, "frame_rate": 25.0}, clip_id="x")


if __name__ == "__main__":
    unittest.main()
