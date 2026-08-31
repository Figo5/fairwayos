import json
import tempfile
import unittest
from pathlib import Path

from ghostcaddie.video.window_selection import (
    append_clip_window_record,
    select_bounded_windows,
)


class WindowSelectionTests(unittest.TestCase):
    def test_selects_ranked_non_overlapping_windows_with_hard_bound(self):
        scores = [0.0, 0.2, 0.9, 0.2, 0.0, 0.1, 0.8, 0.1, 0.0, 0.7]
        windows = select_bounded_windows(scores, radius=1, max_windows=2, min_peak_score=0.5)
        self.assertEqual([(w.start_frame, w.end_frame, w.peak_frame) for w in windows],
                         [(1, 3, 2), (5, 7, 6)])
        self.assertLessEqual(len(windows), 2)

    def test_ties_are_deterministic_and_overlapping_peak_is_suppressed(self):
        windows = select_bounded_windows([0.8, 0.8, 0.8, 0.1], radius=1, max_windows=5)
        self.assertEqual([(w.start_frame, w.end_frame, w.peak_frame) for w in windows], [(0, 1, 0)])

    def test_append_persists_one_json_record_per_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "windows.jsonl"
            append_clip_window_record(path, {"clip_id": "a", "windows": []})
            append_clip_window_record(path, {"clip_id": "b", "windows": []})
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([record["clip_id"] for record in records], ["a", "b"])

    def test_rejects_unbounded_or_invalid_configuration(self):
        with self.assertRaisesRegex(ValueError, "max_windows"):
            select_bounded_windows([1.0], radius=1, max_windows=0)
        with self.assertRaisesRegex(ValueError, "radius"):
            select_bounded_windows([1.0], radius=-1, max_windows=1)

    def test_rejects_non_finite_jsonl_values(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                append_clip_window_record(
                    Path(directory) / "windows.jsonl",
                    {"clip_id": "nan", "score": float("nan")},
                )


if __name__ == "__main__":
    unittest.main()
