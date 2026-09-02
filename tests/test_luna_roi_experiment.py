import unittest

from ghostcaddie.video.luna_roi_experiment import (
    RESEARCH_FLAGS,
    select_temporal_track,
    validate_pseudo_labels,
)


class LunaRoiExperimentTests(unittest.TestCase):
    def test_pseudo_label_flags_are_fail_closed(self):
        labels = [{
            "frame_index": 3,
            "pseudo_label": True,
            "ground_truth": False,
            "research_only": True,
            "production_eligible": False,
            "ball": {"center": [960, 560], "radius_px": 36, "visibility": "visible", "confidence": 0.8},
        }]
        self.assertEqual(validate_pseudo_labels(labels), labels)
        self.assertEqual(RESEARCH_FLAGS, {
            "pseudo_label": True, "ground_truth": False,
            "research_only": True, "production_eligible": False,
        })

    def test_framewise_schema_requires_requested_objects_and_flags(self):
        from ghostcaddie.video.luna_roi_experiment import validate_framewise_labels
        labels = [{
            "ball": {"x": 960, "y": 560, "radius": 30, "visible": True, "confidence": 0.8},
            "clubhead": {"x": 1050, "y": 620, "visible": True, "confidence": 0.7},
            "shaft": {"x1": 700, "y1": 0, "x2": 1000, "y2": 600, "confidence": 0.8},
            "pseudo_label": True, "ground_truth": False, "research_only": True, "production_eligible": False,
        }]
        self.assertEqual(validate_framewise_labels(labels), labels)

    def test_temporal_track_rejects_person_and_club_regions(self):
        candidates = [
            {"frame_index": 0, "center": [950, 560], "radius_px": 18, "confidence": 0.8},
            {"frame_index": 1, "center": [970, 565], "radius_px": 19, "confidence": 0.82},
        ]
        result = select_temporal_track(
            candidates,
            roi=[700, 420, 1250, 760],
            golfer_boxes=[[850, 400, 1100, 760]],
            club_points=[[950, 560], [970, 565]],
            min_consecutive=2,
        )
        self.assertEqual(result["state"], "unavailable")
        self.assertIn("person_or_club_region", result["warnings"])
        self.assertFalse(result["production_eligible"])

    def test_temporal_track_requires_consecutive_motion_supported_candidates(self):
        candidates = [
            {"frame_index": 0, "center": [600, 300], "radius_px": 12, "confidence": 0.8},
            {"frame_index": 2, "center": [650, 300], "radius_px": 12, "confidence": 0.8},
        ]
        result = select_temporal_track(candidates, roi=[500, 200, 900, 500], min_consecutive=2)
        self.assertEqual(result["state"], "unavailable")
        self.assertIn("consecutive_support_required", result["warnings"])


if __name__ == "__main__":
    unittest.main()
