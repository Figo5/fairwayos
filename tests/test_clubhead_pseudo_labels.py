import unittest

from ghostcaddie.video.clubhead_pseudo_labels import build_pseudo_label


class TestClubheadPseudoLabels(unittest.TestCase):
    def test_emits_explicit_pseudo_label_with_provisional_shaft(self):
        result = build_pseudo_label(
            frame_index=10,
            image_size=(600, 480),
            candidate={"point": (300, 360), "confidence": 0.84, "uncertainty_px": 8.0, "evidence": ["line", "motion"]},
            pose={"wrist": (280, 300), "confidence": 0.9},
            ball_point=(315, 365),
            flow_vector=(4.0, 2.0),
            previous_point=(295, 355),
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["provenance"], {"pseudo_label": True, "ground_truth": False, "research_only": True, "production_eligible": False})
        self.assertEqual(result["clubhead"]["source"], "pseudo_label")
        self.assertEqual(result["shaft"]["source"], "pseudo_label")
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertIn("ball_relation", result["evidence"])

    def test_rejects_weak_or_inconsistent_candidate_without_silent_label(self):
        result = build_pseudo_label(
            frame_index=11,
            image_size=(600, 480),
            candidate={"point": (20, 20), "confidence": 0.4, "uncertainty_px": 140.0, "evidence": ["line"]},
            pose={"wrist": (280, 300), "confidence": 0.9},
            ball_point=(315, 365),
            flow_vector=(4.0, 2.0),
            previous_point=(295, 355),
        )
        self.assertFalse(result["available"])
        self.assertIsNone(result["clubhead"]["value"])
        self.assertEqual(result["clubhead"]["source"], "unavailable")
        self.assertIn("pseudo_label_rejected", result["warnings"])
        self.assertFalse(result["provenance"]["ground_truth"])


if __name__ == "__main__":
    unittest.main()
