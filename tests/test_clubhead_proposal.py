import unittest

from ghostcaddie.video.clubhead_proposal import (
    ClubheadCandidate,
    build_clubhead_observation,
    serialize_clubhead_report,
)


class ClubheadProposalTests(unittest.TestCase):
    def test_fuses_roi_pose_line_contour_and_motion_without_promoting_impact(self):
        result = build_clubhead_observation(
            frame_index=12,
            image_size=(1920, 1080),
            roi=(300, 100, 900, 800),
            pose={"wrist": (620, 520), "elbow": (540, 420), "confidence": 0.8},
            line_candidates=[{"endpoint": (760, 700), "score": 0.9, "length": 220}],
            contour_candidates=[{"center": (755, 704), "score": 0.8, "area": 42}],
            motion_candidates=[{"point": (758, 702), "score": 0.7, "speed": 12}],
        )
        self.assertTrue(result.available)
        self.assertEqual(result.state, "observed")
        self.assertGreater(result.confidence, 0.5)
        self.assertLess(result.uncertainty_px, 20.0)
        self.assertIn("roi", result.evidence)
        self.assertIn("pose", result.evidence)
        self.assertIsNone(result.impact)
        self.assertNotIn("impact", result.evidence)

    def test_disagreement_increases_uncertainty_and_does_not_create_exact_impact(self):
        result = build_clubhead_observation(
            frame_index=13,
            image_size=(1920, 1080),
            roi=(300, 100, 900, 800),
            pose={"wrist": (620, 520), "elbow": (540, 420), "confidence": 0.8},
            line_candidates=[{"endpoint": (760, 700), "score": 0.9, "length": 220}],
            contour_candidates=[{"center": (820, 740), "score": 0.8, "area": 42}],
            motion_candidates=[{"point": (700, 640), "score": 0.7, "speed": 12}],
        )
        self.assertTrue(result.available)
        self.assertGreater(result.uncertainty_px, 20.0)
        self.assertIsNone(result.impact)
        self.assertIsNone(result.impact_frame)

    def test_outside_roi_and_golfer_overlap_are_unavailable(self):
        result = build_clubhead_observation(
            frame_index=14,
            image_size=(1920, 1080),
            roi=(300, 100, 900, 800),
            pose={"wrist": (620, 520), "elbow": (540, 420), "confidence": 0.8},
            golfer_bbox=(500, 300, 400, 500),
            line_candidates=[{"endpoint": (620, 520), "score": 0.99, "length": 220}],
            contour_candidates=[], motion_candidates=[],
        )
        self.assertFalse(result.available)
        self.assertEqual(result.state, "unavailable")
        self.assertIn("roi_or_golfer_exclusion", result.warnings)

    def test_single_proposal_family_is_unavailable(self):
        result = build_clubhead_observation(
            frame_index=15, image_size=(100, 100), roi=(0, 0, 100, 100),
            line_candidates=[{"endpoint": (50, 50), "score": 1.0, "length": 40}],
        )
        self.assertFalse(result.available)
        self.assertIn("insufficient_independent_families", result.warnings)

    def test_serialization_is_deterministic_and_research_only(self):
        candidate = ClubheadCandidate(
            frame_index=1, point=(10.0, 20.0), confidence=0.4,
            uncertainty_px=8.0, state="observed", evidence=("roi",),
        )
        report = {"schema_version": "clubhead-proposal.v1", "production_eligible": False,
                  "impact": None, "frames": [candidate]}
        first = serialize_clubhead_report(report)
        second = serialize_clubhead_report(report)
        self.assertEqual(first, second)
        self.assertIn('"production_eligible":false', first)
        self.assertNotIn("NaN", first)


if __name__ == "__main__":
    unittest.main()
