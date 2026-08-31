import unittest

from ghostcaddie.video.research_ball_model import (
    ResearchBallMultiHypothesisTrack,
    ResearchBallTrack,
    normalize_box,
    normalize_point,
)


def candidate(x, y, confidence=0.8):
    return {"center": (float(x), float(y)), "confidence": confidence, "box": [x - 3, y - 3, x + 3, y + 3]}


class TestResearchBallModel(unittest.TestCase):
    def test_scales_normalized_coordinates_to_pixels(self):
        self.assertEqual(normalize_point((0.25, 0.5), 600, 480), (150.0, 240.0))
        self.assertEqual(normalize_box((0.1, 0.2, 0.4, 0.6), 600, 480), (60.0, 96.0, 240.0, 288.0))

    def test_preserves_pixel_coordinates(self):
        self.assertEqual(normalize_point((150.0, 240.0), 600, 480), (150.0, 240.0))
        self.assertEqual(normalize_box((60.0, 96.0, 240.0, 288.0), 600, 480), (60.0, 96.0, 240.0, 288.0))

    def test_rejects_invalid_dimensions_or_shape(self):
        with self.assertRaises(ValueError):
            normalize_point((0.2, 0.3), 0, 480)
        with self.assertRaises(ValueError):
            normalize_box((0.1, 0.2, 0.3), 600, 480)


class TestResearchBallTrack(unittest.TestCase):
    def test_initializes_and_predicts_forward_motion(self):
        track = ResearchBallTrack(min_confidence=0.35, max_misses=2)
        first = track.update([candidate(100, 200, 0.8)])
        second = track.update([candidate(110, 200, 0.8)])
        third = track.update([candidate(120, 200, 0.8)])
        self.assertEqual(first["state"], "observed")
        self.assertEqual(second["state"], "observed")
        self.assertEqual(third["state"], "observed")
        self.assertEqual(third["point"], {"x": 120.0, "y": 200.0})

    def test_rejects_motion_jump_and_terminates_after_misses(self):
        track = ResearchBallTrack(min_confidence=0.35, max_step=25.0, max_misses=2)
        track.update([candidate(100, 200)])
        track.update([candidate(110, 200)])
        rejected = track.update([candidate(400, 100)])
        terminated = track.update([candidate(401, 100)])
        self.assertEqual(rejected["state"], "predicted")
        self.assertEqual(rejected["warning"], "motion_constraint_rejected")
        self.assertEqual(terminated["state"], "terminated")
        self.assertIsNone(terminated["point"])

    def test_confidence_decays_during_short_occlusion(self):
        track = ResearchBallTrack(min_confidence=0.35, confidence_decay=0.2, max_misses=3)
        track.update([candidate(100, 200, 0.8)])
        missed = track.update([])
        self.assertEqual(missed["state"], "predicted")
        self.assertAlmostEqual(missed["confidence"], 0.6)
        self.assertEqual(missed["warning"], "confidence_decayed")

    def test_low_confidence_candidate_does_not_restart_terminated_track(self):
        track = ResearchBallTrack(min_confidence=0.5, max_misses=1)
        track.update([candidate(100, 200, 0.8)])
        track.update([])
        terminated = track.update([candidate(101, 200, 0.4)])
        self.assertEqual(terminated["state"], "terminated")
        self.assertIsNone(terminated["point"])


class TestResearchBallMultiHypothesisTrack(unittest.TestCase):
    def test_weak_detection_cannot_revive_terminated_track(self):
        track = ResearchBallMultiHypothesisTrack(reacquire_confidence=0.75, max_misses=1)
        track.update([candidate(100, 200, 0.9)])
        track.update([])
        result = track.update([candidate(101, 200, 0.6)])
        self.assertEqual(result["state"], "terminated")
        self.assertEqual(result["warning"], "reacquisition_insufficient_evidence")

    def test_reacquisition_requires_two_consistent_strong_frames(self):
        track = ResearchBallMultiHypothesisTrack(reacquire_confidence=0.75, max_misses=1, max_step=30)
        track.update([candidate(100, 200, 0.9)])
        track.update([])
        pending = track.update([candidate(110, 200, 0.8)])
        reacquired = track.update([candidate(120, 200, 0.82)])
        self.assertEqual(pending["state"], "terminated")
        self.assertEqual(pending["warning"], "reacquisition_pending")
        self.assertEqual(reacquired["state"], "reacquired")
        self.assertEqual(reacquired["point"], {"x": 120.0, "y": 200.0})

    def test_continuity_beats_higher_confidence_drift_candidate(self):
        track = ResearchBallMultiHypothesisTrack(max_step=35, max_hypotheses=3)
        track.update([candidate(100, 200, 0.8)])
        track.update([candidate(110, 200, 0.8), candidate(100, 350, 0.95)])
        result = track.update([candidate(120, 200, 0.7), candidate(125, 350, 0.99)])
        self.assertEqual(result["state"], "observed")
        self.assertEqual(result["point"], {"x": 120.0, "y": 200.0})
        self.assertGreaterEqual(result["hypothesis_count"], 2)

    def test_ambiguous_frame_is_not_promoted_to_reacquisition(self):
        track = ResearchBallMultiHypothesisTrack(reacquire_confidence=0.75, max_misses=1)
        track.update([candidate(100, 200, 0.9)])
        track.update([])
        result = track.update([candidate(110, 200, 0.8), candidate(115, 210, 0.81)])
        self.assertEqual(result["state"], "terminated")
        self.assertEqual(result["warning"], "reacquisition_ambiguous")


if __name__ == "__main__":
    unittest.main()
