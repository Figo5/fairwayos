import unittest

from ghostcaddie.video.research_ball_model import ResearchBallTrack


def candidate(x, y, confidence=0.8):
    return {"center": (float(x), float(y)), "confidence": confidence, "box": [x - 3, y - 3, x + 3, y + 3]}


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


if __name__ == "__main__":
    unittest.main()
