import unittest

from ghostcaddie.video.temporal_candidate_gate import (
    Candidate,
    accept_candidate_run,
    filter_candidate,
)


class TemporalCandidateGateTests(unittest.TestCase):
    def test_rejects_candidate_without_measured_residual_motion(self):
        candidate = Candidate(
            frame_index=10,
            x=500,
            y=400,
            radius=5,
            residual_motion=0.0,
            appearance_score=0.9,
        )
        self.assertIsNone(filter_candidate(candidate, width=1920, height=1080))

    def test_rejects_off_frame_and_person_overlap(self):
        off_frame = Candidate(1, -2, 400, 4, 5.0, 0.9)
        person = Candidate(2, 600, 500, 5, 4.0, 0.9)
        self.assertIsNone(filter_candidate(off_frame, width=1920, height=1080))
        self.assertIsNone(
            filter_candidate(person, width=1920, height=1080, person_boxes=[(550, 450, 700, 650)])
        )

    def test_accepts_only_consecutive_motion_supported_run(self):
        candidates = [
            Candidate(8, 300, 400, 4, 3.0, 0.8),
            Candidate(9, 304, 402, 4, 3.2, 0.82),
            Candidate(11, 330, 410, 4, 3.5, 0.85),
        ]
        accepted = accept_candidate_run(candidates, min_consecutive=2, max_step=20)
        self.assertEqual([c.frame_index for c in accepted], [8, 9])

    def test_rejects_ambiguous_reacquisition(self):
        candidates = [
            Candidate(20, 300, 400, 4, 3.0, 0.8),
            Candidate(20, 308, 402, 4, 3.1, 0.79),
        ]
        self.assertEqual(accept_candidate_run(candidates, min_consecutive=2), [])

    def test_run_selector_rejects_invalid_candidates_without_prefilter(self):
        invalid = [
            Candidate(-1, 300, 400, 4, 3.0, 0.8),
            Candidate(0, 300, 400, 4, 0.0, 0.8),
            Candidate(1, float("nan"), 400, 4, 3.0, 0.8),
            Candidate(2, 300, 400, 31, 3.0, 0.8),
        ]
        self.assertEqual(accept_candidate_run(invalid), [])

    def test_run_selector_rejects_nonfinite_step_limit(self):
        candidates = [Candidate(0, 300, 400, 4, 3.0, 0.8)]
        with self.assertRaises(ValueError):
            accept_candidate_run(candidates, max_step=float("nan"))


if __name__ == "__main__":
    unittest.main()
