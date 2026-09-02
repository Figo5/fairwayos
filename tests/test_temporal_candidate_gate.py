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
        accepted = accept_candidate_run(candidates, width=1920, height=1080, min_consecutive=2, max_step=20)
        self.assertEqual([c.frame_index for c in accepted], [8, 9])

    def test_rejects_ambiguous_reacquisition(self):
        candidates = [
            Candidate(20, 300, 400, 4, 3.0, 0.8),
            Candidate(20, 308, 402, 4, 3.1, 0.79),
        ]
        self.assertEqual(accept_candidate_run(candidates, width=1920, height=1080, min_consecutive=2), [])

    def test_run_selector_rejects_invalid_candidates_without_prefilter(self):
        invalid = [
            Candidate(-1, 300, 400, 4, 3.0, 0.8),
            Candidate(0, 300, 400, 4, 0.0, 0.8),
            Candidate(1, float("nan"), 400, 4, 3.0, 0.8),
            Candidate(2, 300, 400, 31, 3.0, 0.8),
        ]
        self.assertEqual(accept_candidate_run(invalid, width=1920, height=1080), [])

    def test_run_selector_rejects_nonfinite_step_limit(self):
        candidates = [Candidate(0, 300, 400, 4, 3.0, 0.8)]
        with self.assertRaises(ValueError):
            accept_candidate_run(candidates, width=1920, height=1080, max_step=float("nan"))

    def test_run_selector_rejects_stationary_and_boolean_numeric_fields(self):
        stationary = [
            Candidate(0, 300, 400, 4, 3.0, 0.8),
            Candidate(1, 300, 400, 4, 3.0, 0.8),
        ]
        self.assertEqual(accept_candidate_run(stationary, width=1920, height=1080), [])
        for field in range(6):
            values = [0, 300, 400, 4, 3.0, 0.8]
            values[field] = True
            self.assertEqual(accept_candidate_run([Candidate(*values)], width=1920, height=1080), [])

    def test_filter_rejects_malformed_and_boolean_values(self):
        for field in range(1, 6):
            values = [0, 300, 400, 4, 3.0, 0.8]
            values[field] = "3.0"
            malformed = Candidate(*values)
            self.assertIsNone(filter_candidate(malformed, width=1920, height=1080))

    def test_run_selector_rejects_partial_dimensions_and_bad_minimum(self):
        candidate = [Candidate(0, 300, 400, 4, 3.0, 0.8)]
        with self.assertRaises(ValueError):
            accept_candidate_run(candidate, width=1920, height=None)
        with self.assertRaises(ValueError):
            accept_candidate_run(candidate, width=None, height=1080)
        with self.assertRaises(ValueError):
            accept_candidate_run(candidate, width=1920, height=1080, min_consecutive=2.5)

    def test_run_selector_rejects_overflowing_step_limit(self):
        candidate = [Candidate(0, 300, 400, 4, 3.0, 0.8)]
        with self.assertRaises(ValueError):
            accept_candidate_run(candidate, width=1920, height=1080, max_step=10**1000)

    def test_filter_rejects_malformed_or_nonfinite_person_boxes(self):
        candidate = Candidate(0, 300, 400, 4, 3.0, 0.8)
        self.assertIsNone(
            filter_candidate(candidate, width=1920, height=1080, person_boxes=[(1, 2, 3)])
        )
        self.assertIsNone(
            filter_candidate(
                candidate, width=1920, height=1080,
                person_boxes=[(1, float("nan"), 3, 4)],
            )
        )

    def test_filter_rejects_invalid_person_box_geometry_and_inputs(self):
        candidate = Candidate(0, 300, 400, 4, 3.0, 0.8)
        for boxes in (
            [(700, 650, 550, 450)],
            [(-1, 2, 3, 4)],
            None,
        ):
            self.assertIsNone(
                filter_candidate(candidate, width=1920, height=1080, person_boxes=boxes)
            )

    def test_run_selector_rejects_malformed_objects_and_iterables(self):
        class Broken:
            @property
            def frame_index(self):
                raise RuntimeError("malformed")

        self.assertEqual(
            accept_candidate_run(None, width=1920, height=1080), []
        )
        self.assertEqual(
            accept_candidate_run([Broken()], width=1920, height=1080), []
        )

    def test_run_selector_rejects_hostile_numeric_subclasses(self):
        class EvilInt(int):
            def __le__(self, other):
                raise RuntimeError("hostile comparison")

        class EvilReal(float):
            def __float__(self):
                raise RuntimeError("hostile conversion")

        candidate = [Candidate(0, 300, 400, 4, 3.0, 0.8)]
        with self.assertRaises(ValueError):
            accept_candidate_run(candidate, width=EvilInt(1920), height=1080)
        with self.assertRaises(ValueError):
            accept_candidate_run(candidate, width=1920, height=1080, max_step=EvilReal(10))

    def test_run_selector_requires_dimensions_with_explicit_error(self):
        with self.assertRaises(ValueError):
            accept_candidate_run([Candidate(0, 300, 400, 4, 3.0, 0.8)])

    def test_filter_rejects_hostile_numeric_subclasses(self):
        class EvilInt(int):
            def __lt__(self, other):
                raise RuntimeError("hostile comparison")

        class EvilFloat(float):
            def __float__(self):
                raise RuntimeError("hostile conversion")

        candidate = Candidate(0, EvilFloat(300), 400, 4, 3.0, 0.8)
        self.assertIsNone(filter_candidate(candidate, width=EvilInt(1920), height=1080))
        self.assertIsNone(filter_candidate(candidate, width=1920, height=1080))

    def test_run_selector_rejects_invalid_duplicate_and_out_of_frame_candidates(self):
        candidates = [
            Candidate(4, 300, 400, 99, 3.0, 0.8),
            Candidate(4, 300, 400, 4, 3.0, 0.8),
            Candidate(5, 5000, 400, 4, 3.0, 0.8),
        ]
        self.assertEqual(
            accept_candidate_run(candidates, width=1920, height=1080), []
        )

    def test_run_selector_skips_malformed_numeric_values(self):
        malformed = [Candidate(0, "bad", 400, 4, 3.0, 0.8)]
        self.assertEqual(accept_candidate_run(malformed, width=1920, height=1080), [])


if __name__ == "__main__":
    unittest.main()
