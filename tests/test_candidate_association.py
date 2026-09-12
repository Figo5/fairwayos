"""Reference-free temporal association over frozen detector candidates.

Context, from two independent audits:
  * the golf-ball ONNX detector DOES place a tiny candidate on the visually
    apparent ball in 40 of 43 moving frames, at raw confidences as low as 0.01,
    buried among background and clubhead distractors. Detection is not the
    bottleneck; SELECTION is.
  * its decoder accepts NaN coordinates and negative/inverted boxes instead of
    rejecting them, so invalid geometry can reach NMS and rendering.

The association below therefore reads ALL frozen candidates and picks one
coherent trajectory. It is initialised by the optimiser, not by a coordinate:
no reference point, no seed, no crop. Frames where the chosen path has no
candidate emit NOTHING -- gaps stay gaps and are never interpolated.
"""
import math
import unittest

from ghostcaddie.tracking.candidates import (
    Candidate, InvalidCandidate, AssociationPolicy, associate,
)


class CandidateValidationTests(unittest.TestCase):
    def test_a_nan_coordinate_is_rejected_not_silently_accepted(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(InvalidCandidate):
                Candidate(frame=1, x1=bad, y1=0.0, x2=10.0, y2=10.0, score=0.5)

    def test_a_nan_score_is_rejected(self):
        with self.assertRaises(InvalidCandidate):
            Candidate(frame=1, x1=0.0, y1=0.0, x2=10.0, y2=10.0,
                      score=float("nan"))

    def test_an_inverted_box_is_rejected_not_reordered(self):
        """Silently swapping corners would hide a decoder bug."""
        with self.assertRaises(InvalidCandidate):
            Candidate(frame=1, x1=10.0, y1=0.0, x2=2.0, y2=10.0, score=0.5)
        with self.assertRaises(InvalidCandidate):
            Candidate(frame=1, x1=0.0, y1=10.0, x2=10.0, y2=2.0, score=0.5)

    def test_a_zero_area_box_is_rejected(self):
        with self.assertRaises(InvalidCandidate):
            Candidate(frame=1, x1=5.0, y1=5.0, x2=5.0, y2=10.0, score=0.5)

    def test_a_valid_box_reports_its_centre_and_size(self):
        c = Candidate(frame=7, x1=100.0, y1=200.0, x2=110.0, y2=214.0, score=0.3)
        self.assertAlmostEqual(c.cx, 105.0)
        self.assertAlmostEqual(c.cy, 207.0)
        self.assertAlmostEqual(c.width, 10.0)
        self.assertAlmostEqual(c.height, 14.0)

    def test_from_xyxy_skips_invalid_boxes_and_counts_them(self):
        from ghostcaddie.tracking.candidates import parse_candidates
        rows = [{"frame": 1, "box_xyxy": [0, 0, 10, 10], "score": 0.2},
                {"frame": 1, "box_xyxy": [10, 0, 2, 10], "score": 0.9},
                {"frame": 1, "box_xyxy": [0, 0, float("nan"), 10], "score": 0.9}]
        kept, rejected = parse_candidates(rows)
        self.assertEqual(len(kept), 1)
        self.assertEqual(rejected, 2)


def _c(frame, cx, cy, size=8.0, score=0.1):
    return Candidate(frame=frame, x1=cx - size / 2, y1=cy - size / 2,
                     x2=cx + size / 2, y2=cy + size / 2, score=score)


class AssociationTests(unittest.TestCase):
    POLICY = AssociationPolicy()

    def test_a_smooth_tiny_trajectory_is_preferred_over_high_scoring_noise(self):
        """The real ball is faint and consistent; distractors are bright and
        incoherent. Association must follow coherence, not confidence."""
        frames = {}
        for i, f in enumerate(range(10, 20)):
            frames[f] = [
                _c(f, 100.0 + 10 * i, 200.0 + 5 * i, score=0.02),   # the ball
                _c(f, 600.0 + 250 * ((i % 2) * 2 - 1), 400.0, score=0.95),  # noise
            ]
        track = associate(frames, self.POLICY)
        self.assertEqual(len(track), 10)
        for i, f in enumerate(range(10, 20)):
            self.assertIsNotNone(track[f], f"frame {f} dropped")
            self.assertAlmostEqual(track[f].cx, 100.0 + 10 * i, places=6)

    def test_a_broad_stationary_high_score_box_loses_to_a_tiny_moving_ball(self):
        """Golf-ball identity includes small object scale; a smooth broad body/
        background box must not win just because it is high confidence and long."""
        frames = {}
        for i, f in enumerate(range(10, 18)):
            frames[f] = [
                _c(f, 100.0 + 8 * i, 200.0, size=8.0, score=0.2),
                Candidate(frame=f, x1=400.0, y1=100.0, x2=570.0,
                          y2=575.0, score=0.99),
            ]
        track = associate(frames, self.POLICY)
        for i, f in enumerate(range(10, 18)):
            self.assertAlmostEqual(track[f].cx, 100.0 + 8 * i, places=6)

    def test_second_order_association_keeps_distinct_incoming_velocities(self):
        """The same current candidate can be reached with different velocities;
        retaining only one velocity per (frame,candidate) loses the lower-
        acceleration continuation."""
        frames = {
            0: [_c(0, 0, 0), _c(0, 100, 0)],
            1: [_c(1, 50, 0)],
            2: [_c(2, 0, 0)],
        }
        policy = AssociationPolicy(score_weight=0.0, size_change_weight=0.0,
                                   min_track_frames=3)
        track = associate(frames, policy)
        self.assertAlmostEqual(track[0].cx, 100.0, places=6)
        self.assertAlmostEqual(track[1].cx, 50.0, places=6)
        self.assertAlmostEqual(track[2].cx, 0.0, places=6)

    def test_candidate_dict_preserves_exact_source_candidate_identity(self):
        c = Candidate(frame=7, x1=1.123456789, y1=2.25, x2=9.5, y2=10.75,
                      score=0.123456789, candidate_index=4)
        self.assertEqual(c.to_dict()["candidate_index"], 4)
        self.assertEqual(c.to_dict()["xyxy"], [1.123456789, 2.25, 9.5, 10.75])

    def test_a_frame_with_no_candidate_emits_nothing_and_is_not_interpolated(self):
        frames = {f: [_c(f, 100.0 + 10 * (f - 10), 200.0)] for f in range(10, 20)}
        frames[14] = []
        track = associate(frames, self.POLICY)
        self.assertIsNone(track[14])
        self.assertIsNotNone(track[13])
        self.assertIsNotNone(track[15])

    def test_a_frame_whose_only_candidates_are_incoherent_emits_nothing(self):
        frames = {f: [_c(f, 100.0 + 10 * (f - 10), 200.0)] for f in range(10, 20)}
        frames[14] = [_c(14, 1200.0, 60.0)]      # far outside any plausible step
        track = associate(frames, self.POLICY)
        self.assertIsNone(track[14], "an incoherent jump must not be selected")

    def test_association_uses_no_reference_coordinate_and_no_seed(self):
        """Same candidates, shifted bodily in space: the track must follow them,
        proving nothing is anchored to a fixed coordinate."""
        mk = lambda off: {f: [_c(f, off + 10 * (f - 10), 200.0 + off)]
                          for f in range(10, 20)}
        a, b = associate(mk(0.0), self.POLICY), associate(mk(500.0), self.POLICY)
        for f in range(10, 20):
            self.assertAlmostEqual(b[f].cx - a[f].cx, 500.0, places=6)

    def test_an_empty_interval_yields_an_empty_track(self):
        self.assertEqual(associate({}, self.POLICY), {})

    def test_every_selected_candidate_is_one_of_the_inputs(self):
        """Nothing is synthesised: each emitted point must be an actual
        detector candidate object from that frame."""
        frames = {f: [_c(f, 100.0 + 10 * (f - 10), 200.0), _c(f, 900.0, 100.0)]
                  for f in range(10, 20)}
        track = associate(frames, self.POLICY)
        for f, sel in track.items():
            if sel is not None:
                self.assertIn(sel, frames[f])

    def test_the_policy_is_serialisable_so_it_can_be_frozen_before_a_run(self):
        d = self.POLICY.to_dict()
        for k in ("max_step_px_per_frame", "miss_cost", "accel_weight",
                  "size_change_weight", "score_weight", "min_track_frames"):
            self.assertIn(k, d)


if __name__ == "__main__":
    unittest.main()
