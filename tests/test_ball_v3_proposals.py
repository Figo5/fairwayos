"""Proposal stage: recall first, and no silent losses.

The v2 pass lost the ball to a score floor and never noticed, because nothing
measured the proposal stage on its own. These tests exist so a future change that
quietly costs recall fails here instead of three stages downstream.
"""
from __future__ import annotations

import unittest

import numpy as np

from ghostcaddie.video import ball_candidates as bc
from ghostcaddie.video import ball_v3_blobs as bblobs
from ghostcaddie.video import ball_v3_proposals as bp


def turf(width: int = 480, height: int = 360, seed: int = 7) -> np.ndarray:
    """Green noise that looks enough like grass to be a fair background."""
    rng = np.random.default_rng(seed)
    frame = np.zeros((height, width, 3), np.uint8)
    frame[..., 0] = rng.integers(20, 55, (height, width))    # B
    frame[..., 1] = rng.integers(90, 150, (height, width))   # G
    frame[..., 2] = rng.integers(30, 70, (height, width))    # R
    return frame


def put_disc(frame: np.ndarray, x: int, y: int, radius: int,
             colour: tuple[int, int, int]) -> np.ndarray:
    import cv2
    out = frame.copy()
    cv2.circle(out, (x, y), radius, colour, -1, lineType=cv2.LINE_AA)
    return out


def contexts_for(frames):
    return bc.build_contexts(frames, list(range(len(frames))), max_side=2880)[0]


class LocalDistinctness(unittest.TestCase):
    def test_finds_a_white_disc_on_turf(self):
        frame = put_disc(turf(), 240, 180, 7, (250, 250, 250))
        found = bblobs.candidates_local_distinctness(contexts_for([frame])[0])
        best = min(found, key=lambda c: (c.x - 240) ** 2 + (c.y - 180) ** 2)
        self.assertLess(np.hypot(best.x - 240, best.y - 180), 10.0)

    def test_finds_a_ball_that_is_not_bright(self):
        """The failure this generator exists for.

        A yellow ball on yellow-green turf and an orange ball on a green mat
        differ from their surround in chroma, not lightness. A generator keyed on
        brightness -- or one whose compactness term is a lightness DoG -- scores
        near zero on exactly these, which cost pexels_6573621 all of its recall.
        """
        for colour in ((40, 200, 230),      # yellow
                       (30, 150, 240),      # orange
                       (120, 120, 120)):    # mid grey
            with self.subTest(colour=colour):
                frame = put_disc(turf(), 200, 150, 7, colour)
                found = bblobs.candidates_local_distinctness(contexts_for([frame])[0])
                self.assertTrue(found)
                best = min(found,
                           key=lambda c: (c.x - 200) ** 2 + (c.y - 150) ** 2)
                self.assertLess(np.hypot(best.x - 200, best.y - 150), 12.0)

    def test_one_hot_region_does_not_starve_the_rest_of_the_frame(self):
        """Normalising by the frame maximum let a chain-link fence spend the whole
        budget on pexels_6573621. A high percentile must not."""
        frame = put_disc(turf(), 120, 180, 7, (250, 250, 250))
        import cv2
        # a violently high-contrast region far from the ball
        cv2.rectangle(frame, (330, 40), (460, 320), (255, 0, 255), -1)
        for offset in range(40, 320, 12):
            cv2.line(frame, (330, offset), (460, offset), (0, 255, 255), 2)
        found = bblobs.candidates_local_distinctness(contexts_for([frame])[0])
        near = [c for c in found if np.hypot(c.x - 120, c.y - 180) < 12.0]
        self.assertTrue(near, "the ball was crowded out by an unrelated hot region")

    def test_coordinates_come_back_in_native_pixels(self):
        """A 4K frame is processed downscaled and on a pyramid; a candidate that
        forgets to map back is the defect that made v2's first radial-symmetry
        implementation useless."""
        big = np.kron(turf(240, 180, seed=3), np.ones((8, 8, 1), np.uint8))
        frame = put_disc(big, 900, 700, 11, (250, 250, 250))
        context = bc.build_contexts([frame], [0], max_side=960)[0][0]
        found = bblobs.candidates_local_distinctness(context)
        best = min(found, key=lambda c: (c.x - 900) ** 2 + (c.y - 700) ** 2)
        self.assertLess(np.hypot(best.x - 900, best.y - 700), 30.0)


class ProposalStage(unittest.TestCase):
    def test_no_score_floor_by_default(self):
        """v2's ConsensusParams.min_score = 0.25 deleted the true ball on 19 of
        the 26 frames where a generator had found it."""
        self.assertEqual(bp.ProposalParams().min_score, 0.0)

    def test_dedupe_keeps_a_small_proposal_next_to_a_large_one(self):
        """Tolerance must come from the SMALLER radius. Using the larger lets a
        coarse-scale blob swallow the fine-scale proposal sitting on the ball."""
        big = bc.Candidate2(frame_index=0, x=100.0, y=100.0, radius=60.0,
                            score=0.9, method="radial_symmetry")
        small = bc.Candidate2(frame_index=0, x=130.0, y=100.0, radius=4.0,
                              score=0.5, method="local_distinctness")
        proposals = bp.propose_frame([big, small], bp.ProposalParams())
        self.assertEqual(len(proposals), 2)

    def test_dedupe_merges_methods_agreeing_on_one_object(self):
        here = [bc.Candidate2(frame_index=0, x=50.0, y=50.0, radius=5.0,
                              score=0.5, method=name)
                for name in ("radial_symmetry", "brightness_cue", "local_contrast")]
        proposals = bp.propose_frame(here, bp.ProposalParams())
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].witnesses, 3)
        self.assertEqual(len(proposals[0].method_scores), 3)

    def test_frame_budget_is_respected(self):
        rng = np.random.default_rng(0)
        many = [bc.Candidate2(frame_index=0, x=float(rng.integers(0, 4000)),
                              y=float(rng.integers(0, 4000)), radius=3.0,
                              score=float(rng.random()), method="brightness_cue")
                for _ in range(900)]
        params = bp.ProposalParams(max_per_frame=25)
        self.assertLessEqual(len(bp.propose_frame(many, params)), 25)

    def test_propose_rejects_an_unknown_generator(self):
        with self.assertRaises(ValueError):
            bp.propose([turf()], [0], methods=("no_such_method",))

    def test_propose_rejects_mismatched_frames_and_indices(self):
        with self.assertRaises(ValueError):
            bp.propose([turf(), turf()], [0])

    def test_recall_counts_a_prediction_on_a_different_object_as_a_miss(self):
        """A prediction further away than the ball is wide is not a near miss."""
        proposal = bp.Proposal(frame_index=0, x=300.0, y=300.0, radius=5.0,
                               best_score=1.0, methods=("brightness_cue",))
        result = bp.proposal_recall({0: [proposal]}, {0: (100.0, 100.0, 5.0)})
        self.assertEqual(result["found"], 0)
        self.assertEqual(result["missed"], 1)

    def test_recall_counts_a_proposal_on_the_ball(self):
        proposal = bp.Proposal(frame_index=0, x=102.0, y=99.0, radius=5.0,
                               best_score=1.0, methods=("local_distinctness",))
        result = bp.proposal_recall({0: [proposal]}, {0: (100.0, 100.0, 8.0)})
        self.assertEqual(result["found"], 1)

    def test_the_new_generator_is_actually_wired_in(self):
        self.assertIn("local_distinctness", bp.METHOD_NAMES)
        self.assertIn("local_distinctness", bp.GENERATORS)

    def test_end_to_end_finds_a_ball_moving_across_turf(self):
        frames = [put_disc(turf(seed=11), 120 + 18 * i, 150, 6, (250, 250, 250))
                  for i in range(4)]
        per_frame, info = bp.propose(frames, list(range(4)))
        truth = {i: (120.0 + 18 * i, 150.0, 6.0) for i in range(4)}
        result = bp.proposal_recall(per_frame, truth)
        self.assertEqual(result["found"], 4, result)
        self.assertGreater(info["mean_proposals"], 0)


if __name__ == "__main__":
    unittest.main()
