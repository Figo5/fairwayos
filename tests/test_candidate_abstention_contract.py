"""Abstention contract for reference-free candidate association.

A coherent proposed path is research evidence, not an accepted ball observation.
Without independent identity qualification the accepted track must abstain.
"""
import unittest

from ghostcaddie.tracking.candidates import (
    AssociationPolicy,
    Candidate,
    associate,
    associate_proposals,
)


def _c(frame, cx, cy=0.0, size=8.0, score=0.5):
    return Candidate(frame=frame, x1=cx - size / 2, y1=cy - size / 2,
                     x2=cx + size / 2, y2=cy + size / 2, score=score)


class AbstentionContractTests(unittest.TestCase):
    def test_small_stationary_distractor_is_proposal_not_accepted_ball(self):
        frames = {f: [_c(f, 100.0, 100.0, size=8.0, score=0.99)] for f in range(6)}

        proposal = associate_proposals(frames, AssociationPolicy())
        accepted = associate(frames, AssociationPolicy())

        self.assertGreater(proposal.proposed_frames, 0)
        self.assertEqual(proposal.acceptance_state, "proposal_only")
        self.assertIn("independent_identity_required", proposal.rejection_reasons)
        self.assertTrue(all(c is None for c in accepted.values()))

    def test_coherent_broadbox_is_proposal_not_accepted_ball(self):
        frames = {
            f: [Candidate(frame=f, x1=400.0, y1=100.0, x2=570.0, y2=575.0, score=0.99)]
            for f in range(6)
        }
        policy = AssociationPolicy(oversize_reject_diag_px=1000.0)

        proposal = associate_proposals(frames, policy)
        accepted = associate(frames, policy)

        self.assertGreater(proposal.proposed_frames, 0)
        self.assertEqual(proposal.accepted_frames, 0)
        self.assertEqual(proposal.acceptance_state, "proposal_only")
        self.assertTrue(all(c is None for c in accepted.values()))

    def test_default_associate_abstains_without_identity_qualification(self):
        frames = {f: [_c(f, 100.0 + 8 * f, 200.0, size=8.0, score=0.2)] for f in range(6)}

        accepted = associate(frames, AssociationPolicy())
        qualified = associate(frames, AssociationPolicy(), identity_qualified=True)

        self.assertTrue(all(c is None for c in accepted.values()))
        self.assertTrue(all(qualified[f] is not None for f in frames))

    def test_second_order_previous_candidate_regression_still_preserved_for_proposals(self):
        frames = {
            0: [_c(0, 0, 0), _c(0, 100, 0)],
            1: [_c(1, 50, 0)],
            2: [_c(2, 0, 0)],
        }
        policy = AssociationPolicy(score_weight=0.0, size_change_weight=0.0,
                                   min_track_frames=3)

        proposal = associate_proposals(frames, policy).proposed_by_frame

        self.assertAlmostEqual(proposal[0].cx, 100.0, places=6)
        self.assertAlmostEqual(proposal[1].cx, 50.0, places=6)
        self.assertAlmostEqual(proposal[2].cx, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
