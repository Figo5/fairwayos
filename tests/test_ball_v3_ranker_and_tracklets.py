"""Listwise ranking, tracklet features, and the acceptance rule.

These pin the three behaviours that separate v3 from v2:

* the ranker is trained on a *comparison within one frame*, so the loss must
  actually move a same-frame competitor down when it moves the ball up;
* abstention competes in that same softmax rather than being a threshold applied
  afterwards;
* a near-tie is never an observation. v2 accepted a winner that beat its rival by
  0.003 and called three clutter tracks accepted.
"""
from __future__ import annotations

import unittest

import numpy as np
import torch

from ghostcaddie.video import ball_v3_proposals as bp
from ghostcaddie.video import ball_v3_ranker as brank
from ghostcaddie.video import ball_v3_tracklets as btrk


def numeric_batch(count: int) -> np.ndarray:
    return np.zeros((count, brank.NUMERIC_FEATURES), dtype=np.float32)


class Ranker(unittest.TestCase):
    def test_feature_width_does_not_depend_on_which_generators_ran(self):
        """An ablation must remove information, not resize the input layer --
        otherwise saved weights stop loading and the comparison is meaningless."""
        self.assertEqual(brank.NUMERIC_FEATURES,
                         4 + 7 + 2 + len(bp.ALL_METHOD_NAMES))
        proposal = bp.Proposal(frame_index=0, x=10.0, y=10.0, radius=4.0,
                               best_score=0.5, methods=("local_distinctness",),
                               method_scores={"local_distinctness": 0.5})
        values = brank.numeric_features(proposal, None, 1920, 1080)
        self.assertEqual(len(values), brank.NUMERIC_FEATURES)

    def test_listwise_loss_prefers_the_target(self):
        null = torch.tensor(-2.0)
        right = brank.listwise_loss(torch.tensor([3.0, 0.0, 0.0]), 0, null)
        wrong = brank.listwise_loss(torch.tensor([0.0, 0.0, 3.0]), 0, null)
        self.assertLess(float(right), float(wrong))

    def test_listwise_loss_can_choose_the_null_slot(self):
        """"None of these is the ball" has to be a first-class answer, not a
        threshold bolted on after the fact."""
        null = torch.tensor(4.0)
        abstain = brank.listwise_loss(torch.tensor([0.0, 0.0]), None, null)
        claim = brank.listwise_loss(torch.tensor([0.0, 0.0]), 0, null)
        self.assertLess(float(abstain), float(claim))

    def test_the_gradient_is_comparative_within_one_frame(self):
        """This is the whole point of the objective. Raising the true ball must
        lower its same-frame competitor; a per-crop binary loss cannot do that."""
        scores = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        null = torch.tensor(-2.0, requires_grad=True)
        brank.listwise_loss(scores, 1, null).backward()
        gradient = scores.grad.numpy()
        self.assertLess(gradient[1], 0.0, "the target should be pushed up")
        self.assertGreater(gradient[0], 0.0, "a competitor should be pushed down")
        self.assertGreater(gradient[2], 0.0)

    def test_rank_frame_reports_a_margin_over_the_runner_up(self):
        model = brank.CandidateRanker()
        images = np.zeros((4, 64, 64, 7), np.uint8)
        result = brank.rank_frame(model, images, numeric_batch(4),
                                  device=torch.device("cpu"))
        self.assertIn("margin", result)
        self.assertGreaterEqual(result["margin"], 0.0)
        self.assertEqual(len(result["probabilities"]), 4)

    def test_rank_frame_abstains_on_an_empty_candidate_list(self):
        model = brank.CandidateRanker()
        result = brank.rank_frame(model, np.zeros((0, 64, 64, 7), np.uint8),
                                  numeric_batch(0), device=torch.device("cpu"))
        self.assertIsNone(result["best"])
        self.assertEqual(result["null_probability"], 1.0)

    def test_weights_round_trip(self):
        import tempfile
        from pathlib import Path
        model = brank.CandidateRanker()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ranker.pt"
            digest = brank.save_ranker(model, path)
            self.assertEqual(len(digest), 64)
            restored = brank.load_ranker(path, device_override=torch.device("cpu"))
            images = np.zeros((3, 64, 64, 7), np.uint8)
            a = brank.rank_frame(model, images, numeric_batch(3),
                                 device=torch.device("cpu"))
            b = brank.rank_frame(restored, images, numeric_batch(3),
                                 device=torch.device("cpu"))
            np.testing.assert_allclose(a["probabilities"], b["probabilities"],
                                       atol=1e-5)


def nodes(positions, *, probability=0.9, torso=None, inside=0.0):
    return {i: [btrk.Node(frame_index=i, x=x, y=y, radius=5.0,
                          probability=probability, margin=0.5, is_top1=True,
                          inside_torso=inside, torso_centre=torso)]
            for i, (x, y) in enumerate(positions)}


class Tracklets(unittest.TestCase):
    def test_linking_follows_a_moving_candidate(self):
        per_frame = nodes([(10.0 + 6 * i, 10.0) for i in range(6)])
        tracks = btrk.build_tracklets(per_frame, {i: (0.0, 0.0) for i in range(6)})
        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(tracks[0].nodes), 6)

    def test_camera_motion_is_removed_before_distances_are_compared(self):
        """On a pan the background moves further between frames than a struck ball
        does. Without compensation every link decision is the camera's."""
        shift = 40.0
        per_frame = nodes([(10.0 + shift * i, 10.0) for i in range(6)])
        shifts = {i: (shift, 0.0) for i in range(6)}
        features = dict(zip(btrk.FEATURE_NAMES, btrk.tracklet_features(
            btrk.build_tracklets(per_frame, shifts)[0], window=6,
            camera_shift=shifts)))
        self.assertLess(features["mean_step_radii"], 0.5,
                        "a candidate pinned to the background should read as static")
        self.assertEqual(features["static_fraction"], 1.0)

    def test_a_genuinely_moving_candidate_is_not_static(self):
        per_frame = nodes([(10.0 + 30.0 * i, 10.0) for i in range(6)])
        shifts = {i: (0.0, 0.0) for i in range(6)}
        features = dict(zip(btrk.FEATURE_NAMES, btrk.tracklet_features(
            btrk.build_tracklets(per_frame, shifts)[0], window=6,
            camera_shift=shifts)))
        self.assertGreater(features["mean_step_radii"], 1.0)
        self.assertEqual(features["static_fraction"], 0.0)

    def test_a_candidate_welded_to_the_torso_is_detected(self):
        """A shirt highlight travels with the golfer. That is the thing appearance
        cannot settle and time can."""
        torso = [(100.0 + 10 * i, 200.0) for i in range(6)]
        per_frame = {i: [btrk.Node(frame_index=i, x=100.0 + 10 * i, y=150.0,
                                   radius=5.0, probability=0.9, is_top1=True,
                                   inside_torso=1.0, torso_centre=torso[i])]
                     for i in range(6)}
        shifts = {i: (0.0, 0.0) for i in range(6)}
        features = dict(zip(btrk.FEATURE_NAMES, btrk.tracklet_features(
            btrk.build_tracklets(per_frame, shifts)[0], window=6,
            camera_shift=shifts)))
        self.assertGreater(features["moves_with_torso"], 0.8)
        self.assertEqual(features["body_inside_fraction"], 1.0)

    def test_a_ball_leaving_the_golfer_does_not_move_with_the_torso(self):
        # the ball departs at about 5 radii per frame -- fast, but inside the
        # plausibility bound the rest of the system uses. A ball moving 12 radii
        # per frame is not something anything here claims to follow.
        torso = [(100.0 + 10 * i, 200.0) for i in range(6)]
        per_frame = {i: [btrk.Node(frame_index=i, x=100.0 + 24 * i, y=150.0 - 8 * i,
                                   radius=5.0, probability=0.9, is_top1=True,
                                   torso_centre=torso[i])]
                     for i in range(6)}
        shifts = {i: (0.0, 0.0) for i in range(6)}
        features = dict(zip(btrk.FEATURE_NAMES, btrk.tracklet_features(
            btrk.build_tracklets(per_frame, shifts)[0], window=6,
            camera_shift=shifts)))
        self.assertLess(features["moves_with_torso"], 0.5)

    def test_features_are_always_the_declared_length(self):
        per_frame = nodes([(10.0, 10.0), (12.0, 10.0), (14.0, 10.0)])
        shifts = {i: (0.0, 0.0) for i in range(3)}
        track = btrk.build_tracklets(per_frame, shifts)[0]
        self.assertEqual(len(btrk.tracklet_features(track, window=3,
                                                    camera_shift=shifts)),
                         len(btrk.FEATURE_NAMES))

    def test_ablation_mask_zeroes_a_named_group(self):
        mask = btrk.mask_for_ablation(["motion"])
        for name in btrk.MOTION_GROUPS["motion"]:
            self.assertEqual(mask[btrk.FEATURE_NAMES.index(name)], 0.0)
        self.assertEqual(mask[btrk.FEATURE_NAMES.index("mean_rank_probability")], 1.0)

    def test_cycle_consistency_is_a_feature_and_not_a_gate(self):
        """Every clip in the v2 final test scored cycle consistency 1.0, including
        the three that tracked pure clutter with nine false positives each."""
        self.assertIn("cycle_consistency", btrk.FEATURE_NAMES)
        per_frame = nodes([(10.0, 10.0), (12.0, 10.0), (14.0, 10.0)])
        shifts = {i: (0.0, 0.0) for i in range(3)}
        track = btrk.build_tracklets(per_frame, shifts)[0]
        features = btrk.tracklet_features(track, window=3, camera_shift=shifts)
        self.assertEqual(len(features), len(btrk.FEATURE_NAMES))

    def test_discriminator_round_trip(self):
        import tempfile
        from pathlib import Path
        model = btrk.TrackletDiscriminator()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracklet.pt"
            digest = btrk.save_discriminator(model, path)
            self.assertEqual(len(digest), 64)
            restored = btrk.load_discriminator(path)
            sample = torch.zeros((1, len(btrk.FEATURE_NAMES)))
            with torch.no_grad():
                self.assertAlmostEqual(float(model(sample)),
                                       float(restored(sample)),
                                       places=5)


if __name__ == "__main__":
    unittest.main()
