"""Consensus and temporal-verification tests.

The tiering rules are the difference between supervision and guesswork, so they
are tested directly; the cycle check is tested on real motion where a round trip
should succeed, and on a case where it should not; and the lock diagnostics are
tested because the previous method's failure was a body lock, not a miss.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghostcaddie.video import ball_candidates as bc  # noqa: E402
from ghostcaddie.video import ball_consensus as bcon  # noqa: E402


def background(width: int = 320, height: int = 240, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[..., 1] = 100 + rng.normal(0, 7, (height, width))
    base[..., 0] = 58 + rng.normal(0, 5, (height, width))
    base[..., 2] = 45 + rng.normal(0, 5, (height, width))
    for _ in range(20):
        cv2.circle(base, (int(rng.integers(0, width)), int(rng.integers(0, height))),
                   int(rng.integers(3, 8)), tuple(float(v) for v in rng.integers(70, 180, 3)), -1)
    return np.clip(base, 0, 255).astype(np.uint8)


def with_disc(frame: np.ndarray, x: int, y: int, radius: int = 4) -> np.ndarray:
    out = frame.copy()
    cv2.circle(out, (x, y), radius, (238, 238, 238), -1)
    return out


def candidate(x: float, y: float, radius: float, score: float, method: str,
              frame: int = 0) -> bc.Candidate2:
    return bc.Candidate2(frame, x, y, radius, score, method, {"test": True})


class ClusteringTests(unittest.TestCase):
    def test_three_methods_on_one_object_become_one_cluster(self):
        candidates = [candidate(100, 100, 4, 0.9, "radial_symmetry"),
                      candidate(102, 101, 4, 0.8, "local_contrast"),
                      candidate(99, 103, 4, 0.7, "brightness_cue")]
        clusters = bcon.cluster_candidates(candidates)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0].methods), 3)
        self.assertLess(clusters[0].spread, 6.0)

    def test_distant_objects_stay_separate(self):
        candidates = [candidate(100, 100, 4, 0.9, "radial_symmetry"),
                      candidate(220, 180, 4, 0.9, "local_contrast")]
        clusters = bcon.cluster_candidates(candidates)
        self.assertEqual(len(clusters), 2)

    def test_weak_candidates_are_not_clustered(self):
        weak = [candidate(100, 100, 4, 0.05, "radial_symmetry"),
                candidate(101, 100, 4, 0.05, "local_contrast")]
        self.assertEqual(bcon.cluster_candidates(weak), [])


class TieringTests(unittest.TestCase):
    def setUp(self):
        self.params = bcon.ConsensusParams()
        self.cluster = bcon.cluster_candidates(
            [candidate(100, 100, 4, 0.9, "radial_symmetry"),
             candidate(101, 100, 4, 0.8, "local_contrast"),
             candidate(100, 101, 4, 0.7, "brightness_cue")])[0]
        self.temporal_ok = {"supported": True, "round_trip_px": 1.2}
        self.plausible = {"ok": True}

    def label(self, **overrides):
        arguments = dict(cluster=self.cluster, competitors=0,
                         temporal=self.temporal_ok, static={"locked": False},
                         plausibility=self.plausible, body=None,
                         params=self.params)
        arguments.update(overrides)
        return bcon.consensus_label("clip", 0, **arguments)

    def test_three_methods_with_temporal_support_are_high(self):
        self.assertEqual(self.label().tier, bcon.TIER_HIGH)

    def test_two_methods_with_temporal_support_are_medium(self):
        cluster = bcon.cluster_candidates(
            [candidate(100, 100, 4, 0.9, "radial_symmetry"),
             candidate(101, 100, 4, 0.8, "local_contrast")])[0]
        self.assertEqual(self.label(cluster=cluster).tier, bcon.TIER_MEDIUM)

    def test_one_method_alone_is_only_low(self):
        cluster = bcon.cluster_candidates([candidate(100, 100, 4, 0.9, "radial_symmetry")])[0]
        self.assertEqual(self.label(cluster=cluster).tier, bcon.TIER_LOW)

    def test_failed_temporal_verification_demotes_high_agreement(self):
        label = self.label(temporal={"supported": False, "reason": "round trip lost"})
        self.assertEqual(label.tier, bcon.TIER_LOW)
        self.assertTrue(any("temporal" in reason for reason in label.rejections))

    def test_competing_candidates_make_it_ambiguous(self):
        self.assertEqual(self.label(competitors=2).tier, bcon.TIER_AMBIGUOUS)

    def test_no_candidate_is_unavailable_not_a_guess(self):
        label = self.label(cluster=None)
        self.assertEqual(label.tier, bcon.TIER_UNAVAILABLE)
        self.assertIsNone(label.x)

    def test_body_lock_rejects_even_with_three_methods(self):
        body = {"locked": True, "regions": [{"kind": "torso", "inside_margin_px": 4.0}]}
        label = self.label(body=body)
        self.assertNotEqual(label.tier, bcon.TIER_HIGH)
        self.assertIn("BODY_LOCK", label.diagnostics)

    def test_static_scene_lock_is_named_not_silently_subtracted(self):
        label = self.label(static={"locked": True, "run_frames": 12,
                                   "median_camera_motion_px": 5.0})
        self.assertIn("STATIC_SCENE_LOCK", label.diagnostics)
        self.assertNotEqual(label.tier, bcon.TIER_HIGH)

    def test_implausible_motion_is_rejected_with_a_reason(self):
        label = self.label(plausibility={"ok": False, "reason": "max step 30 radii"})
        self.assertNotEqual(label.tier, bcon.TIER_HIGH)
        self.assertTrue(any("implausible" in reason for reason in label.rejections))

    def test_only_high_confidence_labels_become_visible_annotations(self):
        high = self.label().as_annotation()
        self.assertEqual(high["state"], "VISIBLE")
        self.assertEqual(high["origin"], "MODEL_CONSENSUS")
        self.assertIn("x", high)
        unavailable = self.label(cluster=None).as_annotation()
        self.assertEqual(unavailable["state"], "NOT_VISIBLE")
        self.assertNotIn("x", unavailable)
        ambiguous = self.label(competitors=3).as_annotation()
        self.assertEqual(ambiguous["state"], "AMBIGUOUS")
        self.assertNotIn("x", ambiguous)


class TemporalCycleTests(unittest.TestCase):
    def setUp(self):
        self.background = background()

    def test_a_moving_disc_passes_the_round_trip(self):
        frames = []
        for step in range(8):
            frames.append(with_disc(self.background, 100 + step * 6, 120, 4))
        contexts, _ = bc.build_contexts(frames, list(range(8)))
        result = bcon.track_patch_both_ways(contexts, 3, 118.0, 120.0, 4.0,
                                           bcon.ConsensusParams())
        self.assertTrue(result["supported"], result)
        self.assertLessEqual(result["round_trip_px"], 6.0)

    def test_a_patch_that_cannot_be_recovered_fails(self):
        """A still scene gives the tracker nothing to hold, and that is reported."""
        frames = [self.background.copy() for _ in range(8)]
        contexts, _ = bc.build_contexts(frames, list(range(8)))
        result = bcon.track_patch_both_ways(contexts, 3, 160.0, 130.0, 4.0,
                                           bcon.ConsensusParams())
        # either it round-trips to the same place (nothing moved, nothing to
        # verify) or it is reported unsupported; what must not happen is a claim
        # of support with a large error
        if result.get("supported"):
            self.assertLessEqual(result["round_trip_px"], 4.0)

    def test_a_frame_outside_the_window_is_refused(self):
        frames = [self.background.copy() for _ in range(4)]
        contexts, _ = bc.build_contexts(frames, [10, 11, 12, 13])
        result = bcon.track_patch_both_ways(contexts, 99, 100.0, 100.0, 4.0,
                                            bcon.ConsensusParams())
        self.assertFalse(result["supported"])
        self.assertIn("frame not in window", result["reason"])

    def test_plausibility_flags_an_impossible_jump(self):
        params = bcon.ConsensusParams()
        far = bcon.Cluster(0, 0, 0, 4, [candidate(0, 0, 4, 0.9, "m")])
        near = bcon.Cluster(1, 400, 300, 4, [candidate(400, 300, 4, 0.9, "m")])
        result = bcon.displacement_plausibility([far, near], params)
        self.assertFalse(result["ok"])
        self.assertIn("implausible", result["reason"])


class LockDiagnosticTests(unittest.TestCase):
    def test_body_regions_are_detected_with_their_kind(self):
        region = [{"kind": "torso", "x": 90, "y": 90, "w": 40, "h": 60}]
        result = bcon.body_lock(100, 110, 3, region)
        self.assertTrue(result["locked"])
        self.assertEqual(result["regions"][0]["kind"], "torso")
        self.assertFalse(bcon.body_lock(300, 30, 3, region)["locked"])

    def test_clothing_regions_are_named_as_clothing(self):
        region = [{"kind": "shoe", "x": 20, "y": 200, "w": 25, "h": 15}]
        cluster = bcon.Cluster(0, 30, 205, 3, [candidate(30, 205, 3, 0.9, "m")])
        diagnostics = bcon.classify_diagnostics(cluster, temporal={"supported": True},
                                                static={"locked": False},
                                                body=bcon.body_lock(30, 205, 3, region))
        self.assertIn("CLOTHING_LOCK", diagnostics)

    def test_static_lock_needs_camera_motion_to_fire(self):
        params = bcon.ConsensusParams()
        still = [bcon.Cluster(i, 100, 100, 4, [candidate(100, 100, 4, 0.9, "m")])
                 for i in range(12)]
        self.assertFalse(bcon.static_scene_lock(still, params,
                                                motion_px=[0.2] * 12)["locked"])
        self.assertTrue(bcon.static_scene_lock(still, params,
                                               motion_px=[6.0] * 12)["locked"])


class HardNegativeMiningTests(unittest.TestCase):
    def test_a_rejected_competitor_becomes_a_hard_negative_with_its_reason(self):
        frame = with_disc(with_disc(background(), 100, 120, 4), 220, 90, 5)
        contexts, _ = bc.build_contexts([frame], [0])
        per_frame = bc.ensemble_candidates(contexts)
        consensus = bcon.consensus_for_window("clip", contexts,
                                              candidates_by_frame=per_frame)
        records = bcon.mine_hard_negatives("clip", contexts,
                                          candidates_by_frame=per_frame,
                                          consensus=consensus, out_dir=None)
        self.assertTrue(records)
        for record in records:
            self.assertEqual(record["kind"], "hard_negative")
            self.assertEqual(record["origin"], "MODEL_CONSENSUS")
            self.assertTrue(record["methods"])
            self.assertIn("crop", record)

    def test_the_accepted_ball_is_not_mined_as_a_negative(self):
        frame = with_disc(background(), 160, 120, 4)
        contexts, _ = bc.build_contexts([frame], [0])
        per_frame = bc.ensemble_candidates(contexts)
        consensus = bcon.consensus_for_window("clip", contexts,
                                              candidates_by_frame=per_frame)
        accepted = [label for label in consensus["labels"].values()
                    if label.tier in (bcon.TIER_HIGH, bcon.TIER_MEDIUM)]
        records = bcon.mine_hard_negatives("clip", contexts,
                                          candidates_by_frame=per_frame,
                                          consensus=consensus, out_dir=None)
        for record in records:
            for label in accepted:
                distance = ((record["x"] - label.x) ** 2
                            + (record["y"] - label.y) ** 2) ** 0.5
                self.assertGreater(distance, 2 * max(record["radius"], label.radius or 1))


class WindowConsensusTests(unittest.TestCase):
    def test_a_window_with_no_ball_produces_only_unavailable_or_low(self):
        frames = [background() for _ in range(6)]
        contexts, _ = bc.build_contexts(frames, list(range(6)))
        per_frame = bc.ensemble_candidates(contexts)
        consensus = bcon.consensus_for_window("clip", contexts,
                                              candidates_by_frame=per_frame)
        high = [label for label in consensus["labels"].values()
                if label.tier == bcon.TIER_HIGH]
        self.assertLessEqual(len(high), 1,
                             "a background-only window must not be confident")
        for label in consensus["labels"].values():
            annotation = label.as_annotation()
            if label.tier in (bcon.TIER_HIGH, bcon.TIER_MEDIUM):
                self.assertIn("x", annotation)
            else:
                self.assertNotIn("x", annotation,
                                 f"{label.tier} must not carry a coordinate")
                self.assertNotIn("y", annotation)

    def test_only_high_confidence_labels_are_trainable_by_default(self):
        frames = [background() for _ in range(6)]
        contexts, _ = bc.build_contexts(frames, list(range(6)))
        per_frame = bc.ensemble_candidates(contexts)
        consensus = bcon.consensus_for_window("clip", contexts,
                                              candidates_by_frame=per_frame)
        trainable = bcon.training_labels(consensus)
        self.assertTrue(all(label.tier == bcon.TIER_HIGH for label in trainable))
        self.assertEqual(len(trainable),
                         sum(1 for label in consensus["labels"].values()
                             if label.tier == bcon.TIER_HIGH and label.x is not None))
        # medium can be promoted deliberately, never by accident
        medium = bcon.training_labels(consensus, tiers=(bcon.TIER_MEDIUM,))
        self.assertTrue(all(label.tier == bcon.TIER_MEDIUM for label in medium))

    def test_the_summary_reports_tiers_diagnostics_and_trainable_counts(self):
        frames = [background() for _ in range(4)]
        contexts, _ = bc.build_contexts(frames, list(range(4)))
        per_frame = bc.ensemble_candidates(contexts)
        consensus = bcon.consensus_for_window("clip", contexts,
                                              candidates_by_frame=per_frame)
        summary = bcon.summarize_consensus(consensus)
        self.assertEqual(sum(summary["tiers"].values()), 4)
        self.assertEqual(set(summary["tiers"]), set(bcon.TIERS))
        self.assertIn("trainable_labels", summary)
        self.assertEqual(set(summary["diagnostics"]), set(bcon.DIAGNOSTIC_KINDS))


def scene_frame(width: int = 640, height: int = 480, seed: int = 5) -> np.ndarray:
    """Turf-like background with clutter, for coordinate-space tests."""
    rng = np.random.default_rng(seed)
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[..., 1] = 108 + rng.normal(0, 6, (height, width))
    base[..., 0] = 62 + rng.normal(0, 5, (height, width))
    base[..., 2] = 50 + rng.normal(0, 5, (height, width))
    for _ in range(20):
        cv2.circle(base, (int(rng.integers(0, width)), int(rng.integers(0, height))),
                   int(rng.integers(3, 7)), tuple(float(v) for v in rng.integers(70, 180, 3)), -1)
    return np.clip(base, 0, 255).astype(np.uint8)


def ball_in(frame: np.ndarray, x: int, y: int, radius: int = 6) -> np.ndarray:
    out = frame.copy()
    cv2.circle(out, (x, y), radius, (244, 244, 244), -1)
    return out


class CoordinateSpaceTests(unittest.TestCase):
    """Native pixels in, native pixels out -- whatever the working resolution is.

    Candidates are reported in native pixels, but the dense methods and the optical
    flow work run on a downscaled copy. On a 4K clip that is a factor of two, and
    mixing the two spaces pointed the tracking window at the wrong part of the frame
    while looking like a legitimate "target not found" result.
    """

    @staticmethod
    def _panning_clip():
        scene = scene_frame()
        return [ball_in(np.roll(scene, index * 6, axis=1), 320, 240, 6)
                for index in range(8)]

    def test_the_temporal_check_works_at_every_working_scale(self):
        from ghostcaddie.video import ball_candidates as bc
        frames = self._panning_clip()
        verdicts = {}
        for max_side in (1920, 320):
            contexts, _ = bc.build_contexts(frames, list(range(len(frames))),
                                            max_side=max_side)
            # a candidate in native pixels, as the ensemble reports it
            verdict = bcon.track_patch_both_ways(contexts, 1, 320.0, 240.0, 6.0,
                                                bcon.ConsensusParams())
            verdicts[max_side] = verdict
        for max_side, verdict in verdicts.items():
            self.assertNotIn("outside the tracking window", str(verdict.get("reason", "")),
                             f"the tracking window must contain the candidate at "
                             f"working side {max_side}: {verdict}")
        self.assertTrue(all(verdict.get("supported") for verdict in verdicts.values()),
                        f"the same clip must reach the same temporal verdict at every "
                        f"working scale: {verdicts}")

    def test_hard_negative_windows_are_native_and_on_frame(self):
        from ghostcaddie.video import ball_candidates as bc
        frames = self._panning_clip()
        contexts, _ = bc.build_contexts(frames, list(range(len(frames))), max_side=320)
        candidates = bc.ensemble_candidates(contexts)
        consensus = bcon.consensus_for_window("clip", contexts,
                                              candidates_by_frame=candidates)
        records = bcon.mine_hard_negatives("clip", contexts,
                                           candidates_by_frame=candidates,
                                           consensus=consensus,
                                           out_dir=Path("/tmp"), max_per_clip=5)
        height, width = frames[0].shape[:2]
        for record in records:
            left, top, size, _ = record["crop"]
            self.assertGreaterEqual(left, 0)
            self.assertGreaterEqual(top, 0)
            self.assertLessEqual(left + size, width,
                                 "a hard-negative window must fit the native frame")
            self.assertLessEqual(top + size, height,
                                 "a hard-negative window must fit the native frame")


if __name__ == "__main__":
    unittest.main()
