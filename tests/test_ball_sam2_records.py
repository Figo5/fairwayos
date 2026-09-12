"""TDD for the pivot ball route: SAM2.1 video segmentation instead of TAPIR.

Why a pivot and not more tuning: the independent review established that the
TAPIR raw coordinates themselves drift away from the sparse AI reference points
(about 39/101/117 px at native f3068/f3080/f3090 full-frame), so the failure is
not an over-strict visibility gate that could be relaxed. The point-tracker route
is treated as falsified on this source.

SAM2.1 tiny is a genuinely different model class (memory-attention video object
segmentation, not point tracking), it is already audited on the repo's safe load
path (weights_only=True, guarded Hiera weights_path), and it already produces
real clubhead masks on this exact source -- so it is the honest next model to
try, with no new download and no new heuristic.

These tests pin the RECORD contract, which is what a renderer and a reviewer
depend on. They deliberately do not assert that SAM2 finds the ball: whether it
does is an empirical question answered by an actual bounded run.
"""
import unittest

import numpy as np

from ghostcaddie.upload.adapters import (
    AdapterUnavailable, BallSam2Adapter, sam2_ball_record, seed_box_from_point,
)

SHA = "b" * 64


class SeedBoxTests(unittest.TestCase):
    def test_box_is_derived_from_the_reviewed_point_and_its_uncertainty(self):
        """No new click: the box is the reviewed point +/- its stated radius."""
        self.assertEqual(seed_box_from_point([836.0, 550.0], 7),
                         [829.0, 543.0, 843.0, 557.0])

    def test_a_zero_radius_seed_is_refused_rather_than_guessed(self):
        with self.assertRaises(ValueError):
            seed_box_from_point([836.0, 550.0], 0)


class RecordTests(unittest.TestCase):
    def _mask(self, x0, y0, x1, y1, shape=(720, 1280)):
        m = np.zeros(shape, bool); m[y0:y1, x0:x1] = True; return m

    def test_a_found_mask_reports_its_centroid_and_area(self):
        r = sam2_ball_record(3063, self._mask(830, 544, 840, 554), SHA, seed_frame=3062)
        self.assertTrue(r["visible"])
        self.assertEqual(r["point_xy"], [834.5, 548.5])
        self.assertEqual(r["area_px"], 100)
        self.assertEqual(r["bbox_xyxy"], [830, 544, 840, 554])

    def test_an_empty_mask_is_a_gap_never_a_stale_point(self):
        r = sam2_ball_record(3064, self._mask(0, 0, 0, 0), SHA, seed_frame=3062)
        self.assertFalse(r["visible"])
        self.assertIsNone(r["point_xy"])
        self.assertIsNone(r["bbox_xyxy"])
        self.assertEqual(r["state"], "unavailable")
        self.assertEqual(r["area_px"], 0)

    def test_the_seed_frame_is_not_counted_as_a_prediction(self):
        seed = sam2_ball_record(3062, self._mask(829, 543, 843, 557), SHA, seed_frame=3062)
        pred = sam2_ball_record(3063, self._mask(830, 544, 840, 554), SHA, seed_frame=3062)
        self.assertTrue(seed["is_seed"])
        self.assertEqual(seed["state"], "seed")
        self.assertFalse(pred["is_seed"])
        from ghostcaddie.upload.adapters import predictive_frames
        self.assertEqual(predictive_frames([seed, pred]), 1)

    def test_a_mask_far_too_large_for_a_golf_ball_is_flagged_not_silently_drawn(self):
        """A memory-attention tracker that latches onto turf or sky must be
        visible as a failure in the record, not rendered as a ball."""
        r = sam2_ball_record(3065, self._mask(100, 100, 500, 500), SHA, seed_frame=3062)
        self.assertFalse(r["visible"])
        self.assertEqual(r["state"], "implausible_area")
        self.assertIsNone(r["point_xy"])
        self.assertEqual(r["area_px"], 160000)

    def test_records_are_source_bound_and_research_only(self):
        r = sam2_ball_record(3063, self._mask(830, 544, 840, 554), SHA, seed_frame=3062)
        self.assertEqual(r["source_sha256"], SHA)
        self.assertEqual(r["initialization"], "assisted")
        self.assertTrue(r["pseudo_label"])
        self.assertFalse(r["ground_truth"])
        self.assertFalse(r["production_eligible"])


class SeedRequirementTests(unittest.TestCase):
    def test_running_without_a_reviewed_seed_raises(self):
        a = BallSam2Adapter()
        cap = a.capability()
        if not cap.available:
            self.skipTest(f"SAM2 not runnable here: {cap.reason}")
        with self.assertRaises(AdapterUnavailable):
            a.run_frames([], source_sha256=SHA, seed=None)


if __name__ == "__main__":
    unittest.main()


class SeedFrameMustBePresentTests(unittest.TestCase):
    """A seed for a frame that is not in the decoded window used to fall back to
    index 0, which silently seeds a DIFFERENT frame than the operator chose.

    That is worse than failing: the operator reviewed frame N, the model was
    initialised on frame 0, and the output still claims the seed was honoured.
    """

    def test_seed_index_refuses_a_frame_outside_the_decoded_order(self):
        from ghostcaddie.upload.adapters import seed_index, AdapterUnavailable
        with self.assertRaises(AdapterUnavailable) as e:
            seed_index([3062, 3063, 3064], 9999)
        self.assertIn("9999", str(e.exception))

    def test_seed_index_returns_the_exact_position_when_present(self):
        from ghostcaddie.upload.adapters import seed_index
        self.assertEqual(seed_index([3062, 3063, 3064], 3063), 1)
