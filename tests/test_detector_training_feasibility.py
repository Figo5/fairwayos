"""Can a detector actually be trained here, right now?

The answer must name the FIRST thing that is missing, in the order that
determines whether work can start at all: rights, then labels, then pixels, then
runtime. "No checkpoint exists" is not an answer -- it is the consequence.

Real state this encodes, all verified on disk:
  * ClubheadDB ships 10,180 YOLO-format clubhead boxes and a 67-swing metadata
    table LOCALLY, and not one pixel. Its image_path column points at the
    original author's desktop.
  * Its videos must be fetched from YouTube/Reddit, which is gated: source-level
    permission is not documented and the Reddit URLs expired in 2025.
  * torch 2.13 + torchvision 0.28 are installed, so the RUNTIME is not the
    blocker and should never be reported as one.
"""
import os
import unittest

from ghostcaddie.training.feasibility import (
    DatasetAsset, TrainingFeasibility, assess, CLUBHEADDB, first_blocker,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BlockerOrderTests(unittest.TestCase):
    def _asset(self, **kw):
        base = dict(name="t", license_declared="CC-BY-4.0", rights_cleared=True,
                    label_count=100, label_path=__file__, pixel_count=10,
                    pixel_root=REPO, runtime_available=True,
                    disjoint_split_field="golfer")
        base.update(kw)
        return DatasetAsset(**base)

    def test_uncleared_rights_block_before_anything_else(self):
        a = self._asset(rights_cleared=False, label_count=0, pixel_count=0,
                        runtime_available=False)
        self.assertEqual(first_blocker(a).kind, "rights")

    def test_missing_labels_block_before_pixels(self):
        a = self._asset(label_count=0, pixel_count=0)
        self.assertEqual(first_blocker(a).kind, "labels")

    def test_missing_pixels_are_reported_even_when_labels_are_present(self):
        """The ClubheadDB case exactly."""
        a = self._asset(pixel_count=0)
        b = first_blocker(a)
        self.assertEqual(b.kind, "pixels")
        self.assertIn("label", b.detail.lower())

    def test_runtime_is_only_blamed_when_it_is_actually_missing(self):
        a = self._asset(runtime_available=False)
        self.assertEqual(first_blocker(a).kind, "runtime")

    def test_a_complete_asset_has_no_blocker(self):
        self.assertIsNone(first_blocker(self._asset()))

    def test_a_split_field_is_required_so_splits_cannot_leak(self):
        a = self._asset(disjoint_split_field="")
        self.assertEqual(first_blocker(a).kind, "split")


class ClubheadDbTests(unittest.TestCase):
    def setUp(self):
        self.a = CLUBHEADDB(REPO)

    @unittest.skipUnless(
        os.path.exists(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "out/clubheaddb_recon/clubhead_db-1.0.1")),
        "ClubheadDB package not present")
    def test_labels_are_present_locally(self):
        self.assertGreater(self.a.label_count, 10000)
        self.assertTrue(os.path.exists(self.a.label_path))

    @unittest.skipUnless(
        os.path.exists(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "out/clubheaddb_recon/clubhead_db-1.0.1")),
        "ClubheadDB package not present")
    def test_no_pixels_ship_with_it(self):
        self.assertEqual(self.a.pixel_count, 0)

    @unittest.skipUnless(
        os.path.exists(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "out/clubheaddb_recon/clubhead_db-1.0.1")),
        "ClubheadDB package not present")
    def test_the_reported_blocker_is_the_pixels_not_the_checkpoint(self):
        b = first_blocker(self.a)
        self.assertIsNotNone(b)
        self.assertIn(b.kind, ("rights", "pixels"))
        self.assertNotIn("checkpoint", b.detail.lower())


class AssessmentTests(unittest.TestCase):
    def test_assess_reports_every_asset_and_refuses_to_start(self):
        r = assess(REPO)
        self.assertIsInstance(r, TrainingFeasibility)
        self.assertFalse(r.can_train_now)
        self.assertTrue(r.blockers)
        for b in r.blockers:
            self.assertTrue(b.detail and b.remedy, b.kind)

    def test_the_runtime_is_not_reported_as_a_blocker_when_torch_exists(self):
        r = assess(REPO)
        kinds = {b.kind for b in r.blockers}
        if r.runtime_present:
            self.assertNotIn("runtime", kinds,
                             "torch is installed; runtime must not be blamed")


if __name__ == "__main__":
    unittest.main()


class BallDetectorFactsTests(unittest.TestCase):
    """The golf-ball ONNX detector: what it actually is, and what it actually did.

    Corrects two overstatements. It is NOT license-undeclared: the ONNX metadata
    declares AGPL-3.0 and names={0: golf_ball}, output shape [1,5,8400]. And it
    did NOT produce zero ball detections: of 8 proximity hits, six are the
    stationary ball at address (f3055-f3060) and TWO -- f3075 and f3079 -- were
    visually confirmed against clean native crops as the real moving ball.

    None of that makes it continuous tracking: 2 moving detections across a
    51-frame interval, against 218 non-ball predictions, is not a track.
    """

    def setUp(self):
        from ghostcaddie.training.feasibility import GOLFBALL_ONNX
        self.a = GOLFBALL_ONNX(REPO)

    def test_the_license_is_recorded_as_declared_not_as_unknown(self):
        self.assertEqual(self.a.license_declared, "AGPL-3.0")

    def test_it_is_recognised_as_a_golf_ball_class_detector(self):
        self.assertIn("golf_ball", self.a.notes)

    def test_the_two_verified_moving_detections_are_preserved(self):
        from ghostcaddie.training.feasibility import BALL_VERIFIED_MOVING_FRAMES
        self.assertEqual(BALL_VERIFIED_MOVING_FRAMES, (3075, 3079))
        self.assertIn("3075", self.a.notes)
        self.assertIn("3079", self.a.notes)

    def test_the_stationary_address_hits_are_not_counted_as_tracking(self):
        from ghostcaddie.training.feasibility import BALL_STATIONARY_HIT_FRAMES
        self.assertEqual(len(BALL_STATIONARY_HIT_FRAMES), 6)
        self.assertEqual(min(BALL_STATIONARY_HIT_FRAMES), 3055)
        self.assertEqual(max(BALL_STATIONARY_HIT_FRAMES), 3060)

    def test_it_is_never_described_as_continuous_tracking(self):
        self.assertIn("not continuous tracking", self.a.notes.lower())

    def test_its_blocker_is_dataset_rights_not_a_missing_checkpoint(self):
        from ghostcaddie.training.feasibility import first_blocker
        b = first_blocker(self.a)
        self.assertIsNotNone(b)
        self.assertEqual(b.kind, "rights")
        self.assertIn("dataset", b.detail.lower())
