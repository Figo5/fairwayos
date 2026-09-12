"""TDD: the ball track must keep its RAW evidence, and never pass a seed off
as a prediction.

Two observability defects this pins down:

  * rejected frames dropped the model's raw coordinates and never recorded
    expected_dist_logit at all, so a rejection could not be diagnosed after the
    fact -- you could not tell "model pointed at the ball but the gate said no"
    from "model was lost".
  * the seed frame was emitted as an ordinary tracked frame, so any count of
    "frames tracked" silently included the frame a human/AI handed the model.

The rendering contract is unchanged and is asserted here too: point_xy stays
null on a rejected frame, so nothing stale can be drawn.
"""
import math
import unittest

from ghostcaddie.upload.adapters import ball_records, predictive_frames

SHA = "a" * 64


def _sig(z):
    return 1 / (1 + math.exp(-z))


class BallRecordTests(unittest.TestCase):
    # native 1280x720 -> model 512x320
    SX, SY = 512 / 1280, 320 / 720

    def _records(self, occ, exp, seed_frame=3062):
        order = [3062, 3063, 3064]
        tracks = [(400.0, 244.0), (405.0, 230.0), (410.0, 215.0)]  # resized px
        return ball_records(order=order, tracks_xy=tracks, occlusion=occ,
                            expected_dist=exp, sx=self.SX, sy=self.SY,
                            seed_frame=seed_frame, source_sha256=SHA)

    def test_raw_coordinates_are_kept_on_rejected_frames(self):
        recs = self._records(occ=[-5.0, 9.0, 9.0], exp=[-5.0, 9.0, 9.0])
        rejected = recs[1]
        self.assertFalse(rejected["visible"])
        self.assertIsNone(rejected["point_xy"], "rejected frames must not render")
        self.assertEqual(rejected["raw_point_xy"],
                         [round(405.0 / self.SX, 2), round(230.0 / self.SY, 2)])

    def test_both_logits_are_recorded_on_every_frame(self):
        recs = self._records(occ=[-5.0, 9.0, 1.5], exp=[-4.0, 8.0, 2.5])
        self.assertEqual([r["occlusion_logit"] for r in recs], [-5.0, 9.0, 1.5])
        self.assertEqual([r["expected_dist_logit"] for r in recs], [-4.0, 8.0, 2.5])

    def test_visibility_score_is_recorded_and_drives_the_gate(self):
        recs = self._records(occ=[-5.0, 9.0, 0.0], exp=[-5.0, 9.0, 0.0])
        for r, o, e in zip(recs, [-5.0, 9.0, 0.0], [-5.0, 9.0, 0.0]):
            self.assertAlmostEqual(r["visibility_score"],
                                   (1 - _sig(o)) * (1 - _sig(e)), places=4)
            self.assertEqual(r["visible"], r["visibility_score"] > 0.5)

    def test_seed_frame_is_labelled_a_seed_not_a_prediction(self):
        recs = self._records(occ=[-5.0, -5.0, -5.0], exp=[-5.0, -5.0, -5.0])
        self.assertTrue(recs[0]["is_seed"])
        self.assertEqual(recs[0]["state"], "seed")
        self.assertFalse(any(r["is_seed"] for r in recs[1:]))

    def test_predictive_frames_excludes_the_seed(self):
        recs = self._records(occ=[-5.0, -5.0, 9.0], exp=[-5.0, -5.0, 9.0])
        # 3 records: seed accepted, one accepted prediction, one rejection
        self.assertEqual(predictive_frames(recs), 1)

    def test_a_seed_frame_outside_the_window_marks_nothing(self):
        recs = self._records(occ=[-5.0] * 3, exp=[-5.0] * 3, seed_frame=9999)
        self.assertFalse(any(r["is_seed"] for r in recs))
        self.assertEqual(predictive_frames(recs), 3)

    def test_records_stay_source_bound_and_research_only(self):
        recs = self._records(occ=[-5.0] * 3, exp=[-5.0] * 3)
        for r in recs:
            self.assertEqual(r["source_sha256"], SHA)
            self.assertTrue(r["pseudo_label"])
            self.assertFalse(r["ground_truth"])
            self.assertFalse(r["production_eligible"])


if __name__ == "__main__":
    unittest.main()
