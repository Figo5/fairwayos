"""Metrics and detector tests.

The metrics decide what "working" means, so they get tested on the cases that are
easy to get wrong: silence where the ball is absent is a success, a prediction on
the wrong object is a false positive rather than a near miss, and a hit outside
tolerance does not count as a detection.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghostcaddie.video import ball_detector as bdet  # noqa: E402
from ghostcaddie.video import ball_eval_v2 as bev  # noqa: E402


def truth(index: int, *, visible: bool, x: float = 100.0, y: float = 100.0,
          radius: float = 4.0) -> dict:
    return {"frame_index": index, "visibility": "VISIBLE" if visible else "OCCLUDED",
            "x": x, "y": y, "radius_px": radius}


def state(index: int, *, observed: bool, x: float = 100.0, y: float = 100.0) -> dict:
    payload = {"frame_index": index, "state": "OBSERVED" if observed else "UNAVAILABLE"}
    if observed:
        payload["x"] = x
        payload["y"] = y
    return payload


class MetricTests(unittest.TestCase):
    def test_silence_where_the_ball_is_absent_is_a_success(self):
        metrics = bev.synthetic_sequence_metrics([state(0, observed=False)],
                                                 [truth(0, visible=False)],
                                                 width=1920, height=1080)
        self.assertEqual(metrics["correct_silence"], 1)
        self.assertEqual(metrics["false_positives"], 0)

    def test_a_prediction_where_the_ball_is_absent_is_a_false_positive(self):
        metrics = bev.synthetic_sequence_metrics([state(0, observed=True, x=50, y=50)],
                                                 [truth(0, visible=False)],
                                                 width=1920, height=1080)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["correct_silence"], 0)
        self.assertEqual(metrics["false_positive_rate_on_absent"], 1.0)

    def test_missing_the_ball_is_a_false_negative(self):
        metrics = bev.synthetic_sequence_metrics([state(0, observed=False)],
                                                 [truth(0, visible=True)],
                                                 width=1920, height=1080)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(metrics["recall"], 0.0)

    def test_a_hit_inside_tolerance_is_a_true_positive_with_its_error(self):
        metrics = bev.synthetic_sequence_metrics(
            [state(0, observed=True, x=112.0, y=100.0)],
            [truth(0, visible=True, x=100.0, y=100.0, radius=8.0)],
            width=1920, height=1080)
        self.assertEqual(metrics["true_positives"], 1)
        self.assertAlmostEqual(metrics["median_center_error_px"], 12.0, places=2)

    def test_a_wrong_object_is_never_a_near_miss(self):
        metrics = bev.synthetic_sequence_metrics(
            [state(0, observed=True, x=400.0, y=700.0)],
            [truth(0, visible=True, x=100.0, y=100.0)],
            width=1920, height=1080)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["true_positives"], 0)

    def test_tolerance_scales_with_the_ball(self):
        self.assertLess(bev.match_tolerance(3.0, 1920, 1080),
                        bev.match_tolerance(80.0, 1920, 1080))

    def test_aggregate_pools_counts_and_keeps_per_clip(self):
        first = bev.synthetic_sequence_metrics([state(0, observed=True)],
                                               [truth(0, visible=True)],
                                               width=1920, height=1080)
        second = bev.synthetic_sequence_metrics([state(0, observed=False)],
                                                [truth(0, visible=False)],
                                                width=1920, height=1080)
        pooled = bev.aggregate_synthetic_metrics([first, second])
        self.assertEqual(pooled["clips"], 2)
        self.assertEqual(pooled["true_positives"], 1)
        self.assertEqual(pooled["correct_silence"], 1)
        self.assertEqual(pooled["precision"], 1.0)
        self.assertEqual(len(pooled["per_clip"]), 2)

    def test_negative_sequences_measure_undeserved_confidence(self):
        report = bev.negative_sequence_report([
            {"accepted": True, "confidence": {"observed_frames": 5}},
            {"accepted": False, "confidence": {"observed_frames": 0}}])
        self.assertEqual(report["sequences"], 2)
        self.assertEqual(report["sequences_with_an_accepted_track"], 1)
        self.assertEqual(report["fabricated_track_rate"], 0.5)

    def test_real_video_diagnostics_never_present_themselves_as_accuracy(self):
        summaries = [{"frames": 4, "tiers": {"HIGH": 2, "MEDIUM": 1, "LOW": 1,
                                             "AMBIGUOUS": 0, "UNAVAILABLE": 0},
                      "diagnostics": {"BODY_LOCK": 1, "CLOTHING_LOCK": 0,
                                      "STATIC_SCENE_LOCK": 0, "CLUB_LOCK": 0,
                                      "FLAG_LOCK": 0}}]
        results = [{"accepted": False,
                    "confidence": {"observed_frames": 2, "span_frames": 4,
                                   "cycle_consistency_rate": 0.5,
                                   "body_lock_rate": 0.5, "static_lock_rate": 0.0}}]
        report = bev.real_video_diagnostics(summaries, results)
        self.assertIn("not accuracy", report["note"])
        self.assertEqual(report["body_lock_rate"], 0.5)
        self.assertEqual(report["consensus_rate"], 0.75)


@unittest.skipUnless(bdet.TORCH_AVAILABLE, "PyTorch is not importable here")
class DetectorTests(unittest.TestCase):
    def test_the_model_serialises_and_reloads_identically(self):
        import tempfile
        import torch
        model = bdet.BallDetector()
        crops = [np.random.default_rng(index).integers(
            0, 255, (bdet.CROP_SIZE, bdet.CROP_SIZE, 3), dtype=np.uint8)
            for index in range(3)]
        before = bdet.score_crops(model, crops, device=torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pt"
            digest = bdet.save_detector(model, path)
            self.assertEqual(len(digest), 64)
            reloaded = bdet.load_detector(path, device_override=torch.device("cpu"))
        after = bdet.score_crops(reloaded, crops, device=torch.device("cpu"))
        for left, right in zip(before, after):
            self.assertAlmostEqual(left["presence"], right["presence"], places=6)
            self.assertAlmostEqual(left["radius_px"], right["radius_px"], places=6)

    def test_weights_are_loaded_as_data_not_code(self):
        import inspect
        source = inspect.getsource(bdet.load_detector)
        self.assertIn("weights_only=True", source,
                      "a state dict must be loaded as data; pickled objects are code")

    def test_the_model_is_small(self):
        model = bdet.BallDetector()
        parameters = sum(p.numel() for p in model.parameters())
        self.assertLess(parameters, 2_000_000,
                        "a huge model is not a better answer to a 4 px ball")

    def test_scores_are_probabilities_with_a_centre_inside_the_crop(self):
        import torch
        model = bdet.BallDetector()
        results = bdet.score_crops(
            model, [np.zeros((bdet.CROP_SIZE, bdet.CROP_SIZE, 3), dtype=np.uint8)],
            device=torch.device("cpu"))
        self.assertEqual(len(results), 1)
        self.assertGreaterEqual(results[0]["presence"], 0.0)
        self.assertLessEqual(results[0]["presence"], 1.0)
        x, y = results[0]["center_in_crop"]
        self.assertTrue(0 <= x < bdet.CROP_SIZE and 0 <= y < bdet.CROP_SIZE)

    def test_subpixel_peak_is_not_quantised_to_whole_pixels(self):
        heat = np.zeros((16, 16), dtype=np.float32)
        heat[7, 8] = 1.0
        heat[7, 9] = 0.6
        heat[7, 7] = 0.4
        x, y = bdet._subpixel_peak(heat)
        self.assertNotEqual(x, round(x))


if __name__ == "__main__":
    unittest.main()
