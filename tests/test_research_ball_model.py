import unittest

from ghostcaddie.video.research_ball_model import (
    ResearchBallMultiHypothesisTrack,
    ResearchBallTrack,
    normalize_box,
    normalize_point,
)


def candidate(x, y, confidence=0.8):
    return {"center": (float(x), float(y)), "confidence": confidence, "box": [x - 3, y - 3, x + 3, y + 3]}


class TestResearchBallModel(unittest.TestCase):
    def test_scales_normalized_coordinates_to_pixels(self):
        self.assertEqual(normalize_point((0.25, 0.5), 600, 480), (150.0, 240.0))
        self.assertEqual(normalize_box((0.1, 0.2, 0.2, 0.3), 600, 480), (60.0, 96.0, 120.0, 144.0))

    def test_implausible_frame_filling_box_is_rejected(self):
        # The local ball model emits full-frame boxes on some clips; the box
        # center of such a hallucination must never seed a ball track.
        with self.assertRaises(ValueError):
            normalize_box((36.7, 0.0, 1920.0, 1080.0), 1920, 1080)
        with self.assertRaises(ValueError):
            normalize_box((42.9, 59.7, 1917.8, 1080.0), 1920, 1080)

    def test_realistic_ball_boxes_pass_plausibility(self):
        # A real ball at 1920x1080 is far smaller than a quarter of any
        # dimension and a tiny fraction of the frame area.
        self.assertEqual(
            normalize_box((1092.6, 813.4, 1156.9, 880.3), 1920, 1080),
            (1092.6, 813.4, 1156.9, 880.3),
        )
        self.assertEqual(
            normalize_box((30.0, 520.0, 90.0, 580.0), 1920, 1080),
            (30.0, 520.0, 90.0, 580.0),
        )

    def test_rejects_invalid_dimensions_or_shape(self):
        with self.assertRaises(ValueError):
            normalize_point((0.2, 0.3), 0, 480)
        with self.assertRaises(ValueError):
            normalize_box((0.1, 0.2, 0.3), 600, 480)

    def test_rejects_nonfinite_normalized_values(self):
        with self.assertRaises(ValueError):
            normalize_point((float("nan"), 0.3), 600, 480)
        with self.assertRaises(ValueError):
            normalize_box((0.1, 0.2, float("inf"), 0.3), 600, 480)

    def test_rejects_degenerate_or_out_of_frame_boxes(self):
        for box in (
            (100.0, 200.0, 100.0, 210.0),
            (100.0, 200.0, 110.0, 200.0),
            (-1.0, 200.0, 10.0, 210.0),
            (100.0, 470.0, 110.0, 481.0),
        ):
            with self.subTest(box=box), self.assertRaises(ValueError):
                normalize_box(box, 600, 480)

    def test_adapter_skips_collapsed_boundary_boxes_but_keeps_valid_pt_candidate(self):
        from ghostcaddie.video import ai_demo

        model = _FakeBallModel(boxes=[
            (0.99, [0.0, 479.0, 600.0, 480.0]),
            (0.65, [109.0, 209.0, 121.0, 221.0]),
        ])
        tracker = _RecordingTracker()

        ball, warning = ai_demo._ball_observation(
            model, tracker, _NumpyishFrame(600, 480), 600, 480
        )

        self.assertIsNone(warning)
        self.assertEqual(len(tracker.seen), 1)
        self.assertEqual(tracker.seen[0]["box"], [109.0, 209.0, 121.0, 221.0])
        self.assertEqual(ball["point"], {"x": 115.0, "y": 215.0})
        self.assertTrue(ball["research_only"])
        self.assertFalse(ball["ground_truth"])
        self.assertFalse(ball["production_eligible"])


class TestBallModelBoxSkipping(unittest.TestCase):
    def test_implausible_box_skipped_and_real_box_kept(self):
        """One frame-filling hallucination must not discard a real detection
        emitted in the same frame; it is skipped individually."""
        try:
            import numpy as np  # noqa: F401
        except ImportError:
            self.skipTest("optional numeric stack unavailable")
        from ghostcaddie.video import ai_demo
        model = _FakeBallModel(boxes=[(0.686, [36.7, 0.0, 1920.0, 1080.0]),
                                      (0.437, [1092.6, 813.4, 1156.9, 880.3])])
        tracker = _RecordingTracker()
        ball, warning = ai_demo._ball_observation(model, tracker, _NumpyishFrame(1920, 1080), 1920, 1080)
        self.assertIsNone(warning)
        self.assertIsNotNone(ball)
        # The kept point is the real box center, not the frame center.
        self.assertAlmostEqual(ball["point"]["x"], 1124.75, places=2)
        self.assertAlmostEqual(ball["point"]["y"], 846.85, places=2)


class _FakeBallModel:
    def __init__(self, boxes):
        self._boxes = boxes

    def __call__(self, frame, verbose=False):
        class _XYXY(list):
            def tolist(self):
                return list(self)

        class _Box:
            def __init__(self, conf, xyxy):
                self.conf = [conf]
                self.xyxy = [_XYXY(xyxy)]

        class _Result:
            def __init__(self, boxes):
                self.boxes = boxes

        return [_Result([_Box(conf, xyxy) for conf, xyxy in self._boxes])]


class _RecordingTracker:
    def __init__(self):
        self.seen = []

    def update(self, candidates):
        self.seen.extend(candidates)
        if candidates:
            best = max(candidates, key=lambda c: c["confidence"])
            return {"state": "observed", "point": {"x": best["center"][0], "y": best["center"][1]},
                    "confidence": best["confidence"]}
        return {"state": "unavailable"}


class _NumpyishFrame:
    """Minimal stand-in satisfying _ball_observation's width/height use."""
    def __init__(self, width, height):
        self.width = width
        self.height = height


class TestResearchBallTrack(unittest.TestCase):
    def test_confidence_semantics_are_detection_quality_not_identity(self):
        result = ResearchBallTrack().update([candidate(100, 200, 0.8)])
        self.assertEqual(result["confidence_semantics"], "detection_quality_not_identity")

    def test_initializes_and_predicts_forward_motion(self):
        track = ResearchBallTrack(min_confidence=0.35, max_misses=2)
        first = track.update([candidate(100, 200, 0.8)])
        second = track.update([candidate(110, 200, 0.8)])
        third = track.update([candidate(120, 200, 0.8)])
        self.assertEqual(first["state"], "observed")
        self.assertEqual(second["state"], "observed")
        self.assertEqual(third["state"], "observed")
        self.assertEqual(third["point"], {"x": 120.0, "y": 200.0})

    def test_rejects_motion_jump_and_terminates_after_misses(self):
        track = ResearchBallTrack(min_confidence=0.35, max_step=25.0, max_misses=2)
        track.update([candidate(100, 200)])
        track.update([candidate(110, 200)])
        rejected = track.update([candidate(400, 100)])
        terminated = track.update([candidate(401, 100)])
        self.assertEqual(rejected["state"], "predicted")
        self.assertEqual(rejected["warning"], "motion_constraint_rejected")
        self.assertEqual(terminated["state"], "terminated")
        self.assertIsNone(terminated["point"])

    def test_confidence_decays_during_short_occlusion(self):
        track = ResearchBallTrack(min_confidence=0.35, confidence_decay=0.2, max_misses=3)
        track.update([candidate(100, 200, 0.8)])
        missed = track.update([])
        self.assertEqual(missed["state"], "predicted")
        self.assertAlmostEqual(missed["confidence"], 0.6)
        self.assertEqual(missed["warning"], "confidence_decayed")

    def test_low_confidence_candidate_does_not_restart_terminated_track(self):
        track = ResearchBallTrack(min_confidence=0.5, max_misses=1)
        track.update([candidate(100, 200, 0.8)])
        track.update([])
        terminated = track.update([candidate(101, 200, 0.4)])
        self.assertEqual(terminated["state"], "terminated")
        self.assertIsNone(terminated["point"])

    def test_rejects_invalid_track_limits(self):
        with self.assertRaises(ValueError):
            ResearchBallTrack(max_step=float("nan"))
        with self.assertRaises(ValueError):
            ResearchBallMultiHypothesisTrack(max_step=float("inf"))

    def test_skips_malformed_and_nonfinite_candidates(self):
        track = ResearchBallTrack(max_misses=2)
        result = track.update([None, {}, {"center": (1,), "confidence": 0.9},
                               {"center": (1, 2), "confidence": "bad"},
                               {"center": (float("nan"), 2), "confidence": 0.9},
                               {"center": (3, 4), "confidence": float("inf")},
                               candidate(100, 200, 0.8)])
        self.assertEqual(result["state"], "observed")
        self.assertEqual(result["point"], {"x": 100.0, "y": 200.0})

    def test_preserves_rejected_candidate_status_instead_of_seeding_track(self):
        track = ResearchBallTrack(max_misses=2)
        rejected = candidate(100, 200, 0.9)
        rejected["status"] = "rejected"
        result = track.update([rejected])
        self.assertEqual(result["state"], "unavailable")
        self.assertIsNone(result["point"])
        self.assertEqual(result["warning"], "candidate_rejected")

    def test_aggregates_bounded_recent_uncertainty_without_identity_claim(self):
        from ghostcaddie.video.research_ball_model import aggregate_temporal_uncertainty

        summary = aggregate_temporal_uncertainty([
            {"state": "observed", "point": {"x": 100, "y": 200}, "confidence": 0.99},
            {"state": "observed", "point": {"x": 10, "y": 20}, "confidence": 0.8},
            {"state": "predicted", "point": {"x": 13, "y": 24}, "confidence": 0.6},
        ], window=2)

        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["observed_count"], 1)
        self.assertEqual(summary["predicted_count"], 1)
        self.assertEqual(summary["confidence_range"], (0.6, 0.8))
        self.assertAlmostEqual(summary["spatial_radius_px"], 2.5)
        self.assertEqual(summary["confidence_semantics"], "detection_quality_not_identity")
        self.assertEqual(summary["provenance"], "research_temporal_aggregation")
        self.assertEqual(summary["identity"], "unavailable")

    def test_temporal_uncertainty_preserves_unavailable_when_no_points_exist(self):
        from ghostcaddie.video.research_ball_model import aggregate_temporal_uncertainty

        summary = aggregate_temporal_uncertainty([
            {"state": "unavailable", "point": None, "confidence": 0.0},
            {"state": "terminated", "point": None, "confidence": 0.0},
        ])

        self.assertEqual(summary["state"], "unavailable")
        self.assertIsNone(summary["spatial_radius_px"])
        self.assertIsNone(summary["confidence_range"])
        self.assertEqual(summary["provenance"], "unavailable")
        self.assertEqual(summary["identity"], "unavailable")


class TestResearchBallMultiHypothesisTrack(unittest.TestCase):
    def test_confidence_semantics_are_detection_quality_not_identity(self):
        result = ResearchBallMultiHypothesisTrack().update([candidate(100, 200, 0.8)])
        self.assertEqual(result["confidence_semantics"], "detection_quality_not_identity")

    def test_weak_detection_cannot_revive_terminated_track(self):
        track = ResearchBallMultiHypothesisTrack(reacquire_confidence=0.75, max_misses=1)
        track.update([candidate(100, 200, 0.9)])
        track.update([])
        result = track.update([candidate(101, 200, 0.6)])
        self.assertEqual(result["state"], "terminated")
        self.assertEqual(result["warning"], "reacquisition_insufficient_evidence")

    def test_reacquisition_requires_two_consistent_strong_frames(self):
        track = ResearchBallMultiHypothesisTrack(reacquire_confidence=0.75, max_misses=1, max_step=30)
        track.update([candidate(100, 200, 0.9)])
        track.update([])
        pending = track.update([candidate(110, 200, 0.8)])
        reacquired = track.update([candidate(120, 200, 0.82)])
        self.assertEqual(pending["state"], "terminated")
        self.assertEqual(pending["warning"], "reacquisition_pending")
        self.assertEqual(reacquired["state"], "reacquired")
        self.assertEqual(reacquired["point"], {"x": 120.0, "y": 200.0})

    def test_continuity_beats_higher_confidence_drift_candidate(self):
        track = ResearchBallMultiHypothesisTrack(max_step=35, max_hypotheses=3)
        track.update([candidate(100, 200, 0.8)])
        track.update([candidate(110, 200, 0.8), candidate(100, 350, 0.95)])
        result = track.update([candidate(120, 200, 0.7), candidate(125, 350, 0.99)])
        self.assertEqual(result["state"], "observed")
        self.assertEqual(result["point"], {"x": 120.0, "y": 200.0})
        self.assertGreaterEqual(result["hypothesis_count"], 2)

    def test_ambiguous_continuity_quality_tie_stays_unavailable(self):
        track = ResearchBallMultiHypothesisTrack(max_step=35, ambiguity_margin=0.05)
        track.update([candidate(100, 200, 0.8)])
        track.update([candidate(110, 200, 0.8)])
        result = track.update([candidate(120, 200, 0.76), candidate(125, 200, 0.77)])
        self.assertEqual(result["state"], "unavailable")
        self.assertIsNone(result["point"])
        self.assertEqual(result["warning"], "ambiguous_candidates")
        self.assertEqual(result["hypothesis_count"], 2)

    def test_ambiguous_frame_is_not_promoted_to_reacquisition(self):
        track = ResearchBallMultiHypothesisTrack(reacquire_confidence=0.75, max_misses=1)
        track.update([candidate(100, 200, 0.9)])
        track.update([])
        result = track.update([candidate(110, 200, 0.8), candidate(115, 210, 0.81)])
        self.assertEqual(result["state"], "terminated")
        self.assertEqual(result["warning"], "reacquisition_ambiguous")


if __name__ == "__main__":
    unittest.main()
