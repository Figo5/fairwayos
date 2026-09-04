import unittest
from typing import get_type_hints

from ghostcaddie.video.automatic_perception import (
    AUTOMATIC_PERCEPTION_SCHEMA_VERSION,
    BodyAnchor,
    ConfidenceMetrics,
    Detection,
    Detector,
    GateDecision,
    OpticalFlowPolicy,
    Provenance,
    Thresholds,
    Track,
    Tracker,
    continuity_metrics,
    select_single_golfer_track,
    AnchorValidation,
    ImpactCandidateInterval,
    SwingPhase,
    SwingPhaseStateMachine,
    evaluate_sequence_gates,
    precision_recall,
    reconstruct_automatic_shot,
    PixelTrackStore,
    CameraMotionCompensator,
)


class TestAutomaticPerceptionContracts(unittest.TestCase):
    def test_camera_motion_compensation_is_separate_from_object_tracking(self):
        compensation = CameraMotionCompensator()
        compensation.update((3.0, -2.0))
        self.assertEqual(compensation.compensate((13.0, 8.0)), (10.0, 10.0))
        self.assertEqual(compensation.total_translation, (3.0, -2.0))

    def test_pixel_track_store_assigns_stable_ids_and_keeps_labels_separate(self):
        store = PixelTrackStore(max_distance=10.0)
        first = store.update([Detection(0, "golfer", (100.0, 100.0), .9, Provenance.DETECTED)])
        second = store.update([Detection(1, "golfer", (104.0, 103.0), .8, Provenance.TRACKED)])
        ball = store.update([Detection(2, "ball", (104.0, 103.0), .8, Provenance.DETECTED)])
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertNotEqual(second[0].track_id, ball[0].track_id)
        self.assertEqual(second[0].frame_indices, (0, 1))

    def test_contract_is_versioned_and_unavailable_is_explicit(self):
        self.assertEqual(AUTOMATIC_PERCEPTION_SCHEMA_VERSION, "automatic-perception.v1")
        unavailable = Detection.unavailable(frame_index=4, label="ball")
        self.assertIsNone(unavailable.value)
        self.assertEqual(unavailable.provenance, Provenance.UNAVAILABLE)
        self.assertFalse(unavailable.visible)
        self.assertIn("unavailable", unavailable.warnings)

    def test_detector_and_tracker_are_runtime_checkable_protocols(self):
        self.assertEqual(get_type_hints(Detector.detect)["return"].__origin__, list)
        self.assertEqual(get_type_hints(Tracker.update)["return"].__origin__, list)

    def test_pose_anchor_uses_feet_when_available_and_is_unavailable_without_pose(self):
        anchor = BodyAnchor.from_pose(
            {"left_ankle": (10, 20), "right_ankle": (14, 24)}, confidence=0.8
        )
        self.assertEqual(anchor.point, (12.0, 22.0))
        self.assertEqual(anchor.provenance, Provenance.POSE)
        missing = BodyAnchor.unavailable("occlusion")
        self.assertIsNone(missing.point)
        self.assertEqual(missing.warnings, ("occlusion",))

    def test_single_golfer_selection_is_deterministic(self):
        tracks = [
            Track(2, "golfer", (0, 1, 2), (0.9, 0.9, 0.9), 100.0),
            Track(1, "golfer", (0, 1, 2), (0.8, 0.8, 0.8), 200.0),
            Track(3, "golfer", (0, 1), (1.0, 1.0), 500.0),
        ]
        self.assertEqual(select_single_golfer_track(tracks, frame_count=3).track_id, 2)
        self.assertIsNone(select_single_golfer_track([], frame_count=3))

    def test_continuity_and_confidence_metrics_are_deterministic(self):
        self.assertEqual(continuity_metrics([0, 1, 2, 4], frame_count=5).coverage, 0.8)
        self.assertEqual(continuity_metrics([0, 1, 2, 4], frame_count=5).longest_gap, 1)
        self.assertEqual(ConfidenceMetrics.from_values([0.2, 0.8, 1.0]).mean, 2 / 3)
        self.assertEqual(ConfidenceMetrics.from_values([]).mean, 0.0)

    def test_flow_policy_refines_only_within_bounded_gap_and_never_promotes_semantics(self):
        policy = OpticalFlowPolicy(max_gap_frames=2)
        self.assertTrue(policy.can_refine(gap_frames=2, source_confidence=0.7))
        self.assertFalse(policy.can_refine(gap_frames=3, source_confidence=0.7))
        self.assertFalse(policy.can_promote_to_semantic_detection)

    def test_thresholds_and_gate_decision_are_provisional_and_explain_blocking(self):
        thresholds = Thresholds()
        self.assertTrue(thresholds.provisional)
        decision = GateDecision.from_metrics(
            continuity=continuity_metrics([0, 1], frame_count=3),
            thresholds=thresholds,
        )
        self.assertFalse(decision.passed)
        self.assertIn("coverage", decision.blocking_reasons)
        self.assertEqual(decision.status, "blocked")

    def test_pose_and_body_anchor_validation_is_deterministic_and_explicit(self):
        valid = AnchorValidation.from_pose(
            {"left_ankle": (10, 20), "right_ankle": (14, 24)},
            image_width=100, image_height=100, confidence=0.8,
        )
        self.assertTrue(valid.available)
        self.assertEqual(valid.anchor.point, (12.0, 22.0))
        invalid = AnchorValidation.from_pose({}, 100, 100, 0.8)
        self.assertFalse(invalid.available)
        self.assertIsNone(invalid.anchor.point)
        out_of_bounds = AnchorValidation(BodyAnchor((101, 1), .9, Provenance.POSE), 100, 100)
        self.assertFalse(out_of_bounds.available)

    def test_swing_state_machine_emits_ordered_phases_and_impact_interval(self):
        machine = SwingPhaseStateMachine()
        phases = [machine.update(i, (float(i), 0.0), (float(i), 0.0)).phase
                  for i in range(6)]
        self.assertEqual(phases[0], SwingPhase.ADDRESS)
        self.assertEqual(phases[-1], SwingPhase.FOLLOW_THROUGH)
        interval = ImpactCandidateInterval.from_frames([3, 4], confidence=.8, frame_rate=60)
        self.assertEqual((interval.start_frame, interval.end_frame), (3, 4))
        self.assertEqual(interval.uncertainty_frames, 2)

    def test_impact_interval_rejects_malformed_frames_and_confidence(self):
        with self.assertRaises(ValueError):
            ImpactCandidateInterval.from_frames([-1, 2], confidence=0.8, frame_rate=60)
        with self.assertRaises(ValueError):
            ImpactCandidateInterval.from_frames([True, 2], confidence=0.8, frame_rate=60)
        with self.assertRaises(ValueError):
            ImpactCandidateInterval.from_frames([1.5, 2], confidence=0.8, frame_rate=60)
        with self.assertRaises(ValueError):
            ImpactCandidateInterval.from_frames([0, 2], confidence=float("nan"), frame_rate=60)
        with self.assertRaises(ValueError):
            ImpactCandidateInterval.from_frames([0, 2], confidence=float("inf"), frame_rate=60)
        with self.assertRaises(ValueError):
            ImpactCandidateInterval.from_frames([0, 2], confidence=True, frame_rate=60)

    def test_sequence_gates_block_motion_and_cuts_with_reasons(self):
        decision = evaluate_sequence_gates(
            camera_motion_displacements=[0.01, 0.05], cut_frames=[4],
            track_coverage=.99, longest_gap=1, anchor_coverage=.99,
            thresholds=Thresholds(),
        )
        self.assertFalse(decision.passed)
        self.assertEqual(decision.blocking_reasons, ("camera_motion", "cut"))

    def test_precision_recall_uses_explicit_empty_unavailable_values(self):
        metrics = precision_recall([True, True, False], [True, False, False])
        self.assertEqual((metrics.precision, metrics.recall), (0.5, 1.0))
        empty = precision_recall([], [])
        self.assertEqual((empty.precision, empty.recall), (None, None))

    def test_automatic_reconstruction_requires_all_evidence_and_maps_once(self):
        from tests.test_video_reconstruction import PAYLOAD, calibration, context
        from ghostcaddie.video.observations import VideoObservations
        observations = VideoObservations.from_dict(PAYLOAD)
        class CountingCalibration:
            width, height = 1920, 1080
            def __init__(self): self.calls = []
            def to_engine(self, point):
                self.calls.append(point)
                from ghostcaddie.geometry import Point2D
                return Point2D(point.x / 10, point.y / 10)
        mapper = CountingCalibration()
        result = reconstruct_automatic_shot(observations, mapper, context())
        self.assertIsNotNone(result.shot_event)
        self.assertEqual(len(mapper.calls), 3)
        self.assertEqual(result.metadata["source"], "video-automatic")

    def test_automatic_reconstruction_rejects_missing_or_low_confidence_evidence(self):
        import copy
        from tests.test_video_reconstruction import PAYLOAD, calibration, context
        from ghostcaddie.video.errors import VideoReconstructionUnavailable
        from ghostcaddie.video.observations import VideoObservations
        for label in ("ball", "clubhead", "contact", "landing"):
            payload = copy.deepcopy(PAYLOAD)
            if label in ("ball", "clubhead"):
                payload["observations"][1][label] = None
            else:
                payload["observations"][1][label] = None
            payload["observations"][1]["warnings"] = ["ball_missing"] if label == "ball" else []
            with self.subTest(label=label), self.assertRaises(VideoReconstructionUnavailable):
                reconstruct_automatic_shot(VideoObservations.from_dict(payload), calibration(), context())
        for label in ("ball", "clubhead", "contact", "landing"):
            payload = copy.deepcopy(PAYLOAD)
            payload["observations"][1][label]["confidence"] = 0.1
            payload["observations"][1]["warnings"] = ["low_confidence"]
            with self.subTest(low_confidence=label), self.assertRaises(VideoReconstructionUnavailable):
                reconstruct_automatic_shot(VideoObservations.from_dict(payload), calibration(), context())

    def test_automatic_reconstruction_rejects_missing_calibration_dimensions(self):
        from tests.test_video_reconstruction import PAYLOAD, context
        from ghostcaddie.video.errors import VideoReconstructionUnavailable
        from ghostcaddie.video.observations import VideoObservations
        class Mapper:
            def to_engine(self, point): return point
        with self.assertRaises(VideoReconstructionUnavailable):
            reconstruct_automatic_shot(VideoObservations.from_dict(PAYLOAD), Mapper(), context())


if __name__ == "__main__":
    unittest.main()
