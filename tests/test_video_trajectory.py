import unittest

from ghostcaddie.video.trajectory import trace_ball_observations, build_trajectory_overlay


class BallTrajectoryTests(unittest.TestCase):
    def test_interpolates_only_short_explicit_observation_gap_and_marks_provenance(self):
        trace = trace_ball_observations(
            [
                {"frame_index": 2, "timestamp_seconds": 0.2, "ball": {"x": 10, "y": 20, "confidence": 0.9}},
                {"frame_index": 4, "timestamp_seconds": 0.4, "ball": {"x": 30, "y": 40, "confidence": 0.7}},
            ],
            max_interpolation_gap=2,
        )
        self.assertEqual([point.frame_index for point in trace.points], [2, 3, 4])
        self.assertEqual([point.provenance for point in trace.points], ["observed", "interpolated", "observed"])
        self.assertEqual((trace.points[1].x, trace.points[1].y), (20.0, 30.0))
        self.assertEqual(trace.points[0].confidence, 0.9)
        self.assertGreater(trace.points[0].uncertainty_radius, 0)
        self.assertEqual(trace.gaps, ())

    def test_long_loss_is_gap_and_faded_lifetime_is_deterministic(self):
        trace = trace_ball_observations(
            [
                {"frame_index": 0, "timestamp_seconds": 0.0, "ball": {"x": 5, "y": 6, "confidence": 1.0}},
                {"frame_index": 5, "timestamp_seconds": 0.5, "ball": {"x": 15, "y": 16, "confidence": 0.5}},
            ],
            max_interpolation_gap=2,
            fade_frames=3,
        )
        self.assertEqual(trace.gaps, ((1, 4),))
        self.assertEqual(trace.points[0].lifetime, 3)
        self.assertEqual(trace.points[-1].lifetime, 1)
        self.assertIn("ball_track_gap", trace.warnings)

    def test_absent_ball_preserves_unavailable_state_and_overlay_warns(self):
        trace = trace_ball_observations([
            {"frame_index": 7, "timestamp_seconds": 0.7, "ball": None},
        ])
        self.assertFalse(trace.available)
        self.assertEqual(trace.points, ())
        self.assertEqual(trace.warnings, ("ball_unavailable",))
        overlay = build_trajectory_overlay(trace)
        self.assertIn("trajectory\\: unavailable", overlay)
        self.assertIn("ball_unavailable", overlay)

    def test_implausible_jump_terminates_continuity_without_relabeling_observation(self):
        trace = trace_ball_observations(
            [
                {"frame_index": 0, "timestamp_seconds": 0.0, "ball": {"x": 10, "y": 10, "confidence": 0.9}},
                {"frame_index": 1, "timestamp_seconds": 0.1, "ball": {"x": 100, "y": 100, "confidence": 0.9}},
            ],
            max_step_pixels=20,
        )
        self.assertEqual([point.provenance for point in trace.points], ["observed", "observed"])
        self.assertIn("implausible_jump", trace.warnings)
        self.assertEqual(trace.gaps, ((1, 1),))

    def test_bounded_prediction_is_opt_in_and_has_increased_uncertainty(self):
        observations = [
            {"frame_index": 0, "timestamp_seconds": 0.0, "ball": {"x": 10, "y": 10, "confidence": 0.9}},
            {"frame_index": 1, "timestamp_seconds": 0.1, "ball": {"x": 14, "y": 12, "confidence": 0.8}},
            {"frame_index": 2, "timestamp_seconds": 0.2, "ball": None},
            {"frame_index": 3, "timestamp_seconds": 0.3, "ball": None},
        ]
        plain = trace_ball_observations(observations)
        predicted = trace_ball_observations(observations, predict_missing=True, max_prediction_gap=2)
        self.assertEqual([point.frame_index for point in plain.points], [0, 1])
        self.assertEqual([point.frame_index for point in predicted.points], [0, 1, 2, 3])
        self.assertEqual([point.provenance for point in predicted.points[-2:]], ["predicted", "predicted"])
        self.assertGreater(predicted.points[-1].uncertainty_radius, predicted.points[1].uncertainty_radius)
        self.assertIn("predicted_track", predicted.warnings)

    def test_prediction_terminates_when_trailing_occlusion_exceeds_bound(self):
        observations = [
            {"frame_index": 0, "timestamp_seconds": 0.0, "ball": {"x": 10, "y": 10, "confidence": 0.9}},
            {"frame_index": 1, "timestamp_seconds": 0.1, "ball": {"x": 14, "y": 12, "confidence": 0.8}},
            {"frame_index": 2, "timestamp_seconds": 0.2, "ball": None},
            {"frame_index": 3, "timestamp_seconds": 0.3, "ball": None},
        ]
        trace = trace_ball_observations(observations, predict_missing=True, max_prediction_gap=1)
        self.assertEqual([point.frame_index for point in trace.points], [0, 1])
        self.assertIn("ball_track_gap", trace.warnings)

    def test_optical_flow_is_optional_and_rejected_when_outside_bound(self):
        observations = [
            {"frame_index": 0, "timestamp_seconds": 0.0, "ball": {"x": 10, "y": 10, "confidence": 0.9}},
            {"frame_index": 1, "timestamp_seconds": 0.1, "ball": None},
        ]
        flow = lambda previous, frame_index, timestamp: (100, 100)
        trace = trace_ball_observations(
            observations, predict_missing=True, max_prediction_gap=1,
            optical_flow=flow, max_step_pixels=20,
        )
        self.assertEqual([point.frame_index for point in trace.points], [0])
        self.assertIn("prediction_rejected", trace.warnings)

    def test_rejects_explicit_observations_that_are_not_in_input_order(self):
        with self.assertRaisesRegex(ValueError, "strictly ordered"):
            trace_ball_observations([
                {"frame_index": 2, "timestamp_seconds": 0.2, "ball": {"x": 2, "y": 2, "confidence": 1}},
                {"frame_index": 1, "timestamp_seconds": 0.1, "ball": {"x": 1, "y": 1, "confidence": 1}},
            ])

    def test_implausible_jump_marks_the_entire_missing_interval_as_a_gap(self):
        trace = trace_ball_observations([
            {"frame_index": 0, "timestamp_seconds": 0.0, "ball": {"x": 0, "y": 0, "confidence": 1}},
            {"frame_index": 5, "timestamp_seconds": 0.5, "ball": {"x": 100, "y": 0, "confidence": 1}},
        ], max_step_pixels=19)
        self.assertEqual(trace.gaps, ((1, 4),))

    def test_camera_motion_hook_stabilizes_coordinates_before_linking(self):
        trace = trace_ball_observations([
            {"frame_index": 0, "timestamp_seconds": 0.0, "ball": {"x": 10, "y": 20, "confidence": 1}},
            {"frame_index": 1, "timestamp_seconds": 0.1, "ball": {"x": 15, "y": 20, "confidence": 1}},
        ], max_step_pixels=2, camera_motion=lambda frame, timestamp: (5 * frame, 0))
        self.assertEqual([(point.x, point.y) for point in trace.points], [(10.0, 20.0), (10.0, 20.0)])
        self.assertNotIn("implausible_jump", trace.warnings)

    def test_overlay_distinguishes_observed_interpolated_confidence_radius_and_warning(self):
        trace = trace_ball_observations([
            {"frame_index": 0, "timestamp_seconds": 0.0, "ball": {"x": 1, "y": 2, "confidence": 0.8}},
            {"frame_index": 2, "timestamp_seconds": 0.2, "ball": {"x": 5, "y": 6, "confidence": 0.6}},
        ])
        overlay = build_trajectory_overlay(trace)
        for token in ("observed", "interpolated", "confidence=", "radius=", "fade="):
            self.assertIn(token, overlay)
        self.assertIn("warnings\\: ", overlay)

    def test_vertical_flight_speed_constraint_breaks_continuity_without_relabeling_observations(self):
        trace = trace_ball_observations(
            [
                {"frame_index": 0, "timestamp_seconds": 0.0, "ball": {"x": 10, "y": 10, "confidence": 1}},
                {"frame_index": 1, "timestamp_seconds": 0.1, "ball": {"x": 10, "y": 30, "confidence": 1}},
            ],
            max_vertical_speed_pixels_per_second=50,
        )
        self.assertEqual([point.provenance for point in trace.points], ["observed", "observed"])
        self.assertEqual(trace.gaps, ((1, 1),))
        self.assertIn("implausible_vertical_speed", trace.warnings)

    def test_ground_boundary_rejects_prediction_below_ground(self):
        observations = [
            {"frame_index": 0, "timestamp_seconds": 0.0, "ball": {"x": 10, "y": 80, "confidence": 1}},
            {"frame_index": 1, "timestamp_seconds": 0.1, "ball": {"x": 12, "y": 81.5, "confidence": 1}},
            {"frame_index": 2, "timestamp_seconds": 0.2, "ball": None},
        ]
        trace = trace_ball_observations(
            observations, predict_missing=True, max_prediction_gap=1,
            ground_y=82, ground_tolerance_pixels=0,
        )
        self.assertEqual([point.provenance for point in trace.points], ["observed", "observed"])
        self.assertIn("ground_constraint", trace.warnings)
        self.assertNotIn("predicted_track", trace.warnings)

    def test_camera_compensated_constant_velocity_prediction_is_marked_predicted(self):
        trace = trace_ball_observations(
            [
                {"frame_index": 0, "timestamp_seconds": 0.0, "ball": {"x": 100, "y": 20, "confidence": 1}},
                {"frame_index": 1, "timestamp_seconds": 0.1, "ball": {"x": 106, "y": 22, "confidence": 1}},
                {"frame_index": 2, "timestamp_seconds": 0.2, "ball": None},
            ],
            predict_missing=True, max_prediction_gap=1,
            camera_motion=lambda frame, timestamp: (5 * frame, 0),
        )
        # Coordinates remain in camera-stabilized space when compensation is used.
        self.assertEqual((trace.points[-1].x, trace.points[-1].y), (102.0, 24.0))
        self.assertEqual(trace.points[-1].provenance, "predicted")
        self.assertIn("predicted_track", trace.warnings)


if __name__ == "__main__":
    unittest.main()
