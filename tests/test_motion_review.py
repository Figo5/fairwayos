"""Tests for the pure image-motion review.

No video, no cv2, no model, no physical units. The cases that matter are the ones
where a rate could otherwise lie: a gap bridged, a manual anchor counted as motion,
an operator/derived point paired, a cut crossed, a normalised value called speed.
"""
from __future__ import annotations

import copy
import unittest

from ghostcaddie.demo.motion_review import PROFILE, summarize_motion

SOURCE = (1920, 1080)
#: Irregular but exactly representable source timeline (frames 0..9, seconds).
PTS = [0.0, 0.25, 0.375, 1.0, 1.125, 2.0, 2.25, 3.0, 3.125, 4.0]
WINDOW = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625]


def pt(frame, x, y, *, origin="automatic", status="observed_template",
       method="template", score=None):
    """A fully declared point mapping (every field the contract names)."""
    return {"source_frame": frame, "x": x, "y": y, "origin": origin,
            "status": status, "method": method, "score": score}


class TimelineTests(unittest.TestCase):
    def test_irregular_pts_uses_the_actual_source_dt(self):
        points = [pt(0, 0.0, 0.0), pt(1, 3.0, 4.0), pt(2, 6.0, 8.0)]
        result = summarize_motion(points, PTS, source_size=SOURCE,
                                  start_frame=0, end_frame=2)
        pairs = result["pairs"]
        self.assertEqual([p["dt_seconds"] for p in pairs], [0.25, 0.125])
        self.assertEqual(pairs[0]["step_native_px"], 5.0)
        self.assertEqual(pairs[0]["rate_native_px_per_video_second"], 20.0)
        self.assertEqual(pairs[1]["rate_native_px_per_video_second"], 40.0)

    def test_gaps_are_never_bridged(self):
        points = [pt(0, 0.0, 0.0), pt(1, 1.0, 0.0), pt(3, 3.0, 0.0), pt(4, 4.0, 0.0)]
        result = summarize_motion(points, PTS, source_size=SOURCE,
                                  start_frame=0, end_frame=4)
        self.assertEqual([p["source_frames"] for p in result["pairs"]],
                         [[0, 1], [3, 4]])
        self.assertEqual(result["pair_count"], 2)

    def test_operator_and_derived_points_are_never_paired(self):
        points = [pt(0, 0.0, 0.0), pt(1, 1.0, 0.0),
                  pt(2, 2.0, 0.0, origin="operator", status="manual_anchor",
                     method="operator_placement"),
                  pt(3, 3.0, 0.0, origin="derived", status="interpolated",
                     method="linear_interpolation"),
                  pt(4, 4.0, 0.0)]
        result = summarize_motion(points, PTS, source_size=SOURCE,
                                  start_frame=0, end_frame=4)
        self.assertEqual([p["source_frames"] for p in result["pairs"]], [[0, 1]])

    def test_a_cut_withholds_only_the_pair_that_crosses_it(self):
        points = [pt(0, 0.0, 0.0), pt(1, 1.0, 0.0), pt(2, 2.0, 0.0), pt(3, 3.0, 0.0)]
        result = summarize_motion(points, PTS, source_size=SOURCE, start_frame=0,
                                  end_frame=3, cut_frames=[2])
        self.assertEqual([p["source_frames"] for p in result["pairs"]],
                         [[0, 1], [2, 3]])

    def test_reacquired_method_starts_a_new_segment(self):
        points = [pt(0, 0.0, 0.0), pt(1, 1.0, 0.0),
                  pt(2, 2.0, 0.0, method="learned_shaft_core_reacquired")]
        result = summarize_motion(points, PTS, source_size=SOURCE,
                                  start_frame=0, end_frame=2)
        self.assertEqual([p["source_frames"] for p in result["pairs"]], [[0, 1]])

    def test_paired_coverage_denominator_is_all_possible_frame_pairs(self):
        points = [pt(0, 0.0, 0.0), pt(1, 1.0, 0.0)]  # one measured pair
        result = summarize_motion(points, PTS, source_size=SOURCE,
                                  start_frame=0, end_frame=4)
        self.assertEqual(result["paired_coverage"], 0.25)  # 1 measured / 4 possible


class SummaryTests(unittest.TestCase):
    def test_duration_weighted_mean_differs_from_arithmetic_mean(self):
        pts = [0.0, 0.25, 1.25]
        points = [pt(0, 0.0, 0.0), pt(1, 10.0, 0.0), pt(2, 19.0, 0.0)]
        result = summarize_motion(points, pts, source_size=SOURCE,
                                  start_frame=0, end_frame=2)
        summary = result["summary"]
        # rates are 40 and 9; duration-weighted (19/1.25) is 15.2, not 24.5.
        self.assertEqual(summary["mean_rate_native_px_per_video_second"], 15.2)
        self.assertNotEqual(summary["mean_rate_native_px_per_video_second"], 24.5)

    def test_earliest_pair_wins_a_peak_tie(self):
        pts = [0.0, 0.125, 0.25]
        points = [pt(0, 0.0, 0.0), pt(1, 5.0, 0.0), pt(2, 10.0, 0.0)]
        result = summarize_motion(points, pts, source_size=SOURCE,
                                  start_frame=0, end_frame=2)
        summary = result["summary"]
        self.assertEqual(summary["peak_observed_rate_native_px_per_video_second"], 40.0)
        self.assertEqual(summary["peak_source_pair"], [0, 1])

    def test_rate_normalised_by_source_height(self):
        pts = [0.0, 0.25]
        points = [pt(0, 0.0, 0.0), pt(1, 3.0, 4.0)]
        result = summarize_motion(points, pts, source_size=SOURCE,
                                  start_frame=0, end_frame=1)
        pair = result["pairs"][0]
        self.assertEqual(pair["rate_native_px_per_video_second"], 20.0)
        self.assertEqual(pair["rate_source_height_per_video_second"], 20.0 / 1080)

    def test_zero_pairs_is_unavailable_with_no_summary(self):
        points = [pt(0, 1.0, 1.0, origin="operator", status="manual_anchor",
                     method="operator_placement")]
        result = summarize_motion(points, PTS, source_size=SOURCE,
                                  start_frame=0, end_frame=0)
        self.assertFalse(result["available"])
        self.assertIsNone(result["summary"])
        self.assertEqual(result["pair_count"], 0)
        self.assertEqual(result["eligible_automatic_frames"], 0)
        self.assertEqual(result["paired_coverage"], 0.0)
        self.assertEqual(result["pairs"], [])


class WindowTests(unittest.TestCase):
    def test_source_window_excludes_outside_points(self):
        points = [pt(i, float(i), 0.0) for i in range(6)]
        result = summarize_motion(points, WINDOW, source_size=SOURCE,
                                  start_frame=2, end_frame=3)
        self.assertEqual([p["source_frames"] for p in result["pairs"]], [[2, 3]])
        self.assertEqual(result["pair_count"], 1)
        self.assertEqual(result["frames_in_window"], 2)
        self.assertEqual(result["eligible_automatic_frames"], 2)

    def test_malformed_point_outside_the_window_is_still_refused(self):
        points = [pt(i, float(i), 0.0) for i in range(6)]
        points[5] = pt(5, float("nan"), 0.0)
        with self.assertRaises(ValueError):
            summarize_motion(points, WINDOW, source_size=SOURCE,
                             start_frame=2, end_frame=3)


class ValidationTests(unittest.TestCase):
    def test_exact_native_coordinate_borders(self):
        ok = summarize_motion([pt(0, 0.0, 0.0), pt(1, 1919.0, 1079.0)], PTS,
                              source_size=SOURCE, start_frame=0, end_frame=1)
        self.assertEqual(ok["pair_count"], 1)
        for bad in ([pt(0, 1920.0, 0.0)], [pt(0, 0.0, 1080.0)], [pt(0, -1.0, 0.0)],
                    [pt(0, 0.0, -0.5)]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    summarize_motion(bad, PTS, source_size=SOURCE,
                                     start_frame=0, end_frame=1)

    def test_malformed_points_are_refused(self):
        cases = {
            "nan": [pt(0, float("nan"), 0.0)],
            "inf": [pt(0, 0.0, float("inf"))],
            "bool_coord": [pt(0, True, 0.0)],
            "partial_even_if_missing": [
                {"source_frame": 0, "x": 1.0, "y": None, "origin": "automatic",
                 "status": "missing", "method": "", "score": None}],
            "missing_xy": [{"source_frame": 0, "origin": "automatic",
                            "status": "observed_template", "method": "", "score": None}],
            "unknown_origin": [pt(0, 0.0, 0.0, origin="robot")],
            "unknown_status": [pt(0, 0.0, 0.0, status="teleported")],
            "duplicate_frame": [pt(0, 0.0, 0.0), pt(0, 1.0, 1.0)],
            "bool_frame": [{"source_frame": True, "x": 0.0, "y": 0.0,
                            "origin": "automatic", "status": "observed_template",
                            "method": "", "score": None}],
            "out_of_range_frame": [pt(50, 0.0, 0.0)],
        }
        for name, bad in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    summarize_motion(bad, PTS, source_size=SOURCE,
                                     start_frame=0, end_frame=1)

    def test_malformed_arguments_and_timeline_are_refused(self):
        points = [pt(0, 0.0, 0.0), pt(1, 1.0, 0.0)]
        for kwargs in ({"source_size": (0, 1080)}, {"source_size": (1920.0, 1080)},
                       {"source_size": (1920, True)}, {"source_size": (1920,)},
                       {"source_size": (1920, -1)}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    summarize_motion(points, PTS, start_frame=0, end_frame=1, **kwargs)
        with self.assertRaises(ValueError):  # non-increasing pts
            summarize_motion(points, [0.0, 0.0], source_size=SOURCE,
                             start_frame=0, end_frame=1)
        with self.assertRaises(ValueError):  # decreasing pts
            summarize_motion(points, [0.5, 0.4], source_size=SOURCE,
                             start_frame=0, end_frame=1)
        with self.assertRaises(ValueError):  # bool in pts
            summarize_motion(points, [0.0, True], source_size=SOURCE,
                             start_frame=0, end_frame=1)
        with self.assertRaises(ValueError):  # pts too short for end_frame
            summarize_motion(points, [0.0], source_size=SOURCE,
                             start_frame=0, end_frame=1)
        with self.assertRaises(ValueError):  # start > end
            summarize_motion(points, PTS, source_size=SOURCE,
                             start_frame=3, end_frame=1)
        with self.assertRaises(ValueError):  # duplicate cuts
            summarize_motion(points, PTS, source_size=SOURCE, start_frame=0,
                             end_frame=2, cut_frames=[1, 1])
        with self.assertRaises(ValueError):  # cut out of range
            summarize_motion(points, PTS, source_size=SOURCE, start_frame=0,
                             end_frame=2, cut_frames=[99])

    def test_input_is_not_mutated(self):
        points = [pt(0, 0.0, 0.0), pt(1, 1.0, 0.0), pt(2, 2.0, 0.0)]
        snapshot = copy.deepcopy(points)
        pts = list(PTS)
        cuts = [1]
        summarize_motion(points, pts, source_size=SOURCE, start_frame=0, end_frame=2,
                         cut_frames=cuts)
        self.assertEqual(points, snapshot)
        self.assertEqual(pts, list(PTS))
        self.assertEqual(cuts, [1])


class ProfileTests(unittest.TestCase):
    def test_pose_status_is_explicit_and_operator_positions_stay_excluded(self):
        points = [pt(0, 0.0, 0.0, status="observed_pose", method="movenet_native_pose"),
                  pt(1, 1.0, 0.0, status="observed_pose", method="movenet_native_pose"),
                  pt(2, 2.0, 0.0, origin="operator", status="observed_pose")]
        result = summarize_motion(points, [0.0, 0.1, 0.2], source_size=SOURCE,
                                  start_frame=0, end_frame=2)
        self.assertEqual([p["source_frames"] for p in result["pairs"]], [[0, 1]])

    def test_equal_coordinates_report_zero_steps_without_inventing_frame_duplicates(self):
        result = summarize_motion([pt(0, 1.0, 1.0), pt(1, 1.0, 1.0), pt(2, 4.0, 5.0)],
                                  [0.0, 0.1, 0.2], source_size=SOURCE,
                                  start_frame=0, end_frame=2)
        self.assertEqual(result["zero_step_pair_count"], 1)
        self.assertEqual(result["nonzero_step_pair_count"], 1)
        self.assertNotIn("duplicate_source_frame_count", result)

    def test_finite_inputs_cannot_emit_infinite_rates(self):
        with self.assertRaises(ValueError):
            summarize_motion([pt(0, 0.0, 0.0), pt(1, 1.0, 0.0)],
                             [0.0, 1e-323], source_size=SOURCE,
                             start_frame=0, end_frame=1)
        with self.assertRaises(ValueError):
            summarize_motion([pt(0, 0.0, 0.0), pt(1, 1.0, 0.0)],
                             [-1e308, 1e308], source_size=SOURCE,
                             start_frame=0, end_frame=1)

    def test_nonfinite_scores_and_unhashable_provenance_are_refused(self):
        for field, value in (("score", float("nan")), ("score", float("inf")),
                             ("origin", []), ("status", {})):
            point = pt(0, 0.0, 0.0)
            point[field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    summarize_motion([point], [0.0], source_size=SOURCE,
                                     start_frame=0, end_frame=0)

    def test_profile_states_image_motion_and_refuses_physical_speed(self):
        self.assertEqual(PROFILE["scope"], "2d_image_motion_source_timeline")
        self.assertEqual(PROFILE["unit"], "native_px/video_s")
        self.assertIsNone(PROFILE["physical_speed_mph"])
        self.assertTrue(PROFILE["limitations"])
        self.assertIn("source-video timeline", PROFILE["reason"])
        self.assertNotIn("mph", PROFILE["unit"])

    def test_summary_carries_the_declared_scope(self):
        result = summarize_motion([pt(0, 0.0, 0.0), pt(1, 1.0, 0.0)], PTS,
                                  source_size=SOURCE, start_frame=0, end_frame=1)
        self.assertEqual(result["scope"], "2d_image_motion_source_timeline")
        self.assertEqual(result["unit"], "native_px/video_s")
        self.assertIsNone(result["physical_speed_mph"])
        self.assertTrue(result["limitations"])


if __name__ == "__main__":
    unittest.main()
