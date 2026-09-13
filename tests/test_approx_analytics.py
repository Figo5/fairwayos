import math
import unittest

from approx_analytics import Point, build_metrics, parse_native_point, projected_speed_mps, scale_from_ball_diameter, speed_px_per_display_second


class ApproxAnalyticsTests(unittest.TestCase):
    def test_missing_prior_has_no_displacement_or_speed(self):
        rows = build_metrics([{"source_frame": 1, "ball": Point(1, 1, 0.9), "head": Point(2, 2, 0.8)}])
        self.assertIsNone(rows[0]["ball_disp_px"])
        self.assertIsNone(rows[0]["ball_speed_px_per_s"])
        self.assertIsNone(rows[0]["head_disp_px"])

    def test_null_gap_does_not_bridge(self):
        rows = build_metrics([
            {"source_frame": 1, "ball": Point(0, 0), "head": Point(0, 0)},
            {"source_frame": 2, "ball": None, "head": None},
            {"source_frame": 3, "ball": Point(3, 4), "head": Point(6, 8)},
        ])
        self.assertIsNone(rows[1]["ball_disp_px"])
        self.assertIsNone(rows[2]["ball_disp_px"])
        self.assertIsNone(rows[2]["head_disp_px"])

    def test_rejected_inconclusive_and_nonfinite_inputs_fail_closed(self):
        self.assertIsNone(parse_native_point({"rejected": "bad", "native_point": [1, 2, 0.9]}, "x"))
        self.assertIsNone(parse_native_point({"inconclusive": "bad", "native_point": [1, 2, 0.9]}, "x"))
        self.assertIsNone(parse_native_point({"semantic_only": True, "native_point": [1, 2, 0.9]}, "x"))
        self.assertIsNone(parse_native_point({"inconclusive": "historic", "semantic_only": True, "semantic_check": {"supported": False}, "native_point": [1, 2, 0.9]}, "x"))
        self.assertIsNone(parse_native_point({"inconclusive": "historic", "semantic_only": True, "semantic_check": {"supported": True}}, "x"))
        with self.assertRaises(ValueError):
            parse_native_point({"native_point": [math.nan, 2, 0.9]}, "x")

    def test_semantic_final_acceptance_survives_historic_inconclusive(self):
        point = parse_native_point({
            "inconclusive": "measurement could not decide before semantic fallback",
            "semantic_only": True,
            "semantic_check": {"supported": True, "verdict": "supported"},
            "native_point": [10, 20, 0.75],
        }, "x")

        self.assertEqual(point, Point(10, 20, 0.75, semantic=True))
        with self.assertRaises(ValueError):
            speed_px_per_display_second(1, fps=math.inf)
        with self.assertRaises(ValueError):
            projected_speed_mps(1, m_per_px=0.001, action_time_scale=float("nan"))

    def test_scale_conversion_requires_explicit_time_and_supports_ball_diameter(self):
        self.assertIsNone(projected_speed_mps(10, m_per_px=0.001, action_time_scale=None))
        self.assertAlmostEqual(projected_speed_mps(10, m_per_px=0.001, action_time_scale=2, fps=30), 0.6)
        scale = scale_from_ball_diameter(20)
        self.assertAlmostEqual(scale["m_per_px"], 0.04267 / 20)
        self.assertIn("at-ball-depth", scale["assumption"])


if __name__ == "__main__":
    unittest.main()
