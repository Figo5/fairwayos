import unittest

from ghostcaddie.video.research_ball_evaluation import summarize_track_report


class TestResearchBallEvaluation(unittest.TestCase):
    def test_reports_continuity_and_termination_metrics_without_inventing_precision(self):
        report = {
            "observations": [
                {"state": "observed", "point": {"x": 1, "y": 1}, "candidate_count": 1},
                {"state": "predicted", "point": {"x": 2, "y": 1}, "candidate_count": 2},
                {"state": "terminated", "point": None, "candidate_count": 1},
                {"state": "terminated", "point": None, "candidate_count": 1},
                {"state": "reacquired", "point": {"x": 5, "y": 1}, "candidate_count": 1},
            ]
        }
        result = summarize_track_report(report)
        self.assertEqual(result["frame_count"], 5)
        self.assertEqual(result["accepted_frame_count"], 3)
        self.assertEqual(result["longest_active_run"], 2)
        self.assertEqual(result["termination_count"], 2)
        self.assertEqual(result["reacquisition_count"], 1)
        self.assertEqual(result["post_termination_candidate_rejections"], 2)
        self.assertIsNone(result["false_positive_rate"])
        self.assertFalse(result["ground_truth_available"])

    def test_empty_report_is_safe_and_deterministic(self):
        result = summarize_track_report({"observations": []})
        self.assertEqual(result["frame_count"], 0)
        self.assertEqual(result["coverage"], 0.0)
        self.assertIsNone(result["false_positive_rate"])

    def test_impact_bracket_never_becomes_exact(self):
        from ghostcaddie.video.clubhead_pseudo_labels import estimate_impact_window

        result = estimate_impact_window([
            {"available": True, "frame_index": 10, "ball_club_alignment": 8.0},
            {"available": True, "frame_index": 11, "ball_club_alignment": 4.0},
        ], 25.0)
        self.assertEqual((result["start_frame"], result["end_frame"]), (9, 13))
        self.assertIsNone(result["exact_impact"])
        self.assertIn("not_ground_truth", result["warning"])


if __name__ == "__main__":
    unittest.main()
