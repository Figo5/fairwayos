import unittest

from ghostcaddie.video.model_comparison_overlay import (
    build_comparison_overlay,
    comparison_filter,
)


class ModelComparisonOverlayTests(unittest.TestCase):
    def test_plan_labels_backends_and_suppresses_unavailable_markers(self):
        plan = build_comparison_overlay(
            frame_index=12,
            width=600,
            height=480,
            candidates={
                "PT": {"state": "candidate", "x": 210, "y": 180, "confidence": 0.71},
                "ONNX": {"state": "unavailable"},
                "GENERIC": {"state": "candidate", "x": 250, "y": 190, "confidence": 0.55},
            },
        )
        self.assertEqual([item["label"] for item in plan["markers"]], ["PT", "GENERIC"])
        self.assertEqual(plan["unavailable"], ["ONNX"])
        self.assertEqual(plan["identity"], "unavailable")
        self.assertFalse(plan["production_eligible"])

    def test_filter_contains_labeled_diagnostic_bars_and_no_identity_claim(self):
        graph = comparison_filter(
            frame_index=12, width=600, height=480,
            candidates={
                "PT": {"state": "candidate", "x": 210, "y": 180, "confidence": 0.71},
                "ONNX": {"state": "unavailable"},
                "GENERIC": {"state": "candidate", "x": 250, "y": 190, "confidence": 0.55},
            },
        )
        self.assertIn("PT", graph)
        self.assertIn("ONNX: UNAVAILABLE", graph)
        self.assertIn("GENERIC", graph)
        self.assertIn("IDENTITY UNAVAILABLE", graph)
        self.assertIn("NOT GOLF-BALL IDENTITY", graph)
        self.assertGreaterEqual(graph.count("drawbox"), 4)

    def test_invalid_candidate_is_rejected_instead_of_rendered(self):
        with self.assertRaises(ValueError):
            build_comparison_overlay(
                frame_index=0, width=600, height=480,
                candidates={"PT": {"state": "candidate", "x": 601, "y": 20, "confidence": 0.5}},
            )

    def test_research_states_are_explicit_without_claiming_ball_identity(self):
        plan = build_comparison_overlay(
            frame_index=12,
            width=600,
            height=480,
            candidates={
                "PT": {"state": "observed", "x": 210, "y": 180, "confidence": 0.71},
                "ONNX": {"state": "predicted", "x": 220, "y": 181, "confidence": 0.51},
                "GENERIC": {"state": "unavailable"},
            },
        )
        self.assertEqual(
            [(item["label"], item["state"]) for item in plan["markers"]],
            [("PT", "observed"), ("ONNX", "predicted")],
        )
        self.assertEqual(plan["unavailable"], ["GENERIC"])
        self.assertEqual(plan["identity"], "unavailable")
        self.assertTrue(plan["research_only"])
        graph = comparison_filter(
            frame_index=12,
            width=600,
            height=480,
            candidates={
                "PT": {"state": "observed", "x": 210, "y": 180, "confidence": 0.71},
                "ONNX": {"state": "predicted", "x": 220, "y": 181, "confidence": 0.51},
                "GENERIC": {"state": "unavailable"},
            },
        )
        self.assertIn("PT: OBSERVED 0.71", graph)
        self.assertIn("ONNX: PREDICTED 0.51", graph)
        self.assertIn("GENERIC: UNAVAILABLE", graph)
        self.assertIn("NOT GOLF-BALL IDENTITY", graph)


if __name__ == "__main__":
    unittest.main()
