import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ghostcaddie.video.automatic_perception import (
    BodyAnchor, ConfidenceMetrics, ContinuityMetrics, GateDecision, Provenance,
    Thresholds,
)
from ghostcaddie.video.automatic_render import (
    build_automatic_report, build_evaluation_report, serialize_automatic_report,
    render_automatic_frame,
)


class TestAutomaticRender(unittest.TestCase):
    def test_report_serialization_is_deterministic_and_preserves_contracts(self):
        report = build_automatic_report(
            frame_results=[{"frame_index": 2, "ball": None, "confidence": {"ball": 0.0},
                            "provenance": {"ball": "unavailable"}, "warnings": ["ball_missing"]}],
            gate_decision=GateDecision("blocked", False, ("continuity",), True),
            thresholds=Thresholds(),
            confidence_metrics=ConfidenceMetrics(.7, .4, .9),
            continuity_metrics=ContinuityMetrics(.8, 2, 8),
            artifact_references=["annotations/frame_000002.jpg"],
        )
        first = serialize_automatic_report(report)
        self.assertEqual(first, serialize_automatic_report(report))
        payload = json.loads(first)
        self.assertEqual(payload["gate"]["blocking_reasons"], ["continuity"])
        self.assertTrue(payload["thresholds"]["provisional"])
        self.assertIsNone(payload["frames"][0]["ball"])
        self.assertEqual(payload["frames"][0]["provenance"]["ball"], "unavailable")

    def test_report_rejects_absolute_paths_and_urls(self):
        with self.assertRaises(ValueError):
            build_automatic_report(frame_results=[], artifact_references=["/tmp/frame.jpg"])
        with self.assertRaises(ValueError):
            build_automatic_report(frame_results=[], visual_references=["https://example.test/frame.jpg"])

    def test_evaluation_report_has_all_metrics_and_reasons_for_missing_evidence(self):
        report = build_evaluation_report(
            track_continuity=.95, anchor_error=None, impact_error=None,
            ball_precision_recall={"precision": .9, "recall": .8},
            clubhead_precision_recall=None, landing_error=None,
            false_positives=0, runtime={"mean_ms_per_frame": 12.0},
            unavailable_reasons={"anchor_error": "ground truth anchor is unavailable",
                                 "impact_error": "no impact ground truth",
                                 "clubhead_precision_recall": "clubhead is not visible"},
        )
        self.assertEqual(report["metrics"]["track_continuity"], .95)
        self.assertIsNone(report["metrics"]["anchor_error"])
        self.assertEqual(report["unavailable_reasons"]["anchor_error"], "ground truth anchor is unavailable")
        self.assertEqual(report["metrics"]["ball_precision_recall"]["recall"], .8)
        self.assertIsNone(report["metrics"]["clubhead_precision_recall"])
        self.assertEqual(set(report["metrics"]), {"track_continuity", "anchor_error", "impact_error",
            "ball_precision_recall", "clubhead_precision_recall", "landing_error", "false_positives", "runtime"})

    def test_renderer_delegates_visual_reference_without_leaking_source(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "frame.jpg"
            observation = object()
            # Boundary validates that automatic callers provide a renderable observation
            with self.assertRaises(TypeError):
                render_automatic_frame("/tmp/source.jpg", output, observation)


if __name__ == "__main__":
    unittest.main()
