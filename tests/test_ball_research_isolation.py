"""The research ball work must stay out of the canonical workflow.

Promotion is a decision recorded in ``docs/ball-tracking-results.md``, not
something that happens by accident because an import crept in. These tests fail
loudly if the canonical path starts depending on the research tracker, or if the
canonical ball layer stops being UNAVAILABLE, so that a future change has to
edit an assertion on purpose rather than slide past review.
"""

import ast
import json
import unittest
from pathlib import Path

from ghostcaddie.video import canonical
from ghostcaddie.video.ball_annotations import annotation_summary, load_annotation
from ghostcaddie.video.research_split import validate_split_manifest

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data" / "ball_annotations"
SPLIT = ROOT / "data" / "ball_split.json"


class CanonicalUnchangedTests(unittest.TestCase):
    def test_the_canonical_ball_layer_is_still_unavailable(self):
        layer = canonical.ball_layer()
        self.assertEqual(layer.state, canonical.LayerState.UNAVAILABLE)
        self.assertEqual(layer.method, "none")
        self.assertEqual(layer.coverage["frames_with_observation"], 0)
        self.assertTrue(layer.limitations)

    def test_the_canonical_module_does_not_import_the_research_tracker(self):
        tree = ast.parse((ROOT / "ghostcaddie" / "video" / "canonical.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
                imported.update(f"{node.module}.{alias.name}" for alias in node.names)
        offenders = {name for name in imported if "ball_pipeline" in name or "ball_evaluation" in name}
        self.assertFalse(offenders, f"canonical.py imported research ball code: {offenders}")

    def test_physical_metrics_still_refuse_to_infer_anything_from_a_ball(self):
        metrics = canonical.physical_metrics(None)
        flat = [metric for group in metrics.values() for metric in group]
        ball_metrics = [m for m in flat if m.name in
                        {"ball_speed", "launch_angle", "carry_distance", "apex_height"}]
        self.assertTrue(ball_metrics)
        self.assertEqual(metrics["supported"], [])
        for metric in ball_metrics:
            self.assertEqual(metric.state, canonical.MetricState.UNAVAILABLE)
            self.assertIsNone(metric.median)


class DatasetIntegrityTests(unittest.TestCase):
    def test_the_split_manifest_is_frozen_and_leak_free(self):
        manifest = validate_split_manifest(json.loads(SPLIT.read_text(encoding="utf-8")))
        self.assertEqual(manifest["status"], "frozen")
        splits = {clip["split"] for clip in manifest["clips"]}
        self.assertEqual(splits, {"train", "validation", "held_out"})

    def test_every_clip_in_the_split_has_labels_bound_to_the_same_bytes(self):
        manifest = validate_split_manifest(json.loads(SPLIT.read_text(encoding="utf-8")))
        for clip in manifest["clips"]:
            path = ANNOTATIONS / f"{clip['clip_id']}.json"
            self.assertTrue(path.exists(), f"missing labels for {clip['clip_id']}")
            document = load_annotation(path)
            self.assertEqual(document["source_sha256"], clip["sha256"],
                             f"{clip['clip_id']} labels are bound to different bytes "
                             "than the split manifest records")

    def test_labels_are_recorded_as_consecutive_runs_not_scattered_singletons(self):
        # Continuity and identity-switch metrics are meaningless on isolated
        # frames, so the dataset is required to carry real sequences.
        for path in sorted(ANNOTATIONS.glob("*.json")):
            summary = annotation_summary(load_annotation(path))
            self.assertTrue(summary["runs"], path.name)
            self.assertGreaterEqual(max(run["length"] for run in summary["runs"]), 10, path.name)

    def test_no_label_file_carries_coordinates_on_a_not_visible_frame(self):
        for path in sorted(ANNOTATIONS.glob("*.json")):
            document = load_annotation(path)
            for key, entry in document["frames"].items():
                if not entry["visible"]:
                    self.assertNotIn("x", entry, f"{path.name} frame {key}")
                    self.assertTrue(entry.get("reason"), f"{path.name} frame {key}")


if __name__ == "__main__":
    unittest.main()
