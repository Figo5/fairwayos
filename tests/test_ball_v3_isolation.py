"""Isolation guards for the v3 research pass.

Two things have to stay true for a v3 number to mean anything, and both are
enforced here rather than remembered:

* the v3 final-test split stays sealed until it is deliberately consumed, and
  nothing that trains can reach the v3 evaluation corpus at all;
* the canonical pipeline does not import any of this. The ball layer is
  ``UNAVAILABLE`` on the canonical path, and it stays that way unless a pass is
  classified PROMOTED -- which this one may not be.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

V3_MODULES = ("ball_v3_blobs", "ball_v3_body", "ball_v3_context",
              "ball_v3_dataset", "ball_v3_proposals", "ball_v3_ranker",
              "ball_v3_tracker", "ball_v3_tracklets")

CANONICAL_MODULES = ("canonical", "canonical_worker")


def load_tool(name: str):
    """Import a research CLI as a module without executing its main()."""
    spec = importlib.util.spec_from_file_location(
        f"v3_tool_{name}", ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FinalTestStaysSealed(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = load_tool("ball_v3_eval")

    def test_final_test_is_refused_without_the_explicit_flag(self):
        with self.assertRaises(SystemExit) as caught:
            self.tool.load_split("final_test", False)
        self.assertIn("sealed", str(caught.exception))

    def test_final_test_opens_only_with_the_flag_and_matches_the_frozen_split(self):
        clips = self.tool.load_split("final_test", True)
        frozen = {c["clip_id"] for c in json.loads(
            (ROOT / "data/ball_dataset_v3_eval/split.json").read_text("utf-8"))["clips"]
            if c["split"] == "final_test"}
        # only clips whose media is on disk can be returned, so this is a subset
        self.assertTrue({c["clip_id"] for c in clips} <= frozen)
        self.assertTrue(clips, "no final-test media on disk")

    def test_validation_never_returns_a_final_test_clip(self):
        split = json.loads(
            (ROOT / "data/ball_dataset_v3_eval/split.json").read_text("utf-8"))
        sealed = {c["clip_id"] for c in split["clips"] if c["split"] == "final_test"}
        for clip in self.tool.load_split("validation", False):
            self.assertNotIn(clip["clip_id"], sealed)

    def test_the_configuration_hash_covers_what_must_be_frozen(self):
        """A final-test number is only reproducible if the configuration that
        produced it is recorded, so the digest must move when any of it moves."""
        from ghostcaddie.video import ball_v3_proposals as bp
        from ghostcaddie.video import ball_v3_tracker as btrack
        missing = Path("/nonexistent/weights.pt")
        base = self.tool.configuration_hash(
            btrack.TrackerParams(), bp.ProposalParams(), missing, None)
        changed = self.tool.configuration_hash(
            btrack.TrackerParams(min_margin=0.99), bp.ProposalParams(), missing, None)
        self.assertNotEqual(base["configuration_sha256"],
                            changed["configuration_sha256"])
        self.assertIn("tracker_params", base["configuration"])
        self.assertIn("proposal_params", base["configuration"])
        self.assertIn("generators", base["configuration"])


class TrainingCannotSeeTheEvaluationCorpus(unittest.TestCase):
    def test_the_trainer_only_reads_v2_train_clips(self):
        tool = load_tool("ball_v3_train")
        split = json.loads(
            (ROOT / "data/ball_dataset_v2/split.json").read_text("utf-8"))
        allowed = {c["clip_id"] for c in split["clips"] if c["split"] == "train"}
        for clip in tool.train_clips():
            self.assertIn(clip["clip_id"], allowed)

    def test_no_training_code_mentions_the_v3_evaluation_corpus(self):
        source = (ROOT / "tools" / "ball_v3_train.py").read_text("utf-8")
        self.assertNotIn("ball_dataset_v3_eval", source,
                         "the trainer must not be able to reach the evaluation "
                         "corpus at all")


class CanonicalStaysIsolated(unittest.TestCase):
    def test_canonical_modules_do_not_import_v3_research_code(self):
        for name in CANONICAL_MODULES:
            source = (ROOT / "ghostcaddie" / "video" / f"{name}.py").read_text("utf-8")
            for module in V3_MODULES:
                self.assertNotIn(module, source,
                                 f"{name}.py imports {module}: the v3 pass is "
                                 "research-only unless it is PROMOTED")

    def test_the_canonical_ball_layer_is_still_unavailable(self):
        from ghostcaddie.video import canonical
        layer = canonical.ball_layer()
        self.assertEqual(layer.state, "unavailable")
        self.assertEqual(layer.method, "none")
        self.assertEqual(layer.coverage["frames_with_observation"], 0)
        self.assertEqual(layer.provenance["routes_admitted"], [])

    def test_v3_results_are_never_marked_production_eligible(self):
        from ghostcaddie.video import ball_v3_tracker as btrack
        provenance = btrack._provenance(None, {}, None)
        self.assertTrue(provenance["research_only"])
        self.assertFalse(provenance["ground_truth"])
        self.assertTrue(provenance["no_network"])


if __name__ == "__main__":
    unittest.main()
