"""Split isolation for the v2 research pass.

The final test split is only meaningful if it stays sealed, and "sealed" has to be
enforced by the code rather than by remembering. These tests check the guards that
stand between development and the held-out split.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghostcaddie.video import ball_detector as bdet  # noqa: E402
from ghostcaddie.video import ball_synthetic as bs  # noqa: E402


def load_cli():
    """Import the research CLI as a module without executing its main()."""
    spec = importlib.util.spec_from_file_location("ball_track_v2_cli",
                                                  ROOT / "tools" / "ball_track_v2.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ball_track_v2_cli"] = module
    spec.loader.exec_module(module)
    return module


class SplitIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = load_cli()

    def test_final_test_is_refused_without_the_explicit_flag(self):
        with self.assertRaises(SystemExit) as caught:
            self.cli.clips_for("final_test")
        self.assertIn("sealed", str(caught.exception))

    def test_final_test_only_opens_with_the_flag(self):
        with self.assertRaises(SystemExit):
            self.cli.clips_for("final_test", allow_final_test=False)
        clips = self.cli.clips_for("final_test", allow_final_test=True)
        self.assertTrue(clips)
        split = json.loads((ROOT / "data" / "ball_dataset_v2" / "split.json").read_text())
        frozen = {clip["clip_id"] for clip in split["clips"]
                  if clip["split"] == "final_test"}
        self.assertEqual({clip["clip_id"] for clip in clips}, frozen)

    def test_synthetic_training_refuses_a_validation_clip(self):
        split = json.loads((ROOT / "data" / "ball_dataset_v2" / "split.json").read_text())
        validation = [clip["clip_id"] for clip in split["clips"]
                      if clip["split"] == "validation"]
        self.assertTrue(validation)
        with self.assertRaises(ValueError) as caught:
            bs.assert_training_clips(split, validation)
        message = str(caught.exception)
        self.assertIn("non-train", message)
        self.assertIn(validation[0], message,
                      "the refusal has to name the clip it is refusing")

    def test_synthetic_training_refuses_a_final_test_clip(self):
        split = json.loads((ROOT / "data" / "ball_dataset_v2" / "split.json").read_text())
        final_test = [clip["clip_id"] for clip in split["clips"]
                      if clip["split"] == "final_test"]
        with self.assertRaises(ValueError) as caught:
            bs.assert_training_clips(split, final_test)
        message = str(caught.exception)
        self.assertIn("non-train", message)
        self.assertIn(final_test[0], message)

    def test_synthetic_training_accepts_train_clips(self):
        split = json.loads((ROOT / "data" / "ball_dataset_v2" / "split.json").read_text())
        train = [clip["clip_id"] for clip in split["clips"] if clip["split"] == "train"]
        bs.assert_training_clips(split, train)          # must not raise

    def test_the_evaluation_seed_differs_from_the_training_seed(self):
        self.assertNotEqual(self.cli.TRAIN_SEED, self.cli.EVAL_SEED,
                            "evaluating on the training seed would silently reuse the "
                            "same insertions the model was trained on")

    def test_the_noise_and_blur_model_is_not_silently_optional(self):
        """A generator that quietly skipped its degradation would still 'work'."""
        import inspect
        source = inspect.getsource(bs)
        self.assertIn("def _degrade", source)
        self.assertIn("noise", source)

    def test_a_downloader_is_never_reached_from_the_tracker(self):
        """No network in the inference path."""
        import ast
        for name in ("ball_tracker_v2.py", "ball_detector.py", "ball_candidates.py",
                     "ball_consensus.py"):
            tree = ast.parse((ROOT / "ghostcaddie" / "video" / name).read_text())
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
                    imported.update(f"{node.module}.{alias.name}"
                                    for alias in node.names)
            offenders = {item for item in imported
                         if any(key in item.lower() for key in
                                ("requests", "urllib", "httpx", "socket", "youtube",
                                 "download"))}
            self.assertFalse(offenders, f"{name} can reach the network: {offenders}")

    def test_model_loading_is_data_not_code(self):
        import inspect
        self.assertIn("weights_only=True", inspect.getsource(bdet.load_detector))


if __name__ == "__main__":
    unittest.main()
