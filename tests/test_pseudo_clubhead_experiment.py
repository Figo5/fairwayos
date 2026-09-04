import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).parents[1] / "out/research_training_gauntlet/run_pseudo_clubhead_experiment.py"
_spec = importlib.util.spec_from_file_location("run_pseudo_clubhead_experiment", SCRIPT)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class TestPseudoClubheadExperiment(unittest.TestCase):
    def test_ball_candidates_skips_oversized_box_and_preserves_valid_box(self):
        result = SimpleNamespace(
            boxes=SimpleNamespace(
                xyxy=__import__("numpy").array([
                    [36.7, 0.0, 1920.0, 1080.0],
                    [1092.6, 813.4, 1156.9, 880.3],
                ]),
                conf=__import__("numpy").array([0.99, 0.8]),
            )
        )

        candidates = _module._ball_candidates(result, 1920, 1080)

        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0]["center"][0], 1124.75)
        self.assertAlmostEqual(candidates[0]["center"][1], 846.85)
        self.assertEqual(candidates[0]["confidence"], 0.8)


if __name__ == "__main__":
    unittest.main()
