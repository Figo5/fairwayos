"""Scenario C: "lie_dispersion" — bunker/rough/green lie differences.

Course: Sandhaven, Hole 15 (Par 4). The same PW shot (start 80,0; pin 155,0;
distance 75yd) is evaluated three times with the ONLY difference being the
shot's lie, which drives lie_modifier multipliers:
  fairway: carry x1.00, stddev x1.00
  rough:   carry x0.93, stddev x1.30
  bunker:  carry x0.78, stddev x1.80
Observed (seed 42, N=300, stable across seeds 7/1234/99999):
  recommended expected_strokes: fairway 2.58-2.61 <= rough 2.68-2.70
                                <= bunker 3.12-3.17
  green-hit probability: fairway 0.95-1.00 >= rough ~0.90 >= bunker 0.45-0.50
"""

import unittest
from dataclasses import replace
from pathlib import Path

from ghostcaddie.adapters.json_file import (
    JsonCourseDataSource,
    JsonPlayerProfileSource,
    JsonShotDataSource,
)
from ghostcaddie.config import Config
from ghostcaddie.pipeline import run_pipeline

DATA = Path(__file__).resolve().parent.parent / "data" / "scenarios" / "lie_dispersion"
COURSE, PLAYER = DATA / "hole.json", DATA / "player.json"

SEED = 626  # explicit — do not rely on Config.default()'s seed


def _run(shot_name: str, seed: int):
    config = Config.default()
    config = replace(config, simulation=replace(config.simulation, random_seed=seed))
    course_source = JsonCourseDataSource(COURSE)
    course = course_source.load_course()
    shot_source = JsonShotDataSource(DATA / shot_name, course.coordinate_system)
    player_source = JsonPlayerProfileSource(PLAYER)
    return run_pipeline(shot_source, course_source, player_source, config)


class TestLieDispersion(unittest.TestCase):
    def setUp(self):
        self.fairway = _run("shot_fairway.json", SEED)
        self.rough = _run("shot_rough.json", SEED)
        self.bunker = _run("shot_bunker.json", SEED)

    def test_expected_strokes_degrade_fairway_le_rough_le_bunker(self):
        e = lambda r: r.recommendation.expected_strokes  # noqa: E731
        self.assertLessEqual(e(self.fairway), e(self.rough))
        self.assertLessEqual(e(self.rough), e(self.bunker))
        # Observed ~2.58 / ~2.68 / ~3.16 — comfortably strict in practice.
        self.assertGreater(e(self.bunker) - e(self.fairway), 0.25)

    def test_green_hit_probability_fairway_ge_rough_ge_bunker(self):
        g = lambda r: r.recommendation.hazard_probabilities.get("green", 0.0)  # noqa: E731
        self.assertGreaterEqual(g(self.fairway), g(self.rough))
        self.assertGreaterEqual(g(self.rough), g(self.bunker))
        # Observed 0.95-1.00 / ~0.90 / 0.45-0.50 — strictly monotone in practice.
        self.assertGreater(g(self.fairway), g(self.bunker))

    def test_deterministic_across_identical_seed_runs(self):
        second = _run("shot_fairway.json", SEED)
        self.assertEqual(self.fairway.recommendation.expected_strokes,
                         second.recommendation.expected_strokes)
        self.assertEqual(self.fairway.recommendation.actual_expected_strokes,
                         second.recommendation.actual_expected_strokes)
        self.assertEqual(self.fairway.recommendation.decision_cost,
                         second.recommendation.decision_cost)


if __name__ == "__main__":
    unittest.main()
