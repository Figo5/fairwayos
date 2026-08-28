"""Scenario A: "layup_vs_attack" — safe layup vs. aggressive attack near water.

Course: Blackwater Links, Hole 11 (Par 5). A water hazard (235-268yd) guards
the green frontage; Hybrid carry is 200yd so the aggressive "pin" line lands
in or around the water, while 7i/PW lay up ~150/120yd, well short of it.

Observed (seed 42, N=300, verified stable across seeds 7/1234/99999):
  Hybrid (pin) candidates: water probability 0.50-0.65, expected 4.06-4.14
  7i / PW (layup) candidates: water probability 0.00,     expected 3.70-3.89
  recommended = 7i (expected 3.70) — strictly better than the best Hybrid.
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
from ghostcaddie.models import RegionType
from ghostcaddie.pipeline import run_pipeline

DATA = Path(__file__).resolve().parent.parent / "data" / "scenarios" / "layup_vs_attack"
SHOT, COURSE, PLAYER = DATA / "shot.json", DATA / "hole.json", DATA / "player.json"

SEED = 424  # explicit — do not rely on Config.default()'s seed


def _run(seed: int):
    config = Config.default()
    config = replace(config, simulation=replace(config.simulation, random_seed=seed))
    course_source = JsonCourseDataSource(COURSE)
    course = course_source.load_course()
    shot_source = JsonShotDataSource(SHOT, course.coordinate_system)
    player_source = JsonPlayerProfileSource(PLAYER)
    return run_pipeline(shot_source, course_source, player_source, config)


class TestLayupVsAttack(unittest.TestCase):
    def setUp(self):
        self.result = _run(SEED)

    def test_aggressive_club_candidate_has_measurable_water_risk(self):
        water_probs = [
            cr.hazard_probabilities.get(RegionType.WATER, 0.0)
            for cr in self.result.candidate_results
            if cr.candidate.club == "Hybrid"
        ]
        # Observed max ~0.65 at N=300 (stable across seeds 42/7/1234/99999).
        self.assertGreater(max(water_probs), 0.05)
        self.assertGreater(max(water_probs), 0.30)  # observed 0.60-0.65

    def test_layup_clubs_have_effectively_zero_water_risk(self):
        layup_water = [
            cr.hazard_probabilities.get(RegionType.WATER, 0.0)
            for cr in self.result.candidate_results
            if cr.candidate.club in ("7i", "PW")
        ]
        # 7i/PW lay up ~60yd short of the water's leading edge at 235.
        for w in layup_water:
            self.assertLess(w, 0.02, "a layup club must not be water-risky")

    def test_recommendation_prefers_safe_layup_over_water_risky_attack(self):
        hybrid = [cr for cr in self.result.candidate_results if cr.candidate.club == "Hybrid"]
        water_risky_hybrid = max(hybrid, key=lambda cr: cr.hazard_probabilities.get(RegionType.WATER, 0.0))
        rec = self.result.recommendation
        # The engine is min-by-expected-strokes; the attacking line carries
        # expected ~4.1 strokes while the safe layup is ~3.7. The robust,
        # non-brittle property is that the recommended option beats the
        # water-risky line on expected strokes (and is not that line).
        self.assertLess(rec.expected_strokes, water_risky_hybrid.expected_strokes)
        self.assertNotEqual(rec.recommended_club, "Hybrid")

    def test_deterministic_across_identical_seed_runs(self):
        second = _run(SEED)
        self.assertEqual(self.result.recommendation.expected_strokes,
                         second.recommendation.expected_strokes)
        self.assertEqual(self.result.recommendation.actual_expected_strokes,
                         second.recommendation.actual_expected_strokes)
        self.assertEqual(self.result.recommendation.decision_cost,
                         second.recommendation.decision_cost)


if __name__ == "__main__":
    unittest.main()
