"""Scenario D: "wind_adjusted_dispersion" — crosswind shifts the preferred line.

Course: Crosswind Pines, Hole 6 (Par 5). A water hazard hugs the positive-y
side of the fairway corridor (x 175-235, y 14-18, i.e. the high-y side of the
engine frame — no handedness implied); the 5i lays up at 205yd, beside it.
shot.json declares a 10mph wind toward 90 deg (+y) on a straight +x shot,
i.e. a crosswind toward +y — documented toward convention. The 5i (miss bias
0) is symmetric so the 10mph crosswind drifts the straight line toward +y
into the water, while the -15yd aim compensates away from it.

Observed (N=300, all five seeds 7/42/1234/4242/99999):
  zero wind: recommended 5i_layup205_+0yd at (205, 0), water 0.00
  windy:     recommended 5i_layup205_-15yd at (205, -15), water 0.00;
             the straight +0yd line carries water risk 0.15-0.20.
So the wind changes the recommended target, and the wind-aware line has a
measurably safer cross-track outcome. The test asserts comparative/robust
properties across seeds — never golden floats.
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

DATA = Path(__file__).resolve().parent.parent / "data" / "scenarios" / "wind_adjusted_dispersion"
SHOT, COURSE, PLAYER = DATA / "shot.json", DATA / "hole.json", DATA / "player.json"

SEEDS = (7, 42, 1234, 4242, 99999)
WIND = {"speed_mph": 10.0, "direction_deg": 90.0}   # toward +y, crosswind for a +x shot
ZERO = {"speed_mph": 0.0, "direction_deg": 0.0}


class _WindOverride:
    """Loads the real shot.json but swaps the wind dict, so the production
    wind path is exercised exactly as the CLI would."""

    def __init__(self, base, wind):
        self.path = base.path
        self._base = base
        self._wind = wind

    def load_shot(self):
        shot = self._base.load_shot()
        shot.wind = dict(self._wind)
        return shot


def _run(seed: int, wind: dict):
    config = Config.default()
    config = replace(config, simulation=replace(config.simulation, random_seed=seed))
    course_source = JsonCourseDataSource(COURSE)
    course = course_source.load_course()
    shot_source = JsonShotDataSource(SHOT, course.coordinate_system)
    player_source = JsonPlayerProfileSource(PLAYER)
    return run_pipeline(
        _WindOverride(shot_source, wind), course_source, player_source, config
    )


def _water_result(result, label: str) -> float:
    return next(
        cr.hazard_probabilities.get("water", 0.0)
        for cr in result.candidate_results
        if cr.candidate.label == label
    )


class TestWindAdjustedDispersion(unittest.TestCase):
    def setUp(self):
        self.windy = _run(SEEDS[1], WIND)
        self.calm = _run(SEEDS[1], ZERO)

    def test_wind_metadata_uses_documented_toward_convention(self):
        shot = self.windy.shot
        self.assertEqual(shot.wind["speed_mph"], 10.0)
        self.assertEqual(shot.wind["direction_deg"], 90.0)
        # Straight +x shot: direction 90 = crosswind TOWARD +y.
        self.assertGreater(shot.wind["speed_mph"], 0)

    def test_wind_changes_recommended_target(self):
        self.assertNotEqual(
            (self.calm.recommendation.recommended_target.x,
             self.calm.recommendation.recommended_target.y),
            (self.windy.recommendation.recommended_target.x,
             self.windy.recommendation.recommended_target.y),
        )
        # Calm prefers the straight line; the crosswind line aims INTO the wind.
        self.assertAlmostEqual(self.calm.recommendation.recommended_target.y, 0.0, delta=0.5)
        self.assertLess(self.windy.recommendation.recommended_target.y, 0.0)

    def test_straight_line_is_wind_risky_but_compensated_line_is_safe(self):
        calm_straight = _water_result(self.calm, "5i_layup205_+0yd")
        windy_straight = _water_result(self.windy, "5i_layup205_+0yd")
        windy_compensated = _water_result(self.windy, "5i_layup205_-15yd")
        # Calm straight line is clean; the wind blows it into the water.
        self.assertEqual(calm_straight, 0.0)
        self.assertGreater(windy_straight, 0.05)  # observed 0.15-0.20
        # The compensated line sidesteps the hazard.
        self.assertLess(windy_compensated, 0.03)  # observed 0.00

    def test_wind_aware_line_has_measurably_safer_cross_track_outcome(self):
        windy_straight = _water_result(self.windy, "5i_layup205_+0yd")
        windy_compensated = _water_result(self.windy, "5i_layup205_-15yd")
        self.assertGreater(windy_straight - windy_compensated, 0.10)

    def test_multi_seed_robustness(self):
        for seed in SEEDS:
            windy = _run(seed, WIND)
            calm = _run(seed, ZERO)
            self.assertNotEqual(
                windy.recommendation.recommended_target,
                calm.recommendation.recommended_target,
                f"seed {seed}: wind must change the recommended target",
            )
            self.assertLess(
                windy.recommendation.recommended_target.y, 0.0,
                f"seed {seed}: windy recommendation must aim into the wind",
            )
            # The windy recommendation is the compensated line, which is safe.
            best_water = _water_result(
                windy, f"{windy.recommendation.recommended_club}_layup205_-15yd"
            )
            self.assertLess(best_water, 0.05, f"seed {seed}: windy best line must be safe")

    def test_same_seed_repeated_runs_are_identical(self):
        repeat = _run(SEED, WIND)
        rec = self.windy.recommendation
        self.assertEqual(rec.expected_strokes, repeat.recommendation.expected_strokes)
        self.assertEqual(rec.actual_expected_strokes, repeat.recommendation.actual_expected_strokes)
        self.assertEqual(rec.decision_cost, repeat.recommendation.decision_cost)
        self.assertEqual(rec.recommended_target, repeat.recommendation.recommended_target)


SEED = SEEDS[1]  # primary seed used by setUp


if __name__ == "__main__":
    unittest.main()
