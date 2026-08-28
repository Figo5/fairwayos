"""Scenario B: "ob_risk" — fairway target vs. OB risk, stroke-and-distance.

Course: Ridgeline National, Hole 4 (Par 4). An OB strip (150-300 x y=20-40)
sits immediately right of the fairway. Driver (miss_bias=6, lateral stddev 16)
pushes a chunk of its 270yd line over the y=20 boundary; 2i (no bias, tight
dispersion) played straight down the line barely reaches OB at all.

Observed (seed 42, N=300, stable across seeds 7/1234/99999):
  Driver right-most (+15yd) candidate: OB ~0.40, expected ~4.91
  Driver straight (0yd) candidate:      OB ~0.10, expected ~4.07
  2i straight (0yd) candidate:          OB 0.00,  expected ~3.96
  (2i's deliberately lateral +15yd aim DOES reach OB ~0.20-0.30 — so the
  "2i is OB-safe" assertion must target the straight 2i play, not any 2i.)
  OB landings score exactly 2.0 + strokes_from_lie(FAIRWAY, 340.0) = 5.6,
  i.e. stroke-and-distance against the ORIGINAL 340yd, never the landing spot.
"""

import random
import unittest
from dataclasses import replace
from pathlib import Path

from ghostcaddie.adapters.json_file import (
    JsonCourseDataSource,
    JsonPlayerProfileSource,
    JsonShotDataSource,
)
from ghostcaddie.config import Config
from ghostcaddie.dispersion import GaussianDispersionModel
from ghostcaddie.expected_strokes import BaselineTourExpectedStrokesModel
from ghostcaddie.models import RegionType
from ghostcaddie.pipeline import run_pipeline
from ghostcaddie.simulation import Candidate, ShotSimulator

DATA = Path(__file__).resolve().parent.parent / "data" / "scenarios" / "ob_risk"
SHOT, COURSE, PLAYER = DATA / "shot.json", DATA / "hole.json", DATA / "player.json"

SEED = 525  # explicit — do not rely on Config.default()'s seed


def _config(seed: int) -> Config:
    config = Config.default()
    return replace(config, simulation=replace(config.simulation, random_seed=seed))


def _run(seed: int):
    config = _config(seed)
    course_source = JsonCourseDataSource(COURSE)
    course = course_source.load_course()
    shot_source = JsonShotDataSource(SHOT, course.coordinate_system)
    player_source = JsonPlayerProfileSource(PLAYER)
    return run_pipeline(shot_source, course_source, player_source, config)


class TestObRisk(unittest.TestCase):
    def setUp(self):
        self.result = _run(SEED)

    def test_driver_line_has_measurable_ob_risk(self):
        ob_probs = [
            cr.hazard_probabilities.get(RegionType.OUT_OF_BOUNDS, 0.0)
            for cr in self.result.candidate_results
            if cr.candidate.club == "Driver"
        ]
        # Observed max 0.35-0.40 across seeds 42/7/1234/99999; assert well below.
        self.assertGreater(max(ob_probs), 0.05)
        self.assertGreater(max(ob_probs), 0.20)

    def test_straight_2i_play_has_negligible_ob_risk(self):
        straight_2i = [
            c.hazard_probabilities.get(RegionType.OUT_OF_BOUNDS, 0.0)
            for c in self.result.candidate_results
            if c.candidate.club == "2i" and c.candidate.label == "2i_layup210_+0yd"
        ]
        self.assertEqual(len(straight_2i), 1)
        # Observed 0.00 across seeds 42/7/1234/99999.
        self.assertLess(straight_2i[0], 0.03)

    def test_ob_risky_line_costs_more_than_clean_line(self):
        driver = [cr for cr in self.result.candidate_results if cr.candidate.club == "Driver"]
        ob_risky = max(driver, key=lambda cr: cr.hazard_probabilities.get(RegionType.OUT_OF_BOUNDS, 0.0))
        clean = min(
            (cr for cr in self.result.candidate_results
             if cr.candidate.club == "2i" and cr.candidate.label == "2i_layup210_+0yd"),
            key=lambda cr: cr.expected_strokes,
        )
        # Observed ~4.90 vs ~3.96 — the OB outcome (~1.6 strokes over a fairway
        # landing) dominates the spread between the two clubs' lines.
        self.assertGreater(ob_risky.expected_strokes, clean.expected_strokes)
        self.assertGreater(ob_risky.expected_strokes - clean.expected_strokes, 0.25)

    def test_ob_uses_original_distance_end_to_end(self):
        """Prove stroke-and-distance threading in the scenario's real data.

        We install a recording wrapper on the strokes model and evaluate the
        Driver +15yd OB-risky candidate against it. Every OB landing that the
        simulator scores must have been scored with original_distance_to_pin
        = 340.0 (never the useless landing-point distance, which is ~60-130yd),
        producing exactly ob_penalty + strokes_from_lie(FAIRWAY, 340) each time.
        """
        config = _config(SEED)
        course_source = JsonCourseDataSource(COURSE)
        course = course_source.load_course()
        shot_source = JsonShotDataSource(SHOT, course.coordinate_system)
        player = JsonPlayerProfileSource(PLAYER).load_player()

        calls = []

        class Recording(BaselineTourExpectedStrokesModel):
            def expected_strokes_for_landing(self, region, dist, orig):
                value = super().expected_strokes_for_landing(region, dist, orig)
                if region == RegionType.OUT_OF_BOUNDS:
                    calls.append((dist, orig, value))
                return value

        strokes_model = Recording(config.expected_strokes)
        simulator = ShotSimulator(
            GaussianDispersionModel(), strokes_model, course, config.simulation
        )
        rng = random.Random(SEED)
        # Use the SAME aim point the pipeline generates for this candidate so
        # the probe exercises identical geometry.
        candidate = next(c for c in simulator.generate_candidates(shot_source.load_shot(), player)
                         if c.label == "Driver_layup270_+15yd")
        simulator.evaluate_candidate(shot_source.load_shot(), player, candidate, rng)

        self.assertGreater(len(calls), 0, "expected OB landings in the probe")
        baseline = BaselineTourExpectedStrokesModel(config.expected_strokes)
        expected_value = config.expected_strokes.ob_penalty_strokes + baseline.strokes_from_lie(
            RegionType.FAIRWAY, 340.0
        )
        for dist, orig, value in calls:
            self.assertEqual(orig, 340.0,
                             "OB must score from the ORIGINAL tee-to-pin distance")
            self.assertNotEqual(orig, dist, "orig distance must not be the landing distance")
            self.assertEqual(value, expected_value)

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
