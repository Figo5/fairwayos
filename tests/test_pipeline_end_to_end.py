import json
import random
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from ghostcaddie.adapters.json_file import (
    JsonCourseDataSource,
    JsonPlayerProfileSource,
    JsonShotDataSource,
)
from ghostcaddie.config import Config
from ghostcaddie.dispersion import DispersionModel, GaussianDispersionModel
from ghostcaddie.pipeline import PipelineResult, run_pipeline
from ghostcaddie.simulation import ShotSimulator

DATA = Path(__file__).resolve().parent.parent / "data"
SHOT, COURSE, PLAYER = DATA / "sample_shot.json", DATA / "sample_hole.json", DATA / "sample_player.json"


class TestPipelineEndToEnd(unittest.TestCase):
    def setUp(self):
        self.config = Config.default()
        self.course_source = JsonCourseDataSource(COURSE)
        self.course = self.course_source.load_course()
        self.shot_source = JsonShotDataSource(SHOT, self.course.coordinate_system)
        self.player_source = JsonPlayerProfileSource(PLAYER)
        self.result = run_pipeline(self.shot_source, self.course_source, self.player_source, self.config)

    def test_recommendation_is_structurally_complete(self):
        rec = self.result.recommendation
        self.assertIsInstance(rec.recommended_club, str)
        self.assertTrue(rec.recommended_club)
        self.assertIsInstance(rec.expected_strokes, float)
        self.assertIsInstance(rec.actual_expected_strokes, float)
        self.assertIsInstance(rec.decision_cost, float)
        self.assertTrue(2.0 <= rec.expected_strokes <= 5.0)
        self.assertTrue(2.0 <= rec.actual_expected_strokes <= 5.0)
        self.assertIsInstance(rec.hazard_probabilities, dict)
        self.assertTrue(rec.hazard_probabilities)
        self.assertIn(rec.confidence, ("low", "medium", "high"))
        self.assertIsInstance(rec.explanation, str)
        self.assertTrue(rec.explanation)
        self.assertIsInstance(rec.provenance, dict)
        self.assertIn("data_disclaimer", rec.provenance)
        self.assertIn("Synthetic", rec.provenance["data_disclaimer"])
        # Note: decision_cost sign is NOT asserted — narrative, not invariant.

    def test_candidates_generated_and_evaluated(self):
        self.assertGreaterEqual(len(self.result.candidate_results), 10)
        for cr in self.result.candidate_results:
            self.assertEqual(cr.sample_count, self.config.simulation.monte_carlo_samples)
            self.assertGreater(cr.expected_strokes, 0)

    def test_recommendation_ranked_best(self):
        best_score = self.result.recommendation.expected_strokes
        all_scores = [cr.expected_strokes for cr in self.result.candidate_results]
        self.assertAlmostEqual(best_score, min(all_scores))

    def test_svg_is_nonempty_and_well_formed(self):
        svg = self.result.svg
        self.assertIsInstance(svg, str)
        self.assertTrue(svg)
        self.assertIn("<svg", svg)
        self.assertIn("</svg>", svg)
        self.assertIn("polygon", svg)

    def test_cli_writes_files_via_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            cmd = [
                sys.executable, "-m", "ghostcaddie", "run",
                "--shot", str(SHOT), "--course", str(COURSE),
                "--player", str(PLAYER), "--out", str(out_dir),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=str(Path(__file__).resolve().parent.parent))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            rec_path = out_dir / "recommendation.json"
            svg_path = out_dir / "overlay.svg"
            self.assertTrue(rec_path.exists())
            self.assertTrue(svg_path.exists())
            rec = json.loads(rec_path.read_text())
            self.assertIn("recommended_club", rec)
            self.assertIn("expected_strokes", rec)
            self.assertIn("decision_cost", rec)
            self.assertIn("explanation", rec)
            self.assertIn("data_disclaimer", rec["provenance"])
            svg_text = svg_path.read_text()
            self.assertIn("<svg", svg_text)
            # stdout should contain the terminal summary
            self.assertIn("RECOMMENDATION vs ACTUAL", proc.stdout)


class TestFourPointAdapterBoundary(unittest.TestCase):
    """Four-point calibration: adapter maps source-image -> engine once."""

    def setUp(self):
        self.config = Config.default()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)

        # Affine four-point calibration: source pixels -> engine yards.
        # (100,80)->(0,0), (900,80)->(300,0), (900,620)->(300,200), (100,620)->(0,200)
        course_raw = json.loads(COURSE.read_text())
        course_raw["coordinate_system"] = {
            "mode": "four_point",
            "units": "yards",
            "source_units": "pixels",
            "source_points": [
                {"x": 100, "y": 80}, {"x": 900, "y": 80},
                {"x": 900, "y": 620}, {"x": 100, "y": 620},
            ],
            "engine_points": [
                {"x": 0, "y": 0}, {"x": 300, "y": 0},
                {"x": 300, "y": 200}, {"x": 0, "y": 200},
            ],
        }
        self.course_path = root / "hole.json"
        self.course_path.write_text(json.dumps(course_raw))

        shot_raw = json.loads(SHOT.read_text())
        shot_raw["start_position"] = {"x": 300, "y": 200}
        shot_raw["target_position"] = {"x": 700, "y": 500}
        shot_raw["actual_landing_position"] = {"x": 600, "y": 400}
        self.shot_path = root / "shot.json"
        self.shot_path.write_text(json.dumps(shot_raw))

        self.course = JsonCourseDataSource(self.course_path).load_course()
        self.shot = JsonShotDataSource(self.shot_path, self.course.coordinate_system).load_shot()

    @staticmethod
    def _affine(u, v):
        # Closed-form affine fit to the four correspondences above.
        return 0.375 * (u - 100.0), (200.0 / 540.0) * (v - 80.0)

    def test_course_mode_and_shot_engine_coordinates(self):
        self.assertEqual(self.course.coordinate_system.mode, "four_point")
        self.assertEqual(self.course.coordinate_system.source_units, "pixels")
        for raw, got in [
            ({"x": 300, "y": 200}, self.shot.start_position),
            ({"x": 700, "y": 500}, self.shot.target_position),
            ({"x": 600, "y": 400}, self.shot.actual_landing_position),
        ]:
            ex, ey = self._affine(raw["x"], raw["y"])
            self.assertAlmostEqual(got.x, ex, places=6)
            self.assertAlmostEqual(got.y, ey, places=6)

    def test_reverse_mapping_returns_source_point(self):
        from ghostcaddie.geometry import CoordinateMapper

        mapper = CoordinateMapper(self.course.coordinate_system)
        back = mapper.from_engine(self.shot.start_position)
        self.assertAlmostEqual(back["x"], 300.0, places=6)
        self.assertAlmostEqual(back["y"], 200.0, places=6)

    def test_pipeline_runs_unchanged_with_four_point_course(self):
        result = run_pipeline(
            JsonShotDataSource(self.shot_path, self.course.coordinate_system),
            JsonCourseDataSource(self.course_path),
            JsonPlayerProfileSource(PLAYER),
            self.config,
        )
        self.assertIsInstance(result.recommendation.recommended_club, str)
        self.assertIn("<svg", result.svg)


class TestWindWiring(unittest.TestCase):
    """Production path threads ShotEvent.wind and the active coefficients."""

    def setUp(self):
        self.config = Config.default()
        self.course_source = JsonCourseDataSource(COURSE)
        self.course = self.course_source.load_course()
        self.shot_source = JsonShotDataSource(SHOT, self.course.coordinate_system)
        self.player_source = JsonPlayerProfileSource(PLAYER)
        self.shot = self.shot_source.load_shot()

    def test_every_dispersion_call_receives_shot_wind(self):
        seen = []

        class Recording(DispersionModel):
            def sample_landing(self, start, aim, club, lie_mod, rng, wind=None):
                seen.append(wind)
                return GaussianDispersionModel().sample_landing(
                    start, aim, club, lie_mod, rng, wind
                )

        from ghostcaddie.expected_strokes import BaselineTourExpectedStrokesModel
        simulator = ShotSimulator(
            Recording(),
            BaselineTourExpectedStrokesModel(self.config.expected_strokes),
            self.course,
            self.config.simulation,
        )
        rng = random.Random(self.config.simulation.random_seed)
        simulator.run(self.shot, self.player_source.load_player(), rng)

        self.assertGreater(len(seen), 0)
        self.assertTrue(all(w == self.shot.wind for w in seen))

    def test_pipeline_builds_model_with_active_config_coefficients(self):
        # The pipeline must construct its dispersion model with the ACTIVE
        # simulation config so custom coefficient values take effect. Probe the
        # construction seam directly: capture the config argument.
        import ghostcaddie.pipeline as pipeline_module

        custom = replace(
            self.config,
            simulation=replace(
                self.config.simulation,
                along_wind_carry_yd_per_mph=3.5,
                crosswind_lateral_drift_yd_per_mph=2.25,
            ),
        )
        captured = []

        original_cls = pipeline_module.GaussianDispersionModel

        class Capturing(original_cls):
            def __init__(self, config=None):
                captured.append(config)
                super().__init__(config)

        pipeline_module.GaussianDispersionModel = Capturing
        try:
            result = run_pipeline(
                self.shot_source, self.course_source, self.player_source, custom
            )
        finally:
            pipeline_module.GaussianDispersionModel = original_cls
        self.assertIsInstance(result, PipelineResult)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0], custom.simulation)
        self.assertEqual(captured[0].along_wind_carry_yd_per_mph, 3.5)
        self.assertEqual(captured[0].crosswind_lateral_drift_yd_per_mph, 2.25)


if __name__ == "__main__":
    unittest.main()
