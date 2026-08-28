"""Session orchestration tests: helper equivalence, envelope validation,
normalized in-memory sources, run_session aggregation, and the session CLI."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from ghostcaddie.adapters.base import (
    CourseDataSource,
    PlayerProfileSource,
    ShotDataSource,
)
from ghostcaddie.adapters.json_file import (
    JsonCourseDataSource,
    JsonPlayerProfileSource,
    JsonShotDataSource,
    _parse_course,
    _parse_player,
    _parse_shot,
)
from ghostcaddie.config import Config
from ghostcaddie.geometry import CoordinateMapper, CoordinateSystem, Point2D
from ghostcaddie.pipeline import run_pipeline
from ghostcaddie.session import (
    SESSION_SCHEMA_VERSION,
    InMemoryCourseSource,
    InMemoryPlayerSource,
    InMemoryShotSource,
    NormalizedShot,
    SessionInput,
    derive_shot_seed,
    parse_session,
    run_session,
    serialize_session_report,
)

DATA = Path(__file__).resolve().parent.parent / "data"
SHOT, COURSE, PLAYER = DATA / "sample_shot.json", DATA / "sample_hole.json", DATA / "sample_player.json"
SESSION_FIXTURE = DATA / "sample_session.json"


def _load_fixture() -> dict:
    return json.loads(SESSION_FIXTURE.read_text())


def _parse_fixture() -> SessionInput:
    return parse_session(_load_fixture())


# --- Task 1: reusable parsing helpers ---


class TestParseHelpersEquivalence(unittest.TestCase):
    """The extracted helpers must normalize the existing sample records identically."""

    def test_course_helper_matches_json_adapter(self):
        raw = json.loads(COURSE.read_text())
        via_adapter = JsonCourseDataSource(COURSE).load_course()
        via_helper = _parse_course(raw)
        self.assertEqual(via_helper.name, via_adapter.name)
        self.assertEqual(via_helper.par, via_adapter.par)
        self.assertEqual(via_helper.coordinate_system, via_adapter.coordinate_system)
        self.assertEqual(via_helper.fairway, via_adapter.fairway)
        self.assertEqual(via_helper.green, via_adapter.green)
        self.assertEqual(via_helper.bunkers, via_adapter.bunkers)
        self.assertEqual(via_helper.water_hazards, via_adapter.water_hazards)
        self.assertEqual(via_helper.out_of_bounds, via_adapter.out_of_bounds)
        self.assertEqual(via_helper.pin_position, via_adapter.pin_position)

    def test_shot_helper_matches_json_adapter(self):
        raw = json.loads(SHOT.read_text())
        course = JsonCourseDataSource(COURSE).load_course()
        mapper = CoordinateMapper(course.coordinate_system)
        via_adapter = JsonShotDataSource(SHOT, course.coordinate_system).load_shot()
        via_helper = _parse_shot(raw, mapper)
        self.assertEqual(via_helper, via_adapter)

    def test_player_helper_matches_json_adapter(self):
        raw = json.loads(PLAYER.read_text())
        via_adapter = JsonPlayerProfileSource(PLAYER).load_player()
        via_helper = _parse_player(raw)
        self.assertEqual(via_helper.player_id, via_adapter.player_id)
        self.assertEqual(via_helper.clubs, via_adapter.clubs)
        self.assertEqual(via_helper.lie_modifiers, via_adapter.lie_modifiers)

    def test_missing_required_field_rejected(self):
        raw = json.loads(COURSE.read_text())
        del raw["pin_position"]
        with self.assertRaises(ValueError):
            _parse_course(raw)
        shot_raw = json.loads(SHOT.read_text())
        del shot_raw["club"]
        with self.assertRaises(ValueError):
            _parse_shot(shot_raw, CoordinateMapper(CoordinateSystem()))
        player_raw = json.loads(PLAYER.read_text())
        del player_raw["player_id"]
        with self.assertRaises(ValueError):
            _parse_player(player_raw)

    def test_wrong_container_types_rejected(self):
        raw = json.loads(COURSE.read_text())
        raw["fairway"] = "not-a-list"
        with self.assertRaises(ValueError):
            _parse_course(raw)
        raw = json.loads(COURSE.read_text())
        raw["coordinate_system"] = "manual"
        with self.assertRaises(ValueError):
            _parse_course(raw)
        shot_raw = json.loads(SHOT.read_text())
        shot_raw["wind"] = [1, 2]
        with self.assertRaises(ValueError):
            _parse_shot(shot_raw, CoordinateMapper(CoordinateSystem()))

    def test_non_empty_string_ids_rejected(self):
        shot_raw = json.loads(SHOT.read_text())
        shot_raw["event_id"] = ""
        with self.assertRaises(ValueError):
            _parse_shot(shot_raw, CoordinateMapper(CoordinateSystem()))
        shot_raw = json.loads(SHOT.read_text())
        shot_raw["player_id"] = 42
        with self.assertRaises(ValueError):
            _parse_shot(shot_raw, CoordinateMapper(CoordinateSystem()))

    def test_positive_integral_ordinals_rejected(self):
        shot_raw = json.loads(SHOT.read_text())
        shot_raw["hole_number"] = 7.5
        with self.assertRaises(ValueError):
            _parse_shot(shot_raw, CoordinateMapper(CoordinateSystem()))
        shot_raw = json.loads(SHOT.read_text())
        shot_raw["shot_number"] = 0
        with self.assertRaises(ValueError):
            _parse_shot(shot_raw, CoordinateMapper(CoordinateSystem()))
        shot_raw = json.loads(SHOT.read_text())
        shot_raw["shot_number"] = True
        with self.assertRaises(ValueError):
            _parse_shot(shot_raw, CoordinateMapper(CoordinateSystem()))

    def test_recursive_finite_value_rejection(self):
        raw = json.loads(COURSE.read_text())
        raw["fairway"][0][0]["x"] = float("nan")
        with self.assertRaises(ValueError):
            _parse_course(raw)
        raw = json.loads(COURSE.read_text())
        raw["pin_position"]["y"] = float("inf")
        with self.assertRaises(ValueError):
            _parse_course(raw)
        shot_raw = json.loads(SHOT.read_text())
        shot_raw["wind"]["speed_mph"] = float("nan")
        with self.assertRaises(ValueError):
            _parse_shot(shot_raw, CoordinateMapper(CoordinateSystem()))
        shot_raw = json.loads(SHOT.read_text())
        shot_raw["distance_to_pin"] = float("inf")
        with self.assertRaises(ValueError):
            _parse_shot(shot_raw, CoordinateMapper(CoordinateSystem()))


# --- Task 2: envelope validation and normalized model ---


class TestParseSessionEnvelope(unittest.TestCase):
    def test_parses_approved_envelope_shape(self):
        s = _parse_fixture()
        self.assertEqual(s.schema_version, SESSION_SCHEMA_VERSION)
        self.assertEqual(s.session_id, "SYN-SESSION-2026-001")
        self.assertEqual(s.tournament_id, "SYN-2026-EXHIBITION-01")
        self.assertEqual(s.player_id, "SYNTH-PLAYER-001")
        self.assertEqual(s.course_id, "SYN-COURSE-001")
        self.assertEqual(s.round_number, 1)
        self.assertEqual(s.seed, 42)
        self.assertEqual(sorted(s.courses), [7, 8])
        self.assertEqual([ns.shot_id for ns in s.shots],
                         ["SHOT-7-1", "SHOT-7-2", "SHOT-8-1", "SHOT-8-2"])

    def test_shared_top_level_player_and_course(self):
        s = _parse_fixture()
        self.assertEqual(s.player.player_id, "SYNTH-PLAYER-001")
        self.assertIn("7i", s.player.clubs)
        self.assertEqual(s.courses[7].name, "Ghost Ridge GC — Hole 7")
        self.assertEqual(s.courses[8].par, 4)

    def test_hole_lookup_and_normalized_shot_identity(self):
        s = _parse_fixture()
        ns = s.shots[0]
        self.assertEqual(ns.hole_number, 7)
        self.assertEqual(ns.shot_number, 1)
        self.assertEqual(ns.shot.event_id, "EVT-SES-001")
        self.assertEqual(ns.shot.hole_number, 7)
        self.assertEqual(ns.shot.shot_number, 1)
        self.assertEqual(ns.shot.start_position, Point2D(0.0, 0.0))

    def test_in_memory_sources_satisfy_protocols(self):
        s = _parse_fixture()
        ns = s.shots[0]
        shot_source = InMemoryShotSource(ns.shot, "session:inline:shot:SHOT-7-1")
        course_source = InMemoryCourseSource(s.courses[7], "session:inline:course:SYN-COURSE-001:hole:7")
        player_source = InMemoryPlayerSource(s.player, "session:inline:player:SYNTH-PLAYER-001")
        self.assertIsInstance(shot_source, ShotDataSource)
        self.assertIsInstance(course_source, CourseDataSource)
        self.assertIsInstance(player_source, PlayerProfileSource)
        self.assertIs(shot_source.load_shot(), ns.shot)
        self.assertIs(course_source.load_course(), s.courses[7])
        self.assertIs(player_source.load_player(), s.player)

    def test_builds_one_coordinate_mapper_per_hole(self):
        import ghostcaddie.session as session_module

        original = session_module.CoordinateMapper
        constructed = []

        class CountingMapper(original):
            def __init__(self, coordinate_system):
                constructed.append(coordinate_system)
                super().__init__(coordinate_system)

        session_module.CoordinateMapper = CountingMapper
        try:
            parse_session(_load_fixture())
        finally:
            session_module.CoordinateMapper = original

        self.assertEqual(len(constructed), 2)

    def test_wrong_schema_version_rejected(self):
        raw = _load_fixture()
        raw["schema_version"] = "0.2"
        with self.assertRaises(ValueError):
            parse_session(raw)

    def test_missing_section_rejected(self):
        raw = _load_fixture()
        del raw["shots"]
        with self.assertRaises(ValueError):
            parse_session(raw)
        raw = _load_fixture()
        del raw["player_profile"]
        with self.assertRaises(ValueError):
            parse_session(raw)

    def test_non_empty_ids_and_positive_round_rejected(self):
        raw = _load_fixture()
        raw["session"]["session_id"] = ""
        with self.assertRaises(ValueError):
            parse_session(raw)
        raw = _load_fixture()
        raw["session"]["round_number"] = 0
        with self.assertRaises(ValueError):
            parse_session(raw)
        raw = _load_fixture()
        raw["session"]["round_number"] = 1.5
        with self.assertRaises(ValueError):
            parse_session(raw)

    def test_non_empty_holes_and_shots_rejected(self):
        raw = _load_fixture()
        raw["course"]["holes"] = []
        with self.assertRaises(ValueError):
            parse_session(raw)
        raw = _load_fixture()
        raw["shots"] = []
        with self.assertRaises(ValueError):
            parse_session(raw)

    def test_duplicate_hole_numbers_rejected(self):
        raw = _load_fixture()
        raw["course"]["holes"].append(copy.deepcopy(raw["course"]["holes"][0]))
        with self.assertRaises(ValueError):
            parse_session(raw)

    def test_duplicate_shot_ids_rejected(self):
        raw = _load_fixture()
        raw["shots"][1]["shot_id"] = "SHOT-7-1"
        with self.assertRaises(ValueError):
            parse_session(raw)

    def test_duplicate_ordinal_pairs_rejected(self):
        raw = _load_fixture()
        raw["shots"][1]["shot_number"] = 1
        with self.assertRaisesRegex(ValueError, "duplicate \\(hole_number, shot_number\\) pairs"):
            parse_session(raw)

    def test_out_of_order_shots_rejected(self):
        raw = _load_fixture()
        raw["shots"][0], raw["shots"][1] = raw["shots"][1], raw["shots"][0]
        with self.assertRaises(ValueError):
            parse_session(raw)

    def test_unknown_shot_hole_rejected(self):
        raw = _load_fixture()
        raw["shots"][0]["hole_number"] = 99
        with self.assertRaises(ValueError):
            parse_session(raw)

    def test_identity_mismatch_rejected(self):
        raw = _load_fixture()
        raw["course"]["course_id"] = "OTHER-COURSE"
        with self.assertRaises(ValueError):
            parse_session(raw)
        raw = _load_fixture()
        raw["shots"][0]["player_id"] = "OTHER-PLAYER"
        with self.assertRaises(ValueError):
            parse_session(raw)
        raw = _load_fixture()
        raw["shots"][0]["tournament_id"] = "OTHER-TOUR"
        with self.assertRaises(ValueError):
            parse_session(raw)
        raw = _load_fixture()
        raw["shots"][0]["course_id"] = "OTHER-COURSE"
        with self.assertRaises(ValueError):
            parse_session(raw)

    def test_top_level_player_profile_identity_must_match_session(self):
        raw = _load_fixture()
        raw["player_profile"]["player_id"] = "OTHER-PLAYER"
        with self.assertRaisesRegex(ValueError, "player_profile.player_id"):
            parse_session(raw)

    def test_explicit_null_optional_identity_is_rejected(self):
        raw = _load_fixture()
        raw["shots"][0]["player_id"] = None
        with self.assertRaisesRegex(ValueError, "player_id must be a non-empty string"):
            parse_session(raw)

    def test_non_finite_values_anywhere_in_envelope_are_rejected(self):
        raw = _load_fixture()
        raw["metadata"] = {"nested": [float("inf")]}
        with self.assertRaisesRegex(ValueError, "non-finite"):
            parse_session(raw)

        raw = _load_fixture()
        raw["course"]["holes"][0]["elevation"] = {"height": float("nan")}
        with self.assertRaisesRegex(ValueError, "non-finite"):
            parse_session(raw)

        # Huge ints (too large for float conversion) must also raise the
        # documented ValueError boundary, not leak OverflowError.
        raw = _load_fixture()
        raw["metadata"] = {"big": 10**400}
        with self.assertRaisesRegex(ValueError, "non-finite"):
            parse_session(raw)
        raw = _load_fixture()
        raw["shots"][0]["distance_to_pin"] = 10**400
        with self.assertRaisesRegex(ValueError, "finite"):
            parse_session(raw)

    def test_envelope_metadata_is_preserved_in_report_provenance(self):
        raw = _load_fixture()
        raw["source_note"] = "synthetic fixture"
        session = parse_session(raw)
        report = run_session(session, Config.default())
        self.assertEqual(report["provenance"]["metadata"], {"source_note": "synthetic fixture"})

    def test_optional_identity_fields_may_be_absent(self):
        raw = _load_fixture()
        for shot in raw["shots"]:
            shot.pop("player_id", None)
            shot.pop("tournament_id", None)
            shot.pop("course_id", None)
        s = parse_session(raw)
        self.assertEqual(len(s.shots), 4)

    def test_metadata_preserved(self):
        raw = _load_fixture()
        raw["source_note"] = "synthetic fixture"
        s = parse_session(raw)
        self.assertEqual(s.metadata, {"source_note": "synthetic fixture"})



class TestFourPointExactOnceMapping(unittest.TestCase):
    """A four-point course: each inline position maps exactly once to engine coords."""

    @staticmethod
    def _affine(u, v):
        return 0.375 * (u - 100.0), (200.0 / 540.0) * (v - 80.0)

    def _four_point_envelope(self):
        raw = _load_fixture()
        hole = raw["course"]["holes"][0]
        hole["coordinate_system"] = {
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
        for shot in raw["shots"]:
            if shot["hole_number"] == 7:
                shot["start_position"] = {"x": 300, "y": 200}
                shot["target_position"] = {"x": 700, "y": 500}
                shot["actual_landing_position"] = {"x": 600, "y": 400}
        return raw

    def test_normalized_positions_are_engine_coordinates(self):
        s = parse_session(self._four_point_envelope())
        ns = next(ns for ns in s.shots if ns.shot_id == "SHOT-7-1")
        for raw, got in [
            ({"x": 300, "y": 200}, ns.shot.start_position),
            ({"x": 700, "y": 500}, ns.shot.target_position),
            ({"x": 600, "y": 400}, ns.shot.actual_landing_position),
        ]:
            ex, ey = self._affine(raw["x"], raw["y"])
            self.assertAlmostEqual(got.x, ex, places=6)
            self.assertAlmostEqual(got.y, ey, places=6)

    def test_in_memory_source_applies_no_second_mapper(self):
        s = parse_session(self._four_point_envelope())
        ns = next(ns for ns in s.shots if ns.shot_id == "SHOT-7-1")
        source = InMemoryShotSource(ns.shot, "session:inline:shot:SHOT-7-1")
        loaded = source.load_shot()
        # The in-memory source returns the SAME already-mapped object; if a
        # second mapper were applied, engine coords would be re-interpreted as
        # source pixels and diverge from the affine expectation.
        self.assertIs(loaded, ns.shot)
        ex, ey = self._affine(300.0, 200.0)
        self.assertAlmostEqual(loaded.start_position.x, ex, places=6)
        self.assertAlmostEqual(loaded.start_position.y, ey, places=6)

    def test_pipeline_runs_on_normalized_four_point_session(self):
        s = parse_session(self._four_point_envelope())
        report = run_session(s, Config.default())
        self.assertEqual(len(report["shot_results"]), 4)
        for sr in report["shot_results"]:
            self.assertIn("recommended_club", sr["recommendation"])


# --- Task 3: run_session and aggregation ---


class TestRunSession(unittest.TestCase):
    def setUp(self):
        self.session = _parse_fixture()
        self.config = Config.default()

    def test_one_pipeline_invocation_per_shot(self):
        calls = []

        def _capture(shot_source, course_source, player_source, config):
            calls.append((shot_source, course_source, player_source, config))
            return run_pipeline(shot_source, course_source, player_source, config)

        import ghostcaddie.session as session_module
        original = session_module.run_pipeline
        session_module.run_pipeline = _capture
        try:
            report = run_session(self.session, self.config)
        finally:
            session_module.run_pipeline = original
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(report["shot_results"]), 4)

    def test_per_shot_seed_is_derived_and_stable(self):
        seeds = []
        import ghostcaddie.session as session_module
        original = session_module.run_pipeline

        def _capture(shot_source, course_source, player_source, config):
            seeds.append(config.simulation.random_seed)
            return run_pipeline(shot_source, course_source, player_source, config)

        session_module.run_pipeline = _capture
        try:
            run_session(self.session, self.config)
        finally:
            session_module.run_pipeline = original
        expected = [
            derive_shot_seed(self.session.seed, i, ns.shot_id)
            for i, ns in enumerate(self.session.shots)
        ]
        self.assertEqual(seeds, expected)
        # Stable across calls.
        self.assertEqual(derive_shot_seed(42, 0, "SHOT-7-1"),
                         derive_shot_seed(42, 0, "SHOT-7-1"))
        self.assertNotEqual(derive_shot_seed(42, 0, "SHOT-7-1"),
                            derive_shot_seed(43, 0, "SHOT-7-1"))
        self.assertNotEqual(derive_shot_seed(42, 0, "SHOT-7-1"),
                            derive_shot_seed(42, 1, "SHOT-7-1"))
        self.assertNotEqual(derive_shot_seed(42, 0, "SHOT-7-1"),
                            derive_shot_seed(42, 0, "SHOT-7-2"))

    def test_fixed_seed_repeated_run_equivalence(self):
        a = run_session(self.session, self.config)
        b = run_session(self.session, self.config)
        for field in ("expected_strokes", "actual_expected_strokes", "decision_cost"):
            self.assertEqual(
                [sr["recommendation"][field] for sr in a["shot_results"]],
                [sr["recommendation"][field] for sr in b["shot_results"]],
            )
        self.assertEqual(a["summary"]["sum_local_decision_cost"],
                         b["summary"]["sum_local_decision_cost"])
        self.assertEqual(a["holes"], b["holes"])

    def test_input_order_preserved(self):
        report = run_session(self.session, self.config)
        self.assertEqual(
            [sr["shot_id"] for sr in report["shot_results"]],
            ["SHOT-7-1", "SHOT-7-2", "SHOT-8-1", "SHOT-8-2"],
        )

    def test_no_aggregate_expected_strokes_field(self):
        report = run_session(self.session, self.config)
        self.assertNotIn("expected_strokes", report["summary"])
        self.assertNotIn("actual_expected_strokes", report["summary"])
        for hole in report["holes"]:
            self.assertNotIn("expected_strokes", hole)
            self.assertNotIn("actual_expected_strokes", hole)
        # Per-shot recommendations still carry them.
        for sr in report["shot_results"]:
            self.assertIn("expected_strokes", sr["recommendation"])
            self.assertIn("actual_expected_strokes", sr["recommendation"])

    def test_local_decision_cost_aggregation(self):
        report = run_session(self.session, self.config)
        expected = round(
            sum(sr["recommendation"]["decision_cost"] for sr in report["shot_results"]), 6
        )
        self.assertEqual(report["summary"]["sum_local_decision_cost"], expected)
        for hole in report["holes"]:
            hole_expected = round(
                sum(
                    sr["recommendation"]["decision_cost"]
                    for sr in report["shot_results"]
                    if sr["hole_number"] == hole["hole_number"]
                ),
                6,
            )
            self.assertEqual(hole["sum_local_decision_cost"], hole_expected)
        self.assertIn("decision_cost_semantics", report["summary"])
        self.assertIn("NOT official Strokes Gained", report["summary"]["decision_cost_semantics"])

    def test_hazard_risk_summary_shape(self):
        report = run_session(self.session, self.config)
        summary = report["summary"]["hazard_risk_summary"]
        self.assertIsInstance(summary, dict)
        for region, agg in summary.items():
            self.assertIn("max", agg)
            self.assertIn("mean", agg)
            self.assertIn("nonzero_shot_count", agg)
        # Omitted hazards treated as zero: mean <= max.
        for agg in summary.values():
            self.assertLessEqual(agg["mean"], agg["max"] + 1e-9)

    def test_provenance_has_no_fabricated_filesystem_paths(self):
        report = run_session(self.session, self.config)
        for sr in report["shot_results"]:
            prov = sr["provenance"]
            for key in ("shot_source", "course_source", "player_source"):
                self.assertIn("session:inline", prov[key])
                self.assertNotIn("/", prov[key])
        self.assertEqual(report["provenance"]["source"], "inline-session-envelope")
        self.assertEqual(report["provenance"]["session_id"], self.session.session_id)

    def test_serialize_rejects_non_finite(self):
        with self.assertRaises(ValueError):
            serialize_session_report({"x": float("nan")})
        with self.assertRaises(ValueError):
            serialize_session_report({"nested": [1.0, float("inf")]})
        with self.assertRaises(ValueError):
            serialize_session_report({"k": float("-inf")})

    def test_serialize_output_is_valid_json_without_nan(self):
        report = run_session(self.session, self.config)
        text = serialize_session_report(report)
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        parsed = json.loads(text)
        self.assertEqual(parsed["schema_version"], SESSION_SCHEMA_VERSION)
        self.assertEqual(parsed["summary"]["shot_count"], 4)


# --- Task 4: session CLI ---


class TestSessionCli(unittest.TestCase):
    def _run_cli(self, *args):
        cmd = [sys.executable, "-m", "ghostcaddie", *args]
        return subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )

    def test_session_command_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "session-out"
            proc = self._run_cli(
                "session", "--input", str(SESSION_FIXTURE), "--out", str(out_dir)
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report_path = out_dir / "session_report.json"
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text())
            self.assertEqual(
                sorted(report),
                ["holes", "provenance", "schema_version", "session", "shot_results", "summary"],
            )
            self.assertEqual(report["schema_version"], SESSION_SCHEMA_VERSION)
            self.assertEqual(report["summary"]["shot_count"], 4)
            self.assertEqual(report["summary"]["hole_count"], 2)
            self.assertEqual(len(report["holes"]), 2)
            self.assertEqual([h["shot_count"] for h in report["holes"]], [2, 2])
            self.assertEqual(
                [sr["shot_id"] for sr in report["shot_results"]],
                ["SHOT-7-1", "SHOT-7-2", "SHOT-8-1", "SHOT-8-2"],
            )
            self.assertIn("sum_local_decision_cost", report["summary"])
            self.assertIn("SESSION", proc.stdout)

    def test_session_command_seed_and_samples_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "session-out"
            proc = self._run_cli(
                "session", "--input", str(SESSION_FIXTURE), "--out", str(out_dir),
                "--seed", "7", "--samples", "50",
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads((out_dir / "session_report.json").read_text())
            self.assertEqual(report["session"]["seed"], 7)
            for sr in report["shot_results"]:
                self.assertEqual(sr["provenance"]["monte_carlo_samples"], 50)

    def test_existing_run_command_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            proc = self._run_cli(
                "run", "--shot", str(SHOT), "--course", str(COURSE),
                "--player", str(PLAYER), "--out", str(out_dir),
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue((out_dir / "recommendation.json").exists())
            self.assertTrue((out_dir / "overlay.svg").exists())
            self.assertIn("RECOMMENDATION vs ACTUAL", proc.stdout)


if __name__ == "__main__":
    unittest.main()
