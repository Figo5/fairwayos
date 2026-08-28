import copy
import json
import unittest
from pathlib import Path

from ghostcaddie.adapters.shotlink import SCHEMA_VERSION as SHOTLINK_SCHEMA
from ghostcaddie.adapters.shotlink import adapt_shotlink, ShotLinkDataSource
from ghostcaddie.adapters.trackman import SCHEMA_VERSION as TRACKMAN_SCHEMA
from ghostcaddie.adapters.trackman import adapt_trackman
from ghostcaddie.config import Config
from ghostcaddie.geometry import CoordinateMapper, CoordinateSystem, Point2D
from ghostcaddie.models import ClubProfile, CourseModel, PlayerProfile
from ghostcaddie.pipeline import run_pipeline
from ghostcaddie.session import NormalizedShot, SessionInput, run_session


class ProviderAdapterTests(unittest.TestCase):
    def setUp(self):
        self.mapper = CoordinateMapper(CoordinateSystem(mode="manual", origin=Point2D(0, 0)))
        self.course_context = {
            "start_position": {"x": 0, "y": 0},
            "aim_position": {"x": 100, "y": 0},
        }
        self.sl = {
            "provider": "shotlink", "schema_version": SHOTLINK_SCHEMA,
            "source_record_id": "SL-001", "event_id": "EV-001",
            "player_id": "P-1", "tournament_id": "T-1", "hole_number": 1,
            "shot_number": 1, "timestamp": "2026-01-01T00:00:00Z", "lie": "tee",
            "club": "7i", "distance_to_pin": 150,
            "wind": {"speed_mph": 0, "direction_deg": 0},
            "geo_frame": {"units": "degrees", "axes": "+x east, +y north",
                          "origin": {"latitude": 35.0, "longitude": -80.0}},
            "start_position": {"latitude": 35.0, "longitude": -80.0},
            "target_position": {"latitude": 35.0, "longitude": -79.999},
            "actual_landing_position": {"latitude": 35.0, "longitude": -79.9995},
        }
        self.tm = {
            "provider": "trackman", "schema_version": TRACKMAN_SCHEMA,
            "source_record_id": "TM-001", "event_id": "EV-002", "player_id": "P-1",
            "tournament_id": "T-1", "hole_number": 1, "shot_number": 1,
            "timestamp": "2026-01-01T00:00:00Z", "club": "7i", "units": "yards",
            "metrics": {"carry_yd": 100, "side_offset_yd": 10},
        }

    def test_schema_and_required_fields_are_strict(self):
        with self.assertRaises(ValueError):
            adapt_shotlink({**self.sl, "schema_version": "wrong"}, self.mapper, self.course_context)
        missing = dict(self.tm); del missing["metrics"]
        with self.assertRaises(ValueError):
            adapt_trackman(missing, self.mapper, self.course_context)

    def test_unknown_fields_permissive_and_strict(self):
        raw = {**self.sl, "vendor_debug": {"x": 1}}
        shot = adapt_shotlink(raw, self.mapper, self.course_context)
        self.assertEqual(shot.provenance["unknown_fields"], ["vendor_debug"])
        with self.assertRaisesRegex(ValueError, "vendor_debug"):
            adapt_shotlink(raw, self.mapper, self.course_context, strict=True)

    def test_shotlink_gps_normalizes_east_to_x_and_preserves_source(self):
        shot = adapt_shotlink(self.sl, self.mapper, self.course_context)
        self.assertGreater(shot.target_position.x, 90)
        self.assertAlmostEqual(shot.target_position.y, 0, places=6)
        self.assertEqual(shot.provenance["source_record_id"], "SL-001")
        self.assertEqual(shot.provenance["provider"], "shotlink")

    def test_shotlink_requires_explicit_geo_frame(self):
        raw = copy.deepcopy(self.sl); raw["geo_frame"]["axes"] = "+x west, +y north"
        with self.assertRaises(ValueError):
            adapt_shotlink(raw, self.mapper, self.course_context)

    def test_trackman_reconstructs_signed_side_offset(self):
        shot = adapt_trackman(self.tm, self.mapper, self.course_context)
        self.assertAlmostEqual(shot.actual_landing_position.x, 100)
        self.assertAlmostEqual(shot.actual_landing_position.y, 10)
        self.assertEqual(shot.provenance["source_record_id"], "TM-001")

    def test_trackman_requires_course_and_player_context(self):
        with self.assertRaises(ValueError):
            adapt_trackman(self.tm, self.mapper, None)
        with self.assertRaises(ValueError):
            adapt_trackman(self.tm, self.mapper, {})
        raw = dict(self.tm); raw["player_id"] = ""
        with self.assertRaises(ValueError):
            adapt_trackman(raw, self.mapper, self.course_context)

    def test_trackman_units_are_strict(self):
        raw = dict(self.tm); raw["units"] = "meters"
        with self.assertRaises(ValueError):
            adapt_trackman(raw, self.mapper, self.course_context)

    def test_adapter_source_runs_existing_single_shot_pipeline(self):
        course = CourseModel("synthetic", 4, CoordinateSystem(),
            [[Point2D(-20, -20), Point2D(200, -20), Point2D(200, 20)]],
            [[Point2D(90, -10), Point2D(110, -10), Point2D(100, 10)]],
            pin_position=Point2D(100, 0))
        player = PlayerProfile("P-1", {"7i": ClubProfile("7i", 100, 5, 5)})
        class Source:
            def __init__(self, value): self.value = value
            def load_course(self): return course
            def load_player(self): return player
        result = run_pipeline(ShotLinkDataSource(self.sl, self.mapper, self.course_context),
                              Source(course), Source(player), Config.default())
        self.assertEqual(result.recommendation.provenance["provider"]["source_record_id"], "SL-001")

    def test_provider_normalized_shots_run_in_existing_session_pipeline(self):
        course = CourseModel("synthetic", 4, CoordinateSystem(),
            [[Point2D(-20, -20), Point2D(200, -20), Point2D(200, 20)]],
            [[Point2D(90, -10), Point2D(110, -10), Point2D(100, 10)]],
            pin_position=Point2D(100, 0))
        player = PlayerProfile("P-1", {"7i": ClubProfile("7i", 100, 5, 5)})
        sl_shot = adapt_shotlink(self.sl, self.mapper, self.course_context)
        tm_raw = {**self.tm, "shot_number": 2}
        tm_shot = adapt_trackman(tm_raw, self.mapper, self.course_context)
        session = SessionInput("0.1", "S-1", "T-1", "P-1", "C-1", 1, 42,
            player, {1: course}, [NormalizedShot("SL-001", 1, 1, sl_shot),
                                  NormalizedShot("TM-001", 1, 2, tm_shot)], {})
        report = run_session(session, Config.default())
        self.assertEqual(report["summary"]["shot_count"], 2)
        self.assertEqual(report["shot_results"][0]["provenance"]["provider"]["provider"], "shotlink")
        self.assertEqual(report["shot_results"][1]["provenance"]["provider"]["provider"], "trackman")

    def test_exact_once_mapping_is_one_call_per_position(self):
        class CountingMapper:
            def __init__(self): self.calls = []
            def to_engine(self, point): self.calls.append(point); return point
        mapper = CountingMapper()
        adapt_shotlink(self.sl, mapper, self.course_context)
        self.assertEqual(len(mapper.calls), 3)

    def test_fixture_shapes_are_json_and_explicit(self):
        for name in ("shotlink.json", "trackman.json"):
            path = Path("data/providers") / name
            with path.open() as fh: raw = json.load(fh)
            self.assertIn("provider", raw)
            self.assertIn("schema_version", raw)
            self.assertIn("source_record_id", raw)


if __name__ == "__main__":
    unittest.main()
