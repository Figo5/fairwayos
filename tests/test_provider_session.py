import json
import unittest
from pathlib import Path

from ghostcaddie.adapters.provider_session import parse_provider_session

class ProviderSessionEnvelopeTests(unittest.TestCase):
    def test_provider_session_boundary_exists_and_dispatches(self):
        self.assertTrue(callable(parse_provider_session))

    def test_fixture_runs_both_providers_in_order_and_without_paths(self):
        from ghostcaddie.adapters.provider_session import load_provider_session
        from ghostcaddie.config import Config
        from ghostcaddie.session import serialize_session_report
        for filename, provider in (("shotlink_session.json", "shotlink"), ("trackman_session.json", "trackman")):
            path = Path("data/providers/sessions") / filename
            report = load_provider_session(path, Config.default(), strict=True)
            self.assertEqual(report["provenance"]["metadata"]["provider"], provider)
            self.assertEqual(report["summary"]["shot_count"], 2)
            self.assertEqual([x["shot_id"] for x in report["shot_results"]],
                             (["PS-SL-1", "PS-SL-2"] if provider == "shotlink" else ["PS-TM-2", "PS-TM-3"]))
            self.assertNotIn(str(path), serialize_session_report(report))

    def test_source_paths_are_relative_only(self):
        path = Path("data/providers/sessions/shotlink_session.json")
        raw = json.loads(path.read_text())
        raw["course_source"]["path"] = "/tmp/course.json"
        with self.assertRaises(ValueError): parse_provider_session(raw, path)

    def test_repeated_provider_session_is_deterministic(self):
        from ghostcaddie.adapters.provider_session import load_provider_session
        from ghostcaddie.config import Config
        from ghostcaddie.session import serialize_session_report
        path = Path("data/providers/sessions/shotlink_session.json")
        a = serialize_session_report(load_provider_session(path, Config.default(), strict=True))
        b = serialize_session_report(load_provider_session(path, Config.default(), strict=True))
        self.assertEqual(a, b)

    def test_permissive_preserves_nested_unknown_diagnostic(self):
        from ghostcaddie.adapters.provider_session import load_provider_session
        from ghostcaddie.config import Config
        path = Path("data/providers/sessions/shotlink_session.json")
        raw = json.loads(path.read_text())
        raw["shots"][0]["provider_record"]["vendor_debug"] = {"trace": 1}
        with self.assertRaises(ValueError):
            parse_provider_session(raw, path, strict=True)
        session = parse_provider_session(raw, path, strict=False)
        self.assertIn("shots[0].provider_record.vendor_debug", session.metadata["unknown_fields"])

    def test_provider_schema_and_missing_context_fail_before_pipeline(self):
        path = Path("data/providers/sessions/trackman_session.json")
        raw = json.loads(path.read_text())
        raw["session"]["provider_schema_version"] = "shotlink.v1"
        with self.assertRaises(ValueError): parse_provider_session(raw, path)
        raw = json.loads(path.read_text())
        del raw["shots"][0]["course_context"]
        with self.assertRaises(ValueError): parse_provider_session(raw, path)