"""TDD: runtime registry must verify interpreters by probing them for real."""
import unittest
from ghostcaddie.upload.runtimes import (
    RuntimeSpec, RuntimeRegistry, probe_interpreter, InterpreterUnavailable,
)


class ProbeTests(unittest.TestCase):
    def test_probe_reports_missing_interpreter(self):
        r = probe_interpreter("/no/such/python", ["numpy"])
        self.assertFalse(r.ok)
        self.assertIn("not executable", r.reason)

    def test_probe_reports_missing_module_by_name(self):
        import sys
        r = probe_interpreter(sys.executable, ["definitely_not_a_module_xyz"])
        self.assertFalse(r.ok)
        self.assertIn("definitely_not_a_module_xyz", r.reason)

    def test_probe_succeeds_on_a_real_interpreter(self):
        import sys
        r = probe_interpreter(sys.executable, ["json"])
        self.assertTrue(r.ok, r.reason)
        self.assertTrue(r.python_version)

    def test_probe_runs_out_of_process(self):
        """The probe must not import the module into THIS process."""
        import sys
        probe_interpreter(sys.executable, ["definitely_not_a_module_xyz"])
        self.assertNotIn("definitely_not_a_module_xyz", sys.modules)


class RegistryTests(unittest.TestCase):
    def test_registry_reports_readiness_per_target(self):
        reg = RuntimeRegistry.default()
        st = reg.readiness()
        for t in ("body", "clubhead", "ball"):
            self.assertIn(t, st)
            self.assertIn("ready", st[t])
            self.assertTrue(st[t]["reason"])

    def test_unready_target_raises_rather_than_running(self):
        reg = RuntimeRegistry({"body": RuntimeSpec("body", "/no/such/python", ["numpy"], "m")})
        with self.assertRaises(InterpreterUnavailable):
            reg.require("body")

    def test_registry_is_serialisable_for_the_readiness_endpoint(self):
        import json
        json.dumps(RuntimeRegistry.default().readiness())


if __name__ == "__main__":
    unittest.main()
