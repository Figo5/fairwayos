"""TDD for the three-target interfaces and bounded job workflow."""
import tempfile, unittest
from ghostcaddie.upload.targets import (
    TargetName, TargetOutcome, RuntimeSafety, plan_targets, describe_runtime,
)
from ghostcaddie.upload.jobs import JobStore, JobState


class RuntimeSafetyTests(unittest.TestCase):
    def test_body_is_no_longer_blocked_by_the_ultralytics_pickle_route(self):
        """Updated: body moved to MoveNet TFLite, which has no pickle surface.

        The old assertion (body blocked by weights_only=False) encoded a fact
        that has since changed. The durable invariant is that body must not be
        permanently blocked by the superseded Ultralytics route.
        """
        b = describe_runtime()[TargetName.BODY]
        # property, not substring: body must never be reported unsafe because of
        # the superseded pickle route
        self.assertNotIn("unsafe", b.reason.lower())
        self.assertNotIn("BLOCKED:", b.reason)

    def test_unrunnable_target_names_its_actual_missing_dependency(self):
        """Whatever the reason, it must be specific and computed, not generic."""
        for name, r in describe_runtime().items():
            if not r.safe_to_run:
                self.assertTrue(len(r.reason) > 20, f"{name} reason too vague")
                self.assertTrue(
                    any(k in r.reason.lower() for k in
                        ("not importable", "model file not found", "unsafe", "seed")),
                    f"{name}: {r.reason}")

    def test_unsafe_or_absent_runtime_never_becomes_runnable_implicitly(self):
        plan = plan_targets(describe_runtime())
        for name, p in plan.items():
            if p.outcome in (TargetOutcome.BLOCKED, TargetOutcome.UNAVAILABLE):
                self.assertIsNone(p.result)

    def test_every_target_has_an_explicit_outcome(self):
        plan = plan_targets(describe_runtime())
        for name in (TargetName.BODY, TargetName.CLUBHEAD, TargetName.BALL):
            self.assertIn(name, plan)
            self.assertIsInstance(plan[name].outcome, str)

    def test_no_target_reports_success_without_a_result(self):
        for t in plan_targets(describe_runtime()).values():
            if t.outcome == TargetOutcome.OBSERVED:
                self.assertIsNotNone(t.result)

    def test_three_target_success_requires_all_three_observed(self):
        from ghostcaddie.upload.targets import is_three_target_success
        plan = plan_targets(describe_runtime())
        self.assertFalse(is_three_target_success(plan))


class JobTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.s = JobStore(self.d)

    def test_new_job_starts_queued_with_zero_progress(self):
        j = self.s.create("clip.mp4")
        self.assertEqual(j.state, JobState.QUEUED)
        self.assertEqual(j.progress, 0.0)

    def test_progress_is_clamped(self):
        j = self.s.create("c.mp4"); self.s.progress(j.id, 5.0)
        self.assertLessEqual(self.s.get(j.id).progress, 1.0)

    def test_cancel_stops_a_running_job(self):
        j = self.s.create("c.mp4"); self.s.start(j.id); self.s.cancel(j.id)
        self.assertEqual(self.s.get(j.id).state, JobState.CANCELLED)

    def test_failure_records_a_reason(self):
        j = self.s.create("c.mp4"); self.s.fail(j.id, "decode failed")
        g = self.s.get(j.id)
        self.assertEqual(g.state, JobState.FAILED)
        self.assertIn("decode", g.error)

    def test_cleanup_removes_job_workdir(self):
        import os
        j = self.s.create("c.mp4")
        wd = self.s.workdir(j.id); os.makedirs(wd, exist_ok=True)
        open(os.path.join(wd, "t.txt"), "w").close()
        self.s.cleanup(j.id)
        self.assertFalse(os.path.exists(wd))

    def test_unknown_job_raises(self):
        with self.assertRaises(KeyError):
            self.s.get("nope")


if __name__ == "__main__":
    unittest.main()
