"""TDD: adapters must PROBE real capability, never hardcode booleans."""
import os, tempfile, unittest
from ghostcaddie.upload.adapters import (
    Capability, BodyMoveNetAdapter, ClubheadSam2Adapter, BallTapirAdapter,
    UltralyticsBodyAdapter,
)


class ProbeTests(unittest.TestCase):
    def test_capability_reports_a_reason_always(self):
        for A in (BodyMoveNetAdapter, ClubheadSam2Adapter, BallTapirAdapter):
            c = A().capability()
            self.assertIsInstance(c, Capability)
            self.assertTrue(c.reason, f"{A.__name__} gave no reason")

    def test_missing_model_file_is_detected_not_assumed(self):
        a = BodyMoveNetAdapter(model_path="/no/such/model.tflite")
        c = a.capability()
        self.assertFalse(c.available)
        self.assertIn("model", c.reason.lower())

    def test_missing_runtime_is_reported_with_the_dependency_name(self):
        a = BodyMoveNetAdapter(model_path=__file__, _force_runtime_missing=True)
        c = a.capability()
        self.assertFalse(c.available)
        self.assertIn("ai_edge_litert", c.reason)

    def test_capability_is_computed_not_a_constant(self):
        """Same adapter class, different model paths -> different answers."""
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.tflite")
            with open(p, "wb") as fh: fh.write(b"x" * 16)
            a1 = BodyMoveNetAdapter(model_path=p)
            a2 = BodyMoveNetAdapter(model_path="/no/such.tflite")
            self.assertNotEqual(a1.capability().model_present,
                                a2.capability().model_present)

    def test_model_hash_is_reported_when_present(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.tflite")
            with open(p, "wb") as fh: fh.write(b"abc")
            c = BodyMoveNetAdapter(model_path=p).capability()
            self.assertEqual(len(c.model_sha256), 64)

    def test_ultralytics_body_adapter_is_permanently_unsafe(self):
        """The pickle path stays blocked; it is not the body route any more."""
        c = UltralyticsBodyAdapter().capability()
        self.assertFalse(c.safe)
        self.assertIn("weights_only=False", c.reason)

    def test_movenet_adapter_is_not_blocked_by_ultralytics(self):
        """Body must not be blocked by the old pickle runtime.

        Asserts the PROPERTY, not the absence of a substring: the earlier
        version failed because the reason string mentions ultralytics only to
        say the blocker does not apply, which is exactly the wording we want.
        """
        c = BodyMoveNetAdapter().capability()
        self.assertTrue(c.safe, "TFLite flatbuffer has no pickle surface")
        self.assertNotEqual(c.runtime, "ultralytics")
        self.assertNotIn("unsafe", c.reason.lower())


class FailurePathTests(unittest.TestCase):
    def test_run_without_capability_raises_not_fakes(self):
        from ghostcaddie.upload.adapters import AdapterUnavailable
        a = BodyMoveNetAdapter(model_path="/no/such.tflite")
        with self.assertRaises(AdapterUnavailable):
            a.run_frames([], source_sha256="0"*64)

    def test_run_requires_a_source_hash(self):
        from ghostcaddie.upload.adapters import AdapterUnavailable
        a = BodyMoveNetAdapter()
        with self.assertRaises((AdapterUnavailable, ValueError)):
            a.run_frames([], source_sha256=None)


if __name__ == "__main__":
    unittest.main()
