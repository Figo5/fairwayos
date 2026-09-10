"""RED-first regression tests for runtime model-route attribution.

Original defect (observed 2026-09-10 on a fresh artifact produced from HEAD
005c2ba): ``build_provenance`` emitted a hardcoded string naming a coordinator
(``gpt-5.6-luna via openai-codex``) and an implementation model
(``glm-5.3-flash via ollama-cloud``) that did not run for that artifact, and
``pga_analyze`` repeated the same claim as a second hardcoded literal in
report.md. The analyzer makes no network calls, so a cloud model cannot have
produced the artifact; the payload asserted ``no_network_calls: true`` and a
``via ollama-cloud`` implementation at the same time.

Coordinator review of c5bd101 raised two further defects, covered here:

1. ``runtime_model_route`` called ANY existing file "loaded". A file
   containing ``not a model`` was reported ``state: loaded``. File discovery
   and hashing is not evidence that a model loaded. The load outcome must be
   OBSERVED and passed in by the caller that actually attempted the load, and
   unavailable/error states must be preserved rather than upgraded.
2. ``test_report_renders_route_from_provenance`` grepped the source file
   instead of exercising the renderer. Replaced with behavioral tests over
   ``_render_model_route`` output, including the missing-model case.

Provenance must report what actually executed. Authorship of the source code
belongs to git history, not to an artifact's runtime provenance record.
"""
import hashlib
import os
import tempfile
import unittest

from ghostcaddie.video.pga_research_analyzer import (
    build_provenance,
    runtime_model_route,
)

# Cloud/agent identities that must never be asserted as artifact runtime
# provenance by a local, no-network analyzer.
FABRICATED = (
    "glm-5.3-flash",
    "ollama-cloud",
    "gpt-5.6-luna",
    "openai-codex",
    "claude-opus-5",
    "Hermes agent",
    "Coordinator:",
)


def _flat(obj):
    """Yield every string anywhere in a nested JSON-ish payload."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _flat(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _flat(v)


class ProvenanceAttributionTests(unittest.TestCase):
    def test_provenance_names_no_model_that_did_not_run(self):
        prov = build_provenance("in.mp4", "out", {"pose": "automatic"})
        blob = "\n".join(_flat(prov))
        for name in FABRICATED:
            self.assertNotIn(
                name, blob,
                f"provenance asserts {name!r}, which did not run for this artifact",
            )

    def test_no_network_claim_is_consistent_with_route(self):
        prov = build_provenance("in.mp4", "out", {"pose": "automatic"})
        route = prov["model_route"]
        if prov.get("no_network_calls"):
            self.assertFalse(
                route.get("remote_models"),
                "payload claims no_network_calls while crediting a remote model",
            )

    def test_route_reports_actual_runtime(self):
        route = runtime_model_route(None)
        self.assertIn("python", route["runtime"])
        self.assertIn("platform", route["runtime"])
        for lib, ver in route["libraries"].items():
            self.assertIsInstance(ver, str, f"{lib} version must be observed")


class ModelLoadOutcomeTests(unittest.TestCase):
    """Discovery/hashing must not be reported as a successful load."""

    def _write(self, td, name, payload):
        p = os.path.join(td, name)
        with open(p, "wb") as fh:
            fh.write(payload)
        return p, hashlib.sha256(payload).hexdigest()

    def test_nonexistent_model_is_unavailable(self):
        route = runtime_model_route("/no/such/model.pt")
        m = route["local_models"][0]
        self.assertFalse(m["discovered"])
        self.assertIsNone(m["sha256"])
        self.assertEqual(m["state"], "unavailable")

    def test_no_path_is_unavailable(self):
        route = runtime_model_route(None)
        m = route["local_models"][0]
        self.assertFalse(m["discovered"])
        self.assertIsNone(m["sha256"])
        self.assertEqual(m["state"], "unavailable")

    def test_existing_file_alone_is_not_loaded(self):
        """The coordinator's reproduction: a .pt containing 'not a model'."""
        with tempfile.TemporaryDirectory() as td:
            p, digest = self._write(td, "fake.pt", b"not a model")
            route = runtime_model_route(p)
            m = route["local_models"][0]
            # discovered and hashed, but NOT loaded -- nobody observed a load
            self.assertTrue(m["discovered"])
            self.assertEqual(m["sha256"], digest)
            self.assertNotEqual(
                m["state"], "loaded",
                "existing file reported as loaded without an observed load",
            )
            self.assertEqual(m["state"], "not_attempted")

    def test_observed_load_failure_is_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            p, digest = self._write(td, "fake.pt", b"not a model")
            route = runtime_model_route(
                p, pose_load_state="load_failed",
                pose_load_error="invalid load key",
            )
            m = route["local_models"][0]
            self.assertTrue(m["discovered"])
            self.assertEqual(m["sha256"], digest)
            self.assertEqual(m["state"], "load_failed")
            self.assertIn("invalid load key", m["load_error"])

    def test_observed_success_is_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            p, digest = self._write(td, "real.pt", b"pretend-weights")
            route = runtime_model_route(p, pose_load_state="loaded")
            m = route["local_models"][0]
            self.assertTrue(m["discovered"])
            self.assertEqual(m["sha256"], digest)
            self.assertEqual(m["state"], "loaded")
            self.assertIsNone(m["load_error"])

    def test_load_success_cannot_be_claimed_for_missing_file(self):
        """A caller must not be able to upgrade a missing file to loaded."""
        route = runtime_model_route("/no/such/model.pt", pose_load_state="loaded")
        m = route["local_models"][0]
        self.assertEqual(m["state"], "unavailable")
        self.assertFalse(m["discovered"])

    def test_unknown_load_state_is_rejected(self):
        with self.assertRaises(ValueError):
            runtime_model_route(None, pose_load_state="probably_fine")


class RenderModelRouteTests(unittest.TestCase):
    """Behavioral tests over the rendered report section (not a source grep)."""

    def _render(self, route):
        from ghostcaddie.video.pga_analyze import _render_model_route
        return "\n".join(_render_model_route({"model_route": route}))

    def test_rendered_route_names_no_model_that_did_not_run(self):
        text = self._render(runtime_model_route(None))
        for name in FABRICATED:
            self.assertNotIn(name, text)

    def test_rendered_route_shows_runtime_and_no_remote_models(self):
        text = self._render(runtime_model_route(None))
        self.assertIn("Runtime: python", text)
        self.assertIn("no network calls", text)

    def test_rendered_missing_model_says_unavailable_without_hash(self):
        text = self._render(runtime_model_route(None))
        self.assertIn("unavailable", text)
        self.assertNotIn("sha256 `None`", text)
        self.assertNotIn("[loaded]", text)

    def test_rendered_loaded_model_shows_path_hash_and_state(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "real.pt")
            with open(p, "wb") as fh:
                fh.write(b"pretend-weights")
            digest = hashlib.sha256(b"pretend-weights").hexdigest()
            text = self._render(runtime_model_route(p, pose_load_state="loaded"))
            self.assertIn("real.pt", text)
            self.assertIn(digest, text)
            self.assertIn("[loaded]", text)

    def test_rendered_failed_load_discloses_failure_not_success(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "fake.pt")
            with open(p, "wb") as fh:
                fh.write(b"not a model")
            text = self._render(runtime_model_route(
                p, pose_load_state="load_failed", pose_load_error="invalid load key"))
            self.assertIn("load_failed", text)
            self.assertIn("invalid load key", text)
            self.assertNotIn("[loaded]", text)

    def test_renderer_tolerates_legacy_string_route(self):
        """Old artifacts stored model_route as a plain string."""
        text = self._render("some legacy string")
        self.assertIn("some legacy string", text)


if __name__ == "__main__":
    unittest.main()
