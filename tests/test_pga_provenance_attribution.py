"""RED-first regression tests for runtime model-route attribution.

Defect (observed 2026-09-10 on a fresh artifact produced from HEAD 005c2ba):
``build_provenance`` emitted a hardcoded string naming a coordinator
(``gpt-5.6-luna via openai-codex``) and an implementation model
(``glm-5.3-flash via ollama-cloud``) that did not run for that artifact, and
``pga_analyze`` repeated the same claim as a second hardcoded literal in
report.md. The analyzer makes no network calls, so a cloud model cannot have
produced the artifact; the payload asserted ``no_network_calls: true`` and a
``via ollama-cloud`` implementation at the same time.

Provenance must report what actually executed: the local interpreter, the
libraries actually imported, and the local weights actually resolved (path +
hash). Authorship of the source code belongs to git history, not to an
artifact's runtime provenance record.
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
        # libraries must be observed versions, not asserted names
        for lib, ver in route["libraries"].items():
            self.assertIsInstance(ver, str, f"{lib} version must be observed")

    def test_local_model_recorded_with_resolved_path_and_hash(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "custom-pose.pt")
            with open(p, "wb") as fh:
                fh.write(b"not-real-weights")
            digest = hashlib.sha256(b"not-real-weights").hexdigest()

            route = runtime_model_route(p)
            models = route["local_models"]
            self.assertEqual(len(models), 1)
            m = models[0]
            # the model actually passed, not a hardcoded filename
            self.assertTrue(m["path"].endswith("custom-pose.pt"))
            self.assertEqual(m["sha256"], digest)
            self.assertTrue(m["exists"])

    def test_absent_model_is_unavailable_not_asserted(self):
        route = runtime_model_route(None)
        for m in route["local_models"]:
            if not m["exists"]:
                self.assertIsNone(m["sha256"])

    def test_report_renders_route_from_provenance(self):
        """report.md must not carry its own second hardcoded attribution."""
        from ghostcaddie.video import pga_analyze
        src = os.path.join(os.path.dirname(pga_analyze.__file__), "pga_analyze.py")
        with open(src) as fh:
            text = fh.read()
        for name in FABRICATED:
            self.assertNotIn(
                name, text,
                f"pga_analyze.py hardcodes {name!r} instead of rendering real provenance",
            )


if __name__ == "__main__":
    unittest.main()
