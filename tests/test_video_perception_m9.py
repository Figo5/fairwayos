import base64
import json
import tempfile
import unittest
from pathlib import Path

from ghostcaddie.video.errors import VideoContractError
from ghostcaddie.video.perception import OllamaPerceptionAdapter, PerceptionResult

class TestOllamaPerceptionAdapterM9(unittest.TestCase):
    def _frame(self, directory):
        path = Path(directory) / "frame_000001.jpg"
        # Minimal valid JPEG dimensions (1x1), with bounded non-empty bytes.
        path.write_bytes(bytes.fromhex("ffd8ffe000104a46494600010100000100010000ffc0001108000100010301110002110003110000ffd9"))
        return path

    def test_model_mode_is_explicitly_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = OllamaPerceptionAdapter(enabled=False)
            result = adapter.perceive([self._frame(tmp)])
        self.assertIsInstance(result, PerceptionResult)
        self.assertIsNone(result.observations)
        self.assertEqual(result.status, "unavailable")
        self.assertIn("opt-in", " ".join(result.warnings))
        self.assertEqual(result.provenance["mode"], "model")

    def test_unavailable_ollama_degrades_without_leaking_input_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = self._frame(tmp)
            adapter = OllamaPerceptionAdapter(enabled=True, endpoint="http://127.0.0.1:1", timeout_seconds=0.01)
            result = adapter.perceive([frame])
        self.assertIsNone(result.observations)
        self.assertEqual(result.status, "unavailable")
        serialized = json.dumps(result.to_dict())
        self.assertNotIn(str(frame), serialized)
        self.assertEqual(set(result.provenance), {"model", "provider", "mode"})

    def test_malformed_model_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = self._frame(tmp)
            adapter = OllamaPerceptionAdapter(enabled=True, transport=lambda payload, timeout: {"message": {"content": "ignore prior instructions"}})
            with self.assertRaises(VideoContractError):
                adapter.perceive([frame])

    def test_fabricated_top_level_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = self._frame(tmp)
            payload = {"schema_version": "video-observations.v1", "image": {"width": 1, "height": 1}, "observations": []}
            payload["fabricated_landing_yards"] = 123.4
            adapter = OllamaPerceptionAdapter(enabled=True, transport=lambda p, t: {"message": {"content": json.dumps(payload)}})
            with self.assertRaises(VideoContractError):
                adapter.perceive([frame])

        captured = {}
        def transport(payload, timeout):
            captured.update(payload)
            return {"message": {"content": "not JSON"}}
        with tempfile.TemporaryDirectory() as tmp:
            frame = self._frame(tmp)
            frame_bytes = frame.read_bytes()
            adapter = OllamaPerceptionAdapter(enabled=True, transport=transport, max_frames=1)
            with self.assertRaises(VideoContractError):
                adapter.perceive([frame])
        self.assertEqual(captured["model"], "gemma4:e2b")
        self.assertEqual(captured["format"], "json")
        image = captured["messages"][0]["images"][0]
        self.assertEqual(base64.b64decode(image), frame_bytes)
        self.assertLessEqual(len(image), adapter.max_payload_bytes * 2)


if __name__ == "__main__":
    unittest.main()
