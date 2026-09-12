import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ghostcaddie.video.fresh_ai_vision import (
    _extract_json,
    build_hermes_command,
    normalize_frame_decision,
    run_child_inference,
    run_smoke_pipeline,
    validate_frames,
)


class FreshAIVisionCliTests(unittest.TestCase):
    def test_build_hermes_command_uses_query_file_and_model_route(self):
        with tempfile.TemporaryDirectory() as d:
            q = Path(d) / "query.md"
            q.write_text("inspect images and emit strict JSON")
            cmd = build_hermes_command(q, max_turns=4)
        self.assertEqual(cmd[:3], ["hermes", "--provider", "openai-codex"])
        self.assertIn("gpt-5.5", cmd)
        self.assertIn("--reasoning", cmd)
        self.assertIn("chat", cmd)
        self.assertEqual("--query", cmd[-2])
        self.assertNotIn("decisions_strict.json", cmd[-1])
        self.assertNotIn("clubhead_framewise", cmd[-1])

    def test_normalize_frame_decision_requires_source_sha_and_nulls_uncertain(self):
        raw = {
            "source_frame": 3058,
            "ball": {"visible": True, "point_xy": [10.2, 20.6], "confidence": 0.72, "uncertainty": "small white dot"},
            "clubhead": {"visible": False, "bbox_xyxy": [1, 2, 3, 4], "confidence": 0.9},
        }
        got = normalize_frame_decision(raw, source_sha256="a" * 64, width=1920, height=1080)
        self.assertEqual(got["source_sha256"], "a" * 64)
        self.assertEqual(got["ball"]["point_xy"], [10.2, 20.6])
        self.assertTrue(got["ball"]["pseudo_label"])
        self.assertFalse(got["clubhead"]["visible"])
        self.assertIsNone(got["clubhead"]["bbox_xyxy"])
        self.assertEqual(got["clubhead"]["confidence"], 0.0)
        self.assertFalse(got["ground_truth"])
        self.assertFalse(got["production_eligible"])

    def test_run_child_inference_extracts_json_fixture_and_writes_logs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            query = root / "prompt.md"
            query.write_text("fixture prompt")

            def fake_run(cmd, **kwargs):
                self.assertEqual(cmd[0], "hermes")
                return subprocess.CompletedProcess(cmd, 0, stdout='banner\n{"frames": [], "provenance": {"model": "gpt-5.5"}}\n', stderr="fixture stderr")

            with mock.patch.object(subprocess, "run", side_effect=fake_run):
                raw, routing = run_child_inference(query, root, max_turns=2)
            self.assertEqual(raw["frames"], [])
            self.assertEqual(routing["returncode"], 0)
            self.assertIn("fixture stderr", (root / "hermes_stderr.log").read_text())

    def test_extract_json_rejects_non_json_fixture_output(self):
        with self.assertRaises(json.JSONDecodeError):
            _extract_json("fixture output without strict JSON")

    def test_validate_frames_rejects_coerced_or_excessive_indices(self):
        for frames in ([False, 1], [1.0], ["1"], [0] * 33):
            with self.subTest(frames=frames):
                with self.assertRaises(ValueError):
                    validate_frames(frames)

    def test_normalize_frame_decision_rejects_strict_schema_violations(self):
        base = {
            "source_frame": 12,
            "ball": {"visible": True, "point_xy": [1, 2], "confidence": 0.5},
            "clubhead": {"visible": False},
        }
        bad_cases = [
            (dict(base, source_frame=True), "source_frame"),
            ({**base, "ball": {"visible": "false", "point_xy": [1, 2], "confidence": 0.5}}, "visible"),
            ({**base, "ball": {"visible": True, "point_xy": [20, 2], "confidence": 0.5}}, "point_xy"),
            ({**base, "ball": {"visible": True, "point_xy": [1, 10], "confidence": 0.5}}, "point_xy"),
            ({**base, "ball": {"visible": True, "point_xy": [1, 2], "confidence": float("nan")}}, "confidence"),
            ({**base, "ball": {"visible": True, "point_xy": [1, 2], "confidence": 1.01}}, "confidence"),
        ]
        for raw, label in bad_cases:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    normalize_frame_decision(raw, source_sha256="a" * 64, width=20, height=10)
        with self.assertRaises(ValueError):
            normalize_frame_decision(base, source_sha256="g" * 64, width=20, height=10)

    def test_run_smoke_pipeline_rejects_duplicate_or_missing_child_frames_fixture(self):
        def run_case(returned_frames):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                video = root / "in.mp4"
                video.write_bytes(b"fake")

                def fake_run(cmd, **kwargs):
                    if cmd[0] == "ffprobe":
                        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"streams": [{"width": 20, "height": 10, "r_frame_rate": "30/1", "nb_frames": "4000"}]}), stderr="")
                    if cmd[0] == "ffmpeg":
                        Path(cmd[-1]).write_bytes(b"jpg")
                        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
                    if cmd[0] == "hermes":
                        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"frames": returned_frames, "provenance": {"model": "gpt-5.5"}}), stderr="fixture")
                    raise AssertionError(cmd)

                with mock.patch.object(subprocess, "run", side_effect=fake_run):
                    run_smoke_pipeline(video, [3058, 3068], root / "out", render=False, max_turns=2)

        duplicate = [
            {"source_frame": 3058, "ball": {"visible": False}, "clubhead": {"visible": False}},
            {"source_frame": 3058, "ball": {"visible": False}, "clubhead": {"visible": False}},
        ]
        missing = [
            {"source_frame": 3058, "ball": {"visible": False}, "clubhead": {"visible": False}},
        ]
        for returned in (duplicate, missing):
            with self.subTest(returned=returned):
                with self.assertRaises(RuntimeError):
                    run_case(returned)

    def test_run_smoke_pipeline_rejects_hardcoded_or_reference_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            video = root / "in.mp4"
            video.write_bytes(b"fake")

            def fake_run(cmd, **kwargs):
                if cmd[0] == "ffprobe":
                    return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"streams": [{"width": 20, "height": 10, "r_frame_rate": "30/1", "nb_frames": "4000"}]}), stderr="")
                if cmd[0] == "ffmpeg":
                    Path(cmd[-1]).write_bytes(b"jpg")
                    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
                if cmd[0] == "hermes":
                    out = json.dumps({
                        "frames": [
                            {"source_frame": 3058, "ball": {"visible": False}, "clubhead": {"visible": False}},
                            {"source_frame": 3068, "ball": {"visible": True, "point_xy": [5, 6], "confidence": 0.8}, "clubhead": {"visible": False}},
                        ],
                        "provenance": {"model": "gpt-5.5", "provider": "openai-codex"},
                    })
                    return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="route ok")
                raise AssertionError(cmd)

            with mock.patch.object(subprocess, "run", side_effect=fake_run):
                result = run_smoke_pipeline(video, [3058, 3068], root / "out", render=False, max_turns=2)
            self.assertTrue(result["ok"])
            self.assertTrue(result["outputs"]["decisions_json"].endswith("fresh_ai_vision_results.json"))
            doc = json.loads(Path(result["outputs"]["decisions_json"]).read_text())
            self.assertEqual(doc["source"]["path"], str(video.resolve()))
            self.assertTrue(doc["source"]["sha256"])
            self.assertEqual(doc["inference"]["provenance"]["input_policy"]["withheld"], ["saved demo decisions", "evaluation references", "annotation seeds"])
            self.assertEqual([r["source_frame"] for r in doc["decisions"]], [3058, 3068])


if __name__ == "__main__":
    unittest.main()
