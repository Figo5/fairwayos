import copy
import json
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from ghostcaddie.cli import main
from tests.test_video_reconstruction import PAYLOAD, calibration

DATA = Path(__file__).resolve().parent.parent / "data"


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
class TestVideoAutomaticAnalyzeCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "calibration.json").write_text(json.dumps(calibration().to_dict()))
        (self.root / "observations.json").write_text(json.dumps(PAYLOAD))
        for name in ("sample_hole.json", "sample_player.json"):
            shutil.copy(DATA / name, self.root / name)
        self.video = self.root / "source.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=green:s=1920x1080:r=2",
                        "-t", "1", "-pix_fmt", "yuv420p", str(self.video)], check=True)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, observations="observations.json", *extra):
        out = self.root / "out"
        argv = ["video-automatic-analyze", "--video", str(self.video),
                "--calibration", "calibration.json", "--course", "sample_hole.json",
                "--player", "sample_player.json", "--observations", observations,
                "--project-root", str(self.root), "--out", str(out), "--event-id", "E1",
                "--tournament-id", "T1", "--hole", "7", "--shot-number", "2",
                "--lie", "fairway", "--club", "7i", "--distance-to-pin", "150",
                "--wind-speed", "8", "--wind-direction", "90", "--timestamp", "automatic-fixture",
                "--target-x", "700", "--target-y", "500"]
        argv.extend(extra)
        with redirect_stdout(StringIO()) as stdout, redirect_stderr(StringIO()) as stderr:
            main(argv)
        return out, stdout.getvalue(), stderr.getvalue()

    def test_help_is_explicit_and_does_not_claim_model_perception(self):
        with redirect_stdout(StringIO()) as stdout:
            with self.assertRaises(SystemExit) as raised:
                main(["video-automatic-analyze", "--help"])
        self.assertEqual(raised.exception.code, 0)
        text = stdout.getvalue().lower()
        self.assertIn("observations", text)
        self.assertIn("approved", text)
        self.assertIn("render-video", text)

    def test_video_auto_analyze_alias_is_available(self):
        with redirect_stdout(StringIO()) as stdout:
            with self.assertRaises(SystemExit) as raised:
                main(["video-auto-analyze", "--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("--calibration", stdout.getvalue())

    def test_missing_required_evidence_blocks_without_analytics_artifacts(self):
        payload = copy.deepcopy(PAYLOAD)
        payload["observations"][1]["ball"] = None
        payload["observations"][1]["warnings"] = ["ball_missing"]
        (self.root / "blocked.json").write_text(json.dumps(payload))
        with self.assertRaises(SystemExit) as raised:
            self.run_cli("blocked.json")
        self.assertNotEqual(raised.exception.code, 0)
        out = self.root / "out"
        self.assertTrue((out / "diagnostics.json").is_file())
        self.assertTrue((out / "evaluation.json").is_file())
        self.assertFalse((out / "recommendation.json").exists())
        self.assertFalse((out / "normalized_shot.json").exists())
        self.assertEqual(json.loads((out / "diagnostics.json").read_text())["status"], "blocked")

    def test_validated_fixture_shaped_input_completes_once_through_automatic_boundary(self):
        out, _, _ = self.run_cli()
        for name in ("diagnostics.json", "evaluation.json", "recommendation.json", "normalized_shot.json"):
            self.assertTrue((out / name).is_file(), name)
        diagnostics = json.loads((out / "diagnostics.json").read_text())
        self.assertEqual(diagnostics["status"], "complete")
        self.assertEqual(diagnostics["gate"]["status"], "passed")
        self.assertEqual(diagnostics["model_provider_provenance"]["mode"], "approved-automatic-adapter")
        self.assertNotIn(str(self.video), (out / "diagnostics.json").read_text())

    def test_success_calls_automatic_reconstruction_and_pipeline_once(self):
        import ghostcaddie.cli as cli
        with (patch.object(cli, "reconstruct_automatic_shot", wraps=cli.reconstruct_automatic_shot) as reconstruct,
              patch.object(cli, "run_pipeline", wraps=cli.run_pipeline) as pipeline):
            self.run_cli()
        self.assertEqual(reconstruct.call_count, 1)
        self.assertEqual(pipeline.call_count, 1)

    def test_project_bound_observation_rejects_absolute_path(self):
        with self.assertRaises(Exception):
            self.run_cli(str(self.root / "observations.json"))


if __name__ == "__main__":
    unittest.main()
