import json
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from ghostcaddie.cli import main
from tests.test_video_reconstruction import PAYLOAD, calibration

DATA = Path(__file__).resolve().parent.parent / "data"


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
class TestVideoAnalyzeCLI(unittest.TestCase):
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

    def run_cli(self, *extra):
        out = self.root / "out"
        with redirect_stdout(StringIO()) as stdout, redirect_stderr(StringIO()) as stderr:
            main(["video-analyze", "--video", str(self.video), "--calibration", "calibration.json",
                  "--course", "sample_hole.json", "--player", "sample_player.json",
                  "--observations", "observations.json", "--project-root", str(self.root),
                  "--out", str(out), "--event-id", "E1", "--tournament-id", "T1", "--hole", "7",
                  "--shot-number", "2", "--lie", "fairway", "--club", "7i", "--distance-to-pin", "150",
                  "--wind-speed", "8", "--wind-direction", "90", "--timestamp", "2026-08-27T12:00:00Z",
                  "--target-x", "700", "--target-y", "500", *extra])
        return out, stdout.getvalue(), stderr.getvalue()

    def test_success_writes_complete_diagnostics_and_annotated_frames(self):
        out, _, _ = self.run_cli()
        for name in ("diagnostics.json", "recommendation.json", "overlay.svg", "normalized_shot.json", "contact_sheet.jpg"):
            self.assertTrue((out / name).is_file(), name)
        self.assertTrue(list((out / "annotated_frames").glob("frame_*.jpg")))
        payload = json.loads((out / "diagnostics.json").read_text())
        self.assertEqual(payload["status"], "complete")
        self.assertTrue(all(not Path(ref).is_absolute() for ref in payload["artifact_references"]))
        self.assertNotIn(str(self.video), (out / "diagnostics.json").read_text())

    def test_help_documents_absolute_video_and_project_resources(self):
        with redirect_stdout(StringIO()) as stdout:
            with self.assertRaises(SystemExit) as raised:
                main(["video-analyze", "--help"])
        self.assertEqual(raised.exception.code, 0)
        text = stdout.getvalue()
        self.assertIn("absolute", text.lower())
        self.assertIn("project", text.lower())

    def test_render_video_writes_deterministic_sampled_mp4(self):
        out, _, _ = self.run_cli("--render-video")
        rendered = out / "annotated_video.mp4"
        self.assertTrue(rendered.is_file())
        self.assertGreater(rendered.stat().st_size, 0)
        self.assertIn("annotated_video.mp4", json.loads((out / "diagnostics.json").read_text())["artifact_references"])


if __name__ == "__main__":
    unittest.main()
