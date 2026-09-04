import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ghostcaddie.cli import main
from ghostcaddie.video.pga_fallback import render_pga_fallback


class TestPgaResearchDemo(unittest.TestCase):
    def test_mode_accepts_bounded_local_flags_and_never_runs_pipeline(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "fixture.mp4"
            subprocess.run([
                "/opt/homebrew/bin/ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                "-i", "color=c=green:s=320x240:r=4", "-t", "1",
                "-pix_fmt", "yuv420p", str(video),
            ], check=True)
            with patch("ghostcaddie.cli.run_local_demo", return_value={"status": "research_only"}) as runner, \
                 patch("ghostcaddie.cli.run_pipeline", side_effect=AssertionError("analytics forbidden")):
                main(["pga-research-demo", "--video", str(video), "--out", str(root / "out"),
                      "--max-duration", "2", "--sample-fps", "2", "--max-frames", "4"])
            self.assertEqual(runner.call_args.kwargs["max_duration_seconds"], 2.0)
            self.assertEqual(runner.call_args.kwargs["sample_fps"], 2.0)
            self.assertEqual(runner.call_args.kwargs["max_frames"], 4)

    def test_fallback_decoded_pixels_contain_required_annotations(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "fixture.mp4"
            subprocess.run(["/opt/homebrew/bin/ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                            "-i", "color=c=green:s=320x240:r=4", "-t", "1", "-pix_fmt", "yuv420p", str(video)], check=True)
            out = root / "rendered.mp4"
            render_pga_fallback(video, out, max_frames=4)
            raw = subprocess.run(["/opt/homebrew/bin/ffmpeg", "-v", "error", "-i", str(out), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
            self.assertEqual(len(raw), 320 * 240 * 3)
            pixels = [raw[i:i + 3] for i in range(0, len(raw), 3)]
            self.assertGreater(sum(pixel != bytes((0, 128, 0)) for pixel in pixels), 100)
            self.assertGreater(sum(all(channel > 180 for channel in pixel) for pixel in pixels), 10)

    def test_blocked_output_has_explicit_research_flags_and_states(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "fixture.mp4"
            subprocess.run([
                "/opt/homebrew/bin/ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                "-i", "color=c=green:s=320x240:r=4", "-t", "1",
                "-pix_fmt", "yuv420p", str(video),
            ], check=True)
            with patch("ghostcaddie.cli.run_local_demo", side_effect=RuntimeError("no frames")):
                with self.assertRaises(SystemExit):
                    main(["pga-research-demo", "--video", str(video), "--out", str(root / "out")])
            payload = json.loads((root / "out" / "diagnostics.json").read_text())
            self.assertEqual(payload["status"], "blocked")
            self.assertTrue(payload["research_only"])
            self.assertFalse(payload["ground_truth"])
            self.assertFalse(payload["production_eligible"])
            self.assertIsNone(payload["analytics"])
            self.assertIsNone(payload["shot_event"])
            self.assertIn("pose", payload["unavailable_layers"])
            self.assertIn("ball", payload["unavailable_layers"])
            self.assertIn("clubhead", payload["unavailable_layers"])
            for name in ("annotated_video.mp4", "contact_sheet.jpg", "diagnostics.json", "provenance.json", "README"):
                self.assertTrue((root / "out" / name).is_file(), name)
            labels = (root / "out" / "README").read_text()
            for label in ("PGA RESEARCH DEMO", "RESEARCH ONLY", "NO PRODUCTION ANALYTICS", "UNAVAILABLE"):
                self.assertIn(label, labels)


if __name__ == "__main__":
    unittest.main()
