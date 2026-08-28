import json
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from ghostcaddie.cli import main


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
class TestVideoPrepareCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.video = self.root / "source.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
            "color=c=green:s=320x180:r=4", "-t", "1", "-pix_fmt", "yuv420p",
            str(self.video),
        ], check=True)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *extra):
        out = self.root / "prepared"
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            main(["video-prepare", "--video", str(self.video), "--out", str(out),
                  "--sample-fps", "2", "--max-frames", "2", *extra])
        return out

    def test_prepare_writes_deterministic_offline_artifacts_and_blank_draft(self):
        out = self.run_cli()
        for relative in (
            "frames/frame_manifest.json", "frames/frame_000001.jpg", "frames/frame_000002.jpg",
            "contact_sheet.jpg", "annotation_workspace.html", "video-human-annotations.v1.json",
        ):
            self.assertTrue((out / relative).is_file(), relative)
        manifest = json.loads((out / "frames/frame_manifest.json").read_text())
        self.assertEqual(manifest["frame_count"], 2)
        self.assertEqual([f["frame_index"] for f in manifest["frames"]], [1, 2])
        self.assertEqual([f["timestamp_seconds"] for f in manifest["frames"]], [0.0, 0.5])
        html = (out / "annotation_workspace.html").read_text()
        self.assertIn('src="contact_sheet.jpg"', html)
        self.assertIn('href",f.filename', html)
        self.assertIn("frames/frame_000001.jpg", html)
        self.assertIn("Save Draft", html)
        self.assertIn("Submit Annotations", html)
        self.assertNotIn(str(self.video), html)
        self.assertNotRegex(html.lower(), r'(?:src|href)=["\'](?:https?:|//)')
        draft = json.loads((out / "video-human-annotations.v1.json").read_text())
        self.assertEqual(set(draft), {"schema_version", "status", "explicit_submit", "video", "calibration_points", "engine_points", "golfer_anchor", "ball", "clubhead", "contact", "target_intended_direction", "landing", "club_selection", "context", "warnings"})
        self.assertEqual(draft["status"], "draft")
        self.assertFalse(draft["explicit_submit"])
        self.assertTrue(all(point["source"] == "unavailable" and point["value"] is None
                            for point in draft["calibration_points"]))
        self.assertTrue(all(point is None for point in draft["engine_points"]))
        for key in ("golfer_anchor", "ball", "clubhead", "contact", "landing"):
            self.assertIsNone(draft[key]["value"])
            self.assertEqual(draft[key]["source"], "unavailable")

    def test_prepare_rejects_malformed_or_unsafe_inputs(self):
        with self.assertRaises((ValueError, SystemExit)):
            self.run_cli("--sample-fps", "0")
        with self.assertRaises((ValueError, SystemExit)):
            main(["video-prepare", "--video", str(self.root / "missing.mp4"), "--out", str(self.root / "out")])
        with self.assertRaises((ValueError, SystemExit)):
            main(["video-prepare", "--video", str(self.video), "--out", str(self.video)])


if __name__ == "__main__":
    unittest.main()
