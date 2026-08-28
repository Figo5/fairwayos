import subprocess
import tempfile
import unittest
import shutil
from pathlib import Path
from unittest.mock import patch

from ghostcaddie.video.errors import VideoExtractionError
from ghostcaddie.video.extraction import generate_contact_sheet


def _ffmpeg_available():
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


class TestContactSheet(unittest.TestCase):
    def test_real_ffmpeg_artifacts_have_expected_dimensions_and_count(self):
        if not _ffmpeg_available():
            self.skipTest("ffmpeg is not installed")
        from ghostcaddie.video.extraction import extract_frames

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "synthetic.mp4"
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "testsrc=size=160x90:rate=4", "-t", "1", "-c:v", "mpeg4", str(source),
            ], check=True)
            frames = extract_frames(str(source), str(root / "frames"), sample_fps=2, max_frames=5)
            self.assertEqual(frames.frame_count, 2)
            self.assertEqual(len(list((root / "frames").glob("frame_*.jpg"))), 2)
            sheet = generate_contact_sheet(frames.output_directory, str(root / "contact.jpg"),
                                           columns=2, frame_width=80, frame_height=45)
            self.assertEqual(frames.frame_count, 2)
            self.assertEqual(sheet.tile_count, 2)
            self.assertTrue(Path(sheet.output_path).is_file())
            probe = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json",
                                    "-show_streams", sheet.output_path], capture_output=True, text=True, check=True)
            self.assertEqual(probe.returncode, 0)
            self.assertIn('"width": 160', probe.stdout)
            self.assertIn('"height": 45', probe.stdout)

    def test_generates_deterministic_tiled_contact_sheet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "frames"
            frames.mkdir()
            for index in (1, 2, 3, 4, 5):
                (frames / f"frame_{index:06d}.jpg").touch()
            output = root / "contact.jpg"
            with patch("ghostcaddie.video.extraction.subprocess.run", return_value=subprocess.CompletedProcess(["ffmpeg"], 0, "", "")) as run:
                result = generate_contact_sheet(str(frames), str(output), columns=3, frame_width=320, frame_height=180)

            self.assertEqual(result.tile_count, 5)
            self.assertEqual(result.columns, 3)
            self.assertEqual(result.rows, 2)
            self.assertEqual(result.width, 960)
            self.assertEqual(result.height, 360)
            args = run.call_args.args[0]
            self.assertTrue(any("tile=3x2" in arg for arg in args))
            self.assertEqual(Path(args[-1]), output.resolve())

    def test_rejects_invalid_contact_sheet_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(VideoExtractionError):
                generate_contact_sheet(str(root / "missing"), str(root / "contact.jpg"), columns=2)
            frames = root / "frames"
            frames.mkdir()
            with self.assertRaises(VideoExtractionError):
                generate_contact_sheet(str(frames), str(root / "contact.jpg"), columns=0)
            with self.assertRaises(VideoExtractionError):
                generate_contact_sheet(str(frames), str(root / "contact.jpg"), columns=2)


if __name__ == "__main__":
    unittest.main()
