import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ghostcaddie.video.annotations import clear_annotated_frames, render_annotated_video


class CompatibleVideoEncodingTests(unittest.TestCase):
    def test_clear_annotated_frames_removes_only_renderer_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = Path(tmp)
            (frames / "frame_000001.jpg").write_bytes(b"old")
            (frames / "other.txt").write_bytes(b"keep")
            outside = frames / "outside.jpg"
            outside.write_bytes(b"outside")
            (frames / "frame_000002.jpg").symlink_to(outside)

            clear_annotated_frames(frames)

            self.assertFalse((frames / "frame_000001.jpg").exists())
            self.assertFalse((frames / "frame_000002.jpg").is_symlink())
            self.assertTrue((frames / "other.txt").is_file())
            self.assertTrue(outside.is_file())

    def test_render_requests_h264_yuv420p_encoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = Path(tmp) / "frames"
            frames.mkdir()
            (frames / "frame_000001.jpg").write_bytes(b"frame")
            output = Path(tmp) / "out.mp4"
            completed = type("Completed", (), {"returncode": 0, "stderr": ""})()
            with patch("ghostcaddie.video.annotations.subprocess.run", return_value=completed) as run:
                with patch.object(Path, "is_file", return_value=True):
                    render_annotated_video(frames, output, frame_rate=2)
            args = run.call_args.args[0]
            self.assertIn("libx264", args)
            self.assertIn("yuv420p", args)
            self.assertTrue(any("format=yuv420p" in value for value in args))


if __name__ == "__main__":
    unittest.main()
