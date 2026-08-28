import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ghostcaddie.video.annotations import render_annotated_video


class CompatibleVideoEncodingTests(unittest.TestCase):
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
