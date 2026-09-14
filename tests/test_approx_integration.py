import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from approx_analytics import render_video


def _write_video(path: Path, *, fps: float = 30.0, frames: int = 1):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError("test writer failed")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for _ in range(frames):
        writer.write(frame)
    writer.release()


def _write_vision(path: Path, *, frames: int = 1, confidence=None):
    data = {"frames": {str(i): {
        "ball_refinement": {"native_point": [10 + i, 20 + i, confidence]},
        "clubhead_refinement": {"native_point": [30 + i, 40 + i, confidence]},
    } for i in range(1, frames + 1)}}
    path.write_text(json.dumps(data))


class IntegrationBlockerTests(unittest.TestCase):
    def test_render_rejects_same_input_and_output_path_before_overwrite(self):
        with TemporaryDirectory() as td:
            video = Path(td) / "in.mp4"
            vision = Path(td) / "vision.json"
            _write_video(video)
            _write_vision(vision)
            before = video.read_bytes()

            with self.assertRaisesRegex(ValueError, "input and output paths must differ"):
                render_video(video, vision, video)

            self.assertEqual(video.read_bytes(), before)

    def test_render_refuses_a_source_with_no_readable_rate_before_writing_output(self):
        with TemporaryDirectory() as td:
            video = Path(td) / "in.mp4"
            vision = Path(td) / "vision.json"
            output = Path(td) / "out.mp4"
            video.write_bytes(b"not a video")
            _write_vision(vision)

            # Either gate may fire first; what matters is that it fires before any
            # output file is created.
            with self.assertRaises((RuntimeError, SystemExit)):
                render_video(video, vision, output)

            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".raw.mp4").exists())

    def test_a_sixty_fps_high_definition_source_renders_at_its_own_rate(self):
        # The second source is 1920x1080 at 60000/1001. Nothing is resized down and
        # the output carries the source's own rational rate back out.
        import subprocess

        with TemporaryDirectory() as td:
            video = Path(td) / "in.mp4"
            vision = Path(td) / "vision.json"
            output = Path(td) / "out.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                 "-i", "testsrc=size=1920x1080:rate=60000/1001:duration=0.05",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True)
            frames = int(subprocess.run(
                ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video)],
                capture_output=True, text=True).stdout.strip())
            _write_vision(vision, frames=frames)

            info = render_video(video, vision, output, golfer_facing=True)

            self.assertEqual(info["fps"], "60000/1001")
            self.assertEqual(info["height"], 1080)
            self.assertGreater(info["width"], 1920)

    def test_render_accepts_null_confidence_without_format_crash(self):
        with TemporaryDirectory() as td:
            video = Path(td) / "in.mp4"
            vision = Path(td) / "vision.json"
            output = Path(td) / "out.mp4"
            _write_video(video, fps=30000 / 1001)
            _write_vision(vision, confidence=None)

            info = render_video(video, vision, output)

            self.assertEqual(info["frames"], 1)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
