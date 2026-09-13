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

    def test_render_rejects_unsupported_source_fps_before_writing_output(self):
        with TemporaryDirectory() as td:
            video = Path(td) / "in.mp4"
            vision = Path(td) / "vision.json"
            output = Path(td) / "out.mp4"
            _write_video(video, fps=25.0)
            _write_vision(vision)

            with self.assertRaisesRegex(ValueError, "expected source FPS 30000/1001"):
                render_video(video, vision, output)

            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".raw.mp4").exists())

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
