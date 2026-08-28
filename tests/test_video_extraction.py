import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ghostcaddie.video.errors import VideoExtractionError
from ghostcaddie.video.extraction import extract_frames


class TestExtractFrames(unittest.TestCase):
    def test_extracts_fps_sample_with_deterministic_names_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "round.mp4"
            source.touch()
            output = root / "frames"
            completed = subprocess.CompletedProcess(["ffmpeg"], 0, "", "")
            def write_frames(args, **kwargs):
                output_pattern = Path(args[-1])
                for index in range(1, 4):
                    output_pattern.parent.mkdir(parents=True, exist_ok=True)
                    (output_pattern.parent / f"frame_{index:06d}.jpg").write_bytes(b"jpeg")
                return completed
            with patch("ghostcaddie.video.extraction.subprocess.run", side_effect=write_frames) as run:
                result = extract_frames(str(source), str(output), sample_fps=2.0, max_frames=3)

            self.assertEqual(result.frame_count, 3)
            self.assertEqual([frame.frame_index for frame in result.frames], [1, 2, 3])
            self.assertEqual([frame.timestamp_seconds for frame in result.frames], [0.0, 0.5, 1.0])
            self.assertEqual([frame.filename for frame in result.frames], [
                "frame_000001.jpg", "frame_000002.jpg", "frame_000003.jpg"
            ])
            manifest = json.loads((output / "frame_manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], "video-diagnostics.v1")
            self.assertEqual(manifest["frame_count"], 3)
            self.assertEqual(manifest["frames"][1]["timestamp_seconds"], 0.5)
            self.assertEqual(manifest["artifact_references"], [
                "frame_000001.jpg", "frame_000002.jpg", "frame_000003.jpg"
            ])
            self.assertEqual(manifest["frame_observations"], manifest["frames"])
            args = run.call_args.args[0]
            self.assertEqual(args[0], "ffmpeg")
            self.assertIn("fps=2", args)
            self.assertIn("-frames:v", args)
            self.assertEqual(Path(args[-1]), output.resolve() / "frame_%06d.jpg")

    def test_rejects_missing_source_and_non_positive_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(VideoExtractionError):
                extract_frames(str(root / "missing.mp4"), str(root / "frames"), sample_fps=2)
            source = root / "round.mp4"
            source.touch()
            for kwargs in ({"sample_fps": 0}, {"max_frames": 0}, {}):
                with self.subTest(kwargs=kwargs), self.assertRaises(VideoExtractionError):
                    extract_frames(str(source), str(root / "frames"), **kwargs)

    def test_reports_ffmpeg_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "round.mp4"
            source.touch()
            failed = subprocess.CompletedProcess(["ffmpeg"], 1, "", "decode failed")
            with patch("ghostcaddie.video.extraction.subprocess.run", return_value=failed):
                with self.assertRaises(VideoExtractionError) as context:
                    extract_frames(str(source), str(root / "frames"), sample_fps=1)
            self.assertIn("decode failed", str(context.exception))


if __name__ == "__main__":
    unittest.main()
