import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ghostcaddie.video.annotations import build_annotation_filter, annotate_frame
from ghostcaddie.video.diagnostics import build_video_diagnostics, serialize_video_diagnostics
from ghostcaddie.video.observations import VideoObservations
from tests.test_video_reconstruction import PAYLOAD, calibration, context


class TestVideoAnnotations(unittest.TestCase):
    def setUp(self):
        self.observation = VideoObservations.from_dict(PAYLOAD).items[1]

    def test_filter_is_deterministic_and_contains_required_overlays(self):
        first = build_annotation_filter(self.observation, calibration())
        second = build_annotation_filter(self.observation, calibration())
        self.assertEqual(first, second)
        for token in ("drawbox", "drawtext", "drawline", "phase=", "confidence", "ball"):
            self.assertIn(token, first)

    def test_annotation_command_uses_input_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "annotated.jpg"
            with patch("ghostcaddie.video.annotations.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")) as run:
                annotate_frame("/tmp/frame.jpg", output, self.observation, calibration(), ffmpeg="ffmpeg")
            args = run.call_args.args[0]
            self.assertEqual(args[:4], ["ffmpeg", "-v", "error", "-i"])
            self.assertEqual(args[-1], str(output))
            self.assertIn("-vf", args)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
    def test_real_ffmpeg_annotation_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            frame = root / "frame.jpg"
            output = root / "annotated.jpg"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=green:s=160x90", "-frames:v", "1", str(frame)], check=True)
            annotate_frame(frame, output, self.observation, calibration())
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
    def test_fallback_artifact_visibly_encodes_labels_and_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = root / "frame.jpg"
            output = root / "annotated.jpg"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=green:s=1920x1080",
                            "-frames:v", "1", str(frame)], check=True)
            annotate_frame(frame, output, self.observation, calibration=None)
            raw = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(output), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                capture_output=True, check=True).stdout
            self.assertEqual(len(raw), 1920 * 1080 * 3)
            # The label panel must contain actual bright glyph pixels, not only a
            # dark replacement box. JPEG tolerance is handled by the threshold.
            top_panel = [raw[(y * 1920 + x) * 3:(y * 1920 + x + 1) * 3]
                         for y in range(8, 34) for x in range(8, 432)]
            self.assertGreater(sum(1 for r, g, b in top_panel if r > 170 and g > 170 and b > 170), 20)
            # The intended direction and estimated landing must form a visible
            # blue trajectory between the anchor and landing marker.
            trajectory = [raw[(y * 1920 + x) * 3:(y * 1920 + x + 1) * 3]
                          for y in range(495, 506) for x in range(280, 870)]
            self.assertGreater(sum(1 for r, g, b in trajectory if b > 120 and b > r * 1.35 and b > g * 1.15), 100)


class TestVideoDiagnosticsM7(unittest.TestCase):
    def test_builder_has_complete_schema_and_deterministic_json(self):
        observations = VideoObservations.from_dict(PAYLOAD)
        diagnostics = build_video_diagnostics(
            observations, {"container_format": "mp4", "codec": "h264", "width": 1920,
                           "height": 1080, "frame_rate": 30.0, "duration_seconds": 2.0},
            calibration=calibration(), artifact_references=["annotated/frame.jpg"],
            warnings=["optional analytics unavailable"])
        payload = diagnostics.to_dict()
        self.assertEqual(set(payload), {"schema_version", "status", "video_metadata", "artifact_references",
            "frame_observations", "contact", "landing", "normalized_shot", "analytics_result",
            "confidence_values", "warnings", "model_provider_provenance"})
        self.assertEqual(serialize_video_diagnostics(diagnostics), serialize_video_diagnostics(diagnostics))
        self.assertEqual(json.loads(serialize_video_diagnostics(diagnostics)), payload)
        self.assertNotIn("/", payload["artifact_references"][0].replace("annotated/frame.jpg", ""))

    def test_builder_rejects_unsafe_artifact_reference(self):
        with self.assertRaises(Exception):
            build_video_diagnostics(VideoObservations.from_dict(PAYLOAD), {}, artifact_references=["../secret.jpg"])


if __name__ == "__main__":
    unittest.main()
