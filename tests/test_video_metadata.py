import json
import subprocess
import unittest
from unittest.mock import patch

from ghostcaddie.video.errors import VideoMetadataError, VideoProbeError
from ghostcaddie.video.metadata import inspect_video, parse_ffprobe_metadata


FFPROBE = {
    "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "20.020"},
    "streams": [{"codec_type": "video", "codec_name": "h264", "width": 1920,
                  "height": 1080, "r_frame_rate": "60000/1001", "nb_frames": "1200"}]
}


class TestParseFfprobeMetadata(unittest.TestCase):
    def test_parses_readable_video_metadata(self):
        metadata = parse_ffprobe_metadata(FFPROBE, source_identifier="/secret/round.mp4")
        self.assertEqual(metadata.container_format, "mov,mp4,m4a,3gp,3g2,mj2")
        self.assertEqual(metadata.codec, "h264")
        self.assertEqual(metadata.width, 1920)
        self.assertAlmostEqual(metadata.frame_rate, 60000 / 1001)
        self.assertEqual(metadata.frame_count, 1200)
        self.assertEqual(metadata.duration_seconds, 20.02)
        self.assertEqual(metadata.source_identifier, "/secret/round.mp4")

    def test_rejects_missing_video_stream(self):
        with self.assertRaises(VideoMetadataError) as ctx:
            parse_ffprobe_metadata({"format": {}, "streams": []})
        self.assertIn("video stream", str(ctx.exception).lower())

    def test_rejects_invalid_metadata(self):
        broken = dict(FFPROBE)
        broken["streams"] = [dict(FFPROBE["streams"][0], width=0)]
        with self.assertRaises(VideoMetadataError):
            parse_ffprobe_metadata(broken)


class TestInspectVideo(unittest.TestCase):
    @patch("ghostcaddie.video.metadata.subprocess.run")
    def test_inspects_with_ffprobe_without_loading_frames(self, run):
        run.return_value = subprocess.CompletedProcess(["ffprobe"], 0, json.dumps(FFPROBE), "")
        metadata = inspect_video("/private/round.mp4")
        self.assertEqual(metadata.codec, "h264")
        args = run.call_args.args[0]
        self.assertIn("-show_streams", args)
        self.assertIn("-show_format", args)
        self.assertEqual(args[-1], "/private/round.mp4")

    @patch("ghostcaddie.video.metadata.subprocess.run", side_effect=FileNotFoundError)
    def test_reports_missing_ffprobe(self, run):
        with self.assertRaises(VideoProbeError):
            inspect_video("round.mp4")

    @patch("ghostcaddie.video.metadata.subprocess.run")
    def test_reports_probe_failure(self, run):
        run.return_value = subprocess.CompletedProcess(["ffprobe"], 1, "", "bad input")
        with self.assertRaises(VideoProbeError):
            inspect_video("round.mp4")


if __name__ == "__main__":
    unittest.main()
