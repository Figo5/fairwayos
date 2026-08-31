import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ghostcaddie.video.youtube_auto_try import (
    AutoTryConfig, DetectorUnavailable, auto_try, validate_segment,
)
from ghostcaddie.video.youtube import DownloadError
from ghostcaddie.video.youtube_auto_try import _download_failure
from ghostcaddie.video.observations import VideoObservations
from tests.test_video_reconstruction import PAYLOAD

URL = "https://youtu.be/dQw4w9WgXcQ"


class YoutubeAutoTryTests(unittest.TestCase):
    def test_download_limit_errors_keep_specific_categories(self):
        self.assertEqual(_download_failure(DownloadError("too long", "duration_limit_exceeded")), "duration_limit")
        self.assertEqual(_download_failure(DownloadError("segment too long", "segment_limit_exceeded")), "segment_limit")

    def test_malformed_url_writes_transparent_blocked_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            result = auto_try(AutoTryConfig("@url:https://www.youtube.com/watch?v=", out))
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["ingestion_status"], "failed")
            self.assertIn("unresolved_url_placeholder", result["blocking_reasons"])
            self.assertTrue((out / "diagnostics.json").is_file())

    def test_validates_strict_url_and_segment(self):
        self.assertEqual(validate_segment(URL, 2, 5), ("dQw4w9WgXcQ", 2.0, 5.0))
        with self.assertRaises(DownloadError):
            validate_segment("https://www.youtube.com/watch?v=", 0, 5)
        with self.assertRaises(DownloadError):
            validate_segment(URL, -1, 5)
        with self.assertRaises(DownloadError):
            validate_segment(URL, 0, 0)

    def test_bounded_download_passes_validated_start_and_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            downloader = Mock()
            downloader.download.return_value = Mock(path=str(Path(tmp) / "source.mp4"))
            config = AutoTryConfig(URL, Path(tmp) / "out", segment_start=3, segment_duration=7)
            with patch("ghostcaddie.video.youtube_auto_try.extract_frames") as extract:
                extract.return_value = Mock(frames=[], output_directory=str(Path(tmp) / "frames"))
                auto_try(config, downloader=downloader, detector=None)
            downloader.download.assert_called_once_with(URL, str((Path(tmp) / "out").resolve() / "ingest"), segment_start=3.0, segment_duration=7.0)

    def test_auto_try_wires_modern_downloader_and_node_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "source.mp4"; source.write_bytes(b"video")
            fake = Mock(); fake.download.return_value = Mock(path=str(source))
            with patch("ghostcaddie.video.youtube_auto_try.YtDlpDownloader") as factory, \
                    patch("ghostcaddie.video.youtube_auto_try.extract_frames") as extract:
                factory.return_value = fake
                extract.return_value = Mock(frames=[], output_directory=str(root / "frames"))
                auto_try(AutoTryConfig(URL, root / "out", yt_dlp="/custom/yt-dlp"), detector=DetectorUnavailable("missing"))
            factory.assert_called_once_with(
                "/custom/yt-dlp", js_runtime="/usr/local/bin/node",
                limits=unittest.mock.ANY, format_selector=unittest.mock.ANY)

    def test_missing_calibration_is_pixel_only_without_recommendation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "source.mp4"; source.write_bytes(b"video")
            out = root / "out"
            downloader = Mock(); downloader.download.return_value = Mock(path=str(source))
            with patch("ghostcaddie.video.youtube_auto_try.extract_frames") as extract:
                extract.return_value = Mock(frames=[], output_directory=str(root / "frames"))
                result = auto_try(AutoTryConfig(URL, out), downloader=downloader, detector=None)
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["coordinate_space"], "pixels")
            self.assertNotIn("recommendation", result)
            self.assertIn("detector_unavailable", result["blocking_reasons"])

    def test_blocked_detector_is_transparent_and_no_fabrication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "source.mp4"; source.write_bytes(b"video")
            downloader = Mock(); downloader.download.return_value = Mock(path=str(source))
            detector = DetectorUnavailable("not installed")
            with patch("ghostcaddie.video.youtube_auto_try.extract_frames") as extract:
                extract.return_value = Mock(frames=[], output_directory=str(root / "frames"))
                result = auto_try(AutoTryConfig(URL, root / "out", calibration=object()), downloader=downloader, detector=detector)
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["observations"], [])
            self.assertIn("detector_unavailable", result["blocking_reasons"])
            self.assertNotIn("recommendation", result)

    def test_detector_evidence_reports_cuts_multiple_golfers_missing_items_and_low_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "source.mp4"; source.write_bytes(b"video")
            downloader = Mock(); downloader.download.return_value = Mock(path=str(source))
            detector = Mock()
            detector.detect.return_value = {
                "observations": [{"frame_index": 0, "person_count": 2, "golfer_count": 2,
                                  "pose": None, "ball": None, "club": None, "clubhead": None,
                                  "contact": None, "confidence": 0.2}],
                "cut_frames": [1],
            }
            with patch("ghostcaddie.video.youtube_auto_try.extract_frames") as extract:
                extract.return_value = Mock(frames=[], output_directory=str(root / "frames"))
                result = auto_try(AutoTryConfig(URL, root / "out", calibration=object()), downloader=downloader, detector=detector)
            self.assertIn("multiple_golfers", result["blocking_reasons"])
            self.assertIn("cut", result["blocking_reasons"])
            self.assertIn("low_confidence", result["blocking_reasons"])
            self.assertIn("ball_unavailable", result["blocking_reasons"])
            self.assertNotIn("recommendation", result)

    def test_validated_observations_can_use_explicit_gated_analytics_runner_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "source.mp4"; source.write_bytes(b"video")
            downloader = Mock(); downloader.download.return_value = Mock(path=str(source))
            detector = Mock(); detector.detect.return_value = VideoObservations.from_dict(PAYLOAD)
            runner = Mock(return_value={"recommendation": {"status": "fixture-shaped"}})
            with patch("ghostcaddie.video.youtube_auto_try.extract_frames") as extract:
                extract.return_value = Mock(frames=[], output_directory=str(root / "frames"))
                result = auto_try(
                    AutoTryConfig(URL, root / "out", calibration=object(), course=object(), player=object()),
                    downloader=downloader, detector=detector, analytics_runner=runner)
            runner.assert_called_once()
            self.assertEqual(result["recommendation"]["status"], "fixture-shaped")
            self.assertNotIn("fabricated", json.dumps(result).lower())


if __name__ == "__main__":
    unittest.main()
