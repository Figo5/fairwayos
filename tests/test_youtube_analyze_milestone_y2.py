import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ghostcaddie.cli import _build_parser, main


class TestYouTubeAnalyzeY2(unittest.TestCase):
    def test_parser_exposes_required_explicit_inputs_and_help(self):
        parser = _build_parser()
        args = parser.parse_args([
            "youtube-analyze", "--url", "https://youtu.be/dQw4w9WgXcQ",
            "--calibration", "calibration.json", "--course", "course.json",
            "--player", "player.json", "--project-root", "/tmp/project",
            "--out", "/tmp/out", "--yt-dlp", "/tmp/yt-dlp",
        ])
        self.assertEqual(args.command, "youtube-analyze")
        self.assertEqual(args.url, "https://youtu.be/dQw4w9WgXcQ")
        self.assertEqual(args.downloader, "/tmp/yt-dlp")

    def test_downloaded_source_hits_unavailable_detector_hard_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp), Path(tmp) / "out"
            source = root / "source.mp4"
            source.write_bytes(b"downloaded")
            downloaded = SimpleNamespace(path=str(source), source=SimpleNamespace(to_dict=lambda: {"platform": "youtube", "video_id": "dQw4w9WgXcQ"}), downloader="yt-dlp")
            with patch("ghostcaddie.cli.YtDlpDownloader") as downloader, patch("ghostcaddie.cli.validate_video_source"), patch("ghostcaddie.cli._write_youtube_diagnostics") as write_diag:
                downloader.return_value.download.return_value = downloaded
                with self.assertRaises(SystemExit) as raised:
                    main(["youtube-analyze", "--url", "https://youtu.be/dQw4w9WgXcQ",
                          "--calibration", "calibration.json", "--course", "course.json",
                          "--player", "player.json", "--project-root", str(root),
                          "--out", str(out), "--yt-dlp", "/tmp/yt-dlp"])
            self.assertNotEqual(raised.exception.code, 0)
            write_diag.assert_called_once()
            payload = write_diag.call_args.args[1]
            self.assertEqual(payload["perception_status"], "unavailable")
            self.assertEqual(payload["overall_status"], "failed")
            self.assertNotIn("https://youtu.be", json.dumps(payload))
            self.assertFalse((out / "recommendation.json").exists())
            self.assertFalse((out / "normalized_shot.json").exists())

    def test_fallback_human_is_explicit_and_prepares_blank_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp), Path(tmp) / "out"
            source = root / "source.mp4"
            source.write_bytes(b"downloaded")
            downloaded = SimpleNamespace(path=str(source), source=SimpleNamespace(to_dict=lambda: {"platform": "youtube", "video_id": "dQw4w9WgXcQ"}), downloader="yt-dlp")
            with patch("ghostcaddie.cli.YtDlpDownloader") as downloader, patch("ghostcaddie.cli.prepare_video", return_value={"output_directory": str(out), "manifest": "frames/frame_manifest.json", "contact_sheet": "contact_sheet.jpg", "workspace": "annotation_workspace.html", "draft": "video-human-annotations.v1.json"}) as prepare, patch("ghostcaddie.cli._write_youtube_diagnostics") as write_diag:
                downloader.return_value.download.return_value = downloaded
                main(["youtube-analyze", "--url", "https://youtu.be/dQw4w9WgXcQ",
                      "--calibration", "calibration.json", "--course", "course.json",
                      "--player", "player.json", "--project-root", str(root),
                      "--out", str(out), "--yt-dlp", "/tmp/yt-dlp", "--fallback-human"])
            prepare.assert_called_once_with(str(source), str(out.resolve()), sample_fps=2.0, max_frames=None)
            payload = write_diag.call_args.args[1]
            self.assertEqual(payload["overall_status"], "fallback-human")
            self.assertEqual(payload["perception_status"], "not_run")
            self.assertIn("video-human-annotations.v1.json", payload["artifact_references"])
            self.assertFalse((out / "recommendation.json").exists())

    def test_status_fields_are_separate_and_diagnostics_are_sanitized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp), Path(tmp) / "out"
            source = root / "source.mp4"
            source.write_bytes(b"downloaded")
            downloaded = SimpleNamespace(path=str(source), source=SimpleNamespace(to_dict=lambda: {"platform": "youtube", "video_id": "dQw4w9WgXcQ"}), downloader="yt-dlp")
            with patch("ghostcaddie.cli.YtDlpDownloader") as downloader:
                downloader.return_value.download.return_value = downloaded
                with self.assertRaises(SystemExit):
                    main(["youtube-analyze", "--url", "https://youtu.be/dQw4w9WgXcQ",
                          "--calibration", "calibration.json", "--course", "course.json",
                          "--player", "player.json", "--project-root", str(root),
                          "--out", str(out), "--yt-dlp", "/tmp/yt-dlp"])
            payload = json.loads((out / "diagnostics.json").read_text())
            for field in ("ingestion_status", "perception_status", "calibration_status", "reconstruction_status", "analytics_status", "overall_status"):
                self.assertIn(field, payload)
            text = (out / "diagnostics.json").read_text()
            self.assertNotIn("https://youtu.be", text)
            self.assertNotIn(str(root), text)


if __name__ == "__main__":
    unittest.main()
