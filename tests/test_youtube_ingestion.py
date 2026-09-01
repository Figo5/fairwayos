import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from ghostcaddie.video.youtube import (
    DownloadLimits,
    DownloadError,
    DownloaderUnavailable,
    YouTubeSource,
    YtDlpDownloader,
    parse_youtube_url,
)


VIDEO_ID = "dQw4w9WgXcQ"


class YouTubeUrlTests(unittest.TestCase):
    def test_accepts_watch_and_short_link_and_sanitizes_provenance(self):
        watch = parse_youtube_url("https://WWW.YouTube.com/watch?v=" + VIDEO_ID + "&utm_source=test")
        short = parse_youtube_url("https://youtu.be/" + VIDEO_ID)
        self.assertEqual(watch, YouTubeSource("youtube", VIDEO_ID))
        self.assertEqual(short.to_dict(), {"platform": "youtube", "video_id": VIDEO_ID})
        self.assertNotIn("url", json.dumps(watch.to_dict()))

    def test_rejects_unsafe_or_non_allowlisted_urls(self):
        bad = (
            "http://youtube.com/watch?v=" + VIDEO_ID,
            "https://evil.example/watch?v=" + VIDEO_ID,
            "file:///tmp/video.mp4",
            "https://user:pass@youtube.com/watch?v=" + VIDEO_ID,
            "https://youtube.com:443/watch?v=" + VIDEO_ID,
            "https://youtube.com/live/" + VIDEO_ID,
            "https://youtube.com/results?search_query=golf",
            "https://youtube.com/@channel",
            "@url:https://youtube.com/watch?v=" + VIDEO_ID,
            "https://youtube.com/watch?v=bad",
            "https://youtube.com/watch?v=" + VIDEO_ID + "&list=PL123",
        )
        for url in bad:
            with self.subTest(url=url):
                with self.assertRaises(DownloadError) as raised:
                    parse_youtube_url(url)
                self.assertTrue(raised.exception.code in {
                    "unsupported_url_scheme", "unsupported_url_host",
                    "unsupported_youtube_url_form", "invalid_video_id",
                    "playlist_not_allowed", "unresolved_url_placeholder",
                })


class YtDlpBoundaryTests(unittest.TestCase):
    def _runner(self, metadata, downloaded_name="source.download.part.mp4", size=4, returncode=0, stderr=""):
        calls = []
        def run(argv, **kwargs):
            calls.append((argv, kwargs))
            if "--dump-single-json" in argv:
                return Mock(returncode=0, stdout=json.dumps(metadata), stderr="")
            target = Path(kwargs["cwd"]) / downloaded_name
            target.write_bytes(b"x" * size)
            return Mock(returncode=returncode, stdout="", stderr=stderr)
        return run, calls

    def test_missing_configured_executable_has_manual_fallback(self):
        with self.assertRaises(DownloaderUnavailable) as raised:
            YtDlpDownloader(None)
        self.assertIn("manual", str(raised.exception).lower())

    def test_symlinked_executable_resolves_when_target_is_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "yt-dlp-target"
            target.write_text("#!/bin/sh\\n")
            target.chmod(target.stat().st_mode | stat.S_IXUSR)
            link = Path(tmp) / "yt-dlp"
            link.symlink_to(target)
            downloader = YtDlpDownloader(str(link))
            self.assertEqual(downloader.executable, str(target.resolve()))

    def test_probe_rejects_private_live_unavailable_and_protected(self):
        for metadata in (
            {"id": VIDEO_ID, "is_live": True, "duration": 10},
            {"id": VIDEO_ID, "availability": "private", "duration": 10},
            {"id": VIDEO_ID, "availability": "unavailable", "duration": 10},
            {"id": VIDEO_ID, "duration": 10, "live_status": "is_upcoming"},
            {"id": VIDEO_ID, "duration": 10, "format": "login required"},
        ):
            runner, _ = self._runner(metadata)
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(DownloadError) as raised:
                    YtDlpDownloader("/bin/sh", runner=runner).download(
                        "https://youtu.be/" + VIDEO_ID, tmp
                    )
                self.assertIn(raised.exception.code, {"live_not_allowed", "protected_content", "unavailable"})

    def test_probe_rejects_returned_video_id_mismatch_before_download(self):
        other_id = "9bZkp7q19f0"
        runner, calls = self._runner({"id": other_id, "duration": 30})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DownloadError) as raised:
                YtDlpDownloader("/bin/sh", runner=runner).download(
                    "https://youtu.be/" + VIDEO_ID, tmp
                )
            self.assertEqual(raised.exception.code, "malformed_metadata")
            self.assertEqual(len(calls), 1)

    def test_probe_rejects_boolean_duration_before_download(self):
        runner, calls = self._runner({"id": VIDEO_ID, "duration": True})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DownloadError) as raised:
                YtDlpDownloader("/bin/sh", runner=runner).download(
                    "https://youtu.be/" + VIDEO_ID, tmp
                )
            self.assertEqual(raised.exception.code, "malformed_metadata")
            self.assertEqual(len(calls), 1)

    def test_probe_rejects_nonfinite_or_nonpositive_duration_before_download(self):
        for duration in ("nan", "inf", "-inf", -1, 0):
            runner, calls = self._runner({"id": VIDEO_ID, "duration": duration})
            with self.subTest(duration=duration), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(DownloadError) as raised:
                    YtDlpDownloader("/bin/sh", runner=runner).download(
                        "https://youtu.be/" + VIDEO_ID, tmp
                    )
                self.assertEqual(raised.exception.code, "malformed_metadata")
                self.assertEqual(len(calls), 1)

    def test_download_uses_fixed_safe_argv_and_atomic_finalization(self):
        runner, calls = self._runner({"id": VIDEO_ID, "duration": 30, "is_live": False})
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "yt-dlp"
            executable.write_text("#!/bin/sh\\n")
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
            with unittest.mock.patch("ghostcaddie.video.youtube.inspect_video"):
                result = YtDlpDownloader(
                    str(executable), runner=runner,
                    limits=DownloadLimits(max_duration_seconds=60, max_download_bytes=10),
                ).download("https://youtu.be/" + VIDEO_ID, tmp)
            self.assertEqual(result.source.video_id, VIDEO_ID)
            self.assertTrue(Path(result.path).is_file())
            self.assertTrue((Path(tmp) / "source_metadata.json").is_file())
            provenance = (Path(tmp) / "source_metadata.json").read_text()
            self.assertIn(VIDEO_ID, provenance)
            self.assertEqual(json.loads(provenance)["status"], "downloaded")
            self.assertNotIn("https://", provenance)
            self.assertNotIn(str(Path(tmp)), provenance)
            self.assertEqual(Path(result.path).name, "source.mp4")
            download_argv, kwargs = calls[1]
            self.assertIn("--no-playlist", download_argv)
            self.assertIn("-o", download_argv)
            self.assertFalse(kwargs["shell"])
            self.assertNotIn("https://youtu.be/" + VIDEO_ID, json.dumps(result.to_dict()))

    def test_download_supports_bounded_low_resolution_segment(self):
        runner, calls = self._runner({"id": VIDEO_ID, "duration": 30})
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "yt-dlp"
            executable.write_text("#!/bin/sh\\n")
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
            with unittest.mock.patch("ghostcaddie.video.youtube.inspect_video"):
                YtDlpDownloader(
                    str(executable),
                    limits=DownloadLimits(max_duration_seconds=60, max_segment_seconds=12, max_download_bytes=10),
                    runner=runner,
                ).download("https://youtu.be/" + VIDEO_ID, tmp)
            download_argv = calls[1][0]
            self.assertIn("--download-sections", download_argv)
            self.assertIn("*0-12", download_argv)
            self.assertTrue(any(arg == "worst" for arg in download_argv))

    def test_download_passes_explicit_node_js_runtime_without_serializing_path(self):
        runner, calls = self._runner({"id": VIDEO_ID, "duration": 30})
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "yt-dlp"
            executable.write_text("#!/bin/sh\\n")
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
            node = Path(tmp) / "node"
            node.write_text("#!/bin/sh\\n")
            node.chmod(node.stat().st_mode | stat.S_IXUSR)
            with unittest.mock.patch("ghostcaddie.video.youtube.inspect_video"):
                result = YtDlpDownloader(
                    str(executable), runner=runner, js_runtime=str(node),
                    limits=DownloadLimits(max_duration_seconds=60, max_segment_seconds=12, max_download_bytes=10),
                ).download("https://youtu.be/" + VIDEO_ID, tmp)
            download_argv = calls[1][0]
            self.assertIn("--js-runtimes", download_argv)
            self.assertIn("node:" + str(node.resolve()), download_argv)
            self.assertNotIn(str(node), json.dumps(result.to_dict()))

    def test_download_accepts_safe_explicit_format_selector(self):
        runner, calls = self._runner({"id": VIDEO_ID, "duration": 30})
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "yt-dlp"
            executable.write_text("#!/bin/sh\\n")
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
            with unittest.mock.patch("ghostcaddie.video.youtube.inspect_video"):
                YtDlpDownloader(str(executable), runner=runner, format_selector="160").download(
                    "https://youtu.be/" + VIDEO_ID, tmp
                )
            self.assertIn("160", calls[1][0])

    def test_probe_rejects_invalid_filesize_metadata_before_download(self):
        for filesize in ("not-a-number", "nan", "inf", "-inf", -1, True):
            runner, calls = self._runner({"id": VIDEO_ID, "duration": 30, "filesize": filesize})
            with self.subTest(filesize=filesize), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(DownloadError) as raised:
                    YtDlpDownloader("/bin/sh", runner=runner).download(
                        "https://youtu.be/" + VIDEO_ID, tmp
                    )
                self.assertEqual(raised.exception.code, "malformed_metadata")
                self.assertEqual(len(calls), 1)

    def test_probe_prefers_zero_filesize_over_approximate_size(self):
        runner, calls = self._runner({
            "id": VIDEO_ID, "duration": 30, "filesize": 0, "filesize_approx": 999,
        })
        with tempfile.TemporaryDirectory() as tmp:
            with unittest.mock.patch("ghostcaddie.video.youtube.inspect_video"):
                YtDlpDownloader(
                    "/bin/sh", runner=runner,
                    limits=DownloadLimits(max_duration_seconds=60, max_download_bytes=10),
                ).download("https://youtu.be/" + VIDEO_ID, tmp)
        self.assertEqual(len(calls), 2)

    def test_download_failure_and_size_limit_are_sanitized(self):
        runner, _ = self._runner({"id": VIDEO_ID, "duration": 30}, returncode=1,
                                 stderr="ERROR https://youtu.be/" + VIDEO_ID + " Authorization: secret")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DownloadError) as raised:
                YtDlpDownloader("/bin/sh", runner=runner).download(
                    "https://youtu.be/" + VIDEO_ID, tmp
                )
            self.assertEqual(raised.exception.code, "download_failed")
            self.assertNotIn(VIDEO_ID, str(raised.exception))
            self.assertNotIn("secret", str(raised.exception))

        runner, _ = self._runner({"id": VIDEO_ID, "duration": 30}, size=11)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DownloadError) as raised:
                YtDlpDownloader(
                    "/bin/sh", runner=runner,
                    limits=DownloadLimits(max_duration_seconds=60, max_download_bytes=10),
                ).download("https://youtu.be/" + VIDEO_ID, tmp)
            self.assertEqual(raised.exception.code, "size_limit_exceeded")

    def test_symlinked_download_is_rejected(self):
        runner, _ = self._runner({"id": VIDEO_ID, "duration": 30})
        def symlink_runner(argv, **kwargs):
            if "--dump-single-json" in argv:
                return Mock(returncode=0, stdout=json.dumps({"id": VIDEO_ID, "duration": 30}), stderr="")
            staging = Path(kwargs["cwd"])
            outside = staging.parent / "outside.mp4"
            outside.write_bytes(b"x")
            (staging / "source.download.part.mp4").symlink_to(outside)
            return Mock(returncode=0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DownloadError) as raised:
                YtDlpDownloader("/bin/sh", runner=symlink_runner).download(
                    "https://youtu.be/" + VIDEO_ID, tmp
                )
            self.assertEqual(raised.exception.code, "unsafe_output")


if __name__ == "__main__":
    unittest.main()
