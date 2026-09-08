"""Clean-checkout/CI portability regressions (commit 9b5cd2b failures).
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import subprocess

TESTS_DIR = Path(__file__).parent

# Split so this guard's own source does not contain the literal it forbids.
_HOMEBREW_FFMPEG = "/opt/homebrew/bin/" + "ffmpeg"


def _runtime_ffmpeg():
    return shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")


class TestHeadlessHighguiCleanup(unittest.TestCase):
    """CI (commit b080e72 installs opencv-python-headless) runs without a GUI.

    Highgui calls such as ``cv2.destroyAllWindows()`` must be conditional on
    highgui actually being usable, and failures must not be hidden by broad
    ``except Exception`` blocks that would also swallow unrelated errors.
    """

    def test_cleanup_is_routed_through_a_conditional_safe_helper(self):
        import inspect

        import ghostcaddie.video.ai_demo as ai_demo

        # The module must expose an explicit helper rather than calling the
        # bare cv2 function inline in run_local_demo.
        self.assertTrue(hasattr(ai_demo, "close_highgui_windows"),
                        "ai_demo must define close_highgui_windows")
        runner_source = inspect.getsource(ai_demo.run_local_demo)
        self.assertNotIn(
            "destroyAllWindows", runner_source,
            "run_local_demo must delegate highgui cleanup to close_highgui_windows",
        )
        helper_source = inspect.getsource(ai_demo.close_highgui_windows)
        self.assertIn("destroyAllWindows", helper_source)

    def test_cleanup_swallows_only_highgui_runtime_errors(self):
        import cv2
        import ghostcaddie.video.ai_demo as ai_demo

        with patch.object(cv2, "destroyAllWindows",
                          side_effect=cv2.error("no GUI server")):
            # A highgui error is tolerated: headless environments have no
            # windows to destroy.
            ai_demo.close_highgui_windows()

        with patch.object(cv2, "destroyAllWindows",
                          side_effect=RuntimeError("unrelated boom")):
            # Unrelated errors must propagate, never be silently swallowed.
            with self.assertRaises(RuntimeError):
                ai_demo.close_highgui_windows()

    def test_cleanup_skips_when_highgui_is_absent(self):
        import cv2
        import ghostcaddie.video.ai_demo as ai_demo

        with patch.object(cv2, "destroyAllWindows",
                          side_effect=AssertionError("must not be called")):
            with patch.object(cv2, "imshow", None):
                # No usable imshow -> highgui unavailable -> no call at all.
                ai_demo.close_highgui_windows()


class TestNoHardcodedToolPaths(unittest.TestCase):
    def test_tracked_tests_do_not_hardcode_homebrew_ffmpeg(self):
        # Regression for 9b5cd2b CI failure: /opt/homebrew/bin/ffmpeg must
        # never appear in tracked test sources.
        for test_file in TESTS_DIR.glob("test_*.py"):
            if test_file.name == Path(__file__).name:
                continue
            text = test_file.read_text()
            self.assertNotIn(
                _HOMEBREW_FFMPEG, text,
                f"{test_file.name} hard-codes a Homebrew-only ffmpeg path",
            )


@unittest.skipUnless(_runtime_ffmpeg(), "ffmpeg is not installed")
class TestPgaResearchDemoPortableRuntime(unittest.TestCase):
    def test_fallback_render_works_with_runtime_discovered_ffmpeg(self):
        # The render path must rely on PATH-discovered ffmpeg, not a
        # hard-coded absolute path.
        import ghostcaddie.video.pga_fallback as pga_fallback

        ffmpeg = _runtime_ffmpeg()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "fixture.mp4"
            subprocess.run([
                ffmpeg, "-y", "-v", "error", "-f", "lavfi",
                "-i", "color=c=green:s=320x240:r=4", "-t", "1",
                "-pix_fmt", "yuv420p", str(video),
            ], check=True)
            out = root / "rendered.mp4"
            pga_fallback.render_pga_fallback(video, out, max_frames=4)
            self.assertGreater(out.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()