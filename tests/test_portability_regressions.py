"""Clean-checkout/CI portability regressions (commit 9b5cd2b failures).
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import subprocess

TESTS_DIR = Path(__file__).parent

# Split so this guard's own source does not contain the literal it forbids.
_HOMEBREW_FFMPEG = "/opt/homebrew/bin/" + "ffmpeg"


def _runtime_ffmpeg():
    return shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")


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