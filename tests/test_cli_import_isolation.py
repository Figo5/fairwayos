"""Finished CLI startup must not load the superseded research demo runtime."""
import subprocess
import sys
import unittest


class CliImportIsolation(unittest.TestCase):
    def test_canonical_and_demo_help_do_not_import_research_demo(self):
        code = '''
import sys
from ghostcaddie.cli import main
try:
    main([sys.argv[1], "--help"])
except SystemExit as exc:
    assert exc.code == 0
for name in ("ghostcaddie.video.ai_demo", "ghostcaddie.video.pga_fallback", "torch", "cv2"):
    assert name not in sys.modules, name
'''
        for command in ("analyze-video", "demo-video"):
            with self.subTest(command=command):
                result = subprocess.run([sys.executable, "-c", code, command],
                                        capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
