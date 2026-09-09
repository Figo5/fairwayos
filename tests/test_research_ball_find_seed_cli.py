"""Focused tests for the research-ball-find-seed CLI command (deterministic
static-ball seed proposal). Verifies it writes a confirmed:false proposed_seed.json
and returns the persistent round ball in a synthetic static clip.
"""
import json, os, tempfile, unittest
from pathlib import Path
import numpy as np

class FindSeedCliTests(unittest.TestCase):
    def _write_synthetic_video(self, path, ball_center=(968,610), n=8):
        cv2 = __import__("cv2")
        w, h = 1600, 1080
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(path), fourcc, 15, (w, h))
        cx, cy = ball_center
        for _ in range(n):
            img = np.full((h, w, 3), (62,126,95), np.uint8)  # grass BGR
            img = cv2.circle(img, (cx, cy), 14, (39,117,121), -1)
            vw.write(img)
        vw.release()

    def test_writes_unconfirmed_proposed_seed(self):
        cv2 = __import__("cv2")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            video = d / "static.mp4"
            self._write_synthetic_video(video)
            out = d / "out"
            from ghostcaddie.cli import main
            main(["research-ball-find-seed", "--video", str(video), "--out", str(out),
                  "--start-frame", "0", "--end-frame", "7",
                  "--roi", "700", "450", "1500", "750"])
            payload = json.loads((out / "proposed_seed.json").read_text())
            self.assertFalse(payload["confirmed"])
            self.assertTrue(payload["research_only"])
            self.assertIsNotNone(payload["proposed_seed"])
            cx, cy = payload["proposed_seed"]
            self.assertTrue(abs(cx - 968) <= 6, (cx, cy))
            self.assertTrue(abs(cy - 610) <= 6, (cx, cy))

    def test_no_ball_reports_none(self):
        cv2 = __import__("cv2")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            video = d / "nograssball.mp4"
            w, h = 1600, 1080
            vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
            for _ in range(5):
                vw.write(np.full((h, w, 3), (62,126,95), np.uint8))  # no ball
            vw.release()
            out = d / "out"
            from ghostcaddie.cli import main
            main(["research-ball-find-seed", "--video", str(video), "--out", str(out),
                  "--start-frame", "0", "--end-frame", "4",
                  "--roi", "700", "450", "1500", "750"])
            payload = json.loads((out / "proposed_seed.json").read_text())
            self.assertIsNone(payload["proposed_seed"])

if __name__ == "__main__":
    unittest.main()
