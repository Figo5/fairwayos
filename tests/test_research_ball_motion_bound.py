"""Focused regression test: ball moving >max_step must still be tracked.
Baseline defect: with default max_step_pixels=24 (search_radius=6, ±30px window),
a ball moving 40px/frame is outside the window -> unavailable -> tracker dies.
This test asserts the tracker can follow a 40px/frame ball by default, i.e. the
motion/search bound is sized to real struck-ball speeds. Written first (RED).
"""
import unittest
import numpy as np
from ghostcaddie.video.research_ball import SeededBallTracker

class BallMotionBoundRegression(unittest.TestCase):
    def _synthetic(self, start=(1370, 574), speed_px=40.0, n=10, radius=14):
        """Generate a small yellow ball moving `speed_px` per frame on turf bg."""
        frames = []
        x, y = start
        for i in range(n):
            img = np.zeros((700, 1600, 3), np.uint8)
            img[:, :] = (95, 126, 62)          # grass-like BGR->RGB? store RGB
            cx, cy = int(round(x)), int(round(y))
            cv2 = __import__("cv2")
            img = cv2.circle(img, (cx, cy), radius, (39, 117, 121), -1, cv2.LINE_AA)  # yellow RGB
            frames.append(img)
            x += speed_px
        return frames

    def test_tracks_ball_moving_40px_per_frame(self):
        frames = self._synthetic(speed_px=40.0)
        r = SeededBallTracker(roi=(700, 450, 1500, 700)).track(frames, seed_frame_index=0, seed_point=(1370, 574))
        provs = [i.provenance for i in r.items]
        # must not die after the seed; at least frames 0..3 tracked
        self.assertEqual(provs[0], "seeded")
        self.assertIn("tracked", provs[1:5], f"ball exceeds default search window: {provs}")

if __name__ == "__main__":
    unittest.main()
