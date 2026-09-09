"""Focused tests for deterministic static-ball candidate search (the reliable
method that found the second-swing ball at (968,610) after raw vision failed).
"""
import unittest
import numpy as np
from ghostcaddie.video.research_ball import find_static_ball

def make_ball_frame(img, cx, cy, radius=12, ball_rgb=(39,117,121), grass=(62,126,95)):
    cv2 = __import__("cv2")
    img[:, :] = grass
    return cv2.circle(img, (int(cx), int(cy)), radius, ball_rgb, -1)

class FindStaticBallTests(unittest.TestCase):
    ROI = (500, 450, 1150, 750)

    def test_finds_persistent_round_ball(self):
        cv2 = __import__("cv2")
        cx, cy = 968.0, 610.0
        frames = []
        for _ in range(5):
            frames.append(make_ball_frame(np.zeros((1080, 1920, 3), np.uint8), cx, cy))
        cand = find_static_ball(frames, roi=self.ROI)
        self.assertIsNotNone(cand)
        self.assertTrue(abs(cand[0] - cx) <= 6, cand)
        self.assertTrue(abs(cand[1] - cy) <= 6, cand)

    def test_rejects_no_ball(self):
        cv2 = __import__("cv2")
        frames = [np.full((1080, 1920, 3), (62,126,95), np.uint8) for _ in range(4)]
        self.assertIsNone(find_static_ball(frames, roi=self.ROI))

    def test_rejects_transient_ball(self):
        """A ball present in only one frame must not be reported (not persistent)."""
        cv2 = __import__("cv2")
        frames = []
        for i in range(4):
            img = np.full((1080, 1920, 3), (62,126,95), np.uint8)
            if i == 1:
                img = make_ball_frame(img, 968, 610)
            frames.append(img)
        self.assertIsNone(find_static_ball(frames, roi=self.ROI, min_persistent_frames=3))

if __name__ == "__main__":
    unittest.main()
