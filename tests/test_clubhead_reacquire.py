"""Focused tests for a NEW clubhead method with explicit reacquisition.
The three existing methods all fail because once the fast-moving head outruns
the narrow search window they either die (lk/region) or clamp a static dark
object (dark_blob). This method: color-segments the dark clubhead, tracks the
largest round dark blob, and on a missed frame RE-SEARCHES a wider window to
reacquire rather than terminating or freezing. Written first (RED, method
not yet in SUPPORTED_METHODS).
"""
import unittest
import numpy as np
from ghostcaddie.video.clubhead_methods import track_candidate, CandidateState

def make_swing_frames(n=12, head_start=(620,730), head_dx=40, head_dy=-25):
    """Synthetic downswing: a dark round blob moving diagonally up-left on a
    brighter turf background, present every frame (no real occlusion)."""
    cv2 = __import__("cv2")
    h, w = 1080, 1920
    frames = []
    x, y = head_start
    for i in range(n):
        img = np.full((h, w, 3), (130, 150, 100), np.uint8)  # brighter turf
        # dark round clubhead
        cv2.circle(img, (int(x), int(y)), 22, (40, 40, 45), -1)
        cv2.circle(img, (int(x), int(y)), 22, (90, 90, 100), 2)
        frames.append(img)
        x += head_dx; y += head_dy
    return frames

class ReacquireClubheadTests(unittest.TestCase):
    def test_method_registered(self):
        from ghostcaddie.video.clubhead_methods import SUPPORTED_METHODS
        # the new method must be a supported, non-alias algorithm name
        self.assertIn("reacquire_color", SUPPORTED_METHODS)

    def test_tracks_accelerating_blob_without_dying(self):
        frames = make_swing_frames()
        cands = track_candidate("reacquire_color", frames, 0, (620,730))
        obs = [c.frame_index for c in cands if c.state is CandidateState.OBSERVED]
        # seed + should track most subsequent frames (reacquisition avoids death)
        self.assertGreater(len(obs), len(frames)//2, f"only {obs}")
        self.assertEqual(cands[0].frame_index, 0)
        self.assertEqual(cands[0].state, CandidateState.OBSERVED)

    def test_missing_blob_fails_closed_not_guessed(self):
        # first frame has head, then it disappears entirely -> must NOT invent points
        cv2 = __import__("cv2")
        h, w = 1080, 1920
        f0 = np.full((h, w, 3), (130,150,100), np.uint8)
        cv2.circle(f0, (620,730), 22, (40,40,45), -1)
        empty = np.full((h, w, 3), (130,150,100), np.uint8)
        cands = track_candidate("reacquire_color", [f0, empty, empty, empty], 0, (620,730))
        obs_after = [c for c in cands if c.state is CandidateState.OBSERVED and c.frame_index>0]
        self.assertEqual(obs_after, [], "must fail closed, not guess a vanished head")

if __name__=="__main__":
    unittest.main()
