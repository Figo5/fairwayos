"""Focused TDD tests: metal-appearance / upward-path RELOCK for reacquire_color.

Failure being fixed (research_stock/pexels_33511561.mp4 frames 176-194): the
true clubhead is a glossy dark-metal mass in the upper half (true y 280-690)
while reacquire_color's prior parks on flat-dark ground/shoe masses and the
tracker emits static_locked storms or false-locks drifting ground junk
(y 990-1007). The relock path must:
  1. reject flat-dark (matte, no enclosed specular highlight) masses outright,
  2. prefer an upward metal candidate over a lower one,
  3. fail closed (UNAVAILABLE) when no metal candidate qualifies.
The NORMAL tracking phase must stay exactly as before (the existing synthetic
dark-blob-on-turf tests have no speculars and must keep passing).
"""
import unittest
import numpy as np

from ghostcaddie.video.clubhead_methods import track_candidate, CandidateState


def _turf(h, w):
    """Mid-value grass: neither 'dark' (<90) nor 'bright' (>=150)."""
    return np.full((h, w, 3), (70, 100, 60), np.uint8)  # BGR, V=100


def _flat_dark_band(img, y0, y1, x0=0, x1=None):
    """Matte flat-dark ground/shoe-like band: dark, zero speculars."""
    h, w = img.shape[:2]
    x1 = w if x1 is None else x1
    img[y0:y1, x0:x1] = (30, 30, 32)
    return img


def _metal_head(img, cx, cy, r=26):
    """Glossy dark head: dark disc body + bright enclosed specular streak."""
    cv2 = __import__("cv2")
    cv2.circle(img, (int(cx), int(cy)), r, (35, 35, 40), -1)
    # horizontal specular streak fully inside the dark body (v ~ 235); minor
    # axis 12 keeps aspect <= ~5 like real head clusters (measured 1.3-5.9)
    cv2.ellipse(img, (int(cx), int(cy)), (max(12, r - 6), 12), 0, 0, 360, (215, 220, 235), -1)
    return img


def _upward_metal_scene(frames_n, head_pts, junk_band=(900, 1010)):
    """Turf scene; glossy metal head at head_pts[i] each frame; static matte
    flat-dark ground band near the bottom (the false-lock trap)."""
    h, w = 1080, 1920
    out = []
    for i in range(frames_n):
        img = _turf(h, w)
        _flat_dark_band(img, junk_band[0], junk_band[1])
        cx, cy = head_pts[i]
        _metal_head(img, cx, cy)
        out.append(img)
    return out


class MetalRelockTests(unittest.TestCase):
    def test_relock_grabs_metal_head_not_flat_dark_ground(self):
        # Seed sits on empty turf; the metal head is far away (upper half);
        # a static flat-dark band sits within the reacquire window below.
        # After stagnation the relock must acquire the METAL head, never the
        # matte band. Head is realistic-sized (r=60) so its enclosed specular
        # streak survives the SID kernel/enclosure gates (see probes).
        pts = [(320.0, 300.0)] * 10
        h, w = 1080, 1920
        frames = []
        for i in range(10):
            img = _turf(h, w)
            _flat_dark_band(img, 900, 1010)
            _metal_head(img, pts[i][0], pts[i][1], r=80)
            frames.append(img)
        seed = (640.0, 730.0)  # on turf, far from head (dist ~ 350) and band
        cands = track_candidate("reacquire_color", frames, 0, seed)
        obs = [c for c in cands[1:] if c.state is CandidateState.OBSERVED]
        self.assertTrue(obs, "relock must eventually acquire the metal head")
        for c in obs:
            self.assertLess(c.point[1], 600,
                            f"locked y={c.point[1]} — flat-dark ground band "
                            f"or lower junk, not the metal head: {c}")
            self.assertLess(abs(c.point[0] - 320), 120, f"not on head x: {c}")
            self.assertLess(abs(c.point[1] - 300), 120, f"not on head y: {c}")

    def test_relock_fails_closed_when_only_flat_dark_mass_exists(self):
        # Same trap but WITHOUT any metal head anywhere: the flat-dark band
        # must never be emitted, ever (fail closed, retry every frame).
        h, w = 1080, 1920
        frames = []
        for _ in range(10):
            img = _turf(h, w)
            _flat_dark_band(img, 900, 1010)
            frames.append(img)
        cands = track_candidate("reacquire_color", frames, 0, (640.0, 730.0))
        obs_after = [c for c in cands if c.state is CandidateState.OBSERVED and c.frame_index > 0]
        self.assertEqual(obs_after, [], f"flat-dark mass must stay unavailable: {obs_after}")

    def test_relock_prefers_upward_metal_over_lower_metal_junk(self):
        # Two glossy masses: true head ABOVE the stuck prior, a metal-looking
        # decoy BELOW it but above the bottom band. Upward-path relock must
        # take the upper one.
        h, w = 1080, 1920
        frames = []
        for _ in range(10):
            img = _turf(h, w)
            _metal_head(img, 320, 300, r=80)   # head, above prior y=730
            _metal_head(img, 620, 700, r=80)   # decoy, near prior y=730
            frames.append(img)
        cands = track_candidate("reacquire_color", frames, 0, (640.0, 730.0))
        obs = [c for c in cands[1:] if c.state is CandidateState.OBSERVED and c.warning == "metal_relock"]
        self.assertTrue(obs, "relock must acquire one of the metal masses")
        for c in obs:
            self.assertLess(c.point[1], 500,
                            f"relock took the LOWER decoy at {c.point}; must "
                            f"prefer the upward metal candidate")

    def test_normal_phase_tracking_unchanged_without_stagnation(self):
        # Existing behaviour: a moving dark blob (no speculars) present every
        # frame within the search window keeps tracking without the metal gate.
        cv2 = __import__("cv2")
        h, w = 1080, 1920
        frames = []
        x, y = 620.0, 730.0
        for _ in range(12):
            img = _turf(h, w)
            cv2.circle(img, (int(x), int(y)), 22, (40, 40, 45), -1)
            frames.append(img)
            x += 40; y -= 25
        cands = track_candidate("reacquire_color", frames, 0, (620.0, 730.0))
        obs = [c.frame_index for c in cands if c.state is CandidateState.OBSERVED]
        self.assertGreater(len(obs), len(frames) // 2, f"regression: {obs}")

    def test_relock_from_high_anchor_follows_descending_head(self):
        # Corner-stuck regime: the prior is anchored HIGH (head exited past the
        # frame corner), then the metal head re-enters and descends BELOW the
        # anchor but stays out of the bottom ground band. The relock must
        # acquire it (strict "upward vs anchor" would deadlock forever).
        cv2 = __import__("cv2")
        h, w = 1080, 1920
        frames = []
        hx, hy = 600.0, 450.0
        for _ in range(9):
            img = _turf(h, w)
            cv2.circle(img, (200, 150), 22, (40, 40, 45), -1)  # static trap, no speculars
            _metal_head(img, hx, hy)
            frames.append(img)
            hx += 35; hy += 20
        cands = track_candidate("reacquire_color", frames, 0, (200.0, 150.0))
        obs = [c for c in cands[1:] if c.state is CandidateState.OBSERVED]
        self.assertTrue(obs, f"descending head must be reacquired: {[(c.frame_index, c.point) for c in cands[1:]]}")
        for c in obs:
            self.assertLess(c.point[1], 734, f"lock in bottom band: {c}")

    def test_relock_never_enters_bottom_ground_band(self):
        # Even from a high anchor, a GLOSSY metal-looking mass inside the bottom
        # ground band must be refused (band ban is unconditional). Seed on open
        # turf so nothing distracts the relock: after stagnation the only metal
        # anywhere is the bottom-band decoy, and it must never be emitted.
        h, w = 1080, 1920
        frames = []
        for _ in range(9):
            img = _turf(h, w)
            _metal_head(img, 450, 800)  # glossy decoy INSIDE bottom band (y > 0.68*h)
            frames.append(img)
        cands = track_candidate("reacquire_color", frames, 0, (200.0, 300.0))
        obs_after = [c for c in cands if c.state is CandidateState.OBSERVED and c.frame_index > 0]
        self.assertEqual(obs_after, [], f"bottom-band glossy decoy must stay unavailable: {obs_after}")

    def test_relock_rejects_ball_against_dark_trees(self):
        # Real-video trap (frames 197-200): the white ball against dark trees
        # produces a bright SID cluster at its edge, but its neighbourhood is
        # mostly BRIGHT (the ball body) — not metal enclosed by darkness. The
        # relock must reject clusters whose surrounding box is not majority-dark
        # (relock_min_dark_frac) and fail closed instead.
        cv2 = __import__("cv2")
        h, w = 1080, 1920
        frames = []
        for _ in range(10):
            img = np.full((h, w, 3), (28, 26, 30), np.uint8)  # dark trees backdrop
            img[900:, :] = (70, 100, 60)
            cv2.circle(img, (1148, 677), 78, (230, 235, 240), -1)  # white ball
            frames.append(img)
        cands = track_candidate("reacquire_color", frames, 0, (648.0, 543.0))
        obs_after = [c for c in cands if c.state is CandidateState.OBSERVED and c.frame_index > 0]
        self.assertEqual(obs_after, [], f"ball-edge SID decoy must stay unavailable: {obs_after}")

    def test_relock_accepts_enclosed_metal_highlight(self):
        # The synthetic head's specular streak (aspect ~7, fully enclosed by
        # dark metal, neighbourhood majority-dark) must pass the enclosure gate
        # and be acquired — the gate may not over-tighten into blindness.
        h, w = 1080, 1920
        frames = []
        for _ in range(10):
            img = _turf(h, w)
            _metal_head(img, 600, 560, r=60)
            frames.append(img)
        cands = track_candidate("reacquire_color", frames, 0, (648.0, 543.0))
        obs = [c for c in cands[1:] if c.state is CandidateState.OBSERVED]
        self.assertTrue(obs, "enclosed metal highlight must be acquirable")
        for c in obs:
            self.assertLess(abs(c.point[0] - 600), 100, f"not on head: {c}")
            self.assertLess(abs(c.point[1] - 560), 100, f"not on head: {c}")

    def test_relock_parameters_are_validated(self):
        frames = _upward_metal_scene(6, [(320.0, 300.0)] * 6)
        for kwargs in (
            {"relock_radius_px": 0},
            {"relock_specular_px": -1},
            {"relock_max_aspect": 0.5},
            {"relock_sid_dark_frac": 1.5},
            {"relock_stagnation_frames": 0},
            {"relock_bottom_band_frac": 0},
            {"relock_bottom_band_frac": 1.5},
            {"relock_min_dark_frac": 0},
            {"relock_min_dark_frac": 1.2},
        ):
            with self.assertRaises(ValueError):
                track_candidate("reacquire_color", frames, 0, (640.0, 730.0), **kwargs)


if __name__ == "__main__":
    unittest.main()