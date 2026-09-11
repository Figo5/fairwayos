"""Behaviour tests for sliced (tiled) local-contrast ball candidates."""
import unittest

import numpy as np

from ghostcaddie.video.pga_sliced_ball import (
    SLICED_DEFAULTS,
    global_candidates,
    sliced_candidates,
)


def scene(bg=90, ball_xy=(300, 400), ball_v=250, radius=4, grad=False):
    """Grass-ish background with an optional small bright ball."""
    img = np.zeros((720, 640, 3), np.uint8)
    img[:, :, 0] = bg // 2          # B
    img[:, :, 1] = bg               # G (grass)
    img[:, :, 2] = bg // 3          # R
    if grad:  # strong global brightness ramp: what breaks a global threshold
        ramp = np.linspace(0, 150, 640).astype(np.uint8)
        img[:, :, 1] = np.clip(img[:, :, 1] + ramp[None, :], 0, 255)
    if ball_xy is not None:
        x, y = ball_xy
        yy, xx = np.ogrid[:720, :640]
        m = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        img[m] = (ball_v, ball_v, ball_v)   # white/desaturated
    return img


class SlicedCandidateTests(unittest.TestCase):
    def test_finds_a_small_bright_desaturated_blob(self):
        c = sliced_candidates(scene())
        self.assertTrue(c)
        top = c[0]
        self.assertLess(abs(top["x"] - 300), 3)
        self.assertLess(abs(top["y"] - 400), 3)

    def test_candidates_are_ranked_by_local_contrast(self):
        c = sliced_candidates(scene())
        self.assertEqual(c, sorted(c, key=lambda d: -d["z"]))

    def test_empty_scene_is_allowed_to_yield_nothing(self):
        # uniform field: no blob exists, so nothing should be invented
        self.assertEqual(sliced_candidates(scene(ball_xy=None)), [])

    def test_large_object_is_not_a_ball_candidate(self):
        c = sliced_candidates(scene(radius=40))
        self.assertTrue(all(d["area"] <= SLICED_DEFAULTS["max_area"] for d in c))

    def test_duplicate_candidates_are_deduped(self):
        c = sliced_candidates(scene())
        for i, a in enumerate(c):
            for b in c[i + 1:]:
                self.assertGreater((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2,
                                   SLICED_DEFAULTS["dedupe_px"] ** 2 - 1e-6)

    def test_local_contrast_survives_a_global_brightness_ramp(self):
        """The transferred idea: a global threshold cannot serve both ends of a
        strong brightness gradient, a per-tile one can."""
        img = scene(ball_xy=(120, 400), grad=True)
        sl = sliced_candidates(img)
        self.assertTrue(sl)
        best_sliced = min((abs(d["x"] - 120) + abs(d["y"] - 400), i)
                          for i, d in enumerate(sl))
        # the ball ranks at or near the top for the sliced detector
        self.assertLess(best_sliced[1], 3)

    def test_global_detector_returns_candidates_unranked_by_contrast(self):
        c = global_candidates(scene())
        self.assertTrue(all("x" in d and "y" in d for d in c))


if __name__ == "__main__":
    unittest.main()


class TemporalRankingTests(unittest.TestCase):
    """Ranking by temporal support, and abstention when support is absent."""

    def _c(self, x, y, z=6.0, area=30):
        return {"x": float(x), "y": float(y), "z": z, "area": area}

    def test_static_object_gets_full_support(self):
        from ghostcaddie.video.pga_sliced_ball import temporal_support
        r = temporal_support([self._c(100, 100)], [self._c(100, 100)],
                             [self._c(100, 100)])
        self.assertAlmostEqual(r[0]["support"], 1.0)

    def test_constant_velocity_object_is_supported(self):
        from ghostcaddie.video.pga_sliced_ball import temporal_support
        r = temporal_support([self._c(100, 300)], [self._c(120, 260)],
                             [self._c(140, 220)])
        self.assertGreater(r[0]["support"], 0.8)

    def test_reversing_object_is_not_supported(self):
        """A speck that jumps back and forth is not a ball."""
        from ghostcaddie.video.pga_sliced_ball import temporal_support
        r = temporal_support([self._c(100, 300)], [self._c(140, 220)],
                             [self._c(100, 300)])
        self.assertEqual(r[0]["support"], 0.0)

    def test_implausibly_fast_jump_is_not_supported(self):
        from ghostcaddie.video.pga_sliced_ball import temporal_support
        r = temporal_support([self._c(10, 10)], [self._c(300, 400)],
                             [self._c(600, 700)])
        self.assertEqual(r[0]["support"], 0.0)

    def test_abstains_with_no_neighbours(self):
        from ghostcaddie.video.pga_sliced_ball import select_with_abstention
        self.assertIsNone(select_with_abstention([], [self._c(100, 100)], []))

    def test_abstains_on_empty_candidates(self):
        from ghostcaddie.video.pga_sliced_ball import select_with_abstention
        self.assertIsNone(select_with_abstention([self._c(1, 1)], [], [self._c(1, 1)]))

    def test_selects_the_supported_candidate_over_a_brighter_unsupported_one(self):
        from ghostcaddie.video.pga_sliced_ball import select_with_abstention
        ball = self._c(200, 200, z=3.0)
        speck = self._c(400, 50, z=9.0)
        sel = select_with_abstention([self._c(200, 200)], [speck, ball],
                                     [self._c(200, 200)])
        self.assertIsNotNone(sel)
        self.assertEqual((sel["x"], sel["y"]), (200.0, 200.0))
