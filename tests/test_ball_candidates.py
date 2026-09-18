"""Candidate-ensemble tests: independence, camera compensation, and honesty.

The ensemble exists because one method's opinion is not evidence. These tests
check that each generator can find a small disc on its own, that the
camera-compensated methods do not simply fire on everything when the camera pans,
and that a frame with no ball does not manufacture candidates.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghostcaddie.video import ball_candidates as bc  # noqa: E402


def textured_background(width: int = 320, height: int = 240, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[..., 1] = 105 + rng.normal(0, 8, (height, width))
    base[..., 0] = 62 + rng.normal(0, 6, (height, width))
    base[..., 2] = 48 + rng.normal(0, 6, (height, width))
    # some structure so "local contrast" has something to reject
    for _ in range(25):
        x, y = rng.integers(0, width), rng.integers(0, height)
        cv2.circle(base, (int(x), int(y)), int(rng.integers(3, 9)),
                   tuple(float(v) for v in rng.integers(70, 190, 3)), -1)
    return np.clip(base, 0, 255).astype(np.uint8)


def add_disc(frame: np.ndarray, x: int, y: int, radius: int = 4,
             colour=(240, 240, 240)) -> np.ndarray:
    out = frame.copy()
    cv2.circle(out, (x, y), radius, colour, -1)
    return out


def shift(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]),
                          borderMode=cv2.BORDER_REFLECT)


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.background = textured_background()
        self.ball_x, self.ball_y = 160, 130
        self.frame = add_disc(self.background, self.ball_x, self.ball_y, 4)
        self.contexts, self.info = bc.build_contexts([self.frame], [0])
        self.context = self.contexts[0]

    def test_each_method_can_find_a_small_bright_disc_on_its_own(self):
        for name in ("radial_symmetry", "local_contrast", "brightness_cue"):
            candidates = bc.GENERATORS[name](self.context)
            self.assertTrue(candidates, f"{name} found nothing on a plain disc")
            nearest = min(math.hypot(c.x - self.ball_x, c.y - self.ball_y)
                          for c in candidates)
            self.assertLess(nearest, 12.0,
                            f"{name} did not place a candidate near the disc "
                            f"(closest {nearest:.1f} px)")

    def test_candidates_carry_their_method_and_cues(self):
        candidates = bc.GENERATORS["local_contrast"](self.context)
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertEqual(candidate.method, "local_contrast")
            self.assertTrue(candidate.cues)
            self.assertGreaterEqual(candidate.score, 0.0)
            self.assertLessEqual(candidate.score, 1.0)

    def test_the_ensemble_keeps_methods_independent_and_grouped_by_frame(self):
        per_frame = bc.ensemble_candidates(self.contexts)
        self.assertEqual(set(per_frame), {0})
        methods = {candidate.method for candidate in per_frame[0]}
        self.assertGreater(len(methods), 1,
                           "on a clean disc several independent methods should fire")
        self.assertTrue(methods <= set(bc.METHOD_NAMES))

    def test_an_unknown_method_is_refused(self):
        with self.assertRaises(ValueError):
            bc.ensemble_candidates(self.contexts, methods=("nonsense",))

    def test_a_featureless_frame_does_not_manufacture_candidates(self):
        flat = np.full((240, 320, 3), 120, dtype=np.uint8)
        contexts, _ = bc.build_contexts([flat], [0])
        per_frame = bc.ensemble_candidates(contexts)
        self.assertLessEqual(len(per_frame[0]), 3,
                             "a blank frame must not generate a crowd of candidates")


class CameraCompensationTests(unittest.TestCase):
    def test_frame_difference_finds_a_mover_and_ignores_a_pan(self):
        base = textured_background()
        mover = add_disc(base, 100, 120, 4)

        # a small object that moves independently of the camera
        first = add_disc(base, 100, 120, 4)
        second = add_disc(base, 106, 122, 4)
        contexts, info = bc.build_contexts([first, second], [0, 1])
        candidates = bc.candidates_frame_difference(contexts[1])
        near = [c for c in candidates
                if math.hypot(c.x - 106, c.y - 122) < 14]
        self.assertTrue(near, "a mover should be proposed by frame differencing")
        self.assertGreater(info["median_camera_motion_fraction"], 0.0)

        # a pure pan: the same scene translated, nothing independent
        panned = [base, shift(base, 12, 4)]
        contexts, _ = bc.build_contexts(panned, [0, 1])
        pan_candidates = bc.candidates_frame_difference(contexts[1])
        strong = [c for c in pan_candidates if c.score > 0.5]
        self.assertLessEqual(
            len(strong), 2,
            "after compensation a pure pan must not light up the frame; got "
            f"{len(strong)} strong candidates")

    def test_camera_motion_estimate_is_close_to_the_applied_shift(self):
        base = textured_background()
        moved = shift(base, 9, -5)
        contexts, _ = bc.build_contexts([base, moved], [0, 1])
        dx, dy = contexts[1].motion
        # convention: motion is how far the scene moved between the two frames,
        # so the second frame shifted by (+9, -5) reports (+9, -5)
        self.assertLess(abs(dx - 9.0), 3.0)
        self.assertLess(abs(dy - (-5.0)), 3.0)

    def test_flow_residual_ignores_static_scene(self):
        base = textured_background()
        contexts, _ = bc.build_contexts([base, base.copy()], [0, 1])
        candidates = bc.candidates_flow_residual(contexts[1])
        strong = [c for c in candidates if c.score > 0.6]
        self.assertLessEqual(len(strong), 2)


class WorkingScaleTests(unittest.TestCase):
    def test_coordinates_are_mapped_back_to_native_pixels(self):
        # a 4000-px-wide frame is processed downscaled but reported natively
        big = np.zeros((200, 4000, 3), dtype=np.uint8)
        big[:] = (60, 110, 60)
        big = add_disc(big, 3000, 100, 6)
        contexts, info = bc.build_contexts([big], [0])
        self.assertLess(info["work_scale"], 1.0)
        per_frame = bc.ensemble_candidates(contexts)
        best = max(per_frame[0], key=lambda c: c.score)
        self.assertGreater(best.x, 2000,
                           "native coordinates must survive the downscale")


if __name__ == "__main__":
    unittest.main()
