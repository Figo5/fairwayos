"""Context geometry, camera compensation and body-relative features.

The v2 pass shipped body rejection that was never run, and a fixed 64 px crop
that inverted completely on balls larger than its training scale. These tests
pin down the two things that must hold for v3: a candidate stays at the centre of
its own description at every scale, and body is a measured feature rather than an
unexercised code path.
"""
from __future__ import annotations

import unittest

import numpy as np

from ghostcaddie.video import ball_v3_body as bbody
from ghostcaddie.video import ball_v3_context as bctx


def frame(width: int = 400, height: int = 300) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


class ContextGeometry(unittest.TestCase):
    def test_stack_has_the_declared_shape(self):
        stack = bctx.candidate_stack(frame(), None, 200.0, 150.0, 6.0)
        self.assertEqual(stack.shape, (bctx.GRID, bctx.GRID, bctx.CHANNELS))
        self.assertEqual(stack.dtype, np.uint8)

    def test_the_candidate_lands_at_the_centre_at_every_scale(self):
        """A bright disc at (x, y) must appear at the middle of its own crop
        whether it is 2 px or 70 px in radius. This is the property v2's fixed
        64 px crop did not have, and it is why a 72 px ball scored 0.069."""
        import cv2
        for radius in (2.0, 6.0, 20.0, 70.0):
            with self.subTest(radius=radius):
                canvas = np.zeros((600, 800, 3), np.uint8)
                cv2.circle(canvas, (400, 300), int(radius), (255, 255, 255), -1)
                stack = bctx.candidate_stack(canvas, None, 400.0, 300.0, radius)
                tight = stack[..., :3].mean(axis=2)
                cy, cx = np.unravel_index(int(np.argmax(tight)), tight.shape)
                self.assertLess(abs(cx - bctx.GRID / 2), 10, "x off centre")
                self.assertLess(abs(cy - bctx.GRID / 2), 10, "y off centre")

    def test_a_candidate_at_the_frame_edge_stays_centred(self):
        """Clamping the window into the frame would move the candidate off the
        centre of its own crop, so the network reads a position nobody asked
        about. The window pads instead -- by replication, so the padding does not
        fabricate a mirrored copy of the ball just inside the crop."""
        import cv2
        canvas = np.zeros((300, 400, 3), np.uint8)
        cv2.circle(canvas, (3, 150), 3, (255, 255, 255), -1)
        stack = bctx.candidate_stack(canvas, None, 3.0, 150.0, 3.0)
        tight = stack[..., :3].mean(axis=2)
        cy, cx = np.unravel_index(int(np.argmax(tight)), tight.shape)
        self.assertLess(abs(cx - bctx.GRID / 2), 10)
        self.assertLess(abs(cy - bctx.GRID / 2), 10)

    def test_context_window_is_wider_than_the_tight_window(self):
        import cv2
        canvas = np.zeros((600, 800, 3), np.uint8)
        cv2.circle(canvas, (400, 300), 6, (255, 255, 255), -1)
        cv2.circle(canvas, (400, 380), 9, (255, 255, 255), -1)   # a neighbour
        stack = bctx.candidate_stack(canvas, None, 400.0, 300.0, 6.0)
        # the neighbour is 80 px below: outside the 48 px tight window, inside the
        # 192 px context window. Count distinct bright blobs, not bright pixels --
        # the tight stream magnifies one disc and so has more bright pixels.

        def blobs(view: np.ndarray) -> int:
            mask = (view.mean(axis=2) > 128).astype(np.uint8)
            count, _labels = cv2.connectedComponents(mask)
            return count - 1

        self.assertEqual(blobs(stack[..., :3]), 1)
        self.assertEqual(blobs(stack[..., 3:6]), 2,
                         "the context stream should see the neighbour the tight "
                         "stream cannot")

    def test_motion_channel_is_empty_without_a_previous_frame(self):
        stack = bctx.candidate_stack(frame(), None, 200.0, 150.0, 6.0)
        self.assertEqual(int(stack[..., 6].max()), 0)

    def test_camera_compensation_cancels_a_pure_pan(self):
        """A panning camera moves the background further between frames than a
        struck ball moves. Without compensation every motion feature is dominated
        by the camera."""
        import cv2
        base = frame(400, 300)
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        shifted = np.roll(gray, 7, axis=1)
        uncompensated = bctx.motion_residual(shifted, gray)
        compensated = bctx.motion_residual(shifted, bctx.compensate(gray, (7.0, 0.0)))
        inner = (slice(20, -20), slice(20, -20))
        self.assertLess(float(compensated[inner].mean()),
                        float(uncompensated[inner].mean()) / 2.0)

    def test_geometry_features_are_resolution_independent(self):
        """The same swing at 1080p and 4K must produce the same numbers."""
        # the same point, half way across and a quarter of the way down
        small = bctx.geometry_features(540.0, 480.0, 6.0, 1080, 1920)
        large = bctx.geometry_features(1920.0, 540.0, 21.3, 3840, 2160)
        self.assertAlmostEqual(small[0], large[0], places=2)
        self.assertAlmostEqual(small[1], large[1], places=2)


POSE = {"keypoints": {
    "sh_l": [100.0, 100.0, 0.9], "sh_r": [160.0, 100.0, 0.9],
    "hip_l": [105.0, 200.0, 0.9], "hip_r": [155.0, 200.0, 0.9],
    "wr_l": [130.0, 150.0, 0.8], "wr_r": [140.0, 150.0, 0.8],
    "ank_l": [110.0, 300.0, 0.8], "ank_r": [150.0, 300.0, 0.8],
    "nose": [130.0, 70.0, 0.9],
}}


class BodyFeatures(unittest.TestCase):
    def test_absent_pose_is_a_state_not_a_crash(self):
        values = bbody.features(None, 50.0, 50.0, 5.0)
        self.assertEqual(len(values), len(bbody.FEATURE_NAMES))
        self.assertEqual(values[0], 0.0, "body_available must say nobody looked")
        self.assertTrue(all(v == 0.0 for v in values))

    def test_available_pose_is_flagged(self):
        values = bbody.features(POSE, 50.0, 50.0, 5.0)
        self.assertEqual(values[0], 1.0)

    def test_a_candidate_inside_the_torso_is_detected(self):
        values = bbody.features(POSE, 130.0, 150.0, 5.0)
        self.assertEqual(values[bbody.FEATURE_NAMES.index("inside_torso")], 1.0)

    def test_a_candidate_on_the_turf_is_outside_the_torso(self):
        values = bbody.features(POSE, 400.0, 400.0, 5.0)
        self.assertEqual(values[bbody.FEATURE_NAMES.index("inside_torso")], 0.0)
        self.assertEqual(values[bbody.FEATURE_NAMES.index("inside_body_box")], 0.0)

    def test_body_proximity_is_a_feature_and_never_an_exclusion(self):
        """At address and through impact the ball is legitimately near the hands
        and feet. A tracker that rejects everything near the body deletes the
        correct answer on exactly the frames that matter most, so this returns a
        distance and nothing here returns a verdict."""
        at_the_feet = bbody.features(POSE, 130.0, 305.0, 5.0)
        far_away = bbody.features(POSE, 700.0, 305.0, 5.0)
        feet = bbody.FEATURE_NAMES.index("distance_to_feet_over_radius")
        self.assertLess(at_the_feet[feet], far_away[feet])
        self.assertTrue(all(isinstance(v, float) for v in at_the_feet))

    def test_torso_centre_needs_enough_keypoints(self):
        self.assertIsNotNone(bbody.torso_centre(POSE))
        self.assertIsNone(bbody.torso_centre(
            {"keypoints": {"nose": [1.0, 1.0, 0.9]}}))

    def test_low_confidence_keypoints_are_ignored(self):
        faint = {"keypoints": {name: [x, y, 0.05]
                               for name, (x, y, _s) in
                               ((k, v) for k, v in POSE["keypoints"].items())}}
        self.assertEqual(bbody.features(faint, 130.0, 150.0, 5.0)[0], 0.0)


if __name__ == "__main__":
    unittest.main()
