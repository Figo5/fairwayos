"""Synthetic scenes for the ball pipeline.

Each scene isolates one failure mode that the research plan names, so a
regression shows up as a named case rather than as a number moving. The scenes
are drawn, not filmed: the ground truth is exact, and a test that fails is a
statement about the code rather than about a clip.
"""

import unittest

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - the portable suite skips without cv2
    cv2 = None

from ghostcaddie.video.ball_pipeline import (
    BallObservation,
    BallParams,
    BallState,
    estimate_camera_motion,
    generate_candidates,
    track_ball,
    validate_track,
)

WIDTH, HEIGHT = 640, 360
RADIUS = 9
BALL = (235, 240, 245)


def turf(seed: int = 7) -> np.ndarray:
    """A textured background with the statistics of mown grass, not flat colour.

    Flat backgrounds make any blob detector look brilliant. The point of these
    scenes is that the detector has to work against clutter.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, (HEIGHT, WIDTH, 3)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 1.6)
    base = np.array([60.0, 120.0, 70.0], dtype=np.float32)
    return np.clip(base + noise * 26.0, 0, 255).astype(np.uint8)


def draw_ball(frame, x, y, radius=RADIUS, colour=BALL):
    cv2.circle(frame, (int(round(x)), int(round(y))), int(radius), colour, -1, cv2.LINE_AA)
    return frame


def scene(positions, *, seed=7, extra=None, shift=(0.0, 0.0)):
    """Frames with the ball at ``positions`` (None means 'not in this frame')."""
    frames = []
    for index, position in enumerate(positions):
        frame = turf(seed).copy()
        if shift != (0.0, 0.0):
            matrix = np.float32([[1, 0, shift[0] * index], [0, 1, shift[1] * index]])
            frame = cv2.warpAffine(frame, matrix, (WIDTH, HEIGHT), borderMode=cv2.BORDER_REFLECT)
        # Ball first, then ``extra`` -- an occluder drawn underneath the ball
        # occludes nothing, which is a way to write a test that always passes.
        if position is not None:
            draw_ball(frame, *position)
        if extra is not None:
            extra(frame, index)
        frames.append(frame)
    return frames


def observed(result):
    return {o.frame_index: (o.x, o.y) for o in result.observations if o.state is BallState.OBSERVED}


def states(result):
    return {o.frame_index: o.state for o in result.observations}


@unittest.skipIf(cv2 is None, "OpenCV is required for the ball pipeline")
class BallContractTests(unittest.TestCase):
    def test_only_positioned_states_may_carry_a_position(self):
        for state in (BallState.AMBIGUOUS, BallState.OCCLUDED, BallState.LOST,
                      BallState.UNAVAILABLE):
            with self.assertRaises(ValueError):
                BallObservation(0, state, x=1.0, y=2.0)
        with self.assertRaises(ValueError):
            BallObservation(0, BallState.OBSERVED)
        BallObservation(0, BallState.INTERPOLATED, x=1.0, y=2.0)

    def test_a_rejected_track_publishes_no_positions_at_all(self):
        rejected = [BallObservation(index, BallState.OBSERVED, 10.0, 10.0, 4.0, 0.05)
                    for index in range(6)]
        state, diagnostics, limitations = validate_track(
            rejected, BallParams(), width=WIDTH, height=HEIGHT, camera_severity=0.0)
        self.assertIs(state, BallState.UNAVAILABLE)
        self.assertTrue(limitations)
        self.assertAlmostEqual(diagnostics["coverage"], 1.0)

    def test_low_coverage_is_rejected_even_when_the_few_points_score_well(self):
        sparse = [BallObservation(0, BallState.OBSERVED, 10.0, 10.0, 4.0, 1.0)]
        sparse += [BallObservation(index, BallState.LOST) for index in range(1, 10)]
        state, _diagnostics, _limitations = validate_track(
            sparse, BallParams(), width=WIDTH, height=HEIGHT, camera_severity=0.0)
        self.assertIs(state, BallState.UNAVAILABLE)

    def test_severe_camera_motion_is_reported_as_a_limitation(self):
        good = [BallObservation(index, BallState.OBSERVED, 10.0 + index, 10.0, 4.0, 0.9)
                for index in range(8)]
        _state, _diagnostics, limitations = validate_track(
            good, BallParams(), width=WIDTH, height=HEIGHT, camera_severity=0.2)
        self.assertTrue(any("camera" in text for text in limitations))

    def test_serialisation_keeps_states_and_never_invents_coordinates(self):
        frames = scene([(320, 180)] * 8)
        payload = track_ball(frames, list(range(8))).as_dict()
        self.assertEqual(len(payload["observations"]), 8)
        for row in payload["observations"]:
            if row["state"] != "observed":
                self.assertIsNone(row["x"])
                self.assertIsNone(row["y"])
        self.assertNotIn("interpolated", {row["state"] for row in payload["observations"]})


@unittest.skipIf(cv2 is None, "OpenCV is required for the ball pipeline")
class CameraMotionTests(unittest.TestCase):
    def test_a_known_pan_is_recovered(self):
        first = cv2.cvtColor(turf(3), cv2.COLOR_BGR2GRAY).astype(np.float32)
        matrix = np.float32([[1, 0, 7], [0, 1, -4]])
        second = cv2.warpAffine(first, matrix, (WIDTH, HEIGHT), borderMode=cv2.BORDER_REFLECT)
        dx, dy = estimate_camera_motion(first, second)
        self.assertAlmostEqual(dx, 7.0, delta=1.0)
        self.assertAlmostEqual(dy, -4.0, delta=1.0)

    def test_a_ball_held_still_by_a_pan_is_not_read_as_a_moving_ball(self):
        # The ball keeps the same pixel position while the background slides,
        # which in compensated coordinates means the ball is moving with the
        # scene -- the association gate has to survive that without breaking.
        frames = scene([(320, 180)] * 12, shift=(3.0, 0.0))
        result = track_ball(frames, list(range(12)))
        self.assertIs(result.state, BallState.OBSERVED)
        self.assertGreaterEqual(len(observed(result)), 8)


@unittest.skipIf(cv2 is None, "OpenCV is required for the ball pipeline")
class BallSceneTests(unittest.TestCase):
    def test_a_small_moving_object_is_tracked_through_the_window(self):
        positions = [(140 + 11 * index, 150 + 4 * index) for index in range(14)]
        result = track_ball(scene(positions), list(range(14)))
        self.assertIs(result.state, BallState.OBSERVED)
        points = observed(result)
        self.assertGreaterEqual(len(points), 10)
        for index, point in points.items():
            self.assertLess(np.hypot(point[0] - positions[index][0],
                                     point[1] - positions[index][1]), 3 * RADIUS)

    def test_a_static_bright_stripe_is_never_reported_as_a_ball(self):
        # A flagstick: bright, persistent, high contrast, and not round. This is
        # the object the withdrawn heuristic tracker followed.
        def flagstick(frame, _index):
            cv2.rectangle(frame, (300, 40), (306, 300), (245, 245, 245), -1)

        result = track_ball(scene([None] * 12, extra=flagstick), list(range(12)))
        for point in observed(result).values():
            self.assertGreater(abs(point[0] - 303), 4 * RADIUS,
                               "a prediction landed on the static stripe")

    def test_a_body_sized_blob_is_not_mistaken_for_a_ball(self):
        def torso(frame, _index):
            cv2.ellipse(frame, (320, 180), (70, 130), 0, 0, 360, (210, 210, 215), -1)

        result = track_ball(scene([None] * 12, extra=torso), list(range(12)))
        for point in observed(result).values():
            self.assertGreater(np.hypot(point[0] - 320, point[1] - 180), 70,
                               "a prediction landed inside the body-sized blob")

    def test_a_temporary_occlusion_produces_occluded_not_a_stale_coordinate(self):
        positions = [(150 + 12 * index, 180) for index in range(14)]
        hidden = {6, 7}

        def cover(frame, index):
            # A wide dark bar, not a disc: occluding a circle with a circle just
            # swaps one detectable object for another and tests nothing.
            if index in hidden:
                x, y = positions[index]
                cv2.rectangle(frame, (int(x) - 120, int(y) - 22), (int(x) + 120, int(y) + 22),
                              (30, 62, 38), -1)

        result = track_ball(scene(positions, extra=cover), list(range(14)))
        points = observed(result)
        for index in hidden:
            self.assertNotIn(index, points, "an occluded frame carried a coordinate")
        self.assertGreaterEqual(len(points), 8)

    def test_an_object_leaving_the_frame_ends_the_track_without_a_last_known_point(self):
        positions = [(120 + 40 * index, 180) for index in range(6)]
        positions += [None] * 6
        result = track_ball(scene(positions), list(range(12)))
        points = observed(result)
        self.assertFalse([index for index in points if index >= 7],
                         "the track kept publishing a position after the ball left")

    def test_two_equally_good_candidates_make_the_frame_ambiguous_not_a_guess(self):
        # A second identical ball a short hop away, appearing mid-window: the
        # honest answer is that this frame does not identify the ball.
        positions = [(200, 180)] * 12

        def twin(frame, index):
            # Far enough apart to be resolved as two objects, close enough that
            # both sit inside one frame's motion gate.
            if index >= 6:
                draw_ball(frame, 200 + 3.0 * RADIUS, 180)

        result = track_ball(scene(positions, extra=twin), list(range(12)))
        later = {index: state for index, state in states(result).items() if index >= 6}
        self.assertTrue(any(state is BallState.AMBIGUOUS for state in later.values()),
                        f"competition never produced an ambiguous frame: {later}")

    def test_an_empty_scene_yields_no_positions_rather_than_a_best_guess(self):
        result = track_ball(scene([None] * 12), list(range(12)))
        if result.state is BallState.UNAVAILABLE:
            self.assertFalse(observed(result))
        else:
            self.assertLessEqual(len(observed(result)), 12)

    def test_candidate_generation_finds_the_ball_and_scores_it_above_the_turf(self):
        frame = draw_ball(turf(11).copy(), 300, 170)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        candidates = generate_candidates(gray, 0, BallParams(), None, frame.astype(np.float32))
        self.assertTrue(candidates)
        best = max(candidates, key=lambda candidate: candidate.score)
        self.assertLess(np.hypot(best.x - 300, best.y - 170), 2 * RADIUS)

    def test_a_ball_hanging_off_the_frame_edge_is_not_scored_at_all(self):
        frame = draw_ball(turf(11).copy(), 4, 170)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        candidates = generate_candidates(gray, 0, BallParams(), None, frame.astype(np.float32))
        self.assertFalse([c for c in candidates if c.x < 30],
                         "a candidate was scored without a measurable background")


if __name__ == "__main__":
    unittest.main()
