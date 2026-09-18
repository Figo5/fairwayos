"""Tracker v2 tests: states, abstention, no stale coordinates, isolation.

The states are the contract with the rest of the system: a viewer must be able to
tell an observation from an interpolation, and a consumer must never receive a
coordinate for a frame the tracker did not actually observe.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghostcaddie.video import ball_tracker_v2 as btr  # noqa: E402
from ghostcaddie.video import canonical  # noqa: E402


def background(width: int = 480, height: int = 320, seed: int = 21) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[..., 1] = 105 + rng.normal(0, 6, (height, width))
    base[..., 0] = 60 + rng.normal(0, 5, (height, width))
    base[..., 2] = 48 + rng.normal(0, 5, (height, width))
    for _ in range(30):
        cv2.circle(base, (int(rng.integers(0, width)), int(rng.integers(0, height))),
                   int(rng.integers(3, 8)), tuple(float(v) for v in rng.integers(70, 190, 3)), -1)
    return np.clip(base, 0, 255).astype(np.uint8)


def with_ball(frame: np.ndarray, x: int, y: int, radius: int = 5) -> np.ndarray:
    out = frame.copy()
    cv2.circle(out, (x, y), radius, (244, 244, 244), -1)
    return out


class StateContractTests(unittest.TestCase):
    def test_an_empty_sequence_is_unavailable_not_invented(self):
        frames = [background() for _ in range(6)]
        result = btr.track_ball_v2(frames, list(range(6)))
        self.assertFalse(result.accepted)
        for state in result.states:
            self.assertNotEqual(state.state, btr.STATE_OBSERVED)
            payload = state.to_dict()
            self.assertNotIn("x", payload)
            self.assertNotIn("y", payload)

    def test_every_state_is_one_of_the_contract(self):
        frames = [with_ball(background(), 100 + i * 8, 150, 5) for i in range(8)]
        result = btr.track_ball_v2(frames, list(range(8)))
        for state in result.states:
            self.assertIn(state.state, btr.TRACK_STATES)

    def test_interpolation_is_marked_and_never_becomes_an_observation(self):
        frames = []
        for index in range(9):
            if index in (4,):
                frames.append(background())              # ball hidden for one frame
            else:
                frames.append(with_ball(background(), 100 + index * 6, 150, 5))
        result = btr.track_ball_v2(frames, list(range(9)), interpolate=True)
        for state in result.states:
            if state.interpolated:
                self.assertNotEqual(state.state, btr.STATE_OBSERVED,
                                    "an interpolated position is not an observation")
                self.assertIn("interpolated", state.to_dict())

    def test_a_candidate_that_never_moves_while_the_camera_pans_is_flagged(self):
        """Long-lived stationarity under camera motion is a caution, not a track.

        A ball genuinely at rest looks like this too, which is why it is a demotion
        and a named diagnostic rather than a silent rejection: the tracker refuses
        the identity and says why instead of guessing.
        """
        scene = background()
        frames = [with_ball(np.roll(scene, index * 4, axis=1), 200, 160, 5)
                  for index in range(14)]
        result = btr.track_ball_v2(frames, list(range(14)))
        flagged = any("STATIC_SCENE_LOCK" in state.diagnostics for state in result.states)
        self.assertTrue(flagged or not result.accepted,
                        "a candidate that rides the frame instead of the world must be "
                        "flagged or the track refused")
        if result.accepted:
            self.assertIn("STATIC_SCENE_LOCK",
                          {kind for state in result.states for kind in state.diagnostics})

    def test_a_short_stationary_ball_is_still_allowed(self):
        scene = background()
        frames = [with_ball(np.roll(scene, index * 4, axis=1), 200, 160, 5)
                  for index in range(6)]
        result = btr.track_ball_v2(frames, list(range(6)))
        self.assertFalse(any("STATIC_SCENE_LOCK" in state.diagnostics
                             for state in result.states),
                         "a ball at rest over a few frames is a ball at rest, not "
                         "scenery")

    def test_reasons_are_given_when_a_track_is_not_accepted(self):
        frames = [background() for _ in range(5)]
        result = btr.track_ball_v2(frames, list(range(5)))
        self.assertFalse(result.accepted)
        self.assertTrue(result.reasons)
        self.assertTrue(all(isinstance(reason, str) and reason for reason in result.reasons))

    def test_confidence_reports_the_metrics_the_rules_use(self):
        frames = [with_ball(background(), 120 + i * 5, 150, 5) for i in range(8)]
        result = btr.track_ball_v2(frames, list(range(8)))
        for key in ("observed_frames", "span_frames", "coverage", "mean_score",
                    "cycle_consistency_rate", "body_lock_rate", "static_lock_rate",
                    "abrupt_jumps", "state_share"):
            self.assertIn(key, result.confidence)


class BodyRegionTests(unittest.TestCase):
    def test_a_body_region_raises_the_bar_rather_than_blinding_the_tracker(self):
        frames = []
        for index in range(8):
            frames.append(with_ball(background(), 200 + index * 4, 160, 5))
        regions = {index: [{"kind": "torso", "x": 150, "y": 90, "w": 200, "h": 180}]
                   for index in range(8)}
        result = btr.track_ball_v2(frames, list(range(8)), body_regions=regions)
        # either it abstains, or the frames it keeps are marked with the lock
        for state in result.states:
            if state.state == btr.STATE_OBSERVED:
                self.assertIn("BODY_LOCK", state.diagnostics)


class ParameterTests(unittest.TestCase):
    def test_no_parameter_requires_an_operator_seed(self):
        params = btr.TrackerParams()
        for field in params.__dict__:
            self.assertNotIn("seed", field.lower())
            self.assertNotIn("roi", field.lower())

    def test_an_empty_sequence_is_refused(self):
        with self.assertRaises(ValueError):
            btr.track_ball_v2([], [])

    def test_mismatched_lengths_are_refused(self):
        with self.assertRaises(ValueError):
            btr.track_ball_v2([background()], [0, 1])


class CanonicalIsolationTests(unittest.TestCase):
    def test_canonical_ball_is_still_unavailable(self):
        layer = canonical.ball_layer()
        self.assertEqual(layer.state, canonical.LayerState.UNAVAILABLE)
        self.assertEqual(layer.method, "none")

    def test_the_canonical_path_does_not_import_the_v2_tracker(self):
        import ast
        for module in ("ghostcaddie/video/canonical.py", "ghostcaddie/cli.py"):
            tree = ast.parse((ROOT / module).read_text(encoding="utf-8"))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
                    imported.update(f"{node.module}.{alias.name}"
                                    for alias in node.names)
            offenders = {name for name in imported
                         if any(key in name for key in
                                ("ball_tracker_v2", "ball_detector", "ball_consensus",
                                 "ball_candidates", "ball_synthetic"))}
            self.assertFalse(offenders, f"{module} imports research tracker code: {offenders}")

    def test_promotion_requires_the_promoted_classification(self):
        """The canonical gate is a classification, and nothing else may open it."""
        from pathlib import Path as _Path
        results = ROOT / "docs" / "ball-tracker-v2-results.md"
        if not results.exists():
            self.skipTest("results document not written yet")
        text = results.read_text(encoding="utf-8").lower()
        layer_unavailable = canonical.ball_layer().state == canonical.LayerState.UNAVAILABLE
        if layer_unavailable:
            self.assertTrue("promoted" not in text.split("classification")[-1][:80]
                            or "not promoted" in text,
                            "canonical ball is UNAVAILABLE, so the results document "
                            "must not claim promotion")


if __name__ == "__main__":
    unittest.main()
