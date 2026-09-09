"""Focused tests for the research-only seeded-ball overlay seam."""
import unittest

from ghostcaddie.video.research_ball import (
    SeededBallTrackItem,
    SeededBallTrackResult,
)
from ghostcaddie.video.research_overlay import build_seeded_ball_overlay_filter


def _item(frame, center, provenance, confidence=0.8, warnings=()):
    return SeededBallTrackItem(frame, center, confidence, provenance, warnings)


def _result(items, **overrides):
    kwargs = {"track_id": "ball-seed-0", "longest_gap": 0,
              "provenance": "research_candidate",
              "production_eligible": False, "ground_truth": False}
    kwargs.update(overrides)
    return SeededBallTrackResult(kwargs.pop("track_id"),
                                 tuple(items),
                                 kwargs.pop("longest_gap"),
                                 **kwargs)


class SeededBallOverlayFilterTests(unittest.TestCase):
    def test_renders_state_distinct_markers_without_bridging_gaps(self):
        result = _result([
            _item(0, (50.0, 60.0), "seeded", 1.0),
            _item(1, (55.0, 62.0), "tracked", 0.9),
            _item(2, None, "unavailable", 0.0, ("appearance_or_motion_unavailable",)),
            _item(5, (70.0, 70.0), "tracked", 0.7),
        ])
        graph = build_seeded_ball_overlay_filter(result, width=200, height=150, radius=8)
        # seeded marker is green, tracked markers are yellow
        self.assertIn("color=green:t=2:enable='eq(n\\,0)'", graph)
        self.assertIn("color=yellow:t=2:enable='eq(n\\,1)'", graph)
        self.assertIn("color=yellow:t=2:enable='eq(n\\,5)'", graph)
        # trail dot only between consecutive supported frames 0->1
        self.assertEqual(graph.count("enable='gte(n\\,1)'"), 1)
        # unavailable frame gets a distinct red tick, no candidate marker
        self.assertIn("color=red", graph)
        self.assertIn("enable='eq(n\\,2)'", graph)
        # only one trail dot: between consecutive frames 0 and 1
        self.assertEqual(graph.count("yellow@0.75"), 1)

    def test_gap_between_supported_frames_adds_no_trail(self):
        result = _result([
            _item(0, (50.0, 60.0), "seeded", 1.0),
            _item(3, (70.0, 70.0), "tracked", 0.7),
        ])
        graph = build_seeded_ball_overlay_filter(result, width=200, height=150, radius=8)
        self.assertNotIn("gte(n", graph)

    def test_requires_research_only_result(self):
        items = [_item(0, (50.0, 60.0), "seeded", 1.0)]
        for overrides in ({"production_eligible": True}, {"ground_truth": True}):
            with self.assertRaises(ValueError):
                build_seeded_ball_overlay_filter(
                    _result(items, **overrides), width=200, height=150, radius=8)

    def test_rejects_unsupported_provenance_or_mismatched_center(self):
        with self.assertRaises(ValueError):
            build_seeded_ball_overlay_filter(
                _result([_item(0, (50.0, 60.0), "predicted")]),
                width=200, height=150, radius=8)
        with self.assertRaises(ValueError):
            build_seeded_ball_overlay_filter(
                _result([_item(0, None, "seeded")]),
                width=200, height=150, radius=8)
        with self.assertRaises(ValueError):
            build_seeded_ball_overlay_filter(
                _result([_item(0, (50.0, 60.0), "unavailable")]),
                width=200, height=150, radius=8)

    def test_validates_bounds_and_geometry(self):
        with self.assertRaises(ValueError):
            build_seeded_ball_overlay_filter(
                _result([_item(0, (250.0, 60.0), "seeded")]),
                width=200, height=150, radius=8)
        with self.assertRaises(ValueError):
            build_seeded_ball_overlay_filter(
                _result([_item(0, (50.0, 60.0), "seeded"),
                         _item(0, (60.0, 60.0), "tracked")]),
                width=200, height=150, radius=8)
        with self.assertRaises(ValueError):
            build_seeded_ball_overlay_filter(
                _result([_item(0, (50.0, 60.0), "seeded")]),
                width=200, height=150, radius=0)
        with self.assertRaises(ValueError):
            build_seeded_ball_overlay_filter(_result([]), width=200, height=150, radius=8)


if __name__ == "__main__":
    unittest.main()