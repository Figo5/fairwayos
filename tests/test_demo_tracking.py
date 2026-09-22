"""Tracking-policy tests: statuses, gap/loss rules, corrections, interpolation.

These run on plain dictionaries — no video, no cv2, no model — because the policy
is deliberately separate from the measurement. The cases that matter are the ones
where a renderer could otherwise lie: a stale marker after a miss, a trail bridged
across a loss or a correction, and an interpolated point counted as an observation.
"""
from __future__ import annotations

import unittest

from ghostcaddie.demo.config import Point, TargetOptions
from ghostcaddie.demo.errors import DemoError
from ghostcaddie.demo.tracking import (ORIGIN_AUTOMATIC, ORIGIN_DERIVED,
                                       ORIGIN_OPERATOR, RawMeasurement,
                                       STATUS_INTERPOLATED, STATUS_LOST,
                                       STATUS_MANUAL, STATUS_MISSING,
                                       STATUS_OBSERVED, STATUS_OBSERVED_LK,
                                       resolve_track, track_from_dict, trail_window)

FRAMES = list(range(0, 20))


def template_options(**overrides) -> TargetOptions:
    values = {"mode": "template", "template_size": 21, "search_radius": 30,
              "min_score": 0.5, "max_jump_px": 40.0, "loss_frames": 3,
              "direction": "forward"}
    values.update(overrides)
    return TargetOptions(**values)


def measurements(found_frames, *, x_step: float = 1.0, start_x: float = 100.0):
    raw = {}
    for frame in range(0, 20):
        if frame in found_frames:
            raw[frame] = RawMeasurement(frame, True, x=start_x + frame * x_step,
                                        y=200.0, score=0.9, method="template")
        else:
            raw[frame] = RawMeasurement(frame, False, method="template",
                                        reason="score_below_min (0.20)")
    return raw


class StatusTests(unittest.TestCase):
    def test_a_seed_is_an_operator_anchor_never_a_detection(self):
        track = resolve_track(
            "ball", options=template_options(), interval_frames=FRAMES,
            raw=measurements(range(1, 20)), seed=Point(0, 100.0, 200.0))
        first = track.points[0]
        self.assertEqual(first.status, STATUS_MANUAL)
        self.assertEqual(first.origin, ORIGIN_OPERATOR)
        self.assertTrue(first.observed)
        self.assertFalse(first.automatic)

    def test_template_matches_are_the_only_automatic_observations(self):
        track = resolve_track(
            "ball", options=template_options(), interval_frames=FRAMES,
            raw=measurements(range(1, 20)), seed=Point(0, 100.0, 200.0))
        automatic = [p for p in track.points if p.automatic]
        self.assertEqual(len(automatic), 19)
        self.assertTrue(all(p.status == STATUS_OBSERVED for p in automatic))
        self.assertTrue(all(p.origin == ORIGIN_AUTOMATIC for p in automatic))

    def test_the_lk_refinement_is_reported_as_its_own_method(self):
        raw = measurements(range(1, 20))
        raw[5] = RawMeasurement(5, True, x=105, y=200, score=0.8, method="template+lk")
        track = resolve_track("ball", options=template_options(lk_refine=True),
                              interval_frames=FRAMES, raw=raw,
                              seed=Point(0, 100.0, 200.0))
        self.assertEqual(track.points[5].status, STATUS_OBSERVED_LK)
        self.assertEqual(track.points[5].method, "template+lk")

    def test_an_unsupported_target_draws_nothing_and_says_why(self):
        track = resolve_track("clubhead", options=template_options(),
                              interval_frames=FRAMES, raw={}, seed=None)
        self.assertFalse(track.available)
        self.assertTrue(all(p.status == STATUS_MISSING for p in track.points))
        self.assertFalse(any(p.drawn for p in track.points))
        self.assertEqual(track.coverage["observed_frames"], 0)
        self.assertEqual(track.accounting["path_length_observed_px"], 0)
        self.assertIn("no operator seed", track.reason)


class StaleMarkerTests(unittest.TestCase):
    def test_nothing_is_drawn_after_a_declared_loss(self):
        # observed 1..10, then nothing for the rest of the interval
        track = resolve_track("ball", options=template_options(loss_frames=3),
                              interval_frames=FRAMES, raw=measurements(range(1, 11)),
                              seed=Point(0, 100.0, 200.0))
        after = [p for p in track.points if p.source_frame >= 11]
        self.assertEqual({p.status for p in after}, {STATUS_LOST})
        self.assertFalse(any(p.drawn for p in after))
        self.assertEqual(track.coverage["lost_frames"], 9)
        self.assertEqual(track.coverage["longest_missing_run"], 9)
        self.assertFalse(any(p.x is not None for p in after))

    def test_a_short_gap_is_a_gap_and_a_run_at_the_threshold_is_a_loss(self):
        raw = measurements(range(1, 20))
        # frames 8 and 9 miss: two frames, below loss_frames=3
        for frame in (8, 9):
            raw[frame] = RawMeasurement(frame, False, method="template",
                                        reason="score_below_min (0.20)")
        track = resolve_track("ball", options=template_options(loss_frames=3),
                              interval_frames=FRAMES, raw=raw,
                              seed=Point(0, 100.0, 200.0),
                              interpolation_enabled=False, max_gap_frames=0)
        self.assertEqual(track.points[8].status, STATUS_MISSING)
        self.assertIn("short gap", track.points[8].note)
        self.assertEqual(track.coverage["lost_frames"], 0)

    def test_three_consecutive_misses_declare_a_loss(self):
        raw = measurements(range(1, 20))
        for frame in (8, 9, 10):
            raw[frame] = RawMeasurement(frame, False, method="template",
                                        reason="score_below_min (0.20)")
        for frame in range(11, 20):
            raw[frame] = RawMeasurement(frame, False, method="template",
                                        reason="track_declared_lost")
        track = resolve_track("ball", options=template_options(loss_frames=3),
                              interval_frames=FRAMES, raw=raw,
                              seed=Point(0, 100.0, 200.0))
        self.assertEqual(track.points[8].status, STATUS_LOST)
        self.assertEqual(track.points[19].status, STATUS_LOST)
        self.assertEqual(track.coverage["lost_frames"], 12)

    def test_a_track_that_resumes_without_a_correction_is_refused(self):
        raw = measurements(range(1, 20))
        for frame in (5, 6, 7):
            raw[frame] = RawMeasurement(frame, False, method="template",
                                        reason="score_below_min (0.20)")
        # frame 8 is an automatic match again: the policy must refuse this, because
        # a lost target is only restarted by an operator correction.
        with self.assertRaises(DemoError) as caught:
            resolve_track("ball", options=template_options(loss_frames=3),
                          interval_frames=FRAMES, raw=raw,
                          seed=Point(0, 100.0, 200.0),
                          interpolation_enabled=False, max_gap_frames=0)
        self.assertIn("follows a declared loss", str(caught.exception))


class StaleTrailTests(unittest.TestCase):
    def test_the_trail_is_the_current_segment_up_to_the_current_frame(self):
        track = resolve_track("ball", options=template_options(),
                              interval_frames=FRAMES, raw=measurements(range(1, 20)),
                              seed=Point(0, 100.0, 200.0))
        window = trail_window(track.points, 12, limit=15)
        self.assertEqual([p.source_frame for p in window], list(range(0, 13)))

    def test_the_trail_is_empty_once_the_target_is_lost(self):
        track = resolve_track("ball", options=template_options(loss_frames=3),
                              interval_frames=FRAMES, raw=measurements(range(1, 11)),
                              seed=Point(0, 100.0, 200.0))
        self.assertEqual(trail_window(track.points, 12), [])
        self.assertEqual(trail_window(track.points, 19), [])
        # ...and the last observed frame still has its own trail
        self.assertTrue(trail_window(track.points, 10))

    def test_the_trail_never_reaches_back_across_a_correction(self):
        track = resolve_track(
            "ball", options=template_options(), interval_frames=FRAMES,
            raw=measurements(range(1, 20)), seed=Point(0, 100.0, 200.0),
            corrections=[Point(12, 300.0, 210.0, "operator restart")])
        window = trail_window(track.points, 15, limit=50)
        self.assertEqual([p.source_frame for p in window], [12, 13, 14, 15])
        self.assertEqual(track.points[12].status, STATUS_MANUAL)
        self.assertEqual(track.points[12].segment, 1)

    def test_the_trail_never_includes_a_missing_or_lost_frame(self):
        raw = measurements(range(1, 20))
        for frame in (6, 7):
            raw[frame] = RawMeasurement(frame, False, method="template", reason="miss")
        track = resolve_track("ball", options=template_options(), interval_frames=FRAMES,
                              raw=raw, seed=Point(0, 100.0, 200.0),
                              interpolation_enabled=False, max_gap_frames=0)
        window = trail_window(track.points, 9, limit=50)
        self.assertNotIn(6, [p.source_frame for p in window])
        self.assertNotIn(7, [p.source_frame for p in window])

    def test_no_point_is_ever_drawn_ahead_of_the_current_frame(self):
        track = resolve_track("ball", options=template_options(),
                              interval_frames=FRAMES, raw=measurements(range(1, 20)),
                              seed=Point(0, 100.0, 200.0))
        window = trail_window(track.points, 3, limit=50)
        self.assertTrue(all(p.source_frame <= 3 for p in window))


class CorrectionTests(unittest.TestCase):
    def test_a_correction_starts_a_new_segment_and_is_an_operator_point(self):
        track = resolve_track(
            "ball", options=template_options(), interval_frames=FRAMES,
            raw=measurements(range(1, 20)), seed=Point(0, 100.0, 200.0),
            corrections=[Point(10, 160.0, 205.0)])
        self.assertEqual(track.points[10].status, STATUS_MANUAL)
        self.assertEqual(track.points[10].origin, ORIGIN_OPERATOR)
        self.assertEqual(track.accounting["segment_count"], 2)
        self.assertEqual(track.points[9].segment, 0)
        self.assertEqual(track.points[10].segment, 1)
        self.assertEqual(track.corrections[0]["frame"], 10)

    def test_the_jump_across_a_correction_is_not_counted_as_motion(self):
        straight = resolve_track("ball", options=template_options(),
                                 interval_frames=FRAMES, raw=measurements(range(1, 20)),
                                 seed=Point(0, 100.0, 200.0))
        near = resolve_track(
            "ball", options=template_options(), interval_frames=FRAMES,
            raw=measurements(range(1, 20)), seed=Point(0, 100.0, 200.0),
            corrections=[Point(10, 111.0, 200.0)])
        far = resolve_track(
            "ball", options=template_options(), interval_frames=FRAMES,
            raw=measurements(range(1, 20)), seed=Point(0, 100.0, 200.0),
            corrections=[Point(10, 900.0, 205.0)])
        self.assertAlmostEqual(straight.accounting["path_length_observed_px"], 19.0,
                               places=2)
        # The step across the correction is a discontinuity, so it is dropped: the
        # corrected tracks have one step fewer than the unbroken one.
        self.assertEqual(straight.accounting["observed_step_count"], 19)
        self.assertEqual(near.accounting["observed_step_count"], 18)
        self.assertEqual(far.accounting["observed_step_count"], 18)
        self.assertAlmostEqual(near.accounting["path_length_observed_px"], 17.0,
                               places=2)
        self.assertEqual(near.accounting["max_observed_step_px"], 1.0)
        # A correction placed far from the automatic chain shows up as a large step
        # from the CORRECTED point to its successor -- real motion relative to the
        # operator's point -- and never as the bridge from the old point to it.
        self.assertGreater(far.accounting["max_observed_step_px"], 700.0)

    def test_a_correction_restarts_a_lost_track(self):
        raw = measurements(range(1, 20))
        for frame in range(6, 12):
            raw[frame] = RawMeasurement(frame, False, method="template",
                                        reason="track_declared_lost")
        track = resolve_track("ball", options=template_options(loss_frames=3),
                              interval_frames=FRAMES, raw=raw,
                              seed=Point(0, 100.0, 200.0),
                              corrections=[Point(12, 500.0, 220.0, "operator")])
        self.assertEqual(track.points[5].status, STATUS_OBSERVED)
        self.assertEqual(track.points[6].status, STATUS_LOST)
        self.assertEqual(track.points[12].status, STATUS_MANUAL)
        self.assertEqual(track.points[13].status, STATUS_OBSERVED)
        self.assertEqual(track.coverage["automatic_observed_frames"], 12)


class JumpTests(unittest.TestCase):
    def test_a_rejected_step_is_reported_with_its_reason_in_the_diagnostics(self):
        raw = measurements(range(1, 20))
        raw[4] = RawMeasurement(4, False, method="template",
                                reason="step_exceeds_max_jump (120px)")
        track = resolve_track("ball", options=template_options(loss_frames=3),
                              interval_frames=FRAMES, raw=raw,
                              seed=Point(0, 100.0, 200.0),
                              interpolation_enabled=False, max_gap_frames=0)
        self.assertEqual(track.points[4].status, STATUS_MISSING)
        self.assertIn("step_exceeds_max_jump", track.points[4].note)
        self.assertTrue(any("step_exceeds_max_jump" in key for key in
                            track.accounting["empty_frame_reasons"]))

    def test_a_rejected_search_window_is_reported_too(self):
        raw = measurements(range(1, 20))
        raw[3] = RawMeasurement(3, False, method="template",
                                reason="search_window_clipped_by_frame_edge")
        track = resolve_track("ball", options=template_options(),
                              interval_frames=FRAMES, raw=raw,
                              seed=Point(0, 100.0, 200.0),
                              interpolation_enabled=False, max_gap_frames=0)
        self.assertEqual(track.points[3].status, STATUS_MISSING)
        self.assertIn("search_window_clipped_by_frame_edge", track.points[3].note)


class InterpolationTests(unittest.TestCase):
    def _gapped(self, gap_frames, *, loss_frames=3, max_gap=2):
        raw = measurements(range(1, 20))
        for frame in gap_frames:
            raw[frame] = RawMeasurement(frame, False, method="template",
                                        reason="score_below_min (0.30)")
        return resolve_track("ball", options=template_options(loss_frames=loss_frames),
                             interval_frames=FRAMES, raw=raw,
                             seed=Point(0, 100.0, 200.0),
                             interpolation_enabled=True, max_gap_frames=max_gap)

    def test_a_bounded_gap_is_filled_and_labelled_as_derived(self):
        track = self._gapped([8, 9])
        for frame in (8, 9):
            point = track.points[frame]
            self.assertEqual(point.status, STATUS_INTERPOLATED)
            self.assertEqual(point.origin, ORIGIN_DERIVED)
            self.assertEqual(point.method, "linear_interpolation")
            self.assertFalse(point.observed)
            self.assertFalse(point.automatic)
            self.assertIsNone(point.score)
            self.assertIn("derived by linear interpolation", point.note)
        self.assertEqual(track.coverage["interpolated_frames"], 2)
        self.assertEqual(track.coverage["observed_frames"], 18)
        self.assertIn("derived", track.limitations[1].lower())

    def test_an_interpolated_point_is_never_counted_in_the_observed_path(self):
        straight = self._gapped([])
        interpolated = self._gapped([8])
        self.assertAlmostEqual(straight.accounting["path_length_observed_px"], 19.0,
                               places=2)
        # one derived frame removes exactly two observed steps from the path
        self.assertAlmostEqual(interpolated.accounting["path_length_observed_px"], 17.0,
                               places=2)
        self.assertGreater(interpolated.accounting["path_length_including_interpolated_px"],
                           interpolated.accounting["path_length_observed_px"])
        self.assertEqual(interpolated.accounting["space"], "2d_image")
        self.assertEqual(interpolated.accounting["unit"], "px")

    def test_a_gap_longer_than_the_limit_stays_missing(self):
        track = self._gapped([7, 8, 9], loss_frames=6, max_gap=2)
        self.assertEqual({track.points[f].status for f in (7, 8, 9)},
                         {STATUS_MISSING})
        self.assertIn("exceeds interpolation.max_gap_frames", track.points[7].note)
        self.assertEqual(track.coverage["interpolated_frames"], 0)

    def test_a_loss_is_never_interpolated_even_when_the_gap_limit_allows_it(self):
        raw = measurements(range(1, 20))
        for frame in range(6, 20):
            raw[frame] = RawMeasurement(frame, False, method="template",
                                        reason="track_declared_lost")
        track = resolve_track("ball", options=template_options(loss_frames=2),
                              interval_frames=FRAMES, raw=raw,
                              seed=Point(0, 100.0, 200.0),
                              interpolation_enabled=True, max_gap_frames=20)
        self.assertEqual({track.points[f].status for f in range(6, 20)}, {STATUS_LOST})
        self.assertEqual(track.coverage["interpolated_frames"], 0)

    def test_a_gap_before_the_first_observation_is_not_filled(self):
        raw = measurements(range(10, 20))
        track = resolve_track("ball",
                              options=template_options(loss_frames=12),
                              interval_frames=FRAMES, raw=raw,
                              seed=Point(10, 110.0, 200.0),
                              interpolation_enabled=True, max_gap_frames=3)
        for frame in range(0, 10):
            self.assertEqual(track.points[frame].status, STATUS_MISSING)
            self.assertIn("not bounded", track.points[frame].note)
        self.assertEqual(track.coverage["interpolated_frames"], 0)

    def test_interpolation_cannot_cross_a_correction(self):
        raw = measurements(range(1, 20))
        for frame in (10, 11):
            raw[frame] = RawMeasurement(frame, False, method="template", reason="miss")
        track = resolve_track("ball", options=template_options(loss_frames=3),
                              interval_frames=FRAMES, raw=raw,
                              seed=Point(0, 100.0, 200.0),
                              corrections=[Point(12, 300.0, 220.0)],
                              interpolation_enabled=True, max_gap_frames=5)
        # A correction is a restart boundary, not an observation to interpolate
        # toward: the gap before it stays missing, so no approach points are drawn
        # to a position the target was never seen at.
        for frame in (10, 11):
            self.assertEqual(track.points[frame].status, STATUS_MISSING)
            self.assertIn("ends at an operator correction",
                          track.points[frame].note)
        self.assertEqual(track.coverage["interpolated_frames"], 0)
        self.assertEqual(trail_window(track.points, 11), [])
        window = trail_window(track.points, 13, limit=50)
        self.assertEqual([p.source_frame for p in window], [12, 13])

    def test_a_gap_ending_at_a_correction_is_still_filled_in_manual_mode(self):
        # In manual mode the anchors ARE the operator's own annotation, so filling a
        # bounded gap between two of them is a documented operator continuity claim.
        raw = {}
        track = resolve_track("clubhead", options=template_options(mode="manual"),
                              interval_frames=FRAMES, raw=raw,
                              seed=Point(1, 100.0, 200.0),
                              corrections=[Point(4, 160.0, 200.0)],
                              interpolation_enabled=True, max_gap_frames=3)
        self.assertEqual({track.points[f].status for f in (2, 3)},
                         {STATUS_INTERPOLATED})


class ManualModeTests(unittest.TestCase):
    def test_manual_mode_draws_only_the_operator_points(self):
        track = resolve_track(
            "clubhead", options=template_options(mode="manual"),
            interval_frames=FRAMES, raw=measurements(range(1, 20)),
            seed=Point(2, 100.0, 200.0),
            corrections=[Point(6, 130.0, 205.0), Point(10, 160.0, 210.0)],
            interpolation_enabled=True, max_gap_frames=4)
        manual = [p for p in track.points if p.status == STATUS_MANUAL]
        self.assertEqual([p.source_frame for p in manual], [2, 6, 10])
        self.assertEqual(track.coverage["automatic_observed_frames"], 0)
        self.assertEqual(track.accounting["path_length_observed_px"], 0)
        self.assertIn("manual anchor mode", track.limitations[0])
        # the operator's own points are still a path, and it is labelled as theirs
        self.assertEqual(track.accounting["operator_step_count"], 0)

    def test_manual_mode_anchors_only_connect_when_they_are_adjacent(self):
        track = resolve_track(
            "clubhead", options=template_options(mode="manual"),
            interval_frames=FRAMES, raw={}, seed=Point(2, 100.0, 200.0),
            corrections=[Point(3, 105.0, 200.0), Point(9, 130.0, 200.0)],
            interpolation_enabled=False, max_gap_frames=0)
        self.assertEqual(track.accounting["operator_step_count"], 1)
        self.assertAlmostEqual(track.accounting["path_length_operator_anchors_px"], 5.0,
                               places=2)


class AutomaticStopTests(unittest.TestCase):
    """automatic_until_frame: the operator stops the MATCHER, not the anchors.

    This exists because a template match on the trousers was published as an
    automatic observation of a ball flying at 100-220 px per frame. The honest answer
    for that stretch is the operator's own flight anchors, and nothing else.
    """

    def test_no_automatic_observation_is_used_past_the_stop_frame(self):
        track = resolve_track(
            "ball", options=template_options(automatic_until_frame=8),
            interval_frames=FRAMES, raw=measurements(range(1, 20)),
            seed=Point(0, 100.0, 200.0))
        by_frame = {p.source_frame: p for p in track.points}
        for frame in range(1, 9):
            self.assertTrue(by_frame[frame].automatic, frame)
        for frame in range(9, 20):
            self.assertEqual(by_frame[frame].status, STATUS_MISSING, frame)
            self.assertEqual(by_frame[frame].method, "operator_automatic_stop")
        self.assertEqual(track.coverage["automatic_observed_frames"], 8)

    def test_operator_anchors_after_the_stop_frame_are_kept(self):
        """track_until_frame would delete the whole flight; this must not."""
        track = resolve_track(
            "ball", options=template_options(automatic_until_frame=8),
            interval_frames=FRAMES, raw=measurements(range(1, 20)),
            seed=Point(0, 100.0, 200.0),
            corrections=[Point(10, 300.0, 150.0), Point(14, 420.0, 120.0)])
        by_frame = {p.source_frame: p for p in track.points}
        self.assertEqual(by_frame[10].status, STATUS_MANUAL)
        self.assertEqual(by_frame[14].status, STATUS_MANUAL)
        self.assertEqual(by_frame[10].x, 300.0)
        self.assertEqual(track.coverage["operator_anchor_frames"], 3)
        # frames 9..19 are past the stop; 10 and 14 hold anchors, so 9 remain empty
        self.assertEqual(track.coverage["automatic_stop_frames"], 9)

    def test_frames_past_the_stop_are_not_declared_lost(self):
        """They are out of scope by the operator's decision, not lost by failure."""
        track = resolve_track(
            "ball", options=template_options(automatic_until_frame=5),
            interval_frames=FRAMES, raw=measurements(range(1, 20)),
            seed=Point(0, 100.0, 200.0))
        self.assertEqual(track.coverage["lost_frames"], 0)
        self.assertEqual(track.coverage["missing_frames"], 14)

    def test_a_stopped_target_is_not_interpolated_across(self):
        track = resolve_track(
            "ball", options=template_options(automatic_until_frame=5),
            interval_frames=FRAMES, raw=measurements(range(1, 20)),
            seed=Point(0, 100.0, 200.0),
            corrections=[Point(9, 300.0, 150.0)])
        interpolated = [p.source_frame for p in track.points
                        if p.status == STATUS_INTERPOLATED]
        self.assertEqual(interpolated, [])

    def test_the_limit_is_recorded_as_a_limitation(self):
        track = resolve_track(
            "ball", options=template_options(automatic_until_frame=5),
            interval_frames=FRAMES, raw=measurements(range(1, 20)),
            seed=Point(0, 100.0, 200.0))
        self.assertTrue(any("automatic_until_frame" in limit
                            for limit in track.limitations), track.limitations)


class RoundTripTests(unittest.TestCase):
    def test_a_serialised_track_rebuilds_with_the_same_facts(self):
        track = resolve_track("ball", options=template_options(),
                              interval_frames=FRAMES, raw=measurements(range(1, 20)),
                              seed=Point(0, 100.0, 200.0),
                              corrections=[Point(12, 300.0, 210.0)])
        payload = track.to_dict()
        rebuilt = track_from_dict("ball", payload)
        self.assertEqual(rebuilt.available, track.available)
        self.assertEqual(rebuilt.coverage, track.coverage)
        self.assertEqual(rebuilt.accounting, track.accounting)
        self.assertEqual([(p.source_frame, p.status, p.segment)
                          for p in rebuilt.points],
                         [(p.source_frame, p.status, p.segment) for p in track.points])
        self.assertEqual(len(trail_window(rebuilt.points, 15)),
                         len(trail_window(track.points, 15)))

    def test_a_rebuilt_track_recomputes_its_derived_flags(self):
        payload = {"available": True, "points": [
            {"source_frame": 0, "status": STATUS_OBSERVED, "origin": ORIGIN_AUTOMATIC,
             "x": 1, "y": 2, "segment": 0},
            {"source_frame": 1, "status": STATUS_LOST, "origin": ORIGIN_DERIVED,
             "x": None, "y": None, "segment": None},
        ]}
        track = track_from_dict("ball", payload)
        self.assertTrue(track.points[0].automatic)
        self.assertTrue(track.points[0].drawn)
        self.assertFalse(track.points[1].drawn)
        self.assertFalse(track.points[1].observed)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
