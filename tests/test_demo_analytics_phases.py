"""Geometry, missingness, phase and tempo tests for demo mode.

Two families of claim are pinned here: the 2D geometry (so an angle is computed in
source pixels with the sample aspect ratio applied, never by mixing normalised and
pixel coordinates) and the honesty of phases and tempo (manual or explicitly
approximate labels, and a stated refusal instead of an invented phase).
"""
from __future__ import annotations

import math
import unittest

from ghostcaddie.demo.analytics import (MIN_METRIC_FRACTION, MIN_METRIC_FRAMES,
                                        body_angles_for_record, body_metrics,
                                        missingness_report, targets_summary)
from ghostcaddie.demo.config import Point, TargetOptions
from ghostcaddie.demo.geometry import (angle_degrees, distance, flexion_degrees,
                                       linear_interpolate, median, path_length,
                                       percentile, tilt_degrees)
from ghostcaddie.demo.phases import (LABEL_APPROX, LABEL_MANUAL, LABEL_UNAVAILABLE,
                                     compute_tempo, phase_cards, resolve_phases)
from ghostcaddie.demo.tracking import RawMeasurement, resolve_track
from tests.demo_fixtures import UPRIGHT_POINTS, body_record, shift_points


class GeometryTests(unittest.TestCase):
    def test_tilt_is_signed_and_measured_from_the_image_horizontal(self):
        self.assertAlmostEqual(tilt_degrees((10, 10), (20, 10)), 0.0, places=6)
        self.assertAlmostEqual(tilt_degrees((10, 10), (20, 20)), 45.0, places=6)
        self.assertAlmostEqual(tilt_degrees((10, 20), (20, 10)), -45.0, places=6)

    def test_a_zero_length_segment_has_no_tilt_rather_than_zero_degrees(self):
        self.assertIsNone(tilt_degrees((5, 5), (5, 5)))

    def test_the_two_landmarks_in_either_order_give_the_same_tilt(self):
        """A shoulder line is a LINE, not an arrow. Before the fold, giving the same
        pair the other way round read as +166 or -170 degrees."""
        pairs = (((10, 10), (20, 20)), ((10, 20), (20, 10)), ((100, 300), (400, 260)),
                 ((400, 260), (100, 300)), ((50, 90), (40, 91)))
        for left, right in pairs:
            forward = tilt_degrees(left, right)
            backward = tilt_degrees(right, left)
            self.assertAlmostEqual(forward, backward, places=6, msg=(left, right))
            self.assertGreater(forward, -90.0, (left, right))
            self.assertLessEqual(forward, 90.0, (left, right))

    def test_a_near_horizontal_left_facing_line_reads_small_not_near_180(self):
        # the golfer facing the other way: 2 px of drop over 50 px of run
        self.assertAlmostEqual(tilt_degrees((100, 100), (50, 98)), 2.291, places=2)
        self.assertAlmostEqual(tilt_degrees((50, 98), (100, 100)), 2.291, places=2)

    def test_the_sample_aspect_ratio_is_applied_before_the_angle_is_taken(self):
        square = tilt_degrees((0, 0), (10, 10), pixel_aspect=1.0)
        anamorphic = tilt_degrees((0, 0), (10, 10), pixel_aspect=2.0)
        self.assertAlmostEqual(square, 45.0, places=6)
        self.assertAlmostEqual(anamorphic, math.degrees(math.atan2(10, 20)), places=6)
        self.assertNotAlmostEqual(square, anamorphic, places=3)

    def test_normalised_coordinates_would_give_a_different_angle(self):
        # The same landmark pair expressed in normalised x/y (each axis divided by
        # its own frame extent) would read differently; the demo computes in source
        # pixels only, with the sample aspect ratio applied.
        pixel_angle = tilt_degrees((0, 0), (640, 480), pixel_aspect=1.0)
        normalised_angle = tilt_degrees((0.0, 0.0), (640 / 1920.0, 480 / 1080.0),
                                        pixel_aspect=1.0)
        self.assertAlmostEqual(pixel_angle, math.degrees(math.atan2(480, 640)),
                               places=6)
        self.assertNotAlmostEqual(pixel_angle, normalised_angle, places=3)

    def test_a_knee_interior_angle_is_180_for_a_straight_leg(self):
        interior = angle_degrees((0, 0), (0, 50), (0, 100))
        self.assertAlmostEqual(interior, 180.0, places=6)
        self.assertAlmostEqual(flexion_degrees(interior), 0.0, places=6)

    def test_a_right_angle_knee_reads_ninety_degrees(self):
        interior = angle_degrees((0, 0), (0, 50), (50, 50))
        self.assertAlmostEqual(interior, 90.0, places=6)
        self.assertAlmostEqual(flexion_degrees(interior), 90.0, places=6)

    def test_a_degenerate_limb_produces_no_angle(self):
        self.assertIsNone(angle_degrees((0, 0), (0, 0), (10, 10)))
        self.assertIsNone(flexion_degrees(None))

    def test_path_length_and_interpolation_are_plain_pixel_maths(self):
        self.assertAlmostEqual(path_length([(0, 0), (3, 4), (6, 8)]), 10.0, places=6)
        self.assertAlmostEqual(distance((0, 0), (3, 4)), 5.0, places=6)
        self.assertEqual(linear_interpolate((0, 0), (10, 20), 0.5), (5.0, 10.0))

    def test_median_and_percentile_ignore_nothing_silently(self):
        self.assertEqual(median([1, 2, 3]), 2)
        self.assertEqual(median([1, 2, 3, 4]), 2.5)
        self.assertIsNone(median([]))
        self.assertAlmostEqual(percentile([0, 10], 0.5), 5.0)


class BodyAngleTests(unittest.TestCase):
    def test_angles_come_from_the_named_landmarks(self):
        record = body_record(0, points=UPRIGHT_POINTS)
        angles = body_angles_for_record(record)
        self.assertAlmostEqual(angles["shoulder_tilt_deg"], 0.0, places=6)
        self.assertAlmostEqual(angles["hip_tilt_deg"], 0.0, places=6)
        self.assertAlmostEqual(angles["knee_interior_deg"]["left"], 180.0, places=6)
        self.assertEqual(angles["lead_side"], "left")
        self.assertEqual(angles["trail_side"], "right")

    def test_handedness_only_relabels_the_sides(self):
        record = body_record(0, points=UPRIGHT_POINTS)
        left = body_angles_for_record(record, handedness="left")
        self.assertEqual(left["lead_side"], "right")
        self.assertEqual(left["knee_interior_deg"]["right"],
                         body_angles_for_record(record)["knee_interior_deg"]["right"])

    def test_an_unavailable_record_yields_no_angles_at_all(self):
        angles = body_angles_for_record(body_record(0, points={}, available=False,
                                                    reason="no_operator_roi"))
        self.assertIsNone(angles["shoulder_tilt_deg"])
        self.assertIsNone(angles["knee_interior_deg"]["left"])

    def test_a_missing_landmark_removes_only_its_own_quantity(self):
        points = dict(UPRIGHT_POINTS)
        points.pop("hip_r")
        angles = body_angles_for_record(body_record(0, points=points))
        self.assertIsNotNone(angles["shoulder_tilt_deg"])
        self.assertIsNone(angles["hip_tilt_deg"])


class BodyMetricTests(unittest.TestCase):
    def _records(self, count, *, skip=(), dx=0.0):
        out = []
        for index in range(count):
            if index in skip:
                out.append(body_record(index, points={}, available=False,
                                       reason="only 4 of 17 keypoints above 0.3"))
            else:
                out.append(body_record(index, points=shift_points(UPRIGHT_POINTS, dx)))
        return out

    def test_a_metric_is_reported_when_enough_frames_carry_it(self):
        metrics = body_metrics(self._records(10), frames_possible=10,
                               handedness="right")
        supported = {m["name"] for m in metrics["supported"]}
        self.assertIn("shoulder_tilt", supported)
        self.assertIn("knee_interior_left", supported)
        self.assertEqual(metrics["frames_with_body_pose"], 10)

    def test_too_few_frames_gives_an_unavailable_metric_with_a_counted_reason(self):
        metrics = body_metrics(self._records(3), frames_possible=10)
        unavailable = {m["name"]: m for m in metrics["unavailable"]}
        self.assertIn("shoulder_tilt", unavailable)
        self.assertIn(f"below the {MIN_METRIC_FRAMES} needed",
                      unavailable["shoulder_tilt"]["reason"])
        self.assertEqual(unavailable["shoulder_tilt"]["frames_used"], 3)

    def test_below_the_coverage_fraction_a_metric_is_refused_not_guessed(self):
        records = self._records(40, skip=tuple(range(8, 40)))
        metrics = body_metrics(records, frames_possible=40)
        unavailable = {m["name"]: m for m in metrics["unavailable"]}
        self.assertIn("shoulder_tilt", unavailable)
        self.assertIn(f"{MIN_METRIC_FRACTION:.0%}", unavailable["shoulder_tilt"]["reason"])
        self.assertEqual(unavailable["shoulder_tilt"]["frames_used"], 8)

    def test_missing_frames_do_not_break_the_knee_chain_on_other_frames(self):
        records = self._records(20, skip=(5, 6))
        metrics = body_metrics(records, frames_possible=20)
        supported = {m["name"]: m for m in metrics["supported"]}
        self.assertIn("knee_interior_left", supported)
        self.assertEqual(supported["knee_interior_left"]["frames_used"], 18)

    def test_head_displacement_is_measured_from_the_first_resolvable_frame(self):
        records = [body_record(i, points=shift_points(UPRIGHT_POINTS, i * 2.0))
                   for i in range(10)]
        metrics = body_metrics(records, frames_possible=10)
        head = {m["name"]: m for m in metrics["supported"]}["head_displacement"]
        self.assertAlmostEqual(head["value"], 9 * 2.0, places=2)

    def test_the_wrist_path_breaks_at_a_frame_without_a_pose(self):
        records = [body_record(i, points=shift_points(UPRIGHT_POINTS, i * 1.0))
                   for i in range(10)]
        records[5] = body_record(5, points={}, available=False, reason="no pose")
        metrics = body_metrics(records, frames_possible=10)
        wrist = {m["name"]: m for m in metrics["supported"]}["wrist_path_length"]
        # runs 0-4 (4 steps) and 6-9 (3 steps), one pixel long each
        self.assertAlmostEqual(wrist["value"], 7.0, places=2)
        self.assertEqual(wrist["detail"]["contiguous_runs"], 2)

    def test_a_body_metric_is_labelled_as_image_space_degrees(self):
        metrics = body_metrics(self._records(10), frames_possible=10)
        shoulder = {m["name"]: m for m in metrics["supported"]}["shoulder_tilt"]
        self.assertEqual(shoulder["unit"], "degrees")
        self.assertEqual(shoulder["space"], "2d_image")
        self.assertIn("pixel_aspect_ratio_applied", metrics["conventions"])

    def test_no_physical_quantity_is_produced_anywhere(self):
        metrics = body_metrics(self._records(10), frames_possible=10)
        text = repr(metrics).lower()
        for forbidden in ("mph", "m/s", "metre", "meter", "rpm", "carry", "spin",
                          "launch_angle"):
            self.assertNotIn(forbidden, text)


class TargetMetricTests(unittest.TestCase):
    def _track(self, *, seed=True, corrections=()):
        frames = list(range(0, 30))
        raw = {f: RawMeasurement(f, True, x=100.0 + f, y=200.0, score=0.9,
                                 method="template") for f in range(1, 30)}
        return resolve_track("ball",
                             options=TargetOptions(mode="template", loss_frames=3),
                             interval_frames=frames, raw=raw,
                             seed=Point(0, 100.0, 200.0) if seed else None,
                             corrections=list(corrections))

    def test_an_unseeded_target_reports_unavailable_with_its_reason(self):
        summary = targets_summary({"ball": self._track(seed=False)},
                                  frames_possible=30)["ball"]
        self.assertFalse(summary["available"])
        self.assertIn("no operator seed", summary["reason"])
        self.assertEqual(summary["observed_frames"], 0)

    def test_a_tracked_target_reports_pixels_and_a_cadence_caveat(self):
        summary = targets_summary({"ball": self._track()}, frames_possible=30)["ball"]
        self.assertTrue(summary["available"])
        self.assertEqual(summary["unit"], "px")
        self.assertEqual(summary["space"], "2d_image")
        self.assertAlmostEqual(summary["path_length_observed_px"], 29.0, places=2)
        self.assertIn("NOT", summary["cadence_note"])
        self.assertIn("speed", summary["cadence_note"])

    def test_missingness_is_reported_in_counts(self):
        report = missingness_report(
            [body_record(0, points=UPRIGHT_POINTS),
             body_record(1, points={}, available=False, reason="no pose")],
            {"ball": self._track()})
        self.assertEqual(report["body_frames_with_pose"], 1)
        self.assertEqual(report["body_frames_without_pose"], 1)
        self.assertIn("no pose", report["body_missing_reasons"])
        self.assertEqual(report["targets"]["ball"]["missing_frames"], 0)
        self.assertIn("never filled in", report["note"])


class PhaseTests(unittest.TestCase):
    def _records(self, count=40):
        return [body_record(i, points=shift_points(UPRIGHT_POINTS,
                                                   dx=0 if i < 20 else (i - 20) * 2.0))
                for i in range(count)]

    def test_operator_keyframes_are_labelled_manual(self):
        phases = resolve_phases(config_keyframes={"address": 0, "top": 20,
                                                  "impact": 30, "finish": 39},
                                interval_frames=list(range(40)),
                                body_records=self._records(), tracks={}, fps=25.0,
                                allow_approx=False,
                                body_angles_fn=lambda record:
                                    body_angles_for_record(record))
        self.assertEqual(phases["phases"]["top"]["label"], LABEL_MANUAL)
        self.assertEqual(phases["phases"]["top"]["method"], "operator_keyframe")
        self.assertEqual(phases["phases"]["top"]["frame"], 20)
        self.assertEqual(phases["phases"]["top"]["seconds_source_timeline"], 0.8)
        self.assertEqual(phases["phases"]["top"]["seconds_basis"],
                         "frames / container cadence")
        self.assertEqual(phases["seconds_basis"], "frames / container cadence")
        self.assertIn("shoulder_tilt_deg",
                      phases["phases"]["top"]["angles_2d"])

    def test_an_unmarked_phase_is_unavailable_with_a_stated_reason(self):
        phases = resolve_phases(config_keyframes={"address": 0, "top": 20},
                                interval_frames=list(range(40)),
                                body_records=self._records(), tracks={}, fps=25.0,
                                allow_approx=False)
        impact = phases["phases"]["impact"]
        self.assertEqual(impact["label"], LABEL_UNAVAILABLE)
        self.assertIsNone(impact["frame"])
        self.assertIn("allow_approx_phases is false", impact["note"])

    def test_evenly_spaced_frames_are_never_used_as_phases(self):
        phases = resolve_phases(config_keyframes={}, interval_frames=list(range(70)),
                                body_records=[], tracks={}, fps=25.0,
                                allow_approx=True)
        for name, info in phases["phases"].items():
            self.assertIsNone(info["frame"], name)
            self.assertEqual(info["label"], LABEL_UNAVAILABLE, name)
        tempo = phases["tempo"]
        self.assertEqual(tempo["state"], "unavailable")
        self.assertIn("not estimated from evenly spaced frames", tempo["note"])

    def test_approximate_phases_are_derived_only_from_named_measurements(self):
        records = self._records()
        raw = {f: RawMeasurement(f, True, x=100.0, y=200.0, score=0.9,
                                 method="template") for f in range(1, 40)}
        for frame in range(30, 40):
            raw[frame] = RawMeasurement(frame, True, x=100.0 + (frame - 30) * 40.0,
                                        y=200.0, score=0.9, method="template")
        ball = resolve_track("ball", options=TargetOptions(loss_frames=3),
                             interval_frames=list(range(40)), raw=raw,
                             seed=Point(0, 100.0, 200.0))
        phases = resolve_phases(config_keyframes={}, interval_frames=list(range(40)),
                                body_records=records, tracks={"ball": ball}, fps=25.0,
                                allow_approx=True)
        self.assertEqual(phases["phases"]["top"]["label"], LABEL_APPROX)
        self.assertEqual(phases["phases"]["top"]["method"], "min_wrist_height")
        self.assertEqual(phases["phases"]["impact"]["label"], LABEL_APPROX)
        self.assertEqual(phases["phases"]["impact"]["method"],
                         "ball_first_large_displacement")
        self.assertEqual(phases["phases"]["finish"]["method"], "last_pose_frame")
        # Phase kinds with no implemented approximation stay unavailable.
        for name in ("takeaway", "downswing", "follow_through"):
            self.assertEqual(phases["phases"][name]["label"], LABEL_UNAVAILABLE, name)
            self.assertIn("not implemented", phases["phases"][name]["note"])
        self.assertTrue(any("approx" in note for note in phases["notes"]))

    def test_cards_cover_the_four_primary_phases_first(self):
        phases = resolve_phases(config_keyframes={"address": 0, "top": 20,
                                                  "finish": 39, "takeaway": 5},
                                interval_frames=list(range(40)),
                                body_records=self._records(), tracks={}, fps=25.0,
                                allow_approx=False)
        cards = phase_cards({name: _record_from(phases, name)
                             for name in ("address", "takeaway", "top", "downswing",
                                          "impact", "follow_through", "finish")})
        primary = [c["phase"] for c in cards if c["primary"]]
        self.assertEqual(primary, ["address", "top", "impact", "finish"])
        extras = [c["phase"] for c in cards if not c["primary"]]
        self.assertEqual(extras, ["takeaway"])
        # the missing impact marker still gets a card, and it says UNAVAILABLE
        missing = [c for c in cards if not c["renderable"]]
        self.assertEqual([c["phase"] for c in missing], ["impact"])
        self.assertEqual(missing[0]["label"], LABEL_UNAVAILABLE)
        self.assertTrue(missing[0]["note"])


class TempoTests(unittest.TestCase):
    def _records(self, keyframes):
        return resolve_phases(config_keyframes=keyframes,
                              interval_frames=list(range(200)),
                              body_records=[], tracks={}, fps=25.0,
                              allow_approx=False)["phases"]

    def test_tempo_prefers_takeaway_and_reports_frame_counts(self):
        tempo = compute_tempo({name: _phase(name, frame, LABEL_MANUAL)
                               for name, frame in (("takeaway", 10), ("top", 40),
                                                   ("impact", 60))}, fps=25.0)
        self.assertEqual(tempo["state"], "supported")
        self.assertEqual(tempo["backswing_frames"], 30)
        self.assertEqual(tempo["downswing_frames"], 20)
        self.assertAlmostEqual(tempo["backswing_downswing_ratio"], 1.5, places=3)
        self.assertEqual(tempo["backswing_start_marker"], "takeaway")
        self.assertEqual(tempo["unit"], "source frames")
        self.assertIn("not measured action time", tempo["caveat"])

    def test_address_is_used_only_as_a_disclosed_fallback(self):
        notes = []
        tempo = compute_tempo({name: _phase(name, frame, LABEL_MANUAL)
                               for name, frame in (("address", 5), ("top", 40),
                                                   ("impact", 60))}, fps=25.0,
                              notes=notes)
        self.assertEqual(tempo["backswing_start_marker"], "address")
        self.assertTrue(any("takeaway is the preferred definition" in n
                            for n in notes))

    def test_tempo_is_unavailable_without_an_impact_marker(self):
        tempo = compute_tempo({name: _phase(name, frame, LABEL_MANUAL)
                               for name, frame in (("takeaway", 10), ("top", 40))},
                              fps=25.0)
        self.assertEqual(tempo["state"], "unavailable")
        self.assertIn("impact", tempo["reason"])

    def test_tempo_is_unavailable_without_a_backswing_start(self):
        tempo = compute_tempo({name: _phase(name, frame, LABEL_MANUAL)
                               for name, frame in (("top", 40), ("impact", 60))},
                              fps=25.0)
        self.assertEqual(tempo["state"], "unavailable")
        self.assertIn("backswing start", tempo["reason"])

    def test_markers_out_of_order_are_refused_not_absolutised(self):
        tempo = compute_tempo({name: _phase(name, frame, LABEL_MANUAL)
                               for name, frame in (("takeaway", 50), ("top", 40),
                                                   ("impact", 60))}, fps=25.0)
        self.assertEqual(tempo["state"], "unavailable")
        self.assertIn("not in chronological order", tempo["reason"])

    def test_an_approximate_marker_makes_the_tempo_approximate_and_says_so(self):
        notes = []
        tempo = compute_tempo({name: _phase(name, frame, label)
                               for name, frame, label in
                               (("takeaway", 10, LABEL_MANUAL), ("top", 40, LABEL_APPROX),
                                ("impact", 60, LABEL_APPROX))}, fps=25.0, notes=notes)
        self.assertEqual(tempo["label"], LABEL_APPROX)
        self.assertTrue(any("approximations of the markers" in n for n in notes))

    def test_manual_keyframes_through_resolve_phases_give_a_manual_tempo(self):
        phases = resolve_phases(config_keyframes={"takeaway": 10, "top": 40,
                                                  "impact": 60},
                                interval_frames=list(range(100)), body_records=[],
                                tracks={}, fps=25.0, allow_approx=False)
        self.assertEqual(phases["tempo"]["label"], LABEL_MANUAL)
        self.assertEqual(phases["tempo"]["backswing_frames"], 30)


def _phase(name, frame, label):
    from ghostcaddie.demo.phases import PhaseRecord
    return PhaseRecord(name=name, label=label, frame=frame, method="test")


def _record_from(phases, name):
    from ghostcaddie.demo.phases import PhaseRecord
    info = phases["phases"][name]
    return PhaseRecord(name=name, label=info["label"], frame=info["frame"],
                       method=info["method"], note=info["note"],
                       angles=info["angles_2d"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
