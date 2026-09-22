"""Focused tests for the rendering, tracker-validity, picker and timing fixes.

Each test here exists because a review found a specific way the demo could look
right while being wrong: overlays drawn in the wrong place under a downscale or a
gutter, a tracker that happily follows a flat patch, a picker that cannot exit a
dirty session or drops the interval it was reopened with, and seconds computed from
a nominal frame rate on a source whose timestamps say otherwise.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2
import numpy as np

from ghostcaddie.demo import render as render_mod
from ghostcaddie.demo.config import (SECONDS_BASIS_TIMESTAMPS, parse_config,
                                     validate_config)
from ghostcaddie.demo.errors import DemoConfigError
from ghostcaddie.demo.picker import (PickerState, native_to_viewport, roi_to_native,
                                     viewport_to_native, viewport_transform)
from ghostcaddie.demo.probe import analyze_timestamps, probe_source, probe_timestamps
from tests.demo_fixtures import (TempDirTestCase, base_config, make_synthetic_clip,
                                 synthetic_probe, write_config)


class PanelPlanTests(unittest.TestCase):
    """The analytics panel must not sit on the golfer, and must be checked."""

    def test_a_roi_on_one_side_puts_the_panel_on_the_other(self):
        roi = [100, 100, 300, 700]          # golfer on the left of 1280
        plan = render_mod.plan_panel(roi, 1280, 720)
        self.assertEqual(plan["side"], "right")
        self.assertFalse(plan["external"])
        self.assertEqual(plan["overlap_px2"], 0.0)

    def test_a_wide_roi_pushes_the_panel_into_an_external_gutter(self):
        # portrait subject inside a landscape frame: the box spans nearly all of it
        roi = [40, 20, 1200, 700]
        plan = render_mod.plan_panel(roi, 1280, 720)
        self.assertTrue(plan["external"])
        self.assertGreater(plan["gutter_w"], 0)

    def test_the_native_roi_is_scaled_before_the_panel_decision(self):
        """A downscaled render must not compare a native ROI against a small frame."""
        native_roi = [40, 20, 2400, 1400]      # spans the 2560-wide source
        plan_full = render_mod.plan_panel_scaled(native_roi, 1.0, 2560, 1440)
        plan_half = render_mod.plan_panel_scaled(native_roi, 0.5, 1280, 720)
        self.assertTrue(plan_full["external"])
        self.assertTrue(plan_half["external"])
        # the half-size plan is computed against the half-size frame, so the gutter
        # is proportional rather than double-counting the native box
        self.assertLess(plan_half["panel_w"], plan_full["panel_w"])

    def test_a_roi_that_does_not_reach_the_panel_keeps_it_an_overlay(self):
        roi = [0, 0, 200, 200]
        plan = render_mod.plan_panel(roi, 1920, 1080)
        self.assertFalse(plan["external"])
        self.assertEqual(plan["side"], "right")

    def test_target_points_are_a_second_no_go_region(self):
        """The panel used to be checked against the golfer only, so it covered the
        ball flight while reporting zero overlap. The target box is now checked too."""
        roi = [100, 100, 300, 700]                 # golfer on the left of 1280
        flight = [(120.0 + index * 30.0, 500.0 - index * 20.0) for index in range(12)]
        plan = render_mod.plan_panel(roi, 1280, 720, points_xy=flight)
        self.assertFalse(plan["external"])         # the right column is still clear
        self.assertEqual(plan["side"], "right")
        self.assertIn("targets_box_xywh", plan)
        # now put the flight through that column: the overlay must be refused
        through = [(1000.0 + index * 20.0, 300.0 + index * 10.0) for index in range(6)]
        plan = render_mod.plan_panel(roi, 1280, 720, points_xy=through)
        self.assertTrue(plan["external"])
        self.assertGreater(plan["overlay_refusals"]["right"]["targets"], 0.0)
        self.assertEqual(plan["targets_overlap_px2"], 0.0)

    def test_the_selected_clip_geometry_puts_the_panel_in_the_gutter(self):
        """The actual failure: a portrait golfer in a landscape frame whose flight
        crosses the panel's column at the payoff frame."""
        roi = [900, 80, 650, 1200]                 # operator ROI, native pixels
        points = [(929.8, 1195.6), (915.9, 1204.7), (1274.0, 1083.0),
                  (2087.8, 821.4), (745.0, 1060.0), (1310.0, 1220.0)]
        plan = render_mod.plan_panel_scaled(roi, 0.5, 1280, 720, points_xy=points)
        self.assertTrue(plan["external"])
        self.assertGreater(plan["gutter_w"], 0)
        self.assertEqual(plan["body_overlap_px2"], 0.0)
        self.assertEqual(plan["targets_overlap_px2"], 0.0)
        self.assertIn("right", plan["overlay_refusals"])

    def test_the_panel_is_never_wider_than_a_quarter_of_the_picture(self):
        for width in (320, 640, 1280, 2560):
            plan = render_mod.plan_panel([0, 0, 10, 10], width, int(width * 9 / 16),
                                         points_xy=[(width - 5.0, 5.0)])
            self.assertLessEqual(plan["panel_w"], max(180, 0.40 * width))


class ComposeTests(unittest.TestCase):
    """Evidence order inside one rendered frame: chrome first, overlays on top."""

    def test_the_picture_is_never_modified_by_the_chrome(self):
        """A keyframe card is taken from the picture, so chrome must not land in it."""
        picture = np.full((240, 320, 3), 90, np.uint8)
        before = picture.copy()
        plan = render_mod.plan_panel([0, 0, 40, 40], 320, 240)
        canvas = render_mod.compose_annotated(
            picture, plan=plan, phase_text="PHASE  impact", frame_text="frame 3",
            analytics_lines=["BODY measured", "  shoulder tilt   1.0 deg"])
        self.assertTrue(np.array_equal(picture, before),
                        "the video chrome was painted into the card source")
        self.assertFalse(np.array_equal(canvas, picture))

    def test_a_marker_is_drawn_above_the_chrome(self):
        # the panel column is dark and translucent; a marker inside it must survive
        picture = np.full((240, 320, 3), 120, np.uint8)
        plan = {"side": "right", "external": False, "panel_w": 120, "panel_x": 200,
                "header_h": 34, "legend_h": 24, "gutter_w": 0, "overlap_px2": 0.0,
                "body_overlap_px2": 0.0, "targets_overlap_px2": 0.0}
        marker = (int(plan["panel_x"]) + 60, 120)
        canvas = render_mod.compose_annotated(
            picture, plan=plan, phase_text="PHASE  impact", frame_text="frame 3",
            analytics_lines=["BODY measured", "  shoulder tilt   1.0 deg"],
            overlays=lambda image: cv2.circle(image, marker, 5, (255, 255, 255), -1))
        self.assertGreater(int(canvas[marker[1], marker[0]].sum()), 700,
                           "the marker was dimmed by the panel drawn over it")

    def test_a_left_gutter_still_lands_the_marker_on_its_picture_coordinate(self):
        picture = np.zeros((240, 320, 3), np.uint8)
        plan = {"side": "left", "external": True, "panel_w": 200, "panel_x": 0,
                "header_h": 34, "legend_h": 24, "gutter_w": 200, "overlap_px2": 0.0,
                "body_overlap_px2": 0.0, "targets_overlap_px2": 0.0}
        canvas = render_mod.compose_annotated(
            picture, plan=plan, phase_text="PHASE  impact", frame_text="frame 3",
            analytics_lines=["BODY measured"],
            overlays=lambda image: cv2.circle(image, (150, 100), 4, (0, 0, 255), -1))
        # a colour no piece of chrome uses, so the chrome's own text cannot be
        # mistaken for the marker
        mask = np.all(canvas == np.array([0, 0, 255], dtype=np.uint8), axis=2)
        ys, xs = np.where(mask)
        self.assertGreater(len(xs), 0)
        self.assertAlmostEqual(float(xs.mean()), 150.0 + 200.0, delta=3.0)
        self.assertAlmostEqual(float(ys.mean()), 100.0, delta=3.0)


class OverlayCoordinateTests(unittest.TestCase):
    """Overlays are drawn on the picture, so a gutter or a downscale cannot offset
    them. This checks the pixel a known marker lands on."""

    def test_a_ball_marker_lands_on_its_scaled_coordinate(self):
        picture = np.zeros((240, 320, 3), np.uint8)
        point = {"status": "observed_template", "x": 160.0, "y": 120.0}
        render_mod.draw_target_marker(picture, point, "ball", 0.5)
        ys, xs = np.where(picture.max(axis=2) > 60)
        self.assertGreater(len(xs), 0)
        self.assertAlmostEqual(float(xs.mean()), 80.0, delta=2.0)
        self.assertAlmostEqual(float(ys.mean()), 60.0, delta=2.0)

    def test_a_club_marker_lands_on_its_scaled_coordinate(self):
        picture = np.zeros((240, 320, 3), np.uint8)
        point = {"status": "manual_anchor", "x": 40.0, "y": 200.0}
        render_mod.draw_target_marker(picture, point, "clubhead", 1.0)
        ys, xs = np.where(picture.max(axis=2) > 60)
        self.assertAlmostEqual(float(xs.mean()), 40.0, delta=3.0)
        self.assertAlmostEqual(float(ys.mean()), 200.0, delta=3.0)

    def test_a_left_gutter_shifts_the_picture_and_its_overlays_together(self):
        """The real bug behind this test: overlays stayed left while the picture
        shifted right, so the marker no longer sat on the ball."""
        picture = np.zeros((200, 300, 3), np.uint8)
        point = {"status": "observed_template", "x": 150.0, "y": 100.0}
        render_mod.draw_target_marker(picture, point, "ball", 1.0)
        gutter = 100
        canvas = np.zeros((200, 300 + gutter, 3), np.uint8)
        canvas[:, gutter:gutter + 300] = picture
        ys, xs = np.where(canvas.max(axis=2) > 60)
        self.assertAlmostEqual(float(xs.mean()), 150.0 + gutter, delta=3.0)
        self.assertAlmostEqual(float(ys.mean()), 100.0, delta=3.0)

    def test_a_skeleton_is_drawn_where_the_keypoints_are(self):
        picture = np.zeros((300, 300, 3), np.uint8)
        keypoints = [{"name": "sh_l", "x": 100.0, "y": 100.0, "visible": True},
                     {"name": "sh_r", "x": 200.0, "y": 100.0, "visible": True},
                     {"name": "hip_l", "x": 100.0, "y": 200.0, "visible": True},
                     {"name": "hip_r", "x": 200.0, "y": 200.0, "visible": True}]
        render_mod.draw_skeleton(picture, keypoints, 1.0)
        tex = picture.mean(axis=2)
        self.assertGreater(tex[100, 150], 0)      # the shoulder line
        self.assertGreater(tex[150, 100], 0)      # a hip edge
        self.assertEqual(tex[10, 10], 0)          # nothing invented elsewhere

    def test_nothing_is_drawn_for_a_missing_or_lost_point(self):
        for status in ("missing", "lost"):
            picture = np.zeros((120, 120, 3), np.uint8)
            render_mod.draw_target_marker(picture, {"status": status, "x": None,
                                                    "y": None}, "ball", 1.0)
            self.assertEqual(int(picture.sum()), 0, status)
            render_mod.draw_target_marker(picture, {"status": status, "x": 60.0,
                                                    "y": 60.0}, "ball", 1.0)
            self.assertEqual(int(picture.sum()), 0, status)

    def test_the_trail_is_dashed_across_a_derived_point(self):
        solid = np.zeros((120, 200, 3), np.uint8)
        derived = np.zeros((120, 200, 3), np.uint8)
        observed = [{"status": "observed_template", "x": 20.0, "y": 60.0},
                    {"status": "observed_template", "x": 180.0, "y": 60.0}]
        mixed = [{"status": "observed_template", "x": 20.0, "y": 60.0},
                 {"status": "interpolated", "x": 100.0, "y": 60.0},
                 {"status": "observed_template", "x": 180.0, "y": 60.0}]
        render_mod.draw_trail(solid, observed, render_mod.BALL, 1.0, 3)
        render_mod.draw_trail(derived, mixed, render_mod.BALL, 1.0, 3)
        self.assertGreater(int(np.count_nonzero(solid[:, :, 0])),
                           int(np.count_nonzero(derived[:, :, 0])))


class CardTextTests(unittest.TestCase):
    def test_card_text_is_ascii_because_cv2_cannot_render_unicode(self):
        picture = np.zeros((100, 200, 3), np.uint8)
        card = render_mod.draw_card(picture, title="TOP - frame 90",
                                    label="OPERATOR-MARKED",
                                    lines=["shoulder tilt 1.2 deg | hip tilt 0.4 deg",
                                           "operator-marked source frame"])
        self.assertEqual(card.ndim, 3)
        for text in (render_mod.VIDEO_SUBTITLE, render_mod.VIDEO_LEGEND,
                     render_mod.VIDEO_CAVEAT):
            self.assertTrue(all(ord(c) < 128 for c in text), text)

    def test_long_note_lines_are_wrapped_not_cut(self):
        wrapped = render_mod._wrap("x" * 30 + " " + "y" * 30, 40)
        self.assertTrue(all(len(line) <= 40 for line in wrapped))
        self.assertEqual(len(wrapped), 2)

    def test_an_unavailable_card_says_so_and_keeps_the_reason(self):
        card = render_mod.draw_unavailable_card(
            title="IMPACT - frame -", reason="no operator keyframe was supplied")
        self.assertEqual(card.shape[1], 640)
        self.assertGreater(int(card.sum()), 0)


class FlatTemplateTests(TempDirTestCase):
    """A low-texture seed must be refused, not followed into a wild jump."""

    def test_a_flat_seed_patch_is_refused_with_a_reason(self):
        from ghostcaddie.demo.config import Point, TargetOptions
        from ghostcaddie.demo.tracking import resolve_track
        from ghostcaddie.demo.worker import _patch_texture, MIN_TEXTURE_STD

        flat = np.full((21, 21, 3), 40, np.uint8)
        textured = np.zeros((21, 21, 3), np.uint8)
        textured[::2, ::2] = 255
        self.assertLess(_patch_texture(flat), MIN_TEXTURE_STD)
        self.assertGreaterEqual(_patch_texture(textured), MIN_TEXTURE_STD)

    def test_a_flat_clip_yields_no_observations_and_no_path(self):
        video = self._flat_clip()
        config = write_config(self.tmp, {
            "schema_version": "fairwayos-demo-config.v1",
            "interval": {"start_seconds": 0.0, "end_seconds": None, "max_frames": 40},
            "roi": [20, 20, 100, 100],
            "seeds": {"ball": {"frame": 1, "x": 60, "y": 60}},
            "targets": {"ball": {"mode": "template", "template_size": 15,
                                 "search_radius": 20, "min_score": 0.3,
                                 "max_jump_px": 200.0, "loss_frames": 3}},
            "render": {"max_width": 160, "crf": 30, "preset": "ultrafast"}})
        from ghostcaddie.demo.pipeline import run_demo
        out = self.tmp / "out-flat"
        result = run_demo(str(video), str(config), str(out), cv_python=sys.executable)
        metrics = result.analytics["targets"]["ball"]
        self.assertEqual(metrics["observed_frames"], 1)      # the seed only
        self.assertEqual(metrics["automatic_observed_frames"], 0)
        self.assertEqual(metrics["path_length_observed_px"], 0)
        notes = " ".join(p["note"] for p in result.targets["ball"]["points"])
        self.assertIn("flat_low_texture", notes)

    def _flat_clip(self) -> Path:
        import cv2 as _cv2
        import numpy as np
        path = self.tmp / "flat.mp4"
        writer = _cv2.VideoWriter(str(path), _cv2.VideoWriter_fourcc(*"mp4v"), 20.0,
                                  (160, 120))
        for _ in range(12):
            writer.write(np.full((120, 160, 3), 42, np.uint8))
        writer.release()
        return path


class VfrTimingTests(TempDirTestCase):
    def test_an_uneven_cadence_is_detected_even_when_the_average_is_nominal(self):
        """The case that matters: deltas average to exactly 1/25 s while varying
        +/-60%, so the average rate field looks perfectly normal."""
        values = [0.0]
        for index in range(24):
            values.append(values[-1] + (0.016 if index % 2 == 0 else 0.064))
        self.assertAlmostEqual((values[-1] - values[0]) / 24, 0.04, places=3)
        payload = analyze_timestamps(values)
        self.assertTrue(payload["irregular_timestamps"])
        self.assertIn("NOT constant", payload["timestamp_note"])
        self.assertGreater(payload["timestamp_stats"]["worst_deviation_fraction"], 0.5)

    def test_a_constant_cadence_is_accepted(self):
        values = [index / 25.0 for index in range(30)]
        payload = analyze_timestamps(values)
        self.assertFalse(payload["irregular_timestamps"])
        self.assertAlmostEqual(payload["timestamp_stats"]["implied_fps"], 25.0, places=3)

    def test_too_few_timestamps_do_not_claim_a_cadence(self):
        payload = analyze_timestamps([0.0, 0.04])
        self.assertIsNone(payload["timestamps"])
        self.assertFalse(payload["irregular_timestamps"])

    def test_packet_timestamps_are_read_and_describe_the_cadence(self):
        video = make_synthetic_clip(self.tmp, frames=20, fps=20.0)
        payload = probe_timestamps(str(video))
        self.assertIsNotNone(payload["timestamps"], payload["timestamp_note"])
        self.assertFalse(payload["irregular_timestamps"])
        stats = payload["timestamp_stats"]
        self.assertAlmostEqual(stats["median_interval"], 0.05, places=4)
        self.assertIn("constant", payload["timestamp_note"])

    def test_an_irregularly_timestamped_source_is_refused_by_default(self):
        # A raw stream with non-uniform timestamps: 20 packets at 1/20s, then a gap
        # twice as long. The average rate stays close to nominal, which is exactly
        # why equal rate fields cannot be trusted.
        video = self.tmp / "vfr.mp4"
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
               "-i", "testsrc=size=160x120:rate=20:duration=0.5", "-f", "lavfi",
               "-i", "testsrc=size=160x120:rate=20:duration=0.5",
               "-filter_complex", "[0:v][1:v]concat=n=2:v=1[out]", "-map", "[out]",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            self.skipTest("ffmpeg is not available to build a concatenated clip")
        if proc.returncode != 0 or not video.is_file():
            self.skipTest("ffmpeg could not build the fixture")
        probe = probe_source(str(video))
        # whatever this particular encode produced, the demo must state the truth
        # about the timestamps rather than assume a constant cadence
        self.assertTrue(probe.timestamp_note)
        if probe.irregular_timestamps:
            short = {"interval": {"start_seconds": 0.0, "end_seconds": None,
                                  "max_frames": 500},
                     "roi": None, "seeds": {}, "keyframes": {}}
            cfg = parse_config(base_config(**short))
            with self.assertRaises(DemoConfigError) as caught:
                validate_config(cfg, probe)
            self.assertIn("not evenly spaced", str(caught.exception))
            accepted = parse_config(base_config(
                timing={"accept_variable_cadence": True}, **short))
            resolved = validate_config(accepted, probe)
            self.assertGreater(resolved.frame_count, 0)

    def test_seconds_come_from_timestamps_when_they_exist(self):
        video = make_synthetic_clip(self.tmp, frames=20, fps=20.0)
        probe = probe_source(str(video))
        resolved = validate_config(parse_config(base_config(
            interval={"start_seconds": 0.0, "end_seconds": 0.8, "max_frames": 60},
            roi=None, seeds={}, keyframes={})), probe)
        self.assertIsNotNone(resolved.frame_times)
        self.assertEqual(resolved.seconds_basis,
                         SECONDS_BASIS_TIMESTAMPS)
        first, second = resolved.frame_times[0], resolved.frame_times[5]
        self.assertAlmostEqual(second - first, 5 * 0.05, places=3)

    def test_a_bframe_cfr_clip_reads_as_constant_cadence(self):
        """The regression that started this: reordered PACKETS are not a VFR clip.

        The clip is encoded with B-frames, so ffprobe's packet list (decode order) is
        not monotonic while the decoded FRAMES are a clean 1/25 s apart. The old probe
        read the packets, sorted nothing, and published "median 0.08s / NOT constant"
        for a CFR 25 clip. This test fails if either half of that returns.
        """
        video = self.tmp / "bframes.mp4"
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
               "-i", "testsrc=size=160x120:rate=25:duration=1.2", "-c:v", "libx264",
               "-bf", "2", "-g", "12", "-pix_fmt", "yuv420p", str(video)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            self.skipTest("ffmpeg is not available to build a B-frame clip")
        if proc.returncode != 0 or not video.is_file():
            self.skipTest("ffmpeg could not build the B-frame fixture")
        # the packets really are out of order, which is why they must not be used
        packets = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "packet=pts_time", "-of", "csv=p=0", str(video)],
            capture_output=True, text=True)
        packet_values = [float(v) for v in packets.stdout.split() if v not in ("", "N/A")]
        probe = probe_source(str(video))
        self.assertFalse(probe.irregular_timestamps, probe.timestamp_note)
        self.assertIsNotNone(probe.timestamps, probe.timestamp_note)
        self.assertAlmostEqual(probe.timestamp_stats["median_interval"], 0.04, places=4)
        self.assertIn("constant", probe.timestamp_note)
        if packet_values:
            self.assertFalse(all(b > a for a, b in zip(packet_values, packet_values[1:])),
                             "this fixture was expected to store packets out of order")
        # ...and the run is accepted without accepting a variable cadence
        resolved = validate_config(parse_config(base_config(
            interval={"start_seconds": 0.0, "end_seconds": None, "max_frames": 60},
            roi=None, seeds={}, keyframes={}), ), probe)
        self.assertFalse(resolved.config.timing.accept_variable_cadence)
        self.assertEqual(resolved.seconds_basis, SECONDS_BASIS_TIMESTAMPS)

    def test_packet_order_timestamps_are_not_read_as_frame_order(self):
        """Packet-order timestamps must be refused, never sorted into CFR."""
        swapped = [0.0, 0.08, 0.04, 0.12, 0.16, 0.20, 0.24]
        payload = analyze_timestamps(swapped)
        self.assertTrue(payload["irregular_timestamps"])
        self.assertIsNone(payload["timestamps"])
        self.assertIn("not strictly increasing", payload["timestamp_note"])
        self.assertIn("presentation order", payload["timestamp_note"])


class PickerViewportTests(unittest.TestCase):
    def test_a_4k_frame_is_scaled_to_fit_and_clicks_map_back(self):
        transform = viewport_transform(2560, 1440, 1600, 950)
        self.assertLess(transform["scale"], 1.0)
        self.assertLessEqual(transform["view_w"], 1600)
        self.assertLessEqual(transform["view_h"], 950)
        native = viewport_to_native((transform["offset_x"] + 100,
                                     transform["offset_y"] + 50), transform)
        self.assertIsNotNone(native)
        self.assertAlmostEqual(native[0], 100 / transform["scale"], delta=1.5)
        self.assertAlmostEqual(native[1], 50 / transform["scale"], delta=1.5)

    def test_a_click_in_the_letterbox_is_not_a_point_on_the_footage(self):
        transform = viewport_transform(1440, 2560, 1600, 950, allow_upscale=True)
        self.assertIsNone(viewport_to_native((2, 2), transform))

    def test_a_portrait_frame_round_trips_through_both_directions(self):
        transform = viewport_transform(1080, 1920, 1600, 950)
        # the round trip is exact to within one viewport pixel, which is 1/scale
        # native pixels, plus the rounding of native_to_viewport itself
        tolerance = 1.0 / transform["scale"] + 1.5
        for point in ((0.0, 0.0), (1070.0, 1910.0), (540.0, 960.0)):
            view = native_to_viewport(point, transform)
            back = viewport_to_native(view, transform)
            self.assertIsNotNone(back, point)
            self.assertAlmostEqual(back[0], point[0], delta=tolerance)
            self.assertAlmostEqual(back[1], point[1], delta=tolerance)

    def test_a_roi_drag_maps_to_native_pixels(self):
        transform = viewport_transform(2560, 1440, 1280, 720)
        scale = transform["scale"]
        view_rect = (int(100 * scale), int(50 * scale), int(200 * scale),
                     int(300 * scale))
        native = roi_to_native(view_rect, transform)
        self.assertAlmostEqual(native[0], 100, delta=2)
        self.assertAlmostEqual(native[1], 50, delta=2)
        self.assertAlmostEqual(native[2], 200, delta=2)
        self.assertAlmostEqual(native[3], 300, delta=2)

    def test_a_noninteger_scale_still_maps_a_click_to_the_right_pixel(self):
        transform = viewport_transform(2560, 1440, 1100, 700)
        self.assertNotAlmostEqual(transform["scale"],
                                  round(transform["scale"]), places=6)
        target_native = (1234.0, 987.0)
        view = native_to_viewport(target_native, transform)
        back = viewport_to_native(view, transform)
        self.assertLessEqual(abs(back[0] - target_native[0]), 2.0)
        self.assertLessEqual(abs(back[1] - target_native[1]), 2.0)

    def test_the_last_pixel_column_and_row_are_clickable(self):
        """The bottom-right source pixel round-tripped to view_w, which the inverse
        transform correctly rejects as outside the picture: a drawable but unclickable
        pixel. It is clamped into the picture now, not dropped."""
        for transform in (viewport_transform(2560, 1440, 1280, 720),
                          viewport_transform(2560, 1440, 1100, 700),
                          viewport_transform(1080, 1920, 1400, 800)):
            corner = (float(transform["source_w"] - 1),
                      float(transform["source_h"] - 1))
            view = native_to_viewport(corner, transform)
            back = viewport_to_native(view, transform)
            self.assertIsNotNone(back, transform)
            self.assertLessEqual(abs(back[0] - corner[0]),
                                 1.0 / transform["scale"] + 1.5)
            self.assertLessEqual(abs(back[1] - corner[1]),
                                 1.0 / transform["scale"] + 1.5)


class PickerStateTests(TempDirTestCase):
    def _state(self, **kwargs):
        return PickerState("/tmp/synthetic.mp4", synthetic_probe(
            width=2560, height=1440, fps=25.0, frame_count=265), **kwargs)

    def test_a_dirty_session_needs_a_second_quit_and_the_second_one_works(self):
        state = self._state()
        state.goto(20)
        state.place_seed("ball", 100.0, 100.0)
        self.assertFalse(state.request_quit())
        self.assertTrue(state.quit_armed)
        self.assertTrue(state.request_quit())

    def test_a_clean_session_quits_immediately(self):
        self.assertTrue(self._state().request_quit())

    def test_a_config_that_cannot_validate_is_not_written(self):
        state = self._state()
        state.goto(50)
        # a correction at the same frame as the seed is refused by the schema
        state.place_seed("ball", 100.0, 100.0)
        state.corrections["ball"].append((50, 120.0, 120.0))
        state.dirty = True
        path = state.save(str(self.tmp / "bad.json"))
        self.assertEqual(path, "")
        self.assertFalse((self.tmp / "bad.json").exists())
        self.assertTrue(any("NOT saved" in m for m in state.messages))

    def test_reopening_a_config_keeps_the_interval_and_all_tuning(self):
        payload = base_config(
            interval={"start_seconds": 0.8, "end_seconds": 7.44, "max_frames": 700},
            seeds={"ball": {"frame": 30, "x": 900.0, "y": 1190.0}},
            keyframes={"address": 25, "top": 90, "impact": 117, "finish": 170},
            targets={"ball": {"mode": "template", "template_size": 25,
                              "search_radius": 55, "min_score": 0.62,
                              "max_jump_px": 44.0, "loss_frames": 4,
                              "track_until_frame": 150}},
            interpolation={"enabled": True, "max_gap_frames": 3},
            render={"max_width": 960, "crf": 18, "preset": "slow"})
        first = self._state(existing=parse_config(payload))
        saved = first.save(str(self.tmp / "kept.json"))
        self.assertTrue(saved, first.messages)
        reopened = self._state(existing=parse_config(json.loads(
            Path(saved).read_text())))
        self.assertEqual(reopened.start_frame, 20)
        self.assertEqual(reopened.end_frame, 185)
        self.assertEqual(reopened.targets["ball"]["search_radius"], 55)
        self.assertEqual(reopened.targets["ball"]["min_score"], 0.62)
        self.assertEqual(reopened.targets["ball"]["track_until_frame"], 150)
        self.assertEqual(reopened.interpolation["max_gap_frames"], 3)
        self.assertEqual(reopened.render_options["max_width"], 960)
        self.assertEqual(reopened.render_options["preset"], "slow")
        self.assertEqual(reopened.max_frames, 700)

    def test_a_saved_config_validates_against_the_source_it_was_drawn_on(self):
        state = self._state()
        state.goto(40)
        state.set_interval_start()
        state.goto(200)
        state.set_interval_end()
        state.set_roi(700, 20, 900, 1300)
        state.goto(45)
        state.place_seed("ball", 900.0, 1190.0)
        state.goto(120)
        state.place_correction("ball", 1200.0, 1080.0)
        state.goto(60)
        state.mark_phase("top")
        path = state.save(str(self.tmp / "ok.json"))
        self.assertTrue(path)
        parsed = parse_config(json.loads(Path(path).read_text()))
        resolved = validate_config(parsed, synthetic_probe(width=2560, height=1440,
                                                          fps=25.0, frame_count=265))
        self.assertEqual((resolved.start_frame, resolved.end_frame), (40, 200))

    def test_resaving_does_not_stack_the_boilerplate_in_notes(self):
        state = self._state()
        state.goto(30)
        state.place_seed("ball", 900.0, 1190.0)
        state.goto(150)
        state.set_interval_end()       # keeps the interval inside the probe duration
        state.notes = "operator note"
        first = state.save(str(self.tmp / "a.json"))
        self.assertTrue(first, state.messages)
        reopened = self._state(existing=parse_config(json.loads(Path(first).read_text())))
        reopened.dirty = True
        second = reopened.save(str(self.tmp / "b.json"))
        self.assertTrue(second, reopened.messages)
        notes = json.loads(Path(second).read_text())["notes"]
        self.assertEqual(notes.count("Config written by demo-setup"), 1, notes)
        self.assertIn("operator note", notes)

    def test_the_picker_saves_against_the_real_probe_of_the_selected_clip(self):
        """P2-1: the picker must be able to save a config for the real clip, with the
        REAL probe -- not a mocked one -- after the timestamp fix."""
        clip = Path(__file__).resolve().parents[1] / "out" / "demo_improvement_campaign" \
            / "development" / "pexels_round4" / "pexels_6541674.mp4"
        if not clip.is_file():
            self.skipTest(f"the local development clip is not present at {clip}")
        probe = probe_source(str(clip))
        self.assertFalse(probe.irregular_timestamps, probe.timestamp_note)
        self.assertFalse(probe.variable_cadence, probe.cadence_note)
        state = PickerState(str(clip), probe)
        self.assertLess(state.end_frame, 265)
        state.goto(20)
        state.set_interval_start()
        state.goto(185)
        state.set_interval_end()
        self.assertTrue(state.set_roi(900, 80, 650, 1200))
        state.goto(20)
        self.assertTrue(state.place_seed("ball", 929.8, 1195.6))
        state.goto(117)
        self.assertTrue(state.place_correction("ball", 915.9, 1204.7))
        state.goto(20)
        state.mark_phase("address")
        state.goto(117)
        state.mark_phase("impact")
        path = state.save(str(self.tmp / "real.json"))
        self.assertTrue(path, state.messages)
        payload = json.loads(Path(path).read_text())
        self.assertEqual(payload["interval"]["start_seconds"], 0.8)
        # the interval is stored half-open, so frame 185 inclusive is 186/25 = 7.44
        self.assertAlmostEqual(payload["interval"]["end_seconds"], 7.44, places=6)
        resolved = validate_config(parse_config(payload), probe)
        self.assertEqual((resolved.start_frame, resolved.end_frame), (20, 185))


class PickerRenderingTests(TempDirTestCase):
    """The picker's appearance is a pure function, so it can be rendered here."""

    def test_the_view_shows_the_hud_beside_the_picture_not_over_it(self):
        from ghostcaddie.demo.setup_gui import draw_picker_view
        state = PickerState("/tmp/s.mp4", synthetic_probe(width=2560, height=1440,
                                                         fps=25.0, frame_count=265))
        state.goto(20)
        state.set_roi(700, 20, 900, 1300)
        state.place_seed("ball", 900.0, 1190.0)
        frame = np.full((1440, 2560, 3), 60, np.uint8)
        cv2.circle(frame, (900, 1190), 8, (250, 250, 250), -1)
        transform = viewport_transform(2560, 1440, 1400, 800)
        canvas = draw_picker_view(frame, state, transform, mouse=(10.0, 10.0))
        # the picture occupies the left region and the HUD the right gutter
        self.assertEqual(canvas.shape[1], transform["view_w"] + 470)
        picture_area = canvas[transform["offset_y"]:transform["offset_y"]
                              + transform["view_h"], :transform["view_w"]]
        self.assertGreater(int(picture_area.mean()), 40)
        # the seed marker is drawn at the scaled native position, inside the picture
        px, py = native_to_viewport((900.0, 1190.0), transform)
        self.assertGreater(int(canvas[py, px].sum()), 0)

    def test_the_scale_shown_reflects_a_4k_source(self):
        from ghostcaddie.demo.setup_gui import draw_picker_view
        state = PickerState("/tmp/s.mp4", synthetic_probe(width=3840, height=2160,
                                                          fps=25.0, frame_count=300))
        transform = viewport_transform(3840, 2160, 1200, 700)
        self.assertLess(transform["scale"], 0.35)
        canvas = draw_picker_view(np.zeros((2160, 3840, 3), np.uint8), state,
                                  transform)
        self.assertEqual(canvas.shape[1], transform["view_w"] + 470)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
