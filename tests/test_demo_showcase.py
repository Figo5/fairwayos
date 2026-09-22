"""Focused tests for the opt-in showcase presentation layer.

Groups, at three levels:

* **stdlib contract** — the config block, the moving-ROI interpolation, the playback
  stride/holds, the layout rectangles, the metric values, the phase badge and the
  wording. These run anywhere, without cv2, because the decisions they pin are exactly
  the ones that must not depend on a renderer.
* **drawing** — the renderer's own promises: the picture keeps its size (no sidebar),
  an unavailable value is drawn dimmed rather than invented, the club marker's amber
  fill survives its own outline, markers are sized in display pixels, and stills and
  the contact sheet are real files with the documented names.
* **end to end** — real pipeline runs in showcase mode on generated synthetic clips,
  asserting the artifact names, the stride/hold playback record, the re-runnable
  config and the provenance.

No source footage, no research media, no model, no network.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ghostcaddie.demo import DEMO_CONFIG_SCHEMA
from ghostcaddie.demo.config import (effective_roi, load_config, parse_config,
                                     roi_envelope_for, validate_config)
from ghostcaddie.demo.errors import DemoConfigError
from ghostcaddie.demo import showcase as sc
from ghostcaddie.demo.paths import (SHOWCASE_STILL_PHASES, out_paths,
                                    resolve_out_paths, showcase_out_paths,
                                    showcase_still_path)
from ghostcaddie.demo.probe import probe_source
from ghostcaddie.demo.showcase_report import (BANNED_CLAIMS, humanise,
                                              render_linkedin_draft,
                                              render_showcase_report)
from tests.demo_fixtures import (TempDirTestCase, make_synthetic_clip, require_cv2,
                                 synthetic_probe)

ROI_KEYFRAMES = [
    {"frame": 0, "roi": [80, 40, 160, 180]},
    {"frame": 20, "roi": [120, 60, 120, 140]},
    {"frame": 39, "roi": [160, 80, 90, 120]},
]


def showcase_payload(**showcase_overrides) -> dict:
    return {
        "schema_version": DEMO_CONFIG_SCHEMA,
        "interval": {"start_seconds": 0.0, "end_seconds": 2.0, "max_frames": 400},
        "roi": [80, 40, 160, 180],
        "seeds": {"ball": {"frame": 1, "x": 61.0, "y": 200.0}},
        "keyframes": {"address": 0, "top": 13, "impact": 26, "finish": 39},
        "targets": {"ball": {"mode": "template", "template_size": 15,
                             "search_radius": 20, "min_score": 0.5}},
        "corrections": {},
        "interpolation": {"enabled": True, "max_gap_frames": 2},
        "render": {"max_width": 320, "crf": 28, "preset": "ultrafast"},
        "showcase": showcase_overrides or {"enabled": True},
    }


# ---------------------------------------------------------------- config


class ShowcaseConfigTests(unittest.TestCase):
    def test_the_shipped_template_parses_as_it_ships(self):
        """A template that cannot be loaded is not a template.

        Its values are illustrative and are never validated against a clip here (that
        needs the real source), but the schema must accept the file exactly as
        committed, or an operator's first experience is a parse error.
        """
        config = load_config(str(REPO / "examples" / "showcase_config.template.json"))
        self.assertTrue(config.showcase.enabled)
        self.assertEqual(config.showcase.metrics,
                         ("shoulder_tilt", "hip_tilt", "knee_flexion", "hand_rise"))
        self.assertEqual(config.render.stride, 3)
        self.assertEqual(len(config.roi_keyframes), 3)
        self.assertEqual(len(config.showcase.holds), 1)

    def test_the_block_is_off_by_default(self):
        payload = showcase_payload()
        payload.pop("showcase")
        config = parse_config(payload)
        self.assertFalse(config.showcase.enabled)
        self.assertEqual(config.showcase.metrics, sc.DEFAULT_METRICS)
        self.assertIn("showcase", config.to_dict())
        self.assertEqual(config.render.stride, 1, "the default pace is untouched")

    def test_enabling_the_block_keeps_every_documented_default(self):
        config = parse_config(showcase_payload(enabled=True))
        block = config.showcase
        self.assertTrue(block.enabled)
        self.assertEqual(block.brand, "FAIRWAY OS")
        self.assertEqual(block.brand_sub, "SWING ANALYSIS")
        self.assertIn("OPERATOR-ASSISTED", block.disclosure)
        self.assertEqual(block.metrics, sc.DEFAULT_METRICS)
        self.assertEqual(block.club_trail_points, sc.CLUB_TRAIL_POINTS)
        self.assertEqual(block.ball_trail_points, sc.BALL_TRAIL_POINTS)
        self.assertIsNone(block.card_row_max_width)
        self.assertEqual(block.holds, ())

    def test_a_metric_typo_is_refused_rather_than_dropped(self):
        with self.assertRaises(DemoConfigError) as caught:
            parse_config(showcase_payload(
                metrics=["shoulder_tilt", "hip_tilt", "hip_tlilt"]))
        self.assertIn("not a known metric", str(caught.exception))

    def test_the_card_count_is_bounded(self):
        with self.assertRaises(DemoConfigError):
            parse_config(showcase_payload(metrics=["shoulder_tilt", "hip_tilt"]))
        with self.assertRaises(DemoConfigError):
            parse_config(showcase_payload(
                metrics=["shoulder_tilt", "hip_tilt", "tempo", "sway", "head_movement"]))
        config = parse_config(showcase_payload(
            metrics=["shoulder_tilt", "hip_tilt", "knee_flexion", "hand_rise"]))
        self.assertEqual(len(config.showcase.metrics), 4)

    def test_a_repeated_metric_is_refused(self):
        with self.assertRaises(DemoConfigError) as caught:
            parse_config(showcase_payload(metrics=["tempo", "tempo", "sway"]))
        self.assertIn("repeats", str(caught.exception))

    def test_an_unknown_field_makes_the_misspelling_visible(self):
        with self.assertRaises(DemoConfigError) as caught:
            parse_config(showcase_payload(enabled=True, metric=["tempo"]))
        self.assertIn("unknown field", str(caught.exception))

    def test_empty_brand_or_disclosure_is_refused(self):
        with self.assertRaises(DemoConfigError):
            parse_config(showcase_payload(brand="  "))
        with self.assertRaises(DemoConfigError):
            parse_config(showcase_payload(disclosure=""))

    def test_trail_encoder_and_card_row_bounds_are_validated(self):
        with self.assertRaises(DemoConfigError):
            parse_config(showcase_payload(club_trail_points=1))
        with self.assertRaises(DemoConfigError):
            parse_config(showcase_payload(preset="nope"))
        with self.assertRaises(DemoConfigError):
            parse_config(showcase_payload(crf=99))
        with self.assertRaises(DemoConfigError):
            parse_config(showcase_payload(card_row_max_width=10))
        config = parse_config(showcase_payload(max_width=None,
                                              card_row_max_width=1450))
        self.assertIsNone(config.showcase.max_width)
        self.assertEqual(config.showcase.card_row_max_width, 1450)

    def test_holds_are_bounded_and_unique(self):
        config = parse_config(showcase_payload(
            holds=[{"frame": 26, "seconds": 0.6, "label": "impact"},
                   {"frame": 30, "seconds": 0.4}]))
        self.assertEqual(len(config.showcase.holds), 2)
        self.assertAlmostEqual(config.showcase.holds[0]["seconds"], 0.6)
        with self.assertRaises(DemoConfigError):
            parse_config(showcase_payload(holds=[{"frame": 26, "seconds": 9.0}]))
        with self.assertRaises(DemoConfigError):
            parse_config(showcase_payload(holds=[{"frame": 26, "seconds": 0.2},
                                                {"frame": 26, "seconds": 0.2}]))
        with self.assertRaises(DemoConfigError):
            parse_config(showcase_payload(
                holds=[{"frame": i, "seconds": 0.5} for i in range(6)]))

    def test_the_render_stride_is_a_validated_integer(self):
        payload = showcase_payload(enabled=True)
        payload["render"]["stride"] = 3
        self.assertEqual(parse_config(payload).render.stride, 3)
        payload["render"]["stride"] = 0
        with self.assertRaises(DemoConfigError):
            parse_config(payload)
        payload["render"]["stride"] = 1.5
        with self.assertRaises(DemoConfigError):
            parse_config(payload)

    def test_the_showcase_contract_does_not_collide_with_the_demo_contract(self):
        demo = {p.name for p in out_paths("/tmp/x").values() if p.suffix}
        show = {p.name for p in showcase_out_paths("/tmp/x").values() if p.suffix}
        self.assertEqual(demo & show, set())
        for phase in SHOWCASE_STILL_PHASES:
            self.assertEqual(showcase_still_path("/tmp/x", phase).name,
                             f"showcase_{phase}.png")
        self.assertEqual(showcase_out_paths("/tmp/x")["annotated"].name,
                         "showcase_annotated.mp4")
        self.assertEqual(resolve_out_paths("/tmp/x", showcase=True)["report"].name,
                         "showcase_report.md")
        self.assertEqual(resolve_out_paths("/tmp/x", showcase=False)["report"].name,
                         "demo_report.md")

    def test_the_resolved_config_carries_the_showcase_block_and_the_moving_roi(self):
        payload = showcase_payload(enabled=True)
        payload["roi_keyframes"] = ROI_KEYFRAMES
        config = parse_config(payload)
        resolved = validate_config(config, synthetic_probe(width=320, height=240))
        effective = resolved.effective()
        self.assertTrue(effective["showcase"]["enabled"])
        self.assertEqual(effective["showcase"]["metrics"], list(sc.DEFAULT_METRICS))
        self.assertEqual(len(effective["roi_keyframes"]), 3)
        self.assertEqual(effective["roi_envelope_xywh"], [80, 40, 170, 180])


class RoiKeyframeTests(unittest.TestCase):
    def test_parsing_keeps_the_author_note(self):
        payload = showcase_payload(enabled=True)
        payload["roi_keyframes"] = [{"frame": 0, "roi": [80, 40, 160, 180],
                                     "note": "PRIMARY: manual visual ROI"}]
        config = parse_config(payload)
        self.assertEqual(config.roi_keyframes[0].note, "PRIMARY: manual visual ROI")
        self.assertEqual(config.to_dict()["roi_keyframes"][0]["roi"],
                         [80, 40, 160, 180])

    def test_a_fractional_duplicate_or_misordered_list_is_refused(self):
        for block in (
                [{"frame": 0.5, "roi": [80, 40, 160, 180]}],
                [{"frame": 10, "roi": [80, 40, 160, 180]},
                 {"frame": 10, "roi": [90, 40, 160, 180]}],
                [{"frame": 30, "roi": [80, 40, 160, 180]},
                 {"frame": 10, "roi": [90, 40, 160, 180]}],
                [{"frame": 10, "roi": [80, 40]}],
                [{"frame": 10, "roi": [80, 40, 0, 180]}],
        ):
            with self.assertRaises(DemoConfigError):
                payload = showcase_payload(enabled=True)
                payload["roi_keyframes"] = block
                parse_config(payload)

    def test_a_box_outside_the_frame_or_the_interval_is_refused_at_validation(self):
        payload = showcase_payload(enabled=True)
        payload["roi_keyframes"] = [{"frame": 0, "roi": [300, 40, 160, 180]}]
        with self.assertRaises(DemoConfigError) as caught:
            validate_config(parse_config(payload), synthetic_probe(width=320,
                                                                    height=240))
        self.assertIn("does not fit inside", str(caught.exception))

        payload["roi_keyframes"] = [{"frame": 400, "roi": [80, 40, 160, 180]}]
        with self.assertRaises(DemoConfigError) as caught:
            validate_config(parse_config(payload), synthetic_probe(width=320,
                                                                    height=240))
        self.assertIn("outside the processed interval", str(caught.exception))

    def test_the_first_box_becomes_the_initial_crop_when_no_static_roi_is_given(self):
        payload = showcase_payload(enabled=True)
        payload.pop("roi")
        payload["roi_keyframes"] = ROI_KEYFRAMES
        config = parse_config(payload)
        self.assertIsNone(config.roi)
        resolved = validate_config(config, synthetic_probe(width=320, height=240))
        self.assertEqual(list(resolved.config.roi), [80, 40, 160, 180])

    def test_interpolation_is_linear_in_position_and_size_and_holds_outside(self):
        keys = [dict(k) for k in ROI_KEYFRAMES]
        self.assertEqual(sc.interpolate_roi(keys, 0), [80, 40, 160, 180])
        self.assertEqual(sc.interpolate_roi(keys, 10), [100, 50, 140, 160])
        self.assertEqual(sc.interpolate_roi(keys, 20), [120, 60, 120, 140])
        # a quarter of the way from the first box to the second
        self.assertEqual(sc.interpolate_roi(keys, 5), [90, 45, 150, 170])
        # held constant after the last box, and at/before the first one
        self.assertEqual(sc.interpolate_roi(keys, 100), [160, 80, 90, 120])
        later = [{"frame": 20, "roi": [120, 60, 120, 140]},
                 {"frame": 39, "roi": [160, 80, 90, 120]}]
        self.assertEqual(sc.interpolate_roi(later, 5), [120, 60, 120, 140])
        self.assertIsNone(sc.interpolate_roi([], 5))
        self.assertEqual(effective_roi(parse_config(showcase_payload(
            enabled=True)), 10), [80, 40, 160, 180])
        moving = showcase_payload(enabled=True)
        moving["roi_keyframes"] = keys
        self.assertEqual(effective_roi(parse_config(moving), 10), [100, 50, 140, 160])
        self.assertIsNone(sc.roi_envelope([]))

    def test_the_envelope_covers_every_box(self):
        envelope = sc.roi_envelope([dict(k) for k in ROI_KEYFRAMES])
        for key in ROI_KEYFRAMES:
            x, y, w, h = key["roi"]
            self.assertLessEqual(envelope[0], x)
            self.assertLessEqual(envelope[1], y)
            self.assertGreaterEqual(envelope[0] + envelope[2], x + w)
            self.assertGreaterEqual(envelope[1] + envelope[3], y + h)
        payload = showcase_payload(enabled=True)
        payload["roi_keyframes"] = ROI_KEYFRAMES
        config = parse_config(payload)
        self.assertEqual(roi_envelope_for(config), sc.roi_envelope(
            [dict(k) for k in ROI_KEYFRAMES]))


class PlaybackPaceTests(unittest.TestCase):
    def test_the_stride_keeps_every_nth_source_frame(self):
        self.assertEqual(sc.stride_frames(48, 56, 3), [48, 51, 54])
        self.assertEqual(sc.stride_frames(48, 1176, 3)[:3], [48, 51, 54])
        self.assertEqual(len(sc.stride_frames(48, 1176, 3)), 377)
        self.assertEqual(sc.stride_frames(0, 9, 1), list(range(10)))
        self.assertEqual(sc.stride_frames(0, 9, 0), list(range(10)))

    def test_a_hold_repeats_one_already_drawn_frame(self):
        frames = [10, 13, 16, 19]
        holds = [{"frame": 13, "seconds": 0.5, "label": "last verified launch"}]
        plan = sc.expand_holds(frames, holds, 20.0)
        self.assertEqual([e["source_frame"] for e in plan],
                         [10] + [13] * 10 + [16, 19])
        self.assertFalse(plan[1]["held"])
        self.assertTrue(all(entry["held"] for entry in plan[2:11]))
        self.assertEqual(plan[2]["hold_label"], "last verified launch")
        self.assertEqual(sc.expand_holds(frames, [], 20.0)[0],
                         {"source_frame": 10, "held": False, "hold_label": None})

    def test_the_hold_record_locates_the_repeat_in_the_output_timeline(self):
        records = sc.hold_records([10, 13, 16], [{"frame": 13, "seconds": 0.5,
                                                  "label": "x"}], 20.0)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["source_frame"], 13)
        self.assertEqual(record["output_frames_held"], 10)
        self.assertEqual(record["output_frames"], [1, 10])
        self.assertIn("held_replay", record["status"])
        self.assertIn("no new observation", record["note"])

    def test_a_hold_on_a_frame_that_is_not_encoded_is_reported_not_invented(self):
        records = sc.hold_records([10, 13, 16], [{"frame": 11, "seconds": 0.5}], 20.0)
        self.assertEqual(records, [])


# ---------------------------------------------------------------- layout


class ShowcaseLayoutTests(unittest.TestCase):
    def test_every_rectangle_is_inside_the_picture_at_any_size(self):
        for width, height in ((320, 240), (640, 360), (1280, 720), (1920, 1080),
                              (1080, 1920)):
            for metrics in (sc.DEFAULT_METRICS,
                            ("shoulder_tilt", "hip_tilt", "knee_flexion", "hand_rise")):
                layout = sc.plan_layout(width, height, metrics)
                self.assertEqual(sc.layout_problems(layout), [],
                                 f"{width}x{height} {metrics}")
                self.assertEqual(layout.width, width)
                self.assertEqual(layout.height, height)

    def test_the_picture_is_never_widened_for_chrome(self):
        layout = sc.plan_layout(640, 360, sc.DEFAULT_METRICS)
        serialised = layout.to_dict()
        self.assertTrue(serialised["frame_aspect_preserved"])
        self.assertEqual(serialised["chrome_outside_picture_px2"], 0)
        for entry in serialised["cards"]:
            self.assertLessEqual(entry["x"] + entry["w"], 640)
            self.assertLessEqual(entry["y"] + entry["h"], 360)

    def test_the_1080p_card_strip_matches_the_operator_request(self):
        layout = sc.plan_layout(1920, 1080, ("shoulder_tilt", "hip_tilt",
                                             "knee_flexion", "hand_rise"),
                                max_row_width=1450)
        self.assertGreaterEqual(layout.strip_top, 900)
        self.assertLessEqual(layout.strip_bottom, 1080)
        self.assertGreaterEqual(layout.cards[0].h, 100)
        self.assertLessEqual(layout.cards[0].h, 120)
        self.assertLessEqual(layout.cards[0].w, 345)
        self.assertGreaterEqual(layout.cards[0].w, 330)
        self.assertLessEqual(layout.cards[-1].right(), 1450)
        self.assertEqual(layout.brand_font, 36)
        self.assertEqual(layout.brand_sub_font, 20)
        self.assertEqual(layout.badge_font, 40)
        self.assertGreaterEqual(layout.card_value_font, 50)
        self.assertLessEqual(layout.card_value_font, 58)
        self.assertGreaterEqual(layout.card_label_font, 24)
        self.assertLessEqual(layout.card_label_font, 28)
        # the strip never reaches the top chrome
        self.assertGreater(layout.strip_top, layout.brand.bottom() + 20)

    def test_the_card_row_stops_early_when_a_width_cap_is_given(self):
        capped = sc.plan_layout(1920, 1080, sc.DEFAULT_METRICS, max_row_width=1450)
        full = sc.plan_layout(1920, 1080, sc.DEFAULT_METRICS)
        self.assertLess(capped.row_width, full.row_width)
        self.assertLessEqual(capped.cards[-1].right(), 1450)
        self.assertGreater(capped.cards[0].x, 0)

    def test_the_cards_never_overlap_and_are_wide_enough_to_read(self):
        layout = sc.plan_layout(1280, 720, sc.DEFAULT_METRICS)
        self.assertEqual(len(layout.cards), 4)
        for i in range(len(layout.cards)):
            for j in range(i + 1, len(layout.cards)):
                self.assertEqual(layout.cards[i].intersection_px2(layout.cards[j]), 0)
        for card in layout.cards:
            self.assertGreater(card.w, layout.card_value_font)
            self.assertGreaterEqual(card.h, 34)

    def test_a_frame_too_small_for_chrome_is_refused(self):
        with self.assertRaises(ValueError):
            sc.plan_layout(100, 80, sc.DEFAULT_METRICS)

    def test_the_layout_check_finds_a_deliberate_overlap(self):
        """The checker must be able to fail, or it proves nothing."""
        layout = sc.plan_layout(640, 360, sc.DEFAULT_METRICS)
        layout.cards[1] = sc.Rect(layout.cards[0].x, layout.cards[0].y,
                                  layout.cards[0].w, layout.cards[0].h)
        layout.card_index[sc.DEFAULT_METRICS[1]] = layout.cards[1]
        problems = sc.layout_problems(layout)
        self.assertTrue(any("overlap" in problem for problem in problems), problems)

    def test_layout_check_finds_chrome_outside_the_picture(self):
        layout = sc.plan_layout(640, 360, sc.DEFAULT_METRICS)
        layout.disclosure = sc.Rect(600, 340, 200, 40)
        problems = sc.layout_problems(layout)
        self.assertTrue(any("not inside" in problem for problem in problems), problems)

    def test_the_measured_coverage_counts_the_roi_and_the_action(self):
        layout = sc.plan_layout(640, 360, sc.DEFAULT_METRICS)
        roi = [260, 60, 120, 200]          # native pixels, right of the cards' band
        points = [(100.0, 320.0)]          # inside the card band
        coverage = sc.chrome_coverage(layout, roi=roi, points_xy=points, scale=1.0)
        self.assertEqual(coverage["drawn_points_total"], 1)
        self.assertEqual(coverage["drawn_points_under_chrome"], 1)
        self.assertEqual(coverage["covered_points"][0]["band"],
                         f"card:{sc.DEFAULT_METRICS[0]}")
        self.assertIsNotNone(coverage["roi_covered_fraction"])
        clear = sc.chrome_coverage(layout, roi=roi, points_xy=[(300.0, 120.0)],
                                   scale=1.0)
        self.assertEqual(clear["drawn_points_under_chrome"], 0)
        self.assertEqual(clear["drawn_points_total"], 1)


# ---------------------------------------------------------------- values


class ShowcaseValueTests(unittest.TestCase):
    def test_missing_values_are_the_placeholder_and_never_a_number(self):
        for metric in sc.METRICS:
            self.assertEqual(sc.format_value(metric, None), sc.PLACEHOLDER)
            self.assertEqual(sc.format_value("nope", 3.0), sc.PLACEHOLDER)
            for bad in (float("nan"), float("inf"), float("-inf")):
                self.assertEqual(sc.format_value(metric, bad), sc.PLACEHOLDER)

    def test_values_are_formatted_with_their_registry_unit(self):
        self.assertEqual(sc.format_value("shoulder_tilt", 12.34), "12.3\u00b0")
        self.assertEqual(sc.format_value("shoulder_tilt", 12.34, unicode_ok=False),
                         "12.3 deg")
        self.assertEqual(sc.format_value("hip_tilt", -4.0), "-4.0\u00b0")
        self.assertEqual(sc.format_value("knee_flexion", 22.0), "22.0\u00b0")
        self.assertEqual(sc.format_value("hand_rise", 412.4), "412 px")
        self.assertEqual(sc.format_value("tempo", 3.5), "3.50:1")
        self.assertEqual(sc.format_value("tempo", 2.0), "2.00:1")
        self.assertEqual(sc.format_value("head_movement", 41.6), "42 px")

    def test_ascii_fallback_removes_typography_the_hershey_font_cannot_draw(self):
        self.assertEqual(sc.ascii_safe("OPERATOR-ASSISTED \u00b7 2D"),
                         "OPERATOR-ASSISTED  -  2D")
        self.assertEqual(sc.ascii_safe("41\u00b0"), "41 deg")
        self.assertEqual(sc.ascii_safe("A\u2192B"), "A->B")
        self.assertEqual(sc.display_text("x\u00b7y", unicode_ok=True), "x\u00b7y")
        self.assertEqual(sc.display_text("x\u00b7y", unicode_ok=False), "x - y")

    def test_per_frame_values_come_from_the_same_angles_the_report_uses(self):
        angles = {"shoulder_tilt_deg": 21.5, "hip_tilt_deg": -6.0,
                  "knee_interior_deg": {"left": 150.0, "right": 170.0},
                  "wrist_midpoint": [1000.0, 800.0]}
        right = sc.frame_values(angles, handedness="right", wrist_reference=1000.0)
        self.assertEqual(right["shoulder_tilt"], 21.5)
        self.assertEqual(right["hip_tilt"], -6.0)
        self.assertEqual(right["knee_flexion"], 30.0)   # 180 - 150 lead leg
        self.assertEqual(right["hand_rise"], 200.0)     # 1000 - 800, image y down
        left = sc.frame_values(angles, handedness="left", wrist_reference=1000.0)
        self.assertEqual(left["knee_flexion"], 10.0)    # lead is the right leg then
        # a rise needs a reference to rise from
        self.assertIsNone(sc.frame_values(angles, handedness="right")["hand_rise"])
        empty = sc.frame_values({}, handedness="right")
        self.assertTrue(all(value is None for value in empty.values()))

    def test_the_wrist_reference_is_the_first_frame_with_both_wrists(self):
        from tests.demo_fixtures import body_record
        records = [
            body_record(0, points={"nose": (100, 50)}, available=True),
            body_record(1, points={"wr_l": (100, 200)}, available=True),
            body_record(2, points={"wr_l": (100, 200), "wr_r": (120, 180)},
                        available=True),
            body_record(3, points={"wr_l": (100, 150), "wr_r": (120, 130)},
                        available=True),
        ]
        reference = sc.wrist_reference(records)
        self.assertEqual(reference["frame"], 2)
        self.assertEqual(reference["y"], 190.0)
        self.assertIn("image y", reference["note"])
        self.assertIsNone(sc.wrist_reference(records[:2]))
        self.assertIsNone(sc.wrist_midpoint({}))

    def test_a_metric_the_report_calls_unavailable_is_unavailable_on_the_card(self):
        analytics = {"body_2d": {"supported": [
            {"name": "head_displacement", "unit": "px", "value": 42.5}]}}
        phases = {"tempo": {"state": "supported", "backswing_downswing_ratio": 2.5,
                            "backswing_frames": 20, "downswing_frames": 8,
                            "backswing_start_marker": "takeaway", "label": "manual"}}
        values = sc.clip_values(analytics, phases)
        self.assertEqual(values["head_movement"]["value"], 42.5)
        self.assertEqual(values["sway"]["state"], "unavailable")
        self.assertIsNone(values["sway"]["value"])
        self.assertEqual(values["tempo"]["value"], 2.5)
        self.assertIn("back 20f", values["tempo"]["sub"])

    def test_tempo_is_unavailable_when_the_phases_say_so(self):
        values = sc.clip_values({"body_2d": {}},
                                {"tempo": {"state": "unavailable",
                                           "reason": "no top marker"}})
        self.assertIsNone(values["tempo"]["value"])
        self.assertEqual(values["tempo"]["reason"], "no top marker")

    def test_the_phase_badge_names_the_span_not_the_nearest_marker(self):
        markers = {"ADDRESS": 10, "TAKEAWAY": 20, "TOP": 40, "DOWNSWING": 50,
                   "IMPACT": 55, "FOLLOW-THROUGH": 60, "FINISH": 90}
        self.assertIsNone(sc.current_phase(9, markers))
        self.assertEqual(sc.current_phase(10, markers), "ADDRESS")
        self.assertEqual(sc.current_phase(19, markers), "ADDRESS")
        self.assertEqual(sc.current_phase(20, markers), "TAKEAWAY")
        self.assertEqual(sc.current_phase(54, markers), "DOWNSWING")
        self.assertEqual(sc.current_phase(55, markers), "IMPACT")
        self.assertEqual(sc.current_phase(500, markers), "FINISH")
        self.assertIsNone(sc.current_phase(30, {}))

    def test_the_badge_skips_a_phase_the_operator_did_not_mark(self):
        markers = {"ADDRESS": 10, "TAKEAWAY": None, "TOP": 40, "DOWNSWING": None,
                   "IMPACT": 55, "FOLLOW-THROUGH": None, "FINISH": None}
        self.assertEqual(sc.current_phase(30, markers), "ADDRESS")
        self.assertEqual(sc.current_phase(41, markers), "TOP")
        self.assertEqual(sc.current_phase(60, markers), "IMPACT")

    def test_the_impact_accent_is_short_and_bounded(self):
        self.assertEqual(sc.impact_accent(50, 50), 1.0)
        self.assertGreater(sc.impact_accent(51, 50), 0.0)
        self.assertLess(sc.impact_accent(52, 50), sc.impact_accent(51, 50))
        self.assertEqual(sc.impact_accent(54, 50), 0.0)
        self.assertEqual(sc.impact_accent(46, 50), 0.0)
        self.assertEqual(sc.impact_accent(50, None), 0.0)

    def test_phase_markers_map_the_config_order_onto_the_badge_names(self):
        phases = {"phases": {"address": {"frame": 3}, "top": {"frame": 9},
                             "impact": {"frame": 12}, "finish": {"frame": 30},
                             "takeaway": {"frame": None}}}
        markers = sc.phase_markers(phases)
        self.assertEqual(markers["ADDRESS"], 3)
        self.assertEqual(markers["TOP"], 9)
        self.assertEqual(markers["IMPACT"], 12)
        self.assertEqual(markers["FINISH"], 30)
        self.assertIsNone(markers["TAKEAWAY"])


# ---------------------------------------------------------------- drawing


class ShowcaseRenderTests(unittest.TestCase):
    """The renderer's promises, on a synthetic picture. No source footage."""

    def setUp(self):
        require_cv2(self)
        import numpy as np
        self.np = np
        self.picture = np.full((360, 640, 3), 70, np.uint8)
        self.picture[180:, :] = np.array([90, 110, 80], np.uint8)

    def chrome(self, *, available=True):
        return {
            "brand": "FAIRWAY OS", "brand_sub": "SWING ANALYSIS",
            "phase": "IMPACT",
            "cards": [
                {"metric": "shoulder_tilt", "label": "SHOULDER TILT",
                 "value": "21.5\u00b0" if available else sc.PLACEHOLDER,
                 "sub": "2D projected", "available": available},
                {"metric": "hip_tilt", "label": "HIP TILT",
                 "value": "-6.0\u00b0", "sub": "2D projected", "available": True},
                {"metric": "knee_flexion", "label": "LEAD KNEE FLEX",
                 "value": "30.0\u00b0", "sub": "lead leg", "available": True},
                {"metric": "hand_rise", "label": "HAND RISE", "value": "412 px",
                 "sub": "wrists above address", "available": True},
            ],
            "disclosure": sc.DEFAULT_DISCLOSURE,
        }

    def test_composing_keeps_the_frame_size_and_draws_something(self):
        from ghostcaddie.demo import render_showcase as rs
        layout = sc.plan_layout(640, 360, sc.DEFAULT_METRICS)
        image, record = rs.compose_frame(self.picture.copy(), layout=layout,
                                         chrome=self.chrome())
        self.assertEqual(image.shape, self.picture.shape)
        self.assertGreater(int(self.np.abs(image.astype(int)
                                          - self.picture.astype(int)).sum()), 0)
        self.assertEqual(record["layout_problems"], [])
        self.assertIn(record["text_engine"], ("pillow", "hershey"))
        self.assertEqual(record["phase_badge"], "IMPACT")
        # the pill is drawn content-fit, not at the layout's maximum slot width
        self.assertLess(record["phase_badge_px"], layout.phase_badge.w)

    def test_the_card_strip_has_no_outline_and_no_corner_circles(self):
        """The preview showed border circles at the card corners; they must be gone.

        The check is a measurement, not a look: with a clean translucent fill the
        card's corner pixel is a blend of card and background, while a drawn circle
        there would be much darker and fully opaque.
        """
        from ghostcaddie.demo import render_showcase as rs
        layout = sc.plan_layout(640, 360, sc.DEFAULT_METRICS)
        image, _ = rs.compose_frame(self.picture.copy(), layout=layout,
                                    chrome=self.chrome())
        card = layout.cards[0]
        corners = [image[card.y, card.x], image[card.y, card.right() - 1],
                   image[card.bottom() - 1, card.x],
                   image[card.bottom() - 1, card.right() - 1]]
        background = self.picture[card.y, card.x].astype(int)
        for corner in corners:
            delta = abs(corner.astype(int) - background)
            self.assertLess(int(delta.max()), 60,
                            "a corner pixel is far from the translucent blend, which "
                            "is what a drawn border circle looks like")

    def test_a_layout_for_another_size_is_refused_rather_than_scaled(self):
        from ghostcaddie.demo import render_showcase as rs
        layout = sc.plan_layout(1280, 720, sc.DEFAULT_METRICS)
        with self.assertRaises(ValueError):
            rs.compose_frame(self.picture.copy(), layout=layout, chrome=self.chrome())

    def test_an_unavailable_card_is_drawn_dimmer_than_an_available_one(self):
        from ghostcaddie.demo import render_showcase as rs
        layout = sc.plan_layout(640, 360, sc.DEFAULT_METRICS)
        bright = rs.compose_frame(self.picture.copy(), layout=layout,
                                  chrome=self.chrome(available=True))[0]
        dim = rs.compose_frame(self.picture.copy(), layout=layout,
                               chrome=self.chrome(available=False))[0]
        card = layout.cards[0]
        region = (slice(card.y, card.bottom()), slice(card.x, card.right()))
        self.assertGreater(float(bright[region].mean()), float(dim[region].mean()))
        self.assertNotAlmostEqual(float(dim[region].mean()),
                                  float(self.picture[region].mean()), places=3)

    def test_the_overlay_layer_draws_a_skeleton_and_markers_without_labels(self):
        from ghostcaddie.demo import render_showcase as rs
        layer, mask = rs.new_layers(640, 360)
        keypoints = [{"name": name, "x": 100 + i * 10, "y": 100 + i * 4,
                      "visible": True} for i, name in enumerate(
            ("nose", "sh_l", "sh_r", "el_l", "el_r", "wr_l", "wr_r", "hip_l",
             "hip_r", "kn_l", "kn_r", "ank_l", "ank_r"))]
        drawn = rs.draw_skeleton(layer, mask, keypoints, 1.0)
        self.assertGreater(drawn, 6)
        self.assertGreater(float(mask.sum()), 0.0)
        rs.draw_club_marker(layer, mask, {"x": 200.0, "y": 200.0,
                                          "status": "manual_anchor"}, 1.0)
        rs.draw_ball_marker(layer, mask, {"x": 210.0, "y": 210.0,
                                          "status": "observed_template"}, 1.0)
        before = float(mask.sum())
        rs.draw_club_marker(layer, mask, {"x": 400.0, "y": 300.0, "status": "lost"}, 1.0)
        rs.draw_ball_marker(layer, mask, {"x": None, "y": None, "status": "missing"},
                            1.0)
        self.assertEqual(float(mask.sum()), before,
                         "a lost or missing target must leave no stale marker")

    def test_the_club_marker_keeps_its_amber_fill_under_the_bright_outline(self):
        """The first render drew the fill and then whitened every selected pixel."""
        from ghostcaddie.demo import render_showcase as rs
        layer, mask = rs.new_layers(480, 320)
        rs.draw_club_marker(layer, mask, {"x": 240.0, "y": 160.0,
                                          "status": "manual_anchor"}, 1.0)
        selected = mask > 0
        self.assertTrue(selected.any())
        pixels = layer[selected]
        amber = self.np.array(rs.CLUBHEAD)
        is_amber = self.np.all(self.np.abs(pixels.astype(int) - amber) < 40, axis=1)
        is_white = self.np.all(pixels.astype(int) > 235, axis=1)
        self.assertGreater(int(is_amber.sum()), 60,
                           "the amber fill is missing: the outline overwrote it")
        self.assertGreater(int(is_white.sum()), 10, "the outline is missing")

    def test_marker_and_line_sizes_follow_the_display_height_not_the_coordinates(self):
        from ghostcaddie.demo import render_showcase as rs
        self.assertAlmostEqual(rs.style_scale(1080), 1.0, places=3)
        self.assertLess(rs.style_scale(360), 0.5)
        self.assertGreater(rs.style_scale(2160), 1.5)
        # a 4K source rendered at 1080p keeps a club marker of ~22px, not ~5px
        layer, mask = rs.new_layers(1920, 1080)
        rs.draw_club_marker(layer, mask, {"x": 900.0, "y": 500.0,
                                          "status": "manual_anchor"},
                            rs.style_scale(1080))
        ys, xs = self.np.where(mask > 0)
        self.assertGreaterEqual(ys.max() - ys.min(), 18)
        self.assertLessEqual(ys.max() - ys.min(), 30)
        layer, mask = rs.new_layers(1920, 1080)
        rs.draw_ball_marker(layer, mask, {"x": 900.0, "y": 500.0,
                                          "status": "observed_template"},
                            rs.style_scale(1080))
        ys, xs = self.np.where(mask > 0)
        self.assertGreaterEqual(xs.max() - xs.min(), 14)
        self.assertLessEqual(xs.max() - xs.min(), 26)

    def test_the_text_engine_shrinks_type_to_fit_rather_than_overflowing(self):
        from ghostcaddie.demo import render_showcase as rs
        engine = rs.TextEngine()
        long_text = "LEAD KNEE FLEXION IN THE LEAD LEG"
        fitted = engine.fit_size(long_text, 260, 20, bold=True, minimum=7)
        self.assertLess(fitted, 20)
        self.assertLessEqual(engine.measure(long_text, fitted, bold=True)[0], 260)
        # the shrink has a floor and is never allowed to run away
        floored = engine.fit_size(long_text, 20, 20, bold=True, minimum=9)
        self.assertEqual(floored, 9)

    def test_the_text_engine_reports_which_engine_drew_the_cards(self):
        from ghostcaddie.demo import render_showcase as rs
        engine = rs.TextEngine()
        report = rs.font_report(engine)
        self.assertIn(report["engine"], ("pillow", "hershey"))
        if report["engine"] == "pillow":
            self.assertTrue(engine.unicode_ok)
            self.assertTrue(Path(report["font_path"]).is_file())
        else:
            self.assertFalse(engine.unicode_ok)
        self.assertGreater(engine.measure("TEMPO", 20)[0], 5)

    def test_a_contact_sheet_holds_the_four_phases_in_order(self):
        from ghostcaddie.demo import render_showcase as rs
        panels = []
        for phase in SHOWCASE_STILL_PHASES:
            image = self.np.full((240, 320, 3), 60, self.np.uint8)
            panels.append((phase, image))
        sheet = rs.contact_sheet(panels, columns=2, gap=10, title="FAIRWAY OS",
                                 subtitle="source frames 0-39",
                                 disclaimer=sc.DEFAULT_DISCLOSURE)
        self.assertGreater(sheet.shape[1], 640)
        self.assertGreater(sheet.shape[0], 480)
        self.assertGreater(sheet.shape[0] * sheet.shape[1], 4 * 240 * 320 * 0.9)


# ---------------------------------------------------------------- end to end


class ShowcaseFixture:
    """Real showcase runs on generated synthetic clips (no source footage)."""

    def run_fixture(self, *, showcase=True, max_width=320, frames=40, stride=1,
                    holds=None, keyframes=None, roi_keyframes=None,
                    metrics=None, extra_render=None, name="out",
                    manual_segments=None, source_description=None):
        require_cv2(self)
        video = make_synthetic_clip(self.tmp, frames=frames, ball_motion="slide",
                                    width=320, height=240, fps=20.0)
        block = {"enabled": True, "max_width": max_width}
        default_keyframes = {"address": 0, "top": max(1, frames // 3),
                            "impact": max(2, 2 * frames // 3), "finish": frames - 1}
        if holds:
            block["holds"] = holds
        if metrics:
            block["metrics"] = metrics
        payload = showcase_payload(**block)
        payload["interval"] = {"start_seconds": 0.0,
                               "end_seconds": round(frames / 20.0, 6),
                               "max_frames": 400}
        payload["corrections"] = {"ball": [{"frame": frames - 2,
                                            "x": 61.0 + frames - 3, "y": 200.0,
                                            "note": "synthetic correction"}]}
        payload["keyframes"] = default_keyframes if keyframes is None else keyframes
        if roi_keyframes is not None:
            payload["roi_keyframes"] = roi_keyframes
        if manual_segments is not None:
            payload["manual_segments"] = manual_segments
        if source_description is not None:
            payload["source_description"] = source_description
        if extra_render:
            payload["render"].update(extra_render)
        payload["render"]["stride"] = stride
        config_path = Path(self.tmp) / f"config_{name}.json"
        config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        out = Path(self.tmp) / name
        from ghostcaddie.demo.pipeline import run_demo
        result = run_demo(str(video), str(config_path), str(out),
                          cv_python=sys.executable, render=True, showcase=showcase)
        return result, out, video, config_path


class ShowcasePipelineTests(ShowcaseFixture, TempDirTestCase):
    def test_the_documented_showcase_artifacts_are_written(self):
        result, out, _video, _config = self.run_fixture()
        for name in ("showcase_annotated.mp4", "showcase_report.md",
                     "showcase_analysis.json", "showcase_provenance.json",
                     "showcase_config.json", "linkedin_post_draft.md",
                     "showcase_contact_sheet.png"):
            self.assertTrue((out / name).is_file(), name)
        for phase in SHOWCASE_STILL_PHASES:
            self.assertTrue((out / f"showcase_{phase}.png").is_file(), phase)
        self.assertFalse((out / "demo_annotated.mp4").exists())
        self.assertFalse((out / "demo_report.md").exists())

    def test_the_video_keeps_the_source_aspect_ratio_with_no_added_width(self):
        result, _out, _video, _config = self.run_fixture()
        picture = result.render["picture_wh"]
        self.assertEqual(result.render["wh"], picture)
        self.assertEqual(result.render["added_width_px"], 0)
        self.assertAlmostEqual(picture[0] / picture[1], 320 / 240, places=2)
        coverage = result.render["chrome_coverage"]
        self.assertIn("not a claim that", coverage["note"])
        self.assertIsNotNone(coverage["drawn_points_total"])

    def test_the_render_is_verified_by_ffprobe_at_the_showcase_size(self):
        result, out, _video, _config = self.run_fixture()
        self.assertTrue(result.render["written"])
        verification = json.loads((out / "showcase_debug" / "verification.json")
                                  .read_text())["render"]["render_ffprobe"]
        self.assertTrue(verification["checked"])
        self.assertTrue(verification["ok"], verification.get("problems"))
        self.assertEqual(verification["codec"], "h264")
        self.assertEqual(verification["pix_fmt"], "yuv420p")
        self.assertTrue(verification["faststart"])
        self.assertEqual(verification["frames"], result.render["frames"])
        self.assertEqual([verification["width"], verification["height"]],
                         result.render["wh"])

    def test_the_stride_is_a_recorded_presentation_pace_change(self):
        result, out, _video, _config = self.run_fixture(stride=3, frames=30)
        playback = result.render["playback"]
        self.assertEqual(playback["stride"], 3)
        self.assertEqual(playback["output_frames"], result.render["frames"])
        self.assertEqual(playback["output_frames"],
                         len(range(result.interval["start_frame"],
                                   result.interval["end_frame"] + 1, 3)))
        self.assertEqual(playback["presentation_speedup"], 3.0)
        self.assertIn("start + i*stride", playback["rule"])
        self.assertIn("PRESENTATION pace only", playback["note"])
        provenance = json.loads((out / "showcase_provenance.json").read_text())
        self.assertEqual(provenance["spec"]["render"]["stride"], 3)
        self.assertEqual(provenance["render"]["playback"]["stride"], 3)
        report = (out / "showcase_report.md").read_text()
        self.assertIn("stride", report.lower())

    def test_a_hold_repeats_one_frame_and_is_recorded_with_its_output_range(self):
        result, out, _video, _config = self.run_fixture(
            stride=1, frames=30, holds=[{"frame": 12, "seconds": 0.5,
                                         "label": "synthetic hold"}])
        playback = result.render["playback"]
        self.assertEqual(len(playback["holds"]), 1)
        record = playback["holds"][0]
        self.assertEqual(record["source_frame"], 12)
        self.assertEqual(record["output_frames_held"], 10)      # 0.5s at 20 fps
        self.assertEqual(result.render["frames"],
                         result.interval["frame_count"] + 9)
        self.assertIn("held_replay", record["status"])
        provenance = json.loads((out / "showcase_provenance.json").read_text())
        self.assertEqual(provenance["spec"]["showcase"]["holds"][0]["frame"], 12)

    def test_a_moving_roi_is_used_per_frame_and_recorded(self):
        result, out, _video, _config = self.run_fixture(
            roi_keyframes=[{"frame": 0, "roi": [80, 40, 160, 180],
                            "note": "synthetic operator box"},
                           {"frame": 39, "roi": [120, 60, 120, 140],
                            "note": "synthetic operator box"}])
        identity = result.body["identity"]
        self.assertEqual(identity["identity_method"], "operator_roi_keyframes")
        self.assertEqual(len(identity["roi_keyframes"]), 2)
        boxes = {record.get("source_frame"): record.get("roi_xywh")
                 for record in result.body["per_frame"]}
        self.assertEqual(boxes[0], [80, 40, 160, 180])
        self.assertEqual(boxes[39], [120, 60, 120, 140])
        provenance = json.loads((out / "showcase_provenance.json").read_text())
        keys = provenance["manual_input_records"]["roi_keyframes"]
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0]["origin"], "operator_crop_geometry")
        self.assertEqual(keys[0]["author"], "synthetic operator box")
        self.assertIn("interpolated", keys[0]["method"])
        self.assertEqual(
            provenance["manual_input_records"]["roi_effective_xywh"],
            [80, 40, 160, 180])

    def test_the_written_config_still_loads_and_binds_to_the_same_frames(self):
        result, out, _video, _config = self.run_fixture()
        written = json.loads((out / "showcase_config.json").read_text())
        self.assertIn("_resolved", written)
        self.assertEqual(written["_resolved"]["source_frames"]["start_frame"],
                         result.interval["start_frame"])
        self.assertEqual(written["_resolved"]["source_frames"]["end_frame"],
                         result.interval["end_frame"])
        reloaded = parse_config(written)
        self.assertTrue(reloaded.showcase.enabled)
        self.assertEqual(reloaded.render.stride, 1)
        self.assertEqual(reloaded.keyframes, result.config["keyframes"])
        probe = probe_source(str(self.tmp / "synthetic.mp4"))
        resolved = validate_config(reloaded, probe)
        self.assertEqual([resolved.start_frame, resolved.end_frame],
                         [result.interval["start_frame"], result.interval["end_frame"]])
        self.assertEqual(load_config(str(out / "showcase_config.json")).keyframes,
                         reloaded.keyframes)

    def test_the_provenance_records_the_style_stills_and_measured_coverage(self):
        result, out, _video, _config = self.run_fixture()
        provenance = json.loads((out / "showcase_provenance.json").read_text())
        presentation = provenance["presentation"]
        self.assertEqual(presentation["style"], "showcase")
        self.assertEqual(presentation["chrome"]["added_width_px"], 0)
        self.assertEqual(presentation["chrome"]["layout_problems"], [])
        self.assertIn("measured_coverage", presentation["chrome"])
        self.assertTrue(presentation["cards"]["video_metrics"])
        roles = {entry["role"] for entry in provenance["outputs"]}
        self.assertIn("annotated", roles)
        self.assertIn("linkedin", roles)
        self.assertIn("config_copy", roles)
        self.assertIn("still:address", roles)
        for entry in provenance["outputs"]:
            self.assertTrue(Path(entry["path"]).is_file(), entry["path"])

    def test_the_stills_carry_at_most_three_metrics_and_the_video_four(self):
        result, out, _video, _config = self.run_fixture()
        keyframes = result.keyframes
        self.assertEqual(len(keyframes["still_metrics"]), 3)
        self.assertEqual(len(keyframes["video_metrics"]), 4)
        for card in keyframes["cards"]:
            self.assertTrue(Path(card["path"]).is_file())
            self.assertLessEqual(len(card["metrics"]), 3)

    def test_a_phase_frame_off_the_stride_grid_still_gets_its_own_still(self):
        """The still must be the operator's frame, not a neighbouring encoded one."""
        result, out, _video, _config = self.run_fixture(
            stride=3, frames=40,
            keyframes={"address": 0, "top": 14, "impact": 28, "finish": 39})
        with_stride = {card["phase"]: card["frame"]
                       for card in result.keyframes["cards"]}
        for phase, frame in with_stride.items():
            self.assertTrue((out / f"showcase_{phase}.png").is_file(), phase)
        stills = result.render["playback"]["extra_stills_from_off_grid_frames"]
        off_grid = [f for f in with_stride.values() if (f % 3) != 0]
        self.assertEqual(stills, len(off_grid))
        self.assertGreaterEqual(stills, 1)

    def test_a_phase_with_no_marker_gets_an_explicit_unavailable_still(self):
        result, out, _video, _config = self.run_fixture(
            keyframes={"address": 0, "finish": 39})
        missing = {item["phase"] for item in result.keyframes["missing"]}
        self.assertIn("top", missing)
        self.assertIn("impact", missing)
        self.assertTrue((out / "showcase_top.png").is_file())
        report = (out / "showcase_report.md").read_text()
        self.assertIn("UNAVAILABLE", report)
        self.assertEqual(len(result.keyframes["contact_sheet_panels"]), 4)

    def test_the_four_metric_cards_are_the_configured_ones(self):
        result, out, _video, _config = self.run_fixture(
            metrics=["shoulder_tilt", "knee_flexion", "hand_rise", "tempo"])
        self.assertEqual(result.presentation["cards"]["video_metrics"],
                         ["shoulder_tilt", "knee_flexion", "hand_rise", "tempo"])
        self.assertEqual(result.presentation["cards"]["still_metrics"],
                         ["shoulder_tilt", "knee_flexion", "hand_rise"])
        self.assertIn("HAND RISE", (out / "showcase_report.md").read_text())

    def test_demo_mode_still_writes_the_demo_contract(self):
        result, out, _video, _config = self.run_fixture(showcase=False)
        self.assertTrue((out / "demo_annotated.mp4").is_file())
        self.assertTrue((out / "demo_report.md").is_file())
        self.assertFalse((out / "showcase_annotated.mp4").exists())
        self.assertEqual(result.presentation["style"], "demo")

    def test_a_manual_anchor_segment_run_is_drawn_recorded_and_never_extrapolated(self):
        """End to end: the opt-in block through the real pipeline, video and JSON."""
        from ghostcaddie.demo.tracking import (MANUAL_HIDDEN_METHOD, ORIGIN_OPERATOR,
                                               STATUS_INTERPOLATED, STATUS_MANUAL)
        result, out, _video, _config = self.run_fixture(
            frames=30, stride=3,
            manual_segments={"ball": {
                "author": "synthetic operator (test)",
                "replace_automatic": False,
                "hide_frames": [[16, 18]],
                "trail": "segments_only",
                "segments": [
                    {"name": "visible_start", "max_anchor_spacing_frames": 4,
                     "anchors": [[0, 60.0, 200.0], [4, 64.0, 200.0]]},
                    {"name": "launch", "max_anchor_spacing_frames": 2,
                     "anchors": [[20, 80.0, 190.0], [22, 86.0, 184.0]]}]}})
        track = result.targets["ball"]
        record = track["manual_segments"]
        self.assertEqual(record["author"], "synthetic operator (test)")
        self.assertEqual(record["anchor_count"], 4)
        self.assertEqual(record["hidden_frames"], [16, 17, 18])
        self.assertEqual(record["trail"], "segments_only")
        points = {int(p["source_frame"]): p for p in track["points"]}
        self.assertEqual(points[0]["status"], STATUS_MANUAL)
        self.assertEqual(points[0]["origin"], ORIGIN_OPERATOR)
        self.assertEqual(points[0]["manual_segment"], "visible_start")
        self.assertIn("synthetic operator (test)", points[0]["note"])
        self.assertEqual(points[2]["status"], STATUS_INTERPOLATED)
        self.assertIn("anchors at source frames 0 and 4", points[2]["note"])
        for frame in (16, 17, 18):
            self.assertEqual(points[frame]["method"], MANUAL_HIDDEN_METHOD)
            self.assertIsNone(points[frame]["x"])
        # between the two segments nothing is invented: those frames are whatever the
        # automatic track measured, or missing, but never part of a manual segment
        for frame in (5, 6, 19, 23):
            self.assertNotIn(points[frame]["manual_segment"],
                             ("visible_start", "launch"))
        report = (out / "showcase_report.md").read_text()
        self.assertIn("manual anchor segments", report)
        self.assertIn("synthetic operator (test)", report)
        provenance = json.loads((out / "showcase_provenance.json").read_text())
        records = provenance["manual_input_records"]["manual_anchor_segments"]
        self.assertEqual(records[0]["target"], "ball")
        self.assertEqual(records[0]["author"], "synthetic operator (test)")
        self.assertEqual(len(records[0]["segments"]), 2)
        self.assertEqual(records[0]["segments"][0]["anchor_count"], 2)
        # the run records the manual segments in its own render record as well
        self.assertEqual(result.render["manual_segments"]["ball"]["anchor_count"], 4)

    def test_showcase_can_be_requested_by_the_flag_without_a_config_block(self):
        require_cv2(self)
        video = make_synthetic_clip(self.tmp, frames=20, ball_motion="slide",
                                    width=320, height=240, fps=20.0)
        payload = showcase_payload()
        payload.pop("showcase")
        payload["interval"] = {"start_seconds": 0.0, "end_seconds": 1.0,
                               "max_frames": 100}
        payload["keyframes"] = {"address": 0, "top": 6, "impact": 13, "finish": 19}
        config_path = Path(self.tmp) / "no_showcase.json"
        config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        out = Path(self.tmp) / "flag_on"
        from ghostcaddie.demo.pipeline import run_demo
        result = run_demo(str(video), str(config_path), str(out),
                          cv_python=sys.executable, render=True, showcase=True)
        self.assertEqual(result.presentation["style"], "showcase")
        self.assertTrue((out / "showcase_annotated.mp4").is_file())
        out_off = Path(self.tmp) / "flag_off"
        off = run_demo(str(video), str(config_path), str(out_off),
                       cv_python=sys.executable, render=True, showcase=False)
        self.assertEqual(off.presentation["style"], "demo")
        self.assertTrue((out_off / "demo_annotated.mp4").is_file())


# ---------------------------------------------------------------- wording


class ShowcaseWordingTests(ShowcaseFixture, TempDirTestCase):
    def test_the_report_states_the_absence_of_a_sidebar_and_the_disclosure(self):
        result, out, _video, _config = self.run_fixture()
        report = (out / "showcase_report.md").read_text()
        self.assertIn("no gutter, no sidebar, no letterbox", report)
        self.assertIn("OPERATOR-ASSISTED", report)
        self.assertIn("metric cards on the video", report)
        self.assertIn("No physical metric", report)
        self.assertIn("operator input", report)

    def test_the_report_and_draft_never_claim_a_physical_quantity(self):
        """Every banned phrase may appear only inside a negated statement.

        File paths are removed before the scan: a temporary directory can contain a
        banned substring by accident (a path segment ending in "...9m" followed by
        "/synthetic.mp4" reads as "m/s"), and a path is not a claim about the swing.
        """
        result, out, _video, _config = self.run_fixture()
        for name in ("showcase_report.md", "linkedin_post_draft.md"):
            text = (out / name).read_text().lower().replace(str(self.tmp).lower(), "")
            for forbidden in BANNED_CLAIMS:
                if forbidden not in text:
                    continue
                for line in text.splitlines():
                    if forbidden not in line or "`/" in line:
                        continue
                    self.assertTrue(
                        any(marker in line for marker in
                            ("no ", "not ", "never", "unavailable", "without")),
                        f"{name}: {forbidden!r} claimed without a negation: {line!r}")

    def test_the_linkedin_draft_is_a_draft_and_claims_no_rights(self):
        result, out, _video, _config = self.run_fixture()
        draft = (out / "linkedin_post_draft.md").read_text()
        self.assertIn("DRAFT", draft)
        self.assertIn("operator review required", draft)
        self.assertIn("No rights to the footage are claimed", draft)
        self.assertIn("never publishes", draft)
        self.assertIn("2D image-space", draft)

    def test_the_playback_map_matches_the_output_frames_and_the_holds(self):
        result, out, _video, _config = self.run_fixture(
            stride=3, frames=30,
            holds=[{"frame": 15, "seconds": 0.5, "label": "a hold"}])
        playback = result.render["playback"]
        mapping = playback["output_to_source_frames"]
        self.assertEqual(len(mapping), result.render["frames"])
        self.assertEqual(len(mapping), playback["output_frames"])
        self.assertEqual(playback["sampled_source_frames"], 10)
        self.assertEqual(mapping[:5], [0, 3, 6, 9, 12])
        hold = playback["holds"][0]
        self.assertEqual(hold["source_frame"], 15)
        positions = [i for i, frame in enumerate(mapping) if frame == 15]
        self.assertEqual(len(positions), hold["output_frames_held"])
        self.assertEqual(positions,
                         list(range(hold["output_frames"][0],
                                    hold["output_frames"][1] + 1)))
        self.assertIn("output_to_source_frames", playback["rule"])
        report = (out / "showcase_report.md").read_text()
        self.assertIn("output-to-source frame map", report)
        self.assertIn("inserted repeat", report)

    def test_the_operator_source_description_is_recorded_verbatim(self):
        description = "a local test clip; not a broadcast of any kind"
        result, out, _video, _config = self.run_fixture(source_description=description)
        self.assertEqual(result.source["description"], description)
        report = (out / "showcase_report.md").read_text()
        self.assertIn(f"source clip: {description}", report)
        draft = (out / "linkedin_post_draft.md").read_text()
        self.assertIn(description, draft)
        provenance = json.loads((out / "showcase_provenance.json").read_text())
        self.assertEqual(provenance["source"]["description"], description)

    def test_the_report_metric_table_covers_the_cards_and_the_placeholder(self):
        result, out, _video, _config = self.run_fixture()
        report = (out / "showcase_report.md").read_text()
        for metric in result.presentation["cards"]["still_metrics"]:
            self.assertIn(sc.METRICS[metric]["label"], report)
        self.assertIn(sc.PLACEHOLDER, report)

    def test_the_report_is_rendered_from_the_result_the_video_came_from(self):
        result, out, _video, _config = self.run_fixture()
        self.assertEqual(render_showcase_report(result),
                         (out / "showcase_report.md").read_text())
        self.assertEqual(render_linkedin_draft(result),
                         (out / "linkedin_post_draft.md").read_text())

    def test_the_draft_humanises_metric_names(self):
        self.assertEqual(humanise("shoulder_tilt"), "shoulder tilt (2D)")
        self.assertEqual(humanise("tempo"), "tempo (backswing:downswing frames)")
        self.assertEqual(humanise("hand_rise"), "hand rise (image space)")
        self.assertEqual(humanise("unknown_metric"), "unknown metric")

    def test_an_empty_result_still_renders_both_documents(self):
        from ghostcaddie.demo.result import DemoResult
        result = DemoResult(
            source={"display_name": "c.mp4", "path": "", "sha256": "a" * 64,
                    "width": 640, "height": 480, "fps": 25.0, "frame_count": 100},
            interval={"start_frame": 0, "end_frame": 24, "frame_count": 25},
            config={"roi_xywh": None, "seeds": {}, "corrections": {}, "keyframes": {}},
            body={"state": "unavailable", "coverage": {}},
            analytics={"body_2d": {"supported": [], "unavailable": []}, "targets": {}},
            phases={"phases": {}, "tempo": {"state": "unavailable", "reason": "none"}},
            targets={}, render={"written": False, "reason": "not requested"},
            keyframes={"cards": [], "missing": [], "contact_sheet": ""},
            presentation={"style": "showcase",
                          "cards": {"video_metrics": list(sc.DEFAULT_METRICS),
                                    "still_metrics": list(sc.DEFAULT_METRICS[:3])},
                          "chrome": {}, "output_paths": {}},
            created_utc="2026-09-22T00:00:00+00:00")
        report = render_showcase_report(result)
        draft = render_linkedin_draft(result)
        self.assertIn("Showcase report", report)
        self.assertIn("unavailable", report)
        self.assertIn("DRAFT", draft)


class ManualAnchorSegmentTests(unittest.TestCase):
    """The opt-in operator-anchored segments: parsing, resolution and provenance."""

    def manual_payload(self, **overrides):
        block = {"segments": [{"name": "takeaway", "max_anchor_spacing_frames": 4,
                               "anchors": [[0, 100.0, 200.0], [4, 140.0, 190.0]]}]}
        block.update(overrides)
        return {"clubhead": block}

    def test_a_segment_is_anchors_plus_bounded_interpolation_and_nothing_else(self):
        from ghostcaddie.demo.tracking import (CLASS_AUTOMATIC, CLASS_MANUAL_SEGMENT,
                                               ORIGIN_DERIVED, ORIGIN_OPERATOR,
                                               STATUS_INTERPOLATED, STATUS_MANUAL,
                                               STATUS_MISSING, TargetTrack, TrackPoint,
                                               apply_manual_segments, point_class)
        frames = list(range(0, 11))
        track = TargetTrack(name="clubhead", mode="template", direction="forward",
                            available=False,
                            points=[TrackPoint(f, STATUS_MISSING, ORIGIN_DERIVED,
                                               note="not measured") for f in frames])
        apply_manual_segments(track, manual=self.manual_payload()["clubhead"],
                              interval_frames=frames)
        by_frame = {p.source_frame: p for p in track.points}
        for frame in (0, 4):
            self.assertEqual(by_frame[frame].status, STATUS_MANUAL)
            self.assertEqual(by_frame[frame].origin, ORIGIN_OPERATOR)
            self.assertIn("takeaway", by_frame[frame].note)
        for frame in (1, 2, 3):
            point = by_frame[frame]
            self.assertEqual(point.status, STATUS_INTERPOLATED)
            self.assertEqual(point.origin, ORIGIN_DERIVED)
            self.assertIn("anchors at source frames 0 and 4", point.note)
            self.assertIn("not an observation", point.note)
            self.assertEqual(point_class(point), CLASS_MANUAL_SEGMENT)
        # no extrapolation either side of the segment
        for frame in (5, 6, 10):
            self.assertEqual(by_frame[frame].status, STATUS_MISSING)
            self.assertIsNone(by_frame[frame].x)
            self.assertEqual(point_class(by_frame[frame]), "empty")
        self.assertTrue(track.available)
        self.assertEqual(track.coverage["operator_anchor_frames"], 2)
        self.assertEqual(track.coverage["interpolated_frames"], 3)
        self.assertEqual(track.manual_segments["anchor_count"], 2)
        self.assertEqual(track.manual_segments["segments"][0]["interpolated_frames"], 3)
        self.assertNotIn(CLASS_AUTOMATIC, {point_class(p) for p in track.points})

    def test_the_record_keeps_the_anchors_the_author_and_the_rule(self):
        from ghostcaddie.demo.tracking import (ORIGIN_DERIVED, STATUS_MISSING,
                                               TargetTrack, TrackPoint,
                                               apply_manual_segments)
        payload = self.manual_payload(author="PRIMARY visual inspection")["clubhead"]
        track = TargetTrack(name="clubhead", mode="template", direction="forward",
                            available=False,
                            points=[TrackPoint(f, STATUS_MISSING, ORIGIN_DERIVED,
                                               note="") for f in range(0, 6)])
        apply_manual_segments(track, manual=payload, interval_frames=list(range(0, 6)))
        record = track.manual_segments
        self.assertEqual(record["origin"], "operator")
        self.assertEqual(record["author"], "PRIMARY visual inspection")
        self.assertEqual(record["segments"][0]["anchor_frames"], [0, 4])
        self.assertIn("interpolated", record["rule"])
        self.assertIn("PRIMARY visual inspection", track.points[0].note)

    def test_hidden_frames_are_emptied_and_never_interpolated_across(self):
        from ghostcaddie.demo.tracking import (MANUAL_HIDDEN_METHOD, ORIGIN_DERIVED,
                                               STATUS_MISSING, TargetTrack, TrackPoint,
                                               apply_manual_segments)
        payload = self.manual_payload(
            segments=[{"name": "launch", "max_anchor_spacing_frames": 2,
                       "anchors": [[0, 10.0, 10.0], [2, 20.0, 12.0], [4, 30.0, 14.0]]}],
            hide_frames=[[2, 2]])["clubhead"]
        frames = list(range(0, 6))
        track = TargetTrack(name="clubhead", mode="template", direction="forward",
                            available=False,
                            points=[TrackPoint(f, STATUS_MISSING, ORIGIN_DERIVED,
                                               note="") for f in frames])
        apply_manual_segments(track, manual=payload, interval_frames=frames)
        by_frame = {p.source_frame: p for p in track.points}
        self.assertEqual(by_frame[2].status, STATUS_MISSING)
        self.assertEqual(by_frame[2].method, MANUAL_HIDDEN_METHOD)
        self.assertEqual(track.manual_segments["hidden_frames"], [2])
        # the hidden frame is not a bridge: 1 and 3 are in different runs
        self.assertNotEqual(by_frame[1].segment, by_frame[3].segment)

    def test_replace_automatic_drops_the_automatic_points_and_says_how_many(self):
        from ghostcaddie.demo.tracking import (ORIGIN_AUTOMATIC, STATUS_OBSERVED,
                                               TargetTrack, TrackPoint,
                                               apply_manual_segments, point_class)
        frames = list(range(0, 6))
        points = [TrackPoint(f, STATUS_OBSERVED, ORIGIN_AUTOMATIC, x=float(f), y=1.0,
                             score=0.9, note="") for f in frames]
        track = TargetTrack(name="clubhead", mode="template", direction="forward",
                            available=True, points=points)
        payload = self.manual_payload(replace_automatic=True,
                                      segments=[{"name": "downswing",
                                                 "max_anchor_spacing_frames": 2,
                                                 "anchors": [[0, 5.0, 5.0],
                                                             [2, 6.0, 6.0]]}])["clubhead"]
        apply_manual_segments(track, manual=payload, interval_frames=frames)
        by_frame = {p.source_frame: p for p in track.points}
        self.assertEqual(by_frame[2].manual_segment, "downswing")
        for frame in (3, 4, 5):
            self.assertEqual(point_class(by_frame[frame]), "empty")
            self.assertIsNone(by_frame[frame].x)
        record = track.manual_segments
        self.assertEqual(record["track_before_manual_segments"]["automatic_observed_frames"],
                         6)
        self.assertEqual(
            record["replaced_automatic_track"]["automatic_observed_frames_dropped"], 6)
        self.assertTrue(any("REPLACED" in note for note in track.limitations))

    def test_a_trail_never_crosses_between_a_manual_segment_and_an_automatic_point(self):
        from ghostcaddie.demo.tracking import (CLASS_AUTOMATIC, CLASS_MANUAL_SEGMENT,
                                               ORIGIN_AUTOMATIC, STATUS_OBSERVED,
                                               TargetTrack, TrackPoint,
                                               apply_manual_segments, point_class)
        frames = list(range(0, 8))
        points = [TrackPoint(f, STATUS_OBSERVED, ORIGIN_AUTOMATIC, x=float(f), y=1.0,
                             score=0.9, note=f"automatic {f}") for f in frames]
        track = TargetTrack(name="ball", mode="template", direction="forward",
                            available=True, points=points)
        payload = {"segments": [{"name": "ground_contact",
                                 "max_anchor_spacing_frames": 1,
                                 "anchors": [[5, 50.0, 50.0]]}],
                   "trail": "segments_only"}
        apply_manual_segments(track, manual=payload, interval_frames=frames)
        by_frame = {p.source_frame: p for p in track.points}
        self.assertEqual(point_class(by_frame[4]), CLASS_AUTOMATIC)
        self.assertEqual(point_class(by_frame[5]), CLASS_MANUAL_SEGMENT)
        self.assertNotEqual(by_frame[4].segment, by_frame[5].segment,
                            "the operator's anchor must not continue the automatic run")

    def test_trail_segments_only_keeps_the_trail_inside_the_manual_segments(self):
        from ghostcaddie.demo.tracking import (ORIGIN_AUTOMATIC, STATUS_OBSERVED,
                                               TargetTrack, TrackPoint,
                                               apply_manual_segments,
                                               manual_segments_only)
        frames = list(range(0, 8))
        points = [TrackPoint(f, STATUS_OBSERVED, ORIGIN_AUTOMATIC, x=float(f), y=1.0,
                             score=0.9, note="") for f in frames]
        track = TargetTrack(name="ball", mode="template", direction="forward",
                            available=True, points=points)
        payload = {"segments": [{"name": "launch", "max_anchor_spacing_frames": 2,
                                 "anchors": [[5, 5.0, 5.0], [7, 7.0, 7.0]]}],
                   "trail": "segments_only"}
        apply_manual_segments(track, manual=payload, interval_frames=frames)
        window = [p for p in track.points if p.drawn]
        kept = manual_segments_only(track, window)
        self.assertEqual([p.source_frame for p in kept], [5, 6, 7])
        # and without the option the whole window is kept, unchanged behaviour
        track.manual_segments["trail"] = "all"
        self.assertEqual(len(manual_segments_only(track, window)), len(window))

    def test_the_trail_bound_withholds_the_trail_but_not_the_marker(self):
        from ghostcaddie.demo.tracking import (ORIGIN_AUTOMATIC, STATUS_OBSERVED,
                                               TargetTrack, TrackPoint,
                                               apply_manual_segments,
                                               manual_segments_only)
        frames = list(range(0, 9))
        points = [TrackPoint(f, STATUS_OBSERVED, ORIGIN_AUTOMATIC, x=float(f), y=1.0,
                             score=0.9, note="") for f in frames]
        track = TargetTrack(name="ball", mode="template", direction="forward",
                            available=True, points=points)
        payload = {"segments": [{"name": "launch", "max_anchor_spacing_frames": 2,
                                 "anchors": [[5, 5.0, 5.0], [7, 7.0, 7.0]]}],
                   "trail": "all", "trail_from_frame": 6}
        apply_manual_segments(track, manual=payload, interval_frames=frames)
        window = [p for p in track.points if p.drawn]
        self.assertEqual([p.source_frame for p in manual_segments_only(track, window)],
                         [6, 7, 8])
        # the marker at frame 3 is still drawn: this is a trail bound, not a hide
        self.assertTrue({p.source_frame: p for p in track.points}[3].drawn)
        self.assertEqual(track.manual_segments["trail_from_frame"], 6)


class ManualAnchorConfigTests(unittest.TestCase):
    def payload(self, block):
        payload = showcase_payload(enabled=True)
        payload["manual_segments"] = {"clubhead": block}
        return payload

    def test_a_gap_above_the_spacing_bound_is_refused_not_interpolated(self):
        with self.assertRaises(DemoConfigError) as caught:
            parse_config(self.payload({"segments": [
                {"name": "s", "max_anchor_spacing_frames": 4,
                 "anchors": [[0, 10.0, 10.0], [9, 20.0, 20.0]]}]}))
        self.assertIn("above max_anchor_spacing_frames", str(caught.exception))

    def test_misordered_duplicate_or_overlapping_input_is_refused(self):
        for block in (
                {"segments": [{"name": "s", "max_anchor_spacing_frames": 40,
                               "anchors": [[5, 1.0, 1.0], [1, 2.0, 2.0]]}]},
                {"segments": [{"name": "s", "max_anchor_spacing_frames": 40,
                               "anchors": [[5, 1.0, 1.0], [5, 2.0, 2.0]]}]},
                {"segments": [{"name": "a", "max_anchor_spacing_frames": 40,
                               "anchors": [[0, 1.0, 1.0], [8, 2.0, 2.0]]},
                              {"name": "b", "max_anchor_spacing_frames": 40,
                               "anchors": [[4, 3.0, 3.0]]}]},
                {"segments": [{"name": "a", "max_anchor_spacing_frames": 40,
                               "anchors": [[4, 1.0, 1.0]]}],
                 "hide_frames": [[4, 6]]},
                {"segments": []},
                {"segments": [{"name": "a", "max_anchor_spacing_frames": 40,
                               "anchors": [[4, 1.0, 1.0]]}], "trail": "sometimes"}):
            with self.assertRaises(DemoConfigError):
                parse_config(self.payload(block))

    def test_replace_automatic_needs_a_segment_and_an_unknown_field_is_refused(self):
        with self.assertRaises(DemoConfigError):
            parse_config(self.payload({"replace_automatic": True, "segments": []}))
        with self.assertRaises(DemoConfigError) as caught:
            parse_config(self.payload({"segments": [], "segements": []}))
        self.assertIn("unknown field", str(caught.exception))

    def test_an_anchor_outside_the_interval_or_the_frame_is_refused_at_validation(self):
        config = parse_config(self.payload({"segments": [
            {"name": "s", "max_anchor_spacing_frames": 40,
             "anchors": [[900, 1.0, 1.0]]}]}))
        with self.assertRaises(DemoConfigError) as caught:
            validate_config(config, synthetic_probe(width=320, height=240))
        self.assertIn("outside the processed interval", str(caught.exception))

        config = parse_config(self.payload({"segments": [
            {"name": "s", "max_anchor_spacing_frames": 40,
             "anchors": [[0, 900.0, 2.0]]}]}))
        with self.assertRaises(DemoConfigError) as caught:
            validate_config(config, synthetic_probe(width=320, height=240))
        self.assertIn("outside the 320px-wide frame", str(caught.exception))

    def test_the_trail_bound_must_lie_inside_a_segment(self):
        with self.assertRaises(DemoConfigError) as caught:
            parse_config(self.payload({"segments": [
                {"name": "s", "max_anchor_spacing_frames": 4,
                 "anchors": [[10, 1.0, 1.0], [14, 2.0, 2.0]]}],
                "trail_from_frame": 5}))
        self.assertIn("not inside any of this target's segments",
                      str(caught.exception))
        config = parse_config(self.payload({"segments": [
            {"name": "s", "max_anchor_spacing_frames": 4,
             "anchors": [[10, 1.0, 1.0], [14, 2.0, 2.0]]}],
            "trail_from_frame": 12}))
        self.assertEqual(config.manual_segments["clubhead"].trail_from_frame, 12)
        self.assertEqual(config.to_dict()["manual_segments"]["clubhead"]
                         ["trail_from_frame"], 12)

    def test_the_anchor_list_form_is_accepted_and_normalised(self):
        config = parse_config(self.payload({"author": "PRIMARY",
                                            "segments": [
                                                {"name": "s",
                                                 "max_anchor_spacing_frames": 4,
                                                 "anchors": [[0, 1.0, 2.0],
                                                             [4, 3.0, 4.0,
                                                              "operator note"]]}]}))
        segment = config.manual_segments["clubhead"].segments[0]
        self.assertEqual([a.frame for a in segment.anchors], [0, 4])
        self.assertEqual(segment.anchors[1].note, "operator note")
        self.assertEqual(config.manual_segments["clubhead"].author, "PRIMARY")
        self.assertEqual(config.to_dict()["manual_segments"]["clubhead"]["anchor_count"], 2)


class CardStabilityTests(unittest.TestCase):
    def test_a_measured_value_is_never_hidden_behind_a_held_one(self):
        values = {frame: None for frame in range(0, 40)}
        for frame in range(10, 25):
            values[frame] = 100.0 + frame        # 110..124, a long available run
        values[26] = 999.0                       # one frame of value inside the gap
        values[35] = 777.0                       # a one-frame blip after the gap
        series = sc.debounce_series(values, min_run=4)
        # the first run is adopted as it is
        self.assertEqual(series[0]["state"], "measured")
        self.assertIsNone(series[0]["value"])
        # a long run is shown as measured
        self.assertEqual(series[10]["value"], 110.0)
        self.assertEqual(series[24]["state"], "measured")
        # one frame after it there is no measurement: the card holds the last one
        self.assertEqual(series[25]["state"], "held")
        self.assertEqual(series[25]["value"], values[24])
        self.assertEqual(series[25]["held_from_frame"], 24)
        # a frame whose own measurement exists is shown as measured, not held ...
        self.assertFalse(series[26]["held"])
        self.assertEqual(series[26]["value"], 999.0)
        # ... and the gap after it is held for at most min_run - 1 frames
        self.assertEqual([f for f in range(27, 34) if series[f]["held"]],
                         [27, 28, 29])
        self.assertEqual(series[29]["value"], 999.0)
        self.assertEqual(series[30]["state"], "measured")
        self.assertIsNone(series[30]["value"])
        # a one-frame value while the card shows the placeholder never flashes
        self.assertEqual(series[35]["state"], "held_unavailable")
        self.assertTrue(series[35]["held"])
        self.assertIsNone(series[35]["value"])
        record = sc.debounce_record(series, min_run=4)
        self.assertEqual(record["min_run_frames"], 4)
        self.assertEqual(record["hold_cap_frames"], 3)
        self.assertEqual(record["longest_held_streak_frames"], 3)
        self.assertLessEqual(record["longest_held_streak_frames"],
                             record["hold_cap_frames"])
        self.assertEqual(record["held_placeholder_frames"], [35])
        self.assertIn("unchanged", record["note"])
        self.assertIn("offline", record["filter"])

    def test_the_hold_bound_is_exact_on_a_seven_and_an_eight_frame_gap(self):
        short_gap = {frame: 10.0 for frame in range(0, 30)}
        for frame in range(20, 27):              # a 7-frame gap
            short_gap[frame] = None
        series = sc.debounce_series(short_gap, min_run=8)
        self.assertTrue(all(series[frame]["held"] for frame in range(20, 27)))
        self.assertEqual(series[26]["value"], 10.0)
        record = sc.debounce_record(series, min_run=8)
        self.assertEqual(record["hold_cap_frames"], 7)
        self.assertEqual(record["longest_held_streak_frames"], 7)

        long_gap = {frame: 10.0 for frame in range(0, 30)}
        for frame in range(20, 28):              # an 8-frame gap
            long_gap[frame] = None
        series = sc.debounce_series(long_gap, min_run=8)
        self.assertTrue(all(series[frame]["held"] for frame in range(20, 27)))
        self.assertFalse(series[27]["held"])
        self.assertIsNone(series[27]["value"])
        record = sc.debounce_record(series, min_run=8)
        self.assertEqual(record["longest_held_streak_frames"], 7)

    def test_a_reappearing_value_is_adopted_only_once_it_has_held(self):
        values = {frame: None for frame in range(0, 40)}
        values[20] = 5.0                          # a blip after a long gap
        for frame in range(30, 40):
            values[frame] = 7.0                   # a run that does hold
        series = sc.debounce_series(values, min_run=8)
        self.assertEqual(series[20]["state"], "held_unavailable")
        self.assertIsNone(series[20]["value"])
        # the card is already showing the placeholder: a gap frame is simply its own
        # state, not a "held" one
        self.assertEqual(series[29]["state"], "measured")
        self.assertIsNone(series[29]["value"])
        self.assertEqual(series[30]["state"], "measured")
        self.assertEqual(series[30]["value"], 7.0)

    def test_a_card_only_ever_shows_a_value_that_was_measured(self):
        values = {0: 5.0, 1: 5.0, 2: 5.0, 3: None, 4: 5.0, 5: 5.0, 6: 5.0}
        series = sc.debounce_series(values, min_run=3)
        measured = {value for value in values.values() if value is not None}
        for entry in series.values():
            if entry["value"] is not None:
                self.assertIn(entry["value"], measured)
        record = sc.debounce_record(series, min_run=3)
        for entry in record["held_value_sources"]:
            self.assertEqual(series[entry["frame"]]["value"], entry["value"])
            self.assertEqual(values[entry["from_frame"]], entry["value"])

    def test_a_negative_zero_pixel_value_prints_as_zero(self):
        self.assertEqual(sc.format_value("hand_rise", -0.4), "0 px")
        self.assertEqual(sc.format_value("sway", -0.0), "0 px")
        self.assertEqual(sc.format_value("hand_rise", 0.4), "0 px")
        self.assertEqual(sc.format_value("hand_rise", -12.4), "-12 px")
        self.assertEqual(sc.format_value("hand_rise", 412.4), "412 px")

    def test_the_impact_accent_span_is_configurable_and_still_bounded(self):
        self.assertEqual(sc.impact_accent(50, 50, half_span=1), 1.0)
        self.assertEqual(sc.impact_accent(52, 50, half_span=1), 0.0)
        self.assertGreater(sc.impact_accent(52, 50, half_span=6), 0.0)
        self.assertEqual(sc.impact_accent(57, 50, half_span=6), 0.0)
        config = parse_config(showcase_payload(enabled=True,
                                               impact_accent_span_frames=18,
                                               card_debounce_frames=12))
        self.assertEqual(config.showcase.impact_accent_span_frames, 18)
        self.assertEqual(config.showcase.card_debounce_frames, 12)
        with self.assertRaises(DemoConfigError):
            parse_config(showcase_payload(enabled=True, impact_accent_span_frames=0))


class ManualAnchorPresentationTests(unittest.TestCase):
    """The contact sheet, the still clearance and the hollow ball ring."""

    def setUp(self):
        require_cv2(self)
        import numpy as np
        self.np = np

    def test_the_contact_sheet_header_keeps_its_text_inside_the_band(self):
        from ghostcaddie.demo import render_showcase as rs
        # A panel colour that text anti-aliasing cannot produce: blending the sheet
        # background into the text colours keeps R and G close together, and this one
        # is far from that diagonal.
        panel_color = (10, 200, 90)
        panel = self.np.full((240, 320, 3), panel_color, self.np.uint8)
        panels = [(phase, panel) for phase in SHOWCASE_STILL_PHASES]
        subtitle = ("source frames 48-1176   ADDRESS f48 TOP f570 IMPACT f840 "
                    "FINISH f1128")
        sheet = rs.contact_sheet(panels, columns=2, gap=10,
                                 title="FAIRWAY OS   SWING ANALYSIS",
                                 subtitle=subtitle,
                                 disclaimer=sc.DEFAULT_DISCLOSURE).astype(int)
        panel_pixels = self.np.abs(sheet - self.np.array(panel_color)).max(axis=2) <= 2
        first_panel_row = int(self.np.where(panel_pixels.any(axis=1))[0].min())
        background = self.np.array(rs.SHEET_BG)
        ink = ((self.np.abs(sheet - background).max(axis=2) > 12)
               & ~panel_pixels)
        ink_rows = self.np.where(ink[:first_panel_row].any(axis=1))[0]
        self.assertGreater(ink_rows.size, 0, "the header text was not drawn at all")
        self.assertLessEqual(int(ink_rows.max()), first_panel_row - 4,
                             "the header text runs into the panels")
        # the footer disclosure is inside the sheet, below the panels
        panel_last_row = int(self.np.where(panel_pixels.any(axis=1))[0].max())
        self.assertGreater(int(self.np.where(ink.any(axis=1))[0].max()),
                           panel_last_row)

    def test_a_contact_sheet_without_a_subtitle_is_smaller_not_broken(self):
        from ghostcaddie.demo import render_showcase as rs
        panel = self.np.full((240, 320, 3), 60, self.np.uint8)
        panels = [(phase, panel) for phase in SHOWCASE_STILL_PHASES]
        with_sub = rs.contact_sheet(panels, columns=2, gap=10, title="FAIRWAY OS",
                                    subtitle="a sub-line")
        without = rs.contact_sheet(panels, columns=2, gap=10, title="FAIRWAY OS")
        self.assertLess(without.shape[0], with_sub.shape[0])
        self.assertEqual(without.shape[1], with_sub.shape[1])

    def test_the_still_card_row_honours_the_same_clearance_as_the_video(self):
        capped = sc.plan_layout(1920, 1080, sc.DEFAULT_METRICS[:3],
                                max_row_width=1450)
        self.assertLessEqual(capped.cards[-1].right(), 1450)
        full = sc.plan_layout(1920, 1080, sc.DEFAULT_METRICS[:3])
        self.assertGreater(full.cards[-1].right(), 1450)

    def test_the_ball_marker_is_hollow_so_the_ball_reads_through_it(self):
        from ghostcaddie.demo import render_showcase as rs
        layer, mask = rs.new_layers(200, 200)
        rs.draw_ball_marker(layer, mask, {"x": 100.0, "y": 100.0,
                                          "status": "manual_anchor"}, 1.0)
        self.assertEqual(float(mask[100, 100]), 0.0,
                         "the centre of the marker must stay clear")
        ring = mask > 0
        self.assertTrue(ring.any(), "the ring was not drawn")

    def test_a_held_card_is_drawn_dimmer_than_a_measured_one(self):
        from ghostcaddie.demo import render_showcase as rs
        picture = self.np.full((360, 640, 3), 70, self.np.uint8)
        layout = sc.plan_layout(640, 360, sc.DEFAULT_METRICS)
        card = {"metric": "hand_rise", "label": "HAND RISE", "value": "412 px",
                "sub": "", "available": True, "held": True}
        measured = dict(card, held=False)
        chrome = {"brand": "FAIRWAY OS", "brand_sub": "SWING ANALYSIS",
                  "phase": "TOP", "disclosure": sc.DEFAULT_DISCLOSURE}
        held_image = rs.compose_frame(
            picture.copy(), layout=layout,
            chrome=dict(chrome, cards=[card] * 4))[0]
        measured_image = rs.compose_frame(
            picture.copy(), layout=layout,
            chrome=dict(chrome, cards=[measured] * 4))[0]
        region = (slice(layout.cards[0].y, layout.cards[0].bottom()),
                  slice(layout.cards[0].x, layout.cards[0].right()))
        self.assertLess(float(held_image[region].mean()),
                        float(measured_image[region].mean()))



if __name__ == "__main__":
    unittest.main()
