"""Config contract tests for demo mode.

The demo config carries every operator decision, so the validation rules are the
product's front door: a fractional frame, an out-of-bounds point, a duplicated
correction or a silently truncated interval would each put something on the video
that nobody asked for. Every rejection asserted here is deliberate.
"""
from __future__ import annotations

import json
import math
import re
import unittest
from pathlib import Path

from ghostcaddie.demo import DEMO_CONFIG_SCHEMA
from ghostcaddie.demo.config import (PHASE_ORDER, load_config, parse_config,
                                     validate_config)
from ghostcaddie.demo.errors import DemoConfigError
from tests.demo_fixtures import TempDirTestCase, base_config, synthetic_probe, write_config


class SchemaTests(unittest.TestCase):
    def test_the_demo_config_schema_is_not_the_canonical_analysis_schema(self):
        self.assertEqual(DEMO_CONFIG_SCHEMA, "fairwayos-demo-config.v1")
        self.assertNotIn("analysis", DEMO_CONFIG_SCHEMA)

    def test_a_foreign_schema_version_is_refused(self):
        with self.assertRaises(DemoConfigError) as caught:
            parse_config(base_config(schema_version="fairwayos-analysis.v1"))
        self.assertIn("fairwayos-demo-config.v1", str(caught.exception))

    def test_an_unknown_top_level_field_is_refused_rather_than_ignored(self):
        with self.assertRaises(DemoConfigError) as caught:
            parse_config(base_config(ball_seed={"frame": 1, "x": 2, "y": 3}))
        self.assertIn("unknown field", str(caught.exception))

    def test_a_key_beginning_with_an_underscore_is_a_comment_not_input(self):
        payload = base_config()
        payload["_comment"] = "explains the file"
        payload["interval"]["_note"] = "0.8s in, 7.44s out"
        payload["targets"]["ball"]["_why"] = "tight search, the ball is small here"
        config = parse_config(payload)          # must not raise
        self.assertEqual(config.roi, [200, 80, 240, 320])

    def test_a_misspelt_field_is_still_refused_next_to_a_comment(self):
        payload = base_config()
        payload["_comment"] = "fine"
        payload["interval"]["start_second"] = 0.0
        with self.assertRaises(DemoConfigError) as caught:
            parse_config(payload)
        self.assertIn("unknown field", str(caught.exception))

    def test_the_shipped_template_parses_and_validates(self):
        template = Path(__file__).resolve().parents[1] / "examples" / (
            "demo_config.template.json")
        self.assertTrue(template.is_file(), template)
        parsed = parse_config(json.loads(template.read_text(encoding="utf-8")))
        self.assertEqual(parsed.schema_version, DEMO_CONFIG_SCHEMA)
        probe = synthetic_probe(width=2560, height=1440, fps=25.0, frame_count=265)
        resolved = validate_config(parsed, probe)
        self.assertEqual((resolved.start_frame, resolved.end_frame), (20, 185))
        self.assertEqual(resolved.config.roi, [640, 60, 760, 1080])
        self.assertIn("illustrative", resolved.config.notes.lower())

    def test_the_documented_config_snippet_parses_and_validates(self):
        """The JSON block in docs/demo-mode.md is shipped output too: parse it."""
        doc = Path(__file__).resolve().parents[1] / "docs" / "demo-mode.md"
        self.assertTrue(doc.is_file(), doc)
        text = doc.read_text(encoding="utf-8")
        blocks = re.findall(r"```json\n(.*?)```", text, flags=re.S)
        self.assertTrue(blocks, "docs/demo-mode.md has no ```json block")
        probe = synthetic_probe(width=2560, height=1440, fps=25.0, frame_count=265)
        for block in blocks:
            parsed = parse_config(json.loads(block))
            validate_config(parsed, probe)
        # the key table must name the keys the handler actually uses
        from ghostcaddie.demo.picker import PHASE_KEYS_UPPER
        for key in PHASE_KEYS_UPPER:
            self.assertIn(f"`{key}`", text, key)
        for unsupported in ("source_timing_caveat", "demo_keyframes.png", "scale: null"):
            self.assertNotIn(unsupported, text)
        self.assertIn("contact_sheet.png", text)

    def test_an_automatic_until_frame_must_be_inside_the_interval_and_after_the_seed(self):
        payload = base_config()
        payload["targets"]["ball"]["automatic_until_frame"] = 500
        with self.assertRaises(DemoConfigError) as caught:
            validate_config(parse_config(payload), synthetic_probe(frame_count=200))
        self.assertIn("outside the processed interval", str(caught.exception))
        payload = base_config()
        payload["targets"]["ball"]["automatic_until_frame"] = 3   # seed is frame 10
        with self.assertRaises(DemoConfigError) as caught:
            validate_config(parse_config(payload), synthetic_probe(frame_count=200))
        self.assertIn("before its seed frame", str(caught.exception))

    def test_an_automatic_until_frame_past_the_track_stop_is_refused(self):
        payload = base_config()
        payload["targets"]["ball"]["track_until_frame"] = 50
        payload["targets"]["ball"]["automatic_until_frame"] = 60
        with self.assertRaises(DemoConfigError) as caught:
            validate_config(parse_config(payload), synthetic_probe(frame_count=200))
        self.assertIn("contradictory", str(caught.exception))

    def test_a_config_from_another_clip_is_warned_about_not_run_silently(self):
        payload = base_config(clip_path_hint="some_other_clip.mp4")
        resolved = validate_config(parse_config(payload),
                                   synthetic_probe(frame_count=200))
        self.assertTrue(any("some_other_clip.mp4" in w and "synthetic.mp4" in w
                            for w in resolved.warnings), resolved.warnings)
        # persisted, not only printed
        self.assertIn("clip_path_hint", json.dumps(resolved.effective()))
        matching = base_config(clip_path_hint="synthetic.mp4")
        resolved = validate_config(parse_config(matching),
                                   synthetic_probe(frame_count=200))
        self.assertFalse([w for w in resolved.warnings if "clip_path_hint" in w])

    def test_an_unknown_nested_field_is_refused(self):
        payload = base_config()
        payload["targets"]["ball"]["colour"] = "yellow"
        with self.assertRaises(DemoConfigError):
            parse_config(payload)

    def test_the_documented_phase_names_are_the_only_phase_names(self):
        self.assertEqual(PHASE_ORDER, ("address", "takeaway", "top", "downswing",
                                       "impact", "follow_through", "finish"))
        with self.assertRaises(DemoConfigError):
            parse_config(base_config(keyframes={"launch": 5}))

    def test_an_unknown_target_is_refused(self):
        with self.assertRaises(DemoConfigError):
            parse_config(base_config(seeds={"putter": {"frame": 1, "x": 1, "y": 1}}))


class StrictValueTests(unittest.TestCase):
    def test_a_fractional_frame_index_is_refused_not_rounded(self):
        with self.assertRaises(DemoConfigError) as caught:
            parse_config(base_config(seeds={"ball": {"frame": 10.5, "x": 1, "y": 1}}))
        self.assertIn("whole number", str(caught.exception))
        self.assertIn("not rounded", str(caught.exception))

    def test_an_integral_float_frame_is_accepted_and_normalised(self):
        cfg = parse_config(base_config(seeds={"ball": {"frame": 10.0, "x": 1, "y": 1}}))
        self.assertIsInstance(cfg.seeds["ball"].frame, int)
        self.assertEqual(cfg.seeds["ball"].frame, 10)

    def test_a_non_finite_coordinate_is_refused(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(DemoConfigError):
                parse_config(base_config(seeds={"ball": {"frame": 3, "x": bad, "y": 1}}))

    def test_a_boolean_is_not_a_number(self):
        with self.assertRaises(DemoConfigError):
            parse_config(base_config(interval={"start_seconds": True}))

    def test_a_string_frame_is_refused(self):
        with self.assertRaises(DemoConfigError):
            parse_config(base_config(keyframes={"top": "60"}))

    def test_an_out_of_range_min_score_is_refused(self):
        payload = base_config()
        payload["targets"]["ball"]["min_score"] = 1.4
        with self.assertRaises(DemoConfigError):
            parse_config(payload)

    def test_an_even_template_size_is_refused_because_the_patch_could_not_centre(self):
        payload = base_config()
        payload["targets"]["ball"]["template_size"] = 20
        with self.assertRaises(DemoConfigError) as caught:
            parse_config(payload)
        self.assertIn("odd", str(caught.exception))

    def test_an_unknown_tracker_mode_or_direction_is_refused(self):
        payload = base_config()
        payload["targets"]["ball"]["mode"] = "magic"
        with self.assertRaises(DemoConfigError):
            parse_config(payload)
        payload = base_config()
        payload["targets"]["ball"]["direction"] = "sideways"
        with self.assertRaises(DemoConfigError):
            parse_config(payload)


class IntervalTests(TempDirTestCase):
    def test_endpoint_semantics_are_half_open_and_frame_indices_are_inclusive(self):
        cfg = parse_config(base_config(
            interval={"start_seconds": 1.0, "end_seconds": 2.0, "max_frames": 100},
            seeds={}, keyframes={}, corrections={}))
        resolved = validate_config(cfg, synthetic_probe(fps=25.0))
        # 1.0s -> frame 25 (inclusive); 2.0s is EXCLUSIVE -> last frame is 49.
        self.assertEqual((resolved.start_frame, resolved.end_frame), (25, 49))
        self.assertEqual(resolved.frame_count, 25)
        self.assertEqual(list(resolved.frames), list(range(25, 50)))
        self.assertIn("end_seconds exclusive", resolved.effective()["endpoint_semantics"])

    def test_a_nonzero_start_shifts_every_record_into_source_frames(self):
        cfg = parse_config(base_config(
            interval={"start_seconds": 2.0, "end_seconds": 3.0, "max_frames": 100},
            seeds={"ball": {"frame": 60, "x": 320, "y": 300}},
            keyframes={"address": 60, "top": 62, "impact": 64, "finish": 66},
            corrections={"ball": [{"frame": 70, "x": 10, "y": 10}]}))
        resolved = validate_config(cfg, synthetic_probe(fps=25.0))
        self.assertEqual(resolved.start_frame, 50)
        self.assertEqual(resolved.end_frame, 74)

    def test_an_end_past_the_source_duration_is_refused_rather_than_truncated(self):
        cfg = parse_config(base_config(
            interval={"start_seconds": 0.0, "end_seconds": 30.0, "max_frames": 2000}))
        with self.assertRaises(DemoConfigError) as caught:
            validate_config(cfg, synthetic_probe(duration=8.0))
        self.assertIn("past the source duration", str(caught.exception))
        self.assertIn("refused rather than truncated", str(caught.exception))

    def test_an_interval_above_max_frames_is_refused_rather_than_truncated(self):
        cfg = parse_config(base_config(
            interval={"start_seconds": 0.0, "end_seconds": 4.0, "max_frames": 10}))
        with self.assertRaises(DemoConfigError) as caught:
            validate_config(cfg, synthetic_probe(fps=25.0))
        self.assertIn("above", str(caught.exception))
        self.assertIn("refused rather than truncated", str(caught.exception))

    def test_an_inverted_interval_is_refused(self):
        cfg = parse_config(base_config(
            interval={"start_seconds": 3.0, "end_seconds": 2.0, "max_frames": 100}))
        with self.assertRaises(DemoConfigError):
            validate_config(cfg, synthetic_probe())

    def test_a_null_end_runs_to_the_last_frame_of_the_source(self):
        cfg = parse_config(base_config(
            interval={"start_seconds": 0.0, "end_seconds": None, "max_frames": 400}))
        resolved = validate_config(cfg, synthetic_probe(fps=25.0, frame_count=200))
        self.assertEqual(resolved.end_frame, 199)

    def test_a_variable_cadence_source_is_refused_unless_the_operator_accepts_it(self):
        probe = synthetic_probe(variable_cadence=True)
        cfg = parse_config(base_config())
        with self.assertRaises(DemoConfigError) as caught:
            validate_config(cfg, probe)
        self.assertIn("not constant", str(caught.exception))
        accepted = parse_config(base_config(
            timing={"accept_variable_cadence": True}))
        resolved = validate_config(accepted, probe)
        self.assertTrue(resolved.config.timing.accept_variable_cadence)


class BoundsTests(TempDirTestCase):
    def test_a_seed_outside_the_interval_is_refused(self):
        cfg = parse_config(base_config(
            interval={"start_seconds": 1.0, "end_seconds": 2.0, "max_frames": 100},
            seeds={"ball": {"frame": 5, "x": 320, "y": 300}}))
        with self.assertRaises(DemoConfigError) as caught:
            validate_config(cfg, synthetic_probe(fps=25.0))
        self.assertIn("outside the processed interval", str(caught.exception))

    def test_a_seed_outside_the_frame_is_refused(self):
        cfg = parse_config(base_config(
            seeds={"ball": {"frame": 10, "x": 5000, "y": 300}}))
        with self.assertRaises(DemoConfigError) as caught:
            validate_config(cfg, synthetic_probe(width=640, height=480))
        self.assertIn("outside the 640px-wide frame", str(caught.exception))

    def test_a_keyframe_outside_the_interval_is_refused(self):
        cfg = parse_config(base_config(
            interval={"start_seconds": 0.0, "end_seconds": 2.0, "max_frames": 100},
            keyframes={"top": 190}))
        with self.assertRaises(DemoConfigError) as caught:
            validate_config(cfg, synthetic_probe(fps=25.0))
        self.assertIn("keyframes.top", str(caught.exception))

    def test_an_roi_that_does_not_fit_the_frame_is_refused(self):
        cfg = parse_config(base_config(roi=[600, 100, 200, 300]))
        with self.assertRaises(DemoConfigError) as caught:
            validate_config(cfg, synthetic_probe(width=640, height=480))
        self.assertIn("does not fit", str(caught.exception))

    def test_an_roi_with_no_area_is_refused(self):
        with self.assertRaises(DemoConfigError):
            parse_config(base_config(roi=[10, 10, 0, 100]))

    def test_duplicate_corrections_at_one_frame_are_refused(self):
        cfg = parse_config(base_config(corrections={
            "ball": [{"frame": 70, "x": 10, "y": 10}, {"frame": 70, "x": 20, "y": 20}]}))
        with self.assertRaises(DemoConfigError) as caught:
            validate_config(cfg, synthetic_probe(fps=25.0))
        self.assertIn("more than one point at frame 70", str(caught.exception))

    def test_a_correction_colliding_with_the_seed_is_refused(self):
        cfg = parse_config(base_config(corrections={
            "ball": [{"frame": 10, "x": 10, "y": 10}]}))
        with self.assertRaises(DemoConfigError):
            validate_config(cfg, synthetic_probe(fps=25.0))

    def test_corrections_are_ordered_by_frame_in_the_resolved_config(self):
        cfg = parse_config(base_config(corrections={
            "ball": [{"frame": 80, "x": 10, "y": 10}, {"frame": 40, "x": 5, "y": 5}]}))
        resolved = validate_config(cfg, synthetic_probe(fps=25.0))
        self.assertEqual([p.frame for p in resolved.config.corrections["ball"]],
                         [40, 80])

    def test_a_backwards_keyframe_sequence_is_refused(self):
        cfg = parse_config(base_config(keyframes={"address": 20, "top": 60,
                                                  "impact": 50, "finish": 70}))
        with self.assertRaises(DemoConfigError) as caught:
            validate_config(cfg, synthetic_probe(fps=25.0))
        self.assertIn("chronological order", str(caught.exception))

    def test_two_phases_on_one_frame_produce_a_warning_not_a_silent_accept(self):
        cfg = parse_config(base_config(keyframes={"address": 20, "takeaway": 20,
                                                  "top": 60, "impact": 90}))
        resolved = validate_config(cfg, synthetic_probe(fps=25.0))
        self.assertTrue(any("share frame" in w for w in resolved.warnings))


class OptionalInputTests(TempDirTestCase):
    def test_no_seeds_means_both_targets_are_declared_unavailable_up_front(self):
        cfg = parse_config(base_config(seeds={}, corrections={}))
        resolved = validate_config(cfg, synthetic_probe())
        self.assertTrue(any("both targets are reported UNAVAILABLE" in n
                            for n in resolved.notes))

    def test_no_keyframes_means_the_phase_cards_are_declared_unavailable(self):
        cfg = parse_config(base_config(keyframes={}))
        resolved = validate_config(cfg, synthetic_probe())
        self.assertTrue(any("UNAVAILABLE" in n for n in resolved.notes))

    def test_manual_mode_without_a_point_is_refused(self):
        payload = base_config(seeds={})
        payload["targets"]["ball"]["mode"] = "manual"
        with self.assertRaises(DemoConfigError) as caught:
            validate_config(parse_config(payload), synthetic_probe())
        self.assertIn("manual", str(caught.exception))

    def test_interpolation_above_the_loss_threshold_is_flagged(self):
        cfg = parse_config(base_config(interpolation={"enabled": True,
                                                      "max_gap_frames": 9}))
        resolved = validate_config(cfg, synthetic_probe())
        self.assertTrue(any("loss" in n and "stays missing" in n
                            for n in resolved.notes))

    def test_a_non_square_pixel_source_is_reported_before_any_angle_is_claimed(self):
        cfg = parse_config(base_config())
        resolved = validate_config(cfg, synthetic_probe(pixel_aspect_ratio=1.21))
        self.assertTrue(any("sample aspect ratio" in n for n in resolved.notes))


class EffectiveConfigTests(TempDirTestCase):
    def test_the_effective_config_records_every_operator_input_and_a_hash_source(self):
        payload = base_config(corrections={"ball": [{"frame": 70, "x": 12.0, "y": 13.0,
                                                    "note": "restart"}]})
        path = write_config(self.tmp, payload)
        cfg = load_config(str(path))
        resolved = validate_config(cfg, synthetic_probe(fps=25.0))
        effective = resolved.effective()
        self.assertEqual(effective["seeds"]["ball"]["frame"], 10)
        self.assertEqual(effective["corrections"]["ball"][0]["note"], "restart")
        self.assertEqual(effective["keyframes"]["top"], 60)
        self.assertIn("endpoint_semantics", effective)
        self.assertEqual(effective["raw_config"]["schema_version"], DEMO_CONFIG_SCHEMA)

    def test_a_missing_config_file_is_an_error_not_an_empty_config(self):
        with self.assertRaises(DemoConfigError):
            load_config(str(self.tmp / "nope.json"))

    def test_malformed_json_is_an_error(self):
        path = Path(self.tmp) / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(DemoConfigError) as caught:
            load_config(str(path))
        self.assertIn("not valid JSON", str(caught.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
