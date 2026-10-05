"""Synthetic-only tests for the classic ball evidence validator.

Everything here is drawn with numpy/cv2. No clip, label, credential or network
resource is touched: this is an isolated research validator, not a canonical
detector, and the tests prove it stays that way.
"""
import copy
import json
import math
import unittest

import cv2
import numpy as np

from ghostcaddie.demo.ball_evidence import (
    PROFILE,
    build_seed_descriptor,
    check_ball_evidence,
)

GRASS = (62, 126, 95)  # BGR turf; grayscale ~= 109, a mid-tone background
WHITE = (255, 255, 255)
SIZE = 64


def grass(size=SIZE):
    return np.full((size, size, 3), GRASS, np.uint8)


def disc(image, cx, cy, radius, color=WHITE):
    cv2.circle(image, (int(cx), int(cy)), int(radius), color, -1)
    return image


def strip(image, x0, x1, y0, y1, color=WHITE):
    image[y0:y1, x0:x1] = color
    return image


def ball_image(cx=32, cy=32, radius=5, color=WHITE, size=SIZE):
    return disc(grass(size), cx, cy, radius, color)


class AcceptedBallTests(unittest.TestCase):
    def test_seed_and_current_accept_a_synthetic_ball_on_grass(self):
        image = ball_image(32, 32, 5)
        built = build_seed_descriptor(image, 32, 32)
        self.assertTrue(built["valid"], built)
        self.assertEqual(built["reason"], "unique_compact_seed_component")
        descriptor = built["descriptor"]
        self.assertIsNotNone(descriptor)

        checked = check_ball_evidence(image, 32, 32, descriptor)
        self.assertTrue(checked["accepted"], checked)
        self.assertEqual(checked["reason"], "unique_compact_component")
        current = checked["current"]
        self.assertIsNotNone(current)
        self.assertEqual(current["coordinates_refined"], False)
        self.assertEqual(current["physical_identity_proven"], False)
        self.assertAlmostEqual(current["area_ratio"], 1.0, places=2)

    def test_the_validator_never_returns_an_updated_point(self):
        image = ball_image()
        descriptor = build_seed_descriptor(image, 32, 32)["descriptor"]
        current = check_ball_evidence(image, 32, 32, descriptor)["current"]
        for forbidden in ("x", "y", "point", "updated_point", "refined_point"):
            self.assertNotIn(forbidden, current)

    def test_seed_descriptor_records_the_audit_geometry(self):
        descriptor = build_seed_descriptor(ball_image(), 32, 32)["descriptor"]
        for key in ("gray_threshold", "background_gray", "central_peak_gray",
                    "area", "bbox", "max_dimension", "elongation",
                    "centroid", "centroid_offset", "centroid_distance"):
            self.assertIn(key, descriptor)
        self.assertGreaterEqual(descriptor["area"], PROFILE["core_min_area_px"])
        self.assertLessEqual(descriptor["area"], PROFILE["core_max_area_px"])


class ThresholdSemanticsTests(unittest.TestCase):
    def test_gray_threshold_is_background_plus_the_declared_margin(self):
        descriptor = build_seed_descriptor(ball_image(), 32, 32)["descriptor"]
        background = descriptor["background_gray"]
        peak = descriptor["central_peak_gray"]
        expected = background + max(PROFILE["min_background_margin_gray"],
                                    PROFILE["peak_margin_fraction"] * (peak - background))
        self.assertAlmostEqual(descriptor["gray_threshold"], expected, places=6)

    def test_gray_threshold_is_frozen_and_reused_by_later_checks(self):
        seed = ball_image(radius=5)
        descriptor = build_seed_descriptor(seed, 32, 32)["descriptor"]
        frozen = descriptor["gray_threshold"]
        # a brighter, differently-lit current frame must NOT recompute the threshold
        brighter = ball_image(radius=5, color=(255, 255, 255))
        brighter[:] = np.clip(brighter.astype(np.int16) + 10, 0, 255).astype(np.uint8)
        checked = check_ball_evidence(brighter, 32, 32, descriptor)
        self.assertTrue(checked["accepted"], checked)
        self.assertEqual(checked["current"]["gray_threshold"], frozen)

    def test_a_saturated_bright_patch_is_excluded_by_the_saturation_ceiling(self):
        # pure yellow is bright but far above the saturation ceiling: it is not
        # treated as "encoded light ball" evidence
        image = disc(grass(), 32, 32, 5, color=(0, 255, 255))
        built = build_seed_descriptor(image, 32, 32)
        self.assertFalse(built["valid"])
        self.assertEqual(built["reason"], "no_compact_seed_component")


class RejectionTests(unittest.TestCase):
    def test_no_foreground_grass_is_rejected_for_seed_and_current(self):
        image = grass()
        built = build_seed_descriptor(image, 32, 32)
        self.assertFalse(built["valid"])
        self.assertEqual(built["reason"], "no_compact_seed_component")

        descriptor = build_seed_descriptor(ball_image(), 32, 32)["descriptor"]
        checked = check_ball_evidence(image, 32, 32, descriptor)
        self.assertFalse(checked["accepted"])
        self.assertEqual(checked["reason"], "no_eligible_component")

    def test_a_long_bright_strip_is_rejected_for_seed_and_current(self):
        strip_image = strip(grass(), 20, 44, 31, 34)  # 24 x 3 white bar
        built = build_seed_descriptor(strip_image, 32, 32)
        self.assertFalse(built["valid"], built)
        self.assertEqual(built["reason"], "no_compact_seed_component")

        descriptor = build_seed_descriptor(ball_image(), 32, 32)["descriptor"]
        checked = check_ball_evidence(strip_image, 32, 32, descriptor)
        self.assertFalse(checked["accepted"], checked)
        self.assertEqual(checked["reason"], "no_eligible_component")

    def test_a_strip_longer_than_the_dimension_cap_is_rejected(self):
        # white cup strip / shaft: over the 25 px bounding-box cap even before
        # elongation is considered
        strip_image = strip(grass(), 16, 48, 31, 33)  # 32 x 2 white bar
        built = build_seed_descriptor(strip_image, 32, 32)
        self.assertFalse(built["valid"], built)

    def test_an_oversized_bright_patch_is_rejected(self):
        image = disc(grass(), 32, 32, 12)  # area far above the 200 px cap
        built = build_seed_descriptor(image, 32, 32)
        self.assertFalse(built["valid"])
        self.assertEqual(built["reason"], "no_compact_seed_component")

        descriptor = build_seed_descriptor(ball_image(), 32, 32)["descriptor"]
        checked = check_ball_evidence(image, 32, 32, descriptor)
        self.assertFalse(checked["accepted"], checked)

    def test_two_separated_plausible_cores_near_the_proposal_are_ambiguous(self):
        two_seed = disc(disc(grass(), 28, 32, 2), 36, 32, 2)
        built = build_seed_descriptor(two_seed, 32, 32)
        self.assertFalse(built["valid"], built)
        self.assertEqual(built["reason"], "ambiguous_seed_component")

        # a valid single-ball seed, then a current frame with two compatible cores
        image = disc(disc(grass(), 28, 32, 3), 36, 32, 3)
        descriptor = build_seed_descriptor(ball_image(radius=5), 32, 32)["descriptor"]
        checked = check_ball_evidence(image, 32, 32, descriptor)
        self.assertFalse(checked["accepted"], checked)
        self.assertEqual(checked["reason"], "ambiguous_evidence")


class BoundedChangeTests(unittest.TestCase):
    def test_size_changes_within_the_declared_ratios_are_accepted(self):
        descriptor = build_seed_descriptor(ball_image(radius=5), 32, 32)["descriptor"]
        for radius in (4, 5, 6):
            checked = check_ball_evidence(ball_image(radius=radius), 32, 32, descriptor)
            self.assertTrue(checked["accepted"], (radius, checked))

    def test_size_beyond_the_area_ratio_is_rejected(self):
        descriptor = build_seed_descriptor(ball_image(radius=2), 32, 32)["descriptor"]
        self.assertIsNotNone(descriptor)
        checked = check_ball_evidence(ball_image(radius=5), 32, 32, descriptor)
        self.assertFalse(checked["accepted"], checked)
        self.assertEqual(checked["reason"], "no_eligible_component")

    def test_illumination_within_the_frozen_threshold_is_accepted(self):
        descriptor = build_seed_descriptor(ball_image(radius=5), 32, 32)["descriptor"]
        dimmer = ball_image(radius=5, color=(215, 215, 215))
        checked = check_ball_evidence(dimmer, 32, 32, descriptor)
        self.assertTrue(checked["accepted"], checked)

    def test_illumination_below_the_frozen_threshold_is_rejected(self):
        descriptor = build_seed_descriptor(ball_image(radius=5), 32, 32)["descriptor"]
        dark = ball_image(radius=5, color=(150, 150, 150))
        checked = check_ball_evidence(dark, 32, 32, descriptor)
        self.assertFalse(checked["accepted"], checked)
        self.assertEqual(checked["reason"], "no_eligible_component")


class SeedIntegrityTests(unittest.TestCase):
    def test_an_invalid_seed_disables_every_observation(self):
        invalid = build_seed_descriptor(grass(), 32, 32)
        self.assertFalse(invalid["valid"])

        # passing None (the inner descriptor of an invalid seed)
        checked = check_ball_evidence(ball_image(), 32, 32, None)
        self.assertFalse(checked["accepted"])
        self.assertEqual(checked["reason"], "seed_invalid")

        # passing the whole wrapper result of an invalid seed
        checked = check_ball_evidence(ball_image(), 32, 32, invalid)
        self.assertFalse(checked["accepted"])
        self.assertEqual(checked["reason"], "seed_invalid")

    def test_a_malformed_descriptor_is_refused(self):
        checked = check_ball_evidence(ball_image(), 32, 32, {})
        self.assertFalse(checked["accepted"])
        self.assertEqual(checked["reason"], "invalid_descriptor")

    def test_the_seed_descriptor_is_not_mutated_by_checks(self):
        descriptor = build_seed_descriptor(ball_image(radius=5), 32, 32)["descriptor"]
        snapshot = copy.deepcopy(descriptor)
        for radius, color in ((4, WHITE), (6, WHITE), (5, (215, 215, 215))):
            check_ball_evidence(ball_image(radius=radius, color=color), 32, 32, descriptor)
        check_ball_evidence(grass(), 32, 32, descriptor)
        self.assertEqual(descriptor, snapshot)

    def test_the_validator_does_not_mutate_the_image(self):
        image = ball_image(radius=5)
        snapshot = image.copy()
        descriptor = build_seed_descriptor(image, 32, 32)["descriptor"]
        check_ball_evidence(image, 32, 32, descriptor)
        self.assertTrue(np.array_equal(image, snapshot))


class DeterminismTests(unittest.TestCase):
    def test_fractional_seed_points_are_deterministic(self):
        image = ball_image(32, 32, 5)
        first = build_seed_descriptor(image, 31.5, 31.5)
        second = build_seed_descriptor(image, 31.5, 31.5)
        self.assertEqual(first, second)
        self.assertTrue(first["valid"], first)

        descriptor = first["descriptor"]
        a = check_ball_evidence(image, 31.5, 31.5, descriptor)
        b = check_ball_evidence(image, 31.5, 31.5, descriptor)
        self.assertEqual(a, b)
        self.assertTrue(a["accepted"], a)

    def test_the_gate_is_stateless_and_prefix_invariant(self):
        image = ball_image(32, 32, 5)
        descriptor = build_seed_descriptor(image, 32, 32)["descriptor"]
        before = check_ball_evidence(image, 32, 32, descriptor)
        # an unrelated intervening call must not change a later result
        build_seed_descriptor(ball_image(radius=6), 32, 32)
        check_ball_evidence(grass(), 32, 32, descriptor)
        after = check_ball_evidence(image, 32, 32, descriptor)
        self.assertEqual(before, after)

    def test_results_are_deterministic_finite_json(self):
        image = ball_image(32, 32, 5)
        built = build_seed_descriptor(image, 32, 32)
        checked = check_ball_evidence(image, 32, 32, built["descriptor"])
        for payload in (built, checked):
            text = json.dumps(payload, allow_nan=False)
            self.assertNotIn("NaN", text)
            self.assertNotIn("Infinity", text)
        self.assertEqual(build_seed_descriptor(image, 32, 32), built)

    def test_repeated_calls_reproduce_identical_json(self):
        image = ball_image(32, 32, 5)
        descriptor = build_seed_descriptor(image, 32, 32)["descriptor"]
        first = json.dumps(check_ball_evidence(image, 32, 32, descriptor))
        second = json.dumps(check_ball_evidence(image, 32, 32, descriptor))
        self.assertEqual(first, second)


class BoundaryInputTests(unittest.TestCase):
    def test_edge_rois_are_rejected_cleanly(self):
        image = grass()
        for x, y in ((2, 32), (32, 2), (62, 32), (32, 62), (0, 0)):
            built = build_seed_descriptor(image, x, y)
            self.assertFalse(built["valid"], (x, y))
            self.assertEqual(built["reason"], "seed_roi_incomplete")

    def test_nonfinite_inputs_are_rejected_cleanly(self):
        image = grass()
        for value in (float("nan"), float("inf"), float("-inf")):
            built = build_seed_descriptor(image, value, 32)
            self.assertFalse(built["valid"])
            self.assertEqual(built["reason"], "seed_not_finite")

        descriptor = build_seed_descriptor(ball_image(), 32, 32)["descriptor"]
        checked = check_ball_evidence(ball_image(), float("nan"), 32, descriptor)
        self.assertFalse(checked["accepted"])
        self.assertEqual(checked["reason"], "proposal_not_finite")

    def test_edge_proposal_is_rejected_cleanly(self):
        descriptor = build_seed_descriptor(ball_image(), 32, 32)["descriptor"]
        checked = check_ball_evidence(ball_image(), 2, 32, descriptor)
        self.assertFalse(checked["accepted"])
        self.assertEqual(checked["reason"], "roi_incomplete")

    def test_malformed_frames_raise_value_error(self):
        bad_frames = [
            None,
            np.zeros((10, 10), np.uint8),            # grayscale, not BGR
            np.zeros((10, 10, 4), np.uint8),         # wrong channel count
            np.zeros((10, 10, 3), np.float32),       # not uint8
        ]
        for frame in bad_frames:
            with self.assertRaises(ValueError, msg=repr(frame)):
                build_seed_descriptor(frame, 5, 5)
            with self.assertRaises(ValueError, msg=repr(frame)):
                check_ball_evidence(frame, 5, 5, None)

    def test_non_numeric_coordinates_raise_value_error(self):
        image = ball_image()
        with self.assertRaises(ValueError):
            build_seed_descriptor(image, "a", 32)
        with self.assertRaises(ValueError):
            check_ball_evidence(image, 32, None, None)


class UnitPixelSupportTests(unittest.TestCase):
    def test_two_row_ball_uses_occupied_pixel_extent(self):
        from ghostcaddie.demo.ball_evidence import _components
        mask = np.zeros((31, 31), dtype=np.uint8)
        mask[15:17, 13:18] = 1
        component = _components(mask)[0]
        self.assertEqual(component["area"], 10)
        self.assertEqual(component["bbox"], [13, 15, 5, 2])
        self.assertAlmostEqual(component["elongation"], 2.5)
        image = grass()
        image[32:34, 30:35] = (235, 235, 235)
        descriptor = build_seed_descriptor(image, 32, 32)
        self.assertTrue(descriptor["valid"])
        self.assertTrue(check_ball_evidence(image, 32, 32, descriptor)["accepted"])

    def test_rotated_thin_shaft_stays_noncompact_without_adding_mask_pixels(self):
        from ghostcaddie.demo.ball_evidence import _components
        mask = np.zeros((31, 31), dtype=np.uint8)
        for i in range(7, 24):
            mask[i, i] = 1
        original = mask.copy()
        component = _components(mask)[0]
        self.assertEqual(component["area"], 17)
        self.assertGreater(component["elongation"], PROFILE["core_max_elongation_seed"])
        np.testing.assert_array_equal(mask, original)

    def test_single_pixel_has_unit_extent_but_fails_minimum_area(self):
        from ghostcaddie.demo.ball_evidence import _components
        mask = np.zeros((31, 31), dtype=np.uint8)
        mask[15, 15] = 1
        component = _components(mask)[0]
        self.assertEqual(component["area"], 1)
        self.assertAlmostEqual(component["elongation"], 1)
        image = grass()
        image[32, 32] = (235, 235, 235)
        self.assertFalse(build_seed_descriptor(image, 32, 32)["valid"])


class ProfileDisclosureTests(unittest.TestCase):
    def test_the_profile_records_the_declared_policy(self):
        expected = {
            "roi_size_px": 31,
            "core_min_area_px": 3,
            "core_max_area_px": 200,
            "core_max_dimension_px": 25,
            "core_max_elongation_seed": 3.5,
            "core_centroid_tolerance_px": 4.0,
            "min_background_margin_gray": 15.0,
            "peak_margin_fraction": 0.65,
            "max_saturation": 140,
            "current_min_area_ratio": 0.3,
            "current_max_area_ratio": 3.0,
            "current_min_dimension_ratio": 0.45,
            "current_max_dimension_ratio": 2.2,
            "current_min_elongation": 2.5,
            "current_elongation_factor": 1.7,
        }
        for key, value in expected.items():
            self.assertEqual(PROFILE[key], value, key)
        self.assertTrue(PROFILE["research_only"])
        self.assertFalse(PROFILE["production_eligible"])
        self.assertFalse(PROFILE["coordinates_refined"])
        self.assertIsNone(PROFILE["exported_confidence"])


if __name__ == "__main__":
    unittest.main()
