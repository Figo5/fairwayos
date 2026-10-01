"""Synthetic-frame tests for ghostcaddie.demo.seed_prompt.

Every image here is drawn in the test file: no real media, no reference labels, no
clubhead ground truth.  The frames are the only input the module is allowed to read.
"""
import copy
import inspect
import unittest

import numpy as np

from ghostcaddie.demo.seed_prompt import PROFILE, derive_head_prompts

GREEN = (70, 130, 60)      # BGR turf: saturated, mid brightness
WHITE = (235, 235, 235)    # round white ball / blade / shoe highlight
DARK = (45, 45, 50)        # dark clubhead: low saturation but dark


def frame(width=225, height=225, colour=GREEN):
    return np.full((height, width, 3), colour, dtype=np.uint8)


def paint_disc(image, cx, cy, radius, colour):
    rows, columns = np.ogrid[:image.shape[0], :image.shape[1]]
    image[(columns - cx) ** 2 + (rows - cy) ** 2 <= radius ** 2] = colour


def paint_rect(image, x, y, width, height, colour):
    image[y:y + height, x:x + width] = colour


class SeedPromptTests(unittest.TestCase):
    def test_ball_near_dark_head_yields_one_negative_centre(self):
        image = frame()
        paint_disc(image, 100, 100, 5, DARK)
        paint_disc(image, 114, 100, 4, WHITE)
        result = derive_head_prompts(image, 100, 100)
        self.assertEqual(result.positive_point, (100, 100))
        self.assertEqual(result.negative_points, ((114, 100),))
        admitted = result.metadata['admitted_regions']
        self.assertEqual(len(admitted), 1)
        self.assertFalse(admitted[0]['contains_positive_seed'])
        self.assertGreaterEqual(admitted[0]['seed_distance_px'], PROFILE['min_negative_gap_px'])
        self.assertLessEqual(admitted[0]['seed_distance_px'], PROFILE['search_radius_px'])
        self.assertFalse(result.metadata['failure_closed'])

    def test_round_white_head_region_is_ambiguous_and_fails_closed(self):
        image = frame()
        paint_disc(image, 100, 100, 5, WHITE)   # round white head under the seed
        paint_disc(image, 114, 100, 4, WHITE)   # round white ball next to it
        result = derive_head_prompts(image, 100, 100)
        self.assertEqual(result.negative_points, ())
        self.assertTrue(result.metadata['failure_closed'])
        self.assertEqual(result.metadata['failure_reason'],
                         'positive_seed_inside_round_white_region_prompts_ambiguous')
        self.assertEqual(result.positive_point, (100, 100))

    def test_two_admissible_round_white_regions_fail_closed(self):
        image = frame()
        paint_disc(image, 100, 100, 5, DARK)
        paint_disc(image, 112, 100, 4, WHITE)
        paint_disc(image, 100, 112, 4, WHITE)
        result = derive_head_prompts(image, 100, 100)
        self.assertEqual(result.negative_points, ())
        self.assertTrue(result.metadata['failure_closed'])
        self.assertEqual(result.metadata['failure_reason'],
                         'more_admissible_candidates_than_max_negative_points')

    def test_elongated_white_blade_is_not_a_ball(self):
        image = frame()
        paint_disc(image, 70, 100, 5, DARK)
        paint_rect(image, 78, 98, 16, 4, WHITE)  # nonspherical blade, 16x4 px
        result = derive_head_prompts(image, 70, 100)
        self.assertEqual(result.negative_points, ())
        failed = [region['failed_gates'] for region in result.metadata['rejected_regions']
                  if region['bbox_native_xywh'] == [78, 98, 16, 4]]
        self.assertTrue(failed, 'blade region was not measured')
        self.assertIn('axis_ratio_above_max', failed[0])
        self.assertIn('circularity_below_min', failed[0])

    def test_bright_oversized_highlight_is_not_a_ball(self):
        image = frame()
        paint_disc(image, 70, 100, 5, DARK)
        paint_disc(image, 86, 100, 12, WHITE)  # large bright shoe-like blob
        result = derive_head_prompts(image, 70, 100)
        self.assertEqual(result.negative_points, ())
        gates = [region['failed_gates'] for region in result.metadata['rejected_regions']
                 if region['center_native_xy'] == [86, 100]][0]
        self.assertIn('area_above_max', gates)
        self.assertIn('half_extent_above_max_ball_radius', gates)

    def test_round_white_blob_beyond_search_radius_is_rejected(self):
        image = frame()
        paint_disc(image, 100, 100, 5, DARK)
        paint_disc(image, 150, 100, 4, WHITE)
        result = derive_head_prompts(image, 100, 100)
        self.assertEqual(result.negative_points, ())
        gates = [region['failed_gates'] for region in result.metadata['rejected_regions']
                 if region['center_native_xy'] == [150, 100]][0]
        self.assertIn('distance_above_search_radius', gates)

    def test_seed_at_image_edge_clamps_box_inside_the_image(self):
        image = frame(width=40, height=30)
        paint_disc(image, 4, 4, 4, DARK)
        result = derive_head_prompts(image, 4, 4)
        box = result.positive_box
        self.assertEqual(box, (0, 0, 12, 12))
        self.assertGreaterEqual(box[0], 0)
        self.assertGreaterEqual(box[1], 0)
        self.assertLessEqual(box[0] + box[2], 40)
        self.assertLessEqual(box[1] + box[3], 30)
        self.assertEqual(result.positive_point, (4, 4))
        corner = derive_head_prompts(image, 0, 0)
        self.assertEqual(corner.positive_box, (0, 0, 8, 8))

    def test_box_matches_declared_half_extent_away_from_edges(self):
        result = derive_head_prompts(frame(), 100, 100)
        half = PROFILE['positive_box_half_px']
        self.assertEqual(result.positive_box, (100 - half, 100 - half, 2 * half + 1, 2 * half + 1))

    def test_coordinates_are_local_to_the_passed_image_not_origin_shifted(self):
        crop = frame(width=60, height=60)
        paint_disc(crop, 20, 20, 5, DARK)
        paint_disc(crop, 34, 20, 4, WHITE)
        local = derive_head_prompts(crop, 20, 20)
        self.assertEqual(local.negative_points, ((34, 20),))
        padded = frame(width=200, height=200)
        padded[40:100, 40:100] = crop
        unshifted = derive_head_prompts(padded, 20, 20)  # same local seed, no caller offset
        self.assertEqual(unshifted.negative_points,
                         ())  # proves no origin translation is applied here
        shifted = derive_head_prompts(padded, 60, 60)    # caller does the translation
        self.assertEqual(shifted.negative_points, ((74, 60),))
        self.assertEqual(shifted.metadata['seed_point_native_xy'], [60, 60])

    def test_negative_centre_never_overlaps_the_positive_seed(self):
        image = frame()
        paint_disc(image, 100, 100, 5, DARK)
        paint_disc(image, 103, 100, 4, WHITE)
        result = derive_head_prompts(image, 100, 100)
        self.assertEqual(result.negative_points, ())
        gates = [region['failed_gates'] for region in result.metadata['rejected_regions']
                 if region['center_native_xy'] == [103, 100]][0]
        self.assertIn('distance_below_min_negative_gap', gates)

    def test_result_depends_only_on_the_passed_frame(self):
        first, second = frame(), frame()
        paint_disc(first, 100, 100, 5, DARK)
        paint_disc(first, 114, 100, 4, WHITE)
        paint_disc(second, 60, 60, 4, WHITE)
        before = copy.deepcopy(PROFILE)
        a = derive_head_prompts(first, 100, 100)
        derive_head_prompts(second, 100, 100)
        b = derive_head_prompts(first, 100, 100)
        self.assertEqual(a.negative_points, b.negative_points)
        self.assertEqual(a.metadata, b.metadata)
        self.assertEqual(PROFILE, before)
        self.assertEqual(a.metadata['frames_read'], 1)
        self.assertFalse(a.metadata['future_frame_access'])

    def test_signature_exposes_no_frame_index_labels_or_track(self):
        parameters = list(inspect.signature(derive_head_prompts).parameters)
        self.assertEqual(parameters, ['image_bgr', 'seed_x', 'seed_y'])

    def test_no_round_white_candidate_returns_unchanged_positive_prompt(self):
        image = frame()
        paint_disc(image, 100, 100, 5, DARK)
        result = derive_head_prompts(image, 100, 100)
        self.assertEqual(result.negative_points, ())
        self.assertEqual(result.positive_point, (100, 100))
        self.assertEqual(result.positive_box, (93, 93, 15, 15))
        self.assertIn('no_admissible_round_white_candidate', result.metadata['reasons'])
        self.assertFalse(result.metadata['failure_closed'])

    def test_metadata_declares_thresholds_and_no_confidence(self):
        result = derive_head_prompts(frame(), 100, 100)
        metadata = result.metadata
        self.assertIsNone(metadata['confidence'])
        self.assertIn('binary', metadata['score_kind'])
        self.assertEqual(metadata['thresholds']['search_radius_px'], PROFILE['search_radius_px'])
        self.assertEqual(metadata['thresholds']['max_axis_ratio'], PROFILE['max_axis_ratio'])
        self.assertEqual(metadata['code_profile'], PROFILE)
        self.assertNotIn('target', metadata)

    def test_invalid_inputs_raise(self):
        image = frame()
        with self.assertRaises(ValueError):
            derive_head_prompts(image, 225, 100)
        with self.assertRaises(ValueError):
            derive_head_prompts(image, -1, 10)
        with self.assertRaises(ValueError):
            derive_head_prompts(image, float('nan'), 10)
        with self.assertRaises(ValueError):
            derive_head_prompts(image[:, :, :1], 10, 10)
        with self.assertRaises(ValueError):
            derive_head_prompts(frame(width=8, height=8), 4, 4)

    def test_fractional_positive_seed_is_preserved_including_image_edge(self):
        image = frame(width=40, height=30)
        result = derive_head_prompts(image, 39.8, 29.7)
        self.assertEqual(result.positive_point, (39.8, 29.7))
        self.assertEqual(result.metadata['positive_point_native_xy'], [39.8, 29.7])
        self.assertEqual(result.positive_box[0], 32.8)
        with self.assertRaises(ValueError):
            derive_head_prompts(image, float('inf'), 10)
        with self.assertRaises(ValueError):
            derive_head_prompts(image, True, 10)


if __name__ == '__main__':
    unittest.main()
