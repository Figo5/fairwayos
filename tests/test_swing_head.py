import copy
import json
import unittest

import cv2
import numpy as np

from ghostcaddie.demo.segmented_head import POINT_POLICY
from ghostcaddie.demo.swing_head import (LAB_CHROMA_DISTANCE_MAX,
                                         LAB_CHROMA_DISTANCE_MAX_POSITIVE_GAIN,
                                         METHOD_CORE, METHOD_REACQUIRED, PROFILE,
                                         SwingHeadPolicy, compare_swing_appearance,
                                         measured_components, oriented_extent,
                                         round_bright)


def fixture(core_color=(60, 60, 62), box=(46, 55, 32, 49)):
    y0, y1, x0, x1 = box
    mask = np.zeros((110, 110), np.uint8)
    mask[y0:y1, x0:x1] = 1
    image = np.full((110, 110, 3), (50, 120, 60), np.uint8)
    image[mask > 0] = core_color
    return mask > 0, image


def block(y0, y1, x0, x1, color=(60, 60, 62), size=(110, 110)):
    mask = np.zeros(size, np.uint8)
    mask[y0:y1, x0:x1] = 1
    image = np.full((size[0], size[1], 3), (50, 120, 60), np.uint8)
    image[mask > 0] = color
    return mask > 0, image


def policy(cuts=(), point_policy=None):
    return SwingHeadPolicy({'frame': 0, 'x': 140., 'y': 250.}, [100, 200], cuts,
                           point_policy=point_policy)


class SwingHeadTests(unittest.TestCase):
    # -- seed is operator-only -------------------------------------------

    def test_seed_is_operator_only_and_scored_null(self):
        mask, image = fixture()
        head = policy()
        anchor, _ = head.step(0, mask, image)
        self.assertTrue(head.seed_valid)
        self.assertEqual(anchor['source_frame'], 0)
        self.assertEqual(anchor['status'], 'manual_anchor')
        self.assertEqual(anchor['origin'], 'operator')
        self.assertEqual((anchor['x'], anchor['y']), (140, 250))
        self.assertIsNone(anchor['score'])

    def test_dark_head_tracked_with_swing_method_and_json_evidence(self):
        mask, image = fixture()
        head = policy()
        head.step(0, mask, image)
        point, raw = head.step(1, mask, image)
        self.assertEqual(point['origin'], 'automatic')
        self.assertEqual(point['method'], METHOD_CORE)
        self.assertEqual((point['x'], point['y']), (140, 250))
        self.assertIsNone(point['score'])
        self.assertEqual(head.seed_parent_area, 153)
        self.assertEqual(raw['admitted'], [[140.0, 250.0]])
        self.assertIsNone(raw['decision']['score'])
        self.assertEqual(raw['decision']['method'], METHOD_CORE)
        self.assertIsNone(raw['decision']['measurement']['score'])
        self.assertEqual(raw['decision']['policy_internal_binary_gate']['score'], 1)
        json.dumps({'point': point, 'raw': raw}, allow_nan=False)

    # -- general measurement defect --------------------------------------

    def test_thin_and_thick_components_remeasured_independently(self):
        mask = np.zeros((140, 140), np.uint8)
        mask[20:40, 20:40] = 1     # thick square: parent area 400
        mask[60:64, 60:80] = 1     # thin bar: parent area 80
        image = np.full((140, 140, 3), (50, 120, 60), np.uint8)
        parts = measured_components(mask, image)
        by_parent = {}
        for part in parts:
            by_parent.setdefault(part['parent_area'], []).append(part)
        self.assertIn(400, by_parent)
        self.assertIn(80, by_parent)
        thick = by_parent[400][0]
        thin = by_parent[80][0]
        # The thin bar would be deleted by any global/thick threshold; its OWN peak
        # sets a small threshold, so it still yields a measurable core.
        self.assertLess(thin['core_threshold_px'], thick['core_threshold_px'])
        self.assertLessEqual(thin['core_threshold_px'], 1.8 + 1e-9)
        for part in parts:
            self.assertGreaterEqual(part['area'], 3)
            self.assertLessEqual(part['area'], 180)
            self.assertLessEqual(max(part['bbox'][2], part['bbox'][3]), 24)

    def test_unitpixel_square_support_regression(self):
        core = np.zeros((40, 40), bool)
        core[10:20, 5:25] = True          # a physical 20 x 10 block
        long_side, short_side, elongation = oriented_extent(core)
        self.assertAlmostEqual(long_side, 20.0, places=4)
        self.assertAlmostEqual(short_side, 10.0, places=4)
        self.assertAlmostEqual(elongation, 2.0, places=4)

    # -- appearance gate --------------------------------------------------

    def test_darkening_face_is_admitted_within_bounds(self):
        mask, image = fixture(core_color=(60, 60, 62))
        head = policy()
        head.step(0, mask, image)
        darker = image.copy()
        darker[mask] = (30, 30, 31)          # DARKER core: a seed-face darkening
        point, raw = head.step(1, mask, darker)
        self.assertEqual(point['origin'], 'automatic')
        self.assertEqual(point['method'], METHOD_CORE)
        check = raw['identity_checks'][0]
        self.assertTrue(check['accepted'])
        self.assertLess(check['feature_differences']['lab_lightness_gain'], 0.0)

    def test_neutral_brightening_is_admitted_within_bounds(self):
        mask, image = fixture(core_color=(60, 60, 62))
        head = policy()
        head.step(0, mask, image)
        brighter = image.copy()
        brighter[mask] = (100, 100, 105)     # brighter, but near-neutral chroma
        point, raw = head.step(1, mask, brighter)
        self.assertEqual(point['origin'], 'automatic')
        self.assertEqual(point['method'], METHOD_CORE)
        check = raw['identity_checks'][0]
        self.assertTrue(check['accepted'])
        self.assertGreater(check['feature_differences']['lab_lightness_gain'], 0.0)
        self.assertTrue(check['feature_differences']['lightness_gain_positive'])
        self.assertLessEqual(check['feature_differences']['lab_chroma_euclidean'],
                             LAB_CHROMA_DISTANCE_MAX_POSITIVE_GAIN)

    def test_brightening_with_chroma_drift_is_refused(self):
        mask, image = fixture(core_color=(60, 60, 62))
        head = policy()
        head.step(0, mask, image)
        drifting = image.copy()
        drifting[mask] = (200, 60, 60)       # brighter AND strongly coloured
        point, raw = head.step(1, mask, drifting)
        self.assertIsNone(point['x'])
        check = raw['identity_checks'][0]
        self.assertFalse(check['accepted'])
        self.assertGreater(check['feature_differences']['lab_lightness_gain'], 0.0)
        self.assertGreater(check['feature_differences']['lab_chroma_euclidean'],
                           LAB_CHROMA_DISTANCE_MAX_POSITIVE_GAIN)

    def test_lighter_than_dark_seed_core_is_refused(self):
        mask, image = fixture(core_color=(60, 60, 62))
        head = policy()
        head.step(0, mask, image)
        lighter = image.copy()
        lighter[mask] = (200, 200, 202)
        point, raw = head.step(1, mask, lighter)
        self.assertIsNone(point['x'])
        self.assertTrue(any('lab_lightness_gain_exceeds_swing_bound' in c['refusal_reasons']
                            for c in raw['identity_checks']))

    def test_stationary_bright_round_ball_is_refused(self):
        mask, image = fixture(core_color=(60, 60, 62))
        head = policy()
        head.step(0, mask, image)
        ball_mask = np.zeros((110, 110), bool)
        cv2.circle(ball_mask, (40, 50), 8, True, -1)
        ball_image = np.full((110, 110, 3), (50, 120, 60), np.uint8)
        ball_image[ball_mask] = (243, 245, 247)
        point, raw = head.step(1, ball_mask, ball_image)
        self.assertIsNone(point['x'])
        self.assertTrue(any('round_bright_core_ambiguous_with_ball' in c['refusal_reasons']
                            or 'lab_lightness_gain_exceeds_swing_bound' in c['refusal_reasons']
                            for c in raw['identity_checks']))

    def test_round_bright_seed_is_refused(self):
        ball_mask = np.zeros((110, 110), bool)
        cv2.circle(ball_mask, (40, 50), 8, True, -1)
        ball_image = np.full((110, 110, 3), (50, 120, 60), np.uint8)
        ball_image[ball_mask] = (243, 245, 247)
        head = policy()
        head.step(0, ball_mask, ball_image)
        self.assertFalse(head.seed_valid)
        for frame in range(1, 5):
            self.assertIsNone(head.step(frame, ball_mask, ball_image)[0]['x'])

    # -- ambiguity / support ---------------------------------------------

    def test_two_valid_candidates_are_ambiguous(self):
        mask, image = fixture()
        two = mask.copy()
        two[46:55, 54:71] = True
        two_image = np.full((110, 110, 3), (50, 120, 60), np.uint8)
        two_image[two] = (60, 60, 62)
        head = policy()
        head.step(0, mask, image)
        point, raw = head.step(1, two, two_image)
        self.assertIsNone(point['x'])
        self.assertEqual(len(raw['admitted']), 2)

    def test_tiny_unsupported_seed_is_invalid(self):
        mask, image = fixture()
        head = SwingHeadPolicy({'frame': 0, 'x': 10., 'y': 10.}, [0, 0], [])
        head.step(0, mask, image)
        self.assertFalse(head.seed_valid)
        for frame in range(1, 5):
            self.assertIsNone(head.step(frame, mask, image)[0]['x'])

    # -- causality / invariance ------------------------------------------

    def test_strict_forward_prefix_invariance_and_frozen_descriptor(self):
        mask, image = fixture()
        first = policy()
        points_a = [first.step(frame, mask, image)[0] for frame in range(6)]
        second = policy()
        points_b = [second.step(frame, mask, image)[0] for frame in range(6)]
        for frame in range(6, 10):
            second.step(frame, mask, image)
        self.assertEqual(points_a, points_b)

        head = policy()
        _, raw = head.step(0, mask, image)
        frozen = copy.deepcopy(head.seed_descriptor)
        raw['parts'][0]['appearance']['lab_median'][0] = 255
        self.assertEqual(head.seed_descriptor, frozen)
        for frame in range(1, 4):
            head.step(frame, mask, image)
        self.assertEqual(head.seed_descriptor, frozen)

    def test_two_confirmations_after_long_gap_within_240_no_backfill(self):
        mask, image = fixture()
        blank = np.zeros_like(mask)
        head = policy()
        head.step(0, mask, image)
        head.step(1, mask, image)
        for frame in range(2, 32):        # 30 frames: would overrun a 24-frame window
            head.step(frame, blank, image)
        early, _ = head.step(32, mask, image)
        recovered, _ = head.step(33, mask, image)
        self.assertIsNone(early['x'])
        self.assertEqual(recovered['method'], METHOD_REACQUIRED)
        self.assertEqual((recovered['x'], recovered['y']), (140, 250))
        self.assertEqual(head.policy_settings['reacq_window_frames'], 240)

    def test_cut_is_terminal_and_resets_without_ghost(self):
        mask, image = fixture()
        head = policy([3])
        for frame in range(3):
            head.step(frame, mask, image)
        cut, evidence = head.step(3, mask, image)
        self.assertEqual(evidence['admitted'], [])
        self.assertIsNone(cut['x'])
        self.assertEqual(cut['origin'], 'derived')
        for frame in range(4, 247):
            self.assertIsNone(head.step(frame, mask, image)[0]['x'])

    def test_seed_on_cut_is_refused_without_operator_marker(self):
        mask, image = fixture()
        head = SwingHeadPolicy({'frame': 0, 'x': 140., 'y': 250.}, [100, 200], [0])
        point, _ = head.step(0, mask, image)
        self.assertFalse(head.seed_valid)
        self.assertEqual(point['origin'], 'derived')
        self.assertIsNone(point['x'])
        for frame in range(1, 4):
            self.assertIsNone(head.step(frame, mask, image)[0]['x'])

    # -- parent-area ratio ------------------------------------------------

    def test_parent_area_ratio_governs_not_seed_core_area(self):
        seed_mask, seed_image = block(40, 60, 31, 51)
        head = SwingHeadPolicy({'frame': 0, 'x': 140.5, 'y': 249.5}, [100, 200], [])
        head.step(0, seed_mask, seed_image)
        self.assertEqual(head.seed_parent_area, 400)
        seed_core_area = head.seed_region['area']

        hole = seed_mask.copy()
        hole[46:54, 37:45] = False        # keep parent ratio in range, change core
        point, raw = head.step(1, hole, seed_image)
        self.assertEqual(point['origin'], 'automatic')
        self.assertNotEqual(raw['parts'][0]['area'], seed_core_area)
        ratios = [check['parent_area_ratio'] for check in raw['identity_checks']]
        self.assertTrue(any(abs(ratio - 336 / 400) < 1e-6 for ratio in ratios))

    # -- profile / policy settings ---------------------------------------

    def test_policy_settings_recorded_and_reacq_window_forced(self):
        head = policy(point_policy={'search_radius_px': 40, 'reacq_confirm_frames': 3})
        self.assertEqual(head.policy_settings['search_radius_px'], 40)
        self.assertEqual(head.policy_settings['reacq_confirm_frames'], 3)
        self.assertEqual(head.policy_settings['max_step_px'], POINT_POLICY['max_step_px'])
        self.assertEqual(head.policy_settings['reacq_window_frames'], 240)
        forced = policy(point_policy={'reacq_window_frames': 10})
        self.assertEqual(forced.policy_settings['reacq_window_frames'], 240)
        mask, image = fixture()
        _, raw = head.step(0, mask, image)
        self.assertEqual(raw['policy_settings'], head.policy_settings)

    def test_invalid_policy_values_refused(self):
        for setting in [{'loss_frames': 0}, {'reacq_confirm_frames': True},
                        {'search_radius_px': float('nan')}, {'max_step_px': -1},
                        {'min_score': 2}, {'ambiguity_margin': -.1}]:
            with self.assertRaises(ValueError):
                policy(point_policy=setting)

    def test_profile_records_every_fixed_numeric_choice(self):
        for key in ('core_min_distance_px', 'core_peak_fraction', 'core_min_area_px',
                    'core_max_area_px', 'core_max_dimension_px', 'seed_support_distance_px',
                    'min_parent_area_ratio', 'max_parent_area_ratio', 'lab_chroma_distance_max',
                    'lab_lightness_gain_max', 'dispersion_ratio_min', 'dispersion_ratio_max',
                    'unit_square_half', 'reacq_window_frames', 'min_core_samples'):
            self.assertIn(key, PROFILE)
        self.assertEqual(PROFILE['reacq_window_frames'], 240)
        self.assertEqual(PROFILE['max_parent_area_ratio'], 5.0)
        self.assertIsNone(PROFILE['exported_confidence'])
        self.assertFalse(PROFILE['physical_identity_established'])
        self.assertFalse(PROFILE['production_eligible'])

    # -- input refusals ---------------------------------------------------

    def test_bad_inputs_are_refused(self):
        seed = {'frame': 0, 'x': 140., 'y': 250.}
        with self.assertRaises(ValueError):
            SwingHeadPolicy(seed, [100, 200], [], 'putt')
        with self.assertRaises(ValueError):
            SwingHeadPolicy(seed, [100, 200], [], 'full_swing', {'bogus': 1})
        with self.assertRaises(ValueError):
            SwingHeadPolicy({'frame': 0, 'x': float('nan'), 'y': 250.}, [100, 200], [])
        with self.assertRaises(ValueError):
            SwingHeadPolicy(seed, [100, float('inf')], [])
        with self.assertRaises(ValueError):
            SwingHeadPolicy('not-a-seed', [100, 200], [])

        mask, image = fixture()
        ordered = policy()
        ordered.step(0, mask, image)
        with self.assertRaises(ValueError):
            ordered.step(2, mask, image)
        with self.assertRaises(ValueError):
            policy().step(0, mask.astype(np.uint8), image)
        with self.assertRaises(ValueError):
            policy().step(0, mask, image.astype(np.float32))
        with self.assertRaises(ValueError):
            policy().step(0, mask, np.zeros((50, 50, 3), np.uint8))
        with self.assertRaises(ValueError):
            policy().step(0, mask[:, :, None], image)

    def test_invalid_frames_cuts_and_empty_images_leave_state_unchanged(self):
        mask,image=fixture()
        for frame in [True, 0., -1]:
            with self.assertRaises(ValueError):
                SwingHeadPolicy({'frame':frame,'x':140,'y':250},[100,200],[])
            h=policy()
            with self.assertRaises(ValueError):h.step(frame,mask,image)
            self.assertEqual(h.last_frame,-1)
        for cut in ['3',True,-1,3.]:
            with self.assertRaises(ValueError):policy([cut])
        h=policy()
        with self.assertRaises(ValueError):h.step(0,np.zeros((0,0),bool),np.zeros((0,0,3),np.uint8))
        self.assertEqual(h.last_frame,-1)

    def test_malformed_appearance_descriptors_refused(self):
        from ghostcaddie.demo.swing_head import compare_swing_appearance
        good={'lab_median':[100.,128.,128.],'gray_dispersion':3.,'sample_count':12}
        for bad in [None,{}, {**good,'lab_median':[100.,128.]},
                    {**good,'lab_median':[float('nan'),128.,128.]},
                    {**good,'gray_dispersion':float('inf')}, {**good,'gray_dispersion':-1},
                    {**good,'sample_count':True}]:
            with self.assertRaises(ValueError):compare_swing_appearance(good,bad)
            with self.assertRaises(ValueError):compare_swing_appearance(bad,good)


if __name__ == '__main__':
    unittest.main()
