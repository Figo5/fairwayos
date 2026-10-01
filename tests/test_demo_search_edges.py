"""Native target pixels remain matchable when only the search margin crosses an edge."""
import unittest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None


@unittest.skipIf(cv2 is None, 'native matching requires cv2 and numpy')
class SearchEdgeTests(unittest.TestCase):
    def setUp(self):
        from ghostcaddie.demo.config import TargetOptions
        from ghostcaddie.demo.worker import _match
        self.match = _match
        self.options = TargetOptions(template_size=13, search_radius=40,
                                     min_score=0.97, max_jump_px=60)
        self.template = np.random.default_rng(123).integers(
            30, 231, (13, 13, 3), dtype=np.uint8)

    def image_at(self, center):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        x, y = center
        x0, y0 = x - 6, y - 6
        x1, y1 = x0 + 13, y0 + 13
        left, top, right, bottom = max(0, x0), max(0, y0), min(64, x1), min(64, y1)
        image[top:bottom, left:right] = self.template[top-y0:bottom-y0, left-x0:right-x0]
        return image

    def test_visible_target_moves_along_every_edge_and_corner(self):
        cases = [((8, 32), (9, 32)), ((55, 32), (54, 32)),
                 ((32, 8), (32, 9)), ((32, 55), (32, 54)),
                 ((8, 8), (9, 9)), ((55, 8), (54, 9)),
                 ((8, 55), (9, 54)), ((55, 55), (54, 54))]
        for previous, destination in cases:
            with self.subTest(previous=previous, destination=destination):
                point, score, reason = self.match(self.image_at(destination), previous,
                                                  self.template, self.options)
                self.assertEqual(point, tuple(float(v) for v in destination))
                self.assertGreater(score, 0.99)
                self.assertEqual(reason, '')

    def test_partially_visible_object_cannot_supply_a_whole_template(self):
        point, _score, reason = self.match(self.image_at((3, 32)), (8, 32),
                                          self.template, self.options)
        self.assertIsNone(point)
        self.assertIn('score_below_min', reason)

    def test_clamping_does_not_remove_the_jump_bound(self):
        from dataclasses import replace
        point, _score, reason = self.match(self.image_at((35, 32)), (8, 32),
                                          self.template, replace(self.options, max_jump_px=5))
        self.assertIsNone(point)
        self.assertIn('step_exceeds_max_jump', reason)

    def test_missing_target_near_edge_does_not_leave_a_marker(self):
        point, _score, reason = self.match(np.zeros((64, 64, 3), dtype=np.uint8),
                                          (8, 32), self.template, self.options)
        self.assertIsNone(point)
        self.assertEqual(reason, 'search_window_is_flat_low_texture')

    def test_seed_patch_still_requires_visible_target_pixels(self):
        from ghostcaddie.demo.worker import _patch
        self.assertIsNone(_patch(self.image_at((3, 32)), 3, 32, 13))
        self.assertIsNotNone(_patch(self.image_at((8, 32)), 8, 32, 13))

    def test_clipped_search_cannot_keep_context_after_target_center_disappears(self):
        from dataclasses import replace
        image = self.image_at((9, 32))
        # The surrounding texture is unchanged and can still match very strongly.
        # The physical target's central pixels have disappeared into flat background.
        image[29:36, 6:13] = 128
        point, score, reason = self.match(image, (8, 32), self.template,
                                          replace(self.options, min_score=0.5))
        self.assertIsNone(point)
        self.assertGreater(score, 0.5)
        self.assertEqual(reason, 'clipped_search_center_evidence_failed')

    def test_interior_search_retains_classic_matching_without_edge_guard(self):
        from unittest.mock import patch
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        image[75:88, 75:88] = self.template
        with patch('ghostcaddie.demo.worker._clipped_match_has_center_evidence',
                   side_effect=AssertionError('interior search invoked edge guard')):
            point, score, reason = self.match(image, (80, 80), self.template, self.options)
        self.assertEqual(point, (81.0, 81.0))
        self.assertGreater(score, 0.99)
        self.assertEqual(reason, '')

    def test_small_grayscale_templates_require_contrast_at_the_actual_center(self):
        from ghostcaddie.demo.worker import _clipped_match_has_center_evidence
        for size in (5, 7, 13):
            with self.subTest(size=size):
                template = np.zeros((size, size), dtype=np.uint8)
                half = size // 2
                template[half-1:half+2, half-1:half+2] = 200
                self.assertTrue(_clipped_match_has_center_evidence(template, template.copy()))
                self.assertFalse(_clipped_match_has_center_evidence(template, np.zeros_like(template)))
