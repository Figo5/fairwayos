import unittest

import numpy as np

from ghostcaddie.video.tracer_weak_supervision import (
    detect_graphic_mask,
    extract_tracer_hint,
    search_clean_ball,
    build_tracer_render_filter,
    provenance_flags,
    accept_ball_track,
    BallObservation,
)


class TracerWeakSupervisionTests(unittest.TestCase):
    def test_tracer_mask_never_becomes_ball_observation(self):
        image = np.full((40, 60, 3), 80, dtype=np.uint8)
        image[10, 10:30] = (245, 220, 20)  # graphic tracer
        mask = detect_graphic_mask(image)
        hint = extract_tracer_hint(mask)
        self.assertGreater(hint.pixel_count, 0)
        clean = image.copy()
        clean[mask] = 80
        self.assertIsNone(search_clean_ball(clean, graphic_mask=mask))

    def test_clean_ball_requires_neutral_compact_motion_supported_pixels(self):
        previous = np.full((50, 70, 3), 80, dtype=np.uint8)
        current = previous.copy()
        current[25:28, 30:33] = (210, 210, 205)
        mask = np.zeros((50, 70), dtype=bool)
        observation = search_clean_ball(
            current, previous_image=previous, graphic_mask=mask,
            max_radius=8,
        )
        self.assertIsNotNone(observation)
        self.assertAlmostEqual(observation.x, 31.0, places=1)
        self.assertAlmostEqual(observation.y, 26.0, places=1)
        self.assertTrue(observation.motion_supported)

    def test_static_logo_is_masked_and_tracer_hint_is_pseudo_only(self):
        image = np.full((40, 60, 3), 80, dtype=np.uint8)
        image[35:39, 2:12] = (230, 220, 20)
        mask = detect_graphic_mask(image, logo_region=(0, 30, 20, 40))
        self.assertTrue(mask[36, 5])
        self.assertEqual(extract_tracer_hint(mask).provenance, "tracer_pseudo_hint")

    def test_malformed_regions_and_previous_frames_fail_closed(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            detect_graphic_mask(image, logo_region=(1, 2, 3))
        with self.assertRaises(ValueError):
            detect_graphic_mask(image, ui_regions=[(1.5, 2, 3, 4)])
        with self.assertRaises(ValueError):
            detect_graphic_mask(image, previous_image=np.zeros((9, 10, 3), dtype=np.uint8))

        render = build_tracer_render_filter(
            width=60, height=40,
            tracer_points=[(10, 10), (20, 10)],
            ball_points=[(31, 26)],
        )
        self.assertIn("color=magenta", render)
        self.assertIn("color=lime", render)
        self.assertIn("w=30", render)
        self.assertNotIn("drawtext", render)
        self.assertNotIn("ground_truth=true", render)
        with self.assertRaises(ValueError):
            build_tracer_render_filter(width=60, height=40, tracer_points=[("10", 10)])

    def test_temporal_track_requires_consecutive_motion_consistency(self):
        good = [BallObservation(10.0, 10.0, 3.0, 0.9, True),
                BallObservation(14.0, 12.0, 3.1, 0.9, True),
                BallObservation(19.0, 15.0, 3.0, 0.8, True)]
        self.assertTrue(accept_ball_track(good, frame_indices=[4, 5, 6],
                                          width=100, height=100, min_frames=3))
        self.assertFalse(accept_ball_track(
            [BallObservation(10.0, 10.0, 3.0, 0.9, True)] * 3,
            frame_indices=[4, 5, 6], width=100, height=100, min_frames=3))

    def test_temporal_track_rejects_jump_and_invalid_provenance(self):
        jump = [BallObservation(10.0, 10.0, 3.0, 0.9, True),
                BallObservation(90.0, 90.0, 3.0, 0.9, True),
                BallObservation(91.0, 91.0, 3.0, 0.9, True)]
        self.assertFalse(accept_ball_track(jump, frame_indices=[4.5, 5.5, 6.5],
                                           width=100, height=100, min_frames=3))
        invalid = BallObservation(14.0, 12.0, 3.0, 0.9, False,
                                  provenance="tracer_pseudo_hint")
        self.assertFalse(accept_ball_track([jump[0], invalid, jump[2]],
                                           frame_indices=[4, 5, 6], width=100, height=100,
                                           min_frames=3))

        self.assertEqual(provenance_flags(), {
            "pseudo_label": True,
            "ground_truth": False,
            "research_only": True,
            "production_eligible": False,
        })


if __name__ == "__main__":
    unittest.main()
