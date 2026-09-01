import unittest

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional research dependency.
    cv2 = None

from ghostcaddie.video.research_ball import ResearchBallTracker


class ResearchBallTrackerTests(unittest.TestCase):
    def test_roi_and_context_cues_prefer_candidate_on_green_away_from_golfer(self):
        image = np.full((60, 90, 3), 70, dtype=np.uint8)
        image[5:8, 10:13] = 245  # bright golfer-area distractor
        image[42:45, 66:69] = 185  # dimmer candidate on the green

        result = ResearchBallTracker(min_confidence=0.45).track(
            [image], contexts=[{"golfer_bbox": (0, 0, 30, 25), "green_bbox": (45, 30, 89, 59)}]
        )

        self.assertIsNotNone(result.items[0].center)
        self.assertAlmostEqual(result.items[0].center[0], 67.0, places=1)
        self.assertAlmostEqual(result.items[0].center[1], 43.0, places=1)

    def test_multiscale_extraction_reports_scale_and_restricts_to_roi(self):
        image = np.full((80, 100, 3), 90, dtype=np.uint8)
        image[20:24, 15:19] = 190
        image[55:58, 75:78] = 190

        candidates = ResearchBallTracker(min_confidence=0.3).extract_candidates(
            image, roi=(60, 45, 90, 70)
        )

        self.assertTrue(candidates)
        self.assertTrue(all(60 <= c.center[0] < 90 and 45 <= c.center[1] < 70 for c in candidates))
        self.assertTrue(any(c.scale != 1.0 for c in candidates))
        self.assertTrue(all(c.provenance == "research_candidate" for c in candidates))

    def test_temporal_differencing_can_surface_candidate_below_static_contrast(self):
        previous = np.full((48, 64, 3), 95, dtype=np.uint8)
        current = previous.copy()
        current[25:28, 31:34] = 125

        candidates = ResearchBallTracker(min_confidence=0.2).extract_candidates(
            current, previous_image=previous
        )

        self.assertTrue(candidates)
        self.assertAlmostEqual(candidates[0].center[0], 32.0, places=1)
        self.assertAlmostEqual(candidates[0].center[1], 26.0, places=1)
        self.assertIn("temporal_difference", candidates[0].cues)

    def test_context_or_roi_with_no_valid_region_keeps_unavailable_state(self):
        image = np.full((20, 20, 3), 90, dtype=np.uint8)
        result = ResearchBallTracker().track([image], contexts=[{"roi": (30, 30, 40, 40)}])

        self.assertIsNone(result.items[0].center)
        self.assertEqual(result.items[0].provenance, "unavailable")
        self.assertIn("roi_unavailable", result.items[0].warnings)

    def test_detects_compact_neutral_contrast_candidate_below_absolute_white_threshold(self):
        image = np.full((48, 64, 3), 95, dtype=np.uint8)
        image[23:26, 31:34] = 205

        result = ResearchBallTracker(min_confidence=0.7).track([image])

        self.assertIsNotNone(result.items[0].center)
        self.assertAlmostEqual(result.items[0].center[0], 32.0, places=1)
        self.assertAlmostEqual(result.items[0].center[1], 24.0, places=1)

    def test_temporal_change_prefers_moving_candidate_over_static_bright_distractor(self):
        frames = []
        for x in (30, 32, 34):
            image = np.full((48, 80, 3), 95, dtype=np.uint8)
            image[10:13, 8:11] = 225  # static highlight
            image[25:28, x:x + 3] = 185  # moving, lower-contrast candidate
            frames.append(image)

        result = ResearchBallTracker(min_confidence=0.55, max_step_pixels=40).track(frames)

        self.assertIsNotNone(result.items[2].center)
        self.assertAlmostEqual(result.items[2].center[0], 35.0, places=1)

    def test_rejects_full_frame_bright_region_as_unavailable(self):
        image = np.full((32, 48, 3), 255, dtype=np.uint8)

        result = ResearchBallTracker(min_confidence=0.8).track([image])

        self.assertIsNone(result.items[0].center)
        self.assertEqual(result.items[0].provenance, "unavailable")
        self.assertIn("no_candidate", result.items[0].warnings)

    def test_rejects_thin_bright_overlay_as_unavailable(self):
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        image[10:12, 5:43] = 255

        result = ResearchBallTracker(min_confidence=0.8).track([image])

        self.assertIsNone(result.items[0].center)
        self.assertEqual(result.items[0].provenance, "unavailable")

    def test_tracks_confident_candidate_and_preserves_an_explicit_gap(self):
        def frame(x=None):
            image = np.zeros((32, 48, 3), dtype=np.uint8)
            if x is not None:
                image[15:18, x:x + 3] = 255
            return image

        result = ResearchBallTracker(min_confidence=0.8, max_gap_frames=1).track(
            [frame(10), frame(), frame(12)], frame_indices=[0, 1, 2]
        )

        self.assertEqual(result.provenance, "research_candidate")
        self.assertEqual(result.track_id, "ball-0")
        self.assertEqual([item.frame_index for item in result.items], [0, 1, 2])
        self.assertEqual(result.items[0].provenance, "candidate")
        self.assertIsNone(result.items[1].center)
        self.assertEqual(result.items[1].provenance, "unavailable")
        self.assertIn("gap", result.items[1].warnings)
        self.assertEqual(result.items[2].provenance, "tracked")
        self.assertEqual(result.longest_gap, 1)
        self.assertFalse(result.production_eligible)

    def test_scales_continuity_bound_for_explicit_frame_gaps(self):
        def frame(x=None):
            image = np.zeros((32, 64, 3), dtype=np.uint8)
            if x is not None:
                image[15:18, x:x + 3] = 255
            return image

        result = ResearchBallTracker(min_confidence=0.8, max_step_pixels=20).track(
            [frame(10), frame(40)], frame_indices=[0, 2]
        )

        self.assertEqual([item.provenance for item in result.items], ["candidate", "tracked"])
        self.assertEqual(result.items[1].center, (41.0, 16.0))
        self.assertEqual(result.longest_gap, 1)

    def test_does_not_bridge_gap_longer_than_configured_tracking_bound(self):
        def frame(x=None):
            image = np.zeros((32, 64, 3), dtype=np.uint8)
            if x is not None:
                image[15:18, x:x + 3] = 255
            return image

        result = ResearchBallTracker(min_confidence=0.8, max_gap_frames=1, max_step_pixels=20).track(
            [frame(10), frame(), frame(), frame(13)], frame_indices=[0, 1, 2, 3]
        )

        self.assertIsNone(result.items[3].center)
        self.assertEqual(result.items[3].provenance, "unavailable")
        self.assertIn("continuity_break", result.items[3].warnings)
        self.assertEqual(result.longest_gap, 2)

    @unittest.skipUnless(cv2 is not None, "requires optional OpenCV research dependency")
    def test_continuity_prefers_nearby_fallback_over_distant_circle_proposal(self):
        def frame(ball_x, include_circle):
            image = np.zeros((120, 220, 3), dtype=np.uint8)
            image[55:58, ball_x:ball_x + 3] = 255
            if include_circle:
                cv2.circle(image, (170, 30), 12, (255, 255, 255), -1)
            return image

        result = ResearchBallTracker(
            min_confidence=0.2, max_gap_frames=0, max_step_pixels=20,
            exclude_bottom_fraction=0,
        ).track([frame(20, False), frame(24, True), frame(28, True)])

        self.assertEqual(
            [item.center for item in result.items],
            [(21.0, 56.0), (25.0, 56.0), (29.0, 56.0)],
        )
        self.assertTrue(all(item.provenance in ("candidate", "tracked") for item in result.items))

    def test_rejects_non_positive_max_step_pixels(self):
        with self.assertRaises(ValueError):
            ResearchBallTracker(max_step_pixels=0)

    def test_rejects_static_bottom_logo_and_tracks_moving_ball(self):
        frames = []
        for x in (12, 16, 20):
            image = np.zeros((60, 80, 3), dtype=np.uint8)
            image[52:58, 66:76] = 255  # persistent lower-scene logo
            image[24:28, x:x + 4] = 230  # translating ball candidate
            frames.append(image)

        result = ResearchBallTracker(min_confidence=0.35, max_step_pixels=20).track(frames)

        centers = [item.center for item in result.items]
        self.assertEqual(centers[0], (13.5, 25.5))
        self.assertEqual(centers[1], (17.5, 25.5))
        self.assertEqual(centers[2], (21.5, 25.5))
        self.assertTrue(all(item.provenance in ("candidate", "tracked") for item in result.items))


if __name__ == "__main__":
    unittest.main()
