import unittest

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional research dependency.
    cv2 = None

from ghostcaddie.video.research_ball import ResearchBallTracker, SeededBallTracker


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
        # Pixel-index mean of the 2x2 temporal component: x=32.5.
        self.assertEqual(candidates[0].center[0], 32.5)
        self.assertEqual(candidates[0].center[1], 26.5)
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
        self.assertEqual(result.items[0].center[0], 32.0)
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

    def test_rejects_non_finite_or_coerced_tracking_bounds(self):
        import math

        invalid = (
            {"max_step_pixels": math.nan},
            {"max_step_pixels": math.inf},
            {"max_step_pixels": True},
            {"max_gap_frames": 1.5},
            {"max_gap_frames": True},
            {"min_pixels": 1.5},
            {"min_pixels": True},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ResearchBallTracker(**kwargs)

    def test_component_centroid_uses_documented_pixel_index_convention(self):
        """Documented convention: a component centroid is the mean of the
        member pixels' integer indices (OpenCV's
        ``connectedComponentsWithStats`` convention, verified identical on
        opencv-python-headless 4.9, 4.12, and 5.0 on this fixture): a 4x4
        block spanning columns 12..15, rows 24..27 centers at (13.5, 25.5).
        This is the pixel-index mean, NOT a +0.5 pixel-center offset; the
        renderer converts to drawing pixels with ``int(round(...))``."""
        image = np.zeros((60, 80, 3), dtype=np.uint8)
        image[24:28, 12:16] = 230

        candidates = ResearchBallTracker(min_confidence=0.35, max_step_pixels=20).extract_candidates(image)
        self.assertEqual(len(candidates), 1)
        # Pixel-index mean of columns 12..15 / rows 24..27.
        self.assertEqual(candidates[0].center, (13.5, 25.5))

    def test_rejects_static_bottom_logo_and_tracks_moving_ball(self):
        frames = []
        for x in (12, 16, 20):
            image = np.zeros((60, 80, 3), dtype=np.uint8)
            image[52:58, 66:76] = 255  # persistent lower-scene logo
            image[24:28, x:x + 4] = 230  # translating ball candidate
            frames.append(image)

        result = ResearchBallTracker(min_confidence=0.35, max_step_pixels=20).track(frames)

        centers = [item.center for item in result.items]
        # Pixel-index mean: 4x4 block at columns x..x+3 centers on x+1.5.
        self.assertEqual(centers[0], (13.5, 25.5))
        self.assertEqual(centers[1], (17.5, 25.5))
        self.assertEqual(centers[2], (21.5, 25.5))
        self.assertTrue(all(item.provenance in ("candidate", "tracked") for item in result.items))


class SeededBallTrackerTests(unittest.TestCase):
    """ROI-seeded appearance/geometry tracker: human-seeded yellow ball only."""

    def _yellow(self, image, cx, cy, r=3):
        # Symmetric (2r+1)-wide block whose pixel-index mean is exactly (cx, cy).
        image[int(cy) - r:int(cy) + r + 1, int(cx) - r:int(cx) + r + 1] = (250, 235, 60)

    def _frames(self, xs, size=(60, 120), r=3):
        frames = []
        for x in xs:
            image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
            self._yellow(image, x, 30, r)
            frames.append(image)
        return frames

    def test_tracks_seeded_yellow_ball_within_explicit_roi(self):
        frames = self._frames([40, 42, 44, 46])

        result = SeededBallTracker(roi=(20, 15, 90, 50)).track(
            frames, seed_frame_index=0, seed_point=(40.0, 30.0)
        )

        self.assertEqual(result.provenance, "research_candidate")
        self.assertFalse(result.production_eligible)
        centers = [item.center for item in result.items]
        self.assertEqual(centers[0], (40.0, 30.0))
        self.assertEqual(centers[3], (46.0, 30.0))
        self.assertTrue(all(item.provenance in ("seeded", "tracked") for item in result.items))

    def test_seed_point_must_lie_inside_required_roi(self):
        image = self._frames([10])[0]
        with self.assertRaises(ValueError):
            SeededBallTracker(roi=(20, 15, 90, 50)).track(
                [image], seed_frame_index=0, seed_point=(10.0, 30.0)
            )
        with self.assertRaises(ValueError):
            SeededBallTracker(roi=(20, 15, 90, 50)).track(
                [image], seed_frame_index=5, seed_point=(40.0, 30.0)
            )

    def test_invalid_roi_is_rejected(self):
        for roi in (None, (30, 30, 20, 40), (0, 0, 5, 0), "bad"):
            with self.subTest(roi=roi):
                with self.assertRaises(ValueError):
                    SeededBallTracker(roi=roi)

    def test_fails_closed_when_ball_leaves_roi_or_disappears(self):
        frames = self._frames([40, 42, 110])  # last frame: ball outside ROI

        result = SeededBallTracker(roi=(20, 15, 90, 50), max_gap_frames=2).track(
            frames, seed_frame_index=0, seed_point=(40.0, 30.0)
        )

        self.assertIsNone(result.items[2].center)
        self.assertEqual(result.items[2].provenance, "unavailable")
        # Adjacent frames after the last observation: gap is zero.
        self.assertEqual(result.longest_gap, 0)

    def test_fails_closed_on_ambiguous_two_yellow_blobs(self):
        frames = self._frames([40, 42])
        self._yellow(frames[1], 52, 30)  # second, equally plausible blob in frame 1

        result = SeededBallTracker(roi=(20, 15, 90, 50), max_step_pixels=15).track(
            frames, seed_frame_index=0, seed_point=(40.0, 30.0)
        )

        self.assertIsNone(result.items[1].center)
        self.assertEqual(result.items[1].provenance, "unavailable")
        self.assertIn("ambiguous", result.items[1].warnings)

    def test_does_not_relink_after_occlusion_longer_than_gap_bound(self):
        frames = [np.zeros((60, 120, 3), dtype=np.uint8) for _ in range(4)]
        frames[0] = self._frames([40])[0]
        self._yellow(frames[3], 46, 30)

        result = SeededBallTracker(roi=(20, 15, 90, 50), max_gap_frames=1).track(
            frames, seed_frame_index=0, seed_point=(40.0, 30.0)
        )

        self.assertIsNone(result.items[3].center)
        self.assertEqual(result.items[3].provenance, "unavailable")

    def test_brief_one_frame_gap_does_not_relink_to_ambiguous_second_blob(self):
        """A single unavailable frame must not let the track silently adopt a
        different blob: when the ball reappears after a one-frame gap and two
        equally plausible blobs are within the (gap-scaled) step bound, the
        tracker must fail closed instead of guessing."""
        frames = self._frames([40, 44], size=(60, 140))
        frames.insert(1, np.zeros((60, 140, 3), dtype=np.uint8))
        self._yellow(frames[2], 56, 30)  # second, equally plausible blob after the gap

        result = SeededBallTracker(roi=(20, 15, 90, 50), max_step_pixels=15,
                                   max_gap_frames=2).track(
            frames, seed_frame_index=0, seed_point=(40.0, 30.0))

        self.assertIsNone(result.items[1].center)
        self.assertEqual(result.items[1].provenance, "unavailable")
        self.assertIsNone(result.items[2].center)
        self.assertEqual(result.items[2].provenance, "unavailable")
        self.assertIn("ambiguous", result.items[2].warnings)

    def test_requires_opencv_for_hsv_appearance_model(self):
        """The HSV appearance model must fail closed without OpenCV rather
        than silently treating RGB values as pseudo-HSV."""
        import ghostcaddie.video.research_ball as rb

        frames = self._frames([40, 42])
        original = rb.cv2
        rb.cv2 = None
        try:
            with self.assertRaises(RuntimeError) as ctx:
                SeededBallTracker(roi=(20, 15, 90, 50)).track(
                    frames, seed_frame_index=0, seed_point=(40.0, 30.0)
                )
            self.assertIn("OpenCV", str(ctx.exception))
        finally:
            rb.cv2 = original

    def test_rejects_seed_moving_faster_than_bound(self):
        frames = self._frames([40, 80])

        result = SeededBallTracker(roi=(20, 15, 110, 50), max_step_pixels=10).track(
            frames, seed_frame_index=0, seed_point=(40.0, 30.0)
        )

        self.assertIsNone(result.items[1].center)
        self.assertEqual(result.items[1].provenance, "unavailable")

    def test_appearance_model_is_fixed_at_seed_not_per_frame_tuned(self):
        # A differently-colored blob near the trajectory must not be adopted;
        # the true ball must still be tracked with the seed appearance model.
        frames = self._frames([40, 42])
        frames[1][26:34, 52:60] = (60, 220, 60)  # green patch, wrong appearance

        result = SeededBallTracker(roi=(20, 15, 90, 50)).track(
            frames, seed_frame_index=0, seed_point=(40.0, 30.0)
        )

        self.assertIsNotNone(result.items[1].center)
        self.assertEqual(result.items[1].center, (42.0, 30.0))


    def test_tracks_large_ball_on_realistic_grass_background(self):
        """Regression for the real-clip (pexels_6573644 frames 72-75) failure.

        The real ball is a large (~70 px diameter) yellow disc moving ~45
        px/frame over sunlit green rough. The per-pixel seed color-distance
        mask (RGB distance <= appearance_tolerance) matches the sunlit grass
        too: at the real seed the ball RGB is (121, 117, 39) and the grass
        (95, 126, 62), a distance of ~0.08 — far below the 0.35 tolerance.
        The mask therefore forms one background-sized component (>5100 px
        cap) and every frame returns 'unavailable'. The fix extracts
        candidate components in the fixed seed-HSV space (hue/saturation are
        stable and discriminative: ball hue ~27-28 vs grass ~50) instead of
        per-pixel RGB distance. This synthetic reproduces that geometry:
        a 68 px disc moving 45 px/frame over textured green rough.
        """
        rng = np.random.default_rng(7)
        ball_rgb = np.array([121, 117, 39], dtype=np.uint8)
        grass_rgb = np.array([95, 126, 62], dtype=np.uint8)
        frames = []
        centers = [400.0, 357.0, 314.0, 273.0]
        for cx in centers:
            image = np.zeros((400, 480, 3), dtype=np.uint8)
            # Textured rough background (spatially varying, sunlit).
            base = np.tile(grass_rgb, (400, 480, 1)).astype(np.int16)
            noise = rng.integers(-14, 15, size=(400, 480, 1), dtype=np.int16)
            image = np.clip(base + noise, 0, 255).astype(np.uint8)
            yy, xx = np.mgrid[0:400, 0:480]
            disc = (xx - cx) ** 2 + (yy - 200.0) ** 2 <= 28.0 ** 2
            image[disc] = ball_rgb
            frames.append(image)

        result = SeededBallTracker(
            roi=(40, 40, 460, 360), max_step_pixels=100, search_radius=24,
        ).track(frames, seed_frame_index=0, seed_point=(centers[0], 200.0))

        self.assertEqual(result.items[0].center, (400.0, 200.0))
        for position, expected_x in enumerate(centers[1:], start=1):
            self.assertIsNotNone(
                result.items[position].center,
                f"frame {position} unavailable: {result.items[position].warnings}",
            )
            self.assertAlmostEqual(result.items[position].center[0], expected_x, delta=2.0)
            self.assertAlmostEqual(result.items[position].center[1], 200.0, delta=2.0)
            self.assertEqual(result.items[position].provenance, "tracked")

    def test_ambiguous_two_blobs_still_fails_closed_with_hsv_extraction(self):
        """The HSV candidate extraction must preserve the ambiguity fail-closed
        contract: two equally plausible seed-colored blobs in the window still
        yield 'unavailable', not a guess."""
        frames = self._frames([40, 42])
        self._yellow(frames[1], 52, 30)

        result = SeededBallTracker(roi=(20, 15, 90, 50), max_step_pixels=15).track(
            frames, seed_frame_index=0, seed_point=(40.0, 30.0)
        )

        self.assertIsNone(result.items[1].center)
        self.assertIn("ambiguous", result.items[1].warnings)

    def test_float_frames_are_rejected(self):
        """Contract: frames must be uint8. Float frames (e.g. [0,1] normalized
        arrays) are explicitly rejected rather than silently collapsed by the
        uint8 cast in _to_hsv, which would zero HSV values and corrupt the
        appearance model. Fail closed, consistent with existing callers that
        all pass uint8 frames."""
        base = self._frames([40])[0]
        float_frame = base.astype(np.float64) / 255.0
        with self.assertRaises(ValueError):
            SeededBallTracker(roi=(20, 15, 90, 50)).track(
                [float_frame], seed_frame_index=0, seed_point=(40.0, 30.0)
            )
        float32_frame = base.astype(np.float32)
        with self.assertRaises(ValueError):
            SeededBallTracker(roi=(20, 15, 90, 50)).track(
                [float32_frame], seed_frame_index=0, seed_point=(40.0, 30.0)
            )


if __name__ == "__main__":
    unittest.main()
