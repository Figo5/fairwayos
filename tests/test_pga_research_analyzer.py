"""Focused RED-first tests for the reusable local PGA research analyzer.

Module under test: ghostcaddie.video.pga_research_analyzer
(pure bookkeeping/detection logic; no network, no cloud, no production
pipeline). Written before the module existed (RED), per the delegated
task contract:

- motion displacement only between the current observation and the previous
  VALID observation, and only when they are consecutive in the sampled
  cadence (no bridging gaps);
- duplicate frame pairs are recorded and excluded from success counts;
- trails clear on any gap (gaps stay gaps);
- every provenance/diagnostics payload carries research_only=true,
  ground_truth=false, production_eligible=false;
- split-screen (frozen panel) detection flags synthetic two-panel footage
  and leaves clean single-view footage unflagged;
- identity segments reset across cuts.
"""
import math
import unittest

import numpy as np

from ghostcaddie.video.pga_research_analyzer import (
    MotionBookkeeper,
    SegmentTracker,
    TrailAccumulator,
    build_provenance,
    detect_split_screen,
)


class MotionIntervalTests(unittest.TestCase):
    def test_consecutive_observations_compute_displacement(self):
        book = MotionBookkeeper(expected_step=2)
        book.observe(source_frame=503, x=465.7, y=300.3, conf=0.75, state="detected", source="brightblob_flight")
        obs = book.observe(source_frame=505, x=463.0, y=290.0, conf=0.75, state="detected", source="brightblob_flight")
        self.assertIsNotNone(obs.prior_valid_source_frame)
        self.assertEqual(obs.prior_valid_source_frame, 503)
        self.assertEqual(obs.source_frame_delta, 2)
        self.assertAlmostEqual(obs.displacement_px, math.hypot(463.0 - 465.7, 290.0 - 300.3))

    def test_gap_clears_displacement_and_never_bridges(self):
        book = MotionBookkeeper(expected_step=2)
        book.observe(source_frame=503, x=465.7, y=300.3, conf=0.75, state="detected", source="brightblob_flight")
        book.observe(source_frame=505, x=None, y=None, conf=0.0, state="unavailable", source="no_confident_observation")
        obs = book.observe(source_frame=507, x=459.5, y=280.0, conf=0.75, state="detected", source="brightblob_flight")
        # prior valid observation is recorded for audit, but the interval spans
        # a gap (delta 4 != expected step 2) so displacement must be None
        self.assertEqual(obs.prior_valid_source_frame, 503)
        self.assertEqual(obs.source_frame_delta, 4)
        self.assertIsNone(obs.displacement_px)

    def test_duplicate_excluded_from_valid_chain_and_counts(self):
        book = MotionBookkeeper(expected_step=2)
        book.observe(source_frame=499, x=474.9, y=336.1, conf=0.8, state="detected", source="brightblob_flight")
        dup = book.record_duplicate(source_frame=501, duplicate_of=499)
        self.assertEqual(dup.state, "duplicate")
        self.assertEqual(dup.duplicate_of, 499)
        obs = book.observe(source_frame=503, x=465.7, y=300.3, conf=0.75, state="detected", source="brightblob_flight")
        # previous VALID observation is f499; f501 was a duplicate and must not
        # become part of the motion chain
        self.assertEqual(obs.prior_valid_source_frame, 499)
        self.assertEqual(obs.source_frame_delta, 4)
        self.assertIsNone(obs.displacement_px)
        counts = book.state_counts()
        self.assertEqual(counts.get("detected", 0), 2)
        self.assertEqual(counts.get("duplicate", 1), 1)
        self.assertNotIn(501, book.valid_source_frames())

    def test_first_observation_has_no_motion(self):
        book = MotionBookkeeper(expected_step=2)
        obs = book.observe(source_frame=341, x=556.0, y=623.0, conf=0.8, state="detected", source="brightblob_tee")
        self.assertIsNone(obs.prior_valid_source_frame)
        self.assertIsNone(obs.displacement_px)


class TrailTests(unittest.TestCase):
    def test_trail_clears_on_gap(self):
        trail = TrailAccumulator()
        trail.append(491, 504.2, 444.2)
        trail.append(495, 487.4, 382.4)
        self.assertEqual(len(trail.points()), 2)
        trail.break_gap()  # an unavailable observation clears the trail
        self.assertEqual(len(trail.points()), 0)
        trail.append(499, 474.9, 336.1)
        self.assertEqual(len(trail.points()), 1)


class ProvenanceFlagTests(unittest.TestCase):
    def test_flags_always_present_and_correct(self):
        prov = build_provenance(
            source_path="in.mp4",
            output_dir="out",
            modes={"pose": "automatic", "ball": "assisted", "clubhead": "assisted"},
        )
        self.assertTrue(prov["research_only"])
        self.assertFalse(prov["ground_truth"])
        self.assertFalse(prov["production_eligible"])
        self.assertEqual(prov["modes"]["ball"], "assisted")

    def test_assisted_never_labeled_automatic(self):
        prov = build_provenance(
            source_path="in.mp4",
            output_dir="out",
            modes={"pose": "automatic", "ball": "assisted_seed", "clubhead": "unavailable"},
        )
        self.assertNotEqual(prov["modes"]["ball"], "automatic")
        self.assertEqual(prov["modes"]["clubhead"], "unavailable")


class SplitScreenTests(unittest.TestCase):
    def test_frozen_right_panel_is_detected(self):
        rng = np.random.default_rng(7)
        frames = []
        base_left = rng.normal(128, 30, (240, 320, 3)).astype(np.float32)
        static_right = rng.normal(90, 25, (240, 160, 3)).astype(np.float32)
        for i in range(6):
            left = np.clip(base_left + rng.normal(0, 18, base_left.shape), 0, 255)
            right = np.clip(static_right + rng.normal(0, 1.0, static_right.shape), 0, 255)
            full = np.concatenate([left.astype(np.uint8), right.astype(np.uint8)], axis=1)
            frames.append(full)
        result = detect_split_screen(frames)
        self.assertTrue(result["split_screen_detected"])
        self.assertEqual(result["live_panel"], "left")
        self.assertIsNotNone(result["seam_x"])
        # frame is 480 wide; the frozen panel occupies x=320..480, so the seam
        # must land near 320 (2/3 of width)
        self.assertGreater(result["seam_x"], 240)
        self.assertLess(result["seam_x"], 400)

    def test_clean_single_view_not_flagged(self):
        rng = np.random.default_rng(11)
        frames = []
        base = rng.normal(120, 30, (240, 320, 3)).astype(np.float32)
        for i in range(6):
            fr = np.clip(base + rng.normal(0, 18, base.shape), 0, 255)
            frames.append(fr.astype(np.uint8))
        result = detect_split_screen(frames)
        self.assertFalse(result["split_screen_detected"])
        self.assertIsNone(result["live_panel"])


class SegmentIdentityTests(unittest.TestCase):
    def test_segment_resets_across_cut(self):
        seg = SegmentTracker()
        s1 = seg.current()
        seg.on_cut()
        s2 = seg.current()
        seg.on_cut()
        s3 = seg.current()
        self.assertEqual(s1, 1)
        self.assertEqual(s2, 2)
        self.assertEqual(s3, 3)
        self.assertTrue(s2 > s1 and s3 > s2)

    def test_no_cut_keeps_segment(self):
        seg = SegmentTracker()
        self.assertEqual(seg.current(), 1)
        self.assertEqual(seg.current(), 1)


if __name__ == "__main__":
    unittest.main()