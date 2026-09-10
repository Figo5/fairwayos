"""Focused tests for MilRegionTracker: a seed-conditioned REGION tracker backed
by OpenCV 5.0.0 TrackerMIL with an appearance-verification gate.

Written FIRST (RED): the module ghostcaddie.video.region_tracker_mil does not
exist yet. Distinctness from lk_point/region_template/reacquire_color: the seed
is a BOX (appearance region, not a point); identity is anchored to a FROZEN
seed patch (MIL's internal model may adapt, but every emitted frame must
re-verify against seed appearance); MIL drift on background fails CLOSED;
gaps never propagate positions; a new segment_id starts only after a strictly
verified bounded-window reacquisition; runtime is bounded by a wall-clock
budget checked between frames.
"""
import math
import unittest

import cv2
import numpy as np

from ghostcaddie.video.region_tracker_mil import MilRegionTracker, MilTrackFrame


def textured_frame(h=1080, w=1920, seed=0):
    """Deterministic textured turf-like background (not flat, NCC-meaningful)."""
    rng = np.random.default_rng(seed)
    base = np.full((h, w, 3), (130, 150, 100), np.float32)
    noise = rng.normal(0, 12, (h, w, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def draw_head(img, cx, cy, rx=26, ry=20):
    """Dark rounded 'clubhead' with a bright rim, MIL-friendly."""
    cv2.ellipse(img, (int(cx), int(cy)), (rx, ry), 0, 0, 360, (40, 40, 45), -1)
    cv2.ellipse(img, (int(cx), int(cy)), (rx, ry), 0, 0, 360, (95, 95, 105), 3)


def swing_frames(n=14, start=(620, 730), dx=12, dy=-8, seed0=0, gap_at=None,
                 gap_len=0, hide_after=None, reappear_offset=None):
    """Synthetic swing: object present every frame unless hidden. When
    reappear_offset is set and the hidden period ended, the object is drawn at
    (start + (i)*d + offset) — e.g. a jump FAR from the last tracked spot."""
    h, w = 1080, 1920
    frames, present, truth_pos = [], [], []
    hidden_until = -1
    if gap_at is not None:
        hidden_until = gap_at + gap_len - 1
    for i in range(n):
        img = textured_frame(h, w, seed=seed0 + i)
        x = start[0] + dx * i
        y = start[1] + dy * i
        visible = not (gap_at is not None and gap_at <= i <= hidden_until)
        if hide_after is not None and i > hide_after:
            visible = False
        if reappear_offset is not None and i == hidden_until + 1:
            x += reappear_offset[0]
            y += reappear_offset[1]
        if visible:
            draw_head(img, x, y)
            truth_pos.append((x, y))
        else:
            truth_pos.append(None)
        present.append(visible)
        frames.append(img)
    return frames, present, truth_pos


SEED_BOX = (594.0, 704.0, 52.0, 44.0)  # around (620, 726) ellipse center


class MilSeedFrameContract(unittest.TestCase):
    def test_seed_frame_emits_visible_box_row(self):
        frames, _, _ = swing_frames(n=3)
        t = MilRegionTracker()
        t.init(frames[0], SEED_BOX, seed_source_frame=0)
        rows = t.track(frames)
        self.assertEqual(len(rows), 3)
        r0 = rows[0]
        self.assertIsInstance(r0, MilTrackFrame)
        self.assertEqual(r0.source_frame_index, 0)
        self.assertEqual(r0.bbox, SEED_BOX)
        self.assertEqual(r0.visibility, "visible")
        self.assertEqual(r0.state, "seed")
        self.assertEqual(r0.confidence, 1.0)
        self.assertEqual(r0.segment_id, 1)
        self.assertEqual(r0.provenance, "ai_assisted_seeded_region")
        self.assertEqual(r0.uncertainty_px, 0.0)
        self.assertGreaterEqual(r0.timestamp, 0.0)


class MilSteadyTracking(unittest.TestCase):
    def test_tracks_slow_motion_with_verified_boxes(self):
        frames, present, truth = swing_frames(n=14)
        t = MilRegionTracker()
        t.init(frames[0], SEED_BOX, 0)
        rows = t.track(frames)
        obs = [r for r in rows if r.visibility == "visible"]
        # most frames stay tracked under slow motion
        self.assertGreaterEqual(len(obs), 10, f"only {len(obs)} visible rows")
        for r in obs:
            if r.source_frame_index == 0:
                continue
            self.assertIn(r.state, ("tracked", "reacquired"))
            self.assertGreater(r.confidence, 0.5, f"f{r.source_frame_index} conf")
            self.assertIsNotNone(r.uncertainty_px)
            cx = r.bbox[0] + r.bbox[2] / 2
            cy = r.bbox[1] + r.bbox[3] / 2
            tx, ty = truth[r.source_frame_index]
            err = math.hypot(cx - tx, cy - ty)
            self.assertLess(err, 40, f"f{r.source_frame_index} center err {err:.0f}px")

    def test_frames_list_starts_at_seed_and_indexes_align(self):
        frames, _, _ = swing_frames(n=5)
        t = MilRegionTracker()
        t.init(frames[0], SEED_BOX, 7)
        rows = t.track(frames)
        self.assertEqual([r.source_frame_index for r in rows], [7, 8, 9, 10, 11])


class MilGapSemantics(unittest.TestCase):
    def test_gap_frames_not_invented_and_new_segment_after_reacquire(self):
        # object hidden frames 4-6, reappears near its true path position
        frames, present, truth = swing_frames(n=10, gap_at=4, gap_len=3)
        t = MilRegionTracker()
        t.init(frames[0], SEED_BOX, 0)
        rows = t.track(frames)
        by = {r.source_frame_index: r for r in rows}
        # during the gap: no visible rows, no fabricated boxes
        for i in (4, 5, 6):
            self.assertIsNone(by[i].bbox, f"f{i} invented a box during a gap")
            self.assertNotEqual(by[i].visibility, "visible")
        # after reappearance: verified reacquisition in a NEW segment
        r7 = by[7]
        self.assertEqual(r7.visibility, "visible")
        self.assertEqual(r7.state, "reacquired")
        self.assertGreater(r7.segment_id, 1, "gap must start a new segment")
        cx = r7.bbox[0] + r7.bbox[2] / 2
        cy = r7.bbox[1] + r7.bbox[3] / 2
        tx, ty = truth[7]
        self.assertLess(math.hypot(cx - tx, cy - ty), 40)
        # segment 1 rows are strictly before the reacquired row
        self.assertEqual({r.segment_id for r in rows[:4]}, {1})
        self.assertEqual(r7.segment_id, max(r.segment_id for r in rows))

    def test_reappearance_far_outside_window_fails_closed(self):
        # object reappears FAR from last verified spot AND STAYS far (new
        # position persists): bounded window must refuse to teleport; frames
        # stay unavailable (honest), never guessed
        far = (400, -260)
        frames, present, truth = swing_frames(n=10, gap_at=4, gap_len=2,
                                              reappear_offset=far)
        # keep the object at the far offset for all remaining frames
        for i in range(6, 10):
            img = textured_frame(1080, 1920, seed=1000 + i)
            draw_head(img, 620 + 12 * i + far[0], 730 - 8 * i + far[1])
            frames[i] = img
        t = MilRegionTracker()
        t.init(frames[0], SEED_BOX, 0)
        rows = t.track(frames)
        for r in rows[6:]:
            self.assertNotEqual(r.visibility, "visible",
                                f"f{r.source_frame_index} teleported to far object")
            self.assertIsNone(r.bbox)

    def test_object_gone_forever_never_emits_again(self):
        frames, present, truth = swing_frames(n=14, gap_at=4, gap_len=100,
                                              hide_after=3)
        t = MilRegionTracker(max_lost_frames=4)
        t.init(frames[0], SEED_BOX, 0)
        rows = t.track(frames)
        later = [r for r in rows if r.source_frame_index >= 4]
        for r in later:
            self.assertNotEqual(r.visibility, "visible")
            self.assertIsNone(r.bbox)
        # after enough consecutive losses the track ENDS (no infinite occlusion)
        states = {r.state for r in rows if r.source_frame_index >= 4 + 4}
        self.assertIn("ended", states)


class MilOffFrameAndRuntime(unittest.TestCase):
    def test_object_driven_off_frame_marks_off_frame(self):
        h, w = 1080, 1920
        frames = []
        x, y = 1500.0, 700.0
        for i in range(10):
            img = textured_frame(h, w, seed=200 + i)
            draw_head(img, x, y)
            frames.append(img)
            x += 80  # crosses the right edge around frame 5
        t = MilRegionTracker()
        t.init(frames[0], (1474.0, 674.0, 52.0, 44.0), 0)
        rows = t.track(frames)
        offs = [r for r in rows if r.visibility == "off_frame"]
        self.assertTrue(offs, "expected off_frame marks once the box exits")
        for r in offs:
            self.assertEqual(r.state, "unavailable")
            self.assertEqual(r.warning, "box_left_frame")
        # no visible row may sit mostly outside the frame
        for r in rows:
            if r.visibility == "visible" and r.source_frame_index > 0:
                bx, by, bw, bh = r.bbox
                self.assertGreaterEqual(bx, -1)
                self.assertLessEqual(bx + bw, w + 1)

    def test_runtime_budget_exceeded_ends_track(self):
        frames, _, _ = swing_frames(n=60)
        # budget so small that only the first frame can possibly complete
        t = MilRegionTracker(max_runtime_s=0.0001, max_lost_frames=100)
        t.init(frames[0], SEED_BOX, 0)
        rows = t.track(frames)
        ended = [r for r in rows if r.state == "ended"]
        self.assertTrue(ended, "budget must terminate the track")
        self.assertEqual(ended[-1].warning, "runtime_budget_exceeded")
        self.assertEqual(len(rows), 60, "all frames must still produce rows")
        # at most one frame (the first, already in flight) may be visible
        vis = [r for r in rows if r.visibility == "visible"]
        self.assertLessEqual(len(vis), 2)
        self.assertEqual(vis[0].state, "seed")

    def test_invalid_inputs_rejected(self):
        frames, _, _ = swing_frames(n=3)
        with self.assertRaises(ValueError):
            MilRegionTracker(min_appearance_ncc=1.5)
        with self.assertRaises(ValueError):
            MilRegionTracker(max_runtime_s=-1)
        t = MilRegionTracker()
        with self.assertRaises(RuntimeError):
            t.track(frames)  # init not called
        t.init(frames[0], SEED_BOX, 0)
        with self.assertRaises(ValueError):
            t.init(frames[0], (0.0, 0.0, -5.0, 10.0), 0)  # bad box
        with self.assertRaises(ValueError):
            t.init(frames[0], (1900.0, 0.0, 50.0, 50.0), 0)  # outside frame


class MilDistinctness(unittest.TestCase):
    def test_seed_is_a_box_not_a_point(self):
        # the tracker requires a region: a zero-area box is invalid
        frames, _, _ = swing_frames(n=3)
        t = MilRegionTracker()
        with self.assertRaises(ValueError):
            t.init(frames[0], (594.0, 704.0, 0.0, 44.0), 0)

    def test_provenance_is_seeded_region(self):
        frames, _, _ = swing_frames(n=2)
        t = MilRegionTracker()
        t.init(frames[0], SEED_BOX, 0)
        rows = t.track(frames)
        self.assertTrue(all(r.provenance == "ai_assisted_seeded_region"
                            for r in rows))


class MilSingleSegmentMode(unittest.TestCase):
    """single_segment=True disables reacquisition entirely: when the track is
    lost the tracker emits ONE 'ended' row and every subsequent frame is
    'unavailable' with no box — it never starts segment 2. This eliminates the
    wrong-object body lock BY CONSTRUCTION (no reacquisition to false-lock on),
    which is the only approach that cannot false-accept. Default stays
    multi-segment (reacquisition on)."""

    def test_single_segment_ends_on_first_loss_and_never_reacquires(self):
        # object hidden frames 4-6, reappears at 7. Multi-segment would
        # reacquire into segment 2; single-segment must end at the loss.
        frames, present, truth = swing_frames(n=10, gap_at=4, gap_len=3)
        t = MilRegionTracker(single_segment=True)
        t.init(frames[0], SEED_BOX, 0)
        rows = t.track(frames)
        # exactly one 'ended' row (the loss frame), then all unavailable
        ended = [r for r in rows if r.state == "ended"]
        self.assertEqual(len(ended), 1, f"expected one ended row, got {len(ended)}")
        self.assertIsNone(ended[0].bbox)
        # every frame after the ended frame is unavailable with no box
        eidx = rows.index(ended[0])
        for r in rows[eidx + 1:]:
            self.assertEqual(r.state, "unavailable")
            self.assertIsNone(r.bbox)
        # no reacquisition ever, segment stays 1
        self.assertNotIn("reacquired", [r.state for r in rows])
        self.assertEqual({r.segment_id for r in rows}, {1})

    def test_single_segment_tracks_normally_when_no_loss(self):
        frames, _, _ = swing_frames(n=10)
        t = MilRegionTracker(single_segment=True)
        t.init(frames[0], SEED_BOX, 0)
        rows = t.track(frames)
        vis = [r for r in rows if r.visibility == "visible"]
        self.assertGreaterEqual(len(vis), 8)
        self.assertNotIn("ended", [r.state for r in rows])
        self.assertNotIn("reacquired", [r.state for r in rows])

    def test_default_is_multi_segment(self):
        # default (single_segment=False) must keep reacquiring into segment 2
        frames, _, _ = swing_frames(n=10, gap_at=4, gap_len=3)
        t = MilRegionTracker()
        t.init(frames[0], SEED_BOX, 0)
        rows = t.track(frames)
        r7 = rows[7]
        self.assertEqual(r7.state, "reacquired")
        self.assertGreater(r7.segment_id, 1)

    def test_single_segment_rejects_non_bool(self):
        with self.assertRaises(ValueError):
            MilRegionTracker(single_segment="yes")


if __name__ == "__main__":
    unittest.main()