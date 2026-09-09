"""RED regressions for FairwayOS MIL comparison evidence repair (run 2224f21).

Defects under test, all in the out/mil_run milestone artifacts produced by the
ignored one-off out/render_mil_mp4s.py:

(1) The renderer resized sources to 1200 px wide but drew native-coordinate
    MIL boxes UNSCALED while baseline points WERE scaled -> spatial
    inconsistency (landscape and portrait).
(2) Rendered windows were truncated: 33511561 rendered 145-160 (must be
    145-200 incl), 6541842 rendered 25-51 (must be 25-70 incl), 6541855 was
    not rendered at all (must be 128-156 incl, pre-seed/unavailable frames).
(3) Timestamps used the tracker's 30 fps default instead of the verified
    per-clip fps (33511561=60, 6541855/6541842=25), conflating source time,
    window-relative time and playback time.
(4) mil_summary.json declared transfer window [25, 51] while its 46 per-frame
    rows span [25, 70] -- summaries must reconcile programmatically.
(5) Renderer/reconciler logic was an untested out/ one-off.

All production logic lives in ghostcaddie.video.mil_overlay (reusable, pure,
tested). Honesty contract mirrored from render_reconcile: a RAW tracked frame
with an AI-review verdict of on_body must display REJECTED, never accepted;
unavailable gaps and segment changes never bridge (no trails, ever); the seed
frame is rendered distinctly.
"""
import json
import math
import os
import unittest

import cv2
import numpy as np

from ghostcaddie.video.mil_overlay import (
    classify_mil_display,
    correct_perframe_timestamps,
    h264_encode_command,
    render_flat_frame,
    render_window,
    reconcile_summary,
    scale_factors,
    segment_runs,
    source_timestamp,
    timestamp_at_source_index,
    transform_box,
    transform_point,
)

BG = (30, 30, 30)


def mil_row(idx, *, bbox=None, state="tracked", segment_id=1, warning=None,
            confidence=0.9):
    return {
        "source_frame_index": idx,
        "timestamp": None,
        "bbox": list(bbox) if bbox else None,
        "visibility": "visible" if bbox else "missing",
        "state": state,
        "confidence": confidence,
        "segment_id": segment_id,
        "provenance": "ai_assisted_seeded_region",
        "uncertainty_px": None,
        "warning": warning,
    }


def base_row(idx, *, point=None, state="observed"):
    return {
        "source_frame_index": idx,
        "bbox": ([point[0] - 30, point[1] - 30, 60, 60] if point else None),
        "point": list(point) if point else None,
        "state": state,
        "confidence": 0.9 if point else 0.0,
        "warning": None if point else "reacquire_failed",
    }


def flat(h, w):
    return np.full((h, w, 3), BG, np.uint8)


# ---------------------------------------------------------------------------
# (1) coordinate transform: landscape + portrait, box and point consistent
# ---------------------------------------------------------------------------
class ScaleFactorTests(unittest.TestCase):
    def test_landscape_box_and_point_share_scale(self):
        # 2560x1440 -> 1200x675
        sx, sy = scale_factors(2560, 1440, 1200, 675)
        self.assertAlmostEqual(sx, 1200 / 2560)
        self.assertAlmostEqual(sy, 675 / 1440)
        box = transform_box((500.0, 600.0, 240.0, 260.0), 2560, 1440, 1200, 675)
        self.assertAlmostEqual(box[0], 500 * sx, places=9)
        self.assertAlmostEqual(box[1], 600 * sy, places=9)
        self.assertAlmostEqual(box[2], 240 * sx, places=9)
        self.assertAlmostEqual(box[3], 260 * sy, places=9)
        pt = transform_point((620.0, 730.0), 2560, 1440, 1200, 675)
        self.assertAlmostEqual(pt[0], 620 * sx, places=9)
        self.assertAlmostEqual(pt[1], 730 * sy, places=9)

    def test_portrait_box_and_point_share_scale(self):
        # 1440 wide x 2560 tall -> 1200 x 2133 (rounded)
        sx, sy = scale_factors(1440, 2560, 1200, 2133)
        self.assertAlmostEqual(sx, 1200 / 1440)
        self.assertAlmostEqual(sy, 2133 / 2560)
        box = transform_box((820.0, 1665.0, 135.0, 130.0), 1440, 2560, 1200, 2133)
        self.assertAlmostEqual(box[0], 820 * sx, places=9)
        self.assertAlmostEqual(box[1], 1665 * sy, places=9)
        self.assertAlmostEqual(box[2], 135 * sx, places=9)
        self.assertAlmostEqual(box[3], 130 * sy, places=9)
        pt = transform_point((887.5, 1730.0), 1440, 2560, 1200, 2133)
        self.assertAlmostEqual(pt[0], 887.5 * sx, places=9)
        self.assertAlmostEqual(pt[1], 1730.0 * sy, places=9)

    def test_box_center_maps_to_transformed_center_point(self):
        # the old bug drew boxes unscaled while points were scaled; the box
        # center must land exactly on the transformed center point.
        for nw, nh, ow, oh in ((2560, 1440, 1200, 675), (1440, 2560, 1200, 2133)):
            box = (100.0, 200.0, 300.0, 150.0)
            tb = transform_box(box, nw, nh, ow, oh)
            cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
            tp = transform_point((cx, cy), nw, nh, ow, oh)
            self.assertAlmostEqual(tb[0] + tb[2] / 2, tp[0], places=6)
            self.assertAlmostEqual(tb[1] + tb[3] / 2, tp[1], places=6)

    def test_rounded_output_dims(self):
        box = transform_box((820.0, 1665.0, 135.0, 130.0), 1440, 2560,
                            1200, 2133, round_output=True)
        for v in box:
            self.assertIsInstance(v, int)
        fx = transform_box((820.0, 1665.0, 135.0, 130.0), 1440, 2560, 1200, 2133)
        for rv, fv in zip(box, fx):
            self.assertLessEqual(abs(rv - fv), 1.0)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            transform_box((0, 0, 0, 0), 2560, 1440, 1200, 675)
        with self.assertRaises(ValueError):
            transform_box((0, 0, 10, 10), 2560, 1440, 0, 675)
        with self.assertRaises(ValueError):
            transform_point((1e9, 0), 2560, 1440, 1200, 675)


class DecodedPixelAlignmentTests(unittest.TestCase):
    """The transform must agree with what cv2.resize actually does to pixels."""

    MARKER = (0, 0, 220)  # BGR red-ish

    def _marker_frame(self, nw, nh, box, point):
        img = flat(nh, nw)
        x, y, w, h = (int(v) for v in box)
        img[y:y + h, x:x + w] = self.MARKER
        cv2.circle(img, (int(point[0]), int(point[1])), 40, (0, 220, 0), -1)
        return img

    def _check(self, nw, nh, ow, oh, box, point):
        img = self._marker_frame(nw, nh, box, point)
        resized = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA)
        tb = transform_box(box, nw, nh, ow, oh)
        x0, y0, bw, bh = tb
        # interior of the transformed box must be marker pixels
        x0i, y0i = int(round(x0)) + 3, int(round(y0)) + 3
        x1i, y1i = int(round(x0 + bw)) - 3, int(round(y0 + bh)) - 3
        region = resized[y0i:y1i, x0i:x1i]
        self.assertTrue(region.size, "transformed box empty")
        b_mean = float(region[:, :, 0].mean())
        r_mean = float(region[:, :, 2].mean())
        self.assertGreater(r_mean, 150, "box not on resized marker (red)")
        self.assertLess(b_mean, 90)
        # transformed point must land on the green circle
        tp = transform_point(point, nw, nh, ow, oh)
        px = resized[int(round(tp[1])), int(round(tp[0]))]
        self.assertGreater(int(px[1]), 150, "point not on resized marker (green)")

    def test_landscape_decoded_pixels(self):
        self._check(2560, 1440, 1200, 675, (500.0, 600.0, 240.0, 260.0),
                    (620.0, 730.0))

    def test_portrait_decoded_pixels(self):
        self._check(1440, 2560, 1200, 2133, (820.0, 1665.0, 135.0, 130.0),
                    (887.5, 1730.0))

    def test_render_flat_frame_places_box_on_resized_object(self):
        # end-to-end through the renderer itself, landscape AND portrait
        for nw, nh, ow, oh, box in (
                (2560, 1440, 1200, 675, (500.0, 600.0, 240.0, 260.0)),
                (1440, 2560, 1200, 2133, (820.0, 1665.0, 135.0, 130.0))):
            img = flat(nh, nw)
            x, y, w, h = (int(v) for v in box)
            img[y:y + h, x:x + w] = self.MARKER
            row = mil_row(0, bbox=box, state="tracked")
            out = render_flat_frame(img, mil_row=row, baseline_row=None,
                                    native_w=nw, native_h=nh, out_w=ow,
                                    out_h=oh, source_frame_index=0,
                                    window_start=0, fps=25.0)
            self.assertEqual(out.shape[:2], (oh, ow))
            tb = transform_box(box, nw, nh, ow, oh)
            ix, iy = int(round(tb[0])) + 4, int(round(tb[1])) + 4
            self.assertGreater(int(out[iy, ix, 2]), 150,
                               f"rendered box not on object at {(nw, nh)}")


# ---------------------------------------------------------------------------
# (3) timing: passed source fps, never a 30 default; distinct time bases
# ---------------------------------------------------------------------------
class TimestampTests(unittest.TestCase):
    def test_fps_60_offset_55(self):
        self.assertAlmostEqual(source_timestamp(55, 60.0), 55 / 60.0, places=12)
        self.assertAlmostEqual(timestamp_at_source_index(200, 145, 60.0),
                               55 / 60.0, places=12)

    def test_fps_25(self):
        self.assertAlmostEqual(timestamp_at_source_index(70, 25, 25.0), 1.8,
                               places=12)

    def test_not_the_30fps_default(self):
        for fps in (60.0, 25.0):
            ts = timestamp_at_source_index(200, 145, fps)
            wrong = 55 / 30.0
            self.assertNotAlmostEqual(ts, wrong, places=3)

    def test_correct_perframe_timestamps_rewrites_all_bases(self):
        rows = [mil_row(i) for i in range(145, 201)]  # old ts all offset/30
        out = correct_perframe_timestamps(rows, window_start=145, fps=60.0,
                                          seed_frame=145)
        self.assertEqual(len(out), 56)
        last = out[-1]
        self.assertAlmostEqual(last["timestamp"], 200 / 60.0, places=12)
        self.assertAlmostEqual(last["timestamp_window"], 55 / 60.0, places=12)
        self.assertAlmostEqual(last["timestamp_seed"], 55 / 60.0, places=12)
        first = out[0]
        self.assertAlmostEqual(first["timestamp"], 145 / 60.0, places=12)
        self.assertAlmostEqual(first["timestamp_window"], 0.0, places=12)
        # inputs not mutated
        self.assertIsNone(rows[0]["timestamp"])

    def test_correct_perframe_timestamps_fps25_window_offset(self):
        rows = [mil_row(i) for i in range(25, 71)]
        out = correct_perframe_timestamps(rows, window_start=25, fps=25.0,
                                          seed_frame=25)
        self.assertAlmostEqual(out[-1]["timestamp"], 70 / 25.0, places=12)
        self.assertAlmostEqual(out[-1]["timestamp_window"], 45 / 25.0, places=12)

    def test_correct_perframe_timestamps_validates(self):
        rows = [mil_row(5), mil_row(7)]  # gap
        with self.assertRaises(ValueError):
            correct_perframe_timestamps(rows, window_start=5, fps=25.0)
        with self.assertRaises(ValueError):
            correct_perframe_timestamps([mil_row(5)], window_start=5, fps=0)
        with self.assertRaises(ValueError):
            correct_perframe_timestamps([mil_row(9)], window_start=5, fps=25.0)


# ---------------------------------------------------------------------------
# (2) full-window rendering: (end-start+1) frames incl unavailable
# ---------------------------------------------------------------------------
class FullWindowRenderTests(unittest.TestCase):
    TMP = None

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.TMP = tempfile.mkdtemp(prefix="mil_overlay_test_")
        cls.clip = os.path.join(cls.TMP, "synthetic.mp4")
        wr = cv2.VideoWriter(cls.clip, cv2.VideoWriter_fourcc(*"mp4v"), 25.0,
                             (320, 240))
        for i in range(40):
            img = flat(240, 320)
            img[10, i % 320] = (i, i, i)
            wr.write(img)
        wr.release()

    def _rows(self, start, end):
        rows = []
        for i in range(start, end + 1):
            if i in (12, 13, 14):
                rows.append(mil_row(i, bbox=(50, 50, 40, 40), state="tracked"))
            elif i == 15:
                rows.append(mil_row(i, state="unavailable", warning="reacquire_failed"))
            elif i == 16:
                rows.append(mil_row(i, state="ended", warning="track_terminated"))
            else:
                rows.append(mil_row(i, state="unavailable"))
        return rows

    def test_window_frame_count_inclusive(self):
        res = render_window(self.clip, (10, 25), mil_rows=self._rows(10, 25),
                            out_dir=self.TMP, fps=25.0, prefix="full")
        self.assertEqual(res["frame_count"], 16)  # 25-10+1
        self.assertEqual(res["window"], [10, 25])
        self.assertEqual(len(res["frame_files"]), 16)
        for f in res["frame_files"]:
            self.assertTrue(os.path.exists(f), f)
        names = sorted(os.path.basename(f) for f in res["frame_files"])
        self.assertEqual(names[0], "frame_000001.png")
        self.assertEqual(names[-1], "frame_000016.png")

    def test_truncated_source_raises_instead_of_skipping(self):
        # window extends past decoded frames -> hard error, no silent skip
        with self.assertRaises(RuntimeError):
            render_window(self.clip, (30, 45), mil_rows=self._rows(30, 45),
                          out_dir=self.TMP, fps=25.0, prefix="trunc")

    def test_unavailable_frames_still_rendered_with_state_text(self):
        res = render_window(self.clip, (10, 25), mil_rows=self._rows(10, 25),
                            out_dir=self.TMP, fps=25.0, prefix="unav")
        # frame 15 (local output #6) is unavailable: no box, gray text present
        img = cv2.imread(res["frame_files"][15 - 10])
        self.assertIsNotNone(img)
        for color in ((0, 255, 0), (0, 255, 255), (255, 0, 255)):
            self.assertFalse(np.all(img == np.array(color), axis=2).any(),
                             f"unexpected MIL mark {color} on unavailable frame")


# ---------------------------------------------------------------------------
# (4) summary reconciliation from per-frame records
# ---------------------------------------------------------------------------
def transfer_like_rows():
    rows = [mil_row(25, bbox=(820, 1665, 135, 130), state="seed",
                    segment_id=1, confidence=1.0)]
    for i in range(26, 32):
        rows.append(mil_row(i, bbox=(700 + i, 1500, 135, 130), state="tracked",
                            segment_id=1))
    for i in range(32, 36):
        rows.append(mil_row(i, state="unavailable", segment_id=1,
                            warning="appearance_verification_failed"))
    for i in range(36, 52):
        rows.append(mil_row(i, bbox=(600, 1235, 135, 130), state="tracked",
                            segment_id=2))
    for i in range(52, 57):
        rows.append(mil_row(i, bbox=(610, 1240, 135, 130), state="reacquired",
                            segment_id=3, warning="seed_appearance_reacquired"))
    for i in range(57, 64):
        rows.append(mil_row(i, state="unavailable", segment_id=3))
    for i in range(64, 71):
        rows.append(mil_row(i, state="ended", segment_id=3,
                            warning="track_terminated_no_reacquire"))
    return rows


class SummaryReconcileTests(unittest.TestCase):
    QA = {i: "on_clubhead" for i in range(25, 32)}
    QA.update({i: "on_body" for i in range(36, 52)})

    def test_counts_derived_from_rows(self):
        rows = transfer_like_rows()
        s = reconcile_summary(rows, window=(25, 70), fps=25.0,
                              clip="pexels_6541842", qa_verdicts=self.QA)
        self.assertEqual(s["window_frames"], [25, 70])
        self.assertEqual(s["total_frames"], 46)
        self.assertEqual(s["rows_present"], 46)
        states = s["mil_state_counts"]
        self.assertEqual(states["seed"], 1)
        self.assertEqual(states["tracked"], 22)
        self.assertEqual(states["reacquired"], 5)
        self.assertEqual(states["unavailable"], 11)
        self.assertEqual(states["ended"], 7)
        self.assertEqual(sum(states.values()), 46)
        # display layer: raw tracked frames QA'd on_body are REJECTED, never
        # accepted because the tracker state says tracked
        d = s["display_counts"]
        self.assertEqual(d["on_clubhead"], 6)    # 26-31 (seed shown separately)
        self.assertEqual(d["rejected"], 16)      # 36-51 on_body
        self.assertEqual(d["unreviewed"], 5)     # 52-56
        self.assertEqual(d["unavailable"], 11)
        self.assertEqual(d["ended"], 7)
        # segments: bbox runs 25-31(seg1), 36-51(seg2), 52-56(seg3)
        self.assertEqual(s["segments"], 3)
        # longest CONSECUTIVE run is 36-51 (seg2); 25-31 and 52-56 are 7 & 5
        self.assertEqual(s["longest_bbox_run"], 16)
        runs = segment_runs(rows)
        self.assertEqual([(r["start"], r["end"], r["segment_id"]) for r in runs],
                         [(25, 31, 1), (36, 51, 2), (52, 56, 3)])

    def test_baseline_counts(self):
        rows = transfer_like_rows()
        no_base = set(range(32, 36)) | set(range(57, 64))  # 4 + 7 unavailable
        brows = [base_row(i, point=(600 + (i % 5), 1240)
                          if i not in no_base else None,
                          state="observed" if i not in no_base
                          else "unavailable")
                 for i in range(25, 71)]
        s = reconcile_summary(rows, baseline_rows=brows, window=(25, 70),
                              fps=25.0, clip="c", qa_verdicts=self.QA)
        # synthetic construction: 4 + 7 = 11 unavailable frames, 35 observed.
        # Internal consistency is the invariant under test (sums to 46).
        self.assertEqual(s["baseline_state_counts"]["observed"], 35)
        self.assertEqual(s["baseline_state_counts"]["unavailable"], 11)
        self.assertEqual(s["baseline_state_counts"]["observed"] +
                         s["baseline_state_counts"]["unavailable"], 46)

    def test_declared_window_truncating_rows_raises(self):
        # the actual defect: summary said [25, 51] but rows span [25, 70]
        rows = transfer_like_rows()
        with self.assertRaises(ValueError) as ctx:
            reconcile_summary(rows, window=(25, 51), fps=25.0, clip="c")
        self.assertIn("70", str(ctx.exception))

    def test_rows_outside_window_raise(self):
        with self.assertRaises(ValueError):
            reconcile_summary([mil_row(24, bbox=(1, 1, 5, 5))],
                              window=(25, 70), fps=25.0, clip="c")

    def test_preseed_window_rows_subset(self):
        # 6541855: window 128-156 but MIL rows start at the seed frame 140
        rows = [mil_row(140, bbox=(846, 1250, 190, 190), state="seed",
                        segment_id=1, confidence=1.0)]
        rows += [mil_row(i, state="unavailable", segment_id=1) for i in range(141, 153)]
        rows += [mil_row(i, state="ended", segment_id=1,
                         warning="track_terminated_no_reacquire")
                 for i in range(153, 157)]
        s = reconcile_summary(rows, window=(128, 156), fps=25.0, clip="pexels_6541855")
        self.assertEqual(s["total_frames"], 29)
        self.assertEqual(s["rows_present"], 17)
        self.assertEqual(s["pre_seed_frames"], 12)
        self.assertEqual(s["seed_frame"], 140)


# ---------------------------------------------------------------------------
# seed rendering distinct; gap/segment no-trail; QA border layers
# ---------------------------------------------------------------------------
class SeedRenderingTests(unittest.TestCase):
    def test_seed_drawn_distinctly(self):
        img = flat(480, 854)
        row = mil_row(0, bbox=(100, 100, 60, 60), state="seed", confidence=1.0)
        out = render_flat_frame(img, mil_row=row, baseline_row=None,
                                native_w=854, native_h=480, out_w=854,
                                out_h=480, source_frame_index=0,
                                window_start=0, fps=25.0)
        magenta = np.all(out == np.array((255, 0, 255)), axis=2).sum()
        self.assertGreater(int(magenta), 100, "seed box not drawn magenta")
        # tracked frame must NOT contain the seed color
        out2 = render_flat_frame(img, mil_row=mil_row(0, bbox=(100, 100, 60, 60),
                                                     state="tracked"),
                                 baseline_row=None, native_w=854, native_h=480,
                                 out_w=854, out_h=480, source_frame_index=0,
                                 window_start=0, fps=25.0)
        self.assertEqual(int(np.all(out2 == np.array((255, 0, 255)), axis=2).sum()), 0)


class GapSegmentNoTrailTests(unittest.TestCase):
    def test_no_trail_across_unavailable_gap(self):
        img = flat(240, 854)
        before = render_flat_frame(img, mil_row=mil_row(0, bbox=(50, 50, 40, 40),
                                                        state="tracked", segment_id=1),
                                   baseline_row=None, native_w=854, native_h=240,
                                   out_w=854, out_h=240, source_frame_index=0,
                                   window_start=0, fps=25.0)
        gap = render_flat_frame(img, mil_row=mil_row(1, state="unavailable",
                                                     warning="appearance_verification_failed"),
                                baseline_row=None, native_w=854, native_h=240,
                                out_w=854, out_h=240, source_frame_index=1,
                                window_start=0, fps=25.0)
        after = render_flat_frame(img, mil_row=mil_row(2, bbox=(300, 50, 40, 40),
                                                       state="tracked", segment_id=2),
                                  baseline_row=None, native_w=854, native_h=240,
                                  out_w=854, out_h=240, source_frame_index=2,
                                  window_start=0, fps=25.0)
        # gap frame: zero MIL marks anywhere
        for color in ((0, 255, 0), (0, 255, 255), (255, 0, 255)):
            self.assertEqual(int(np.all(gap == np.array(color), axis=2).sum()), 0)
        # after frame: green confined to its own box neighborhood; nothing
        # between the old center (70,70) and the new center (320,70) — no
        # bridging trail. Legend pixels are excluded by the y-band.
        green = np.all(after[55:85, :] == np.array((0, 255, 0)), axis=2)
        self.assertGreater(int(green.sum()), 0)
        ys, xs = np.nonzero(green)
        self.assertGreaterEqual(int(xs.min()), 294)
        self.assertLessEqual(int(xs.max()), 346)

    def test_segment_change_no_connector(self):
        img = flat(240, 854)
        # consecutive frames, different segment ids, far-apart boxes
        a = render_flat_frame(img, mil_row=mil_row(0, bbox=(50, 50, 40, 40),
                                                  state="reacquired", segment_id=1),
                              baseline_row=None, native_w=854, native_h=240,
                              out_w=854, out_h=240, source_frame_index=0,
                              window_start=0, fps=25.0)
        b = render_flat_frame(img, mil_row=mil_row(1, bbox=(700, 50, 40, 40),
                                                   state="reacquired", segment_id=2),
                              baseline_row=None, native_w=854, native_h=240,
                              out_w=854, out_h=240, source_frame_index=1,
                              window_start=0, fps=25.0)
        for fr in (a, b):
            yellow = np.all(fr[55:85, :] == np.array((0, 255, 255)), axis=2)
            ys, xs = np.nonzero(yellow)
            self.assertGreater(int(yellow.sum()), 0)
            # yellow strictly inside its own box neighborhood
            self.assertTrue(((xs >= 44) & (xs <= 96)).all() or
                            ((xs >= 694) & (xs <= 746)).all(),
                            "mark outside own box (connector drawn?)")
        # midpoint between the two centers must be clean in both frames
        for fr in (a, b):
            self.assertEqual(int(np.all(fr[60:80, 350:420] == np.array((0, 255, 255)),
                                        axis=2).sum()), 0)


class QABorderTests(unittest.TestCase):
    def test_rejected_verdict_red_border_not_green(self):
        img = flat(240, 854)
        row = mil_row(36, bbox=(100, 100, 60, 60), state="tracked", segment_id=2)
        out = render_flat_frame(img, mil_row=row, baseline_row=None,
                                native_w=854, native_h=240, out_w=854, out_h=240,
                                source_frame_index=36, window_start=25, fps=25.0,
                                mil_qa_verdict="on_body")
        self.assertEqual(tuple(out[1, 1]), (0, 0, 255), "red QA border missing")
        self.assertEqual(tuple(out[2, 2]), (0, 0, 255))
        # box itself drawn red (rejected), not green
        green = int(np.all(out == np.array((0, 255, 0)), axis=2).sum())
        self.assertEqual(green, 0, "rejected frame rendered as accepted green")

    def test_on_clubhead_verdict_green_border(self):
        img = flat(240, 854)
        row = mil_row(26, bbox=(100, 100, 60, 60), state="tracked", segment_id=1)
        out = render_flat_frame(img, mil_row=row, baseline_row=None,
                                native_w=854, native_h=240, out_w=854, out_h=240,
                                source_frame_index=26, window_start=25, fps=25.0,
                                mil_qa_verdict="on_clubhead")
        self.assertEqual(tuple(out[1, 1]), (0, 255, 0))

    def test_unreviewed_frame_has_no_border(self):
        img = flat(240, 854)
        row = mil_row(52, bbox=(100, 100, 60, 60), state="reacquired", segment_id=3)
        out = render_flat_frame(img, mil_row=row, baseline_row=None,
                                native_w=854, native_h=240, out_w=854, out_h=240,
                                source_frame_index=52, window_start=25, fps=25.0,
                                mil_qa_verdict=None)
        self.assertEqual(tuple(out[1, 1]), BG)

    def test_classify_mil_display(self):
        self.assertEqual(classify_mil_display("tracked", True, "on_body"), "rejected")
        self.assertEqual(classify_mil_display("reacquired", True, "on_shaft"), "rejected")
        self.assertEqual(classify_mil_display("tracked", True, "on_clubhead"),
                         "on_clubhead")
        self.assertEqual(classify_mil_display("tracked", True, None), "unreviewed")
        self.assertEqual(classify_mil_display("unavailable", False, "on_clubhead"),
                         "unavailable")
        self.assertEqual(classify_mil_display("ended", False, None), "ended")
        self.assertEqual(classify_mil_display("seed", True, "on_clubhead"), "seed")


class BaselinePointTests(unittest.TestCase):
    def test_baseline_point_drawn_scaled_only_when_observed(self):
        img = flat(240, 854)
        out = render_flat_frame(img, mil_row=None,
                                baseline_row=base_row(0, point=(400, 120)),
                                native_w=854, native_h=240, out_w=854, out_h=240,
                                source_frame_index=0, window_start=0, fps=25.0)
        blue = np.all(out == np.array((255, 0, 0)), axis=2)
        self.assertGreater(int(blue.sum()), 0)
        ys, xs = np.nonzero(blue)
        self.assertAlmostEqual(float(xs.mean()), 400.0, delta=12)
        self.assertAlmostEqual(float(ys.mean()), 120.0, delta=12)
        # unavailable baseline -> nothing drawn
        out2 = render_flat_frame(img, mil_row=None,
                                 baseline_row=base_row(1, state="unavailable"),
                                 native_w=854, native_h=240, out_w=854, out_h=240,
                                 source_frame_index=1, window_start=0, fps=25.0)
        self.assertEqual(int(np.all(out2 == np.array((255, 0, 0)), axis=2).sum()), 0)

    def test_baseline_qa_reject_hollow_red(self):
        img = flat(240, 854)
        out = render_flat_frame(img, mil_row=None,
                                baseline_row=base_row(0, point=(400, 120)),
                                native_w=854, native_h=240, out_w=854, out_h=240,
                                source_frame_index=0, window_start=0, fps=25.0,
                                baseline_qa_verdict="on_shaft")
        blue = int(np.all(out == np.array((255, 0, 0)), axis=2).sum())
        self.assertEqual(blue, 0, "QA-rejected baseline still drawn accepted-blue")


class EncodeCommandTests(unittest.TestCase):
    def test_h264_command_shape(self):
        argv = h264_encode_command("/tmp/frames", "/tmp/out.mp4", 25.0)
        joined = " ".join(argv)
        self.assertIn("libx264", joined)
        self.assertIn("yuv420p", joined)
        self.assertIn("+faststart", joined)
        self.assertIn("25", joined)
        self.assertIn("frame_%06d.png", joined)
        self.assertTrue(argv[-1].endswith("out.mp4"))


if __name__ == "__main__":
    unittest.main()