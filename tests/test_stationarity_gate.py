"""Regression tests for the sustained-stationarity identity gate.

RED-first contract:
- a near-static wrong-object lock (fps=25, ~5px/f sustained > 0.2s) -> rejected
  stationary_false_lock;
- a useful moving track (fps=60, most frames >8px/f, static runs < 0.2s) retained;
- a gap (unavailable frame) resets the static counter (no rejection on the rejoin);
- fps-aware: 0.2s == 12 frames @60fps, == 5 frames @25fps;
- baseline (gate not applied) is untouched (this filter is an isolated option);
- input rows are never mutated.
Only ONE mechanism is added (stationarity); thresholds 8px/0.2s frozen.
"""
import math, unittest
from ghostcaddie.video.stationarity_gate import (StationarityIdentityFilter,
                                                 STATIONARY_MAX_DISP_PX,
                                                 STATIONARY_MIN_DURATION_S)

def row(f, x, state="tracked", bbox_w=135, bbox_h=130):
    return {"source_frame_index": f, "state": state, "bbox": [x, 1000, bbox_w, bbox_h],
            "visibility": "visible", "confidence": 0.7, "segment_id": 1, "warning": None}


class TestStationarityGate(unittest.TestCase):
    def test_constant_slow_lock_rejected_at_25fps(self):
        # near-static: center advances 5px/frame at 25fps; 0.2s = 5 frames
        rows = [row(25 + i, 100 + i * 5) for i in range(10)]
        out = StationarityIdentityFilter(fps=25.0).filter(rows)
        rej = [r["source_frame_index"] for r in out if r["warning"] == "stationary_false_lock"]
        self.assertTrue(len(rej) >= 1, f"expected some rejected, got {rej}")
        # verify all rejections are state=unavailable, bbox=None
        for r in out:
            if r["warning"] == "stationary_false_lock":
                self.assertEqual(r["state"], "unavailable")
                self.assertIsNone(r["bbox"])

    def test_fast_moving_useful_track_retained_at_60fps(self):
        # clubhead moves a lot per frame at 60fps (e.g. 30px/f): no sustained <8px
        rows = [row(145 + i, 100 + i * 30) for i in range(16)]
        out = StationarityIdentityFilter(fps=60.0).filter(rows)
        rej = [r["source_frame_index"] for r in out if r["warning"] == "stationary_false_lock"]
        self.assertEqual(rej, [], f"useful moving track must be retained, got rejected {rej}")
        emitted = [r["source_frame_index"] for r in out if r["state"] in ("tracked", "reacquired", "seed")]
        self.assertEqual(emitted, [145 + i for i in range(16)])

    def test_gap_resets_counter(self):
        # 5 slow frames, then a gap (unavailable), then 5 fast frames -> no false reject
        rows = ([row(0 + i, 100 + i * 5) for i in range(5)] +
                [{"source_frame_index": 5, "state": "unavailable", "bbox": None,
                  "visibility": "missing", "confidence": 0.0, "segment_id": 1, "warning": "gap"}] +
                [row(6 + i, 600 + i * 40) for i in range(5)])
        out = StationarityIdentityFilter(fps=25.0).filter(rows)
        # the 5 slow frames may be rejected (true sustained static), but the fast rejoin must persist
        rej_after = [r["source_frame_index"] for r in out
                     if r["warning"] == "stationary_false_lock" and r["source_frame_index"] >= 6]
        self.assertEqual(rej_after, [], f"gap must reset counter; got post-gap rejects {rej_after}")

    def test_fps_aware_half_frames(self):
        # sustained slow lock: at 60fps fewer consecutive frames needed to hit 0.2s
        # 6 frames @60fps = 0.1s < 0.2s -> NOT rejected; 13 frames @60 = 0.216s -> rejected
        short = StationarityIdentityFilter(fps=60.0).filter([row(0+i, 100+i*4) for i in range(6)])
        self.assertFalse(any(r["warning"] == "stationary_false_lock" for r in short),
                         "6 frames @60fps (0.1s) must NOT trip the 0.2s gate")
        long = StationarityIdentityFilter(fps=60.0).filter([row(0+i, 100+i*4) for i in range(13)])
        self.assertTrue(any(r["warning"] == "stationary_false_lock" for r in long),
                        "13 frames @60fps (0.216s) must trip the gate")

    def test_input_rows_not_mutated(self):
        rows = [row(0 + i, 100 + i * 5) for i in range(10)]
        orig = [dict(r) for r in rows]
        StationarityIdentityFilter(fps=25.0).filter(rows)
        self.assertEqual(rows, orig, "filter must not mutate input rows")

    def test_thresholds_frozen(self):
        self.assertEqual(STATIONARY_MAX_DISP_PX, 8.0)
        self.assertEqual(STATIONARY_MIN_DURATION_S, 0.2)


if __name__ == "__main__":
    unittest.main()
