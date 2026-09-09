"""Identity-verification candidate: SUSTAINED-STATIONARITY wrong-object reject.

ONE bounded identity-verification change (tested, isolated research option).
Wraps the MilRegionTracker's per-frame output rows and rejects wrong-object
static locks while retaining useful clubhead observations.

Mechanism (single hypothesis, from deterministic diagnosis of the 6541842
body-lock): a wrong-object reacquisition holds the box near-static for a
sustained interval, whereas a real clubhead in these clips keeps moving. When
inter-frame box-center displacement stays < ``stationary_max_disp_px`` for a
sustained ``>= stationary_min_duration_s`` of CLIP TIME (fps-aware), the row is
treated as a wrong-object false lock: emit state=unavailable with warning
stationary_false_lock, reset the static counter immediately (so it does not
suppress later real frames), and keep the rows that follow (the tracker is
still allowed to reacquire).

This is explicitly NOT a scene/coordinate/floor ban and NOT an NCC raise. The
frozen thresholds are 8 px and 0.2 s, predeclared, not tuned after results.
Baseline (gate disabled) is byte-identical to the input rows.

research_only=true, ground_truth=false, production_eligible=false.
"""
from __future__ import annotations
import math
from typing import Optional, Sequence

# --- predeclared frozen thresholds (do not tune after seeing results) -------
STATIONARY_MAX_DISP_PX = 8.0
STATIONARY_MIN_DURATION_S = 0.2


class StationarityIdentityFilter:
    """Post-track identity-verification gate over per-frame tracker rows.

    Rows must be dicts with source_frame_index, bbox (or None), state,
    fps. Produces a new list of rows with the same keys; rejected static-lock
    frames get state=unavailable, bbox=None, warning=stationary_false_lock.
    Input rows are not mutated.
    """

    def __init__(self, *, max_disp_px: float = STATIONARY_MAX_DISP_PX,
                 min_duration_s: float = STATIONARY_MIN_DURATION_S,
                 fps: float = 30.0) -> None:
        if isinstance(max_disp_px, bool) or not isinstance(max_disp_px, (int, float)) \
                or not math.isfinite(max_disp_px) or max_disp_px < 0:
            raise ValueError("max_disp_px must be finite >=0")
        if isinstance(min_duration_s, bool) or not isinstance(min_duration_s, (int, float)) \
                or not math.isfinite(min_duration_s) or min_duration_s <= 0:
            raise ValueError("min_duration_s must be finite >0")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) \
                or not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite >0")
        self.max_disp_px = float(max_disp_px)
        self.min_duration_s = float(min_duration_s)
        self.fps = float(fps)

    def filter(self, rows: Sequence[dict]) -> list:
        out: list = []
        static_seconds = 0.0
        lock_suspected = False  # once sustained-static, keep rejecting until motion resumes/gap
        prev_f: Optional[int] = None
        prev_c: Optional[tuple] = None
        for r in rows:
            row = dict(r)  # do not mutate input
            f = row.get("source_frame_index")
            bbox = row.get("bbox")
            state = row.get("state") or "unavailable"
            accepted = (bbox is not None and state in ("tracked", "reacquired", "seed")
                        and isinstance(f, int))
            if accepted:
                c = (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)
                moving = True
                if prev_f is not None and f == prev_f + 1 and prev_c is not None:
                    d = math.hypot(c[0] - prev_c[0], c[1] - prev_c[1])
                    moving = d >= self.max_disp_px  # true object motion resumes
                    if moving:
                        static_seconds = 0.0
                        lock_suspected = False
                    else:
                        static_seconds += 1.0 / self.fps
                else:
                    # gap (unavailable) or non-consecutive frame resets the lock
                    lock_suspected = False
                    static_seconds = 0.0
                if lock_suspected or static_seconds >= self.min_duration_s - 1e-9:
                    # sustained static lock (or still within one): reject as wrong-object.
                    # Once locked, keep rejecting while the object stays static.
                    lock_suspected = True
                    row["state"] = "unavailable"
                    row["bbox"] = None
                    row["visibility"] = "missing"
                    row["warning"] = "stationary_false_lock"
                    row["segment_id"] = r.get("segment_id", 1)
                # IMPORTANT: always advance position, whether or not we emit,
                # so the static-run accounting is continuous (no phantom gap).
                prev_f, prev_c = f, c
            else:
                # unavailable/ended resets the lock (a gap breaks it)
                static_seconds = 0.0
                lock_suspected = False
                prev_f, prev_c = None, None
            out.append(row)
        return out


def apply_stationarity_filter(rows, *, fps) -> list:
    return StationarityIdentityFilter(fps=fps).filter(rows)
