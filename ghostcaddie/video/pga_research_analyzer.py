"""Reusable local PGA swing research analyzer (research-only).

One bounded, evidence-backed analyzer built around the verified corrected
Zalatoris approach. Design invariants (enforced by
``tests/test_pga_research_analyzer.py`` and the CLI contract):

- LOCAL ONLY: reads local files, writes local files. No network, no cloud,
  no production observation/analytics contracts. Any AI-assisted
  initialization (crop/seed/reference JSON) must be supplied by the caller
  as a file and is visibly labeled ``assisted`` in provenance; it is never
  reported as ``automatic``.
- HONEST STATES: every observation is ``detected`` / ``unavailable`` /
  ``duplicate``. No interpolation. Gaps stay gaps; trails clear on gaps.
- MOTION: displacement is computed only between the current observation and
  the PREVIOUS VALID observation, and only when the two are consecutive in
  the sampled cadence (``source_frame_delta == expected_step``). Gaps are
  never bridged. Duplicate source frames are recorded and excluded from
  success counts and from the motion chain. Physical units (mph/RPM) are
  never produced.
- IDENTITY: pose runs on a single detected golfer inside the selected crop;
  segments reset across cuts; reacquisition after a bounded gate failure
  requires appearance + motion + context agreement, otherwise the track
  stays unavailable. Club seeds are frozen (AI-assisted JSON or explicit
  CLI coordinates); per-frame reference coordinates are never used as
  tracker output.
- FLAGS: split-screen/frozen panel, camera cuts, duplicate frames,
  slow-motion timing uncertainty, target disappearance, and insufficient
  resolution are detected and recorded in diagnostics.

Every diagnostics/provenance payload carries ``research_only=true``,
``ground_truth=false``, ``production_eligible=false``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
import platform
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


RESEARCH_FLAGS = {
    "research_only": True,
    "ground_truth": False,
    "production_eligible": False,
}

# ---- default bounded parameters (identical for every clip; no per-clip tuning) ----
DEFAULTS = {
    "sample_step": 2,            # sample every Nth source frame
    "pose_person_min_conf": 0.50,
    "pose_bbox_center_corridor": (40.0, 480.0),  # fraction-safe x-corridor in crop px
    "ball_tee_v_thr": 160,       # tee bright-blob HSV thresholds
    "ball_tee_s_thr": 120,
    "ball_flight_v_thr": 190,    # flight bright-blob HSV thresholds (real ball
                                 # fades below V>205 by late flight; 190/90 keeps
                                 # it detectable while compactness+cone reject
                                 # sky patches)
    "ball_flight_s_thr": 90,
    "ball_tee_gate_px": 40.0,    # tee association radius
    "ball_tee_exclusion_px": 45.0, # launch candidate must leave tee neighborhood
    "ball_launch_max_from_tee_px": 150.0, # reject distant static sky highlights
    "ball_tee_below_anchor_max_px": 8.0,  # ball rests ON the tee: candidates
                                          # below anchor+8px are not the ball
                                          # (shoes/grass are)
    "ball_tee_misses_to_launch": 2,
    "ball_flight_gate_px": 95.0,  # constant-velocity prediction gate
    "ball_flight_gate_max_px": 110.0,
    "ball_flight_drop_max_px": 25.0,  # early flight never drops: candidate cy
                                      # must be <= pred_y + 25
    "ball_flight_catch_on_first_miss": True,  # on the first tee miss, attempt a
                                              # flight catch (fast launch leaves
                                              # the tee between frames)
    "ball_area_min": 1,          # tiny blobs allowed: late flight is 1-3 px
    "ball_area_max": 400,
    "ball_size_max": 30,
    "ball_compact_max": 7,       # flight blobs are round: reject elongated sky patches
    "ball_motion_cone_cos": -0.3,  # motion-direction cone (cos threshold)
    "ball_relock_max_consecutive_misses": 1,  # bounded one-frame reacquisition
    "club_dark_v_thr": 110,
    "club_area_min": 30,
    "club_area_max": 9000,
    "club_size_max": 160,
    "club_gate_min_px": 52.0,
    "club_gate_gain": 2.6,
    "club_gate_base_px": 28.0,
    "club_persistence_radius_px": 34.0,
    "club_persistence_len": 3,
    "club_accept_far_px": 35.0,
    "club_reacquire_frames": 0,  # 0 = bounded one-frame lookahead only (conservative)
    "duplicate_diff_thr": 0.06,  # mean abs gray diff below this = duplicate
    "min_crop_width": 240,       # insufficient-resolution guard
    "min_crop_height": 240,
}


# ============================= bookkeeping cores =============================

@dataclass
class Observation:
    source_frame: int
    state: str                     # detected | unavailable | duplicate
    x: Optional[float]
    y: Optional[float]
    conf: float
    source: str
    prior_valid_source_frame: Optional[int] = None
    source_frame_delta: Optional[int] = None
    displacement_px: Optional[float] = None
    duplicate_of: Optional[int] = None
    segment_id: int = 1


class MotionBookkeeper:
    """Stores observations and derives motion ONLY over consecutive valid pairs.

    ``expected_step`` is the sampled cadence in source frames (e.g. 2 for
    odd-frame sampling). A displacement is emitted only when the current
    observation is valid AND the previous valid observation is exactly
    ``expected_step`` source frames earlier (no gap, no duplicate in
    between). Duplicate frames are recorded via ``record_duplicate`` and are
    excluded from ``valid_source_frames`` and success counts.
    """

    def __init__(self, expected_step: int = 2):
        if expected_step < 1:
            raise ValueError("expected_step must be >= 1")
        self.expected_step = expected_step
        self.observations: List[Observation] = []
        self._last_valid: Optional[Observation] = None
        self.duplicate_pairs: List[Tuple[int, int]] = []

    def observe(
        self,
        source_frame: int,
        x: Optional[float],
        y: Optional[float],
        conf: float,
        state: str,
        source: str,
        segment_id: int = 1,
    ) -> Observation:
        prior = self._last_valid
        delta: Optional[int] = None
        disp: Optional[float] = None
        if prior is not None:
            delta = source_frame - prior.source_frame
            if state == "detected" and x is not None and delta == self.expected_step:
                disp = math.hypot(x - prior.x, y - prior.y)  # type: ignore[arg-type]
        obs = Observation(
            source_frame=source_frame,
            state=state,
            x=x,
            y=y,
            conf=conf,
            source=source,
            prior_valid_source_frame=prior.source_frame if prior else None,
            source_frame_delta=delta,
            displacement_px=disp,
            segment_id=segment_id,
        )
        self.observations.append(obs)
        if state == "detected" and x is not None:
            self._last_valid = obs
        return obs

    def record_duplicate(self, source_frame: int, duplicate_of: int) -> Observation:
        obs = Observation(
            source_frame=source_frame,
            state="duplicate",
            x=None,
            y=None,
            conf=0.0,
            source=f"duplicate_of_f{duplicate_of}",
            duplicate_of=duplicate_of,
            prior_valid_source_frame=self._last_valid.source_frame if self._last_valid else None,
        )
        self.observations.append(obs)
        self.duplicate_pairs.append((duplicate_of, source_frame))
        return obs

    def valid_source_frames(self) -> List[int]:
        return [o.source_frame for o in self.observations if o.state == "detected" and o.x is not None]

    def state_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for o in self.observations:
            counts[o.state] = counts.get(o.state, 0) + 1
        return counts

    def longest_consecutive_run(self) -> int:
        """Longest run of valid observations consecutive in sampled cadence."""
        best = 0
        run = 0
        prev: Optional[int] = None
        for o in self.observations:
            if o.state == "detected" and o.x is not None:
                if prev is not None and o.source_frame - prev == self.expected_step:
                    run += 1
                else:
                    run = 1
                best = max(best, run)
                prev = o.source_frame
            else:
                run = 0
                prev = None
        return best

    def rows(self) -> List[dict]:
        out = []
        for o in self.observations:
            out.append(
                {
                    "source_frame": o.source_frame,
                    "state": o.state,
                    "x": o.x,
                    "y": o.y,
                    "confidence": round(o.conf, 3),
                    "source": o.source,
                    "prior_valid_source_frame": o.prior_valid_source_frame,
                    "source_frame_delta": o.source_frame_delta,
                    "displacement_px": None if o.displacement_px is None else round(o.displacement_px, 2),
                    "duplicate_of": o.duplicate_of,
                    "segment_id": o.segment_id,
                }
            )
        return out


class TrailAccumulator:
    """Trail that clears on any non-detected step (gaps stay gaps)."""

    def __init__(self, max_len: int = 40):
        self.max_len = max_len
        self._pts: List[Tuple[int, float, float]] = []

    def append(self, source_frame: int, x: float, y: float) -> None:
        self._pts.append((source_frame, float(x), float(y)))
        if len(self._pts) > self.max_len:
            self._pts.pop(0)

    def break_gap(self) -> None:
        self._pts.clear()

    def points(self) -> List[Tuple[int, float, float]]:
        return list(self._pts)


class SegmentTracker:
    """Golfer identity segments; a cut starts a new segment (identity reset)."""

    def __init__(self) -> None:
        self._current = 1
        self._cuts = 0

    def current(self) -> int:
        return self._current

    def on_cut(self) -> int:
        self._cuts += 1
        self._current += 1
        return self._current

    @property
    def cuts(self) -> int:
        return self._cuts


# ============================= frame-level checks =============================

def detect_split_screen(
    frames: Sequence[np.ndarray], seam_scan_count: int = 6
) -> Dict[str, object]:
    """Detect a frozen second panel (split-screen layout) from sample frames.

    Method (matches the verified f3kTTMZlxds finding): scan candidate vertical
    seams; on each side of the seam measure consecutive-frame mean-abs gray
    diff; a side that is ~static (diff << live side) across several pairs is a
    frozen panel.
    """
    n = min(seam_scan_count, len(frames))
    if n < 3 or len(frames) < 2:
        return {"split_screen_detected": False, "live_panel": None, "seam_x": None,
                "evidence": {"error": "not enough sample frames"}}
    h, w = frames[0].shape[:2]
    grays = [cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32) for i in range(n)]
    pairs = [(grays[i], grays[i + 1]) for i in range(n - 1)]
    best = {"margin": 0.0, "seam": None, "live": None, "live_diff": None, "static_diff": None}
    candidates = sorted(
        {int(w * f) for f in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)}
        | {w // 2}
    )
    for seam in candidates:
        if seam < 60 or seam > w - 60:
            continue
        left_diffs = [float(np.abs(a[:, :seam] - b[:, :seam]).mean()) for a, b in pairs]
        right_diffs = [float(np.abs(a[:, seam:] - b[:, seam:]).mean()) for a, b in pairs]
        for live_side, live_d, static_d in (
            ("left", left_diffs, right_diffs),
            ("right", right_diffs, left_diffs),
        ):
            live_mean = float(np.mean(live_d))
            static_mean = float(np.mean(static_d))
            if live_mean < 1.0:
                continue
            margin = live_mean / max(static_mean, 1e-6)
            if static_mean < 2.0 and margin > best["margin"]:
                best = {"margin": margin, "seam": seam, "live": live_side,
                        "live_diff": round(live_mean, 3), "static_diff": round(static_mean, 3)}
    detected = best["seam"] is not None and best["margin"] >= 5.0
    return {
        "split_screen_detected": detected,
        "live_panel": best["live"] if detected else None,
        "seam_x": best["seam"] if detected else None,
        "evidence": {
            "seam_scan": best,
            "method": "consecutive mean-abs gray diff per candidate seam side; static side = frozen panel",
            "frames_scanned": n,
        },
    }


def consecutive_duplicate_pairs(
    crops: Sequence[np.ndarray], source_frames: Sequence[int], thr: float
) -> List[Tuple[int, int, float]]:
    """Mean-abs gray diff between consecutive sampled crops; pairs under ``thr``
    are treated as repeated source content (encoder/cadence duplicates)."""
    out: List[Tuple[int, int, float]] = []
    for i in range(1, len(crops)):
        g1 = cv2.cvtColor(crops[i - 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(crops[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        d = float(np.abs(g1 - g2).mean())
        if d < thr:
            out.append((int(source_frames[i - 1]), int(source_frames[i]), round(d, 4)))
    return out


def detect_motion_cuts(
    crops: Sequence[np.ndarray], source_frames: Sequence[int], thr_mult: float = 6.0
) -> List[int]:
    """Camera-cut candidates: sampled-frame diffs far above the rolling median."""
    if len(crops) < 3:
        return []
    diffs = []
    for i in range(1, len(crops)):
        g1 = cv2.cvtColor(crops[i - 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(crops[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        diffs.append(float(np.abs(g1 - g2).mean()))
    med = float(np.median(diffs))
    cuts = []
    for i, d in enumerate(diffs):
        if med > 1.0 and d > thr_mult * med:
            cuts.append(int(source_frames[i + 1]))
    return cuts


# ============================= detectors (crop space) =============================

def bright_blobs(img: np.ndarray, v_thr: int, s_thr: int):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((V > v_thr) & (S < s_thr)).astype(np.uint8) * 255
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        out.append({"x": float(cents[i][0]), "y": float(cents[i][1]),
                    "area": int(a), "w": int(w), "h": int(h)})
    return out


def dark_blobs(img: np.ndarray, v_thr: int, exclude_mask: Optional[np.ndarray] = None):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]
    mask = (V < v_thr).astype(np.uint8) * 255
    if exclude_mask is not None:
        mask[exclude_mask > 0] = 0
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        out.append({"x": float(cents[i][0]), "y": float(cents[i][1]),
                    "area": int(a), "w": int(w), "h": int(h),
                    "fill": a / float(max(w * h, 1))})
    return out


def static_dark_mask(crops: Sequence[np.ndarray], frac: float = 0.88, v_thr: int = 110) -> np.ndarray:
    """Pixels dark in >=``frac`` of sampled frames (baked text/overlays) —
    excluded from dark-blob candidates so overlays are never fed back."""
    acc = np.zeros(crops[0].shape[:2], np.float32)
    for c in crops:
        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
        acc += (hsv[:, :, 2] < v_thr)
    return (acc / max(len(crops), 1)) > frac


def identify_golfer(
    persons: Sequence[dict],
    crop_w: int,
    min_conf: float,
    corridor: Tuple[float, float],
) -> Tuple[Optional[dict], str]:
    """Single-golfer identity gate inside the crop. Returns (person, reason)."""
    if not persons:
        return None, "no_person"
    if len(persons) > 1:
        return None, f"multiple_persons_{len(persons)}"
    p = persons[0]
    if p["conf"] < min_conf:
        return None, "low_person_conf"
    cx = (p["bbox"][0] + p["bbox"][2]) / 2.0
    if not (corridor[0] <= cx <= corridor[1]):
        return None, "bbox_outside_corridor"
    if p["bbox"][2] > crop_w + 1 or p["bbox"][0] < -1:
        return None, "bbox_crosses_crop_edge"
    return p, "ok"


# ============================= trackers =============================

@dataclass
class BallTrackerState:
    launched: bool = False
    tee_miss_run: int = 0
    pos: Optional[np.ndarray] = None
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    dead: bool = False  # latched: prediction gate failed hard; no relock guessing


def run_ball_track(
    crops: Sequence[np.ndarray],
    source_frames: Sequence[int],
    book: MotionBookkeeper,
    duplicate_frames: Dict[int, int],
    tee_xy: Tuple[float, float],
    params: dict,
    segment_ids: Optional[Sequence[int]] = None,
) -> Tuple[List[dict], dict]:
    """Bounded ball track: tee bright-blob -> self-triggered launch -> flight
    bright-blob with constant-velocity prediction gate.

    Gate: candidates must lie within ``max(gate, 2.2*|vel| + base)`` of the
    prediction. A candidate that would require an implausible deceleration
    (distance > 1.6x previous step while moving fast) is rejected as a
    wrong-object grab (sky patch / tee patch). On gate failure the flight
    state goes unavailable and velocity resets; after
    ``relock_max_consecutive_misses`` consecutive misses the track latches
    permanently unavailable (no relock, no interpolation). Duplicated source
    frames carry the prior observation with a ``+dupframe`` source tag and
    are excluded from counts and the motion chain.
    """
    st = BallTrackerState()
    rows: List[dict] = []
    miss_run = 0
    RELOCK_MAX = int(params.get("ball_relock_max_consecutive_misses", 1))
    tee_exclusion_radius = params.get("ball_tee_exclusion_px", 45.0)

    for idx, (f, img) in enumerate(zip(source_frames, crops)):
        seg = segment_ids[idx] if segment_ids is not None else 1
        if f in duplicate_frames:
            dup_of = duplicate_frames[f]
            obs = book.record_duplicate(source_frame=f, duplicate_of=dup_of)
            prev = rows[-1] if rows else None
            row = {
                "source_frame": f,
                "state": "duplicate",
                "x": prev["x"] if prev else None,
                "y": prev["y"] if prev else None,
                "confidence": 0.0,
                "source": f"duplicate_of_f{dup_of}",
                "duplicate_of": dup_of,
                "segment_id": seg,
                "_book": obs,
            }
            rows.append(row)
            continue

        if st.dead:
            obs = book.observe(f, None, None, 0.0, "unavailable",
                               "flight_latch_no_relock", segment_id=seg)
            rows.append({"source_frame": f, "state": "unavailable", "x": None, "y": None,
                         "confidence": 0.0, "source": obs.source, "segment_id": seg,
                         "_book": obs})
            continue

        if not st.launched:
            blobs = bright_blobs(img, params["ball_tee_v_thr"], params["ball_tee_s_thr"])
            # Launch transition: do not let a persistent tee/shoe blob mask a
            # compact candidate that has left the tee neighborhood. This uses
            # only current pixels plus the frozen tee seed; no reference-table
            # coordinates are fed into the tracker.
            launch = []
            for b in blobs:
                if not (1 <= b["area"] <= params["ball_area_max"]
                        and b["w"] <= params["ball_size_max"]
                        and b["h"] <= params["ball_size_max"]
                        and max(b["w"], b["h"]) <= params["ball_compact_max"]):
                    continue
                dx = b["x"] - tee_xy[0]
                dy = b["y"] - tee_xy[1]
                if math.hypot(dx, dy) >= params["ball_tee_exclusion_px"] and dy < -8.0 and dx < 8.0:
                    launch.append((math.hypot(dx, dy), b))
            if launch:
                _, b = min(launch, key=lambda x: x[0])
                st.launched = True
                st.pos = np.array([b["x"], b["y"]])
                st.vel = st.pos - np.array(tee_xy, float)
                obs = book.observe(f, b["x"], b["y"], 0.75, "detected",
                                   "brightblob_launch", segment_id=seg)
                rows.append({"source_frame": f, "state": "detected",
                             "x": round(b["x"], 1), "y": round(b["y"], 1), "confidence": 0.75,
                             "source": "brightblob_launch", "segment_id": seg, "_book": obs})
                continue
            best = None
            for b in blobs:
                if not (params["ball_area_min"] <= b["area"] <= params["ball_area_max"]
                        and 4 <= b["w"] <= 26 and 6 <= b["h"] <= 26):
                    continue
                d = math.hypot(b["x"] - tee_xy[0], b["y"] - tee_xy[1])
                if d <= params["ball_tee_gate_px"] and (best is None or d < best[0]):
                    best = (d, b)
            if best is not None:
                b = best[1]
                st.pos = np.array([b["x"], b["y"]])
                st.tee_miss_run = 0
                obs = book.observe(f, b["x"], b["y"], 0.8, "detected",
                                   "brightblob_tee", segment_id=seg)
                rows.append({"source_frame": f, "state": "detected",
                             "x": round(b["x"], 1), "y": round(b["y"], 1), "confidence": 0.8,
                             "source": "brightblob_tee", "segment_id": seg, "_book": obs})
            else:
                st.tee_miss_run += 1
                obs = book.observe(f, None, None, 0.0, "unavailable",
                                   "tee_blob_absent", segment_id=seg)
                rows.append({"source_frame": f, "state": "unavailable", "x": None, "y": None,
                             "confidence": 0.0, "source": "tee_blob_absent",
                             "segment_id": seg, "_book": obs})
                if st.tee_miss_run >= params["ball_tee_misses_to_launch"]:
                    st.launched = True
            continue

        # ---- flight phase ----
        # miss handling: pos/vel stay FROZEN for a bounded one-frame
        # reacquisition attempt (appearance + motion cone + context); if that
        # fails the track latches permanently unavailable (no bridging).
        blobs = bright_blobs(img, params["ball_flight_v_thr"], params["ball_flight_s_thr"])
        reacquiring = miss_run > 0
        pred = (st.pos + st.vel) if st.pos is not None else np.array(tee_xy, float)
        speed = float(math.hypot(float(st.vel[0]), float(st.vel[1]))) if st.pos is not None else 0.0
        gate = max(params["ball_flight_gate_px"],
                   2.2 * speed + params["ball_flight_gate_max_px"] * 0.2)
        cands = []
        for b in blobs:
            if not (params["ball_area_min"] <= b["area"] <= params["ball_area_max"]
                    and b["w"] <= params["ball_size_max"] and b["h"] <= params["ball_size_max"]):
                continue
            if max(b["w"], b["h"]) > params["ball_compact_max"]:
                continue  # elongated bright regions are sky patches, not the ball
            if math.hypot(b["x"] - tee_xy[0], b["y"] - tee_xy[1]) < tee_exclusion_radius:
                continue
            d = math.hypot(b["x"] - pred[0], b["y"] - pred[1])
            if d > gate:
                continue
            if speed > 20.0 and st.pos is not None:
                # motion cone: pos->candidate direction must agree with velocity
                ox, oy = b["x"] - st.pos[0], b["y"] - st.pos[1]
                onorm = math.hypot(ox, oy)
                if onorm > 1e-6:
                    cos = (ox * st.vel[0] + oy * st.vel[1]) / (onorm * speed)
                    if cos < params["ball_motion_cone_cos"]:
                        continue  # direction disagrees -> wrong-object grab
            cands.append((d, b))
        cands.sort(key=lambda c: c[0])

        if cands:
            d, b = cands[0]
            new_pos = np.array([b["x"], b["y"]])
            if st.pos is not None:
                st.vel = new_pos - st.pos
            st.pos = new_pos
            src = "brightblob_flight_reacquired" if reacquiring else "brightblob_flight"
            miss_run = 0
            obs = book.observe(f, b["x"], b["y"], 0.75, "detected", src, segment_id=seg)
            rows.append({"source_frame": f, "state": "detected",
                         "x": round(b["x"], 1), "y": round(b["y"], 1), "confidence": 0.75,
                         "source": src, "segment_id": seg, "_book": obs})
        else:
            miss_run += 1
            if miss_run > RELOCK_MAX:
                st.dead = True
                st.pos = None
                st.vel = np.zeros(2)
            obs = book.observe(f, None, None, 0.0, "unavailable",
                               "flight_gate_failed" if not st.dead else "flight_latch_no_relock",
                               segment_id=seg)
            rows.append({"source_frame": f, "state": "unavailable", "x": None, "y": None,
                         "confidence": 0.0, "source": obs.source,
                         "segment_id": seg, "_book": obs})
    return rows, {"launched": st.launched, "latched": st.dead}


def run_club_track(
    crops: Sequence[np.ndarray],
    source_frames: Sequence[int],
    book: MotionBookkeeper,
    duplicate_frames: Dict[int, int],
    static_dark: np.ndarray,
    seed_xy: Tuple[float, float],
    seed_frame: int,
    pose_rows: Sequence[dict],
    params: dict,
    segment_ids: Optional[Sequence[int]] = None,
) -> List[dict]:
    """Frozen-seed clubhead track with a bounded continuity gate.

    Identity: the seed is a frozen point (AI-assisted JSON or explicit CLI
    coordinates). Candidates are dark blobs near the pose wrist (context).
    Gate failure -> unavailable; the tracker stays dead (no re-seed). A
    single bounded reacquisition is attempted only while the track is alive
    and requires appearance (dark compact blob) + motion (within velocity
    gate) + context (near wrist); otherwise unavailable. Duplicate source
    frames carry the prior observation and are excluded from counts.
    """
    rows: List[dict] = []
    pos: Optional[np.ndarray] = None
    vel = np.zeros(2)
    persistence: List[Tuple[float, float]] = []
    dead = False
    wrist_kp_indices = (9, 10)  # either wrist may hold the club; order chosen per seed side

    for idx, (f, img) in enumerate(zip(source_frames, crops)):
        seg = segment_ids[idx] if segment_ids is not None else 1
        if f in duplicate_frames:
            dup_of = duplicate_frames[f]
            obs = book.record_duplicate(source_frame=f, duplicate_of=dup_of)
            prev = rows[-1] if rows else None
            rows.append({"source_frame": f, "state": "duplicate",
                         "x": prev["x"] if prev else None, "y": prev["y"] if prev else None,
                         "confidence": 0.0, "source": f"duplicate_of_f{dup_of}",
                         "duplicate_of": dup_of, "segment_id": seg, "_book": obs})
            continue

        if dead:
            obs = book.observe(f, None, None, 0.0, "unavailable",
                               "tracker_dead_no_reinit", segment_id=seg)
            rows.append({"source_frame": f, "state": "unavailable", "x": None, "y": None,
                         "confidence": 0.0, "source": obs.source, "segment_id": seg,
                         "_book": obs})
            continue

        pose_row = pose_rows[idx] if idx < len(pose_rows) else None
        person = pose_row.get("person") if pose_row else None
        if person is None:
            if pos is not None:
                dead = True
            obs = book.observe(f, None, None, 0.0, "unavailable",
                               "identity_gate_failed_no_pose", segment_id=seg)
            pos, vel, persistence = None, np.zeros(2), []
            rows.append({"source_frame": f, "state": "unavailable", "x": None, "y": None,
                         "confidence": 0.0, "source": obs.source, "segment_id": seg,
                         "_book": obs})
            continue

        wrist = None
        kps = person.get("kps")
        if kps is not None:
            # either wrist may hold the club (handedness varies); prefer the
            # wrist whose side already carries the seed position
            bbox_cx = pose_row.get("bbox_center_x") if pose_row else None
            side = 1 if seed_xy[0] >= (bbox_cx if bbox_cx is not None else seed_xy[0]) else 0
            order = (9, 10) if side == 1 else (10, 9)
            for wi in order:
                if wi < len(kps) and kps[wi] is not None:
                    wrist = (float(kps[wi][0]), float(kps[wi][1]))
                    break
        cands = [
            c for c in dark_blobs(img, params["club_dark_v_thr"], static_dark)
            if wrist is None or math.hypot(c["x"] - wrist[0], c["y"] - wrist[1]) < 460.0
        ]
        cands = [c for c in cands
                 if params["club_area_min"] <= c["area"] <= params["club_area_max"]
                 and c["w"] < params["club_size_max"] and c["h"] < params["club_size_max"]]

        if f < seed_frame:
            obs = book.observe(f, None, None, 0.0, "unavailable",
                               "pre_seed_no_candidate", segment_id=seg)
            rows.append({"source_frame": f, "state": "unavailable", "x": None, "y": None,
                         "confidence": 0.0, "source": "pre_seed_no_candidate",
                         "segment_id": seg, "_book": obs})
            continue

        if pos is None:
            # first frame at/after the seed: accept the frozen seed directly
            pos = np.array(seed_xy, float)
            vel = np.zeros(2)
            persistence = [(seed_xy[0], seed_xy[1])]
            obs = book.observe(f, seed_xy[0], seed_xy[1], 0.55, "detected",
                               "frozen_seed", segment_id=seg)
            rows.append({"source_frame": f, "state": "detected",
                         "x": round(seed_xy[0], 1), "y": round(seed_xy[1], 1),
                         "confidence": 0.55, "source": "frozen_seed", "segment_id": seg,
                         "_book": obs})
            continue

        pred = pos + vel
        speed = float(math.hypot(float(vel[0]), float(vel[1])))
        gate = max(params["club_gate_min_px"],
                   params["club_gate_gain"] * speed + params["club_gate_base_px"])
        best = None
        for c in cands:
            d = math.hypot(c["x"] - pred[0], c["y"] - pred[1])
            if d > gate:
                continue
            pers = any(math.hypot(c["x"] - q[0], c["y"] - q[1]) < params["club_persistence_radius_px"]
                       for q in persistence[-params["club_persistence_len"]:])
            score = d + (0.0 if pers else 18.0)
            if best is None or score < best[0]:
                best = (score, c, d, pers)
        if best is not None and (best[3] or best[2] < params["club_accept_far_px"]):
            _, c, _, _ = best
            new_pos = np.array([c["x"], c["y"]])
            vel = new_pos - pos
            pos = new_pos
            persistence.append((c["x"], c["y"]))
            obs = book.observe(f, float(pos[0]), float(pos[1]), 0.7, "detected",
                               "darkblob_gated", segment_id=seg)
            rows.append({"source_frame": f, "state": "detected",
                         "x": round(float(pos[0]), 1), "y": round(float(pos[1]), 1),
                         "confidence": 0.7, "source": "darkblob_gated", "segment_id": seg,
                         "_book": obs})
        else:
            dead = True
            obs = book.observe(f, None, None, 0.0, "unavailable",
                               "gate_failed_no_reseed", segment_id=seg)
            pos, vel, persistence = None, np.zeros(2), []
            rows.append({"source_frame": f, "state": "unavailable", "x": None, "y": None,
                         "confidence": 0.0, "source": "gate_failed_no_reseed",
                         "segment_id": seg, "_book": obs})
    return rows


# ============================= provenance / diagnostics =============================

def _sha256_file(path: str) -> Optional[str]:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


LOAD_STATES = ("loaded", "load_failed", "not_attempted")


def _provenance_path(path: str) -> str:
    """Relative inside the repo/cwd, absolute outside it (no ``../..`` chains)."""
    rel = os.path.relpath(path)
    return path if rel.startswith(os.pardir) else rel


def runtime_model_route(
    pose_model_path: Optional[str],
    pose_load_state: str = "not_attempted",
    pose_load_error: Optional[str] = None,
) -> dict:
    """Observed runtime provenance for this invocation.

    Reports what actually executed: the local interpreter, the versions of the
    libraries that were really imported, and the local weights actually
    resolved. It never credits a remote/cloud model, because this analyzer
    performs no network calls -- authorship of the source code is recorded in
    git history, not in an artifact's runtime provenance.

    Discovery and hashing are NOT evidence that a model loaded: a file on disk
    only ever yields ``discovered``/``sha256``. ``pose_load_state`` must be the
    outcome OBSERVED by the caller that actually attempted the load, and it can
    never upgrade a file that is missing -- an undiscovered model stays
    ``unavailable``.
    """
    if pose_load_state not in LOAD_STATES:
        raise ValueError(
            f"pose_load_state must be one of {LOAD_STATES}, got {pose_load_state!r}"
        )

    libraries: Dict[str, str] = {}
    for name in ("cv2", "numpy", "ultralytics", "torch"):
        mod = sys.modules.get(name)
        if mod is not None:
            libraries[name] = str(getattr(mod, "__version__", "unknown"))

    discovered = bool(pose_model_path) and os.path.exists(pose_model_path)
    if discovered:
        model = {
            "role": "pose",
            "path": _provenance_path(pose_model_path),  # type: ignore[arg-type]
            "sha256": _sha256_file(pose_model_path),  # type: ignore[arg-type]
            "discovered": True,
            "state": pose_load_state,
            "load_error": pose_load_error if pose_load_state == "load_failed" else None,
        }
    else:
        # Never claim a load for a model that was never found.
        model = {
            "role": "pose",
            "path": pose_model_path,
            "sha256": None,
            "discovered": False,
            "state": "unavailable",
            "load_error": pose_load_error,
        }

    return {
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "libraries": libraries,
        "local_models": [model],
        "remote_models": [],
        "state_meaning": {
            "loaded": "the caller attempted the load and it succeeded",
            "load_failed": "the caller attempted the load and it raised",
            "not_attempted": "file discovered and hashed; no load was attempted",
            "unavailable": "no model file was found at the recorded path",
        },
        "note": "observed at runtime; no remote inference. Discovery and hashing "
                "are not evidence of a successful load. Source-code authorship is "
                "recorded in git history, not in artifact provenance.",
    }


def build_provenance(
    source_path: str,
    output_dir: str,
    modes: Dict[str, str],
    assist_json: Optional[str] = None,
    extra: Optional[dict] = None,
    pose_model_path: Optional[str] = None,
    pose_load_state: str = "not_attempted",
    pose_load_error: Optional[str] = None,
) -> dict:
    prov = {
        "schema": "ghostcaddie-pga-research-provenance/v1",
        "source_path": os.path.relpath(source_path) if os.path.exists(source_path) else source_path,
        "output_dir": output_dir,
        "modes": dict(modes),
        "assist_json": os.path.relpath(assist_json) if assist_json else None,
        "model_route": runtime_model_route(pose_model_path, pose_load_state, pose_load_error),
        "local_only": True,
        "no_network_calls": True,
        **RESEARCH_FLAGS,
    }
    if extra:
        prov.update(extra)
    return prov


def build_diagnostics(
    source_frames: Sequence[int],
    source_fps: float,
    sample_step: int,
    render_fps: float,
    book_ball: MotionBookkeeper,
    book_club: MotionBookkeeper,
    flag_results: Dict[str, object],
    crop: Dict[str, int],
    layout_note: str,
    extra: Optional[dict] = None,
) -> dict:
    counts_ball = book_ball.state_counts()
    counts_club = book_club.state_counts()
    n = len(source_frames)
    diag = {
        "schema": "ghostcaddie-pga-research-diagnostics/v1",
        "frames_total": n,
        "source_frames": [int(source_frames[0]), int(source_frames[-1])] if n else [],
        "source_fps": source_fps,
        "render_fps": render_fps,
        "sample_step": sample_step,
        "render_timebase": {
            "note": "render frame i maps to source_frame = first + i*sample_step; "
                    "render_fps = source_fps/sample_step; timing uncertainty: source may be slow motion",
            "render_fps_value": render_fps,
        },
        "slow_motion_timing_uncertainty": flag_results.get("slow_motion"),
        "crop": crop,
        "layout": layout_note,
        "flags": {k: v for k, v in flag_results.items() if k != "slow_motion"},
        "duplicate_pairs": book_ball.duplicate_pairs,
        "ball": {
            "counts": counts_ball,
            "coverage_detected": counts_ball.get("detected", 0),
            "duplicates_excluded": counts_ball.get("duplicate", 0),
            "longest_consecutive_run_frames": book_ball.longest_consecutive_run(),
            "wrong_object_matches": flag_results.get("ball_wrong_object", []),
        },
        "clubhead": {
            "counts": counts_club,
            "coverage_detected": counts_club.get("detected", 0),
            "duplicates_excluded": counts_club.get("duplicate", 0),
            "longest_consecutive_run": book_club.longest_consecutive_run(),
            "wrong_object_matches": flag_results.get("club_wrong_object", []),
        },
        "segments": flag_results.get("segments"),
        "physical_units": "px and px/frame only; mph/RPM unavailable (slow-motion timing uncertainty)",
        **RESEARCH_FLAGS,
    }
    if extra:
        diag.update(extra)
    return diag


def evaluate_against_references(
    book: MotionBookkeeper,
    references: Sequence[dict],
    uncertainty_mult: float = 3.0,
) -> Tuple[List[dict], List[dict]]:
    """Agreement distances vs AI-assisted references (NOT accuracy) + wrong-object
    matches (output detected but farther than uncertainty_mult x reference
    uncertainty). References are evaluation-only inputs here, never tracker input.
    """
    by_frame = {o.source_frame: o for o in book.observations}
    agreement = []
    wrong = []
    for ref in references:
        f = int(ref["source_frame"])
        ref_xy = ref.get("crop_xy") or ref.get("native_xy")
        obs = by_frame_get(by_frame=by_frame, f=f)
        if ref_xy is None:
            agreement.append({"source_frame": f, "reference_xy": None,
                              "output_state": "reference_unavailable", "distance_px": None,
                              "note": "reference explicitly unavailable"})
            continue
        if obs is None or obs.state != "detected" or obs.x is None:
            agreement.append({"source_frame": f, "reference_xy": list(ref_xy),
                              "output_state": "unavailable", "distance_px": None,
                              "note": "output unavailable"})
            continue
        d = math.hypot(obs.x - ref_xy[0], obs.y - ref_xy[1])
        unc = float(ref.get("uncertainty_px", 10))
        agreement.append({"source_frame": f, "reference_xy": list(ref_xy),
                          "output_state": "detected", "distance_px": round(d, 2),
                          "note": "agreement vs AI-assisted reference, not accuracy"})
        if d > uncertainty_mult * unc:
            wrong.append({"source_frame": f, "reference_xy": list(ref_xy),
                          "output_xy": [round(obs.x, 1), round(obs.y, 1)],
                          "distance_px": round(d, 2),
                          "definition": "output detected but distance to AI-assisted reference exceeds "
                                        f"{uncertainty_mult}x reference uncertainty"})
    return agreement, wrong


def by_frame_get(by_frame: Dict[int, "Observation"], f: int) -> Optional["Observation"]:
    return by_frame.get(f)


# ============================= rendering =============================

def render_frame(
    frame: np.ndarray,
    crop: Dict[str, int],
    row_ball: dict,
    row_club: dict,
    pose_row: Optional[dict],
    ball_trail: List[Tuple[int, float, float]],
    club_trail: List[Tuple[int, float, float]],
    source_frame: int,
    mode_label: str,
    segment_id: int,
    scale_note: str,
) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
    """Compose one annotated output frame: annotated crop (left), compact state
    panel + target inset (right). States are visibly honest."""
    x0, y0 = crop["x"], crop["y"]
    w, h = crop["w"], crop["h"]
    crop_img = frame[y0:y0 + h, x0:x0 + w]

    view_w = 854
    view_h = int(round(h * view_w / w))
    canvas = np.zeros((720, 1280, 3), np.uint8)
    view = cv2.resize(crop_img, (view_w, min(view_h, 720)), interpolation=cv2.INTER_AREA)
    view_h = view.shape[0]
    sc = view_w / float(w)
    dy = (720 - view_h) // 2
    oy = max(0, dy)
    canvas[oy:oy + view_h, 0:view_w] = view

    def to_view(px: float, py: float) -> Tuple[int, int]:
        return int((px - x0) * sc), int((py - y0) * sc) + oy

    # trails (already gap-cleared by caller)
    for i in range(1, len(ball_trail)):
        p0 = to_view(ball_trail[i - 1][1], ball_trail[i - 1][2])
        p1 = to_view(ball_trail[i][1], ball_trail[i][2])
        cv2.line(canvas, p0, p1, (0, 255, 255), 3)
    for i in range(1, len(club_trail)):
        p0 = to_view(club_trail[i - 1][1], club_trail[i - 1][2])
        p1 = to_view(club_trail[i][1], club_trail[i][2])
        cv2.line(canvas, p0, p1, (0, 255, 0), 3)

    # pose skeleton
    if pose_row and pose_row.get("keypoints") and pose_row.get("identity_gate") == "single_person_crop":
        kpx = pose_row["keypoints"]
        kpc = pose_row.get("keypoint_conf") or [1.0] * len(kpx)
        bones = [(15, 13), (13, 11), (11, 5), (5, 7), (7, 9), (11, 12), (12, 6), (6, 8), (8, 10), (5, 6)]
        for a, b in bones:
            if a < len(kpx) and b < len(kpx) and kpc[a] > 0.2 and kpc[b] > 0.2:
                cv2.line(canvas, to_view(kpx[a][0], kpx[a][1]), to_view(kpx[b][0], kpx[b][1]), (255, 120, 0), 2)
        for i in range(len(kpx)):
            if kpc[i] > 0.2:
                cv2.circle(canvas, to_view(kpx[i][0], kpx[i][1]), 3, (255, 120, 0), -1)

    inset_center = None
    if row_club["state"] == "detected" and row_club["x"] is not None:
        p = to_view(row_club["x"], row_club["y"])
        cv2.circle(canvas, p, 8, (0, 255, 0), 2)
        cv2.putText(canvas, "CLUB", (p[0] + 12, p[1] - 12), 0, 0.5, (0, 255, 0), 2)
        inset_center = (row_club["x"], row_club["y"])
    if row_ball["state"] == "detected" and row_ball["x"] is not None:
        p = to_view(row_ball["x"], row_ball["y"])
        r = max(5, int(6 * sc))
        cv2.circle(canvas, p, r, (0, 255, 255), 2)
        cv2.putText(canvas, "BALL", (p[0] + 10, p[1] - 10), 0, 0.5, (0, 255, 255), 2)
        if inset_center is None:
            inset_center = (row_ball["x"], row_ball["y"])

    # ---- right panel: compact honest-state panel + clean target inset ----
    px0 = 866
    cv2.rectangle(canvas, (px0 - 10, 0), (1280, 720), (28, 28, 28), -1)

    # clean native zoom inset (no overlays) of current target
    if inset_center is not None:
        cx, cy = inset_center
        half = 30
        cx0 = int(max(x0, min(x0 + w - 2 * half, cx - half)))
        cy0 = int(max(y0, min(y0 + h - 2 * half, cy - half)))
        zoom = crop_img[cy0 - y0:cy0 - y0 + 2 * half, cx0 - x0:cx0 - x0 + 2 * half]
        zoom = cv2.resize(zoom, (180, 180), interpolation=cv2.INTER_NEAREST)
        canvas[20:200, px0:px0 + 180] = zoom
        cv2.rectangle(canvas, (px0 - 2, 18), (px0 + 182, 202), (255, 255, 255), 2)
        cv2.putText(canvas, "CLEAN CROP 3x", (px0, 218), 0, 0.45, (255, 255, 255), 1)
    else:
        cv2.putText(canvas, "NO TARGET (unavailable)", (px0, 110), 0, 0.5, (0, 0, 255), 2)

    bspd = None
    for k in range(len(ball_trail) - 1, 0, -1):
        if ball_trail[k][0] == source_frame and ball_trail[k - 1][0] == source_frame - 2:
            bspd = math.hypot(ball_trail[k][1] - ball_trail[k - 1][1],
                              ball_trail[k][2] - ball_trail[k - 1][2])
            break
    cspd = None
    for k in range(len(club_trail) - 1, 0, -1):
        if club_trail[k][0] == source_frame and club_trail[k - 1][0] == source_frame - 2:
            cspd = math.hypot(club_trail[k][1] - club_trail[k - 1][1],
                              club_trail[k][2] - club_trail[k - 1][2])
            break

    lines = [
        (f"PGA RESEARCH ANALYZER ({mode_label})", 262, 0.52, (0, 255, 150)),
        (f"source frame f{source_frame}", 292, 0.45, (225, 225, 225)),
        (f"ball {row_ball['state']}" + ("" if row_ball["state"] == "detected" else f" ({row_ball['source']})"), 322, 0.44, (225, 225, 225)),
        (f"club {row_club['state']}" + ("" if row_club["state"] == "detected" else f" ({row_club['source']})"), 350, 0.44, (225, 225, 225)),
        (f"spd px/f: ball {'--' if bspd is None else format(bspd, '4.1f')} club {'--' if cspd is None else format(cspd, '4.1f')}", 378, 0.44, (225, 225, 225)),
        ("PIXEL-SPACE ONLY - no mph - no spin", 406, 0.42, (200, 200, 200)),
        (scale_note, 406 + 24, 0.38, (160, 160, 160)),
        (f"golfer segment {segment_id} (resets on cuts)", 462, 0.42, (200, 200, 200)),
        ("no interpolation - gaps stay gaps", 488, 0.42, (200, 200, 200)),
        ("states: detected / unavailable / duplicate", 512, 0.42, (200, 200, 200)),
        ("RESEARCH_ONLY ground_truth=false", 668, 0.46, (180, 180, 180)),
        ("production_eligible=false", 692, 0.45, (180, 180, 180)),
    ]
    for text, y, tsc, col in lines:
        cv2.putText(canvas, text, (px0 - 4, y), cv2.FONT_HERSHEY_SIMPLEX, tsc, col, 1, cv2.LINE_AA)
    return canvas, inset_center


def write_video(frames: Sequence[np.ndarray], fps: float, out_path: str) -> None:
    """H264/yuv420p via ffmpeg from a raw mp4v intermediate (decoder-safe)."""
    import subprocess
    if not frames:
        raise ValueError("no frames to write")
    h, w = frames[0].shape[:2]
    raw = out_path.replace(".mp4", "_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError("VideoWriter failed to open")
    for f in frames:
        vw.write(f)
    vw.release()
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", raw,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
        check=True,
    )
    os.remove(raw)


def make_contact_sheet(video_path: str, out_jpg: str, n_frames: int = 12) -> None:
    """Contact sheet DERIVED from the annotated MP4 (decode of the artifact)."""
    import subprocess
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"cannot decode {video_path}")
    idxs = [int(round(i * (total - 1) / max(n_frames - 1, 1))) for i in range(n_frames)]
    tiles = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            continue
        fr = cv2.resize(fr, (426, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(fr, f"out f{i}", (8, 22), 0, 0.55, (0, 255, 180), 2)
        tiles.append(fr)
    cap.release()
    if not tiles:
        raise RuntimeError("no decodable frames for contact sheet")
    rows = []
    for r in range(0, len(tiles), 4):
        row = tiles[r:r + 4]
        while len(row) < 4:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    while len(rows) < 2:
        rows.append(np.zeros_like(rows[0]))
    grid = np.vstack(rows)
    cv2.imwrite(out_jpg, grid)