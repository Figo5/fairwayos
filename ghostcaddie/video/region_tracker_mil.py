"""Research-only seed-conditioned REGION tracking via OpenCV TrackerMIL.

Distinct from the point/region/color clubhead methods in
ghostcaddie.video.clubhead_methods: the seed is a bounding BOX that both
localizes and initializes the appearance model; tracking maintains MIL
appearance memory (TrackerMIL adapts online) while every emitted frame must
independently re-verify against the FROZEN seed patch with normalized
cross-correlation. MIL's ``update`` keeps returning boxes even when the object
is gone, so verification is not optional — drift fails CLOSED (frame marked
unavailable, position never propagated). Reacquisition after a failed frame is
an appearance re-detection against the seed patch inside a bounded window and
starts a NEW segment_id. Runtime is bounded by a wall-clock budget checked
between frames; after ``max_lost_frames`` consecutive failures the track ends
instead of guessing. No ground truth, no domain events, no production use.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Optional, Tuple

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 required at init() time anyway
    cv2 = None  # type: ignore[assignment]

import numpy as np


@dataclass(frozen=True)
class MilTrackFrame:
    """One row of a seeded region track. bbox is (x, y, w, h) floats."""
    source_frame_index: int
    timestamp: float
    bbox: Optional[Tuple[float, float, float, float]]
    visibility: str            # visible | missing | occluded | off_frame
    state: str                 # seed | tracked | reacquired | unavailable | ended
    confidence: float
    segment_id: int
    provenance: str
    uncertainty_px: Optional[float]
    warning: Optional[str] = None


_PROVENANCE = "ai_assisted_seeded_region"

# verification thresholds (NCC against the frozen seed patch)
DEFAULT_MIN_APPEARANCE_NCC = 0.50     # accept a MIL-proposed box above this
DEFAULT_REACQUIRE_MIN_SCORE = 0.60    # re-detection must be more certain
DEFAULT_LOCAL_REFINE_RADIUS_PX = 24   # small window for verify+refine
DEFAULT_REACQUIRE_RADIUS_PX = 160.0   # bounded re-detection window radius
DEFAULT_MAX_RUNTIME_S = 120.0         # wall-clock budget for a whole track()
DEFAULT_MAX_LOST_FRAMES = 12          # consecutive failures before track ends
DEFAULT_FPS = 30.0


def _validate_box(seed_box, w: int, h: int) -> Tuple[float, float, float, float]:
    if seed_box is None or len(seed_box) != 4:
        raise ValueError("seed_box must be (x, y, w, h)")
    vals = tuple(seed_box)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in vals):
        raise ValueError("seed_box values must be finite numbers")
    x, y, bw, bh = (float(v) for v in vals)
    if bw <= 0 or bh <= 0:
        raise ValueError("seed_box width/height must be positive")
    if x < 0 or y < 0 or x + bw > w or y + bh > h:
        raise ValueError("seed_box must lie inside the frame")
    return (x, y, bw, bh)


def _ncc_of(patch_a, patch_b) -> float:
    """Normalized cross-correlation of two same-size grayscale patches."""
    if patch_a is None or patch_b is None:
        return -1.0
    if patch_a.size == 0 or patch_b.size == 0 or patch_a.shape != patch_b.shape:
        return -1.0
    return float(cv2.matchTemplate(patch_a, patch_b, cv2.TM_CCOEFF_NORMED)[0, 0])


class MilRegionTracker:
    """Seed-conditioned MIL region tracker (fail-closed adapter over
    cv2.TrackerMIL). See module docstring for the honesty contract."""

    def __init__(self, *, min_appearance_ncc: float = DEFAULT_MIN_APPEARANCE_NCC,
                 reacquire_min_score: float = DEFAULT_REACQUIRE_MIN_SCORE,
                 local_refine_radius_px: int = DEFAULT_LOCAL_REFINE_RADIUS_PX,
                 reacquire_radius_px: float = DEFAULT_REACQUIRE_RADIUS_PX,
                 max_runtime_s: float = DEFAULT_MAX_RUNTIME_S,
                 max_lost_frames: int = DEFAULT_MAX_LOST_FRAMES,
                 fps: float = DEFAULT_FPS) -> None:
        for name, v in (("min_appearance_ncc", min_appearance_ncc),
                        ("reacquire_min_score", reacquire_min_score)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) \
                    or not math.isfinite(v) or not 0.0 < v < 1.0:
                raise ValueError(f"{name} must be a finite number in (0, 1)")
        for name, v in (("reacquire_radius_px", reacquire_radius_px),
                        ("max_runtime_s", max_runtime_s), ("fps", fps)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) \
                    or not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        for name, v in (("local_refine_radius_px", local_refine_radius_px),
                        ("max_lost_frames", max_lost_frames)):
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if reacquire_min_score < min_appearance_ncc:
            raise ValueError("reacquire_min_score must be >= min_appearance_ncc")
        self.min_appearance_ncc = float(min_appearance_ncc)
        self.reacquire_min_score = float(reacquire_min_score)
        self.local_refine_radius_px = int(local_refine_radius_px)
        self.reacquire_radius_px = float(reacquire_radius_px)
        self.max_runtime_s = float(max_runtime_s)
        self.max_lost_frames = int(max_lost_frames)
        self.fps = float(fps)
        self._seed_box: Optional[Tuple[float, float, float, float]] = None
        self._seed_source_frame: Optional[int] = None
        self._seed_patch = None
        self._frame_size: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------ init
    def init(self, frame_bgr, seed_box, seed_source_frame: int) -> None:
        if frame_bgr is None or getattr(frame_bgr, "ndim", 0) != 3:
            raise ValueError("frame_bgr must be an HxWx3 image")
        h, w = frame_bgr.shape[:2]
        if isinstance(seed_source_frame, bool) or not isinstance(seed_source_frame, int) \
                or seed_source_frame < 0:
            raise ValueError("seed_source_frame must be a non-negative int")
        self._seed_box = _validate_box(seed_box, w, h)
        self._seed_source_frame = int(seed_source_frame)
        self._frame_size = (h, w)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        sx, sy, sw, sh = (int(round(v)) for v in self._seed_box)
        self._seed_patch = gray[sy:sy + sh, sx:sx + sw].copy()

    # ---------------------------------------------------------------- helpers
    def _crop(self, gray, box):
        h, w = self._frame_size
        x, y, bw, bh = box
        x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
        x1, y1 = min(w, int(round(x + bw))), min(h, int(round(y + bh)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        return gray[y0:y1, x0:x1]

    def _local_search(self, gray, box, radius):
        """Match the frozen seed patch around ``box`` within ``radius``.

        Returns (score, refined_box_or_None, uncertainty_px_or_None). The
        uncertainty is the score-weighted spatial std of near-peak matches:
        a sharp single peak is low-uncertainty; a flat/ambiguous surface is
        high-uncertainty.
        """
        h, w = self._frame_size
        x, y, bw, bh = box
        cx, cy = x + bw / 2.0, y + bh / 2.0
        x0 = max(0, int(round(cx - radius - bw / 2)))
        y0 = max(0, int(round(cy - radius - bh / 2)))
        x1 = min(w, int(round(cx + radius + bw / 2)) + 1)
        y1 = min(h, int(round(cy + radius + bh / 2)) + 1)
        win = gray[y0:y1, x0:x1]
        if win.size == 0 or win.shape[0] < self._seed_patch.shape[0] \
                or win.shape[1] < self._seed_patch.shape[1]:
            return -1.0, None, None
        scores = cv2.matchTemplate(win, self._seed_patch, cv2.TM_CCOEFF_NORMED)
        _, best_val, _, best_loc = cv2.minMaxLoc(scores)
        bx = float(x0 + best_loc[0])
        by = float(y0 + best_loc[1])
        refined = (bx, by, float(bw), float(bh))
        # near-peak spread -> positional uncertainty
        thresh = float(best_val) - 0.05
        ys, xs = np.where(scores >= thresh)
        if xs.size == 0:
            unc = float(radius)
        else:
            wts = scores[ys, xs] - thresh + 1e-6
            mx = float(np.average(xs, weights=wts))
            my = float(np.average(ys, weights=wts))
            unc = float(math.sqrt(max(0.0, float(np.average((xs - mx) ** 2, weights=wts)
                                              + np.average((ys - my) ** 2, weights=wts)))))
            unc = max(1.0, min(float(radius), unc))
        return float(best_val), refined, unc

    # ------------------------------------------------------------------ track
    def track(self, frames_bgr) -> list:
        """Track the seeded region across ``frames_bgr``.

        ``frames_bgr[0]`` must be the seed frame itself (the image passed to
        ``init``); subsequent elements are the following frames. Rows carry
        absolute ``source_frame_index = seed_source_frame + offset``. Exactly
        one row is produced per frame.
        """
        if self._seed_box is None or self._seed_patch is None:
            raise RuntimeError("init() must be called before track()")
        if not frames_bgr:
            raise ValueError("frames_bgr must contain at least the seed frame")
        h, w = self._frame_size
        deadline = time.monotonic() + self.max_runtime_s
        rows: list = []
        rows.append(MilTrackFrame(
            source_frame_index=self._seed_source_frame, timestamp=0.0,
            bbox=self._seed_box, visibility="visible", state="seed",
            confidence=1.0, segment_id=1, provenance=_PROVENANCE,
            uncertainty_px=0.0))

        mil = cv2.TrackerMIL_create()
        sb = self._seed_box
        mil.init(frames_bgr[0],
                 (int(round(sb[0])), int(round(sb[1])),
                  int(round(sb[2])), int(round(sb[3]))))  # type: ignore[call-arg]
        cur = sb
        cur_center = (sb[0] + sb[2] / 2.0, sb[1] + sb[3] / 2.0)
        segment_id = 1
        lost_streak = 0
        ended = False
        end_reason: Optional[str] = None

        for offset in range(1, len(frames_bgr)):
            idx = self._seed_source_frame + offset
            ts = offset / self.fps

            # ---- terminal states: keep emitting honest rows, no positions
            if ended:
                rows.append(MilTrackFrame(
                    source_frame_index=idx, timestamp=ts, bbox=None,
                    visibility="missing", state="ended", confidence=0.0,
                    segment_id=segment_id, provenance=_PROVENANCE,
                    uncertainty_px=None, warning=end_reason))
                continue
            if time.monotonic() > deadline:
                ended, end_reason = True, "runtime_budget_exceeded"
                rows.append(MilTrackFrame(
                    source_frame_index=idx, timestamp=ts, bbox=None,
                    visibility="missing", state="ended", confidence=0.0,
                    segment_id=segment_id, provenance=_PROVENANCE,
                    uncertainty_px=None, warning=end_reason))
                continue

            frame = frames_bgr[offset]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ---- verdict for this frame (exactly one row will be appended)
            verdict = None  # MilTrackFrame
            mil_proposed = False
            # --- 1) MIL continuity proposal
            try:
                ok, bb = mil.update(frame)
            except cv2.error:
                ok, bb = False, None
            if ok:
                mil_proposed = True
                box = tuple(float(v) for v in bb)
                area = box[2] * box[3]
                cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
                step = math.hypot(cx - cur_center[0], cy - cur_center[1])
                x0, y0 = max(0.0, box[0]), max(0.0, box[1])
                x1, y1 = min(float(w), box[0] + box[2]), min(float(h), box[1] + box[3])
                inside_frac = max(0.0, x1 - x0) * max(0.0, y1 - y0) / area if area > 0 else 0.0
                outside_frac = 1.0 - inside_frac
                patch = self._crop(gray, box)
                if patch is None or outside_frac > 0.5:
                    verdict = MilTrackFrame(
                        source_frame_index=idx, timestamp=ts, bbox=None,
                        visibility="off_frame", state="unavailable", confidence=0.0,
                        segment_id=segment_id, provenance=_PROVENANCE,
                        uncertainty_px=None, warning="box_left_frame")
                    lost_streak += 1
                    if lost_streak >= self.max_lost_frames:
                        ended, end_reason = True, "track_terminated_no_reacquire"
                else:
                    if outside_frac > 0.0:
                        # candidate boxes must stay on the canvas
                        box = (x0, y0, x1 - x0, y1 - y0)
                    if step > self.reacquire_radius_px:
                        # MIL's whole-image scan teleported: refuse it and
                        # fall through to bounded-window reacquisition
                        mil_proposed = False
                    else:
                        ncc = _ncc_of(patch, self._seed_patch)
                        if ncc >= self.min_appearance_ncc:
                            score, refined, unc = self._local_search(
                                gray, box, self.local_refine_radius_px)
                            if refined is not None and score >= self.min_appearance_ncc:
                                resuming = lost_streak > 0
                                if resuming:
                                    segment_id += 1
                                conf = max(0.0, min(1.0, score))
                                verdict = MilTrackFrame(
                                    source_frame_index=idx, timestamp=ts,
                                    bbox=refined, visibility="visible",
                                    state="reacquired" if resuming else "tracked",
                                    confidence=conf, segment_id=segment_id,
                                    provenance=_PROVENANCE, uncertainty_px=unc,
                                    warning="seed_appearance_reacquired" if resuming else None)
                                cur = refined
                                cur_center = (refined[0] + refined[2] / 2.0,
                                              refined[1] + refined[3] / 2.0)
                                if resuming:
                                    # new segment = new MIL episode
                                    mil.init(frame, (int(round(refined[0])),
                                                     int(round(refined[1])),
                                                     int(round(refined[2])),
                                                     int(round(refined[3]))))  # type: ignore[call-arg]
                                    mil_proposed = False
                                else:
                                    # re-anchor MIL only when its proposal
                                    # disagreed with the verified box (keep
                                    # adaptive memory when they agree)
                                    mcx = box[0] + box[2] / 2.0
                                    mcy = box[1] + box[3] / 2.0
                                    rcx = refined[0] + refined[2] / 2.0
                                    rcy = refined[1] + refined[3] / 2.0
                                    if math.hypot(mcx - rcx, mcy - rcy) > 8.0:
                                        mil.init(frame, (int(round(refined[0])),
                                                         int(round(refined[1])),
                                                         int(round(refined[2])),
                                                         int(round(refined[3]))))  # type: ignore[call-arg]
                                lost_streak = 0
            # --- 2) bounded-window reacquisition (when MIL path failed)
            if verdict is None and not ended:
                score, found, unc = self._local_search(
                    gray, cur, self.reacquire_radius_px)
                if found is not None and score >= self.reacquire_min_score:
                    segment_id += 1
                    conf = max(0.0, min(1.0, score)) * 0.9
                    verdict = MilTrackFrame(
                        source_frame_index=idx, timestamp=ts, bbox=found,
                        visibility="visible", state="reacquired", confidence=conf,
                        segment_id=segment_id, provenance=_PROVENANCE,
                        uncertainty_px=unc, warning="seed_appearance_reacquired")
                    cur = found
                    cur_center = (found[0] + found[2] / 2.0,
                                  found[1] + found[3] / 2.0)
                    # a new segment is a new MIL episode
                    mil.init(frame, (int(round(found[0])), int(round(found[1])),
                                     int(round(found[2])),
                                     int(round(found[3]))))  # type: ignore[call-arg]
                    lost_streak = 0
                else:
                    # unverifiable this frame. Only report the region as
                    # having left the canvas when the SEED box had clearance
                    # (so "at the edge" is evidence of exit, not seed placement)
                    # and the last verified box now sits at the boundary.
                    bw_, bh_ = cur[2], cur[3]
                    edge_gap = min(cur_center[0], cur_center[1],
                                   w - cur_center[0], h - cur_center[1])
                    sx, sy, sw_, sh_ = self._seed_box
                    seed_clearance = min(sx, sy, w - sx - sw_, h - sy - sh_)
                    if seed_clearance > max(sw_, sh_) and edge_gap <= max(bw_, bh_):
                        verdict = MilTrackFrame(
                            source_frame_index=idx, timestamp=ts, bbox=None,
                            visibility="off_frame", state="unavailable",
                            confidence=0.0, segment_id=segment_id,
                            provenance=_PROVENANCE, uncertainty_px=None,
                            warning="box_left_frame")
                    else:
                        reason = "appearance_verification_failed" if mil_proposed \
                            else "reacquire_failed"
                        verdict = MilTrackFrame(
                            source_frame_index=idx, timestamp=ts, bbox=None,
                            visibility="missing", state="unavailable",
                            confidence=0.0, segment_id=segment_id,
                            provenance=_PROVENANCE, uncertainty_px=None,
                            warning=reason)
                    lost_streak += 1
                    if lost_streak >= self.max_lost_frames:
                        ended, end_reason = True, "track_terminated_no_reacquire"
            elif verdict is None:
                verdict = MilTrackFrame(
                    source_frame_index=idx, timestamp=ts, bbox=None,
                    visibility="missing", state="unavailable", confidence=0.0,
                    segment_id=segment_id, provenance=_PROVENANCE,
                    uncertainty_px=None, warning="reacquire_failed")
            rows.append(verdict)
        return rows