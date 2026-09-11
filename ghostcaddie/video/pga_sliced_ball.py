"""Sliced (tiled) local-contrast golf-ball candidate detection.

Transfers the *idea* behind the soccer reference's sliced 640px ball inference
-- process small regions at native resolution instead of thresholding the whole
frame once -- to our classical detector. No code is copied from that repository
(it carries no licence, so it grants no reuse rights), and no weights are used:
this needs none, which is why it is runnable today.

Why slicing should help here: a global HSV threshold has to satisfy the ball
against bright sky AND against dark grass shadow with one pair of numbers. A
golf ball in flight is a handful of pixels whose absolute brightness varies
hugely with background, so a global threshold either floods the frame with
candidates or misses the ball. A per-tile threshold asks the local question --
"is this blob much brighter than ITS OWN neighbourhood?" -- which is the
question that actually distinguishes a ball.

Candidates are CANDIDATES. Nothing here establishes ball identity.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SLICED_DEFAULTS = {
    "tile": 160,
    "overlap": 0.5,
    "local_sigma": 3.2,      # blob must exceed local mean by this many local SDs
    "min_area": 2,
    "max_area": 260,
    "max_side": 24,
    "max_aspect": 2.2,
    "min_sat_margin": 90,    # ball is desaturated relative to grass
    "dedupe_px": 6.0,
}


def _tiles(h: int, w: int, tile: int, overlap: float):
    step = max(1, int(tile * (1.0 - overlap)))
    for y0 in range(0, max(1, h - 1), step):
        for x0 in range(0, max(1, w - 1), step):
            yield x0, y0, min(x0 + tile, w), min(y0 + tile, h)


def sliced_candidates(img: np.ndarray, params: Optional[dict] = None) -> List[dict]:
    """Small, round, locally-bright, locally-desaturated blobs.

    Returns candidates sorted by descending local contrast (``z``). Never
    returns a ball identity claim -- only ranked candidate points.
    """
    p = dict(SLICED_DEFAULTS)
    if params:
        p.update(params)
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)

    found: List[dict] = []
    for x0, y0, x1, y1 in _tiles(h, w, int(p["tile"]), float(p["overlap"])):
        v = V[y0:y1, x0:x1]
        s = S[y0:y1, x0:x1]
        if v.size < 64:
            continue
        mu, sd = float(v.mean()), float(v.std())
        if sd < 1e-3:
            continue
        # locally bright AND locally desaturated
        mask = ((v > mu + p["local_sigma"] * sd) &
                (s < max(0.0, float(s.mean()) - p["min_sat_margin"] * 0.0) + 255)).astype(np.uint8)
        mask &= (s < 110).astype(np.uint8)
        mask *= 255
        if not mask.any():
            continue
        n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
        for k in range(1, n):
            bx, by, bw, bh, a = stats[k]
            if not (p["min_area"] <= a <= p["max_area"]):
                continue
            if bw > p["max_side"] or bh > p["max_side"]:
                continue
            if max(bw, bh) / max(1, min(bw, bh)) > p["max_aspect"]:
                continue
            cx, cy = float(cent[k][0]) + x0, float(cent[k][1]) + y0
            region = lab[by:by + bh, bx:bx + bw] == k
            z = float((v[by:by + bh, bx:bx + bw][region].mean() - mu) / sd)
            found.append({"x": cx, "y": cy, "area": int(a), "z": z})

    found.sort(key=lambda c: -c["z"])
    out: List[dict] = []
    for c in found:
        if all((c["x"] - o["x"]) ** 2 + (c["y"] - o["y"]) ** 2 > p["dedupe_px"] ** 2
               for o in out):
            out.append(c)
    return out


def global_candidates(img: np.ndarray, v_thr: int = 190, s_thr: int = 90,
                      params: Optional[dict] = None) -> List[dict]:
    """Frozen-baseline style: one global HSV threshold over the whole frame."""
    p = dict(SLICED_DEFAULTS)
    if params:
        p.update(params)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > v_thr) & (hsv[:, :, 1] < s_thr)).astype(np.uint8) * 255
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    out = []
    for k in range(1, n):
        bx, by, bw, bh, a = stats[k]
        if not (p["min_area"] <= a <= p["max_area"]):
            continue
        if bw > p["max_side"] or bh > p["max_side"]:
            continue
        if max(bw, bh) / max(1, min(bw, bh)) > p["max_aspect"]:
            continue
        out.append({"x": float(cent[k][0]), "y": float(cent[k][1]),
                    "area": int(a), "z": float(a)})
    out.sort(key=lambda c: -c["z"])
    return out
