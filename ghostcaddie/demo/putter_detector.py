"""Isolated UNTRAINED STRUCTURAL putter-head re-detection candidate (research-only).

This module is a finite first arm of ``worker-brief.md`` (tiger-redetection-cycle). It
is deliberately NOT a semantic model: there are no trained weights, no COCO detector and
no object class. It detects *compact structural evidence* in the native frame afresh on
EVERY frame and asks a few geometric questions of it:

* is there one compact component whose size, area, local contrast and colour are
  compatible with the head the operator clicked, and whose interior is thick enough that
  it cannot be a thin shaft segment (``compact_core``)?
* does a THIN structure sit next to it whose near endpoint attaches to it and whose far
  endpoint extends away (local thin-shaft endpoint geometry)? A round blob with no such
  attachment is rejected as a ball-like distractor.

Four consequences of that design, stated plainly:

* **Polarity is measured, not assumed.** Some putter heads are bright metal and some are
  dark or neutral-grey, so the contrast polarity is DERIVED from the seed component
  against its own local background (``brighter`` or ``darker``) and re-used for every
  frame. A bright object is not treated as a head, and a dark one is not excluded for
  being dark.
* **The point is the compact CORE's centre, never the blob's.** A head and its shaft can
  merge into one connected component, and the seed's 53 px ROI truncates a longer shaft
  that a 53/97 px search crop fully contains, so the merged blob's area, aspect, diameter
  and centroid all depend on the crop. The reported point and the size/geometry
  comparisons are therefore taken from the compact core and from per-connected-structure
  shaft geometry, not from the merged blob's centroid.
* **A crop is never upscaled.** Everything is measured in native pixels and mapped back
  with :func:`to_native`; :func:`detect` REFUSES a non-unit measurement scale rather than
  dividing coordinates for a crop it did not resize.
* **Nothing is claimed semantically.** A measured frame is reported under the benchmark's
  EXISTING automatic-observation status, distinguished only by this candidate's own method
  names (``putter_detector`` / ``putter_detector_reacquired``): no new status, no class,
  no "this is a putter".

What it refuses to do: follow a template or the surrounding patch (the 31 px masked
template regressed and followed clothing, so this arm carries NO appearance template and
never updates one), ban the bottom of the frame, assume a travel direction, use per-clip
/ per-hole / per-frame special cases, or emit a point it did not measure. All the
constants are module-level and fixed in :data:`PROFILE`; the operator's seed is read once
for initialisation and no later coordinate, manual path or label is ever read.

Two frames of confirmation are required before a re-acquisition is reported, and the
earlier candidate frame stays reported as lost (no backfill). If the seed carries no
usable compact-head evidence the candidate fails closed: nothing is measured at all.

Diagnostic geometry: the measurement is taken in a native crop and converted back with
:func:`to_native`, which is also the hook any future upscaled crop must use.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ghostcaddie.demo.foreground_tracker import (
    CandidateMatch, ForegroundPolicy, FrameDecision, ORIGIN_AUTOMATIC, ORIGIN_DERIVED,
    STATUS_LOST, STATUS_MISSING, STATUS_OBSERVED, SearchRequest)

PROFILE_NAME = "putter-structural-redetect-v1"
METHOD_DETECTOR = "putter_detector"
METHOD_REACQUIRED = "putter_detector_reacquired"
METHOD_SEED = "operator_placement"

# -- read-only seed ROI (no user box is supplied to this arm) ----------------------
SEED_ROI_PX = 53
SEED_ROI_MIN_PX = 13
SEED_PICK_PX = 3.0
SEED_MIN_CONTRAST = 12.0
SEED_MIN_PIXELS = 4

# -- native head-size envelope (frames carry native 6-12 px putter heads) ----------
HEAD_MIN_DIAM_PX = 3.0
HEAD_MAX_DIAM_PX = 14.0
HEAD_CORE_DIST = 2.0
MIN_CORE_PX = 3

# -- per-frame component gates (every value fixed for every clip/hole/frame) -------
MIN_AREA_PX = 3
MAX_AREA_PX = 260
#: Fill is a SCORE term, never a gate: a rotated head-plus-shaft blob fills very little
#: of its own bounding box, so gating on fill would break rotation tolerance while
#: adding nothing the ``compact_core`` test does not already provide. The structural
#: rejections are size, area, compact core, contrast, colour and ball-likeness.
MIN_FILL = 0.10
MAX_FILL = 1.0
MAX_ASPECT = 3.2
MAX_SIZE_DEV = 0.55
SEED_SIZE_SPAN = 0.62
MIN_CONTRAST = 12.0
COLOUR_MIN_SCORE = 0.35
GRAY_DELTA_MAX = 52.0
LAB_DELTA_MAX = 46.0
ROUNDNESS_MAX = 0.84
BALL_ASPECT_MAX = 1.35
IDENTICAL_DIAM_PX = 1.5
IDENTICAL_COLOUR_MIN = 0.60

# -- brightness / metal evidence (grayscale + local contrast + HSV + Lab) ----------
LOCAL_WINDOW = 9
BRIGHT_SIGMA = 0.75
BRIGHT_MIN_DELTA = 6.0
MIN_LOCAL_STD = 3.0
#: A contrast band marks where an object DIFFERS from its neighbourhood, which for an
#: object as wide as the contrast window is only its edges -- its smooth interior is
#: unmarked (and for a dark head there is no metal branch to fall back on). The band's
#: connected region of similar intensity is therefore grown back in, which is what gives
#: a dark or neutral head a compact core at all.
SIMILARITY_TOL = 22.0
METAL_MIN_VALUE = 110.0
METAL_MAX_SAT = 95.0
METAL_MIN_L = 120.0
#: Contrast POLARITY is DERIVED from the seed component against its own local background.
#: Some putter heads are bright metal, others are dark or neutral-grey: treating "bright"
#: as the definition of a head would be a false-acceptance rule, not a cue. The metal
#: branch below is a bright-object cue and is therefore only used for bright polarity.
#: Exactly one polarity is ever in force: a both-polarity union would admit the halo of
#: opposite-contrast pixels around any object (in a small seed ROI, around a dark head on
#: a bright green, that halo is the whole ROI).
POLARITY_BRIGHTER = "brighter"
POLARITY_DARKER = "darker"
#: How far out a thin candidate pixel is checked for a genuinely LINE-like
#: neighbourhood: a line pixel has foreground on BOTH sides of it along one axis, while a
#: blob's corner or rim only has foreground on one side. That asymmetry is what separates
#: a shaft from a head's own boundary pixels, which no local density threshold can do (a
#: 2 px line and a rectangle's corner fill almost the same fraction of a 5x5 window).
LINE_RING_PX = 2
#: The four antipodal offset pairs tested at that radius: vertical, horizontal, and the
#: two diagonals.
LINE_ANTIPODAL_PAIRS = (((-2, 0), (2, 0)), ((0, -2), (0, 2)),
                        ((-1, 2), (1, -2)), ((-2, -1), (2, 1)))

# -- thin-structure (shaft) geometry ----------------------------------------------
THIN_DIST = 1.6
SHAFT_MIN_LEN_PX = 5.0
SHAFT_MIN_PIXELS = 6
SHAFT_MAX_THICK_PX = 3.2
SHAFT_ATTACH_PX = 3.5
#: How close a thin structure's near endpoint must come to the head's compact CORE
#: before it counts as attached (the core is inset by HEAD_CORE_DIST).
SHAFT_REACH_PX = HEAD_CORE_DIST + SHAFT_ATTACH_PX
#: How far around the head's core thin structure is looked for.
SHAFT_SEARCH_PX = 16.0

# -- point policy (inherits foreground_tracker.ForegroundPolicy) -------------------
SEARCH_RADIUS_PX = 26.0
REACQ_RADIUS_PX = 48.0
MAX_STEP_PX = 30.0
REACQ_MAX_STEP_PX = 52.0
LOSS_FRAMES = 3
REACQ_WINDOW_FRAMES = 24
REACQ_CONFIRM_FRAMES = 2
MIN_SCORE = 0.55
SEED_IDENTITY_MIN_SCORE = 0.45
REACQ_MIN_SCORE = 0.70
AMBIGUITY_MARGIN = 0.06

WEIGHTS = {"contrast": 0.20, "bright": 0.14, "size": 0.20, "colour": 0.16,
           "shape": 0.14, "shaft": 0.16}

LIMITATIONS = (
    "Untrained STRUCTURAL detector: it segments compact bright/metal-like evidence and "
    "checks thin-structure endpoint geometry. It has no trained weights, no object class "
    "and cannot recognise a putter as an object.",
    "No template and no following: every frame is segmented afresh from the frame alone; "
    "there is no appearance model to drift onto clothing.",
    "The operator's seed is read once for initialisation; no later coordinate, manual "
    "path, correction or label is read by this candidate.",
    "A refused seed means nothing is measured for the whole clip: the candidate fails "
    "closed instead of guessing a head.",
    "Shaft support is one weighted evidence term and the tie-breaker of an otherwise "
    "ambiguous pair; it is not proof that a club is present. It needs a genuinely thin "
    "structure (about 1-2 px at this native scale): a thicker shaft, or a head whose "
    "shaft merges into it at a wider angle, is accepted as a head but reported without "
    "shaft support.",
    "Synthetic fixtures prove the gates and the policy, never accuracy on real clips.",
)

PROFILE: Dict[str, Any] = {
    "head_region": "each separate thick core, dilated inside foreground mask",
    "head_reconstruction_kernel": [3, 3],
    "head_reconstruction_iterations": 2,
    "shaft_excludes_other_thick_components": True,
    "seed_context_is_fixed_larger_native_crop": True,
    "connectivity": 8,
    "distance_transform_mask_size": 3,
    "ring_padding_px": 3,
    "shape_normalization": [0.80, 0.20],
    "contrast_normalization": 26.0,
    "brightness_normalization": 100.0,
    "identity_weights": [0.5, 0.5],
    "closing_kernel": [3, 3],
    "name": PROFILE_NAME,
    "kind": "untrained structural detector",
    "appearance_template": None,
    "template_following": False,
    "appearance_updates": 0,
    "seed_inputs": ["frame", "x", "y"],
    "seed_roi_px": SEED_ROI_PX,
    "seed_roi_min_px": SEED_ROI_MIN_PX,
    "head_diam_px_range": [HEAD_MIN_DIAM_PX, HEAD_MAX_DIAM_PX],
    "cues": ["grayscale_local_contrast", "hsv", "lab", "contours",
             "min_area_rect_geometry", "distance_transform", "thin_line_fit"],
    "area_px_range": [MIN_AREA_PX, MAX_AREA_PX],
    "fill_range": [MIN_FILL, MAX_FILL],
    "fill_is_a_gate": False,
    "max_aspect": MAX_ASPECT,
    "max_size_deviation": MAX_SIZE_DEV,
    "seed_pick_px": SEED_PICK_PX,
    "seed_min_contrast": SEED_MIN_CONTRAST,
    "seed_min_pixels": SEED_MIN_PIXELS,
    "min_contrast": MIN_CONTRAST,
    "colour_min_score": COLOUR_MIN_SCORE,
    "seed_size_span": SEED_SIZE_SPAN,
    "ball_aspect_max": BALL_ASPECT_MAX,
    "identical_diam_px": IDENTICAL_DIAM_PX,
    "identical_colour_min": IDENTICAL_COLOUR_MIN,
    "head_core_dist_px": HEAD_CORE_DIST,
    "local_window_px": LOCAL_WINDOW,
    "bright_sigma": BRIGHT_SIGMA,
    "bright_min_delta": BRIGHT_MIN_DELTA,
    "min_local_std": MIN_LOCAL_STD,
    "metal_min_value": METAL_MIN_VALUE,
    "metal_max_sat": METAL_MAX_SAT,
    "metal_min_l": METAL_MIN_L,
    "similarity_tol": SIMILARITY_TOL,
    "polarity_values": [POLARITY_BRIGHTER, POLARITY_DARKER],
    "polarity_source": ("DERIVED from the seed component against its own local background "
                        "(brighter or darker) and re-used for every frame; a bright head "
                        "is never assumed"),
    "grey_delta_max": GRAY_DELTA_MAX,
    "lab_delta_max": LAB_DELTA_MAX,
    "roundness_max": ROUNDNESS_MAX,
    "min_core_px": MIN_CORE_PX,
    "shaft_min_len_px": SHAFT_MIN_LEN_PX,
    "shaft_min_pixels": SHAFT_MIN_PIXELS,
    "shaft_max_thick_px": SHAFT_MAX_THICK_PX,
    "shaft_attach_px": SHAFT_ATTACH_PX,
    "shaft_reach_px": SHAFT_REACH_PX,
    "shaft_search_px": SHAFT_SEARCH_PX,
    "thin_dist_px": THIN_DIST,
    "line_ring_px": LINE_RING_PX,
    "line_antipodal_pairs": [list(pair) for pair in LINE_ANTIPODAL_PAIRS],
    "search_radius_px": SEARCH_RADIUS_PX,
    "reacq_radius_px": REACQ_RADIUS_PX,
    "max_step_px": MAX_STEP_PX,
    "reacq_max_step_px": REACQ_MAX_STEP_PX,
    "loss_frames": LOSS_FRAMES,
    "reacq_window_frames": REACQ_WINDOW_FRAMES,
    "reacq_confirm_frames": REACQ_CONFIRM_FRAMES,
    "min_score": MIN_SCORE,
    "seed_identity_min_score": SEED_IDENTITY_MIN_SCORE,
    "reacq_min_score": REACQ_MIN_SCORE,
    "ambiguity_margin": AMBIGUITY_MARGIN,
    "weights": dict(WEIGHTS),
    "velocity_extrapolation": False,
    "global_bottom_ban": False,
    "methods": {"observation": METHOD_DETECTOR, "reacquired": METHOD_REACQUIRED,
                "seed_anchor": METHOD_SEED},
}


# --------------------------------------------------------------------------
# frame and crop helpers (cv2/numpy imported lazily, like foreground_tracker)

def _cv():
    import cv2
    return cv2


def _np():
    import numpy as np
    return np


def to_native(x: float, y: float, *, origin_x: float, origin_y: float,
              scale: float = 1.0) -> Tuple[float, float]:
    """Map a measurement taken in a crop back to NATIVE frame pixels.

    ``scale`` is the factor a crop was upscaled by for measurement (1.0 in this arm:
    nothing is upscaled). Any future scaled arm MUST route its coordinates through
    here, otherwise a crop coordinate would be emitted as a frame coordinate.
    """
    if scale <= 0:
        raise ValueError("crop scale must be positive")
    return (float(origin_x) + float(x) / float(scale),
            float(origin_y) + float(y) / float(scale))


def _crop(image, x: float, y: float, radius: float):
    """A native square crop around ``(x, y)``: the crop, its native origin, clipped?"""
    np = _np()
    height, width = image.shape[:2]
    half = int(math.floor(radius))
    center_x, center_y = int(round(x)), int(round(y))
    x0, y0 = center_x - half, center_y - half
    x1, y1 = center_x + half + 1, center_y + half + 1
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return None, (cx0, cy0), True
    clipped = (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1)
    return np.ascontiguousarray(image[cy0:cy1, cx0:cx1]), (cx0, cy0), bool(clipped)


def _maps(crop):
    """Grayscale, HSV, Lab and whether the frame actually carried colour."""
    cv2, np = _cv(), _np()
    colour_known = crop.ndim == 3 and crop.shape[2] >= 3
    if colour_known:
        bgr = np.ascontiguousarray(crop[:, :, :3])
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop if crop.ndim == 2 else crop[:, :, 0]
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return (gray, cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV),
            cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB), bool(colour_known))


def _grow_similar(mask, gray):
    """Union the contrast band with the object it belongs to.

    The band is the seed; the object is the set of pixels CONNECTED to that seed whose
    intensity is within :data:`SIMILARITY_TOL` of the band's own median. Components of
    similar intensity that are not connected to the band (another dark object elsewhere in
    the crop) are left out.
    """
    cv2, np = _cv(), _np()
    band = mask > 0
    if not band.any():
        return mask
    reference = float(np.median(gray[band]))
    similar = np.abs(gray.astype(np.float32) - reference) <= SIMILARITY_TOL
    count, labels = cv2.connectedComponents((band | similar).astype(np.uint8), 8)
    attached = sorted(set(int(label) for label in np.unique(labels[band])) - {0})
    return (np.isin(labels, attached).astype(np.uint8)) * 255


def _fill_holes(mask):
    """Add the regions fully enclosed by the mask: an object's own smooth interior.

    Local contrast only marks where an object differs from its NEIGHBOURHOOD, so the
    interior of an object wider than the contrast window is not marked at all -- for a
    dark head (which has no metal branch to fall back on) that leaves a hollow ring with
    no compact core, and the head would be refused as a thin structure. The enclosed
    region is part of the object.
    """
    cv2, np = _cv(), _np()
    height, width = mask.shape[:2]
    padded = np.zeros((height + 2, width + 2), dtype=np.uint8)
    padded[1:height + 1, 1:width + 1] = (mask > 0).astype(np.uint8)
    flooded = padded.copy()
    cv2.floodFill(flooded, np.zeros((height + 4, width + 4), np.uint8), (0, 0), 1)
    enclosed = (padded > 0) | ((padded == 0) & (flooded == 0))
    return (enclosed[1:height + 1, 1:width + 1].astype(np.uint8)) * 255


def _evidence_mask(gray, hsv, lab, colour_known, polarity):
    """Foreground evidence at ONE polarity: pixels differing from their local background.

    ``polarity`` is ``brighter`` or ``darker``. Metal-like colour must also
    differ from its local background; pale grass alone is not head evidence.
    """
    cv2, np = _cv(), _np()
    f32 = gray.astype(np.float32)
    size = (LOCAL_WINDOW, LOCAL_WINDOW)
    mean = cv2.blur(f32, size)
    std = np.sqrt(np.maximum(cv2.blur(f32 * f32, size) - mean * mean, 0.0))
    spread = BRIGHT_SIGMA * np.maximum(std, MIN_LOCAL_STD)
    brighter = (f32 >= mean + spread) & (f32 >= mean + BRIGHT_MIN_DELTA)
    darker = (f32 <= mean - spread) & (f32 <= mean - BRIGHT_MIN_DELTA)
    if polarity == POLARITY_BRIGHTER:
        local = brighter
    elif polarity == POLARITY_DARKER:
        local = darker
    else:
        raise ValueError(f"unknown polarity {polarity!r}; use {POLARITY_BRIGHTER!r} or "
                         f"{POLARITY_DARKER!r}")
    metal = local
    if colour_known and polarity == POLARITY_BRIGHTER:
        metal = (local | ((hsv[:, :, 2].astype(np.float32) >= METAL_MIN_VALUE)
                          & (hsv[:, :, 1].astype(np.float32) <= METAL_MAX_SAT)
                          & (lab[:, :, 0].astype(np.float32) >= METAL_MIN_L)))
    metal = metal & (f32 >= mean + BRIGHT_MIN_DELTA)
    mask = ((local | metal).astype(np.uint8)) * 255
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return _fill_holes(_grow_similar(closed, gray))


def _colour_mean(hsv, lab, box, comp, colour_known, inside_mean) -> dict:
    x, y, w, h = box
    values = {"gray": round(float(inside_mean), 2), "lab": None, "hsv": None}
    if colour_known:
        selection = comp > 0
        values["lab"] = [round(float(lab[y:y + h, x:x + w, index][selection].mean()), 2)
                         for index in range(3)]
        values["hsv"] = [round(float(hsv[y:y + h, x:x + w, index][selection].mean()), 2)
                         for index in range(3)]
    return values


def _ring_mean(gray, box, pad: int = 3) -> float:
    """Mean of the pixels just outside the component's box: its local background."""
    np = _np()
    x, y, w, h = box
    height, width = gray.shape[:2]
    outer = gray[max(0, y - pad):min(height, y + h + pad),
                 max(0, x - pad):min(width, x + w + pad)]
    inner = gray[y:y + h, x:x + w]
    count = float(outer.size - inner.size)
    if count <= 0:
        return float(gray.mean())
    return float((float(outer.sum()) - float(inner.sum())) / count)


def _point_box_distance(px: float, py: float, box) -> float:
    x, y, w, h = box
    dx = max(x - px, 0.0, px - (x + w))
    dy = max(y - py, 0.0, py - (y + h))
    return math.hypot(dx, dy)


def _line_like(mask):
    """Pixels with foreground on BOTH sides along one axis: a line, not a blob's edge.

    A 2 px line's pixel and a rectangle's corner pixel fill almost the same fraction of a
    small window, so density cannot tell them apart; antipodal coverage can. This is what
    stops a head's own corner pixels from being fitted as a "shaft".
    """
    np = _np()
    height, width = mask.shape[:2]
    pad = LINE_RING_PX
    padded = np.zeros((height + 2 * pad, width + 2 * pad), dtype=bool)
    padded[pad:pad + height, pad:pad + width] = mask > 0

    def shifted(dy, dx):
        return padded[pad + dy:pad + dy + height, pad + dx:pad + dx + width]

    line = None
    for (near_y, near_x), (far_y, far_x) in LINE_ANTIPODAL_PAIRS:
        pair = shifted(near_y, near_x) & shifted(far_y, far_x)
        line = pair if line is None else (line | pair)
    return line


def _shaft_geometry(mask, dist, core_box) -> dict:
    """Local thin-structure endpoint geometry around a head's compact CORE.

    Thin, line-like pixels (small distance-transform radius AND foreground on both sides
    along one axis, see :func:`_line_like`) are grouped by connectivity and each group is
    fitted separately (see :func:`_structure_geometry`). A structure supports the head
    only when it is long enough, thin enough, AND asymmetric about the head: its NEAR
    endpoint attaches to the compact core, its far endpoint extends at least
    ``SHAFT_MIN_LEN_PX`` beyond it, and it reaches strictly further away. Distance is
    measured to the head's compact core rather than to its bounding box, which is what
    makes this work when a head and its shaft merge into one component; and requiring a
    real reach away from the core is what refuses a head's own outline, a rectangle's
    corner pixels and a disk's rim -- all of which are line-like but do not leave the
    head.
    """
    cv2, np = _cv(), _np()
    x, y, w, h = core_box
    reach = int(round(SHAFT_SEARCH_PX))
    height, width = mask.shape[:2]
    x0, y0 = max(0, x - reach), max(0, y - reach)
    x1, y1 = min(width, x + w + reach), min(height, y + h + reach)
    window = np.zeros_like(mask)
    window[y0:y1, x0:x1] = 255
    thin = ((mask > 0) & (dist < THIN_DIST) & _line_like(mask) & (window > 0)).astype(np.uint8)
    label_count, labels = cv2.connectedComponents(thin, 8)
    structures = []
    for label in range(1, label_count):
        pixels = np.argwhere(labels == label)[:, ::-1].astype(np.float32)
        if pixels.shape[0] >= SHAFT_MIN_PIXELS:
            structures.append(_structure_geometry(cv2, pixels, core_box))
    geometry: Dict[str, Any] = {"supported": False, "thin_pixels": int(thin.sum()),
                                "structures": len(structures), "supported_structures": 0,
                                "length_px": 0.0, "thickness_px": 0.0, "near_end": None,
                                "far_end": None, "attach_distance_px": None,
                                "extend_distance_px": None, "reach_px": SHAFT_REACH_PX,
                                "reason": "no_thin_structure_near_head"}
    if not structures:
        return geometry
    supported = [item for item in structures if item["supported"]]
    chosen = max(supported or structures, key=lambda item: item["length_px"])
    geometry.update(chosen)
    geometry["structures"] = len(structures)
    geometry["supported_structures"] = len(supported)
    return geometry


def _structure_geometry(cv2, pixels, core_box) -> dict:
    """One CONNECTED thin structure: is it a shaft attached to, and reaching away from,
    the head's compact core?

    Each structure is fitted on its own pixels: fitting one line through pixels that
    belong to different structures (a head's own outline plus a shaft plus a second
    head's outline) would draw a line between two objects and read as support for both.
    """
    count = int(pixels.shape[0])
    line = cv2.fitLine(pixels, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    dx, dy = float(line[0]), float(line[1])
    norm = math.hypot(dx, dy) or 1.0
    dx, dy = dx / norm, dy / norm
    projection = ((pixels[:, 0] - float(line[2])) * dx
                  + (pixels[:, 1] - float(line[3])) * dy)
    low, high = float(projection.min()), float(projection.max())
    first = (float(line[2]) + dx * low, float(line[3]) + dy * low)
    last = (float(line[2]) + dx * high, float(line[3]) + dy * high)
    length = high - low
    thickness = count / max(length, 1.0)
    first_gap = _point_box_distance(first[0], first[1], core_box)
    last_gap = _point_box_distance(last[0], last[1], core_box)
    near, far = (first, last) if first_gap <= last_gap else (last, first)
    near_gap, far_gap = min(first_gap, last_gap), max(first_gap, last_gap)
    geometry = {"supported": False, "pixels": count, "length_px": round(length, 2),
                "thickness_px": round(thickness, 2),
                "near_end": [round(near[0], 2), round(near[1], 2)],
                "far_end": [round(far[0], 2), round(far[1], 2)],
                "attach_distance_px": round(near_gap, 2),
                "extend_distance_px": round(far_gap, 2),
                "reach_px": SHAFT_REACH_PX, "reason": ""}
    if length < SHAFT_MIN_LEN_PX:
        geometry["reason"] = "thin_structure_too_short"
    elif thickness > SHAFT_MAX_THICK_PX:
        geometry["reason"] = "thin_structure_not_thin_enough"
    elif near_gap > SHAFT_REACH_PX:
        geometry["reason"] = "thin_structure_endpoint_does_not_attach"
    elif far_gap < SHAFT_MIN_LEN_PX or far_gap <= near_gap:
        geometry["reason"] = "thin_structure_does_not_extend_away_from_head"
    else:
        geometry["supported"] = True
        geometry["reason"] = "thin_structure_endpoint_attaches_to_head"
    return geometry


def _components(gray, hsv, lab, mask, dist, colour_known) -> List[dict]:
    """Split thick cores and reconstruct each head separately from its thin shaft."""
    cv2, np = _cv(), _np()
    count, core_labels, core_stats, _ = cv2.connectedComponentsWithStats(
        (dist >= HEAD_CORE_DIST).astype(np.uint8), 8)
    out = []
    kernel = np.ones((3, 3), np.uint8)
    for label in range(1, count):
        core = core_labels == label
        core_pixels = np.argwhere(core)
        if len(core_pixels) > MAX_AREA_PX:
            continue
        # Reconstruct the head thickness without reattaching an arbitrarily long shaft.
        selection = (cv2.dilate(core.astype(np.uint8), kernel, iterations=2) > 0) & (mask > 0)
        pixels = np.argwhere(selection)
        if not len(pixels):
            continue
        y, x = pixels.min(axis=0)
        bottom, right = pixels.max(axis=0)
        w, h = int(right-x+1), int(bottom-y+1)
        x, y = int(x), int(y)
        area = len(pixels)
        if not MIN_AREA_PX <= area <= MAX_AREA_PX:
            continue
        points = pixels[:, ::-1].astype(np.float32)
        centre = points.mean(axis=0)
        core_centre = core_pixels[:, ::-1].mean(axis=0)
        cy, cx = core_pixels.min(axis=0)
        cby, cbx = core_pixels.max(axis=0)
        core_box = [int(cx), int(cy), int(cbx-cx+1), int(cby-cy+1)]
        rect_w, rect_h = cv2.minAreaRect(points)[1]
        long_side, short_side = float(max(rect_w,rect_h)), float(min(rect_w,rect_h))
        radii = np.linalg.norm(points-centre, axis=1)
        comp = selection[y:y+h,x:x+w].astype(np.uint8)
        inside = float(gray[selection].mean())
        background = _ring_mean(gray,(x,y,w,h))
        component = {
            'label':label,'area':area,'bbox_crop':[x,y,w,h],
            'centroid_crop':centre.tolist(),
            'core_centroid_crop':core_centre.tolist(),'core_box_crop':core_box,
            'aspect':max(w,h)/max(min(w,h),1),'fill':area/(w*h),
            'equivalent_diam_px':math.sqrt(4*area/math.pi),
            'long_side_px':long_side,'short_side_px':short_side,
            'elongation':long_side/max(short_side,.5),
            'roundness':area/(math.pi*float(radii.max())**2+1e-6),
            'core_px':len(core_pixels),'inside_mean':inside,
            'local_background_mean':background,'contrast':inside-background,
            'colour':_colour_mean(hsv,lab,(x,y,w,h),comp,colour_known,inside)}
        # Thick clothing/shoes have thin boundary pixels too. Exclude those from
        # shaft support; only this head's own core may retain adjacent thin pixels.
        other_thick = (dist >= HEAD_CORE_DIST) & ~core
        exclusion = cv2.dilate(other_thick.astype(np.uint8),kernel,iterations=2)>0
        shaft_mask = mask.copy()
        shaft_mask[exclusion] = 0
        component['shaft'] = _shaft_geometry(shaft_mask,dist,core_box)
        out.append(component)
    return out


def _analyse(crop, polarity) -> dict:
    """Everything one crop yields: maps, components and their shaft geometry."""
    cv2, np = _cv(), _np()
    gray, hsv, lab, colour_known = _maps(crop)
    mask = _evidence_mask(gray, hsv, lab, colour_known, polarity)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    return {"gray": gray, "colour_known": colour_known, "mask": mask,
            "polarity": polarity,
            "components": _components(gray, hsv, lab, mask, dist, colour_known)}


def _colour_score(one: dict, other: dict, colour_known: bool) -> float:
    if not colour_known or one.get("lab") is None or other.get("lab") is None:
        return _clamp01(1.0 - abs(one["gray"] - other["gray"]) / GRAY_DELTA_MAX)
    delta = math.sqrt(sum((one["lab"][i] - other["lab"][i]) ** 2 for i in range(3)))
    return _clamp01(1.0 - delta / LAB_DELTA_MAX)


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else float(value))


def _shape_score(component: dict) -> float:
    fill_term = _clamp01((component["fill"] - MIN_FILL) / (0.80 - MIN_FILL))
    not_round = _clamp01((ROUNDNESS_MAX - component["roundness"]) / 0.20)
    return 0.5 * fill_term + 0.5 * not_round


def _gates(component: dict, seed: dict, colour_known: bool) -> Dict[str, bool]:
    ratio = component["equivalent_diam_px"] / max(seed["equivalent_diam_px"], 1e-6)
    return {
        "area": MIN_AREA_PX <= component["area"] <= MAX_AREA_PX,
        "aspect": component["aspect"] <= MAX_ASPECT,
        "compact_core": component["core_px"] >= MIN_CORE_PX,
        "contrast": abs(component["contrast"]) >= MIN_CONTRAST,
        "size": (1.0 - MAX_SIZE_DEV) <= ratio <= (1.0 + MAX_SIZE_DEV),
        "colour": _colour_score(component["colour"], seed["colour"],
                                colour_known) >= COLOUR_MIN_SCORE,
        "not_ball_like": not (component["roundness"] >= ROUNDNESS_MAX
                              and component["aspect"] <= BALL_ASPECT_MAX
                              and not component["shaft"]["supported"]),
    }


def _scored(component: dict, seed: dict, colour_known: bool) -> dict:
    size = _clamp01(1.0 - abs(component["equivalent_diam_px"]
                              - seed["equivalent_diam_px"])
                    / max(SEED_SIZE_SPAN * seed["equivalent_diam_px"], 1e-6))
    colour = _colour_score(component["colour"], seed["colour"], colour_known)
    bright = _clamp01(1.0 - abs(component["inside_mean"] - seed["inside_mean"]) / 100.0)
    values = {"contrast": _clamp01((abs(component["contrast"]) - MIN_CONTRAST) / 26.0),
              "bright": bright, "size": size, "colour": colour,
              "shape": _shape_score(component),
              "shaft": 1.0 if component["shaft"]["supported"] else 0.0}
    score = sum(WEIGHTS[key] * value for key, value in values.items())
    identity = 0.5 * size + 0.5 * colour
    return {"component": component, "terms": {key: round(value, 4)
                                              for key, value in values.items()},
            "score": round(score, 4), "seed_identity": round(identity, 4)}


def detect(image, request, seed: dict, *, scale: float = 1.0):
    """Measure one frame: one compact head candidate, or an honest refusal.

    Returns ``(CandidateMatch, evidence)``. The detector applies the STRUCTURAL gates
    only (size/area/contrast/colour/shape/compact-core and the ball-like rejection,
    plus the two-identical-heads abstention it alone can see); the score, seed-identity
    and step thresholds stay in the point policy, where they are testable.
    """
    if float(scale) != 1.0:
        raise ValueError("this arm measures in native pixels and never upscales a crop: "
                         "measurement scale must be 1.0")
    crop, origin, clipped = _crop(image, request.anchor_x, request.anchor_y,
                                  float(request.radius_px))
    evidence: Dict[str, Any] = {"frame": int(request.frame), "clipped": bool(clipped),
                                "crop_origin_native": [int(origin[0]), int(origin[1])],
                                "crop_scale": float(scale),
                                "anchor_native": [round(float(request.anchor_x), 2),
                                                  round(float(request.anchor_y), 2)],
                                "reacquiring": bool(request.reacquiring),
                                "components": [], "admissible": 0, "chosen": None,
                                "reason": ""}
    if crop is None:
        evidence["reason"] = "search_window_off_frame"
        return CandidateMatch(int(request.frame), False,
                              reason="search_window_off_frame",
                              clipped=True), evidence
    analysis = _analyse(crop, seed.get("polarity", POLARITY_BRIGHTER))
    evidence["polarity"] = analysis["polarity"]
    colour_known = analysis["colour_known"]
    admissible: List[dict] = []
    for component in analysis["components"]:
        gates = _gates(component, seed, colour_known)
        scored = _scored(component, seed, colour_known)
        box = component["bbox_crop"]
        core_x, core_y = component["core_centroid_crop"]
        native_x, native_y = to_native(core_x, core_y, origin_x=origin[0],
                                       origin_y=origin[1], scale=scale)
        record = {"bbox_crop": box,
                  "bbox_native": [int(origin[0] + box[0]), int(origin[1] + box[1]),
                                  int(box[2]), int(box[3])],
                  "centroid_crop": component["centroid_crop"],
                  "core_centroid_crop": component["core_centroid_crop"],
                  "point_native": [round(native_x, 2), round(native_y, 2)],
                  "point_note": ("the reported point is the compact head CORE's centre, "
                                 "converted from crop to native pixels"),
                  "area": component["area"],
                  "equivalent_diam_px": component["equivalent_diam_px"],
                  "aspect": component["aspect"], "fill": component["fill"],
                  "contrast": component["contrast"], "roundness": component["roundness"],
                  "core_px": component["core_px"], "elongation": component["elongation"],
                  "colour": component["colour"], "shaft": component["shaft"],
                  "terms": scored["terms"], "score": scored["score"],
                  "seed_identity": scored["seed_identity"], "gates": gates,
                  "gates_passed": all(gates.values())}
        evidence["components"].append(record)
        if all(gates.values()):
            admissible.append({"record": record, "match": scored})
    evidence["admissible"] = len(admissible)
    if not admissible:
        failed: Dict[str, int] = {}
        for record in evidence["components"]:
            for name, ok in record["gates"].items():
                if not ok:
                    failed[name] = failed.get(name, 0) + 1
        evidence["reason"] = "no_admissible_component"
        evidence["failed_gates"] = dict(sorted(failed.items()))
        return CandidateMatch(int(request.frame), False,
                              reason="no_admissible_component"), evidence
    admissible.sort(key=lambda item: -item["match"]["score"])
    best = admissible[0]
    second_score = admissible[1]["match"]["score"] if len(admissible) > 1 else 0.0
    if len(admissible) > 1:
        runner = admissible[1]
        gap = best["match"]["score"] - runner["match"]["score"]
        identical = (abs(best["record"]["equivalent_diam_px"]
                         - runner["record"]["equivalent_diam_px"]) <= IDENTICAL_DIAM_PX
                     and _colour_score(best["record"]["colour"],
                                       runner["record"]["colour"],
                                       colour_known) >= IDENTICAL_COLOUR_MIN)
        if gap < AMBIGUITY_MARGIN and identical:
            # Two near-identical compact heads with nothing structural to tell them apart:
            # abstain. There is deliberately no "prefer the one with a shaft" branch here:
            # shaft support is worth more than the ambiguity margin in the composite score
            # (WEIGHTS["shaft"] > AMBIGUITY_MARGIN), so a head that carries a thin shaft
            # can never be part of a within-margin pair in the first place -- the shaft is
            # what separates the scores.
            evidence["reason"] = "ambiguous_two_identical_heads"
            evidence["candidates_top2"] = [best["record"], runner["record"]]
            return CandidateMatch(int(request.frame), False,
                                  reason="ambiguous_two_identical_heads"), evidence
    chosen = best["record"]
    evidence["chosen"] = chosen
    evidence["reason"] = "admissible_compact_head_evidence"
    native_x, native_y = chosen["point_native"]
    transform = ("head+shaft" if chosen["shaft"]["supported"] else "head-only")
    return CandidateMatch(int(request.frame), True, x=native_x, y=native_y,
                          score=float(chosen["score"]),
                          seed_score=float(chosen["seed_identity"]),
                          second_score=float(round(second_score, 4)),
                          transform=transform, clipped=bool(clipped), reason=""), evidence


def _seed_candidates(crop, polarity, origin, seed_relative):
    """Seed-viable components of one ROI at ONE polarity, plus every record seen."""
    analysis = _analyse(crop, polarity)
    records: List[dict] = []
    viable: List[dict] = []
    for component in analysis["components"]:
        box = component["bbox_crop"]
        gates = {
            "pixels": component["area"] >= SEED_MIN_PIXELS,
            "compact_core": component["core_px"] >= MIN_CORE_PX,
            "area": component["area"] <= MAX_AREA_PX,
            "aspect": component["aspect"] <= MAX_ASPECT,
            "contrast": abs(component["contrast"]) >= SEED_MIN_CONTRAST,
            "native_size": HEAD_MIN_DIAM_PX <= component["equivalent_diam_px"]
            <= HEAD_MAX_DIAM_PX,
            "contains_seed": _point_box_distance(seed_relative[0], seed_relative[1],
                                                 box) <= SEED_PICK_PX,
        }
        record = {"bbox_crop": box,
                  "bbox_native": [int(origin[0] + box[0]), int(origin[1] + box[1]),
                                  int(box[2]), int(box[3])],
                  "area": component["area"],
                  "equivalent_diam_px": component["equivalent_diam_px"],
                  "aspect": component["aspect"], "fill": component["fill"],
                  "contrast": component["contrast"], "roundness": component["roundness"],
                  "core_px": component["core_px"], "colour": component["colour"],
                  "gates": gates, "gates_passed": all(gates.values())}
        records.append(record)
        if all(gates.values()):
            viable.append(record)
    return analysis, records, viable


def describe_seed(image, seed_x: float, seed_y: float):
    """The read-only seed evidence, or an honest refusal.

    The 53 px ROI is segmented at each polarity in turn (the polarity is not known yet) and
    exactly one of them must yield exactly one compact component inside the native
    head-size envelope that contains (or is within :data:`SEED_PICK_PX` of) the seed pixel.
    That component's contrast sign against its own local background IS the seed's polarity,
    which every later frame then re-uses -- a bright head is never assumed, and a dark one
    is never excluded for being dark. A pass yielding two candidates, two passes each
    yielding one, or no pass yielding any, is refused rather than guessed.

    Nothing else is read: never a label, and never a coordinate from another frame.
    """
    crop, origin, clipped = _crop(image, seed_x, seed_y, SEED_ROI_PX / 2.0)
    evidence: Dict[str, Any] = {"seed_native": [round(float(seed_x), 2),
                                                round(float(seed_y), 2)],
                                "roi_px": SEED_ROI_PX, "clipped": bool(clipped),
                                "crop_origin_native": [int(origin[0]), int(origin[1])],
                                "polarity_passes": {}, "components": [], "reason": ""}
    if crop is None:
        evidence["reason"] = "seed_roi_off_frame"
        return None, "seed_roi_off_frame", evidence
    seed_relative = (float(seed_x) - origin[0], float(seed_y) - origin[1])
    passes: Dict[str, Any] = {}
    accepted: List[tuple] = []
    ambiguous = False
    for polarity in (POLARITY_BRIGHTER, POLARITY_DARKER):
        analysis, records, viable = _seed_candidates(crop, polarity, origin, seed_relative)
        passes[polarity] = {"components": records, "viable": len(viable),
                            "colour_known": bool(analysis["colour_known"])}
        ambiguous = ambiguous or len(viable) > 1
        if len(viable) == 1:
            accepted.append((polarity, dict(viable[0])))
    evidence["polarity_passes"] = passes
    if len(accepted) > 1:
        evidence["reason"] = "seed_evidence_ambiguous_polarity"
        return None, "seed_evidence_ambiguous_polarity", evidence
    if not accepted:
        reason = ("seed_evidence_ambiguous" if ambiguous
                  else "seed_has_no_compact_head_evidence")
        evidence["reason"] = reason
        return None, reason, evidence
    polarity, chosen = accepted[0]
    evidence["components"] = passes[polarity]["components"]
    evidence["polarity"] = polarity
    descriptor = dict(chosen)
    descriptor.update({"colour_known": passes[polarity]["colour_known"],
                       "roi_px": SEED_ROI_PX,
                       "roi_origin_native": [int(origin[0]), int(origin[1])],
                       "roi_clipped": bool(clipped),
                       "equivalent_diam_px": float(descriptor["equivalent_diam_px"]),
                       "inside_mean": float(descriptor["colour"]["gray"]),
                       "polarity": polarity,
                       "polarity_note": ("derived from this seed component against its own "
                                         "local background and re-used for every frame; a "
                                         "bright head is never assumed")})
    evidence["reason"] = "seed_compact_head_evidence_accepted"
    evidence["chosen"] = descriptor
    return descriptor, None, evidence


class PutterPolicy(ForegroundPolicy):
    """``ForegroundPolicy`` with this arm's own fixed constants and honest method names.

    Inherits the accept/abstain/lose/re-acquire state machine unchanged (bounded step,
    loss declaration, finite ELAPSED re-acquisition window, causal two-frame
    confirmation with no backfill, structural cut reset), and changes exactly three
    things: the search is a plain square around the last OBSERVED anchor (no velocity
    extrapolation, so no assumed travel direction and a stationary head at address is
    just a stationary head), nothing is ever followed/appearance-updated, and an
    accepted frame is reported under this candidate's own method names.
    """

    def __init__(self, *, seed_frame: int, seed_x: float, seed_y: float,
                 target: str = "clubhead") -> None:
        super().__init__(seed_frame=seed_frame, seed_x=seed_x, seed_y=seed_y,
                         target=target, loss_frames=LOSS_FRAMES,
                         search_radius_px=SEARCH_RADIUS_PX,
                         reacq_radius_px=REACQ_RADIUS_PX, max_step_px=MAX_STEP_PX,
                         reacq_max_step_px=REACQ_MAX_STEP_PX, min_score=MIN_SCORE,
                         seed_identity_min_score=SEED_IDENTITY_MIN_SCORE,
                         reacq_min_score=REACQ_MIN_SCORE,
                         reacq_confirm_frames=REACQ_CONFIRM_FRAMES,
                         reacq_window_frames=REACQ_WINDOW_FRAMES,
                         ambiguity_margin=AMBIGUITY_MARGIN)

    def request_for(self, frame: int) -> Optional[SearchRequest]:
        frame = int(frame)
        if not self.measure or self.reset or frame <= self.seed_frame:
            return None
        anchor = self.reacq_anchor if self.lost else self.last
        if self.lost and self.reacq_frames_left <= 0:
            return None
        if anchor is None:
            return None
        if self.lost:
            return SearchRequest(frame, anchor[0], anchor[1], self.reacq_radius_px, True,
                                 "bounded re-detection in a square around the last "
                                 "OBSERVED physical anchor inside a finite elapsed "
                                 "window (never re-centred on an unconfirmed candidate)")
        return SearchRequest(frame, anchor[0], anchor[1], self.search_radius_px, False,
                             "square search around the last OBSERVED physical anchor "
                             "(no velocity extrapolation and no assumed travel "
                             "direction, so a stationary head stays searchable)")

    def wants_appearance_update(self, observation: CandidateMatch) -> bool:
        """Never: this arm carries no appearance template to refresh."""
        return False

    def _tracking(self, frame: int, observation: CandidateMatch, gates, measurement):
        decision = super()._tracking(frame, observation, gates, measurement)
        decision.method = METHOD_DETECTOR
        decision.note = (
            f"structural putter-head detection: compact head evidence segmented afresh "
            f"in THIS frame (no template following), composite evidence score "
            f"{observation.score:.3f}, seed identity {observation.seed_score:.3f}, second "
            f"structure {observation.second_score:.3f}, geometry {observation.transform}"
            + ("; search window clamped at a frame edge" if observation.clipped else "")
            + (f"; {observation.reason}" if observation.reason else ""))
        self.velocity = (0.0, 0.0)
        return decision

    def _confirming(self, frame: int, observation: CandidateMatch, gates, measurement):
        decision = super()._confirming(frame, observation, gates, measurement)
        if decision.observed:
            decision.method = METHOD_REACQUIRED
            decision.note = (
                f"structural re-detection at source frame {frame}: the putter head was "
                f"segmented afresh and confirmed by a SECOND consecutive current frame "
                f"(no backfill, the earlier candidate frame stays lost); composite "
                f"evidence score {observation.score:.3f}, seed identity "
                f"{observation.seed_score:.3f}, geometry {observation.transform}; "
                f"declared loss at source frame {self.loss_frame}")
        return decision


@dataclass
class PutterRun:
    """One forward leg of this candidate, with its own raw evidence."""
    target: str
    seed: dict
    interval_frames: List[int]
    available: bool = False
    reason: str = ""
    decisions: List[FrameDecision] = field(default_factory=list)
    raw_frames: List[dict] = field(default_factory=list)
    seed_evidence: Dict[str, Any] = field(default_factory=dict)
    policy_summary: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=lambda: list(LIMITATIONS))

    def points(self) -> List[dict]:
        return [decision.to_point() for decision in self.decisions]

    def summary(self) -> dict:
        counts: Dict[str, int] = {}
        for decision in self.decisions:
            counts[decision.status] = counts.get(decision.status, 0) + 1
        reasons: Dict[str, int] = {}
        for raw in self.raw_frames:
            if raw.get("decision", {}).get("status") in (STATUS_OBSERVED, "operator_placed"):
                continue
            reason = raw.get("evidence", {}).get("reason")
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
        reacquired = [decision.source_frame for decision in self.decisions
                      if decision.reacquired]
        return {"target": self.target, "available": bool(self.available),
                "reason": self.reason, "seed": dict(self.seed),
                "frames": len(self.decisions),
                "status_counts": dict(sorted(counts.items())),
                "observed_frames": sum(1 for d in self.decisions if d.observed),
                "automatic_observed_frames": sum(1 for d in self.decisions
                                                 if d.origin == ORIGIN_AUTOMATIC),
                "lost_frames": counts.get(STATUS_LOST, 0),
                "missing_frames": counts.get(STATUS_MISSING, 0),
                "reacquired_frames": len(reacquired),
                "reacquired_frame_range": ([min(reacquired), max(reacquired)]
                                           if reacquired else None),
                "methods": sorted({d.method for d in self.decisions if d.method}),
                "appearance_updates": 0,
                "empty_frame_reasons": dict(sorted(reasons.items(),
                                                   key=lambda item: -item[1])),
                "seed_evidence": dict(self.seed_evidence),
                "policy": dict(self.policy_summary),
                "profile": PROFILE_NAME,
                "engine": "untrained_structural_detector"}


def track_putter_sequence(frames, *, seed_frame: int, seed_x: float, seed_y: float,
                          interval_frames: Sequence[int], cut_frames: Sequence[int] = (),
                          target: str = "clubhead") -> PutterRun:
    """Run one forward leg over an ascending stream of ``(frame, image)`` pairs.

    Identical contract to ``foreground_tracker.track_sequence``: frames before the seed
    are decoded and dropped, a frame missing from the stream is reported unmeasured, and
    nothing outside ``interval_frames`` is ever reported.
    """
    interval = [int(frame) for frame in interval_frames]
    cuts = {int(frame) for frame in cut_frames}
    seed_frame = int(seed_frame)
    run = PutterRun(target=target, seed={"frame": seed_frame, "x": float(seed_x),
                                         "y": float(seed_y)}, interval_frames=interval)
    policy = PutterPolicy(seed_frame=seed_frame, seed_x=seed_x, seed_y=seed_y,
                          target=target)
    stream = iter(frames)
    buffered: Optional[Tuple[int, Any]] = None
    last_seen: Optional[int] = None

    def image_for(frame: int):
        nonlocal buffered, last_seen
        while True:
            if buffered is None:
                try:
                    buffered = next(stream)
                except StopIteration:
                    return None
                if last_seen is not None and int(buffered[0]) <= last_seen:
                    raise ValueError("frames must arrive in ascending order")
                last_seen = int(buffered[0])
            if int(buffered[0]) == frame:
                image, buffered = buffered[1], None
                return image
            if int(buffered[0]) > frame:
                return None
            buffered = None

    seed_image = image_for(seed_frame)
    if seed_image is None:
        policy.refuse_seed("seed_frame_not_decodable")
    else:
        descriptor, reason, seed_evidence = describe_seed(seed_image, seed_x, seed_y)
        run.seed_evidence = seed_evidence
        if descriptor is None:
            policy.refuse_seed(reason or "seed_uninformative")
        else:
            run.available = True
    run.reason = policy.refusal_reason
    seed_descriptor = run.seed_evidence.get("chosen")
    raw_by_frame: Dict[int, dict] = {}
    for frame in interval:
        request_record = None
        observation = None
        evidence: Dict[str, Any] = {}
        if frame < seed_frame:
            policy.step(frame)
        elif frame in cuts and frame != seed_frame:
            policy.step(frame, cut=True)
        else:
            request = policy.request_for(frame) if frame > seed_frame else None
            if request is not None and seed_descriptor is not None:
                request_record = request.to_dict()
                image = image_for(frame)
                if image is None:
                    observation = CandidateMatch(frame, False,
                                                 reason="frame_not_decodable")
                    evidence = {"frame": frame, "reason": "frame_not_decodable"}
                else:
                    observation, evidence = detect(image, request, seed_descriptor)
            elif request is not None:
                observation = CandidateMatch(frame, False, reason="seed_uninformative")
            policy.step(frame, observation)
        raw_by_frame[frame] = {"frame": frame, "request": request_record,
                               "decision": policy.decisions[-1].to_raw(),
                               "appearance_updated": False, "evidence": evidence}
    run.decisions = list(policy.decisions)
    run.raw_frames = [raw_by_frame.get(int(decision.source_frame),
                                       {"frame": int(decision.source_frame)})
                      for decision in run.decisions]
    run.policy_summary = policy.summary()
    return run


def track_putter(cap, *, seed_frame: int, seed_x: float, seed_y: float, start_frame: int,
                 end_frame: int, cut_frames: Sequence[int] = (),
                 target: str = "clubhead") -> PutterRun:
    """Decode the interval sequentially and re-detect one forward leg from the seed."""
    cv2 = _cv()

    def decoded():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(seed_frame))
        frame = int(seed_frame)
        while frame <= int(end_frame):
            ok, image = cap.read()
            if not ok:
                return
            yield frame, image
            frame += 1

    return track_putter_sequence(decoded(), seed_frame=seed_frame, seed_x=seed_x,
                                 seed_y=seed_y,
                                 interval_frames=list(range(int(start_frame),
                                                            int(end_frame) + 1)),
                                 cut_frames=cut_frames, target=target)
