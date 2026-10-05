"""Classic bright-compact-component ball evidence validator (research only).

WHY THIS EXISTS
---------------
This is an isolated research validator of a *classic* ball-tracking proposal: a
golf ball in a native broadcast frame is a small, bright, low-saturation, compact
blob against turf. The validator answers one narrow question -- does the current
frame still carry a single bright compact component consistent with the one the
seed showed? -- and nothing else.

WHAT IT IS
----------
* one frozen, seed-derived absolute gray threshold. The threshold is computed once
  from the seed's own pixels (border-ring median grayscale background plus a
  margin driven by the seed's central 7x7 95th-percentile brightness) and then
  FROZEN into the descriptor. Every later check reuses that absolute threshold;
  it never re-derives one from the current frame;
* a binary mask of ``gray >= threshold AND saturation <= 140`` inside a 31 px
  native ROI centred on the seed / proposal point. The saturation ceiling exists
  to admit an encoded light ball whose colour is not pure white;
* plain connected components with NO morphology that could fill or bridge
  evidence; a component is eligible only if it is small (area and bounding-box
  cap), centred within 4 px of the requested point, and compact (elongation cap);
* elongation is measured as the oriented minimum-area rectangle of the component's
  FULL UNIT-PIXEL SUPPORT: each occupied pixel centre is expanded into its four
  square corners (x +/- 0.5, y +/- 0.5) before the rectangle is fitted. This
  reports the physical footprint a tiny encoded target actually paints (a 5x2
  block is 5x2, elongation 2.5) rather than the span of pixel centres, which
  under-counts one pixel per axis and inflates elongation for tiny cores;
* the seed must produce EXACTLY ONE eligible component. An ambiguous or
  non-compact seed disables every observation;
* the current frame must additionally keep the component's area and maximum
  dimension within frozen ratios of the seed's, and its elongation within a
  frozen bound derived from the seed's, and must again show exactly ONE eligible
  component.

WHAT IT IS NOT
--------------
* NOT a detector and NOT a new canonical ball detector. Nothing here is promoted;
  the canonical workflow does not import this module;
* NOT a coordinate refiner: the validator never returns an updated point. It
  reports the current component's geometry for audit only;
* NOT proof of physical ball identity and NOT a learned confidence. The existence
  of a bright compact blob is evidence, not identity.

CAUSALITY / ISOLATION
---------------------
* checks read CURRENT pixels only. No label, no frame number, no identity, no
  future frame, no run id and no adaptive descriptor update enter the algorithm;
* the module is stateless: the same inputs always return the same result, so
  appending frames can never rewrite what an earlier call reported (prefix
  invariance);
* the descriptor never mutates the image and never mutates its arguments.

All returned containers are plain deterministic JSON-native Python values (no
NaN, no numpy scalars).
"""
from __future__ import annotations

import math

__all__ = ["PROFILE", "build_seed_descriptor", "check_ball_evidence"]

BALL_EVIDENCE_SCHEMA = "fairwayos-ball-evidence.v1"

# Every parameter the algorithm uses lives here. Nothing is read from an
# environment variable, a caller argument or a later frame.
PROFILE = {
    # native ROI
    "roi_size_px": 31,
    "roi_half_px": 15,
    "connectivity": 8,
    # seed threshold construction
    "central_window_px": 7,
    "central_percentile": 95.0,
    "min_background_margin_gray": 15.0,
    "peak_margin_fraction": 0.65,
    # evidence mask
    "max_saturation": 140,
    # seed eligibility
    "core_min_area_px": 3,
    "core_max_area_px": 200,
    "core_max_dimension_px": 25,
    "core_max_elongation_seed": 3.5,
    "core_centroid_tolerance_px": 4.0,
    # geometry disclosure
    "elongation_geometry": "oriented_rectangle_of_unit_pixel_support",
    # current eligibility (relative to the frozen seed descriptor)
    "current_min_area_ratio": 0.3,
    "current_max_area_ratio": 3.0,
    "current_min_dimension_ratio": 0.45,
    "current_max_dimension_ratio": 2.2,
    "current_min_elongation": 2.5,
    "current_elongation_factor": 1.7,
    # disclosures
    "coordinates_refined": False,
    "exported_confidence": None,
    "physical_identity_proven": False,
    "research_only": True,
    "production_eligible": False,
}

_REQUIRED_DESCRIPTOR_KEYS = ("gray_threshold", "area", "max_dimension", "elongation")


def _is_number(value):
    if isinstance(value, bool):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _validate_image(image):
    """Reject malformed frames loudly; never mutate or copy the caller's array."""
    import numpy as np
    if image is None:
        raise ValueError("image must be a native uint8 BGR array, not None")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must be an HxWx3 BGR array")
    if array.dtype != np.uint8:
        raise ValueError("image must be uint8")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("image must be non-empty")
    return array


def _center(value):
    # Deterministic integer centre for a possibly fractional point.
    return int(math.floor(float(value) + 0.5))


def _roi_bounds(x, y, width, height):
    """Inclusive ROI bounds, or None if the 31 px ROI cannot be complete."""
    half = PROFILE["roi_half_px"]
    cx, cy = _center(x), _center(y)
    left, top = cx - half, cy - half
    right, bottom = cx + half, cy + half
    if left < 0 or top < 0 or right > width - 1 or bottom > height - 1:
        return None
    return left, top, right, bottom


def _gray_saturation(bgr_roi):
    import cv2
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)[:, :, 1]
    return gray, saturation


def _seed_threshold(gray_roi):
    import numpy as np
    half = PROFILE["roi_half_px"]
    window = PROFILE["central_window_px"] // 2
    ring = np.concatenate(
        (gray_roi[0, :], gray_roi[-1, :], gray_roi[:, 0], gray_roi[:, -1])
    ).astype(np.float64)
    background = float(np.median(ring))
    centre = gray_roi[half - window:half + window + 1,
                     half - window:half + window + 1].astype(np.float64)
    peak = float(np.percentile(centre, PROFILE["central_percentile"]))
    margin = max(PROFILE["min_background_margin_gray"],
                 PROFILE["peak_margin_fraction"] * (peak - background))
    return background, peak, background + margin


def _evidence_mask(gray_roi, saturation_roi, threshold):
    import numpy as np
    bright = gray_roi.astype(np.float64) >= float(threshold)
    unsaturated = saturation_roi <= PROFILE["max_saturation"]
    return (bright & unsaturated).astype(np.uint8)


def _components(mask_roi):
    """Connected components of the mask, no morphology. Geometry only.

    Elongation is the oriented minimum-area rectangle of the component's FULL
    UNIT-PIXEL SUPPORT: every occupied pixel centre is expanded into its four
    square corners before the rectangle is fitted. Measuring pixel centres
    instead would treat a 5x2 block as 4x1 (elongation 4) rather than the
    physical 5x2 footprint (elongation 2.5), artificially elongating tiny cores.
    Only the corners are added for measurement; no pixel is admitted to the mask
    and the mask itself is never modified.
    """
    import cv2
    import numpy as np
    count, labels, stats, centres = cv2.connectedComponentsWithStats(
        mask_roi, PROFILE["connectivity"])
    # (x, y) offsets to the four corners of a unit pixel centred on (0, 0).
    corners_offsets = np.array(
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=np.float32)
    result = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if area <= 0:
            continue
        pixels = np.argwhere(labels == index)[:, ::-1].astype(np.float32)
        support = (pixels[:, None, :] + corners_offsets[None, :, :]).reshape(-1, 2)
        _, (rect_w, rect_h), _ = cv2.minAreaRect(support)
        major, minor = max(rect_w, rect_h), min(rect_w, rect_h)
        # Occupied-pixel support guarantees a positive minor extent (>= 1); the
        # check below is a numerical guard only, never a geometric floor.
        if minor <= 0.0:
            continue
        elongation = major / minor
        if not math.isfinite(elongation):
            continue
        cx, cy = (float(v) for v in centres[index])
        result.append({
            "area": area,
            "bbox": [x, y, w, h],
            "max_dimension": max(w, h),
            "elongation": float(elongation),
            "centroid": [cx, cy],
        })
    return result


def _eligible(components, x, y, left, top, elongation_limit):
    """Base eligibility shared by seed and current, bound to a requested point."""
    tolerance = PROFILE["core_centroid_tolerance_px"]
    eligible = []
    for component in components:
        if not (PROFILE["core_min_area_px"] <= component["area"] <= PROFILE["core_max_area_px"]):
            continue
        if component["max_dimension"] > PROFILE["core_max_dimension_px"]:
            continue
        if component["elongation"] > elongation_limit:
            continue
        full_x = left + component["centroid"][0]
        full_y = top + component["centroid"][1]
        offset_x, offset_y = full_x - x, full_y - y
        distance = math.hypot(offset_x, offset_y)
        if distance > tolerance:
            continue
        entry = dict(component)
        entry["centroid_full"] = [float(full_x), float(full_y)]
        entry["centroid_offset"] = [float(offset_x), float(offset_y)]
        entry["centroid_distance"] = float(distance)
        eligible.append(entry)
    return eligible


def build_seed_descriptor(image, x, y):
    """Freeze the seed's bright compact component into a copyable descriptor.

    Returns ``{"valid": bool, "reason": str, "descriptor": dict | None}``. An
    invalid seed yields ``descriptor=None`` so that observations are disabled.
    """
    frame = _validate_image(image)
    if not _is_number(x) or not _is_number(y):
        raise ValueError("seed coordinates must be numeric")
    seed_x, seed_y = float(x), float(y)
    if not (math.isfinite(seed_x) and math.isfinite(seed_y)):
        return {"valid": False, "reason": "seed_not_finite", "descriptor": None}

    height, width = frame.shape[0], frame.shape[1]
    bounds = _roi_bounds(seed_x, seed_y, width, height)
    if bounds is None:
        return {"valid": False, "reason": "seed_roi_incomplete", "descriptor": None}
    left, top, right, bottom = bounds
    roi = frame[top:bottom + 1, left:right + 1]
    gray, saturation = _gray_saturation(roi)
    background, peak, threshold = _seed_threshold(gray)
    mask = _evidence_mask(gray, saturation, threshold)
    components = _components(mask)
    eligible = _eligible(components, seed_x, seed_y, left, top,
                         PROFILE["core_max_elongation_seed"])
    if not eligible:
        return {"valid": False, "reason": "no_compact_seed_component", "descriptor": None}
    if len(eligible) > 1:
        return {"valid": False, "reason": "ambiguous_seed_component", "descriptor": None}

    core = eligible[0]
    descriptor = {
        "schema": BALL_EVIDENCE_SCHEMA,
        "seed_point": [seed_x, seed_y],
        "roi_origin": [int(left), int(top)],
        "roi_size_px": PROFILE["roi_size_px"],
        "gray_threshold": float(threshold),
        "background_gray": float(background),
        "central_peak_gray": float(peak),
        "area": int(core["area"]),
        "bbox": [int(v) for v in core["bbox"]],
        "max_dimension": int(core["max_dimension"]),
        "elongation": float(core["elongation"]),
        "centroid": [float(v) for v in core["centroid_full"]],
        "centroid_offset": [float(v) for v in core["centroid_offset"]],
        "centroid_distance": float(core["centroid_distance"]),
        "coordinates_refined": PROFILE["coordinates_refined"],
        "physical_identity_proven": PROFILE["physical_identity_proven"],
    }
    return {"valid": True, "reason": "unique_compact_seed_component",
            "descriptor": descriptor}


def _resolve_descriptor(descriptor):
    """Accept a bare descriptor or a wrapper returned by build_seed_descriptor."""
    if descriptor is None:
        return None
    if not isinstance(descriptor, dict):
        return None
    if "descriptor" in descriptor and "valid" in descriptor:
        if not descriptor.get("valid"):
            return None
        descriptor = descriptor.get("descriptor")
    if not isinstance(descriptor, dict):
        return None
    for key in _REQUIRED_DESCRIPTOR_KEYS:
        if key not in descriptor:
            return None
        try:
            float(descriptor[key])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(float(descriptor[key])):
            return None
    return descriptor


def check_ball_evidence(image, x, y, descriptor):
    """Check current pixels for one frozen-seed-consistent compact component.

    Returns ``{"accepted": bool, "reason": str, "current": dict | None}``. The
    current geometry is audit-only: no updated point is ever returned.
    """
    frame = _validate_image(image)
    if not _is_number(x) or not _is_number(y):
        raise ValueError("proposal coordinates must be numeric")
    wrapper_invalid = isinstance(descriptor, dict) and descriptor.get("valid") is False
    resolved = _resolve_descriptor(descriptor)
    if resolved is None:
        reason = "seed_invalid" if (descriptor is None or wrapper_invalid) else "invalid_descriptor"
        return {"accepted": False, "reason": reason, "current": None}

    proposal_x, proposal_y = float(x), float(y)
    if not (math.isfinite(proposal_x) and math.isfinite(proposal_y)):
        return {"accepted": False, "reason": "proposal_not_finite", "current": None}

    height, width = frame.shape[0], frame.shape[1]
    bounds = _roi_bounds(proposal_x, proposal_y, width, height)
    if bounds is None:
        return {"accepted": False, "reason": "roi_incomplete", "current": None}
    left, top, right, bottom = bounds
    roi = frame[top:bottom + 1, left:right + 1]
    gray, saturation = _gray_saturation(roi)
    threshold = float(resolved["gray_threshold"])
    mask = _evidence_mask(gray, saturation, threshold)
    components = _components(mask)

    seed_area = float(resolved["area"])
    seed_dimension = float(resolved["max_dimension"])
    elongation_limit = max(PROFILE["current_min_elongation"],
                           PROFILE["current_elongation_factor"] * float(resolved["elongation"]))
    eligible = _eligible(components, proposal_x, proposal_y, left, top, elongation_limit)

    kept = []
    for component in eligible:
        if seed_area > 0:
            area_ratio = component["area"] / seed_area
            if not (PROFILE["current_min_area_ratio"] <= area_ratio <= PROFILE["current_max_area_ratio"]):
                continue
        if seed_dimension > 0:
            dimension_ratio = component["max_dimension"] / seed_dimension
            if not (PROFILE["current_min_dimension_ratio"] <= dimension_ratio <= PROFILE["current_max_dimension_ratio"]):
                continue
        kept.append(component)

    if not kept:
        return {"accepted": False, "reason": "no_eligible_component", "current": None}
    if len(kept) > 1:
        return {"accepted": False, "reason": "ambiguous_evidence", "current": None}

    core = kept[0]
    current = {
        "proposal": [proposal_x, proposal_y],
        "area": int(core["area"]),
        "bbox": [int(v) for v in core["bbox"]],
        "max_dimension": int(core["max_dimension"]),
        "elongation": float(core["elongation"]),
        "centroid": [float(v) for v in core["centroid_full"]],
        "centroid_offset": [float(v) for v in core["centroid_offset"]],
        "centroid_distance": float(core["centroid_distance"]),
        "gray_threshold": threshold,
        "area_ratio": float(core["area"] / seed_area) if seed_area > 0 else None,
        "dimension_ratio": float(core["max_dimension"] / seed_dimension) if seed_dimension > 0 else None,
        "coordinates_refined": PROFILE["coordinates_refined"],
        "physical_identity_proven": PROFILE["physical_identity_proven"],
    }
    return {"accepted": True, "reason": "unique_compact_component", "current": current}
