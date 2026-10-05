"""Swing-family native SAM2 mask-core head measurement, isolated from detection.

WHY THIS EXISTS
---------------
The earlier compact-core experiments erode a mask with ONE fixed thickness: a
global seed threshold (``adaptive_head``) or the seed neighborhood's peak
(``appearance_head``). On a full swing that single number is wrong in both
directions at once. A thin end-on clubhead has a SMALL local peak, so a fixed
thick erosion deletes its core entirely; a turned driver face has a LARGE local
peak and the same fixed threshold leaves a core that is far too large to be a
compact head. This module measures every connected native mask component
independently -- each parent component is eroded at
``max(1.8, 0.75 * that component's own distance-transform peak)`` -- so a thin
end-on head and a turned face are each rendered to a core whose size reflects
their own thickness rather than a neighbour's.

WHAT IT IS
----------
* one contiguous forward leg from the operator seed frame; no frame before the
  seed is measured, no future frame is read, no coordinate is ever back-filled;
* the native SAM2 mask and the native BGR image are handed in by the caller and
  are the ONLY evidence: no label, no reference, no clip identity, no correction;
* the original seed is OPERATOR-ONLY: the seed frame's exported point is the
  operator's own placement, carries ``score=None`` and is never re-derived;
* the frozen original-seed core descriptor and the seed's parent-block area are
  deep-copied once and never refreshed;
* current candidates must be a single eligible measured core inside the
  ``ForegroundPolicy`` search window, with a parent-area ratio in ``0.2..5`` and
  a frozen appearance gate that lets a silver driver face darken into a dark
  shell and allows neutral reflective brightening, but refuses a simultaneous
  brightening and colour change and any lighter-than-bound turf/ball core;
* ``ForegroundPolicy`` owns the causal confirmation, the bounded re-acquisition
  window (pre-declared at 240 elapsed frames, bounded further by the caller's own
  stroke window/crop), the loss rule and the no-back-fill guarantee;
* structural cuts are terminal, and a cut ON the seed frame is itself refused.

WHAT IT IS NOT
--------------
* not a detector and not physical clubhead identity: the appearance bounds are
  constraints on NATIVE CORE PIXELS, not a semantic or physical identity claim;
* a development revision, not validation: the algorithm itself accepts no
  reference, label, clip identity or later frame, but these constants were
  CHANGED in a DISCLOSED development cycle after inspecting this module's own v1
  outputs, so they are NOT untuned and this is NOT holdout validation;
* not a source of probabilistic confidence: all admissions are binary policy
  gates and the exported score is null;
* not accuracy evidence: the unit tests are synthetic behaviour contracts.

``describe_core`` is the shared, already-frozen native-pixel descriptor, and its
home module (``head_identity``) imports ``cv2`` and ``numpy`` EAGERLY at import
time; because this module re-uses that descriptor, importing ``swing_head`` also
requires ``cv2`` and ``numpy``, and the local imports inside the measurement
helpers are tidiness, not isolation.
"""
from __future__ import annotations

import copy
import math

from .foreground_tracker import CandidateMatch, ForegroundPolicy
from .head_identity import MIN_CORE_SAMPLES, SUBJECT, describe_core
from .segmented_head import POINT_POLICY

# --------------------------------------------------------------------------
# every fixed numeric choice, in one place

CORE_MIN_DISTANCE_PX = 1.8
CORE_PEAK_FRACTION = 0.75
CORE_MIN_AREA_PX = 3
CORE_MAX_AREA_PX = 180
CORE_MAX_DIMENSION_PX = 24
SEED_SUPPORT_DISTANCE_PX = 3
MIN_ELONGATION = 1.25
MIN_PARENT_AREA_RATIO = 0.2
MAX_PARENT_AREA_RATIO = 5.0
ROUND_BRIGHT_FRACTION_REFUSAL = 0.5
LAB_CHROMA_DISTANCE_MAX = 35.0
#: Stricter chroma bound for ANY positive lightness gain: a core that brightens
#: must also keep its colour, so a simultaneous brightening and colour change is
#: refused while neutral reflective brightening (small chroma) still passes.
LAB_CHROMA_DISTANCE_MAX_POSITIVE_GAIN = 10.0
LAB_LIGHTNESS_GAIN_MAX = 60.0
DISPERSION_RATIO_MIN = 0.1
DISPERSION_RATIO_MAX = 5.0
DISPERSION_TEXTURE_FLOOR = 1.0
DISPERSION_DENOMINATOR_FLOOR = 1.0
UNIT_SQUARE_HALF = 0.5
SWING_REACQ_WINDOW_FRAMES = 240
DISTANCE_TRANSFORM_MASK_SIZE = 3
CONNECTIVITY = 8

METHOD_CORE = "sam2_swing_core"
METHOD_REACQUIRED = "sam2_swing_core_reacquired"
SUPPORTED_STROKE = "full_swing"

PROFILE = {
    "stroke_type": SUPPORTED_STROKE,
    "supported_strokes": [SUPPORTED_STROKE],
    "method": METHOD_CORE,
    "method_reacquired": METHOD_REACQUIRED,
    "core_min_distance_px": CORE_MIN_DISTANCE_PX,
    "core_peak_fraction": CORE_PEAK_FRACTION,
    "core_min_area_px": CORE_MIN_AREA_PX,
    "core_max_area_px": CORE_MAX_AREA_PX,
    "core_max_dimension_px": CORE_MAX_DIMENSION_PX,
    "seed_support_distance_px": SEED_SUPPORT_DISTANCE_PX,
    "min_elongation": MIN_ELONGATION,
    "min_parent_area_ratio": MIN_PARENT_AREA_RATIO,
    "max_parent_area_ratio": MAX_PARENT_AREA_RATIO,
    "round_bright_fraction_refusal": ROUND_BRIGHT_FRACTION_REFUSAL,
    "min_nonround_elongation": MIN_ELONGATION,
    "lab_chroma_distance_max": LAB_CHROMA_DISTANCE_MAX,
    "lab_chroma_distance_max_positive_gain": LAB_CHROMA_DISTANCE_MAX_POSITIVE_GAIN,
    "lab_lightness_gain_max": LAB_LIGHTNESS_GAIN_MAX,
    "dispersion_ratio_min": DISPERSION_RATIO_MIN,
    "dispersion_ratio_max": DISPERSION_RATIO_MAX,
    "dispersion_texture_floor": DISPERSION_TEXTURE_FLOOR,
    "dispersion_denominator_floor": DISPERSION_DENOMINATOR_FLOOR,
    "unit_square_half": UNIT_SQUARE_HALF,
    "min_core_samples": MIN_CORE_SAMPLES,
    "distance_transform_mask_size": DISTANCE_TRANSFORM_MASK_SIZE,
    "connectivity": CONNECTIVITY,
    "reacq_window_frames": SWING_REACQ_WINDOW_FRAMES,
    "measurement": ("each connected native mask parent component is eroded at "
                    "max(core_min_distance_px, core_peak_fraction * that component's "
                    "own distance-transform peak); connected cores are independently "
                    "bounded by area and max bbox dimension"),
    "oriented_extent": ("minAreaRect over the UNIT-PIXEL SQUARE SUPPORT (the four "
                        "corners x +/- 0.5, y +/- 0.5 of every core pixel), never "
                        "pixel centres only"),
    "morphology": "none: no closing and no dilation ever fills the mask",
    "appearance_gate": ("frozen original-seed appearance: Lab chroma (a,b) distance "
                        "<= 35, and <= 10 whenever the lightness gain is positive; "
                        "current L - seed L <= 60; gray dispersion ratio in 0.1..5 "
                        "when max(seed, current) dispersion >= 1.0. A simultaneous "
                        "brightening and colour change is refused"),
    "original_seed_descriptor_frozen": True,
    "template_refresh": False,
    "seed_is_operator_only": True,
    "seed_selected_from_later_frames": False,
    "algorithm_accepts_references": False,
    "development_revision": ("constants revised in a DISCLOSED development cycle after "
                             "inspecting this module's own v1 outputs; NOT untuned and "
                             "NOT holdout validation"),
    "exported_confidence": None,
    "physical_identity_established": False,
    "appearance_is_not_semantic_identity": True,
    "subject": SUBJECT,
    "reference_labelled": False,
    "research_only": True,
    "production_eligible": False,
}


# --------------------------------------------------------------------------
# measurement: per-component remeasurement, unit-pixel oriented extent

def oriented_extent(core_mask):
    """Oriented extent of a core from its UNIT-PIXEL SQUARE SUPPORT.

    Every core pixel contributes the four corners of its unit square
    ``(x +/- 0.5, y +/- 0.5)``. ``minAreaRect`` over that support measures the
    PHYSICAL extent of the region: a solid ``w x h`` block reports ``w x h``,
    where a centres-only measurement would report ``(w-1) x (h-1)``. Returns
    ``(long_side_px, short_side_px, elongation)``.
    """
    import cv2
    import numpy as np
    if not core_mask.any():
        raise ValueError("core region has no pixels to measure")
    ys, xs = np.nonzero(core_mask)
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    half = UNIT_SQUARE_HALF
    corner_x = np.concatenate([xs - half, xs + half, xs - half, xs + half])
    corner_y = np.concatenate([ys - half, ys - half, ys + half, ys + half])
    support = np.stack([corner_x, corner_y], axis=1).astype(np.float32)
    _, (side_a, side_b), _ = cv2.minAreaRect(support)
    long_side = float(max(side_a, side_b))
    short_side = float(min(side_a, side_b))
    return long_side, short_side, float(long_side / max(short_side, 0.5))


def measured_components(mask, image, seed_point=None):
    """Measure every connected native mask component independently.

    For each parent component the distance transform of the WHOLE mask is read
    locally, that component's own peak sets the erosion
    ``max(CORE_MIN_DISTANCE_PX, CORE_PEAK_FRACTION * peak)``, and the connected
    core components inside it are bounded by area and max bbox dimension. Each
    surviving core reports its parent area and its own threshold. No morphology
    is ever applied, so nothing fills the mask.
    """
    import cv2
    import numpy as np
    if mask.shape != image.shape[:2]:
        raise ValueError("Image and mask dimensions differ")
    binary = mask.astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, DISTANCE_TRANSFORM_MASK_SIZE)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, CONNECTIVITY)
    parts = []
    for index in range(1, count):
        px, py, pw, ph, parent_area = map(int, stats[index])
        if parent_area <= 0:
            continue
        parent = labels[py:py + ph, px:px + pw] == index
        local_distance = np.where(parent, distance[py:py + ph, px:px + pw], 0.0)
        peak = float(local_distance.max()) if parent.any() else 0.0
        threshold = max(CORE_MIN_DISTANCE_PX, CORE_PEAK_FRACTION * peak)
        core = (local_distance >= threshold) & parent
        if not core.any():
            continue
        c_count, c_labels, c_stats, _ = cv2.connectedComponentsWithStats(
            core.astype(np.uint8), CONNECTIVITY)
        for c_index in range(1, c_count):
            cx, cy, cw, ch, area = map(int, c_stats[c_index])
            if not CORE_MIN_AREA_PX <= area <= CORE_MAX_AREA_PX:
                continue
            if max(cw, ch) > CORE_MAX_DIMENSION_PX:
                continue
            local_core = c_labels[cy:cy + ch, cx:cx + cw] == c_index
            long_side, short_side, elongation = oriented_extent(local_core)
            ys, xs = np.nonzero(local_core)
            abs_x = (px + cx + xs).astype(np.float64)
            abs_y = (py + cy + ys).astype(np.float64)
            centroid = [float(abs_x.mean()), float(abs_y.mean())]
            try:
                appearance = describe_core(
                    image[py + cy:py + cy + ch, px + cx:px + cx + cw], local_core)
            except ValueError as exc:
                appearance = {"unavailable": str(exc)}
            core_gray = gray[py + cy:py + cy + ch, px + cx:px + cx + cw][local_core]
            xa, ya = max(0, px - 6), max(0, py - 6)
            xb, yb = min(mask.shape[1], px + pw + 6), min(mask.shape[0], py + ph + 6)
            ring_gray = gray[ya:yb, xa:xb][mask[ya:yb, xa:xb] == 0]
            contrast = (abs(float(np.median(core_gray)) - float(np.median(ring_gray)))
                        if ring_gray.size else None)
            region = {"xy": centroid, "area": area,
                      "bbox": [px + cx, py + cy, cw, ch],
                      "parent_area": parent_area, "parent_bbox": [px, py, pw, ph],
                      "elongation": float(elongation), "long_side_px": long_side,
                      "short_side_px": short_side, "core_threshold_px": float(threshold),
                      "appearance": appearance}
            region['local_contrast'] = {'core_gray_median': float(np.median(core_gray)),
                                        'background_samples': int(ring_gray.size),
                                        'absolute_median_gray_difference': contrast,
                                        'ring_margin_px': 6}
            if seed_point is not None:
                region["seed_support_distance_px"] = float(np.min(
                    np.hypot(abs_x - seed_point[0], abs_y - seed_point[1])))
            parts.append(region)
    return parts


def round_bright(region):
    """A round, bright, low-saturation core: the ball-confusion refusal."""
    appearance = region["appearance"]
    return (region["elongation"] < MIN_ELONGATION and
            appearance.get("bright_low_sat_fraction", 0) >= ROUND_BRIGHT_FRACTION_REFUSAL)


def _checked_appearance_descriptor(descriptor, name):
    """Fail closed on a malformed or non-finite Lab3 appearance descriptor.

    Only the fields this gate reads are required: a three-element finite
    ``lab_median``, a finite non-negative ``gray_dispersion`` and a non-negative
    integer ``sample_count``. Anything else is refused with ``ValueError`` before
    any comparison is attempted.
    """
    if not isinstance(descriptor, dict):
        raise ValueError(f"{name} must be a descriptor dict")
    lab = descriptor.get("lab_median")
    if (not isinstance(lab, (list, tuple)) or len(lab) != 3
            or not all(_finite_number(v) for v in lab)):
        raise ValueError(f"{name} lab_median must be three finite numbers")
    dispersion = descriptor.get("gray_dispersion")
    if isinstance(dispersion, bool) or not isinstance(dispersion, (int, float)):
        raise ValueError(f"{name} gray_dispersion must be a finite non-negative number")
    if not math.isfinite(float(dispersion)) or dispersion < 0:
        raise ValueError(f"{name} gray_dispersion must be a finite non-negative number")
    count = descriptor.get("sample_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"{name} sample_count must be a non-negative integer")
    return descriptor


def compare_swing_appearance(seed_descriptor, current_descriptor):
    """Frozen swing appearance constraints on native core pixels.

    These are APPEARANCE constraints, not semantic or physical identity. They
    allow a silver driver face to DARKEN into a dark shell (chroma within 35) and
    allow NEUTRAL reflective brightening, but require ANY positive lightness gain
    to keep chroma within 10 as well, so a simultaneous brightening and colour
    change is refused rather than latched onto a background feature.
    """
    import numpy as np
    seed = _checked_appearance_descriptor(seed_descriptor, "seed_descriptor")
    current = _checked_appearance_descriptor(current_descriptor, "current_descriptor")
    seed_lab = np.asarray(seed["lab_median"], dtype=np.float64)
    current_lab = np.asarray(current["lab_median"], dtype=np.float64)
    chroma_distance = float(np.linalg.norm(current_lab[1:] - seed_lab[1:]))
    lightness_gain = float(current_lab[0] - seed_lab[0])
    seed_dispersion = float(seed["gray_dispersion"])
    current_dispersion = float(current["gray_dispersion"])
    dispersion_ratio = current_dispersion / max(seed_dispersion,
                                                DISPERSION_DENOMINATOR_FLOOR)
    textured = max(seed_dispersion,
                   current_dispersion) >= DISPERSION_TEXTURE_FLOOR
    brightness_raised = lightness_gain > 0
    refusal = []
    if seed["sample_count"] < MIN_CORE_SAMPLES:
        refusal.append("seed_core_sample_count_below_minimum")
    if current["sample_count"] < MIN_CORE_SAMPLES:
        refusal.append("current_core_sample_count_below_minimum")
    if chroma_distance > LAB_CHROMA_DISTANCE_MAX:
        refusal.append("lab_chroma_distance_exceeds_swing_bound")
    if lightness_gain > LAB_LIGHTNESS_GAIN_MAX:
        refusal.append("lab_lightness_gain_exceeds_swing_bound")
    if brightness_raised and chroma_distance > LAB_CHROMA_DISTANCE_MAX_POSITIVE_GAIN:
        refusal.append("positive_lightness_gain_with_chroma_drift")
    if textured and not DISPERSION_RATIO_MIN <= dispersion_ratio <= DISPERSION_RATIO_MAX:
        refusal.append("gray_dispersion_ratio_out_of_swing_bounds")
    differences = {"lab_chroma_euclidean": chroma_distance,
                   "lab_lightness_gain": lightness_gain,
                   "lightness_gain_positive": brightness_raised,
                   "gray_dispersion_ratio": dispersion_ratio,
                   "gray_dispersion_compared": textured}
    if not all(np.isfinite(v) for v in [chroma_distance, lightness_gain,
                                        dispersion_ratio]):
        raise ValueError("swing identity comparison produced non-finite differences")
    return {"accepted": not refusal, "stroke_type": SUPPORTED_STROKE,
            "feature_differences": differences,
            "thresholds": {"lab_chroma_distance_max": LAB_CHROMA_DISTANCE_MAX,
                           "lab_chroma_distance_max_positive_gain":
                               LAB_CHROMA_DISTANCE_MAX_POSITIVE_GAIN,
                           "lab_lightness_gain_max": LAB_LIGHTNESS_GAIN_MAX,
                           "dispersion_ratio_min": DISPERSION_RATIO_MIN,
                           "dispersion_ratio_max": DISPERSION_RATIO_MAX,
                           "dispersion_texture_floor": DISPERSION_TEXTURE_FLOOR,
                           "dispersion_denominator_floor": DISPERSION_DENOMINATOR_FLOOR},
            "refusal_reasons": refusal,
            "decision_basis": ("fixed swing appearance thresholds on native core "
                               "pixels, not semantic identity"),
            "physical_identity_established": False, "subject": SUBJECT}


# --------------------------------------------------------------------------
# policy

def _finite_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


class SwingHeadPolicy:
    """Full-swing clubhead core policy over native SAM2 masks.

    Frames arrive once, in contiguous forward order from the seed frame. The
    raw evidence dict returned alongside each point carries every measured
    component, every admission decision and its reasons, and the appearance
    differences behind them. Nothing here is a confidence and no future frame
    or reference is ever read.
    """

    def __init__(self, seed, origin, cut_frames, stroke_type="full_swing",
                 point_policy=None):
        if stroke_type != SUPPORTED_STROKE:
            raise ValueError(f"Only {SUPPORTED_STROKE!r} is supported, not {stroke_type!r}")
        if not isinstance(seed, dict):
            raise ValueError("seed must be a dict with frame/x/y")
        if not _finite_number(seed.get("x")) or not _finite_number(seed.get("y")):
            raise ValueError("seed x/y must be finite numbers")
        frame = seed.get("frame")
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise ValueError("seed frame must be an integer")
        if frame < 0:
            raise ValueError("seed frame must be non-negative")
        if not isinstance(origin, (list, tuple)) or len(origin) != 2 \
                or not all(_finite_number(v) for v in origin):
            raise ValueError("origin must be a finite (x, y) pair")

        cuts = set()
        for cut_frame in cut_frames:
            if isinstance(cut_frame, bool) or not isinstance(cut_frame, int):
                raise ValueError("cut frames must be integers")
            if cut_frame < 0:
                raise ValueError("cut frames must be non-negative")
            cuts.add(cut_frame)

        settings = dict(POINT_POLICY)
        settings["reacq_window_frames"] = SWING_REACQ_WINDOW_FRAMES
        if point_policy is not None:
            unknown = set(point_policy) - set(POINT_POLICY)
            if unknown:
                raise ValueError("Unknown point-policy settings: " + str(sorted(unknown)))
            settings.update(point_policy)
            # Only this option is overridden on a supplied policy: the bounded
            # re-acquisition window is a swing-family constant, further bounded
            # by whatever stroke window/crop the caller feeds.
            settings["reacq_window_frames"] = SWING_REACQ_WINDOW_FRAMES

        for key, value in settings.items():
            if key in ('loss_frames', 'reacq_confirm_frames', 'reacq_window_frames'):
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f'{key} must be a positive integer')
            elif not _finite_number(value) or (value <= 0 if key.endswith('_px') else not 0 <= value <= 1):
                raise ValueError(f'{key} must be a finite valid policy bound')

        self.seed = dict(seed)
        self.origin = [origin[0], origin[1]]
        self.cuts = cuts
        self.stroke_type = stroke_type
        self.policy_settings = settings
        self.policy = ForegroundPolicy(seed_frame=frame, seed_x=seed["x"],
                                       seed_y=seed["y"], **settings)
        self.last_frame = frame - 1
        self.seed_valid = None
        self.seed_region = None
        self.seed_descriptor = None
        self.seed_parent_area = None
        self.refusal_reason = ""

    # -- frame intake -----------------------------------------------------

    def step(self, frame, mask, image):
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise ValueError("frame must be an integer")
        if frame < 0:
            raise ValueError("frame must be non-negative")
        if frame != self.last_frame + 1:
            raise ValueError("Frames must arrive once in contiguous forward order")
        self._validate_inputs(mask, image)
        self.last_frame = frame
        if frame == self.seed["frame"]:
            return self._seed_step(mask, image)
        return self._track_step(frame, mask, image)

    @staticmethod
    def _validate_inputs(mask, image):
        import numpy as np
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
            raise ValueError("mask must be a boolean (native SAM2) numpy array")
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise ValueError("image must be a uint8 numpy array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (height,width,3)")
        if mask.ndim != 2 or mask.shape != image.shape[:2]:
            raise ValueError("Image and mask dimensions differ")
        if mask.size == 0:
            raise ValueError("mask must not be empty")
        if image.size == 0:
            raise ValueError("image must not be empty")

    # -- seed -------------------------------------------------------------

    def _seed_step(self, mask, image):
        if self.seed["frame"] in self.cuts:
            decision = self.policy.step(self.seed["frame"], None, cut=True)
            self.seed_valid = False
            self.policy.measure = False
            self.refusal_reason = "seed_frame_is_a_structural_cut"
            self.policy.refusal_reason = self.refusal_reason
            return self._point(decision), self._evidence(
                self.seed["frame"], mask, [], [], [], decision)
        local_seed = [self.seed["x"] - self.origin[0], self.seed["y"] - self.origin[1]]
        parts = measured_components(mask, image, local_seed)
        supported = [p for p in parts
                     if p.get("seed_support_distance_px", math.inf) <= SEED_SUPPORT_DISTANCE_PX]
        usable = [p for p in supported if "unavailable" not in p["appearance"]]
        eligible = [p for p in usable if not round_bright(p) and not self._extra_refusals(p, image)]
        self.seed_region = eligible[0] if len(eligible) == 1 else None
        self.seed_valid = self.seed_region is not None
        if self.seed_valid:
            self.seed_descriptor = copy.deepcopy(self.seed_region["appearance"])
            self.seed_parent_area = self.seed_region["parent_area"]
        else:
            self.policy.measure = False
            self.refusal_reason = ("no unique seed-supported core with usable "
                                   "non-round-bright appearance")
            self.policy.refusal_reason = self.refusal_reason
        decision = self.policy.seed_decision()
        return self._point(decision), self._evidence(
            self.seed["frame"], mask, parts, [], [p["xy"] for p in eligible], decision)

    # -- tracking ---------------------------------------------------------

    def _track_step(self, frame, mask, image):
        if frame in self.cuts:
            decision = self.policy.step(frame, None, cut=True)
            return self._point(decision), self._evidence(frame, mask, [], [], [], decision)
        parts = measured_components(mask, image)
        request = self.policy.request_for(frame)
        checks = []
        admitted = []
        if (request is not None and self.seed_valid and self.seed_parent_area is not None
                and self.seed_descriptor is not None):
            for part in parts:
                native_xy = [part["xy"][0] + self.origin[0],
                             part["xy"][1] + self.origin[1]]
                ratio = part["parent_area"] / self.seed_parent_area
                entry = {"region_xy": native_xy, "parent_area": part["parent_area"],
                         "parent_area_ratio": float(ratio),
                         "elongation": part["elongation"]}
                if not MIN_PARENT_AREA_RATIO <= ratio <= MAX_PARENT_AREA_RATIO:
                    checks.append({**entry, "accepted": False,
                                   "refusal_reasons": ["parent_area_ratio_out_of_bounds"]})
                    continue
                measured_distance = math.dist(native_xy, [request.anchor_x, request.anchor_y])
                if measured_distance > request.radius_px:
                    checks.append({**entry, "accepted": False,
                                   "refusal_reasons": ["outside_search_radius"],
                                   "measured_distance_px": float(measured_distance),
                                   "search_radius_px": float(request.radius_px),
                                   "search_anchor": [request.anchor_x, request.anchor_y]})
                    continue
                if "unavailable" in part["appearance"]:
                    checks.append({**entry, "accepted": False,
                                   "refusal_reasons": [part["appearance"]["unavailable"]]})
                    continue
                match = compare_swing_appearance(self.seed_descriptor, part["appearance"])
                reasons = list(match["refusal_reasons"])
                reasons.extend(self._extra_refusals(part, image))
                if round_bright(part):
                    reasons.append("round_bright_core_ambiguous_with_ball")
                checks.append({**entry, **match, "accepted": not reasons,
                               "refusal_reasons": reasons})
                if not reasons:
                    admitted.append({**part, "native_xy": native_xy})
        candidate = None
        if len(admitted) == 1:
            xy = admitted[0]["native_xy"]
            candidate = CandidateMatch(frame, True, x=xy[0], y=xy[1], score=1,
                                       seed_score=1, second_score=0,
                                       transform="swing native mask-core appearance gate")
        decision = self.policy.step(frame, candidate, cut=frame in self.cuts)
        return self._point(decision), self._evidence(
            frame, mask, parts, checks, [a["native_xy"] for a in admitted], decision)

    def _extra_refusals(self, part, image):
        return []

    # -- reporting --------------------------------------------------------

    def _point(self, decision):
        point = decision.to_point()
        if point["origin"] == "automatic":
            point["method"] = (METHOD_REACQUIRED if decision.method.endswith("reacquired")
                               else METHOD_CORE)
            point["score"] = None
        return point

    def _evidence(self, frame, mask, parts, checks, admitted_xy, decision):
        raw_decision = decision.to_raw()
        # The inherited policy accepts binary admissions via its score gates.
        # Preserve those numbers only under an explicit internal-gate name.
        measurement = raw_decision.get('measurement', {})
        binary = {key: measurement.get(key) for key in
                  ('score', 'seed_identity_score', 'second_best_score', 'margin') if key in measurement}
        for key in binary:
            measurement[key] = None
        raw_decision['score'] = None
        if raw_decision['origin'] == 'automatic':
            raw_decision['method'] = self._point(decision)['method']
            raw_decision['note'] = 'Current swing mask core accepted by binary policy gates; score is not model confidence.'
        raw_decision['policy_internal_binary_gate'] = binary
        raw_decision['status_contract_note'] = 'observed_template is the legacy automatic-observation status; method identifies swing mask measurement.'
        return {"frame": frame, "mask_area": int(mask.sum()), "parts": parts,
                "identity_checks": checks, "admitted": admitted_xy,
                "seed_valid": self.seed_valid, "seed_parent_area": self.seed_parent_area,
                "decision": raw_decision,
                "policy_settings": dict(self.policy_settings),
                "appearance_thresholds": {"parent_area_ratio_min": MIN_PARENT_AREA_RATIO,
                                          "parent_area_ratio_max": MAX_PARENT_AREA_RATIO,
                                          "lab_chroma_distance_max": LAB_CHROMA_DISTANCE_MAX,
                                          "lab_chroma_distance_max_positive_gain":
                                              LAB_CHROMA_DISTANCE_MAX_POSITIVE_GAIN,
                                          "lab_lightness_gain_max": LAB_LIGHTNESS_GAIN_MAX,
                                          "dispersion_ratio_min": DISPERSION_RATIO_MIN,
                                          "dispersion_ratio_max": DISPERSION_RATIO_MAX},
                "score_note": ("Binary gate only; no model confidence and no established "
                               "physical identity.")}
