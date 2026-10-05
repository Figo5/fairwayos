"""Independent CURRENT-FRAME shaft + compact-head pixel detector (no SAM masks).

WHAT THIS IS
------------
A single forward leg from the operator's seed frame. Each later frame is measured
from the CURRENT frame alone: Hough shaft segments, a native-contrast compact-core
search at each accepted segment's measured distal end, current body-pose wrists and
a current body ROI. Nothing from the operator's seed image, no invisible template,
no future frame and no target label/ID/correction ever enters detection.

The head point exported on a confirmation is the EXACT measured compact-core pixel
centroid (plus the crop origin); it is never a Hough line endpoint, a wrist/pose
average, or a refined/synthetic point.

CAUSALITY
---------
* body ROI and wrists are supplied by the caller frame by frame. The ROI is
  operator-assisted and may be interpolated between operator keyframes; the only
  causal assumption is that the supplied crop schedule is fixed. No LATER measured
  pose or target output ever enters the current detection.
* the original seed frame is OPERATOR-ONLY: it exports the operator's coordinates
  with ``score=None`` and is never re-derived.
* a current candidate is withheld until it is confirmed by the required number of
  consecutive current measurements, and a confirmation is only ever reported at the
  frame that completes the run -- earlier candidate frames are never back-filled.

WHAT THIS IS NOT
----------------
* not physical/semantic clubhead identity and not a calibrated probability: all
  admissions are binary gates and every exported score is ``None``;
* a research-only development revision: ``PROFILE`` records every fixed constant
  chosen from a brief informed by previously inspected footage, not an unseen test;
* not accuracy evidence: the unit tests are synthetic behaviour contracts.

The measurement helpers import ``cv2``/``numpy`` lazily; ``describe_core`` comes from
``head_identity`` (which imports them eagerly) and ``oriented_extent`` from
``swing_head``, so importing this module requires both packages.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

from .head_identity import describe_core
from .swing_head import oriented_extent

# --------------------------------------------------------------------------
# every fixed numeric choice, in one place. ``_720`` constants are native-pixel
# values at a 720-line source and are multiplied by s = source_height / 720.
BASE_HEIGHT = 720.0
CANNY_LOW = 50
CANNY_HIGH = 150
CANNY_APERTURE = 3
HOUGH_RHO = 1.0
HOUGH_THETA = math.pi / 180.0
HOUGH_THRESHOLD = 25
HOUGH_MIN_LINE_LENGTH_720 = 40.0
HOUGH_MAX_LINE_GAP_720 = 10.0
MAX_SEGMENTS = 4096
MAX_ELIGIBLE_SEGMENTS = 256
SEGMENT_MIN_LENGTH_720 = 40.0
SEGMENT_MAX_LENGTH_720 = 350.0
WRIST_PROXIMAL_MAX_720 = 40.0
WRIST_DISTAL_MIN_720 = 60.0
WRIST_MIN_SCORE = 0.3
DIRECTION_MARGIN_720 = 30.0
ROI_MARGIN_720 = 8.0
PATCH_RADIUS_720 = 24.0
BORDER_RING_720 = 4.0
MIN_BACKGROUND_SAMPLES = 16
CONTRAST_DELTA = 15
CORE_MIN_DISTANCE_720 = 1.8
CORE_PEAK_FRACTION = 0.75
CORE_MIN_AREA_720 = 8.0
CORE_MAX_AREA_720 = 180.0
CORE_MAX_DIMENSION_720 = 24.0
CORE_MAX_ELONGATION = 4.0
MIN_CORE_SAMPLES = 8
SHAFT_PEAK_MIN_720 = 2.0
SUPPORT_DISTANCE_720 = 20.0
PERPENDICULAR_MAX_720 = 12.0
ALONG_MARGIN_720 = 20.0
DEDUP_RADIUS_720 = 8.0
CONFIRM_FRAMES = 2
MAX_DELTA_720 = 240.0
ROUND_BRIGHT_ELONGATION = 1.25
ROUND_BRIGHT_FRACTION = 0.5
CONNECTIVITY = 8
DISTANCE_TRANSFORM_MASK_SIZE = 3

METHOD_SHAFT = "shaft_head"
METHOD_REACQUIRED = "shaft_head_reacquired"
SUPPORTED_STROKE = "full_swing"

PROFILE = {
    "stroke_type": SUPPORTED_STROKE,
    "supported_strokes": [SUPPORTED_STROKE],
    "method": METHOD_SHAFT,
    "method_reacquired": METHOD_REACQUIRED,
    "base_height": BASE_HEIGHT,
    "scale_rule": "s = source_size height / base_height; every _720 constant is s-scaled",
    "canny_low": CANNY_LOW,
    "canny_high": CANNY_HIGH,
    "canny_aperture": CANNY_APERTURE,
    "canny_l2gradient": True,
    "hough_rho": HOUGH_RHO,
    "hough_theta_radians": HOUGH_THETA,
    "hough_threshold": HOUGH_THRESHOLD,
    "hough_min_line_length_720": HOUGH_MIN_LINE_LENGTH_720,
    "hough_max_line_gap_720": HOUGH_MAX_LINE_GAP_720,
    "max_segments": MAX_SEGMENTS,
    "max_eligible_segments": MAX_ELIGIBLE_SEGMENTS,
    "wrist_rule": "At least one visible current native wrist with score in [.3,1]; no stale or inferred wrist",
    "segment_min_length_720": SEGMENT_MIN_LENGTH_720,
    "segment_max_length_720": SEGMENT_MAX_LENGTH_720,
    "wrist_proximal_max_720": WRIST_PROXIMAL_MAX_720,
    "shaft_ribbon": {"along_fraction_start": .2, "along_fraction_end": .8,
                     "along_spacing_720": 2., "minimum_profiles": 10,
                     "perpendicular_radius_720": 10., "outer_band_inner_720": 7.,
                     "outer_background_max_gray_difference": 20., "contrast_min": 15.,
                     "run_center_max_offset_720": 3., "run_max_width_720": 6.,
                     "same_sign_profile_fraction_min": .7,
                     "competition": "Veto the single selected core only; never prune competing clusters",
                     "contrast_support": "Connected absolute-contrast run permits adjacent bright/dark compression ringing; sign is strongest absolute pixel contrast",
                     "sampling": "nearest native pixels only; duplicate pixels within each profile deduplicated before measuring width"},
    "wrist_distal_min_720": WRIST_DISTAL_MIN_720,
    "wrist_min_score": WRIST_MIN_SCORE,
    "direction_margin_720": DIRECTION_MARGIN_720,
    "roi_margin_720": ROI_MARGIN_720,
    "patch_radius_720": PATCH_RADIUS_720,
    "border_ring_720": BORDER_RING_720,
    "min_background_samples": MIN_BACKGROUND_SAMPLES,
    "contrast_delta": CONTRAST_DELTA,
    "core_min_distance_720": CORE_MIN_DISTANCE_720,
    "core_peak_fraction": CORE_PEAK_FRACTION,
    "core_min_area_720": CORE_MIN_AREA_720,
    "core_max_area_720": CORE_MAX_AREA_720,
    "core_area_scaling": "area bounds are scaled by s**2",
    "core_max_dimension_720": CORE_MAX_DIMENSION_720,
    "core_max_elongation": CORE_MAX_ELONGATION,
    "min_core_samples": MIN_CORE_SAMPLES,
    "shaft_peak_min_720": SHAFT_PEAK_MIN_720,
    "support_distance_720": SUPPORT_DISTANCE_720,
    "perpendicular_max_720": PERPENDICULAR_MAX_720,
    "along_margin_720": ALONG_MARGIN_720,
    "dedup_radius_720": DEDUP_RADIUS_720,
    "confirm_frames": CONFIRM_FRAMES,
    "max_delta_720": MAX_DELTA_720,
    "round_bright_elongation": ROUND_BRIGHT_ELONGATION,
    "round_bright_fraction": ROUND_BRIGHT_FRACTION,
    "connectivity": CONNECTIVITY,
    "distance_transform_mask_size": DISTANCE_TRANSFORM_MASK_SIZE,
    "measurement": ("Canny -> HoughLinesP shaft segments gated by current wrists and "
                    "body ROI -> per measured distal end, two independent native "
                    "contrast masks, distance transform, per-parent erosion at "
                    "max(core_min_distance, fraction * own parent peak), connected "
                    "cores bounded by area/dimension/elongation/support"),
    "background_estimate": ("median of the grayscale patch border ring only; no "
                            "semantic classes and no future information"),
    "morphology": "none: no closing, dilation or opening ever fills pixels",
    "patch_clipping_rule": "Only complete native patches; border-crossing parents require thin proximal shaft support; no boundary-touching core",
    "shaft_boundary_half_width_720": 3.0,
    "body_coordinate_space": "native_source_pixels",
    "contrast_gate_dependency": "Contrast is implied by the native threshold mask, not an independent classifier",
    "line_extrapolation": False,
    "exported_point_source": "exact measured compact-core pixel centroid plus crop origin",
    "dedup_rule": ("cores within dedup_radius of a cluster representative are grouped; "
                   "the cluster's greatest-area measured core is selected; no "
                   "coordinate averaging; no transitive merging of distinct blobs"),
    "candidate_rule": "exactly one admitted cluster, otherwise no candidate",
    "temporal_rule": ("the first current candidate, and any candidate after a missing/"
                      "ambiguous frame, is withheld until confirm_frames consecutive "
                      "current measurements; later valid frames emit immediately; a "
                      "candidate farther than max_delta from the last confirmation "
                      "restarts the pending run; nothing is ever back-filled"),
    "score_kind": "binary gate only, never a calibrated probability",
    "raw_confirmations_are_not_probability": True,
    "seed_is_operator_only": True,
    "seed_image_descriptor_used": False,
    "template_refresh": False,
    "future_frame_buffering": False,
    "algorithm_accepts_references": False,
    "physical_identity_established": False,
    "research_only": True,
    "production_eligible": False,
    "exported_confidence": None,
}


# --------------------------------------------------------------------------
# small geometry helpers (pure)

def _finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _px(base, s):
    return float(base) * s


def _canonical(x1, y1, x2, y2):
    """Order a segment's endpoints deterministically as native float tuples."""
    a = (float(x1), float(y1))
    b = (float(x2), float(y2))
    return (a[0], a[1], b[0], b[1]) if a <= b else (b[0], b[1], a[0], a[1])


def _point_rect_distance(px, py, rect):
    rx, ry, rw, rh = rect
    dx = max(rx - px, 0.0, px - (rx + rw))
    dy = max(ry - py, 0.0, py - (ry + rh))
    return math.hypot(dx, dy)


def _rect_distance(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(bx - (ax + aw), ax - (bx + bw), 0.0)
    dy = max(by - (ay + ah), ay - (by + bh), 0.0)
    return math.hypot(dx, dy)


def _along_perp(point, proximal, distal):
    dx = distal[0] - proximal[0]
    dy = distal[1] - proximal[1]
    length = math.hypot(dx, dy)
    vx = point[0] - proximal[0]
    vy = point[1] - proximal[1]
    if length <= 0:
        return 0.0, math.hypot(vx, vy)
    along = (vx * dx + vy * dy) / length
    perp = abs(vx * dy - vy * dx) / length
    return along, perp


def _border_pixels(gray, ring):
    import numpy as np
    border = np.zeros(gray.shape, dtype=bool)
    border[:ring, :] = border[-ring:, :] = True
    border[:, :ring] = border[:, -ring:] = True
    return gray[border].tolist()


def _support_hash(pair_list):
    return hashlib.sha256(repr(sorted(pair_list)).encode("ascii")).hexdigest()


def _shaft_ribbon(gray, origin, proximal, distal, s):
    """Require a thin contrast stripe bounded by background on both sides.

    A Hough edge is only a proposal. Native nearest-neighbour samples on its
    perpendicular profiles distinguish a narrow shaft from a filled region edge.
    No sampled point is ever exported as a head position.
    """
    import numpy as np
    length = math.dist(proximal, distal)
    dx, dy = (distal[0]-proximal[0])/length, (distal[1]-proximal[1])/length
    offsets = list(range(-int(math.ceil(10*s)), int(math.ceil(10*s))+1))
    alongs = np.arange(.2*length, .8*length + .001, 2*s)
    profiles = []
    counts = {"dark": 0, "bright": 0}
    for along in alongs:
        center = [proximal[0]+dx*float(along), proximal[1]+dy*float(along)]
        samples = {}
        for offset in offsets:
            coordinate = (round(center[0]-dy*offset-origin[0]), round(center[1]+dx*offset-origin[1]))
            samples.setdefault(coordinate, offset)
        coordinates, sample_offsets = list(samples), list(samples.values())
        row = {"center_native": center, "accepted_signs": [], "gray": None,
               "offsets_native_px":sample_offsets,
               "pixels_native":[[x+origin[0],y+origin[1]] for x,y in coordinates]}
        profiles.append(row)
        if any(x<0 or y<0 or x>=gray.shape[1] or y>=gray.shape[0] for x,y in coordinates):
            row['refusal'] = 'clipped_profile_samples'
            continue
        values = np.array([int(gray[y,x]) for x,y in coordinates])
        row['gray'] = values.tolist()
        left = float(np.median(values[np.array(sample_offsets) <= -7*s]))
        right = float(np.median(values[np.array(sample_offsets) >= 7*s]))
        row.update({"left_background": left, "right_background": right})
        if abs(left-right)>20:
            row['refusal'] = 'different_background_on_two_sides'
            continue
        bg=(left+right)/2
        mask = np.abs(values-bg)>=15
        changes = np.diff(np.r_[False,mask,False].astype(int))
        starts,ends = np.flatnonzero(changes==1),np.flatnonzero(changes==-1)
        signs = set()
        for start,end in zip(starts,ends):
            if (start>0 and end<len(values) and sample_offsets[end-1]-sample_offsets[start]+1<=6*s and
                    any(abs(sample_offsets[i])<=3*s for i in range(start,end))):
                strongest = values[start:end][np.argmax(np.abs(values[start:end]-bg))]-bg
                signs.add('bright' if strongest>=0 else 'dark')
        for kind in sorted(signs):
            row['accepted_signs'].append(kind)
            counts[kind]+=1
    fraction = max(counts.values())/len(profiles) if profiles else 0.
    return {"accepted":len(profiles)>=10 and fraction>=.7,
            "same_sign_fraction":fraction,"sign_counts":counts,
            "offsets_native_px":offsets,"profiles":profiles}


# --------------------------------------------------------------------------
# current-frame detection

def _measure_patch(patch_bgr, patch_gray, base, background, s, proximal, distal,
                   line_length, wrists, roi):
    """Measure every compact core in one accepted line's distal patch.

    ``base`` is the native coordinate of the patch's (0, 0) pixel. Returns a list of
    core records; each carries ``accepted`` and every ``refusal_reasons`` entry.
    """
    import cv2
    import numpy as np
    area_min = _px(CORE_MIN_AREA_720, s * s)
    area_max = _px(CORE_MAX_AREA_720, s * s)
    max_dim = _px(CORE_MAX_DIMENSION_720, s)
    min_distance = _px(CORE_MIN_DISTANCE_720, s)
    shaft_peak = _px(SHAFT_PEAK_MIN_720, s)
    support_max = _px(SUPPORT_DISTANCE_720, s)
    perp_max = _px(PERPENDICULAR_MAX_720, s)
    along_margin = _px(ALONG_MARGIN_720, s)
    wrist_min = _px(WRIST_DISTAL_MIN_720, s)
    roi_margin = _px(ROI_MARGIN_720, s)
    records = []
    for kind, sign in (("dark", -1), ("bright", 1)):
        cutoff = background + sign * CONTRAST_DELTA
        mask = ((patch_gray <= cutoff) if sign < 0
                else (patch_gray >= cutoff)).astype(np.uint8)
        if not mask.any():
            continue
        distance = cv2.distanceTransform(mask, cv2.DIST_L2,
                                         DISTANCE_TRANSFORM_MASK_SIZE)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, CONNECTIVITY)
        for index in range(1, count):
            px, py, pw, ph, parent_area = map(int, stats[index])
            if parent_area <= 0:
                continue
            parent = labels[py:py + ph, px:px + pw] == index
            border_y, border_x = np.nonzero(mask * (labels == index))
            on_edge = ((border_x == 0) | (border_y == 0) |
                       (border_x == patch_gray.shape[1] - 1) |
                       (border_y == patch_gray.shape[0] - 1))
            truncated = any(
                perp > 3.0 * s or along > line_length - support_max
                for along, perp in (_along_perp((base[0] + x, base[1] + y), proximal, distal)
                                    for x, y in zip(border_x[on_edge], border_y[on_edge])))
            local_distance = np.where(parent, distance[py:py + ph, px:px + pw], 0.0)
            peak = float(local_distance.max()) if parent.any() else 0.0
            threshold = max(min_distance, CORE_PEAK_FRACTION * peak)
            core = (local_distance >= threshold) & parent
            if not core.any():
                continue
            c_count, c_labels, c_stats, _ = cv2.connectedComponentsWithStats(
                core.astype(np.uint8), CONNECTIVITY)
            for c_index in range(1, c_count):
                cx, cy, cw, ch, area = map(int, c_stats[c_index])
                cbox = c_labels[cy:cy + ch, cx:cx + cw] == c_index
                ys, xs = np.nonzero(cbox)
                abs_x = (px + cx + xs).astype(np.float64)
                abs_y = (py + cy + ys).astype(np.float64)
                centroid = [base[0] + float(abs_x.mean()), base[1] + float(abs_y.mean())]
                long_side, short_side, elongation = oriented_extent(cbox)
                core_gray = patch_gray[py + cy:py + cy + ch, px + cx:px + cx + cw][cbox]
                contrast = abs(float(np.median(core_gray)) - background)
                along, perp = _along_perp(centroid, proximal, distal)
                bbox = [base[0] + px + cx, base[1] + py + cy, cw, ch]
                wrist_distance = min(math.dist(centroid, w) for w in wrists)
                refusals = []
                if truncated:
                    refusals.append("parent_truncated_by_patch_outside_thin_proximal_shaft")
                if (abs_x.min() <= 0 or abs_y.min() <= 0 or
                        abs_x.max() >= patch_gray.shape[1] - 1 or abs_y.max() >= patch_gray.shape[0] - 1):
                    refusals.append("core_touches_patch_boundary")
                if peak < shaft_peak:
                    refusals.append("thin_shaft_only_parent")
                if not area_min <= area <= area_max:
                    refusals.append("core_area_out_of_bounds")
                if max(cw, ch) > max_dim:
                    refusals.append("core_max_bbox_dimension_exceeds_bound")
                if elongation > CORE_MAX_ELONGATION:
                    refusals.append("core_elongation_exceeds_bound")
                if area < MIN_CORE_SAMPLES:
                    refusals.append("core_sample_count_below_minimum")
                if contrast < CONTRAST_DELTA:
                    refusals.append("local_gray_contrast_below_bound")
                if math.dist(centroid, distal) > support_max:
                    refusals.append("core_support_too_far_from_distal_endpoint")
                if perp > perp_max:
                    refusals.append("perpendicular_distance_to_line_exceeds_bound")
                if not line_length - along_margin <= along <= line_length + along_margin:
                    refusals.append("along_line_projection_out_of_bounds")
                if _rect_distance(bbox, roi) < roi_margin:
                    refusals.append("core_bbox_inside_or_near_body_roi")
                if wrist_distance < wrist_min:
                    refusals.append("core_too_close_to_a_wrist")
                appearance = None
                try:
                    appearance = describe_core(
                        patch_bgr[py + cy:py + cy + ch, px + cx:px + cx + cw], cbox)
                except ValueError:
                    refusals.append("core_appearance_unavailable")
                bright_fraction = (appearance or {}).get("bright_low_sat_fraction", 0.0)
                if (elongation < ROUND_BRIGHT_ELONGATION
                        and bright_fraction >= ROUND_BRIGHT_FRACTION):
                    refusals.append("round_bright_core_refused")
                records.append({
                    "mask": kind, "centroid_native": centroid, "area": int(area),
                    "bbox_native": bbox, "elongation": float(elongation),
                    "long_side_px": float(long_side), "short_side_px": float(short_side),
                    "parent_peak_px": float(peak), "core_threshold_px": float(threshold),
                    "local_gray_contrast": float(contrast),
                    "support_distance_px": float(math.dist(centroid, distal)),
                    "perpendicular_distance_px": float(perp),
                    "along_line_projection_px": float(along),
                    "nearest_wrist_distance_px": float(wrist_distance),
                    "bright_low_sat_fraction": float(bright_fraction),
                    "support_pixels_native": [[base[0] + float(x), base[1] + float(y)]
                                              for x, y in zip(abs_x, abs_y)],
                    "support_sha256": _support_hash([(base[0] + float(x), base[1] + float(y))
                                                    for x, y in zip(abs_x, abs_y)]),
                    "accepted": not refusals, "refusal_reasons": refusals,
                })
    return records


def _cluster_cores(cores, radius):
    """Group cores within ``radius`` of a representative; never merge transitively."""
    order = sorted(cores, key=lambda c: (-c["area"], c["centroid_native"][0],
                                         c["centroid_native"][1], c["mask"]))
    clusters = []
    for core in order:
        for cluster in clusters:
            if math.dist(core["centroid_native"],
                         cluster["representative"]["centroid_native"]) <= radius:
                cluster["members"].append(core)
                break
        else:
            clusters.append({"representative": core, "members": [core]})
    for cluster in clusters:
        cluster["selected"] = max(cluster["members"],
                                  key=lambda c: (c["area"], -c["centroid_native"][0],
                                                 -c["centroid_native"][1]))
    return clusters


def _detect(image, origin, source_size, s, body):
    """One frame of current evidence. Returns a JSON-native evidence dict."""
    import cv2
    import numpy as np
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = int(round(_px(PATCH_RADIUS_720, s)))
    ring = max(1, int(round(_px(BORDER_RING_720, s))))
    min_length = _px(SEGMENT_MIN_LENGTH_720, s)
    max_length = _px(SEGMENT_MAX_LENGTH_720, s)
    proximal_max = _px(WRIST_PROXIMAL_MAX_720, s)
    distal_min = _px(WRIST_DISTAL_MIN_720, s)
    direction_margin = _px(DIRECTION_MARGIN_720, s)
    roi_margin = _px(ROI_MARGIN_720, s)

    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH, apertureSize=CANNY_APERTURE,
                      L2gradient=True)
    raw_lines = cv2.HoughLinesP(
        edges, HOUGH_RHO, HOUGH_THETA, threshold=HOUGH_THRESHOLD,
        minLineLength=int(round(_px(HOUGH_MIN_LINE_LENGTH_720, s))),
        maxLineGap=int(round(_px(HOUGH_MAX_LINE_GAP_720, s))))
    segments = set()
    if raw_lines is not None:
        for row in np.asarray(raw_lines).reshape(-1, 4):
            x1, y1, x2, y2 = (int(v) for v in row)
            segments.add(_canonical(x1 + origin[0], y1 + origin[1],
                                    x2 + origin[0], y2 + origin[1]))
    evidence = {"segment_count": len(segments), "cap_exceeded": False, "lines": [],
                "measurements": [], "clusters": [], "selected": None}
    if len(segments) > MAX_SEGMENTS:
        evidence["cap_exceeded"] = True
        evidence["segments"] = sorted(segments)
        evidence["refusal_reason"] = "segment_cap_exceeded"
        return evidence
    segments = sorted(segments)
    evidence["segments"] = [list(segment) for segment in segments]

    wrists = [[w["x"], w["y"]] for w in body["wrists"].values()]
    grip_mean = [sum(w[0] for w in wrists) / len(wrists),
                 sum(w[1] for w in wrists) / len(wrists)]
    roi = body["roi_xywh"]
    accepted_lines = []
    for segment in segments:
        x1, y1, x2, y2 = segment
        a, b = (x1, y1), (x2, y2)
        length = math.dist(a, b)
        record = {"segment": list(segment), "length_px": float(length),
                  "accepted": False, "refusal_reasons": []}
        refusals = record["refusal_reasons"]
        if not min_length <= length <= max_length:
            refusals.append("segment_length_out_of_bounds")
            evidence["lines"].append(record)
            continue
        distance_a = min(math.dist(a, w) for w in wrists)
        distance_b = min(math.dist(b, w) for w in wrists)
        if distance_a <= distance_b:
            proximal, distal, d_prox, d_dist = a, b, distance_a, distance_b
        else:
            proximal, distal, d_prox, d_dist = b, a, distance_b, distance_a
        record.update({"proximal": list(proximal), "distal": list(distal),
                       "proximal_wrist_distance_px": float(d_prox),
                       "distal_wrist_distance_px": float(d_dist)})
        if d_prox > proximal_max:
            refusals.append("proximal_endpoint_not_near_a_wrist")
        if d_dist < distal_min:
            refusals.append("distal_endpoint_too_close_to_a_wrist")
        if d_dist < d_prox + direction_margin:
            refusals.append("direction_does_not_lead_away_from_grip")
        if _point_rect_distance(distal[0], distal[1], roi) < roi_margin:
            refusals.append("distal_endpoint_inside_or_near_body_roi")
        record["accepted"] = not refusals
        evidence["lines"].append(record)
        if record["accepted"]:
            accepted_lines.append(record)

    accepted_lines.sort(key=lambda rec: (-math.dist(rec["distal"], grip_mean),
                                         tuple(rec["segment"])))
    if len(accepted_lines) > MAX_ELIGIBLE_SEGMENTS:
        evidence["cap_exceeded"] = True
        evidence["refusal_reason"] = "eligible_segment_cap_exceeded"
        return evidence
    measurements = []
    for record in accepted_lines:
        distal = record["distal"]
        proximal = record["proximal"]
        local_x = int(round(distal[0] - origin[0]))
        local_y = int(round(distal[1] - origin[1]))
        x0, x1 = max(0, local_x - radius), min(width, local_x + radius + 1)
        y0, y1 = max(0, local_y - radius), min(height, local_y + radius + 1)
        patch_bgr = image[y0:y1, x0:x1]
        patch_gray = gray[y0:y1, x0:x1]
        if patch_gray.shape != (2 * radius + 1, 2 * radius + 1):
            record["measurement_refusal"] = "patch_clipped_by_crop"
            continue
        border = _border_pixels(patch_gray, ring)
        if len(border) < MIN_BACKGROUND_SAMPLES:
            record["measurement_refusal"] = "insufficient_background_samples"
            continue
        background = float(np.median(np.asarray(border, dtype=np.float64)))
        base = (origin[0] + x0, origin[1] + y0)
        cores = _measure_patch(patch_bgr, patch_gray, base, background, s,
                               tuple(proximal), tuple(distal), record["length_px"],
                               wrists, roi)
        for core in cores:
            core["line"] = list(record["segment"])
            core["distal"] = list(distal)
            core["patch_background_gray"] = background
            core["patch_bbox_native"] = [base[0], base[1], x1 - x0, y1 - y0]
            measurements.append(core)
    evidence["measurements"] = measurements
    accepted = [m for m in measurements if m["accepted"]]
    clusters = _cluster_cores(accepted, _px(DEDUP_RADIUS_720, s))
    evidence["clusters"] = [
        {"representative_native": cluster["representative"]["centroid_native"],
         "member_centroids_native": [m["centroid_native"] for m in cluster["members"]],
         "member_areas": [m["area"] for m in cluster["members"]],
         "selected_native": cluster["selected"]["centroid_native"],
         "selected_area": cluster["selected"]["area"]}
        for cluster in clusters]
    if len(clusters) == 1:
        selected = clusters[0]["selected"]
        selected_line = next(record for record in accepted_lines if record["segment"] == selected["line"])
        ribbon = _shaft_ribbon(gray, origin, selected_line["proximal"], selected_line["distal"], s)
        evidence["selected_shaft_ribbon"] = ribbon
        if not ribbon["accepted"]:
            evidence["refusal_reason"] = "single_core_has_no_current_thin_shaft_ribbon"
            return evidence
        evidence["selected"] = {
            "centroid_native": selected["centroid_native"],
            "centroid_local": [selected["centroid_native"][0] - origin[0],
                               selected["centroid_native"][1] - origin[1]],
            "area": selected["area"], "mask": selected["mask"],
            "elongation": selected["elongation"],
            "support_sha256": selected["support_sha256"]}
    elif not clusters:
        evidence["refusal_reason"] = "no_compact_core"
    else:
        evidence["refusal_reason"] = "ambiguous_multiple_core_clusters"
    return evidence


# --------------------------------------------------------------------------
# policy

class ShaftHeadPolicy:
    """One forward leg of independent current-frame shaft+head measurement."""

    def __init__(self, seed, origin, cut_frames, source_size,
                 stroke_type=SUPPORTED_STROKE):
        if stroke_type != SUPPORTED_STROKE:
            raise ValueError(f"Only {SUPPORTED_STROKE!r} is supported, not {stroke_type!r}")
        if not isinstance(seed, dict):
            raise ValueError("seed must be a dict with frame/x/y")
        frame = seed.get("frame")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError("seed frame must be a non-negative integer")
        if not _finite(seed.get("x")) or not _finite(seed.get("y")):
            raise ValueError("seed x/y must be finite numbers")
        if (not isinstance(source_size, (list, tuple)) or len(source_size) != 2
                or not all(type(v) is int and v > 0 for v in source_size)):
            raise ValueError("source_size must be a positive (width, height) pair")
        source_w, source_h = float(source_size[0]), float(source_size[1])
        if (not isinstance(origin, (list, tuple)) or len(origin) != 2
                or not all(type(v) is int and v >= 0 for v in origin)):
            raise ValueError("origin must be a finite non-negative (x, y) pair")
        if not 0 <= float(seed["x"]) < source_w or not 0 <= float(seed["y"]) < source_h:
            raise ValueError("seed point must lie inside the native source")
        cuts = set()
        for cut_frame in cut_frames:
            if isinstance(cut_frame, bool) or not isinstance(cut_frame, int) \
                    or cut_frame < 0:
                raise ValueError("cut frames must be non-negative integers")
            cuts.add(cut_frame)

        self.seed = {"frame": int(frame), "x": float(seed["x"]), "y": float(seed["y"])}
        self.origin = [float(origin[0]), float(origin[1])]
        self.source_size = [source_w, source_h]
        self.cuts = cuts
        self.stroke_type = stroke_type
        self.scale_s = source_h / BASE_HEIGHT
        self.last_frame = int(frame) - 1
        self.terminal = False
        self.seed_emitted = False
        self.pending = None
        self.last_confirmation = None
        self.confirmed = False
        self.gapped = False

    # -- intake -----------------------------------------------------------

    def step(self, frame, image, body_frame=None):
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError("frame must be a non-negative integer")
        if frame != self.last_frame + 1:
            raise ValueError("Frames must arrive once in contiguous forward order")
        self._validate_image(image)
        body = None
        body_reason = ""
        detect = frame != self.seed["frame"] and frame not in self.cuts \
            and not self.terminal
        if detect:
            body, body_reason = self._read_body(frame, body_frame)
        if self.terminal:
            self.last_frame = frame
            return self._derived(frame, "after_structural_cut")
        if frame == self.seed["frame"]:
            self.last_frame = frame
            return self._seed_step(frame)
        if frame in self.cuts:
            self.last_frame = frame
            self.terminal = True
            self.pending = None
            self.gapped = True
            return self._derived(frame, "structural_cut_is_terminal")
        evidence = _detect(image, self.origin, tuple(self.source_size), self.scale_s,
                           body) if body is not None else {"selected": None,
                                                           "body_fail_closed": body_reason}
        self.last_frame = frame
        return self._track_step(frame, body, body_reason, evidence)

    def _validate_image(self, image):
        import numpy as np
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise ValueError("image must be a uint8 numpy array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (height,width,3)")
        if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("image must not be empty")
        if (self.origin[0] + image.shape[1] > self.source_size[0]
                or self.origin[1] + image.shape[0] > self.source_size[1]):
            raise ValueError("image crop exceeds the native source bounds")

    def _read_body(self, frame, body_frame):
        """Return ``(body, "")`` or ``(None, reason)``; raise on malformed input."""
        if body_frame is None:
            return None, "body_frame_absent"
        if not isinstance(body_frame, Mapping):
            raise ValueError("body_frame must be a mapping")
        source_frame = body_frame.get("source_frame")
        if isinstance(source_frame, bool) or not isinstance(source_frame, int):
            raise ValueError("body_frame.source_frame must be an integer")
        if source_frame != frame:
            raise ValueError("body_frame.source_frame must equal the current frame")
        if body_frame.get("coordinate_space") != "native_source_pixels":
            raise ValueError("body_frame needs explicit native_source_pixels coordinate space")
        if body_frame.get("normalized") is True:
            raise ValueError("body_frame coordinates must be native, not normalized")
        if body_frame.get("available") is not True:
            return None, "body_pose_unavailable"
        roi = body_frame.get("ROI_xywh", body_frame.get("roi_xywh"))
        if roi is None:
            return None, "body_roi_absent"
        if (not isinstance(roi, (list, tuple)) or len(roi) != 4
                or not all(_finite(v) for v in roi)):
            raise ValueError("body ROI_xywh must be four finite numbers")
        rx, ry, rw, rh = (float(v) for v in roi)
        if rw <= 0 or rh <= 0:
            raise ValueError("body ROI must have positive width and height")
        if rx < 0 or ry < 0 or rx + rw > self.source_size[0] \
                or ry + rh > self.source_size[1]:
            raise ValueError("body ROI lies outside the native source bounds")
        entries = body_frame.get("wrists")
        if not isinstance(entries, Mapping):
            keypoints = body_frame.get("keypoints")
            entries = ({k.get("name"): k for k in keypoints
                        if isinstance(k, Mapping)}
                       if isinstance(keypoints, (list, tuple)) else {})
        wrists = {}
        unavailable = []
        for name in ("wr_l", "wr_r"):
            wrist = entries.get(name) if isinstance(entries, Mapping) else None
            if not isinstance(wrist, Mapping):
                unavailable.append(f"wrist_{name}_absent")
                continue
            if wrist.get("visible") is not True:
                unavailable.append(f"wrist_{name}_not_visible")
                continue
            score = wrist.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) \
                    or not math.isfinite(float(score)) or not WRIST_MIN_SCORE <= float(score) <= 1:
                unavailable.append(f"wrist_{name}_score_outside_reliable_range")
                continue
            x, y = wrist.get("x"), wrist.get("y")
            if not _finite(x) or not _finite(y):
                raise ValueError(f"wrist {name} coordinates must be finite numbers")
            if not 0 <= float(x) < self.source_size[0] \
                    or not 0 <= float(y) < self.source_size[1]:
                raise ValueError(f"wrist {name} lies outside the native source bounds")
            wrists[name] = {"x": float(x), "y": float(y), "score": float(score)}
        if not wrists:
            return None, "no_reliable_current_wrist: " + ";".join(unavailable)
        return {"roi_xywh": [rx, ry, rw, rh], "wrists": wrists}, ""

    # -- seed / derived ---------------------------------------------------

    def _operator_point(self, frame):
        return _point_dict(frame, "manual_anchor", "operator", "", self.seed["x"],
                           self.seed["y"], None)

    def _derived(self, frame, reason):
        point = _point_dict(frame, "missing", "derived", "", None, None, None)
        return point, {"frame": int(frame), "stage": "derived", "reason": reason,
                       "decision": _raw_decision(point, reason),
                       "probability": False, "score": None}

    def _seed_step(self, frame):
        if frame in self.cuts:
            self.terminal = True
            return self._derived(frame, "seed_frame_is_a_structural_cut")
        self.seed_emitted = True
        point = self._operator_point(frame)
        raw = {"frame": int(frame), "stage": "seed", "scale_s": self.scale_s,
               "source_size": list(self.source_size), "origin": list(self.origin),
               "operator": {"x": self.seed["x"], "y": self.seed["y"]},
               "decision": _raw_decision(
                   point, "seed (operator placement); never re-derived"),
               "score": None, "probability": False}
        return point, raw

    # -- tracking ---------------------------------------------------------

    def _track_step(self, frame, body, body_reason, evidence):
        candidate = evidence.get("selected")
        centroid = candidate["centroid_native"] if candidate else None
        max_delta = _px(MAX_DELTA_720, self.scale_s)
        decision_note = ""
        if centroid is None:
            self.pending = None
            self.gapped = True
            decision = _point_dict(frame, "missing", "derived", "", None, None, None)
            decision_note = body_reason or evidence.get("refusal_reason") \
                or "no_current_evidence"
        else:
            delta = (math.dist(centroid, self.last_confirmation)
                     if self.last_confirmation is not None else None)
            established = (self.last_confirmation is not None and not self.gapped
                           and delta is not None and delta <= max_delta)
            if established:
                self._confirm(frame, centroid, acquired=False)
                decision = self._observed(frame, centroid, METHOD_SHAFT)
                decision_note = "continuing confirmed track"
            else:
                pending_ok = (self.pending is not None
                              and self.pending[0] == frame - 1
                              and math.dist(centroid, self.pending[1]) <= max_delta)
                if pending_ok:
                    method = METHOD_REACQUIRED if self.confirmed else METHOD_SHAFT
                    self._confirm(frame, centroid, acquired=method == METHOD_REACQUIRED)
                    decision = self._observed(frame, centroid, method)
                    decision_note = (f"confirmation {CONFIRM_FRAMES}/{CONFIRM_FRAMES} "
                                     f"consecutive current measurements")
                else:
                    if self.pending is not None:
                        self.gapped = True
                    if delta is not None and delta > max_delta:
                        self.gapped = True
                    self.pending = (frame, centroid)
                    decision = _point_dict(frame, "missing", "derived", "", None,
                                           None, None)
                    decision_note = ("withheld pending "
                                     f"{CONFIRM_FRAMES} consecutive current "
                                     "measurements; never back-filled")
        raw = dict(evidence)
        raw.update({"frame": int(frame), "stage": "detection", "scale_s": self.scale_s,
                    "source_size": list(self.source_size),
                    "origin": list(self.origin),
                    "body": body, "body_fail_closed": body_reason,
                    "temporal": {"pending": (None if self.pending is None
                                             else {"frame": self.pending[0],
                                                   "centroid_native": self.pending[1]}),
                                 "last_confirmation": self.last_confirmation,
                                 "gapped": self.gapped, "confirmed": self.confirmed},
                    "decision": _raw_decision(decision, decision_note),
                    "probability": False, "score": None,
                    "note": ("measured compact-core centroid, not a line endpoint; "
                             "binary gates, not a probability")})
        return decision, raw

    def _confirm(self, frame, centroid, acquired):
        self.last_confirmation = centroid
        self.confirmed = True
        self.gapped = False
        self.pending = None

    def _observed(self, frame, centroid, method):
        return _point_dict(frame, "observed_template", "automatic", method,
                           centroid[0], centroid[1], None)


def _point_dict(frame, status, origin, method, x, y, score):
    return {"source_frame": int(frame), "status": status, "origin": origin,
            "method": method,
            "x": None if x is None else round(float(x), 2),
            "y": None if y is None else round(float(y), 2),
            "score": None if score is None else round(float(score), 4)}


def _raw_decision(point, note):
    return {"frame": point["source_frame"], "status": point["status"],
            "origin": point["origin"], "method": point["method"],
            "x": point["x"], "y": point["y"], "score": None, "note": note}
