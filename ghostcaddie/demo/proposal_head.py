"""Current-frame LEARNED-BOX proposal LOCALIZED to measured compact head pixels.

WHAT THIS IS
------------
A single current-frame measurement. A learned object detector proposes XYXY boxes;
each box is used ONLY as a search restriction around its centre. Every exported
point is an EXACT measured compact-core pixel centroid produced by re-using the
already-frozen ``shaft_head`` Hough + native-contrast measurement, gated by the
current body-pose wrists and by a real thin shaft ribbon. No bbox centre, no Hough
endpoint, no pose average and no detector confidence is ever exported as a point.

The detector is treated as a proposer, never a measurement:

* the fixed proposal confidence floor ``0.001`` is the SAME floor the frozen
  inference used to collect its raw predictions; it is a candidate-admission
  floor, NOT an operating threshold and NOT a selection score;
* every eligible box is measured; candidates are never ranked and the highest
  confidence is NEVER chosen -- a candidate is admitted only by binary geometry
  and pixel gates, and the frame is refused whenever more than one independent
  measured cluster remains.

WHAT THIS IS NOT
----------------
* not a calibrated probability and not physical/semantic clubhead identity: every
  admission is a binary gate and the exported confidence is ``None``;
* not a temporal policy: there is no confirmation, smoothing, back-fill or future
  frame here -- ``measure_proposals`` measures one frame from current evidence
  only. The primary driver re-uses the shaft confirmation behaviour separately;
* research-only: ``PROFILE`` records every fixed constant and its dependencies,
  and the constants were chosen from an already-inspected development brief, not
  from an unseen test.

CAUSALITY / INPUTS
------------------
* ``image`` is a complete uint8 BGR native crop; ``origin`` is its native
  top-left; ``source_size`` is the native (width, height) of the full source.
* ``body`` is the canonical already-validated current-frame pose in native
  coordinates: ``{'roi_xywh': [x, y, w, h], 'wrists': {'wr_l': {...}, 'wr_r':
  {...}}}`` with each wrist carrying finite native ``x``/``y`` and a reliable
  ``score``. The body ROI is ASSISTED CONTEXT (an exclusion region), never an
  independent person identity. A body with no reliable wrist yields
  ``selected=None`` with a reason; a structurally malformed body raises.
* ``detections`` is the raw detector output: mappings with crop-local XYXY
  ``xyxy`` (``box`` accepted as an alias), a finite ``confidence`` in ``[0, 1]``
  and an integer ``class_id``. The artifact is a one-class clubhead model, so
  ``class_id`` must equal ``0`` (clubhead); any other value raises ``ValueError``
  rather than being silently ignored. Malformed detections raise ``ValueError``;
  the absence of any proposal is a withheld frame, not an exception.

The measurement helpers/constants are re-used from ``shaft_head`` WITHOUT
modification; ``head_identity``/``swing_head`` are imported transitively by it.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from . import shaft_head

METHOD_PROPOSAL = "shaft_head_proposal_localized"
CLASS_ID_CLUBHEAD = 0
PROPOSAL_CONFIDENCE_FLOOR = 0.001
MAX_PROPOSALS = 30
PROPOSAL_SUPPORT_INSIDE_FRACTION = 0.8

PROFILE = {
    "method": METHOD_PROPOSAL,
    "class_id_clubhead": CLASS_ID_CLUBHEAD,
    "base_height": shaft_head.BASE_HEIGHT,
    "scale_rule": "s = source_size height / base_height; every reused _720 constant is s-scaled",
    "proposal_confidence_floor": PROPOSAL_CONFIDENCE_FLOOR,
    "confidence_role": ("fixed detector candidate-admission floor only, the same "
                        "value the frozen inference used to collect raw predictions; "
                        "NOT an operating threshold and NOT a selection score"),
    "class_constraint": ("the artifact is a one-class clubhead model: class_id must "
                         "equal class_id_clubhead (0); any other class_id raises "
                         "ValueError rather than being silently ignored"),
    "max_segments": shaft_head.MAX_SEGMENTS,
    "max_eligible_segments": shaft_head.MAX_ELIGIBLE_SEGMENTS,
    "max_proposals": MAX_PROPOSALS,
    "raw_proposal_cap": ("MAX_PROPOSALS bounds the RAW detector proposal count "
                         "before any Hough measurement, not only the above-floor "
                         "subset; exceeded raw count fails closed"),
    "proposal_support_inside_fraction": PROPOSAL_SUPPORT_INSIDE_FRACTION,
    "proposal_centre_used_as": ("patch location only; a proposal bbox centre is "
                                "never exported as a measured point"),
    "centroid_inside_proposal_box_required": True,
    "containment_coordinate_convention": "Detector boxes use continuous pixel edges; support indices and their mean are tested at x+0.5,y+0.5. Exported coordinates remain the exact mean of native pixel indices, as used by OpenCV drawing and Hough.",
    "wrist_min_score": shaft_head.WRIST_MIN_SCORE,
    "segment_min_length_720": shaft_head.SEGMENT_MIN_LENGTH_720,
    "segment_max_length_720": shaft_head.SEGMENT_MAX_LENGTH_720,
    "wrist_proximal_max_720": shaft_head.WRIST_PROXIMAL_MAX_720,
    "wrist_distal_min_720": shaft_head.WRIST_DISTAL_MIN_720,
    "direction_margin_720": shaft_head.DIRECTION_MARGIN_720,
    "roi_margin_720": shaft_head.ROI_MARGIN_720,
    "patch_radius_720": shaft_head.PATCH_RADIUS_720,
    "border_ring_720": shaft_head.BORDER_RING_720,
    "min_background_samples": shaft_head.MIN_BACKGROUND_SAMPLES,
    "support_distance_720": shaft_head.SUPPORT_DISTANCE_720,
    "dedup_radius_720": shaft_head.DEDUP_RADIUS_720,
    "canny": {"low": shaft_head.CANNY_LOW, "high": shaft_head.CANNY_HIGH,
              "aperture": shaft_head.CANNY_APERTURE, "l2gradient": True},
    "hough": {"rho": shaft_head.HOUGH_RHO, "theta_radians": shaft_head.HOUGH_THETA,
              "threshold": shaft_head.HOUGH_THRESHOLD,
              "min_line_length_720": shaft_head.HOUGH_MIN_LINE_LENGTH_720,
              "max_line_gap_720": shaft_head.HOUGH_MAX_LINE_GAP_720},
    "proposal_role": "a detector bbox is ONLY a search restriction; never a measurement",
    "measurement": ("current Hough shaft candidates exactly as shaft_head; segments "
                    "gated by reused length/current-wrist direction/body-distance "
                    "rules; for each eligible box, complete native contrast patches "
                    "measured with shaft_head._measure_patch around the rounded box "
                    "centre against a line whose measured distal lies within "
                    "SUPPORT_DISTANCE_720*s of the box rectangle (no extrapolation)"),
    "core_gate": ("a measured core is kept only if _measure_patch accepts it AND its "
                  "measured centroid lies inside its proposing rectangle AND at least "
                  "proposal_support_inside_fraction of its actual measured support "
                  "pixels lie inside that rectangle (centroid alone is insufficient); "
                  "existing shaft_head core checks (complete patch, border-truncation, "
                  "thin-shaft, area/dimension/elongation, round bright ball, contrast, "
                  "ROI/wrist distance) all remain"),
    "ribbon_gate": ("shaft_head._shaft_ribbon is required per candidate's supporting "
                    "line BEFORE ambiguity resolution; every qualifying independent "
                    "candidate is retained, none is discarded by score; a core that "
                    "physically passes but has no thin ribbon is reported as a refusal"),
    "cluster_rule": ("qualifying candidates clustered by shaft_head._cluster_cores at "
                     "radius 8*s; exactly one cluster is selected, more than one is "
                     "withheld (no averaging, no scored winner)"),
    "exported_point_source": "exact measured compact-core pixel centroid plus crop origin",
    "bbox_centre_or_endpoint_exported": False,
    "confidence_based_selection": False,
    "line_extrapolation": False,
    "temporal_policy": False,
    "template_refresh": False,
    "future_frame_buffering": False,
    "algorithm_accepts_references": False,
    "body_roi_is_assisted_context_only": True,
    "score_kind": "binary gate only, never a calibrated probability",
    "exported_confidence": None,
    "physical_identity_established": False,
    "dependencies": ["ghostcaddie.demo.shaft_head (geometry + measurement helpers and constants)",
                     "ghostcaddie.demo.head_identity.describe_core (transitive)",
                     "ghostcaddie.demo.swing_head.oriented_extent (transitive)"],
    "research_only": True,
    "production_eligible": False,
}


# --------------------------------------------------------------------------
# validation (pure)

def _validate_geometry(image, origin, source_size):
    """Validate the uint8 BGR crop and its native provenance; raise on malformed."""
    import numpy as np
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("image must be a uint8 numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height,width,3)")
    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must not be empty")
    if (not isinstance(origin, (list, tuple)) or len(origin) != 2
            or not all(type(v) is int and v >= 0 for v in origin)):
        raise ValueError("origin must be a non-negative (x, y) integer pair")
    if (not isinstance(source_size, (list, tuple)) or len(source_size) != 2
            or not all(type(v) is int and v > 0 for v in source_size)):
        raise ValueError("source_size must be a positive (width, height) integer pair")
    width, height = int(image.shape[1]), int(image.shape[0])
    ox, oy = int(origin[0]), int(origin[1])
    if ox + width > int(source_size[0]) or oy + height > int(source_size[1]):
        raise ValueError("image crop exceeds the native source bounds")
    return ox, oy, width, height


def _validate_detections(detections, width, height):
    """Validate raw detector boxes; return normalised crop-local records."""
    if not isinstance(detections, (list, tuple)):
        raise ValueError("detections must be a sequence of mappings")
    records = []
    for index, detection in enumerate(detections):
        if not isinstance(detection, Mapping):
            raise ValueError(f"detection {index} must be a mapping")
        class_id = detection.get("class_id")
        if isinstance(class_id, bool) or not isinstance(class_id, int):
            raise ValueError(f"detection {index} class_id must be an integer")
        if int(class_id) != CLASS_ID_CLUBHEAD:
            raise ValueError(
                f"detection {index} class_id {class_id} violates the one-class "
                f"clubhead artifact contract (expected {CLASS_ID_CLUBHEAD})")
        confidence = detection.get("confidence")
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence))
                or not 0.0 <= float(confidence) <= 1.0):
            raise ValueError(f"detection {index} confidence must be finite in [0,1]")
        box = detection.get("xyxy", detection.get("box"))
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or not all(shaft_head._finite(v) for v in box)):
            raise ValueError(f"detection {index} box must be four finite numbers")
        x1, y1, x2, y2 = (float(v) for v in box)
        if not (x2 > x1 and y2 > y1):
            raise ValueError(f"detection {index} box edges must increase")
        if x1 < 0.0 or y1 < 0.0 or x2 > width or y2 > height:
            raise ValueError(f"detection {index} box must lie within the crop")
        records.append({"index": index, "class_id": int(class_id),
                        "confidence": float(confidence), "box": [x1, y1, x2, y2]})
    return records


def _validate_body(body, source_w, source_h):
    """Return ``(body_context, "")`` or ``(None, reason)``; raise on malformed."""
    if body is None:
        return None, "body_absent"
    if not isinstance(body, Mapping):
        raise ValueError("body must be a mapping")
    if body.get("normalized") is True:
        raise ValueError("body coordinates must be native, not normalized")
    roi = body.get("roi_xywh")
    if roi is None:
        return None, "body_roi_absent"
    if (not isinstance(roi, (list, tuple)) or len(roi) != 4
            or not all(shaft_head._finite(v) for v in roi)):
        raise ValueError("body roi_xywh must be four finite numbers")
    rx, ry, rw, rh = (float(v) for v in roi)
    if rw <= 0 or rh <= 0:
        raise ValueError("body roi_xywh must have positive width and height")
    if rx < 0 or ry < 0 or rx + rw > source_w or ry + rh > source_h:
        raise ValueError("body roi_xywh lies outside the native source bounds")
    entries = body.get("wrists")
    if not isinstance(entries, Mapping):
        raise ValueError("body wrists must be a mapping")
    wrists = {}
    unavailable = []
    for name in ("wr_l", "wr_r"):
        wrist = entries.get(name)
        if not isinstance(wrist, Mapping):
            unavailable.append(f"wrist_{name}_absent")
            continue
        score = wrist.get("score")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not shaft_head.WRIST_MIN_SCORE <= float(score) <= 1.0):
            unavailable.append(f"wrist_{name}_score_outside_reliable_range")
            continue
        x, y = wrist.get("x"), wrist.get("y")
        if not shaft_head._finite(x) or not shaft_head._finite(y):
            raise ValueError(f"wrist {name} must carry finite native x/y")
        if not 0 <= float(x) < source_w or not 0 <= float(y) < source_h:
            raise ValueError(f"wrist {name} lies outside the native source bounds")
        wrists[name] = {"x": float(x), "y": float(y), "score": float(score)}
    if not wrists:
        return None, "no_reliable_current_wrist: " + ";".join(unavailable)
    return {"roi_xywh": [rx, ry, rw, rh], "wrists": wrists}, ""


# --------------------------------------------------------------------------
# current-frame shaft candidates (same geometry as shaft_head._detect)

def _current_segments(gray, origin, s):
    """Hough segments exactly as the frozen shaft detector computes them."""
    import cv2
    import numpy as np
    edges = cv2.Canny(gray, shaft_head.CANNY_LOW, shaft_head.CANNY_HIGH,
                      apertureSize=shaft_head.CANNY_APERTURE, L2gradient=True)
    raw_lines = cv2.HoughLinesP(
        edges, shaft_head.HOUGH_RHO, shaft_head.HOUGH_THETA,
        threshold=shaft_head.HOUGH_THRESHOLD,
        minLineLength=int(round(shaft_head._px(shaft_head.HOUGH_MIN_LINE_LENGTH_720, s))),
        maxLineGap=int(round(shaft_head._px(shaft_head.HOUGH_MAX_LINE_GAP_720, s))))
    segments = set()
    if raw_lines is not None:
        for row in np.asarray(raw_lines).reshape(-1, 4):
            x1, y1, x2, y2 = (int(v) for v in row)
            segments.add(shaft_head._canonical(x1 + origin[0], y1 + origin[1],
                                               x2 + origin[0], y2 + origin[1]))
    return segments


def _gate_segments(segments, s, wrists, roi):
    """Apply the reused length / wrist-direction / body-distance gates."""
    min_length = shaft_head._px(shaft_head.SEGMENT_MIN_LENGTH_720, s)
    max_length = shaft_head._px(shaft_head.SEGMENT_MAX_LENGTH_720, s)
    proximal_max = shaft_head._px(shaft_head.WRIST_PROXIMAL_MAX_720, s)
    distal_min = shaft_head._px(shaft_head.WRIST_DISTAL_MIN_720, s)
    direction_margin = shaft_head._px(shaft_head.DIRECTION_MARGIN_720, s)
    roi_margin = shaft_head._px(shaft_head.ROI_MARGIN_720, s)
    records, accepted = [], []
    for segment in sorted(segments):
        x1, y1, x2, y2 = segment
        a, b = (x1, y1), (x2, y2)
        length = math.dist(a, b)
        record = {"segment": list(segment), "length_px": float(length),
                  "accepted": False, "refusal_reasons": []}
        refusals = record["refusal_reasons"]
        if not min_length <= length <= max_length:
            refusals.append("segment_length_out_of_bounds")
            records.append(record)
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
        if shaft_head._point_rect_distance(distal[0], distal[1], roi) < roi_margin:
            refusals.append("distal_endpoint_inside_or_near_body_roi")
        record["accepted"] = not refusals
        records.append(record)
        if record["accepted"]:
            accepted.append(record)
    return records, accepted


def _cluster_entry(cluster):
    return {"representative_native": cluster["representative"]["centroid_native"],
            "member_centroids_native": [m["centroid_native"] for m in cluster["members"]],
            "member_areas": [m["area"] for m in cluster["members"]],
            "member_proposal_indices": sorted({m["proposal_index"]
                                              for m in cluster["members"]}),
            "selected_native": cluster["selected"]["centroid_native"],
            "selected_area": cluster["selected"]["area"]}


# --------------------------------------------------------------------------
# single-frame measurement

def measure_proposals(image, origin, source_size, body, detections):
    """Localize learned-box proposals to measured compact head pixels on ONE frame.

    Pure measurement: no temporal policy, no reference/seed argument, no future
    data. Returns a JSON-serializable evidence dict whose ``selected`` is the
    exact measured core centroid (plus support hash/pixels/box/line/ribbon and
    proposing ids) when exactly one qualifying cluster remains, otherwise ``None``
    with a ``refusal_reason``. All refusals and competing clusters are preserved.
    """
    import cv2
    import numpy as np
    ox, oy, width, height = _validate_geometry(image, origin, source_size)
    source_w, source_h = float(source_size[0]), float(source_size[1])
    s = source_h / shaft_head.BASE_HEIGHT
    proposals = _validate_detections(detections, width, height)

    evidence = {"method": METHOD_PROPOSAL, "selected": None, "clusters": [],
                "cluster_count": 0, "proposals": [], "lines": [], "refusals": [],
                "segment_count": 0, "cap_exceeded": False, "body": None,
                "body_reason": "", "scale_s": s, "origin": [ox, oy],
                "source_size": [source_w, source_h], "refusal_reason": None}

    body_context, body_reason = _validate_body(body, source_w, source_h)
    evidence["body"] = body_context
    evidence["body_reason"] = body_reason
    if body_context is None:
        evidence["refusal_reason"] = body_reason
        return evidence

    class0 = [p for p in proposals if p["class_id"] == CLASS_ID_CLUBHEAD]
    if not class0:
        evidence["refusal_reason"] = "no_class0_proposal"
        return evidence
    # Fail closed on the RAW detector proposal count before any Hough work: the
    # cap bounds the one-class artifact's raw output, not just the above-floor
    # subset, so an unbounded proposal list cannot drive unbounded measurement.
    if len(proposals) > MAX_PROPOSALS:
        evidence["cap_exceeded"] = True
        evidence["refusal_reason"] = "proposal_cap_exceeded"
        return evidence
    eligible = [p for p in class0 if p["confidence"] >= PROPOSAL_CONFIDENCE_FLOOR]
    if not eligible:
        evidence["refusal_reason"] = "no_eligible_proposal_above_confidence_floor"
        return evidence

    wrists = [[w["x"], w["y"]] for w in body_context["wrists"].values()]
    roi = body_context["roi_xywh"]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    segments = _current_segments(gray, (ox, oy), s)
    evidence["segment_count"] = len(segments)
    if len(segments) > shaft_head.MAX_SEGMENTS:
        evidence["cap_exceeded"] = True
        evidence["refusal_reason"] = "segment_cap_exceeded"
        evidence["segments"] = [list(segment) for segment in sorted(segments)]
        return evidence
    line_records, accepted_lines = _gate_segments(segments, s, wrists, roi)
    evidence["lines"] = line_records
    if len(accepted_lines) > shaft_head.MAX_ELIGIBLE_SEGMENTS:
        evidence["cap_exceeded"] = True
        evidence["refusal_reason"] = "eligible_segment_cap_exceeded"
        return evidence

    radius = int(round(shaft_head._px(shaft_head.PATCH_RADIUS_720, s)))
    ring = max(1, int(round(shaft_head._px(shaft_head.BORDER_RING_720, s))))
    support_max = shaft_head._px(shaft_head.SUPPORT_DISTANCE_720, s)
    candidates = []
    for proposal in eligible:
        box = proposal["box"]
        nbox = [box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy]
        rect = [nbox[0], nbox[1], nbox[2] - nbox[0], nbox[3] - nbox[1]]
        center = [(nbox[0] + nbox[2]) / 2.0, (nbox[1] + nbox[3]) / 2.0]
        entry = {"index": proposal["index"], "class_id": proposal["class_id"],
                 "confidence": proposal["confidence"], "box_local": list(box),
                 "box_native_xyxy": list(nbox), "supporting_line_count": 0,
                 "measured_cores": [], "qualifying": 0, "refusal_reasons": []}
        evidence["proposals"].append(entry)
        supporting = [rec for rec in accepted_lines
                      if shaft_head._point_rect_distance(rec["distal"][0],
                                                         rec["distal"][1], rect) <= support_max]
        entry["supporting_line_count"] = len(supporting)
        if not supporting:
            entry["refusal_reasons"].append("no_supporting_segment_near_proposal_box")
            continue
        local_x = int(round(center[0] - ox))
        local_y = int(round(center[1] - oy))
        x0, x1 = max(0, local_x - radius), min(width, local_x + radius + 1)
        y0, y1 = max(0, local_y - radius), min(height, local_y + radius + 1)
        patch_bgr = image[y0:y1, x0:x1]
        patch_gray = gray[y0:y1, x0:x1]
        if patch_gray.shape != (2 * radius + 1, 2 * radius + 1):
            entry["refusal_reasons"].append("patch_clipped_by_crop")
            continue
        border = shaft_head._border_pixels(patch_gray, ring)
        if len(border) < shaft_head.MIN_BACKGROUND_SAMPLES:
            entry["refusal_reasons"].append("insufficient_background_samples")
            continue
        background = float(np.median(np.asarray(border, dtype=np.float64)))
        base = (ox + x0, oy + y0)
        for record in supporting:
            ribbon = shaft_head._shaft_ribbon(gray, (ox, oy),
                                              tuple(record["proximal"]),
                                              tuple(record["distal"]), s)
            cores = shaft_head._measure_patch(patch_bgr, patch_gray, base, background, s,
                                              tuple(record["proximal"]),
                                              tuple(record["distal"]),
                                              record["length_px"], wrists, roi)
            for core in cores:
                total = len(core["support_pixels_native"])
                inside = sum(1 for x, y in core["support_pixels_native"]
                             if rect[0] <= x+.5 <= rect[0] + rect[2]
                             and rect[1] <= y+.5 <= rect[1] + rect[3])
                fraction = inside / total if total else 0.0
                centroid = core["centroid_native"]
                centroid_inside = (rect[0] <= centroid[0]+.5 <= rect[0] + rect[2]
                                   and rect[1] <= centroid[1]+.5 <= rect[1] + rect[3])
                entry["measured_cores"].append(
                    {"centroid_native": centroid, "mask": core["mask"],
                     "area": core["area"], "accepted": core["accepted"],
                     "centroid_inside_proposal_box": bool(centroid_inside),
                     "refusal_reasons": list(core["refusal_reasons"]),
                     "support_inside_fraction": float(fraction)})
                support_contained = fraction >= PROPOSAL_SUPPORT_INSIDE_FRACTION
                contained = centroid_inside and support_contained
                physically_accepted = core["accepted"] and contained
                if physically_accepted and ribbon["accepted"]:
                    candidate = dict(core)
                    candidate["proposal_index"] = proposal["index"]
                    candidate["proposal_box_native_xyxy"] = list(nbox)
                    candidate["support_inside_fraction"] = float(fraction)
                    candidate["centroid_inside_proposal_box"] = bool(centroid_inside)
                    candidate["ribbon"] = ribbon
                    candidate["line"] = list(record["segment"])
                    candidate["distal"] = list(record["distal"])
                    candidates.append(candidate)
                    entry["qualifying"] += 1
                else:
                    reasons = list(core["refusal_reasons"])
                    if core["accepted"] and not centroid_inside:
                        reasons.append("core_centroid_outside_proposal_box")
                    if core["accepted"] and not support_contained:
                        reasons.append("core_support_not_inside_proposal_box")
                    if physically_accepted and not ribbon["accepted"]:
                        reasons.append("core_accepted_but_no_thin_shaft_ribbon")
                    evidence["refusals"].append({
                        "proposal_index": proposal["index"],
                        "core_centroid_native": core["centroid_native"],
                        "core_mask": core["mask"], "line": list(record["segment"]),
                        "support_inside_fraction": float(fraction),
                        "centroid_inside_proposal_box": bool(centroid_inside),
                        "core_accepted": bool(core["accepted"]),
                        "ribbon_accepted": bool(ribbon["accepted"]),
                        "refusal_reasons": reasons})

    clusters = (shaft_head._cluster_cores(
        candidates, shaft_head._px(shaft_head.DEDUP_RADIUS_720, s))
        if candidates else [])
    evidence["clusters"] = [_cluster_entry(cluster) for cluster in clusters]
    evidence["cluster_count"] = len(clusters)
    if len(clusters) == 1:
        selected = clusters[0]["selected"]
        evidence["selected"] = {
            "centroid_native": selected["centroid_native"],
            "support_sha256": selected["support_sha256"],
            "support_pixels_native": selected["support_pixels_native"],
            "bbox_native": selected["bbox_native"],
            "area": selected["area"], "mask": selected["mask"],
            "line": selected["line"], "distal": selected["distal"],
            "proposal_index": selected["proposal_index"],
            "proposal_box_native_xyxy": selected["proposal_box_native_xyxy"],
            "proposal_indices": sorted({m["proposal_index"]
                                        for m in clusters[0]["members"]}),
            "support_inside_fraction": selected["support_inside_fraction"],
            "ribbon": selected["ribbon"],
        }
    elif not clusters:
        has_line = any(p["supporting_line_count"] for p in evidence["proposals"])
        if not has_line:
            evidence["refusal_reason"] = "no_supporting_segment_for_any_proposal"
        elif evidence["refusals"]:
            evidence["refusal_reason"] = "no_qualifying_proposal_core"
        else:
            evidence["refusal_reason"] = "no_measured_proposal_core"
    else:
        evidence["refusal_reason"] = "ambiguous_multiple_proposal_core_clusters"
    return evidence
