"""2D image-space analytics for the demo path. Pure Python.

Everything here is measured in **source pixels** and reported in **degrees** or
**pixels**. There is no physical scale, no action time and no 3D: the clip is one
uncalibrated camera view, so nothing in this module may produce m/s, mph, metres,
rpm or a launch angle.

Missingness is first-class. A metric whose landmarks are not visible on enough
frames is reported ``unavailable`` with the count and the reason, and no value is
invented from the frames that were visible but insufficient.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, TYPE_CHECKING

from .geometry import (MetricValue, angle_degrees, flexion_degrees, median,
                       path_length, percentile, tilt_degrees)

if TYPE_CHECKING:  # pragma: no cover - typing only, no import at runtime
    from .tracking import TargetTrack

#: A metric needs this many usable frames before it is reported at all.
MIN_METRIC_FRAMES = 5
#: ...and this share of the frames that could carry it.
MIN_METRIC_FRACTION = 0.30

LANDMARK_NAMES = ("nose", "eye_l", "eye_r", "ear_l", "ear_r", "sh_l", "sh_r", "el_l",
                  "el_r", "wr_l", "wr_r", "hip_l", "hip_r", "kn_l", "kn_r", "ank_l",
                  "ank_r")

TILT_CONVENTION = ("Tilt is the undirected angle of the LINE through the two landmarks "
                   "from the image horizontal, in degrees, folded into (-90, 90]: the "
                   "landmarks in either order give the same tilt, and no tilt is ever "
                   "reported near 180. The sign is the slope of the endpoint at the "
                   "larger image x (the image-right end): positive means that end sits "
                   "lower in the image. Image y grows downwards.")
KNEE_DEFINITION = ("Knee interior angle is the angle at the knee between the hip and the "
                   "ankle, in degrees: 180 is a straight leg, smaller is more flexed. "
                   "Flexion is reported as 180 - interior angle.")
SEGMENT_DEFINITION = ("Left/right landmarks are the model's own left/right, which for a "
                      "golfer facing the camera may not match the golfer's anatomy; the "
                      "lead/trail labels follow the configured handedness.")


def _visible_points(record: dict) -> Dict[str, dict]:
    """Visible landmarks of a record that is actually available.

    A record the identity check rejected contributes NOTHING, even if the model
    happened to return confident keypoints for it: an unattributed pose is not a
    measurement, so it may not reach an angle, a displacement or a path length.
    """
    if not record or not record.get("available"):
        return {}
    out: Dict[str, dict] = {}
    for point in record.get("keypoints") or []:
        if point.get("visible"):
            out[point["name"]] = point
    return out


def body_angles_for_record(record: Optional[dict], *, pixel_aspect: float = 1.0,
                           handedness: str = "right") -> dict:
    """Shoulder tilt, hip tilt and both knee angles at one frame, or None per key.

    Used by the metrics and by the keyframe cards, so a card and the metric table
    can never disagree about what was measured at a frame.
    """
    out = {"shoulder_tilt_deg": None, "hip_tilt_deg": None,
           "knee_interior_deg": {"left": None, "right": None},
           "knee_flexion_deg": {"left": None, "right": None},
           "lead_side": "left" if handedness == "right" else "right",
           "trail_side": "right" if handedness == "right" else "left",
           "source_frame": None}
    if not record or not record.get("available"):
        return out
    out["source_frame"] = record.get("source_frame")
    pts = _visible_points(record)
    if "sh_l" in pts and "sh_r" in pts:
        out["shoulder_tilt_deg"] = tilt_degrees(
            (pts["sh_l"]["x"], pts["sh_l"]["y"]), (pts["sh_r"]["x"], pts["sh_r"]["y"]),
            pixel_aspect=pixel_aspect)
    if "hip_l" in pts and "hip_r" in pts:
        out["hip_tilt_deg"] = tilt_degrees(
            (pts["hip_l"]["x"], pts["hip_l"]["y"]), (pts["hip_r"]["x"], pts["hip_r"]["y"]),
            pixel_aspect=pixel_aspect)
    for side in ("left", "right"):
        hip, knee, ankle = pts.get(f"hip_{side[0]}"), pts.get(f"kn_{side[0]}"), \
            pts.get(f"ank_{side[0]}")
        if not (hip and knee and ankle):
            continue
        interior = angle_degrees((hip["x"], hip["y"]), (knee["x"], knee["y"]),
                                 (ankle["x"], ankle["y"]), pixel_aspect=pixel_aspect)
        out["knee_interior_deg"][side] = interior
        out["knee_flexion_deg"][side] = flexion_degrees(interior)
    # Both wrists or nothing: the midpoint is the same quantity the wrist-path metric
    # uses, exposed here so a card and the metric table cannot disagree about it.
    if "wr_l" in pts and "wr_r" in pts:
        out["wrist_midpoint"] = [round((pts["wr_l"]["x"] + pts["wr_r"]["x"]) / 2.0, 2),
                                 round((pts["wr_l"]["y"] + pts["wr_r"]["y"]) / 2.0, 2)]
    return out


def _metric(name: str, unit: str, values: List[float], *, frames_possible: int,
            description: str = "", unavailable_reason: str) -> MetricValue:
    usable = [v for v in values if v is not None]
    if len(usable) < MIN_METRIC_FRAMES:
        return MetricValue(name=name, unit=unit, state="unavailable",
                           reason=(f"{unavailable_reason}: {len(usable)} usable "
                                   f"frame(s), below the {MIN_METRIC_FRAMES} needed"),
                           frames_used=len(usable), frames_possible=frames_possible,
                           detail={"description": description} if description else None)
    if frames_possible and len(usable) / frames_possible < MIN_METRIC_FRACTION:
        return MetricValue(
            name=name, unit=unit, state="unavailable",
            reason=(f"{unavailable_reason}: landmarks were usable on only "
                    f"{len(usable)} of {frames_possible} frames, below the "
                    f"{MIN_METRIC_FRACTION:.0%} needed for a median to mean anything"),
            frames_used=len(usable), frames_possible=frames_possible,
            detail={"description": description} if description else None)
    return MetricValue(name=name, unit=unit, state="supported", value=median(usable),
                       frames_used=len(usable), frames_possible=frames_possible,
                       detail={"description": description,
                               "p10": percentile(usable, 0.10),
                               "p90": percentile(usable, 0.90)} if description else None)


def body_metrics(body_records: Sequence[dict], *, frames_possible: int,
                 pixel_aspect: float = 1.0,
                 handedness: str = "right") -> Dict[str, object]:
    """The supported / unavailable split for the body, in 2D image space."""
    angles = [body_angles_for_record(r, pixel_aspect=pixel_aspect,
                                     handedness=handedness) for r in body_records]
    available = [r for r in body_records if r.get("available")]
    possible = frames_possible or len(body_records)
    lead = "left" if handedness == "right" else "right"
    trail = "right" if handedness == "right" else "left"

    values = {
        "shoulder_tilt": [a["shoulder_tilt_deg"] for a in angles],
        "hip_tilt": [a["hip_tilt_deg"] for a in angles],
        f"knee_interior_{lead}": [a["knee_interior_deg"][lead] for a in angles],
        f"knee_interior_{trail}": [a["knee_interior_deg"][trail] for a in angles],
        f"knee_flexion_{lead}": [a["knee_flexion_deg"][lead] for a in angles],
    }

    supported: List[MetricValue] = []
    unavailable: List[MetricValue] = []
    specs = (
        ("shoulder_tilt", "degrees", "shoulder line tilt from the image horizontal",
         "shoulder landmarks not resolvable"),
        ("hip_tilt", "degrees", "hip line tilt from the image horizontal",
         "hip landmarks not resolvable"),
        (f"knee_interior_{lead}", "degrees",
         f"lead ({lead}) " + KNEE_DEFINITION, f"lead {lead} knee chain not resolvable"),
        (f"knee_interior_{trail}", "degrees",
         f"trail ({trail}) " + KNEE_DEFINITION, f"trail {trail} knee chain not resolvable"),
        (f"knee_flexion_{lead}", "degrees", "180 - lead knee interior angle",
         "lead knee chain not resolvable"),
    )
    for key, unit, description, reason in specs:
        metric = _metric(key, unit, values[key], frames_possible=possible,
                         description=description, unavailable_reason=reason)
        (supported if metric.state == "supported" else unavailable).append(metric)

    # --- translations: nose, body centre, wrists
    nose = [(p["nose"]["x"], p["nose"]["y"]) for p in
            (_visible_points(r) for r in body_records) if "nose" in p]
    head_metric = _translation_metric(
        "head_displacement", nose, possible,
        "largest image displacement of the nose from its first resolvable frame",
        "nose not resolvable")
    (supported if head_metric.state == "supported" else unavailable).append(head_metric)

    centre = []
    for record in body_records:
        pts = _visible_points(record)
        if "hip_l" in pts and "hip_r" in pts:
            centre.append(((pts["hip_l"]["x"] + pts["hip_r"]["x"]) / 2.0,
                           (pts["hip_l"]["y"] + pts["hip_r"]["y"]) / 2.0))
    sway_metric = _translation_metric(
        "body_centre_lateral_sway", centre, possible,
        "range of the mid-hip x coordinate across resolvable frames",
        "hips not resolvable", axis="x")
    (supported if sway_metric.state == "supported" else unavailable).append(sway_metric)

    wrists = []
    for record in body_records:
        pts = _visible_points(record)
        # Both wrists, or nothing: switching between a two-wrist midpoint and a
        # single wrist would create a fake step in the path.
        if "wr_l" in pts and "wr_r" in pts:
            wrists.append(((pts["wr_l"]["x"] + pts["wr_r"]["x"]) / 2.0,
                           (pts["wr_l"]["y"] + pts["wr_r"]["y"]) / 2.0))
        else:
            wrists.append(None)
    wrist_path = _path_metric("wrist_path_length", wrists, body_records, possible,
                              "summed consecutive-frame steps of the wrist midpoint",
                              "wrists not resolvable")
    (supported if wrist_path.state == "supported" else unavailable).append(wrist_path)

    heights = _wrist_height_metric(wrists, possible)
    (supported if heights.state == "supported" else unavailable).append(heights)

    auto_metrics = _motion_metric(body_records, possible)
    (supported if auto_metrics.state == "supported" else unavailable).append(auto_metrics)

    return {
        "supported": [m.to_dict() for m in supported],
        "unavailable": [m.to_dict() for m in unavailable],
        "frames_with_body_pose": len(available),
        "frames_possible": possible,
        "conventions": {"tilt": TILT_CONVENTION, "knee": KNEE_DEFINITION,
                        "sides": SEGMENT_DEFINITION,
                        "space": "source pixels, single uncalibrated camera",
                        "pixel_aspect_ratio_applied": pixel_aspect},
    }


def _translation_metric(name: str, points: Sequence[tuple], possible: int,
                        description: str, reason: str,
                        axis: Optional[str] = None) -> MetricValue:
    if len(points) < MIN_METRIC_FRAMES:
        return MetricValue(name, "px", "unavailable",
                           reason=f"{reason}: {len(points)} usable frame(s)",
                           frames_used=len(points), frames_possible=possible)
    if axis == "x":
        xs = [p[0] for p in points]
        value = max(xs) - min(xs)
    else:
        first = points[0]
        value = max(((p[0] - first[0]) ** 2 + (p[1] - first[1]) ** 2) ** 0.5
                    for p in points)
    return MetricValue(name, "px", "supported", value=value, frames_used=len(points),
                       frames_possible=possible, detail={"description": description})


def _path_metric(name: str, points: Sequence[Optional[tuple]],
                 records: Sequence[dict], possible: int, description: str,
                 reason: str) -> MetricValue:
    """Path length over CONTIGUOUS source frames only.

    A frame without a pose ends the run, and so does any break in the source frame
    numbering: a list of records that happens to be adjacent is not the same thing
    as adjacent source frames, and counting a step across a sampling break would
    report motion that was never observed.
    """
    runs: List[List[tuple]] = []
    current: List[tuple] = []
    previous_frame: Optional[int] = None
    for index, record in enumerate(records):
        point = points[index] if index < len(points) else None
        frame = int(record.get("source_frame", index))
        contiguous = previous_frame is not None and frame == previous_frame + 1
        if point is None or not record.get("available") or not contiguous:
            if current:
                runs.append(current)
                current = []
        if point is not None and record.get("available"):
            current.append(point)
        previous_frame = frame
    if current:
        runs.append(current)
    total = 0.0
    used = 0
    for run in runs:
        used += len(run)
        if len(run) >= 2:
            total += path_length(run)
    if used < MIN_METRIC_FRAMES:
        return MetricValue(name, "px", "unavailable",
                           reason=f"{reason}: {used} usable frame(s)",
                           frames_used=used, frames_possible=possible)
    return MetricValue(name, "px", "supported", value=total, frames_used=used,
                       frames_possible=possible,
                       detail={"description": description,
                               "contiguous_runs": len(runs)})


def _wrist_height_metric(points: Sequence[Optional[tuple]], possible: int) -> MetricValue:
    usable = [p for p in points if p is not None]
    if len(usable) < MIN_METRIC_FRAMES:
        return MetricValue("wrist_height_range", "px", "unavailable",
                           reason=f"wrists not resolvable: {len(usable)} frame(s)",
                           frames_used=len(usable), frames_possible=possible)
    ys = [p[1] for p in usable]
    return MetricValue(
        "wrist_height_range", "px", "supported", value=max(ys) - min(ys),
        frames_used=len(usable), frames_possible=possible,
        detail={"description": ("vertical image excursion of the wrist midpoint; smaller "
                                "y is higher in the image"),
                "highest_frame_y": min(ys), "lowest_frame_y": max(ys)})


def _motion_metric(records: Sequence[dict], possible: int) -> MetricValue:
    """Median per-frame movement of all visible keypoints: a 2D activity measure.

    Only CONSECUTIVE frames with a pose are compared; a missing frame ends the run
    so that no step is measured across a gap.
    """
    steps: List[float] = []
    previous: Optional[Dict[str, dict]] = None
    previous_frame: Optional[int] = None
    for record in records:
        frame = int(record.get("source_frame", -1))
        if not record.get("available"):
            previous, previous_frame = None, None
            continue
        pts = _visible_points(record)
        if previous is not None and previous_frame is not None \
                and frame == previous_frame + 1:
            shared = [n for n in pts if n in previous]
            if shared:
                value = median([((pts[n]["x"] - previous[n]["x"]) ** 2
                                 + (pts[n]["y"] - previous[n]["y"]) ** 2) ** 0.5
                                for n in shared])
                if value is not None:
                    steps.append(value)
        previous, previous_frame = pts, frame
    if len(steps) < MIN_METRIC_FRAMES:
        return MetricValue("keypoint_motion", "px/frame", "unavailable",
                           reason=f"too few contiguous pose frames: {len(steps)}",
                           frames_used=len(steps), frames_possible=possible)
    return MetricValue("keypoint_motion", "px/frame", "supported", value=median(steps),
                       frames_used=len(steps), frames_possible=possible,
                       detail={"description": ("median per-frame movement of the "
                                               "keypoints that are visible on two "
                                               "consecutive frames; image-space "
                                               "activity, at container cadence")})


def _count(coverage: dict, key: str) -> int:
    """A count from a coverage block, never None and never a surprise type."""
    value = coverage.get(key, 0)
    return int(value) if isinstance(value, (int, float)) else 0


# ---------------------------------------------------------------- targets

def target_metrics(track: "TargetTrack", *, frames_possible: int) -> dict:
    """Image-space accounting for one tracked target. No speed, no physics."""
    coverage = track.coverage
    accounting = track.accounting
    if not track.available:
        return {"available": False, "reason": track.reason,
                "observed_frames": 0, "frames_total": frames_possible,
                "limitations": track.limitations}
    observed = coverage["observed_frames"]
    fps_note = ("px/frame is a step per source frame at container cadence; it is NOT "
                "a speed and it is not measured action time")
    missing = _count(coverage, "missing_frames")
    lost = _count(coverage, "lost_frames")
    return {
        "available": True,
        "reason": "",
        "observed_frames": observed,
        "automatic_observed_frames": coverage["automatic_observed_frames"],
        "operator_anchor_frames": coverage["operator_anchor_frames"],
        "interpolated_frames": coverage["interpolated_frames"],
        "missing_frames": missing,
        "lost_frames": lost,
        # the two counts above are disjoint, so the total is what a reader wants
        "unobserved_frames": _count(coverage, "unobserved_frames") or (missing + lost),
        "operator_stop_frames": _count(coverage, "operator_stop_frames"),
        "automatic_stop_frames": _count(coverage, "automatic_stop_frames"),
        "frames_total": coverage["frames_total"],
        "observed_fraction": coverage["observed_fraction"],
        "longest_missing_run": coverage["longest_missing_run"],
        "segment_count": accounting["segment_count"],
        "path_length_observed_px": accounting["path_length_observed_px"],
        "max_observed_step_px": accounting["max_observed_step_px"],
        "path_length_operator_anchors_px": accounting["path_length_operator_anchors_px"],
        "path_length_including_interpolated_px":
            accounting["path_length_including_interpolated_px"],
        "space": "2d_image",
        "unit": "px",
        "cadence_note": fps_note,
        "limitations": track.limitations,
    }


def targets_summary(tracks: Dict[str, "TargetTrack"], *, frames_possible: int) -> dict:
    return {name: target_metrics(track, frames_possible=frames_possible)
            for name, track in tracks.items()}


def reason_key(reason: str) -> str:
    """A stable key for a frame-missing reason.

    The canonical pose reasons carry the measured share in their text ("pose covers
    24.9% of the identified subject region"), so the raw strings differ per frame and
    a histogram keyed on them is noise. The numbers are masked out, which leaves one
    key per KIND of reason while the full text is kept alongside as an example.
    """
    text = str(reason or "no_pose").strip()
    masked = re.sub(r"\d+(?:\.\d+)?", "#", text)
    head = masked.split(";")[0].split(":")[0].strip()
    return (head or masked or "no_pose")[:90]


def missingness_report(body_records: Sequence[dict],
                       tracks: Dict[str, "TargetTrack"]) -> dict:
    """What was and was not measurable, stated as counts rather than adjectives."""
    posed = [r for r in body_records if r.get("available")]
    reasons: Dict[str, int] = {}
    examples: Dict[str, str] = {}
    for record in body_records:
        if record.get("available"):
            continue
        raw = str(record.get("reason") or "no_pose")
        key = reason_key(raw)
        reasons[key] = reasons.get(key, 0) + 1
        examples.setdefault(key, raw)
    return {
        "body_frames_total": len(body_records),
        "body_frames_with_pose": len(posed),
        "body_frames_without_pose": len(body_records) - len(posed),
        "body_missing_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        #: the un-masked text behind each grouped key, so the report can show the
        #: actual measurement instead of a masked pattern
        "body_missing_reason_examples": examples,
        "targets": {name: {"observed_frames": track.coverage.get("observed_frames", 0),
                           "missing_frames": track.coverage.get("missing_frames", 0),
                           "lost_frames": track.coverage.get("lost_frames", 0),
                           "unobserved_frames": _count(track.coverage,
                                                      "unobserved_frames")
                           or (_count(track.coverage, "missing_frames")
                               + _count(track.coverage, "lost_frames")),
                           "operator_stop_frames": _count(track.coverage,
                                                          "operator_stop_frames"),
                           "automatic_stop_frames": _count(track.coverage,
                                                           "automatic_stop_frames"),
                           "interpolated_frames": track.coverage.get("interpolated_frames", 0)}
                    for name, track in tracks.items()},
        "note": ("Missing frames are reported, never filled in. ``missing_frames`` and "
                 "``lost_frames`` are disjoint status counts; ``unobserved_frames`` is "
                 "their sum. Interpolated frames are derived and are counted separately "
                 "from observations, and frames past a stop frame are counted apart "
                 "because the operator put them out of scope."),
    }
