"""Local-first, research-only AI Demo Mode contracts and orchestration helpers.

This module never calls the validated analytics pipeline. Optional model adapters
may add observations, but missing or ambiguous evidence remains explicit.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

DEMO_SCHEMA_VERSION = "fairwayos-ai-demo.v1"


class ObservationState(str, Enum):
    OBSERVED = "observed"
    INTERPOLATED = "interpolated"
    PREDICTED = "predicted"
    UNAVAILABLE = "unavailable"


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _evidence(value: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    item = dict(value or {})
    state = item.get("state", ObservationState.UNAVAILABLE.value)
    if state not in {member.value for member in ObservationState}:
        raise ValueError("invalid observation state")
    item["state"] = state
    item["confidence"] = max(0.0, min(1.0, _finite(item.get("confidence"))))
    item.setdefault("uncertainty", None)
    return item


def build_demo_observation(*, frame_index: int, timestamp_seconds: float,
                           golfer: Optional[Mapping[str, Any]] = None,
                           pose: Optional[Mapping[str, Any]] = None,
                           ball: Optional[Mapping[str, Any]] = None,
                           clubhead: Optional[Mapping[str, Any]] = None,
                           impact: Optional[Mapping[str, Any]] = None,
                           warnings: Iterable[str] = ()) -> dict[str, Any]:
    """Build one explicit observation without promoting any label to truth."""
    if isinstance(frame_index, bool) or int(frame_index) < 0:
        raise ValueError("frame_index must be non-negative")
    return {
        "frame_index": int(frame_index),
        "timestamp_seconds": round(_finite(timestamp_seconds), 6),
        "golfer": _evidence(golfer),
        "pose": _evidence(pose),
        "ball": _evidence(ball),
        "clubhead": _evidence(clubhead),
        "impact": _evidence(impact),
        "warnings": sorted({str(item) for item in warnings if str(item)}),
        "research_only": True,
        "ground_truth": False,
        "production_eligible": False,
    }


def select_swing_window(scores: Sequence[float], *, frame_rate: float,
                        max_duration_seconds: float = 8.0) -> dict[str, Any]:
    """Select a deterministic bounded motion window around the strongest peak."""
    fps = _finite(frame_rate)
    duration = _finite(max_duration_seconds)
    if fps <= 0 or duration <= 0:
        raise ValueError("frame_rate and max_duration_seconds must be positive")
    values = tuple(_finite(value) for value in scores)
    if not values:
        return {"start_frame": None, "end_frame": None, "peak_frame": None,
                "peak_score": 0.0, "status": "unavailable"}
    peak = max(range(len(values)), key=lambda index: (values[index], -index))
    limit = max(1, int(math.floor(fps * duration)))
    length = min(len(values), limit + 1)
    left = max(0, peak - length // 2)
    right = min(len(values) - 1, left + length - 1)
    return {"start_frame": left, "end_frame": right, "peak_frame": peak,
            "peak_score": round(values[peak], 6), "status": "candidate"}


def reject_obvious_false_positive(candidate: Mapping[str, Any], *, image_width: int,
                                  image_height: int) -> dict[str, Any]:
    """Reject geometry that cannot be a defensible tracked clubhead candidate."""
    reasons = []
    point = candidate.get("point")
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        reasons.append("point_unavailable")
    else:
        x, y = _finite(point[0], -1), _finite(point[1], -1)
        if not (0 <= x < image_width and 0 <= y < image_height):
            reasons.append("point_out_of_bounds")
    if not candidate.get("inside_golfer", False):
        reasons.append("not_supported_by_golfer_geometry")
    if int(candidate.get("temporal_support", 0) or 0) < 2:
        reasons.append("insufficient_temporal_support")
    if _finite(candidate.get("confidence")) < 0.5:
        reasons.append("low_confidence")
    return {"accepted": not reasons, "reasons": sorted(set(reasons)),
            "research_only": True, "ground_truth": False,
            "production_eligible": False}


def build_demo_report(*, source: Mapping[str, Any], media: Mapping[str, Any],
                      swing_window: Mapping[str, Any], observations: Sequence[Mapping[str, Any]],
                      artifact_references: Iterable[str], warnings: Iterable[str]) -> dict[str, Any]:
    """Create the stable report contract; analytics fields are always unavailable."""
    refs = sorted({str(ref) for ref in artifact_references})
    if any(not ref or os.path.isabs(ref) or ".." in Path(ref).parts for ref in refs):
        raise ValueError("artifact references must be safe relative paths")
    return {
        "schema_version": DEMO_SCHEMA_VERSION,
        "status": "research_only",
        "source": dict(source),
        "media": dict(media),
        "swing_window": dict(swing_window),
        "observations": [dict(item) for item in observations],
        "artifact_references": refs,
        "methods": ["local_yolo_pose", "local_golf_ball", "classical_frame_difference", "guarded_candidate_rejection"],
        "warnings": sorted({str(item) for item in warnings if str(item)}),
        "research_only": True,
        "ground_truth": False,
        "production_eligible": False,
        "coordinate_space": "pixels",
        "analytics": None,
        "shot_event": None,
        "landing": None,
        "calibration": None,
        "recommendation": None,
    }


def _default_model_path(name: str) -> Optional[str]:
    root = Path(__file__).resolve().parents[2]
    candidates = {
        "pose": root / "out/research_training_gauntlet/yolo11n-pose.pt",
        "ball": root / "out/research_training_gauntlet/models/notjulietxd_golf_ball_tracker/best.pt",
    }
    path = candidates.get(name)
    return str(path) if path is not None and path.is_file() else None


def _load_yolo(path: Optional[str], task: str):
    if not path:
        return None, "model_unavailable"
    try:
        from ultralytics import YOLO
        return YOLO(str(Path(path).expanduser().resolve(strict=True)), task=task), None
    except Exception:
        return None, "model_load_failed"


def _pose_observation(model, frame, width: int, height: int):
    if model is None:
        return None, None
    try:
        result = model(frame, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None, "golfer_not_detected"
        best = None
        for index, cls in enumerate(result.boxes.cls.tolist()):
            if int(cls) != 0:
                continue
            confidence = float(result.boxes.conf[index])
            if best is None or confidence > best[0]:
                best = (confidence, index, result.boxes.xyxy[index].tolist())
        if best is None:
            return None, "golfer_not_detected"
        confidence, index, raw_box = best
        x1, y1, x2, y2 = [max(0.0, min(float(value), limit)) for value, limit in zip(raw_box, (width, height, width, height))]
        keypoints = []
        if result.keypoints is not None:
            points = result.keypoints.xy[index].cpu().numpy()
            point_conf = result.keypoints.conf[index].cpu().numpy() if result.keypoints.conf is not None else []
            for point_index, point in enumerate(points):
                score = float(point_conf[point_index]) if len(point_conf) > point_index else 0.0
                keypoints.append([round(float(point[0]), 2), round(float(point[1]), 2), round(score, 4)])
        return {
            "state": ObservationState.OBSERVED.value,
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "uncertainty": round((1.0 - confidence) * max(width, height) * 0.05, 2),
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "keypoints": keypoints,
            "model": "local_yolo_pose",
        }, None
    except Exception:
        return None, "pose_inference_failed"


def _ball_observation(model, tracker, frame, width: int, height: int):
    if model is None:
        return None, "ball_model_unavailable"
    try:
        from .research_ball_model import normalize_box
        result = model(frame, verbose=False)[0]
        candidates = []
        if result.boxes is not None:
            for box in result.boxes:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = normalize_box(box.xyxy[0].tolist(), width, height)
                candidates.append({"center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0), "confidence": confidence,
                                  "box": [x1, y1, x2, y2]})
        tracked = tracker.update(candidates)
        raw_state = tracked.get("state", ObservationState.UNAVAILABLE.value)
        state = _normalize_tracker_state(raw_state)
        warning = tracked.get("warning") or ("tracker_" + str(raw_state) if raw_state != state else None)
        point = tracked.get("point") if state != ObservationState.UNAVAILABLE.value else None
        if point is None:
            return {"state": state, "confidence": 0.0,
                    "uncertainty": None, "candidate_count": len(candidates), "model": "local_golf_ball",
                    "tracker_state": raw_state, "tracker_warning": warning}, None
        return {"state": state,
                "confidence": round(float(tracked.get("confidence", 0.0)), 4),
                "uncertainty": round(max(2.0, (1.0 - float(tracked.get("confidence", 0.0))) * 30.0), 2),
                "point": {"x": round(float(point["x"]), 2), "y": round(float(point["y"]), 2)},
                "candidate_count": len(candidates), "model": "local_golf_ball",
                "tracker_warning": warning, "tracker_state": raw_state}, None
    except Exception:
        return None, "ball_inference_failed"


def _draw_pose(frame, pose):
    if not pose:
        return
    x1, y1, x2, y2 = [int(value) for value in pose["bbox"]]
    import cv2
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    points = pose.get("keypoints") or []
    for point in points:
        if point[2] >= 0.25:
            cv2.circle(frame, (int(point[0]), int(point[1])), 3, (255, 120, 0), -1)


def _normalize_tracker_state(raw_state: Any) -> str:
    return raw_state if raw_state in {member.value for member in ObservationState} else ObservationState.UNAVAILABLE.value


def _draw_ball_overlay(frame, ball, trail):
    if not ball or not ball.get("point"):
        return {"marker": False, "tracer_points": 0, "zoom_inset": False}
    import cv2
    import numpy as np
    point = ball["point"]
    cx, cy = int(round(point["x"])), int(round(point["y"]))
    trail_points = [(int(x), int(y)) for x, y in trail]
    if len(trail_points) > 1:
        cv2.polylines(frame, [np.asarray(trail_points, dtype="int32")], False, (0, 165, 255), 6, cv2.LINE_AA)
    radius = max(14, int(round((ball.get("uncertainty") or 6.0) * 1.5)))
    cv2.circle(frame, (cx, cy), radius, (0, 0, 255), 5, cv2.LINE_AA)
    cv2.line(frame, (cx - radius - 8, cy), (cx + radius + 8, cy), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - radius - 8), (cx, cy + radius + 8), (255, 255, 255), 2, cv2.LINE_AA)
    label = "BALL %s conf=%.2f u=%.1fpx" % (ball.get("state", "unavailable").upper(), ball.get("confidence", 0.0), ball.get("uncertainty") or 0.0)
    cv2.putText(frame, label, (max(4, cx + radius + 10), max(22, cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    inset_w, inset_h = min(260, frame.shape[1] // 3), min(200, frame.shape[0] // 3)
    half_w, half_h = max(20, inset_w // 4), max(20, inset_h // 4)
    x0, x1 = max(0, cx - half_w), min(frame.shape[1], cx + half_w)
    y0, y1 = max(0, cy - half_h), min(frame.shape[0], cy + half_h)
    crop = frame[y0:y1, x0:x1]
    if crop.size:
        inset = cv2.resize(crop, (inset_w, inset_h), interpolation=cv2.INTER_NEAREST)
        cv2.circle(inset, (inset_w // 2, inset_h // 2), max(10, radius // 2), (0, 0, 255), 4, cv2.LINE_AA)
        ix = max(0, frame.shape[1] - inset_w - 12)
        iy = 12
        frame[iy:iy + inset_h, ix:ix + inset_w] = inset
        cv2.rectangle(frame, (ix, iy), (ix + inset_w - 1, iy + inset_h - 1), (0, 0, 255), 4)
        cv2.putText(frame, "BALL ZOOM", (ix + 8, iy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return {"marker": True, "tracer_points": len(trail_points), "zoom_inset": bool(crop.size)}


def _draw_point(frame, evidence, color, label):
    if not evidence or not evidence.get("point"):
        return
    import cv2
    point = evidence["point"]
    center = (int(point["x"]), int(point["y"]))
    radius = max(4, int(round(evidence.get("uncertainty") or 4)))
    cv2.circle(frame, center, radius, color, 2)
    cv2.circle(frame, center, 3, color, -1)
    cv2.putText(frame, "%s %s conf=%.2f" % (label, evidence.get("state", "unavailable").upper(), evidence.get("confidence", 0.0)),
                (center[0] + 8, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _candidate_evidence(frame, pose, scores, index):
    """Return a visibly marked but non-promotable classical clubhead candidate."""
    if not pose or index <= 0 or index >= len(scores):
        return {"state": ObservationState.UNAVAILABLE.value, "confidence": 0.0, "uncertainty": None,
                "rejection": "pose_or_motion_unavailable"}
    points = [p for p in pose.get("keypoints", []) if len(p) >= 3 and p[2] >= 0.25]
    if not points or scores[index] <= 0:
        return {"state": ObservationState.UNAVAILABLE.value, "confidence": 0.0, "uncertainty": None,
                "rejection": "no_separable_clubhead_evidence"}
    wrist = points[9] if len(points) > 9 else points[-1]
    candidate = {"point": [wrist[0], wrist[1]], "confidence": 0.2, "inside_golfer": True,
                 "temporal_support": 1}
    decision = reject_obvious_false_positive(candidate, image_width=10**9, image_height=10**9)
    return {"state": ObservationState.UNAVAILABLE.value, "confidence": 0.0, "uncertainty": None,
            "candidate_point": candidate["point"], "rejection": ";".join(decision["reasons"]),
            "research_only": True, "ground_truth": False, "production_eligible": False}


def motion_scores(frames: Sequence[Any]) -> list[float]:
    """Return classical grayscale-difference scores when OpenCV is available."""
    try:
        import cv2
    except ImportError:
        return [0.0 for _ in frames]
    scores = []
    previous = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scores.append(0.0 if previous is None else float(cv2.absdiff(gray, previous).mean()))
        previous = gray
    return scores


def _unavailable(frame_index: int, timestamp: float, warning: str) -> dict[str, Any]:
    item = {"state": ObservationState.UNAVAILABLE.value, "confidence": 0.0,
            "uncertainty": None, "warning": warning}
    return build_demo_observation(frame_index=frame_index, timestamp_seconds=timestamp,
                                  golfer=item, pose=item, ball=item, clubhead=item,
                                  impact=item, warnings=[warning])


def build_demo_provenance(*, source: Mapping[str, Any], video_path: Path,
                          media: Mapping[str, Any]) -> dict[str, Any]:
    """Return reproducibility metadata without leaking local paths."""
    digest = hashlib.sha256()
    with video_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "schema_version": "fairwayos-ai-demo-provenance.v1",
        "mode": "ai-demo",
        "source": dict(source),
        "acquisition": {"status": "local_copy", "local_artifact": video_path.name},
        "media": dict(media),
        "sha256": digest.hexdigest(),
        "research_only": True,
        "ground_truth": False,
        "production_eligible": False,
    }


def run_local_demo(video_path: str, output_dir: str, *, sample_fps: float = 4.0,
                   max_duration_seconds: float = 8.0, max_frames: Optional[int] = None,
                   source: Optional[Mapping[str, Any]] = None,
                   pose_model: Optional[str] = None, ball_model: Optional[str] = None) -> dict[str, Any]:
    """Run bounded local demo perception and render H.264 output.

    Model adapters are intentionally optional. The guaranteed fallback emits
    explicit unavailable observations and a valid annotated H.264 copy.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv is required for AI Demo Mode rendering") from exc
    video = Path(video_path).expanduser().resolve(strict=True)
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / max(0.1, sample_fps))))
    frames, frame_numbers = [], []
    index = 0
    cap_limit = None if max_frames is None else max(0, int(max_frames))
    while cap_limit is None or len(frames) < cap_limit:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            frames.append(frame)
            frame_numbers.append(index)
        index += 1
    cap.release()
    if not frames:
        raise RuntimeError("no decodable frames in bounded demo input")
    scores = motion_scores(frames)
    window = select_swing_window(scores, frame_rate=max(0.1, fps / step),
                                 max_duration_seconds=max_duration_seconds)
    if window["start_frame"] is not None:
        start_index = int(window["start_frame"])
        end_index = int(window["end_frame"]) + 1
        frames = frames[start_index:end_index]
        frame_numbers = frame_numbers[start_index:end_index]
        scores = scores[start_index:end_index]
        window["source_sample_start"] = start_index
        window["source_sample_end"] = end_index - 1
        window["peak_frame"] = max(0, int(window["peak_frame"]) - start_index)
    pose_model, pose_warning = _load_yolo(pose_model or _default_model_path("pose"), "pose")
    ball_model, ball_warning = _load_yolo(ball_model or _default_model_path("ball"), "detect")
    try:
        from .research_ball_model import ResearchBallMultiHypothesisTrack
        ball_tracker = ResearchBallMultiHypothesisTrack(reacquire_confidence=0.75, max_step=80.0, max_misses=2, max_hypotheses=3)
    except Exception:
        ball_tracker = None
        ball_warning = ball_warning or "ball_tracker_unavailable"
    annotated = out / "annotated_frames"
    annotated.mkdir(exist_ok=True)
    observations = []
    trail = []
    for ordinal, (frame, number) in enumerate(zip(frames, frame_numbers)):
        item = frame.copy()
        pose, pose_frame_warning = _pose_observation(pose_model, frame, width, height)
        ball, ball_frame_warning = _ball_observation(ball_model, ball_tracker, frame, width, height) if ball_tracker else (None, "ball_tracker_unavailable")
        overlay_flags = {"marker": False, "tracer_points": 0, "zoom_inset": False}
        if not ball or not ball.get("point") or ball.get("state") == ObservationState.UNAVAILABLE.value:
            trail.clear()
        else:
            trail.append((int(ball["point"]["x"]), int(ball["point"]["y"])))
            if len(trail) > 40:
                trail.pop(0)
            overlay_flags = _draw_ball_overlay(item, ball, trail)
            ball["rendered_overlay"] = overlay_flags
        clubhead = _candidate_evidence(item, pose, scores, ordinal)
        impact = {"state": ObservationState.UNAVAILABLE.value, "confidence": 0.0, "uncertainty": None,
                  "rejection": "exact_contact_unavailable"}
        warnings = ["research_only", "ground_truth_false", "production_analytics_unavailable"]
        warnings.extend(value for value in (pose_warning, ball_warning, pose_frame_warning, ball_frame_warning) if value)
        if ball and ball.get("tracker_warning"):
            warnings.append(str(ball["tracker_warning"]))
        observations.append(build_demo_observation(
            frame_index=number, timestamp_seconds=number / fps,
            golfer=pose or {"state": ObservationState.UNAVAILABLE.value, "confidence": 0.0},
            pose=pose or {"state": ObservationState.UNAVAILABLE.value, "confidence": 0.0},
            ball=ball or {"state": ObservationState.UNAVAILABLE.value, "confidence": 0.0},
            clubhead=clubhead, impact=impact, warnings=warnings,
        ))
        _draw_pose(item, pose)
        cv2.putText(item, "FAIRWAYOS AI DEMO | RESEARCH ONLY", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(item, "golfer/pose: %s | ball/tracer: %s" % ("observed" if pose else "unavailable", ball.get("state", "unavailable") if ball else "unavailable"), (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(item, "clubhead: REJECTED CANDIDATE | impact: unavailable", (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
        cv2.putText(item, "confidence/uncertainty: model output; not ground truth", (12, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(item, "WARNING: pixel-space research evidence; no production analytics", (12, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(annotated / f"frame_{ordinal + 1:06d}.jpg"), item)
    cv2.destroyAllWindows()
    rendered = out / "annotated_video.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for H.264 demo output")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", f"{fps / step:g}", "-i", str(annotated / "frame_%06d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(rendered)], check=True)
    media = {"fps": fps, "width": width, "height": height, "frame_count": total, "sample_fps": fps / step}
    provenance = build_demo_provenance(source=source or {"platform": "local", "video_id": video.stem},
                                       video_path=video, media=media)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    report = build_demo_report(
        source=source or {"platform": "local", "video_id": video.stem},
        media=media,
        swing_window=window, observations=observations,
        artifact_references=["annotated_video.mp4", "annotated_frames/", "diagnostics.json", "provenance.json"],
        warnings=["research_only", "ground_truth_false", "production_analytics_unavailable", "clubhead_not_validated"],
    )
    (out / "diagnostics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
