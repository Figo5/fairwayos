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
import signal
import subprocess
import time
from dataclasses import dataclass
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


class _InferenceTimeout(TimeoutError):
    pass


def _run_with_timeout(function, *, seconds: float):
    """Run one optional model call with a clean local wall-clock boundary."""
    if not hasattr(signal, "setitimer"):
        return function()
    def alarm_handler(signum, frame):
        raise _InferenceTimeout("bounded model inference timed out")
    previous = signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, max(0.1, float(seconds)))
    try:
        return function()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


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


def build_research_impact_bracket(events: Sequence[Mapping[str, Any]], *,
                                  frame_numbers: Sequence[int]) -> dict[str, Any]:
    """Build a research-only bracket around a predicted Impact sample."""
    impact_frames = [int(event["frame_index"]) for event in events
                     if event.get("event") == "Impact" and "frame_index" in event]
    samples = sorted({int(frame) for frame in frame_numbers})
    if not impact_frames or not samples:
        return {"state": "unavailable", "frames": [],
                "reason": "no SwingNet impact event or sampled frames",
                "research_only": True, "ground_truth": False,
                "production_eligible": False}
    if len(samples) < 2:
        return {"state": "unavailable", "frames": [],
                "reason": "impact bracket requires two distinct sampled frames",
                "research_only": True, "ground_truth": False,
                "production_eligible": False}
    predicted = impact_frames[0]
    nearest = min(range(len(samples)), key=lambda index: (abs(samples[index] - predicted), samples[index]))
    left = samples[max(0, nearest - 1)]
    right = samples[min(len(samples) - 1, nearest + 1)]
    return {"state": "candidate_bracket_only", "frames": [left, right],
            "reason": "SwingNet event prediction; exact contact unavailable",
            "research_only": True, "ground_truth": False,
            "production_eligible": False}


def build_demo_encoding_command(ffmpeg: str, frames_dir: str, output_path: str,
                                frame_rate: float) -> list[str]:
    """Return the deterministic H.264/yuv420p demo encoding command."""
    return [ffmpeg, "-y", "-loglevel", "error", "-framerate", f"{frame_rate:g}",
            "-i", str(Path(frames_dir) / "frame_%06d.jpg"), "-vf",
            "scale=in_range=pc:out_range=tv,format=yuv420p", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)]


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


def build_visual_review(verdicts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a research-only visual review block; flags can never be promoted."""
    items = []
    for verdict in verdicts:
        item = dict(verdict)
        action = str(item.get("recommended_action", item.get("verdict", ""))).lower()
        if action == "reject" or item.get("false_positive") is True:
            item["ball_marker_aligned"] = False
        item["research_only"] = True
        item["ground_truth"] = False
        item["production_eligible"] = False
        items.append(item)
    return {
        "verdicts": items,
        "research_only": True,
        "ground_truth": False,
        "production_eligible": False,
    }


def build_demo_report(*, source: Mapping[str, Any], media: Mapping[str, Any],
                      swing_window: Mapping[str, Any], observations: Sequence[Mapping[str, Any]],
                      artifact_references: Iterable[str], warnings: Iterable[str],
                      swingnet_events: Sequence[Mapping[str, Any]] = (),
                      impact_bracket: Optional[Mapping[str, Any]] = None,
                      render: Optional[Mapping[str, Any]] = None,
                      visual_review: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Create the stable report contract; analytics fields are always unavailable."""
    refs = sorted({str(ref) for ref in artifact_references})
    if any(not ref or os.path.isabs(ref) or ".." in Path(ref).parts for ref in refs):
        raise ValueError("artifact references must be safe relative paths")
    report = {
        "schema_version": DEMO_SCHEMA_VERSION,
        "status": "research_only",
        "source": dict(source),
        "media": dict(media),
        "swing_window": dict(swing_window),
        "observations": [dict(item) for item in observations],
        "swingnet_events": [dict(item) for item in swingnet_events],
        "impact_bracket": dict(impact_bracket or {
            "state": "unavailable", "frames": [],
            "reason": "no SwingNet impact event or sampled frames",
            "research_only": True, "ground_truth": False,
            "production_eligible": False,
        }),
        "artifact_references": refs,
        "methods": ["local_yolo_pose", "local_golf_ball", "local_swingnet_research_only", "classical_frame_difference", "guarded_candidate_rejection"],
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
    if render is not None:
        render_view = dict(render)
        render_view["research_only"] = True
        render_view["ground_truth"] = False
        render_view["production_eligible"] = False
        report["render"] = render_view
    if visual_review is not None:
        review_view = dict(visual_review)
        review_view["research_only"] = True
        review_view["ground_truth"] = False
        review_view["production_eligible"] = False
        report["visual_review"] = review_view
    return report


def _default_model_path(name: str) -> Optional[str]:
    root = Path(__file__).resolve().parents[2]
    candidates = {
        "pose": root / "out/research_training_gauntlet/yolo11n-pose.pt",
        "ball": root / "out/research_training_gauntlet/models/notjulietxd_golf_ball_tracker/best.pt",
    }
    path = candidates.get(name)
    return str(path) if path is not None and path.is_file() else None


def _load_swingnet(path: Optional[str] = None):
    """Load the local SwingNet checkpoint when the optional research stack exists."""
    root = Path(__file__).resolve().parents[2]
    weights = Path(path).expanduser() if path else root / "out/golfdb_evaluation/swingnet_1800.pth.tar"
    module_dir = weights.parent
    model_file = module_dir / "model.py"
    if not weights.is_file() or not model_file.is_file():
        return None, "swingnet_unavailable"
    try:
        import sys
        import torch
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))
        import model as swingnet_model
        original_load = torch.load
        torch.load = lambda *args, **kwargs: {}
        try:
            net = swingnet_model.EventDetector(pretrain=False, width_mult=1., lstm_layers=1,
                                               lstm_hidden=256, bidirectional=True, dropout=False)
        finally:
            torch.load = original_load
        checkpoint = torch.load(str(weights), map_location="cpu", weights_only=False)
        net.load_state_dict(checkpoint["model_state_dict"])
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        net.to(device).eval()
        from torch.autograd import Variable
        def init_hidden(batch_size):
            layers = 2 * 1
            return (Variable(torch.zeros(layers, batch_size, 256, device=device), requires_grad=False),
                    Variable(torch.zeros(layers, batch_size, 256, device=device), requires_grad=False))
        net.init_hidden = init_hidden
        return (net, device), None
    except Exception:
        return None, "swingnet_load_failed"


def _swingnet_events(bundle, frames, frame_numbers, fps):
    if bundle is None or not frames:
        return [], "swingnet_unavailable"
    try:
        import cv2
        import numpy as np
        import torch
        net, device = bundle
        images = np.asarray([cv2.cvtColor(cv2.resize(frame, (160, 160)), cv2.COLOR_BGR2RGB)
                             for frame in frames]).transpose(0, 3, 1, 2).astype("float32") / 255.0
        tensor = torch.from_numpy(images)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        with torch.no_grad():
            logits = net(tensor.unsqueeze(0).to(device)).detach().cpu()
        probs = torch.softmax(logits, dim=1).numpy()
        labels = ('Address', 'Toe-up', 'Mid-backswing', 'Top', 'Mid-downswing',
                  'Impact', 'Mid-follow-through', 'Finish')
        events = []
        for event_index, label in enumerate(labels):
            position = int(np.argmax(probs[:, event_index]))
            source_index = int(frame_numbers[position])
            events.append({"event": label, "frame_index": source_index,
                           "timestamp_seconds": round(source_index / fps, 6),
                           "confidence": round(float(probs[position, event_index]), 6),
                           "provenance": "SwingNet model prediction; research-only",
                           "research_only": True, "ground_truth": False,
                           "production_eligible": False})
        return events, None
    except Exception:
        return [], "swingnet_inference_failed"


def _load_yolo(path: Optional[str], task: str):
    if not path:
        return None, "model_unavailable"
    try:
        from ultralytics import YOLO
        model = _run_with_timeout(
            lambda: YOLO(str(Path(path).expanduser().resolve(strict=True)), task=task),
            seconds=10.0)
        return model, None
    except _InferenceTimeout:
        return None, "model_load_timeout"
    except Exception:
        return None, "model_load_failed"


def _pose_observation(model, frame, width: int, height: int):
    if model is None:
        return None, None
    try:
        result = _run_with_timeout(lambda: model(frame, verbose=False)[0], seconds=2.0)
        if result.boxes is None or len(result.boxes) == 0:
            return None, "golfer_not_detected"
        best = None
        person_count = 0
        for index, cls in enumerate(result.boxes.cls.tolist()):
            if int(cls) != 0:
                continue
            person_count += 1
            raw_confidence = result.boxes.conf[index]
            confidence_value = raw_confidence.tolist() if hasattr(raw_confidence, "tolist") else raw_confidence
            confidence = float(confidence_value)
            if best is None or confidence > best[0]:
                best = (confidence, index, result.boxes.xyxy[index].tolist())
        if best is None:
            return None, "golfer_not_detected"
        confidence, index, raw_box = best
        x1, y1, x2, y2 = [max(0.0, min(float(value), limit)) for value, limit in zip(raw_box, (width, height, width, height))]
        keypoints = []
        if result.keypoints is not None:
            raw_points = result.keypoints.xy[index]
            points = raw_points.cpu().numpy() if hasattr(raw_points, "cpu") else raw_points.tolist()
            raw_conf = result.keypoints.conf[index] if result.keypoints.conf is not None else []
            point_conf = raw_conf.cpu().numpy() if hasattr(raw_conf, "cpu") else (raw_conf.tolist() if hasattr(raw_conf, "tolist") else raw_conf)
            for point_index, point in enumerate(points):
                score = float(point_conf[point_index]) if len(point_conf) > point_index else 0.0
                keypoints.append([round(float(point[0]), 2), round(float(point[1]), 2), round(score, 4)])
        feet = [point for point in keypoints[15:17] if len(point) >= 3 and point[2] >= 0.25]
        anchor = {"x": round(sum(point[0] for point in feet) / len(feet), 2),
                  "y": round(sum(point[1] for point in feet) / len(feet), 2)} if feet else None
        return {
            "state": ObservationState.OBSERVED.value,
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "uncertainty": round((1.0 - confidence) * max(width, height) * 0.05, 2),
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "keypoints": keypoints,
            "anchor": anchor,
            "track_id": "golfer-0",
            "track_confidence": round(max(0.0, min(1.0, confidence)), 4),
            "person_count": person_count,
            "second_person_count": max(0, person_count - 1),
            "multi_person_frame": person_count > 1,
            "model": "local_yolo_pose",
        }, None
    except _InferenceTimeout:
        return None, "pose_inference_timeout"
    except Exception:
        return None, "pose_inference_failed"


def _box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _boxes_overlap(left, right):
    return not (left[2] <= right[0] or right[2] <= left[0] or
                left[3] <= right[1] or right[3] <= left[1])


def _distance_to_segment(point, start, end):
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - x1, py - y1)
    scale = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (x1 + scale * dx), py - (y1 + scale * dy))


def _pose_exclusion_reasons(box, pose):
    if not pose:
        return []
    reasons = []
    bbox = pose.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            expanded = (float(bbox[0]) - 8.0, float(bbox[1]) - 8.0,
                        float(bbox[2]) + 8.0, float(bbox[3]) + 8.0)
            if _boxes_overlap(box, expanded):
                reasons.append("golfer_bbox_overlap")
        except (TypeError, ValueError):
            pass
    points = []
    for index, raw in enumerate(pose.get("keypoints") or ()):
        if not isinstance(raw, (list, tuple)) or len(raw) < 3:
            continue
        try:
            x, y, confidence = float(raw[0]), float(raw[1]), float(raw[2])
        except (TypeError, ValueError):
            continue
        if confidence < 0.25 or not all(math.isfinite(value) for value in (x, y, confidence)):
            continue
        points.append((index, (x, y)))
        radius = 24.0 if index <= 4 else 16.0
        if math.hypot(max(box[0] - x, 0.0, x - box[2]),
                      max(box[1] - y, 0.0, y - box[3])) <= radius:
            reasons.append("head_region" if index <= 4 else "pose_region")
    point_map = dict(points)
    for left, right in POSE_SKELETON:
        if left not in point_map or right not in point_map:
            continue
        if _distance_to_segment(((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0),
                                point_map[left], point_map[right]) <= 16.0:
            reasons.append("pose_limb_region")
    # Hands/wrists are the usual entry point for club/shaft false positives;
    # exclude a conservative corridor from each wrist toward the ground.
    for wrist in (9, 10):
        if wrist not in point_map:
            continue
        wx, wy = point_map[wrist]
        corridor_end = (wx, wy + max(40.0, (box[3] - box[1]) * 0.8))
        if _distance_to_segment(((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0),
                                (wx, wy), corridor_end) <= 12.0:
            reasons.append("club_or_shaft_region")
    return sorted(set(reasons))


def filter_ball_model_candidates(candidates, *, pose=None, image_width, image_height):
    """Apply strict pixel-space gates before a model candidate can be tracked."""
    accepted, rejected = [], []
    for candidate in candidates:
        reasons = []
        if not isinstance(candidate, dict):
            rejected.append({"candidate": candidate, "reasons": ["malformed_candidate"]})
            continue
        try:
            box = tuple(float(value) for value in candidate["box"])
            confidence = float(candidate.get("confidence", 0.0))
        except (KeyError, TypeError, ValueError):
            rejected.append({"candidate": dict(candidate), "reasons": ["malformed_candidate"]})
            continue
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            reasons.append("nonfinite_geometry")
        elif not (0.0 <= box[0] < box[2] <= image_width and
                  0.0 <= box[1] < box[3] <= image_height):
            reasons.append("box_out_of_bounds")
        else:
            width, height = box[2] - box[0], box[3] - box[1]
            if min(width, height) < 2.0:
                reasons.append("ball_too_small")
            if max(width, height) > min(image_width, image_height) * 0.12:
                reasons.append("ball_too_large")
            if _box_area(box) > image_width * image_height * 0.015:
                reasons.append("ball_area_implausible")
            if max(width, height) / min(width, height) > 2.2:
                reasons.append("ball_shape_implausible")
        if not math.isfinite(confidence) or confidence < 0.35:
            reasons.append("low_confidence")
        reasons.extend(_pose_exclusion_reasons(box, pose) if len(box) == 4 else ())
        if reasons:
            rejected.append({"candidate": dict(candidate), "reasons": sorted(set(reasons))})
        else:
            item = dict(candidate)
            item["box"] = list(box)
            item["confidence"] = confidence
            accepted.append(item)
    return accepted, rejected


def validate_rendered_ball_markers(observations):
    """Return render-safety violations for accepted points inside golfer boxes."""
    violations = []
    for observation in observations:
        ball = observation.get("ball") or {}
        point = ball.get("point")
        bbox = (observation.get("golfer") or {}).get("bbox")
        if not isinstance(point, dict) or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x, y = _finite(point.get("x"), -1), _finite(point.get("y"), -1)
        if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
            violations.append({"frame_index": observation.get("frame_index"),
                               "reason": "accepted_marker_inside_golfer_bbox"})
    return violations


class ResearchBallEvidenceGate:
    """Require pose-safe model/heuristic agreement on consecutive clean frames."""

    def __init__(self, tracker, research_tracker, *, min_consecutive=2,
                 match_distance=28.0, static_distance=2.0, static_streak=2):
        if min_consecutive < 2 or match_distance <= 0:
            raise ValueError("invalid evidence gate limits")
        self.tracker = tracker
        self.research_tracker = research_tracker
        self.min_consecutive = int(min_consecutive)
        self.match_distance = float(match_distance)
        self.static_distance = float(static_distance)
        self.static_streak = int(static_streak)
        self.previous_frame = None
        self.agreement_streak = 0
        self.static_streak_count = 0
        self.static_anchor = None

    def update(self, model_candidates, frame, *, pose, image_width, image_height):
        accepted, rejected = filter_ball_model_candidates(
            model_candidates, pose=pose, image_width=image_width, image_height=image_height)
        context = {}
        if pose and pose.get("bbox"):
            context["golfer_bbox"] = pose["bbox"]
        research_candidates = self.research_tracker.extract_candidates(
            frame, self.previous_frame, context=context)
        self.previous_frame = frame.copy() if hasattr(frame, "copy") else frame
        research_points = [(candidate.center, float(candidate.confidence))
                           for candidate in research_candidates]
        agreements = []
        for candidate in accepted:
            center = tuple(float(value) for value in candidate["center"])
            if any(math.hypot(center[0] - point[0], center[1] - point[1]) <= self.match_distance
                   for point, _ in research_points):
                agreements.append(candidate)
            else:
                rejected.append({"candidate": dict(candidate), "reasons": ["research_tracker_disagreement"]})
        if len(agreements) == 1:
            self.agreement_streak += 1
            center = tuple(float(value) for value in agreements[0]["center"])
            if (self.static_anchor is not None and
                    math.hypot(center[0] - self.static_anchor[0], center[1] - self.static_anchor[1]) <= self.static_distance):
                self.static_streak_count += 1
            else:
                self.static_streak_count = 1
            self.static_anchor = center
        else:
            self.agreement_streak = 0
            self.static_streak_count = 0
            self.static_anchor = None
        if (len(agreements) == 1 and image_height >= 720 and
                agreements[0]["center"][1] >= image_height * 0.72 and
                self.static_streak_count >= self.static_streak):
            rejected.append({"candidate": dict(agreements[0]),
                             "reasons": ["static_ground_or_trouser_false_positive"]})
            self.tracker.update([])
            return {
                "state": ObservationState.UNAVAILABLE.value, "point": None,
                "confidence": 0.0, "uncertainty": None,
                "candidate_count": len(model_candidates),
                "research_candidate_count": len(research_candidates),
                "agreement_streak": self.agreement_streak,
                "rejected_candidates": rejected,
                "tracker_warning": "static_ground_or_trouser_false_positive",
                "research_only": True, "ground_truth": False, "production_eligible": False,
            }
        if self.agreement_streak < self.min_consecutive or len(agreements) != 1:
            self.tracker.update([])
            return {
                "state": ObservationState.UNAVAILABLE.value, "point": None,
                "confidence": 0.0, "uncertainty": None,
                "candidate_count": len(model_candidates),
                "research_candidate_count": len(research_candidates),
                "agreement_streak": self.agreement_streak,
                "rejected_candidates": rejected,
                "tracker_warning": "temporal_agreement_required" if agreements else "candidate_rejected",
                "research_only": True, "ground_truth": False, "production_eligible": False,
            }
        tracked = self.tracker.update(agreements)
        state = _normalize_tracker_state(tracked.get("state"))
        point = tracked.get("point") if state in {"observed", "predicted"} else None
        if point is None:
            self.agreement_streak = 0
        return {
            "state": state if point is not None else ObservationState.UNAVAILABLE.value,
            "point": point, "confidence": float(tracked.get("confidence", 0.0)) if point else 0.0,
            "uncertainty": None if point is None else max(2.0, (1.0 - float(tracked.get("confidence", 0.0))) * 30.0),
            "candidate_count": len(model_candidates),
            "research_candidate_count": len(research_candidates),
            "agreement_streak": self.agreement_streak,
            "rejected_candidates": rejected,
            "tracker_warning": tracked.get("warning"),
            "tracker_state": tracked.get("state"),
            "research_only": True, "ground_truth": False, "production_eligible": False,
        }
def _ball_observation(model, tracker, frame, width: int, height: int,
                      pose=None, evidence_gate=None, roi=None):
    if model is None:
        return None, "ball_model_unavailable"
    try:
        from .research_ball_model import normalize_box
        offset_x = offset_y = 0
        inference_frame = frame
        inference_width, inference_height = width, height
        if roi is not None:
            offset_x, offset_y, roi_x2, roi_y2 = [int(value) for value in roi]
            inference_frame = frame[offset_y:roi_y2, offset_x:roi_x2]
            inference_height, inference_width = inference_frame.shape[:2]
        result = _run_with_timeout(lambda: model(inference_frame, verbose=False)[0], seconds=2.0)
        candidates = []
        if result.boxes is not None:
            for box in result.boxes:
                confidence = float(box.conf[0])
                try:
                    x1, y1, x2, y2 = normalize_box(
                        box.xyxy[0].tolist(), inference_width, inference_height)
                except ValueError:
                    # Implausible ball geometry (e.g. a frame-filling box) is
                    # skipped per box so one hallucination cannot discard a
                    # genuine ball detection emitted in the same frame.
                    continue
                x1, x2 = x1 + offset_x, x2 + offset_x
                y1, y2 = y1 + offset_y, y2 + offset_y
                candidates.append({"center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0), "confidence": confidence,
                                  "box": [x1, y1, x2, y2], "research_only": True,
                                  "ground_truth": False, "production_eligible": False})
        if evidence_gate is not None:
            gated = evidence_gate.update(
                candidates, frame, pose=pose, image_width=width, image_height=height,
            )
            gated["model"] = "local_golf_ball"
            return gated, None
        tracked = tracker.update(candidates)
        raw_state = tracked.get("state", ObservationState.UNAVAILABLE.value)
        state = _normalize_tracker_state(raw_state)
        warning = tracked.get("warning") or ("tracker_" + str(raw_state) if raw_state != state else None)
        point = tracked.get("point") if state != ObservationState.UNAVAILABLE.value else None
        if point is None:
            return {"state": state, "confidence": 0.0,
                    "uncertainty": None, "candidate_count": len(candidates), "model": "local_golf_ball",
                    "tracker_state": raw_state, "tracker_warning": warning,
                    "research_only": True, "ground_truth": False,
                    "production_eligible": False}, None
        return {"state": state,
                "confidence": round(float(tracked.get("confidence", 0.0)), 4),
                "uncertainty": round(max(2.0, (1.0 - float(tracked.get("confidence", 0.0))) * 30.0), 2),
                "point": {"x": round(float(point["x"]), 2), "y": round(float(point["y"]), 2)},
                "candidate_count": len(candidates), "model": "local_golf_ball",
                "tracker_warning": warning, "tracker_state": raw_state,
                "research_only": True, "ground_truth": False,
                "production_eligible": False}, None
    except _InferenceTimeout:
        return None, "ball_inference_timeout"
    except Exception:
        return None, "ball_inference_failed"


def _coarse_ball_candidates(model, frames, width: int, height: int, *, limit: int = 8,
                            max_frames: int = 4):
    """Probe sparse full frames only to locate a bounded native-FPS ROI."""
    if model is None:
        return [], "coarse_ball_model_unavailable"
    candidates = []
    try:
        from .research_ball_model import normalize_box
        for frame in list(frames)[:max(0, int(max_frames))]:
            result = _run_with_timeout(lambda: model(frame, verbose=False)[0], seconds=2.0)
            if result.boxes is None:
                continue
            for box in result.boxes:
                if len(candidates) >= limit:
                    return candidates, "coarse_candidate_limit"
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = normalize_box(box.xyxy[0].tolist(), width, height)
                candidates.append({"box": [x1, y1, x2, y2], "confidence": confidence,
                                   "research_only": True, "ground_truth": False,
                                   "production_eligible": False})
        return candidates, None
    except _InferenceTimeout:
        return candidates, "coarse_ball_inference_timeout"
    except Exception:
        return candidates, "coarse_ball_inference_failed"


POSE_SKELETON = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                 (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
                 (12, 14), (14, 16), (0, 5), (0, 6))


def _draw_pose(frame, pose):
    if not pose:
        return {"bbox": False, "skeleton": False, "anchor": False}
    x1, y1, x2, y2 = [int(value) for value in pose["bbox"]]
    import cv2
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
    points = pose.get("keypoints") or []
    drawn = set()
    for left, right in POSE_SKELETON:
        if left >= len(points) or right >= len(points):
            continue
        if points[left][2] >= 0.25 and points[right][2] >= 0.25:
            cv2.line(frame, (int(points[left][0]), int(points[left][1])),
                     (int(points[right][0]), int(points[right][1])), (255, 80, 0), 4, cv2.LINE_AA)
            drawn.update((left, right))
    for point_index, point in enumerate(points):
        if len(point) >= 3 and point[2] >= 0.25:
            cv2.circle(frame, (int(point[0]), int(point[1])), 6, (255, 180, 0), -1)
            drawn.add(point_index)
    anchor = pose.get("anchor")
    anchor_drawn = bool(anchor)
    if anchor_drawn:
        ax, ay = int(anchor["x"]), int(anchor["y"])
        cv2.drawMarker(frame, (ax, ay), (255, 255, 0), cv2.MARKER_CROSS, 28, 4)
        cv2.putText(frame, "FEET ANCHOR", (ax + 10, ay - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
    return {"bbox": True, "skeleton": bool(drawn), "anchor": anchor_drawn}


def _normalize_tracker_state(raw_state: Any) -> str:
    return raw_state if raw_state in {member.value for member in ObservationState} else ObservationState.UNAVAILABLE.value


def _draw_ball_overlay(frame, ball, trail, source_frame=None):
    import cv2
    rejected = (ball or {}).get("rejected_candidates") or []
    rejected_drawn = 0
    for record in rejected:
        candidate = record.get("candidate") if isinstance(record, dict) else None
        box = candidate.get("box") if isinstance(candidate, dict) else None
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 80, 255), 2)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 80, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x1, y2), (0, 80, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "BALL REJECTED", (max(4, x1), max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 2, cv2.LINE_AA)
        rejected_drawn += 1
    if not ball or not ball.get("point"):
        return {"marker": False, "tracer_points": 0, "zoom_inset": False,
                "rejected_markers": rejected_drawn}
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
    crop_source = source_frame if source_frame is not None else frame
    crop = crop_source[y0:y1, x0:x1]
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


def clean_frame_for_components(frame: Any) -> Any:
    """Return the unannotated frame shared by all inference components."""
    return frame.copy()


@dataclass
class BoundedProcessingBudget:
    """Hard limits for the native-FPS ROI pass; never opens production gates."""
    max_frames: int
    max_seconds: float
    max_memory_bytes: int
    max_candidates: int
    started_at: float = 0.0
    frames: int = 0
    memory_bytes: int = 0
    candidates: int = 0
    reason: Optional[str] = None

    def allow(self, *, frame_bytes: int, candidate_count: int, now: Optional[float] = None) -> bool:
        if self.started_at == 0.0:
            self.started_at = time.monotonic() if now is None else float(now)
        current = time.monotonic() if now is None else float(now)
        limits = (
            (self.frames >= self.max_frames, "frame_limit"),
            (current - self.started_at > self.max_seconds, "time_limit"),
            (self.memory_bytes + max(0, int(frame_bytes)) > self.max_memory_bytes, "memory_limit"),
            (self.candidates + max(0, int(candidate_count)) > self.max_candidates, "candidate_limit"),
        )
        for exceeded, reason in limits:
            if exceeded:
                self.reason = reason
                return False
        self.frames += 1
        self.memory_bytes += max(0, int(frame_bytes))
        self.candidates += max(0, int(candidate_count))
        return True


def build_bounded_roi(candidates: Sequence[Mapping[str, Any]], *, image_width: int,
                      image_height: int, padding: int = 64, max_candidates: int = 8) -> dict[str, Any]:
    """Build one clipped ROI from coarse candidates, with explicit hard bounds."""
    valid = []
    for candidate in list(candidates)[:max(0, int(max_candidates))]:
        try:
            box = [float(value) for value in candidate["box"]]
            if len(box) == 4 and all(math.isfinite(value) for value in box):
                x1, y1, x2, y2 = box
                if 0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height:
                    valid.append(box)
        except (KeyError, TypeError, ValueError):
            continue
    if not valid:
        return {"state": "unavailable", "candidate_count": 0, "reason": "no_coarse_candidate",
                "research_only": True, "ground_truth": False, "production_eligible": False}
    x1 = max(0, int(math.floor(min(box[0] for box in valid) - padding)))
    y1 = max(0, int(math.floor(min(box[1] for box in valid) - padding)))
    x2 = min(int(image_width), int(math.ceil(max(box[2] for box in valid) + padding)))
    y2 = min(int(image_height), int(math.ceil(max(box[3] for box in valid) + padding)))
    return {"state": "candidate_region", "box": [x1, y1, x2, y2],
            "candidate_count": len(valid), "padding": int(padding),
            "research_only": True, "ground_truth": False, "production_eligible": False}


def run_local_demo(video_path: str, output_dir: str, *, sample_fps: float = 4.0,
                   max_duration_seconds: float = 8.0, max_frames: Optional[int] = None,
                   source: Optional[Mapping[str, Any]] = None,
                   pose_model: Optional[str] = None, ball_model: Optional[str] = None,
                   visual_review: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
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
    sample_limit = None if max_frames is None else max(0, int(max_frames))
    duration_source_limit = max(
        1,
        int(math.floor(max(0.0, _finite(max_duration_seconds)) * max(0.1, fps))) + 1,
    )
    while index < duration_source_limit and (sample_limit is None or len(frames) < sample_limit):
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
    # Four full-frame probes are enough to seed a bounded ROI; never let a
    # requested output frame count silently turn the coarse pass into a scan.
    coarse_frames, coarse_numbers = list(frames[:4]), list(frame_numbers[:4])
    coarse_candidates, coarse_warning = _coarse_ball_candidates(
        ball_model, coarse_frames, width, height, max_frames=4)
    roi_plan = build_bounded_roi(coarse_candidates, image_width=width, image_height=height,
                                 padding=64, max_candidates=8)
    native_budget = None
    budget_warning = None
    pose_cache = {}
    if roi_plan["state"] == "candidate_region":
        # Re-read only the selected motion window at native FPS. The model sees
        # the ROI, while the heuristic tracker still sees the clean full frame.
        native_budget = BoundedProcessingBudget(
            max_frames=min(sample_limit if sample_limit is not None else 8, 8),
            max_seconds=120.0, max_memory_bytes=64 * 1024 * 1024, max_candidates=64)
        source_start, source_end = frame_numbers[0], frame_numbers[-1]
        native_cap = cv2.VideoCapture(str(video))
        native_frames, native_numbers = [], []
        native_index = 0
        while native_index <= source_end:
            ok, native_frame = native_cap.read()
            if not ok:
                break
            if native_index >= source_start:
                if not native_budget.allow(frame_bytes=getattr(native_frame, "nbytes", 0),
                                           candidate_count=8):
                    budget_warning = "native_roi_" + str(native_budget.reason)
                    break
                native_frames.append(native_frame)
                native_numbers.append(native_index)
            native_index += 1
        native_cap.release()
        if native_frames:
            frames, frame_numbers, scores, step = native_frames, native_numbers, motion_scores(native_frames), 1
            for coarse_frame, coarse_number in zip(coarse_frames[:8], coarse_numbers[:8]):
                pose_cache[coarse_number] = _pose_observation(
                    pose_model, coarse_frame, width, height)
            window["native_roi"] = True
            window["roi"] = dict(roi_plan)
        else:
            budget_warning = budget_warning or "native_roi_no_frames"
    else:
        window["native_roi"] = False
        window["roi"] = dict(roi_plan)
    if roi_plan["state"] == "candidate_region":
        swingnet_bundle, swingnet_warning = None, "swingnet_skipped_for_native_roi_budget"
    else:
        swingnet_bundle, swingnet_warning = _load_swingnet()
    try:
        from .research_ball_model import ResearchBallMultiHypothesisTrack
        from .research_ball import ResearchBallTracker
        ball_tracker = ResearchBallMultiHypothesisTrack(reacquire_confidence=0.75, max_step=80.0, max_misses=2, max_hypotheses=3)
        research_tracker = ResearchBallTracker(min_confidence=0.35, max_gap_frames=1,
                                               max_step_pixels=80.0, min_pixels=3,
                                               max_component_fraction=0.008,
                                               max_aspect_ratio=2.2)
        ball_evidence_gate = ResearchBallEvidenceGate(ball_tracker, research_tracker,
                                                       min_consecutive=2, match_distance=28.0)
    except Exception:
        ball_tracker = None
        ball_evidence_gate = None
        ball_warning = ball_warning or "ball_tracker_unavailable"
    swingnet_events, swingnet_inference_warning = _swingnet_events(
        swingnet_bundle, coarse_frames, coarse_numbers, fps)
    event_by_frame = {}
    for event in swingnet_events:
        event_by_frame.setdefault(event["frame_index"], []).append(event)
    impact_bracket = build_research_impact_bracket(swingnet_events, frame_numbers=frame_numbers)
    annotated = out / "annotated_frames"
    annotated.mkdir(exist_ok=True)
    for stale_frame in annotated.glob("frame_*.jpg"):
        if stale_frame.is_file() and not stale_frame.is_symlink():
            stale_frame.unlink()
    observations = []
    trail = []
    for ordinal, (frame, number) in enumerate(zip(frames, frame_numbers)):
        clean_frame = clean_frame_for_components(frame)
        item = clean_frame.copy()
        if roi_plan["state"] == "candidate_region":
            pose, pose_frame_warning = pose_cache.get(number, (None, "coarse_pose_not_available"))
        else:
            pose, pose_frame_warning = _pose_observation(pose_model, clean_frame, width, height)
        roi_box = roi_plan.get("box") if roi_plan["state"] == "candidate_region" else None
        if ball_tracker:
            if roi_box is None:
                ball, ball_frame_warning = _ball_observation(
                    ball_model, ball_tracker, clean_frame, width, height, pose, ball_evidence_gate)
            else:
                ball, ball_frame_warning = _ball_observation(
                    ball_model, ball_tracker, clean_frame, width, height, pose,
                    ball_evidence_gate, roi=roi_box)
        else:
            ball, ball_frame_warning = None, "ball_tracker_unavailable"
        if not ball or not ball.get("point") or ball.get("state") == ObservationState.UNAVAILABLE.value:
            trail.clear()
        else:
            trail.append((int(ball["point"]["x"]), int(ball["point"]["y"])))
            if len(trail) > 40:
                trail.pop(0)
        overlay_flags = _draw_ball_overlay(item, ball or {}, trail, source_frame=clean_frame)
        if ball is not None:
            ball["rendered_overlay"] = overlay_flags
        if pose:
            pose.setdefault("track_id", "golfer-0")
            pose.setdefault("track_confidence", pose.get("confidence", 0.0))
        pose_overlay = _draw_pose(item, pose)
        if pose:
            pose["skeleton_rendered"] = pose_overlay["skeleton"]
            pose["bbox_rendered"] = pose_overlay["bbox"]
            pose["anchor_rendered"] = pose_overlay["anchor"]
        events = event_by_frame.get(number, [])
        bracket_frames = impact_bracket.get("frames", [])
        impact_applicable = (impact_bracket.get("state") == "candidate_bracket_only"
                             and len(bracket_frames) == 2
                             and bracket_frames[0] <= number <= bracket_frames[1])
        impact = {"state": ObservationState.UNAVAILABLE.value, "confidence": 0.0, "uncertainty": None,
                  "rejection": "exact_contact_unavailable"}
        if impact_applicable:
            impact["bracket"] = list(bracket_frames)
            impact["bracket_state"] = "candidate_bracket_only"
        warnings = ["research_only", "ground_truth_false", "production_analytics_unavailable",
                    "camera_motion:not_assessed", "blur:not_assessed", "occlusion:not_assessed"]
        warnings.extend(value for value in (pose_warning, ball_warning, pose_frame_warning,
                                             ball_frame_warning, coarse_warning, budget_warning, swingnet_warning,
                                             swingnet_inference_warning) if value)
        if not pose:
            warnings.append("golfer_track_lost")
        if ball and ball.get("tracker_warning"):
            warnings.append(str(ball["tracker_warning"]))
        observations.append(build_demo_observation(
            frame_index=number, timestamp_seconds=number / fps,
            golfer=pose or {"state": ObservationState.UNAVAILABLE.value, "confidence": 0.0},
            pose=pose or {"state": ObservationState.UNAVAILABLE.value, "confidence": 0.0},
            ball=ball or {"state": ObservationState.UNAVAILABLE.value, "confidence": 0.0},
            clubhead=_candidate_evidence(item, pose, scores, ordinal), impact=impact, warnings=warnings,
        ))
        cv2.putText(item, "FAIRWAYOS AI DEMO | RESEARCH ONLY | NO PRODUCTION CLAIM", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(item, "FRAME %d | t=%.3fs | golfer=%s id=%s conf=%s" % (
            number + 1, number / fps, "observed" if pose else "unavailable",
            pose.get("track_id", "unavailable") if pose else "unavailable",
            "%.2f" % pose["confidence"] if pose else "0.00"),
            (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(item, "ball=%s conf=%s u=%spx tracer=%d" % (
            ball.get("state", "unavailable") if ball else "unavailable",
            "%.2f" % ball.get("confidence", 0.0) if ball else "0.00",
            "%.1f" % ball.get("uncertainty", 0.0) if ball and ball.get("uncertainty") is not None else "n/a",
            len(trail)), (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
        event_text = " | ".join(event["event"] for event in events) or "none on this frame"
        cv2.putText(item, "SwingNet research-only: " + event_text, (12, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 180, 255), 1, cv2.LINE_AA)
        bracket_text = (" | bracket %d-%d" % tuple(bracket_frames)) if impact_applicable else ""
        cv2.putText(item, "impact: unavailable" + bracket_text, (12, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 255), 1, cv2.LINE_AA)
        cv2.putText(item, "WARNINGS: camera/blur/occlusion not assessed | analytics unavailable", (12, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 80, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(annotated / f"frame_{ordinal + 1:06d}.jpg"), item)
    render_violations = validate_rendered_ball_markers(observations)
    if render_violations:
        # Last-resort render safety: an accepted point inside the golfer box is
        # downgraded before diagnostics and MP4 publication.
        bad_frames = {item["frame_index"] for item in render_violations}
        for observation in observations:
            if observation.get("frame_index") not in bad_frames:
                continue
            observation["ball"] = {"state": ObservationState.UNAVAILABLE.value,
                                    "confidence": 0.0, "uncertainty": None,
                                    "rejection": "accepted_marker_inside_golfer_bbox",
                                    "research_only": True, "ground_truth": False,
                                    "production_eligible": False}
            observation["warnings"] = sorted(set(observation.get("warnings", ())) |
                                              {"accepted_marker_inside_golfer_bbox"})
    cv2.destroyAllWindows()
    rendered = out / "annotated_video.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for H.264 demo output")
    subprocess.run(build_demo_encoding_command(
        ffmpeg, str(annotated), str(rendered), fps / step), check=True)
    media = {"fps": fps, "width": width, "height": height, "frame_count": total, "sample_fps": fps / step}
    sample_fps = media["sample_fps"]
    render_block = {
        "rendered_frames": len(observations),
        "sample_fps": sample_fps,
        "processing": {"mode": "coarse_full_frame_then_native_roi" if roi_plan["state"] == "candidate_region" else "bounded_sampled_full_frame",
                        "roi": dict(roi_plan),
                        "budget": None if native_budget is None else {
                            "frames": native_budget.frames, "max_frames": native_budget.max_frames,
                            "memory_bytes": native_budget.memory_bytes,
                            "max_memory_bytes": native_budget.max_memory_bytes,
                            "candidates": native_budget.candidates,
                            "max_candidates": native_budget.max_candidates,
                            "termination": native_budget.reason,
                        }},
        "duration_seconds": len(observations) / sample_fps if sample_fps else 0.0,
        "audio": "unavailable_dropped_by_reencode",
        "reason": "annotated re-render covers sampled frames only; source audio not carried into re-encode",
    }
    provenance = build_demo_provenance(source=source or {"platform": "local", "video_id": video.stem},
                                       video_path=video, media=media)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    report = build_demo_report(
        source=source or {"platform": "local", "video_id": video.stem},
        media=media,
        swing_window=window, observations=observations,
        swingnet_events=swingnet_events,
        impact_bracket=impact_bracket,
        artifact_references=["annotated_video.mp4", "annotated_frames/", "diagnostics.json", "provenance.json"],
        warnings=["research_only", "ground_truth_false", "production_analytics_unavailable", "clubhead_not_validated"],
        render=render_block,
        visual_review=visual_review,
    )
    (out / "diagnostics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
