#!/usr/bin/env python3
"""Local-only pseudo-label clubhead/contact experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ghostcaddie.video.clubhead_proposal import build_clubhead_observation
from ghostcaddie.video.clubhead_pseudo_labels import build_pseudo_label, estimate_impact_window
from ghostcaddie.video.research_ball_model import ResearchBallMultiHypothesisTrack, normalize_box

MODEL = ROOT / "out/research_training_gauntlet/models/notjulietxd_golf_ball_tracker/best.pt"


def _pose(result):
    if result.boxes is None or len(result.boxes) == 0:
        return None, None
    idx = int(result.boxes.conf.argmax())
    if int(result.boxes.cls[idx]) != 0:
        return None, None
    x1, y1, x2, y2 = [float(v) for v in result.boxes.xyxy[idx].tolist()]
    bbox = (x1, y1, x2 - x1, y2 - y1)
    pose = {"confidence": float(result.boxes.conf[idx])}
    if result.keypoints is not None and result.keypoints.conf is not None:
        pts = result.keypoints.xy[idx].cpu().numpy()
        conf = result.keypoints.conf[idx].cpu().numpy()
        if conf[9] >= 0.25 and conf[7] >= 0.25:
            pose["wrist"] = tuple(float(v) for v in pts[9])
            pose["elbow"] = tuple(float(v) for v in pts[7])
    return bbox, pose


def _ball_candidates(result, width, height):
    output = []
    if result.boxes is None:
        return output
    for box, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
        try:
            x1, y1, x2, y2 = normalize_box(box, width, height)
        except ValueError:
            continue
        output.append({"center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0), "confidence": float(confidence)})
    return output


def _flow(prev_gray, gray, point):
    if prev_gray is None or point is None:
        return None
    p0 = np.asarray([[point]], dtype=np.float32)
    p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(21, 21), maxLevel=3)
    if p1 is None or status is None or int(status[0][0]) != 1:
        return None
    return (float(p1[0][0][0] - point[0]), float(p1[0][0][1] - point[1]))


def run(source: Path, output: Path, max_frames: int = 0):
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"unable to open {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(total, max_frames) if max_frames else total
    output.mkdir(parents=True, exist_ok=True)
    ball_model = YOLO(str(MODEL))
    pose_model = YOLO("yolo11n-pose.pt")
    tracker = ResearchBallMultiHypothesisTrack(min_confidence=0.35, reacquire_confidence=0.75, max_step=80.0, max_misses=2, max_hypotheses=3)
    labels, frames = [], []
    previous_gray = None
    previous_clubhead = None
    for frame_index in range(limit):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ball_result = ball_model(frame, verbose=False)[0]
        ball_candidates = _ball_candidates(ball_result, width, height)
        ball_state = tracker.update(ball_candidates)
        ball_point = None if ball_state["point"] is None else (ball_state["point"]["x"], ball_state["point"]["y"])
        pose_result = pose_model(frame, verbose=False)[0]
        golfer_bbox, pose = _pose(pose_result)
        if golfer_bbox:
            bx, by, bw, bh = golfer_bbox
            roi = (max(0.0, bx - bw), max(0.0, by - bh), min(float(width), 3 * bw), min(float(height), 3 * bh))
        else:
            roi = (0.0, 0.0, float(width), float(height))
        rx, ry, rw, rh = [int(v) for v in roi]
        crop = gray[ry:min(height, ry + rh), rx:min(width, rx + rw)]
        edges = cv2.Canny(crop, 50, 150)
        detected = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=35, minLineLength=25, maxLineGap=8)
        lines = []
        for line in detected[:80] if detected is not None else ():
            ax, ay, bx2, by2 = [int(v) for v in line.reshape(-1)[:4]]
            length = float(np.hypot(bx2 - ax, by2 - ay))
            lines.append({"endpoint": (float(rx + bx2), float(ry + by2)), "score": min(1.0, length / 250.0), "length": length})
        contours = []
        if previous_gray is not None:
            diff = cv2.absdiff(gray, previous_gray)
            _, mask = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
            for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                area = cv2.contourArea(contour)
                if 8 <= area <= width * height * 0.1:
                    mx, my, mw, mh = cv2.boundingRect(contour)
                    contours.append({"center": (mx + mw / 2.0, my + mh / 2.0), "score": min(1.0, area / 500.0), "area": area})
        motion = [{"point": item["center"], "score": item["score"], "speed": item["score"]} for item in contours]
        candidate = build_clubhead_observation(frame_index=frame_index, image_size=(width, height), roi=roi, pose=pose, golfer_bbox=golfer_bbox, line_candidates=lines, contour_candidates=contours, motion_candidates=motion)
        flow = _flow(previous_gray, gray, previous_clubhead)
        label = build_pseudo_label(frame_index=frame_index, image_size=(width, height), candidate={"point": candidate.point, "confidence": candidate.confidence, "uncertainty_px": candidate.uncertainty_px, "evidence": candidate.evidence}, pose=pose or {}, ball_point=ball_point, flow_vector=flow, previous_point=previous_clubhead)
        label["ball_state"] = ball_state["state"]
        label["ball_point"] = ball_point
        label["ball_candidate_count"] = len(ball_candidates)
        labels.append(label)
        frames.append(frame)
        if label["available"]:
            previous_clubhead = (label["clubhead"]["value"]["x"], label["clubhead"]["value"]["y"])
        previous_gray = gray
    cap.release()
    impact_window = estimate_impact_window(labels, fps)
    for label in labels:
        label["impact_window"] = impact_window
    video_path = output / "pseudo_clubhead_experiment.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    accepted = 0
    rejected = 0
    trail = []
    for index, frame in enumerate(frames):
        label = labels[index]
        ball_point = label.get("ball_point")
        if ball_point is not None and label.get("ball_state") != "terminated":
            trail.append((int(ball_point[0]), int(ball_point[1])))
            trail = trail[-45:]
            if len(trail) > 1:
                cv2.polylines(frame, [np.asarray(trail, dtype=np.int32)], False, (0, 165, 255), 2)
            cv2.circle(frame, trail[-1], 5, (0, 165, 255), 1)
        else:
            trail = []
        if label["available"]:
            accepted += 1
            cx, cy = label["clubhead"]["value"]["x"], label["clubhead"]["value"]["y"]
            gx, gy = label["shaft"]["value"]["grip"]["x"], label["shaft"]["value"]["grip"]["y"]
            cv2.line(frame, (int(gx), int(gy)), (int(cx), int(cy)), (255, 255, 0), 2)
            cv2.circle(frame, (int(cx), int(cy)), max(5, int(label["uncertainty_px"])), (0, 255, 0), 2)
        else:
            rejected += 1
        if label.get("ball_club_alignment") is not None:
            cv2.putText(frame, f"PSEUDO CLUB {label['confidence']:.2f} +/-{label['uncertainty_px']:.1f}px", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(frame, f"ALIGN {label['ball_club_alignment']:.1f}px", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
        else:
            cv2.putText(frame, "PSEUDO CLUBHEAD UNAVAILABLE", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        if impact_window["available"]:
            cv2.putText(frame, f"IMPACT BRACKET {impact_window['start_frame']}-{impact_window['end_frame']} (NOT EXACT)", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)
        writer.write(frame)
    writer.release()
    report = {"schema_version": "clubhead-pseudo-experiment.v1", "source": source.name, "frame_count": len(labels), "accepted_pseudo_labels": accepted, "rejected_candidates": rejected, "impact_window": impact_window, "production_eligible": False, "ground_truth": False, "pseudo_label": True, "research_only": True, "clubhead": None, "impact": None, "trajectory": None, "landing": None, "calibration": None, "shot_event": None, "analytics": None, "recommendation": None, "frames": labels}
    (output / "pseudo_clubhead_experiment.json").write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    result = run(args.source, args.output, args.max_frames)
    print(json.dumps({k: result[k] for k in ("frame_count", "accepted_pseudo_labels", "rejected_candidates", "impact_window")}, sort_keys=True))
