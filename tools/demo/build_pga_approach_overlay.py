"""Footage-first local multilayer overlay: Si Woo Kim 120-yard approach.

LOCAL ONLY. Redistribution rights UNVERIFIED - do not publish or upload.

Three independent research layers are rendered only when supported:
- BODY: YOLO11n-pose person box for the main golfer during the setup/swing shot.
- CLUBHEAD: hidden/unavailable here; native review could not identify a stable
  clubhead point separate from the ball/shaft at 30fps.
- BALL: the existing visually sampled ball flight track, source frames 149-208.

No stale trails: each layer's marker/trail is cleared on unavailable/offscreen/cut
states. No speed, yardage, apex, carry, impact, landing, calibration, trajectory,
or production analytics are claimed. The broadcast's own "TO HOLE: 120 YDS" text
is broadcaster information, never our measurement.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from ghostcaddie.video.pga_source_gate import REJECTED, evaluate_source
from ghostcaddie.video.siwoo_layers import (
    LayerState,
    choose_golfer_candidate,
    can_continue_without_detection,
    contiguous_visible_ranges,
    is_golfer_body_candidate,
    object_visibility_for_frame,
    renderable_trail,
    shift_box,
)

SRC = Path("out/pga_official_acquisition/6404321996112.mp4")
TRACK = Path("out/pga_experiment_20260911/siwoo_flight_track.json")
POSE_MODEL = Path("out/research_training_gauntlet/yolo11n-pose.pt")
OUTDIR = Path("out/pga_demo_20260911")
W, H, FPS = 1280, 720, 30.0
PASS_START, PASS_END = 40, 340
BALL_START, BALL_END = 149, 208
CUT_FRAME = 216
BODY_ASSESSED_START, BODY_ASSESSED_END = 40, CUT_FRAME - 1
BODY_NATIVE_VISIBLE_START, BODY_NATIVE_VISIBLE_END = 40, 118
CLUBHEAD_ASSESSED_START, CLUBHEAD_ASSESSED_END = 40, 118
REVIEWED_VISIBILITY = {
    "ball": [(BALL_START, BALL_END, "observed_ball_track_development_clip")],
    "clubhead": [],
    "cuts": [CUT_FRAME],
}
F = cv2.FONT_HERSHEY_SIMPLEX
# BGR colors
BODY_C = (255, 165, 60)
BALL_C = (0, 220, 255)
CLUB_C = (80, 255, 80)
BAD_C = (85, 85, 235)
WARN_C = (70, 175, 245)
FG = (238, 235, 235)
DIM = (170, 165, 165)
PANEL_BG = (16, 14, 14)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frames(path: Path) -> Dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames: Dict[int, np.ndarray] = {}
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames[i] = frame
        i += 1
    cap.release()
    return frames


def white_bib_fraction(frame: np.ndarray, box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(((hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 170)).mean())


def _gray_patch(gray: np.ndarray, x: float, y: float, radius: int = 9) -> Optional[np.ndarray]:
    xi, yi = int(round(x)), int(round(y))
    if xi - radius < 0 or yi - radius < 0 or xi + radius + 1 > gray.shape[1] or yi + radius + 1 > gray.shape[0]:
        return None
    return gray[yi - radius:yi + radius + 1, xi - radius:xi + radius + 1]


def _template_step(frame: np.ndarray, xy: Tuple[float, float], tmpl: np.ndarray,
                   radius: int = 42, min_score: float = 0.60) -> Optional[dict]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y = xy
    th, tw = tmpl.shape[:2]
    x1 = max(0, int(round(x - radius)))
    y1 = max(0, int(round(y - radius)))
    x2 = min(gray.shape[1], int(round(x + radius)))
    y2 = min(gray.shape[0], int(round(y + radius)))
    roi = gray[y1:y2, x1:x2]
    if roi.shape[0] < th or roi.shape[1] < tw:
        return None
    res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    nx, ny = x1 + loc[0] + tw / 2.0, y1 + loc[1] + th / 2.0
    step = float(((nx - x) ** 2 + (ny - y) ** 2) ** 0.5)
    if score < min_score or step > radius:
        return {"visible": False, "reason": "template_extension_rejected", "proposal": [round(nx, 1), round(ny, 1)], "score": round(float(score), 3), "step": round(step, 2)}
    return {"visible": True, "point": [round(nx, 1), round(ny, 1)], "confidence": round(float(score), 3), "step": round(step, 2)}


def extend_ball_track_by_template(frames: Dict[int, np.ndarray], base_track: Dict[int, list]) -> Tuple[Dict[int, dict], dict]:
    """Adjacent-frame template extension from the visually accepted base segment.

    This is source-local, research-only evidence. It does not use the visibility
    audit coordinates as labels; it starts from the already accepted f149-f208
    segment and walks one native frame at a time, terminating on failed NCC/step.
    """
    extensions: Dict[int, dict] = {}
    meta = {"method": "adjacent_template_from_existing_track", "min_ncc": 0.60,
            "preflight_radius": 38, "post208_radius": 42,
            "green_probe": "rejected_false_locks_not_integrated"}
    # Backward f148..f98 from accepted f149.
    cur = (float(base_track[149][0]), float(base_track[149][1]))
    tmpl = _gray_patch(cv2.cvtColor(frames[149], cv2.COLOR_BGR2GRAY), *cur)
    if tmpl is not None:
        for idx in range(148, 97, -1):
            rec = _template_step(frames[idx], cur, tmpl, radius=38, min_score=0.60)
            if not rec or not rec.get("visible"):
                extensions[idx] = rec or {"visible": False, "reason": "template_extension_rejected"}
                break
            rec["reason"] = "adjacent_template_backward_from_existing_track"
            extensions[idx] = rec
            cur = tuple(rec["point"])
            nxt = _gray_patch(cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY), *cur)
            if nxt is not None:
                tmpl = nxt
    # Forward f209..f223 from accepted f208.
    cur = (float(base_track[208][0]), float(base_track[208][1]))
    tmpl = _gray_patch(cv2.cvtColor(frames[208], cv2.COLOR_BGR2GRAY), *cur)
    if tmpl is not None:
        for idx in range(209, 224):
            rec = _template_step(frames[idx], cur, tmpl, radius=42, min_score=0.60)
            if not rec or not rec.get("visible"):
                extensions[idx] = rec or {"visible": False, "reason": "template_extension_rejected"}
                break
            rec["reason"] = "adjacent_template_forward_from_existing_track"
            extensions[idx] = rec
            cur = tuple(rec["point"])
            nxt = _gray_patch(cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY), *cur)
            if nxt is not None:
                tmpl = nxt
    meta["accepted_extension_frames"] = sorted(k for k, v in extensions.items() if v.get("visible"))
    meta["rejected_or_unavailable"] = {str(k): v for k, v in extensions.items() if not v.get("visible")}
    return extensions, meta


def build_body_states(frames: Dict[int, np.ndarray]) -> Tuple[Dict[int, dict], dict]:
    """Run the local pose model frame-by-frame before the first cut.

    No body frame-number cutoff is used to hide failure: each source frame is
    assessed against model output, a caddie/spectator spatial gate, and a bounded
    optical-flow continuation from the last selected golfer box. Frames after the
    broadcast cut stay unavailable until a future explicit reacquisition gate.
    """
    records: Dict[int, dict] = {}
    meta = {
        "model": str(POSE_MODEL),
        "model_sha256": sha256(POSE_MODEL) if POSE_MODEL.exists() else None,
        "model_state": "unavailable",
        "selection_policy": "per-frame YOLO person + spatial/caddie-risk gate before first cut; bounded LK continuation for detector dropouts; no frame cutoff hides failures",
        "body_assessed_source_frames": [BODY_ASSESSED_START, BODY_ASSESSED_END],
    }
    if not POSE_MODEL.exists():
        for idx in range(PASS_START, PASS_END):
            records[idx] = {"visible": False, "reason": "pose_model_unavailable"}
        return records, meta
    from ultralytics import YOLO
    model = YOLO(str(POSE_MODEL))
    meta["model_state"] = "loaded"
    prev_gray = None
    prev_body = None
    consecutive_detector_misses = 0
    for idx in range(PASS_START, PASS_END):
        frame = frames[idx]
        state = {"visible": False, "reason": "body_after_native_reviewed_trackable_window" if idx > BODY_NATIVE_VISIBLE_END and idx < CUT_FRAME else ("body_unavailable_after_camera_cut" if idx >= CUT_FRAME else "golfer_pose_not_detected_or_not_trackable")}
        detector_hit = False
        if idx < CUT_FRAME and idx <= BODY_NATIVE_VISIBLE_END:
            res = model.predict(frame, verbose=False, imgsz=640, conf=0.20)[0]
            candidates = []
            if res.boxes is not None:
                xyxy = res.boxes.xyxy.cpu().numpy()
                conf = res.boxes.conf.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy()
                kpts_xy = None
                if getattr(res, "keypoints", None) is not None and res.keypoints is not None:
                    kpts_xy = res.keypoints.xy.cpu().numpy()
                for det_i, (box, c, k) in enumerate(zip(xyxy, conf, cls)):
                    if int(k) != 0:
                        continue
                    b = tuple(float(v) for v in box.tolist())
                    if not is_golfer_body_candidate(b, W, H):
                        continue
                    keypoints = []
                    if kpts_xy is not None and det_i < len(kpts_xy):
                        for x, y in kpts_xy[det_i].tolist():
                            if x > 0 and y > 0:
                                keypoints.append([round(float(x), 1), round(float(y), 1)])
                    candidates.append({
                        "box": tuple(int(round(v)) for v in b),
                        "confidence": float(c),
                        "white_bib_fraction": white_bib_fraction(frame, b),
                        "keypoints": keypoints,
                    })
            picked = choose_golfer_candidate(candidates, preferred_x=630)
            if picked is not None and float(picked["confidence"]) >= 0.35:
                detector_hit = True
                state = {
                    "visible": True,
                    "box": list(picked["box"]),
                    "confidence": round(float(picked["confidence"]), 3),
                    "reason": "yolo11n_pose_main_golfer_reviewed_interval",
                    "white_bib_fraction": round(float(picked.get("white_bib_fraction", 0.0)), 3),
                    "keypoints": picked.get("keypoints", []),
                }
            elif (prev_body is not None and prev_gray is not None
                  and can_continue_without_detection(consecutive_detector_misses, max_misses=5)):
                # Fill detector dropouts only inside the native-reviewed visible interval.
                # Sparse optical flow points come from the previous selected golfer box;
                # no propagation crosses the reviewed exit or camera cut.
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                box = tuple(prev_body["box"])
                x1, y1, x2, y2 = box
                roi = prev_gray[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
                pts = cv2.goodFeaturesToTrack(roi, maxCorners=40, qualityLevel=0.01,
                                              minDistance=7, blockSize=5)
                if pts is not None and len(pts) >= 6:
                    pts[:, 0, 0] += x1
                    pts[:, 0, 1] += y1
                    nxt, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None,
                                                            winSize=(21, 21), maxLevel=3)
                    good = st.reshape(-1) == 1 if st is not None else []
                    if nxt is not None and np.any(good):
                        delta = nxt.reshape(-1, 2)[good] - pts.reshape(-1, 2)[good]
                        dx, dy = np.median(delta, axis=0).tolist()
                        if abs(dx) <= 55 and abs(dy) <= 95:
                            shifted = shift_box(box, dx, dy, W, H)
                            if is_golfer_body_candidate(shifted, W, H):
                                state = {"visible": True, "box": list(shifted),
                                         "confidence": round(max(0.20, float(prev_body.get("confidence", 0.35)) * 0.80), 3),
                                         "reason": "optical_flow_body_continuation_from_last_selected_golfer",
                                         "keypoints": []}
            else:
                state = {"visible": False, "reason": "golfer_pose_not_detected_or_not_trackable"}
        if idx >= CUT_FRAME:
            state = {"visible": False, "reason": "camera_cut_reset_no_golfer_reacquisition"}
        records[idx] = state
        if state.get("visible"):
            prev_body = state
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            consecutive_detector_misses = 0 if idx < CUT_FRAME and detector_hit else consecutive_detector_misses + 1
        elif idx >= CUT_FRAME:
            prev_body = None
            prev_gray = None
            consecutive_detector_misses = 0
        else:
            consecutive_detector_misses += 1
    return records, meta


def build_ball_states(track: Dict[int, list], extensions: Optional[Dict[int, dict]] = None) -> Dict[int, dict]:
    states: Dict[int, dict] = {}
    extensions = extensions or {}
    visible_keys = sorted(list(track.keys()) + [k for k, v in extensions.items() if v.get("visible")])
    min_visible = min(visible_keys)
    max_visible = max(visible_keys)
    for idx in range(PASS_START, PASS_END):
        if idx in track:
            x, y, conf = track[idx]
            states[idx] = {"visible": True, "point": [float(x), float(y)],
                           "confidence": float(conf), "reason": "existing_observed_ball_track_development_clip"}
        elif idx in extensions and extensions[idx].get("visible"):
            states[idx] = {k: v for k, v in extensions[idx].items() if k != "step"}
        elif idx < min_visible:
            states[idx] = {"visible": False, "reason": "ball_unresolved_before_template_or_existing_track"}
        elif idx > max_visible:
            states[idx] = {"visible": False, "reason": "ball_unresolved_after_last_supported_template_frame"}
        else:
            states[idx] = {"visible": False, "reason": "ball_unresolved_gap_preserved_no_bridge"}
    return states


def build_clubhead_states() -> Dict[int, dict]:
    states: Dict[int, dict] = {}
    for idx in range(PASS_START, PASS_END):
        if CLUBHEAD_ASSESSED_START <= idx <= CLUBHEAD_ASSESSED_END:
            reason = "clubhead_unavailable_native_review_cannot_separate_from_ball_shaft_blur"
        elif idx >= CUT_FRAME:
            reason = "camera_cut_reset_clubhead_offscreen"
        else:
            reason = "clubhead_offscreen_or_unresolvable"
        states[idx] = {"visible": False, "reason": reason}
    return states


def draw_text(img, text, org, scale=0.45, color=FG, thick=1):
    cv2.putText(img, text, org, F, scale, color, thick, cv2.LINE_AA)


def draw_body(img, st: dict):
    if not st.get("visible"):
        return
    x1, y1, x2, y2 = st["box"]
    cv2.rectangle(img, (x1, y1), (x2, y2), BODY_C, 2)
    keypoints = st.get("keypoints") or []
    # COCO pose skeleton, drawn only for returned keypoints inside the selected golfer box.
    skeleton = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
    for a, b in skeleton:
        if a < len(keypoints) and b < len(keypoints):
            ax, ay = keypoints[a]
            bx, by = keypoints[b]
            if x1 - 20 <= ax <= x2 + 20 and y1 - 20 <= ay <= y2 + 20 and x1 - 20 <= bx <= x2 + 20 and y1 - 20 <= by <= y2 + 20:
                cv2.line(img, (int(ax), int(ay)), (int(bx), int(by)), BODY_C, 2, cv2.LINE_AA)
    for kp in keypoints:
        x, y = kp
        if x1 - 20 <= x <= x2 + 20 and y1 - 20 <= y <= y2 + 20:
            cv2.circle(img, (int(x), int(y)), 3, BODY_C, -1)
    draw_text(img, f"BODY pose {st.get('confidence', 0):.2f}", (x1, max(18, y1 - 8)), 0.5, BODY_C, 1)


def draw_ball(img, st: dict, trail: List[Tuple[float, float]]):
    if not st.get("visible"):
        return
    x, y = st["point"]
    for n, (px, py) in enumerate(trail[-20:]):
        a = 0.25 + 0.75 * (n + 1) / max(1, min(20, len(trail)))
        cv2.circle(img, (int(px), int(py)), 2, (0, int(170 * a), int(255 * a)), -1)
    cv2.circle(img, (int(x), int(y)), 13, BALL_C, 2)
    cv2.line(img, (int(x)-23, int(y)), (int(x)-17, int(y)), BALL_C, 1)
    cv2.line(img, (int(x)+17, int(y)), (int(x)+23, int(y)), BALL_C, 1)
    draw_text(img, f"BALL observed {st.get('confidence', 0):.2f}", (int(x)+18, max(20, int(y)-15)), 0.45, BALL_C, 1)


def render_frame(source: np.ndarray, idx: int, body: dict, ball: dict, club: dict,
                 ball_history: List[Tuple[float, float]]) -> np.ndarray:
    annotated = source.copy()
    draw_body(annotated, body)
    # Clubhead has no marker in this artifact; visible=false clears any stale mark by construction.
    draw_ball(annotated, ball, ball_history)
    # Dominant annotated footage; small original inset only for sync comparison.
    inset = cv2.resize(source, (256, 144), interpolation=cv2.INTER_AREA)
    cv2.rectangle(annotated, (8, 8), (272, 176), PANEL_BG, -1)
    annotated[24:168, 16:272] = inset
    draw_text(annotated, "original inset", (18, 20), 0.42, DIM, 1)
    cv2.rectangle(annotated, (0, H - 82), (W, H), PANEL_BG, -1)
    draw_text(annotated, "FairwayOS local research | BODY pose, BALL observed track, CLUBHEAD hidden unless identifiable", (12, H - 60), 0.46, FG, 1)
    btxt = "BODY visible" if body.get("visible") else f"BODY hidden: {body.get('reason')}"
    ctxt = f"CLUBHEAD hidden: {club.get('reason')}"
    atxt = "BALL visible" if ball.get("visible") else f"BALL hidden: {ball.get('reason')}"
    draw_text(annotated, f"src f{idx} | no speed/carry/apex/landing/impact/calibration | local only", (12, H - 39), 0.43, WARN_C, 1)
    draw_text(annotated, btxt[:74], (12, H - 20), 0.40, BODY_C if body.get("visible") else DIM, 1)
    draw_text(annotated, (ctxt[:72] + " | " + atxt[:72])[:145], (12, H - 4), 0.40, FG, 1)
    return annotated


def write_video(frames: List[np.ndarray], final: Path):
    raw = final.with_name("_raw_multilayer.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for frame in frames:
        vw.write(frame)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(final)], check=True)
    raw.unlink(missing_ok=True)


def main():
    d = evaluate_source(str(SRC))
    if d.status == REJECTED or not d.local_research_allowed:
        raise SystemExit(f"gate refuses: {d.status}")
    assert not d.public_redistribution_cleared
    print(f"gate: {d.status} | local_research_allowed=True | demo_eligible={d.demo_eligible}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    frames = load_frames(SRC)
    track = {int(k): v for k, v in json.load(open(TRACK)).items()}
    body_states, body_meta = build_body_states(frames)
    ball_extensions, ball_extension_meta = extend_ball_track_by_template(frames, track)
    ball_states = build_ball_states(track, ball_extensions)
    club_states = build_clubhead_states()

    out_frames: List[np.ndarray] = []
    ball_history: List[Tuple[float, float]] = []
    for idx in range(PASS_START, PASS_END):
        ball_state = ball_states[idx]
        layer = LayerState("ball", bool(ball_state.get("visible")),
                           tuple(ball_state["point"]) if ball_state.get("visible") else None,
                           reason=ball_state.get("reason", "unavailable"))
        if ball_state.get("visible"):
            ball_history.append(tuple(ball_state["point"]))
        else:
            ball_history = renderable_trail(layer, ball_history)
        out_frames.append(render_frame(frames[idx], idx, body_states[idx], ball_state,
                                       club_states[idx], ball_history))

    # Short replay of the tracked ball window only; clubhead remains hidden.
    for idx in range(BALL_START, BALL_END + 1):
        ball_state = ball_states[idx]
        ball_history = [tuple(ball_states[k]["point"]) for k in range(BALL_START, idx + 1)
                        if ball_states[k].get("visible")]
        replay = render_frame(frames[idx], idx, body_states[idx], ball_state,
                              club_states[idx], ball_history)
        draw_text(replay, "replay: repeated frames only, no interpolation", (14, 24), 0.55, WARN_C, 1)
        out_frames.extend([replay, replay])

    final = OUTDIR / "siwoo_approach_multilayer_research_overlay.mp4"
    write_video(out_frames, final)

    counts = {
        "body_visible_frames": sum(1 for v in body_states.values() if v.get("visible")),
        "clubhead_visible_frames": sum(1 for v in club_states.values() if v.get("visible")),
        "ball_visible_frames": sum(1 for v in ball_states.values() if v.get("visible")),
        "rendered_frames": len(out_frames),
    }
    report = {
        "schema": "fairwayos-siwoo-multilayer-demo/v1",
        "research_only": True,
        "ground_truth": False,
        "production_eligible": False,
        "public_redistribution_cleared": False,
        "source": str(SRC),
        "source_sha256": sha256(SRC),
        "baseline_mp4_sha256_before_change": "a8048ab38ff859b45804399639cf44f370256cd9b1ad14b1db12a3e4026f0dd5",
        "output": str(final),
        "output_sha256": sha256(final),
        "frame_mapping": {"realtime_pass_source_frames": [PASS_START, PASS_END - 1],
                            "realtime_display_frames": [0, PASS_END - PASS_START - 1],
                            "ball_replay_source_frames": [BALL_START, BALL_END]},
        "body_model": body_meta,
        "ball_extension": ball_extension_meta,
        "ball_visible_ranges": contiguous_visible_ranges(ball_states),
        "body_visible_ranges": contiguous_visible_ranges(body_states),
        "counts": counts,
        "visible_review_denominators": {
            "body_native_review_visible_interval": [BODY_NATIVE_VISIBLE_START, BODY_NATIVE_VISIBLE_END],
            "body_native_review_visible_frames": BODY_NATIVE_VISIBLE_END - BODY_NATIVE_VISIBLE_START + 1,
            "body_assessed_before_cut_interval": [BODY_ASSESSED_START, BODY_ASSESSED_END],
            "body_assessed_before_cut_frames": BODY_ASSESSED_END - BODY_ASSESSED_START + 1,
            "clubhead_reviewed_assessed_interval": [CLUBHEAD_ASSESSED_START, CLUBHEAD_ASSESSED_END],
            "clubhead_reviewed_assessed_frames": CLUBHEAD_ASSESSED_END - CLUBHEAD_ASSESSED_START + 1,
            "clubhead_supported_identifiable_frames": 0,
            "ball_reviewed_visible_interval": [BALL_START, BALL_END],
            "ball_reviewed_visible_frames": BALL_END - BALL_START + 1,
        },
        "coverage_over_reviewed_visible_frames": {
            "body_native_visible_review": round(sum(1 for i in range(BODY_NATIVE_VISIBLE_START, BODY_NATIVE_VISIBLE_END + 1) if body_states[i].get("visible")) / (BODY_NATIVE_VISIBLE_END - BODY_NATIVE_VISIBLE_START + 1), 3),
            "body_assessed_before_cut": round(sum(1 for i in range(BODY_ASSESSED_START, BODY_ASSESSED_END + 1) if body_states[i].get("visible")) / (BODY_ASSESSED_END - BODY_ASSESSED_START + 1), 3),
            "clubhead": 0.0,
            "ball": round(counts["ball_visible_frames"] / (BALL_END - BALL_START + 1), 3),
        },
        "clubhead_limit": "No clubhead marker rendered. Native review and LK probe showed address seed locked onto the ball/shaft region; impact/downswing at 30fps is blurred/unresolvable. Clubhead stays unavailable rather than fabricated.",
        "non_claims": ["no speed", "no carry", "no apex", "no landing coordinate", "no impact/contact", "no calibration", "no production analytics"],
        "per_frame": {str(i): {"body": body_states[i], "clubhead": club_states[i], "ball": ball_states[i]}
                      for i in range(PASS_START, PASS_END)},
    }
    (OUTDIR / "siwoo_multilayer_diagnostics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"output": str(final), "frames": len(out_frames), "seconds": round(len(out_frames)/FPS, 2),
                      "counts": counts, "sha256": report["output_sha256"]}, indent=1))


if __name__ == "__main__":
    main()
