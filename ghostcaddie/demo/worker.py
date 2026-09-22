"""Out-of-process demo worker: decode, pose, track, render, cards.

Run as ``<cv interpreter> -m ghostcaddie.demo.worker`` with a JSON spec on stdin;
one JSON document comes back on stdout. This is the ONLY demo module (with
``render``) that imports cv2/numpy, which is what keeps the whole output contract,
the config validation and the report testable on a bare interpreter.

What it does, in two decodes of the bounded interval:

  pass 1  every frame of the interval: MoveNet on the operator ROI (when a model
          and an ROI exist), plus template/optical-flow tracking for each seeded
          target. Per-frame records only; no image is held after its frame is done.
  pass 2  every frame again: overlay and hand the frame straight to ffmpeg, so a
          long clip is never assembled in memory. Keyframe cards are taken from
          this same pass.

Nothing here decides what a measurement *means* -- gaps, losses, corrections and
interpolation are decided by :mod:`ghostcaddie.demo.tracking`, which is pure and
tested without a video. The worker only measures, renders, and reports failures.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import cv2
import numpy as np

from . import report as report_mod
from . import render as render_mod
from . import render_showcase as render_showcase_mod
from .showcase import interpolate_roi
from .tracking import (RawMeasurement, apply_manual_segments, manual_segments_only,
                       resolve_track)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .tracking import TargetTrack

# Undecoded protection: a template match must beat this before it is called a
# match, and a frame-to-frame step above the configured bound is refused.
LK_AGREEMENT_FRACTION = 0.35
LK_MIN_TRACKED = 0.5
#: A patch this flat carries no appearance information. TM_CCOEFF_NORMED reports a
#: near-perfect match for a constant template anywhere in a constant window, so a
#: flat template is refused outright rather than allowed to jump the search window.
MIN_TEXTURE_STD = 6.0


def _patch_texture(patch) -> float:
    """Grey-level standard deviation of a patch: how much there is to match on."""
    if patch is None or patch.size == 0:
        return 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    return float(gray.std())


# --------------------------------------------------------------------------
# decoding

def _open(video: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open the source: {video}")
    return cap


def _seek(cap: cv2.VideoCapture, frame: int) -> None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))


def _frame(cap: cv2.VideoCapture, frame: int) -> Optional[np.ndarray]:
    _seek(cap, frame)
    ok, img = cap.read()
    return img if ok else None


def _forward_leg(cap: cv2.VideoCapture, start: int, stop: int):
    """Sequential frames [start, stop) -- one decode, no per-frame seeking."""
    _seek(cap, start)
    frame = start
    while frame < stop:
        ok, img = cap.read()
        if not ok:
            return
        yield frame, img
        frame += 1


def _backward_frames(cap: cv2.VideoCapture, start: int, stop: int):
    """Descending frames (start, stop] -- used only for backward tracking.

    This seeks per frame, which is slower than sequential decoding; it is only
    reached when a config asks for backward or both.
    """
    for frame in range(start, stop, -1):
        img = _frame(cap, frame)
        if img is None:
            return
        yield frame, img


# --------------------------------------------------------------------------
# measurement

def _patch(img: np.ndarray, x: float, y: float, size: int) -> Optional[np.ndarray]:
    half = size // 2
    h, w = img.shape[:2]
    x0, y0 = int(round(x)) - half, int(round(y)) - half
    x1, y1 = x0 + size, y0 + size
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return img[y0:y1, x0:x1].copy()


def _match(img: np.ndarray, previous: Tuple[float, float], template: np.ndarray,
           options) -> Tuple[Optional[Tuple[float, float]], Optional[float], str]:
    """Template match inside a bounded window around the previous point."""
    h, w = img.shape[:2]
    size = template.shape[0]
    half = size // 2
    radius = int(options.search_radius)
    px, py = int(round(previous[0])), int(round(previous[1]))
    x0, y0 = px - half - radius, py - half - radius
    x1, y1 = px + half + radius + 1, py + half + radius + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None, None, "search_window_clipped_by_frame_edge"
    window = img[y0:y1, x0:x1]
    if window.shape[0] < size or window.shape[1] < size:
        return None, None, "search_window_smaller_than_template"
    if _patch_texture(window) < MIN_TEXTURE_STD:
        return None, None, "search_window_is_flat_low_texture"
    try:
        result = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    except cv2.error as exc:  # pragma: no cover - depends on the cv2 build
        return None, None, f"match_template_failed:{type(exc).__name__}"
    _, max_value, _, max_loc = cv2.minMaxLoc(result)
    point = (float(x0 + max_loc[0] + half), float(y0 + max_loc[1] + half))
    if float(max_value) < float(options.min_score):
        return None, float(max_value), f"score_below_min ({max_value:.2f})"
    step = math.hypot(point[0] - previous[0], point[1] - previous[1])
    if step > float(options.max_jump_px):
        return None, float(max_value), f"step_exceeds_max_jump ({step:.0f}px)"
    return point, float(max_value), ""


def _lk_agrees(previous_img: np.ndarray, img: np.ndarray, previous: Tuple[float, float],
               candidate: Tuple[float, float], template_size: int) -> Tuple[bool, str]:
    """Does optical flow agree with the template match? Only agreement is reported."""
    half = template_size // 2
    grid = [(previous[0] - half + template_size * fx, previous[1] - half + template_size * fy)
            for fx in (0.25, 0.5, 0.75) for fy in (0.25, 0.5, 0.75)]
    pts = np.array([[p[0], p[1]] for p in grid], dtype=np.float32).reshape(-1, 1, 2)
    prev_gray = cv2.cvtColor(previous_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        moved = np.zeros_like(pts)
        status_arr = np.zeros((len(grid), 1), dtype=np.uint8)
        moved, status_arr, _err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, pts, moved, winSize=(21, 21), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        status = status_arr
    except cv2.error:
        return False, "lk_failed"
    if moved is None or status is None:
        return False, "lk_failed"
    good = status.reshape(-1) == 1
    if good.sum() < LK_MIN_TRACKED * len(grid):
        return False, "lk_too_few_tracked_points"
    deltas = (moved.reshape(-1, 2) - pts.reshape(-1, 2))[good]
    median = np.median(deltas, axis=0)
    predicted = (previous[0] + float(median[0]), previous[1] + float(median[1]))
    tolerance = max(3.0, LK_AGREEMENT_FRACTION * template_size)
    if math.hypot(candidate[0] - predicted[0], candidate[1] - predicted[1]) > tolerance:
        return False, "lk_disagrees_with_template_match"
    return True, ""


def measure_target(cap: cv2.VideoCapture, name: str, options, seed, corrections,
                   start_frame: int, end_frame: int) -> Dict[int, RawMeasurement]:
    """Track one target from its seed, restarting only at operator anchors."""
    raw: Dict[int, RawMeasurement] = {}
    anchors = {int(seed.frame): (float(seed.x), float(seed.y), "seed")}
    for correction in corrections or []:
        anchors[int(correction.frame)] = (float(correction.x), float(correction.y),
                                          "correction")
    if options.mode == "manual":
        return raw

    directions = ("forward", "backward") if options.direction == "both" else \
        (options.direction,)
    stop_frame = (None if options.track_until_frame is None
                  else int(options.track_until_frame))
    # automatic_until_frame bounds the MEASUREMENT, not the track: past it there is
    # nothing to look for, so the legs stop early and no template match on a stretch
    # the operator took the tracker off can ever reach the output.
    automatic_stop = (None if getattr(options, "automatic_until_frame", None) is None
                      else int(options.automatic_until_frame))

    for direction in directions:
        if direction == "forward":
            legs = sorted(f for f in anchors if f < end_frame)
            for anchor_frame in legs:
                boundary = min([f for f in anchors if f > anchor_frame] or [end_frame + 1])
                if stop_frame is not None:
                    boundary = min(boundary, stop_frame + 1)
                if automatic_stop is not None:
                    boundary = min(boundary, automatic_stop + 1)
                if boundary <= anchor_frame + 1:
                    continue
                _track_leg(cap, name, options, raw, anchors, anchor_frame,
                           _forward_leg(cap, anchor_frame + 1, boundary), forward=True)
        else:
            first = _frame(cap, int(seed.frame))
            if first is None:
                continue
            _track_leg(cap, name, options, raw, anchors, int(seed.frame),
                       _backward_frames(cap, int(seed.frame) - 1, start_frame - 1),
                       forward=False, seed_image=first)
    return raw


def _track_leg(cap: cv2.VideoCapture, name: str, options, raw: Dict[int, RawMeasurement],
               anchors: Dict[int, Tuple[float, float, str]], anchor_frame: int,
               stream, *, forward: bool, seed_image: Optional[np.ndarray] = None) -> None:
    """Measure frames from one anchor (the seed or a correction) until a loss.

    A correction anchor inside the stream restarts the leg; after a declared loss
    the leg stops measuring and does not re-acquire until the next anchor.
    """
    anchor_image = seed_image if seed_image is not None else _frame(cap, anchor_frame)
    if anchor_image is None:
        return
    direction = "forward" if forward else "backward"
    previous_image = anchor_image
    previous = (anchors[anchor_frame][0], anchors[anchor_frame][1])
    template = _patch(previous_image, previous[0], previous[1], options.template_size)
    if template is None:
        # A seed too close to the frame edge cannot carry a centred patch: every
        # frame of the leg is reported unmeasured rather than seeded at a shifted
        # position. Path and coverage stay truthful.
        for frame, _img in stream:
            if frame in anchors:
                previous = (anchors[frame][0], anchors[frame][1])
                continue
            raw[frame] = RawMeasurement(frame, False, method="template",
                                        reason="seed_patch_not_fully_inside_frame",
                                        direction=direction)
        return
    if _patch_texture(template) < MIN_TEXTURE_STD:
        # A flat seed patch has no appearance to follow; TM_CCOEFF_NORMED would
        # happily "match" it anywhere, so the track is refused instead of drifting.
        for frame, _img in stream:
            if frame in anchors:
                previous = (anchors[frame][0], anchors[frame][1])
                continue
            raw[frame] = RawMeasurement(frame, False, method="template",
                                        reason="seed_patch_is_flat_low_texture",
                                        direction=direction)
        return
    lost = False
    misses = 0
    since_refresh = 0
    for frame, img in stream:
        if frame in anchors:
            previous = (anchors[frame][0], anchors[frame][1])
            new_template = _patch(img, previous[0], previous[1], options.template_size)
            if new_template is not None:
                template = new_template
            previous_image = img
            lost, misses, since_refresh = False, 0, 0
            continue
        if lost:
            raw[frame] = RawMeasurement(frame, False, method="template",
                                        reason="track_declared_lost",
                                        direction=direction)
            continue
        point, score, reason = _match(img, previous, template, options)
        method = "template"
        if point is not None and options.lk_refine:
            agrees, why = _lk_agrees(previous_image, img, previous, point,
                                     options.template_size)
            if agrees:
                method = "template+lk"
            else:
                point, reason = None, why
        if point is None:
            raw[frame] = RawMeasurement(frame, False, method=method, reason=reason or
                                        "no_bounded_match", direction=direction)
            misses += 1
            previous_image = img
            if misses >= int(options.loss_frames):
                lost = True
            continue
        raw[frame] = RawMeasurement(frame, True, x=point[0], y=point[1],
                                    score=score or 0.0, method=method,
                                    direction=direction)
        previous = point
        previous_image = img
        misses = 0
        since_refresh += 1
        if options.template_refresh_frames and \
                since_refresh >= int(options.template_refresh_frames) and \
                (score or 0.0) >= float(options.template_refresh_min_score):
            refreshed = _patch(img, point[0], point[1], options.template_size)
            if refreshed is not None and _patch_texture(refreshed) >= MIN_TEXTURE_STD:
                template = refreshed
            since_refresh = 0


# --------------------------------------------------------------------------
# pose

def body_pass(cap: cv2.VideoCapture, spec: dict) -> dict:
    """MoveNet on the operator ROI for every frame of the interval.

    No ROI means no golfer identity, and the demo refuses to attribute a pose to
    the golfer on the strength of a whole-frame skeleton: the layer is AMBIGUOUS
    and no pose is reported. A frame whose pose is not confidently the subject is
    missing -- it is never interpolated into a skeleton that was not measured.
    """
    from ghostcaddie.video.canonical_worker import MoveNet, attribute, SCORE_THRESHOLD, \
        MIN_VISIBLE_KEYPOINTS

    start_frame = int(spec["interval"]["start_frame"])
    end_frame = int(spec["interval"]["end_frame"])
    roi = spec.get("roi")
    roi_keyframes = spec.get("roi_keyframes") or []
    model = spec.get("body", {})
    identity: Dict[str, object] = {"identity_method": "none", "roi_xywh": None}
    limitations: List[str] = []
    net = None
    model_error = ""
    if model.get("model_found"):
        try:
            net = MoveNet(model["model_path"])
        except Exception as exc:
            model_error = f"{type(exc).__name__}: {exc}"

    # Who the subject is, recorded BEFORE the model is consulted: the identity here is
    # the operator's ROI (or ROI keyframes), and that is true whether or not a body
    # model was available on this run. Leaving it as "none" when the model is missing
    # would misdescribe the operator's input as absent.
    def effective_box(frame: int) -> List[int]:
        """The crop box for one frame: the interpolated operator box when the ROI
        moves, otherwise the static ROI. Recorded on every per-frame record so the run
        says which crop each measurement came from."""
        if roi_keyframes:
            box = interpolate_roi(roi_keyframes, frame)
            if box:
                return [int(v) for v in box]
        return [int(v) for v in roi] if roi else []

    per_frame: List[dict] = []
    if roi:
        identity["roi_xywh"] = [int(v) for v in roi]
        identity["identity_method"] = "operator_roi"
    if roi and roi_keyframes:
        identity["identity_method"] = "operator_roi_keyframes"
        identity["roi_keyframes"] = [dict(key) for key in roi_keyframes]
        identity["roi_interpolation"] = ("linear in position and size between the "
                                         "operator's boxes, held constant outside them")
        identity["roi_author"] = spec.get("roi_author") or "operator"
    if not roi:
        limitations.append(
            "No operator ROI was supplied. The demo does not attribute a whole-frame "
            "single-pose skeleton to the golfer, so the body layer is ambiguous and "
            "no pose, no skeleton and no 2D angle is reported. Draw an ROI in "
            "`demo-setup` to enable body analysis.")
        for frame in range(start_frame, end_frame + 1):
            per_frame.append({"source_frame": frame, "available": False,
                              "reason": "no_operator_roi",
                              "visible_keypoint_count": 0, "keypoints": [],
                              "roi_xywh": None})
        return {"state": "ambiguous", "method": "none", "identity": identity,
                "model_error": model_error, "limitations": limitations,
                "per_frame": per_frame, "roi_xywh": None}
    if net is None:
        limitations.append(
            "The body model is not available, so no pose could be measured. Body "
            "metrics are unavailable rather than estimated.")
        for frame in range(start_frame, end_frame + 1):
            per_frame.append({"source_frame": frame, "available": False,
                              "reason": model_error or "body_model_not_found",
                              "visible_keypoint_count": 0, "keypoints": [],
                              "roi_xywh": effective_box(frame)})
        return {"state": "unavailable", "method": "none", "identity": identity,
                "model_error": model_error or "body model not found",
                "limitations": limitations, "per_frame": per_frame,
                "roi_xywh": list(roi)}

    x, y, rw, rh = (int(v) for v in roi)
    if rw <= 0 or rh <= 0 or x < 0 or y < 0:
        raise RuntimeError(f"roi {list(roi)} is not usable")
    if roi_keyframes:
        limitations.append(
            "The golfer region is an operator crop that MOVES: the operator placed "
            "boxes at specific frames and the crop is interpolated between them. It is "
            "crop geometry from the operator, not a detected or tracked subject, and "
            "each frame's effective box is recorded.")
    limitations.append(
        "The golfer region was supplied by the operator. It is user input, not "
        "automatic perception, and the pose inside it is attributed to the golfer on "
        "the operator's authority.")
    limitations.append(
        "MoveNet SinglePose Lightning returns exactly one skeleton per crop; it is a "
        "single-pose model, not a person detector.")
    if model_error:
        limitations.append(f"the body runtime reported: {model_error}")

    for frame, img in _forward_leg(cap, start_frame, end_frame + 1):
        bx, by, bw, bh = effective_box(frame)
        crop = img[by:by + bh, bx:bx + bw]
        if crop.size == 0:
            per_frame.append({"source_frame": frame, "available": False,
                              "reason": "empty_roi_crop", "visible_keypoint_count": 0,
                              "keypoints": [], "roi_xywh": [bx, by, bw, bh]})
            continue
        try:
            points, transform = net.infer(crop, (bx, by))
        except Exception as exc:
            per_frame.append({"source_frame": frame, "available": False,
                              "reason": f"pose_inference_failed:{type(exc).__name__}",
                              "visible_keypoint_count": 0, "keypoints": [],
                              "roi_xywh": [bx, by, bw, bh]})
            continue
        ok, why = attribute(points, [bx, by, bw, bh])
        visible = sum(1 for p in points if p["visible"])
        per_frame.append({"source_frame": frame, "available": bool(ok),
                          "reason": why, "visible_keypoint_count": visible,
                          "keypoints": points, "transform": transform,
                          "roi_xywh": [bx, by, bw, bh]})
    posed = [r for r in per_frame if r["available"]]
    coverage = len(posed) / float(len(per_frame)) if per_frame else 0.0
    if coverage >= 0.80:
        state = "supported"
    elif coverage >= 0.40:
        state = "partial"
    else:
        state = "ambiguous"
        limitations.append(
            f"a confident pose was measured on only {len(posed)} of {len(per_frame)} "
            f"frames ({coverage:.0%}); that is too little to treat the pose as the "
            f"golfer, so no 2D quantity is reported from it")
        # The aggregate decision governs the whole artifact family: if the layer is
        # declared ambiguous, no per-frame pose may still feed an angle, a metric or
        # an overlay. The measurements are kept for diagnostics and marked as
        # measured-but-not-attributed.
        reason = (f"aggregate_state_ambiguous: a pose was measured on "
                  f"{len(posed)}/{len(per_frame)} frames ({coverage:.0%}), below the "
                  f"share needed to attribute any of them to the golfer")
        for record in per_frame:
            if record.get("available"):
                record["measured"] = True
                record["available"] = False
                record["reason"] = reason
    return {"state": state,
            "method": ("movenet_singlepose_lightning_v3_on_operator_roi"
                       if posed else "none"),
            "identity": identity, "model_error": model_error,
            "limitations": limitations, "per_frame": per_frame,
            "roi_xywh": [x, y, rw, rh], "coverage_fraction": round(coverage, 4)}


# --------------------------------------------------------------------------
# rendering

def _ffmpeg_path() -> str:
    return shutil.which("ffmpeg") or ""


def render_pass(cap: cv2.VideoCapture, spec: dict, body: dict,
                tracks: Dict[str, "TargetTrack"],
                phases: dict) -> Tuple[dict, Dict[str, np.ndarray]]:
    """Stream the interval to ffmpeg, drawing per frame. No clip in memory."""
    from .tracking import trail_window

    out_path = spec["render"]["path"]
    start_frame = int(spec["interval"]["start_frame"])
    end_frame = int(spec["interval"]["end_frame"])
    native_w, native_h = (int(v) for v in spec["native_wh"])
    max_width = spec["render"].get("max_width")
    scale = 1.0
    if max_width and native_w > int(max_width):
        scale = float(max_width) / float(native_w)
    out_w = int(round(native_w * scale))
    out_h = int(round(native_h * scale))
    trimmed = False
    if out_w % 2 or out_h % 2:
        out_w -= out_w % 2
        out_h -= out_h % 2
        trimmed = True
    # The no-go check runs on the RENDERED picture, from the points that will be
    # drawn on it -- the ball and club action, not just the golfer's ROI.
    plan = render_mod.plan_panel_scaled(spec.get("roi"), scale, out_w, out_h,
                                        points_xy=_drawn_points(tracks))
    gutter = int(plan["gutter_w"]) if plan.get("external") else 0
    if gutter and (out_w + gutter) % 2:
        # H.264 yuv420p wants even dimensions: shrink the gutter, not the picture, so
        # the panel plan and the encoder agree on the canvas width.
        gutter -= 1
        plan = dict(plan, gutter_w=gutter)
    canvas_w = out_w + gutter

    ffmpeg = _ffmpeg_path()
    if not ffmpeg:
        return {"written": False, "frames": 0, "fps": float(spec["native_fps"]),
                "wh": [out_w, out_h], "pix_fmt": "", "faststart": False,
                "bytes": 0, "encoder": "",
                "reason": ("ffmpeg was not found on PATH. The demo writes H.264 / "
                           "yuv420p / faststart output with ffmpeg and does not "
                           "substitute another container silently.")}, {}

    fps = float(spec["native_fps"])
    if canvas_w % 2:
        canvas_w -= 1
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-s", f"{canvas_w}x{out_h}", "-r", f"{fps}",
           "-i", "-", "-an", "-c:v", "libx264", "-preset",
           str(spec["render"].get("preset", "medium")), "-crf",
           str(spec["render"].get("crf", 20)), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", out_path]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    body_by_frame = {int(r["source_frame"]): r for r in body["per_frame"]}
    point_by_frame = {name: {int(p.source_frame): p for p in track.points}
                      for name, track in tracks.items()}
    phase_frames = {name: phase.get("frame")
                    for name, phase in (phases.get("phases") or {}).items()}
    frame_to_phase = {frame: name for name, frame in phase_frames.items()
                      if frame is not None}
    side = str(plan["side"])
    cards: Dict[str, np.ndarray] = {}
    # The panel's static content: a small phase badge, the three body angles and the
    # tempo. Raw counts, window bounds and per-target tallies are NOT here -- they
    # belong in demo_report.md and debug/, where they can be read without competing
    # with the swing (and without needing a wide dark panel over the picture).
    tempo_line = _tempo_line(phases)
    rendered = 0
    log_path = os.path.join(spec["debug_dir"], "ffmpeg.log")
    os.makedirs(spec["debug_dir"], exist_ok=True)

    def draw_overlays(image, frame: int, record: Optional[dict]) -> None:
        """Skeleton, trails and markers, in PICTURE coordinates, nothing else."""
        if record is not None and record.get("available"):
            render_mod.draw_skeleton(image, record["keypoints"], scale)
        for name, track in tracks.items():
            if not track.available:
                continue
            point = point_by_frame[name].get(frame)
            if point is None or not point.drawn:
                continue
            # One rule decides the trail: the current segment up to this frame. A
            # loss empties it, a correction starts a new one.
            window = trail_window(track.points, frame, render_mod.TRAIL_POINTS)
            render_mod.draw_trail(
                image, [p.to_dict() for p in window],
                render_mod.BALL if name == "ball" else render_mod.CLUB, scale, 3)
            render_mod.draw_target_marker(image, point.to_dict(), name, scale)

    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log, stderr=log)
        stdin = proc.stdin
        if stdin is None:  # pragma: no cover - Popen with PIPE always provides it
            raise RuntimeError("ffmpeg stdin pipe could not be opened")
        try:
            for frame, img in _forward_leg(cap, start_frame, end_frame + 1):
                picture = (img if scale == 1.0
                           else cv2.resize(img, (out_w, out_h),
                                           interpolation=cv2.INTER_AREA))
                record = body_by_frame.get(frame)
                # Overlays are drawn on the PICTURE in picture coordinates, so they
                # can be copied into a keyframe card without any video chrome. The
                # copy is taken BEFORE the canvas exists, which is what keeps the
                # card clean.
                draw_overlays(picture, frame, record)
                phase_name = frame_to_phase.get(frame)
                if phase_name is not None and phase_name not in cards:
                    cards[phase_name] = picture.copy()
                phase_text = _phase_badge(phase_name, phase_frames, phases, frame)
                analytics = _analytics_lines(record, tempo_line)
                annotated = render_mod.compose_annotated(
                    picture, plan=plan, phase_text=phase_text,
                    frame_text=f"source frame {frame} / {end_frame}",
                    analytics_lines=analytics, operator_note="1",
                    overlays=lambda target: draw_overlays(target, frame, record))
                stdin.write(np.ascontiguousarray(
                    annotated if annotated.shape[0] == out_h
                    and annotated.shape[1] == canvas_w
                    else cv2.resize(annotated, (canvas_w, out_h),
                                    interpolation=cv2.INTER_AREA)).tobytes())
                rendered += 1
        except BrokenPipeError:
            pass
        finally:
            try:
                stdin.close()
            except Exception:
                pass
            returncode = proc.wait()
    ok = returncode == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 0
    result = {
        "written": bool(ok),
        "frames": rendered,
        "fps": fps,
        "wh": [canvas_w, out_h],
        "picture_wh": [out_w, out_h],
        "source_wh": [native_w, native_h],
        "scale": round(scale, 6),
        "dimensions_trimmed_to_even": trimmed,
        "encoder": "libx264 (ffmpeg)",
        "reason": "" if ok else (f"ffmpeg exited {returncode}; see debug/ffmpeg.log"),
        "bytes": os.path.getsize(out_path) if os.path.isfile(out_path) else 0,
        "pipeline": "rawvideo bgr24 | libx264 -pix_fmt yuv420p -movflags +faststart",
        "panel_side": side,
        "panel_external_gutter_px": gutter,
        "panel_placement": "external_gutter" if plan.get("external") else "overlay",
        # Measured on the CHOSEN panel region: an external gutter is outside the
        # picture, so it covers nothing, and these stay 0 rather than reporting the
        # overlap of a candidate the plan rejected.
        "panel_overlap_px2": plan.get("overlap_px2"),
        "panel_overlap_body_px2": plan.get("body_overlap_px2"),
        "panel_overlap_targets_px2": plan.get("targets_overlap_px2"),
        "panel_covers_the_golfer": bool(plan.get("body_overlap_px2")),
        "panel_covers_the_action": bool(plan.get("targets_overlap_px2")),
        "panel_clearance_required_px": plan.get("clearance_required_px"),
        "panel_no_go_box_body_xywh": list(spec["roi"]) if spec.get("roi") else None,
        "panel_no_go_box_targets_xywh": plan.get("targets_box_xywh"),
        "panel_rejected_overlay_candidates": plan.get("overlay_refusals") or {},
        "ffmpeg_log": log_path,
    }
    return result, cards


def render_keyframe_cards(spec: dict, cards: Dict[str, np.ndarray],
                          phases: dict) -> dict:
    """Write one card per phase; a phase with no frame gets an explicit UNAVAILABLE."""
    keyframes_dir = spec["keyframes_dir"]
    os.makedirs(keyframes_dir, exist_ok=True)
    written: List[dict] = []
    unavailable: List[dict] = []
    panels: Dict[str, np.ndarray] = {}
    order = list(phases.get("cards") or [])
    index = 0
    for card in order:
        name = card["phase"]
        index += 1
        title = f"{name.replace('_', ' ').upper()} - frame " + (
            str(card["frame"]) if card["frame"] is not None else "-")
        label = report_mod.PHASE_LABEL_TEXT.get(card["label"], str(card["label"]))
        path = os.path.join(keyframes_dir, f"{index:02d}_{name}.png")
        if not card.get("renderable"):
            image = render_mod.draw_unavailable_card(
                title=title,
                reason=f"{card.get('note', '')} (method: {card.get('method', 'none')})")
            ok = render_mod.write_png(path, image)
            record = {"phase": name, "path": path, "written": ok,
                      "label": card["label"], "frame": None}
            (written if ok else unavailable).append(record)
            if ok:
                panels[name] = image
            continue
        source = cards.get(name)
        if source is None:
            reason = ("the annotated frame for this phase was not captured during the "
                      "render pass")
            image = render_mod.draw_unavailable_card(title=title, reason=reason)
            ok = render_mod.write_png(path, image)
            record = {"phase": name, "path": path, "written": ok,
                      "label": card["label"], "frame": card["frame"]}
            (written if ok else unavailable).append(record)
            if ok:
                panels[name] = image
            continue
        angles = card.get("angles") or {}
        bits = []
        if angles.get("shoulder_tilt_deg") is not None:
            bits.append(f"shoulder tilt {angles['shoulder_tilt_deg']:.1f} deg")
        if angles.get("hip_tilt_deg") is not None:
            bits.append(f"hip tilt {angles['hip_tilt_deg']:.1f} deg")
        lead = angles.get("lead_side")
        knee = (angles.get("knee_interior_deg") or {}).get(lead) if lead else None
        if knee is not None:
            bits.append(f"{lead} knee interior {knee:.1f} deg")
        if not bits:
            bits.append("no 2D angle was measurable at this frame")
        notes = [card.get("note", "")[:150], report_mod.VIDEO_CAVEAT]
        image = render_mod.draw_card(source, title=title, label=label,
                                     lines=[" | ".join(bits)] + notes)
        ok = render_mod.write_png(path, image)
        record = {"phase": name, "path": path, "written": ok, "label": card["label"],
                  "frame": card["frame"], "angles": angles}
        (written if ok else unavailable).append(record)
        if ok:
            panels[name] = image

    sheet_path = spec["contact_sheet"]
    priority = [cards.get("primary_cards") or ["address", "top", "impact", "finish"]][0]
    sheet_cards = [panels[name] for name in priority if name in panels]
    sheet_written = False
    if sheet_cards:
        sheet_written = render_mod.write_png(
            sheet_path, render_mod.contact_sheet(sheet_cards, columns=2))
    return {"cards": written, "unavailable": unavailable,
            "contact_sheet": sheet_path if sheet_written else "",
            "contact_sheet_written": bool(sheet_written),
            "contact_sheet_panels": [n for n in priority if n in panels]}


def _phase_badge(phase_name: Optional[str], phase_frames: Dict[str, Optional[int]],
                 phases: dict, frame: int) -> str:
    if phase_name is not None:
        card = next((c for c in phases.get("cards", []) if c["phase"] == phase_name),
                    None)
        label = report_mod.PHASE_LABEL_TEXT.get(
            (card or {}).get("label", "unavailable"), "UNAVAILABLE")
        return f"PHASE  {phase_name.replace('_', ' ')}  -  {label}"
    past = [(f, n) for n, f in phase_frames.items() if f is not None and f < frame]
    if not past:
        return "PHASE  before first marker"
    f, name = max(past)
    return f"PHASE  after {name.replace('_', ' ')} (f{f})"


def _drawn_points(tracks: Dict[str, "TargetTrack"]) -> List[Tuple[float, float]]:
    """Every target point that will be drawn, in native source pixels.

    Used only to keep the analytics panel away from the action. Missing and lost
    points are not included: they draw nothing, so they are not a region the panel
    could cover.
    """
    points: List[Tuple[float, float]] = []
    for track in tracks.values():
        if not track.available:
            continue
        for point in track.points:
            if point.drawn and point.x is not None and point.y is not None:
                points.append((float(point.x), float(point.y)))
    return points


def _tempo_line(phases: dict) -> str:
    """One compact tempo line for the panel, or nothing when there is no tempo."""
    tempo = (phases or {}).get("tempo") or {}
    if tempo.get("state") != "supported":
        return ""
    seconds = tempo.get("seconds_source_timeline") or {}
    return (f"tempo  back {tempo['backswing_frames']}f"
            + (f" ({seconds['backswing']:.2f}s)" if seconds.get("backswing") is not None
               else "")
            + f" / down {tempo['downswing_frames']}f"
            + (f" ({seconds['downswing']:.2f}s)" if seconds.get("downswing") is not None
               else "")
            + f"  = {tempo['backswing_downswing_ratio']:.2f}:1")


def _analytics_lines(record: Optional[dict], tempo_line: str = "") -> List[str]:
    """The panel's few lines: body state, three angles, tempo. No raw counts.

    Every per-frame tally that used to be drawn here (observed / interpolated /
    missing, the window bounds) is a debug quantity: it lives in demo_report.md and
    debug/, and printing it on every frame is what made the panel wide enough to
    cover the ball flight.
    """
    lines = ["BODY   " + ("measured" if record and record.get("available")
                          else "no pose at this frame")]
    angles = {}
    if record and record.get("available"):
        from .analytics import body_angles_for_record
        angles = body_angles_for_record(record)

    def fmt(value, unit="deg"):
        return "n/a" if value is None else f"{value:.1f} {unit}"

    lines.append(f"  shoulder tilt   {fmt(angles.get('shoulder_tilt_deg'))}")
    lines.append(f"  hip tilt        {fmt(angles.get('hip_tilt_deg'))}")
    lead = angles.get("lead_side")
    knee = (angles.get("knee_interior_deg") or {}).get(lead) if lead else None
    lines.append(f"  {lead or 'lead'} knee int.  {fmt(knee)}")
    if tempo_line:
        lines.append(tempo_line)
    return lines


# --------------------------------------------------------------------------
# showcase rendering (opt-in)
#
# The demo pass above puts its chrome in a gutter beside the picture. The showcase
# pass instead fills the frame with the picture and draws translucent cards ON it,
# which is the look social playback needs and the reason the two are separate code
# paths rather than one renderer full of branches. Everything about what is drawn
# lives in ``demo.showcase`` (stdlib) and ``demo.render_showcase`` (cv2/PIL); this
# section only supplies the per-frame measurements those modules expect.

def _showcase_stage(spec: dict) -> dict:
    """Output size, scale and layout for a showcase render.

    ``showcase.max_width`` may upscale as well as downscale: the footage carries no
    extra detail, but the CHROME is rendered at the output size, so a 2x render of a
    640px source gives cards that stay sharp on a phone while the picture is honest
    about what it is.
    """
    from . import showcase as showcase_mod

    options = spec.get("showcase") or {}
    native_w, native_h = (int(v) for v in spec["native_wh"])
    scale = 1.0
    target = options.get("max_width")
    if target:
        scale = min(3.0, max(0.1, float(target) / float(native_w)))
    out_w = int(round(native_w * scale))
    out_h = int(round(native_h * scale))
    trimmed = False
    if out_w % 2 or out_h % 2:
        out_w -= out_w % 2
        out_h -= out_h % 2
        trimmed = True
        scale = out_w / float(native_w)
    metrics = tuple(options.get("metrics") or showcase_mod.DEFAULT_METRICS)
    layout = showcase_mod.plan_layout(out_w, out_h, metrics,
                                      max_row_width=options.get("card_row_max_width"))
    return {"scale": scale, "out_w": out_w, "out_h": out_h, "trimmed": trimmed,
            "metrics": metrics, "layout": layout, "options": options,
            # Display-pixel factor for line weights and markers: separate from the
            # coordinate scale, which is 0.5 for a 4K source rendered at 1080p.
            "style": render_showcase_mod.style_scale(out_h)}


def _frame_metric_series(body_records: Sequence[dict], metrics: Sequence[str], *,
                         handedness: str, wrist_reference: Optional[dict] = None
                         ) -> Dict[str, Dict[int, Optional[float]]]:
    """The measured per-frame series behind each per-frame card.

    One series per metric for the whole run, because the card's stability rule is about
    consecutive frames and cannot be decided one frame at a time.
    """
    from . import showcase as showcase_mod
    from .analytics import body_angles_for_record

    wanted = [metric for metric in metrics
              if (showcase_mod.METRICS.get(metric) or {}).get("source") == "frame"]
    series: Dict[str, Dict[int, Optional[float]]] = {metric: {} for metric in wanted}
    if not wanted:
        return series
    for record in body_records:
        frame = int(record["source_frame"])
        angles = (body_angles_for_record(record, handedness=handedness)
                  if record.get("available") else {})
        values = showcase_mod.frame_values(
            angles, handedness=handedness,
            wrist_reference=(wrist_reference or {}).get("y"))
        for metric in wanted:
            series[metric][frame] = values.get(metric)
    return series


def _card_display_series(body_records: Sequence[dict], metrics: Sequence[str], *,
                         handedness: str, wrist_reference: Optional[dict],
                         min_run: int) -> dict:
    """What each per-frame card SHOWS, after the bounded display hysteresis.

    The measured series is returned untouched next to it, so the report, the analysis
    JSON and the provenance can always be checked against what a card displayed.
    """
    from . import showcase as showcase_mod

    measured = _frame_metric_series(body_records, metrics, handedness=handedness,
                                    wrist_reference=wrist_reference)
    display: Dict[str, Dict[int, dict]] = {}
    records: Dict[str, dict] = {}
    for metric, values in measured.items():
        series = showcase_mod.debounce_series(values, min_run=min_run)
        display[metric] = series
        records[metric] = {
            **showcase_mod.debounce_record(series, min_run=min_run),
            "measured_frames": len(values),
            "measured_available_frames": sum(1 for value in values.values()
                                             if value is not None),
        }
    return {"display": display, "measured": measured, "records": records}


def _showcase_card_payload(metrics, record: Optional[dict], clip: Dict[str, Any], *,
                           handedness: str,
                           wrist_ref: Optional[Dict[str, Any]] = None,
                           display: Optional[Dict[str, Dict[int, dict]]] = None
                           ) -> List[dict]:
    """The cards for one frame: per-frame angles, clip values where applicable.

    ``display`` is the debounced per-metric table (see ``_card_display_series``). When a
    metric has no entry for this frame the raw per-frame value is used, so this function
    is still correct without a series (for example on a single-frame still).
    """
    from . import showcase as showcase_mod
    from .analytics import body_angles_for_record

    angles: Dict[str, Any] = {}
    if record is not None and record.get("available"):
        angles = body_angles_for_record(record, handedness=handedness)
    frame_metrics = showcase_mod.frame_values(
        angles, handedness=handedness,
        wrist_reference=(wrist_ref or {}).get("y"))
    frame = int(record["source_frame"]) if record is not None else None
    cards: List[dict] = []
    for metric in metrics:
        spec = showcase_mod.METRICS.get(metric) or {}
        held = False
        if spec.get("source") == "clip":
            value = (clip.get(metric) or {}).get("value")
            state = (clip.get(metric) or {}).get("state")
            sub = (clip.get(metric) or {}).get("sub") or showcase_mod.format_sub(
                metric, unicode_ok=True)
        else:
            entry = ((display or {}).get(metric) or {}).get(frame) \
                if frame is not None else None
            if entry is not None:
                value = entry.get("value")
                held = bool(entry.get("held"))
                state = ("held" if held and value is not None
                         else "held_unavailable" if held
                         else "supported" if value is not None else "unavailable")
            else:
                value = frame_metrics.get(metric)
                state = "supported" if value is not None else "unavailable"
            sub = showcase_mod.format_sub(metric, unicode_ok=True)
        cards.append({
            "metric": metric, "label": showcase_mod.format_label(metric),
            "value": showcase_mod.format_value(metric, value, unicode_ok=True),
            "value_ascii": showcase_mod.format_value(metric, value, unicode_ok=False),
            "sub": sub, "available": bool(value is not None),
            "state": state, "held": held,
            "raw_value": None if value is None else round(float(value), 3),
        })
    return cards


def _render_pass_showcase(cap: cv2.VideoCapture, spec: dict, body: dict,
                          tracks: Dict[str, "TargetTrack"], phases: dict) -> tuple:
    """Stream the interval to ffmpeg with the showcase chrome, capturing stills."""
    from . import render_showcase as sc_render
    from . import showcase as showcase_mod
    from .tracking import trail_window

    stage = _showcase_stage(spec)
    # hand_rise is measured from the first frame in which both wrists were
    # resolvable; the reference is computed once here and reported with the run.
    stage["wrist_reference"] = showcase_mod.wrist_reference(body["per_frame"])
    layout = stage["layout"]
    scale = stage["scale"]
    out_w, out_h = stage["out_w"], stage["out_h"]
    options = stage["options"]
    out_path = spec["render"]["path"]
    start_frame = int(spec["interval"]["start_frame"])
    end_frame = int(spec["interval"]["end_frame"])
    native_w, native_h = (int(v) for v in spec["native_wh"])
    fps = float(spec["native_fps"])
    ffmpeg = _ffmpeg_path()
    empty_stills: Dict[str, Any] = {}
    if not ffmpeg:
        return ({"written": False, "frames": 0, "fps": fps, "wh": [out_w, out_h],
                 "picture_wh": [out_w, out_h], "source_wh": [native_w, native_h],
                 "scale": round(scale, 6), "trimmed_to_even": stage["trimmed"],
                 "reason": ("ffmpeg was not found on PATH. The showcase writes "
                            "H.264 / yuv420p / faststart output with ffmpeg."),
                 "style": "showcase"}, empty_stills)

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", f"{fps}", "-i", "-",
           "-an", "-c:v", "libx264", "-preset", str(options.get("preset", "slow")),
           "-crf", str(options.get("crf", 18)), "-pix_fmt", "yuv420p", "-movflags",
           "+faststart", out_path]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    body_by_frame = {int(r["source_frame"]): r for r in body["per_frame"]}
    point_by_frame = {name: {int(p.source_frame): p for p in track.points}
                      for name, track in tracks.items()}
    markers = showcase_mod.phase_markers(phases)
    impact_frame = markers.get("IMPACT")
    handedness = str(spec.get("handedness", "right"))
    # Clip-level cards (peak head movement, sway) are read from the same pure
    # analytic the report prints, computed here on the same per-frame records, so a
    # card and the metric table cannot disagree about a peak.
    from .analytics import body_metrics as _body_metrics
    frames_possible = (int(spec["interval"]["end_frame"])
                       - int(spec["interval"]["start_frame"]) + 1)
    clip_analytics = {"body_2d": _body_metrics(
        body["per_frame"], frames_possible=frames_possible,
        pixel_aspect=float(spec.get("pixel_aspect", 1.0)), handedness=handedness)}
    clip = showcase_mod.clip_values(clip_analytics, phases, handedness=handedness)
    trail_points = {"ball": int(options.get("ball_trail_points") or
                                showcase_mod.BALL_TRAIL_POINTS),
                    "clubhead": int(options.get("club_trail_points") or
                                    showcase_mod.CLUB_TRAIL_POINTS)}
    accent_on = bool(options.get("impact_accent", True))
    accent_span = int(options.get("impact_accent_span_frames") or 3)
    debounce_frames = int(options.get("card_debounce_frames")
                          or showcase_mod.DEFAULT_CARD_DEBOUNCE_FRAMES)
    # The card's display series is computed once, from the whole run: the stability rule
    # is about CONSECUTIVE frames, so it cannot be decided frame by frame in the encode
    # loop. The measured series is kept beside it and reported.
    card_display = _card_display_series(body["per_frame"], stage["metrics"],
                                        handedness=handedness,
                                        wrist_reference=stage.get("wrist_reference"),
                                        min_run=debounce_frames)
    stage["card_display"] = card_display
    engine = sc_render.TextEngine()
    stills: Dict[str, Any] = {}
    phase_frames = {name: info.get("frame")
                    for name, info in (phases.get("phases") or {}).items()}
    frame_to_phases: Dict[int, List[str]] = {}
    for name, frame in phase_frames.items():
        if frame is not None:
            frame_to_phases.setdefault(int(frame), []).append(name)
    rendered = 0
    overlay_elements = 0
    log_path = os.path.join(spec["debug_dir"], "ffmpeg.log")
    os.makedirs(spec["debug_dir"], exist_ok=True)

    # --- presentation pace: a uniform stride plus any operator holds.
    stride = int((spec.get("render") or {}).get("stride") or 1)
    source_frames = showcase_mod.stride_frames(start_frame, end_frame, stride)
    holds = list(options.get("holds") or [])
    output_plan = showcase_mod.expand_holds(source_frames, holds, fps)
    slots: Dict[int, int] = {}
    for entry in output_plan:
        slots[entry["source_frame"]] = slots.get(entry["source_frame"], 0) + 1
    style = float(stage["style"])

    def scaled_keypoints(record: Optional[dict]) -> List[dict]:
        """Move the pose into OUTPUT pixels; the renderer never scales coordinates."""
        if record is None or not record.get("available"):
            return []
        return [dict(p, x=p["x"] * scale, y=p["y"] * scale)
                for p in (record.get("keypoints") or [])]

    def scaled_points(name: str, frame: int) -> List[dict]:
        track = tracks[name]
        window = trail_window(track.points, frame, trail_points.get(name, 10))
        # ``trail: segments_only`` (opt-in, per target) keeps the trail inside the
        # operator's own anchored segments: behind a stationary ball the trail would
        # otherwise trace the camera's pan, not the ball.
        window = manual_segments_only(track, window)
        return [dict(p.to_dict(), x=p.x * scale, y=p.y * scale) for p in window
                if p.x is not None and p.y is not None]

    def overlay_draw(layer, mask, record, frame):
        nonlocal overlay_elements
        drawn = 0
        keypoints = scaled_keypoints(record)
        if keypoints:
            drawn += sc_render.draw_skeleton(layer, mask, keypoints, style)
        for name, track in tracks.items():
            if not track.available:
                continue
            point = point_by_frame[name].get(frame)
            if point is None or not point.drawn:
                continue
            payload = scaled_points(name, frame)
            payload.append(dict(point.to_dict(), x=point.x * scale, y=point.y * scale))
            if name == "clubhead":
                sc_render.draw_trail(layer, mask, payload, sc_render.CLUB_TRAIL,
                                     alpha=sc_render.CLUB_TRAIL_ALPHA,
                                     thickness=max(3, int(round(3.4 * style))), dash=5)
                sc_render.draw_club_marker(
                    layer, mask, dict(point.to_dict(), x=point.x * scale,
                                      y=point.y * scale), style)
            else:
                sc_render.draw_trail(layer, mask, payload, sc_render.BALL_TRAIL,
                                     alpha=sc_render.TRAIL_ALPHA,
                                     thickness=max(2, int(round(2.0 * style))), dash=4)
                sc_render.draw_ball_marker(
                    layer, mask, dict(point.to_dict(), x=point.x * scale,
                                      y=point.y * scale), style)
        if accent_on:
            strength = showcase_mod.impact_accent(frame, impact_frame,
                                                  half_span=accent_span)
            if strength > 0:
                anchor = None
                for name in ("clubhead", "ball"):
                    candidate = point_by_frame.get(name, {}).get(frame)
                    if candidate is not None and candidate.drawn:
                        anchor = dict(candidate.to_dict(), x=candidate.x * scale,
                                      y=candidate.y * scale)
                        break
                sc_render.draw_impact_accent(layer, mask, anchor, style, strength)
        overlay_elements = drawn
        return drawn

    def draw_slot(frame: int, img) -> Optional[np.ndarray]:
        """Draw one output frame from one decoded source frame. Returns the canvas."""
        picture = (img if abs(scale - 1.0) < 1e-9 else
                   cv2.resize(img, (out_w, out_h),
                              interpolation=(cv2.INTER_AREA if scale < 1.0
                                             else cv2.INTER_CUBIC)))
        record = body_by_frame.get(frame)
        cards = _showcase_card_payload(stage["metrics"], record, clip,
                                       handedness=handedness,
                                       wrist_ref=stage.get("wrist_reference"),
                                       display=(stage.get("card_display") or {}).get(
                                           "display"))
        chrome = {
            "brand": options.get("brand", showcase_mod.DEFAULT_BRAND),
            "brand_sub": options.get("brand_sub", showcase_mod.DEFAULT_BRAND_SUB),
            "phase": showcase_mod.current_phase(frame, markers),
            "cards": cards,
            "disclosure": options.get("disclosure", showcase_mod.DEFAULT_DISCLOSURE),
        }
        staged = sc_render.stage_overlays(
            picture, overlay_draw=lambda layer, mask: overlay_draw(layer, mask, record,
                                                                   frame))
        annotated, _drawing = sc_render.compose_chrome(staged, layout=layout,
                                                       chrome=chrome, engine=engine)
        for phase_name in frame_to_phases.get(frame, []):
            if phase_name not in stills:
                stills[phase_name] = {"staged": staged.copy(), "cards": cards,
                                      "frame": frame}
        return annotated

    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log, stderr=log)
        stdin = proc.stdin
        if stdin is None:  # pragma: no cover - Popen with PIPE always provides one
            raise RuntimeError("ffmpeg stdin pipe could not be opened")
        try:
            for frame, img in _forward_leg(cap, start_frame, end_frame + 1):
                count = slots.get(frame)
                if not count:
                    continue
                annotated = draw_slot(frame, img)
                if annotated is None:
                    continue
                payload = np.ascontiguousarray(annotated).tobytes()
                # A hold repeats the SAME drawn bytes: nothing new is measured or
                # drawn during the held frames, which is what makes the hold a replay
                # of a verified frame rather than a stretch of invented motion.
                for _ in range(count):
                    stdin.write(payload)
                    rendered += 1
        except BrokenPipeError:
            pass
        finally:
            try:
                stdin.close()
            except Exception:
                pass
            returncode = proc.wait()

    # --- stills for marked phases that the stride skipped.
    #
    # The stills must be the operator's own frames, not a neighbouring encoded frame,
    # so a phase that does not land on the stride grid is decoded separately and drawn
    # with exactly the same overlay code.
    extra_stills = 0
    for phase_frame, phase_names in sorted(frame_to_phases.items()):
        missing = [name for name in phase_names if name not in stills]
        if not missing:
            continue
        image = _frame(cap, phase_frame)
        if image is None:
            continue
        still_picture = (image if abs(scale - 1.0) < 1e-9 else
                         cv2.resize(image, (out_w, out_h),
                                    interpolation=(cv2.INTER_AREA if scale < 1.0
                                                   else cv2.INTER_CUBIC)))
        record = body_by_frame.get(phase_frame)
        staged = sc_render.stage_overlays(
            still_picture,
            overlay_draw=lambda layer, mask: overlay_draw(layer, mask, record,
                                                          phase_frame))
        cards = _showcase_card_payload(stage["metrics"], record, clip,
                                       handedness=handedness,
                                       wrist_ref=stage.get("wrist_reference"),
                                       display=(stage.get("card_display") or {}).get(
                                           "display"))
        for name in missing:
            stills[name] = {"staged": staged.copy(), "cards": cards,
                            "frame": phase_frame}
            extra_stills += 1

    ok = returncode == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 0
    result = {
        "written": bool(ok), "frames": rendered, "fps": fps, "wh": [out_w, out_h],
        "picture_wh": [out_w, out_h], "source_wh": [native_w, native_h],
        "scale": round(scale, 6), "dimensions_trimmed_to_even": stage["trimmed"],
        "encoder": "libx264 (ffmpeg)", "style": "showcase",
        "pipeline": "rawvideo bgr24 | libx264 -pix_fmt yuv420p -movflags +faststart",
        "reason": "" if ok else f"ffmpeg exited {returncode}; see debug/ffmpeg.log",
        "bytes": os.path.getsize(out_path) if os.path.isfile(out_path) else 0,
        "chrome_inside_picture": True, "added_width_px": 0,
        "chrome_coverage": showcase_mod.chrome_coverage(
            layout, roi=spec.get("roi_effective"),
            points_xy=_drawn_points(tracks), scale=scale),
        "layout": layout.to_dict(),
        "layout_problems": showcase_mod.layout_problems(layout),
        "text_engine": sc_render.font_report(engine),
        "card_metrics": list(stage["metrics"]),
        "card_values": {
            "debounce": (stage.get("card_display") or {}).get("records") or {},
            "rule": ("the card holds its previous state until a new state has held for "
                     "card_debounce_frames consecutive source frames. Held frames show "
                     "the last measured value (dimmed) or keep the placeholder; the "
                     "measured per-frame series is unchanged and is reported in "
                     "showcase_analysis.json."),
        },
        "manual_segments": {name: (track.manual_segments or None)
                            for name, track in tracks.items()},
        "wrist_reference": stage.get("wrist_reference"),
        "trail_points": trail_points,
        "style_scale": round(style, 4),
        "playback": {
            "stride": stride,
            "rule": ("the base sequence is source frame start + i*stride for i = 0..N-1; "
                     "a hold then repeats an ALREADY DRAWN frame in place, so from the "
                     "first hold onwards the output index is the base index plus the "
                     "repeats inserted before it. `output_to_source_frames` lists the "
                     "exact source frame of every output frame, and `holds` locates "
                     "each repeat in the output timeline."),
            "source_frames_encoded": len(source_frames),
            "sampled_source_frames": len(source_frames),
            "output_frames": len(output_plan),
            "output_to_source_frames": [entry["source_frame"] for entry in output_plan],
            "source_frame_span": [start_frame, end_frame],
            "source_fps_unchanged": fps,
            "presentation_speedup": float(stride),
            "output_seconds": round(len(output_plan) / fps, 3) if fps else None,
            "source_interval_seconds": round((end_frame - start_frame + 1) / fps, 3)
            if fps else None,
            "holds": showcase_mod.hold_records(source_frames, holds, fps),
            "extra_stills_from_off_grid_frames": extra_stills,
            "note": ("the stride and the holds change PRESENTATION pace only. Every "
                     "analysis value is still measured per source frame, and neither "
                     "the speed-up nor the source's own slow motion is treated as "
                     "measured action time."),
        },
        "impact_accent_frames": ([f for f in source_frames
                                  if showcase_mod.impact_accent(f, impact_frame,
                                                                half_span=accent_span)
                                  > 0]
                                 if accent_on and impact_frame is not None else []),
        "impact_accent_span_frames": (accent_span if accent_on and impact_frame
                                      is not None else None),
        "ffmpeg_log": log_path,
    }
    return result, stills


def _render_showcase_stills(spec: dict, stills: Dict[str, Any], phases: dict) -> dict:
    """Write ``showcase_<phase>.png`` for every marked phase, then the contact sheet.

    A still carries at most three cards: it is read in a feed, not paused on a
    timeline. The still is composed from the SAME staged frame the video showed (same
    skeleton, same markers, same trails) with its own layout, so a card can never
    disagree with the video it was taken from. A phase with no frame gets an explicit
    UNAVAILABLE card rather than a blank or a repeat of another phase.
    """
    from . import render_showcase as sc_render
    from . import showcase as showcase_mod
    from .paths import SHOWCASE_STILL_PHASES, showcase_still_path

    out_dir = os.path.dirname(spec["contact_sheet"])
    options = spec.get("showcase") or {}
    metrics = tuple(options.get("metrics") or showcase_mod.DEFAULT_METRICS)
    still_metrics = metrics[:3]
    written: List[dict] = []
    unavailable: List[dict] = []
    panels: Dict[str, np.ndarray] = {}
    #: What goes on the contact sheet: the same stills, composed without their own
    #: disclosure line because the sheet prints one for all four panels.
    panel_images: Dict[str, np.ndarray] = {}
    frames: Dict[str, Optional[int]] = {}
    card_lookup = {c["phase"]: c for c in (phases.get("cards") or [])}
    engine = sc_render.TextEngine()

    for card in (phases.get("cards") or []):
        name = card["phase"]
        entry = stills.get(name)
        if not card.get("renderable") or entry is None:
            continue
        staged = entry["staged"]
        height, width = staged.shape[:2]
        # The still honours the SAME card-row clearance as the video, so a still cannot
        # spread its cards over a mark in the footage that the video leaves visible.
        layout = showcase_mod.plan_layout(width, height, still_metrics,
                                          max_row_width=options.get(
                                              "card_row_max_width"))
        subset = _still_cards(entry["cards"], still_metrics, showcase_mod)
        label = report_mod.PHASE_LABEL_TEXT.get(str(card.get("label")), "UNAVAILABLE")
        # ONE sub-brand string everywhere: the video and the stills used to disagree
        # (`SWING ANALYSIS` vs `SWING ANALYSIS · OPERATOR-MARKED`), which is a difference
        # a viewer cannot explain. The phase is on the pill.
        chrome = {
            "brand": options.get("brand", showcase_mod.DEFAULT_BRAND),
            "brand_sub": options.get("brand_sub", showcase_mod.DEFAULT_BRAND_SUB),
            "phase": name.replace("_", "-").upper(),
            "cards": subset,
            "disclosure": options.get("disclosure", showcase_mod.DEFAULT_DISCLOSURE),
        }
        image, drawing = sc_render.compose_chrome(staged, layout=layout, chrome=chrome,
                                                  engine=engine, mode="still")
        path = str(showcase_still_path(out_dir, name))
        ok = sc_render.write_png(path, image)
        record = {"phase": name, "path": path, "written": bool(ok),
                  "label": card.get("label"), "frame": card.get("frame"),
                  "metrics": [c["metric"] for c in subset],
                  "unavailable_metrics": [c["metric"] for c in subset
                                          if not c.get("available")],
                  "held_metrics": [c["metric"] for c in subset if c.get("held")],
                  "layout_problems": drawing.get("layout_problems") or []}
        (written if ok else unavailable).append(record)
        if ok:
            panels[name] = image
            frames[name] = card.get("frame")
            # The sheet carries ONE disclosure line in its footer, so its panels are
            # composed without the per-panel line: same staged frame, same cards, same
            # markers, one line fewer.
            if chrome["disclosure"]:
                sheet_chrome = dict(chrome, disclosure="")
                sheet_panel, _ = sc_render.compose_chrome(staged, layout=layout,
                                                          chrome=sheet_chrome,
                                                          engine=engine, mode="still")
                panel_images[name] = sheet_panel
            else:
                panel_images[name] = image

    missing: List[dict] = []
    for phase in SHOWCASE_STILL_PHASES:
        if phase in panels:
            continue
        card = card_lookup.get(phase) or {}
        reason = str(card.get("note")
                     or "no operator keyframe was supplied for this phase")[:220]
        title = phase.replace("_", " ").upper()
        image = render_mod.draw_unavailable_card(
            title=f"{title} - UNAVAILABLE", reason=reason, card_width=640, height=360)
        path = str(showcase_still_path(out_dir, phase))
        ok = render_mod.write_png(path, image)
        missing.append({"phase": phase, "path": path, "written": bool(ok),
                        "reason": reason})
        if ok:
            panels[phase] = image

    sheet_path = spec["contact_sheet"]
    ordered = [(name, panels[name]) for name in SHOWCASE_STILL_PHASES
               if name in panels]
    sheet_written = False
    if ordered:
        # No frame indices and no per-run numbers on the graphic: those live in
        # showcase_report.md and showcase_provenance.json, and on the sheet they were
        # the debug clutter that crowded the header. The footer carries the one
        # disclosure line.
        sheet_panels: List[Tuple[str, Any]] = []
        for name, panel in ordered:
            chosen = panel_images.get(name)
            sheet_panels.append((name, panel if chosen is None else chosen))
        image = sc_render.contact_sheet(
            sheet_panels, columns=2, gap=14,
            title=(f"{options.get('brand', showcase_mod.DEFAULT_BRAND)}"
                   f"  \u00b7  "
                   f"{options.get('brand_sub', showcase_mod.DEFAULT_BRAND_SUB)}"),
            subtitle="",
            disclaimer=options.get("disclosure", showcase_mod.DEFAULT_DISCLOSURE))
        sheet_written = sc_render.write_png(sheet_path, image)
    return {"cards": written, "unavailable": unavailable, "missing": missing,
            "contact_sheet": sheet_path if sheet_written else "",
            "contact_sheet_written": bool(sheet_written),
            "contact_sheet_panels": [name for name, _ in ordered],
            "contact_sheet_panel_disclosure_dropped": bool(panel_images),
            "still_metrics": list(still_metrics),
            "video_metrics": list(metrics),
            "still_paths": [record["path"] for record in written],
            "still_frames": frames}


def _still_cards(cards: Sequence[dict], still_metrics: Sequence[str],
                 showcase_mod) -> List[dict]:
    """The first cards of the video, in the still's own order, placeholders filled.

    A still has room for fewer cards than the video, so the metrics it drops are
    simply not shown on it -- they are never merged into one card or relabelled.
    """
    by_metric = {c.get("metric"): c for c in cards}
    out: List[dict] = []
    for metric in still_metrics:
        card = by_metric.get(metric)
        if card is None:
            card = {"metric": metric, "label": showcase_mod.format_label(metric),
                    "value": showcase_mod.PLACEHOLDER, "sub": "", "available": False,
                    "state": "unavailable"}
        out.append(card)
    return out


# --------------------------------------------------------------------------
# entry point

def run(spec: dict) -> dict:
    from .analytics import body_angles_for_record
    from .phases import frame_seconds_factory, resolve_phases

    cap = _open(spec["video"])
    try:
        body = body_pass(cap, spec)
        tracks: Dict[str, dict] = {}
        track_objects: Dict[str, "TargetTrack"] = {}
        raw_dump: Dict[str, dict] = {}
        interval_frames = list(range(int(spec["interval"]["start_frame"]),
                                     int(spec["interval"]["end_frame"]) + 1))
        manual_blocks = spec.get("manual_segments") or {}
        for name in ("ball", "clubhead"):
            target_spec = spec.get("targets", {}).get(name) or {}
            seed = spec.get("seeds", {}).get(name)
            manual = manual_spec(manual_blocks.get(name))
            options = _options(target_spec.get("options"))
            corrections = spec.get("corrections", {}).get(name) or []
            if manual and manual.get("replace_automatic"):
                # The operator replaced this target's automatic track with their own
                # anchor segments, so there is nothing to measure: the segments ARE the
                # presentation. Nothing is drawn outside them, and the run says so.
                raw: Dict[int, RawMeasurement] = {}
                raw_dump[name] = {}
                track = resolve_track(
                    name, options=options, interval_frames=interval_frames, raw={},
                    seed=None, corrections=[],
                    interpolation_enabled=False, max_gap_frames=0)
            elif not seed:
                continue
            else:
                raw = measure_target(cap, name, options, _point(seed),
                                     [_point(c) for c in corrections],
                                     int(spec["interval"]["start_frame"]),
                                     int(spec["interval"]["end_frame"]))
                raw_dump[name] = {str(k): v.to_dict()
                                  for k, v in sorted(raw.items())}
                track = resolve_track(
                    name, options=options, interval_frames=interval_frames, raw=raw,
                    seed=_point(seed), corrections=[_point(c) for c in corrections],
                    interpolation_enabled=bool(spec["interpolation"]["enabled"]),
                    max_gap_frames=int(spec["interpolation"]["max_gap_frames"]))
            if manual:
                apply_manual_segments(track, manual=manual,
                                      interval_frames=interval_frames)
            track_objects[name] = track
            tracks[name] = track.to_dict()

        angles_fn = (lambda record: body_angles_for_record(
            record, pixel_aspect=float(spec.get("pixel_aspect", 1.0)),
            handedness=str(spec.get("handedness", "right"))))
        phase_config = spec.get("phases") or {}
        frame_times = phase_config.get("frame_times")
        phase_payload = resolve_phases(
            config_keyframes={k: int(v) for k, v in
                              (phase_config.get("keyframes") or {}).items()},
            interval_frames=interval_frames, body_records=body["per_frame"],
            tracks=track_objects, fps=float(phase_config.get("fps")
                                            or spec["native_fps"]),
            allow_approx=bool(phase_config.get("allow_approx", False)),
            body_angles_fn=angles_fn,
            frame_seconds=frame_seconds_factory(
                frame_times, int(phase_config.get("start_frame",
                                                  interval_frames[0]))))

        style = str((spec.get("render") or {}).get("style") or "demo")
        if style == "showcase":
            # The showcase pass captures its stills from the same staged frames it
            # encoded, so the stills are produced inside it rather than from a second
            # decode that could draw something slightly different.
            render, showcase_stills = _render_pass_showcase(cap, spec, body,
                                                            track_objects,
                                                            phase_payload)
            keyframes = _render_showcase_stills(spec, showcase_stills, phase_payload)
        else:
            render, cards = render_pass(cap, spec, body, track_objects, phase_payload)
            keyframes = render_keyframe_cards(spec, cards, phase_payload)
    finally:
        cap.release()
    return {
        "source_frames": {"start_frame": int(spec["interval"]["start_frame"]),
                          "end_frame": int(spec["interval"]["end_frame"]),
                          "count": int(spec["interval"]["end_frame"])
                          - int(spec["interval"]["start_frame"]) + 1},
        "body": body,
        "targets": tracks,
        "phases": phase_payload,
        "render": render,
        "keyframes": keyframes,
        "raw_measurements": raw_dump,
        "environment": _environment(),
    }


def _options(payload):
    """Target options come from the orchestrator as the resolved dataclass dict."""
    from .config import TargetOptions
    if payload is None:
        return TargetOptions()
    if isinstance(payload, TargetOptions):
        return payload
    fields = {k: v for k, v in payload.items() if k in TargetOptions.__dataclass_fields__}
    return TargetOptions(**fields)


def _point(payload):
    from .config import Point
    if isinstance(payload, Point):
        return payload
    return Point(frame=int(payload["frame"]), x=float(payload["x"]),
                 y=float(payload["y"]), note=str(payload.get("note", "")))


def manual_spec(payload: Optional[dict]) -> Dict[str, Any]:
    """Normalise one target's ``manual_segments`` spec block for the tracker.

    The orchestrator sends the resolved block, in which every anchor is a mapping. The
    ``[frame, x, y]`` list form the operator's own annotation files use is accepted here
    too, so a spec built straight from one of those files cannot be mis-read: both shapes
    end up as one.
    """
    if not payload:
        return {}
    out: Dict[str, Any] = {
        "author": str(payload.get("author") or ""),
        "replace_automatic": bool(payload.get("replace_automatic")),
        "trail": str(payload.get("trail") or "all"),
        "trail_from_frame": (None if payload.get("trail_from_frame") is None
                             else int(payload["trail_from_frame"])),
        "hide_frames": [[int(pair[0]), int(pair[1])]
                        for pair in (payload.get("hide_frames") or [])],
        "segments": [],
    }
    for segment in (payload.get("segments") or []):
        anchors: List[Dict[str, Any]] = []
        for item in (segment.get("anchors") or []):
            if isinstance(item, dict):
                anchors.append({"frame": int(item["frame"]), "x": float(item["x"]),
                                "y": float(item["y"]),
                                "note": str(item.get("note") or "")})
            else:
                values = list(item)
                anchors.append({"frame": int(values[0]), "x": float(values[1]),
                                "y": float(values[2]),
                                "note": str(values[3]) if len(values) > 3 else ""})
        out["segments"].append({
            "name": str(segment.get("name") or "segment"),
            "max_anchor_spacing_frames": int(segment.get("max_anchor_spacing_frames")
                                             or 1),
            "anchors": anchors,
            "note": str(segment.get("note") or ""),
        })
    return out


def _environment() -> dict:
    versions = {"cv2": cv2.__version__, "numpy": np.__version__}
    try:
        import ai_edge_litert
        versions["ai_edge_litert"] = getattr(ai_edge_litert, "__version__", "unknown")
    except Exception:
        versions["ai_edge_litert"] = "unavailable"
    return {"python": sys.version.split()[0], "versions": versions}


def main() -> None:
    spec = json.loads(sys.stdin.read())
    try:
        payload = run(spec)
    except Exception as exc:
        print(f"demo worker error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    json.dump(payload, sys.stdout)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
