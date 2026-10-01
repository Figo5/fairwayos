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
from typing import (Any, Dict, List, Mapping, Optional, Sequence, Tuple,
                    TYPE_CHECKING)

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

def _measure(cap: cv2.VideoCapture, spec: dict) -> tuple:
    """Body, targets and phases for ONE interval. Pure measurement, no rendering.

    Shared by the single-interval workflow and by every scene of a broadcast run, so a
    scene is measured by exactly the same code as a whole clip. There is no second
    tracker, no second angle function and no second phase resolver to drift -- and
    because ``spec["interval"]`` is the scene window, nothing can be tracked, seeded,
    interpolated or phase-resolved outside the frames the operator declared visible.
    """
    from .analytics import body_angles_for_record
    from .phases import frame_seconds_factory, resolve_phases

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
            # No seed and no operator segments: this target is absent from this
            # interval, and absent is what the run reports.
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
    return body, track_objects, tracks, phase_payload, raw_dump


def run(spec: dict) -> dict:
    cap = _open(spec["video"])
    try:
        style = str((spec.get("render") or {}).get("style") or "demo")
        if style == "broadcast":
            return _run_broadcast(cap, spec)
        body, track_objects, tracks, phase_payload, raw_dump = _measure(cap, spec)
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


# --------------------------------------------------------------------------
# broadcast: several independent scene windows in one whole-shot render

#: The display vocabulary for a scene's phase badge. The full-swing names are the
#: existing badge spellings, so the look is unchanged; a putt gets its own words and
#: therefore can never show TOP.
_BROADCAST_BADGES: Dict[str, Dict[str, str]] = {
    "full_swing": {"address": "ADDRESS", "takeaway": "TAKEAWAY", "top": "TOP",
                   "downswing": "DOWNSWING", "impact": "IMPACT",
                   "follow_through": "FOLLOW-THROUGH", "finish": "FINISH"},
    "chip": {"address": "ADDRESS", "takeaway": "TAKEAWAY", "top": "TOP",
             "downswing": "DOWNSWING", "impact": "IMPACT",
             "follow_through": "FOLLOW-THROUGH", "finish": "FINISH"},
    "pitch": {"address": "ADDRESS", "takeaway": "TAKEAWAY", "top": "TOP",
              "downswing": "DOWNSWING", "impact": "IMPACT",
              "follow_through": "FOLLOW-THROUGH", "finish": "FINISH"},
    "bunker": {"address": "ADDRESS", "takeaway": "TAKEAWAY", "top": "TOP",
               "downswing": "DOWNSWING", "impact": "IMPACT",
               "follow_through": "FOLLOW-THROUGH", "finish": "FINISH"},
    "unknown": {"address": "ADDRESS", "takeaway": "TAKEAWAY", "top": "TOP",
                "downswing": "DOWNSWING", "impact": "IMPACT",
                "follow_through": "FOLLOW-THROUGH", "finish": "FINISH"},
    "putt": {"address": "ADDRESS", "backstroke": "BACKSTROKE",
             "forward_stroke": "FORWARD STROKE", "contact": "CONTACT",
             "finish": "FINISH"},
}
#: canonical analytics slot -> the scene's own vocabulary word.
_BROADCAST_SLOT_TO_NAME: Dict[str, str] = {
    "address": "address", "takeaway": "takeaway", "top": "top",
    "downswing": "downswing", "impact": "impact", "follow_through": "follow_through",
    "finish": "finish",
}
#: The vocabulary word that owns each canonical slot, per stroke type. The analytics
#: run on canonical slots (so tempo and the sequence check keep one definition); every
#: artifact the operator reads is emitted in the scene's own words. A putt therefore
#: never emits the string TOP: there is no vocabulary entry that maps to it.
_BROADCAST_SLOT_VOCAB: Dict[str, Dict[str, str]] = {
    "full_swing": dict(_BROADCAST_SLOT_TO_NAME),
    "chip": dict(_BROADCAST_SLOT_TO_NAME),
    "pitch": dict(_BROADCAST_SLOT_TO_NAME),
    "bunker": dict(_BROADCAST_SLOT_TO_NAME),
    "unknown": dict(_BROADCAST_SLOT_TO_NAME),
    "putt": {"address": "address", "takeaway": "backstroke",
             "downswing": "forward_stroke", "impact": "contact", "finish": "finish"},
}


def _broadcast_scene_phases(phases: Mapping[str, Any], *, stroke_type: str,
                            vocabulary: Sequence[str], keyframes: Mapping[str, int]
                            ) -> dict:
    """Re-key a scene's phase payload into the scene's own vocabulary.

    The canonical slots stay inside the measurement; what is published is the words
    this stroke actually has. A putt's payload therefore contains ``contact`` and
    never ``impact``, ``top`` or ``downswing`` -- not in a key, not in a label and not
    in a reason string.
    """
    records = (phases or {}).get("phases") or {}
    slot_vocab = _BROADCAST_SLOT_VOCAB.get(stroke_type) or _BROADCAST_SLOT_TO_NAME
    out: Dict[str, Any] = {}
    for slot, name in slot_vocab.items():
        if name not in vocabulary:
            continue
        record = records.get(slot)
        if record is None:
            continue
        entry = dict(record)
        entry["name"] = name
        out[name] = entry
    return {"phases": out, "vocabulary": list(vocabulary),
            "stroke_type": stroke_type,
            "tempo": _broadcast_tempo(stroke_type, vocabulary, keyframes,
                                      canonical=(phases or {}).get("tempo") or {}),
            "source_seconds_basis": (phases or {}).get("source_seconds_basis"),
            "notes": [n for n in ((phases or {}).get("notes") or [])]}


def _broadcast_tempo(stroke_type: str, vocabulary: Sequence[str],
                     keyframes: Mapping[str, int],
                     canonical: Optional[Mapping[str, Any]] = None) -> dict:
    """Tempo in the stroke's own terms, or an honest refusal.

    * **full swing** — the canonical record the existing resolver computed, unchanged.
      Delegating is the whole point: the pilot showed that record working, and a second
      definition here could only disagree with it.
    * **putt** — measured from the operator's own marks in source frames:
      ``back = forward_stroke - backstroke`` (the hands travelling away and back to the
      starting point) and ``through = contact - forward_stroke`` (the strike travel).
      Both are unavailable, never estimated, unless all three marks are present.

    Seconds are source-timeline positions throughout and are never measured action time.
    """
    def frame(name: str) -> Optional[int]:
        value = keyframes.get(name)
        return None if value is None else int(value)

    if stroke_type == "putt":
        backstroke = frame("backstroke")
        forward = frame("forward_stroke")
        contact = frame("contact")
        if backstroke is None or forward is None or contact is None:
            return {"state": "unavailable", "label": "PUTT TEMPO",
                    "unit": ":1",
                    "reason": ("a putt's tempo needs the operator's backstroke, "
                               "forward_stroke and contact marks; without all three "
                               "there is no back : through pair to report")}
        back_frames = forward - backstroke
        through_frames = contact - forward
        if back_frames <= 0 or through_frames <= 0:
            return {"state": "unavailable", "label": "PUTT TEMPO", "unit": ":1",
                    "reason": ("the operator's marks are not in stroke order "
                               f"(backstroke {backstroke}, forward_stroke {forward}, "
                               f"contact {contact}), so no back : through pair can be "
                               f"formed")}
        return {"state": "supported", "label": "PUTT TEMPO back : through",
                "backswing_frames": back_frames, "downswing_frames": through_frames,
                "backswing_downswing_ratio": round(back_frames / float(through_frames),
                                                   3),
                "backswing_start_marker": "backstroke",
                "marks": {"backstroke": backstroke, "forward_stroke": forward,
                          "contact": contact},
                "unit": ":1",
                "note": ("source frames only: this is a position in the source "
                         "timeline, not measured action time")}
    if canonical:
        return dict(canonical)
    return {"state": "unavailable", "label": "TEMPO", "unit": ":1",
            "reason": ("the phase resolver produced no tempo record for this scene; "
                       "it is reported unavailable rather than recomputed here")}


def _broadcast_badge(stroke_type: str, frame: int,
                     markers: Dict[str, Optional[int]]) -> Optional[str]:
    """The phase badge for one frame, in the scene's own vocabulary.

    Bounded by construction: the marker table only holds this scene's own marks and
    the caller only draws inside the window, so FINISH cannot ride into a reaction
    shot behind the cut.
    """
    badges = _BROADCAST_BADGES.get(stroke_type) or _BROADCAST_BADGES["full_swing"]
    order = [slot for slot in ("address", "takeaway", "top", "downswing", "impact",
                               "follow_through", "finish")
             if markers.get(slot) is not None]
    current: Optional[str] = None
    for slot in order:
        marker = markers.get(slot)
        if marker is not None and int(marker) <= int(frame):
            current = slot
    if current is None:
        return None
    slot_vocab = _BROADCAST_SLOT_VOCAB.get(stroke_type) or _BROADCAST_SLOT_TO_NAME
    name = slot_vocab.get(current, current)
    return badges.get(name)


def _run_broadcast(cap: cv2.VideoCapture, spec: dict) -> dict:
    """Measure every scene, then composite the whole clip once.

    Each scene is measured by ``_measure`` with its own interval, so a scene's body,
    tracks, phases, card series and hand_rise reference are built from that scene's
    frames alone. Nothing is shared between scenes, and no record exists for a frame
    that is not inside a golfer-stroke window.
    """
    broadcast = spec.get("broadcast") or {}
    measured: List[dict] = []
    for entry in (broadcast.get("scenes") or []):
        scene_spec = entry.get("spec")
        if str(entry.get("kind") or "golfer_stroke") != "golfer_stroke" or not scene_spec:
            measured.append({"entry": entry, "measured": False, "body": None,
                             "track_objects": {}, "tracks": {}, "phases": None,
                             "raw": {}})
            continue
        body, track_objects, tracks, phases, raw = _measure(cap, scene_spec)
        measured.append({"entry": entry, "measured": True, "body": body,
                         "track_objects": track_objects, "tracks": tracks,
                         "phases": phases, "raw": raw})

    render, stills = _render_pass_broadcast(cap, spec, measured)
    keyframes = _render_broadcast_stills(spec, stills)

    scenes_out: List[dict] = []
    for item in measured:
        entry = item["entry"]
        body = item["body"] or {}
        coverage = {"frames_with_pose": sum(1 for r in (body.get("per_frame") or [])
                                            if r.get("available")),
                    "frames_total": len(body.get("per_frame") or [])}
        scenes_out.append({
            "id": entry.get("id"), "label": entry.get("label"),
            "kind": entry.get("kind"), "stroke_type": entry.get("stroke_type"),
            "window": dict(entry.get("window") or {}),
            "phase_vocabulary": list(entry.get("vocabulary") or []),
            "operator_keyframes": dict(entry.get("keyframes") or {}),
            "measured": bool(item["measured"]),
            "body": {"state": body.get("state"), "method": body.get("method"),
                     "coverage": coverage,
                     "identity": body.get("identity") or {},
                     "roi_xywh": body.get("roi_xywh"),
                     "limitations": list(body.get("limitations") or []),
                     "per_frame": body.get("per_frame") or []},
            "targets": item["tracks"],
            # Published in the scene's own vocabulary: a putt has no TOP here.
            "phases": (_broadcast_scene_phases(
                item["phases"] or {}, stroke_type=str(entry.get("stroke_type") or
                                                      "unknown"),
                vocabulary=entry.get("vocabulary") or [],
                keyframes=entry.get("keyframes") or {})
                if item["measured"] else
                {"phases": {}, "vocabulary": list(entry.get("vocabulary") or []),
                 "stroke_type": entry.get("stroke_type"),
                 "tempo": {"state": "unavailable",
                           "reason": "this window is not a golfer stroke, so no "
                                     "phase was measured in it", "unit": ":1"},
                 "source_seconds_basis": None, "notes": []}),
            "raw_measurements": item["raw"],
        })
    return {
        "source_frames": {"start_frame": int(spec["interval"]["start_frame"]),
                          "end_frame": int(spec["interval"]["end_frame"]),
                          "count": int(spec["interval"]["end_frame"])
                          - int(spec["interval"]["start_frame"]) + 1},
        "scenes": scenes_out,
        "render": render,
        "keyframes": keyframes,
        "environment": _environment(),
    }


def _render_broadcast_stills(spec: dict, stills: Dict[str, Any]) -> dict:
    """Write ``scene-<id>_<phase>.png`` for every marked phase, then a contact sheet.

    Each still is composed from the SAME staged frame the video showed (same skeleton,
    same markers, same trails) with that scene's own card row, so a still can never
    disagree with the video. The key includes the scene id, so a second window's
    ``address`` still cannot overwrite the first window's.
    """
    from . import render_showcase as sc_render

    out_dir = spec.get("keyframes_dir") or "."
    options = spec.get("showcase") or {}
    os.makedirs(out_dir, exist_ok=True)
    written: List[dict] = []
    panels: List[tuple] = []
    engine = sc_render.TextEngine()
    # Stroke order, not alphabetical: the sheet is read as a sequence, so ADDRESS
    # must come before TAKEAWAY. Sorting the keys would put DOWNSWING first.
    order: List[tuple] = []
    for key in sorted(stills):
        entry = stills[key]
        scene_id = str(entry.get("scene_id") or "")
        vocabulary = list(entry.get("vocabulary_order") or [])
        phase = str(entry.get("phase") or "")
        try:
            rank = vocabulary.index(phase)
        except ValueError:
            rank = len(vocabulary)
        order.append((scene_id, rank, phase, key))
    for _scene, _rank, _phase, key in sorted(order):
        entry = stills[key]
        staged = entry["staged"]
        height, width = staged.shape[:2]
        card_lookup = entry.get("card_metrics") or []
        layout = _broadcast_layout(width, height, card_lookup,
                                   options.get("card_row_max_width"),
                                   entry.get("card_position", "bottom"))
        chrome = {
            "text_scrim": True,
            "brand": options.get("brand", ""),
            "brand_sub": options.get("brand_sub", ""),
            # The same badge table the video uses, so a still and the video cannot
            # spell the same phase differently.
            "phase": str(entry.get("phase_badge") or entry.get("phase", "")),
            "cards": entry.get("cards") or [],
            "disclosure": options.get("disclosure", ""),
        }
        image, _drawing = sc_render.compose_chrome(staged, layout=layout, chrome=chrome,
                                                   engine=engine, mode="still")
        path = os.path.join(out_dir, f"{key}.png")
        ok = sc_render.write_png(path, image)
        written.append({"key": key, "path": path, "written": bool(ok),
                        "frame": entry.get("frame"), "scene_id": entry.get("scene_id"),
                        "phase": entry.get("phase"),
                        "cards": [c.get("metric") for c in (entry.get("cards") or [])]})
        if ok:
            panels.append((f"{entry.get('scene_id')} / {entry.get('phase')}", image))
    sheet_path = spec.get("contact_sheet") or ""
    sheet_written = False
    if panels and sheet_path:
        sheet = sc_render.contact_sheet(
            panels, columns=min(3, len(panels)),
            title=f"{options.get('brand', '')} — scene stills",
            subtitle="source frame order within each scene",
            disclaimer=str(options.get("disclosure", "")))
        sheet_written = sc_render.write_png(sheet_path, sheet)
    return {"stills": written, "contact_sheet": sheet_path,
            "contact_sheet_written": bool(sheet_written),
            "count": len([s for s in written if s["written"]])}


def _broadcast_layout(width: int, height: int, metrics, max_row_width,
                      card_position: str = "bottom"):
    from . import showcase as showcase_mod

    return showcase_mod.plan_layout(width, height, tuple(metrics),
                                    max_row_width=max_row_width,
                                    card_position=card_position)


def _broadcast_content_bottom(body_records, tracks, *, scale: float) -> Optional[float]:
    """The lowest output-pixel y that carries measured content in this scene.

    The union of every visible pose keypoint and every drawn ball/club point: this is
    the thing the card row must not cover. It is computed over the WHOLE scene, not per
    frame, because a row that jumped between top and bottom during playback would be
    worse than either position.
    """
    lowest: Optional[float] = None
    for record in body_records or []:
        if not record.get("available"):
            continue
        for point in record.get("keypoints") or []:
            if point.get("visible") and point.get("y") is not None:
                y = float(point["y"]) * scale
                lowest = y if lowest is None else max(lowest, y)
    for track in (tracks or {}).values():
        for point in getattr(track, "points", []) or []:
            if point.drawn and point.y is not None:
                y = float(point.y) * scale
                lowest = y if lowest is None else max(lowest, y)
    return lowest


def _broadcast_card_position(request: str, *, body, tracks, scale: float, out_w: int,
                             out_h: int, metrics, max_row_width) -> tuple:
    """Resolve the card row's position for one scene, with its reason.

    ``auto`` moves the row to the top when the measured content would otherwise sit
    under it. The decision is made once for the whole scene and recorded, so the video,
    the stills, the analysis and the provenance cannot disagree about where the row is.
    """
    from . import showcase as showcase_mod

    if request in ("top", "bottom"):
        return request, f"the operator set card_position to {request!r}"
    bottom_layout = showcase_mod.plan_layout(out_w, out_h, metrics,
                                             max_row_width=max_row_width,
                                             card_position="bottom")
    lowest = _broadcast_content_bottom(body, tracks, scale=scale)
    if lowest is None:
        return "bottom", ("no measured content in this scene, so the row took the "
                          "default bottom position")
    clearance = max(2.0, 0.006 * out_h)
    if lowest > bottom_layout.strip_top - clearance:
        return "top", (f"measured content reaches output y={lowest:.0f}, which is "
                       f"inside the bottom card band "
                       f"(y={bottom_layout.strip_top}..{bottom_layout.strip_bottom}); "
                       f"the row moved to the top rather than covering it")
    return "bottom", (f"measured content stops at output y={lowest:.0f}, above the "
                      f"bottom card band (y={bottom_layout.strip_top}), so the row "
                      f"stayed at the bottom")


def _broadcast_metric_availability(display: dict, clip: dict,
                                   metrics: Sequence[str]) -> Dict[str, bool]:
    """Which cards actually have something to show anywhere in this scene.

    A metric with no measured frame in the scene is left out for the WHOLE scene, so
    a card cannot pop in and out frame by frame. If none survives, the card row is
    omitted and the reason is recorded rather than four placeholders being drawn.
    """
    from . import showcase as showcase_mod

    out: Dict[str, bool] = {}
    for metric in metrics:
        spec = showcase_mod.METRICS.get(metric) or {}
        if spec.get("source") == "frame":
            series = ((display.get("measured") or {}).get(metric) or {})
            out[metric] = any(value is not None for value in series.values())
        else:
            out[metric] = (clip.get(metric) or {}).get("state") == "supported"
    return out


def _broadcast_metric_record(record: Optional[dict], hidden: Sequence[Sequence[int]]) -> Optional[dict]:
    if record is None:
        return None
    frame = int(record["source_frame"])
    if any(int(first) <= frame <= int(last) for first, last in hidden):
        return dict(record, available=False, metric_gate="operator_fast_motion")
    return record


def _broadcast_card_display(body_records, metrics, *, handedness, wrist_reference,
                            min_run, hidden):
    """Reset display hysteresis at every explicit operator metric gate."""
    from itertools import groupby
    from . import showcase as showcase_mod
    if not hidden:
        return _card_display_series(body_records, metrics, handedness=handedness,
                                    wrist_reference=wrist_reference, min_run=min_run)
    combined = {"display": {}, "measured": {}, "records": {}}
    gated = [_broadcast_metric_record(record, hidden) for record in body_records]
    for is_hidden, records in groupby(gated, lambda r: bool(r.get("metric_gate"))):
        block = _card_display_series(list(records), metrics, handedness=handedness,
                                     wrist_reference=wrist_reference,
                                     min_run=1 if is_hidden else min_run)
        for kind in ("display", "measured"):
            for metric, values in block[kind].items():
                combined[kind].setdefault(metric, {}).update(values)
    for metric, series in combined["display"].items():
        values = combined["measured"][metric]
        combined["records"][metric] = {
            **showcase_mod.debounce_record(series, min_run=min_run),
            "measured_frames": len(values),
            "measured_available_frames": sum(v is not None for v in values.values()),
            "operator_gated_frames": sum(bool(r.get("metric_gate")) for r in gated)}
    return combined


def _render_pass_broadcast(cap: cv2.VideoCapture, spec: dict, measured: List[dict]
                           ) -> tuple:
    """Stream the WHOLE clip once, drawing only inside the scene windows.

    Frame *n* of the output is frame *n* of the source: the identity mapping is the
    whole point of the broadcast render. On a frame in no window the picture is the
    source picture (plus, at most, the persistent brand mark) -- no skeleton, no
    marker, no trail, no card, no badge, no held value. On a frame inside a window,
    exactly that window's own state is drawn.
    """
    from . import showcase as showcase_mod
    from .analytics import body_metrics as _body_metrics
    from .tracking import trail_window

    broadcast = spec.get("broadcast") or {}
    options = spec.get("showcase") or {}
    render_cfg = spec.get("render") or {}
    native_w, native_h = (int(v) for v in spec["native_wh"])
    start_frame = int(spec["interval"]["start_frame"])
    end_frame = int(spec["interval"]["end_frame"])
    fps = float(spec["native_fps"])
    fps_rational = str(spec.get("native_fps_rational") or "") or _rational(fps)
    ffmpeg = _ffmpeg_path()
    video_path = str(render_cfg.get("path") or "")
    final_path = str(broadcast.get("final_path") or video_path)
    audio = broadcast.get("audio") or {}
    outside = str(broadcast.get("outside_windows") or "brand_only")
    debug_dir = spec.get("debug_dir") or "."
    os.makedirs(debug_dir, exist_ok=True)

    scale = 1.0
    if render_cfg.get("max_width"):
        scale = min(3.0, max(0.1, float(render_cfg["max_width"]) / float(native_w)))
    out_w = int(round(native_w * scale))
    out_h = int(round(native_h * scale))
    trimmed = False
    if out_w % 2 or out_h % 2:
        out_w -= out_w % 2
        out_h -= out_h % 2
        trimmed = True
        scale = out_w / float(native_w)
    style = render_showcase_mod.style_scale(out_h)

    base_result = {
        "written": False, "frames": 0, "fps": fps, "fps_rational": fps_rational,
        "wh": [out_w, out_h], "picture_wh": [out_w, out_h],
        "source_wh": [native_w, native_h], "scale": round(scale, 6),
        "trimmed_to_even": trimmed, "style": "broadcast",
        "reason": "", "audio": dict(audio), "muxed": False,
        "output_to_source_frames": "identity (frame i of the output is frame i of the source)",
    }
    if not render_cfg.get("enabled", True):
        base_result["reason"] = "rendering off; measurement only"
        return base_result, {}

    if not ffmpeg:
        base_result["reason"] = (
            "ffmpeg was not found on PATH. The broadcast render writes H.264 / "
            "yuv420p / faststart output with ffmpeg and muxes the source audio.")
        return base_result, {}

    card_row_max_width = options.get("card_row_max_width")
    debounce_frames = int(options.get("card_debounce_frames")
                          or showcase_mod.DEFAULT_CARD_DEBOUNCE_FRAMES)
    accent_on = bool(options.get("impact_accent", True))
    accent_span = int(options.get("impact_accent_span_frames") or 3)
    trail_points = {"ball": int(options.get("ball_trail_points")
                                or showcase_mod.BALL_TRAIL_POINTS),
                    "clubhead": int(options.get("club_trail_points")
                                    or showcase_mod.CLUB_TRAIL_POINTS)}
    handedness_default = str(spec.get("handedness", "right"))

    states: List[dict] = []
    for item in measured:
        entry = item["entry"]
        window = entry.get("window") or {}
        state = {
            "entry": entry, "id": entry.get("id"),
            "start": int(window.get("start_frame") or 0),
            "end": int(window.get("end_frame") or -1),
            "measured": bool(item["measured"]),
            "metrics": [], "omitted_metrics": [], "card_row": False,
            "row_reason": "", "layout": None, "display_names": {}, "markers": {},
            "body_by_frame": {}, "point_by_frame": {}, "clip": {}, "card_display": {},
            "wrist_reference": None, "handedness": handedness_default,
            "scene_id": entry.get("id"),
            "contact_frame": entry.get("contact_frame"),
            "vocabulary": list(entry.get("vocabulary") or []),
            "card_records": {},
        }
        if item["measured"]:
            body = item["body"]
            phases = item["phases"] or {}
            stroke_type = str(entry.get("stroke_type") or "unknown")
            handedness = str(entry.get("handedness") or handedness_default)
            metrics = tuple(entry.get("metrics") or options.get("metrics")
                            or showcase_mod.DEFAULT_METRICS)
            frames_possible = state["end"] - state["start"] + 1
            clip_analytics = {"body_2d": _body_metrics(
                body["per_frame"], frames_possible=frames_possible,
                pixel_aspect=float(spec.get("pixel_aspect", 1.0)),
                handedness=handedness)}
            published_phases = _broadcast_scene_phases(
                phases, stroke_type=stroke_type, vocabulary=entry.get("vocabulary") or [],
                keyframes=entry.get("keyframes") or {})
            clip = showcase_mod.clip_values(clip_analytics, published_phases,
                                            handedness=handedness)
            if stroke_type == "putt" and (clip.get("tempo") or {}).get("state") == "supported":
                tempo = published_phases["tempo"]
                clip["tempo"]["sub"] = f"back {tempo['backswing_frames']}f / through {tempo['downswing_frames']}f"
            wrist_reference = showcase_mod.wrist_reference(body["per_frame"])
            display = _broadcast_card_display(
                body["per_frame"], metrics, handedness=handedness,
                wrist_reference=wrist_reference, min_run=debounce_frames,
                hidden=entry.get("metric_hide_frames") or [])
            available = _broadcast_metric_availability(display, clip, metrics)
            kept = [m for m in metrics if available.get(m)]
            omitted = [m for m in metrics if not available.get(m)]
            position, why = _broadcast_card_position(
                str(options.get("card_position") or "auto"),
                body=body["per_frame"], tracks=item["track_objects"], scale=scale,
                out_w=out_w, out_h=out_h, metrics=kept,
                max_row_width=card_row_max_width)
            state.update({
                "handedness": handedness,
                "display_names": dict(_BROADCAST_BADGES.get(
                    stroke_type) or _BROADCAST_BADGES["full_swing"]),
                "slot_vocab": dict(_BROADCAST_SLOT_VOCAB.get(stroke_type)
                                   or _BROADCAST_SLOT_TO_NAME),
                "markers": _broadcast_phase_markers(phases),
                "body_by_frame": {int(r["source_frame"]): r
                                  for r in body["per_frame"]},
                "point_by_frame": {name: {int(p.source_frame): p for p in track.points}
                                   for name, track in item["track_objects"].items()},
                "track_objects": dict(item["track_objects"]),
                "clip": clip, "card_display": display,
                "wrist_reference": wrist_reference,
                "card_records": (display or {}).get("records") or {},
                "metrics": kept, "omitted_metrics": omitted,
            })
            state["layout"] = showcase_mod.plan_layout(
                out_w, out_h, kept, max_row_width=card_row_max_width,
                card_position=position)
            state["card_position"] = position
            state["card_position_reason"] = why
            if kept:
                state["card_row"] = True
            else:
                state["row_reason"] = (
                    "no card in this scene's metric set had a single measured frame, "
                    "so the row is omitted rather than drawn as four placeholders")
        states.append(state)

    def scene_for_frame(frame: int) -> Optional[dict]:
        for state in states:
            if state["start"] <= frame <= state["end"]:
                return state
        return None

    # The chrome layout for a frame in no window: brand only, no card row, no badge.
    brand_only_layout = showcase_mod.plan_layout(out_w, out_h, (),
                                                 max_row_width=card_row_max_width)
    # The brand-only frames still report the layout the report measures coverage from.
    chrome_engine = render_showcase_mod.TextEngine()
    stills: Dict[str, Any] = {}
    rendered = 0
    empty_frames = 0
    overlay_frames = 0

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", fps_rational, "-i", "-",
           "-an", "-c:v", "libx264", "-preset", str(render_cfg.get("preset", "medium")),
           "-crf", str(render_cfg.get("crf", 18)), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", video_path]
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
    log_path = os.path.join(debug_dir, "ffmpeg.log")

    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log, stderr=log)
        stdin = proc.stdin
        if stdin is None:  # pragma: no cover - Popen with PIPE always provides one
            raise RuntimeError("ffmpeg stdin pipe could not be opened")
        try:
            for frame, img in _forward_leg(cap, start_frame, end_frame + 1):
                picture = (img if abs(scale - 1.0) < 1e-9 else
                           cv2.resize(img, (out_w, out_h),
                                      interpolation=(cv2.INTER_AREA if scale < 1.0
                                                     else cv2.INTER_CUBIC)))
                state = scene_for_frame(frame)
                if state is None or not state["measured"]:
                    # A frame in no window, or in a window that is not a golfer
                    # stroke: the source picture, and at most the persistent mark.
                    if outside == "clean":
                        annotated = picture
                    else:
                        annotated, _draw = render_showcase_mod.compose_chrome(
                            picture, layout=brand_only_layout,
                            chrome={"brand": options.get("brand", ""),
                                    "brand_sub": options.get("brand_sub", ""),
                                    "phase": None, "cards": [], "disclosure": ""},
                            engine=chrome_engine)
                    empty_frames += 1
                else:
                    annotated = _draw_broadcast_slot(
                        picture, frame, state, scale, style, trail_window,
                        trail_points, accent_on, accent_span, options,
                        chrome_engine, stills)
                    overlay_frames += 1
                payload = np.ascontiguousarray(annotated).tobytes()
                stdin.write(payload)
                rendered += 1
        except BaseException:
            stdin.close()
            proc.kill()
            # Reap the child: killing without waiting leaves a zombie and can leave
            # ffmpeg writing into the output file after the next run starts.
            try:
                proc.wait(timeout=30)
            except Exception:
                pass
            raise
        finally:
            try:
                stdin.close()
            except Exception:
                pass
        code = proc.wait()

    if code != 0:
        base_result["reason"] = (f"ffmpeg exited with status {code}; see {log_path}")
        base_result["frames"] = rendered
        return base_result, stills

    muxed, mux_note = _mux_source_audio(ffmpeg, video_path, final_path, audio,
                                        os.path.join(debug_dir, "ffmpeg_mux.log"))
    card_disclosure = []
    for state in states:
        if not state["measured"]:
            continue
        records = state.get("card_records") or {}
        held = sum(int((records.get(metric) or {}).get("held_count") or 0)
                   for metric in state["metrics"])
        card_disclosure.append({
            "scene_id": state["id"],
            "metrics": list(state["metrics"]),
            "omitted_metrics": list(state["omitted_metrics"]),
            "card_row": bool(state["card_row"]),
            "row_reason": state["row_reason"],
            "card_position": state.get("card_position", "bottom"),
            "card_position_reason": state.get("card_position_reason", ""),
            "card_debounce_frames": debounce_frames,
            "metric_hide_frames": state["entry"].get("metric_hide_frames") or [],
            "metric_gate_reason": "operator withheld frame body values during visually marked fast motion; clip tempo remains available",
            "held_value_frames": held,
            "wrist_reference": state.get("wrist_reference"),
            "debounce_records": {metric: (records.get(metric) or {})
                                 for metric in state["metrics"]},
        })
    base_result.update({
        "written": True, "frames": rendered, "muxed": muxed, "mux_note": mux_note,
        "empty_frames": empty_frames, "overlay_frames": overlay_frames,
        "path": final_path, "video_only_path": video_path,
        "frame_plan": {"stride": 1, "holds": [],
                       "output_frames": rendered,
                       "scenes_drawn": sorted({s["id"] for s in states
                                               if s["measured"]})},
        "scenes": card_disclosure,
        "audio_verification": _audio_fingerprint(ffmpeg, final_path, audio),
    })
    return base_result, stills


def _rational(value: float) -> str:
    """A float frame rate as an ffmpeg rational, only when it is already one."""
    from fractions import Fraction

    frac = Fraction(value).limit_denominator(1000000)
    return f"{frac.numerator}/{frac.denominator}"


def _broadcast_phase_markers(phases: Mapping[str, Any]) -> Dict[str, Optional[int]]:
    records = (phases or {}).get("phases") or {}
    return {name: (None if (records.get(name) or {}).get("frame") is None
                   else int(records[name]["frame"]))
            for name in ("address", "takeaway", "top", "downswing", "impact",
                         "follow_through", "finish")}


def _draw_broadcast_slot(picture, frame: int, state: dict, scale: float,
                         style: float, trail_window, trail_points: dict,
                         accent_on: bool, accent_span: int, options: dict,
                         engine, stills: dict):
    """Draw one in-window frame: this scene's overlays, then this scene's chrome."""
    from . import showcase as showcase_mod

    record = state["body_by_frame"].get(frame)
    markers = state["markers"]

    def scaled_keypoints(item: Optional[dict]) -> List[dict]:
        if item is None or not item.get("available"):
            return []
        return [dict(p, x=p["x"] * scale, y=p["y"] * scale)
                for p in (item.get("keypoints") or [])]

    def scaled_points(name: str, at: int) -> List[dict]:
        points = state["point_by_frame"].get(name) or {}
        ordered = [points[k] for k in sorted(points)]
        window = trail_window(ordered, at, trail_points.get(name, 10))
        # ``trail: segments_only`` (opt-in, per target) keeps this scene's trail inside
        # the operator's own anchored segments. The track object is this scene's own,
        # so the filter cannot see a point from another scene.
        track = (state.get("track_objects") or {}).get(name)
        if track is not None:
            window = manual_segments_only(track, window)
        return [dict(p.to_dict(), x=p.x * scale, y=p.y * scale)
                for p in window if p.x is not None and p.y is not None]

    def overlay_draw(layer, mask, item, at):
        drawn = 0
        keypoints = scaled_keypoints(item)
        if keypoints:
            drawn += render_showcase_mod.draw_skeleton(layer, mask, keypoints, style)
        for name, points in (state["point_by_frame"] or {}).items():
            point = points.get(at)
            if point is None or not point.drawn:
                continue
            payload = scaled_points(name, at)
            payload.append(dict(point.to_dict(), x=point.x * scale, y=point.y * scale))
            if name == "clubhead":
                render_showcase_mod.draw_trail(
                    layer, mask, payload, render_showcase_mod.CLUB_TRAIL,
                    alpha=render_showcase_mod.CLUB_TRAIL_ALPHA,
                    thickness=max(3, int(round(3.4 * style))), dash=5)
                render_showcase_mod.draw_club_marker(
                    layer, mask, dict(point.to_dict(), x=point.x * scale,
                                      y=point.y * scale), style)
            else:
                render_showcase_mod.draw_trail(
                    layer, mask, payload, render_showcase_mod.BALL_TRAIL,
                    alpha=render_showcase_mod.TRAIL_ALPHA,
                    thickness=max(2, int(round(2.0 * style))), dash=4)
                render_showcase_mod.draw_ball_marker(
                    layer, mask, dict(point.to_dict(), x=point.x * scale,
                                      y=point.y * scale), style)
        if accent_on:
            contact = state.get("contact_frame")
            strength = showcase_mod.impact_accent(at, contact, half_span=accent_span)
            if strength > 0:
                anchor = None
                for name in ("clubhead", "ball"):
                    candidate = (state["point_by_frame"].get(name) or {}).get(at)
                    if candidate is not None and candidate.drawn:
                        anchor = dict(candidate.to_dict(), x=candidate.x * scale,
                                      y=candidate.y * scale)
                        break
                render_showcase_mod.draw_impact_accent(layer, mask, anchor, style,
                                                       strength)
        return drawn

    card_record = _broadcast_metric_record(record, state["entry"].get("metric_hide_frames") or [])
    cards = _showcase_card_payload(state["metrics"], card_record, state["clip"],
                                   handedness=state["handedness"],
                                   wrist_ref=state["wrist_reference"],
                                   display=(state["card_display"] or {}).get(
                                       "display")) if state["card_row"] else []
    chrome = {
        "text_scrim": True,
            "brand": options.get("brand", ""),
        "brand_sub": options.get("brand_sub", ""),
        "phase": _broadcast_badge(str(state["entry"].get("stroke_type") or "unknown"),
                                  frame, markers),
        "cards": cards,
        "disclosure": (options.get("disclosure", "")
                       if (cards or state["measured"]) else ""),
    }
    staged = render_showcase_mod.stage_overlays(
        picture, overlay_draw=lambda layer, mask: overlay_draw(layer, mask, record,
                                                               frame))
    annotated, _drawing = render_showcase_mod.compose_chrome(
        staged, layout=state["layout"], chrome=chrome, engine=engine)
    for slot, marker in markers.items():
        if marker is not None and int(marker) == int(frame):
            # The still is named in the scene's OWN vocabulary, so a putt writes
            # scene-<id>_contact.png and never ..._impact.png.
            name = (state.get("slot_vocab") or {}).get(slot, slot)
            key = f"scene-{state['id']}_{name}"
            # A still is read on its own, out of context, so it must never carry a
            # HELD value: it gets the measured value at this frame or a placeholder,
            # and the badge spelling comes from the same table the video uses.
            measured = _showcase_card_payload(
                state["metrics"], card_record, state["clip"],
                handedness=state["handedness"],
                wrist_ref=state["wrist_reference"], display=None
            ) if state["card_row"] else []
            stills.setdefault(key, {"staged": staged.copy(), "cards": measured,
                                    "frame": frame, "scene_id": state["id"],
                                    "phase": name,
                                    "phase_badge": (state.get("display_names") or {}).get(
                                        name, name.replace("_", " ").upper()),
                                    "vocabulary_order": list(
                                        (state.get("vocabulary") or [])),
                                    "card_position": state.get("card_position", "bottom"),
                                    "card_metrics": list(state["metrics"])})
    return annotated


def _mux_source_audio(ffmpeg: str, video_path: str, final_path: str, audio: dict,
                      log_path: str) -> tuple:
    """Mux the clip's own audio onto the rendered video, without re-encoding it.

    ``-c:a copy`` is what makes the audio bit-exact; there is no resample, no volume
    change and no ``-shortest``, so a stream that is a few milliseconds longer or
    shorter than the picture is preserved rather than silently trimmed.
    """
    source = str(audio.get("source") or "")
    if not audio.get("present") or str(audio.get("mode") or "copy") == "none" or not source:
        shutil.copyfile(video_path, final_path)
        return False, ("no audio was present in the source, so the render is video "
                       "only" if not audio.get("present")
                       else "audio muxing was switched off for this run")
    mux_tmp = final_path + ".muxing.mp4"
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", video_path,
           "-i", source, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
           "-c:a", "copy", "-movflags", "+faststart", mux_tmp]
    with open(log_path, "wb") as log:
        code = subprocess.call(cmd, stdout=log, stderr=log)
    if code != 0:
        shutil.copyfile(video_path, final_path)
        return False, (f"the audio mux failed (ffmpeg status {code}); the video-only "
                       f"render was kept and the failure is recorded")
    # Publish by rename, so a --force rerun can never overwrite a finished video in
    # place while it is still being muxed.
    os.replace(mux_tmp, final_path)
    return True, "source audio copied bit-exactly (-c:a copy)"


def _audio_fingerprint(ffmpeg: str, path: str, audio: dict) -> Dict[str, Any]:
    """The output's audio stream, and an md5 of its audio packets."""
    out: Dict[str, Any] = {"checked": False}
    if not os.path.isfile(path):
        out["reason"] = "no output file"
        return out
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=codec_name,sample_rate,channels,time_base,duration,nb_frames",
             "-of", "json", path], capture_output=True, text=True, timeout=60)
        payload = json.loads(probe.stdout or "{}")
        streams = payload.get("streams") or []
    except Exception as exc:  # pragma: no cover - ffprobe missing/unparsable
        out["reason"] = f"ffprobe could not read the output audio: {exc}"
        return out
    if not streams:
        out["present"] = False
        out["checked"] = True
        out["reason"] = "the output has no audio stream"
        return out
    stream = streams[0]
    out.update({"checked": True, "present": True,
                "codec_name": stream.get("codec_name"),
                "sample_rate": stream.get("sample_rate"),
                "channels": stream.get("channels")})
    digest = _audio_md5(ffmpeg, path)
    out["audio_md5"] = digest
    return out


def _audio_md5(ffmpeg: str, path: str) -> str:
    try:
        p = subprocess.run([ffmpeg, "-v", "error", "-i", path, "-map", "0:a:0",
                            "-c", "copy", "-f", "md5", "-"],
                           capture_output=True, text=True, timeout=300)
        return (p.stdout or "").strip().splitlines()[-1].strip() if p.stdout else ""
    except Exception:  # pragma: no cover - ffmpeg missing
        return ""


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
