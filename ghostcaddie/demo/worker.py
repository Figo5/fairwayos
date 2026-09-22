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
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import cv2
import numpy as np

from . import report as report_mod
from . import render as render_mod
from .tracking import RawMeasurement, resolve_track

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

    per_frame: List[dict] = []
    if not roi:
        limitations.append(
            "No operator ROI was supplied. The demo does not attribute a whole-frame "
            "single-pose skeleton to the golfer, so the body layer is ambiguous and "
            "no pose, no skeleton and no 2D angle is reported. Draw an ROI in "
            "`demo-setup` to enable body analysis.")
        for frame in range(start_frame, end_frame + 1):
            per_frame.append({"source_frame": frame, "available": False,
                              "reason": "no_operator_roi",
                              "visible_keypoint_count": 0, "keypoints": []})
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
                              "visible_keypoint_count": 0, "keypoints": []})
        return {"state": "unavailable", "method": "none", "identity": identity,
                "model_error": model_error or "body model not found",
                "limitations": limitations, "per_frame": per_frame,
                "roi_xywh": list(roi)}

    x, y, rw, rh = (int(v) for v in roi)
    if rw <= 0 or rh <= 0 or x < 0 or y < 0:
        raise RuntimeError(f"roi {list(roi)} is not usable")
    identity["roi_xywh"] = [x, y, rw, rh]
    identity["identity_method"] = "operator_roi"
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
        crop = img[y:y + rh, x:x + rw]
        if crop.size == 0:
            per_frame.append({"source_frame": frame, "available": False,
                              "reason": "empty_roi_crop", "visible_keypoint_count": 0,
                              "keypoints": []})
            continue
        try:
            points, transform = net.infer(crop, (x, y))
        except Exception as exc:
            per_frame.append({"source_frame": frame, "available": False,
                              "reason": f"pose_inference_failed:{type(exc).__name__}",
                              "visible_keypoint_count": 0, "keypoints": []})
            continue
        ok, why = attribute(points, [x, y, rw, rh])
        visible = sum(1 for p in points if p["visible"])
        per_frame.append({"source_frame": frame, "available": bool(ok),
                          "reason": why, "visible_keypoint_count": visible,
                          "keypoints": points, "transform": transform})
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
        for name in ("ball", "clubhead"):
            target_spec = spec.get("targets", {}).get(name) or {}
            seed = spec.get("seeds", {}).get(name)
            if not seed:
                continue
            options = _options(target_spec.get("options"))
            corrections = spec.get("corrections", {}).get(name) or []
            raw = measure_target(cap, name, options, _point(seed),
                                 [_point(c) for c in corrections],
                                 int(spec["interval"]["start_frame"]),
                                 int(spec["interval"]["end_frame"]))
            raw_dump[name] = {str(k): v.to_dict() for k, v in sorted(raw.items())}
            track = resolve_track(
                name, options=options, interval_frames=interval_frames, raw=raw,
                seed=_point(seed), corrections=[_point(c) for c in corrections],
                interpolation_enabled=bool(spec["interpolation"]["enabled"]),
                max_gap_frames=int(spec["interpolation"]["max_gap_frames"]))
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
