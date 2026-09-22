"""Draw the demo overlay. cv2-side only: imported by the worker, never by the CLI.

Visual hierarchy, in order of what the eye should find first:

1. the golfer's skeleton in a single subtle teal, with slightly heavier joints at
   shoulders, hips, knees, elbows and wrists, and a stable ring on the head;
2. the operator-seeded ball (small bright circle) and club (contrasting diamond),
   each with its own trail;
3. dotted/dashed strokes for derived interpolation, so nothing that was not
   observed ever looks like a measurement;
4. one compact phase badge, and the analytics in a dedicated panel placed on the
   side of the frame the golfer is not using -- or outside the picture entirely in
   a gutter, when both sides would cover the operator's ROI.

There is deliberately no confidence number, no frame-score and no identifier text
on the picture: that detail lives in `debug/`, where it can be read without
competing with the swing.

Every string drawn here is ASCII. cv2's Hershey fonts cannot render middle dots,
en/em dashes or degree signs, so a Unicode separator shows up as visible `??`
artifacts on the video. Markdown outputs keep their typography; pixels do not.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .report import (VIDEO_CAVEAT, VIDEO_LEGEND, VIDEO_SUBTITLE, VIDEO_TITLE,
                     PHASE_LABEL_TEXT)

# BGR
TEAL = (172, 196, 52)
TEAL_DIM = (128, 148, 40)
BALL = (70, 235, 255)
CLUB = (255, 96, 208)
PANEL_BG = (24, 22, 20)
TEXT = (238, 238, 238)
MUTED = (170, 170, 170)
AMBER = (90, 205, 255)
SHEET_BG = (18, 18, 18)

SKELETON_EDGES = [("sh_l", "sh_r"), ("sh_l", "el_l"), ("el_l", "wr_l"),
                  ("sh_r", "el_r"), ("el_r", "wr_r"), ("sh_l", "hip_l"),
                  ("sh_r", "hip_r"), ("hip_l", "hip_r"), ("hip_l", "kn_l"),
                  ("kn_l", "ank_l"), ("hip_r", "kn_r"), ("kn_r", "ank_r")]
HEAVY_JOINTS = ("sh_l", "sh_r", "hip_l", "hip_r")
MEDIUM_JOINTS = ("el_l", "el_r", "kn_l", "kn_r", "wr_l", "wr_r")

TRAIL_POINTS = 15
DASH_LENGTH = 7


def font_scale(width: int) -> float:
    return max(0.38, min(0.95, width / 1750.0))


def thickness(width: int) -> int:
    return max(1, int(round(width / 900.0)))


def _dashed_line(img, a, b, color, thick, dash=DASH_LENGTH) -> None:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    length = math.hypot(bx - ax, by - ay)
    if length < 1:
        return
    steps = max(1, int(length // dash))
    for i in range(steps):
        if i % 2:
            continue
        t0, t1 = i / steps, min(1.0, (i + 0.75) / steps)
        cv2.line(img, (int(ax + (bx - ax) * t0), int(ay + (by - ay) * t0)),
                 (int(ax + (bx - ax) * t1), int(ay + (by - ay) * t1)),
                 color, thick, cv2.LINE_AA)


def _text(img, text, org, scale, color, thick=1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                cv2.LINE_AA)


def _overlap(box, panel_x0: int, panel_x1: int, top: int, bottom: int) -> float:
    """How much of an axis-aligned box a panel on that side would cover (pixels²)."""
    if not box:
        return 0.0
    x, y, w, h = (int(v) for v in box)
    ix = max(0, min(x + w, panel_x1) - max(x, panel_x0))
    iy = max(0, min(y + h, bottom) - max(y, top))
    return float(ix * iy)


def _roi_overlap(roi, panel_x0: int, panel_x1: int, top: int, bottom: int) -> float:
    """How much of the ROI a panel on that side would cover (in pixels²)."""
    return _overlap(roi, panel_x0, panel_x1, top, bottom)


def bounding_box(points: Optional[Sequence[Sequence[float]]],
                 *, pad: int = 0) -> Optional[List[int]]:
    """``[x, y, w, h]`` around drawn target points, in the coordinates given.

    Missing points are simply absent: the box is built from the points that will
    actually be drawn, so it is the region of the picture the ball and club action
    occupies rather than a guess about where the flight might have been.
    """
    xs = [float(p[0]) for p in (points or [])
          if p is not None and len(p) >= 2
          and p[0] is not None and p[1] is not None]
    ys = [float(p[1]) for p in (points or [])
          if p is not None and len(p) >= 2
          and p[0] is not None and p[1] is not None]
    if not xs or not ys:
        return None
    pad = max(0, int(pad))
    x0, y0 = int(math.floor(min(xs))) - pad, int(math.floor(min(ys))) - pad
    x1, y1 = int(math.ceil(max(xs))) + pad, int(math.ceil(max(ys))) + pad
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def panel_width_for(width: int) -> int:
    """The analytics panel's width. Compact: it carries angles, not counts."""
    if width < 720:
        return int(max(180, 0.40 * width))
    return int(max(200, min(0.26 * width, 380)))


def plan_panel(roi: Optional[Sequence[int]], width: int, height: int,
               *, points_xy: Optional[Sequence[Sequence[float]]] = None,
               points_pad: int = 0) -> Dict[str, Any]:
    """Where the analytics panel goes, checked against everything it must not cover.

    Two no-go regions, not one:

    1. the operator's ROI -- the golfer; and
    2. the bounding box of every target point that will be DRAWN (observed, anchored
       or interpolated) -- the ball and club action.

    Order of preference:

    1. an overlay column on the side the ROI is not using, but only when the picture
       leaves a band at least TWICE the panel's width clear of both no-go regions.
       Covering either is the failure this check exists to prevent, and a panel that
       clears the golfer by a few pixels still crowds the swing and darkens a third
       of the picture, which is what the first render actually did -- the ball flight
       ran straight through the panel's column while the body-only check reported
       zero overlap;
    2. an EXTERNAL gutter -- the output frame is widened and the panel sits outside
       the picture, so it cannot cover anything, which is the right answer for a
       portrait golfer in a landscape frame whose flight crosses the other side.

    The chosen plan records the measured overlap of BOTH regions, so the debug record
    says what was actually covered instead of claiming the panel is out of the way.
    """
    # The header band must actually CONTAIN the type it carries. At 34 px the title
    # baseline (0.45 * h) put the top of the glyphs above y=0, so the delivered
    # 1612x720 render clipped the first ~4 px of "FAIRWAY OS / DEMO" and left no top
    # margin. 60 px gives a >=5 px margin above the title and keeps the subtitle and
    # the frame counter inside the band at every output height.
    header_h = int(max(60, 0.055 * height))
    legend_h = int(max(24, 0.040 * height))
    panel_w = panel_width_for(width)
    roi_centre = None
    if roi:
        roi_centre = int(roi[0]) + int(roi[2]) / 2.0
    first = "right" if (roi_centre is None or roi_centre < width / 2.0) else "left"
    order = [first, "left" if first == "right" else "right"]
    targets_box = bounding_box(points_xy, pad=points_pad)

    def coverage(side: str) -> Dict[str, float]:
        x0 = 0 if side == "left" else width - panel_w
        return {
            "body": _overlap(roi, x0, x0 + panel_w, header_h, height - legend_h),
            "targets": _overlap(targets_box, x0, x0 + panel_w, header_h,
                                height - legend_h),
        }

    refusals: Dict[str, Any] = {}

    for side in order:
        measured = coverage(side)
        if measured["body"] > 0.0 or measured["targets"] > 0.0:
            refusals[side] = dict(measured, reason="covers_the_no_go_region")
            continue
        # the panel must not merely miss the action: it needs a clear column of twice
        # its own width beside it, or it is crowding the swing
        x0 = 0 if side == "left" else width - panel_w
        band = [max(0, x0 - panel_w), min(width, x0 + panel_w + panel_w)]
        if roi and _overlap(roi, band[0], band[1], header_h, height - legend_h) > 0.0:
            refusals[side] = dict(measured, reason="clearance_band_touches_the_roi",
                                  band_xy=[band[0], band[1]])
            continue
        if targets_box and _overlap(targets_box, band[0], band[1], header_h,
                                    height - legend_h) > 0.0:
            refusals[side] = dict(measured, band_xy=[band[0], band[1]],
                                  reason="clearance_band_touches_a_target_point")
            continue
        return {"side": side, "external": False, "panel_w": panel_w, "panel_x": x0,
                "header_h": header_h, "legend_h": legend_h, "overlap_px2": 0.0,
                "body_overlap_px2": 0.0, "targets_overlap_px2": 0.0,
                "targets_box_xywh": targets_box,
                "clearance_required_px": panel_w, "gutter_w": 0,
                "overlay_refusals": refusals}
    side = first
    # Keep the gutter proportionate: a panel must never eat most of the picture
    # width, or a small frame turns into mostly chrome.
    gutter_w = min(panel_w, max(90, int(round(0.45 * width))))
    return {"side": side, "external": True, "panel_w": panel_w, "panel_x": 0,
            "header_h": header_h, "legend_h": legend_h,
            # the chosen panel is outside the picture, so it covers nothing: the
            # overlaps below are for the CHOSEN region, and the rejected overlay
            # candidates are recorded separately rather than being reported as if
            # the gutter covered them
            "overlap_px2": 0.0, "body_overlap_px2": 0.0, "targets_overlap_px2": 0.0,
            "targets_box_xywh": targets_box, "clearance_required_px": panel_w,
            "overlay_refusals": refusals, "gutter_w": gutter_w}


def panel_side(roi: Optional[Sequence[int]], width: int) -> Tuple[str, int]:
    """Kept for callers that only need the side; use :func:`plan_panel` for layout."""
    if not roi:
        return "right", 0
    x, _y, w, _h = (int(v) for v in roi)
    centre = x + w / 2.0
    return ("right", 0) if centre < width / 2.0 else ("left", 0)


def plan_panel_scaled(roi_xywh: Optional[Sequence[int]], scale: float,
                      width: int, height: int,
                      *, points_xy: Optional[Sequence[Sequence[float]]] = None
                      ) -> Dict[str, Any]:
    """Panel plan for the RENDERED frame size.

    The ROI and the target points are in native source pixels; the panel decision has
    to be made in the output's pixels, or a downscaled render would compare a native
    box against a smaller frame and place the panel wrongly. Both no-go regions are
    scaled through the same transform that draws them, so the check cannot drift away
    from what ends up on the picture.
    """
    scaled = None
    if roi_xywh:
        scaled = [int(round(float(v) * scale)) for v in roi_xywh]
    points = None
    if points_xy:
        points = [(float(p[0]) * scale, float(p[1]) * scale) for p in points_xy
                  if p is not None and len(p) >= 2
                  and p[0] is not None and p[1] is not None]
    # A small margin so the panel does not sit exactly on the flight line.
    pad = max(2, int(round(0.010 * width)))
    return plan_panel(scaled, width, height, points_xy=points, points_pad=pad)


def compose_annotated(picture, *, plan: Dict[str, Any], phase_text: str,
                      frame_text: str, analytics_lines: Sequence[str],
                      operator_note: str = "", overlays=None):
    """Picture -> chrome -> overlays, in that order, on one canvas.

    ``picture`` is the frame with its overlays already drawn on it and is never
    modified, which is what keeps a keyframe card free of video chrome. The chrome
    (header band, analytics panel or gutter, legend band) is drawn on a canvas that
    contains the picture, and then ``overlays`` is called LAST with a view of the
    canvas's picture region, so a skeleton, a trail or a marker is always drawn on
    top of the chrome and can never be dimmed by it.

    Chrome-free keyframe cards and never-dimmed markers are the two things the
    first render got wrong; both are structural here rather than a matter of luck.
    """
    height, width = picture.shape[:2]
    gutter = int(plan.get("gutter_w") or 0) if plan.get("external") else 0
    if gutter:
        canvas_w = width + gutter
        frame_x = gutter if plan.get("side") == "left" else 0
        canvas = np.zeros((height, canvas_w, 3), dtype=np.uint8)
        canvas[:, frame_x:frame_x + width] = picture
    else:
        canvas_w = width
        frame_x = 0
        canvas = picture.copy()
    draw_header_and_panel(canvas, plan=plan, phase_text=phase_text,
                          frame_text=frame_text, analytics_lines=analytics_lines,
                          operator_note=operator_note)
    if overlays is not None:
        # a basic slice of a numpy array is a VIEW, so drawing into it draws onto the
        # canvas -- no coordinate offset arithmetic anywhere in the overlay code
        overlays(canvas[:, frame_x:frame_x + width] if frame_x else canvas)
    return canvas


def _translucent(img, x0, y0, x1, y1, color, alpha) -> None:
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(img.shape[1], int(x1)), min(img.shape[0], int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def draw_skeleton(img, keypoints: Sequence[dict], scale: float) -> None:
    """Only what the record says is visible. Nothing is carried between frames."""
    by = {p["name"]: p for p in keypoints if p.get("visible")}
    if not by:
        return
    thin = max(1, int(round(1.6 * scale)))
    for a, b in SKELETON_EDGES:
        if a in by and b in by:
            cv2.line(img, (int(by[a]["x"] * scale), int(by[a]["y"] * scale)),
                     (int(by[b]["x"] * scale), int(by[b]["y"] * scale)),
                     TEAL, thin, cv2.LINE_AA)
    for name, p in by.items():
        if name == "nose":
            continue
        if name in HEAVY_JOINTS:
            radius, color = int(round(5 * scale)), TEXT
        elif name in MEDIUM_JOINTS:
            radius, color = int(round(4 * scale)), TEAL
        else:
            continue
        cv2.circle(img, (int(p["x"] * scale), int(p["y"] * scale)), radius, color,
                   -1, cv2.LINE_AA)
    if "nose" in by:
        cv2.circle(img, (int(by["nose"]["x"] * scale), int(by["nose"]["y"] * scale)),
                   int(round(9 * scale)), TEAL, max(1, int(round(2 * scale))),
                   cv2.LINE_AA)


def draw_trail(img, points: Sequence[dict], color, scale: float,
               radius: int) -> None:
    """Trail for ONE segment, up to the current frame. Dashed where derived.

    ``points`` must already be limited to the current segment and ordered in time;
    the caller guarantees that a loss or a correction was a segment boundary, so a
    trail can never bridge one.
    """
    if len(points) < 2:
        return
    thick = max(1, int(round(1.4 * scale)))
    for a, b in zip(points, points[1:]):
        if a.get("x") is None or b.get("x") is None:
            continue
        pa = (a["x"] * scale, a["y"] * scale)
        pb = (b["x"] * scale, b["y"] * scale)
        if a.get("status") == "interpolated" or b.get("status") == "interpolated":
            _dashed_line(img, pa, pb, color, thick)
        else:
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     color, thick, cv2.LINE_AA)


def draw_target_marker(img, point: dict, kind: str, scale: float) -> None:
    """One point, in the style its status deserves. Nothing is drawn for missing."""
    status = point.get("status")
    if status in ("missing", "lost") or point.get("x") is None:
        return
    x, y = int(point["x"] * scale), int(point["y"] * scale)
    derived = status == "interpolated"
    operator = status == "manual_anchor"
    color = BALL if kind == "ball" else CLUB
    if derived:
        color = tuple(int(c * 0.55 + 40) for c in color)
    if kind == "ball":
        radius = max(2, int(round((4 if not derived else 3) * scale)))
        cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), radius + max(1, int(round(2 * scale))), color,
                   1, cv2.LINE_AA)
    else:
        size = max(4, int(round((7 if not derived else 5) * scale)))
        pts = np.array([[x, y - size], [x + size, y], [x, y + size], [x - size, y]],
                       dtype=np.int32)
        cv2.polylines(img, [pts], True, color, max(1, int(round(1.6 * scale))),
                      cv2.LINE_AA)
        if not derived:
            cv2.circle(img, (x, y), max(1, int(round(1.6 * scale))), color, -1,
                       cv2.LINE_AA)
    if operator:
        cv2.circle(img, (x, y), int(round(11 * scale)), TEXT,
                   max(1, int(round(1.2 * scale))), cv2.LINE_AA)


def _fit_text(img, text, x: int, y: int, scale: float, color, thick: int,
              max_w: int) -> None:
    """Draw text, shrinking it until it fits ``max_w`` instead of running off.

    Layout is fixed (header band, panel gutter, legend band), so a long operator
    note must give a little type size rather than paint over the picture.
    """
    size = scale
    while size > 0.34 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size,
                                          thick)[0][0] > max_w:
        size -= 0.02
    _text(img, text, (x, y), size, color, thick)


def draw_header_and_panel(img, *, plan: Dict[str, Any], phase_text: str,
                          frame_text: str, analytics_lines: Sequence[str],
                          operator_note: str = "") -> None:
    """Header, phase badge, analytics panel and legend, inside ``plan``'s layout.

    When the plan widened the output with an EXTERNAL gutter, the panel is drawn
    inside that gutter -- the picture keeps its full width and the panel never sits
    on the golfer.
    """
    height, width = img.shape[:2]
    layout = {"header_h": int(plan["header_h"]), "legend_h": int(plan["legend_h"])}
    gutter = int(plan.get("gutter_w") or 0)
    if plan.get("external") and gutter > 0:
        layout["panel_w"] = gutter
        layout["panel_x"] = 0 if plan["side"] == "left" else width - gutter
    else:
        layout["panel_w"] = int(plan["panel_w"])
        layout["panel_x"] = int(plan["panel_x"])
    layout["panel_y"] = layout["header_h"]
    scale = font_scale(width)
    thick = thickness(width)

    _translucent(img, 0, 0, width, layout["header_h"], PANEL_BG, 0.74)
    _text(img, VIDEO_TITLE, (int(10 * scale) + 4, int(layout["header_h"] * 0.45)),
          scale * 0.95, TEXT, thick)
    _text(img, VIDEO_SUBTITLE, (int(10 * scale) + 4, int(layout["header_h"] * 0.85)),
          scale * 0.62, AMBER, max(1, thick - 1))
    title_w = int(cv2.getTextSize(VIDEO_SUBTITLE, cv2.FONT_HERSHEY_SIMPLEX,
                                  scale * 0.62, max(1, thick - 1))[0][0])
    frame_w = int(cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.68,
                                  thick)[0][0])
    fx = max(int(10 * scale) + 20 + title_w, width - frame_w - int(14 * scale))
    _text(img, frame_text, (fx, int(layout["header_h"] * 0.82)), scale * 0.68, MUTED,
          max(1, thick - 1))

    _translucent(img, layout["panel_x"], layout["panel_y"],
                 layout["panel_x"] + layout["panel_w"], height - layout["legend_h"],
                 PANEL_BG, 0.55)
    pad = int(10 * scale) + 4
    text_w = layout["panel_w"] - 2 * pad
    banner_h = int(max(22, 0.030 * height))
    _translucent(img, layout["panel_x"], layout["panel_y"],
                 layout["panel_x"] + layout["panel_w"],
                 layout["panel_y"] + banner_h, PANEL_BG, 0.85)
    _fit_text(img, phase_text, layout["panel_x"] + pad,
              layout["panel_y"] + int(banner_h * 0.75), scale * 0.66, TEXT, thick,
              text_w)
    y = layout["panel_y"] + banner_h + int(22 * scale)
    for line in analytics_lines:
        if y > height - layout["legend_h"] - int(10 * scale):
            break
        color = MUTED if line.startswith((" ", "-")) else TEXT
        _fit_text(img, line, layout["panel_x"] + pad, y, scale * 0.60, color,
                  max(1, thick - 1), text_w)
        y += int(22 * scale)

    _translucent(img, 0, height - layout["legend_h"], width, height, PANEL_BG, 0.80)
    ly = height - int(layout["legend_h"] * 0.35)
    legend_w = width - 2 * int(10 * scale) - 8
    if operator_note:
        caveat_w = int(cv2.getTextSize(VIDEO_CAVEAT, cv2.FONT_HERSHEY_SIMPLEX,
                                       scale * 0.52, max(1, thick - 1))[0][0])
        legend_w -= caveat_w + int(20 * scale)
    _fit_text(img, VIDEO_LEGEND, int(10 * scale) + 4, ly, scale * 0.52, MUTED,
              max(1, thick - 1), max(80, legend_w))
    if operator_note:
        size = cv2.getTextSize(VIDEO_CAVEAT, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.52,
                               max(1, thick - 1))[0][0]
        _text(img, VIDEO_CAVEAT, (max(int(10 * scale) + 4, width - size - int(10 * scale)),
                                  height - layout["legend_h"] - int(6 * scale)),
              scale * 0.52, AMBER, max(1, thick - 1))


def _wrap(text: str, width: int) -> List[str]:
    """Word-wrap for cv2 text: no mid-word cuts, no overflowing lines."""
    words = str(text).split()
    lines: List[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > max(8, width):
            if current:
                lines.append(current)
            current = word
        else:
            current = (current + " " + word).strip()
    if current:
        lines.append(current)
    return lines


def draw_card(frame_img, *, title: str, label: str, lines: Sequence[str],
              card_width: int = 640) -> np.ndarray:
    """One keyframe card: the annotated frame plus a labelled strip underneath."""
    h, w = frame_img.shape[:2]
    strip_h = 118
    scale = card_width / float(w)
    body = cv2.resize(frame_img, (card_width, max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_AREA)
    card = np.zeros((body.shape[0] + strip_h, card_width, 3), dtype=np.uint8)
    card[:body.shape[0], :, :] = body
    card[body.shape[0]:, :, :] = PANEL_BG
    fs = max(0.42, min(0.72, card_width / 1100.0))
    thick = max(1, int(round(card_width / 700.0)))
    _text(card, title, (12, body.shape[0] + int(30 * fs) + 6), fs * 1.15, TEXT, thick)
    label_color = AMBER if label != "OPERATOR-MARKED" else TEAL
    size = int(cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs * 0.92, thick)[0][0])
    _text(card, label, (card_width - size - 12, body.shape[0] + int(30 * fs) + 6),
          fs * 0.92, label_color, thick)
    # Wrapped to the card width so a long note cannot run off the edge.
    char_width = max(30, int(card_width / (fs * 12.0)))
    wrapped: List[str] = []
    for line in lines:
        wrapped.extend(_wrap(line, char_width))
    y = body.shape[0] + int(58 * fs) + 6
    for line in wrapped[:3]:
        _text(card, line, (12, y), fs * 0.88, MUTED, max(1, thick - 1))
        y += int(24 * fs)
    return card


def draw_unavailable_card(*, title: str, reason: str, card_width: int = 640,
                          height: int = 360) -> np.ndarray:
    card = np.full((height, card_width, 3), SHEET_BG, dtype=np.uint8)
    _text(card, title, (14, 40), 0.72, TEXT, 2)
    _text(card, "UNAVAILABLE", (14, 78), 0.62, AMBER, 2)
    y = 112
    for line in _wrap(reason, 78)[:6]:
        _text(card, line, (14, y), 0.50, MUTED, 1)
        y += 22
    return card


def contact_sheet(cards: Sequence[np.ndarray], *, columns: int = 2,
                  gap: int = 10) -> np.ndarray:
    """A 2x2 panel grid of the four primary cards (or however many were given)."""
    if not cards:
        return np.full((360, 640, 3), SHEET_BG, dtype=np.uint8)
    width = max(c.shape[1] for c in cards)
    rows: List[np.ndarray] = []
    for start in range(0, len(cards), columns):
        row_cards = list(cards[start:start + columns])
        row_h = max(c.shape[0] for c in row_cards)
        row = np.full((row_h, width * len(row_cards) + gap * (len(row_cards) - 1), 3),
                      SHEET_BG, dtype=np.uint8)
        x = 0
        for card in row_cards:
            row[:card.shape[0], x:x + card.shape[1]] = card
            x += card.shape[1] + gap
        rows.append(row)
    total_h = sum(r.shape[0] for r in rows) + gap * (len(rows) - 1)
    sheet = np.full((total_h, width * columns + gap * (columns - 1), 3), SHEET_BG,
                    dtype=np.uint8)
    y = 0
    for row in rows:
        sheet[y:y + row.shape[0], :row.shape[1]] = row
        y += row.shape[0] + gap
    return sheet


def write_png(path: str, image: np.ndarray) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return bool(cv2.imwrite(path, image))
    except Exception:
        return False
