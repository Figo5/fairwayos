"""Showcase chrome renderer: cv2 drawing, optional Pillow typography.

Only drawing lives here. Everything about *what* to draw and *where* is in
:mod:`ghostcaddie.demo.showcase`, which is stdlib-only and tested without cv2, so a
layout or wording change cannot silently disagree with the renderer.

Compositing model, and why it is two blends instead of draw-in-place:

1. the picture is the source frame at output size, untouched in geometry: no gutter,
   no letterbox, no sidebar;
2. an OVERLAY layer holds the skeleton, the trails, the markers and the impact
   accent, each with its own alpha in a coverage mask, and is blended onto the
   picture once -- this is what gives the skeleton "restrained opacity" without
   per-line colour arithmetic;
3. a CHROME layer holds the translucent cards, the pills and the text, blended on
   top, so a card dims what is under it instead of being crossed by it.

Typography: Pillow with a system TrueType font when the worker interpreter has it
(real antialiasing and the disclosure's middle dot), otherwise cv2's Hershey faces
with ``LINE_AA``, which are antialiased too but ASCII-only. The engine that was used
is reported in the run's debug record rather than assumed.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import showcase as sc

try:  # pragma: no cover - import guard, exercised by whichever branch is taken
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except Exception:  # pragma: no cover
    PIL_AVAILABLE = False
    Image = ImageDraw = ImageFont = None  # type: ignore

# BGR
TEAL = (176, 212, 112)
TEAL_SOFT = (128, 158, 88)
JOINT = (238, 246, 232)
CLUBHEAD = (48, 176, 255)
CLUB_TRAIL = (40, 140, 220)
BALL = (250, 250, 250)
BALL_RING = (24, 26, 28)
BALL_TRAIL = (96, 214, 255)
ACCENT = (96, 214, 255)
TEXT = (246, 246, 246)
TEXT_MUTED = (188, 196, 202)
TEXT_FAINT = (140, 148, 154)
TEXT_SHADOW = (10, 12, 14)
CARD_BG = (22, 24, 26)
CARD_EDGE = (86, 96, 100)
ACCENT_RULE = (196, 224, 128)
SHEET_BG = (14, 15, 16)

SKELETON_ALPHA = 0.58
TRAIL_ALPHA = 0.72
MARKER_ALPHA = 1.0
CLUB_TRAIL_ALPHA = 0.85
CARD_ALPHA = 0.58
CARD_ALPHA_MISSING = 0.38
PILL_ALPHA = 0.42
TEXT_ALPHA = 1.0

SKELETON_EDGES = [("sh_l", "sh_r"), ("sh_l", "el_l"), ("el_l", "wr_l"),
                  ("sh_r", "el_r"), ("el_r", "wr_r"), ("sh_l", "hip_l"),
                  ("sh_r", "hip_r"), ("hip_l", "hip_r"), ("hip_l", "kn_l"),
                  ("kn_l", "ank_l"), ("hip_r", "kn_r"), ("kn_r", "ank_r")]
JOINT_NAMES = ("sh_l", "sh_r", "el_l", "el_r", "wr_l", "wr_r", "hip_l", "hip_r",
               "kn_l", "kn_r", "ank_l", "ank_r")

#: System fonts, best first. Only read, never installed.
FONT_CANDIDATES_REGULAR = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
FONT_CANDIDATES_BOLD = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Helvetica-Bold.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


def _first_existing(paths: Sequence[str]) -> str:
    for path in paths:
        if os.path.isfile(path):
            return path
    return ""


class TextEngine:
    """Draw text with the best engine this interpreter has, and say which one.

    Pillow gives real antialiasing, kerning and the full character set; cv2's Hershey
    fonts are the portable fallback and force ASCII. The object reports ``engine`` and
    ``font_path`` so the run can record what actually rendered the cards.
    """

    def __init__(self) -> None:
        self.engine = "hershey"
        self.font_path = ""
        self.bold_path = ""
        self._cache: Dict[Tuple[int, bool], Any] = {}
        if not PIL_AVAILABLE:
            return
        regular = _first_existing(FONT_CANDIDATES_REGULAR)
        bold = _first_existing(FONT_CANDIDATES_BOLD)
        if not regular:
            return
        self.engine = "pillow"
        self.font_path = regular
        self.bold_path = bold or regular

    @property
    def unicode_ok(self) -> bool:
        return self.engine == "pillow"

    def _font(self, size: int, bold: bool):
        key = (max(6, int(size)), bool(bold))
        font = self._cache.get(key)
        if font is None:
            path = self.bold_path if bold else self.font_path
            font = ImageFont.truetype(path, key[0])
            self._cache[key] = font
        return font

    def measure(self, text: str, size: int, *, bold: bool = False) -> Tuple[int, int]:
        """Text extent in pixels for the engine that will draw it."""
        value = sc.display_text(text, unicode_ok=self.unicode_ok)
        if self.engine == "pillow":
            left, top, right, bottom = self._font(size, bold).getbbox(value)
            return int(max(1, right - left)), int(max(1, bottom - top))
        scale = max(0.24, size / 30.0)
        (width, height), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_DUPLEX, scale, 1)
        return int(max(1, width)), int(max(1, height))

    def fit_size(self, text: str, box_w: int, size: int, *, bold: bool = False,
                 minimum: int = 7) -> int:
        """Shrink ``size`` until the text fits ``box_w``, down to ``minimum``."""
        candidate = max(minimum, int(size))
        while candidate > minimum and self.measure(text, candidate, bold=bold)[0] > box_w:
            candidate -= 1
        return candidate

    def draw(self, layer: np.ndarray, mask: np.ndarray, text: str, xy: Tuple[int, int],
             size: int, color: Tuple[int, int, int], *,
             bold: bool = False, alpha: float = TEXT_ALPHA) -> Tuple[int, int]:
        """Paste text with its top-left at ``xy``; returns the drawn extent."""
        value = sc.display_text(text, unicode_ok=self.unicode_ok)
        if not value:
            return 0, 0
        if self.engine == "pillow":
            width, height = self.measure(value, size, bold=bold)
            pad = 2
            image = Image.new("RGBA", (width + 2 * pad, height + 2 * pad), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            font = self._font(size, bold)
            left, top, _, _ = font.getbbox(value)
            # Pillow's draw.text positions by the ascender box; subtracting the
            # glyph's own bbox origin lands the ink exactly on the requested rect.
            draw.text((pad - left, pad - top), value, font=font,
                      fill=(int(color[2]), int(color[1]), int(color[0]), 255))
            arr = np.array(image)
            rgb = arr[:, :, :3]
            alpha_img = (arr[:, :, 3].astype(np.float32) / 255.0) * float(alpha)
            _paste(layer, mask, rgb, alpha_img, xy[0] - pad, xy[1] - pad, swap=True)
            return width, height
        scale = max(0.24, size / 30.0)
        (width, height), baseline = cv2.getTextSize(value, cv2.FONT_HERSHEY_DUPLEX,
                                                    scale, 1)
        local = np.zeros((height + baseline + 6, width + 8, 3), np.uint8)
        cv2.putText(local, value, (4, height + 2), cv2.FONT_HERSHEY_DUPLEX, scale,
                    color, 1, cv2.LINE_AA)
        selected = local.any(axis=2)
        if not selected.any():
            return 0, 0
        alpha_img = selected.astype(np.float32) * float(alpha)
        _paste(layer, mask, local, alpha_img, xy[0], xy[1], swap=False)
        return width, height


# ------------------------------------------------------------------ compositing

def _window(x: int, y: int, w: int, h: int, width: int, height: int
            ) -> Optional[Tuple[int, int, int, int, int, int]]:
    """Clip a request to the frame, keeping BOTH the source and target offsets.

    A raster (a text tile) can start above or left of the frame; without the source
    offset its visible part would be taken from the wrong side of the tile. Returns
    ``(source_x, source_y, dest_x, dest_y, w, h)`` or None when nothing is visible.
    """
    x, y, w, h = int(x), int(y), int(w), int(h)
    dx = max(0, x)
    dy = max(0, y)
    sx = max(0, -x)
    sy = max(0, -y)
    dw = min(w - sx, width - dx)
    dh = min(h - sy, height - dy)
    if dw <= 0 or dh <= 0:
        return None
    return sx, sy, dx, dy, dw, dh


def _paste(layer: np.ndarray, mask: np.ndarray, rgb: np.ndarray,
           alpha_img: np.ndarray, x: int, y: int, *, swap: bool) -> None:
    """Blend a small RGBA raster into ``layer``/``mask`` at ``(x, y)``.

    ``swap`` says the raster is RGB and the layer is BGR. Alpha is combined with
    ``maximum`` rather than addition: overlapping chrome must not accumulate into an
    opaque blob where a card and a marker meet.
    """
    height, width = mask.shape[:2]
    window = _window(x, y, rgb.shape[1], rgb.shape[0], width, height)
    if window is None:
        return
    sx, sy, dx, dy, w, h = window
    piece = rgb[sy:sy + h, sx:sx + w]
    alpha_piece = alpha_img[sy:sy + h, sx:sx + w]
    selected = alpha_piece > 0.002
    if not selected.any():
        return
    colors = piece[:, :, ::-1] if swap else piece
    sub_layer = layer[dy:dy + h, dx:dx + w]
    sub_mask = mask[dy:dy + h, dx:dx + w]
    sub_layer[selected] = colors[selected]
    np.maximum(sub_mask, alpha_piece, out=sub_mask)


def blend_float(picture: np.ndarray, layer: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Alpha composite in float32, one pass. uint8 in, uint8 out."""
    top = float(mask.max())
    if top <= 0.0:
        return picture
    alpha = mask if top <= 1.0 else (mask / top)
    alpha3 = cv2.merge([alpha, alpha, alpha])
    out = (layer.astype(np.float32) * alpha3
           + picture.astype(np.float32) * (1.0 - alpha3))
    return out.astype(np.uint8)


def new_layers(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    return (np.zeros((height, width, 3), np.uint8),
            np.zeros((height, width), np.float32))


def filled_rect(layer: np.ndarray, mask: np.ndarray, x: int, y: int, w: int, h: int,
                color, alpha: float, *, radius: int = 0) -> None:
    """One rectangle (optionally rounded) into the layer/mask pair."""
    height, width = mask.shape[:2]
    window = _window(x, y, w, h, width, height)
    if window is None:
        return
    sx, sy, dx, dy, w, h = window
    if radius <= 0:
        local = np.full((h, w), 255, np.uint8)
    else:
        radius = int(min(radius, w // 2, h // 2))
        local = np.zeros((h, w), np.uint8)
        cv2.rectangle(local, (radius, 0), (w - radius - 1, h - 1), 255, -1)
        cv2.rectangle(local, (0, radius), (w - 1, h - radius - 1), 255, -1)
        for cx, cy in ((radius, radius), (w - radius - 1, radius),
                       (radius, h - radius - 1), (w - radius - 1, h - radius - 1)):
            cv2.circle(local, (cx, cy), radius, 255, -1)
    selected = local > 0
    sub_layer = layer[dy:dy + h, dx:dx + w]
    sub_mask = mask[dy:dy + h, dx:dx + w]
    sub_layer[selected] = color
    np.maximum(sub_mask, np.float32(alpha) * (local.astype(np.float32) / 255.0),
               out=sub_mask)


def _stack_line(layer, mask, color, alpha, thickness, p0, p1) -> None:
    height, width = mask.shape[:2]
    local = np.zeros((height, width), np.uint8)
    cv2.line(local, p0, p1, 255, max(1, int(thickness)), cv2.LINE_AA)
    selected = local > 0
    if not selected.any():
        return
    layer[selected] = color
    np.maximum(mask, np.float32(alpha) * selected.astype(np.float32), out=mask)


def line(layer: np.ndarray, mask: np.ndarray, p0, p1, color, alpha: float,
         thickness: int = 1) -> None:
    _stack_line(layer, mask, color, alpha, thickness,
                (int(round(p0[0])), int(round(p0[1]))),
                (int(round(p1[0])), int(round(p1[1]))))


def dashed_line(layer: np.ndarray, mask: np.ndarray, p0, p1, color, alpha: float,
                thickness: int = 1, dash: int = 7) -> None:
    """Dashed where a point is derived, so it never reads as an observation."""
    import math

    ax, ay = float(p0[0]), float(p0[1])
    bx, by = float(p1[0]), float(p1[1])
    length = math.hypot(bx - ax, by - ay)
    if length < 1:
        return
    steps = max(1, int(length // max(3, dash)))
    for i in range(steps):
        if i % 2:
            continue
        t0, t1 = i / steps, min(1.0, (i + 0.75) / steps)
        line(layer, mask, (ax + (bx - ax) * t0, ay + (by - ay) * t0),
             (ax + (bx - ax) * t1, ay + (by - ay) * t1), color, alpha, thickness)


def circle(layer: np.ndarray, mask: np.ndarray, center, radius: int, color,
           alpha: float, *, thickness: int = -1) -> None:
    height, width = mask.shape[:2]
    local = np.zeros((height, width), np.uint8)
    cv2.circle(local, (int(round(center[0])), int(round(center[1]))),
               max(1, int(radius)), 255, thickness, cv2.LINE_AA)
    selected = local > 0
    if not selected.any():
        return
    layer[selected] = color
    np.maximum(mask, np.float32(alpha) * selected.astype(np.float32), out=mask)


# ------------------------------------------------------------------ overlays
#
# Coordinates reaching these functions are ALREADY in the output frame's pixels: the
# caller applies the coordinate scale, and the ``style`` factor below only sizes line
# weights and markers in DISPLAY pixels (frame height / 1080). Keeping the two
# separate is what stops a 4K->1080 render from drawing 5px markers, which is what
# tying the marker size to the coordinate scale did.

STYLE_REFERENCE_HEIGHT = 1080.0


def style_scale(frame_height: int) -> float:
    """Display-pixel factor for line weights and markers at this output height."""
    return max(0.4, min(4.0, float(frame_height) / STYLE_REFERENCE_HEIGHT))


def _style_px(value: float, minimum: int = 1) -> int:
    return int(max(minimum, round(value)))


def draw_skeleton(layer: np.ndarray, mask: np.ndarray, keypoints: Sequence[dict],
                  style: float) -> int:
    """Teal skeleton, small joints, no labels. Returns the joints drawn.

    At 1080p this is a 3px line with ~2.4px joints: visible over 4K footage without
    turning the golfer into a diagram.
    """
    visible = {p["name"]: p for p in keypoints if p.get("visible")}
    if not visible:
        return 0
    thickness = _style_px(3.0 * style)
    for a, b in SKELETON_EDGES:
        if a in visible and b in visible:
            line(layer, mask, (visible[a]["x"], visible[a]["y"]),
                 (visible[b]["x"], visible[b]["y"]), TEAL, SKELETON_ALPHA, thickness)
    drawn = 0
    for name in JOINT_NAMES:
        point = visible.get(name)
        if point is None:
            continue
        circle(layer, mask, (point["x"], point["y"]), _style_px(2.4 * style), JOINT,
               SKELETON_ALPHA)
        drawn += 1
    return drawn


def draw_trail(layer: np.ndarray, mask: np.ndarray, points: Sequence[dict],
               color, *, alpha: float, thickness: int = 1, dot: int = 0,
               taper: bool = True, dash: int = 6) -> None:
    """Trail for ONE segment, in output pixels, under the marker. Never bridged.

    Two presentation rules, both of them about not over-claiming:

    * an edge with a derived endpoint is DASHED, so an interpolated stretch is visibly
      different from an observed one;
    * the trail TAPERS: the oldest edge is thinner and fainter than the edge next to the
      marker, so a short trail reads as a direction of travel rather than as a drawn
      shape.

    The caller passes the trail window for one segment only (see
    ``tracking.trail_window``), so nothing here can join two segments.
    """
    if len(points) < 2:
        return
    span = max(1, len(points) - 1)
    for index, (a, b) in enumerate(zip(points, points[1:])):
        if a.get("x") is None or b.get("x") is None:
            continue
        pa = (a["x"], a["y"])
        pb = (b["x"], b["y"])
        weight = 1.0 if not taper else 0.45 + 0.55 * ((index + 1) / float(span))
        edge_alpha = float(alpha) * weight
        edge_thickness = max(1, int(round(int(thickness) * (0.6 + 0.4 * weight))))
        if a.get("status") == "interpolated" or b.get("status") == "interpolated":
            dashed_line(layer, mask, pa, pb, color, edge_alpha * 0.9, edge_thickness,
                        dash=dash)
        else:
            line(layer, mask, pa, pb, color, edge_alpha, edge_thickness)
    if dot:
        for point in points:
            if point.get("x") is None:
                continue
            circle(layer, mask, (point["x"], point["y"]), dot, color, alpha * 0.55)


def draw_club_marker(layer: np.ndarray, mask: np.ndarray, point: dict,
                     style: float) -> None:
    """The prominent one: an amber chevron with a bright outline.

    The outline is drawn into its OWN mask. Drawing it into the mask that was just
    filled and then colouring every selected pixel white is what turned the whole
    marker white in the first render: the fill and the outline have to be separate
    passes over separate masks, or the later colour wins everywhere.
    """
    if point.get("x") is None or point.get("status") in ("missing", "lost"):
        return
    x, y = float(point["x"]), float(point["y"])
    derived = point.get("status") == "interpolated"
    size = 11.0 * style
    polygon = np.array([[[x, y - size], [x + size * 0.86, y], [x, y + size * 1.15],
                         [x - size * 0.86, y]]], dtype=np.int32)
    alpha = 0.5 if derived else CLUB_TRAIL_ALPHA
    height, width = mask.shape[:2]

    fill = np.zeros((height, width), np.uint8)
    cv2.fillPoly(fill, polygon, 255)
    selected = fill > 0
    if selected.any():
        layer[selected] = CLUBHEAD
        np.maximum(mask, np.float32(alpha) * selected.astype(np.float32), out=mask)

    outline = np.zeros((height, width), np.uint8)
    cv2.polylines(outline, polygon, True, 255, _style_px(1.8 * style), cv2.LINE_AA)
    # the outline is the polygon's edge: subtract the eroded fill so the bright ring
    # is a ring and the amber interior survives
    edge = (outline > 0) & ~(cv2.erode(fill, np.ones((3, 3), np.uint8)) > 0)
    if edge.any():
        layer[edge] = (255, 255, 255)
        np.maximum(mask, np.float32(MARKER_ALPHA) * edge.astype(np.float32), out=mask)


def draw_ball_marker(layer: np.ndarray, mask: np.ndarray, point: dict,
                     style: float) -> None:
    """Small, precise, high contrast: a HOLLOW ring around the ball.

    A filled disc hides the thing it claims to mark. A ring with a dark seat ring just
    outside it leaves the ball visible through the middle, which is also the honest
    picture: the operator's point is ON the ball, not in place of it.
    """
    if point.get("x") is None or point.get("status") in ("missing", "lost"):
        return
    x, y = float(point["x"]), float(point["y"])
    derived = point.get("status") == "interpolated"
    radius = _style_px((5.5 if derived else 7.5) * style, 3)
    stroke = _style_px(3.0 * style, 2)
    alpha = 0.55 if derived else 0.95
    # The dark seat ring is what makes the marker findable at feed size, where a thin
    # ring on a white ball on green disappears: it is drawn a touch heavier than the
    # bright ring so the marker still reads while the ball stays visible through it.
    circle(layer, mask, (x, y), radius + stroke - 1, BALL_RING, 1.0 if not derived else 0.8,
           thickness=_style_px(2.5 * style, 1))
    circle(layer, mask, (x, y), radius, BALL, alpha, thickness=stroke)


def draw_impact_accent(layer: np.ndarray, mask: np.ndarray, point: Optional[dict],
                       style: float, strength: float) -> None:
    """A short accent ring at the operator's impact marker. Bounded, never a claim."""
    if strength <= 0.0 or not point or point.get("x") is None:
        return
    x, y = float(point["x"]), float(point["y"])
    radius = _style_px((18.0 + 26.0 * strength) * style, 4)
    circle(layer, mask, (x, y), radius, ACCENT, 0.34 * strength,
           thickness=_style_px(1.6 * style))
    circle(layer, mask, (x, y), _style_px(3.0 * style, 2), ACCENT, 0.5 * strength)


# ------------------------------------------------------------------ chrome

def draw_card(layer: np.ndarray, mask: np.ndarray, rect: sc.Rect, card: Dict[str, Any],
              engine: TextEngine, fonts: Dict[str, int]) -> None:
    """One metric card: accent rule, small label, one large value.

    There is deliberately no caption sub-line: the metric definitions belong in the
    report, and a third dense line is what makes an otherwise clean strip read as
    debug output. A value that is unavailable keeps the card and its label and shows
    the placeholder, dimmed, so the absence is visible rather than filled in.

    A card whose value is HELD (``card["held"]``) is a value measured at an earlier
    frame and shown across a flicker of fewer than ``card_debounce_frames`` frames: it
    is drawn dimmer than a measured value, and the provenance records exactly which
    frames were held.
    """
    available = bool(card.get("available"))
    held = bool(card.get("held"))
    radius = max(2, int(rect.h * 0.12))
    # A clean translucent rounded fill and NOTHING else: no outline and no corner
    # circles. The first render drew the border with four full circles at the corners,
    # which read as debug handles on the picture.
    filled_rect(layer, mask, rect.x, rect.y, rect.w, rect.h, CARD_BG,
                CARD_ALPHA if available else CARD_ALPHA_MISSING, radius=radius)
    rule_h = max(2, int(rect.h * 0.05))
    filled_rect(layer, mask, rect.x + radius, rect.y + max(1, int(rect.h * 0.035)),
                max(6, rect.w - 2 * radius), rule_h, ACCENT_RULE,
                (0.65 if held else 0.85) if available else 0.35, radius=rule_h)
    pad_x = max(4, int(rect.w * 0.05))
    inner_w = max(12, rect.w - 2 * pad_x)
    label = card.get("label", "")
    label_size = engine.fit_size(label, inner_w, fonts["label"], bold=True,
                                 minimum=max(7, fonts["label"] - 6))
    label_y = rect.y + max(3, int(rect.h * 0.16))
    engine.draw(layer, mask, label, (rect.x + pad_x, label_y), label_size,
                TEXT_MUTED if available else TEXT_FAINT, bold=True)
    value = card.get("value", sc.PLACEHOLDER)
    value_size = engine.fit_size(value, inner_w, fonts["value"], bold=True,
                                 minimum=max(10, fonts["value"] - 10))
    value_y = label_y + engine.measure(label, label_size, bold=True)[1] + \
        max(0, int(rect.h * 0.02))
    # A placeholder is two dashes: its ink is only a few pixels tall, and drawing that
    # ink at the top of the value line tucked it under the label, where it read as a
    # smudge. Bottom-aligning short-ink strings to the digit box puts the dashes on the
    # value's own baseline and leaves every normal value exactly where it was.
    reference_h = engine.measure("0", value_size, bold=True)[1]
    ink_h = engine.measure(value, value_size, bold=True)[1]
    value_y += max(0, reference_h - ink_h)
    color = TEXT if available and not held else (TEXT_MUTED if available else TEXT_FAINT)
    engine.draw(layer, mask, value, (rect.x + pad_x, value_y), value_size,
                color, bold=True)


def draw_pill(layer: np.ndarray, mask: np.ndarray, rect: sc.Rect, text: str,
              engine: TextEngine, font: int, *, align: str = "center",
              color: Tuple[int, int, int] = TEXT, bold: bool = True,
              alpha: float = PILL_ALPHA) -> None:
    radius = max(2, rect.h // 2)
    filled_rect(layer, mask, rect.x, rect.y, rect.w, rect.h, CARD_BG, alpha,
                radius=radius)
    size = engine.fit_size(text, max(10, rect.w - int(rect.h * 0.5)), font,
                           bold=bold, minimum=max(7, font - 6))
    width, height = engine.measure(text, size, bold=bold)
    if align == "center":
        x = rect.x + max(2, (rect.w - width) // 2)
    else:
        x = rect.x + max(2, int(rect.h * 0.25))
    y = rect.y + max(1, (rect.h - height) // 2)
    engine.draw(layer, mask, text, (x, y), size, color, bold=bold)


def stage_overlays(picture: np.ndarray, *, overlay_draw=None) -> np.ndarray:
    """Blend the overlay layer (skeleton, trails, markers, accent) onto the picture.

    Separated from the chrome so a still can be taken from the exact staged frame the
    video showed, and so the still cannot be composed from a differently drawn
    overlay than the one in the video.
    """
    height, width = picture.shape[:2]
    layer, mask = new_layers(width, height)
    if overlay_draw is not None:
        overlay_draw(layer, mask)
    return blend_float(picture, layer, mask)


def compose_chrome(picture: np.ndarray, *, layout: sc.ShowcaseLayout, chrome: dict,
                   engine: Optional[TextEngine] = None,
                   mode: str = "video") -> Tuple[np.ndarray, Dict[str, Any]]:
    """Draw the cards, brand, phase badge and disclosure onto a staged picture."""
    height, width = picture.shape[:2]
    if (width, height) != (layout.width, layout.height):
        raise ValueError(f"layout is {layout.width}x{layout.height} but the picture is "
                         f"{width}x{height}: layout and render must share pixels")
    engine = engine or TextEngine()
    drawn_badge_px = 0
    layer, mask = new_layers(width, height)
    for rect, card in zip(layout.cards, chrome.get("cards") or []):
        draw_card(layer, mask, rect, card, engine, {
            "value": layout.card_value_font, "label": layout.card_label_font,
            "sub": layout.card_sub_font})
    brand = chrome.get("brand") or ""
    brand_sub = chrome.get("brand_sub") or ""
    # A small dark drop shadow under the brand text. The mint sub-brand was nearly
    # illegible on pale sky, and a shadow is the fix that keeps the look while making
    # the type readable at feed size (a chip would have added a second panel). The
    # sub-brand is white at ~75%: mint disappeared into pale sky entirely.
    shadow = max(1, int(round(1.6 * (height / 1080.0))))

    def draw_brand_text(text: str, xy, size: int, color, *, bold: bool,
                        alpha: float = TEXT_ALPHA) -> None:
        engine.draw(layer, mask, text, (xy[0] + shadow, xy[1] + shadow), size,
                    TEXT_SHADOW, bold=bold, alpha=0.7)
        engine.draw(layer, mask, text, (xy[0], xy[1]), size, color, bold=bold,
                    alpha=alpha)

    if brand:
        size = engine.fit_size(brand, layout.brand.w, layout.brand_font, bold=True)
        draw_brand_text(brand, (layout.brand.x, layout.brand.y), size, TEXT, bold=True)
        drawn = engine.measure(brand, size, bold=True)
        if brand_sub:
            sub_size = engine.fit_size(brand_sub, layout.brand.w, layout.brand_sub_font,
                                       bold=False)
            draw_brand_text(brand_sub, (layout.brand.x, layout.brand.y + drawn[1] + 2),
                            sub_size, TEXT, bold=False, alpha=0.75)
    phase = chrome.get("phase") or ""
    if phase:
        badge = layout.phase_badge
        # Content-fit, right-aligned: the pill is as wide as its own text plus a
        # little padding, not a fixed fraction of the frame. A 420px pill around the
        # word IMPACT was one of the things the operator's preview flagged.
        text_w = engine.measure(phase, layout.badge_font, bold=True)[0]
        pad = max(6, int(badge.h * 0.55))
        badge_w = min(layout.width - 2 * layout.margin, text_w + 2 * pad)
        rect = sc.Rect(badge.x + badge.w - badge_w, badge.y, badge_w, badge.h)
        # One pill colour everywhere: the video and the stills used to differ (gold vs
        # white) for the same badge, which is a difference a viewer cannot explain.
        draw_pill(layer, mask, rect, phase, engine, layout.badge_font, color=ACCENT)
        drawn_badge_px = badge_w
    disclosure = chrome.get("disclosure") or ""
    if disclosure:
        size = engine.fit_size(disclosure, layout.disclosure.w, layout.disclosure_font,
                               minimum=6)
        measure = engine.measure(disclosure, size)
        disc_y = layout.disclosure.y + max(0, (layout.disclosure.h - measure[1]) // 2)
        # The disclosure is the line that carries the honesty story into a feed, and
        # grey on sunlit grass is close to invisible there: it gets the same drop shadow
        # treatment as the brand, and a brighter grey.
        engine.draw(layer, mask, disclosure,
                    (layout.disclosure.x + shadow, disc_y + shadow), size, TEXT_SHADOW,
                    alpha=0.7)
        engine.draw(layer, mask, disclosure, (layout.disclosure.x, disc_y), size,
                    TEXT_MUTED)
    out = blend_float(picture, layer, mask)
    return out, {"text_engine": engine.engine, "font_path": engine.font_path,
                 "phase_badge": phase,
                 # the pill's drawn width, so a report can state it instead of
                 # quoting the layout's maximum slot width
                 "phase_badge_px": drawn_badge_px,
                 "cards": [c.get("metric") for c in (chrome.get("cards") or [])],
                 "layout_problems": sc.layout_problems(layout)}


def compose_frame(picture: np.ndarray, *, layout: sc.ShowcaseLayout, chrome: dict,
                  overlay_draw=None, engine: Optional[TextEngine] = None,
                  mode: str = "video") -> Tuple[np.ndarray, Dict[str, Any]]:
    """Picture -> overlays -> chrome. Returns the frame and a drawing record.

    ``overlay_draw`` is called with the overlay layer/mask so the caller keeps its
    own object model (tracks, pose records) out of this module.
    """
    staged = stage_overlays(picture, overlay_draw=overlay_draw)
    image, record = compose_chrome(staged, layout=layout, chrome=chrome, engine=engine,
                                  mode=mode)
    record["overlay_elements"] = int(getattr(overlay_draw, "drawn", 0) or 0)
    return image, record


def write_png(path: str, image: np.ndarray) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return bool(cv2.imwrite(path, image))
    except Exception:
        return False


def contact_sheet(panels: Sequence[Tuple[str, np.ndarray]], *, columns: int = 2,
                  gap: int = 14, title: str = "", subtitle: str = "",
                  disclaimer: str = "", max_width: Optional[int] = None) -> np.ndarray:
    """The LinkedIn sheet: the phase panels on one grid with a real header band.

    The header is MEASURED before it is sized: the band is built from the text that goes
    into it plus deliberate padding, so a sub-line cannot straddle the band edge and land
    across a panel's sky (which is what the first sheet did). Every per-run number —
    frame indices, metrics, provenance — stays in the report and the JSON: the graphic
    carries the brand, each panel's own phase badge and one disclosure line in the
    footer.

    ``max_width`` optionally downscales the finished sheet for a feed; it is applied
    after compositing so nothing is drawn at a scale it was not measured for.
    """
    panels = [(name, image) for name, image in panels if image is not None]
    if not panels:
        return np.full((360, 640, 3), SHEET_BG, np.uint8)
    cell_w = max(image.shape[1] for _, image in panels)
    cell_h = max(image.shape[0] for _, image in panels)
    rows = (len(panels) + columns - 1) // columns
    width = columns * cell_w + (columns + 1) * gap
    # The header and footer text is inset from the sheet edge in proportion to a panel,
    # not by the panel gap: on a 3882px sheet a 10px inset put the title in the corner.
    pad = max(20, int(round(cell_w * 0.02)))
    band_pad = max(8, int(round(cell_h * 0.035)))
    engine = TextEngine()

    title_size = 0
    title_h = 0
    sub_size = 0
    sub_h = 0
    lead = 0
    if title or subtitle:
        content_w = max(40, width - 2 * pad)
        if title:
            title_size = engine.fit_size(title, content_w,
                                         max(20, int(cell_h * 0.075)), bold=True)
            title_h = engine.measure(title, title_size, bold=True)[1]
        if subtitle:
            sub_size = engine.fit_size(subtitle, content_w,
                                       max(13, int((title_size or 24) * 0.55)))
            sub_h = engine.measure(subtitle, sub_size)[1]
            lead = max(6, int(round(cell_h * 0.03))) if title else 0
    head = (band_pad + title_h + lead + sub_h + band_pad) if (title or subtitle) else 0
    foot = 0
    disc_size = 0
    disc_h = 0
    if disclaimer:
        disc_size = engine.fit_size(disclaimer, max(40, width - 2 * pad),
                                    max(11, int(cell_h * 0.03)), minimum=8)
        disc_h = engine.measure(disclaimer, disc_size)[1]
        foot = band_pad + disc_h + band_pad

    height = head + rows * cell_h + (rows + 1) * gap + foot
    sheet = np.full((height, width, 3), SHEET_BG, np.uint8)
    layer, mask = new_layers(width, height)
    if title:
        engine.draw(layer, mask, title, (pad, band_pad), title_size, TEXT, bold=True)
    if subtitle:
        engine.draw(layer, mask, subtitle, (pad, band_pad + title_h + lead), sub_size,
                    ACCENT_RULE)
    for index, (name, image) in enumerate(panels):
        row, column = divmod(index, columns)
        x = gap + column * (cell_w + gap)
        y = head + gap + row * (cell_h + gap)
        h, w = image.shape[:2]
        sheet[y:y + h, x:x + w] = image
    if disclaimer:
        engine.draw(layer, mask, disclaimer, (pad, height - foot + band_pad), disc_size,
                    TEXT_FAINT)
    composed = blend_float(sheet, layer, mask)
    if max_width and composed.shape[1] > int(max_width):
        scale = int(max_width) / float(composed.shape[1])
        composed = cv2.resize(composed,
                              (int(max_width), max(1, int(round(height * scale)))),
                              interpolation=cv2.INTER_AREA)
    return composed


def phase_still(picture: np.ndarray, *, layout: sc.ShowcaseLayout, chrome: dict,
                overlay_draw=None, engine: Optional[TextEngine] = None
                ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """A still: the same chrome, no video-only elements, phase shown prominently."""
    return compose_frame(picture, layout=layout, chrome=chrome,
                         overlay_draw=overlay_draw, engine=engine, mode="still")


def font_report(engine: TextEngine) -> dict:
    return {"engine": engine.engine, "font_path": engine.font_path,
            "bold_path": engine.bold_path, "unicode_ok": engine.unicode_ok}
