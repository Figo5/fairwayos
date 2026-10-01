"""Showcase presentation layer: what is drawn, where, and with which numbers.

This module is **stdlib only** and deliberately contains no drawing: it decides the
layout rectangles, the four metric cards, the phase badge and the disclosure string,
so the same decisions can be tested without cv2 and so the renderer and the report
cannot disagree about what a card says.

Two hard rules are encoded here:

* **The picture keeps its own aspect ratio.** Every rectangle this module returns is
  inside ``(width, height)``. There is no gutter, no sidebar and no letterbox: a card
  is translucent chrome ON the footage, never a second panel beside it. The demo
  renderer's external gutter exists to keep chrome off a small portrait golfer; the
  showcase deliberately trades that for the frame-filling look social playback needs,
  and compensates with translucent cards in bands the operator can see through.
* **Only image-space quantities reach a card.** The registry below names each metric
  with its unit and its space, and 2D projected shoulder/hip tilt is labelled as a 2D
  tilt so it can never be read as a 3D turn.

Missing values are simply missing: a card whose value is unavailable formats as the
placeholder and is drawn dimmed. Nothing is interpolated, held for a long stretch or
carried forward to keep a card looking full.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: The seven phase names the badge can show, in swing order.
PHASE_BADGES = ("ADDRESS", "TAKEAWAY", "TOP", "DOWNSWING", "IMPACT",
                "FOLLOW-THROUGH", "FINISH")

#: Metric registry. ``space`` is what the number is measured in and ``source`` says
#: whether it changes per frame (``frame``) or is one value for the whole clip
#: (``clip``). ``unit`` is the display unit; ``sub`` is the small secondary line.
METRICS: Dict[str, Dict[str, str]] = {
    "shoulder_tilt": {
        "label": "SHOULDER TILT", "unit": "deg", "space": "2D image tilt",
        "source": "frame", "sub": "2D projected, not a turn",
        "description": "tilt of the shoulder line from the image horizontal"},
    "hip_tilt": {
        "label": "HIP TILT", "unit": "deg", "space": "2D image tilt",
        "source": "frame", "sub": "2D projected, not a turn",
        "description": "tilt of the hip line from the image horizontal"},
    "knee_flexion": {
        "label": "LEAD KNEE FLEX", "unit": "deg", "space": "2D image angle",
        "source": "frame", "sub": "lead leg, 180 - interior",
        "description": "flexion of the lead knee (180 minus the interior angle)"},
    "head_movement": {
        "label": "HEAD MOVE", "unit": "px", "space": "2D image displacement",
        "source": "clip", "sub": "peak from first resolvable frame",
        "description": "largest image displacement of the nose"},
    "hand_rise": {
        "label": "HAND RISE", "unit": "px", "space": "2D image displacement",
        "source": "frame", "sub": "wrists above the address midpoint",
        "description": ("how far the wrist midpoint sits ABOVE the first frame in which "
                        "both wrists were resolvable, in image pixels; image y grows "
                        "downwards, and the camera's own movement is included")},
    "sway": {
        "label": "SWAY", "unit": "px", "space": "2D image range",
        "source": "clip", "sub": "mid-hip horizontal range",
        "description": "range of the mid-hip x coordinate over the clip"},
    "tempo": {
        "label": "TEMPO", "unit": ":1", "space": "source frames",
        "source": "clip", "sub": "backswing : downswing",
        "description": "backswing / downswing source-frame ratio"},
}

#: The default four cards. Four is what fits legibly at social playback size.
DEFAULT_METRICS: Tuple[str, ...] = ("shoulder_tilt", "hip_tilt", "knee_flexion",
                                    "tempo")
MIN_CARDS = 3
MAX_CARDS = 4

DEFAULT_DISCLOSURE = "OPERATOR-ASSISTED \u00b7 2D IMAGE-SPACE ESTIMATES"
DEFAULT_BRAND = "FAIRWAY OS"
DEFAULT_BRAND_SUB = "SWING ANALYSIS"

PLACEHOLDER = "--"

#: Trail lengths, in drawn points, for the two operator-anchored targets. Short on
#: purpose: a long trail is a shape the operator never supplied.
CLUB_TRAIL_POINTS = 12
BALL_TRAIL_POINTS = 10

#: How many consecutive source frames a card's availability must hold before the card
#: switches state. A per-frame resolvability flag can toggle for one or two frames as a
#: wrist edge leaves the crop; without this the card strobes between a number and the
#: placeholder, which is the one chrome element that moves on its own.
DEFAULT_CARD_DEBOUNCE_FRAMES = 8


# ---------------------------------------------------------------- text safety

_ASCII_MAP = {
    "\u00b7": " - ", "\u2013": "-", "\u2014": "-", "\u00b0": " deg", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u2192": "->", "\u2248": "~", "\u00d7": "x",
}


def ascii_safe(text: str) -> str:
    """Rewrite typography that cv2's Hershey fonts would render as ``?`` boxes.

    The PIL path can carry the real characters; the Hershey fallback cannot, so this
    is applied only there. A character with no mapping is dropped rather than drawn
    as a question mark, because a broken glyph in a card value reads as data.
    """
    out = str(text)
    for source, replacement in _ASCII_MAP.items():
        out = out.replace(source, replacement)
    return "".join(ch if ord(ch) < 128 else "" for ch in out)


def display_text(text: str, *, unicode_ok: bool) -> str:
    return str(text) if unicode_ok else ascii_safe(text)


# ---------------------------------------------------------------- value formatting

def format_value(metric: str, value: Optional[float], *,
                 unicode_ok: bool = True) -> str:
    """One card's big number, or the placeholder when there is no value.

    Units are attached here so the card and the report use the same rounding. A
    value that is not finite is treated as missing: a ``nan`` on a card is worse
    than no card value at all.
    """
    spec = METRICS.get(metric)
    if spec is None:
        return PLACEHOLDER
    if value is None or not _finite(value):
        return PLACEHOLDER
    number = float(value)
    if metric in ("shoulder_tilt", "hip_tilt", "knee_flexion"):
        text = f"{number:.1f}\u00b0" if unicode_ok else f"{number:.1f} deg"
    elif metric == "tempo":
        text = f"{number:.2f}:1"
    else:
        text = f"{number:.0f} px"
        if text == "-0 px":
            # -0.4 rounds to "-0", which reads as a bug on a card and is the same
            # number as zero. The measured value is untouched; only the glyph is.
            text = "0 px"
    return text


def format_sub(metric: str, *, unicode_ok: bool = True) -> str:
    return display_text((METRICS.get(metric) or {}).get("sub", ""), unicode_ok=unicode_ok)


def format_label(metric: str) -> str:
    return (METRICS.get(metric) or {}).get("label", metric.upper())


def _finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


# ---------------------------------------------------------------- metric values

def frame_values(angles: Mapping[str, Any], *, handedness: str = "right",
                 wrist_reference: Optional[float] = None
                 ) -> Dict[str, Optional[float]]:
    """The per-frame metrics for one body record, from its 2D angles.

    ``knee_flexion`` is the LEAD leg for the configured handedness, computed as
    ``180 - interior`` with the same helper the report uses, so a card and the
    metric table cannot disagree. A missing landmark leaves the value ``None``.

    ``hand_rise`` needs a reference: the wrist midpoint's image y on the first frame
    where both wrists were resolvable (see :func:`wrist_reference`). Without one it is
    ``None``, because a rise needs something to rise from.
    """
    from .geometry import flexion_degrees

    lead = angles.get("lead_side") or ("left" if handedness == "right" else "right")
    interior = (angles.get("knee_interior_deg") or {}).get(lead)
    mid = wrist_midpoint(angles)
    rise = None
    if mid is not None and wrist_reference is not None:
        rise = float(wrist_reference) - float(mid[1])
    return {
        "shoulder_tilt": _maybe(angles.get("shoulder_tilt_deg")),
        "hip_tilt": _maybe(angles.get("hip_tilt_deg")),
        "knee_flexion": _maybe(flexion_degrees(interior)),
        "hand_rise": _maybe(rise),
    }


def wrist_midpoint(angles: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    """The midpoint of both wrists, or None when either is missing.

    Both wrists or nothing: switching between a two-wrist midpoint and a single wrist
    would create a step in the series that is not a movement of the golfer.
    """
    wrists = angles.get("wrist_midpoint")
    if isinstance(wrists, (list, tuple)) and len(wrists) >= 2 and _finite(wrists[1]):
        return (float(wrists[0]), float(wrists[1]))
    return None


def wrist_reference(records: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The first frame in which BOTH wrists were resolvable, for ``hand_rise``.

    Recorded rather than assumed: the reference frame and its y are reported next to
    every hand-rise value, because the number means "above this frame", not "above a
    true address position" and not anything in metres.
    """
    for record in records:
        if not record.get("available"):
            continue
        points = {p.get("name"): p for p in (record.get("keypoints") or [])}
        left, right = points.get("wr_l"), points.get("wr_r")
        if not left or not right or not left.get("visible") or not right.get("visible"):
            continue
        frame = record.get("source_frame")
        if frame is None:
            continue
        y = (float(left["y"]) + float(right["y"])) / 2.0
        return {"frame": int(frame), "y": y,
                "note": ("image y of the wrist midpoint on the first frame where both "
                         "wrists were resolvable; hand rise is measured from here in "
                         "image pixels, and the camera's own movement is included")}
    return None


def _maybe(value: Any) -> Optional[float]:
    return float(value) if _finite(value) else None


# ---------------------------------------------------------------- card stability

def debounce_series(values_by_frame: Mapping[int, Optional[float]], *,
                    min_run: int = DEFAULT_CARD_DEBOUNCE_FRAMES
                    ) -> Dict[int, Dict[str, Any]]:
    """What a per-frame card should SHOW at each frame, given its measured series.

    The measurement itself is never changed: this only decides what the card displays,
    and every decision is recorded per frame so the run can say which frames showed a
    held presentation rather than a value measured at that frame.

    The rule, stated exactly (it is an OFFLINE, whole-run filter and says so):

    * the card has a settled state, ``available`` or ``unavailable``, adopted from the
      first run of consecutive frames and changed only by the rules below;
    * while the card is settled on a value, a frame whose own measurement exists is
      shown as MEASURED (never hidden behind a held value);
    * a frame whose measurement is missing is shown as the last measured value, marked
      held, for at most ``min_run - 1`` consecutive frames. A gap that outlasts that
      bound switches the card to ``unavailable`` — it shows the placeholder from the
      eighth frame on instead of holding a value across a gap longer than the bound;
    * while the card is settled on the placeholder, a reappearing value is only shown
      once its run has held for ``min_run`` consecutive frames. A one-frame blip
      therefore never flashes a number.

    Because a run's LENGTH is known to this function, the card re-enables at the first
    frame of a run that turns out to be long enough, not ``min_run`` frames later. That
    lookahead is a presentation choice and is disclosed; the frame-by-frame decisions
    are recorded, and no value is ever shown that was not measured somewhere in the run
    (the held frames carry the frame their value came from).

    Both bounds are in source frames: with the default of 8 that is a hold of at most 7
    frames (~1/3 s at this source's cadence). A real loss of the landmark outlasts it
    and switches the card, so nothing long is ever hidden.
    """
    order = [int(frame) for frame in sorted(values_by_frame)]
    out: Dict[int, Dict[str, Any]] = {}
    if not order:
        return out
    runs: List[Dict[str, Any]] = []
    for frame in order:
        state = "available" if values_by_frame.get(frame) is not None else "unavailable"
        if runs and runs[-1]["state"] == state and runs[-1]["frames"][-1] == frame - 1:
            runs[-1]["frames"].append(frame)
        else:
            runs.append({"state": state, "frames": [frame]})
    threshold = max(1, int(min_run))
    hold_cap = threshold - 1
    settled: Optional[str] = None
    held_value: Optional[float] = None
    held_from: Optional[int] = None
    held_streak = 0
    for run in runs:
        run_state = run["state"]
        first_run = settled is None
        if first_run:
            settled = run_state
        elif run_state == "available" and len(run["frames"]) >= threshold \
                and settled != "available":
            # the card was showing the placeholder and this value has now held long
            # enough to be adopted (the reappearance is debounced the same way)
            settled = "available"
        for frame in run["frames"]:
            own = values_by_frame.get(frame)
            if run_state == "available" and own is not None \
                    and (first_run or settled == "available"):
                # A measured value is always preferred over a held one.
                held_value, held_from = float(own), frame
                held_streak = 0
                out[frame] = {"value": float(own), "state": "measured", "held": False,
                              "held_from_frame": None,
                              "reason": "measured at this frame"}
            elif run_state == "available":
                # settled on the placeholder and this value has not held long enough
                held_streak = 0
                out[frame] = {"value": None, "state": "held_unavailable", "held": True,
                              "held_from_frame": None,
                              "reason": (f"a value flickered for fewer than {threshold} "
                                         f"consecutive frames: the card holds the "
                                         f"placeholder rather than flashing a number")}
            elif settled == "available" and held_value is not None \
                    and held_streak + 1 <= hold_cap:
                held_streak += 1
                out[frame] = {"value": held_value, "state": "held", "held": True,
                              "held_from_frame": held_from,
                              "reason": (f"no measurement at this frame; the card holds "
                                         f"the last measured value (frame {held_from}) "
                                         f"for at most {hold_cap} frames")}
            else:
                # the gap outlasted the hold bound (or there is nothing to hold):
                # settle on the placeholder instead of freezing a stale value
                if settled == "available" and held_streak >= hold_cap:
                    settled = "unavailable"
                held_streak = 0
                out[frame] = {"value": None, "state": "measured", "held": False,
                              "held_from_frame": None,
                              "reason": "no value at this frame"}
    return out


def debounce_record(series: Mapping[int, Mapping[str, Any]], *,
                    min_run: int = DEFAULT_CARD_DEBOUNCE_FRAMES) -> Dict[str, Any]:
    """The provenance for one debounced card: what was held, and where it came from."""
    threshold = max(1, int(min_run))
    held = sorted(int(frame) for frame, entry in series.items() if entry.get("held"))
    held_value = [frame for frame in held if series[frame].get("value") is not None]
    held_placeholder = [frame for frame in held if frame not in set(held_value)]
    longest = 0
    streak = 0
    for frame in sorted(int(f) for f in series):
        if series[frame].get("held"):
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0
    return {
        "min_run_frames": threshold,
        "hold_cap_frames": threshold - 1,
        "held_frames": held,
        "held_count": len(held),
        "held_value_frames": held_value,
        "held_placeholder_frames": held_placeholder,
        "longest_held_streak_frames": longest,
        "held_value_sources": [
            {"frame": frame, "from_frame": series[frame].get("held_from_frame"),
             "value": series[frame].get("value")} for frame in held_value],
        "filter": ("offline whole-run hysteresis: a state must hold for "
                   f"{threshold} consecutive source frames to be adopted; a missing "
                   f"measurement is held for at most {threshold - 1} consecutive "
                   "frames, after which the card settles on the placeholder until a "
                   "value holds again"),
        "note": ("display only. Frames listed here showed the previous state; the "
                 "measured per-frame series in the analysis JSON is unchanged and no "
                 "value was invented. A held value is the last MEASURED value, and "
                 "`held_value_sources` names the frame it was measured at."),
    }


def clip_values(analytics: Mapping[str, Any], phases: Mapping[str, Any],
                *, handedness: str = "right") -> Dict[str, Dict[str, Any]]:
    """The clip-level metrics: peak head displacement, sway range and tempo.

    Every one of them is read from the same records the report prints: a metric that
    the report calls ``unavailable`` is ``unavailable`` on the card too. Nothing is
    recomputed here, so there is no second definition to drift.
    """
    supported = {str(m.get("name")): m
                 for m in (analytics.get("body_2d", {}).get("supported") or [])}
    out: Dict[str, Dict[str, Any]] = {}
    for key, name in (("head_movement", "head_displacement"),
                      ("sway", "body_centre_lateral_sway")):
        metric = supported.get(name)
        out[key] = {"value": _maybe((metric or {}).get("value")),
                    "state": "supported" if metric else "unavailable",
                    "reason": "" if metric else "the report reports it unavailable",
                    "unit": (metric or {}).get("unit", METRICS[key]["unit"])}
    tempo = (phases or {}).get("tempo") or {}
    if tempo.get("state") == "supported":
        out["tempo"] = {
            "value": _maybe(tempo.get("backswing_downswing_ratio")),
            "state": "supported",
            "reason": "",
            "unit": ":1",
            "detail": {"backswing_frames": tempo.get("backswing_frames"),
                       "downswing_frames": tempo.get("downswing_frames"),
                       "backswing_start_marker": tempo.get("backswing_start_marker"),
                       "label": tempo.get("label")},
            "sub": (f"back {tempo.get('backswing_frames')}f / "
                    f"down {tempo.get('downswing_frames')}f"),
        }
    else:
        out["tempo"] = {"value": None, "state": "unavailable",
                        "reason": tempo.get("reason") or "tempo is unavailable",
                        "unit": ":1"}
    return out


# ---------------------------------------------------------------- phase badge

def current_phase(frame: int, markers: Mapping[str, Optional[int]],
                  order: Sequence[str] = PHASE_BADGES) -> Optional[str]:
    """Which phase the frame is IN, from the operator's own markers.

    The badge names a span, not a nearest marker: the current phase is the last
    marked phase at or before this frame. Before the first marker there is no phase
    to name and the badge is hidden rather than showing the next phase early, which
    would label setup frames as ADDRESS.
    """
    steps = [name for name in order if markers.get(name) is not None]
    if not steps:
        return None
    current: Optional[str] = None
    for name in steps:
        frame_value = markers.get(name)
        if frame_value is not None and int(frame_value) <= int(frame):
            current = name
    return current


def phase_markers(phases: Mapping[str, Any]) -> Dict[str, Optional[int]]:
    """``{PHASE_BADGE_NAME: frame}`` from a resolved phase payload."""
    from .config import PHASE_ORDER

    records = (phases or {}).get("phases") or {}
    out: Dict[str, Optional[int]] = {}
    for name, badge in zip(PHASE_ORDER, PHASE_BADGES):
        record = records.get(name) or {}
        frame = record.get("frame")
        out[badge] = None if frame is None else int(frame)
    return out


def impact_accent(frame: int, impact_frame: Optional[int], *,
                  half_span: int = 3) -> float:
    """How strong the impact accent is at this frame, 0..1. Zero away from impact.

    A short bounded accent: full strength at the impact frame, falling to nothing
    within ``half_span`` frames either side. It marks the operator's own impact
    marker and asserts nothing about the strike itself.
    """
    if impact_frame is None:
        return 0.0
    distance = abs(int(frame) - int(impact_frame))
    if distance > half_span:
        return 0.0
    return 1.0 - (distance / float(half_span + 1))


# ---------------------------------------------------------------- layout

@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    def right(self) -> int:
        return self.x + self.w

    def bottom(self) -> int:
        return self.y + self.h

    def overlaps(self, other: "Rect") -> bool:
        return not (self.right() <= other.x or other.right() <= self.x
                    or self.bottom() <= other.y or other.bottom() <= self.y)

    def intersection_px2(self, other: "Rect") -> int:
        width = max(0, min(self.right(), other.right()) - max(self.x, other.x))
        height = max(0, min(self.bottom(), other.bottom()) - max(self.y, other.y))
        return width * height


@dataclass
class ShowcaseLayout:
    """Every rectangle the showcase draws, all of them inside the picture."""
    width: int
    height: int
    margin: int
    brand: Rect
    phase_badge: Rect
    cards: List[Rect] = field(default_factory=list)
    disclosure: Rect = Rect(0, 0, 0, 0)
    brand_font: int = 12
    brand_sub_font: int = 9
    badge_font: int = 10
    card_value_font: int = 16
    card_label_font: int = 9
    card_sub_font: int = 8
    disclosure_font: int = 8
    card_index: Dict[str, Rect] = field(default_factory=dict)
    #: Total width of the card strip (may be capped so a logo in the footage stays
    #: visible) and the vertical band the cards occupy.
    row_width: int = 0
    strip_top: int = 0
    strip_bottom: int = 0

    def to_dict(self) -> dict:
        return {
            "width": self.width, "height": self.height, "margin": self.margin,
            "brand": self.brand.to_dict(), "phase_badge": self.phase_badge.to_dict(),
            "cards": [{"metric": metric, **rect.to_dict()}
                      for metric, rect in self.card_index.items()],
            "card_row": {"x": self.margin, "w": self.row_width,
                         "top": self.strip_top, "bottom": self.strip_bottom},
            "disclosure": self.disclosure.to_dict(),
            "fonts": {"brand": self.brand_font, "brand_sub": self.brand_sub_font,
                      "badge": self.badge_font, "card_value": self.card_value_font,
                      "card_label": self.card_label_font, "card_sub": self.card_sub_font,
                      "disclosure": self.disclosure_font},
            "chrome_outside_picture_px2": 0,
            "frame_aspect_preserved": True,
        }


def _clamp(value: float, low: int, high: int) -> int:
    return int(max(low, min(high, round(value))))


def plan_layout(width: int, height: int, metrics: Sequence[str], *,
                max_row_width: Optional[int] = None,
                card_position: str = "bottom") -> ShowcaseLayout:
    """Rectangles for one rendered frame, all inside ``(width, height)``.

    Sizes are fractions of the frame HEIGHT, so the same design holds at 360p and at
    1080p. The numbers below are the ones the operator asked for at 1080p and they
    are expressed as fractions so the proportions survive a different render size:

    * a compact card strip of about 0.105 h (113px at 1080p) that starts low enough
      to clear a golfer's feet on a full-body framing, with the disclosure line in
      the margin UNDER the cards rather than a second band above them;
    * label ~0.24 of the card height (27px at 1080p) and value ~0.46 (52px), which is
      what stays readable in a social feed;
    * no caption sub-line: the definitions live in the report, and a third dense line
      on a card is what makes a strip look like debug output.

    ``max_row_width`` caps the card strip's total width in OUTPUT pixels so it can
    stop short of something in the footage (a watermark or a logo) that must stay
    visible: the row is laid out from the left margin and simply ends earlier.
    """
    if width < 160 or height < 120:
        raise ValueError(f"frame {width}x{height} is too small for showcase chrome")
    margin = _clamp(0.044 * height, 6, 90)
    brand_font = _clamp(0.033 * height, 11, 64)
    brand_sub_font = max(8, int(round(brand_font * 0.55)))
    badge_font = _clamp(0.037 * height, 10, 54)
    brand_h = brand_font + brand_sub_font + _clamp(0.012 * height, 2, 18)
    badge_h = badge_font + _clamp(0.028 * height, 8, 40)
    top_h = max(brand_h, badge_h)
    disclosure_font = _clamp(0.026 * height, 8, 30)
    bottom_band = max(margin, disclosure_font + _clamp(0.015 * height, 3, 18))

    card_h = _clamp(0.105 * height, 34, 200)
    if card_position not in ("bottom", "top"):
        raise ValueError(f"card_position must be 'bottom' or 'top', got "
                         f"{card_position!r}")
    if card_position == "top":
        # The card row moves UNDER the brand/badge band instead of above the
        # disclosure line. It exists because a golfer low in the frame has their feet
        # exactly where the bottom row would sit, and covering the feet of the swing
        # is worse than any layout nicety. Nothing else about the design changes: same
        # card size, same fonts, same inside-the-picture guarantee.
        card_row_top = margin + top_h + _clamp(0.012 * height, 2, 14)
        available = (height - bottom_band - card_row_top)
        if available < card_h:
            card_h = max(24, available)
        card_row_bottom = card_row_top + card_h
    else:
        card_row_bottom = height - bottom_band
        card_row_top = card_row_bottom - card_h
        # Bounded, not silent: if the frame is so short that the card band would touch
        # the brand block, the cards shrink to the space that is actually free.
        available = card_row_bottom - (margin + top_h) - _clamp(0.020 * height, 3, 24)
        if available < card_h:
            card_h = max(24, available)
            card_row_top = card_row_bottom - card_h

    count = max(1, len(metrics))
    gap = _clamp(0.012 * width, 3, 40)
    full_width = width - 2 * margin
    row_width = full_width if max_row_width is None else min(
        full_width, max(count * 24 + gap * (count - 1), int(max_row_width) - margin))
    usable = row_width - gap * (count - 1)
    card_w = max(24, int(usable // count))
    cards: List[Rect] = []
    index: Dict[str, Rect] = {}
    x = margin
    for metric in metrics:
        rect = Rect(x=x, y=card_row_top, w=card_w, h=card_h)
        cards.append(rect)
        index[metric] = rect
        x += card_w + gap

    layout = ShowcaseLayout(
        width=width, height=height, margin=margin,
        brand=Rect(margin, margin, max(40, int(width * 0.30)), brand_h),
        phase_badge=Rect(max(0, width - margin - int(width * 0.22)), margin,
                         int(width * 0.22), badge_h),
        cards=cards,
        disclosure=Rect(margin, card_row_bottom, row_width, bottom_band),
        brand_font=brand_font, brand_sub_font=brand_sub_font, badge_font=badge_font,
        card_value_font=_clamp(card_h * 0.50, 13, 128),
        card_label_font=_clamp(card_h * 0.24, 8, 64),
        card_sub_font=_clamp(card_h * 0.17, 7, 52),
        disclosure_font=disclosure_font, card_index=index)
    layout.row_width = row_width
    layout.strip_top = card_row_top
    layout.strip_bottom = card_row_bottom
    return layout


def layout_problems(layout: ShowcaseLayout) -> List[str]:
    """Everything wrong with a layout, measured rather than asserted.

    Returned to the renderer's debug record as well as used by the tests, so the
    claim "the picture is untouched" is backed by a measurement of every rectangle.
    """
    problems: List[str] = []
    named = [("brand", layout.brand), ("phase_badge", layout.phase_badge),
             ("disclosure", layout.disclosure)]
    named += [(f"card:{m}", rect) for m, rect in layout.card_index.items()]
    for name, rect in named:
        if rect.w <= 0 or rect.h <= 0:
            problems.append(f"{name} has a zero or negative size {rect.to_dict()}")
        if rect.x < 0 or rect.y < 0 or rect.right() > layout.width \
                or rect.bottom() > layout.height:
            problems.append(f"{name} {rect.to_dict()} is not inside the "
                            f"{layout.width}x{layout.height} picture")
    for i in range(len(layout.cards)):
        for j in range(i + 1, len(layout.cards)):
            overlap = layout.cards[i].intersection_px2(layout.cards[j])
            if overlap:
                problems.append(f"cards {i} and {j} overlap by {overlap}px2")
    if layout.brand.overlaps(layout.phase_badge):
        problems.append("the brand block and the phase badge overlap")
    for name, rect in named:
        if name in ("brand", "phase_badge"):
            continue
        if rect.overlaps(layout.brand) or rect.overlaps(layout.phase_badge):
            problems.append(f"{name} overlaps the headline chrome")
    return problems


# ---------------------------------------------------------------- coverage

def chrome_coverage(layout: ShowcaseLayout, *, roi: Optional[Sequence[int]] = None,
                    points_xy: Sequence[Sequence[float]] = (),
                    scale: float = 1.0) -> Dict[str, Any]:
    """How much of the golfer and of the action the chrome actually covers.

    The showcase deliberately draws over the picture, so the honest thing is to
    MEASURE the overlap of every chrome band with the operator's ROI and with the
    points that will be drawn, and to record it. The demo renderer avoids the overlap
    by construction (external gutter); the showcase reports it instead of claiming
    there is none, and the report says what it measured.

    ``roi`` and ``points_xy`` are in NATIVE source pixels; ``scale`` maps them into
    the rendered frame, so the measurement cannot drift from what is drawn.
    """
    roi_rect = None
    if roi:
        roi_rect = Rect(int(round(float(roi[0]) * scale)),
                        int(round(float(roi[1]) * scale)),
                        int(round(float(roi[2]) * scale)),
                        int(round(float(roi[3]) * scale)))
    bands = {"brand": layout.brand, "phase_badge": layout.phase_badge,
             "disclosure": layout.disclosure}
    bands.update({f"card:{name}": rect for name, rect in layout.card_index.items()})
    body: Dict[str, int] = {}
    total_body = 0
    for name, rect in bands.items():
        pixels = roi_rect.intersection_px2(rect) if roi_rect else 0
        body[name] = pixels
        total_body += pixels
    action = 0
    action_points = 0
    covered: List[Dict[str, Any]] = []
    for index, point in enumerate(points_xy or []):
        if point is None or len(point) < 2 or point[0] is None or point[1] is None:
            continue
        x = float(point[0]) * scale
        y = float(point[1]) * scale
        for name, rect in bands.items():
            if rect.x <= x < rect.right() and rect.y <= y < rect.bottom():
                action += 1
                covered.append({"index": index, "band": name,
                                "x": round(x, 1), "y": round(y, 1)})
                break
        action_points += 1
    roi_area = (roi_rect.w * roi_rect.h) if roi_rect else 0
    return {
        "roi_xywh_rendered": roi_rect.to_dict() if roi_rect else None,
        "roi_area_px2": roi_area,
        "roi_covered_px2_total": total_body,
        "roi_covered_fraction": (round(total_body / roi_area, 4) if roi_area else None),
        "roi_covered_px2_by_band": body,
        "drawn_points_total": action_points,
        "drawn_points_under_chrome": action,
        "drawn_points_under_chrome_fraction": (round(action / action_points, 4)
                                               if action_points else None),
        "covered_points": covered[:50],
        "note": ("chrome is translucent by design and the picture fills the frame; "
                 "these are measurements of what the chrome covers, not a claim that "
                 "it covers nothing. The demo renderer (showcase.enabled=false) keeps "
                 "chrome off the picture entirely with an external gutter."),
    }


# ---------------------------------------------------------------- moving ROI

def interpolate_roi(keyframes: Sequence[Mapping[str, Any]], frame: int
                    ) -> Optional[List[int]]:
    """The effective ROI box at one source frame, from the operator's own boxes.

    The operator draws a crop box at a few frames because the camera pans; between
    two boxes the box is interpolated LINEARLY in position and size, and outside the
    first/last box it is held constant. Nothing here is a pose: this is crop geometry,
    recorded with its origin, and it decides only which pixels the pose model is shown.

    Returns ``[x, y, w, h]`` in source pixels, or None when there are no keyframes.
    """
    if not keyframes:
        return None
    ordered = sorted((dict(k) for k in keyframes), key=lambda k: int(k["frame"]))
    frame = int(frame)
    if frame <= int(ordered[0]["frame"]):
        return [int(round(float(v))) for v in ordered[0]["roi"]]
    if frame >= int(ordered[-1]["frame"]):
        return [int(round(float(v))) for v in ordered[-1]["roi"]]
    for before, after in zip(ordered, ordered[1:]):
        a, b = int(before["frame"]), int(after["frame"])
        if not (a <= frame <= b):
            continue
        if b == a:
            return [int(round(float(v))) for v in before["roi"]]
        t = (frame - a) / float(b - a)
        return [int(round(float(x) + (float(y) - float(x)) * t))
                for x, y in zip(before["roi"], after["roi"])]
    return [int(round(float(v))) for v in ordered[-1]["roi"]]


def roi_envelope(keyframes: Sequence[Mapping[str, Any]]) -> Optional[List[int]]:
    """The bounding box of every ROI keyframe: the region the crop moves through.

    Used for measurements that must cover the whole pan (for example how much of the
    golfer's region the cards cover), rather than only the first frame's box.
    """
    boxes = [list(k["roi"]) for k in keyframes if k.get("roi")]
    if not boxes:
        return None
    x0 = min(int(b[0]) for b in boxes)
    y0 = min(int(b[1]) for b in boxes)
    x1 = max(int(b[0]) + int(b[2]) for b in boxes)
    y1 = max(int(b[1]) + int(b[3]) for b in boxes)
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


# ---------------------------------------------------------------- playback pace

def stride_frames(start_frame: int, end_frame: int, stride: int) -> List[int]:
    """The source frames that are actually encoded, at a uniform presentation stride.

    ``stride=3`` keeps every third source frame, which makes playback three times
    faster while the container cadence stays the source's own. The mapping is exact and
    reportable: output frame *i* shows source frame ``start + i * stride``.
    """
    step = max(1, int(stride))
    return list(range(int(start_frame), int(end_frame) + 1, step))


def expand_holds(frames: Sequence[int], holds: Sequence[Mapping[str, Any]],
                 fps: float) -> List[Dict[str, Any]]:
    """The output frame list: every encoded frame, with holds expanded in place.

    A hold repeats ONE already-drawn frame for ``seconds`` of playback. It is
    presentation only: the held frames are marked ``held`` so the run can state that
    nothing new was observed during them, and a hold on a frame whose target was lost
    is refused rather than dragging a stale marker along.

    Returns one record per output frame: ``{source_frame, held, hold_label}``.
    """
    by_frame = {int(hold["frame"]): dict(hold) for hold in (holds or [])}
    out: List[Dict[str, Any]] = []
    for frame in frames:
        frame = int(frame)
        out.append({"source_frame": frame, "held": False, "hold_label": None})
        hold = by_frame.get(frame)
        if not hold:
            continue
        repeats = max(1, int(round(float(hold.get("seconds", 0.0)) * float(fps or 1))))
        for _ in range(repeats - 1):
            out.append({"source_frame": frame, "held": True,
                        "hold_label": str(hold.get("label") or "")})
    return out


def hold_records(frames: Sequence[int], holds: Sequence[Mapping[str, Any]],
                 fps: float) -> List[Dict[str, Any]]:
    """Where each hold landed in the OUTPUT timeline, for the provenance record."""
    records: List[Dict[str, Any]] = []
    expanded = expand_holds(frames, holds, fps)
    for hold in (holds or []):
        frame = int(hold["frame"])
        indices = [i for i, entry in enumerate(expanded)
                   if entry["source_frame"] == frame]
        if not indices:
            continue
        records.append({
            "source_frame": frame,
            "label": str(hold.get("label") or ""),
            "seconds": float(hold.get("seconds", 0.0)),
            "output_frames": [min(indices), max(indices)],
            "output_frames_held": len(indices),
            "output_frame_drawn": min(indices),
            "output_frames_repeated": len(indices) - 1,
            "status": "held_replay_of_an_already_drawn_frame",
            "note": ("the first frame of the span is the drawn frame; the rest are "
                     "repeats of it, so no new observation, position or trail is produced "
                     "during the hold."),
        })
    return records


# ---------------------------------------------------------------- config block
@dataclass(frozen=True)
class ShowcaseOptions:
    """The opt-in ``showcase`` config block. Off unless the operator turns it on."""
    enabled: bool = False
    brand: str = DEFAULT_BRAND
    brand_sub: str = DEFAULT_BRAND_SUB
    metrics: Tuple[str, ...] = DEFAULT_METRICS
    disclosure: str = DEFAULT_DISCLOSURE
    club_trail_points: int = CLUB_TRAIL_POINTS
    ball_trail_points: int = BALL_TRAIL_POINTS
    impact_accent: bool = True
    #: Half-width of the impact accent in SOURCE frames: the accent is drawn at full
    #: strength on the impact marker and fades out within this many frames either side.
    #: At stride 3, the default of 3 shows the accent for three output frames; a larger
    #: value is what makes it visible for longer than a blink.
    impact_accent_span_frames: int = 3
    #: Card hysteresis, in consecutive source frames (see ``debounce_series``).
    card_debounce_frames: int = DEFAULT_CARD_DEBOUNCE_FRAMES
    max_width: Optional[int] = 1280
    crf: int = 18
    preset: str = "slow"
    #: Cap on the card strip's total width in OUTPUT pixels, so the strip can stop
    #: short of a logo or watermark in the footage and leave it visible.
    card_row_max_width: Optional[int] = None
    #: Short holds on already-drawn frames, as ``{"frame", "seconds", "label"}``.
    holds: Tuple[Dict[str, Any], ...] = ()

    def to_dict(self) -> dict:
        return {"enabled": self.enabled, "brand": self.brand,
                "brand_sub": self.brand_sub, "metrics": list(self.metrics),
                "disclosure": self.disclosure,
                "club_trail_points": self.club_trail_points,
                "ball_trail_points": self.ball_trail_points,
                "impact_accent": self.impact_accent,
                "impact_accent_span_frames": self.impact_accent_span_frames,
                "card_debounce_frames": self.card_debounce_frames,
                "max_width": self.max_width, "crf": self.crf, "preset": self.preset,
                "card_row_max_width": self.card_row_max_width,
                "holds": [dict(hold) for hold in self.holds]}


SHOWCASE_FIELDS = ("enabled", "brand", "brand_sub", "metrics", "disclosure",
                   "club_trail_points", "ball_trail_points", "impact_accent",
                   "impact_accent_span_frames", "card_debounce_frames",
                   "max_width", "crf", "preset", "card_row_max_width", "holds")
