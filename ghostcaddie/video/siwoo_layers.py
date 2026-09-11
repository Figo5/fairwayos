"""Si Woo Kim approach demo layer contracts.

Research-only helpers used by the local overlay renderer. These functions do not
promote predictions to ground truth and do not call calibration/analytics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

Point = Tuple[float, float]
Box = Tuple[int, int, int, int]


@dataclass(frozen=True)
class LayerState:
    name: str
    visible: bool
    point: Optional[Point] = None
    box: Optional[Box] = None
    reason: str = "unavailable"
    confidence: Optional[float] = None


def renderable_trail(state: LayerState, history: Sequence[Point]) -> List[Point]:
    """Return the trail to draw for a layer, clearing it when not visible.

    Rendering code should call this at the object boundary so unavailable,
    off-screen, cut, or occluded states never leave ghost trails behind.
    """
    if not state.visible:
        return []
    if state.point is None and state.box is None:
        return []
    return list(history)


def reset_layers_on_cut(layers: Mapping[str, LayerState],
                        cut_between: Tuple[int, int]) -> Dict[str, LayerState]:
    """Return layer states reset after a camera cut.

    The input is not mutated; diagnostic reasons on the old states are preserved
    by callers in their per-frame records. The reset state explicitly names the
    cut so hidden overlays are distinguishable from ordinary low confidence.
    """
    a, b = cut_between
    return {
        name: LayerState(name=state.name, visible=False, point=None, box=None,
                         reason=f"camera_cut_reset_between_{a}_{b}",
                         confidence=None)
        for name, state in layers.items()
    }


def _box_center_x(box: Sequence[float]) -> float:
    return (float(box[0]) + float(box[2])) / 2.0


def choose_golfer_candidate(candidates: Iterable[Mapping],
                            preferred_x: Optional[float] = None) -> Optional[Mapping]:
    """Choose the main golfer from person candidates without caddie promotion.

    Candidate dictionaries may include ``white_bib_fraction``. Large white-bib
    fractions are penalized because in the Si Woo reaction shot the caddie/bib is
    a major false-player risk. This is a selection heuristic, not identity truth.
    """
    best = None
    best_score = None
    for cand in candidates:
        box = cand.get("box")
        if box is None:
            continue
        conf = float(cand.get("confidence", 0.0))
        bib = float(cand.get("white_bib_fraction", 0.0))
        width = max(1.0, float(box[2]) - float(box[0]))
        height = max(1.0, float(box[3]) - float(box[1]))
        score = conf + min(0.4, height / 900.0) - 0.75 * bib
        if preferred_x is not None:
            score -= abs(_box_center_x(box) - float(preferred_x)) / 1800.0
        if best_score is None or score > best_score:
            best_score = score
            best = cand
    return best


def source_to_display_frame(source_frame: int, pass_start: int) -> int:
    """Map a native source frame to the real-time pass display frame index."""
    if source_frame < pass_start:
        raise ValueError("source frame precedes pass_start")
    return int(source_frame) - int(pass_start)


def object_visibility_for_frame(frame: int, object_name: str, visibility: Mapping) -> Tuple[str, str]:
    """Return reviewed per-object visibility state for a source frame.

    ``visibility`` maps object names to inclusive ``(start, end, reason)`` tuples
    plus an optional ``cuts`` list. A cut wins over an interval, because all
    temporal history must reset at the edit before any object can reacquire.
    """
    if any(int(frame) >= int(cut) for cut in visibility.get("cuts", [])):
        return "unavailable", f"camera_cut_reset_at_{max(c for c in visibility.get('cuts', []) if int(frame) >= int(c))}"
    for start, end, reason in visibility.get(object_name, []):
        if int(start) <= int(frame) <= int(end):
            return "visible", str(reason)
    return "unavailable", f"{object_name}_not_visible_or_not_resolvable"


def contiguous_visible_ranges(states: Mapping[int, Mapping]) -> List[Tuple[int, int]]:
    """Return inclusive contiguous ranges for visible frame states.

    Invisible/ambiguous frames split segments so renderers/reports do not bridge
    gaps with implied observations.
    """
    ranges: List[Tuple[int, int]] = []
    start = prev = None
    for frame in sorted(int(k) for k in states.keys()):
        if states[frame].get("visible") is True:
            if start is None or prev is None or frame != prev + 1:
                if start is not None and prev is not None:
                    ranges.append((start, prev))
                start = frame
            prev = frame
        elif start is not None and prev is not None:
            ranges.append((start, prev))
            start = prev = None
    if start is not None and prev is not None:
        ranges.append((start, prev))
    return ranges


def can_continue_without_detection(consecutive_misses: int, max_misses: int) -> bool:
    """Bound optical-flow continuation so a lost body cannot ghost indefinitely."""
    if not isinstance(consecutive_misses, int) or not isinstance(max_misses, int):
        raise TypeError("miss counters must be integers")
    if max_misses < 0:
        raise ValueError("max_misses must be non-negative")
    return 0 <= consecutive_misses < max_misses


def is_golfer_body_candidate(box: Sequence[float], frame_width: int, frame_height: int) -> bool:
    """Return whether a person box is in the reviewed golfer region.

    This is a spatial/shape gate, not a frame-number gate: late partial
    follow-through boxes can pass if they still occupy the golfer's center-left
    source pixels, while right-side caddie/spectator boxes are rejected.
    """
    if box is None or len(box) < 4:
        return False
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    if x2 <= x1 or y2 <= y1:
        return False
    cx = (x1 + x2) / 2.0
    width = x2 - x1
    height = y2 - y1
    return (0.37 * frame_width <= cx <= 0.61 * frame_width
            and width <= 0.28 * frame_width
            and height >= 0.30 * frame_height
            and y2 >= 0.78 * frame_height)


def shift_box(box: Box, dx: float, dy: float, frame_width: int, frame_height: int) -> Box:
    """Shift a box by measured image motion while preserving size and bounds."""
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = x2 - x1, y2 - y1
    nx1, ny1 = x1 + float(dx), y1 + float(dy)
    nx1 = min(max(0.0, nx1), max(0.0, float(frame_width) - w))
    ny1 = min(max(0.0, ny1), max(0.0, float(frame_height) - h))
    return (int(round(nx1)), int(round(ny1)),
            int(round(nx1 + w)), int(round(ny1 + h)))
