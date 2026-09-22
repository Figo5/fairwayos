"""Phase resolution and tempo, with the label rules that keep them honest.

Three rules matter here:

1. **A phase frame is either the operator's or explicitly approximate.** A card
   labelled ``manual`` came from a config keyframe. A card labelled ``approx`` was
   derived by the single named method recorded next to it. A phase that satisfies
   neither is ``unavailable`` — never silently spaced out across the clip.
2. **Evenly spaced frames are never a phase source.** There is deliberately no code
   path that divides the interval into N and calls the results phases.
3. **Tempo needs three real markers.** Backswing and downswing durations are only
   reported when a top and an impact marker exist and the takeaway (or, disclosed,
   the address) sits before the top. Durations are source frames; seconds are
   container-cadence seconds, which is not measured action time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from .config import (PHASE_ORDER, PRIMARY_PHASES, SECONDS_BASIS_CADENCE,
                     SECONDS_BASIS_TIMESTAMPS)
from .geometry import median

if TYPE_CHECKING:  # pragma: no cover
    from .tracking import TargetTrack

LABEL_MANUAL = "manual"
LABEL_APPROX = "approx"
LABEL_UNAVAILABLE = "unavailable"

#: How much bigger than the typical observed step a ball step must be before the
#: approximation calls it the strike. Deliberately conservative.
STRIKE_STEP_FACTOR = 5.0
STRIKE_STEP_MIN_PX = 12.0
STRIKE_PRIOR_OBSERVATIONS = 3

NOT_IMPLEMENTED_REASON = (
    "no operator keyframe was supplied and this approximation is not implemented, so "
    "the phase is reported unavailable rather than guessed from the interval")

# ``SECONDS_BASIS_TIMESTAMPS`` / ``SECONDS_BASIS_CADENCE`` are imported from
# ``demo.config`` above, so the resolved config and these records cannot disagree
# about what a second means.


@dataclass
class PhaseRecord:
    name: str
    label: str
    frame: Optional[int] = None
    method: str = ""
    note: str = ""
    seconds: Optional[float] = None
    seconds_basis: str = SECONDS_BASIS_CADENCE
    #: The same frame measured from the FIRST frame of the processed interval. It is
    #: emitted under its own name so it can never be mistaken for a source-timeline
    #: position: ``seconds_source_timeline`` above is the position in the source, so
    #: frame 20 of a 25 fps clip reads 0.80 s and not 0.00 s.
    seconds_from_interval_start: Optional[float] = None
    angles: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        round4 = lambda value: None if value is None else round(value, 4)  # noqa: E731
        return {"name": self.name, "label": self.label, "frame": self.frame,
                "method": self.method, "note": self.note,
                "seconds_source_timeline": round4(self.seconds),
                "seconds_from_interval_start": round4(self.seconds_from_interval_start),
                "seconds_basis": self.seconds_basis,
                "angles_2d": self.angles}


class FrameSeconds:
    """``frame -> seconds`` from the source's own decoded frame timestamps.

    ``absolute`` is a position in the SOURCE timeline and is what the phase records
    and the report mean by "seconds in the source timeline". ``relative`` is the
    same frame measured from the first processed frame; it is available for callers
    that explicitly want interval-relative time, and it is never labelled as a
    source-timeline position.
    """

    def __init__(self, frame_times: Sequence[float], start_frame: int) -> None:
        self.times: List[float] = [float(v) for v in frame_times]
        self.start_frame = int(start_frame)

    def absolute(self, frame: int) -> Optional[float]:
        index = int(frame) - self.start_frame
        if 0 <= index < len(self.times):
            return float(self.times[index])
        return None

    def relative(self, frame: int) -> Optional[float]:
        value = self.absolute(frame)
        if value is None:
            return None
        return value - float(self.times[0])

    def __call__(self, frame: int) -> Optional[float]:
        return self.absolute(frame)


def frame_seconds_factory(frame_times: Optional[Sequence[float]], start_frame: int):
    """Build the frame->seconds mapper from the source's own frame timestamps.

    Seconds are the decoded frame's own presentation timestamp, i.e. its position in
    the source timeline, so they do not depend on a nominal frame rate and they do
    not silently rebase onto the processed interval. When the timestamps are missing
    the factory returns None and callers fall back to frames / cadence with that
    basis stated.
    """
    if not frame_times:
        return None
    return FrameSeconds(frame_times, start_frame)


def resolve_phases(*, config_keyframes: Dict[str, int], interval_frames: Sequence[int],
                   body_records: Sequence[dict], tracks: Dict[str, "TargetTrack"],
                   fps: float, allow_approx: bool,
                   body_angles_fn=None,
                   frame_seconds=None) -> dict:
    """Build one record per phase, then the tempo that follows from them."""
    poses = [r for r in body_records if r.get("available")]
    pose_by_frame = {int(r["source_frame"]): r for r in poses}
    approx_used: List[str] = []

    # A derived 'top' is only meaningful inside a bounded pre-impact window. Taking
    # the highest wrist across the whole clip commonly lands on the finish, so
    # without an upper bound the phase stays unavailable instead of being wrong.
    impact_hint, impact_note = _strike_from_ball(tracks.get("ball"))
    if impact_hint is None and "impact" in config_keyframes:
        impact_hint = int(config_keyframes["impact"])
        impact_note = "bounded by the operator's impact keyframe"
    upper_bound = impact_hint

    address_hint, address_note = _first_pose_frame(poses)
    top_hint, top_note = _min_wrist_height(poses, upper_bound=upper_bound)
    finish_hint, finish_note = _last_pose_frame(poses)

    approx_sources = {
        "address": (address_hint, address_note),
        "top": (top_hint, top_note),
        "impact": (impact_hint, impact_note),
        "finish": (finish_hint, finish_note),
    }

    seconds_basis = (SECONDS_BASIS_TIMESTAMPS if frame_seconds is not None
                     else SECONDS_BASIS_CADENCE)

    def seconds_of(frame: Optional[int]) -> Optional[float]:
        if frame is None:
            return None
        if frame_seconds is not None:
            return frame_seconds(frame)
        if not fps:
            return None
        return frame / float(fps)

    def interval_relative_seconds(frame: Optional[int]) -> Optional[float]:
        """The same frame measured from the first processed frame, or None."""
        if frame is None or frame_seconds is None:
            return None
        relative = getattr(frame_seconds, "relative", None)
        return relative(frame) if relative is not None else None

    records: Dict[str, PhaseRecord] = {}
    for name in PHASE_ORDER:
        if name in config_keyframes:
            frame = int(config_keyframes[name])
            record = PhaseRecord(name=name, label=LABEL_MANUAL, frame=frame,
                                 method="operator_keyframe",
                                 note="operator-marked source frame",
                                 seconds=seconds_of(frame),
                                 seconds_from_interval_start=
                                 interval_relative_seconds(frame))
        elif allow_approx and approx_sources.get(name, (None, ""))[0] is not None:
            frame, note = approx_sources[name]
            frame = int(frame)
            record = PhaseRecord(name=name, label=LABEL_APPROX, frame=frame,
                                 method=_method_for(name),
                                 note=f"approximate: {note}; not an observation",
                                 seconds=seconds_of(frame),
                                 seconds_from_interval_start=
                                 interval_relative_seconds(frame))
            approx_used.append(name)
        else:
            reason = (NOT_IMPLEMENTED_REASON if allow_approx
                      else "no operator keyframe was supplied and approximate phase "
                           "derivation is disabled (allow_approx_phases is false)")
            record = PhaseRecord(name=name, label=LABEL_UNAVAILABLE,
                                 method="none", note=reason)
        if record.frame is not None and body_angles_fn is not None:
            record.angles = body_angles_fn(pose_by_frame.get(record.frame))
        records[name] = record

    notes: List[str] = []
    for name in PHASE_ORDER:
        records[name].seconds_basis = seconds_basis
    # Derived phases must be chronological as a set, not only inside the tempo
    # calculation: a card sequence that cannot have happened is a claim about the
    # swing, so a derived phase that breaks the order is withdrawn.
    ordered = [records[name] for name in PHASE_ORDER]
    previous: Optional[PhaseRecord] = None
    for record in ordered:
        if record.frame is None:
            continue
        if previous is not None and previous.frame is not None \
                and record.frame < previous.frame:
            if record.label == LABEL_APPROX:
                withdrawn = (f"withdrawn: the approximation put {record.name} at frame "
                             f"{record.frame}, before {previous.name} at frame "
                             f"{previous.frame}, which cannot be the swing order")
                records[record.name] = PhaseRecord(
                    name=record.name, label=LABEL_UNAVAILABLE, method="none",
                    note=withdrawn)
                approx_used = [name for name in approx_used if name != record.name]
                notes.append(withdrawn)
                continue
        previous = record

    if approx_used:
        notes.append("Approximate phase frames were derived from measurement: "
                     + ", ".join(f"{name} ({records[name].method})"
                                 for name in approx_used)
                     + ". They are marked approx on the cards and in the video.")
    tempo = compute_tempo(records, fps=fps, notes=notes, frame_seconds=frame_seconds,
                          seconds_basis=seconds_basis)
    cards = phase_cards(records)
    return {"phases": {name: records[name].to_dict() for name in PHASE_ORDER},
            "order": list(PHASE_ORDER),
            "primary_cards": list(PRIMARY_PHASES),
            "cards": cards,
            "approx_used": approx_used,
            "seconds_basis": seconds_basis,
            "notes": notes,
            "tempo": tempo}


def _method_for(name: str) -> str:
    return {"address": "first_pose_frame", "top": "min_wrist_height",
            "impact": "ball_first_large_displacement", "finish": "last_pose_frame",
            }.get(name, "none")


def _first_pose_frame(poses: Sequence[dict]) -> Tuple[Optional[int], str]:
    if len(poses) < 5:
        return None, "too few frames carry a body pose"
    return int(poses[0]["source_frame"]), "first frame in the interval with a body pose"


def _last_pose_frame(poses: Sequence[dict]) -> Tuple[Optional[int], str]:
    if len(poses) < 5:
        return None, "too few frames carry a body pose"
    return int(poses[-1]["source_frame"]), "last frame in the interval with a body pose"


def _min_wrist_height(poses: Sequence[dict],
                      upper_bound: Optional[int] = None) -> Tuple[Optional[int], str]:
    """Highest resolvable wrist, restricted to frames before ``upper_bound``.

    Without the bound this measurement is not the top of a swing: the highest hands
    in a clip are usually the finish, and an unbounded argmin would label the
    follow-through as the top and invert the whole phase order.
    """
    if upper_bound is None:
        return None, ("no impact marker or ball-strike frame is available to bound the "
                      "search, so the highest-wrist approximation cannot be restricted "
                      "to a pre-impact window and is not used")
    best: Optional[Tuple[int, float]] = None
    for record in poses:
        frame = int(record["source_frame"])
        if frame >= int(upper_bound):
            continue
        ys = [p["y"] for p in (record.get("keypoints") or [])
              if p.get("visible") and p["name"] in ("wr_l", "wr_r")]
        if not ys:
            continue
        y = min(ys)
        if best is None or y < best[1]:
            best = (frame, y)
    if best is None or len(poses) < 5:
        return None, "wrists were not resolvable on enough frames"
    return best[0], (f"frame with the highest resolvable wrist before frame "
                     f"{int(upper_bound)} (image y={best[1]:.0f}), used as a stand-in "
                     f"for the top of the swing")


def _strike_from_ball(track: Optional["TargetTrack"]) -> tuple:
    """The first large displacement of the operator-seeded ball, or nothing."""
    if track is None or not track.available:
        return None, "no ball track, so the strike frame cannot be approximated"
    observed = [p for p in track.points if p.automatic and p.x is not None
                and p.y is not None]
    if len(observed) < STRIKE_PRIOR_OBSERVATIONS + 1:
        return None, ("the ball was observed on too few frames to approximate the "
                      "strike")
    steps = []
    for a, b in zip(observed, observed[1:]):
        if b.source_frame != a.source_frame + 1:
            continue
        steps.append(((b.x - a.x) ** 2 + (b.y - a.y) ** 2) ** 0.5)
    typical = median(steps) if steps else None
    threshold = max(STRIKE_STEP_MIN_PX,
                    (typical or 0.0) * STRIKE_STEP_FACTOR)
    run = 0
    for a, b in zip(observed, observed[1:]):
        if b.source_frame != a.source_frame + 1:
            run = 0
            continue
        step = ((b.x - a.x) ** 2 + (b.y - a.y) ** 2) ** 0.5
        if run >= STRIKE_PRIOR_OBSERVATIONS and step >= threshold:
            return int(a.source_frame), (
                f"last frame of a run of {run + 1} consecutive ball observations "
                f"before a {step:.0f}px step (threshold {threshold:.0f}px): treated as "
                f"the strike")
        run += 1
    return None, ("the ball track never showed a step large enough to be called a "
                  "strike")


def compute_tempo(records: Dict[str, PhaseRecord], *, fps: float,
                  notes: Optional[List[str]] = None,
                  frame_seconds=None,
                  seconds_basis: str = SECONDS_BASIS_CADENCE) -> dict:
    """Backswing / downswing durations and their ratio, or a stated refusal."""
    notes = notes if notes is not None else []

    def elapsed(a: int, b: int) -> float:
        """Seconds between two source frames, from timestamps when available."""
        if frame_seconds is not None:
            first, second = frame_seconds(a), frame_seconds(b)
            if first is not None and second is not None:
                return float(second) - float(first)
        return (int(b) - int(a)) / float(fps) if fps else 0.0

    blank = lambda name: PhaseRecord(name=name, label=LABEL_UNAVAILABLE)  # noqa: E731
    top = records.get("top") or blank("top")
    impact = records.get("impact") or blank("impact")
    takeaway = records.get("takeaway")
    address = records.get("address")

    top_frame = top.frame
    impact_frame = impact.frame
    if top_frame is None or impact_frame is None:
        return _tempo_unavailable(
            "tempo needs both a top-of-swing marker and an impact marker; "
            + _missing_text(top, "top") + _missing_text(impact, "impact"))

    start = takeaway if (takeaway is not None and takeaway.frame is not None) else address
    if start is None:
        return _tempo_unavailable(
            "tempo needs a backswing start: a takeaway marker, or an address marker "
            "as the disclosed fallback. Neither is available.")
    start_frame = start.frame
    if start_frame is None:
        return _tempo_unavailable(
            "tempo needs a backswing start: a takeaway marker, or an address marker "
            "as the disclosed fallback. Neither is available.")

    backswing = int(top_frame) - int(start_frame)
    downswing = int(impact_frame) - int(top_frame)
    basis_marker = "takeaway" if start.name == "takeaway" else "address"
    basis_label = _label_for(start, top, impact)
    if backswing <= 0 or downswing <= 0:
        return _tempo_unavailable(
            f"the markers are not in chronological order: {start.name}={start_frame}, "
            f"top={top_frame}, impact={impact_frame}")
    if basis_marker == "address":
        notes.append("Tempo used the address marker as the backswing start because no "
                     "takeaway marker exists; takeaway is the preferred definition and "
                     "address overstates the backswing by the waggle/pre-swing time.")
    if basis_label == LABEL_APPROX:
        notes.append("At least one tempo marker is approximate, so the durations and "
                     "the ratio below are approximations of the markers, not measured "
                     "action times.")
    return {
        "state": "supported",
        "label": basis_label,
        "backswing_start_marker": basis_marker,
        "backswing_frames": backswing,
        "downswing_frames": downswing,
        "backswing_downswing_ratio": round(backswing / downswing, 3),
        "unit": "source frames",
        "seconds_source_timeline": {
            "backswing": round(elapsed(int(start_frame), int(top_frame)), 4),
            "downswing": round(elapsed(int(top_frame), int(impact_frame)), 4),
            "basis": seconds_basis,
        },
        "seconds_basis": seconds_basis,
        "definition": (f"backswing = {basis_marker} frame -> top frame; downswing = "
                       f"top frame -> impact frame; ratio = backswing / downswing"),
        "caveat": ("durations are counts of source frames within the source timeline. "
                   + ("Seconds are differences of the source's own decoded frame "
                      "presentation timestamps, so they locate each marker in the source "
                      "timeline; they are not measured action time."
                      if seconds_basis == SECONDS_BASIS_TIMESTAMPS else
                      "Seconds are container-cadence seconds and are not measured "
                      "action time: a replay or slow-motion source keeps its own "
                      "cadence.")),
        "markers": {"start": start.to_dict(), "top": top.to_dict(),
                    "impact": impact.to_dict()},
    }


def _missing_text(record: Optional[PhaseRecord], name: str) -> str:
    if record is None or record.frame is None:
        return f"no {name} marker is available"
    return ""


def _label_for(*records: PhaseRecord) -> str:
    if any(r.label == LABEL_APPROX for r in records):
        return LABEL_APPROX
    if all(r.label == LABEL_MANUAL for r in records):
        return LABEL_MANUAL
    return LABEL_UNAVAILABLE


def _tempo_unavailable(reason: str) -> dict:
    return {"state": "unavailable", "reason": reason,
            "note": "Tempo is not estimated from evenly spaced frames or from the "
                    "clip length; without the markers it is simply unavailable."}


def phase_cards(records: Dict[str, PhaseRecord]) -> List[dict]:
    """Which cards to render: the four primary phases, then any extra phase given."""
    cards = []
    for name in PRIMARY_PHASES:
        record = records[name]
        cards.append(_card(record, primary=True))
    for name in PHASE_ORDER:
        if name in PRIMARY_PHASES:
            continue
        record = records[name]
        if record.frame is not None:
            cards.append(_card(record, primary=False))
    return cards


def _card(record: PhaseRecord, *, primary: bool) -> dict:
    return {"phase": record.name, "label": record.label, "frame": record.frame,
            "primary": primary, "method": record.method, "note": record.note,
            "angles": record.angles,
            "renderable": record.frame is not None}
