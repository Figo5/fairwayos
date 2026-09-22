"""Point-tracking policy: statuses, gaps, losses, corrections, interpolation.

This module is **pure** — it contains no cv2, no numpy and no measurement. The
worker measures one frame at a time (template match, optional optical-flow
refinement) and hands the raw result here; this module decides what that
measurement *is* and what may be drawn from it. Keeping the policy out of the
measurement loop is what makes it testable without decoding a video.

The rules, stated once:

* ``manual_anchor`` — a point the operator placed (seed or correction). Origin is
  ``operator``. It is an observation, but it is never called a detection.
* ``observed_template`` / ``observed_template_lk`` — the tracker matched. Origin is
  ``automatic``. **These are the only automatic observations**, and they are the
  only points that contribute to automatic path length.
* ``interpolated`` — a point derived between two real observations across a bounded
  gap. Origin is ``derived``. It is drawn dashed, counted separately, and never
  counted as an observation.
* ``missing`` — nothing is drawn. Never a stale marker, never a bridged trail.
* ``lost`` — the tracker missed ``loss_frames`` consecutive frames: the target was
  lost, so every remaining frame of that run stays missing. A loss is never
  interpolated, however large ``max_gap_frames`` is, and it is never re-acquired
  automatically: only an operator correction restarts a track.

A gap is interpolated only when it is bounded on both sides by real observations,
is shorter than the loss threshold, and is no longer than ``max_gap_frames``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .errors import DemoError
from .geometry import linear_interpolate, max_step, path_length

STATUS_MANUAL = "manual_anchor"
STATUS_OBSERVED = "observed_template"
STATUS_OBSERVED_LK = "observed_template_lk"
STATUS_INTERPOLATED = "interpolated"
STATUS_MISSING = "missing"
STATUS_LOST = "lost"

ORIGIN_OPERATOR = "operator"
ORIGIN_AUTOMATIC = "automatic"
ORIGIN_DERIVED = "derived"

OBSERVED_STATUSES = frozenset({STATUS_MANUAL, STATUS_OBSERVED, STATUS_OBSERVED_LK})
AUTOMATIC_STATUSES = frozenset({STATUS_OBSERVED, STATUS_OBSERVED_LK})
EMPTY_STATUSES = frozenset({STATUS_MISSING, STATUS_LOST})


@dataclass
class RawMeasurement:
    """One frame of measurement, straight from the worker. No interpretation."""
    source_frame: int
    found: bool
    x: float = 0.0
    y: float = 0.0
    score: float = 0.0
    method: str = "template"
    reason: str = ""
    #: Which way the leg walked over this frame. It matters for losses: a forward
    #: leg's loss forbids later automatic observations, a backward leg's loss
    #: forbids earlier ones (the leg walked from the seed towards the interval
    #: start and gave up further back).
    direction: str = "forward"

    def to_dict(self) -> dict:
        return {"source_frame": int(self.source_frame), "found": bool(self.found),
                "x": round(float(self.x), 3), "y": round(float(self.y), 3),
                "score": round(float(self.score), 4), "method": self.method,
                "reason": self.reason, "direction": self.direction}


@dataclass
class TrackPoint:
    source_frame: int
    status: str
    origin: str
    method: str = ""
    x: Optional[float] = None
    y: Optional[float] = None
    score: Optional[float] = None
    segment: Optional[int] = None
    note: str = ""
    direction: str = "forward"
    #: Which operator-anchored segment this point belongs to, when it came from the
    #: opt-in ``manual_segments`` block. Empty for every classic track.
    manual_segment: Optional[str] = None

    @property
    def observed(self) -> bool:
        return self.status in OBSERVED_STATUSES

    @property
    def automatic(self) -> bool:
        return self.status in AUTOMATIC_STATUSES

    @property
    def drawn(self) -> bool:
        return self.status != STATUS_MISSING and self.status != STATUS_LOST

    def to_dict(self) -> dict:
        return {"source_frame": int(self.source_frame), "status": self.status,
                "origin": self.origin, "method": self.method,
                "x": None if self.x is None else round(float(self.x), 2),
                "y": None if self.y is None else round(float(self.y), 2),
                "score": None if self.score is None else round(float(self.score), 4),
                "segment": self.segment, "observed": self.observed,
                "interpolated": self.status == STATUS_INTERPOLATED,
                "direction": self.direction,
                "manual_segment": self.manual_segment,
                "note": self.note}


@dataclass
class TargetTrack:
    name: str
    mode: str
    direction: str
    available: bool
    points: List[TrackPoint] = field(default_factory=list)
    segments: List[dict] = field(default_factory=list)
    coverage: Dict[str, object] = field(default_factory=dict)
    accounting: Dict[str, object] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    reason: str = ""
    seed: Optional[dict] = None
    corrections: List[dict] = field(default_factory=list)
    #: The opt-in operator-anchored segments applied to this track, if any: the
    #: segments themselves (with every anchor's own note), the hidden ranges, the
    #: trail rule, and what became of the automatic track they replaced.
    manual_segments: Optional[dict] = None

    def to_dict(self) -> dict:
        return {"name": self.name, "mode": self.mode, "direction": self.direction,
                "available": self.available, "reason": self.reason,
                "seed": self.seed, "corrections": self.corrections,
                "coverage": self.coverage, "accounting": self.accounting,
                "limitations": self.limitations, "segments": self.segments,
                "manual_segments": self.manual_segments,
                "points": [p.to_dict() for p in self.points]}

    # -- drawable views (the renderer uses these; nothing else may) -----------
    def drawn_points(self) -> List[TrackPoint]:
        return [p for p in self.points if p.drawn]

    def observed_points(self) -> List[TrackPoint]:
        return [p for p in self.points if p.observed]

    def automatic_points(self) -> List[TrackPoint]:
        return [p for p in self.points if p.automatic]

    def interpolated_points(self) -> List[TrackPoint]:
        return [p for p in self.points if p.status == STATUS_INTERPOLATED]


NO_SEED_REASON = ("no operator seed or correction was supplied for this target, so "
                  "nothing is tracked and nothing is drawn")
MANUAL_MODE_LIMITATION = (
    "This target is in manual anchor mode: every point is an operator placement, "
    "there is no automatic search at all.")
TEMPLATE_MODE_LIMITATION = (
    "Points are template matches around the previous point, bounded by a search "
    "radius and a maximum per-frame step. This is a local appearance tracker, not a "
    "detector: it follows what it was seeded on, and it prefers to lose the target "
    "over following something else.")
INTERPOLATION_LIMITATION = (
    "Dashed points are linear interpolation across a bounded gap between two real "
    "observations. They are derived, not observed: they carry no measurement and are "
    "excluded from every observed quantity and from the automatic path length.")
LOSS_LIMITATION = (
    "After the tracker misses the target for the configured number of consecutive "
    "frames the target is declared lost. Lost frames stay missing: nothing is drawn, "
    "no trail bridges the loss, and the track is only restarted by an operator "
    "correction.")
CORRECTION_LIMITATION = (
    "Operator corrections are hard restart points. The trail does not bridge a "
    "correction, and the image-space step across one is not counted as motion.")


def _anchor_records(seed, corrections: Sequence) -> Dict[int, dict]:
    anchors: Dict[int, dict] = {}
    if seed is not None:
        anchors[int(seed.frame)] = {"frame": int(seed.frame), "x": float(seed.x),
                                    "y": float(seed.y), "source": "seed",
                                    "note": getattr(seed, "note", "")}
    for c in corrections or []:
        anchors[int(c.frame)] = {"frame": int(c.frame), "x": float(c.x),
                                 "y": float(c.y), "source": "correction",
                                 "note": getattr(c, "note", "")}
    return anchors


def resolve_track(name: str, *, options, interval_frames: Sequence[int],
                  raw: Dict[int, RawMeasurement], seed, corrections: Sequence = (),
                  interpolation_enabled: bool = True,
                  max_gap_frames: int = 2) -> TargetTrack:
    """Build the per-frame status record for one target.

    In ``manual`` mode the measurements are ignored outright: the operator said the
    points are theirs, so a template match cannot sneak into that track.
    """
    if options.mode == "manual":
        raw = {}
    anchors = _anchor_records(seed, corrections)
    corrections_used = [a for a in anchors.values() if a["source"] == "correction"]
    corrections_used.sort(key=lambda a: a["frame"])
    stop_frame = (None if options.track_until_frame is None
                  else int(options.track_until_frame))
    #: The operator's declared end of AUTOMATIC matching. Unlike track_until_frame
    #: this never suppresses an operator anchor: the flight is exactly the anchors
    #: the operator placed, and no template match is used past this frame.
    auto_stop = (None if getattr(options, "automatic_until_frame", None) is None
                 else int(options.automatic_until_frame))
    correction_frames = {a["frame"] for a in corrections_used}

    if not anchors:
        track = TargetTrack(name=name, mode=options.mode, direction=options.direction,
                            available=False, reason=NO_SEED_REASON)
        track.points = [TrackPoint(f, STATUS_MISSING, ORIGIN_DERIVED,
                                   note="no seed: nothing was tracked")
                        for f in interval_frames]
        track.coverage = _coverage(track.points)
        track.accounting = _accounting(track.points, [])
        return track

    # --- statuses frame by frame
    points: List[TrackPoint] = []
    for frame in interval_frames:
        if stop_frame is not None and frame > stop_frame:
            points.append(TrackPoint(
                frame, STATUS_MISSING, ORIGIN_DERIVED, method="operator_stop_frame",
                note=(f"past the operator-declared stop frame {stop_frame}: this "
                      f"target is out of scope here, so nothing is drawn and "
                      f"nothing is inferred")))
            continue
        anchor = anchors.get(frame)
        if anchor is not None:
            points.append(TrackPoint(
                frame, STATUS_MANUAL, ORIGIN_OPERATOR, method="operator_placement",
                x=anchor["x"], y=anchor["y"], score=None,
                note=("seed" if anchor["source"] == "seed" else "correction")
                     + (f": {anchor['note']}" if anchor["note"] else "")))
            continue
        if auto_stop is not None and frame > auto_stop:
            # Past the operator's automatic stop the ONLY points that exist are the
            # operator's own anchors. A template match here would be reported as an
            # automatic observation on a stretch of video the operator took the
            # tracker off, so it is not used at all -- measured or not.
            points.append(TrackPoint(
                frame, STATUS_MISSING, ORIGIN_DERIVED,
                method="operator_automatic_stop", direction="forward",
                note=(f"past automatic_until_frame={auto_stop}: the operator stopped "
                      f"automatic matching here, so only operator anchors are used "
                      f"and nothing is inferred between them")))
            continue
        measurement = raw.get(frame)
        if measurement is not None and measurement.found:
            status = (STATUS_OBSERVED_LK if measurement.method == "template+lk"
                      else STATUS_OBSERVED)
            points.append(TrackPoint(frame, status, ORIGIN_AUTOMATIC,
                                     method=measurement.method, x=measurement.x,
                                     y=measurement.y, score=measurement.score,
                                     direction=measurement.direction))
        else:
            reason = measurement.reason if measurement is not None else "not_measured"
            points.append(TrackPoint(frame, STATUS_MISSING, ORIGIN_DERIVED,
                                     method=(measurement.method if measurement else ""),
                                     note=reason,
                                     direction=(measurement.direction
                                                if measurement is not None
                                                else "forward")))

    # --- gaps vs losses, then bounded interpolation
    protected = ({frame for frame in interval_frames
                  if stop_frame is not None and frame > stop_frame}
                 | {frame for frame in interval_frames
                    if auto_stop is not None and frame > auto_stop
                    and frame not in anchors})
    _classify_gaps(points, loss_frames=int(options.loss_frames),
                   protected=protected)
    if interpolation_enabled and max_gap_frames > 0:
        _interpolate(points, max_gap_frames=int(max_gap_frames),
                     protected=protected,
                     blocked_bound_frames=(correction_frames
                                           if options.mode == "template" else set()))

    # --- invariant: nothing automatic may follow a loss without a correction
    _enforce_loss_rule(points, anchors)

    segments = _segments(points, mode=options.mode, anchors=anchors)
    coverage = _coverage(points)
    accounting = _accounting(points, segments)

    track = TargetTrack(name=name, mode=options.mode, direction=options.direction,
                        available=True, points=points, segments=segments,
                        coverage=coverage, accounting=accounting,
                        seed=anchors.get(int(seed.frame)) if seed is not None else None,
                        corrections=corrections_used)
    track.limitations = _limitations(options, coverage, accounting)
    return track


# ---------------------------------------------------------------- internals

def _classify_gaps(points: List[TrackPoint], *, loss_frames: int,
                   protected: Optional[set] = None) -> None:
    """Label each run of empty frames as a short gap or a declared loss.

    Frames past an operator stop frame are ``protected``: they are out of scope by
    the operator's decision, so they are neither a gap nor a loss.
    """
    protected = protected or set()
    i = 0
    n = len(points)
    while i < n:
        if points[i].status not in EMPTY_STATUSES or points[i].source_frame in protected:
            i += 1
            continue
        start = i
        while (i < n and points[i].status in EMPTY_STATUSES
               and points[i].source_frame not in protected):
            i += 1
        run = points[start:i]
        is_loss = len(run) >= loss_frames
        for point in run:
            original = point.note
            detail = f"; measurement: {original}" if original else ""
            if is_loss:
                point.status = STATUS_LOST
                point.note = (f"tracker missed {len(run)} consecutive frames "
                              f"(>= loss_frames={loss_frames}); target declared lost"
                              + detail)
            else:
                point.status = STATUS_MISSING
                point.note = (f"short gap: {len(run)} consecutive frame(s) with no "
                              f"bounded match (loss_frames={loss_frames})" + detail)


def _keep(note: str) -> str:
    """Keep an earlier measurement reason when a later stage rewrites a note."""
    return f"; measurement: {note}" if note else ""


def _interpolate(points: List[TrackPoint], *, max_gap_frames: int,
                 protected: Optional[set] = None,
                 blocked_bound_frames: Optional[set] = None) -> None:
    """Fill bounded gaps only. A loss is not a gap and is never filled.

    Two further refusals, both because a filled point would be a claim the operator
    never made:

    * a gap that ENDS at a correction (in template mode) is not filled: the
      correction may be a large relocation, and interpolating toward it would draw
      invented approach points to a place the target never was;
    * frames past the operator's stop frame are never filled.
    """
    protected = protected or set()
    blocked = blocked_bound_frames or set()
    i = 0
    n = len(points)
    while i < n:
        if points[i].status != STATUS_MISSING or points[i].source_frame in protected:
            i += 1
            continue
        start = i
        while (i < n and points[i].status == STATUS_MISSING
               and points[i].source_frame not in protected):
            i += 1
        stop = i
        before = points[start - 1] if start > 0 else None
        after = points[stop] if stop < n else None
        run = points[start:stop]
        if (before is None or after is None or not before.observed or not after.observed
                or before.x is None or before.y is None
                or after.x is None or after.y is None):
            for point in run:
                point.note = ("gap is not bounded by observations on both sides, so "
                              "nothing is filled" + _keep(point.note))
            continue
        if after.source_frame in blocked:
            for point in run:
                point.note = ("this gap ends at an operator correction, which is a "
                              "restart boundary: it is not filled, because "
                              "interpolating toward a corrected position would draw "
                              "points the target was never observed at"
                              + _keep(point.note))
            continue
        if len(run) > max_gap_frames:
            for point in run:
                point.note = (f"gap of {len(run)} frames exceeds "
                              f"interpolation.max_gap_frames={max_gap_frames}; left "
                              f"missing" + _keep(point.note))
            continue
        anchor_a = (float(before.x), float(before.y))
        anchor_b = (float(after.x), float(after.y))
        for index, point in enumerate(run):
            t = (index + 1) / (len(run) + 1)
            x, y = linear_interpolate(anchor_a, anchor_b, t)
            point.status = STATUS_INTERPOLATED
            point.origin = ORIGIN_DERIVED
            point.method = "linear_interpolation"
            point.x, point.y = x, y
            point.score = None
            point.note = (f"derived by linear interpolation across a {len(run)}-frame "
                          f"gap between source frames {before.source_frame} and "
                          f"{after.source_frame}")


def _enforce_loss_rule(points: List[TrackPoint], anchors: Dict[int, dict]) -> None:
    """A track may not resume automatically after a declared loss.

    A loss is terminal *in the direction the leg was walking*:

    * a forward leg that lost the target may not be followed by a later automatic
      observation, because that would be the tracker quietly re-acquiring on its own;
    * a backward leg (walking from the operator's seed towards the interval start)
      that gave up may not be followed by an *earlier* automatic observation, for the
      same reason.

    In both cases an operator anchor between the loss and the observation restarts
    the track legitimately, so that is allowed. The rule is recorded here rather than
    in the worker because a track assembled from several legs is where a re-acquired
    point could slip in.
    """
    anchor_frames = sorted(int(f) for f in anchors)
    for index, point in enumerate(points):
        if point.status != STATUS_LOST:
            continue
        backward = point.direction == "backward"
        for other in (points[index + 1:] if not backward
                      else reversed(points[:index])):
            if other.status == STATUS_MANUAL:
                break                       # an operator restart ends the objection
            if other.status == STATUS_LOST:
                continue
            if other.status not in AUTOMATIC_STATUSES:
                continue
            low, high = sorted((point.source_frame, other.source_frame))
            if any(low < f <= high for f in anchor_frames):
                continue
            raise DemoError(
                f"internal inconsistency ({other.direction} leg): an automatic "
                f"observation at frame {other.source_frame} follows a declared loss "
                f"at frame {point.source_frame} with no operator anchor in between; "
                f"refusing to render a re-acquired track")


def _segments(points: List[TrackPoint], *, mode: str,
              anchors: Dict[int, dict]) -> List[dict]:
    """Maximal drawable runs. A trail never bridges a gap, a loss or a correction."""
    segments: List[dict] = []
    previous: Optional[TrackPoint] = None
    for point in points:
        if not point.drawn:
            point.segment = None
            previous = point
            continue
        breaks = False
        if previous is None or not previous.drawn:
            breaks = True
        elif (mode == "template" and point.source_frame in anchors
              and anchors[point.source_frame]["source"] == "correction"):
            breaks = True
        if breaks:
            segments.append({"index": len(segments), "start_frame": point.source_frame,
                             "end_frame": point.source_frame, "frames": 1,
                             "observed_frames": 0, "interpolated_frames": 0})
        else:
            current = segments[-1]
            current["end_frame"] = point.source_frame
            current["frames"] += 1
        if point.observed:
            segments[-1]["observed_frames"] += 1
        if point.status == STATUS_INTERPOLATED:
            segments[-1]["interpolated_frames"] += 1
        point.segment = segments[-1]["index"]
        previous = point
    return segments


def _coverage(points: Sequence[TrackPoint]) -> dict:
    total = len(points)
    observed = [p for p in points if p.observed]
    manual = [p for p in points if p.status == STATUS_MANUAL]
    automatic = [p for p in points if p.automatic]
    interpolated = [p for p in points if p.status == STATUS_INTERPOLATED]
    missing = [p for p in points if p.status == STATUS_MISSING]
    lost = [p for p in points if p.status == STATUS_LOST]
    longest_gap = 0
    run = 0
    for p in points:
        if p.status in EMPTY_STATUSES:
            run += 1
            longest_gap = max(longest_gap, run)
        else:
            run = 0
    protected = [p for p in points
                 if p.method == "operator_stop_frame"]
    automatic_stopped = [p for p in points
                         if p.method == "operator_automatic_stop"]
    return {
        "frames_total": total,
        "observed_frames": len(observed),
        "automatic_observed_frames": len(automatic),
        "operator_anchor_frames": len(manual),
        "interpolated_frames": len(interpolated),
        # ``missing_frames`` and ``lost_frames`` are DISJOINT status counts: a frame
        # is either a short miss/no-match or a declared loss, never both. The report
        # adds them; a reader must not treat "lost" as a subset of "missing".
        "missing_frames": len(missing),
        "lost_frames": len(lost),
        "unobserved_frames": len(missing) + len(lost),
        #: frames the operator deliberately put out of scope with track_until_frame
        "operator_stop_frames": len(protected),
        #: frames between the operator's anchors past automatic_until_frame: no
        #: automatic observation exists there by the operator's decision
        "automatic_stop_frames": len(automatic_stopped),
        "longest_missing_run": longest_gap,
        "observed_fraction": round(len(observed) / total, 4) if total else 0.0,
        "interpolated_fraction": round(len(interpolated) / total, 4) if total else 0.0,
    }


def _steps(points: Sequence[TrackPoint], predicate) -> List[float]:
    """Consecutive-frame steps between points satisfying ``predicate``.

    Non-consecutive frames are not steps: a gap, a loss or a correction is a
    discontinuity, and counting the jump across one would report motion that was
    never observed.
    """
    ordered = [p for p in points if p.drawn]
    steps: List[float] = []
    for a, b in zip(ordered, ordered[1:]):
        if a.x is None or a.y is None or b.x is None or b.y is None:
            continue
        if b.source_frame != a.source_frame + 1:
            continue
        if a.segment != b.segment:
            continue
        if not (predicate(a) and predicate(b)):
            continue
        steps.append(math.hypot(float(b.x) - float(a.x), float(b.y) - float(a.y)))
    return steps


def _accounting(points: Sequence[TrackPoint], segments: Sequence[dict]) -> dict:
    observed_steps = _steps(points, lambda p: p.observed)
    operator_steps = _steps(points, lambda p: p.status == STATUS_MANUAL)
    any_steps = _steps(points, lambda p: True)
    rejected = [p.note for p in points
                if p.status in EMPTY_STATUSES and p.note]
    return {
        "path_length_observed_px": round(sum(observed_steps), 2),
        "observed_step_count": len(observed_steps),
        "max_observed_step_px": round(max(observed_steps), 2) if observed_steps else None,
        "path_length_operator_anchors_px": round(sum(operator_steps), 2),
        "operator_step_count": len(operator_steps),
        "path_length_including_interpolated_px": round(sum(any_steps), 2),
        "segment_count": len(segments),
        "space": "2d_image",
        "unit": "px",
        "note": ("Path lengths sum consecutive-frame steps inside one segment only, "
                 "between points that were actually observed (template matches or the "
                 "operator's own placements). Steps across a gap, a loss or a "
                 "correction are excluded, as are interpolated (derived) steps."),
        "empty_frame_reasons": _reason_histogram(rejected),
    }


def _reason_histogram(notes: Iterable[str]) -> Dict[str, int]:
    """Count why frames carry no point, keeping the measurement reason visible.

    A frame can be classified twice (first as a short gap, then as a gap the
    interpolation limit refused to fill), so both the classification and the
    tracker's own measurement reason are counted.
    """
    out: Dict[str, int] = {}
    for note in notes:
        head = note.split("; measurement: ")[0][:60]
        out[head] = out.get(head, 0) + 1
        if "; measurement: " in note:
            detail = note.split("; measurement: ", 1)[1][:60]
            out[detail] = out.get(detail, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def trail_window(points: Sequence[TrackPoint], current_frame: int,
                 limit: int = 15) -> List[TrackPoint]:
    """The drawable trail for ``current_frame``: one segment, no bridging, no stale.

    The renderer must never connect two points that were not observed back to back
    inside the same segment, so this helper is the single place that decides which
    points may be joined:

    * nothing is drawn at all unless the CURRENT frame has a drawable point, so a
      miss, a gap or a loss leaves the picture clean instead of leaving a tail
      pointing at where the target used to be;
    * points after ``current_frame`` are excluded (nothing is drawn ahead of time);
    * only the current segment is drawn, so a trail cannot bridge a loss or a
      correction;
    * missing and lost frames are never in the window.
    """
    frame = int(current_frame)
    current = next((p for p in points if p.source_frame == frame), None)
    if current is None or not current.drawn:
        return []
    window = [p for p in points
              if p.drawn and p.segment == current.segment and p.source_frame <= frame]
    return window[-int(limit):] if limit else window


# --------------------------------------------------------------------------
# opt-in operator-anchored segments
#
# The classic path above is a tracker with operator restarts. This path is the other
# way round: the operator's own points ARE the presentation, and the only thing
# derived from them is a LINEAR interpolation inside a segment whose anchor spacing
# the config already bounded. Nothing here can create a point outside a segment, and
# nothing can bridge a hidden or unanchored stretch.

MANUAL_ANCHOR_METHOD = "operator_manual_anchor"
MANUAL_INTERPOLATED_METHOD = "linear_interpolation_within_manual_anchor_segment"
MANUAL_HIDDEN_METHOD = "operator_hidden_frame"
MANUAL_OUTSIDE_METHOD = "outside_manual_anchor_segments"

CLASS_MANUAL_SEGMENT = "manual_segment"
CLASS_OPERATOR_ANCHOR = "operator_anchor"
CLASS_AUTOMATIC = "automatic"
CLASS_EMPTY = "empty"

MANUAL_SEGMENTS_LIMITATION = (
    "This target's presentation comes from the operator's own anchor segments "
    "(`manual_segments`): every anchor is an operator placement, every frame between "
    "two anchors of one segment is a linear interpolation labelled derived, and no point "
    "outside those segments is added by this block. There is no extrapolation past "
    "either end of a segment and no bridging across an unanchored stretch.")
MANUAL_MIXED_LIMITATION = (
    "Runs outside the operator's segments are the automatic tracker's own points "
    "(measured, missing or lost): the operator's anchors add the stretches they "
    "verified and do not extend or relabel those frames.")
MANUAL_REPLACED_LIMITATION = (
    "This target's automatic track was REPLACED for presentation by the operator's "
    "anchor segments (`replace_automatic`). The automatic measurements are not drawn "
    "and are not counted as this target's path; the segments are the whole of what is "
    "shown, and every frame outside them is missing.")


def point_class(point: TrackPoint) -> str:
    """What a point is, for the one rule that decides whether a trail may join two.

    The classes are deliberately coarse, and a trail never crosses between them: an
    operator-placed segment is not joined to an automatic observation, so a trail can
    never imply that the tracker followed the operator's anchors (or the reverse).
    """
    if not point.drawn:
        return CLASS_EMPTY
    if point.method in (MANUAL_ANCHOR_METHOD, MANUAL_INTERPOLATED_METHOD):
        return CLASS_MANUAL_SEGMENT
    if point.status in AUTOMATIC_STATUSES:
        return CLASS_AUTOMATIC
    return CLASS_OPERATOR_ANCHOR


def is_manual_segment_point(point: TrackPoint) -> bool:
    return point_class(point) == CLASS_MANUAL_SEGMENT


def _anchor_xy(item: Any) -> Dict[str, Any]:
    """One anchor in either shape: ``{"frame", "x", "y", "note"}`` or ``[f, x, y]``.

    The list form is the shape the operator's own annotation files use; normalising it
    here (as well as in the config parser) means the rest of this module only ever sees
    one shape.
    """
    if isinstance(item, Mapping):
        return {"frame": int(item["frame"]), "x": float(item["x"]),
                "y": float(item["y"]), "note": str(item.get("note") or "")}
    values = list(item)
    return {"frame": int(values[0]), "x": float(values[1]), "y": float(values[2]),
            "note": str(values[3]) if len(values) > 3 else ""}


def apply_manual_segments(track: TargetTrack, *, manual: Mapping[str, Any],
                          interval_frames: Sequence[int]) -> TargetTrack:
    """Overlay the operator's anchor segments onto a track, in place.

    ``manual`` is the resolved ``manual_segments.<target>`` payload. The rules:

    * every anchor becomes a ``manual_anchor`` point (origin ``operator``) carrying the
      operator's own note and the segment it belongs to;
    * every frame between two anchors of the SAME segment becomes ``interpolated``
      (origin ``derived``) and records the two anchors it was derived from;
    * ``hide_frames`` are emptied explicitly, and they are never interpolated across;
    * with ``replace_automatic`` the automatic points are dropped for presentation and
      every frame outside the segments is emptied; without it they are kept, so the
      operator's anchors sit on top of a track that is still reported as automatic;
    * the frame count of the track never changes: a frame the operator did not anchor
      and the tracker did not measure stays ``missing``.
    """
    segments_in = list(manual.get("segments") or [])
    hidden: List[int] = []
    for first, last in (manual.get("hide_frames") or []):
        hidden.extend(range(int(first), int(last) + 1))
    hidden_set = set(hidden)
    author = str(manual.get("author") or "")
    replace = bool(manual.get("replace_automatic"))

    by_frame: Dict[int, TrackPoint] = {int(p.source_frame): p for p in track.points}
    before = {
        "automatic_observed_frames": sum(1 for p in track.points if p.automatic),
        "operator_anchor_frames": sum(1 for p in track.points
                                      if p.status == STATUS_MANUAL),
        "interpolated_frames": sum(1 for p in track.points
                                   if p.status == STATUS_INTERPOLATED),
        "segment_count": len(track.segments),
        "note": ("what the track held BEFORE the operator's segments were applied: "
                 "this is the automatic measurement this block overrides or replaces"),
    }
    for frame in interval_frames:
        frame = int(frame)
        if frame not in by_frame:
            by_frame[frame] = TrackPoint(frame, STATUS_MISSING, ORIGIN_DERIVED,
                                        note="this frame carries no measurement")

    if replace:
        for frame, point in by_frame.items():
            if point.drawn:
                point.status = STATUS_MISSING
                point.origin = ORIGIN_DERIVED
                point.method = MANUAL_OUTSIDE_METHOD
                point.x = point.y = None
                point.score = None
                point.manual_segment = None
                point.note = ("outside the operator's verified visible segments: "
                              "nothing is drawn here and nothing is inferred")

    for frame in sorted(hidden_set):
        point = by_frame.get(frame)
        if point is None:
            continue
        point.status = STATUS_MISSING
        point.origin = ORIGIN_DERIVED
        point.method = MANUAL_HIDDEN_METHOD
        point.x = point.y = None
        point.score = None
        point.manual_segment = None
        point.note = ("the operator declared this frame unusable (occlusion or an "
                      "unreliable reading): nothing is drawn and nothing is "
                      "interpolated across it")

    segment_records: List[dict] = []
    for segment in segments_in:
        name = str(segment.get("name") or "segment")
        spacing = int(segment.get("max_anchor_spacing_frames") or 1)
        anchors = sorted((_anchor_xy(a) for a in (segment.get("anchors") or [])),
                         key=lambda a: int(a["frame"]))
        if not anchors:
            continue
        anchor_frames = [int(a["frame"]) for a in anchors]
        anchor_by_frame = {int(a["frame"]): a for a in anchors}
        derived = 0
        for frame in range(anchor_frames[0], anchor_frames[-1] + 1):
            if frame in hidden_set:
                continue
            point = by_frame.get(frame)
            if point is None:
                continue
            anchor = anchor_by_frame.get(frame)
            if anchor is not None:
                note = f"operator anchor in segment '{name}'"
                if author:
                    note += f" (author: {author})"
                if anchor.get("note"):
                    note += f": {anchor['note']}"
                point.status = STATUS_MANUAL
                point.origin = ORIGIN_OPERATOR
                point.method = MANUAL_ANCHOR_METHOD
                point.x = float(anchor["x"])
                point.y = float(anchor["y"])
                point.score = None
                point.manual_segment = name
                point.note = note
                continue
            before_frame = max(f for f in anchor_frames if f < frame)
            after_frame = min(f for f in anchor_frames if f > frame)
            if after_frame - before_frame > spacing:  # pragma: no cover - config refuses
                continue
            t = (frame - before_frame) / float(after_frame - before_frame)
            x, y = linear_interpolate((float(anchor_by_frame[before_frame]["x"]),
                                       float(anchor_by_frame[before_frame]["y"])),
                                      (float(anchor_by_frame[after_frame]["x"]),
                                       float(anchor_by_frame[after_frame]["y"])), t)
            point.status = STATUS_INTERPOLATED
            point.origin = ORIGIN_DERIVED
            point.method = MANUAL_INTERPOLATED_METHOD
            point.x, point.y = x, y
            point.score = None
            point.manual_segment = name
            point.note = (f"derived by linear interpolation between the operator "
                          f"anchors at source frames {before_frame} and "
                          f"{after_frame} in segment "
                          f"'{name}' (max spacing {spacing} frame(s)); not an "
                          f"observation and not a measurement")
            derived += 1
        segment_records.append({
            "name": name,
            "start_frame": anchor_frames[0],
            "end_frame": anchor_frames[-1],
            "max_anchor_spacing_frames": spacing,
            "anchor_frames": anchor_frames,
            "anchor_count": len(anchor_frames),
            "interpolated_frames": derived,
            "derived_rule": ("linear between the two surrounding anchors of THIS "
                             "segment only; no extrapolation past either end"),
            "note": str(segment.get("note") or ""),
            "author": author or "operator",
        })

    ordered = [by_frame[int(f)] for f in interval_frames if int(f) in by_frame]
    track.points = ordered
    track.segments = _class_runs(ordered)
    track.coverage = _coverage(ordered)
    track.accounting = _accounting(ordered, track.segments)
    track.available = any(p.drawn for p in ordered)
    if track.available:
        track.reason = ""
    manual_record = {
        "origin": "operator",
        "author": author or "operator",
        "replace_automatic": replace,
        "trail": str(manual.get("trail") or "all"),
        "trail_from_frame": (None if manual.get("trail_from_frame") is None
                             else int(manual["trail_from_frame"])),
        "hide_frames": [[int(a), int(b)] for a, b in (manual.get("hide_frames") or [])],
        "hidden_frames": sorted(hidden_set),
        "segments": segment_records,
        "anchor_count": sum(record["anchor_count"] for record in segment_records),
        "interpolated_frames": sum(record["interpolated_frames"]
                                   for record in segment_records),
        "track_before_manual_segments": before,
        "rule": ("anchors are operator placements; frames between two anchors of one "
                 "segment are linearly interpolated (derived); no point outside the "
                 "segments is added by this block and none is extrapolated"
                 + ("; every frame outside them is missing (the automatic track was "
                    "replaced)" if replace else
                    "; outside them the target is the automatic track's own points, "
                    "unchanged")),
    }
    if replace:
        manual_record["replaced_automatic_track"] = {
            "automatic_observed_frames_dropped": before["automatic_observed_frames"],
            "note": ("the automatic measurement still exists in the run's raw records "
                     "and in the analysis JSON, but it is not drawn and is not counted "
                     "as this target's presented path"),
        }
    track.manual_segments = manual_record
    if MANUAL_SEGMENTS_LIMITATION not in track.limitations:
        track.limitations.append(MANUAL_SEGMENTS_LIMITATION)
    if not replace and MANUAL_MIXED_LIMITATION not in track.limitations:
        track.limitations.append(MANUAL_MIXED_LIMITATION)
    if replace and MANUAL_REPLACED_LIMITATION not in track.limitations:
        track.limitations.append(MANUAL_REPLACED_LIMITATION)
    return track


def manual_segments_only(track: TargetTrack, points: Sequence[TrackPoint]
                         ) -> List[TrackPoint]:
    """Clip a drawn window for presentation: the target's own trail rules.

    Two opt-in rules, both per target, both recorded:

    * ``trail_from_frame``: nothing is trailed before that frame, so a trail cannot
      start on a stretch the operator did not ask to trail (the marker itself is
      unaffected — this is the trail only);
    * ``trail: "segments_only"``: nothing is trailed outside the operator's own
      anchored segments. That is how a trail stops tracing the camera's pan behind a
      ball that is sitting still.
    """
    manual = track.manual_segments or {}
    out = list(points)
    start = manual.get("trail_from_frame")
    if start is not None:
        out = [p for p in out if int(p.source_frame) >= int(start)]
    if str(manual.get("trail") or "all") != "segments_only":
        return out
    return [p for p in out if is_manual_segment_point(p)]


def _class_runs(points: Sequence[TrackPoint]) -> List[dict]:
    """Maximal drawable runs at one frame per frame and one class per run.

    A run breaks on a missing frame, on a non-consecutive frame (the trail cannot
    bridge a gap) and whenever the class of the point changes, so an operator-anchored
    stretch is never presented as a continuation of an automatic one.
    """
    segments: List[dict] = []
    previous: Optional[TrackPoint] = None
    for point in points:
        point.segment = None
        if not point.drawn:
            previous = None
            continue
        cls = point_class(point)
        joins = (previous is not None and previous.drawn
                 and point.source_frame == previous.source_frame + 1
                 and point_class(previous) == cls
                 and point.manual_segment == previous.manual_segment)
        if joins:
            current = segments[-1]
            current["end_frame"] = point.source_frame
            current["frames"] += 1
        else:
            segments.append({"index": len(segments), "start_frame": point.source_frame,
                             "end_frame": point.source_frame, "frames": 1,
                             "observed_frames": 0, "interpolated_frames": 0,
                             "class": cls, "manual_segment": point.manual_segment})
        current = segments[-1]
        if point.observed:
            current["observed_frames"] += 1
        if point.status == STATUS_INTERPOLATED:
            current["interpolated_frames"] += 1
        point.segment = current["index"]
        previous = point
    return segments


def track_from_dict(name: str, payload: dict) -> TargetTrack:
    """Rebuild a TargetTrack from its serialised form.

    The orchestrator works on the worker's own records so the report, the analysis
    and the video cannot be describing different runs. ``points`` round-trip; the
    derived properties (``observed``, ``automatic``, ``drawn``) are recomputed from
    the status instead of being trusted from the file.
    """
    track = TargetTrack(
        name=str(payload.get("name") or name),
        mode=str(payload.get("mode") or "template"),
        direction=str(payload.get("direction") or "forward"),
        available=bool(payload.get("available")),
        reason=str(payload.get("reason") or ""),
        seed=payload.get("seed") or None,
        corrections=list(payload.get("corrections") or []),
        coverage=dict(payload.get("coverage") or {}),
        accounting=dict(payload.get("accounting") or {}),
        limitations=list(payload.get("limitations") or []),
        segments=list(payload.get("segments") or []),
        manual_segments=dict(payload.get("manual_segments") or {}) or None)
    for record in payload.get("points") or []:
        track.points.append(TrackPoint(
            source_frame=int(record["source_frame"]),
            status=str(record["status"]),
            origin=str(record.get("origin") or ORIGIN_DERIVED),
            method=str(record.get("method") or ""),
            x=None if record.get("x") is None else float(record["x"]),
            y=None if record.get("y") is None else float(record["y"]),
            score=None if record.get("score") is None else float(record["score"]),
            segment=record.get("segment"),
            direction=str(record.get("direction") or "forward"),
            manual_segment=record.get("manual_segment"),
            note=str(record.get("note") or "")))
    return track


def _limitations(options, coverage: dict, accounting: dict) -> List[str]:
    out: List[str] = []
    out.append(MANUAL_MODE_LIMITATION if options.mode == "manual"
               else TEMPLATE_MODE_LIMITATION)
    stop = getattr(options, "automatic_until_frame", None)
    if stop is not None and options.mode != "manual":
        out.append(f"Automatic matching for this target was stopped by the operator "
                   f"after source frame {int(stop)} "
                   f"(`automatic_until_frame`), so no template match past that frame is "
                   f"used or reported; beyond it the target is exactly the operator's "
                   f"own anchors.")
    out.append(INTERPOLATION_LIMITATION if coverage["interpolated_frames"]
               else "No gap was interpolated on this target.")
    if coverage["lost_frames"]:
        out.append(LOSS_LIMITATION)
    if accounting["segment_count"] > 1:
        out.append(CORRECTION_LIMITATION)
    return out
