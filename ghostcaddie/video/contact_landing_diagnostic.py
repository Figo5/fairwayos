"""Research-only contact/landing candidate diagnostics.

This module combines *explicit* ball-candidate track points with timestamped
SwingNet/native event hints and generic pose cues. It is deliberately a
candidate generator: it never emits a validated event, impact, or landing.
No production module imports this file.
"""

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Optional, Tuple


@dataclass(frozen=True)
class ContactLandingCandidate:
    kind: str
    timestamp_seconds: float
    confidence: float
    event: str
    event_source: str
    evidence: Tuple[str, ...]
    validated: bool = False


@dataclass(frozen=True)
class ContactLandingDiagnostic:
    available: bool
    candidates: Optional[Tuple[ContactLandingCandidate, ...]]
    reason: Optional[str]
    validated_events: Tuple[Any, ...] = ()
    provenance: str = "research_candidate_diagnostic"


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be numeric" % name)
    if not math.isfinite(result):
        raise ValueError("%s must be finite" % name)
    return result


def _explicit_points(ball_track: Iterable[Any]):
    points = []
    for item in ball_track or ():
        raw_provenance = _value(item, "provenance")
        if not isinstance(raw_provenance, str):
            continue
        provenance = raw_provenance.strip().lower()
        if provenance not in {"observed", "native", "user_confirmed"}:
            continue
        # Accept trajectory points as well as simple mapping records.
        timestamp = _value(item, "timestamp_seconds")
        x, y = _value(item, "x"), _value(item, "y")
        confidence = _value(item, "confidence")
        if timestamp is None or x is None or y is None or confidence is None:
            continue
        points.append((_number(timestamp, "ball timestamp"), _number(x, "ball x"),
                      _number(y, "ball y"), _number(confidence, "ball confidence"), provenance))
    if any(not 0 <= point[3] <= 1 for point in points):
        raise ValueError("ball confidence must be between 0 and 1")
    return sorted(points, key=lambda point: (point[0], point[1], point[2]))


def _pose_score(pose_cues: Iterable[Any], timestamp: float, cue: str, window: float) -> float:
    best = 0.0
    for item in pose_cues or ():
        item_time = _value(item, "timestamp_seconds")
        cues = _value(item, "cues", {}) or {}
        if item_time is None or not isinstance(cues, Mapping):
            continue
        delta = abs(_number(item_time, "pose timestamp") - timestamp)
        if delta <= window and cue in cues:
            value = _number(cues[cue], "pose cue")
            best = max(best, max(0.0, min(1.0, value)))
    return best


def diagnose_contact_landing(
    ball_track: Iterable[Any],
    event_timestamps: Iterable[Any],
    pose_cues: Iterable[Any] = (),
    *,
    min_event_confidence: float = 0.55,
    max_event_ball_delta: float = 0.12,
    max_pose_delta: float = 0.18,
    min_landing_pose_confidence: float = 0.85,
) -> ContactLandingDiagnostic:
    """Return conservative contact/landing candidates or an unavailable result.

    Contact requires a sufficiently confident impact/contact timestamp and a
    nearby explicit ball point. Landing requires either an explicit native
    landing timestamp plus two nearby explicit points, or two nearby points and
    a strong generic ``landing`` pose cue. SwingNet impact is not treated as a
    landing timestamp. All returned candidates remain ``validated=False`` and
    ``validated_events`` is always empty by design.
    """
    if not 0 <= min_event_confidence <= 1 or not 0 <= min_landing_pose_confidence <= 1:
        raise ValueError("confidence thresholds must be between 0 and 1")
    if max_event_ball_delta < 0 or max_pose_delta < 0:
        raise ValueError("time windows must be non-negative")

    points = _explicit_points(ball_track)
    events = []
    for item in event_timestamps or ():
        event = str(_value(item, "event", "")).strip()
        timestamp = _value(item, "timestamp_seconds")
        confidence = _value(item, "confidence", 1.0)
        if not event or timestamp is None:
            continue
        events.append((event, _number(timestamp, "event timestamp"),
                       _number(confidence, "event confidence"),
                       str(_value(item, "source", "unknown")).strip().lower()))
    events.sort(key=lambda item: (item[1], item[0].lower(), item[3]))
    if not events:
        return ContactLandingDiagnostic(False, None, "no_event_timestamp")
    relevant = [event for event in events if event[0].lower() in {"impact", "contact", "hit", "landing", "bounce", "ground_contact"}]
    if not relevant:
        return ContactLandingDiagnostic(False, None, "no_contact_or_landing_event")
    if any(event[2] < min_event_confidence for event in relevant):
        return ContactLandingDiagnostic(False, None, "event_confidence_below_threshold")
    if not points:
        return ContactLandingDiagnostic(False, None, "no_explicit_ball_candidate_near_event")

    candidates = []
    for event, timestamp, event_confidence, source in relevant:
        label = event.lower()
        near = [point for point in points if abs(point[0] - timestamp) <= max_event_ball_delta]
        if label in {"impact", "contact", "hit"} and near:
            point = min(near, key=lambda value: (abs(value[0] - timestamp), -value[3]))
            pose = _pose_score(pose_cues, timestamp, "contact", max_pose_delta)
            score = round(min(1.0, 0.6 * event_confidence + 0.3 * point[3] + 0.1 * pose), 6)
            candidates.append(ContactLandingCandidate("contact", point[0], score, event, source,
                ("explicit_ball_candidate", "event_timestamp") + (("generic_pose_contact",) if pose else ())))
        is_native_landing = label in {"landing", "bounce", "ground_contact"} and source == "native"
        landing_pose = _pose_score(pose_cues, timestamp, "landing", max_pose_delta)
        if (is_native_landing or landing_pose >= min_landing_pose_confidence) and len(near) >= 2:
            score = round(min(1.0, 0.55 * event_confidence + 0.25 * min(p[3] for p in near) + 0.2 * landing_pose), 6)
            evidence = ("explicit_ball_candidate", "event_timestamp")
            if landing_pose >= min_landing_pose_confidence:
                evidence += ("generic_pose_landing",)
            candidates.append(ContactLandingCandidate("landing", timestamp, score, event, source, evidence))

    # A generic pose cue can supply the landing time, but it still needs two
    # explicit ball points. This is a candidate timestamp, not a validated one.
    for item in pose_cues or ():
        cues = _value(item, "cues", {}) or {}
        pose_timestamp = _value(item, "timestamp_seconds")
        if pose_timestamp is None or not isinstance(cues, Mapping) or "landing" not in cues:
            continue
        pose_timestamp = _number(pose_timestamp, "pose timestamp")
        landing_pose = max(0.0, min(1.0, _number(cues["landing"], "pose cue")))
        near = [point for point in points if abs(point[0] - pose_timestamp) <= max_event_ball_delta]
        if landing_pose >= min_landing_pose_confidence and len(near) >= 2:
            ball_confidence = min(point[3] for point in near)
            score = round(min(1.0, 0.55 * landing_pose + 0.45 * ball_confidence), 6)
            candidates.append(ContactLandingCandidate(
                "landing", pose_timestamp, score, "generic_pose_landing", "generic_pose",
                ("explicit_ball_candidate", "generic_pose_landing")))

    if not candidates:
        reason = "no_explicit_ball_candidate_near_event"
        return ContactLandingDiagnostic(False, None, reason)
    candidates.sort(key=lambda candidate: (candidate.timestamp_seconds, candidate.kind, candidate.event_source))
    return ContactLandingDiagnostic(True, tuple(candidates), None)
