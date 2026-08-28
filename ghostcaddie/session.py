"""Multi-shot / multi-hole session orchestration.

A thin, validated session layer on top of the unchanged single-shot pipeline.
`parse_session()` is the sole inline-ingestion boundary: it parses one
versioned envelope exactly once, applies each hole's CoordinateMapper exactly
once, and produces a normalized in-memory `SessionInput` whose shot positions
are already engine coordinates. `run_session()` then invokes the unchanged
`run_pipeline()` once per ordered shot with a per-shot SHA-256-derived seed and
aggregates only local recommendation metadata (decision_cost and rounded
hazard probabilities) — never aggregate expected-strokes metrics.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Optional

from .adapters.json_file import (
    _parse_course,
    _parse_player,
    _parse_shot,
    _require_dict,
    _require_str,
    _positive_integral,
    _finite_number,
)
from .config import Config
from .geometry import CoordinateMapper
from .models import CourseModel, PlayerProfile, ShotEvent
from .pipeline import run_pipeline

SESSION_SCHEMA_VERSION = "0.1"
_DEFAULT_SESSION_SEED = 42
_MAX_HIGHLIGHTED_DECISIONS = 3


@dataclass
class NormalizedShot:
    """One ordered shot: session identity plus the parsed engine-coordinate ShotEvent."""

    shot_id: str
    hole_number: int
    shot_number: int
    shot: ShotEvent


@dataclass
class SessionInput:
    """Validated, normalized session. All shot positions are engine coordinates."""

    schema_version: str
    session_id: str
    tournament_id: str
    player_id: str
    course_id: str
    round_number: int
    seed: int
    player: PlayerProfile
    courses: Dict[int, CourseModel]  # hole_number -> CourseModel
    shots: List[NormalizedShot]  # strictly ordered by (hole_number, shot_number)
    metadata: Dict[str, object]  # optional envelope metadata for provenance


class InMemoryShotSource:
    """Protocol-compatible shot source returning a pre-parsed engine-coordinate ShotEvent."""

    def __init__(self, shot: ShotEvent, identifier: str):
        self.shot = shot
        self.path = identifier  # descriptive non-path identifier for provenance

    def load_shot(self) -> ShotEvent:
        return self.shot


class InMemoryCourseSource:
    def __init__(self, course: CourseModel, identifier: str):
        self.course = course
        self.path = identifier

    def load_course(self) -> CourseModel:
        return self.course


class InMemoryPlayerSource:
    def __init__(self, player: PlayerProfile, identifier: str):
        self.player = player
        self.path = identifier

    def load_player(self) -> PlayerProfile:
        return self.player


def _parse_session_section(raw) -> dict:
    _require_dict(raw, "session")
    session_id = _require_str(raw.get("session_id"), "session.session_id")
    tournament_id = _require_str(raw.get("tournament_id"), "session.tournament_id")
    player_id = _require_str(raw.get("player_id"), "session.player_id")
    course_id = _require_str(raw.get("course_id"), "session.course_id")
    round_number = _positive_integral(raw.get("round_number"), "session.round_number")
    seed = _finite_number(raw.get("seed", _DEFAULT_SESSION_SEED), "session.seed")
    if seed != int(seed):
        raise ValueError(f"session.seed must be an integer, got {seed!r}")
    return {
        "session_id": session_id,
        "tournament_id": tournament_id,
        "player_id": player_id,
        "course_id": course_id,
        "round_number": round_number,
        "seed": int(seed),
    }


def _parse_course_section(raw) -> Dict[int, CourseModel]:
    _require_dict(raw, "course")
    course_id = _require_str(raw.get("course_id"), "course.course_id")
    holes_raw = raw.get("holes")
    if not isinstance(holes_raw, list) or not holes_raw:
        raise ValueError("course.holes must be a non-empty list")
    courses: Dict[int, CourseModel] = {}
    for i, hole_raw in enumerate(holes_raw):
        _require_dict(hole_raw, f"course.holes[{i}]")
        hole_number = _positive_integral(hole_raw.get("hole_number"), f"course.holes[{i}].hole_number")
        if hole_number in courses:
            raise ValueError(f"duplicate hole_number {hole_number} in course.holes")
        courses[hole_number] = _parse_course(hole_raw)
    return courses


def _parse_shot_record(
    raw,
    index: int,
    session: dict,
    courses: Dict[int, CourseModel],
    mappers: Dict[int, CoordinateMapper],
) -> NormalizedShot:
    _require_dict(raw, f"shots[{index}]")
    shot_id = _require_str(raw.get("shot_id"), f"shots[{index}].shot_id")
    hole_number = _positive_integral(raw.get("hole_number"), f"shots[{index}].hole_number")
    shot_number = _positive_integral(raw.get("shot_number"), f"shots[{index}].shot_number")
    # Optional identity fields must match the session when present; when absent
    # they are inherited from the session (ShotEvent requires them).
    shot_raw = dict(raw)
    for field in ("player_id", "tournament_id", "course_id"):
        if field in raw:
            value = raw[field]
            _require_str(value, f"shots[{index}].{field}")
            if value != session[field]:
                raise ValueError(
                    f"shots[{index}].{field} ({value!r}) does not match session "
                    f"{field} ({session[field]!r})"
                )
        else:
            shot_raw[field] = session[field]
    if hole_number not in courses:
        raise ValueError(
            f"shot {shot_id!r} references hole_number {hole_number} which is not "
            "declared in course.holes"
        )
    shot = _parse_shot(shot_raw, mappers[hole_number])
    return NormalizedShot(
        shot_id=shot_id,
        hole_number=hole_number,
        shot_number=shot_number,
        shot=shot,
    )


def parse_session(raw: dict) -> SessionInput:
    """Parse and validate the complete envelope exactly once. No analytics."""
    _require_dict(raw, "envelope")
    _walk_finite(raw)
    schema_version = raw.get("schema_version")
    if schema_version != SESSION_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SESSION_SCHEMA_VERSION!r}, got {schema_version!r}"
        )
    for section in ("session", "player_profile", "course", "shots"):
        if section not in raw:
            raise ValueError(f"envelope missing required section {section!r}")

    session = _parse_session_section(raw["session"])
    player = _parse_player(raw["player_profile"])
    if player.player_id != session["player_id"]:
        raise ValueError(
            f"player_profile.player_id ({player.player_id!r}) does not match "
            f"session.player_id ({session['player_id']!r})"
        )
    courses = _parse_course_section(raw["course"])
    if raw["course"].get("course_id") != session["course_id"]:
        raise ValueError(
            f"course.course_id ({raw['course'].get('course_id')!r}) does not match "
            f"session.course_id ({session['course_id']!r})"
        )
    mappers = {
        hole_number: CoordinateMapper(course.coordinate_system)
        for hole_number, course in courses.items()
    }

    shots_raw = raw["shots"]
    if not isinstance(shots_raw, list) or not shots_raw:
        raise ValueError("shots must be a non-empty list")
    shots: List[NormalizedShot] = []
    seen_ids = set()
    seen_ordinals = set()
    prev = None
    for i, shot_raw in enumerate(shots_raw):
        ns = _parse_shot_record(shot_raw, i, session, courses, mappers)
        if ns.shot_id in seen_ids:
            raise ValueError(f"duplicate shot_id {ns.shot_id!r}")
        seen_ids.add(ns.shot_id)
        ordinal = (ns.hole_number, ns.shot_number)
        if ordinal in seen_ordinals:
            raise ValueError("duplicate (hole_number, shot_number) pairs")
        seen_ordinals.add(ordinal)
        if prev is not None and ordinal <= prev:
            raise ValueError(
                "shots must be strictly ordered by (hole_number, shot_number); "
                f"found {ordinal!r} after {prev!r}"
            )
        prev = ordinal
        shots.append(ns)

    known = {"schema_version", "session", "player_profile", "course", "shots"}
    metadata = {k: v for k, v in raw.items() if k not in known}

    return SessionInput(
        schema_version=schema_version,
        session_id=session["session_id"],
        tournament_id=session["tournament_id"],
        player_id=session["player_id"],
        course_id=session["course_id"],
        round_number=session["round_number"],
        seed=session["seed"],
        player=player,
        courses=courses,
        shots=shots,
        metadata=metadata,
    )


def derive_shot_seed(session_seed: int, ordinal: int, shot_id: str) -> int:
    """Stable per-shot seed: SHA-256 over session seed, zero-based ordinal, shot_id."""
    digest = hashlib.sha256(
        f"{session_seed}:{ordinal}:{shot_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest, byteorder="big")


def _hazard_risk_summary(prob_lists: List[Dict[str, float]]) -> Dict[str, dict]:
    """Aggregate rounded hazard probabilities, treating omitted hazards as zero."""
    regions = sorted({r for d in prob_lists for r in d})
    summary: Dict[str, dict] = {}
    for region in regions:
        vals = [d.get(region, 0.0) for d in prob_lists]
        summary[region] = {
            "max": max(vals),
            "mean": sum(vals) / len(vals),
            "nonzero_shot_count": sum(1 for v in vals if v > 0),
        }
    return summary


def run_session(session: SessionInput, config: Config) -> dict:
    """Run unchanged run_pipeline() once per ordered shot; aggregate local metadata."""
    shot_results = []
    for ordinal, ns in enumerate(session.shots):
        seed = derive_shot_seed(session.seed, ordinal, ns.shot_id)
        shot_config = replace(
            config, simulation=replace(config.simulation, random_seed=seed)
        )
        course = session.courses[ns.hole_number]
        shot_source = InMemoryShotSource(ns.shot, f"session:inline:shot:{ns.shot_id}")
        course_source = InMemoryCourseSource(
            course, f"session:inline:course:{session.course_id}:hole:{ns.hole_number}"
        )
        player_source = InMemoryPlayerSource(
            session.player, f"session:inline:player:{session.player_id}"
        )
        result = run_pipeline(shot_source, course_source, player_source, shot_config)
        rec = result.recommendation
        shot_results.append(
            {
                "shot_id": ns.shot_id,
                "hole_number": ns.hole_number,
                "shot_number": ns.shot_number,
                "recommendation": asdict(rec),
                "provenance": dict(rec.provenance),
            }
        )

    # Group ordered hole entries.
    holes: List[dict] = []
    hole_order = sorted(session.courses)
    for hole_number in hole_order:
        hole_shots = [sr for sr in shot_results if sr["hole_number"] == hole_number]
        hole_shots.sort(key=lambda sr: sr["shot_number"])
        prob_lists = [
            sr["recommendation"]["hazard_probabilities"] for sr in hole_shots
        ]
        holes.append(
            {
                "hole_number": hole_number,
                "shot_count": len(hole_shots),
                "shot_ids": [sr["shot_id"] for sr in hole_shots],
                "sum_local_decision_cost": round(
                    sum(sr["recommendation"]["decision_cost"] for sr in hole_shots), 6
                ),
                "hazard_risk_summary": _hazard_risk_summary(prob_lists),
                "recommendations": [
                    {
                        "shot_id": sr["shot_id"],
                        "shot_number": sr["shot_number"],
                        "recommended_club": sr["recommendation"]["recommended_club"],
                        "decision_cost": sr["recommendation"]["decision_cost"],
                    }
                    for sr in hole_shots
                ],
            }
        )

    total_cost = round(
        sum(sr["recommendation"]["decision_cost"] for sr in shot_results), 6
    )
    all_probs = [sr["recommendation"]["hazard_probabilities"] for sr in shot_results]
    highest = sorted(
        shot_results, key=lambda sr: sr["recommendation"]["decision_cost"], reverse=True
    )[:_MAX_HIGHLIGHTED_DECISIONS]

    summary = {
        "shot_count": len(shot_results),
        "hole_count": len(holes),
        "sum_local_decision_cost": total_cost,
        "decision_cost_semantics": (
            "sum_local_decision_cost is a LOCAL, non-additive diagnostic sum of "
            "per-shot recommendation decision costs. It is NOT official Strokes "
            "Gained and NOT an official round stroke total."
        ),
        "highest_cost_decisions": [
            {
                "shot_id": sr["shot_id"],
                "hole_number": sr["hole_number"],
                "shot_number": sr["shot_number"],
                "decision_cost": sr["recommendation"]["decision_cost"],
                "recommended_club": sr["recommendation"]["recommended_club"],
            }
            for sr in highest
        ],
        "hazard_risk_summary": _hazard_risk_summary(all_probs),
    }

    return {
        "schema_version": SESSION_SCHEMA_VERSION,
        "session": {
            "session_id": session.session_id,
            "tournament_id": session.tournament_id,
            "player_id": session.player_id,
            "course_id": session.course_id,
            "round_number": session.round_number,
            "seed": session.seed,
            "shot_count": len(shot_results),
            "hole_count": len(holes),
        },
        "summary": summary,
        "holes": holes,
        "shot_results": shot_results,
        "provenance": {
            "engine_version": "0.1.0",
            "session_schema_version": SESSION_SCHEMA_VERSION,
            "session_id": session.session_id,
            "source": "inline-session-envelope",
            "metadata": dict(session.metadata),
            "data_disclaimer": (
                "Synthetic/mock data only. Not sourced from ShotLink, TrackMan, "
                "TOURCAST, or any official PGA TOUR system. Not for competitive "
                "or broadcast use."
            ),
        },
    }


def _walk_finite(obj, path: str = "root") -> None:
    """Recursively reject non-finite numeric leaves (dict keys/values, sequences)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk_finite(k, f"{path}.key")
            _walk_finite(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _walk_finite(v, f"{path}[{i}]")
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        try:
            finite = math.isfinite(obj)
        except OverflowError:
            # int too large for float conversion: reject under the documented
            # ValueError boundary rather than leaking OverflowError.
            raise ValueError(f"non-finite numeric value at {path}: {obj!r}") from None
        if not finite:
            raise ValueError(f"non-finite numeric value at {path}: {obj!r}")


def serialize_session_report(report) -> str:
    """Recursively validate finiteness, then serialize with allow_nan=False."""
    if hasattr(report, "__dataclass_fields__"):
        report = asdict(report)
    _walk_finite(report)
    return json.dumps(report, allow_nan=False, indent=2)
