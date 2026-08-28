"""Secure provider-aware session envelope ingestion.

This module is deliberately an ingestion boundary: provider-shaped records are
adapted once to engine-coordinate ShotEvents, then the existing session runner
is reused unchanged.
"""
import copy
import json
from pathlib import Path
from typing import Dict, Tuple

from ..adapters.json_file import _parse_course, _parse_player
from ..geometry import CoordinateMapper
from ..session import NormalizedShot, SessionInput, run_session
from .provider import ProviderAdapterError
from .shotlink import SCHEMA_VERSION as SHOTLINK_SCHEMA, adapt_shotlink
from .trackman import SCHEMA_VERSION as TRACKMAN_SCHEMA, adapt_trackman

SCHEMA_VERSION = "provider-session.v1"


def _dict(value, name):
    if not isinstance(value, dict):
        raise ProviderAdapterError("%s must be a dict" % name)
    return value


def _str(value, name):
    if not isinstance(value, str) or not value:
        raise ProviderAdapterError("%s must be a non-empty string" % name)
    return value


def _int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProviderAdapterError("%s must be a positive integer" % name)
    return value


def _unknown(raw, allowed, name, strict, diagnostics):
    extras = sorted(set(raw) - set(allowed))
    if extras and strict:
        raise ProviderAdapterError("unknown fields at %s: %s" % (name, ", ".join(extras)))
    diagnostics.extend("%s.%s" % (name, field) for field in extras)


def _safe_source(envelope_path, value, role):
    value = _dict(value, role)
    path_value = _str(value.get("path"), role + ".path")
    path = Path(path_value)
    if path.is_absolute():
        raise ProviderAdapterError("%s.path must be relative" % role)
    envelope = Path(envelope_path).expanduser().resolve()
    # A repository root is the first ancestor containing the package. This
    # permits data/ fixtures while keeping source resolution project-local.
    root = envelope.parent
    for candidate in (envelope.parent,) + tuple(envelope.parents):
        if (candidate / "ghostcaddie").is_dir():
            root = candidate
            break
    resolved = (envelope.parent / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ProviderAdapterError("%s.path escapes project root" % role)
    if not resolved.is_file():
        raise ProviderAdapterError("%s.path must name an existing regular file" % role)
    return resolved


def _load_json(path, role):
    try:
        with open(path) as fh:
            value = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ProviderAdapterError("invalid %s JSON" % role) from exc
    return _dict(value, role)


def _course_payload(raw):
    if "holes" in raw:
        return raw
    # A source may be a single existing course record; normalize it to the
    # multi-hole shape expected by the established session parser.
    hole = dict(raw)
    hole_number = hole.pop("hole_number", 1)
    return {"course_id": raw.get("course_id", "course"),
            "holes": [dict(hole, hole_number=hole_number)]}


def _player_payload(raw):
    return raw.get("player_profile", raw)


def _course_context(course_raw, hole_number, wrapper):
    explicit = wrapper.get("course_context")
    if explicit is not None:
        return explicit
    contexts = course_raw.get("provider_context", {})
    if isinstance(contexts, dict):
        return contexts.get(str(hole_number), contexts.get(hole_number, {}))
    return {}


def _validate_nested_record(record, provider, strict, diagnostics, index):
    # Adapters validate the concrete provider fields. This supplements them
    # with strict nested-field policy, without changing adapter behavior.
    allowed = {
        "shotlink": {"provider", "schema_version", "source_record_id", "event_id", "player_id", "tournament_id", "hole_number", "shot_number", "timestamp", "start_position", "target_position", "actual_landing_position", "lie", "club", "distance_to_pin", "wind", "geo_frame"},
        "trackman": {"provider", "schema_version", "source_record_id", "event_id", "player_id", "tournament_id", "hole_number", "shot_number", "timestamp", "lie", "club", "distance_to_pin", "wind", "metrics", "units"},
    }[provider]
    _unknown(record, allowed, "shots[%d].provider_record" % index, strict, diagnostics)
    nested = {"start_position": {"latitude", "longitude"}, "target_position": {"latitude", "longitude"}, "actual_landing_position": {"latitude", "longitude"}, "geo_frame": {"units", "axes", "origin"}, "wind": {"speed_mph", "direction_deg"}, "metrics": {"carry_yd", "side_offset_yd"}}
    for field, fields in nested.items():
        if field in record and isinstance(record[field], dict):
            _unknown(record[field], fields, "shots[%d].provider_record.%s" % (index, field), strict, diagnostics)
    if isinstance(record.get("geo_frame"), dict) and isinstance(record["geo_frame"].get("origin"), dict):
        _unknown(record["geo_frame"]["origin"], {"latitude", "longitude"}, "shots[%d].provider_record.geo_frame.origin" % index, strict, diagnostics)


def parse_provider_session(raw, envelope_path, strict=True):
    """Return a normalized :class:`SessionInput` from provider-session.v1."""
    diagnostics = []
    raw = _dict(raw, "envelope")
    _unknown(raw, {"schema_version", "session", "course_source", "player_source", "shots"}, "envelope", strict, diagnostics)
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ProviderAdapterError("schema_version must be %r" % SCHEMA_VERSION)
    session_raw = _dict(raw.get("session"), "session")
    _unknown(session_raw, {"session_id", "tournament_id", "player_id", "course_id", "round_number", "seed", "provider", "provider_schema_version"}, "session", strict, diagnostics)
    provider = _str(session_raw.get("provider"), "session.provider")
    schemas = {"shotlink": SHOTLINK_SCHEMA, "trackman": TRACKMAN_SCHEMA}
    if provider not in schemas:
        raise ProviderAdapterError("unsupported provider %r" % provider)
    if session_raw.get("provider_schema_version") != schemas[provider]:
        raise ProviderAdapterError("session.provider_schema_version does not match provider")
    session = {k: session_raw.get(k) for k in ("session_id", "tournament_id", "player_id", "course_id")}
    for k, v in session.items(): _str(v, "session." + k)
    session["round_number"] = _int(session_raw.get("round_number"), "session.round_number")
    seed = session_raw.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ProviderAdapterError("session.seed must be an integer")
    session["seed"] = seed
    course_path = _safe_source(envelope_path, raw.get("course_source"), "course_source")
    player_path = _safe_source(envelope_path, raw.get("player_source"), "player_source")
    course_raw, player_raw = _load_json(course_path, "course_source"), _load_json(player_path, "player_source")
    _unknown(raw.get("course_source"), {"path"}, "course_source", strict, diagnostics)
    _unknown(raw.get("player_source"), {"path"}, "player_source", strict, diagnostics)
    course_source = _course_payload(course_raw)
    if course_source.get("course_id") != session["course_id"]:
        raise ProviderAdapterError("course source course_id does not match session")
    player = _parse_player(_player_payload(player_raw))
    if player.player_id != session["player_id"]:
        raise ProviderAdapterError("player source player_id does not match session")
    holes = course_source["holes"]
    courses = {}
    for hole in holes:
        hole_number = hole.get("hole_number")
        if hole_number in courses: raise ProviderAdapterError("duplicate course hole_number")
        courses[hole_number] = _parse_course(hole)
    shots_raw = raw.get("shots")
    if not isinstance(shots_raw, list) or not shots_raw: raise ProviderAdapterError("shots must be a non-empty list")
    normalized, seen_ids, previous = [], set(), None
    mappers = {n: CoordinateMapper(c.coordinate_system) for n, c in courses.items()}
    for i, wrapper in enumerate(shots_raw):
        wrapper = _dict(wrapper, "shots[%d]" % i)
        _unknown(wrapper, {"shot_id", "hole_number", "shot_number", "provider_record", "course_context"}, "shots[%d]" % i, strict, diagnostics)
        shot_id = _str(wrapper.get("shot_id"), "shots[%d].shot_id" % i)
        hole_number, shot_number = _int(wrapper.get("hole_number"), "shots[%d].hole_number" % i), _int(wrapper.get("shot_number"), "shots[%d].shot_number" % i)
        if shot_id in seen_ids: raise ProviderAdapterError("duplicate shot_id %r" % shot_id)
        ordinal = (hole_number, shot_number)
        if previous is not None and ordinal <= previous: raise ProviderAdapterError("shots must be strictly ordered")
        if hole_number not in courses: raise ProviderAdapterError("shot references undeclared hole")
        record = _dict(wrapper.get("provider_record"), "shots[%d].provider_record" % i)
        _validate_nested_record(record, provider, strict, diagnostics, i)
        record = copy.deepcopy(record)
        record.setdefault("player_id", session["player_id"])
        record.setdefault("tournament_id", session["tournament_id"])
        record.setdefault("hole_number", hole_number)
        record.setdefault("shot_number", shot_number)
        context = _course_context(course_raw, hole_number, wrapper)
        if provider == "shotlink": shot = adapt_shotlink(record, mappers[hole_number], context, strict=strict)
        else: shot = adapt_trackman(record, mappers[hole_number], context, strict=strict)
        if shot.player_id != session["player_id"] or shot.tournament_id != session["tournament_id"]: raise ProviderAdapterError("provider record identity mismatch")
        normalized.append(NormalizedShot(shot_id, hole_number, shot_number, shot))
        shot.provenance["session_provider"] = provider
        seen_ids.add(shot_id); previous = ordinal
    metadata = {"provider": provider, "provider_schema_version": schemas[provider], "unknown_fields": diagnostics, "source_roles": ["course", "player"]}
    return SessionInput(SCHEMA_VERSION, session["session_id"], session["tournament_id"], session["player_id"], session["course_id"], session["round_number"], session["seed"], player, courses, normalized, metadata)


def run_provider_session(raw, envelope_path, config, strict=True):
    """Parse a provider envelope and run the unchanged analytics session."""
    report = run_session(parse_provider_session(raw, envelope_path, strict=strict), config)
    # Provider-session reports are replay artifacts: omit the volatile pipeline
    # wall-clock stamp so identical input and seed serialize identically.
    for result in report.get("shot_results", []):
        result.get("recommendation", {}).get("provenance", {}).pop("generated_at", None)
        result.get("provenance", {}).pop("generated_at", None)
    return report


def load_provider_session(path, config, strict=True):
    path = Path(path)
    with open(path) as fh: raw = json.load(fh)
    return run_provider_session(raw, path, config, strict=strict)
