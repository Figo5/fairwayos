"""Import saved automatic research predictions into preview / export.

The demo path can already *review* a saved ``fairwayos-target-predictions.v1``
file (``demo.tracking_review``): it binds each clip to a registered source hash and
summarises gaps. This module is the next, narrower step: it turns those same saved
automatic observations into the ``TargetTrack`` records the renderer and the
preview UI already understand, so a saved research run can be drawn *without*
re-measuring anything and *without* ever passing a derived or operator point off as
an automatic observation.

Two entry points, both stdlib-only:

``load_saved_tracks(path, records, require_all_registered=True)``
    read and validate a saved predictions file against the registered media
    records. Validation is delegated to the benchmark's own prediction contract
    (``target_benchmark.validate_predictions``) driven with a synthetic clip built
    from the *registered* source geometry, so no reference labels are read and no
    accuracy is implied. The strict UI sets ``require_all_registered=True`` and
    refuses a file that names a hash it does not have; ``False`` lets a collection
    file carry other clips but only ever imports the registered matches, and
    duplicate ids/hashes are refused everywhere either way.

``build_saved_track(name, block, interval_frames, expected_seed=None, cut_frames=())``
    build one ``TargetTrack`` from a validated target block. The saved operator
    seed is the only operator anchor; automatic observations keep their own status,
    method and score (a SAM2 score stays ``None``); a saved interpolation is *not*
    rendered (its frame is emitted ``missing``); every frame the saved run did not
    represent is emitted ``missing``; and because this API has no distinct reseed
    step, no automatic observation is carried across a structural source cut.

Nothing here fabricates a position, a confidence or an interpolation. A frame the
saved run did not observe is absent, not invented.
"""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .errors import DemoConfigError
from . import target_benchmark as bench
from .tracking import (NO_SEED_REASON, ORIGIN_AUTOMATIC, ORIGIN_DERIVED,
                       ORIGIN_OPERATOR, STATUS_INTERPOLATED, STATUS_LOST,
                       STATUS_MANUAL, STATUS_MISSING, STATUS_OBSERVED,
                       STATUS_OBSERVED_LK, TargetTrack, TrackPoint,
                       _accounting, _coverage, _segments)

#: A saved predictions file is a research artifact; 16 MiB is the same ceiling the
#: review path uses, so the two cannot disagree about what is importable.
MAX_SAVED_TRACK_BYTES = 16 * 1024 * 1024

#: The targets the tracker and the renderer know. A saved block naming anything
#: else is refused rather than ignored.
TARGET_NAMES = ("ball", "clubhead")

_SAVED_AUTOMATIC_LIMITATION = (
    "Positions are automatic research observations imported from a saved "
    "predictions file: no measurement is repeated at import, missing and lost "
    "frames stay absent, and interpolated (derived) points are never drawn.")

_SAVED_INTERPOLATION_EXCLUDED = (
    "%d saved interpolated (derived) frame(s) were excluded from rendering at "
    "import: a derived point is not a measurement and is emitted missing.")

_SAVED_CUT_SUPPRESSED = (
    "%d automatic observation(s) at or after the structural cut frame %s were "
    "suppressed: this importer carries no distinct reseed step, so an automatic "
    "track is never continued across a source splice.")

_SAVED_REACQUISITION_PRESERVED = (
    "Automatic re-acquisition already recorded in the saved run is preserved: a "
    "saved re-acquired observation after a gap is kept as an automatic point under "
    "its own method, and the trail does not bridge the gap.")


# --------------------------------------------------------------------------
# small accessors (records are objects on the UI path, dicts in tests)


def _field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _iter_records(records: Any) -> list:
    if isinstance(records, Mapping):
        return list(records.values())
    return list(records)


def _seed_xyz(value: Any) -> Optional[dict]:
    """Normalise a seed to ``{'frame','x','y'}`` from a mapping or an object."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        frame, x, y = value.get("frame"), value.get("x"), value.get("y")
    else:
        frame = getattr(value, "frame", None)
        x = getattr(value, "x", None)
        y = getattr(value, "y", None)
    return {"frame": frame, "x": x, "y": y}


def _finite_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int)


def _finite_number(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _validate_saved_seed(seed: Any, width: int, height: int, frame_count: int) -> None:
    """Bound the saved seed to the registered source geometry.

    The benchmark's own seed check pins a prediction's seed to the *reference*
    seed; here the synthetic reference seed is the saved seed itself, so that
    check cannot catch a seed outside the frame. The same native bounds are
    enforced directly instead.
    """
    if seed is None:
        return
    if not isinstance(seed, Mapping):
        raise ValueError("seed must be an object or null")
    frame = seed.get("frame")
    if not _finite_int(frame) or not 0 <= frame < frame_count:
        raise ValueError("Seed outside source frames")
    for key, extent in (("x", width), ("y", height)):
        value = seed.get(key)
        if not _finite_number(value) or not 0 <= float(value) <= float(extent):
            raise ValueError("Seed outside native source pixels")


# --------------------------------------------------------------------------
# reading a saved predictions file


def load_saved_tracks(path: str | Path, records: Any,
                      require_all_registered: bool = True) -> dict:
    """Validate a saved predictions file against registered source records.

    Returns a JSON-native dict::

        {"prediction_sha256": <hash of the file bytes>,
         "variant": <the run's declared variant>,
         "clips_by_sha": {<source sha256>: {"id", "sha256", "width", "height",
                                            "frame_count",
                                            "targets": {<target>: {"seed",
                                                                   "points",
                                                                   "options"}}}}}

    ``require_all_registered=True`` refuses a file that binds to a hash this
    collection does not have (the strict UI path). ``False`` permits other
    collection clips but only returns the registered matches; either way a
    duplicate clip id or duplicate source hash anywhere in the file is refused.
    Every refusal is a ``DemoConfigError``.
    """
    source = Path(path)
    try:
        if source.stat().st_size > MAX_SAVED_TRACK_BYTES:
            raise DemoConfigError("Saved track results exceed 16 MiB")
        with source.open('rb') as handle:
            raw_bytes = handle.read(MAX_SAVED_TRACK_BYTES + 1)
        if len(raw_bytes) > MAX_SAVED_TRACK_BYTES:
            raise DemoConfigError("Saved track results exceed 16 MiB")
        text = raw_bytes.decode('utf-8')
    except (OSError, UnicodeError) as exc:
        raise DemoConfigError(
            f"Saved track results could not be read: {source} ({exc})") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DemoConfigError(
            f"Saved track results are not valid JSON: {source} ({exc})") from exc
    try:
        return _load_payload(payload, hashlib.sha256(raw_bytes).hexdigest(),
                             _iter_records(records), bool(require_all_registered))
    except DemoConfigError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
        raise DemoConfigError(f"Invalid saved track results: {exc}") from exc


def _load_payload(payload: Any, digest: str, records: list,
                  require_all_registered: bool) -> dict:
    if not isinstance(payload, Mapping):
        raise ValueError("saved track results must be a JSON object")
    raw_clips = payload.get("clips")
    if not isinstance(raw_clips, list):
        raise ValueError("saved track results must carry a list of clips")

    known = {}
    for record in records:
        sha = _field(record, "sha256")
        if not isinstance(sha, str) or not sha.strip():
            raise ValueError("a registered record has no sha256")
        known[sha.strip().lower()] = record

    seen_ids: set = set()
    seen_sha: set = set()
    matched: list = []
    for index, block in enumerate(raw_clips):
        if not isinstance(block, Mapping):
            raise ValueError(f"clip[{index}] must be an object")
        clip_id, sha = block.get("id"), block.get("sha256")
        if not isinstance(clip_id, str) or not clip_id.strip():
            raise ValueError(f"clip[{index}].id must be a non-empty string")
        if not isinstance(sha, str) or not sha.strip():
            raise ValueError(f"clip[{index}].sha256 must be a non-empty string")
        sha = sha.strip().lower()
        if clip_id in seen_ids:
            raise ValueError(f"saved results repeat clip id {clip_id!r}")
        if sha in seen_sha:
            raise ValueError(f"saved results repeat source hash {sha}")
        seen_ids.add(clip_id)
        seen_sha.add(sha)
        if sha in known:
            matched.append((block, sha, known[sha]))
        elif require_all_registered:
            raise ValueError(
                f"saved results bind to an unregistered source hash {sha[:12]}..; "
                f"the strict reader rejects unknown hashes")

    clips: list = []
    for block, sha, record in matched:
        width = int(_field(record, "width"))
        height = int(_field(record, "height"))
        frame_count = int(_field(record, "frame_count"))
        if width <= 0 or height <= 0 or frame_count <= 0:
            raise ValueError("registered source geometry must be positive")
        targets = block.get("targets")
        if not isinstance(targets, Mapping) or not targets:
            raise ValueError("saved clip carries no targets")
        unknown = sorted(set(targets) - set(TARGET_NAMES))
        if unknown:
            raise ValueError(f"unknown tracking target {unknown[0]!r}")
        targets = {name: dict(targets[name]) for name in targets}
        for target_name in sorted(targets):
            _validate_saved_seed(targets[target_name].get("seed"),
                                 width, height, frame_count)
        clips.append({
            "id": block["id"], "sha256": sha, "width": width, "height": height,
            "frame_count": frame_count,
            "window": {"start_frame": 0, "end_frame": frame_count - 1},
            "cut_frames": [], "targets": targets})

    dataset = {"clips": clips, "clip_index": {c["id"]: c for c in clips}}
    run_payload = {"schema_version": payload.get("schema_version"),
                   "variant": payload.get("variant"),
                   "clips": [block for block, _, _ in matched]}
    validated = bench.validate_predictions(run_payload, dataset)
    validated_by_id = {c["id"]: c for c in validated["clips"]}

    clips_by_sha: dict = {}
    for block, sha, record in matched:
        vclip = validated_by_id[block["id"]]
        targets_out: dict = {}
        for target_name in sorted(block["targets"]):
            vtarget = vclip["targets"][target_name]
            points = vtarget["points"]
            seed = block["targets"][target_name].get("seed")
            if any(p["origin"] == ORIGIN_AUTOMATIC
                   and (seed is None or p["source_frame"] <= seed["frame"])
                   for p in points):
                raise ValueError("Automatic observations require an earlier seed")
            for point in points:
                if point['status'] == STATUS_MANUAL:
                    if not _seed_agrees({'frame': point['source_frame'], 'x': point['x'], 'y': point['y']}, seed):
                        raise ValueError('Saved seed anchor coordinates disagree with its declared seed')
                if point['origin'] == ORIGIN_AUTOMATIC and point['method'].startswith('sam2_') and point['score'] is not None:
                    raise ValueError('SAM2 region observations carry no calibrated point confidence')
            targets_out[target_name] = {"seed": seed, "points": points,
                                        "options": vtarget["options"]}
        clips_by_sha[sha] = {
            "id": block["id"], "sha256": sha,
            "width": int(_field(record, "width")),
            "height": int(_field(record, "height")),
            "frame_count": int(_field(record, "frame_count")),
            "targets": targets_out}
    return {"prediction_sha256": digest, "variant": validated["variant"],
            "clips_by_sha": clips_by_sha}


# --------------------------------------------------------------------------
# building a renderable track from a validated block


def build_saved_track(name: str, block: Mapping[str, Any],
                      interval_frames: Sequence[int],
                      expected_seed: Any = None,
                      cut_frames: Sequence[int] = ()) -> TargetTrack:
    """Build a ``TargetTrack`` from one validated saved target block.

    ``block`` is ``clips_by_sha[sha]["targets"][name]``: a ``seed`` (or null) and
    the validated ``points``. ``expected_seed`` is the operator's configured seed,
    when the caller has one: if it disagrees with the saved operator seed the call
    is refused, and when it is absent the saved seed supplies it. ``cut_frames``
    are the structural source cuts; because this API has no distinct reseed step,
    every automatic observation at or after the first cut *following the seed* is
    suppressed.
    """
    try:
        return _build_saved_track(name, block, list(interval_frames),
                                 expected_seed, tuple(cut_frames))
    except DemoConfigError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
        raise DemoConfigError(f"Invalid saved track: {exc}") from exc


def _build_saved_track(name: str, block: Mapping[str, Any], interval_frames: list,
                       expected_seed: Any, cut_frames: tuple) -> TargetTrack:
    if name not in TARGET_NAMES:
        raise ValueError(f"unknown tracking target {name!r}")
    if not isinstance(block, Mapping):
        raise ValueError("saved target block must be an object")

    saved_xyz = _seed_xyz(block.get("seed"))
    conf_xyz = _seed_xyz(expected_seed) if expected_seed is not None else None
    if conf_xyz is not None:
        if saved_xyz is None:
            raise ValueError("a configured seed was supplied but the saved track "
                             "declares none")
        if not _seed_agrees(conf_xyz, saved_xyz):
            raise ValueError("the configured seed disagrees with the saved operator "
                             "seed")
    seed_frame = saved_xyz["frame"] if saved_xyz is not None else None

    points_raw = block.get("points") or []
    if not isinstance(points_raw, list):
        raise ValueError("saved target points must be a list")
    by_frame: dict = {}
    for point in points_raw:
        if not isinstance(point, Mapping):
            raise ValueError("each saved point must be an object")
        frame = point.get("source_frame")
        if not _finite_int(frame):
            raise ValueError("saved point source_frame must be an integer")
        if frame in by_frame:
            raise ValueError(f"a saved point repeats frame {frame}")
        by_frame[frame] = point

    cut_limit: Optional[int] = None
    if seed_frame is not None:
        after = [int(c) for c in cut_frames if int(c) > seed_frame]
        if after:
            cut_limit = min(after)

    interval = [int(f) for f in interval_frames]
    points: list = []
    suppressed_cut = 0
    excluded_interpolation = 0
    reacquired = False
    seen_empty = False

    for frame in interval:
        if cut_limit is not None and frame >= cut_limit:
            saved = by_frame.get(frame)
            if saved is not None and saved.get("status") in bench.AUTOMATIC_STATUSES:
                suppressed_cut += 1
                points.append(TrackPoint(
                    frame, STATUS_MISSING, ORIGIN_DERIVED, method="cut_suppressed",
                    note=(f"automatic tracking does not cross the structural cut at "
                          f"frame {cut_limit} without a distinct reseed, so nothing "
                          f"is drawn here")))
                seen_empty = True
                continue
        saved = by_frame.get(frame)
        if frame == seed_frame:
            x, y = _seed_pixel(saved, saved_xyz)
            point = TrackPoint(frame, STATUS_MANUAL, ORIGIN_OPERATOR,
                               method="operator_placement", x=x, y=y, score=None,
                               note="saved operator seed")
            points.append(point)
            seen_empty = False
            continue
        if saved is None:
            points.append(TrackPoint(
                frame, STATUS_MISSING, ORIGIN_DERIVED,
                note="no saved observation for this frame"))
            seen_empty = True
            continue
        status = saved.get("status")
        if status in bench.AUTOMATIC_STATUSES:
            x, y = saved.get("x"), saved.get("y")
            if not (_finite_number(x) and _finite_number(y)):
                raise ValueError(f"automatic point at frame {frame} needs "
                                 f"coordinates")
            score = saved.get("score")
            if score is not None and not _finite_number(score):
                raise ValueError(f"automatic point at frame {frame} has a "
                                 f"non-finite score")
            points.append(TrackPoint(
                frame, status, ORIGIN_AUTOMATIC, method=str(saved.get("method") or ""),
                x=float(x), y=float(y),
                score=None if score is None else float(score), direction="forward"))
            if seen_empty:
                reacquired = True
            seen_empty = False
        elif status == STATUS_MANUAL:
            raise ValueError(f"a saved operator anchor at frame {frame} is not the "
                             f"target seed: this importer draws no manual anchors "
                             f"except the saved seed")
        elif status == STATUS_INTERPOLATED:
            excluded_interpolation += 1
            points.append(TrackPoint(
                frame, STATUS_MISSING, ORIGIN_DERIVED,
                note="saved interpolation excluded at import: a derived point is "
                     "not a measurement"))
            seen_empty = True
        elif status in (STATUS_MISSING, STATUS_LOST):
            points.append(TrackPoint(
                frame, status, ORIGIN_DERIVED,
                note=f"saved track carries no position for this frame ({status})"))
            seen_empty = True
        else:
            raise ValueError(f"unknown saved point status {status!r} at frame {frame}")

    anchors = {}
    if seed_frame is not None:
        anchors[seed_frame] = {"frame": seed_frame, "x": float(saved_xyz["x"]),
                               "y": float(saved_xyz["y"]), "source": "seed", "note": ""}
    segments = _segments(points, mode="template", anchors=anchors)
    coverage = _coverage(points)
    accounting = _accounting(points, segments)
    available = any(p.drawn for p in points)

    track = TargetTrack(name=name, mode="saved_automatic_research", direction="forward",
                        available=available, points=points, segments=segments,
                        coverage=coverage, accounting=accounting,
                        seed=(dict(anchors[seed_frame]) if seed_frame is not None
                              else None),
                        corrections=[])
    track.reason = "" if available else NO_SEED_REASON
    limitations = [_SAVED_AUTOMATIC_LIMITATION]
    if seed_frame is None:
        limitations.append(
            "The saved track declares no operator seed, so it has no automatic "
            "track and every frame is absent.")
    if excluded_interpolation:
        limitations.append(_SAVED_INTERPOLATION_EXCLUDED % excluded_interpolation)
    if suppressed_cut:
        limitations.append(_SAVED_CUT_SUPPRESSED % (suppressed_cut, cut_limit))
    if reacquired:
        limitations.append(_SAVED_REACQUISITION_PRESERVED)
    track.limitations = limitations
    return track


def _seed_pixel(saved: Any, saved_xyz: Mapping[str, Any]) -> tuple:
    """The seed pixel: the saved anchor's own coordinates when it carries them."""
    if isinstance(saved, Mapping):
        x, y = saved.get("x"), saved.get("y")
        if _finite_number(x) and _finite_number(y):
            return float(x), float(y)
    return float(saved_xyz["x"]), float(saved_xyz["y"])


def _seed_agrees(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if not _finite_int(a["frame"]) or not _finite_int(b["frame"]):
        return False
    if int(a["frame"]) != int(b["frame"]):
        return False
    for key in ("x", "y"):
        if not (_finite_number(a[key]) and _finite_number(b[key])):
            return False
        if abs(float(a[key]) - float(b[key])) > 1e-6:
            return False
    return True


__all__ = ["MAX_SAVED_TRACK_BYTES", "TARGET_NAMES", "load_saved_tracks",
           "build_saved_track"]
