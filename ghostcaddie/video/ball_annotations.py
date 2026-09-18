"""Human-authored golf-ball ground-truth annotations, bound to source bytes.

This module is deliberately detector-free. Nothing here proposes, refines or
scores a ball position; it only validates, loads and stores labels that a person
recorded by looking at magnified native-resolution pixels. Any code path that
writes tracker output into one of these files is a bug, because the whole point
of the format is to be the thing a tracker is measured *against*.

A label is one of three states, and the format refuses to blur them:

* ``visible: true``  -- a centre was read off the image; ``x``/``y`` required.
* ``visible: false`` -- the ball is not in the frame or cannot be located;
  ``reason`` required, coordinates forbidden.
* ``ambiguous``      -- an orthogonal flag on either state, meaning the
  annotator would not defend the call. Ambiguous frames are excluded from
  strict precision/recall and from centre-error statistics, and reported
  separately rather than silently dropped.

Every read and every write re-checks the media bytes through
``source_binding.bind_source``: a label file whose ``source_sha256`` no longer
matches its video is not a weaker label, it is a label about a different video.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from .errors import VideoContractError
from .source_binding import bind_source

SCHEMA_VERSION = "golf-ball-annotation.v1"

_TOP_LEVEL = frozenset({
    "schema_version", "clip_id", "video", "source_sha256",
    "width", "height", "fps", "annotator", "notes", "frames",
})
_FRAME_FIELDS = frozenset({"visible", "x", "y", "radius", "ambiguous", "reason"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FRAME_KEY = re.compile(r"^(0|[1-9][0-9]*)$")

PathLike = Union[str, Path]


def _fail(message: str) -> None:
    raise VideoContractError(message)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{name} must be a non-empty string")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        _fail(f"{name} must be a finite number")
    return number


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{name} must be a positive integer")
    return int(value)


def _validate_frame(key: str, entry: Any, *, width: int, height: int) -> dict:
    where = f"frames[{key!r}]"
    if not _FRAME_KEY.fullmatch(key):
        _fail(f"{where} key must be a decimal frame index with no leading zeros")
    if not isinstance(entry, Mapping):
        _fail(f"{where} must be an object")
    extra = set(entry) - _FRAME_FIELDS
    if extra:
        _fail(f"{where} has unknown field {sorted(extra)[0]!r}")
    if "visible" not in entry or not isinstance(entry["visible"], bool):
        _fail(f"{where}.visible must be present and boolean")
    ambiguous = entry.get("ambiguous", False)
    if not isinstance(ambiguous, bool):
        _fail(f"{where}.ambiguous must be boolean")

    out: dict[str, Any] = {"visible": bool(entry["visible"]), "ambiguous": bool(ambiguous)}
    if entry["visible"]:
        if "x" not in entry or "y" not in entry:
            _fail(f"{where} is visible and must carry x and y")
        x = _finite(entry["x"], f"{where}.x")
        y = _finite(entry["y"], f"{where}.y")
        if not (0.0 <= x < float(width)) or not (0.0 <= y < float(height)):
            _fail(f"{where} centre ({x}, {y}) is outside the {width}x{height} frame")
        out["x"], out["y"] = x, y
        if "radius" in entry:
            radius = _finite(entry["radius"], f"{where}.radius")
            if radius <= 0:
                _fail(f"{where}.radius must be positive")
            out["radius"] = radius
    else:
        for forbidden in ("x", "y", "radius"):
            if forbidden in entry:
                _fail(f"{where} is not visible and must not carry {forbidden}")
        if not isinstance(entry.get("reason"), str) or not entry["reason"].strip():
            _fail(f"{where} is not visible and must carry a non-empty reason")
    if "reason" in entry:
        out["reason"] = _text(entry["reason"], f"{where}.reason")
    return out


def validate_annotation(payload: Mapping[str, Any]) -> dict:
    """Return a validated deep copy, or raise ``VideoContractError``."""
    if not isinstance(payload, Mapping):
        _fail("annotation must be an object")
    keys = set(payload)
    if keys != _TOP_LEVEL:
        missing, extra = _TOP_LEVEL - keys, keys - _TOP_LEVEL
        if missing:
            _fail(f"missing required field: {sorted(missing)[0]}")
        _fail(f"fabricated field: {sorted(extra)[0]}")
    if payload["schema_version"] != SCHEMA_VERSION:
        _fail("invalid schema_version")
    clip_id = _text(payload["clip_id"], "clip_id")
    video = _text(payload["video"], "video")
    if Path(video).is_absolute() or video.startswith(("~/", "~\\")):
        _fail("video must be a repository-relative path")
    sha = _text(payload["source_sha256"], "source_sha256")
    if not _SHA256.fullmatch(sha):
        _fail("source_sha256 must be a lowercase SHA-256 hex digest")
    width = _positive_int(payload["width"], "width")
    height = _positive_int(payload["height"], "height")
    fps = _finite(payload["fps"], "fps")
    if fps <= 0:
        _fail("fps must be positive")
    annotator = _text(payload["annotator"], "annotator")
    notes = payload["notes"]
    if not isinstance(notes, str):
        _fail("notes must be a string")
    frames = payload["frames"]
    if not isinstance(frames, Mapping):
        _fail("frames must be an object keyed by frame index")

    validated = {
        key: _validate_frame(key, entry, width=width, height=height)
        for key, entry in frames.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "clip_id": clip_id,
        "video": video,
        "source_sha256": sha,
        "width": width,
        "height": height,
        "fps": fps,
        "annotator": annotator,
        "notes": notes,
        "frames": validated,
    }


def new_annotation(*, clip_id: str, video: str, source_sha256: str, width: int,
                   height: int, fps: float, annotator: str, notes: str = "") -> dict:
    """Create an empty, validated annotation document."""
    return validate_annotation({
        "schema_version": SCHEMA_VERSION, "clip_id": clip_id, "video": video,
        "source_sha256": source_sha256, "width": width, "height": height,
        "fps": fps, "annotator": annotator, "notes": notes, "frames": {},
    })


def set_frame(payload: Mapping[str, Any], frame_index: int, **label: Any) -> dict:
    """Return a copy of ``payload`` with one frame label added or replaced."""
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        _fail("frame_index must be a non-negative integer")
    document = validate_annotation(payload)
    document["frames"][str(frame_index)] = label
    return validate_annotation(document)


def drop_frame(payload: Mapping[str, Any], frame_index: int) -> dict:
    """Return a copy of ``payload`` with one frame label removed."""
    document = validate_annotation(payload)
    document["frames"].pop(str(frame_index), None)
    return validate_annotation(document)


def serialize_annotation(payload: Mapping[str, Any]) -> str:
    """Serialize deterministically, rejecting non-finite JSON."""
    document = validate_annotation(payload)
    document["frames"] = dict(sorted(document["frames"].items(), key=lambda kv: int(kv[0])))
    return json.dumps(document, allow_nan=False, indent=2, sort_keys=False) + "\n"


def bind_annotation(payload: Mapping[str, Any], *, project_root: PathLike) -> dict:
    """Prove the annotated video bytes still match ``source_sha256``.

    Returns the ``bind_source`` record. Raises ``SourceBindingError`` when the
    media has moved, been re-encoded, or been swapped for another clip.
    """
    document = validate_annotation(payload)
    video = Path(project_root).expanduser() / document["video"]
    return bind_source(video, document["source_sha256"])


def load_annotation(path: PathLike, *, project_root: Optional[PathLike] = None) -> dict:
    """Load and validate an annotation file, optionally re-binding source bytes."""
    text = Path(path).expanduser().read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VideoContractError(f"invalid annotation JSON at {path}: {exc}") from exc
    document = validate_annotation(payload)
    if project_root is not None:
        bind_annotation(document, project_root=project_root)
    return document


def save_annotation(path: PathLike, payload: Mapping[str, Any], *,
                    project_root: Optional[PathLike] = None) -> dict:
    """Validate, optionally re-bind source bytes, then write deterministically."""
    document = validate_annotation(payload)
    if project_root is not None:
        bind_annotation(document, project_root=project_root)
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(serialize_annotation(document), encoding="utf-8")
    return document


def annotation_summary(payload: Mapping[str, Any]) -> dict:
    """Counts an evaluator can trust without re-parsing the frame table."""
    document = validate_annotation(payload)
    frames = document["frames"]
    indices = sorted(int(key) for key in frames)
    visible = [key for key, entry in frames.items() if entry["visible"]]
    ambiguous = [key for key, entry in frames.items() if entry["ambiguous"]]
    runs = []
    for index in indices:
        if runs and index == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], index)
        else:
            runs.append((index, index))
    return {
        "clip_id": document["clip_id"],
        "labeled_frames": len(frames),
        "visible_frames": len(visible),
        "not_visible_frames": len(frames) - len(visible),
        "ambiguous_frames": len(ambiguous),
        "runs": [{"start": start, "end": end, "length": end - start + 1} for start, end in runs],
    }


__all__ = [
    "SCHEMA_VERSION", "validate_annotation", "new_annotation", "set_frame",
    "drop_frame", "serialize_annotation", "bind_annotation", "load_annotation",
    "save_annotation", "annotation_summary",
]
