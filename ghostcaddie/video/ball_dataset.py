"""Ball dataset v2: sources manifest, annotation contract, frozen split, QC.

This module is the *contract* for the second dataset. It is deliberately
detector-free: nothing here proposes, refines or scores a ball position. It
validates, loads, summarises and flags labels that a reviewer recorded, and its
whole reason to exist is to be the thing a tracker is later measured against.

Four things it refuses to blur:

* **States.** ``VISIBLE`` requires coordinates. ``NOT_VISIBLE``, ``OCCLUDED`` and
  ``AMBIGUOUS`` forbid them, and each requires a reason or an explanation.
  ``AMBIGUOUS`` is a real state, not a soft ``VISIBLE``: it is excluded from
  strict precision/recall and reported separately rather than quietly promoted.

* **Label origin.** ``HUMAN_CONFIRMED`` is the only origin that may be described
  as human ground truth. ``AGENT_REVIEWED`` is machine-reviewed and weaker.
  ``AGENT_PROPOSED`` is a guess and lives in a *separate namespace*: a proposal
  can never appear as a frame label, and a rejected proposal is kept because it
  is useful negative evidence.

* **Source binding.** Every read and write re-checks the media bytes. A label
  file whose ``source_sha256`` no longer matches its video is not a weaker
  label, it is a label about a different video.

* **Split integrity.** No clip, no source hash and no shoot group may appear in
  two splits; no clip from the consumed v1 corpus or its shoot families may sit
  in the v2 final test. Final-test data cannot be loaded without an explicit
  opt-in, so leakage has to be deliberate rather than accidental.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from .errors import VideoContractError
from .source_binding import SourceBindingError, bind_source

SCHEMA_SOURCES = "golf-ball-dataset-sources.v1"
SCHEMA_ANNOTATION = "golf-ball-annotation.v2"
SCHEMA_SPLIT = "golf-ball-dataset-split.v1"
SCHEMA_ACQUISITION_PLAN = "golf-ball-dataset-acquisition-plan.v1"

STATE_VISIBLE = "VISIBLE"
STATE_NOT_VISIBLE = "NOT_VISIBLE"
STATE_OCCLUDED = "OCCLUDED"
STATE_AMBIGUOUS = "AMBIGUOUS"
STATES = (STATE_VISIBLE, STATE_NOT_VISIBLE, STATE_OCCLUDED, STATE_AMBIGUOUS)

#: States that carry no defended position.
POSITIONLESS_STATES = frozenset({STATE_NOT_VISIBLE, STATE_OCCLUDED, STATE_AMBIGUOUS})

ORIGIN_HUMAN = "HUMAN_CONFIRMED"
ORIGIN_AGENT_REVIEWED = "AGENT_REVIEWED"
ORIGIN_AGENT_PROPOSED = "AGENT_PROPOSED"
LABEL_ORIGINS = (ORIGIN_HUMAN, ORIGIN_AGENT_REVIEWED)
PROPOSAL_ORIGIN = ORIGIN_AGENT_PROPOSED

HARD_NEGATIVE_TAGS = (
    "body", "clothing", "shoe", "club", "tee", "flagstick", "glare",
    "background_ball", "unknown_bright_object",
)

SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "validation"
SPLIT_FINAL_TEST = "final_test"
SPLITS = (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_FINAL_TEST)
#: Splits a development tool may read by default. Final test is opt-in.
DEVELOPMENT_SPLITS = (SPLIT_TRAIN, SPLIT_VALIDATION)

SOURCE_REQUIRED_FIELDS = (
    "clip_id", "provider", "source_url", "source_identifier", "acquisition_date",
    "license", "license_url", "original_filename", "sha256", "source_group",
    "rights_class",
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
FRAME_KEY = re.compile(r"^(0|[1-9][0-9]*)$")

PathLike = Union[str, Path]


class DatasetError(VideoContractError):
    """Raised when the dataset contract is violated."""


class LeakageError(DatasetError):
    """Raised when something tries to reach final-test data without opting in."""


def _fail(message: str) -> None:
    raise DatasetError(message)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{name} must be a non-empty string")
    return value


def _optional_text(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        _fail(f"{name} must be a string when present")
    return value or None


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


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SHA256.match(value):
        _fail(f"{name} must be a lowercase hex SHA-256")
    return value


def _string_list(value: Any, name: str) -> list[str]:
    """Free-form vocabulary (diversity axes). Unknown values are not rejected."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        _fail(f"{name} must be a list of strings")
    out = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            _fail(f"{name} must contain non-empty strings")
        if item not in out:
            out.append(item)
    return out


def _tags(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        _fail(f"{name} must be a list of tags")
    out = []
    for item in value:
        if item not in HARD_NEGATIVE_TAGS:
            _fail(f"{name} contains unknown tag {item!r}; known tags: "
                  f"{', '.join(HARD_NEGATIVE_TAGS)}")
        if item not in out:
            out.append(item)
    return out


def _bind_media(video_path: PathLike, expected_sha256: str) -> None:
    """Re-check the media bytes, reporting one exception type for the dataset API."""
    try:
        bind_source(Path(video_path), expected_sha256)
    except SourceBindingError as exc:
        raise DatasetError(str(exc)) from exc


def _read_json_no_duplicate_keys(path: PathLike) -> Any:
    """Parse JSON, refusing duplicated keys instead of silently keeping the last."""
    text = Path(path).read_text(encoding="utf-8")

    def hook(pairs):
        seen = {}
        for key, value in pairs:
            if key in seen:
                _fail(f"duplicate key {key!r} in {Path(path).name}")
            seen[key] = value
        return seen

    try:
        return json.loads(text, object_pairs_hook=hook)
    except json.JSONDecodeError as exc:
        _fail(f"{Path(path).name} is not valid JSON: {exc}")


# ---------------------------------------------------------------------------
# sources manifest
# ---------------------------------------------------------------------------

def validate_sources(payload: Mapping[str, Any]) -> dict:
    if not isinstance(payload, Mapping):
        _fail("sources manifest must be an object")
    version = payload.get("schema_version")
    if version != SCHEMA_SOURCES:
        _fail(f"sources schema_version must be {SCHEMA_SOURCES!r}, got {version!r}")
    clips = payload.get("clips")
    if not isinstance(clips, list) or not clips:
        _fail("sources manifest must carry a non-empty clips list")

    seen_ids, seen_hashes, normalised = set(), {}, []
    for index, clip in enumerate(clips):
        where = f"clips[{index}]"
        if not isinstance(clip, Mapping):
            _fail(f"{where} must be an object")
        record = dict(clip)
        for field in SOURCE_REQUIRED_FIELDS:
            if field not in record:
                _fail(f"{where} is missing required field {field!r}")
        clip_id = _text(record["clip_id"], f"{where}.clip_id")
        if clip_id in seen_ids:
            _fail(f"duplicate clip_id {clip_id!r} in sources manifest")
        seen_ids.add(clip_id)
        _text(record["provider"], f"{where}.provider")
        _text(record["source_url"], f"{where}.source_url")
        _text(record["source_identifier"], f"{where}.source_identifier")
        _text(record["acquisition_date"], f"{where}.acquisition_date")
        _text(record["original_filename"], f"{where}.original_filename")
        _text(record["source_group"], f"{where}.source_group")
        _text(record["rights_class"], f"{where}.rights_class")
        _text(record["license"], f"{where}.license")
        license_url = record.get("license_url")
        _text(license_url, f"{where}.license_url")
        sha = _sha(record["sha256"], f"{where}.sha256")
        if sha in seen_hashes:
            _fail(f"duplicate source sha256 for {clip_id!r} and "
                  f"{seen_hashes[sha]!r}: one file cannot be two clips")
        seen_hashes[sha] = clip_id
        if record.get("license_text_reference") is not None:
            _optional_text(record["license_text_reference"],
                           f"{where}.license_text_reference")
        media_path = _optional_text(record.get("media_path"), f"{where}.media_path")
        if media_path is None:
            _fail(f"{where}.media_path is required (media is local, never committed)")
        if not media_path.startswith("out/"):
            _fail(f"{where}.media_path must live under the gitignored out/ tree; "
                  f"got {media_path!r}. Media is never committed.")
        record["intended_tags"] = _string_list(record.get("intended_tags"),
                                               f"{where}.intended_tags")
        record["tags"] = _tags(record.get("tags"), f"{where}.tags")
        for numeric in ("bytes", "width", "height"):
            if record.get(numeric) is not None:
                _positive_int(record[numeric], f"{where}.{numeric}")
        for optional_number in ("fps", "duration_s"):
            if record.get(optional_number) is not None:
                _finite(record[optional_number], f"{where}.{optional_number}")
        normalised.append(record)

    return {
        "schema_version": SCHEMA_SOURCES,
        "dataset_id": _text(payload.get("dataset_id", "fairwayos-ball-v2"),
                            "dataset_id"),
        "clips": normalised,
    }


def load_sources(path: PathLike) -> dict:
    return validate_sources(_read_json_no_duplicate_keys(path))


def sources_by_id(manifest: Mapping[str, Any]) -> dict:
    return {clip["clip_id"]: clip for clip in manifest["clips"]}


# ---------------------------------------------------------------------------
# annotations
# ---------------------------------------------------------------------------

def _validate_frame(key: str, entry: Any, *, width: int, height: int,
                    frames: Optional[int]) -> dict:
    where = f"frames[{key!r}]"
    if not FRAME_KEY.fullmatch(str(key)):
        _fail(f"{where} key must be a decimal frame index with no leading zeros")
    index = int(key)
    if frames is not None and index >= frames:
        _fail(f"{where} is beyond the clip length ({frames} frames)")
    if not isinstance(entry, Mapping):
        _fail(f"{where} must be an object")
    allowed = {"state", "x", "y", "radius", "bbox", "reason", "explanation",
               "candidate_region", "origin", "reviewer", "confidence", "tags",
               "note", "review_flag"}
    extra = set(entry) - allowed
    if extra:
        _fail(f"{where} has unexpected fields: {sorted(extra)}")
    state = entry.get("state")
    if state is None:
        _fail(f"{where} has no state; every reviewed frame must carry exactly one "
              f"of {', '.join(STATES)}")
    if state not in STATES:
        _fail(f"{where}.state must be one of {', '.join(STATES)}, got {state!r}")
    origin = entry.get("origin")
    if origin not in LABEL_ORIGINS:
        _fail(f"{where}.origin must be one of {', '.join(LABEL_ORIGINS)}; "
              f"proposals belong in the proposals namespace, not in frames")
    if origin == ORIGIN_HUMAN and not _optional_text(entry.get("reviewer"),
                                                     f"{where}.reviewer"):
        _fail(f"{where} claims human ground truth without naming the reviewer")
    out = dict(entry)
    out["tags"] = _tags(entry.get("tags"), f"{where}.tags")

    if state == STATE_VISIBLE:
        for axis, limit in (("x", width), ("y", height)):
            if entry.get(axis) is None:
                _fail(f"{where} is VISIBLE and must carry {axis}")
            value = _finite(entry[axis], f"{where}.{axis}")
            if not 0 <= value <= limit - 1:
                _fail(f"{where}.{axis}={value} is outside the frame (0..{limit - 1})")
            out[axis] = value
        if entry.get("radius") is not None:
            radius = _finite(entry["radius"], f"{where}.radius")
            if radius <= 0 or radius > min(width, height):
                _fail(f"{where}.radius={radius} is not a plausible ball radius")
            out["radius"] = radius
        if entry.get("bbox") is not None:
            box = entry["bbox"]
            if (not isinstance(box, (list, tuple)) or len(box) != 4
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                               for v in box)):
                _fail(f"{where}.bbox must be four numbers")
            x1, y1, x2, y2 = (float(v) for v in box)
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                _fail(f"{where}.bbox is outside the frame or not a box")
            out["bbox"] = [x1, y1, x2, y2]
        if _optional_text(entry.get("reason"), f"{where}.reason") is not None:
            _fail(f"{where} is VISIBLE and must not carry a reason string")
    else:
        for forbidden in ("x", "y", "radius", "bbox", "candidate_region"):
            if entry.get(forbidden) is not None:
                _fail(f"{where} is {state} and must not carry {forbidden}; "
                      "coordinates on a non-visible frame are a fabricated position")
        if state in (STATE_NOT_VISIBLE, STATE_OCCLUDED):
            reason = _optional_text(entry.get("reason"), f"{where}.reason")
            if reason is None and state == STATE_NOT_VISIBLE:
                _fail(f"{where} is NOT_VISIBLE and must give a reason")
            if state == STATE_OCCLUDED and reason is None and not _optional_text(
                    entry.get("explanation"), f"{where}.explanation"):
                _fail(f"{where} is OCCLUDED and must give a reason or explanation")
        if entry.get("confidence") not in (None, "high", "medium", "low"):
            _fail(f"{where}.confidence must be high, medium or low when present")
    return out


def validate_annotation(payload: Mapping[str, Any], *, sources: Optional[Mapping] = None,
                        project_root: Optional[PathLike] = None) -> dict:
    if not isinstance(payload, Mapping):
        _fail("annotation must be an object")
    version = payload.get("schema_version")
    if version != SCHEMA_ANNOTATION:
        _fail(f"annotation schema_version must be {SCHEMA_ANNOTATION!r}, got {version!r}")
    clip_id = _text(payload.get("clip_id"), "clip_id")
    sha = _sha(payload.get("source_sha256"), "source_sha256")
    width = _positive_int(payload.get("width"), "width")
    height = _positive_int(payload.get("height"), "height")
    video = _text(payload.get("video"), "video")
    frames_total = payload.get("frames_total")
    if frames_total is not None:
        frames_total = _positive_int(frames_total, "frames_total")

    manifest_clip = None
    if sources is not None:
        manifest_clip = sources_by_id(sources).get(clip_id)
        if manifest_clip is None:
            _fail(f"annotation for {clip_id!r} is not in the sources manifest")
        if manifest_clip["sha256"] != sha:
            _fail(f"annotation for {clip_id!r} is bound to a different source hash "
                  f"than the manifest records")
    if project_root is not None:
        # The media bytes are the authority: a mismatch means these labels are
        # about a different video, not a weaker version of the same one.
        _bind_media(Path(project_root) / video, sha)

    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, Mapping):
        _fail("frames must be an object")
    frames = {str(key): _validate_frame(str(key), value, width=width, height=height,
                                        frames=frames_total)
              for key, value in raw_frames.items()}

    proposals = payload.get("proposals") or {}
    if not isinstance(proposals, Mapping):
        _fail("proposals must be an object")
    clean_proposals = {}
    for key, entry in proposals.items():
        where = f"proposals[{key!r}]"
        if not FRAME_KEY.fullmatch(str(key)):
            _fail(f"{where} key must be a decimal frame index")
        if not isinstance(entry, Mapping):
            _fail(f"{where} must be an object")
        if entry.get("origin") != PROPOSAL_ORIGIN:
            _fail(f"{where}.origin must be {PROPOSAL_ORIGIN!r}")
        status = entry.get("status")
        if status not in ("pending", "accepted", "rejected"):
            _fail(f"{where}.status must be pending, accepted or rejected")
        if status == "pending" and key in frames:
            _fail(f"{where} is pending yet frame {key} already carries a label; "
                  "a proposal is not a label")
        x, y = entry.get("x"), entry.get("y")
        if x is None or y is None:
            if status != "rejected":
                _fail(f"{where} must carry a proposed x and y unless rejected")
        else:
            for axis, value, limit in (("x", x, width), ("y", y, height)):
                number = _finite(value, f"{where}.{axis}")
                if not 0 <= number <= limit - 1:
                    _fail(f"{where}.{axis} is outside the frame")
        clean_proposals[str(key)] = dict(entry)

    reviewer = payload.get("reviewer")
    if reviewer is not None:
        if not isinstance(reviewer, Mapping):
            _fail("reviewer must be an object")
        kind = reviewer.get("kind")
        if kind not in ("human", "agent", "mixed"):
            _fail("reviewer.kind must be human, agent or mixed")
        _text(reviewer.get("id"), "reviewer.id")

    return {
        "schema_version": SCHEMA_ANNOTATION,
        "clip_id": clip_id,
        "video": video,
        "source_sha256": sha,
        "width": width,
        "height": height,
        "fps": _finite(payload["fps"], "fps") if payload.get("fps") is not None else None,
        "frames_total": frames_total,
        "annotator": payload.get("annotator") or "unattributed",
        "reviewer": dict(reviewer) if reviewer else None,
        "source_group": payload.get("source_group"),
        "tags": _tags(payload.get("tags"), "tags"),
        "notes": payload.get("notes") or "",
        "frames": frames,
        "proposals": clean_proposals,
    }


def load_annotation(path: PathLike, *, sources: Optional[Mapping] = None,
                    project_root: Optional[PathLike] = None,
                    bind: bool = True) -> dict:
    payload = _read_json_no_duplicate_keys(path)
    return validate_annotation(payload, sources=sources,
                               project_root=project_root if bind else None)


def annotation_summary(annotation: Mapping[str, Any]) -> dict:
    counts = {state: 0 for state in STATES}
    per_origin: dict[str, int] = {}
    tags: dict[str, int] = {}
    for entry in annotation["frames"].values():
        counts[entry["state"]] += 1
        per_origin[entry["origin"]] = per_origin.get(entry["origin"], 0) + 1
        for tag in entry.get("tags", []):
            tags[tag] = tags.get(tag, 0) + 1
    runs = frame_runs(annotation)
    return {
        "clip_id": annotation["clip_id"],
        "labelled_frames": len(annotation["frames"]),
        "states": counts,
        "origins": per_origin,
        "tags": tags,
        "proposals": len(annotation.get("proposals") or {}),
        "visible_runs": runs["visible_runs"],
        "absent_runs": runs["absent_runs"],
        "longest_visible_run": runs["longest_visible_run"],
        "longest_absent_run": runs["longest_absent_run"],
        "longest_run": runs["longest_run"],
        "isolated_visible": runs["isolated_visible"],
        "radii": sorted(round(entry["radius"], 2)
                        for entry in annotation["frames"].values()
                        if entry.get("radius") is not None),
    }


def frame_runs(annotation: Mapping[str, Any]) -> dict:
    """Consecutive index runs. The tracker must solve temporal identity, so
    isolated frames are counted and reported rather than averaged away."""
    indices = sorted(int(key) for key in annotation["frames"])
    if not indices:
        return {"visible_runs": 0, "absent_runs": 0, "longest_visible_run": 0,
                "longest_absent_run": 0, "longest_run": 0, "isolated_visible": 0}

    def visible(index: int) -> bool:
        return annotation["frames"][str(index)]["state"] == STATE_VISIBLE

    runs = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index == previous + 1 and visible(index) == visible(previous):
            previous = index
            continue
        runs.append((start, previous, visible(previous)))
        start = previous = index
    runs.append((start, previous, visible(previous)))

    visible_runs = [r for r in runs if r[2]]
    absent_runs = [r for r in runs if not r[2]]
    return {
        "visible_runs": len(visible_runs),
        "absent_runs": len(absent_runs),
        "longest_visible_run": max((r[1] - r[0] + 1 for r in visible_runs),
                                   default=0),
        "longest_absent_run": max((r[1] - r[0] + 1 for r in absent_runs), default=0),
        "longest_run": max((r[1] - r[0] + 1 for r in runs), default=0),
        "isolated_visible": sum(1 for r in visible_runs if r[1] == r[0]),
    }


# ---------------------------------------------------------------------------
# split manifest
# ---------------------------------------------------------------------------

def validate_split(payload: Mapping[str, Any], *, sources: Optional[Mapping] = None,
                   legacy_groups: Optional[Iterable[str]] = None) -> dict:
    if not isinstance(payload, Mapping):
        _fail("split manifest must be an object")
    version = payload.get("schema_version")
    if version != SCHEMA_SPLIT:
        _fail(f"split schema_version must be {SCHEMA_SPLIT!r}, got {version!r}")
    status = payload.get("status")
    if status != "frozen":
        _fail(f"split status must be 'frozen', got {status!r}")
    clips = payload.get("clips")
    if not isinstance(clips, list) or not clips:
        _fail("split manifest must carry a non-empty clips list")

    by_clip, by_hash, by_group = {}, {}, {}
    for index, clip in enumerate(clips):
        where = f"clips[{index}]"
        if not isinstance(clip, Mapping):
            _fail(f"{where} must be an object")
        for field in ("clip_id", "split", "source_group", "sha256"):
            if field not in clip:
                _fail(f"{where} is missing required field {field!r}")
        clip_id = _text(clip["clip_id"], f"{where}.clip_id")
        split = clip["split"]
        if split not in SPLITS:
            _fail(f"{where}.split must be one of {', '.join(SPLITS)}, got {split!r}")
        group = _text(clip["source_group"], f"{where}.source_group")
        sha = _sha(clip["sha256"], f"{where}.sha256")
        if clip_id in by_clip:
            _fail(f"clip {clip_id!r} appears twice in the split manifest")
        if sha in by_hash:
            _fail(f"source hash of {clip_id!r} is already assigned to "
                  f"{by_hash[sha]!r}; the same clip cannot be in two splits")
        if group in by_group and by_group[group] != split:
            _fail(f"shoot group {group!r} is split across {by_group[group]!r} and "
                  f"{split!r}; a shared golfer, wardrobe or location may not leak")
        by_clip[clip_id], by_hash[sha], by_group[group] = clip_id, clip_id, split
        if sources is not None:
            manifest_clip = sources_by_id(sources).get(clip_id)
            if manifest_clip is None:
                _fail(f"split references {clip_id!r} which is not in the sources "
                      "manifest")
            if manifest_clip["sha256"] != sha:
                _fail(f"split records a different sha256 for {clip_id!r} than the "
                      "sources manifest")

    legacy_groups = set(legacy_groups or ())
    for clip in clips:
        if clip["split"] == SPLIT_FINAL_TEST:
            if clip.get("legacy_clip"):
                _fail(f"{clip['clip_id']!r} is a consumed v1 clip and must not be in "
                      "the v2 final test")
            if clip["source_group"] in legacy_groups:
                _fail(f"{clip['clip_id']!r} shares shoot group "
                      f"{clip['source_group']!r} with the consumed v1 corpus and must "
                      "not be in the v2 final test")
    return dict(payload)


def load_split(path: PathLike, *, sources: Optional[Mapping] = None,
               legacy_groups: Optional[Iterable[str]] = None,
               allow_final_test: bool = False) -> dict:
    payload = validate_split(_read_json_no_duplicate_keys(path), sources=sources,
                             legacy_groups=legacy_groups)
    if not allow_final_test:
        # Reading the manifest is fine; it states where final test is. Loading the
        # *labels* is what has to be deliberate.
        pass
    return payload


def clips_for_split(split_manifest: Mapping[str, Any], split: str, *,
                    allow_final_test: bool = False) -> list[str]:
    if split not in SPLITS:
        _fail(f"unknown split {split!r}; expected one of {', '.join(SPLITS)}")
    if split == SPLIT_FINAL_TEST and not allow_final_test:
        raise LeakageError(
            "final test is opt-in: pass --final-test (or allow_final_test=True) to "
            "reach it. Development tooling must not read it by accident.")
    return sorted(clip["clip_id"] for clip in split_manifest["clips"]
                  if clip["split"] == split)


def development_clips(split_manifest: Mapping[str, Any]) -> list[str]:
    return sorted(clip["clip_id"] for clip in split_manifest["clips"]
                  if clip["split"] in DEVELOPMENT_SPLITS)


def legacy_groups_from_split(path: PathLike) -> set[str]:
    """Shoot families consumed by a previous (already-scored) corpus."""
    payload = _read_json_no_duplicate_keys(path)
    groups = set()
    for clip in payload.get("clips", []):
        source_id = clip.get("source_id") or ""
        if source_id:
            groups.add(source_id)
        groups.add(f"pexels_{str(clip.get('clip_id', ''))[7:11]}")
    groups.discard("pexels_")
    return groups


# ---------------------------------------------------------------------------
# dataset-level validation, statistics, media status
# ---------------------------------------------------------------------------

def validate_dataset(*, sources: Mapping[str, Any], split: Mapping[str, Any],
                     annotation_paths: Sequence[PathLike],
                     project_root: PathLike,
                     legacy_groups: Optional[Iterable[str]] = None,
                     check_media: bool = True) -> dict:
    manifest = validate_sources(sources)
    split_payload = validate_split(split, sources=manifest,
                                   legacy_groups=legacy_groups)
    problems: list[str] = []
    seen_clips = set()
    summaries = []
    for path in sorted(Path(p) for p in annotation_paths):
        try:
            annotation = load_annotation(path, sources=manifest,
                                         project_root=project_root,
                                         bind=check_media)
        except VideoContractError as exc:
            problems.append(f"{path.name}: {exc}")
            continue
        if annotation["clip_id"] in seen_clips:
            problems.append(f"{annotation['clip_id']}: more than one annotation file")
        seen_clips.add(annotation["clip_id"])
        clip = sources_by_id(manifest)[annotation["clip_id"]]
        if "source_group" in annotation and annotation["source_group"] and \
                annotation["source_group"] != clip["source_group"]:
            problems.append(f"{annotation['clip_id']}: annotation source_group "
                            f"{annotation['source_group']!r} disagrees with the "
                            f"manifest's {clip['source_group']!r}")
        summaries.append(annotation_summary(annotation))
    if problems:
        _fail("dataset validation failed:\n  - " + "\n  - ".join(problems))
    return {"clips": len(manifest["clips"]), "annotated_clips": len(summaries),
            "split": split_payload, "summaries": summaries}


def media_status(*, sources: Mapping[str, Any], split: Optional[Mapping[str, Any]] = None,
                 project_root: PathLike = ".") -> dict:
    """Which media is present, and never accept a differently-hashed file."""
    root = Path(project_root)
    manifest = validate_sources(sources)
    split_of = {}
    if split is not None:
        for clip in split["clips"]:
            split_of[clip["clip_id"]] = clip["split"]
    order = {name: index for index, name in enumerate(SPLITS)}
    buckets: dict[str, dict] = {
        name: {"present": 0, "total": 0, "missing": []} for name in SPLITS}
    other = {"present": 0, "total": 0, "missing": []}
    for clip in sorted(manifest["clips"],
                       key=lambda c: (order.get(split_of.get(c["clip_id"], ""), 99),
                                      c["clip_id"])):
        split_name = split_of.get(clip["clip_id"], "unassigned")
        bucket = buckets.get(split_name, other)
        bucket["total"] += 1
        path = root / clip["media_path"]
        if not path.exists():
            bucket["missing"].append({"clip_id": clip["clip_id"],
                                      "expected_sha256": clip["sha256"],
                                      "source": clip["source_url"]})
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != clip["sha256"]:
            bucket["missing"].append({
                "clip_id": clip["clip_id"], "expected_sha256": clip["sha256"],
                "source": clip["source_url"],
                "reason": "on-disk file has a different hash; a differently hashed "
                          "file is a different clip, not a substitute"})
            continue
        bucket["present"] += 1
    buckets["unassigned"] = other
    return buckets


def dataset_statistics(*, sources: Mapping[str, Any], split: Mapping[str, Any],
                       annotation_paths: Sequence[PathLike],
                       project_root: PathLike, bind: bool = False) -> dict:
    manifest = validate_sources(sources)
    split_of = {clip["clip_id"]: clip["split"] for clip in split["clips"]}
    by_id = sources_by_id(manifest)
    annotations = {}
    for path in sorted(Path(p) for p in annotation_paths):
        annotation = load_annotation(path, sources=manifest, project_root=project_root,
                                     bind=bind)
        annotations[annotation["clip_id"]] = annotation

    def blank():
        return {
            "clips": 0, "annotated_clips": 0, "labelled_frames": 0,
            "states": {state: 0 for state in STATES},
            "origin": {}, "tags": {}, "intended_tags": {},
            "visible_runs": 0, "absent_runs": 0, "longest_visible_run": 0,
            "longest_absent_run": 0, "isolated_visible": 0,
            "resolutions": {}, "fps": {}, "providers": {}, "groups": {},
            "radii": [], "annotators": {},
        }

    report = {name: blank() for name in SPLITS}
    report["unassigned"] = blank()
    for clip in manifest["clips"]:
        split_name = split_of.get(clip["clip_id"], "unassigned")
        bucket = report[split_name]
        bucket["clips"] += 1
        bucket["providers"][clip["provider"]] = \
            bucket["providers"].get(clip["provider"], 0) + 1
        bucket["groups"][clip["source_group"]] = \
            bucket["groups"].get(clip["source_group"], 0) + 1
        for tag in clip.get("intended_tags", []):
            bucket["intended_tags"][tag] = bucket["intended_tags"].get(tag, 0) + 1
        if clip.get("width") and clip.get("height"):
            key = f"{clip['width']}x{clip['height']}"
            bucket["resolutions"][key] = bucket["resolutions"].get(key, 0) + 1
        if clip.get("fps"):
            key = f"{round(float(clip['fps']), 2):g}"
            bucket["fps"][key] = bucket["fps"].get(key, 0) + 1
        annotation = annotations.get(clip["clip_id"])
        if annotation is None:
            continue
        bucket["annotated_clips"] += 1
        summary = annotation_summary(annotation)
        bucket["labelled_frames"] += summary["labelled_frames"]
        for state, count in summary["states"].items():
            bucket["states"][state] += count
        for origin, count in summary["origins"].items():
            bucket["origin"][origin] = bucket["origin"].get(origin, 0) + count
        for tag, count in summary["tags"].items():
            bucket["tags"][tag] = bucket["tags"].get(tag, 0) + count
        for tag in clip.get("tags", []) + annotation.get("tags", []):
            bucket["tags"][f"clip:{tag}"] = bucket["tags"].get(f"clip:{tag}", 0) + 1
        bucket["visible_runs"] += summary["visible_runs"]
        bucket["absent_runs"] += summary["absent_runs"]
        bucket["longest_visible_run"] = max(bucket["longest_visible_run"],
                                            summary["longest_visible_run"])
        bucket["longest_absent_run"] = max(bucket["longest_absent_run"],
                                           summary["longest_absent_run"])
        bucket["isolated_visible"] += frame_runs(annotation)["isolated_visible"]
        bucket["radii"] += summary["radii"]
        annotator = annotation.get("annotator") or "unattributed"
        bucket["annotators"][annotator] = bucket["annotators"].get(annotator, 0) + 1

    for bucket in report.values():
        radii = sorted(bucket["radii"])
        bucket["radii"] = radii
        bucket["ball_diameter_px"] = ({
            "min": round(2 * radii[0], 1), "median": round(2 * radii[len(radii) // 2], 1),
            "max": round(2 * radii[-1], 1)} if radii else None)
        bucket["radii_with_radius"] = len(radii)
        labelled = max(bucket["labelled_frames"], 1)
        bucket["state_share"] = {state: round(count / labelled, 4)
                                 for state, count in bucket["states"].items()}
    return report


def review_flags(annotation: Mapping[str, Any], *, project_root: PathLike,
                 jump_fraction: float = 0.25,
                 tolerance_frame_fraction: float = 0.02,
                 tolerance_radius_multiple: float = 1.5) -> list[dict]:
    """Flag suspicious labels for a human to adjudicate. Never rewrites them."""
    width, height = annotation["width"], annotation["height"]
    flags: list[dict] = []
    indices = sorted(int(key) for key in annotation["frames"])

    def entry(index: int) -> Mapping:
        return annotation["frames"][str(index)]

    jump_limit = jump_fraction * width
    for previous, current in zip(indices, indices[1:]):
        if current != previous + 1:
            continue
        a, b = entry(previous), entry(current)
        if a["state"] == STATE_VISIBLE and b["state"] == STATE_VISIBLE:
            distance = math.dist((a["x"], a["y"]), (b["x"], b["y"]))
            radius = max(a.get("radius") or 0.0, b.get("radius") or 0.0)
            tolerance = max(tolerance_frame_fraction * max(width, height),
                            tolerance_radius_multiple * radius)
            if distance > jump_limit:
                flags.append({"clip_id": annotation["clip_id"], "frame": current,
                              "kind": "large_jump", "detail":
                              f"{distance:.0f} px from frame {previous} exceeds "
                              f"{jump_limit:.0f} px ({jump_fraction:.0%} of width)"})
            elif distance > tolerance:
                flags.append({"clip_id": annotation["clip_id"], "frame": current,
                              "kind": "near_tolerance", "detail":
                              f"{distance:.1f} px from frame {previous} is beyond the "
                              f"match tolerance ({tolerance:.1f} px)"})
    for index in indices:
        state = entry(index)["state"]
        neighbours = [entry(n)["state"] for n in (index - 1, index + 1) if n in indices]
        if state == STATE_VISIBLE and len(neighbours) == 2 and \
                all(n != STATE_VISIBLE for n in neighbours):
            flags.append({"clip_id": annotation["clip_id"], "frame": index,
                          "kind": "isolated_visible",
                          "detail": "a single visible frame between non-visible frames"})
        if state == STATE_AMBIGUOUS:
            flags.append({"clip_id": annotation["clip_id"], "frame": index,
                          "kind": "ambiguous", "detail": "reviewer would not defend "
                          "this call; excluded from strict metrics"})
    for key, proposal in (annotation.get("proposals") or {}).items():
        if proposal.get("status") == "pending":
            flags.append({"clip_id": annotation["clip_id"], "frame": int(key),
                          "kind": "pending_proposal",
                          "detail": "proposal not yet accepted or rejected; it is not "
                                    "a label"})
    return sorted(flags, key=lambda item: (item["clip_id"], item["frame"], item["kind"]))


def dataset_review_flags(*, sources: Mapping[str, Any],
                         annotation_paths: Sequence[PathLike],
                         project_root: PathLike, bind: bool = True) -> list[dict]:
    manifest = validate_sources(sources)
    flags: list[dict] = []
    for path in sorted(Path(p) for p in annotation_paths):
        annotation = load_annotation(path, sources=manifest, project_root=project_root,
                                     bind=bind)
        flags.extend(review_flags(annotation, project_root=project_root))
    return flags


# ---------------------------------------------------------------------------
# annotation store (shared by the CLI and the local annotation UI)
# ---------------------------------------------------------------------------

def new_annotation(*, clip_id: str, video: str, source_sha256: str, width: int,
                   height: int, fps: Optional[float] = None,
                   frames_total: Optional[int] = None, source_group: Optional[str] = None,
                   annotator: str = "unattributed", reviewer: Optional[Mapping] = None,
                   tags: Optional[Sequence[str]] = None, notes: str = "") -> dict:
    payload = {
        "schema_version": SCHEMA_ANNOTATION,
        "clip_id": clip_id,
        "video": video,
        "source_sha256": source_sha256,
        "width": width,
        "height": height,
        "fps": fps,
        "frames_total": frames_total,
        "source_group": source_group,
        "annotator": annotator,
        "reviewer": dict(reviewer) if reviewer else None,
        "tags": list(tags or []),
        "notes": notes,
        "frames": {},
        "proposals": {},
    }
    return validate_annotation(payload)


def annotation_path_for(annotations_dir: PathLike, clip_id: str) -> Path:
    return Path(annotations_dir) / f"{clip_id}.json"


def load_or_new_annotation(*, annotations_dir: PathLike, sources: Mapping[str, Any],
                           clip_id: str, project_root: PathLike,
                           annotator: str, reviewer: Optional[Mapping] = None,
                           bind: bool = True) -> dict:
    """Resume an existing annotation file, or start one for a known clip."""
    path = annotation_path_for(annotations_dir, clip_id)
    if path.exists():
        return load_annotation(path, sources=sources, project_root=project_root,
                               bind=bind)
    clip = sources_by_id(validate_sources(sources)).get(clip_id)
    if clip is None:
        _fail(f"{clip_id!r} is not in the sources manifest; refusing to start an "
              "annotation for an unknown clip")
    return new_annotation(clip_id=clip_id, video=clip["media_path"],
                          source_sha256=clip["sha256"], width=clip["width"],
                          height=clip["height"], fps=clip.get("fps"),
                          frames_total=clip.get("frames"),
                          source_group=clip.get("source_group"),
                          annotator=annotator, reviewer=reviewer,
                          tags=clip.get("tags") or [])


def set_label(payload: Mapping[str, Any], frame_index: int, *, state: str,
              origin: str, reviewer: Optional[str] = None, x: Optional[float] = None,
              y: Optional[float] = None, radius: Optional[float] = None,
              reason: Optional[str] = None, explanation: Optional[str] = None,
              candidate_region: Optional[Sequence[float]] = None,
              tags: Optional[Sequence[str]] = None,
              confidence: Optional[str] = None,
              note: Optional[str] = None) -> dict:
    """Record one reviewed frame. Raises rather than guessing a coordinate."""
    entry: dict[str, Any] = {"state": state, "origin": origin}
    if reviewer:
        entry["reviewer"] = reviewer
    if state == STATE_VISIBLE:
        if x is None or y is None:
            _fail(f"frame {frame_index} is VISIBLE and needs both x and y; a visible "
                  "frame without a coordinate is not a label")
        entry["x"], entry["y"] = float(x), float(y)
        if radius is not None:
            entry["radius"] = float(radius)
    else:
        if x is not None or y is not None:
            _fail(f"frame {frame_index} is {state}; coordinates are forbidden on a "
                  "non-visible frame")
        if reason:
            entry["reason"] = reason
        if explanation:
            entry["explanation"] = explanation
        if candidate_region is not None:
            if state != STATE_AMBIGUOUS:
                _fail("only AMBIGUOUS frames may carry a candidate_region")
            entry["candidate_region"] = [float(v) for v in candidate_region]
    if tags:
        entry["tags"] = list(tags)
    if confidence:
        entry["confidence"] = confidence
    if note:
        entry["note"] = note
    frames = dict(payload["frames"])
    frames[str(int(frame_index))] = entry
    updated = dict(payload)
    updated["frames"] = frames
    return validate_annotation(updated)


def drop_label(payload: Mapping[str, Any], frame_index: int) -> dict:
    frames = dict(payload["frames"])
    if str(int(frame_index)) not in frames:
        _fail(f"frame {frame_index} has no label to drop")
    del frames[str(int(frame_index))]
    updated = dict(payload)
    updated["frames"] = frames
    return validate_annotation(updated)


def set_proposal(payload: Mapping[str, Any], frame_index: int, *, x: Optional[float],
                 y: Optional[float], method: str, status: str = "pending",
                 confidence: Optional[str] = None) -> dict:
    """Record a machine guess. It stays a proposal until a reviewer decides."""
    if status not in ("pending", "accepted", "rejected"):
        _fail("proposal status must be pending, accepted or rejected")
    entry: dict[str, Any] = {"origin": PROPOSAL_ORIGIN, "method": method,
                             "status": status}
    if x is not None and y is not None:
        entry["x"], entry["y"] = float(x), float(y)
    elif status != "rejected":
        _fail("a proposal must carry x and y unless it is rejected")
    if confidence:
        entry["confidence"] = confidence
    proposals = dict(payload.get("proposals") or {})
    proposals[str(int(frame_index))] = entry
    updated = dict(payload)
    updated["proposals"] = proposals
    return validate_annotation(updated)


def resolve_proposal(payload: Mapping[str, Any], frame_index: int, *, status: str,
                     label: Optional[Mapping[str, Any]] = None) -> dict:
    """Accept or reject a proposal. Accepting does not create a label: the
    reviewer's explicit label is written separately, and rejecting keeps the guess
    as negative evidence."""
    proposals = dict(payload.get("proposals") or {})
    key = str(int(frame_index))
    if key not in proposals:
        _fail(f"frame {frame_index} has no proposal to resolve")
    if status not in ("accepted", "rejected"):
        _fail("resolving a proposal means accepting or rejecting it")
    proposals[key] = {**proposals[key], "status": status}
    updated = dict(payload)
    updated["proposals"] = proposals
    updated = validate_annotation(updated)
    if label is not None and status == "accepted":
        updated = set_label(updated, frame_index, **label)
    return updated


def serialize_annotation(payload: Mapping[str, Any]) -> str:
    ordered = dict(payload)
    ordered["frames"] = {key: payload["frames"][key]
                         for key in sorted(payload["frames"], key=int)}
    ordered["proposals"] = {key: payload["proposals"][key]
                            for key in sorted(payload.get("proposals") or {}, key=int)}
    return json.dumps(ordered, indent=1, sort_keys=False) + "\n"


def save_annotation(path: PathLike, payload: Mapping[str, Any], *,
                    sources: Optional[Mapping] = None,
                    project_root: Optional[PathLike] = None) -> Path:
    validated = validate_annotation(payload, sources=sources,
                                    project_root=project_root)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(serialize_annotation(validated), encoding="utf-8")
    return target


def load_split_labels(*, split_manifest: Mapping[str, Any],
                      annotations_dir: PathLike, sources: Mapping[str, Any],
                      split: str, project_root: PathLike,
                      allow_final_test: bool = False, bind: bool = True) -> dict:
    """Load the labels of one split. Final test refuses without an explicit opt-in,
    so a training or tuning helper cannot reach it by accident."""
    clip_ids = clips_for_split(split_manifest, split, allow_final_test=allow_final_test)
    loaded = {}
    for clip_id in clip_ids:
        path = annotation_path_for(annotations_dir, clip_id)
        if not path.exists():
            continue
        loaded[clip_id] = load_annotation(path, sources=sources,
                                          project_root=project_root, bind=bind)
    return loaded
