"""Research-only target-tracking benchmark: reference contract, predictions
contract, and a stdlib-only evaluator with a markdown report.

Three things live here, and all three are deliberately **stdlib-only** so they run
in an interpreter that has no cv2/numpy:

``fairwayos-target-reference.v1``
    the *dataset contract*: a sparse **AI-visual reference** for the twelve Tiger
    clips. It is not ground truth and no field may make it look like it is. The
    provenance block must state ``ground_truth=false``, ``human_confirmed=false``,
    ``label_type="ai_visual_reference"``, ``research_only=true`` and
    ``production_eligible=false``, and a payload that contradicts any of those is
    refused rather than repaired.

``fairwayos-target-predictions.v1``
    the *prediction contract*: one point per source frame per target, as emitted by
    a runner. Validated with the same rigour as the reference, because a prediction
    file is where an operator placement, a derived point or a seed could otherwise
    be passed off as an automatic observation. An ``observed_template*`` point must
    carry origin ``automatic``; a point at the seed frame may never be automatic.

the evaluator
    what the numbers mean. Stated once, because every one of them is a place a
    benchmark could lie:

    * only frames the reference explicitly labelled ``visible`` are scored, minus
      the target's seed frame; a frame the reference never mentions is **absent
      from the denominator** and is not evidence that the target was absent;
    * an ``uncertain`` label is excluded from scoring altogether: it is neither a
      positive nor a negative;
    * a *negative* (``occluded`` / ``out_of_view``) is a claim the reference made,
      never an assumption read off an unlabelled frame;
    * nothing counts as correct unless it is an automatic observation within the
      label's own tolerance;
    * interpolated (derived) points, operator anchors and the seed are counted
      separately and are never evidence of an automatic observation;
    * a recovery time is only reported across explicitly consecutive labels with an
      occlusion -> visible transition; a sparse reference returns ``None`` with the
      reason, not a number;
    * the model score is carried for provenance and is used in no metric at all;
    * development and evaluation splits are reported separately and are never
      pooled into one "overall" number. These twelve clips were already used to
      build the assisted showcase, so the split is bookkeeping, not a statistical
      holdout, and no statement here claims otherwise.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, NoReturn, Optional, Sequence

from .config import TargetOptions as _TargetOptions
from .errors import DemoError
from .tracking import (AUTOMATIC_STATUSES, ORIGIN_AUTOMATIC, ORIGIN_DERIVED,
                       ORIGIN_OPERATOR, STATUS_INTERPOLATED, STATUS_LOST,
                       STATUS_MANUAL, STATUS_MISSING, STATUS_OBSERVED,
                       STATUS_OBSERVED_LK)

DATASET_SCHEMA = "fairwayos-target-reference.v1"
PREDICTIONS_SCHEMA = "fairwayos-target-predictions.v1"
REPORT_SCHEMA = "fairwayos-target-benchmark-report.v1"

#: The only provenance this contract accepts. Nothing here is negotiable: the
#: twelve clips are AI-visual references, not human ground truth.
REQUIRED_PROVENANCE = {
    "ground_truth": False,
    "human_confirmed": False,
    "label_type": "ai_visual_reference",
    "research_only": True,
    "production_eligible": False,
}

TARGET_NAMES = ("ball", "clubhead")
LABEL_STATES = ("visible", "occluded", "out_of_view", "uncertain")
NEGATIVE_STATES = ("occluded", "out_of_view")
SPLITS = ("development", "evaluation")
DIRECTIONS = ("forward", "backward", "both")

_TOP_LEVEL_REQUIRED = frozenset({"schema_version", "provenance", "clips"})
_TOP_LEVEL_OPTIONAL = frozenset({"notes"})
_CLIP_FIELDS = frozenset({"id", "path", "sha256", "width", "height", "frame_count",
                          "fps_rational", "stroke_type", "split", "tags", "window",
                          "cut_frames", "targets"})
_TARGET_REQUIRED = frozenset({"labels"})
_TARGET_OPTIONAL = frozenset({"seed", "options"})
_SEED_FIELDS = frozenset({"frame", "x", "y", "note"})
_LABEL_FIELDS = frozenset({"frame", "state", "x", "y", "tolerance_px", "note",
                           "reviewer"})
_WINDOW_REQUIRED = frozenset({"start_frame", "end_frame"})
_WINDOW_OPTIONAL = frozenset({"note"})
_PREDICTION_TOP = frozenset({"schema_version", "variant", "clips"})
_PREDICTION_CLIP = frozenset({"id", "sha256", "targets"})
_PREDICTION_TARGET_REQUIRED = frozenset({"seed", "points"})
_PREDICTION_TARGET_OPTIONAL = frozenset({"options"})
_PREDICTION_SEED_REQUIRED = frozenset({"frame", "x", "y"})
_PREDICTION_POINT = frozenset({"source_frame", "x", "y", "status", "origin",
                               "method", "score"})
_PREDICTION_STATUSES = frozenset({STATUS_OBSERVED, STATUS_OBSERVED_LK, STATUS_MANUAL,
                                  STATUS_INTERPOLATED, STATUS_MISSING, STATUS_LOST})
#: The methods that may carry an automatic observation. ``template`` and
#: ``template+lk`` are the classic demo tracker's. The next two are the isolated
#: experimental masked-foreground candidate's
#: (``ghostcaddie.demo.foreground_tracker``): it reports a re-acquired frame under its
#: OWN method name so a bounded automatic re-acquisition can never be read as an
#: uninterrupted continuation of the same track. The following pair belongs to the second
#: isolated candidate (``ghostcaddie.demo.putter_detector``), an untrained STRUCTURAL
#: detector that re-segments compact head evidence in each frame: it likewise keeps a
#: re-detected frame under its own name. This tuple is the ONLY contract surface the
#: candidates extend: no metric, label, split or option changes with it. The final pair
#: names measured compact regions from the optional SAM2 mask experiment. The
#: ball-evidence pair names seed/current-pixel validation of classic proposals;
#: neither candidate establishes semantic or physical target identity. Nothing else
#: may claim to be a tracker method, and an empty method (which the classic policy
#: writes on non-observations) stays accepted exactly as before.
AUTOMATIC_METHODS = ("template", "template+lk", "foreground_template",
                     "foreground_template_reacquired", "putter_detector",
                     "putter_detector_reacquired", "sam2_head_region",
                     "sam2_head_region_reacquired", "ball_evidence_template",
                     "ball_evidence_template_reacquired", "sam2_swing_core",
                     "sam2_swing_core_reacquired", "shaft_head", "shaft_head_reacquired",
                     "learned_bbox_center")
#: Options a reference may set on a target. This is the tracker's own option set
#: (``demo.config.TargetOptions``) plus the one interpolation switch the runner
#: passes to ``resolve_track``. Nothing here is a label.
_OPTION_FIELDS = {
    "mode": str, "template_size": int, "search_radius": int, "min_score": float,
    "max_jump_px": float, "loss_frames": int, "direction": str, "lk_refine": bool,
    "template_refresh_frames": int, "template_refresh_min_score": float,
    "track_radius_frames": int, "track_until_frame": int, "automatic_until_frame": int,
}
#: The tracker options that are ``Optional[int]``: null means "no bound", and a
#: reference or a prediction is allowed to declare one as null rather than omit it.
_OPTIONAL_INT_OPTIONS = ("track_radius_frames", "track_until_frame",
                         "automatic_until_frame")
#: The tracker's own defaults, read from the dataclass rather than copied, so a
#: reference that sets only some options still records the options its run used.
_OPTION_DEFAULTS = {field.name: getattr(_TargetOptions(), field.name)
                    for field in _TargetOptions.__dataclass_fields__.values()}
if set(_OPTION_DEFAULTS) != set(_OPTION_FIELDS):  # pragma: no cover - a guard
    raise RuntimeError("target_benchmark option contract drifted from "
                       "demo.config.TargetOptions: "
                       f"{sorted(set(_OPTION_DEFAULTS) ^ set(_OPTION_FIELDS))}")
_VARIANT_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")

#: The same opening every report carries. It is not decoration: each line closes a
#: claim a reader could otherwise infer from the tables.
SCOPE_NOTES = (
    "Reference labels are sparse AI-visual references, not ground truth "
    "(ground_truth=false, human_confirmed=false, label_type=ai_visual_reference): "
    "recall is measured only on the frames the reference explicitly labelled "
    "visible. Frames the reference does not mention are absent from the "
    "denominator and are not evidence that the target was absent.",
    "These clips were already used to build the assisted showcase: the "
    "development/evaluation split is bookkeeping, not a statistical holdout, and "
    "no number in this report is a held-out generalisation estimate.",
    "The clips come from one broadcast family and share camera, encoder and "
    "tournament. No family-independence or cross-family claim is made or implied "
    "by scoring them separately.",
    "A target's seed frame is excluded from every score. The seed is a reference "
    "point, and an automatic observation at the seed frame is refused outright.",
    "Only automatic observations can score: the classic tracker's `template` and "
    "`template+lk`, or a candidate's own declared experimental method "
    "(foreground_template, putter_detector, sam2_head_region, or "
    "ball_evidence_template, including each method's _reacquired form). "
    "Re-acquired frames carry their own method name so a re-acquisition is never "
    "readable as a continuation. Interpolated (derived) "
    "points, operator anchors and the seed are counted separately and are never "
    "evidence of an automatic observation.",
    "Scope of the seeded track: one FORWARD leg per target. Labels at or before the "
    "seed frame are pre-initialization -- retained and counted separately "
    "(preseed_labels), excluded from the recall denominator, never scored as misses. "
    "A target with no seed has no automatic track and therefore no accuracy "
    "denominator at all (unseeded_labels), and nothing is claimed about its "
    "acquisition or re-acquisition.",
    "Label states other than visible are scored as negatives only when the "
    "reference said so (occluded / out_of_view); uncertain labels are excluded "
    "from scoring entirely, and an unlabelled frame is never assumed empty.",
    "Labels outside the clip window, and labels on a structural cut frame, are "
    "retained and counted separately (out_of_window_labels, on_cut_frame_labels) but "
    "excluded from every tracker-quality denominator: nothing is predicted outside "
    "the window by structural policy, so an outside-window negative is zero by "
    "construction and would only flatter the negative false-positive rate.",
    "The tracker's per-point score is recorded for provenance and is used in no "
    "metric: it is never treated as accuracy.",
)


class TargetBenchmarkError(DemoError):
    """The reference dataset, the predictions, or the comparison is not usable.

    Every refusal is stated with the field that caused it. Nothing is repaired,
    defaulted or dropped silently.
    """


# --------------------------------------------------------------------------
# small validators (strict: a wrong type is a refusal, not a coercion)


def _fail(message: str) -> NoReturn:
    raise TargetBenchmarkError(message)


def _mapping(value: Any, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> list:
    if not isinstance(value, list):
        _fail(f"{name} must be a list")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{name} must be a finite number")
    return float(value)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{name} must be an integer (non-integer frame numbers are refused)")
    return int(value)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{name} must be a non-empty string")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{name} must be a boolean")
    return value


def _fields(value: Mapping, required: Iterable[str], optional: Iterable[str],
            name: str) -> None:
    required, optional = frozenset(required), frozenset(optional)
    keys = set(value)
    missing = sorted(required - keys)
    if missing:
        _fail(f"{name} is missing required field: {missing[0]}")
    extra = sorted(keys - required - optional)
    if extra:
        _fail(f"{name} carries a field this contract does not define: {extra[0]}")


def _sha(value: Any, name: str) -> str:
    text = _text(value, name).strip().lower()
    if not _SHA_RE.match(text):
        _fail(f"{name} must be a lowercase 64-character sha256 hex digest")
    return text


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it; the frozen input binding of every clip."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# dataset contract


def _fps_rational(value: Any, name: str) -> dict:
    """Accept ``"24000/1001"`` or ``{"numerator": .., "denominator": ..}``.

    The declared cadence is recorded as given plus its float value; nothing here
    claims the real capture cadence, which a container's nominal frame rate is not
    evidence of.
    """
    if isinstance(value, str):
        parts = value.strip().split("/")
        if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
            _fail(f"{name} must be 'numerator/denominator' or an object with both")
        numerator, denominator = int(parts[0]), int(parts[1])
    else:
        _fields(value, ("numerator", "denominator"), (), name)
        numerator = _integer(value["numerator"], f"{name}.numerator")
        denominator = _integer(value["denominator"], f"{name}.denominator")
    if numerator <= 0 or denominator <= 0:
        _fail(f"{name} must be positive")
    return {"numerator": numerator, "denominator": denominator,
            "value": round(numerator / denominator, 6)}


def _window(value: Any, name: str, frame_count: int) -> dict:
    mapping = _mapping(value, name)
    _fields(mapping, _WINDOW_REQUIRED, _WINDOW_OPTIONAL, name)
    start = _integer(mapping["start_frame"], f"{name}.start_frame")
    end = _integer(mapping["end_frame"], f"{name}.end_frame")
    if start < 0 or end < start:
        _fail(f"{name} must satisfy 0 <= start_frame <= end_frame")
    if end >= frame_count:
        _fail(f"{name}.end_frame {end} is past the clip's {frame_count} frames")
    note = mapping.get("note", "")
    if not isinstance(note, str):
        _fail(f"{name}.note must be a string")
    return {"start_frame": start, "end_frame": end, "note": note}


def _cut_frames(value: Any, name: str, frame_count: int) -> list:
    frames = []
    seen = set()
    for index, raw in enumerate(_sequence(value, name)):
        frame = _integer(raw, f"{name}[{index}]")
        if frame < 0 or frame >= frame_count:
            _fail(f"{name}[{index}] {frame} is outside the clip's frame range")
        if frame in seen:
            _fail(f"{name} repeats frame {frame}: a duplicate structural frame is refused")
        seen.add(frame)
        frames.append(frame)
    return sorted(frames)


def _seed(value: Any, name: str, window: Mapping, width: int, height: int) -> Optional[dict]:
    if value is None:
        return None
    mapping = _mapping(value, name)
    _fields(mapping, _SEED_FIELDS, (), name)
    frame = _integer(mapping["frame"], f"{name}.frame")
    if not window["start_frame"] <= frame <= window["end_frame"]:
        _fail(f"{name}.frame {frame} is outside the clip window "
              f"[{window['start_frame']}, {window['end_frame']}]")
    x = _finite(mapping["x"], f"{name}.x")
    y = _finite(mapping["y"], f"{name}.y")
    if not 0 <= x <= width or not 0 <= y <= height:
        _fail(f"{name} ({x}, {y}) is outside the {width}x{height} frame")
    return {"frame": frame, "x": x, "y": y, "note": _text(mapping["note"], f"{name}.note")}


def _label(value: Any, name: str, frame_count: int, width: int, height: int) -> dict:
    """One reference label.

    A label is RETAINED wherever it sits in the clip: the window and cut frames are
    structural metadata, so a label outside the window (or on a cut frame) is kept and
    counted but excluded from every tracker-quality denominator downstream. Only a
    frame the clip does not contain at all, a wrong state, or a non-visible state
    carrying coordinates is refused.
    """
    mapping = _mapping(value, name)
    _fields(mapping, _LABEL_FIELDS, (), name)
    frame = _integer(mapping["frame"], f"{name}.frame")
    if not 0 <= frame < frame_count:
        _fail(f"{name}.frame {frame} is outside the clip's {frame_count} frames")
    state = _text(mapping["state"], f"{name}.state")
    if state not in LABEL_STATES:
        _fail(f"{name}.state {state!r} is not one of {list(LABEL_STATES)}")
    note = _text(mapping["note"], f"{name}.note")
    reviewer = _text(mapping["reviewer"], f"{name}.reviewer")
    if state == "visible":
        x, y = mapping["x"], mapping["y"]
        if x is None or y is None:
            _fail(f"{name} is visible and therefore requires coordinates")
        x, y = _finite(x, f"{name}.x"), _finite(y, f"{name}.y")
        if not 0 <= x <= width or not 0 <= y <= height:
            _fail(f"{name} ({x}, {y}) is outside the {width}x{height} frame")
        tolerance = _finite(mapping["tolerance_px"], f"{name}.tolerance_px")
        if tolerance <= 0:
            _fail(f"{name}.tolerance_px must be positive")
        return {"frame": frame, "state": state, "x": x, "y": y,
                "tolerance_px": tolerance, "note": note, "reviewer": reviewer}
    for field in ("x", "y", "tolerance_px"):
        if mapping[field] is not None:
            _fail(f"{name} is {state!r} and must carry null {field}: a non-visible "
                  f"state has no coordinate and no tolerance")
    return {"frame": frame, "state": state, "x": None, "y": None,
            "tolerance_px": None, "note": note, "reviewer": reviewer}


def _options(value: Any, name: str, window: Mapping) -> dict:
    mapping = _mapping(value, name)
    unknown = sorted(set(mapping) - set(_OPTION_FIELDS) - {"interpolation"})
    if unknown:
        _fail(f"{name} carries an option this runner does not know: {unknown[0]}")
    out: dict = dict(_OPTION_DEFAULTS)
    for field, kind in _OPTION_FIELDS.items():
        if field not in mapping:
            continue
        raw = mapping[field]
        if kind is bool:
            out[field] = _boolean(raw, f"{name}.{field}")
            continue
        if kind is int:
            if raw is None and field in _OPTIONAL_INT_OPTIONS:
                # Optional[int] in the tracker's own options: an unset bound is null.
                out[field] = None
                continue
            out[field] = _integer(raw, f"{name}.{field}")
            continue
        if kind is float:
            value_f = _finite(raw, f"{name}.{field}")
            if field in ("min_score",) and not 0 <= value_f <= 1:
                _fail(f"{name}.{field} must be within [0, 1]")
            if field in ("template_refresh_min_score",) and not 0 <= value_f <= 1:
                _fail(f"{name}.{field} must be within [0, 1]")
            if value_f <= 0 and field in ("max_jump_px",):
                _fail(f"{name}.{field} must be positive")
            out[field] = value_f
            continue
        out[field] = _text(raw, f"{name}.{field}")
    if out.get("mode", "template") != "template":
        _fail(f"{name}.mode must be 'template': a manual track has no automatic "
              f"observation to score")
    if out.get("direction", "forward") not in DIRECTIONS:
        _fail(f"{name}.direction must be one of {list(DIRECTIONS)}")
    if out.get("direction", "forward") != "forward":
        # The benchmark's scope is one forward leg per target: nothing is measured
        # before the seed, which is what makes "pre-seed labels are pre-initialization"
        # and "a loss is terminal without an operator anchor" true. A backward or
        # both-direction track would need its own denominator rules, so it is refused
        # here rather than scored with the wrong ones.
        _fail(f"{name}.direction must be 'forward': this benchmark scores a single "
              f"forward leg from the target's seed")
    if "template_size" in out and out["template_size"] < 3:
        _fail(f"{name}.template_size must be at least 3")
    if "search_radius" in out and out["search_radius"] < 0:
        _fail(f"{name}.search_radius must not be negative")
    if "loss_frames" in out and out["loss_frames"] < 1:
        _fail(f"{name}.loss_frames must be at least 1")
    if "template_refresh_frames" in out and out["template_refresh_frames"] < 0:
        _fail(f"{name}.template_refresh_frames must not be negative")
    for stop in ("track_until_frame", "automatic_until_frame"):
        if out.get(stop) is not None and not window["start_frame"] <= out[stop] <= window["end_frame"]:
            _fail(f"{name}.{stop} must lie inside the clip window")
    interpolation = {"enabled": False, "max_gap_frames": 0}
    if "interpolation" in mapping:
        block = _mapping(mapping["interpolation"], f"{name}.interpolation")
        _fields(block, (), ("enabled", "max_gap_frames"), f"{name}.interpolation")
        if "enabled" in block:
            interpolation["enabled"] = _boolean(block["enabled"],
                                                f"{name}.interpolation.enabled")
        if "max_gap_frames" in block:
            gap = _integer(block["max_gap_frames"], f"{name}.interpolation.max_gap_frames")
            if gap < 0:
                _fail(f"{name}.interpolation.max_gap_frames must not be negative")
            interpolation["max_gap_frames"] = gap
    out["interpolation"] = interpolation
    return out


def _target(value: Any, name: str, window: Mapping, frame_count: int,
            width: int, height: int) -> dict:
    mapping = _mapping(value, name)
    _fields(mapping, _TARGET_REQUIRED, _TARGET_OPTIONAL, name)
    # The seed and the options are bound to the clip WINDOW (that is the interval the
    # runner measures); labels are only bound to the clip's frames, because a label
    # outside the window is retained and counted rather than refused.
    seed = _seed(mapping.get("seed"), f"{name}.seed", window, width, height)
    options = _options(mapping.get("options") or {}, f"{name}.options", window)
    labels = []
    seen: set = set()
    for index, raw in enumerate(_sequence(mapping["labels"], f"{name}.labels")):
        label = _label(raw, f"{name}.labels[{index}]", frame_count, width, height)
        if label["frame"] in seen:
            _fail(f"{name}.labels repeats frame {label['frame']}: duplicate labels "
                  f"are refused")
        seen.add(label["frame"])
        labels.append(label)
    if seed is not None and seed["frame"] in seen:
        # Not a refusal: the label is kept, and the evaluator drops the seed frame
        # from the denominator while the refusal-on-unlabelled rule stays intact.
        pass
    labels.sort(key=lambda item: item["frame"])
    return {"seed": seed, "options": options, "labels": labels}


def _clip(value: Any, name: str, base_dir: Optional[Path], verify_hashes: bool) -> dict:
    mapping = _mapping(value, name)
    _fields(mapping, _CLIP_FIELDS, (), name)
    clip_id = _text(mapping["id"], f"{name}.id")
    path = _text(mapping["path"], f"{name}.path")
    if not Path(path).is_absolute() and base_dir is not None:
        path = str((base_dir / path).resolve())
    declared_sha = _sha(mapping["sha256"], f"{name}.sha256")
    width = _integer(mapping["width"], f"{name}.width")
    height = _integer(mapping["height"], f"{name}.height")
    frame_count = _integer(mapping["frame_count"], f"{name}.frame_count")
    if width <= 0 or height <= 0 or frame_count <= 0:
        _fail(f"{name} width, height and frame_count must be positive")
    fps = _fps_rational(mapping["fps_rational"], f"{name}.fps_rational")
    stroke_type = _text(mapping["stroke_type"], f"{name}.stroke_type")
    split = _text(mapping["split"], f"{name}.split")
    if split not in SPLITS:
        _fail(f"{name}.split {split!r} is not one of {list(SPLITS)}")
    tags = []
    for index, raw in enumerate(_sequence(mapping["tags"], f"{name}.tags")):
        tags.append(_text(raw, f"{name}.tags[{index}]"))
    if len(set(tags)) != len(tags):
        _fail(f"{name}.tags repeats a tag")
    window = _window(mapping["window"], f"{name}.window", frame_count)
    cuts = _cut_frames(mapping["cut_frames"], f"{name}.cut_frames", frame_count)
    targets_raw = _mapping(mapping["targets"], f"{name}.targets")
    if not targets_raw:
        _fail(f"{name}.targets is empty: a clip with no target has nothing to score")
    targets: dict = {}
    for target_name, block in targets_raw.items():
        if target_name not in TARGET_NAMES:
            _fail(f"{name}.targets.{target_name} is not one of {list(TARGET_NAMES)}")
        targets[target_name] = _target(block, f"{name}.targets.{target_name}", window,
                                       frame_count, width, height)
    if verify_hashes:
        resolved = Path(path)
        if not resolved.is_absolute() and base_dir is not None:
            resolved = base_dir / resolved
        if not resolved.exists():
            _fail(f"{name}.path does not exist: {resolved}")
        actual = sha256_file(resolved)
        if actual != declared_sha:
            _fail(f"{name}.sha256 does not match the file: frozen input hash "
                  f"mismatch for {resolved} (declared {declared_sha[:12]}.., "
                  f"actual {actual[:12]}..)")
    return {"id": clip_id, "path": path, "sha256": declared_sha, "width": width,
            "height": height, "frame_count": frame_count, "fps_rational": fps,
            "stroke_type": stroke_type, "split": split, "tags": tags,
            "window": window, "cut_frames": cuts, "targets": dict(sorted(targets.items())),
            "target_names": sorted(targets)}


def validate_dataset(payload: Mapping[str, Any], *, base_dir: str | Path | None = None,
                     verify_hashes: bool = True) -> dict:
    """Validate one ``fairwayos-target-reference.v1`` payload.

    Returns a normalised copy: labels sorted by frame, cut frames sorted, options
    filled with the contract's defaults. Refuses duplicates, non-integer frames,
    non-finite numbers, out-of-bounds coordinates, a non-visible state that carries
    coordinates, a missing reviewer or note, unknown fields, an unsupported schema
    version, a provenance that is not an AI-visual reference, and any clip whose
    file does not hash to its declared ``sha256``.
    """
    mapping = _mapping(payload, "dataset")
    _fields(mapping, _TOP_LEVEL_REQUIRED, _TOP_LEVEL_OPTIONAL, "dataset")
    if mapping["schema_version"] != DATASET_SCHEMA:
        _fail(f"dataset.schema_version must be {DATASET_SCHEMA!r}, got "
              f"{mapping['schema_version']!r}")
    provenance = _mapping(mapping["provenance"], "dataset.provenance")
    for field, expected in REQUIRED_PROVENANCE.items():
        if field not in provenance:
            _fail(f"dataset.provenance must state {field}={expected!r} explicitly")
        if provenance[field] != expected or type(provenance[field]) is not type(expected):
            _fail(f"dataset.provenance.{field} must be {expected!r}; this contract "
                  f"accepts only sparse AI-visual references, never ground truth")
    notes_raw = mapping.get("notes") or []
    notes = []
    for index, raw in enumerate(_sequence(notes_raw, "dataset.notes")):
        notes.append(_text(raw, f"dataset.notes[{index}]"))
    directory = Path(base_dir) if base_dir is not None else None
    clips = []
    seen_ids: set = set()
    for index, raw in enumerate(_sequence(mapping["clips"], "dataset.clips")):
        clip = _clip(raw, f"dataset.clips[{index}]", directory, verify_hashes)
        if clip["id"] in seen_ids:
            _fail(f"dataset.clips repeats id {clip['id']!r}: duplicate clips are refused")
        seen_ids.add(clip["id"])
        clips.append(clip)
    if not clips:
        _fail("dataset.clips is empty: there is nothing to benchmark")
    return {"schema_version": DATASET_SCHEMA,
            "provenance": dict(provenance),
            "notes": notes,
            "clips": clips,
            "clip_index": {clip["id"]: clip for clip in clips}}


def load_dataset(path: str | Path, *, verify_hashes: bool = True) -> tuple[dict, str]:
    """Read and validate a dataset file. Returns ``(dataset, file_sha256)``."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise TargetBenchmarkError(f"dataset could not be read: {source} ({exc})") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TargetBenchmarkError(f"dataset is not valid JSON: {source} ({exc})") from exc
    dataset = validate_dataset(payload, base_dir=source.parent,
                              verify_hashes=verify_hashes)
    return dataset, sha256_text(text)


# --------------------------------------------------------------------------
# predictions contract


def _prediction_point(value: Any, name: str, window: Mapping, cuts: Sequence[int],
                      seed_frame: Optional[int], width: int, height: int) -> dict:
    mapping = _mapping(value, name)
    _fields(mapping, _PREDICTION_POINT, (), name)
    frame = _integer(mapping["source_frame"], f"{name}.source_frame")
    if not window["start_frame"] <= frame <= window["end_frame"]:
        _fail(f"{name}.source_frame {frame} is outside the clip window "
              f"[{window['start_frame']}, {window['end_frame']}]: a point outside the "
              f"window is suppressed at source, not scored")
    if frame in cuts:
        _fail(f"{name}.source_frame {frame} is a structural cut frame: a cut resets "
              f"the source, so no prediction may be reported there")
    status = _text(mapping["status"], f"{name}.status")
    if status not in _PREDICTION_STATUSES:
        _fail(f"{name}.status {status!r} is not one of {sorted(_PREDICTION_STATUSES)}")
    origin = _text(mapping["origin"], f"{name}.origin")
    method = mapping["method"]
    if not isinstance(method, str):
        _fail(f"{name}.method must be a string")
    score = mapping["score"]
    if score is not None:
        score = _finite(score, f"{name}.score")
    x, y = mapping["x"], mapping["y"]
    if status in AUTOMATIC_STATUSES:
        if origin != ORIGIN_AUTOMATIC:
            _fail(f"{name} is {status!r} but origin is {origin!r}: only an automatic "
                  f"origin may carry an automatic observation")
        if x is None or y is None:
            _fail(f"{name} is {status!r} and requires coordinates")
        x, y = _finite(x, f"{name}.x"), _finite(y, f"{name}.y")
        if not 0 <= x <= width or not 0 <= y <= height:
            _fail(f"{name} ({x}, {y}) is outside the {width}x{height} frame")
        if seed_frame is not None and frame == seed_frame:
            _fail(f"{name} is an automatic observation at the seed frame "
                  f"{seed_frame}: the seed frame is an operator anchor and is never "
                  f"scored")
        if method not in AUTOMATIC_METHODS and method != "":
            _fail(f"{name}.method {method!r} is not a tracker method; this contract "
                  f"accepts {list(AUTOMATIC_METHODS)} (or an empty method on a "
                  f"non-observation)")
    elif status == STATUS_MANUAL:
        if origin != ORIGIN_OPERATOR:
            _fail(f"{name} is {status!r} but origin is {origin!r}")
        if x is None or y is None:
            _fail(f"{name} is {status!r} and requires coordinates")
        x, y = _finite(x, f"{name}.x"), _finite(y, f"{name}.y")
        if seed_frame is None or frame != seed_frame:
            _fail(f"{name} is an operator anchor at frame {frame}, but this target is "
                  f"seeded at "
                  + ("no frame at all" if seed_frame is None else str(seed_frame))
                  + ": a runner may not place operator anchors of its own (a carried "
                    "correction or a seed taken from the reference labels after a loss "
                    "would be evidence the benchmark must not manufacture)")
    elif status == STATUS_INTERPOLATED:
        if origin != ORIGIN_DERIVED:
            _fail(f"{name} is {status!r} but origin is {origin!r}: a derived point "
                  f"may never be presented as an observation")
        if x is None or y is None:
            _fail(f"{name} is {status!r} and requires coordinates")
        x, y = _finite(x, f"{name}.x"), _finite(y, f"{name}.y")
    else:  # missing / lost
        if origin != ORIGIN_DERIVED:
            _fail(f"{name} is {status!r} but origin is {origin!r}")
        if x is not None or y is not None or score is not None:
            _fail(f"{name} is {status!r} and must carry null coordinates and no score")
    return {"source_frame": frame, "x": x, "y": y, "status": status, "origin": origin,
            "method": method, "score": score}


def _prediction_seed(value: Any, name: str, reference_seed: Optional[Mapping[str, Any]]) -> None:
    """The seed a runner used must be the seed the reference declares.

    This is the only reference coordinate a runner is allowed to read, so it is
    pinned exactly: same frame, same pixel. A seed of its own making makes the run a
    different measurement and is refused rather than scored.
    """
    if reference_seed is None:
        if value is not None:
            _fail(f"{name} is not null, but the reference declares no seed for this "
                  f"target: a runner may not invent one")
        return
    if value is None:
        _fail(f"{name} is null, but the reference seeds this target at frame "
              f"{reference_seed['frame']}")
    mapping = _mapping(value, name)
    _fields(mapping, _PREDICTION_SEED_REQUIRED, ("note",), name)
    frame = _integer(mapping["frame"], f"{name}.frame")
    x = _finite(mapping["x"], f"{name}.x")
    y = _finite(mapping["y"], f"{name}.y")
    if frame != reference_seed["frame"]:
        _fail(f"{name}.frame {frame} is not the reference seed frame "
              f"{reference_seed['frame']}")
    for field, actual in (("x", x), ("y", y)):
        if abs(actual - float(reference_seed[field])) > 1e-6:
            _fail(f"{name}.{field} {actual} is not the reference seed "
                  f"{reference_seed[field]}: this prediction was seeded elsewhere")


def validate_predictions(payload: Mapping[str, Any], dataset: Mapping[str, Any]) -> dict:
    """Validate a ``fairwayos-target-predictions.v1`` payload against a dataset.

    The binding is by hash: every prediction clip must name a dataset clip and carry
    that clip's frozen ``sha256``. Every declared target must be reported and must
    echo the *same seed* the reference declares -- a runner that seeded a target
    somewhere else produced a different measurement, so its numbers are not
    comparable. A point on a cut frame, a point outside the window, a duplicate
    frame, or an automatic observation at the seed frame is refused.
    """
    mapping = _mapping(payload, "predictions")
    _fields(mapping, _PREDICTION_TOP, (), "predictions")
    if mapping["schema_version"] != PREDICTIONS_SCHEMA:
        _fail(f"predictions.schema_version must be {PREDICTIONS_SCHEMA!r}, got "
              f"{mapping['schema_version']!r}")
    variant = _text(mapping["variant"], "predictions.variant")
    if not _VARIANT_RE.match(variant):
        _fail("predictions.variant must match ^[a-z0-9][a-z0-9_.-]{0,63}$")
    clips = []
    seen_ids: set = set()
    for index, raw in enumerate(_sequence(mapping["clips"], "predictions.clips")):
        name = f"predictions.clips[{index}]"
        block = _mapping(raw, name)
        _fields(block, _PREDICTION_CLIP, (), name)
        clip_id = _text(block["id"], f"{name}.id")
        if clip_id in seen_ids:
            _fail(f"{name}.id {clip_id!r} is repeated")
        seen_ids.add(clip_id)
        clip = dataset["clip_index"].get(clip_id)
        if clip is None:
            _fail(f"{name}.id {clip_id!r} is not a clip of this dataset")
        declared = _sha(block["sha256"], f"{name}.sha256")
        if declared != clip["sha256"]:
            _fail(f"{name}.sha256 does not match the dataset's frozen hash for "
                  f"{clip_id!r}: predictions and reference are not the same clip")
        targets_raw = _mapping(block["targets"], f"{name}.targets")
        missing = sorted(set(clip["targets"]) - set(targets_raw))
        extra = sorted(set(targets_raw) - set(clip["targets"]))
        if missing:
            _fail(f"{name}.targets omits {missing[0]!r}: a target the reference "
                  f"declares must be reported, present or empty")
        if extra:
            _fail(f"{name}.targets carries {extra[0]!r}, which the reference does not "
                  f"declare")
        targets: dict = {}
        for target_name in sorted(targets_raw):
            block_in = _mapping(targets_raw[target_name], f"{name}.targets.{target_name}")
            _fields(block_in, _PREDICTION_TARGET_REQUIRED, _PREDICTION_TARGET_OPTIONAL,
                    f"{name}.targets.{target_name}")
            reference_seed = clip["targets"][target_name]["seed"]
            seed_frame = reference_seed["frame"] if reference_seed else None
            _prediction_seed(block_in["seed"], f"{name}.targets.{target_name}.seed",
                             reference_seed)
            # A prediction may declare the options it actually ran with (a variant's
            # whole point). They are validated against the same option contract as the
            # reference and recorded, never used to change what is scored.
            declared_options = None
            if "options" in block_in:
                declared_options = _options(block_in["options"],
                                            f"{name}.targets.{target_name}.options",
                                            clip["window"])
            points = []
            seen_frames: set = set()
            for point_index, raw_point in enumerate(
                    _sequence(block_in["points"], f"{name}.targets.{target_name}.points")):
                point = _prediction_point(
                    raw_point, f"{name}.targets.{target_name}.points[{point_index}]",
                    clip["window"], clip["cut_frames"], seed_frame, clip["width"],
                    clip["height"])
                if point["source_frame"] in seen_frames:
                    _fail(f"{name}.targets.{target_name} repeats frame "
                          f"{point['source_frame']}")
                seen_frames.add(point["source_frame"])
                points.append(point)
            points.sort(key=lambda item: item["source_frame"])
            targets[target_name] = {"points": points,
                                    "options": declared_options,
                                    "point_index": {p["source_frame"]: p for p in points}}
        clips.append({"id": clip_id, "sha256": declared, "targets": targets})
    if len(clips) != len(dataset["clips"]):
        _fail(f"predictions cover {len(clips)} clip(s) but the dataset has "
              f"{len(dataset['clips'])}: every declared clip must be reported")
    return {"schema_version": PREDICTIONS_SCHEMA, "variant": variant, "clips": clips}


# --------------------------------------------------------------------------
# structural suppression (window + cut frames), pure and testable


def suppress_structural(points: Sequence[Mapping[str, Any]], *, window: Mapping[str, int],
                        cut_frames: Sequence[int]) -> tuple[list[dict], dict]:
    """Drop points the structure of the source forbids, and say how many.

    Two rules, both from the brief: nothing is reported outside the declared window,
    and a cut is a reset. A point whose ``source_frame`` is a cut frame is dropped,
    and an interpolated point whose two bracketing observations straddle a cut frame
    is dropped too -- that point would bridge a reset the tracker was never entitled
    to bridge. Reconsecutive observations are unaffected.
    """
    start, end = int(window["start_frame"]), int(window["end_frame"])
    cuts = sorted(int(frame) for frame in cut_frames)
    kept: list[dict] = []
    dropped_outside = 0
    dropped_cut = 0
    dropped_bridged = 0
    ordered = sorted(points, key=lambda item: int(item["source_frame"]))
    previous_observed: Optional[int] = None
    for point in ordered:
        frame = int(point["source_frame"])
        if frame < start or frame > end:
            dropped_outside += 1
            continue
        if frame in cuts:
            dropped_cut += 1
            continue
        if point["status"] in AUTOMATIC_STATUSES or point["status"] == STATUS_MANUAL \
                or point["status"] in (STATUS_MISSING, STATUS_LOST):
            kept.append(dict(point))
            if point["status"] in AUTOMATIC_STATUSES or point["status"] == STATUS_MANUAL:
                previous_observed = frame
            continue
        # interpolated: is it bracketed across a cut?
        bridging = False
        if previous_observed is not None and cuts:
            bridging = any(previous_observed < cut < frame for cut in cuts)
        if not bridging and cuts:
            # ...and is the closing observation on the far side of a cut?
            later = [int(item["source_frame"]) for item in ordered
                     if int(item["source_frame"]) > frame
                     and (item["status"] in AUTOMATIC_STATUSES
                          or item["status"] == STATUS_MANUAL)]
            if later and any(frame < cut < min(later) for cut in cuts):
                bridging = True
        if bridging:
            dropped_bridged += 1
            continue
        kept.append(dict(point))
    return kept, {"dropped_outside_window": dropped_outside, "dropped_at_cut_frames": dropped_cut,
                  "dropped_bridging_a_cut": dropped_bridged,
                  "kept": len(kept), "window": [start, end], "cut_frames": cuts}


# --------------------------------------------------------------------------
# evaluation


def _percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    """Nearest-rank percentile: deterministic, no interpolation, no numpy."""
    ordered = sorted(values)
    if not ordered:
        return None
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return round(ordered[index], 3)


def _error_summary(errors: Sequence[float]) -> dict:
    ordered = sorted(errors)
    if not ordered:
        return {"count": 0, "median_px": None, "p90_px": None, "max_px": None}
    middle = len(ordered) // 2
    median = (ordered[middle] if len(ordered) % 2
              else (ordered[middle - 1] + ordered[middle]) / 2.0)
    return {"count": len(ordered), "median_px": round(median, 3),
            "p90_px": _percentile(ordered, 0.9), "max_px": round(ordered[-1], 3)}


def _consecutive_runs(frames: Sequence[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for frame in sorted(frames):
        if runs and frame == runs[-1][-1] + 1:
            runs[-1].append(frame)
        else:
            runs.append([frame])
    return runs


def _longest_missing_run(scored_frames: Sequence[int],
                         automatic_frames: Sequence[int]) -> dict:
    """The longest run of consecutive labelled visible frames with no automatic
    observation. Sparse labels (no two consecutive) return ``None`` with the reason
    instead of a fabricated number."""
    runs = _consecutive_runs(scored_frames)
    usable = [run for run in runs if len(run) >= 2]
    basis = {"visible_labelled_frames": len(scored_frames),
             "consecutive_runs": len(runs),
             "longest_consecutive_label_run": max((len(run) for run in runs), default=0),
             "runs_of_two_or_more": len(usable)}
    if not usable:
        return {"frames": None, "basis": basis,
                "reason": ("no two labelled visible frames are consecutive: the "
                           "reference is sparse, so a missing run cannot be measured "
                           "rather than measured as zero")}
    automatic = set(automatic_frames)
    longest = 0
    best_run: list[int] = []
    for run in usable:
        current: list[int] = []
        for frame in run:
            if frame in automatic:
                current = []
                continue
            current = current + [frame]
            if len(current) > longest:
                longest, best_run = len(current), list(current)
    return {"frames": longest, "basis": basis,
            "run_frames": best_run,
            "reason": ("longest consecutive run of labelled visible frames with no "
                       "automatic observation")}


def _recovery(labels: Sequence[Mapping[str, Any]], automatic: Mapping[int, Mapping],
              correct_frames: Mapping[int, bool],
              seed_frame: Optional[int] = None) -> dict:
    """Recovery across occlusion -> visible transitions.

    Three rules, each of which has been a way for this metric to lie:

    * the labels must be explicitly consecutive (a dense run of visible labels
      starting at the frame right after the occlusion). A sparse reference returns
      ``None`` with a reason instead of a number;
    * the *latency* is measured from the FIRST VISIBLE frame after the occlusion
      (``latency = recovered_frame - first_visible_frame``), so re-acquiring on that
      first frame is 0. It is not measured from the last occluded frame, which would
      silently add a frame to every case;
    * a transition at or before the target's seed frame is not a recovery at all:
      the tracker is acquiring for the first time there (or the track does not exist
      yet), so the case is excluded with an explicit reason rather than counted. The
      report states the excluded cases; it does not quietly drop them.
    """
    by_frame = {int(label["frame"]): label for label in labels}
    cases: list[dict] = []
    latencies: list[int] = []
    for frame in sorted(by_frame):
        first = by_frame.get(frame)
        following = by_frame.get(frame + 1)
        if first is None or following is None:
            continue
        if first["state"] not in NEGATIVE_STATES or following["state"] != "visible":
            continue
        dense_end = frame + 1
        while True:
            nxt = by_frame.get(dense_end + 1)
            if nxt is None or nxt["state"] != "visible":
                break
            dense_end += 1
        dense = by_frame.get(frame + 2) is not None and by_frame[frame + 2]["state"] == "visible"
        case = {"occlusion_frame": frame, "occlusion_state": first["state"],
                "first_visible_frame": frame + 1, "visible_run": [frame + 1, dense_end],
                "recovered_frame": None, "latency_frames": None, "evaluated": dense}
        if seed_frame is None or frame <= seed_frame:
            # Initial acquisition, not a recovery. The seed is where the operator's
            # own reference point sits; before it there is no automatic track, and at
            # it the point is the operator anchor itself.
            case["evaluated"] = False
            case["excluded_before_acquisition"] = True
            case["reason"] = (
                "this occlusion-to-visible transition lies at or before the target's "
                "seed frame "
                + ("(the target has no seed at all)" if seed_frame is None
                   else f"({seed_frame})")
                + ": the tracker is acquiring the target for the first time there, so "
                  "this is not an automatic recovery and is excluded, not counted")
            cases.append(case)
            continue
        if not dense:
            case["reason"] = ("the labelled visible frames after the occlusion are not "
                              "consecutive (sparse reference): a recovery latency "
                              "cannot be measured here")
            cases.append(case)
            continue
        recovered = next((f for f in range(frame + 1, dense_end + 1)
                          if correct_frames.get(f)), None)
        if recovered is None:
            case["reason"] = ("no correct automatic observation inside the consecutive "
                              "labelled run after the occlusion")
        else:
            case["recovered_frame"] = recovered
            case["latency_frames"] = recovered - case["first_visible_frame"]
            latencies.append(case["latency_frames"])
        cases.append(case)
    evaluated = [case for case in cases if case["evaluated"]]
    excluded = [case for case in cases if case.get("excluded_before_acquisition")]
    return {
        "occlusion_to_visible_transitions": len(cases),
        "evaluated_transitions": len(evaluated),
        "excluded_before_acquisition": len(excluded),
        "recovered_within_labelled_run": sum(1 for case in evaluated
                                             if case["recovered_frame"] is not None),
        "latency_definition": ("recovered_frame - first_visible_frame, i.e. frames to "
                               "re-acquire counted from the first visible frame after "
                               "the occlusion (0 = re-acquired immediately)"),
        "median_latency_frames": _error_summary(latencies)["median_px"],
        "mean_latency_frames": (round(sum(latencies) / len(latencies), 3)
                                if latencies else None),
        "max_latency_frames": (max(latencies) if latencies else None),
        "latencies_frames": sorted(latencies),
        "cases": cases,
        "null_reason": None if evaluated else
        ("no occlusion -> visible transition after this target's acquisition is "
         "labelled with consecutive visible frames, so recovery latency is not "
         "measured (the reference is sparse) and is reported as null rather than "
         "assumed"),
    }


def score_target(clip: Mapping[str, Any], target_name: str,
                 points: Sequence[Mapping[str, Any]],
                 suppression: Optional[Mapping[str, Any]] = None,
                 effective_options: Optional[Mapping[str, Any]] = None,
                 options_source: str = "reference") -> dict:
    """Score one target of one clip. Only the reference's own visible labels count.

    ``effective_options`` is what the run actually used (the prediction's own
    declaration when it made one, otherwise the reference's options): the report
    records it, because two variants of the same tracker are only comparable when
    their effective options are written down.
    """
    block = clip["targets"][target_name]
    seed = block["seed"]
    seed_frame = seed["frame"] if seed else None
    cuts = set(clip["cut_frames"])
    by_frame = {int(point["source_frame"]): point for point in points}
    automatic = {frame: point for frame, point in by_frame.items()
                 if point["status"] in AUTOMATIC_STATUSES}
    derived = {frame for frame, point in by_frame.items()
               if point["status"] == STATUS_INTERPOLATED}
    manual = {frame for frame, point in by_frame.items()
              if point["status"] == STATUS_MANUAL}

    labels = block["labels"]
    window_start = int(clip["window"]["start_frame"])
    window_end = int(clip["window"]["end_frame"])
    # Structural partitions, in this order. A label is never dropped: it is either
    # scored or it lands in one named bucket that says why it was not.
    in_window = [label for label in labels
                 if window_start <= int(label["frame"]) <= window_end]
    out_of_window = [label for label in labels
                     if not window_start <= int(label["frame"]) <= window_end]
    on_cut = [label for label in in_window if int(label["frame"]) in cuts]
    scorable = [label for label in in_window if int(label["frame"]) not in cuts]

    # Seeded-tracker scope. A seeded target is a SINGLE FORWARD LEG: nothing is
    # measured before the seed, so a label at or before the seed frame is
    # pre-initialization, and a target with no seed never has a track at all. Both
    # are excluded from the accuracy denominator and both are RETAINED and counted,
    # because a label the reference really wrote must not quietly disappear from the
    # record just because it cannot be scored.
    if seed_frame is None:
        scored_labels: list = []
        preseed_labels: list = list(scorable)
        denominator_reason = (
            "this target has no seed, so it is unseeded: no automatic track exists, "
            "no accuracy denominator is formed, and nothing can be claimed about "
            "initial acquisition or re-acquisition here")
    else:
        scored_labels = [label for label in scorable if int(label["frame"]) > seed_frame]
        preseed_labels = [label for label in scorable if int(label["frame"]) <= seed_frame]
        denominator_reason = (
            f"the denominator is exactly the frames this target's reference labelled "
            f"visible INSIDE the clip window and after its seed frame {seed_frame}; "
            f"labels at or before the seed are pre-initialization on a single forward "
            f"leg, labels outside the window (or on a structural cut frame) are "
            f"retained and counted separately, and none of them is scored")
    visible = [label for label in scored_labels if label["state"] == "visible"]
    uncertain = [label for label in scored_labels if label["state"] == "uncertain"]
    negatives = [label for label in scored_labels if label["state"] in NEGATIVE_STATES]

    def _bucket(entries: Sequence[Mapping[str, Any]]) -> dict:
        return {"total": len(entries),
                "visible": sum(1 for item in entries if item["state"] == "visible"),
                "occluded": sum(1 for item in entries if item["state"] == "occluded"),
                "out_of_view": sum(1 for item in entries if item["state"] == "out_of_view"),
                "uncertain": sum(1 for item in entries if item["state"] == "uncertain"),
                "frames": sorted(int(item["frame"]) for item in entries)}

    preseed_record = _bucket(preseed_labels)
    preseed_record["rule"] = (
        "labels at or before the seed frame (including the seed frame itself) are "
        "pre-initialization for a single forward leg: retained in this record, never "
        "scored as misses")
    out_of_window_record = _bucket(out_of_window)
    out_of_window_record["rule"] = (
        "labels outside the clip window are retained and counted here but excluded "
        "from every tracker-quality denominator: the window is structural policy, so "
        "an outside-window negative is 0 by construction, not detector precision")
    on_cut_record = _bucket(on_cut)
    on_cut_record["rule"] = (
        "labels on a structural cut frame are retained and counted here, and excluded "
        "from scoring: the source was spliced there")
    unseeded_record = ({**_bucket(scorable),
                        "rule": ("unseeded target: the labels are retained and counted "
                                 "here, but no denominator, recall, error, miss or "
                                 "negative-FP metric exists for this target")}
                       if seed_frame is None else None)

    correct = 0
    incorrect = 0
    misses = 0
    miss_with_derived = 0
    miss_with_manual = 0
    errors: list[float] = []
    per_frame: list[dict] = []
    correct_frames: dict = {}
    for label in visible:
        frame = int(label["frame"])
        point = automatic.get(frame)
        record = {"frame": frame, "tolerance_px": label["tolerance_px"],
                  "note": label["note"], "reviewer": label["reviewer"]}
        if point is None:
            misses += 1
            record["outcome"] = "miss"
            if frame in derived:
                miss_with_derived += 1
                record["derived_point_present"] = True
            if frame in manual:
                miss_with_manual += 1
                record["operator_anchor_present"] = True
            correct_frames[frame] = False
        else:
            error = math.hypot(float(point["x"]) - float(label["x"]),
                               float(point["y"]) - float(label["y"]))
            errors.append(error)
            within = error <= float(label["tolerance_px"])
            correct_frames[frame] = within
            if within:
                correct += 1
                record["outcome"] = "correct"
            else:
                incorrect += 1
                record["outcome"] = "incorrect"
            record["error_px"] = round(error, 3)
        per_frame.append(record)

    # Retained-but-unscored labels keep their own rows: a reader must be able to see
    # every label the reference wrote, including the ones this scope cannot score.
    unscopped_outcome = ("unseeded_not_scored" if seed_frame is None
                         else "preseed_not_scored")
    for label in preseed_labels:
        frame = int(label["frame"])
        per_frame.append({"frame": frame, "state": label["state"],
                          "outcome": unscopped_outcome,
                          "reason": ("at or before the seed frame" if seed_frame is not None
                                     else "the target is unseeded"),
                          "note": label["note"], "reviewer": label["reviewer"]})
    for label in out_of_window:
        per_frame.append({"frame": int(label["frame"]), "state": label["state"],
                          "outcome": "out_of_window_not_scored",
                          "reason": (f"outside the clip window "
                                     f"[{window_start}, {window_end}]"),
                          "note": label["note"], "reviewer": label["reviewer"]})
    for label in on_cut:
        per_frame.append({"frame": int(label["frame"]), "state": label["state"],
                          "outcome": "on_cut_frame_not_scored",
                          "reason": "the source was spliced at this structural frame",
                          "note": label["note"], "reviewer": label["reviewer"]})
    for label in scored_labels:
        if label["state"] == "visible":
            continue
        per_frame.append({"frame": int(label["frame"]), "state": label["state"],
                          "outcome": ("negative" if label["state"] in NEGATIVE_STATES
                                      else "uncertain_excluded"),
                          "auto_prediction_present": int(label["frame"]) in automatic,
                          "note": label["note"], "reviewer": label["reviewer"]})
    per_frame.sort(key=lambda item: item["frame"])

    negative_fp_frames = sorted(int(label["frame"]) for label in negatives
                                if int(label["frame"]) in automatic)
    negatives_total = len(negatives)
    automatic_emitted = len(automatic)
    automatic_in_window = clip["window"]["end_frame"] - clip["window"]["start_frame"] + 1
    unlabelled = automatic_in_window - len(in_window)

    return {
        "target": target_name,
        "seed": seed,
        "seed_frame_excluded": seed_frame is not None,
        "seed_frame": seed_frame,
        "options": block["options"],
        "effective_options": dict(effective_options or block["options"]),
        "effective_options_source": options_source,
        "evaluated_visible_frames": len(visible),
        "correct_within_tolerance": correct,
        "incorrect_visible_count": incorrect,
        "miss_count": misses,
        "recall_within_tolerance": (round(correct / len(visible), 4) if visible else None),
        "recall_denominator_note": denominator_reason,
        "preseed_labels": preseed_record,
        "preseed_visible_count": preseed_record["visible"],
        "unseeded_labels": unseeded_record,
        "unseeded_visible_count": (unseeded_record["visible"]
                                   if unseeded_record else 0),
        "out_of_window_labels": out_of_window_record,
        "out_of_window_labels_count": out_of_window_record["total"],
        "on_cut_frame_labels": on_cut_record,
        "on_cut_frame_labels_count": on_cut_record["total"],
        "errors_px": [round(value, 3) for value in sorted(errors)],
        "error_summary": _error_summary(errors),
        "negative_labeled_frames": negatives_total,
        "negative_fp_count": len(negative_fp_frames),
        "negative_fp_rate": (round(len(negative_fp_frames) / negatives_total, 4)
                             if negatives_total else None),
        "negative_fp_frames": negative_fp_frames,
        "negative_states_used": list(NEGATIVE_STATES),
        "uncertain_labels_excluded": len(uncertain),
        "unlabelled_window_frames_absent_from_denominator": unlabelled,
        "longest_missing_run_among_consecutive_visible_labels": _longest_missing_run(
            [int(label["frame"]) for label in visible], list(automatic)),
        "recovery": _recovery(scorable, automatic, correct_frames, seed_frame),
        "accounting": {
            "automatic_points_emitted": automatic_emitted,
            "automatic_points_at_labelled_frames": len(errors),
            "derived_points_emitted": len(derived),
            "derived_points_at_visible_labels": miss_with_derived,
            "missing_or_lost_points_emitted": sum(
                1 for point in by_frame.values()
                if point["status"] in (STATUS_MISSING, STATUS_LOST)),
            "operator_anchor_points_emitted": len(manual),
            "operator_anchor_points_at_visible_labels": miss_with_manual,
            "seed_point_emitted": seed_frame is not None and seed_frame in by_frame,
            "structural_suppression": dict(suppression or {}),
            "score_used_in_any_metric": False,
        },
        "per_frame": per_frame,
        "limitations": _target_limitations(seed, visible, negatives, uncertain, errors,
                                          automatic_emitted, preseed_record,
                                          unseeded_record, out_of_window_record,
                                          on_cut_record),
    }


def _target_limitations(seed, visible, negatives, uncertain, errors,
                        automatic_emitted, preseed=None, unseeded=None,
                        out_of_window=None, on_cut=None) -> list:
    out: list = []
    if seed is None:
        out.append("This target has no seed, so it is unseeded: there is no automatic "
                   "track, no accuracy denominator, and nothing can be claimed about "
                   "initial acquisition or re-acquisition. Its labels are retained and "
                   f"counted separately ({unseeded['total'] if unseeded else 0} label(s)).")
    if preseed and preseed["total"]:
        out.append(f"{preseed['total']} label(s) at or before the seed frame "
                   f"({preseed['visible']} visible) are pre-initialization on a single "
                   f"forward leg: retained in the record, excluded from the "
                   f"denominator, never scored as misses.")
    if out_of_window and out_of_window["total"]:
        out.append(f"{out_of_window['total']} label(s) lie OUTSIDE the clip window "
                   f"({out_of_window['visible']} visible, {out_of_window['occluded'] + out_of_window['out_of_view']} "
                   f"negative): retained and counted, excluded from every "
                   f"tracker-quality denominator, because an outside-window prediction "
                   f"is forbidden by structural policy and would make the negative "
                   f"rate look better than it is.")
    if on_cut and on_cut["total"]:
        out.append(f"{on_cut['total']} label(s) sit on a structural cut frame: retained "
                   f"and counted, excluded from scoring.")
    if not visible:
        out.append("The reference labels no visible frame for this target, so there "
                   "is no denominator: recall, error and miss counts are null, not "
                   "zero.")
    if not negatives:
        out.append("The reference labels no occluded/out_of_view frame for this "
                   "target, so the negative false-positive rate is null: absence of "
                   "a negative label is not evidence that the target was never "
                   "predicted where it was absent.")
    if uncertain:
        out.append(f"{len(uncertain)} uncertain label(s) are excluded from scoring "
                   f"entirely: neither positive nor negative.")
    if not errors:
        out.append("No automatic observation landed on a labelled frame, so no error "
                   "distribution exists.")
    out.append(f"{automatic_emitted} automatic point(s) were emitted in the window; "
               f"only those landing on a labelled frame are scored, and the tracker's "
               f"own score is used in no metric.")
    return out


_POOL_KEYS = ("evaluated_visible_frames", "correct_within_tolerance",
              "incorrect_visible_count", "miss_count", "negative_labeled_frames",
              "negative_fp_count", "uncertain_labels_excluded",
              "unlabelled_window_frames_absent_from_denominator",
              "preseed_visible_count", "unseeded_visible_count",
              "out_of_window_labels_count", "on_cut_frame_labels_count")


def pool_target(entries: Sequence[Mapping[str, Any]], *, scope: str) -> dict:
    """Pool per-(clip,target) scores. Rates are recomputed from the pooled counts so
    a clip with a bigger denominator carries its own weight, and no rate is an
    average of rates."""
    totals = {key: 0 for key in _POOL_KEYS}
    errors: list[float] = []
    automatic = 0
    derived = 0
    anchors = 0
    absent_points = 0
    for entry in entries:
        for key in _POOL_KEYS:
            totals[key] += int(entry[key])
        errors.extend(float(value) for value in entry["errors_px"])
        accounting = entry["accounting"]
        automatic += int(accounting["automatic_points_emitted"])
        derived += int(accounting["derived_points_emitted"])
        anchors += int(accounting["operator_anchor_points_emitted"])
        absent_points += int(accounting["missing_or_lost_points_emitted"])
    evaluated = totals["evaluated_visible_frames"]
    negatives = totals["negative_labeled_frames"]
    longest = [entry["longest_missing_run_among_consecutive_visible_labels"]["frames"]
               for entry in entries]
    measured_longest = [value for value in longest if value is not None]
    recovery_cases = [case for entry in entries for case in entry["recovery"]["cases"]]
    evaluated_cases = [case for case in recovery_cases if case["evaluated"]]
    latencies = [case["latency_frames"] for case in evaluated_cases
                 if case["latency_frames"] is not None]
    return {
        "scope": scope,
        "contributing_entries": len(entries),
        "evaluated_visible_frames": evaluated,
        "correct_within_tolerance": totals["correct_within_tolerance"],
        "incorrect_visible_count": totals["incorrect_visible_count"],
        "miss_count": totals["miss_count"],
        "recall_within_tolerance": (round(totals["correct_within_tolerance"] / evaluated, 4)
                                    if evaluated else None),
        "error_summary": _error_summary(errors),
        "negative_labeled_frames": negatives,
        "negative_fp_count": totals["negative_fp_count"],
        "negative_fp_rate": (round(totals["negative_fp_count"] / negatives, 4)
                             if negatives else None),
        "uncertain_labels_excluded": totals["uncertain_labels_excluded"],
        "unlabelled_window_frames_absent_from_denominator":
            totals["unlabelled_window_frames_absent_from_denominator"],
        "longest_missing_run_among_consecutive_visible_labels":
            (max(measured_longest) if measured_longest else None),
        "longest_missing_run_basis": ("max over the contributing (clip, target) pairs; "
                                      "None when none of them has two consecutive "
                                      "labelled visible frames"),
        "recovery": {
            "occlusion_to_visible_transitions": len(recovery_cases),
            "evaluated_transitions": len(evaluated_cases),
            "excluded_before_acquisition": sum(
                1 for case in recovery_cases if case.get("excluded_before_acquisition")),
            "recovered_within_labelled_run": len(latencies),
            "latency_definition": ("recovered_frame - first_visible_frame, per case; "
                                   "pooled from the contributing clips"),
            "median_latency_frames": _error_summary(latencies)["median_px"],
            "mean_latency_frames": (round(sum(latencies) / len(latencies), 3)
                                    if latencies else None),
            "max_latency_frames": (max(latencies) if latencies else None),
            "latencies_frames": sorted(latencies),
            "null_reason": None if evaluated_cases else
            ("no occlusion -> visible transition after the targets' acquisition is "
             "labelled with consecutive visible frames in the contributing clips, so "
             "recovery latency is not measured"),
        },
        "accounting": {"automatic_points_emitted": automatic,
                       "derived_points_emitted": derived,
                       "missing_or_lost_points_emitted": absent_points,
                       "operator_anchor_points_emitted": anchors,
                       "score_used_in_any_metric": False},
    }


def evaluate(dataset: Mapping[str, Any], predictions: Mapping[str, Any], *,
             suppressions: Optional[Mapping[str, Any]] = None,
             dataset_path: Optional[str] = None,
             dataset_sha256: Optional[str] = None,
             predictions_path: Optional[str] = None,
             predictions_sha256: Optional[str] = None,
             runtime: Optional[Mapping[str, Any]] = None) -> dict:
    """Evaluate validated predictions. Callers pass `validate_dataset` and
    `validate_predictions` output; the runner, the tests and the CLI all do."""
    clips_out: dict = {}
    split_clips: dict = {split: [] for split in SPLITS}
    tag_clips: dict = {}
    for clip in dataset["clips"]:
        prediction = next(item for item in predictions["clips"]
                          if item["id"] == clip["id"])
        targets_out: dict = {}
        for target_name in clip["target_names"]:
            prediction_target = prediction["targets"][target_name]
            points = prediction_target["points"]
            suppression = (suppressions or {}).get(clip["id"], {}).get(target_name)
            declared = prediction_target.get("options")
            targets_out[target_name] = score_target(
                clip, target_name, points, suppression,
                effective_options=declared,
                options_source="prediction" if declared else "reference")
        clips_out[clip["id"]] = {
            "id": clip["id"], "sha256": clip["sha256"], "split": clip["split"],
            "stroke_type": clip["stroke_type"], "tags": clip["tags"],
            "window": clip["window"], "cut_frames": clip["cut_frames"],
            "fps_rational": clip["fps_rational"], "targets": targets_out,
        }
        split_clips[clip["split"]].append(clip["id"])
        for tag in clip["tags"]:
            tag_clips.setdefault(tag, []).append(clip["id"])

    splits_out: dict = {}
    for split in SPLITS:
        ids = split_clips[split]
        splits_out[split] = {
            "clips": ids,
            "targets": {name: pool_target(
                [clips_out[clip_id]["targets"][name] for clip_id in ids
                 if name in clips_out[clip_id]["targets"]],
                scope=f"{split}/target={name}") for name in TARGET_NAMES},
            # A tag is reported inside a split only when at least one of the clips
            # carrying it is in that split: a tag that exists only in the other split
            # is not a development (or evaluation) result, and an empty table for it
            # would invite a reader to read the zero as a measurement.
            "tags": {tag: {name: pool_target(
                [clips_out[clip_id]["targets"][name] for clip_id in tag_clips[tag]
                 if clip_id in ids and name in clips_out[clip_id]["targets"]],
                scope=f"{split}/tag={tag}/target={name}") for name in TARGET_NAMES}
                for tag in sorted(tag_clips)
                if set(tag_clips[tag]) & set(ids)},
        }
    all_ids = [clip["id"] for clip in dataset["clips"]]
    return {
        "schema_version": REPORT_SCHEMA,
        "reference": {"path": dataset_path, "sha256": dataset_sha256,
                      "schema_version": dataset["schema_version"],
                      "clip_count": len(dataset["clips"]),
                      "provenance": dataset["provenance"],
                      "notes": dataset.get("notes", [])},
        "predictions": {"path": predictions_path, "sha256": predictions_sha256,
                        "variant": predictions["variant"],
                        "schema_version": predictions["schema_version"]},
        "runtime": dict(runtime or {}),
        "scope_notes": list(SCOPE_NOTES),
        "splits": splits_out,
        "pooled_all_splits": {
            "targets": {name: pool_target(
                [clips_out[clip_id]["targets"][name] for clip_id in all_ids
                 if name in clips_out[clip_id]["targets"]],
                scope=f"all-splits/target={name}") for name in TARGET_NAMES},
            "note": ("Development and evaluation are pooled here for bookkeeping "
                     "only. This is NOT a held-out number and must not be quoted as "
                     "one; the split is not a statistical holdout for these "
                     "already-showcased clips."),
        },
        "clips": clips_out,
    }


# --------------------------------------------------------------------------
# markdown report


def _rate(value: Optional[float]) -> str:
    return "null" if value is None else f"{value:.4f}"


def _count(value: Optional[int]) -> str:
    return "null" if value is None else str(value)


def render_markdown(results: Mapping[str, Any]) -> str:
    """Render the report. Every null says why it is null."""
    reference = results["reference"]
    predictions = results["predictions"]
    lines: list[str] = []
    lines.append("# Tiger target benchmark (research only, sparse AI-visual reference)")
    lines.append("")
    lines.append(f"- reference: `{reference['path']}` sha256 `{reference['sha256']}`")
    lines.append(f"- predictions: `{predictions['path']}` variant "
                 f"`{predictions['variant']}` sha256 `{predictions['sha256']}`")
    lines.append(f"- report schema: `{results['schema_version']}`; clips: "
                 f"{reference['clip_count']}")
    runtime = results.get("runtime") or {}
    if runtime:
        lines.append(f"- runtime: python {runtime.get('python')} / {runtime.get('interpreter')}"
                     + (f"; cv2 {runtime['cv2']}" if runtime.get("cv2") else ""))
        if runtime.get("tool_sha256"):
            lines.append(f"- runner: `{runtime.get('tool')}` sha256 "
                         f"`{runtime['tool_sha256']}`")
        if runtime.get("module_sha256"):
            lines.append(f"- evaluator module sha256 `{runtime['module_sha256']}`")
        for name, record in (runtime.get("algorithm_code_sha256") or {}).items():
            lines.append(f"- algorithm `{name}` sha256 `{record['sha256']}`")
        if runtime.get("tracker_scope"):
            lines.append(f"- tracker scope: {runtime['tracker_scope']}")
        profile = runtime.get("candidate_profile")
        if profile:
            lines.append(f"- experimental profile: `{profile['name']}`; applies to "
                         f"{', '.join(profile['targets'])}; constants sha256 "
                         f"`{profile['constants_sha256']}`")
            lines.append("- candidate target options below retain the classic reference "
                         "options for comparison; the candidate actually uses this "
                         "fixed profile: " + json.dumps(profile['constants'], sort_keys=True))
    lines.append(f"- provenance: {json.dumps(reference['provenance'], sort_keys=True)}")
    lines.append("")
    lines.append("## Scope of the seeded track")
    lines.append("")
    lines.append("- Each target is scored as a SINGLE FORWARD LEG from its seed: "
                 "nothing is measured before the seed, and backward / "
                 "both-direction tracks are not supported by this contract.")
    lines.append("- Labels at or before the seed frame are pre-initialization. They "
                 "are retained in the record (`preseed_labels`) and counted "
                 "separately, and they are excluded from the recall denominator: "
                 "they are never scored as misses.")
    lines.append("- A target with no seed is unseeded: it has no automatic track and "
                 "therefore no accuracy denominator at all. Its labels are retained "
                 "and counted separately (`unseeded_labels`).")
    lines.append("- Labels outside the window and labels on a structural cut frame "
                 "are retained and counted (`out_of_window_labels`, "
                 "`on_cut_frame_labels`), never scored: nothing is predicted "
                 "outside the window by policy, so such negatives measure policy, "
                 "not the tracker.")
    lines.append("")
    lines.append("## What these numbers are, and are not")
    lines.append("")
    for note in results["scope_notes"]:
        lines.append(f"- {note}")
    lines.append("")

    for split in SPLITS:
        block = results["splits"][split]
        lines.append(f"## {split.capitalize()} split")
        lines.append("")
        lines.append(f"Clips: {', '.join(block['clips']) if block['clips'] else 'none'}")
        lines.append("")
        lines.append("| target | evaluated visible frames | correct | recall | incorrect | "
                     "miss | error median/p90/max px | neg labels | negative FP | FP rate | "
                     "longest missing run |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for name, pooled in block["targets"].items():
            if not pooled["contributing_entries"]:
                lines.append(f"- {name}: no clip in this split declares this target "
                             f"(nothing to score; the row is omitted rather than "
                             f"printed as a zero).")
                continue
            summary = pooled["error_summary"]
            lines.append(
                f"| {name} | {pooled['evaluated_visible_frames']} | "
                f"{pooled['correct_within_tolerance']} | "
                f"{_rate(pooled['recall_within_tolerance'])} | "
                f"{pooled['incorrect_visible_count']} | {pooled['miss_count']} | "
                f"{_count(summary['median_px'])}/{_count(summary['p90_px'])}/"
                f"{_count(summary['max_px'])} | {pooled['negative_labeled_frames']} | "
                f"{pooled['negative_fp_count']} | {_rate(pooled['negative_fp_rate'])} | "
                f"{_count(pooled['longest_missing_run_among_consecutive_visible_labels'])} |")
        lines.append("")
        for name, pooled in block["targets"].items():
            if not pooled["contributing_entries"]:
                continue
            recovery = pooled["recovery"]
            lines.append(f"**{name}**: recovery across occlusion -> visible: "
                         f"{recovery['evaluated_transitions']} evaluated transition(s) of "
                         f"{recovery['occlusion_to_visible_transitions']}"
                         + (f" ({recovery['excluded_before_acquisition']} excluded as "
                            f"at/before acquisition)"
                            if recovery.get("excluded_before_acquisition") else "")
                         + f", recovered {recovery['recovered_within_labelled_run']}, "
                         f"latency median {_count(recovery['median_latency_frames'])} / "
                         f"mean {_count(recovery['mean_latency_frames'])} / "
                         f"max {_count(recovery['max_latency_frames'])} frame(s) from the "
                         f"first visible frame."
                         + (f" {recovery['null_reason']}" if recovery["null_reason"] else ""))
            lines.append("")
        if block["tags"]:
            lines.append("Per tag (pooled inside the split):")
            lines.append("")
            lines.append("| tag | target | evaluated | correct | recall | miss | negative FP rate |")
            lines.append("|---|---|---|---|---|---|---|")
            for tag, per_target in block["tags"].items():
                for name, pooled in per_target.items():
                    if not pooled["contributing_entries"]:
                        continue
                    lines.append(f"| {tag} | {name} | {pooled['evaluated_visible_frames']} | "
                                 f"{pooled['correct_within_tolerance']} | "
                                 f"{_rate(pooled['recall_within_tolerance'])} | "
                                 f"{pooled['miss_count']} | "
                                 f"{_rate(pooled['negative_fp_rate'])} |")
            lines.append("")

    lines.append("## Per clip")
    lines.append("")
    for clip_id, clip in results["clips"].items():
        lines.append(f"### {clip_id} ({clip['split']}, stroke `{clip['stroke_type']}`, "
                     f"tags: {', '.join(clip['tags']) or 'none'})")
        lines.append("")
        lines.append(f"sha256 `{clip['sha256']}`; window "
                     f"[{clip['window']['start_frame']}, {clip['window']['end_frame']}]; "
                     f"cut frames {clip['cut_frames'] or 'none'}; declared fps "
                     f"{clip['fps_rational']['numerator']}/{clip['fps_rational']['denominator']}.")
        lines.append("")
        for name, score in clip["targets"].items():
            summary = score["error_summary"]
            lines.append(f"- **{name}**: seed {'none' if not score['seed'] else str(score['seed']['frame'])} "
                         f"(excluded from scoring: {score['seed_frame_excluded']}); "
                         f"effective options ({score['effective_options_source']}): "
                         f"{json.dumps(score['effective_options'], sort_keys=True)}; "
                         f"evaluated {score['evaluated_visible_frames']}, correct "
                         f"{score['correct_within_tolerance']}, recall "
                         f"{_rate(score['recall_within_tolerance'])}, incorrect "
                         f"{score['incorrect_visible_count']}, miss {score['miss_count']}, "
                         f"error median/p90/max {_count(summary['median_px'])}/"
                         f"{_count(summary['p90_px'])}/{_count(summary['max_px'])} px, "
                         f"negatives {score['negative_labeled_frames']} with "
                         f"{score['negative_fp_count']} false positive(s) (rate "
                         f"{_rate(score['negative_fp_rate'])}), longest missing run "
                         f"{_count(score['longest_missing_run_among_consecutive_visible_labels']['frames'])}.")
            limitation = score["longest_missing_run_among_consecutive_visible_labels"]
            if limitation["frames"] is None:
                lines.append(f"  - longest missing run is null: {limitation['reason']}")
            if score["preseed_labels"]["total"]:
                lines.append(f"  - pre-seed labels retained and NOT scored: "
                             f"{score['preseed_labels']['total']} "
                             f"({score['preseed_labels']['visible']} visible, frames "
                             f"{score['preseed_labels']['frames']}).")
            if score["unseeded_labels"]:
                lines.append(f"  - unseeded target: {score['unseeded_labels']['total']} "
                             f"label(s) retained, no accuracy denominator.")
            if score["out_of_window_labels"]["total"]:
                lines.append(f"  - labels outside the window retained and NOT scored: "
                             f"{score['out_of_window_labels']['total']} "
                             f"({score['out_of_window_labels']['visible']} visible, "
                             f"{score['out_of_window_labels']['occluded'] + score['out_of_window_labels']['out_of_view']} negative).")
            if score["on_cut_frame_labels"]["total"]:
                lines.append(f"  - labels on a structural cut frame retained and NOT "
                             f"scored: {score['on_cut_frame_labels']['total']}.")
            lines.append(f"  - automatic points emitted {score['accounting']['automatic_points_emitted']}, "
                         f"derived {score['accounting']['derived_points_emitted']} "
                         f"(of which {score['accounting']['derived_points_at_visible_labels']} on a "
                         f"labelled visible frame, counted as a miss and never as evidence), "
                         f"operator anchors {score['accounting']['operator_anchor_points_emitted']}, "
                         f"structural suppression {score['accounting']['structural_suppression']}.")
            for note in score["limitations"]:
                lines.append(f"  - {note}")
        lines.append("")

    lines.append("## Pooled all splits (bookkeeping only, NOT held out)")
    lines.append("")
    lines.append(results["pooled_all_splits"]["note"])
    lines.append("")
    lines.append("| target | evaluated | correct | recall | miss | negative FP rate |")
    lines.append("|---|---|---|---|---|---|")
    for name, pooled in results["pooled_all_splits"]["targets"].items():
        lines.append(f"| {name} | {pooled['evaluated_visible_frames']} | "
                     f"{pooled['correct_within_tolerance']} | "
                     f"{_rate(pooled['recall_within_tolerance'])} | {pooled['miss_count']} | "
                     f"{_rate(pooled['negative_fp_rate'])} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def canonical_json(payload: Mapping[str, Any]) -> str:
    """The one serialisation used for outputs and for the results hash."""
    return json.dumps(payload, allow_nan=False, sort_keys=True, indent=2) + "\n"
