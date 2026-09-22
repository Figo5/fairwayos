"""Demo config: schema, strict validation, and the effective resolved config.

The config is the operator's contract and it is validated, never repaired. Every
rejection below exists because silently "fixing" the input would put a number or a
marker on the video that the operator did not ask for:

* non-finite or non-numeric values are refused (no ``NaN`` sneaking through);
* frame indices must be whole numbers — a fractional frame is refused, not rounded;
* nothing may point outside the source frame, the processed interval, or the
  source duration — an out-of-range keyframe is an error, never clipped;
* corrections must be unique per target frame and cannot collide with the seed;
* phase keyframes must be in chronological order;
* a variable-cadence source is refused unless the operator explicitly accepts it,
  because a nominal fps is not a measured action time.

Endpoint semantics (documented here because it is the one thing an operator cannot
guess): ``start_seconds`` is INCLUSIVE and ``end_seconds`` is EXCLUSIVE — the
processed interval is the half-open source range ``[start_seconds, end_seconds)``,
mapped to inclusive source frame indices
``start_frame = floor(start_seconds * fps)`` and
``end_frame = ceil(end_seconds * fps) - 1``. With ``end_seconds: null`` the
interval runs to the last frame of the source. ``fps`` here is the average frame
rate ffprobe reports; see ``demo.probe`` for why that is not an action time.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from . import DEMO_CONFIG_SCHEMA
from .errors import DemoConfigError
from .probe import SourceProbe

PHASE_ORDER = ("address", "takeaway", "top", "downswing", "impact",
               "follow_through", "finish")
#: Cards are produced for these four; the others are optional extras.
PRIMARY_PHASES = ("address", "top", "impact", "finish")
TARGET_NAMES = ("ball", "clubhead")

SOURCE_TIMING_DEFAULT_CAVEAT = (
    "unknown: the source may be a replay, a slow-motion capture or a re-timed copy, "
    "so container cadence is used only to place frames in the source timeline and is "
    "never treated as measured action time")

#: What the seconds in a phase/tempo record are measured from. Defined here because
#: both the resolved config and the phase records report it, and a single definition
#: is what stops the two from disagreeing about what a second means.
SECONDS_BASIS_TIMESTAMPS = ("source frame presentation timestamps (ffprobe, decoded "
                            "frame order)")
SECONDS_BASIS_CADENCE = "frames / container cadence"

DEFAULT_MAX_FRAMES = 1200
DEFAULT_RENDER_MAX_WIDTH = 1280

_TARGET_MODES = ("template", "manual")
_DIRECTIONS = ("forward", "backward", "both")
_PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium",
            "slow", "slower", "veryslow")


# ---------------------------------------------------------------- value helpers

def _finite(value: Any, path: str, *, minimum: Optional[float] = None,
            maximum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DemoConfigError(f"{path} must be a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise DemoConfigError(f"{path} must be finite, got {value!r}")
    if minimum is not None and out < minimum:
        raise DemoConfigError(f"{path} must be >= {minimum}, got {out}")
    if maximum is not None and out > maximum:
        raise DemoConfigError(f"{path} must be <= {maximum}, got {out}")
    return out


def _whole(value: Any, path: str, *, minimum: Optional[int] = None,
           maximum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DemoConfigError(f"{path} must be an integer, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise DemoConfigError(f"{path} must be finite, got {value!r}")
    if number != int(number):
        raise DemoConfigError(
            f"{path} must be a whole number; {value!r} is fractional and is not "
            f"rounded silently")
    out = int(number)
    if minimum is not None and out < minimum:
        raise DemoConfigError(f"{path} must be >= {minimum}, got {out}")
    if maximum is not None and out > maximum:
        raise DemoConfigError(f"{path} must be <= {maximum}, got {out}")
    return out


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise DemoConfigError(f"{path} must be true or false, got {value!r}")
    return value


def _text(value: Any, path: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise DemoConfigError(f"{path} must be a string, got {value!r}")
    if not allow_empty and not value.strip():
        raise DemoConfigError(f"{path} must not be empty")
    return value


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DemoConfigError(f"{path} must be a JSON object, got {type(value).__name__}")
    return value


def _named_blocks(block: Mapping[str, Any], path: str,
                  names: Sequence[str]) -> Dict[str, Any]:
    """The target-named sub-blocks of ``seeds``/``targets``/``corrections``.

    Underscore-prefixed keys are comments (they can explain a block), so they are
    skipped here exactly as they are in the unknown-field check.
    """
    out: Dict[str, Any] = {}
    for name, value in block.items():
        if str(name).startswith("_"):
            continue
        if name not in names:
            raise DemoConfigError(f"{path}.{name} is not one of {list(names)}")
        out[name] = value
    return out


def _unknown_keys(block: Mapping[str, Any], allowed: Sequence[str], path: str) -> List[str]:
    # A key beginning with "_" is a comment, not operator input: templates need a
    # place to explain themselves and JSON has no comment syntax. Everything else
    # is still refused when unknown.
    return [k for k in block.keys() if k not in allowed and not str(k).startswith("_")]


def _reject_unknown(block: Mapping[str, Any], allowed: Sequence[str], path: str) -> None:
    extra = _unknown_keys(block, allowed, path)
    if extra:
        raise DemoConfigError(
            f"{path} has unknown field(s) {sorted(extra)}; known fields are "
            f"{sorted(allowed)}. A misspelt field is refused rather than ignored, "
            f"because ignoring it would silently drop operator input.")


# ---------------------------------------------------------------- schema pieces

@dataclass(frozen=True)
class Point:
    frame: int
    x: float
    y: float
    note: str = ""

    def to_dict(self) -> dict:
        return {"frame": self.frame, "x": round(self.x, 3), "y": round(self.y, 3),
                "note": self.note}


@dataclass(frozen=True)
class TargetOptions:
    mode: str = "template"
    template_size: int = 31
    search_radius: int = 40
    min_score: float = 0.50
    max_jump_px: float = 60.0
    loss_frames: int = 3
    direction: str = "forward"
    lk_refine: bool = False
    template_refresh_frames: int = 0
    template_refresh_min_score: float = 0.70
    track_radius_frames: Optional[int] = None
    #: Operator-declared end of the target's scope: after this source frame the
    #: target is out of scope (leaves the frame, is gone, was never visible) and
    #: stays missing. It is not a loss and nothing is inferred past it.
    track_until_frame: Optional[int] = None
    #: Operator-declared end of AUTOMATIC matching. This one does NOT suppress
    #: operator anchors: past this frame the target is exactly the anchors the
    #: operator placed, which is how a flight that outruns the template search
    #: (100-220 px per frame) is reported without publishing a false match.
    automatic_until_frame: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class InterpolationOptions:
    enabled: bool = True
    max_gap_frames: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TimingOptions:
    source_cadence_caveat: str = SOURCE_TIMING_DEFAULT_CAVEAT
    nominal_fps_trusted: bool = False
    accept_variable_cadence: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RenderOptions:
    max_width: Optional[int] = DEFAULT_RENDER_MAX_WIDTH
    crf: int = 20
    preset: str = "medium"
    render: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DemoConfig:
    schema_version: str = DEMO_CONFIG_SCHEMA
    clip_path_hint: str = ""
    start_seconds: float = 0.0
    end_seconds: Optional[float] = None
    max_frames: int = DEFAULT_MAX_FRAMES
    roi: Optional[List[int]] = None
    seeds: Dict[str, Point] = field(default_factory=dict)
    keyframes: Dict[str, int] = field(default_factory=dict)
    targets: Dict[str, TargetOptions] = field(default_factory=dict)
    corrections: Dict[str, List[Point]] = field(default_factory=dict)
    interpolation: InterpolationOptions = InterpolationOptions()
    timing: TimingOptions = TimingOptions()
    render: RenderOptions = RenderOptions()
    handedness: str = "right"
    allow_approx_phases: bool = False
    notes: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def target(self, name: str) -> TargetOptions:
        return self.targets.get(name) or TargetOptions()

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "clip_path_hint": self.clip_path_hint,
            "interval": {"start_seconds": self.start_seconds,
                         "end_seconds": self.end_seconds,
                         "max_frames": self.max_frames},
            "roi": list(self.roi) if self.roi else None,
            "seeds": {k: v.to_dict() for k, v in sorted(self.seeds.items())},
            "keyframes": {k: self.keyframes[k] for k in sorted(self.keyframes)},
            "targets": {k: v.to_dict() for k, v in sorted(self.targets.items())},
            "corrections": {k: [p.to_dict() for p in v]
                            for k, v in sorted(self.corrections.items())},
            "interpolation": self.interpolation.to_dict(),
            "timing": self.timing.to_dict(),
            "render": self.render.to_dict(),
            "handedness": self.handedness,
            "allow_approx_phases": self.allow_approx_phases,
            "notes": self.notes,
        }


# ---------------------------------------------------------------- parsing

_TARGET_FIELDS = ("mode", "template_size", "search_radius", "min_score",
                  "max_jump_px", "loss_frames", "direction", "lk_refine",
                  "template_refresh_frames", "template_refresh_min_score",
                  "track_radius_frames", "track_until_frame",
                  "automatic_until_frame")
_TOP_FIELDS = ("schema_version", "clip_path_hint", "interval", "roi", "seeds",
               "keyframes", "targets", "corrections", "interpolation", "timing",
               "render", "handedness", "allow_approx_phases", "notes")


def parse_config(payload: Any) -> DemoConfig:
    """Turn a decoded JSON object into a DemoConfig. Unknown fields are refused."""
    data = _mapping(payload, "config")
    if not data:
        raise DemoConfigError("config is empty")
    _reject_unknown(data, _TOP_FIELDS, "config")
    version = data.get("schema_version", DEMO_CONFIG_SCHEMA)
    if version != DEMO_CONFIG_SCHEMA:
        raise DemoConfigError(
            f"config schema_version must be {DEMO_CONFIG_SCHEMA!r}, got {version!r}. "
            f"This is deliberately not the canonical analysis schema.")

    interval = _mapping(data.get("interval"), "interval")
    _reject_unknown(interval, ("start_seconds", "end_seconds", "max_frames"), "interval")
    start_seconds = _finite(interval.get("start_seconds", 0.0), "interval.start_seconds",
                            minimum=0.0)
    end_raw = interval.get("end_seconds", None)
    end_seconds = None if end_raw is None else _finite(end_raw, "interval.end_seconds",
                                                       minimum=0.0)
    max_frames = _whole(interval.get("max_frames", DEFAULT_MAX_FRAMES),
                        "interval.max_frames", minimum=1)

    roi = None
    if data.get("roi") is not None:
        raw_roi = data["roi"]
        if not isinstance(raw_roi, (list, tuple)) or len(raw_roi) != 4:
            raise DemoConfigError("roi must be [x, y, w, h] in source pixels")
        roi = [_whole(v, f"roi[{i}]", minimum=0) for i, v in enumerate(raw_roi)]
        if roi[2] <= 0 or roi[3] <= 0:
            raise DemoConfigError("roi width and height must be positive")

    seeds: Dict[str, Point] = {}
    for name, block in _named_blocks(_mapping(data.get("seeds"), "seeds"), "seeds",
                                     TARGET_NAMES).items():
        b = _mapping(block, f"seeds.{name}")
        _reject_unknown(b, ("frame", "x", "y", "note"), f"seeds.{name}")
        seeds[name] = Point(
            frame=_whole(b.get("frame"), f"seeds.{name}.frame", minimum=0),
            x=_finite(b.get("x"), f"seeds.{name}.x"),
            y=_finite(b.get("y"), f"seeds.{name}.y"),
            note=_text(b.get("note", ""), f"seeds.{name}.note"))

    keyframes: Dict[str, int] = {}
    for name, value in _mapping(data.get("keyframes"), "keyframes").items():
        if name not in PHASE_ORDER:
            raise DemoConfigError(
                f"keyframes.{name} is not a phase; known phases are {list(PHASE_ORDER)}")
        keyframes[name] = _whole(value, f"keyframes.{name}", minimum=0)

    targets: Dict[str, TargetOptions] = {}
    for name, block in _named_blocks(_mapping(data.get("targets"), "targets"),
                                     "targets", TARGET_NAMES).items():
        b = _mapping(block, f"targets.{name}")
        _reject_unknown(b, _TARGET_FIELDS, f"targets.{name}")
        mode = _text(b.get("mode", "template"), f"targets.{name}.mode")
        if mode not in _TARGET_MODES:
            raise DemoConfigError(f"targets.{name}.mode must be one of {list(_TARGET_MODES)}")
        direction = _text(b.get("direction", "forward"), f"targets.{name}.direction")
        if direction not in _DIRECTIONS:
            raise DemoConfigError(
                f"targets.{name}.direction must be one of {list(_DIRECTIONS)}")
        template_size = _whole(b.get("template_size", 31),
                               f"targets.{name}.template_size", minimum=5, maximum=401)
        if template_size % 2 == 0:
            raise DemoConfigError(
                f"targets.{name}.template_size must be odd so the patch centres on the "
                f"seed pixel, got {template_size}")
        track_radius = b.get("track_radius_frames")
        track_until = b.get("track_until_frame")
        automatic_until = b.get("automatic_until_frame")
        targets[name] = TargetOptions(
            mode=mode, template_size=template_size,
            search_radius=_whole(b.get("search_radius", 40),
                                 f"targets.{name}.search_radius", minimum=1, maximum=2000),
            min_score=_finite(b.get("min_score", 0.50), f"targets.{name}.min_score",
                              minimum=0.0, maximum=1.0),
            max_jump_px=_finite(b.get("max_jump_px", 60.0),
                                f"targets.{name}.max_jump_px", minimum=0.5),
            loss_frames=_whole(b.get("loss_frames", 3), f"targets.{name}.loss_frames",
                               minimum=1, maximum=1000),
            direction=direction,
            lk_refine=_boolean(b.get("lk_refine", False), f"targets.{name}.lk_refine"),
            template_refresh_frames=_whole(
                b.get("template_refresh_frames", 0),
                f"targets.{name}.template_refresh_frames", minimum=0),
            template_refresh_min_score=_finite(
                b.get("template_refresh_min_score", 0.70),
                f"targets.{name}.template_refresh_min_score", minimum=0.0, maximum=1.0),
            track_radius_frames=None if track_radius is None else _whole(
                track_radius, f"targets.{name}.track_radius_frames", minimum=0),
            track_until_frame=None if track_until is None else _whole(
                track_until, f"targets.{name}.track_until_frame", minimum=0),
            automatic_until_frame=None if automatic_until is None else _whole(
                automatic_until, f"targets.{name}.automatic_until_frame", minimum=0))

    corrections: Dict[str, List[Point]] = {}
    for name, items in _named_blocks(_mapping(data.get("corrections"), "corrections"),
                                     "corrections", TARGET_NAMES).items():
        if not isinstance(items, list):
            raise DemoConfigError(f"corrections.{name} must be a list of "
                                  f"{{frame, x, y}} objects")
        parsed: List[Point] = []
        for i, item in enumerate(items):
            b = _mapping(item, f"corrections.{name}[{i}]")
            _reject_unknown(b, ("frame", "x", "y", "note"), f"corrections.{name}[{i}]")
            parsed.append(Point(
                frame=_whole(b.get("frame"), f"corrections.{name}[{i}].frame", minimum=0),
                x=_finite(b.get("x"), f"corrections.{name}[{i}].x"),
                y=_finite(b.get("y"), f"corrections.{name}[{i}].y"),
                note=_text(b.get("note", ""), f"corrections.{name}[{i}].note")))
        corrections[name] = parsed

    interp_block = _mapping(data.get("interpolation"), "interpolation")
    _reject_unknown(interp_block, ("enabled", "max_gap_frames"), "interpolation")
    interpolation = InterpolationOptions(
        enabled=_boolean(interp_block.get("enabled", True), "interpolation.enabled"),
        max_gap_frames=_whole(interp_block.get("max_gap_frames", 2),
                              "interpolation.max_gap_frames", minimum=0, maximum=600))

    timing_block = _mapping(data.get("timing"), "timing")
    _reject_unknown(timing_block, ("source_cadence_caveat", "nominal_fps_trusted",
                                   "accept_variable_cadence"), "timing")
    timing = TimingOptions(
        source_cadence_caveat=_text(timing_block.get("source_cadence_caveat",
                                                     SOURCE_TIMING_DEFAULT_CAVEAT),
                                    "timing.source_cadence_caveat"),
        nominal_fps_trusted=_boolean(timing_block.get("nominal_fps_trusted", False),
                                     "timing.nominal_fps_trusted"),
        accept_variable_cadence=_boolean(
            timing_block.get("accept_variable_cadence", False),
            "timing.accept_variable_cadence"))

    render_block = _mapping(data.get("render"), "render")
    _reject_unknown(render_block, ("max_width", "crf", "preset", "render"), "render")
    max_width_raw = render_block.get("max_width", DEFAULT_RENDER_MAX_WIDTH)
    max_width = None if max_width_raw is None else _whole(
        max_width_raw, "render.max_width", minimum=64)
    preset = _text(render_block.get("preset", "medium"), "render.preset")
    if preset not in _PRESETS:
        raise DemoConfigError(f"render.preset must be one of {list(_PRESETS)}")
    render = RenderOptions(
        max_width=max_width,
        crf=_whole(render_block.get("crf", 20), "render.crf", minimum=0, maximum=51),
        preset=preset,
        render=_boolean(render_block.get("render", True), "render.render"))

    handedness = _text(data.get("handedness", "right"), "handedness")
    if handedness not in ("right", "left"):
        raise DemoConfigError("handedness must be 'right' or 'left'")

    return DemoConfig(
        schema_version=version,
        clip_path_hint=_text(data.get("clip_path_hint", ""), "clip_path_hint"),
        start_seconds=start_seconds, end_seconds=end_seconds, max_frames=max_frames,
        roi=roi, seeds=seeds, keyframes=keyframes, targets=targets,
        corrections=corrections, interpolation=interpolation, timing=timing,
        render=render, handedness=handedness,
        allow_approx_phases=_boolean(data.get("allow_approx_phases", False),
                                     "allow_approx_phases"),
        notes=_text(data.get("notes", ""), "notes"),
        raw=dict(data))


def load_config(path: str) -> DemoConfig:
    p = Path(path).expanduser()
    if not p.is_file():
        raise DemoConfigError(f"config file does not exist: {path}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DemoConfigError(f"config is not valid JSON: {exc}") from exc
    return parse_config(payload)


# ---------------------------------------------------------------- resolution

@dataclass
class ResolvedConfig:
    config: DemoConfig
    source: SourceProbe
    start_frame: int
    end_frame: int
    frame_count: int
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    #: Frame presentation timestamps for the processed frames, when the container
    #: provided them. Seconds in the report come from these, not from frames / fps.
    frame_times: Optional[List[float]] = None

    @property
    def seconds_basis(self) -> str:
        return (SECONDS_BASIS_TIMESTAMPS if self.frame_times
                else "frames / container cadence")

    @property
    def frames(self) -> range:
        return range(self.start_frame, self.end_frame + 1)

    def effective(self) -> dict:
        """The config as it was actually applied, in source frames."""
        cfg = self.config
        return {
            "source_frames": self.start_frame,
            "source_frame_range": [self.start_frame, self.end_frame],
            "processed_frame_count": self.frame_count,
            "start_seconds": cfg.start_seconds,
            "end_seconds": cfg.end_seconds,
            "max_frames": cfg.max_frames,
            "endpoint_semantics": ("start_seconds inclusive, end_seconds exclusive; "
                                   "start_frame = floor(start_seconds*fps), "
                                   "end_frame = ceil(end_seconds*fps) - 1"),
            "roi_xywh": list(cfg.roi) if cfg.roi else None,
            "seeds": {k: v.to_dict() for k, v in sorted(cfg.seeds.items())},
            "keyframes": {k: cfg.keyframes[k] for k in sorted(cfg.keyframes)},
            "targets": {k: v.to_dict() for k, v in sorted(cfg.targets.items())},
            "corrections": {k: [p.to_dict() for p in v]
                            for k, v in sorted(cfg.corrections.items())},
            "interpolation": cfg.interpolation.to_dict(),
            "timing": cfg.timing.to_dict(),
            "render": cfg.render.to_dict(),
            "handedness": cfg.handedness,
            "allow_approx_phases": cfg.allow_approx_phases,
            "warnings": list(self.warnings),
            "notes_from_config": cfg.notes,
            "raw_config": cfg.raw,
            "seconds_basis": self.seconds_basis,
            "source_timestamps": {
                "available": self.frame_times is not None,
                "frames": (self.source.timestamp_stats or {}).get("frames"),
                "first_pts": (self.frame_times or [None])[0],
                "last_pts": (self.frame_times or [None])[-1],
                "stats": self.source.timestamp_stats,
                "note": self.source.timestamp_note,
                "irregular": self.source.irregular_timestamps,
            },
        }


def _frame_for_seconds(seconds: float, fps: float, *, upper: bool) -> int:
    value = float(seconds) * float(fps)
    return int(math.ceil(value)) if upper else int(math.floor(value))


def validate_config(cfg: DemoConfig, source: SourceProbe) -> ResolvedConfig:
    """Bind the config to the source. Raises rather than repairing anything."""
    warnings: List[str] = []
    notes: List[str] = []

    if source.variable_cadence and not cfg.timing.accept_variable_cadence:
        raise DemoConfigError(
            f"the container cadence is not constant, so a frame time is not one "
            f"rate: {source.cadence_note} Refusing to run: set "
            f"timing.accept_variable_cadence to true to accept that every frame time "
            f"in this report is container-cadence only.")
    if source.irregular_timestamps and not cfg.timing.accept_variable_cadence:
        raise DemoConfigError(
            f"the decoded frame timestamps are not evenly spaced, so there is no one "
            f"frame time to convert frames into seconds: {source.timestamp_note}. "
            f"Refusing to run as if the cadence were constant; set "
            f"timing.accept_variable_cadence to true to accept frame-index-only "
            f"timing.")

    fps = source.fps
    if not fps or not math.isfinite(fps) or fps <= 0:
        raise DemoConfigError("the source reports no usable frame rate")

    # A config carried over from another clip runs silently otherwise: the hint is
    # the operator's own record of which source the numbers were measured on, so a
    # mismatch is worth saying out loud. It is a warning, not a refusal: the same
    # clip can legitimately be renamed or copied, and the source SHA-256 in the
    # provenance is the authoritative binding.
    hint = (cfg.clip_path_hint or "").strip()
    if hint and Path(hint).name != source.display_name:
        warnings.append(
            f"clip_path_hint is '{Path(hint).name}' but the source being analysed is "
            f"'{source.display_name}': this config was written for a different file "
            f"name. Check that its seeds, corrections and keyframes are frame numbers "
            f"of THIS clip before trusting the output.")

    if cfg.end_seconds is not None and cfg.end_seconds <= cfg.start_seconds:
        raise DemoConfigError(
            f"interval.end_seconds ({cfg.end_seconds}) must be greater than "
            f"interval.start_seconds ({cfg.start_seconds}); the interval is half-open "
            f"[start, end)")

    start_frame = _frame_for_seconds(cfg.start_seconds, fps, upper=False)
    if cfg.end_seconds is None:
        end_frame = (source.last_frame_index if source.last_frame_index is not None
                     else int(math.ceil(source.duration_seconds * fps)) - 1)
        if source.last_frame_index is None:
            notes.append("the container reports no frame count; the interval end is "
                         "duration x cadence")
    else:
        if cfg.end_seconds > source.duration_seconds + 1.0 / fps:
            raise DemoConfigError(
                f"interval.end_seconds ({cfg.end_seconds}) is past the source duration "
                f"({source.duration_seconds:g}s); this is refused rather than truncated")
        end_frame = _frame_for_seconds(cfg.end_seconds, fps, upper=True) - 1
        if source.last_frame_index is not None:
            end_frame = min(end_frame, source.last_frame_index)

    if end_frame < start_frame:
        raise DemoConfigError(
            f"the interval [start_seconds={cfg.start_seconds}, end_seconds="
            f"{cfg.end_seconds}) maps to source frames [{start_frame}, {end_frame}], "
            f"which contains no frame")

    frame_count = end_frame - start_frame + 1
    if frame_count > cfg.max_frames:
        raise DemoConfigError(
            f"the interval covers {frame_count} source frames, above "
            f"interval.max_frames={cfg.max_frames}. This is refused rather than "
            f"truncated: lower the interval or raise max_frames deliberately.")

    # --- ROI against the source frame
    if cfg.roi:
        x, y, w, h = cfg.roi
        if x + w > source.width or y + h > source.height:
            raise DemoConfigError(
                f"roi [x={x}, y={y}, w={w}, h={h}] does not fit inside the "
                f"{source.width}x{source.height} source frame")
        if w * h < 16:
            raise DemoConfigError("roi is too small to contain a golfer")

    # --- seeds, corrections, keyframes against the interval and the frame
    for name in TARGET_NAMES:
        seed = cfg.seeds.get(name)
        if seed is None:
            continue
        _check_point(seed, f"seeds.{name}", source, start_frame, end_frame)

    for name in TARGET_NAMES:
        seen: Dict[int, Point] = {}
        seed = cfg.seeds.get(name)
        if seed is not None:
            seen[seed.frame] = seed
        for point in cfg.corrections.get(name, []):
            key = point.frame
            if key in seen:
                raise DemoConfigError(
                    f"corrections.{name} has more than one point at frame {key} "
                    f"(the seed occupies frame {seed.frame}); corrections must be "
                    f"unique per frame" if seed else
                    f"corrections.{name} has more than one point at frame {key}; "
                    f"corrections must be unique per frame")
            seen[key] = point
            _check_point(point, f"corrections.{name}[frame={point.frame}]", source,
                         start_frame, end_frame)
        existing = list(cfg.corrections.get(name, []))
        if existing:
            cfg.corrections[name] = sorted(existing, key=lambda p: p.frame)

    present = [(phase, cfg.keyframes[phase]) for phase in PHASE_ORDER
               if phase in cfg.keyframes]
    for phase, frame in present:
        if not (start_frame <= frame <= end_frame):
            raise DemoConfigError(
                f"keyframes.{phase} = {frame} is outside the processed interval "
                f"[{start_frame}, {end_frame}]")
    for (prev_phase, prev_frame), (phase, frame) in zip(present, present[1:]):
        if frame < prev_frame:
            raise DemoConfigError(
                f"keyframes.{phase} = {frame} is before keyframes.{prev_phase} = "
                f"{prev_frame}; phase keyframes must be in chronological order "
                f"({' < '.join(PHASE_ORDER)})")
        if frame == prev_frame and prev_frame != 0:
            warnings.append(f"keyframes.{phase} and keyframes.{prev_phase} share frame "
                            f"{frame}; phase durations that depend on them are 0")

    for name, options in cfg.targets.items():
        if options.mode == "manual" and name not in cfg.seeds:
            raise DemoConfigError(
                f"targets.{name}.mode is 'manual', which needs an operator point, but "
                f"no seeds.{name} is given; manual mode never invents a track")
        if options.mode == "manual" and options.direction != "forward":
            raise DemoConfigError(
                f"targets.{name}.direction only applies to template tracking; manual "
                f"mode uses exactly the operator's points")
        if options.max_jump_px < 2:
            warnings.append(f"targets.{name}.max_jump_px is {options.max_jump_px}px, "
                            f"which will reject almost every frame")
        stop = options.track_until_frame
        if stop is not None:
            if not (start_frame <= stop <= end_frame):
                raise DemoConfigError(
                    f"targets.{name}.track_until_frame = {stop} is outside the "
                    f"processed interval [{start_frame}, {end_frame}]")
            seed = cfg.seeds.get(name)
            if seed is not None and stop < seed.frame:
                raise DemoConfigError(
                    f"targets.{name}.track_until_frame = {stop} is before its seed "
                    f"frame {seed.frame}; a stop frame ends the track, it does not "
                    f"start it")
            for point in cfg.corrections.get(name, []):
                if point.frame > stop:
                    raise DemoConfigError(
                        f"corrections.{name} has a point at frame {point.frame}, past "
                        f"targets.{name}.track_until_frame = {stop}")
        automatic = options.automatic_until_frame
        if automatic is not None:
            if not (start_frame <= automatic <= end_frame):
                raise DemoConfigError(
                    f"targets.{name}.automatic_until_frame = {automatic} is outside the "
                    f"processed interval [{start_frame}, {end_frame}]")
            seed = cfg.seeds.get(name)
            if seed is not None and automatic < seed.frame:
                raise DemoConfigError(
                    f"targets.{name}.automatic_until_frame = {automatic} is before its "
                    f"seed frame {seed.frame}; a stop frame ends automatic matching, it "
                    f"does not start it")
            if options.track_until_frame is not None \
                    and automatic > options.track_until_frame:
                raise DemoConfigError(
                    f"targets.{name}.automatic_until_frame = {automatic} is past "
                    f"targets.{name}.track_until_frame = {options.track_until_frame}, "
                    f"so the target would be out of scope before automatic matching "
                    f"stopped; the two bounds are contradictory")
            if not any(point.frame > automatic
                       for point in cfg.corrections.get(name, [])):
                warnings.append(
                    f"targets.{name}.automatic_until_frame = {automatic} has no "
                    f"operator correction after it: past that frame this target will "
                    f"report nothing at all, which is only useful if the target really "
                    f"is out of scope there")

    if cfg.interpolation.enabled and cfg.interpolation.max_gap_frames == 0:
        notes.append("interpolation is enabled with max_gap_frames=0, so no gap is "
                     "filled: every non-observed frame stays missing")
    if cfg.interpolation.enabled:
        for name in TARGET_NAMES:
            options = cfg.target(name)
            if cfg.interpolation.max_gap_frames >= options.loss_frames:
                notes.append(
                    f"interpolation.max_gap_frames ({cfg.interpolation.max_gap_frames}) "
                    f"is at or above targets.{name}.loss_frames ({options.loss_frames}); "
                    f"a declared loss still stays missing, because a loss is not a gap")

    if not cfg.seeds and not cfg.corrections:
        notes.append("no ball or clubhead seed was supplied: both targets are reported "
                     "UNAVAILABLE and nothing is drawn for them")
    if not cfg.keyframes:
        notes.append("no phase keyframes were supplied: phase cards report UNAVAILABLE "
                     "unless allow_approx_phases is set")
    if source.pixel_aspect_ratio != 1.0:
        notes.append(f"source sample aspect ratio is {source.pixel_aspect_ratio:g}:1; "
                     f"angles are corrected for it before being reported")

    frame_times = None
    try:
        from .probe import frame_times_for_interval
        frame_times = frame_times_for_interval(source, start_frame, end_frame)
    except Exception:  # pragma: no cover - timestamp lookup is best effort here
        frame_times = None
    if frame_times:
        notes.append("seconds are the decoded frames' own presentation timestamps "
                     "(ffprobe), not frames divided by a nominal rate: "
                     + source.timestamp_note)
    else:
        notes.append("per-frame frame timestamps were not available, so any seconds "
                     "below are frames / container cadence and carry that caveat"
                     + (f" ({source.timestamp_note})" if source.timestamp_note else ""))

    return ResolvedConfig(config=cfg, source=source, start_frame=start_frame,
                          end_frame=end_frame, frame_count=frame_count,
                          warnings=warnings, notes=notes, frame_times=frame_times)


def _check_point(point: Point, path: str, source: SourceProbe,
                 start_frame: int, end_frame: int) -> None:
    if not (start_frame <= point.frame <= end_frame):
        raise DemoConfigError(
            f"{path}.frame = {point.frame} is outside the processed interval "
            f"[{start_frame}, {end_frame}]")
    if not (0 <= point.x <= source.width - 1):
        raise DemoConfigError(
            f"{path}.x = {point.x} is outside the {source.width}px-wide frame")
    if not (0 <= point.y <= source.height - 1):
        raise DemoConfigError(
            f"{path}.y = {point.y} is outside the {source.height}px-tall frame")
