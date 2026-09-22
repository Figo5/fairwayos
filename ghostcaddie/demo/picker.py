"""The operator picker's state machine — pure, testable, no GUI.

`demo-setup` is what an operator actually runs, and it needs to be usable with a
mouse and a keyboard only: choose an interval, drag a ROI, click a ball seed and a
clubhead seed, mark the phases, add corrections, save a JSON config. All of that
decision-making lives here so that it can be tested without a window, and the GUI
module is then a thin event loop over these methods.

Everything the operator places is recorded with its source frame and marked as
operator input. Nothing here invents a point, and anything out of bounds is
rejected with a message instead of being clamped.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import DEMO_CONFIG_SCHEMA
from .config import (DEFAULT_MAX_FRAMES, PHASE_ORDER, TARGET_NAMES,
                     InterpolationOptions, RenderOptions, TargetOptions,
                     TimingOptions, DemoConfig, parse_config, validate_config)
from .errors import DemoConfigError
from .probe import SourceProbe

PHASE_KEYS = {"a": "address", "t": "takeaway", "p": "top", "d": "downswing",
              "i": "impact", "f": "follow_through", "z": "finish"}
#: The keys that MARK a phase. Only the upper-case forms do: ``a``/``d``/``i`` are
#: navigation (previous/next frame, interval start), so a lower-case mark would make
#: the documented key table and the handler disagree.
PHASE_KEYS_UPPER = {key.upper(): name for key, name in PHASE_KEYS.items()}

#: Defaults the picker writes for a brand-new config. They are ordinary
#: TargetOptions/... values, so a saved config and a hand-written one are the same
#: kind of object and cannot drift apart.
DEFAULT_TARGET_OPTIONS = {
    "ball": TargetOptions(mode="template", template_size=31, search_radius=48,
                          min_score=0.5, max_jump_px=60.0, loss_frames=3,
                          direction="forward").to_dict(),
    "clubhead": TargetOptions(mode="template", template_size=41, search_radius=70,
                              min_score=0.45, max_jump_px=90.0, loss_frames=3,
                              direction="forward").to_dict(),
}
DEFAULT_INTERPOLATION = InterpolationOptions().to_dict()
DEFAULT_TIMING = TimingOptions().to_dict()
DEFAULT_RENDER = RenderOptions().to_dict()

HELP_LINES = [
    "demo-setup — operator picker (local only, nothing is uploaded)",
    "",
    "  left / right, a / d   step one source frame",
    "  n / N                 step 10 frames",
    "  [ / ]                 jump 1 second",
    "  i / o                 set interval START / END at this frame",
    "  g                     clear the interval (whole clip)",
    "  drag the mouse        draw the golfer ROI (operator input, not detection)",
    "  x                     clear the ROI",
    "  1                     place the BALL seed at the mouse position",
    "  2                     place the CLUBHEAD seed at the mouse position",
    "  3 / 4                 add a BALL / CLUBHEAD correction at the mouse",
    "  5 / 6                 clear BALL / CLUBHEAD corrections",
    "  mouse position        the crosshair shown at the current frame",
    "  H                     toggle trajectory numbers (positions, this run only)",
    "  h                     toggle this help",
    "  s                     save the config JSON",
    "  q / ESC               quit (unsaved changes are reported, never discarded "
    "silently)",
    "",
    "  phases (mark at the current frame, UPPER-CASE keys only):",
    "    A address   T takeaway   P top   D downswing   I impact",
    "    F follow-through        Z finish",
    "    C clear all phase marks",
    "    (lower-case a/d/i are navigation, so only the capitals mark a phase)",
    "",
    "  Every point you place is operator input. It is recorded as such in the config,",
    "  in demo_provenance.json and in the report; it is never called a detection.",
]


class PickerError(RuntimeError):
    pass


def viewport_transform(source_w: int, source_h: int, max_w: int, max_h: int,
                       *, allow_upscale: bool = False) -> dict:
    """Fit a source frame into a viewport, and how to map a click back.

    Native 4K footage does not fit a laptop screen, so the picker shows the frame
    scaled and converts every mouse coordinate back to native pixels. The transform
    is stored explicitly — scale, offsets, and the letterbox margins — so a click
    can be inverted exactly instead of approximately.
    """
    if source_w <= 0 or source_h <= 0 or max_w <= 0 or max_h <= 0:
        raise PickerError("viewport transform needs positive dimensions")
    scale = min(max_w / float(source_w), max_h / float(source_h))
    if not allow_upscale:
        scale = min(scale, 1.0)
    view_w = max(1, int(round(source_w * scale)))
    view_h = max(1, int(round(source_h * scale)))
    return {"scale": scale, "view_w": view_w, "view_h": view_h,
            "offset_x": int((max_w - view_w) // 2),
            "offset_y": int((max_h - view_h) // 2),
            "viewport_w": int(max_w), "viewport_h": int(max_h),
            "source_w": int(source_w), "source_h": int(source_h)}


def viewport_to_native(point: Sequence[float], transform: dict) -> Optional[Tuple[float, float]]:
    """Map a click in the viewport back to native source pixels.

    A click outside the picture (in the letterbox margin) returns None: it is not a
    point on the footage, and clamping it to the nearest pixel would place an
    operator point the operator did not choose.
    """
    x = float(point[0]) - float(transform["offset_x"])
    y = float(point[1]) - float(transform["offset_y"])
    # the boundary is inclusive: the last viewport column belongs to the last
    # source column, so a point on the edge round-trips instead of vanishing
    if not (0 <= x <= transform["view_w"] and 0 <= y <= transform["view_h"]):
        return None
    scale = float(transform["scale"]) or 1.0
    nx, ny = x / scale, y / scale
    if not (0 <= nx < transform["source_w"] and 0 <= ny < transform["source_h"]):
        return None
    return (nx, ny)


def native_to_viewport(point: Sequence[float], transform: dict) -> Tuple[int, int]:
    """Map a native coordinate onto the scaled viewport, for drawing only.

    The result is clamped INSIDE the picture: rounding a native x at the right-hand
    edge can land exactly on ``view_w``, which ``viewport_to_native`` correctly
    rejects as outside the picture, so without the clamp the last pixel column and
    row of the source could be drawn but never clicked.
    """
    scale = float(transform["scale"]) or 1.0
    x = int(round(float(point[0]) * scale)) + int(transform["offset_x"])
    y = int(round(float(point[1]) * scale)) + int(transform["offset_y"])
    # clamped into the PICTURE area, which starts at the letterbox offset, not at 0
    return (min(max(int(transform["offset_x"]), x),
                int(transform["offset_x"]) + int(transform["view_w"]) - 1),
            min(max(int(transform["offset_y"]), y),
                int(transform["offset_y"]) + int(transform["view_h"]) - 1))


def roi_to_native(roi: Sequence[int], transform: dict) -> Tuple[int, int, int, int]:
    """Map a ROI rectangle drawn in viewport pixels back to native pixels.

    A drag is clamped to the picture area (the letterbox margin is not footage), so
    a drag that starts or ends slightly outside still yields the operator's box on
    the image rather than a box that cannot exist.
    """
    scale = float(transform["scale"]) or 1.0
    ox, oy = float(transform["offset_x"]), float(transform["offset_y"])
    vw, vh = float(transform["view_w"]), float(transform["view_h"])
    x0 = min(max(float(roi[0]) - ox, 0.0), vw)
    y0 = min(max(float(roi[1]) - oy, 0.0), vh)
    x1 = min(max(float(roi[0]) + float(roi[2]) - ox, 0.0), vw)
    y1 = min(max(float(roi[1]) + float(roi[3]) - oy, 0.0), vh)
    return (int(round(min(x0, x1) / scale)), int(round(min(y0, y1) / scale)),
            int(round(abs(x1 - x0) / scale)), int(round(abs(y1 - y0) / scale)))


class PickerState:
    def __init__(self, video_path: str, source: SourceProbe, *,
                 existing: Optional[DemoConfig] = None,
                 out_path: Optional[str] = None) -> None:
        self.video_path = str(video_path)
        self.source = source
        self.out_path = out_path or str(Path(video_path).with_suffix(".demo-config.json"))
        self.frame = 0
        self.start_frame = 0
        self.end_frame = max(0, (source.last_frame_index if source.last_frame_index
                                 is not None
                                 else int(math.ceil(source.duration_seconds * source.fps)) - 1))
        self.roi: Optional[List[int]] = None
        self.seeds: Dict[str, Tuple[int, float, float]] = {}
        self.corrections: Dict[str, List[Tuple[int, float, float]]] = {
            "ball": [], "clubhead": []}
        self.keyframes: Dict[str, int] = {}
        self.notes = ""
        #: Clip-specific tuning carried over from an existing config untouched, so
        #: reopening a config to add one correction cannot silently reset the
        #: tracker, timing, interpolation or render settings it was tuned with.
        self.targets: Dict[str, dict] = {}
        self.interpolation: Optional[dict] = None
        self.timing: Optional[dict] = None
        self.render_options: Optional[dict] = None
        self.handedness = "right"
        self.allow_approx_phases = False
        self.max_frames = DEFAULT_MAX_FRAMES
        self.end_seconds: Optional[float] = None
        self.start_seconds = 0.0
        self.messages: List[str] = []
        self.dirty = False
        self.quit_armed = False
        if existing is not None:
            self._load_existing(existing)

    # -- loading -----------------------------------------------------------
    def _load_existing(self, config: DemoConfig) -> None:
        """Load a config without losing anything the operator already decided."""
        self.roi = list(config.roi) if config.roi else None
        for name, point in config.seeds.items():
            self.seeds[name] = (int(point.frame), float(point.x), float(point.y))
        for name in TARGET_NAMES:
            points = config.corrections.get(name) or []
            self.corrections[name] = [(int(p.frame), float(p.x), float(p.y))
                                      for p in points]
        self.keyframes = {k: int(v) for k, v in config.keyframes.items()}
        self.targets = {name: options.to_dict()
                        for name, options in config.targets.items()}
        self.interpolation = config.interpolation.to_dict()
        self.timing = config.timing.to_dict()
        self.render_options = config.render.to_dict()
        self.handedness = config.handedness
        self.allow_approx_phases = config.allow_approx_phases
        self.max_frames = config.max_frames
        self.notes = config.notes
        # The interval is restored as written: start_seconds is inclusive and
        # end_seconds is exclusive, so the frame range comes back exactly.
        self.start_seconds = float(config.start_seconds)
        self.start_frame = int(math.floor(config.start_seconds * self.source.fps))
        if config.end_seconds is not None:
            self.end_seconds = float(config.end_seconds)
            self.end_frame = min(
                self.end_frame,
                max(self.start_frame,
                    int(math.ceil(config.end_seconds * self.source.fps)) - 1))
        self._say(f"loaded interval {self.start_frame}-{self.end_frame} and every "
                  f"other setting from the existing config")

    # -- navigation --------------------------------------------------------
    def _say(self, message: str) -> None:
        self.messages.append(message)
        self.messages = self.messages[-6:]

    def goto(self, frame: int) -> None:
        last = self.source.last_frame_index
        limit = last if last is not None else int(
            math.ceil(self.source.duration_seconds * self.source.fps))
        self.frame = max(0, min(int(frame), limit))

    def step(self, delta: int) -> None:
        self.goto(self.frame + int(delta))

    def jump_seconds(self, seconds: float) -> None:
        self.goto(self.frame + int(round(seconds * self.source.fps)))

    # -- interval ----------------------------------------------------------
    def set_interval_start(self) -> None:
        if self.frame > self.end_frame:
            self._say("interval start must not be after the end; not changed")
            return
        self.start_frame = self.frame
        self.start_seconds = self.frame / self.source.fps
        self.dirty = True
        self._say(f"interval start = frame {self.start_frame} (inclusive)")

    def set_interval_end(self) -> None:
        if self.frame < self.start_frame:
            self._say("interval end must not be before the start; not changed")
            return
        self.end_frame = self.frame
        # end_seconds is EXCLUSIVE, so the written value is the boundary after the
        # marked frame: frame N inclusive is (N+1)/fps seconds.
        self.end_seconds = (self.frame + 1) / self.source.fps
        self.dirty = True
        self._say(f"interval end = frame {self.end_frame} (inclusive; written as the "
                  f"exclusive boundary {self.end_seconds:g}s)")

    def clear_interval(self) -> None:
        self.start_frame = 0
        self.start_seconds = 0.0
        last = self.source.last_frame_index
        self.end_frame = (last if last is not None else
                          int(math.ceil(self.source.duration_seconds * self.source.fps)) - 1)
        self.end_seconds = None
        self.dirty = True
        self._say("interval reset to the whole clip")

    # -- ROI ---------------------------------------------------------------
    def set_roi(self, x: float, y: float, w: float, h: float) -> bool:
        roi = [int(round(x)), int(round(y)), int(round(w)), int(round(h))]
        if roi[2] <= 0 or roi[3] <= 0:
            self._say("ROI has no area; not set")
            return False
        if roi[0] < 0 or roi[1] < 0 or roi[0] + roi[2] > self.source.width \
                or roi[1] + roi[3] > self.source.height:
            self._say("ROI does not fit inside the frame; not set")
            return False
        self.roi = roi
        self.dirty = True
        self._say(f"ROI = {roi} (operator input, not automatic perception)")
        return True

    def clear_roi(self) -> None:
        self.roi = None
        self.dirty = True
        self._say("ROI cleared: without an ROI the body layer is reported AMBIGUOUS")

    # -- points ------------------------------------------------------------
    def _check_point(self, x: float, y: float) -> bool:
        if not (math.isfinite(x) and math.isfinite(y)):
            self._say("point is not finite; not placed")
            return False
        if not (0 <= x < self.source.width and 0 <= y < self.source.height):
            self._say(f"point ({x:.0f}, {y:.0f}) is outside the "
                      f"{self.source.width}x{self.source.height} frame; not placed")
            return False
        return True

    def place_seed(self, target: str, x: float, y: float) -> bool:
        if target not in TARGET_NAMES:
            raise PickerError(f"unknown target {target!r}")
        if not self._check_point(x, y):
            return False
        if not (self.start_frame <= self.frame <= self.end_frame):
            self._say("the current frame is outside the interval; move inside first")
            return False
        self.seeds[target] = (self.frame, float(x), float(y))
        self.corrections[target] = [c for c in self.corrections[target]
                                    if c[0] != self.frame]
        self.dirty = True
        self._say(f"{target} seed = frame {self.frame} at ({x:.0f}, {y:.0f})")
        return True

    def place_correction(self, target: str, x: float, y: float) -> bool:
        if target not in TARGET_NAMES:
            raise PickerError(f"unknown target {target!r}")
        if not self._check_point(x, y):
            return False
        if not (self.start_frame <= self.frame <= self.end_frame):
            self._say("the current frame is outside the interval; move inside first")
            return False
        if target in self.seeds and self.seeds[target][0] == self.frame:
            self.seeds.pop(target)
            self._say(f"{target} seed was replaced by a correction at frame {self.frame}")
        if any(c[0] == self.frame for c in self.corrections[target]):
            self._say(f"{target} already has a point at frame {self.frame}; "
                      f"corrections must be unique per frame")
            return False
        self.corrections[target].append((self.frame, float(x), float(y)))
        self.corrections[target].sort(key=lambda c: c[0])
        self.dirty = True
        self._say(f"{target} correction = frame {self.frame} at ({x:.0f}, {y:.0f})")
        return True

    def clear_corrections(self, target: str) -> None:
        self.corrections[target] = []
        self.dirty = True
        self._say(f"{target} corrections cleared")

    # -- phases ------------------------------------------------------------
    def mark_phase(self, phase: str) -> bool:
        if phase not in PHASE_ORDER:
            raise PickerError(f"unknown phase {phase!r}")
        self.keyframes[phase] = self.frame
        self.dirty = True
        self._say(f"{phase} = frame {self.frame}")
        return True

    def clear_keyframes(self) -> None:
        self.keyframes = {}
        self.dirty = True
        self._say("phase marks cleared")

    # -- output ------------------------------------------------------------
    def to_config_dict(self) -> dict:
        interval_end = (self.end_seconds if self.end_seconds is not None
                        else (self.end_frame + 1) / self.source.fps)
        targets = {name: dict(self.targets.get(name) or DEFAULT_TARGET_OPTIONS[name])
                   for name in TARGET_NAMES}
        # The boilerplate is written once and REPLACED on a resave. Appending it
        # blindly stacked the same sentence up on every save, and the interval can
        # change between saves, so the old sentence is cut at its marker first.
        boilerplate = ("Config written by demo-setup. Interval end is EXCLUSIVE: "
                       f"end_seconds corresponds to source frame {self.end_frame + 1}.")
        marker = "Config written by demo-setup"
        notes = self.notes.split(marker)[0].strip()
        notes = (notes + " " if notes else "") + boilerplate
        payload = {
            "schema_version": DEMO_CONFIG_SCHEMA,
            "clip_path_hint": Path(self.video_path).name,
            "interval": {
                "start_seconds": round(self.start_seconds, 6),
                "end_seconds": round(interval_end, 6),
                "max_frames": self.max_frames,
            },
            "roi": list(self.roi) if self.roi else None,
            "seeds": {name: {"frame": frame, "x": round(x, 2), "y": round(y, 2),
                             "note": "operator-placed in demo-setup"}
                      for name, (frame, x, y) in self.seeds.items()},
            "keyframes": {phase: self.keyframes[phase]
                          for phase in PHASE_ORDER if phase in self.keyframes},
            "targets": targets,
            "corrections": {name: [{"frame": frame, "x": round(x, 2), "y": round(y, 2),
                                    "note": "operator correction in demo-setup"}
                                   for (frame, x, y) in self.corrections.get(name, [])]
                            for name in TARGET_NAMES},
            "interpolation": dict(self.interpolation or DEFAULT_INTERPOLATION),
            "timing": dict(self.timing or DEFAULT_TIMING),
            "render": dict(self.render_options or DEFAULT_RENDER),
            "handedness": self.handedness,
            "allow_approx_phases": self.allow_approx_phases,
            "notes": notes,
        }
        return payload

    def save(self, path: Optional[str] = None) -> str:
        """Write the config, or report why it is not writable yet.

        The picker validates against the probe before writing, so an operator cannot
        spend a session building a config that the pipeline will refuse. A refusal is
        recorded in the HUD and nothing is written.
        """
        target = Path(path or self.out_path).expanduser()
        payload = self.to_config_dict()
        try:
            parsed = parse_config(payload)
            validate_config(parsed, self.source)
        except DemoConfigError as exc:
            self._say(f"NOT saved: {exc}")
            return ""
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self.out_path = str(target)
        self.dirty = False
        self.quit_armed = False
        self._say(f"saved {target}")
        return str(target)

    def status_lines(self, *, show_help: bool) -> List[str]:
        lines = [
            f"frame {self.frame}  ({self.frame / self.source.fps:g}s of "
            f"{self.source.duration_seconds:g}s)  {self.source.width}x"
            f"{self.source.height} @ {self.source.fps:g} fps",
            f"interval  {self.start_frame}-{self.end_frame} "
            f"({self.end_frame - self.start_frame + 1} frames)",
            f"roi       {self.roi if self.roi else 'none (body will be AMBIGUOUS)'}",
        ]
        for name in TARGET_NAMES:
            seed = self.seeds.get(name)
            corrections = len(self.corrections.get(name, []))
            lines.append(f"{name:<9} "
                         + (f"seed f{seed[0]} ({seed[1]:.0f},{seed[2]:.0f})" if seed
                            else "no seed (reported UNAVAILABLE)")
                         + f"   corrections {corrections}")
        marked = ", ".join(f"{p}=f{self.keyframes[p]}" for p in PHASE_ORDER
                           if p in self.keyframes)
        lines.append(f"phases    {marked or 'none (cards will be UNAVAILABLE)'}")
        # A long absolute path runs off the gutter, so show the file name and the
        # parent separately (the full path is printed to the terminal on exit).
        out = Path(self.out_path)
        lines.append(f"output    {out.name}"
                     + (f"  (folder {out.parent.name})" if out.parent.name else "")
                     + ("  [UNSAVED]" if self.dirty else ""))
        if show_help:
            lines += [""] + HELP_LINES
        else:
            lines.append("press h for help, s to save, q to quit")
        if self.messages:
            lines.append("")
            lines += [f"- {m}" for m in self.messages[-3:]]
        return lines

    # -- quit policy -------------------------------------------------------
    def request_quit(self) -> bool:
        """True when the session may end. A dirty session needs a second 'q'.

        This exists because the previous policy told the operator "press q again to
        discard" without ever honouring it: a dirty session could not be left except
        by killing the process.
        """
        if not self.dirty:
            return True
        if not self.quit_armed:
            self.quit_armed = True
            self._say("unsaved changes: press q again (or Escape) to quit WITHOUT "
                      "saving, or s to save first")
            return False
        self._say("quitting with unsaved changes at the operator's request")
        return True

    def discard_changes(self) -> None:
        self.dirty = False
        self.quit_armed = False
        self._say("unsaved changes discarded")
