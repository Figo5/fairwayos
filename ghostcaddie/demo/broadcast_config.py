"""The BROADCAST SHOWCASE config: several scene windows inside one whole shot.

Why this exists
---------------
The demo/showcase workflow renders **one** interval with one set of overlays. A real
broadcast shot is not one interval: the golfer is visible for a few seconds, then the
director cuts to a reaction, a flight tracer or a green view, and a second golfer
window may follow. Drawing the first window's skeleton, trails, cards and phase badge
across the cut is wrong for the same reason a stale metric is wrong: it is a claim
about frames nobody measured.

This module defines the opt-in *broadcast* config on top of the existing, unchanged
demo config:

* a **whole-shot interval** (the clip that is rendered end to end, at the source's own
  cadence, with its own audio);
* a list of **scenes**, each an ordinary demo config scoped to a
  ``{"start_frame", "end_frame"}`` golfer-visible window, with its own ROI/seeds/
  phases/targets;
* scenes may not overlap. A frame belongs to at most one scene, so nothing is ever
  interpolated across a cut and no scene inherits the previous one's tracking,
  card debounce or ``hand_rise`` reference;
* outside every window nothing is drawn: no body, no trails, no cards, no phase badge.

Everything that does not depend on the extra scene structure is parsed by the existing
:func:`ghostcaddie.demo.config.parse_config` — the scene blocks are the demo blocks,
validated by the demo validator, so there is exactly one implementation of each rule.

Phase vocabulary
----------------
A putt has no TOP. Each scene declares a ``shot_type`` and its keyframes use that
type's own vocabulary; a putt is marked ``address / backstroke / forward_stroke /
contact / finish``. The vocabulary is mapped onto the existing canonical phase slots
for the analytics that already exist (so tempo stays defined), and the *display*
keeps the operator's vocabulary, so a putt never shows a fictitious TOP label.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import (DemoConfig, InterpolationOptions, RoiKeyframe, parse_config,
                     parse_showcase)
from .errors import DemoConfigError

BROADCAST_CONFIG_SCHEMA = "fairwayos-broadcast-config.v1"

#: The stroke types a scene may declare. ``unknown`` is honest: it means the operator
#: did not classify the stroke, so the full-swing vocabulary is used as a default and
#: the report says the type was not declared.
STROKE_TYPES = ("full_swing", "chip", "pitch", "bunker", "putt", "unknown")
#: Backwards-compatible alias: earlier drafts of the contract called this ``shot_type``.
SHOT_TYPES = STROKE_TYPES

#: What a window contains. Only ``golfer_stroke`` may carry tracking inputs; the other
#: kinds exist so the operator can say *why* a stretch has no overlay, and so a cut
#: proposal has somewhere to land. A reaction/flight/green window is rendered with no
#: overlay, no cards and no badge.
SCENE_KINDS = ("golfer_stroke", "reaction", "flight", "green", "replay", "other")

_FULL_SWING_VOCAB = ("address", "takeaway", "top", "downswing", "impact",
                     "follow_through", "finish")
_PUTT_VOCAB = ("address", "backstroke", "forward_stroke", "contact", "finish")

STROKE_TYPE_PHASES: Dict[str, Tuple[str, ...]] = {
    "full_swing": _FULL_SWING_VOCAB,
    "chip": _FULL_SWING_VOCAB,
    "pitch": _FULL_SWING_VOCAB,
    "bunker": _FULL_SWING_VOCAB,
    "putt": _PUTT_VOCAB,
    "unknown": _FULL_SWING_VOCAB,
}
#: Alias kept for the first published draft of the contract.
SHOT_TYPE_PHASES = STROKE_TYPE_PHASES

#: vocabulary name -> canonical slot used by the existing phase/tempo analytics.
#: ``backstroke`` is the putt's backswing and ``forward_stroke`` its downswing, which
#: is what makes the tempo record mean the same thing on a putt as on a full swing.
PHASE_SLOT: Dict[str, str] = {
    "address": "address", "takeaway": "takeaway", "top": "top",
    "downswing": "downswing", "impact": "impact", "follow_through": "follow_through",
    "finish": "finish",
    "backstroke": "takeaway", "forward_stroke": "downswing", "contact": "impact",
}

#: The frame a scene's impact accent is anchored to, per vocabulary.
CONTACT_NAMES: Dict[str, str] = {
    "full_swing": "impact", "chip": "impact", "pitch": "impact", "bunker": "impact",
    "unknown": "impact", "putt": "contact",
}

MAX_SCENES = 64
#: A window shorter than this cannot hold a stroke; a longer one is not a scene, it is
#: the whole clip (which is what the plain demo workflow is for).
MIN_SCENE_FRAMES = 3
MAX_SCENE_FRAMES = 20000

#: A broadcast-profile cap on operator anchor spacing. A 30 fps downswing is a handful
#: of frames between anchors, so a long gap is not a slow swing, it is an invented
#: stretch. The tight cap applies within +/- this many frames of the contact mark.
BROADCAST_MAX_ANCHOR_SPACING = 12
BROADCAST_CONTACT_SPACING = 6
BROADCAST_CONTACT_WINDOW_FRAMES = 12

AUDIO_MODES = ("copy", "none")
#: What is drawn on a frame that lies in no window. ``brand_only`` keeps the small
#: persistent mark; ``clean`` draws the untouched source picture.
OUTSIDE_WINDOW_MODES = ("brand_only", "clean")

#: A scene that is not a golfer stroke may carry nothing but its window and a label:
#: there is no golfer to track, so any seed, anchor or phase mark on it is a mistake.
_NON_STROKE_SCENE_KEYS = ("id", "label", "kind", "window", "notes")

_SCENE_KEYS = ("id", "label", "kind", "stroke_type", "window", "keyframes", "metrics",
               "notes", "metric_hide_frames")
#: Keys handed straight to the demo validator. The broadcast layer adds nothing to
#: them, which is the point: one implementation per rule.
_DEMO_SCENE_KEYS = ("roi", "roi_keyframes", "seeds", "targets", "corrections",
                    "manual_segments", "interpolation", "handedness",
                    "allow_approx_phases", "source_description", "notes")

_BROADCAST_FIELDS = ("schema_version", "clip_path_hint", "shot", "interval", "scenes",
                     "presentation", "outside_windows", "cuts", "overrides", "timing",
                     "render", "source_description", "notes")
_SHOT_FIELDS = ("id", "label", "stroke_type", "expected_source_sha256", "expected_fps",
                "expected_frames", "source_notes", "parent_source_manifest")
_WINDOW_FIELDS = ("start_frame", "end_frame", "note")
_PRESENTATION_FIELDS = ("brand", "brand_sub", "disclosure", "metrics",
                        "card_debounce_frames", "card_row_max_width",
                        "ball_trail_points", "club_trail_points", "impact_accent",
                        "impact_accent_span_frames", "card_position",
                        "metrics_required")
_CUT_FIELDS = ("frame", "kind", "span", "confidence", "method", "note", "decision",
               "decided_by", "decided_at")
_CUT_KINDS = ("hard_cut", "dissolve")
_CUT_DECISIONS = ("proposed", "accepted", "rejected", "operator_added")
_OVERRIDE_FIELDS = ("frame", "decision", "note", "decided_by", "decided_at")


def _text(value: Any, path: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise DemoConfigError(f"{path} must be a string")
    if not value.strip() and not allow_empty:
        raise DemoConfigError(f"{path} must not be empty")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise DemoConfigError(f"{path} must be true or false")
    return value


def _whole(value: Any, path: str, *, minimum: Optional[int] = None,
           maximum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float) and float(value).is_integer():
            value = int(value)
        else:
            raise DemoConfigError(f"{path} must be a whole number")
    if minimum is not None and value < minimum:
        raise DemoConfigError(f"{path} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise DemoConfigError(f"{path} must be <= {maximum}, got {value}")
    return value


def _finite(value: Any, path: str, *, minimum: Optional[float] = None,
            maximum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DemoConfigError(f"{path} must be a number")
    out = float(value)
    if out != out or out in (float("inf"), float("-inf")):
        raise DemoConfigError(f"{path} must be a finite number")
    if minimum is not None and out < minimum:
        raise DemoConfigError(f"{path} must be >= {minimum}, got {out}")
    if maximum is not None and out > maximum:
        raise DemoConfigError(f"{path} must be <= {maximum}, got {out}")
    return out


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DemoConfigError(f"{path} must be an object")
    return value


def _reject_unknown(block: Mapping[str, Any], allowed: Sequence[str], path: str) -> None:
    unknown = sorted(k for k in block if not str(k).startswith("_")
                     and k not in allowed)
    if unknown:
        raise DemoConfigError(
            f"{path} has unknown field(s) {unknown}; allowed keys are {list(allowed)}. "
            f"Unknown keys are refused rather than ignored so a typo cannot silently "
            f"drop an operator input.")


# ---------------------------------------------------------------- data model

def _manual_segments_for_emit(blocks: Mapping[str, Any]) -> Dict[str, dict]:
    """Re-emit ``manual_segments`` using only keys the demo parser accepts.

    The demo layer's own ``ManualSegments.to_dict`` carries *derived* facts
    (``segment_count``, ``anchor_count``, ``rule``, ``interpolated_rule``,
    ``start_frame``, ``end_frame``). Those are useful in a report, but the demo
    parser REFUSES them as unknown fields, so a config copy that carries them cannot
    be loaded again -- which is the one thing a config copy must do. Here the derived
    counts move under ``_``-prefixed keys, which the schema ignores by design, and the
    anchors, spacing bound, hide ranges, trail mode and author are re-stated exactly.
    """
    out: Dict[str, dict] = {}
    for name, block in sorted(blocks.items()):
        out[name] = {
            "author": block.author,
            "replace_automatic": bool(block.replace_automatic),
            "hide_frames": [[int(first), int(last)] for first, last in block.hide_frames],
            "trail": block.trail,
            "trail_from_frame": (None if block.trail_from_frame is None
                                 else int(block.trail_from_frame)),
            "segments": [{
                "name": segment.name,
                "max_anchor_spacing_frames": int(segment.max_anchor_spacing_frames),
                "anchors": [anchor.to_dict() for anchor in segment.anchors],
                "note": segment.note,
                "_start_frame": segment.start_frame(),
                "_end_frame": segment.end_frame(),
                "_anchor_count": len(segment.anchors),
            } for segment in block.segments],
            "_segment_count": len(block.segments),
        }
    return out


@dataclass
class SceneWindow:
    start_frame: int
    end_frame: int
    note: str = ""

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1

    def frames(self) -> range:
        return range(self.start_frame, self.end_frame + 1)

    def contains(self, frame: int) -> bool:
        return self.start_frame <= frame <= self.end_frame

    def to_dict(self) -> dict:
        # Only keys the parser accepts: a run's own config copy is meant to be
        # re-runnable, so emitting a derived key here would make the file it writes
        # unloadable. Derived facts go under `_`-prefixed keys, which the schema
        # ignores by design.
        return {"start_frame": self.start_frame, "end_frame": self.end_frame,
                "note": self.note, "_frame_count": self.frame_count}


@dataclass
class SceneConfig:
    """One window and everything drawn inside it.

    ``kind`` says what the window contains. Only a ``golfer_stroke`` window carries
    tracking inputs and is measured at all; the other kinds render as bare source
    frames, which is how a reaction / flight / green stretch is declared instead of
    being left implicit.
    """

    scene_id: str
    label: str
    stroke_type: str
    window: SceneWindow
    #: Operator keyframes in the scene's own vocabulary (``contact`` for a putt).
    keyframes: Dict[str, int]
    #: The same keyframes mapped onto the canonical slots the analytics use.
    canonical_keyframes: Dict[str, int]
    demo: DemoConfig
    kind: str = "golfer_stroke"
    metrics: Tuple[str, ...] = ()
    metric_hide_frames: Tuple[Tuple[int, int], ...] = ()
    notes: str = ""

    @property
    def is_stroke(self) -> bool:
        return self.kind == "golfer_stroke"

    @property
    def phase_vocabulary(self) -> Tuple[str, ...]:
        return STROKE_TYPE_PHASES[self.stroke_type]

    @property
    def shot_type(self) -> str:
        """Alias for the first published draft of the contract."""
        return self.stroke_type

    @property
    def contact_name(self) -> Optional[str]:
        return CONTACT_NAMES.get(self.stroke_type)

    @property
    def contact_frame(self) -> Optional[int]:
        name = self.contact_name
        return self.keyframes.get(name) if name else None

    def roi_keyframes_in_window(self) -> List[RoiKeyframe]:
        return list(self.demo.roi_keyframes)

    def to_dict(self) -> dict:
        # Every key here must be one the parser accepts: this dict is what the run
        # writes as its own `broadcast_config.json`, and a config copy that cannot be
        # passed back to the CLI is not a config copy. Derived facts are emitted under
        # `_`-prefixed keys, which the loader ignores.
        out = {
            "id": self.scene_id, "label": self.label, "kind": self.kind,
            "stroke_type": self.stroke_type,
            "window": self.window.to_dict(),
            "keyframes": {k: self.keyframes[k] for k in sorted(self.keyframes)},
            "notes": self.notes,
            "_phase_vocabulary": list(self.phase_vocabulary),
            "_contact_frame": self.contact_frame,
            "_contact_mark": self.contact_name,
        }
        if self.metric_hide_frames:
            out["metric_hide_frames"] = [list(pair) for pair in self.metric_hide_frames]
        if self.metrics:
            out["metrics"] = list(self.metrics)
        # The demo-shaped keys (ROI, its keyframes, seeds, targets, corrections, manual
        # segments, interpolation, handedness, approximation and the operator's own
        # description) are emitted from the VALIDATED payload the demo parser accepted,
        # so an assistance mark the schema models -- and a key it merely tolerates --
        # survives a save unchanged. A scene built in code rather than parsed has no
        # such payload, so it falls back to the model; that path emits derived facts
        # under `_` keys for the same reason.
        raw = self.demo.raw or {}
        for key in _DEMO_SCENE_KEYS:
            if key == "notes":
                continue  # already emitted from the scene model, above
            if key in raw:
                out[key] = raw[key]
                continue
            if key == "roi" and self.demo.roi:
                out["roi"] = list(self.demo.roi)
            elif key == "roi_keyframes" and self.demo.roi_keyframes:
                out["roi_keyframes"] = [k.to_dict() for k in self.demo.roi_keyframes]
            elif key == "seeds" and self.demo.seeds:
                out["seeds"] = {k: v.to_dict() for k, v in sorted(self.demo.seeds.items())}
            elif key == "targets" and self.demo.targets:
                out["targets"] = {k: v.to_dict()
                                  for k, v in sorted(self.demo.targets.items())}
            elif key == "corrections" and self.demo.corrections:
                out["corrections"] = {k: [p.to_dict() for p in v]
                                      for k, v in sorted(self.demo.corrections.items())}
            elif key == "manual_segments" and self.demo.manual_segments:
                out["manual_segments"] = _manual_segments_for_emit(
                    self.demo.manual_segments)
            elif (key == "interpolation"
                  and self.demo.interpolation.to_dict() != InterpolationOptions().to_dict()):
                out["interpolation"] = self.demo.interpolation.to_dict()
            elif key == "handedness" and self.demo.handedness != "right":
                out["handedness"] = self.demo.handedness
            elif key == "allow_approx_phases" and self.demo.allow_approx_phases:
                out["allow_approx_phases"] = True
            elif key == "source_description" and self.demo.source_description:
                out["source_description"] = self.demo.source_description
        return out


@dataclass
class BroadcastPresentation:
    brand: str = "FAIRWAY OS"
    brand_sub: str = "SWING ANALYSIS"
    disclosure: str = "OPERATOR-ASSISTED \u00b7 2D IMAGE-SPACE ESTIMATES"
    metrics: Tuple[str, ...] = ("shoulder_tilt", "hip_tilt", "knee_flexion",
                               "hand_rise")
    card_debounce_frames: int = 8
    card_row_max_width: Optional[int] = None
    ball_trail_points: int = 10
    club_trail_points: int = 12
    impact_accent: bool = True
    impact_accent_span_frames: int = 12
    #: Where the card row sits. ``auto`` decides per scene from the scene's own
    #: measured content: when the golfer's keypoint union or the drawn ball/club path
    #: reaches into the bottom band, the row moves to the top instead of covering the
    #: feet of the swing. The choice is made once per scene (never frame by frame) and
    #: is recorded with its reason.
    card_position: str = "auto"
    #: When true (the default) a scene with every card unavailable hides the card row
    #: rather than drawing four placeholders.
    metrics_required: bool = False

    def to_dict(self) -> dict:
        return {"brand": self.brand, "brand_sub": self.brand_sub,
                "disclosure": self.disclosure, "metrics": list(self.metrics),
                "card_debounce_frames": self.card_debounce_frames,
                "card_row_max_width": self.card_row_max_width,
                "ball_trail_points": self.ball_trail_points,
                "club_trail_points": self.club_trail_points,
                "impact_accent": self.impact_accent,
                "impact_accent_span_frames": self.impact_accent_span_frames,
                "card_position": self.card_position,
                "metrics_required": self.metrics_required}


@dataclass
class CutCandidate:
    """A proposed cut. Always a suggestion: it is never applied on its own."""

    frame: int
    kind: str = "hard_cut"
    confidence: float = 0.0
    method: str = ""
    note: str = ""
    #: For a dissolve: the inclusive frame span the blend occupies.
    span: Optional[List[int]] = None
    decision: str = "proposed"
    decided_by: str = ""
    decided_at: str = ""

    def to_dict(self) -> dict:
        return {"frame": self.frame, "kind": self.kind,
                "span": list(self.span) if self.span else None,
                "confidence": round(float(self.confidence), 4),
                "method": self.method, "note": self.note,
                "decision": self.decision, "decided_by": self.decided_by,
                "decided_at": self.decided_at}


@dataclass
class BroadcastConfig:
    schema_version: str = BROADCAST_CONFIG_SCHEMA
    clip_path_hint: str = ""
    shot_id: str = ""
    shot_label: str = ""
    stroke_type: str = "unknown"
    expected_source_sha256: str = ""
    #: Where the parent (pre-trim) source digest and the native trim frames of this
    #: clip are recorded. An operator names it explicitly so provenance never has to
    #: guess; when it is empty the run looks for a manifest beside the clip itself.
    parent_source_manifest: str = ""
    #: Operator-declared source facts. They are re-stated in provenance, never
    #: derived: a claim about the footage that the tool did not check is the
    #: operator's claim, and it must survive a save.
    source_notes: str = ""
    expected_fps: Optional[float] = None
    expected_frames: Optional[int] = None
    start_seconds: float = 0.0
    end_seconds: Optional[float] = None
    max_frames: int = 1200
    scenes: List[SceneConfig] = field(default_factory=list)
    presentation: BroadcastPresentation = field(default_factory=BroadcastPresentation)
    outside_windows: str = "brand_only"
    cuts: List[CutCandidate] = field(default_factory=list)
    overrides: List[dict] = field(default_factory=list)
    render: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, Any] = field(default_factory=dict)
    source_description: str = ""
    notes: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)
    #: Configured vs effective ``max_anchor_spacing_frames`` per operator segment.
    spacing_records: List[dict] = field(default_factory=list)

    @property
    def shot_type(self) -> str:
        return self.stroke_type

    def scene(self, scene_id: str) -> Optional[SceneConfig]:
        for scene in self.scenes:
            if scene.scene_id == scene_id:
                return scene
        return None

    def stroke_scenes(self) -> List[SceneConfig]:
        return [s for s in self.scenes if s.is_stroke]

    def to_dict(self) -> dict:
        shot: Dict[str, Any] = {"id": self.shot_id, "label": self.shot_label,
                "stroke_type": self.stroke_type,
                "expected_source_sha256": self.expected_source_sha256}
        # Only what the operator supplied is re-stated: an absent key stays absent, so
        # a save cannot turn "not declared" into "declared as empty".
        if self.parent_source_manifest:
            shot["parent_source_manifest"] = self.parent_source_manifest
        if self.source_notes:
            shot["source_notes"] = self.source_notes
        if self.expected_fps is not None:
            shot["expected_fps"] = self.expected_fps
        if self.expected_frames is not None:
            shot["expected_frames"] = self.expected_frames
        return {
            "schema_version": self.schema_version,
            "clip_path_hint": self.clip_path_hint,
            "shot": shot,
            "interval": {"start_seconds": self.start_seconds,
                         "end_seconds": self.end_seconds,
                         "max_frames": self.max_frames},
            "scenes": [s.to_dict() for s in self.scenes],
            "presentation": self.presentation.to_dict(),
            "outside_windows": self.outside_windows,
            "cuts": [c.to_dict() for c in self.cuts],
            "overrides": list(self.overrides),
            "render": dict(self.render),
            "timing": dict(self.timing),
            "source_description": self.source_description,
            "notes": self.notes,
        }


# ---------------------------------------------------------------- parsing

def _parse_window(block: Any, path: str, *, index: int) -> SceneWindow:
    data = _mapping(block, path)
    _reject_unknown(data, _WINDOW_FIELDS, path)
    start = _whole(data.get("start_frame"), f"{path}.start_frame", minimum=0)
    end = _whole(data.get("end_frame"), f"{path}.end_frame", minimum=0)
    if end < start:
        raise DemoConfigError(
            f"{path}.end_frame ({end}) is before start_frame ({start}); a scene window "
            f"must be an ordered inclusive source-frame range")
    count = end - start + 1
    if count < MIN_SCENE_FRAMES:
        raise DemoConfigError(
            f"{path} covers {count} frame(s); a scene window needs at least "
            f"{MIN_SCENE_FRAMES} frames to hold a stroke")
    if count > MAX_SCENE_FRAMES:
        raise DemoConfigError(
            f"{path} covers {count} frames; the maximum is {MAX_SCENE_FRAMES} "
            f"(scene {index})")
    return SceneWindow(start_frame=start, end_frame=end,
                       note=_text(data.get("note", ""), f"{path}.note"))


def _parse_scene_keyframes(block: Any, path: str, *, stroke_type: str,
                           window: SceneWindow) -> Tuple[Dict[str, int], Dict[str, int]]:
    vocab = STROKE_TYPE_PHASES[stroke_type]
    data = _mapping(block, path)
    keyframes: Dict[str, int] = {}
    for name, value in data.items():
        if str(name).startswith("_"):
            continue
        if name not in vocab:
            raise DemoConfigError(
                f"{path}.{name} is not a phase of a {stroke_type!r} scene; its phases "
                f"are {list(vocab)}. A putt has no TOP: name the phases this stroke "
                f"actually has.")
        frame = _whole(value, f"{path}.{name}", minimum=0)
        if not window.contains(frame):
            raise DemoConfigError(
                f"{path}.{name} = {frame} lies outside the scene window "
                f"[{window.start_frame}, {window.end_frame}]; a scene may only mark "
                f"frames inside its own visible window")
        keyframes[name] = frame
    canonical: Dict[str, int] = {}
    for name, frame in keyframes.items():
        canonical[PHASE_SLOT[name]] = frame
    return keyframes, canonical


def _enforce_anchor_spacing(cfg: DemoConfig, *, path: str, contact_frame: Optional[int]) -> List[dict]:
    """Cap operator anchor spacing, harder close to contact.

    A clubhead crossing 60 frames of a 30 fps downswing is not a slow swing: it is a
    gap the operator never verified. The demo validator allows up to 240 frames between
    anchors; the broadcast profile refuses anything past
    :data:`BROADCAST_MAX_ANCHOR_SPACING`, and past :data:`BROADCAST_CONTACT_SPACING`
    within +/- :data:`BROADCAST_CONTACT_WINDOW_FRAMES` frames of the contact mark.
    Both the configured and the effective value are returned so the run can record them.
    """
    records: List[dict] = []
    for name, block in sorted(cfg.manual_segments.items()):
        for segment in block.segments:
            configured = int(segment.max_anchor_spacing_frames)
            effective = configured
            near_contact = (contact_frame is not None and any(
                abs(int(a.frame) - contact_frame) <= BROADCAST_CONTACT_WINDOW_FRAMES
                for a in segment.anchors))
            if near_contact:
                effective = min(effective, BROADCAST_CONTACT_SPACING)
            effective = min(effective, BROADCAST_MAX_ANCHOR_SPACING)
            if configured > effective:
                limit = ("BROADCAST_CONTACT_SPACING"
                         if near_contact and effective == BROADCAST_CONTACT_SPACING
                         else "BROADCAST_MAX_ANCHOR_SPACING")
                raise DemoConfigError(
                    f"{path}.manual_segments.{name}.{segment.name}."
                    f"max_anchor_spacing_frames = {configured} exceeds the broadcast cap "
                    f"of {effective} ({limit})"
                    + (" within "
                       f"{BROADCAST_CONTACT_WINDOW_FRAMES} frames of the contact mark"
                       if near_contact else "")
                    + ". A gap that long is an invented stretch, not a tracked club; "
                      "refused at parse time.")
            records.append({"target": name, "segment": segment.name,
                            "configured": configured, "effective": effective,
                            "near_contact": bool(near_contact)})
    return records


def _scene_demo_payload(scene_block: Mapping[str, Any], *, path: str,
                        window: SceneWindow, stroke_type: str,
                        canonical_keyframes: Dict[str, int]) -> DemoConfig:
    """Build the demo payload for one scene out of its demo-shaped keys.

    Everything in :data:`_DEMO_SCENE_KEYS` is handed to the demo parser unchanged, so
    ROI interpolation, manual anchors, target options and interpolation limits are all
    validated by the code that already validates them.
    """
    payload: Dict[str, Any] = {
        "schema_version": "fairwayos-demo-config.v1",
        "clip_path_hint": "",
        # A scene carries its phases as canonical slots; the display vocabulary lives
        # on the scene. ``impact`` is the canonical name of ``contact``.
        "keyframes": dict(canonical_keyframes),
    }
    for key in _DEMO_SCENE_KEYS:
        if key in scene_block:
            payload[key] = scene_block[key]
    if stroke_type == "putt":
        # The only approximate phase the demo can derive is TOP, from the minimum wrist
        # height. A putt has no top, so the derivation is switched off for real rather
        # than relabelled: a putt's phases are the operator's marks or nothing.
        payload["allow_approx_phases"] = False
    cfg = parse_config(payload)

    # The demo validator has no idea about scene windows, so the frame bounds are
    # checked here, against the same window, for every operator frame the scene holds.
    for name, point in cfg.seeds.items():
        if not window.contains(point.frame):
            raise DemoConfigError(
                f"{path}.seeds.{name}.frame = {point.frame} lies outside the scene "
                f"window [{window.start_frame}, {window.end_frame}]")
    for name, points in cfg.corrections.items():
        for point in points:
            if not window.contains(point.frame):
                raise DemoConfigError(
                    f"{path}.corrections.{name} has frame {point.frame} outside the "
                    f"scene window [{window.start_frame}, {window.end_frame}]")
    for name, block in cfg.manual_segments.items():
        for segment in block.segments:
            for anchor in segment.anchors:
                if not window.contains(anchor.frame):
                    raise DemoConfigError(
                        f"{path}.manual_segments.{name}.{segment.name} has anchor frame "
                        f"{anchor.frame} outside the scene window "
                        f"[{window.start_frame}, {window.end_frame}]")
        for hidden in block.hide_frames:
            for frame in range(int(hidden[0]), int(hidden[1]) + 1):
                if not window.contains(frame):
                    raise DemoConfigError(
                        f"{path}.manual_segments.{name}.hide_frames={hidden} leaves the "
                        f"scene window [{window.start_frame}, {window.end_frame}]")
    for keyframe in cfg.roi_keyframes:
        if not window.contains(keyframe.frame):
            raise DemoConfigError(
                f"{path}.roi_keyframes has a box at frame {keyframe.frame} outside the "
                f"scene window [{window.start_frame}, {window.end_frame}]")
    for name, options in cfg.targets.items():
        for field_name in ("track_until_frame", "automatic_until_frame"):
            value = getattr(options, field_name)
            if value is not None and not window.contains(value):
                raise DemoConfigError(
                    f"{path}.targets.{name}.{field_name} = {value} lies outside the "
                    f"scene window [{window.start_frame}, {window.end_frame}]")
    return cfg


def parse_broadcast(payload: Any) -> BroadcastConfig:
    """Turn a decoded JSON object into a BroadcastConfig. Unknown fields are refused."""
    data = _mapping(payload, "config")
    if not data:
        raise DemoConfigError("config is empty")
    _reject_unknown(data, _BROADCAST_FIELDS, "config")
    version = data.get("schema_version", BROADCAST_CONFIG_SCHEMA)
    if version != BROADCAST_CONFIG_SCHEMA:
        raise DemoConfigError(
            f"config schema_version must be {BROADCAST_CONFIG_SCHEMA!r}, got "
            f"{version!r}. A single-interval demo/showcase config uses "
            f"'fairwayos-demo-config.v1' and the demo-video command instead.")

    shot_block = _mapping(data.get("shot"), "shot")
    _reject_unknown(shot_block, _SHOT_FIELDS, "shot")
    stroke_type = _text(shot_block.get("stroke_type", "unknown"),
                        "shot.stroke_type")
    if stroke_type not in STROKE_TYPES:
        raise DemoConfigError(
            f"shot.stroke_type must be one of {list(STROKE_TYPES)}, got "
            f"{stroke_type!r}")

    interval = _mapping(data.get("interval"), "interval")
    _reject_unknown(interval, ("start_seconds", "end_seconds", "max_frames"),
                    "interval")
    start_seconds = _finite(interval.get("start_seconds", 0.0),
                            "interval.start_seconds", minimum=0.0)
    end_raw = interval.get("end_seconds", None)
    end_seconds = (None if end_raw is None else
                   _finite(end_raw, "interval.end_seconds", minimum=0.0))
    if end_seconds is not None and end_seconds <= start_seconds:
        raise DemoConfigError("interval.end_seconds must be after start_seconds")
    max_frames = _whole(interval.get("max_frames", 1200), "interval.max_frames",
                        minimum=1)

    raw_scenes = data.get("scenes")
    if not isinstance(raw_scenes, (list, tuple)):
        raise DemoConfigError("scenes must be a list of scene objects")
    if not raw_scenes:
        raise DemoConfigError(
            "scenes is empty: a broadcast config with no scene draws nothing. Supply "
            "at least one golfer-visible window, or use demo-video for the whole clip.")
    if len(raw_scenes) > MAX_SCENES:
        raise DemoConfigError(f"scenes has {len(raw_scenes)} entries; at most "
                              f"{MAX_SCENES} are allowed")

    scenes: List[SceneConfig] = []
    seen_ids: Dict[str, int] = {}
    spacing_records: List[dict] = []
    for index, item in enumerate(raw_scenes):
        path = f"scenes[{index}]"
        block = _mapping(item, path)
        _reject_unknown(block, _SCENE_KEYS + _DEMO_SCENE_KEYS, path)
        scene_id = _text(block.get("id"), f"{path}.id", allow_empty=False)
        if scene_id in seen_ids:
            raise DemoConfigError(
                f"{path}.id = {scene_id!r} repeats scenes[{seen_ids[scene_id]}].id; "
                f"scene ids name output artifacts, so they must be unique")
        seen_ids[scene_id] = index
        window = _parse_window(block.get("window"), f"{path}.window", index=index)
        kind = _text(block.get("kind", "golfer_stroke"), f"{path}.kind")
        if kind not in SCENE_KINDS:
            raise DemoConfigError(
                f"{path}.kind must be one of {list(SCENE_KINDS)}, got {kind!r}")
        if kind != "golfer_stroke":
            extra = sorted(k for k in block
                           if not str(k).startswith("_")
                           and k not in _NON_STROKE_SCENE_KEYS)
            if extra:
                raise DemoConfigError(
                    f"{path} has kind={kind!r} but also carries {extra}. A "
                    f"reaction/flight/green/replay window has no golfer to track, so "
                    f"seeds, targets, anchors, ROI or phase marks on it would be "
                    f"tracking nothing. Remove them, or declare the scene a "
                    f"golfer_stroke.")
            scenes.append(SceneConfig(
                scene_id=scene_id, label=_text(block.get("label", scene_id),
                                               f"{path}.label"),
                stroke_type="unknown", window=window, keyframes={},
                canonical_keyframes={}, demo=parse_config(
                    {"schema_version": "fairwayos-demo-config.v1"}),
                kind=kind, metrics=(), notes=_text(block.get("notes", ""),
                                                   f"{path}.notes")))
            continue
        scene_type = _text(block.get("stroke_type", stroke_type),
                           f"{path}.stroke_type")
        if scene_type not in STROKE_TYPES:
            raise DemoConfigError(
                f"{path}.stroke_type must be one of {list(STROKE_TYPES)}, got "
                f"{scene_type!r}")
        keyframes, canonical = _parse_scene_keyframes(
            block.get("keyframes"), f"{path}.keyframes", stroke_type=scene_type,
            window=window)
        demo = _scene_demo_payload(block, path=path, window=window,
                                   stroke_type=scene_type,
                                   canonical_keyframes=canonical)
        contact = keyframes.get(CONTACT_NAMES[scene_type])
        spacing_records += _enforce_anchor_spacing(demo, path=path,
                                                   contact_frame=contact)
        metrics: Tuple[str, ...] = ()
        if block.get("metrics") is not None:
            parsed = parse_showcase({"metrics": block["metrics"]},
                                    f"{path}.metrics")
            metrics = tuple(parsed.metrics)
        hidden = []
        for index_range, pair in enumerate(_mapping_list(block.get("metric_hide_frames", []),
                                                 f"{path}.metric_hide_frames")):
            if not isinstance(pair, list) or len(pair) != 2:
                raise DemoConfigError(f"{path}.metric_hide_frames[{index_range}] must be [first, last]")
            first, last = (_whole(v, f"{path}.metric_hide_frames[{index_range}]") for v in pair)
            if first > last or not window.contains(first) or not window.contains(last):
                raise DemoConfigError(f"{path}.metric_hide_frames must be ascending ranges inside the scene")
            if hidden and first <= hidden[-1][1]:
                raise DemoConfigError(f"{path}.metric_hide_frames ranges must not overlap")
            hidden.append((first, last))
        scenes.append(SceneConfig(
            scene_id=scene_id, label=_text(block.get("label", scene_id),
                                           f"{path}.label"),
            stroke_type=scene_type, window=window, keyframes=keyframes,
            canonical_keyframes=canonical, demo=demo, kind=kind, metrics=metrics,
            metric_hide_frames=tuple(hidden),
            notes=_text(block.get("notes", ""), f"{path}.notes")))

    ordered = sorted(scenes, key=lambda s: s.window.start_frame)
    for previous, current in zip(ordered, ordered[1:]):
        if current.window.start_frame <= previous.window.end_frame:
            raise DemoConfigError(
                f"scenes {previous.scene_id!r} [{previous.window.start_frame}, "
                f"{previous.window.end_frame}] and {current.scene_id!r} "
                f"[{current.window.start_frame}, {current.window.end_frame}] overlap. "
                f"A frame belongs to at most one scene: overlapping windows would let "
                f"one scene's tracks, cards or hand_rise reference carry into another, "
                f"which is exactly what the broadcast contract forbids.")
    if [s.scene_id for s in ordered] != [s.scene_id for s in scenes]:
        raise DemoConfigError(
            "scenes must be listed in ascending window order; the list is refused "
            "rather than sorted so a mis-ordered file cannot silently change which "
            "scene owns a frame")

    presentation = _parse_presentation(data.get("presentation"), "presentation")

    outside_windows = _text(data.get("outside_windows", "brand_only"),
                            "outside_windows")
    if outside_windows not in OUTSIDE_WINDOW_MODES:
        raise DemoConfigError(
            f"outside_windows must be one of {list(OUTSIDE_WINDOW_MODES)}, got "
            f"{outside_windows!r}")

    cuts: List[CutCandidate] = []
    for index, item in enumerate(_mapping_list(data.get("cuts"), "cuts")):
        path = f"cuts[{index}]"
        block = _mapping(item, path)
        _reject_unknown(block, _CUT_FIELDS, path)
        kind = _text(block.get("kind", "hard_cut"), f"{path}.kind")
        if kind not in _CUT_KINDS:
            raise DemoConfigError(f"{path}.kind must be one of {list(_CUT_KINDS)}")
        span_raw = block.get("span")
        span: Optional[List[int]] = None
        if span_raw is not None:
            if not isinstance(span_raw, (list, tuple)) or len(span_raw) != 2:
                raise DemoConfigError(
                    f"{path}.span must be [first_frame, last_frame] for a dissolve")
            span = [_whole(span_raw[0], f"{path}.span[0]", minimum=0),
                    _whole(span_raw[1], f"{path}.span[1]", minimum=0)]
            if span[1] < span[0]:
                raise DemoConfigError(f"{path}.span is reversed: {span}")
        if kind == "dissolve" and span is None:
            raise DemoConfigError(
                f"{path} is a dissolve with no span: a blend has a width, and a "
                f"one-frame blend is a hard cut. Say which frames it covers.")
        if kind == "hard_cut" and span is not None:
            raise DemoConfigError(
                f"{path} is a hard cut with a span; a hard cut is one frame boundary")
        decision = _text(block.get("decision", "proposed"), f"{path}.decision")
        if decision not in _CUT_DECISIONS:
            raise DemoConfigError(
                f"{path}.decision must be one of {list(_CUT_DECISIONS)}")
        cuts.append(CutCandidate(
            frame=_whole(block.get("frame"), f"{path}.frame", minimum=0),
            kind=kind, span=span,
            confidence=_finite(block.get("confidence", 0.0), f"{path}.confidence",
                               minimum=0.0, maximum=1.0),
            method=_text(block.get("method", ""), f"{path}.method"),
            note=_text(block.get("note", ""), f"{path}.note"),
            decision=decision,
            decided_by=_text(block.get("decided_by", ""), f"{path}.decided_by"),
            decided_at=_text(block.get("decided_at", ""), f"{path}.decided_at")))
    frames = [c.frame for c in cuts]
    if frames != sorted(frames):
        raise DemoConfigError("cuts must be in ascending frame order")

    overrides: List[dict] = []
    for index, item in enumerate(_mapping_list(data.get("overrides"), "overrides")):
        path = f"overrides[{index}]"
        block = _mapping(item, path)
        _reject_unknown(block, _OVERRIDE_FIELDS, path)
        decision = _text(block.get("decision"), f"{path}.decision", allow_empty=False)
        if decision not in ("accept", "reject"):
            raise DemoConfigError(
                f"{path}.decision must be 'accept' or 'reject'; a proposal is never "
                f"applied without the operator saying so")
        overrides.append({"frame": _whole(block.get("frame"), f"{path}.frame",
                                          minimum=0),
                          "decision": decision,
                          "note": _text(block.get("note", ""), f"{path}.note"),
                          "decided_by": _text(block.get("decided_by", ""),
                                              f"{path}.decided_by"),
                          "decided_at": _text(block.get("decided_at", ""),
                                              f"{path}.decided_at")})

    render = _mapping(data.get("render"), "render")
    _reject_unknown(render, ("max_width", "crf", "preset", "render"), "render")
    timing = _mapping(data.get("timing"), "timing")
    _reject_unknown(timing, ("source_cadence_caveat", "nominal_fps_trusted",
                             "accept_variable_cadence"), "timing")

    expected_sha = _text(shot_block.get("expected_source_sha256", ""),
                         "shot.expected_source_sha256")
    if expected_sha:
        import re
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise DemoConfigError(
                "shot.expected_source_sha256 must be 64 lowercase hex characters, or "
                "omitted. A malformed digest is refused rather than compared.")

    # The parent-source manifest is a path the operator names. It may be absolute or
    # relative to the config file; it is resolved (and checked to exist) at provenance
    # time, where a manifest that cannot be read is reported as an absence rather than
    # failing a render that is otherwise sound.
    parent_manifest = _text(shot_block.get("parent_source_manifest", ""),
                            "shot.parent_source_manifest")
    source_notes = _text(shot_block.get("source_notes", ""), "shot.source_notes")
    expected_fps_raw = shot_block.get("expected_fps")
    expected_fps = (None if expected_fps_raw is None else
                    _finite(expected_fps_raw, "shot.expected_fps", minimum=0.0))
    if expected_fps is not None and expected_fps <= 0:
        raise DemoConfigError("shot.expected_fps must be greater than zero")
    expected_frames_raw = shot_block.get("expected_frames")
    expected_frames = (None if expected_frames_raw is None else
                       _whole(expected_frames_raw, "shot.expected_frames", minimum=1))

    return BroadcastConfig(
        schema_version=version,
        clip_path_hint=_text(data.get("clip_path_hint", ""), "clip_path_hint"),
        shot_id=_text(shot_block.get("id", ""), "shot.id"),
        shot_label=_text(shot_block.get("label", ""), "shot.label"),
        stroke_type=stroke_type,
        expected_source_sha256=expected_sha,
        parent_source_manifest=parent_manifest,
        source_notes=source_notes,
        expected_fps=expected_fps,
        expected_frames=expected_frames,
        start_seconds=start_seconds, end_seconds=end_seconds, max_frames=max_frames,
        scenes=scenes, presentation=presentation, outside_windows=outside_windows,
        cuts=cuts, overrides=overrides, spacing_records=spacing_records,
        render={k: render[k] for k in render if not str(k).startswith("_")},
        timing={k: timing[k] for k in timing if not str(k).startswith("_")},
        source_description=_text(data.get("source_description", ""),
                                 "source_description"),
        notes=_text(data.get("notes", ""), "notes"),
        raw=dict(data))


def _mapping_list(value: Any, path: str) -> List[Any]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise DemoConfigError(f"{path} must be a list")
    return list(value)


def _parse_presentation(block: Any, path: str) -> BroadcastPresentation:
    data = _mapping(block, path)
    _reject_unknown(data, _PRESENTATION_FIELDS, path)
    defaults = BroadcastPresentation()
    metrics = defaults.metrics
    if data.get("metrics") is not None:
        metrics = tuple(parse_showcase({"metrics": data["metrics"]},
                                       f"{path}.metrics").metrics)
    return BroadcastPresentation(
        brand=_text(data.get("brand", defaults.brand), f"{path}.brand"),
        brand_sub=_text(data.get("brand_sub", defaults.brand_sub),
                        f"{path}.brand_sub"),
        disclosure=_text(data.get("disclosure", defaults.disclosure),
                         f"{path}.disclosure"),
        metrics=metrics,
        card_debounce_frames=_whole(data.get("card_debounce_frames",
                                             defaults.card_debounce_frames),
                                    f"{path}.card_debounce_frames", minimum=1,
                                    maximum=600),
        card_row_max_width=None if data.get("card_row_max_width") is None else _whole(
            data.get("card_row_max_width"), f"{path}.card_row_max_width", minimum=64),
        ball_trail_points=_whole(data.get("ball_trail_points",
                                          defaults.ball_trail_points),
                                 f"{path}.ball_trail_points", minimum=1, maximum=240),
        club_trail_points=_whole(data.get("club_trail_points",
                                          defaults.club_trail_points),
                                 f"{path}.club_trail_points", minimum=1, maximum=240),
        impact_accent=_boolean(data.get("impact_accent", defaults.impact_accent),
                               f"{path}.impact_accent"),
        impact_accent_span_frames=_whole(
            data.get("impact_accent_span_frames", defaults.impact_accent_span_frames),
            f"{path}.impact_accent_span_frames", minimum=0, maximum=240),
        card_position=_card_position(data.get("card_position",
                                              defaults.card_position),
                                     f"{path}.card_position"),
        metrics_required=_boolean(data.get("metrics_required",
                                           defaults.metrics_required),
                                  f"{path}.metrics_required"))


def _card_position(value: Any, path: str) -> str:
    text = _text(value, path)
    if text not in ("auto", "top", "bottom"):
        raise DemoConfigError(
            f"{path} must be 'auto', 'top' or 'bottom', got {text!r}")
    return text


def load_broadcast(path: str) -> BroadcastConfig:
    import json
    from pathlib import Path as _Path

    p = _Path(path).expanduser()
    if not p.is_file():
        raise DemoConfigError(f"config file does not exist: {path}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DemoConfigError(f"config is not valid JSON: {exc}") from exc
    return parse_broadcast(payload)


# ---------------------------------------------------------------- resolution

@dataclass
class ResolvedScene:
    scene: SceneConfig
    start_frame: int
    end_frame: int
    frame_count: int
    warnings: List[str] = field(default_factory=list)

    def frames(self) -> range:
        return range(self.start_frame, self.end_frame + 1)


@dataclass
class ResolvedBroadcast:
    config: BroadcastConfig
    source: Any
    start_frame: int
    end_frame: int
    frame_count: int
    scenes: List[ResolvedScene] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    frame_times: Optional[List[float]] = None


def validate_broadcast(cfg: BroadcastConfig, probe) -> ResolvedBroadcast:
    """Bind a parsed broadcast config to a real source and check the frame bounds.

    Mirrors :func:`ghostcaddie.demo.config.validate_config` where it applies, and adds
    the two things a broadcast run needs on top: every scene window must lie inside the
    rendered interval, and every scene must actually see pixels.
    """
    from .config import _frame_for_seconds  # single implementation of the mapping

    if cfg.expected_source_sha256:
        pass  # the digest is checked against the real file by the runner
    warnings: List[str] = []
    fps = float(probe.fps)
    last = probe.last_frame_index
    if last is None:
        raise DemoConfigError(
            "the source's frame count could not be determined, so a broadcast window "
            "cannot be bound to it")
    if probe.variable_cadence:
        raise DemoConfigError(
            "the source cadence is not constant, so the frames in a scene window "
            "cannot be located reliably; the broadcast workflow renders at source "
            "cadence and refuses this rather than guessing")
    if getattr(probe, "irregular_timestamps", False):
        # r_frame_rate == avg_frame_rate does NOT prove a constant cadence: an
        # irregular stream can average to exactly the nominal rate. The decoded frame
        # timestamps are the real test, so they are what this refusal uses.
        raise DemoConfigError(
            "the source's decoded frame timestamps are irregular ("
            + (probe.timestamp_note or "no note") + "), so a scene window cannot be "
            "bound to reliable frame positions; a broadcast render encodes at a "
            "constant cadence and would silently re-time this footage")
    start_frame = _frame_for_seconds(cfg.start_seconds, fps, upper=False)
    end_frame = (_frame_for_seconds(cfg.end_seconds, fps, upper=True) if
                 cfg.end_seconds is not None else last)
    end_frame = min(end_frame, last)
    if end_frame < start_frame:
        raise DemoConfigError(
            f"the interval [{cfg.start_seconds}, {cfg.end_seconds}] is empty in this "
            f"source ({start_frame}..{end_frame})")
    frame_count = end_frame - start_frame + 1
    if frame_count > cfg.max_frames:
        raise DemoConfigError(
            f"the interval covers {frame_count} frames, more than "
            f"interval.max_frames={cfg.max_frames}. That is a hard refusal, not a "
            f"truncation: raise the cap deliberately or narrow the interval.")

    resolved_scenes: List[ResolvedScene] = []
    #: A window may not cross an ACCEPTED cut: the frames either side of a hard cut are
    #: different camera and different subject, so one scene would be tracking two. A
    #: proposal that the operator did not accept is only a suggestion and does not bind.
    cut_points: List[Tuple[str, int]] = []
    for cut in cfg.cuts:
        if cut.decision not in ("accepted", "operator_added"):
            continue
        if cut.kind == "dissolve" and cut.span:
            cut_points.append(("dissolve", int(cut.span[0])))
            cut_points.append(("dissolve", int(cut.span[1])))
        else:
            cut_points.append((cut.kind, int(cut.frame)))
    for index, item in enumerate(cfg.overrides):
        if item.get("decision") == "accept":
            cut_points.append(("override", int(item["frame"])))
    for scene in cfg.scenes:
        window = scene.window
        for kind, point in cut_points:
            if window.start_frame < point <= window.end_frame:
                raise DemoConfigError(
                    f"scene {scene.scene_id!r} window [{window.start_frame}, "
                    f"{window.end_frame}] crosses the accepted cut at frame {point} "
                    f"({kind}). Split the scene at the cut: the frames either side are "
                    f"not the same shot, and a window that spans both would carry the "
                    f"first side's tracking into the second.")
        if window.end_frame > last:
            raise DemoConfigError(
                f"scene {scene.scene_id!r} window ends at frame {window.end_frame}, "
                f"past the source's last frame {last}")
        if window.start_frame < start_frame or window.end_frame > end_frame:
            raise DemoConfigError(
                f"scene {scene.scene_id!r} window "
                f"[{window.start_frame}, {window.end_frame}] lies outside the rendered "
                f"interval [{start_frame}, {end_frame}]")
        resolved_scenes.append(ResolvedScene(
            scene=scene, start_frame=window.start_frame, end_frame=window.end_frame,
            frame_count=window.frame_count))
    covered = sum(s.frame_count for s in resolved_scenes)
    if covered == 0:  # pragma: no cover - scenes cannot be empty by construction
        raise DemoConfigError("no scene covers any frame of the interval")
    warnings.append(
        f"{len(resolved_scenes)} scene window(s) cover {covered} of {frame_count} "
        f"frames; the other {frame_count - covered} frames are rendered with no "
        f"overlay and no chrome.")
    return ResolvedBroadcast(config=cfg, source=probe, start_frame=start_frame,
                             end_frame=end_frame, frame_count=frame_count,
                             scenes=resolved_scenes, warnings=warnings,
                             frame_times=None)
