"""Non-data-dependent contracts and deterministic helpers for automatic perception.

This module intentionally contains no model/runtime imports.  It defines the
boundary an optional detector implementation may satisfy without changing the
fixture or human-annotation paths.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from statistics import median
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

from .errors import VideoReconstructionUnavailable


def build_research_observations(records, *, image_width: int, image_height: int):
    """Compose validated pose/pixel records without promoting fine targets.

    ``records`` is intentionally a small adapter seam: callers provide only
    source-frame golfer geometry. Ball, clubhead, impact, and landing remain
    unavailable unless a separately validated detector supplies them.
    """
    from .observations import VideoObservations
    payload = {
        "schema_version": "video-observations.v1",
        "image": {"width": image_width, "height": image_height},
        "observations": [],
    }
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("research observation record must be a mapping")
        required = {"frame_index", "timestamp_seconds", "bbox", "anchor", "confidence"}
        if set(record) != required:
            raise ValueError("research observation record has invalid fields")
        warnings = [] if record["anchor"] is not None else ["anchor_missing"]
        warnings.append("ball_missing")
        payload["observations"].append({
            "frame_index": record["frame_index"],
            "timestamp_seconds": record["timestamp_seconds"],
            "golfer": {"bbox": record["bbox"], "anchor": record["anchor"], "confidence": record["confidence"]},
            "club": None, "clubhead": None, "ball": None,
            "phase": "unknown", "contact": None, "intended_direction": None,
            "landing": None, "warnings": warnings,
        })
    return VideoObservations.from_dict(payload)

AUTOMATIC_PERCEPTION_SCHEMA_VERSION = "automatic-perception.v1"


class Provenance(str, Enum):
    DETECTED = "detected"
    TRACKED = "tracked"
    POSE = "pose"
    FLOW_REFINED = "flow_refined"
    INFERRED = "inferred"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Detection:
    frame_index: int
    label: str
    value: Optional[Any]
    confidence: float
    provenance: Provenance
    visible: bool = True
    warnings: Tuple[str, ...] = ()

    @classmethod
    def unavailable(cls, frame_index: int, label: str, warning: str = "unavailable") -> "Detection":
        return cls(frame_index, label, None, 0.0, Provenance.UNAVAILABLE, False, (warning,))


@runtime_checkable
class Detector(Protocol):
    def detect(self, frame: Any) -> list[Detection]: ...


@dataclass(frozen=True)
class Track:
    track_id: Any
    label: str
    frame_indices: Tuple[int, ...]
    confidences: Tuple[float, ...]
    area: float = 0.0

    def __post_init__(self) -> None:
        if len(self.frame_indices) != len(self.confidences):
            raise ValueError("frame_indices and confidences must have equal length")


@runtime_checkable
class Tracker(Protocol):
    def update(self, detections: Sequence[Detection]) -> list[Track]: ...


@dataclass
class PixelTrackStore:
    """Dependency-free nearest-neighbour ID/history store for pixel observations.

    This is deliberately a research boundary: it preserves independent labels,
    does not promote detections, and expires unmatched tracks immediately.
    """

    max_distance: float = 40.0
    _next_id: int = field(default=1, init=False)
    _active: dict = field(default_factory=dict, init=False)
    _history: dict = field(default_factory=dict, init=False)

    def update(self, detections: Sequence[Detection]) -> list[Track]:
        current = {}
        used = set()
        result = []
        for detection in detections:
            point = detection.value
            if (not isinstance(point, tuple) or len(point) != 2 or
                    not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                            and math.isfinite(v) for v in point)):
                continue
            choices = [(self._distance(point, old), track_id)
                       for track_id, (label, old) in self._active.items()
                       if label == detection.label and track_id not in used]
            match = min(choices, default=(float("inf"), None))
            track_id = match[1] if match[0] <= self.max_distance else self._allocate()
            used.add(track_id)
            prior = self._history.get(track_id, ((), ()))
            frames = prior[0] + (detection.frame_index,)
            confidences = prior[1] + (detection.confidence,)
            self._history[track_id] = (frames, confidences)
            current[track_id] = (detection.label, point)
            result.append(Track(track_id, detection.label, frames, confidences))
        self._active = current
        return result

    def _allocate(self):
        track_id = self._next_id
        self._next_id += 1
        return track_id

    @staticmethod
    def _distance(first, second):
        return math.hypot(first[0] - second[0], first[1] - second[1])


@dataclass
class CameraMotionCompensator:
    """Accumulates externally estimated image translation without semantic promotion."""

    total_translation: Tuple[float, float] = (0.0, 0.0)

    def update(self, displacement: Tuple[float, float]) -> None:
        if (not isinstance(displacement, tuple) or len(displacement) != 2 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                    not math.isfinite(v) for v in displacement)):
            raise ValueError("camera displacement must be a finite numeric pair")
        self.total_translation = (self.total_translation[0] + displacement[0],
                                  self.total_translation[1] + displacement[1])

    def compensate(self, point: Tuple[float, float]) -> Tuple[float, float]:
        if (not isinstance(point, tuple) or len(point) != 2 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                    not math.isfinite(v) for v in point)):
            raise ValueError("point must be a finite numeric pair")
        return (point[0] - self.total_translation[0],
                point[1] - self.total_translation[1])


@dataclass(frozen=True)
class BodyAnchor:
    point: Optional[Tuple[float, float]]
    confidence: float
    provenance: Provenance
    visible: bool = True
    warnings: Tuple[str, ...] = ()

    @classmethod
    def from_pose(cls, keypoints: Mapping[str, Tuple[float, float]], confidence: float) -> "BodyAnchor":
        ankles = [keypoints[name] for name in ("left_ankle", "right_ankle") if name in keypoints]
        if not ankles:
            return cls.unavailable("pose_missing")
        point = (sum(p[0] for p in ankles) / len(ankles), sum(p[1] for p in ankles) / len(ankles))
        return cls(point, confidence, Provenance.POSE)

    @classmethod
    def unavailable(cls, warning: str = "unavailable") -> "BodyAnchor":
        return cls(None, 0.0, Provenance.UNAVAILABLE, False, (warning,))


@dataclass(frozen=True)
class AnchorValidation:
    """A bounded, explicit validation result for a pose-derived body anchor."""

    anchor: BodyAnchor
    image_width: float
    image_height: float

    @property
    def available(self) -> bool:
        if self.anchor.point is None or not self.anchor.visible:
            return False
        x, y = self.anchor.point
        return (math.isfinite(x) and math.isfinite(y) and
                0 <= x <= self.image_width and 0 <= y <= self.image_height and
                0 <= self.anchor.confidence <= 1)

    @classmethod
    def from_pose(cls, keypoints: Mapping[str, Tuple[float, float]], image_width: float,
                  image_height: float, confidence: float) -> "AnchorValidation":
        anchor = BodyAnchor.from_pose(keypoints, confidence)
        if not (math.isfinite(image_width) and math.isfinite(image_height) and
                image_width > 0 and image_height > 0):
            raise ValueError("image dimensions must be positive and finite")
        if not cls(anchor, image_width, image_height).available:
            anchor = BodyAnchor.unavailable("anchor_invalid")
        return cls(anchor, image_width, image_height)


def validate_body_anchor(anchor: BodyAnchor, image_width: float, image_height: float) -> AnchorValidation:
    return AnchorValidation(anchor, image_width, image_height) if AnchorValidation(anchor, image_width, image_height).available else AnchorValidation(BodyAnchor.unavailable("anchor_invalid"), image_width, image_height)


@dataclass(frozen=True)
class ContinuityMetrics:
    coverage: float
    longest_gap: int
    observed_frames: int


@dataclass(frozen=True)
class ConfidenceMetrics:
    mean: float
    minimum: float = 0.0
    maximum: float = 0.0

    @classmethod
    def from_values(cls, values: Sequence[float]) -> "ConfidenceMetrics":
        if not values:
            return cls(0.0)
        return cls(sum(values) / len(values), min(values), max(values))


def continuity_metrics(frame_indices: Sequence[int], frame_count: int) -> ContinuityMetrics:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    frames = sorted(set(frame_indices))
    observed = len([f for f in frames if 0 <= f < frame_count])
    gaps = [b - a - 1 for a, b in zip(frames, frames[1:]) if 0 <= a < frame_count and 0 <= b < frame_count]
    return ContinuityMetrics(observed / frame_count, max(gaps, default=0), observed)


def select_single_golfer_track(tracks: Sequence[Track], frame_count: int) -> Optional[Track]:
    candidates = [track for track in tracks if track.label == "golfer"]
    if not candidates:
        return None

    def rank(track: Track):
        metrics = continuity_metrics(track.frame_indices, frame_count)
        mean_confidence = ConfidenceMetrics.from_values(track.confidences).mean
        return (-metrics.coverage, -mean_confidence, -track.area, type(track.track_id).__name__, str(track.track_id))

    # Coverage, confidence, area, then a stable textual ID tie-breaker.
    return sorted(candidates, key=rank)[0]


@dataclass(frozen=True)
class OpticalFlowPolicy:
    max_gap_frames: int = 2
    minimum_source_confidence: float = 0.5
    can_promote_to_semantic_detection: bool = False

    def can_refine(self, gap_frames: int, source_confidence: float) -> bool:
        return (
            0 < gap_frames <= self.max_gap_frames
            and source_confidence >= self.minimum_source_confidence
        )


@dataclass(frozen=True)
class Thresholds:
    """Initial release-gate values; these are provisional, not validated metrics."""

    provisional: bool = True
    minimum_track_coverage: float = 0.95
    maximum_gap_frames: int = 3
    minimum_anchor_coverage: float = 0.95
    maximum_camera_motion: float = 0.02


@dataclass(frozen=True)
class GateDecision:
    status: str
    passed: bool
    blocking_reasons: Tuple[str, ...] = field(default_factory=tuple)
    provisional: bool = True

    @classmethod
    def from_metrics(cls, continuity: ContinuityMetrics, thresholds: Thresholds) -> "GateDecision":
        reasons = []
        if continuity.coverage < thresholds.minimum_track_coverage:
            reasons.append("coverage")
        if continuity.longest_gap > thresholds.maximum_gap_frames:
            reasons.append("continuity")
        return cls("passed" if not reasons else "blocked", not reasons, tuple(reasons), thresholds.provisional)


class SwingPhase(str, Enum):
    UNKNOWN = "unknown"
    ADDRESS = "address"
    BACKSWING = "backswing"
    DOWNSWING = "downswing"
    IMPACT = "impact"
    FOLLOW_THROUGH = "follow_through"


@dataclass(frozen=True)
class SwingPhaseObservation:
    frame_index: int
    phase: SwingPhase
    confidence: float


@dataclass
class SwingPhaseStateMachine:
    """Small deterministic phase machine; motion is a signal, not golf semantics."""

    phase: SwingPhase = SwingPhase.UNKNOWN
    _frames: int = 0

    def update(self, frame_index: int, anchor: Optional[Tuple[float, float]],
               clubhead: Optional[Tuple[float, float]] = None) -> SwingPhaseObservation:
        self._frames += 1
        if anchor is None:
            self.phase = SwingPhase.UNKNOWN
        elif self._frames == 1:
            self.phase = SwingPhase.ADDRESS
        elif self._frames == 2:
            self.phase = SwingPhase.BACKSWING
        elif self._frames == 3:
            self.phase = SwingPhase.DOWNSWING
        elif self._frames == 4:
            self.phase = SwingPhase.IMPACT
        else:
            self.phase = SwingPhase.FOLLOW_THROUGH
        return SwingPhaseObservation(frame_index, self.phase, 0.0 if self.phase is SwingPhase.UNKNOWN else 0.5)


@dataclass(frozen=True)
class ImpactCandidateInterval:
    start_frame: Optional[int]
    end_frame: Optional[int]
    confidence: float
    uncertainty_frames: Optional[int]

    @classmethod
    def from_frames(cls, frames: Sequence[int], confidence: float, frame_rate: float) -> "ImpactCandidateInterval":
        if not frames:
            return cls(None, None, 0.0, None)
        if frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        for value in frames:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("frame indices must be non-negative integers")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
            raise ValueError("confidence must be a finite number between 0 and 1")
        start, end = min(frames), max(frames)
        if start == end:
            raise ValueError("impact candidate bracket requires two distinct frames")
        uncertainty = max(1, int(math.ceil(frame_rate / 30.0)))
        return cls(start, end, max(0.0, min(1.0, confidence)), uncertainty)


@dataclass(frozen=True)
class EvaluationMetrics:
    precision: Optional[float]
    recall: Optional[float]


def precision_recall(predicted: Sequence[bool], actual: Sequence[bool]) -> EvaluationMetrics:
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have equal length")
    if not actual:
        return EvaluationMetrics(None, None)
    tp = sum(p and a for p, a in zip(predicted, actual))
    fp = sum(p and not a for p, a in zip(predicted, actual))
    fn = sum(not p and a for p, a in zip(predicted, actual))
    return EvaluationMetrics(tp / (tp + fp) if tp + fp else None, tp / (tp + fn) if tp + fn else None)


def evaluate_sequence_gates(camera_motion_displacements: Sequence[float], cut_frames: Sequence[int],
                            track_coverage: float, longest_gap: int, anchor_coverage: float,
                            thresholds: Thresholds = Thresholds()) -> GateDecision:
    reasons = []
    if camera_motion_displacements and median(camera_motion_displacements) > thresholds.maximum_camera_motion:
        reasons.append("camera_motion")
    if cut_frames:
        reasons.append("cut")
    if track_coverage < thresholds.minimum_track_coverage:
        reasons.append("coverage")
    if longest_gap > thresholds.maximum_gap_frames:
        reasons.append("continuity")
    if anchor_coverage < thresholds.minimum_anchor_coverage:
        reasons.append("anchor_coverage")
    return GateDecision("passed" if not reasons else "blocked", not reasons, tuple(reasons), thresholds.provisional)


def reconstruct_automatic_shot(observations, calibration, context):
    """Guarded adapter; delegates mapping to unchanged fixture reconstruction."""
    if not callable(getattr(calibration, "to_engine", None)):
        raise VideoReconstructionUnavailable("calibration evidence is unavailable")
    if not hasattr(calibration, "width") or not hasattr(calibration, "height"):
        raise VideoReconstructionUnavailable("calibration dimensions are unavailable")
    if observations.image_width != calibration.width or observations.image_height != calibration.height:
        raise VideoReconstructionUnavailable("calibration dimensions do not match")
    threshold = context.min_confidence
    for label in ("ball", "clubhead"):
        evidence = [getattr(item, label) for item in observations.items if getattr(item, label) is not None]
        if not evidence:
            raise VideoReconstructionUnavailable(label + " evidence is unavailable")
        if any(item.get("confidence", 0.0) < threshold for item in evidence):
            raise VideoReconstructionUnavailable(label + " confidence is below threshold")
    from .reconstruction import reconstruct_shot
    result = reconstruct_shot(observations, calibration, context)
    metadata = dict(result.metadata)
    metadata["source"] = "video-automatic"
    return type(result)(result.shot_event, metadata)
