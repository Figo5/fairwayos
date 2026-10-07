"""Ground-scene overlays, with observed motion separate from model advice.

A camera transform reprojects existing ground observations; it never supplies
an observation of a missing ball. Reference-image pixels are not yards. The
strategy bridge uses the existing pipeline only with explicit course/profile
inputs and a source/scene-bound calibration supplied by the caller.
"""
from dataclasses import dataclass
import math
from typing import Optional

from ..config import Config
from ..geometry import Point2D
from ..pipeline import PipelineResult, run_pipeline
from ..session import InMemoryCourseSource, InMemoryPlayerSource, InMemoryShotSource
from .calibration import VideoCalibration


def _point(value):
    if value is None:
        return None
    if len(value) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError("point must contain two finite coordinates")
    return tuple(float(v) for v in value)


def transform_point(matrix, point):
    """Project a ground point with a validated 3x3 camera transformation."""
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("camera transform must be 3x3")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for row in matrix for v in row):
        raise ValueError("camera transform must be finite")
    x, y = _point(point)
    values = [row[0] * x + row[1] * y + row[2] for row in matrix]
    if abs(values[2]) < 1e-9:
        raise ValueError("ground projection is at infinity")
    return _point((values[0] / values[2], values[1] / values[2]))


class ObservedGroundTrail:
    """Short ground trail in CURRENT camera pixels; no held/missing positions.

    A failed registration, absent detection, frame gap, or scene change clears
    the trail. This is deliberately stricter than drawing a guessed connector.
    """
    def __init__(self, max_frames=90):
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 2:
            raise ValueError("max_frames must be an integer >= 2")
        self.max_frames = max_frames
        self.points = []
        self.last_frame = None
        self.scene = None

    def update(self, frame, scene, ball, previous_to_current=None):
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError("frame must be a non-negative integer")
        if not isinstance(scene, str) or not scene:
            raise ValueError("scene must be a non-empty identifier")
        ball = _point(ball)
        contiguous = scene == self.scene and self.last_frame is not None and frame == self.last_frame + 1
        if not contiguous or previous_to_current is None or ball is None:
            self.points = []
        else:
            self.points = [(n, transform_point(previous_to_current, point)) for n, point in self.points if frame - n < self.max_frames]
        if ball is not None:
            self.points.append((frame, ball))
        self.last_frame, self.scene = frame, scene
        return tuple(point for _, point in self.points)


@dataclass(frozen=True)
class StrategyBinding:
    source_sha256: str
    scene_id: str
    reference_frame: int
    calibration_basis: str

    def __post_init__(self):
        if len(self.source_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA256 digest")
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise ValueError("scene_id is required")
        if isinstance(self.reference_frame, bool) or not isinstance(self.reference_frame, int) or self.reference_frame < 0:
            raise ValueError("reference_frame must be non-negative")
        if self.calibration_basis not in ('operator', 'measured', 'synthetic'):
            raise ValueError("calibration basis must be explicit")


@dataclass(frozen=True)
class StrategyLayer:
    pipeline: PipelineResult
    binding: StrategyBinding
    reference_target: tuple
    label: str

    def target_in_frame(self, source_sha256, scene_id, reference_to_current, *, reference_frame):
        if (source_sha256 != self.binding.source_sha256 or scene_id != self.binding.scene_id
                or isinstance(reference_frame, bool) or not isinstance(reference_frame, int)
                or reference_frame != self.binding.reference_frame or reference_to_current is None):
            return None
        return transform_point(reference_to_current, self.reference_target)


def build_strategy_layer(shot, course, player, calibration: VideoCalibration,
                         binding: StrategyBinding, config: Optional[Config] = None):
    """Connect supplied golf inputs to an on-video model target.

    Missing real inputs do not trigger sample-data substitution. Synthetic
    inputs remain usable in an explicitly labelled scenario. The caller must
    not treat this coarse expected-strokes engine as a putting/flight model.
    """
    if any(v is None for v in (shot, course, player, calibration, binding)):
        raise ValueError("strategy needs shot, course, player, calibration and source binding")
    if calibration.engine_units != 'yards' or course.coordinate_system.units != 'yards':
        raise ValueError("strategy requires yard-calibrated course coordinates; image pixels are not yards")
    if player.player_id != shot.player_id or shot.club not in player.clubs:
        raise ValueError("shot must match the supplied player and club profile")
    if shot.lie == 'green':
        raise ValueError("the current strategy engine does not support calibrated putting recommendations")
    pipeline = run_pipeline(InMemoryShotSource(shot, 'scene:inline:shot'),
                            InMemoryCourseSource(course, 'scene:inline:course'),
                            InMemoryPlayerSource(player, 'scene:inline:player'), config or Config.default())
    target = calibration.from_engine(pipeline.recommendation.recommended_target)
    label = 'SCENARIO TARGET' if binding.calibration_basis == 'synthetic' else 'MODEL AIM TARGET'
    pipeline.recommendation.provenance['scene_binding'] = {
        'source_sha256': binding.source_sha256, 'scene_id': binding.scene_id,
        'reference_frame': binding.reference_frame, 'calibration_basis': binding.calibration_basis,
        'model_status': 'illustrative expected-strokes estimate; not a physical flight or putting model',
    }
    return StrategyLayer(pipeline, binding, (target['x'], target['y']), label)
