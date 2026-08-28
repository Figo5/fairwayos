"""Validated manual four-point calibration for video image coordinates."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from ..geometry import CoordinateMapper, CoordinateSystem, Point2D
from .errors import VideoCalibrationError, VideoPathError
from .paths import ProjectBoundary


def _point(value, name):
    if isinstance(value, Point2D):
        point = value
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        point = Point2D(value[0], value[1])
    elif isinstance(value, dict):
        point = Point2D(value.get("x"), value.get("y"))
    else:
        raise VideoCalibrationError(f"{name} must be a 2D point")
    for axis in ("x", "y"):
        number = getattr(point, axis)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise VideoCalibrationError(f"{name}.{axis} must be finite")
    return Point2D(float(point.x), float(point.y))


@dataclass(frozen=True)
class VideoCalibration:
    image_width: int
    image_height: int
    source_units: str
    engine_units: str
    source_points: tuple
    engine_points: tuple

    def __post_init__(self):
        if isinstance(self.image_width, bool) or not isinstance(self.image_width, int) or self.image_width <= 0:
            raise VideoCalibrationError("image_width must be a positive integer")
        if isinstance(self.image_height, bool) or not isinstance(self.image_height, int) or self.image_height <= 0:
            raise VideoCalibrationError("image_height must be a positive integer")
        for value, name in ((self.source_units, "source_units"), (self.engine_units, "engine_units")):
            if not isinstance(value, str) or not value.strip():
                raise VideoCalibrationError(f"{name} must be a non-empty string")
        if len(self.source_points) != 4 or len(self.engine_points) != 4:
            raise VideoCalibrationError("calibration requires exactly four paired points")
        source = tuple(_point(p, f"source_points[{i}]") for i, p in enumerate(self.source_points))
        engine = tuple(_point(p, f"engine_points[{i}]") for i, p in enumerate(self.engine_points))
        for i, point in enumerate(source):
            if not (0 <= point.x <= self.image_width and 0 <= point.y <= self.image_height):
                raise VideoCalibrationError(f"source_points[{i}] is outside image bounds")
        try:
            mapper = CoordinateMapper(CoordinateSystem(
                mode="four_point", units=self.engine_units, source_units=self.source_units,
                source_points=source, engine_points=engine,
            ))
        except (TypeError, ValueError) as exc:
            raise VideoCalibrationError(str(exc)) from exc
        object.__setattr__(self, "source_points", source)
        object.__setattr__(self, "engine_points", engine)
        object.__setattr__(self, "_mapper", mapper)

    @property
    def mapper(self):
        return self._mapper

    @property
    def units(self):
        return self.engine_units

    @property
    def width(self):
        return self.image_width

    @property
    def height(self):
        return self.image_height

    def to_engine(self, point):
        try:
            return self._mapper.to_engine(_point(point, "source_point").__dict__)
        except (TypeError, KeyError, ValueError) as exc:
            raise VideoCalibrationError(str(exc)) from exc

    def from_engine(self, point):
        try:
            return self._mapper.from_engine(_point(point, "engine_point"))
        except (TypeError, KeyError, ValueError) as exc:
            raise VideoCalibrationError(str(exc)) from exc

    forward = to_engine
    inverse = from_engine
    map_forward = to_engine
    map_inverse = from_engine

    def to_dict(self):
        def pair(p): return {"x": p.x, "y": p.y}
        return {"image_width": self.image_width, "image_height": self.image_height,
                "source_units": self.source_units, "engine_units": self.engine_units,
                "source_points": [pair(p) for p in self.source_points],
                "engine_points": [pair(p) for p in self.engine_points]}


def load_video_calibration(path, project_boundary):
    if not isinstance(project_boundary, ProjectBoundary):
        raise VideoPathError("a ProjectBoundary is required for calibration resources")
    calibration_path = project_boundary.resolve_calibration(path)
    try:
        raw = json.loads(calibration_path.read_text())
    except (OSError, ValueError) as exc:
        raise VideoCalibrationError(f"unable to load calibration JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise VideoCalibrationError("calibration JSON must be an object")
    image = raw.get("image", {})
    if not isinstance(image, dict):
        raise VideoCalibrationError("image must be an object")
    width = raw.get("image_width", image.get("width"))
    height = raw.get("image_height", image.get("height"))
    try:
        return VideoCalibration(width, height, raw.get("source_units", "pixels"),
                                raw.get("engine_units", raw.get("units", "yards")),
                                tuple(raw.get("source_points", ())), tuple(raw.get("engine_points", ())))
    except (TypeError, ValueError) as exc:
        raise VideoCalibrationError(str(exc)) from exc
