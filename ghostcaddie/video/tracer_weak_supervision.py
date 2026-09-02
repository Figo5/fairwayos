"""Research-only shot-tracer weak supervision primitives.

Tracer pixels are treated as graphics and trajectory hints only. They are never
returned as ball observations, ground truth, or production analytics input.
"""

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional research dependency.
    np = None


@dataclass(frozen=True)
class TracerHint:
    pixel_count: int
    centroid: Optional[Tuple[float, float]]
    provenance: str = "tracer_pseudo_hint"
    pseudo_label: bool = True
    ground_truth: bool = False
    research_only: bool = True
    production_eligible: bool = False


@dataclass(frozen=True)
class BallObservation:
    x: float
    y: float
    radius: float
    appearance_score: float
    motion_supported: bool
    provenance: str = "clean_frame_candidate"
    pseudo_label: bool = True
    ground_truth: bool = False
    research_only: bool = True
    production_eligible: bool = False


def provenance_flags():
    return {
        "pseudo_label": True,
        "ground_truth": False,
        "research_only": True,
        "production_eligible": False,
    }


def _require_image(image):
    if np is None:
        raise RuntimeError("numpy is required for tracer research")
    try:
        array = np.asarray(image)
    except Exception as exc:
        raise ValueError("image must be a numeric HxWx3 array") from exc
    if array.ndim != 3 or array.shape[2] != 3 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("image must be a non-empty HxWx3 array")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("image must contain numeric pixels")
    return array.astype(np.float32, copy=False)


def _region_mask(shape, region):
    mask = np.zeros(shape, dtype=bool)
    if region is None:
        return mask
    try:
        values = list(region)
    except Exception as exc:
        raise ValueError("region must contain four integer coordinates") from exc
    if (len(values) != 4 or any(type(value) is not int for value in values)):
        raise ValueError("region must contain four integer coordinates")
    x1, y1, x2, y2 = values
    height, width = shape
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def detect_graphic_mask(image, *, logo_region=None, ui_regions=(), previous_image=None):
    """Return a conservative mask for tracer/logo/UI-like graphic pixels."""
    rgb = _require_image(image)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    yellow_tracer = (red >= 150) & (green >= 110) & (blue <= 150) & (red + green >= blue * 2.0)
    high_chroma = chroma >= 100
    mask = yellow_tracer | high_chroma
    height, width = mask.shape
    # Fixed edge strips are UI/logo candidates, not ball search regions.
    edge = max(2, min(width, height) // 80)
    mask[:edge, :] |= high_chroma[:edge, :]
    mask[-edge:, :] |= high_chroma[-edge:, :]
    mask[:, :edge] |= high_chroma[:, :edge]
    mask[:, -edge:] |= high_chroma[:, -edge:]
    mask |= _region_mask(mask.shape, logo_region)
    if ui_regions is None:
        raise ValueError("ui_regions must be an iterable of regions")
    try:
        regions = list(ui_regions)
    except Exception as exc:
        raise ValueError("ui_regions must be an iterable of regions") from exc
    for region in regions:
        mask |= _region_mask(mask.shape, region)
    if previous_image is not None:
        previous = _require_image(previous_image)
        if previous.shape != rgb.shape:
            raise ValueError("previous_image must match image dimensions")
    return mask


def extract_tracer_hint(graphic_mask):
    """Summarize graphic pixels as a trajectory hint, never as a ball."""
    if np is None:
        raise RuntimeError("numpy is required for tracer research")
    mask = np.asarray(graphic_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("graphic_mask must be two-dimensional")
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return TracerHint(0, None)
    return TracerHint(int(len(xs)), (float(xs.mean()), float(ys.mean())))


def _components(mask):
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            points = []
            while stack:
                cy, cx = stack.pop()
                points.append((cy, cx))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            yield points


def search_clean_ball(image, *, previous_image=None, graphic_mask=None,
                      min_radius=1.5, max_radius=20.0):
    """Search clean pixels independently of tracer coordinates."""
    rgb = _require_image(image)
    height, width = rgb.shape[:2]
    if graphic_mask is None:
        graphic_mask = np.zeros((height, width), dtype=bool)
    else:
        graphic_mask = np.asarray(graphic_mask, dtype=bool)
        if graphic_mask.shape != (height, width):
            raise ValueError("graphic_mask must match image dimensions")
    luminance = rgb.mean(axis=2)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    clean = (~graphic_mask & (luminance >= 130.0) & (spread <= 35.0))
    if previous_image is not None:
        previous = _require_image(previous_image)
        if previous.shape != rgb.shape:
            return None
        motion = np.max(np.abs(rgb - previous), axis=2)
        clean &= motion >= max(8.0, float(np.median(motion)) + 3.0)
    candidates = []
    for points in _components(clean):
        area = len(points)
        if area < 2:
            continue
        ys = np.asarray([point[0] for point in points], dtype=float)
        xs = np.asarray([point[1] for point in points], dtype=float)
        box_width = xs.max() - xs.min() + 1.0
        box_height = ys.max() - ys.min() + 1.0
        radius = max(box_width, box_height) / 2.0
        if not (min_radius <= radius <= max_radius):
            continue
        compactness = area / max(1.0, box_width * box_height)
        appearance = min(1.0, float(luminance[ys.astype(int), xs.astype(int)].mean()) / 255.0)
        score = appearance * min(1.0, compactness * 1.5)
        candidates.append((score, float(xs.mean()), float(ys.mean()), radius))
    if not candidates:
        return None
    score, x, y, radius = max(candidates)
    return BallObservation(x, y, radius, score, previous_image is not None)


def accept_ball_track(observations, *, frame_indices, width, height, min_frames=3,
                      max_step=80.0, max_radius_ratio=2.5, max_score_delta=0.5):
    """Accept only consecutive, typed, bounded, appearance-consistent motion."""
    if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0
               for value in (width, height)):
        return False
    if not isinstance(min_frames, int) or isinstance(min_frames, bool) or min_frames < 2:
        return False
    if any(isinstance(value, bool) for value in (max_step, max_radius_ratio, max_score_delta)):
        return False
    try:
        limits = (float(max_step), float(max_radius_ratio), float(max_score_delta))
    except (TypeError, ValueError, OverflowError):
        return False
    if (not all(math.isfinite(value) and value > 0.0 for value in limits[:2]) or
            not math.isfinite(limits[2]) or limits[2] < 0.0):
        return False
    max_step, max_radius_ratio, max_score_delta = limits
    try:
        values, indices = list(observations), list(frame_indices)
    except Exception:
        return False
    if len(values) < min_frames or len(values) != len(indices):
        return False
    if any(not isinstance(index, int) or isinstance(index, bool) or index < 0
           for index in indices):
        return False
    if any(second != first + 1 for first, second in zip(indices, indices[1:])):
        return False
    previous = None
    moved = False
    radii, scores = [], []
    for value in values:
        if not isinstance(value, BallObservation):
            return False
        if (type(value.motion_supported) is not bool or value.motion_supported is not True or
                type(value.pseudo_label) is not bool or value.pseudo_label is not True or
                type(value.ground_truth) is not bool or value.ground_truth is not False or
                type(value.research_only) is not bool or value.research_only is not True or
                type(value.production_eligible) is not bool or value.production_eligible is not False or
                type(value.provenance) is not str or value.provenance != "clean_frame_candidate"):
            return False
        try:
            x, y, radius, score = (float(value.x), float(value.y), float(value.radius),
                                   float(value.appearance_score))
        except (TypeError, ValueError, OverflowError):
            return False
        if not all(math.isfinite(v) for v in (x, y, radius, score)):
            return False
        if not (0.5 <= radius <= min(width, height) / 4.0 and 0.0 <= score <= 1.0):
            return False
        if not (radius <= x < width - radius and radius <= y < height - radius):
            return False
        if previous is not None:
            step = math.hypot(x - previous[0], y - previous[1])
            if not math.isfinite(step) or step > max_step:
                return False
            moved = moved or step >= max(1.0, radius * 0.5)
        previous = (x, y)
        radii.append(radius)
        scores.append(score)
    if min(radii) <= 0.0 or max(radii) / min(radii) > max_radius_ratio:
        return False
    if max(scores) - min(scores) > max_score_delta:
        return False
    return moved


def build_tracer_render_filter(*, width, height, tracer_points=(), ball_points=()):
    """Build distinct FFmpeg layers for tracer hints and clean-ball candidates."""
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError("width must be a positive integer")
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise ValueError("height must be a positive integer")
    if width < 30 or height < 9:
        raise ValueError("frame is too small for diagnostic legend")
    def checked_points(points, label):
        try:
            values = list(points)
        except Exception as exc:
            raise ValueError(label + " must be an iterable of points") from exc
        checked = []
        for point in values:
            try:
                if len(point) != 2 or any(type(value) not in (int, float) for value in point):
                    raise ValueError
                x, y = float(point[0]), float(point[1])
            except Exception as exc:
                raise ValueError(label + " contains an invalid point") from exc
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
                raise ValueError(label + " contains an out-of-frame point")
            checked.append((x, y))
        return checked
    tracer_points = checked_points(tracer_points, "tracer_points")
    ball_points = checked_points(ball_points, "ball_points")
    filters = [
        # Portable legend bars: magenta=tracer pseudo-hint, lime=clean-ball
        # candidate. Text labels remain in the JSON diagnostics/docs because
        # this FFmpeg build may not include drawtext.
        f"drawbox=x=0:y=0:w=30:h=4:color=magenta:t=fill",
        f"drawbox=x=0:y=5:w=30:h=4:color=lime:t=fill",
    ]
    for x, y in tracer_points:
        filters.append(f"drawbox=x={float(x)-3:g}:y={float(y)-3:g}:w=6:h=6:color=magenta:t=fill")
    for x, y in ball_points:
        filters.append(f"drawbox=x={float(x)-5:g}:y={float(y)-5:g}:w=10:h=10:color=lime:t=2")
    return ",".join(filters)
