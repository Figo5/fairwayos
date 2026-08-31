"""Coordinate normalization for local research model outputs."""

from __future__ import annotations

from typing import Iterable, Tuple


def _dimensions(width: int, height: int) -> None:
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")


def _values(values: Iterable[float], count: int) -> Tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("coordinates must be numeric") from exc
    if len(result) != count:
        raise ValueError("unexpected coordinate shape")
    return result


def normalize_point(point: Iterable[float], width: int, height: int) -> Tuple[float, float]:
    """Convert normalized or pixel-space point coordinates to pixels."""
    _dimensions(width, height)
    x, y = _values(point, 2)
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return x * width, y * height
    return x, y


def normalize_box(box: Iterable[float], width: int, height: int) -> Tuple[float, float, float, float]:
    """Convert normalized or pixel-space ``x1,y1,x2,y2`` coordinates to pixels."""
    _dimensions(width, height)
    x1, y1, x2, y2 = _values(box, 4)
    if all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
        return x1 * width, y1 * height, x2 * width, y2 * height
    return x1, y1, x2, y2
