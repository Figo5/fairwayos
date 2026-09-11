"""Display <-> native coordinate mapping for reviewed seed selection.

The browser shows a scaled, possibly letterboxed preview of one exactly-decoded
native frame. A click must map back to the NATIVE pixel it actually points at.

Clicks in the letterbox padding are REJECTED, not clamped: clamping would invent
a coordinate on the image edge that the operator never indicated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


class MappingError(ValueError):
    pass


@dataclass(frozen=True)
class DisplayTransform:
    native_w: int
    native_h: int
    display_w: int
    display_h: int

    def validate(self) -> "DisplayTransform":
        for n, v in (("native_w", self.native_w), ("native_h", self.native_h),
                     ("display_w", self.display_w), ("display_h", self.display_h)):
            if not isinstance(v, (int, float)) or v <= 0:
                raise MappingError(f"{n} must be positive, got {v!r}")
        return self

    @property
    def scale(self) -> float:
        """Uniform contain-scale, matching CSS object-fit: contain."""
        self.validate()
        return min(self.display_w / self.native_w, self.display_h / self.native_h)

    @property
    def content_size(self) -> Tuple[float, float]:
        s = self.scale
        return self.native_w * s, self.native_h * s

    @property
    def pad(self) -> Tuple[float, float]:
        cw, ch = self.content_size
        return (self.display_w - cw) / 2.0, (self.display_h - ch) / 2.0

    def to_dict(self) -> dict:
        cw, ch = self.content_size
        px, py = self.pad
        return {"native_w": self.native_w, "native_h": self.native_h,
                "display_w": self.display_w, "display_h": self.display_h,
                "scale": self.scale, "content_w": cw, "content_h": ch,
                "pad_x": px, "pad_y": py}


def map_display_to_native(t: DisplayTransform, dx: float, dy: float) -> Tuple[float, float]:
    t.validate()
    if not (0 <= dx <= t.display_w and 0 <= dy <= t.display_h):
        raise MappingError(f"click ({dx},{dy}) outside the display area")
    px, py = t.pad
    s = t.scale
    cw, ch = t.content_size
    if not (px - 1e-9 <= dx <= px + cw + 1e-9 and py - 1e-9 <= dy <= py + ch + 1e-9):
        raise MappingError(
            f"click ({dx},{dy}) is in the letterbox padding, not on the image; "
            "refusing to clamp it onto the edge")
    return ((dx - px) / s, (dy - py) / s)


def map_native_to_display(t: DisplayTransform, nx: float, ny: float) -> Tuple[float, float]:
    t.validate()
    px, py = t.pad
    s = t.scale
    return (nx * s + px, ny * s + py)


def build_seed_payload(source_sha256: str, frame: int, target: str,
                       transform: DisplayTransform,
                       display_points: Sequence[Tuple[float, float]]) -> dict:
    """Turn operator clicks into a hash-bound, frame-bound seed payload."""
    if target == "body":
        raise MappingError(
            "body runs automatically and takes no seed; supplying one would "
            "misrepresent an automatic layer as assisted")
    if target not in ("clubhead", "ball"):
        raise MappingError(f"unknown seed target {target!r}")
    if not source_sha256 or len(source_sha256) != 64:
        raise MappingError("a 64-char source_sha256 is required")
    if not isinstance(frame, int) or frame < 0:
        raise MappingError("frame must be a non-negative integer")
    pts = [map_display_to_native(transform, x, y) for x, y in display_points]
    out = {"source_sha256": source_sha256, "frame": frame, "target": target,
           "initialization": "assisted",
           "disclosure": ("operator-selected on an exactly decoded native frame; "
                          "AI/human reviewed assistance, NOT ground truth"),
           "display_transform": transform.to_dict(),
           "pseudo_label": True, "ground_truth": False, "production_eligible": False}
    if target == "clubhead":
        if len(pts) != 2:
            raise MappingError("clubhead seed needs exactly two corner clicks")
        (x1, y1), (x2, y2) = pts
        out["box_xyxy"] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    else:
        if len(pts) != 1:
            raise MappingError("ball seed needs exactly one click")
        out["point_xy"] = [pts[0][0], pts[0][1]]
    return out
