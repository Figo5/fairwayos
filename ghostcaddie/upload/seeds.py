"""Source-specific assistance, bound to the source hash and always disclosed.

A seed is human/AI-supplied initialisation for a target that cannot start
unattended. It is assistance, never ground truth, and it is refused unless it
names the exact source it was drawn on - the same binding failure that once let
one clip's coordinates be used on another.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


class SeedRejected(ValueError):
    pass


@dataclass(frozen=True)
class Seed:
    target: str
    frame: int
    box_xyxy: Optional[List[float]] = None
    point_xy: Optional[List[float]] = None
    initialization: str = "assisted"
    disclosure: str = ("source-specific assisted initialisation supplied by the "
                       "operator; AI/human reviewed, NOT ground truth")

    def to_dict(self) -> dict:
        return {"target": self.target, "frame": self.frame,
                "box_xyxy": self.box_xyxy, "point_xy": self.point_xy,
                "initialization": self.initialization,
                "disclosure": self.disclosure,
                "pseudo_label": True, "ground_truth": False,
                "production_eligible": False}


class SeedBundle:
    def __init__(self, source_sha256: str, seeds: List[Seed]):
        self.source_sha256 = source_sha256
        self.seeds = seeds

    @classmethod
    def from_dict(cls, data: dict, source_sha256: str) -> "SeedBundle":
        if not isinstance(data, dict):
            raise SeedRejected("seed bundle must be an object")
        declared = data.get("source_sha256")
        if not declared:
            raise SeedRejected(
                "seed bundle has no source_sha256; assistance must name the exact "
                "source it was drawn on")
        if declared != source_sha256:
            raise SeedRejected(
                f"seed bundle is bound to a different source "
                f"({declared[:16]}...) than the uploaded video "
                f"({source_sha256[:16]}...); refusing to transplant coordinates")
        out = []
        for s in data.get("seeds", []):
            t, f = s.get("target"), s.get("frame")
            if t not in ("body", "clubhead", "ball"):
                raise SeedRejected(f"unknown seed target {t!r}")
            if not isinstance(f, int) or f < 0:
                raise SeedRejected(f"seed frame must be a non-negative int, got {f!r}")
            box = s.get("box_xyxy")
            if box is not None:
                if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                    raise SeedRejected(f"invalid box_xyxy {box!r}")
            out.append(Seed(target=t, frame=f, box_xyxy=box,
                            point_xy=s.get("point_xy")))
        return cls(source_sha256, out)

    def for_target(self, target: str) -> Optional[Seed]:
        for s in self.seeds:
            if s.target == target:
                return s
        return None

    def to_dict(self) -> dict:
        return {"source_sha256": self.source_sha256,
                "seeds": [s.to_dict() for s in self.seeds]}
