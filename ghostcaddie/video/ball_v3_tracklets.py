"""Tracklets, temporal features, and the discriminator that scores them.

The per-frame ranker answers "which candidate on this frame looks most like the
ball, in context". That is still an appearance question, and appearance alone
measurably does not settle it: with an oracle crop placed exactly on a real
labelled ball, the v2 detector beat the best same-frame distractor only 62.5% of
the time. Some distractors are simply going to win a still-frame argument.

What they cannot do is *keep* winning while behaving like a ball. A shirt
highlight travels with the torso. A tee, a divot, a sprinkler head and a range
ball sit still while the camera pans. A clubhead accelerates through a short arc
and stops. A struck ball departs on its own. Those are all temporal facts, and
they are what this module measures.

Design note, from Phase 0: **forward/backward cycle consistency is not a gate.**
Every clip in the v2 final test scored ``cycle_consistency_rate = 1.0``, including
the three that confidently tracked clutter with nine false positives each. Static
clutter passes a round-trip check trivially -- it is exactly the thing optical
flow tracks best. It survives here as one weak feature among many, never as a
filter.

The discriminator is a small MLP over these features rather than a video network.
That is a deliberate ceiling: the appearance work is already done by the ranker,
the per-frame evidence is already summarised, and there is not enough supervision
in this corpus to justify fitting a 3D CNN. If the temporal features turn out to
carry the signal, a heavier temporal model is the upgrade path.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    torch = None

    class _Missing:
        class Module:
            pass

    nn = _Missing
    TORCH_AVAILABLE = False

#: Feature order for the tracklet discriminator. Named so ablations can drop a
#: group by name and the results table can say what was dropped.
FEATURE_NAMES = (
    "support_fraction",          # frames held / frames in the window
    "mean_rank_probability",
    "max_rank_probability",
    "mean_rank_margin",
    "top1_fraction",             # how often it won its own frame
    "mean_step_radii",           # camera-compensated displacement per frame
    "std_step_radii",
    "max_step_radii",
    "mean_abs_accel_radii",
    "direction_consistency",     # mean cosine between successive velocities
    "static_fraction",           # frames barely moving after compensation
    "radius_stability",
    "body_inside_fraction",
    "moves_with_torso",          # agreement between candidate and torso motion
    "mean_distance_to_hands",
    "cycle_consistency",         # kept as a weak feature, never as a gate
)

MOTION_GROUPS = {
    "motion": ("mean_step_radii", "std_step_radii", "max_step_radii",
               "mean_abs_accel_radii", "direction_consistency", "static_fraction"),
    "body": ("body_inside_fraction", "moves_with_torso", "mean_distance_to_hands"),
    "ranking": ("mean_rank_probability", "max_rank_probability",
                "mean_rank_margin", "top1_fraction"),
}

#: A candidate this many radii from the prediction is a different object. Set to
#: match the fastest ball the rest of the system considers plausible: the
#: synthetic generator draws "fast" trajectories at up to 6.0 radii per frame and
#: v2's plausibility bound was 8.0, so a tighter link radius here would refuse to
#: follow a struck ball on its first frame -- before any velocity estimate exists
#: to widen the search.
LINK_RADII = 8.0
#: Below this compensated displacement a candidate counts as not moving.
STATIC_STEP_RADII = 0.25


@dataclass
class Node:
    """One candidate, on one frame, already scored by the ranker."""

    frame_index: int
    x: float
    y: float
    radius: float
    probability: float
    margin: float = 0.0
    is_top1: bool = False
    inside_torso: float = 0.0
    distance_to_hands: float = 0.0
    torso_centre: Optional[tuple[float, float]] = None
    cycle_ok: float = 1.0


@dataclass
class Tracklet:
    nodes: list[Node] = field(default_factory=list)
    #: Per-frame camera shift already removed from the geometry, in native pixels.
    compensated: bool = True

    @property
    def span(self) -> int:
        if not self.nodes:
            return 0
        return self.nodes[-1].frame_index - self.nodes[0].frame_index + 1

    def to_dict(self) -> dict:
        return {"frames": [n.frame_index for n in self.nodes],
                "x": [round(n.x, 2) for n in self.nodes],
                "y": [round(n.y, 2) for n in self.nodes],
                "radius": [round(n.radius, 2) for n in self.nodes],
                "probability": [round(n.probability, 4) for n in self.nodes]}


def build_tracklets(per_frame: dict[int, Sequence[Node]],
                    camera_shift: dict[int, tuple[float, float]], *,
                    top_k: int = 8, link_radii: float = LINK_RADII,
                    min_length: int = 3) -> list[Tracklet]:
    """Greedy forward linking of the top-K ranked candidates per frame.

    Camera motion is removed before distances are compared: on a panning shot the
    background moves further between frames than a struck ball does, and without
    compensation every link decision is dominated by the camera.
    """
    indices = sorted(per_frame)
    open_tracks: list[Tracklet] = []
    finished: list[Tracklet] = []
    for position, index in enumerate(indices):
        nodes = list(per_frame.get(index) or [])[:top_k]
        shift = camera_shift.get(index, (0.0, 0.0))
        used: set[int] = set()
        still_open: list[Tracklet] = []
        for track in open_tracks:
            last = track.nodes[-1]
            # Predict where the last node lands on this frame: its own velocity
            # through the world, plus the camera's motion into this frame.
            #
            # The velocity has to be CAMERA-COMPENSATED first. Node positions are
            # native pixels, so a raw frame-to-frame difference already contains
            # the camera's motion; adding the shift on top of that double-counts
            # it, and a track on a panning shot then breaks after two frames --
            # measured directly, every track capped at length 2 on a 40 px/frame
            # pan.
            velocity = (0.0, 0.0)
            if len(track.nodes) >= 2:
                previous = track.nodes[-2]
                gap = max(1, last.frame_index - previous.frame_index)
                previous_shift = camera_shift.get(last.frame_index, (0.0, 0.0))
                velocity = ((last.x - previous.x - previous_shift[0]) / gap,
                            (last.y - previous.y - previous_shift[1]) / gap)
            predicted = (last.x + velocity[0] + shift[0],
                         last.y + velocity[1] + shift[1])
            tolerance = link_radii * max(2.0, last.radius)
            best, best_distance = None, tolerance
            for slot, node in enumerate(nodes):
                if slot in used:
                    continue
                distance = math.hypot(node.x - predicted[0], node.y - predicted[1])
                if distance <= best_distance:
                    best, best_distance = slot, distance
            if best is None:
                finished.append(track)
                continue
            used.add(best)
            track.nodes.append(nodes[best])
            still_open.append(track)
        for slot, node in enumerate(nodes):
            if slot not in used:
                still_open.append(Tracklet(nodes=[node]))
        open_tracks = still_open
    finished.extend(open_tracks)
    return [t for t in finished if len(t.nodes) >= min_length]


def tracklet_features(track: Tracklet, *, window: int,
                      camera_shift: dict[int, tuple[float, float]]) -> list[float]:
    """Summarise a tracklet into ``FEATURE_NAMES``. Never raises on a short track."""
    nodes = track.nodes
    count = len(nodes)
    radius = max(1.0, float(np.median([n.radius for n in nodes])))

    steps: list[float] = []
    velocities: list[tuple[float, float]] = []
    for previous, current in zip(nodes, nodes[1:]):
        gap = max(1, current.frame_index - previous.frame_index)
        shift = camera_shift.get(current.frame_index, (0.0, 0.0))
        dx = (current.x - previous.x - shift[0]) / gap
        dy = (current.y - previous.y - shift[1]) / gap
        velocities.append((dx, dy))
        steps.append(math.hypot(dx, dy) / radius)

    accelerations = [
        math.hypot(b[0] - a[0], b[1] - a[1]) / radius
        for a, b in zip(velocities, velocities[1:])]

    cosines = []
    for a, b in zip(velocities, velocities[1:]):
        na, nb = math.hypot(*a), math.hypot(*b)
        if na > 1e-6 and nb > 1e-6:
            cosines.append((a[0] * b[0] + a[1] * b[1]) / (na * nb))

    # does this candidate move the way the torso moves? A shirt highlight does.
    torso_agreement = 0.0
    paired = [(p, c) for p, c in zip(nodes, nodes[1:])
              if p.torso_centre is not None and c.torso_centre is not None]
    if paired:
        agreements = []
        for previous, current in paired:
            tdx = current.torso_centre[0] - previous.torso_centre[0]
            tdy = current.torso_centre[1] - previous.torso_centre[1]
            cdx, cdy = current.x - previous.x, current.y - previous.y
            torso_motion = math.hypot(tdx, tdy)
            if torso_motion < 0.5:
                continue
            # 1.0 when the candidate moved exactly with the torso, 0.0 when its
            # motion is unrelated in size
            agreements.append(
                max(0.0, 1.0 - math.hypot(cdx - tdx, cdy - tdy) / (torso_motion + 1e-6)))
        torso_agreement = float(np.mean(agreements)) if agreements else 0.0

    radii = [n.radius for n in nodes]
    return [
        min(1.0, count / max(1, window)),
        float(np.mean([n.probability for n in nodes])),
        float(np.max([n.probability for n in nodes])),
        float(np.mean([n.margin for n in nodes])),
        float(np.mean([1.0 if n.is_top1 else 0.0 for n in nodes])),
        float(np.mean(steps)) if steps else 0.0,
        float(np.std(steps)) if steps else 0.0,
        float(np.max(steps)) if steps else 0.0,
        float(np.mean(accelerations)) if accelerations else 0.0,
        float(np.mean(cosines)) if cosines else 0.0,
        float(np.mean([1.0 if s < STATIC_STEP_RADII else 0.0 for s in steps]))
        if steps else 1.0,
        1.0 / (1.0 + float(np.std(radii)) / radius),
        float(np.mean([n.inside_torso for n in nodes])),
        torso_agreement,
        float(np.mean([min(n.distance_to_hands, 200.0) / 200.0 for n in nodes])),
        float(np.mean([n.cycle_ok for n in nodes])),
    ]


def mask_for_ablation(dropped: Sequence[str]) -> np.ndarray:
    """Zero out named feature groups, so an ablation removes information rather
    than changing the architecture."""
    mask = np.ones(len(FEATURE_NAMES), dtype=np.float32)
    names = set()
    for item in dropped:
        names.update(MOTION_GROUPS.get(item, (item,)))
    for position, name in enumerate(FEATURE_NAMES):
        if name in names:
            mask[position] = 0.0
    return mask


if TORCH_AVAILABLE:

    class TrackletDiscriminator(nn.Module):
        """Is this tracklet the played ball? A small MLP, on purpose."""

        def __init__(self, size: int = len(FEATURE_NAMES), width: int = 64) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(size, width), nn.ReLU(inplace=True),
                nn.Linear(width, width), nn.ReLU(inplace=True),
                nn.Linear(width, 1))

        def forward(self, features: Any) -> Any:
            return self.net(features).squeeze(-1)


def save_discriminator(model: Any, path: Path) -> str:
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required to save the tracklet discriminator")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_discriminator(path: Path, *, device_override: Any = None) -> Any:
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required to load the tracklet discriminator")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = TrackletDiscriminator(size=int(state["net.0.weight"].shape[1]))
    model.load_state_dict(state)
    if device_override is not None:
        model = model.to(device_override)
    return model.eval()
