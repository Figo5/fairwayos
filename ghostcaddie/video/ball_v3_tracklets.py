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
from typing import Any, Mapping, Optional, Sequence

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
    "cycle_consistency",         # forward/backward link agreement; weak, never a gate
    "mean_rank_position",        # where it sits in its own frame's ranked list
    "competitor_pressure",       # the strongest candidate on the frame that is not it
    "chroma_score_mean",         # local_distinctness generator response
    "chroma_score_max",
    "brightness_score_mean",     # brightness_cue generator response
    "brightness_score_std",      # how much that response changes over the track
    "tight_mean_intensity",      # the candidate's own crop
    "tight_std_intensity",
    "context_contrast",          # how busy the surrounding scene is
    "torso_relative_speed",      # |candidate - torso| velocity, in radii per frame
    "radius_over_short_side",    # scale of the candidate relative to the frame
)

MOTION_GROUPS = {
    "motion": ("mean_step_radii", "std_step_radii", "max_step_radii",
               "mean_abs_accel_radii", "direction_consistency", "static_fraction",
               "cycle_consistency"),
    "body": ("body_inside_fraction", "moves_with_torso", "mean_distance_to_hands",
             "torso_relative_speed"),
    "ranking": ("mean_rank_probability", "max_rank_probability",
                "mean_rank_margin", "top1_fraction", "mean_rank_position",
                "competitor_pressure"),
    "appearance": ("tight_mean_intensity", "tight_std_intensity", "context_contrast"),
    "chroma": ("chroma_score_mean", "chroma_score_max", "brightness_score_mean",
               "brightness_score_std"),
    "scale": ("radius_stability", "radius_over_short_side"),
}

#: Features this pipeline is *not* able to compute, with the reason. They are
#: deliberately absent from ``FEATURE_NAMES`` rather than present as a constant
#: zero: a zero that means "nobody looked" is indistinguishable from a zero that
#: means "measured, and it was zero", and a model trained on the first will
#: confidently report the second.
UNAVAILABLE_FEATURES = {
    "club_lock": "no clubhead tracker exists on any path: canonical CLUBHEAD is "
                 "UNAVAILABLE and the research clubhead path was falsified, so "
                 "there is no club geometry to lock a candidate to",
    "club_relative_velocity": "same reason: no club geometry",
    "club_candidate_agreement": "same reason: no club geometry",
    "crop_embedding": "the tight/context crops are summarised by three scalar "
                      "statistics; a learned crop embedding is not carried here "
                      "because the listwise ranker already scores the same crops",
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
    """One candidate, on one frame, already scored by the ranker.

    The last six fields are supplied by the caller and default to a sentinel so
    that ``node_gaps`` can tell "not computed" apart from "computed and zero".
    ``build_tracklets`` fills the first three itself.
    """

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
    #: filled by ``build_tracklets`` from the frame's own candidate pool
    rank: int = -1
    #: how many candidates were ranked on this frame, and how many the proposal
    #: stage produced in total before the frame budget trimmed them
    frame_nodes: int = 0
    pool_size: int = 0
    competitor_probability: float = 0.0
    #: supplied by the caller; -1.0 / None means "not computed"
    chroma_score: float = -1.0
    brightness_score: float = -1.0
    radius_over_short_side: float = -1.0
    #: (tight mean intensity, tight std, context std) from the stored stack
    appearance: Optional[tuple[float, float, float]] = None
    #: Was a usable pose available on this frame? ``None`` means the caller never
    #: said. A zero body vector is *not* evidence of a missing pose: a candidate
    #: can be legitimately far from every landmark and outside the torso, which
    #: produces zeros from real geometry. Only this flag separates the two, and
    #: ``node_gaps`` reads it rather than inferring availability from the other
    #: fields being set.
    body_available: Optional[bool] = None


#: Node fields a caller must supply for a complete feature vector.
CALLER_NODE_FIELDS = ("chroma_score", "brightness_score",
                      "radius_over_short_side", "appearance")


def node_gaps(track: "Tracklet", *, include_body: bool = True) -> list[str]:
    """Which optional node fields were left unset anywhere in this track.

    The training-data builder asserts the caller fields are all present, so a
    feature vector that would be zero by absence cannot silently enter a training
    set.

    ``include_body`` adds the pose check. It is separate because a missing pose
    is *not* a builder bug -- the pose cache legitimately has no entry for some
    clips -- so the training builder drops on ``include_body=False`` and records
    the pose gaps as coverage instead. Before this, a tracklet whose pose was
    missing everywhere reported no gaps at all, because the other five caller
    fields were set and the body vector came back as zeros.
    """
    gaps: list[str] = []
    for node in track.nodes:
        if node.frame_nodes <= 0 or node.rank < 0:
            gaps.append("rank")
        if node.chroma_score < 0.0:
            gaps.append("chroma_score")
        if node.brightness_score < 0.0:
            gaps.append("brightness_score")
        if node.radius_over_short_side < 0.0:
            gaps.append("radius_over_short_side")
        if node.appearance is None:
            gaps.append("appearance")
        if include_body and node.body_available is not True:
            gaps.append("body_pose")
    return sorted(set(gaps))


def body_coverage(track: "Tracklet") -> Optional[float]:
    """Fraction of frames with a usable pose, or ``None`` if nobody said.

    ``None`` rather than 0.0 for an unannotated tracklet: "no pose" and "not
    asked" are different answers and only one of them is a measurement.
    """
    if not track.nodes:
        return None
    known = [n for n in track.nodes if n.body_available is not None]
    if not known:
        return None
    return float(sum(1 for n in known if n.body_available) / len(known))


@dataclass
class Tracklet:
    nodes: list[Node] = field(default_factory=list)
    #: Per-frame camera shift already removed from the geometry, in native pixels.
    compensated: bool = True
    #: Fraction of consecutive node pairs whose forward link is reproducible in
    #: reverse from the *following* step's velocity. Computed by
    #: ``build_tracklets``; 1.0 for a track too short to check, which is why it is
    #: reported and never gated on.
    cycle_agreement: float = 1.0
    #: Where this tracklet came from. A tracklet built from a synthetic insertion
    #: is never human ground truth.
    provenance: dict = field(default_factory=dict)

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
                "probability": [round(n.probability, 4) for n in self.nodes],
                "cycle_agreement": round(self.cycle_agreement, 4),
                "provenance": dict(self.provenance)}


def _cycle_agreement(track: Tracklet, pool: Mapping[int, Sequence[Node]],
                     camera_shift: dict[int, tuple[float, float]],
                     link_radii: float) -> float:
    """How often the track's forward links survive a reverse check.

    For each consecutive pair (a, b) that has a following node c, the position of
    ``a`` is predicted backwards from ``b`` using the velocity of the *next* step
    (b to c) and compared with where ``a`` actually is. Using the next step's
    velocity rather than the pair's own is what makes this a check rather than a
    tautology: the pair's own velocity would predict ``a`` from ``b`` exactly by
    construction.

    The v2 pass reported ``cycle_consistency_rate = 1.0`` on every final-test
    clip, including three that tracked pure clutter, so this number is reported
    and never gated on.
    """
    nodes = track.nodes
    checked = passed = 0
    for a, b, c in zip(nodes, nodes[1:], nodes[2:]):
        gap = max(1, c.frame_index - b.frame_index)
        shift_c = camera_shift.get(c.frame_index, (0.0, 0.0))
        velocity = ((c.x - b.x - shift_c[0]) / gap, (c.y - b.y - shift_c[1]) / gap)
        shift_b = camera_shift.get(b.frame_index, (0.0, 0.0))
        predicted = (b.x - velocity[0] - shift_b[0], b.y - velocity[1] - shift_b[1])
        tolerance = link_radii * max(2.0, b.radius)
        checked += 1
        if math.hypot(predicted[0] - a.x, predicted[1] - a.y) <= tolerance:
            passed += 1
    if checked == 0:
        return 1.0
    return passed / checked


def build_tracklets(per_frame: Mapping[int, Sequence[Node]],
                    camera_shift: dict[int, tuple[float, float]], *,
                    top_k: int = 8, link_radii: float = LINK_RADII,
                    min_length: int = 3, max_gap: int = 1,
                    pool_sizes: Optional[dict[int, int]] = None
                    ) -> list[Tracklet]:
    """Greedy forward linking of the top-K ranked candidates per frame.

    Camera motion is removed before distances are compared: on a panning shot the
    background moves further between frames than a struck ball does, and without
    compensation every link decision is dominated by the camera.

    ``max_gap`` is the largest frame step a track may be continued across. It
    defaults to 1, i.e. only consecutive frames link. A wider link would bridge a
    hole in the input -- an interrupted shard, a dropped decode -- and manufacture
    continuity that was never observed, which is the failure this whole stage
    exists to avoid.
    """
    indices = sorted(per_frame)
    open_tracks: list[Tracklet] = []
    finished: list[Tracklet] = []
    for position, index in enumerate(indices):
        nodes = list(per_frame.get(index) or [])[:top_k]
        # rank, pool size and the strongest competitor are properties of the
        # frame, not of the link, so they are recorded here rather than guessed
        # downstream.
        size = int((pool_sizes or {}).get(index, len(nodes)))
        for rank, node in enumerate(nodes):
            node.rank = rank
            node.frame_nodes = len(nodes)
            node.pool_size = size
            node.competitor_probability = max(
                (other.probability for other in nodes if other is not node),
                default=0.0)
        shift = camera_shift.get(index, (0.0, 0.0))
        used: set[int] = set()
        still_open: list[Tracklet] = []
        for track in open_tracks:
            last = track.nodes[-1]
            if index - last.frame_index > max_gap:
                finished.append(track)
                continue
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
    kept = [t for t in finished if len(t.nodes) >= min_length]
    for track in kept:
        track.cycle_agreement = _cycle_agreement(track, per_frame, camera_shift,
                                                 link_radii)
    return kept


def tracklet_features(track: Tracklet, *, window: int,
                      camera_shift: dict[int, tuple[float, float]]) -> list[float]:
    """Summarise a tracklet into ``FEATURE_NAMES``. Never raises on a short track.

    Fields the caller did not fill read as zero here; ``node_gaps`` says which,
    and the training-data builder refuses to write a tracklet with gaps.
    """
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
    # Both sides are camera-compensated: an uncompensated comparison measures the
    # pan, not the relationship.
    torso_agreement = 0.0
    torso_relative: list[float] = []
    paired = [(p, c) for p, c in zip(nodes, nodes[1:])
              if p.torso_centre is not None and c.torso_centre is not None]
    if paired:
        agreements = []
        for previous, current in paired:
            previous_centre = previous.torso_centre
            current_centre = current.torso_centre
            if previous_centre is None or current_centre is None:  # pragma: no cover
                continue
            gap = max(1, current.frame_index - previous.frame_index)
            shift = camera_shift.get(current.frame_index, (0.0, 0.0))
            tdx = (current_centre[0] - previous_centre[0] - shift[0]) / gap
            tdy = (current_centre[1] - previous_centre[1] - shift[1]) / gap
            cdx = (current.x - previous.x - shift[0]) / gap
            cdy = (current.y - previous.y - shift[1]) / gap
            torso_motion = math.hypot(tdx, tdy)
            torso_relative.append(math.hypot(cdx - tdx, cdy - tdy) / radius)
            if torso_motion < 0.5:
                continue
            # 1.0 when the candidate moved exactly with the torso, 0.0 when its
            # motion is unrelated in size
            agreements.append(
                max(0.0, 1.0 - math.hypot(cdx - tdx, cdy - tdy) / (torso_motion + 1e-6)))
        torso_agreement = float(np.mean(agreements)) if agreements else 0.0

    def appearance(index: int) -> float:
        values = [n.appearance[index] for n in nodes if n.appearance is not None]
        return float(np.mean(values)) if values else 0.0

    brightness = [max(0.0, n.brightness_score) for n in nodes]
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
        float(track.cycle_agreement),
        float(np.mean([n.rank / max(1, n.frame_nodes - 1)
                       for n in nodes if n.frame_nodes > 0])) if any(
            n.frame_nodes > 0 for n in nodes) else 0.0,
        float(np.mean([n.competitor_probability for n in nodes])),
        float(np.mean([max(0.0, n.chroma_score) for n in nodes])),
        float(np.max([max(0.0, n.chroma_score) for n in nodes])),
        float(np.mean(brightness)),
        float(np.std(brightness)),
        appearance(0),
        appearance(1),
        appearance(2),
        float(np.mean(torso_relative)) if torso_relative else 0.0,
        float(np.mean([max(0.0, n.radius_over_short_side) for n in nodes])),
    ]


def groups_for_mask(value: Optional[str]) -> list[str]:
    """Parse ``--mask body``/``--drop motion,chroma`` into group names."""
    if not value:
        return []
    raw = str(value).replace("+", ",")
    return [item.strip() for item in raw.split(",") if item.strip()]


def dropped_feature_names(dropped: Sequence[str]) -> list[str]:
    """The concrete feature names a list of groups expands to."""
    names: set[str] = set()
    for item in dropped:
        names.update(MOTION_GROUPS.get(item, (item,)))
    return [name for name in FEATURE_NAMES if name in names]


def mask_for_ablation(dropped: Sequence[str]) -> np.ndarray:
    """Zero out named feature groups, so an ablation removes information rather
    than changing the architecture."""
    mask = np.ones(len(FEATURE_NAMES), dtype=np.float32)
    for name in dropped_feature_names(dropped):
        mask[FEATURE_NAMES.index(name)] = 0.0
    return mask


if TORCH_AVAILABLE:

    class TrackletDiscriminator(nn.Module):
        """Is this tracklet the played ball? A small MLP, on purpose.

        The ablation mask is a *buffer*, not a training-time trick. Dropped
        columns have random first-layer weights, so a model trained with a mask
        and then run without one multiplies those random weights by live feature
        values -- for the raw 0-255 intensity group that was measured to collapse
        validation AUC from 0.9713 to 0.5612. Carrying the mask in the module
        means the checkpoint cannot be deployed in a configuration it was not
        trained in.
        """

        def __init__(self, size: int = len(FEATURE_NAMES), width: int = 64,
                     mask: Any = None) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(size, width), nn.ReLU(inplace=True),
                nn.Linear(width, width), nn.ReLU(inplace=True),
                nn.Linear(width, 1))
            if mask is None:
                mask = np.ones(size, dtype=np.float32)
            values = np.asarray(list(mask), dtype=np.float32)
            if values.shape != (size,):
                raise ValueError(
                    f"mask has {values.shape} entries, model has {size} inputs")
            self.register_buffer("feature_mask", torch.from_numpy(values))

        def forward(self, features: Any) -> Any:
            return self.net(features * self.feature_mask).squeeze(-1)


def discriminator_mask(model: Any) -> Optional[np.ndarray]:
    """The mask a loaded discriminator applies, or ``None`` if it has none."""
    buffer = getattr(model, "feature_mask", None)
    if buffer is None:
        return None
    return np.asarray(buffer.detach().cpu().numpy(), dtype=np.float32)


def masked_feature_names(model: Any) -> list[str]:
    """Which features a loaded model actually reads. Empty means it reads all."""
    mask = discriminator_mask(model)
    if mask is None:
        return []
    return [name for name, keep in zip(FEATURE_NAMES, mask) if keep == 0.0]


def save_discriminator(model: Any, path: Path) -> str:
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required to save the tracklet discriminator")
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = discriminator_mask(model)
    payload = {
        "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "mask": None if mask is None else [float(v) for v in mask],
        "feature_names": list(FEATURE_NAMES),
        "format": "tracklet-discriminator/2",
    }
    torch.save(payload, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_discriminator(path: Path, *, device_override: Any = None) -> Any:
    """Load a discriminator, mask included.

    Accepts the older bare ``state_dict`` payloads too, but a bare payload has no
    mask and is therefore assumed to be a *full* model. An ablation checkpoint
    saved in that format must be migrated with
    ``migrate_ablation_checkpoint`` rather than loaded and hoped about.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required to load the tracklet discriminator")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "net.0.weight" in payload:  # legacy bare state_dict: no mask recorded
        state, mask = payload, None
    else:
        state = payload["state"]
        mask = payload.get("mask")
    model = TrackletDiscriminator(size=int(state["net.0.weight"].shape[1]),
                                 mask=mask)
    model.load_state_dict(state, strict=False)
    if mask is not None:
        model.feature_mask.copy_(
            torch.as_tensor(np.asarray(mask, dtype=np.float32)))
    if device_override is not None:
        model = model.to(device_override)
    return model.eval()


def migrate_ablation_checkpoint(source: Path, target: Path,
                                dropped: Sequence[str]) -> dict:
    """Rewrite a legacy mask-less ablation checkpoint with its mask recorded.

    The weights are untouched; the only change is that the dropped columns are
    now zeroed inside ``forward`` as they were during training. Returns metadata
    describing what was migrated, including whether the recorded drop groups
    actually match the architecture's input width.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required to migrate a checkpoint")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    state = payload["state"] if "net.0.weight" not in payload else payload
    size = int(state["net.0.weight"].shape[1])
    if size != len(FEATURE_NAMES):
        raise ValueError(
            f"{source} has {size} inputs but FEATURE_NAMES has "
            f"{len(FEATURE_NAMES)}; its drop groups cannot be reconstructed")
    model = TrackletDiscriminator(size=size, mask=mask_for_ablation(dropped))
    model.load_state_dict(state, strict=False)
    model.eval()
    digest = save_discriminator(model, target)
    return {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "target": str(target),
        "target_sha256": digest,
        "drop_groups": list(dropped),
        "masked_features": dropped_feature_names(dropped),
        "migrated": True,
    }
