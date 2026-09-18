"""The v3 tracker: propose widely, rank comparatively, decide over time, abstain.

Four stages, each of which exists because Phase 0 measured the v2 equivalent
failing:

1. **propose** (``ball_v3_proposals``) -- v2 proposed the ball on 31.0% of real
   labelled frames, so 69% of its loss was upstream of any model. v3 proposes
   generously and applies no score floor.
2. **rank** (``ball_v3_ranker``) -- a listwise softmax over the whole frame's
   candidates plus a learned null slot, so the model is trained on the comparison
   the tracker actually needs and abstention is a first-class answer.
3. **link and score over time** (``ball_v3_tracklets``) -- appearance alone does
   not settle it; what separates a ball from a shirt highlight is that the
   highlight keeps travelling with the torso.
4. **accept, or say so** -- and the bar is deliberately hard to clear.

On that last point, the thing this pass is most concerned not to repeat: the v2
final test accepted three clips that were tracking pure clutter, at nine false
positives each, and every one of them scored ``cycle_consistency_rate = 1.0``.
Static clutter passes a forward/backward check *trivially* -- it is the easiest
thing in the frame for optical flow to follow. A confident-looking consistency
number is therefore not evidence, and this module never gates on one.

``UNAVAILABLE`` and ``AMBIGUOUS`` are successful outputs. A position is emitted
only for frames the accepted tracklet actually holds; a frame it does not hold
gets a state and no coordinates, and no last-known position is ever carried
forward.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

from . import ball_v3_body as bbody
from . import ball_v3_context as bctx
from . import ball_v3_proposals as bp
from . import ball_v3_ranker as brank
from . import ball_v3_tracklets as btrk

VERSION = "ball_tracker_v3_context"

OBSERVED = "OBSERVED"
AMBIGUOUS = "AMBIGUOUS"
OCCLUDED = "OCCLUDED"
LOST = "LOST"
UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class TrackerParams:
    """The acceptance surface. Every number is argued for where it is used."""

    #: Candidates per frame carried into tracklet linking.
    top_k: int = 8
    #: Frames a tracklet must hold before it can be accepted at all.
    min_support: int = 4
    #: Softmax probability the winning candidate needs on a frame to be OBSERVED.
    min_probability: float = 0.35
    #: Gap to the runner-up. This is the signal v2 did not have: a candidate that
    #: wins by 0.003 has not won. A near-tie becomes AMBIGUOUS, never OBSERVED.
    min_margin: float = 0.15
    #: Tracklet-model probability required to accept a track.
    min_track_probability: float = 0.5
    #: A track whose frames are mostly the model saying "none of these" is not a
    #: track, however well its candidates link up.
    max_mean_null_probability: float = 0.6
    #: Fraction of frames a track may spend essentially motionless after camera
    #: compensation before it is treated as scenery rather than a ball.
    max_static_fraction: float = 0.9
    #: Fraction of frames a track may spend inside the torso polygon.
    max_body_inside_fraction: float = 0.5
    #: Interpolation is off by default and can never produce OBSERVED.
    interpolate: bool = False


@dataclass
class FrameState:
    frame_index: int
    state: str
    x: Optional[float] = None
    y: Optional[float] = None
    radius: Optional[float] = None
    probability: Optional[float] = None
    margin: Optional[float] = None
    null_probability: Optional[float] = None
    interpolated: bool = False
    diagnostics: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index, "state": self.state,
            "x": None if self.x is None else round(self.x, 2),
            "y": None if self.y is None else round(self.y, 2),
            "radius": None if self.radius is None else round(self.radius, 2),
            "probability": None if self.probability is None else round(self.probability, 4),
            "margin": None if self.margin is None else round(self.margin, 4),
            "null_probability": None if self.null_probability is None
            else round(self.null_probability, 4),
            "interpolated": self.interpolated,
            "diagnostics": list(self.diagnostics), "reasons": list(self.reasons),
        }


@dataclass
class TrackResult:
    states: list[FrameState]
    accepted: bool
    confidence: dict
    provenance: dict
    rejection_reasons: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)

    @property
    def observed(self) -> list[FrameState]:
        return [s for s in self.states if s.state == OBSERVED]

    def to_dict(self) -> dict:
        return {"method": VERSION, "accepted": self.accepted,
                "states": [s.to_dict() for s in self.states],
                "confidence": self.confidence, "provenance": self.provenance,
                "rejection_reasons": list(self.rejection_reasons),
                "timings": self.timings}


def camera_shifts(frames: Sequence[np.ndarray], indices: Sequence[int]
                  ) -> tuple[dict[int, tuple[float, float]], list[Optional[np.ndarray]]]:
    """Global translation per frame, and the compensated motion residual."""
    shifts: dict[int, tuple[float, float]] = {}
    residuals: list[Optional[np.ndarray]] = []
    previous_gray = None
    for frame, index in zip(frames, indices):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous_gray is None:
            shifts[index] = (0.0, 0.0)
            residuals.append(None)
        else:
            window = cv2.createHanningWindow((gray.shape[1], gray.shape[0]), cv2.CV_32F)
            (dx, dy), _response = cv2.phaseCorrelate(
                previous_gray.astype(np.float32), gray.astype(np.float32), window)
            shifts[index] = (float(dx), float(dy))
            residuals.append(bctx.motion_residual(
                gray, bctx.compensate(previous_gray, (float(dx), float(dy)))))
        previous_gray = gray
    return shifts, residuals


def track_ball_v3(frames: Sequence[np.ndarray], indices: Sequence[int], *,
                  ranker: Any, discriminator: Any = None,
                  poses: Optional[Mapping[int, Mapping]] = None,
                  params: Optional[TrackerParams] = None,
                  proposal_params: Optional[bp.ProposalParams] = None,
                  provenance: Optional[dict] = None) -> TrackResult:
    """Run the whole pipeline over consecutive frames. Local only, no network."""
    import time

    if len(frames) != len(indices):
        raise ValueError("frames and indices must be the same length")
    if not frames:
        raise ValueError("at least one frame is required")
    params = params or TrackerParams()
    proposal_params = proposal_params or bp.ProposalParams()
    poses = poses or {}
    device = brank.device()
    timings: dict[str, float] = {}

    height, width = frames[0].shape[:2]
    started = time.time()
    shifts, residuals = camera_shifts(frames, indices)
    timings["motion_s"] = round(time.time() - started, 3)

    started = time.time()
    per_frame_proposals, info = bp.propose(frames, indices, params=proposal_params)
    timings["proposals_s"] = round(time.time() - started, 3)

    started = time.time()
    ranked: dict[int, dict] = {}
    nodes_by_frame: dict[int, list[btrk.Node]] = {}
    for position, index in enumerate(indices):
        proposals = per_frame_proposals.get(index) or []
        if not proposals:
            ranked[index] = {"best": None, "null_probability": 1.0, "margin": 0.0,
                             "probabilities": np.zeros(0)}
            nodes_by_frame[index] = []
            continue
        pose = poses.get(index)
        stacks = np.stack([
            bctx.candidate_stack(frames[position], residuals[position],
                                 p.x, p.y, p.radius) for p in proposals])
        numeric = np.stack([
            np.asarray(brank.numeric_features(p, pose, width, height),
                       dtype=np.float32) for p in proposals])
        result = brank.rank_frame(ranker, stacks, numeric, device=device)
        ranked[index] = result
        order = np.argsort(-result["probabilities"])[:params.top_k]
        centre = bbody.torso_centre(pose)
        nodes = []
        for rank, slot in enumerate(order):
            proposal = proposals[int(slot)]
            body = bbody.features(pose, proposal.x, proposal.y, proposal.radius)
            nodes.append(btrk.Node(
                frame_index=index, x=proposal.x, y=proposal.y,
                radius=proposal.radius,
                probability=float(result["probabilities"][int(slot)]),
                margin=result["margin"] if rank == 0 else 0.0,
                is_top1=rank == 0,
                inside_torso=body[bbody.FEATURE_NAMES.index("inside_torso")],
                distance_to_hands=body[
                    bbody.FEATURE_NAMES.index("distance_to_hands_over_radius")] * 200.0,
                torso_centre=centre))
        nodes_by_frame[index] = nodes
    timings["ranking_s"] = round(time.time() - started, 3)

    started = time.time()
    tracklets = btrk.build_tracklets(nodes_by_frame, shifts, top_k=params.top_k,
                                     min_length=max(2, params.min_support - 1))
    scored = []
    for track in tracklets:
        features = btrk.tracklet_features(track, window=len(indices),
                                          camera_shift=shifts)
        probability = _track_probability(discriminator, features, device)
        scored.append((probability, features, track))
    scored.sort(key=lambda item: item[0], reverse=True)
    timings["association_s"] = round(time.time() - started, 3)

    mean_null = float(np.mean([ranked[i]["null_probability"] for i in indices])) \
        if indices else 1.0

    states = {index: FrameState(frame_index=index, state=UNAVAILABLE)
              for index in indices}
    if not scored:
        for state in states.values():
            state.state = LOST
            state.reasons.append("no tracklet survived linking")
        confidence = _confidence(None, None, states, mean_null, indices)
        return TrackResult(states=[states[i] for i in indices], accepted=False,
                           confidence=confidence,
                           provenance=_provenance(provenance, info, discriminator),
                           rejection_reasons=["no tracklet survived linking"],
                           timings=timings)

    probability, features, best = scored[0]
    named = dict(zip(btrk.FEATURE_NAMES, features))
    rival_probability = scored[1][0] if len(scored) > 1 else 0.0

    reasons: list[str] = []
    if len(best.nodes) < params.min_support:
        reasons.append(f"track holds {len(best.nodes)} frames, below "
                       f"{params.min_support}")
    if probability < params.min_track_probability:
        reasons.append(f"tracklet probability {probability:.2f} below "
                       f"{params.min_track_probability:.2f}")
    if mean_null > params.max_mean_null_probability:
        reasons.append(f"the ranker's mean 'no ball here' probability is "
                       f"{mean_null:.2f}, above {params.max_mean_null_probability:.2f}")
    if named["static_fraction"] > params.max_static_fraction:
        reasons.append(f"static for {named['static_fraction']:.0%} of its frames "
                       "after camera compensation: scenery, not a ball")
    if named["body_inside_fraction"] > params.max_body_inside_fraction:
        reasons.append(f"inside the torso for {named['body_inside_fraction']:.0%} "
                       "of its frames")
    accepted = not reasons

    held = {node.frame_index: node for node in best.nodes}
    for index in indices:
        state = states[index]
        node = held.get(index)
        result = ranked[index]
        state.null_probability = result["null_probability"]
        if node is None:
            state.state = LOST if accepted else UNAVAILABLE
            state.reasons.append("the accepted track does not hold this frame"
                                 if accepted else "no track was accepted")
            continue
        state.probability = node.probability
        state.margin = result["margin"]
        if not accepted:
            state.state = AMBIGUOUS
            state.reasons.extend(reasons)
            continue
        if node.probability < params.min_probability:
            state.state = AMBIGUOUS
            state.reasons.append(
                f"candidate probability {node.probability:.2f} below "
                f"{params.min_probability:.2f}")
        elif result["margin"] < params.min_margin:
            state.state = AMBIGUOUS
            state.reasons.append(
                f"beats the runner-up by only {result['margin']:.3f}; a near-tie "
                "is not an observation")
        else:
            state.state = OBSERVED
            state.x, state.y, state.radius = node.x, node.y, node.radius
        if node.inside_torso >= 0.5:
            state.diagnostics.append("BODY_LOCK")

    confidence = _confidence(probability, named, states, mean_null, indices)
    confidence["rival_track_probability"] = round(float(rival_probability), 4)
    confidence["track_probability_margin"] = round(
        float(probability - rival_probability), 4)
    return TrackResult(states=[states[i] for i in indices], accepted=accepted,
                       confidence=confidence,
                       provenance=_provenance(provenance, info, discriminator),
                       rejection_reasons=reasons, timings=timings)


def _track_probability(discriminator: Any, features: Sequence[float],
                       device: Any) -> float:
    """The tracklet model's answer, or an explicit heuristic when it is absent.

    The fallback is stated rather than hidden because an ablation runs with the
    discriminator removed, and a silent 1.0 there would make the ablation
    meaningless.
    """
    if discriminator is None:
        named = dict(zip(btrk.FEATURE_NAMES, features))
        return float(np.clip(
            0.5 * named["mean_rank_probability"] + 0.3 * named["top1_fraction"]
            + 0.2 * (1.0 - named["static_fraction"]), 0.0, 1.0))
    import torch
    with torch.no_grad():
        tensor = torch.from_numpy(
            np.asarray([features], dtype=np.float32)).to(device)
        return float(torch.sigmoid(discriminator(tensor)).cpu().numpy()[0])


def _confidence(track_probability, named, states, mean_null, indices) -> dict:
    observed = [s for s in states.values() if s.state == OBSERVED]
    margins = [s.margin for s in states.values() if s.margin is not None]
    counts: dict[str, int] = {}
    for state in states.values():
        counts[state.state] = counts.get(state.state, 0) + 1
    total = max(1, len(indices))
    body_locked = sum(1 for s in states.values() if "BODY_LOCK" in s.diagnostics)
    return {
        "track_probability": None if track_probability is None
        else round(float(track_probability), 4),
        "observed_frames": len(observed),
        "span_frames": len(indices),
        "coverage": round(len(observed) / total, 4),
        "abstention_rate": round(
            sum(1 for s in states.values()
                if s.state in (AMBIGUOUS, UNAVAILABLE, LOST)) / total, 4),
        "mean_null_probability": round(mean_null, 4),
        "median_margin": round(float(np.median(margins)), 4) if margins else None,
        "body_lock_rate": round(body_locked / max(1, len(observed)), 4)
        if observed else 0.0,
        "static_fraction": None if named is None
        else round(float(named["static_fraction"]), 4),
        "body_inside_fraction": None if named is None
        else round(float(named["body_inside_fraction"]), 4),
        "state_counts": counts,
    }


def _provenance(provenance, info, discriminator) -> dict:
    out = dict(provenance or {})
    out.setdefault("method", VERSION)
    out.setdefault("no_network", True)
    out.setdefault("operator_seed", None)
    out.setdefault("research_only", True)
    out.setdefault("ground_truth", False)
    out["proposals"] = info
    out["tracklet_model"] = ("local TrackletDiscriminator" if discriminator is not None
                             else "none (stated heuristic)")
    return out
