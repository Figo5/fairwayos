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
#: Bumped whenever ``extract_frames`` changes shape or meaning, so a stale on-disk
#: cache cannot be silently reused across a code change. It is one input to
#: :func:`extraction_fingerprint`, which is what a cache file is actually keyed by.
EXTRACTION_VERSION = 1

#: The subset of ``extract_frames``' timings that a cache hit inherits from the
#: run that *built* the extraction. A run reusing a cached extraction must report
#: these as inherited, never as its own measured runtime.
TIMING_KEYS_FROM_EXTRACTION = ("motion_s", "proposals_s", "ranking_s")

#: The source files whose contents decide what an extraction means. Any edit to
#: one of these changes the fingerprint, so a cache built before the edit cannot
#: be reused after it.
PREPROCESSING_MODULES = ("ball_v3_body", "ball_v3_blobs", "ball_v3_context",
                         "ball_v3_proposals", "ball_v3_ranker", "ball_v3_tracker")

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


def file_sha256(path) -> str:
    """SHA256 of a file's bytes, streamed so a large video is not held in memory."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pose_digest(poses: Optional[Mapping[int, Mapping]]) -> str:
    """Digest of the *selected* pose data actually handed to the ranker.

    Not the body-input mode: two runs can both say "body" and still feed the
    ranker different poses (a different cache, a different frame subset), and
    those produce different ranker probabilities.
    """
    import hashlib
    import json as _json

    canonical = _json.dumps(
        {str(index): {str(k): v for k, v in sorted((pose or {}).items())}
         for index, pose in sorted((poses or {}).items())},
        sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def extraction_fingerprint(*, source_sha256: str, indices: Sequence[int],
                           ranker_sha256: str, proposal_params: Any,
                           tracker_params: Any, body_mode: str,
                           poses_digest: str) -> dict:
    """Everything an extraction depends on, as one digest plus its parts.

    The old cache key was ``clip_id__first_index__count__body|nobody__vN``. It
    bound no source digest, no ranker weights, no parameters and only the first
    index, so a different stride over the same endpoints with a different
    ``top_k`` reused the same file and silently scored another configuration.
    This binds:

    * the source media's SHA256 (the pixels, not the filename);
    * the ORDERED frame indices in full;
    * the ranker weights' SHA256;
    * the proposal *and* tracker parameters, values included (so ``top_k`` and any
      generator knob is inside the key);
    * the digests of the code and preprocessing that builds an extraction;
    * the body-input mode *and* the selected poses' own digest;
    * ``EXTRACTION_VERSION``.

    A cache file carries the fingerprint it was written under and is refused if
    it disagrees with the fingerprint of the call, so a stale entry is an error,
    never a silent reuse.
    """
    import hashlib
    import json as _json

    components = {
        "extraction_version": EXTRACTION_VERSION,
        "source_sha256": source_sha256,
        "indices": [int(i) for i in indices],
        "ranker_sha256": ranker_sha256,
        "proposal_params": _canonical_params(proposal_params),
        "tracker_params": _canonical_params(tracker_params),
        "body_mode": body_mode,
        "poses_digest": poses_digest,
        "code_digest": preprocessing_digest(),
    }
    digest = hashlib.sha256(_json.dumps(components, sort_keys=True).encode(
        "utf-8")).hexdigest()
    return {"fingerprint": digest, "components": components}


def _canonical_params(params: Any) -> dict:
    if params is None:
        return {}
    values = params if isinstance(params, Mapping) else vars(params)
    return {str(k): v for k, v in sorted(values.items())}


def preprocessing_digest() -> str:
    """Digest of the code that decides what an extraction contains."""
    import hashlib
    from pathlib import Path as _Path

    here = _Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in PREPROCESSING_MODULES:
        path = here / f"{name}.py"
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def extract_frames(ranker: Any, frames: Sequence[np.ndarray],
                   indices: Sequence[int], *,
                   poses: Optional[Mapping[int, Mapping]] = None,
                   params: Optional[TrackerParams] = None,
                   proposal_params: Optional[bp.ProposalParams] = None) -> dict:
    """Everything the tracker computes that the temporal model cannot change.

    Camera motion, the proposal pool, the context stacks, the ranker
    probabilities and the body geometry are all decided per frame by the ranker
    and the generators. An ablation of the *tracklet* model cannot alter any of
    them, so recomputing them per variant is pure waste -- and at 0.74 s/frame it
    is the difference between measuring three variants and measuring eight.

    The result is JSON-serialisable so it can be cached on disk. It records
    ``body_inputs`` because the one variant that *does* change the ranker's inputs
    (feeding it no poses at all) must not be allowed to reuse a cache built with
    poses; ``track_ball_v3`` refuses that combination rather than silently
    scoring stale probabilities.
    """
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
    short_side = max(1, min(width, height))
    per_frame: dict[str, dict] = {}
    for position, index in enumerate(indices):
        proposals = per_frame_proposals.get(index) or []
        pose = poses.get(index)
        centre = bbody.torso_centre(pose)
        entry: dict = {
            "pool_size": len(proposals),
            "shift": [float(shifts[index][0]), float(shifts[index][1])],
            "torso_centre": None if centre is None else [float(centre[0]),
                                                        float(centre[1])],
            # recorded, never inferred from a zero body vector
            "body_available": bool(len(bbody.visible_points(pose)) >= 4),
            "null_probability": 1.0, "margin": 0.0,
            "probabilities": [], "candidates": [],
        }
        if proposals:
            views = bctx.FrameViews(frames[position], residuals[position])
            stacks = views.stack_many([(p.x, p.y, p.radius) for p in proposals])
            numeric = np.stack([
                np.asarray(brank.numeric_features(p, pose, width, height),
                           dtype=np.float32) for p in proposals])
            result = brank.rank_frame(ranker, stacks, numeric, device=device)
            order = np.argsort(-result["probabilities"])[:params.top_k]
            entry["null_probability"] = float(result["null_probability"])
            entry["margin"] = float(result["margin"])
            for rank, slot in enumerate(order):
                slot = int(slot)
                proposal = proposals[slot]
                body = bbody.features(pose, proposal.x, proposal.y,
                                      proposal.radius)
                scores = getattr(proposal, "method_scores", {}) or {}
                entry["candidates"].append({
                    "x": float(proposal.x), "y": float(proposal.y),
                    "radius": float(proposal.radius),
                    "probability": float(result["probabilities"][slot]),
                    "margin": float(result["margin"]) if rank == 0 else 0.0,
                    "is_top1": bool(rank == 0),
                    "inside_torso": float(
                        body[bbody.FEATURE_NAMES.index("inside_torso")]),
                    "distance_to_hands": float(body[bbody.FEATURE_NAMES.index(
                        "distance_to_hands_over_radius")] * 200.0),
                    "chroma_score": float(scores.get("local_distinctness", -1.0)),
                    "brightness_score": float(scores.get("brightness_cue", -1.0)),
                    "radius_over_short_side": float(proposal.radius) / short_side,
                    "appearance": [float(v) for v in _stack_appearance(stacks[slot])],
                })
            entry["probabilities"] = [float(result["probabilities"][int(s)])
                                      for s in order]
        per_frame[str(index)] = entry
    timings["ranking_s"] = round(time.time() - started, 3)

    return {
        "version": EXTRACTION_VERSION,
        "body_inputs": bool(poses),
        "short_side": short_side,
        "info": info,
        "timings": timings,
        "shifts": {str(k): [float(v[0]), float(v[1])] for k, v in shifts.items()},
        "frames": per_frame,
    }


def _nodes_from_extraction(extraction: Mapping, indices: Sequence[int]
                           ) -> tuple[dict[int, dict], dict[int, list], dict[int, int]]:
    """Rebuild the tracker's per-frame node pools from an extraction."""
    ranked: dict[int, dict] = {}
    nodes_by_frame: dict[int, list] = {}
    pool_sizes: dict[int, int] = {}
    for index in indices:
        entry = extraction["frames"].get(str(index)) or {}
        pool_sizes[index] = int(entry.get("pool_size") or 0)
        ranked[index] = {
            "probabilities": np.asarray(entry.get("probabilities") or [],
                                        dtype=np.float32),
            "null_probability": float(entry.get("null_probability", 1.0)),
            "margin": float(entry.get("margin", 0.0)),
        }
        raw_centre = entry.get("torso_centre")
        centre = (None if raw_centre is None
                  else (float(raw_centre[0]), float(raw_centre[1])))
        body_available = bool(entry.get("body_available", False))
        nodes = []
        for candidate in entry.get("candidates") or []:
            appearance = candidate.get("appearance")
            nodes.append(btrk.Node(
                frame_index=index, x=candidate["x"], y=candidate["y"],
                radius=candidate["radius"],
                probability=candidate["probability"],
                margin=candidate.get("margin", 0.0),
                is_top1=bool(candidate.get("is_top1")),
                inside_torso=candidate.get("inside_torso", 0.0),
                distance_to_hands=candidate.get("distance_to_hands", 0.0),
                torso_centre=centre,
                chroma_score=candidate.get("chroma_score", -1.0),
                brightness_score=candidate.get("brightness_score", -1.0),
                radius_over_short_side=candidate.get("radius_over_short_side", -1.0),
                appearance=(None if appearance is None else tuple(appearance)),
                body_available=body_available))
        nodes_by_frame[index] = nodes
    return ranked, nodes_by_frame, pool_sizes


def track_ball_v3(frames: Sequence[np.ndarray], indices: Sequence[int], *,
                  ranker: Any, discriminator: Any = None,
                  poses: Optional[Mapping[int, Mapping]] = None,
                  params: Optional[TrackerParams] = None,
                  proposal_params: Optional[bp.ProposalParams] = None,
                  provenance: Optional[dict] = None,
                  extraction: Optional[dict] = None) -> TrackResult:
    """Run the whole pipeline over consecutive frames. Local only, no network.

    ``extraction`` is an optional cached result of :func:`extract_frames`. Passing
    it skips decode-adjacent work that does not depend on the temporal model, so
    the same frames can be re-decided under several ablation variants without
    re-running proposals and the ranker each time.
    """
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

    if extraction is None:
        extraction = extract_frames(ranker, frames, indices, poses=poses,
                                    params=params,
                                    proposal_params=proposal_params)
    else:
        # Reusing a cached extraction. Everything that does not depend on the
        # temporal model -- camera motion, proposals, context stacks, ranker
        # probabilities, body geometry -- is already in the dict, so only
        # association and the tracklet model rerun. This is what makes eight
        # ablation variants affordable instead of eight full passes; the
        # alternative was measuring only the variants that fit in the budget.
        if bool(extraction.get("body_inputs")) != bool(poses):
            raise ValueError(
                "cached extraction was built with different body inputs than "
                "this call supplies; ranker probabilities would be stale")
        timings["extraction_reused"] = 1.0
    assert extraction is not None

    info = extraction.get("info") or {}
    shifts = {int(key): (float(value[0]), float(value[1]))
              for key, value in extraction["shifts"].items()}
    ranked, nodes_by_frame, pool_sizes = _nodes_from_extraction(
        extraction, indices)
    timings.update({k: v for k, v in (extraction.get("timings") or {}).items()})

    started = time.time()
    tracklets = btrk.build_tracklets(nodes_by_frame, shifts, top_k=params.top_k,
                                     min_length=max(2, params.min_support - 1),
                                     pool_sizes=pool_sizes)
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


def _stack_appearance(stack: np.ndarray) -> tuple[float, float, float]:
    """Three scalar statistics of a candidate's own stored view.

    ``candidate_stack`` lays out tight BGR (0:3), context BGR (3:6) and the
    motion residual (6). The tight crop is the candidate itself, so its mean and
    spread describe the object; the context crop's spread describes how busy the
    scene around it is. This is a summary, not a learned embedding -- the ranker
    already reads the same crops through a CNN.
    """
    tight = stack[..., :3].astype(np.float32)
    context = stack[..., 3:6].astype(np.float32)
    return (float(tight.mean()), float(tight.std()), float(context.std()))


def _track_probability(discriminator: Any, features: Sequence[float],
                       device: Any) -> float:
    """The tracklet model's answer, or an explicit heuristic when it is absent.

    The fallback is stated rather than hidden because an ablation runs with the
    discriminator removed, and a silent 1.0 there would make the ablation
    meaningless.

    The tensor is placed on the *discriminator's own* device rather than on the
    tracker's: ``load_discriminator`` returns a CPU model, and sending an MPS
    tensor into a CPU convolution raises rather than silently falling back. That
    is a real failure this pass hit, so the placement is derived from the weights.
    """
    if discriminator is None:
        named = dict(zip(btrk.FEATURE_NAMES, features))
        return float(np.clip(
            0.5 * named["mean_rank_probability"] + 0.3 * named["top1_fraction"]
            + 0.2 * (1.0 - named["static_fraction"]), 0.0, 1.0))
    import torch
    parameter = next(discriminator.parameters())
    with torch.no_grad():
        tensor = torch.from_numpy(
            np.asarray([features], dtype=np.float32)).to(parameter.device)
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
