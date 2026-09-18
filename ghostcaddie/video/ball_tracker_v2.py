"""Ball tracker v2: ensemble + learned detector + temporal verification.

The v1 pipeline failed by tracking one clip of ten correctly and, on the held-out
split, locking onto the golfer's shirt for fifteen frames. This pipeline is built
around that failure rather than around a hopeful detector:

* fusing three independent evidence sources (a learned detector, a classical
  candidate ensemble, and temporal verification) instead of trusting one;
* naming failure modes rather than silently subtracting them;
* preferring abstention over a wrong identity -- a track is either accepted with
  evidence behind it, or reported with the reason it was not accepted;
* never inventing an observation: a missing frame is OCCLUDED, LOST or
  UNAVAILABLE, and any interpolated position is labelled as interpolation and can
  never be counted as an observation.

Per-frame states: ``OBSERVED``, ``AMBIGUOUS``, ``OCCLUDED``, ``LOST``,
``UNAVAILABLE``.

Everything this module needs is local: no network, no external service, no seed
from the operator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

from . import ball_candidates as bc
from . import ball_consensus as bcon
from . import ball_detector as bdet

STATE_OBSERVED = "OBSERVED"
STATE_AMBIGUOUS = "AMBIGUOUS"
STATE_OCCLUDED = "OCCLUDED"
STATE_LOST = "LOST"
STATE_UNAVAILABLE = "UNAVAILABLE"
TRACK_STATES = (STATE_OBSERVED, STATE_AMBIGUOUS, STATE_OCCLUDED, STATE_LOST,
                STATE_UNAVAILABLE)

VERSION = "fairwayos-ball-v2"


@dataclass(frozen=True)
class TrackerParams:
    """The decision surface, each number argued where it is used."""

    #: Learned-detector probability required for an observation. Set from the
    #: detector's own validation operating point, not from the test set.
    detector_threshold: float = 0.5
    #: Independent classical methods needed on top of a detector hit. Two means the
    #: detector plus one physical cue; a detector hit alone is not an observation.
    min_methods: int = 2
    #: Spatial agreement for two candidates to describe one object, in radii.
    agreement_radii: float = 2.0
    #: Frames the ball may be unobserved while a track stays alive.
    max_gap: int = 5
    #: Forward/backward round-trip tolerance, in radii.
    cycle_radii: float = 1.5
    cycle_span: int = 3
    #: Largest displacement per frame, and change in displacement, in radii.
    max_step_radii: float = 8.0
    max_accel_radii: float = 5.0
    #: Rival hypothesis with a score within this fraction of the best one makes the
    #: frame ambiguous instead of decided.
    competitor_ratio: float = 0.85
    #: How close a rival's *track-level* mean detector probability has to be before it
    #: counts as a genuinely competing hypothesis. Frame-level ties are everywhere on
    #: grass; a rival detected worse over its whole track is not a rival.
    competitor_detector_margin: float = 0.05
    #: Track acceptance rules. Coverage is the observed fraction of the span; the
    #: lock ceilings exist because those were the v1 failure modes.
    min_observed_frames: int = 3
    min_coverage: float = 0.30
    min_mean_score: float = 0.45
    min_cycle_rate: float = 0.50
    max_body_lock_rate: float = 0.10
    max_static_lock_rate: float = 0.15
    max_abrupt_jumps: int = 2
    #: Body/clothing regions shrink the accepted area: a candidate inside one needs
    #: the detector at this probability *and* a passing cycle check to be observed.
    body_region_detector_bonus: float = 0.25


@dataclass
class FrameState:
    frame_index: int
    state: str
    x: Optional[float] = None
    y: Optional[float] = None
    radius: Optional[float] = None
    detector_probability: Optional[float] = None
    methods: list[str] = field(default_factory=list)
    score: float = 0.0
    cycle: dict = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    interpolated: bool = False

    def to_dict(self) -> dict:
        payload = {"frame_index": self.frame_index, "state": self.state}
        if self.state == STATE_OBSERVED or self.interpolated:
            payload["x"] = None if self.x is None else round(self.x, 2)
            payload["y"] = None if self.y is None else round(self.y, 2)
            if self.radius:
                payload["radius"] = round(self.radius, 2)
        payload["methods"] = list(self.methods)
        payload["score"] = round(self.score, 4)
        if self.detector_probability is not None:
            payload["detector_probability"] = round(self.detector_probability, 4)
        if self.cycle:
            payload["cycle"] = self.cycle
        if self.diagnostics:
            payload["diagnostics"] = list(self.diagnostics)
        if self.reasons:
            payload["reasons"] = list(self.reasons)
        if self.interpolated:
            payload["interpolated"] = True
        return payload


@dataclass
class BallTrackResult:
    states: list[FrameState]
    accepted: bool
    confidence: dict
    reasons: list[str]
    provenance: dict

    @property
    def observed(self) -> list[FrameState]:
        return [state for state in self.states if state.state == STATE_OBSERVED]

    def states_by_index(self) -> dict[int, FrameState]:
        return {state.frame_index: state for state in self.states}

    def to_dict(self) -> dict:
        return {
            "version": VERSION,
            "accepted": self.accepted,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "provenance": self.provenance,
            "frames": [state.to_dict() for state in self.states],
        }


def _crop_for_detector(frame: np.ndarray, x: float, y: float, radius: float
                       ) -> tuple[Optional[np.ndarray], float]:
    """Crop around a hypothesis and resize to the detector's input size.

    The window is scaled to the ball so a macro ball and a 3 px ball both arrive
    as a 64x64 crop; the returned factor maps the detector's answer back to native
    pixels.
    """
    size = int(max(48, min(192, round(radius * 8))))
    half = size // 2
    left, top = int(round(x)) - half, int(round(y)) - half
    if left < 0 or top < 0 or left + size > frame.shape[1] or top + size > frame.shape[0]:
        return None, 1.0
    patch = frame[top:top + size, left:left + size]
    resized = cv2.resize(patch, (bdet.CROP_SIZE, bdet.CROP_SIZE),
                         interpolation=cv2.INTER_AREA if size > bdet.CROP_SIZE
                         else cv2.INTER_LINEAR)
    return resized, size / float(bdet.CROP_SIZE)


def track_ball_v2(frames: Sequence[np.ndarray], indices: Sequence[int], *,
                  detector: Any = None, params: Optional[TrackerParams] = None,
                  body_regions: Optional[Mapping[int, Sequence[Mapping]]] = None,
                  interpolate: bool = False,
                  provenance: Optional[dict] = None) -> BallTrackResult:
    """Run the full pipeline over consecutive frames."""
    if len(frames) != len(indices):
        raise ValueError("frames and indices must be the same length")
    if not frames:
        raise ValueError("at least one frame is required")
    params = params or TrackerParams()
    provenance = dict(provenance or {})
    provenance.setdefault("method", VERSION)
    provenance.setdefault("no_network", True)
    provenance.setdefault("operator_seed", None)
    provenance.setdefault("detector", "none (classical ensemble only)"
                          if detector is None else "local BallDetector heatmap CNN")

    contexts, context_info = bc.build_contexts(frames, indices)
    per_frame_candidates = bc.ensemble_candidates(contexts)
    # Working-scale shift converted to native pixels: the candidate positions these
    # are compared against are native.
    motion_px = [math.hypot(*context.motion) * float(context.scale or 1.0)
                 for context in contexts]

    # --- hypotheses per frame -------------------------------------------------
    hypotheses: dict[int, list[dict]] = {}
    for position, context in enumerate(contexts):
        frame = frames[position]
        clusters = bcon.cluster_candidates(per_frame_candidates.get(context.index, []),
                                           params=bcon.ConsensusParams(
                                               agreement_radii=params.agreement_radii))
        entries = []
        for cluster in clusters:
            crop, factor = _crop_for_detector(frame, cluster.x, cluster.y, cluster.radius)
            probability, refined = None, (cluster.x, cluster.y)
            if crop is not None and detector is not None:
                scored = bdet.score_crops(detector, [crop],
                                          device=bdet.device())[0]
                probability = scored["presence"]
                centre_x, centre_y = scored["center_in_crop"]
                refined = (cluster.x + (centre_x - bdet.CROP_SIZE / 2.0) * factor,
                           cluster.y + (centre_y - bdet.CROP_SIZE / 2.0) * factor)
            entries.append({
                "cluster": cluster,
                "x": refined[0],
                "y": refined[1],
                "radius": cluster.radius,
                "methods": cluster.methods,
                "detector": probability,
            })
        # score: the detector decides, agreement and cluster strength support it
        for entry in entries:
            detector_term = 0.5 if entry["detector"] is None else entry["detector"]
            agreement = min(1.0, len(entry["methods"]) / 3.0)
            entry["score"] = round(0.65 * detector_term + 0.25 * agreement
                                   + 0.10 * entry["cluster"].best_score, 4)
        entries.sort(key=lambda item: item["score"], reverse=True)
        hypotheses[context.index] = entries

    # --- track-level rival comparison ----------------------------------------
    # Optical flow will happily track a grass tuft, a divot or a shoe for three
    # frames, so "the rival is temporally coherent" is not enough to call a frame
    # ambiguous: on real footage nearly every clutter blob passes that test. The rival
    # has to hold up as a *track* -- present in a comparable number of frames and
    # detected about as well -- before two answers are treated as equally defensible.
    def _rival_for(entries: list[dict]) -> Optional[dict]:
        if len(entries) < 2:
            return None
        best = entries[0]
        for entry in entries[1:]:
            far_apart = math.hypot(entry["x"] - best["x"], entry["y"] - best["y"]) > \
                4.0 * max(entry["radius"], best["radius"])
            if far_apart and entry["score"] >= params.competitor_ratio * best["score"]:
                return entry
        return None

    rival_by_index: dict[int, Optional[dict]] = {
        index: _rival_for(entries) for index, entries in hypotheses.items()}

    def _evidence(sequence: list[Optional[dict]]) -> tuple[float, int]:
        probabilities = [entry["detector"] for entry in sequence
                         if entry is not None and entry["detector"] is not None]
        if not probabilities:
            return 0.0, 0
        return float(sum(probabilities) / len(probabilities)), len(probabilities)

    best_mean, best_count = _evidence([entries[0] if entries else None
                                       for entries in hypotheses.values()])
    rival_mean, rival_count = _evidence(list(rival_by_index.values()))
    rival_is_credible = bool(
        rival_count >= max(3, int(0.6 * best_count))
        and rival_mean >= best_mean - params.competitor_detector_margin)
    competitor_report = {
        "best_track_mean_detector": round(best_mean, 4),
        "best_track_frames": best_count,
        "rival_track_mean_detector": round(rival_mean, 4),
        "rival_track_frames": rival_count,
        "rival_credible": rival_is_credible,
    }

    chosen: dict[int, Optional[dict]] = {}
    states: list[FrameState] = []
    for position, context in enumerate(contexts):
        entries = hypotheses[context.index]
        frame = frames[position]
        state = FrameState(frame_index=context.index, state=STATE_UNAVAILABLE)
        if not entries:
            previous = [s for s in states if s.state in (STATE_OBSERVED, STATE_OCCLUDED)]
            state.state = STATE_LOST if previous else STATE_UNAVAILABLE
            if previous:
                state.reasons.append("no candidate survived clustering")
            states.append(state)
            continue

        best = entries[0]
        rival = rival_by_index.get(context.index)
        if rival is not None and rival_is_credible:
            state.state = STATE_AMBIGUOUS
            state.x, state.y = best["x"], best["y"]
            state.radius = best["radius"]
            state.methods = best["methods"]
            state.score = best["score"]
            state.detector_probability = best["detector"]
            state.reasons.append(
                f"competing hypothesis {rival['score']:.2f} vs {best['score']:.2f}, and "
                f"the rival survives as a track too (mean detector {rival_mean:.2f} over "
                f"{rival_count} frames vs {best_mean:.2f} over {best_count}); two "
                "defensible answers is not a decision")
            states.append(state)
            continue
        if rival is not None:
            state.reasons.append(
                f"a rival candidate scored {rival['score']:.2f} vs {best['score']:.2f} "
                "but does not hold up as a track; the decision stands on this one")

        # temporal verification on the candidate itself
        cycle = bcon.track_patch_both_ways(contexts, context.index, best["x"], best["y"],
                                           best["radius"], bcon.ConsensusParams(
                                               cycle_radii=params.cycle_radii,
                                               cycle_span=params.cycle_span))
        regions = (body_regions or {}).get(context.index, [])
        body = bcon.body_lock(best["x"], best["y"], best["radius"], regions) if regions \
            else None
        diagnostics = bcon.classify_diagnostics(best["cluster"],
                                                temporal=cycle,
                                                static={"locked": False}, body=body)
        state.methods = best["methods"]
        state.score = best["score"]
        state.detector_probability = best["detector"]
        state.cycle = {key: value for key, value in cycle.items()
                       if key in ("supported", "round_trip_px", "appearance_ncc")}
        state.diagnostics = diagnostics

        required_methods = params.min_methods
        if best["detector"] is None:
            # No learned detector: the bar rises to three independent methods rather
            # than silently accepting what a weaker evidence base supports.
            detector_ok = len(best["methods"]) >= 3
            required_methods = 3
            detector_note = ("no learned detector is loaded, so three independent "
                             "methods are required")
        else:
            threshold = params.detector_threshold
            if body and body.get("locked"):
                threshold += params.body_region_detector_bonus
                state.reasons.append("inside a body or clothing region, so a higher "
                                     "detector probability is required")
            detector_ok = best["detector"] >= threshold
            detector_note = (f"detector probability {best['detector']:.2f} below "
                             f"{threshold:.2f}")
        methods_ok = len(best["methods"]) >= required_methods
        cycle_ok = bool(cycle.get("supported"))
        if detector_ok and methods_ok and cycle_ok:
            state.state = STATE_OBSERVED
            state.x, state.y, state.radius = best["x"], best["y"], best["radius"]
        elif detector_ok and methods_ok and not cycle_ok:
            state.state = STATE_AMBIGUOUS
            state.x, state.y, state.radius = best["x"], best["y"], best["radius"]
            state.reasons.append("temporal verification failed: "
                                 + str(cycle.get("reason", "round trip did not return")))
        elif not methods_ok:
            state.state = STATE_AMBIGUOUS
            state.x, state.y, state.radius = best["x"], best["y"], best["radius"]
            state.reasons.append(f"only {len(best['methods'])} independent method(s); "
                                 f"{required_methods} required")
        else:
            state.state = STATE_AMBIGUOUS
            state.x, state.y, state.radius = best["x"], best["y"], best["radius"]
            state.reasons.append(detector_note)
        states.append(state)

    # --- bracketed gaps ------------------------------------------------------
    observed_indices = [s.frame_index for s in states if s.state == STATE_OBSERVED]
    for state in states:
        if state.state in (STATE_LOST, STATE_UNAVAILABLE):
            before = [i for i in observed_indices if i < state.frame_index]
            after = [i for i in observed_indices if i > state.frame_index]
            if before and after and (after[0] - before[-1]) <= params.max_gap + 1:
                state.state = STATE_OCCLUDED
                state.reasons.append("bracketed by observations on both sides")

    # --- optional interpolation (never an observation) -----------------------
    if interpolate:
        by_index = {state.frame_index: state for state in states}
        for position, state in enumerate(states):
            if state.state != STATE_OCCLUDED:
                continue
            before = [states[i] for i in range(position) if states[i].state == STATE_OBSERVED]
            after = [states[i] for i in range(position + 1, len(states))
                     if states[i].state == STATE_OBSERVED]
            if not before or not after:
                continue
            left, right = before[-1], after[0]
            if left.x is None or right.x is None:
                continue
            span = right.frame_index - left.frame_index
            weight = (state.frame_index - left.frame_index) / max(1, span)
            state.x = left.x + (right.x - left.x) * weight
            state.y = (left.y or 0.0) + ((right.y or left.y or 0.0) - (left.y or 0.0)) * weight
            state.radius = left.radius
            state.interpolated = True
            state.reasons.append("position interpolated across a bracketed gap; this "
                                 "is not an observation")

    # --- static-scene lock, track wide ---------------------------------------
    static_report = bcon.static_scene_lock(
        [bcon.Cluster(state.frame_index, state.x or 0.0, state.y or 0.0,
                      state.radius or 3.0, []) if state.state == STATE_OBSERVED
         else None for state in states],
        bcon.ConsensusParams(static_lock_frames=max(6, params.max_gap + 4)),
        motion_px=motion_px)
    if static_report.get("locked"):
        for state in states:
            if state.state == STATE_OBSERVED:
                state.diagnostics = sorted(set(state.diagnostics + ["STATIC_SCENE_LOCK"]))
                state.reasons.append("the whole track holds position while the camera "
                                     "moves; that is scenery, not flight")

    # --- track-level confidence and acceptance -------------------------------
    def state_at(position: int) -> Optional[FrameState]:
        return states[position] if 0 <= position < len(states) else None

    observed = [s for s in states if s.state == STATE_OBSERVED]
    span = (states[-1].frame_index - states[0].frame_index + 1) if states else 0
    coverage = len(observed) / max(1, span)
    probabilities = [s.detector_probability for s in observed
                     if s.detector_probability is not None]
    mean_score = float(np.mean([s.score for s in observed])) if observed else 0.0
    cycles = [s for s in observed if s.cycle.get("supported")]
    cycle_rate = len(cycles) / max(1, len(observed))
    body_locked = sum(1 for s in observed if "BODY_LOCK" in s.diagnostics
                      or "CLOTHING_LOCK" in s.diagnostics)
    static_locked = sum(1 for s in observed if "STATIC_SCENE_LOCK" in s.diagnostics)
    abrupt_jumps = 0
    for left, right in zip(observed, observed[1:]):
        if right.frame_index - left.frame_index > 1 or left.x is None or right.x is None:
            continue
        step = math.hypot(right.x - left.x, right.y - left.y)
        if step > params.max_step_radii * max(left.radius or 3.0, 1.0):
            abrupt_jumps += 1

    confidence = {
        "competitor": competitor_report,
        "observed_frames": len(observed),
        "span_frames": span,
        "coverage": round(coverage, 4),
        "mean_detector_probability": (round(float(np.mean(probabilities)), 4)
                                      if probabilities else None),
        "mean_score": round(mean_score, 4),
        "cycle_consistency_rate": round(cycle_rate, 4),
        "body_lock_rate": round(body_locked / max(1, len(observed)), 4),
        "static_lock_rate": round(static_locked / max(1, len(observed)), 4),
        "abrupt_jumps": abrupt_jumps,
        "ambiguous_frames": sum(1 for s in states if s.state == STATE_AMBIGUOUS),
        "occluded_frames": sum(1 for s in states if s.state == STATE_OCCLUDED),
        "lost_frames": sum(1 for s in states if s.state == STATE_LOST),
        "unavailable_frames": sum(1 for s in states if s.state == STATE_UNAVAILABLE),
        "state_share": {state: round(sum(1 for s in states if s.state == state)
                                     / max(1, len(states)), 4)
                        for state in TRACK_STATES},
    }

    reasons: list[str] = []
    if len(observed) < params.min_observed_frames:
        reasons.append(f"only {len(observed)} observed frame(s); "
                       f"{params.min_observed_frames} required to accept a track")
    if coverage < params.min_coverage:
        reasons.append(f"coverage {coverage:.2f} below {params.min_coverage:.2f}")
    if mean_score < params.min_mean_score:
        reasons.append(f"mean evidence {mean_score:.2f} below {params.min_mean_score:.2f}")
    if cycle_rate < params.min_cycle_rate:
        reasons.append(f"cycle consistency {cycle_rate:.2f} below "
                       f"{params.min_cycle_rate:.2f}")
    if confidence["body_lock_rate"] > params.max_body_lock_rate:
        reasons.append(f"body/clothing lock rate {confidence['body_lock_rate']:.2f} above "
                       f"{params.max_body_lock_rate:.2f}")
    if confidence["static_lock_rate"] > params.max_static_lock_rate:
        reasons.append(f"static-scene lock rate {confidence['static_lock_rate']:.2f} "
                       f"above {params.max_static_lock_rate:.2f}")
    if abrupt_jumps > params.max_abrupt_jumps:
        reasons.append(f"{abrupt_jumps} abrupt jumps above {params.max_abrupt_jumps}")
    accepted = not reasons
    if not accepted:
        for state in states:
            if state.state == STATE_OBSERVED:
                state.state = STATE_AMBIGUOUS
                state.reasons.append("track not accepted: " + "; ".join(reasons))

    provenance.update({"ensemble_work_scale": context_info["work_scale"],
                       "median_camera_motion_fraction":
                           context_info["median_camera_motion_fraction"],
                       "methods": list(bc.METHOD_NAMES)})
    return BallTrackResult(states=states, accepted=accepted, confidence=confidence,
                           reasons=reasons, provenance=provenance)
