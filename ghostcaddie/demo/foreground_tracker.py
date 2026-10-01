"""Isolated experimental masked-foreground appearance tracker (clubhead research).

WHY THIS EXISTS
---------------
The classic demo tracker follows a whole ``template_size`` (31 px) patch around the
previous point. Around a clubhead most of that patch is not clubhead: trousers,
turf and shaft share the window, and the match locks onto whichever 31 px square
fits best. The failure the primary reported is a loss of clubhead *identity*, not
of local appearance.

This module is the candidate: it matches only the target-centred FOREGROUND
evidence the seed patch actually carries, and it may re-acquire a genuinely lost
target inside a bounded window instead of staying lost forever.

WHAT IT IS
----------
* one forward leg from the seed, no operator anchors, no back-fill;
* the ONLY reference input is the seed's native frame and native coordinate: the
  patch and the mask are built from that frame alone. No label, no manual path and
  no later frame is read anywhere in this module;
* masked normalised cross-correlation over a bounded local search around the
  motion-extrapolated position; the search window is CLAMPED at a frame edge
  instead of rejecting the whole window;
* a frozen seed patch + mask is the identity gate: a candidate that matches the
  drifting appearance model but not the seed is refused;
* ambiguity is detected (a second competing peak inside the same margin) and the
  frame is then reported with NO point -- never a stale marker at the last
  position;
* after ``LOSS_FRAMES`` consecutive refusals the target is declared lost FROM THAT
  FRAME (the earlier short gaps of the run stay ``missing``: they are never rewritten
  retroactively), and a
  BOUNDED re-acquisition window (``REACQ_WINDOW_FRAMES`` ELAPSED frames within
  ``REACQ_RADIUS_PX`` of the last physical position, at the higher
  ``REACQ_MIN_SCORE``) may re-acquire it only after ``REACQ_CONFIRM_FRAMES``
  consecutive measured confirmations. The search stays centred on that last OBSERVED
  physical position and is never re-centred on an unconfirmed candidate, and every
  candidate must satisfy BOTH its consecutive compatibility with the previous
  confirming candidate AND the bounded distance from that physical anchor
  (``global_step``), so a chain of individually-modest confirmations cannot walk the
  window away from the target. CAUSALLY: only the CURRENT frame may be reported
  as re-acquired -- an earlier candidate frame stays lost with no coordinates and is
  never back-filled by a later frame that confirms. The reported latency therefore
  includes the confirmation delay, the new segment begins at the confirmed frame, and
  the run is prefix-invariant: appending frames can never rewrite what an earlier
  frame already reported. Past the window nothing is reported at all;
* a structural cut resets the tracker, and no re-acquisition ever crosses a cut.

WHAT IT IS NOT
--------------
* not a detector and not a general tracker: it follows what the seed showed;
* not driven by the reference's per-clip options. Every constant is fixed at module
  level (see ``PROFILE``) and is the same for every clip, hole and frame. The
  ``options`` a prediction declares are the classic tracker's options (what the ball
  leg really runs with), not this profile;
* not a change to demo defaults: ``ghostcaddie.demo.worker``, ``tracking`` and
  ``config`` are untouched, the classic loss rule is NOT reused here, and this
  module owns its own explicit point policy;
* not evidence of accuracy: the unit tests are synthetic behaviour tests, not
  performance claims.

STATUS AND METHOD VOCABULARY
----------------------------
``manual_anchor`` (the seed only), ``observed_template`` with method
``foreground_template`` or ``foreground_template_reacquired``, ``missing`` and
``lost``. These are the benchmark's own statuses plus the two method names the
predictions contract accepts; nothing here invents a status, and a predicted
position is never reported as an observation.

cv2 and numpy are imported INSIDE the measurement functions, so importing this
module -- and unit-testing its policy layer -- needs neither.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .tracking import (ORIGIN_AUTOMATIC, ORIGIN_DERIVED, ORIGIN_OPERATOR,
                       STATUS_LOST, STATUS_MANUAL, STATUS_MISSING, STATUS_OBSERVED)

# --------------------------------------------------------------------------
# the frozen profile: every constant, fixed globally, never per clip/hole/frame

PROFILE_NAME = "clubhead-foreground-v1"

#: Seed patch size (native pixels). The same number as the classic tracker's default
#: ``template_size``, but read from here, never from the reference's options.
TEMPLATE_SIZE = 31
#: Local search radius around the motion-extrapolated position, in native pixels.
SEARCH_RADIUS_PX = 24
#: Bounded re-acquisition radius around the last PHYSICAL position.
REACQ_RADIUS_PX = 48
#: Bounded per-frame step from the last physical position while tracking.
MAX_STEP_PX = 30.0
#: Bounded step while re-acquiring (the re-acquisition radius, plus slack).
REACQ_MAX_STEP_PX = 52.0
#: Consecutive refusals before the target is declared lost.
LOSS_FRAMES = 3
#: Masked-correlation floor for a normal observation.
MIN_SCORE = 0.55
#: The frozen seed patch must still correlate at least this well (identity gate).
SEED_IDENTITY_MIN_SCORE = 0.45
#: Higher floor inside the re-acquisition window.
REACQ_MIN_SCORE = 0.70
#: Consecutive confirming frames required before a re-acquired run is reported.
REACQ_CONFIRM_FRAMES = 2
#: How many frames the bounded re-acquisition window lasts.
REACQ_WINDOW_FRAMES = 10
#: A second competing peak this close to the best refuses the frame as ambiguous.
AMBIGUITY_MARGIN = 0.06
#: Two peaks closer than this are the same match, not a competitor.
PEAK_SEPARATION_PX = 8.0
#: How much of the last step extrapolates the search centre (a search CENTRE only;
#: the extrapolation is never reported as an observation).
VELOCITY_WEIGHT = 1.0
#: Modest fixed rotation hypotheses, in degrees.
ROTATIONS_DEG = (-6.0, 0.0, 6.0)
#: Modest fixed scale hypotheses.
SCALES = (0.94, 1.0, 1.06)
#: Appearance update is guarded: only a well-scoring, identity-confirmed match may
#: refresh the model, and the frozen seed patch stays the gate afterwards.
APPEARANCE_UPDATE_MIN_SCORE = 0.80
#: Seed-patch grey-level standard deviation floor (below this there is nothing to
#: match on and the target is refused outright).
MASK_MIN_CONTRAST = 18.0
#: Gradient magnitude above which a seed pixel counts as foreground evidence.
MASK_GRADIENT_MIN = 12.0
#: Grey difference from the estimated background above which a seed pixel counts.
MASK_BACKGROUND_DELTA = 12.0
#: ...and how many robust background sigmas above that estimate a pixel must sit
#: before it counts either. The background estimate comes from the patch's own
#: border ring (median and MAD-scaled sigma), so a busy surround raises the bar
#: instead of flooding the mask with background.
MASK_BACKGROUND_SIGMA_FACTOR = 2.5
#: The sigma term is capped so a wildly textured surround cannot make the mask
#: unreachable for a genuinely contrasting head.
MASK_BACKGROUND_MAX_SIGMA = 24.0
#: Width of the background-estimating border ring of the seed patch, in pixels.
MASK_BORDER_RING_PX = 2
#: The foreground mask must cover a plausible part of the patch: neither a few
#: speckles nor nearly the whole 31 px context.
MASK_FOREGROUND_MIN_FRACTION = 0.05
MASK_FOREGROUND_MAX_FRACTION = 0.80
#: Absolute floor on the number of masked pixels.
MASK_MIN_PIXELS = 24
#: Central prior: weight >= this keeps the mask centred on the target.
MASK_CENTER_PRIOR_MIN = 0.25
MASK_CENTER_PRIOR_FALLOFF = 1.5
#: A search window this flat carries no appearance information.
MIN_WINDOW_STD = 4.0
#: Numerical guard for the masked normalisation (grey-squared units).
MIN_MASK_VARIANCE = 1.0
EPSILON = 1e-6

METHOD_FOREGROUND = "foreground_template"
METHOD_REACQUIRED = "foreground_template_reacquired"
METHOD_SEED = "operator_placement"

#: (rotation, scale) hypotheses, in a fixed order. The identity transform is exact.
TRANSFORMS: Tuple[Tuple[float, float], ...] = tuple(
    (angle, scale) for angle in ROTATIONS_DEG for scale in SCALES)


def transform_label(angle: float, scale: float) -> str:
    return f"rot{angle:+.1f}/scale{scale:.2f}"


#: Every constant above, in one place, so the runtime record can hash what the run
#: actually used instead of restating it. A unit test fails if a module constant is
#: added without being listed here.
PROFILE: Dict[str, Any] = {
    "name": PROFILE_NAME,
    "method": METHOD_FOREGROUND,
    "method_reacquired": METHOD_REACQUIRED,
    "method_seed": METHOD_SEED,
    "template_size": TEMPLATE_SIZE,
    "search_radius_px": SEARCH_RADIUS_PX,
    "reacq_radius_px": REACQ_RADIUS_PX,
    "max_step_px": MAX_STEP_PX,
    "reacq_max_step_px": REACQ_MAX_STEP_PX,
    "loss_frames": LOSS_FRAMES,
    "min_score": MIN_SCORE,
    "seed_identity_min_score": SEED_IDENTITY_MIN_SCORE,
    "reacq_min_score": REACQ_MIN_SCORE,
    "reacq_confirm_frames": REACQ_CONFIRM_FRAMES,
    "reacq_window_frames": REACQ_WINDOW_FRAMES,
    "ambiguity_margin": AMBIGUITY_MARGIN,
    "peak_separation_px": PEAK_SEPARATION_PX,
    "velocity_weight": VELOCITY_WEIGHT,
    "rotations_deg": list(ROTATIONS_DEG),
    "scales": list(SCALES),
    "transforms": [transform_label(angle, scale) for angle, scale in TRANSFORMS],
    "appearance_update_min_score": APPEARANCE_UPDATE_MIN_SCORE,
    "mask_min_contrast": MASK_MIN_CONTRAST,
    "mask_gradient_min": MASK_GRADIENT_MIN,
    "mask_background_delta": MASK_BACKGROUND_DELTA,
    "mask_background_sigma_factor": MASK_BACKGROUND_SIGMA_FACTOR,
    "mask_background_max_sigma": MASK_BACKGROUND_MAX_SIGMA,
    "mask_border_ring_px": MASK_BORDER_RING_PX,
    "mask_foreground_min_fraction": MASK_FOREGROUND_MIN_FRACTION,
    "mask_foreground_max_fraction": MASK_FOREGROUND_MAX_FRACTION,
    "mask_min_pixels": MASK_MIN_PIXELS,
    "mask_center_prior_min": MASK_CENTER_PRIOR_MIN,
    "mask_center_prior_falloff": MASK_CENTER_PRIOR_FALLOFF,
    "min_window_std": MIN_WINDOW_STD,
    "min_mask_variance": MIN_MASK_VARIANCE,
    "epsilon": EPSILON,
    "reacq_search_anchor": ("the last OBSERVED physical position before the loss; the "
                            "search is never re-centred on an unconfirmed candidate"),
    "gates": ["score", "seed_identity", "unambiguous", "step", "global_step"],
    "step_gate_meaning": ("'step' is consecutive compatibility with the previous "
                          "confirming candidate; 'global_step' is the bounded distance "
                          "from the last observed physical anchor. Both must hold."),
    "distance_metric": "grey masked zero-mean normalised cross-correlation",
    "driven_by_reference_options": False,
    "one_forward_leg_only": True,
    "statuses": [STATUS_MANUAL, STATUS_OBSERVED, STATUS_MISSING, STATUS_LOST],
}


# --------------------------------------------------------------------------
# pure policy layer: what a measurement means (no cv2, no numpy)

@dataclass(frozen=True)
class SearchRequest:
    """Where the measurement layer is asked to look, and how strictly."""
    frame: int
    anchor_x: float
    anchor_y: float
    radius_px: float
    reacquiring: bool
    because: str

    def to_dict(self) -> dict:
        return {"frame": int(self.frame),
                "anchor": [round(self.anchor_x, 2), round(self.anchor_y, 2)],
                "radius_px": self.radius_px, "reacquiring": bool(self.reacquiring),
                "because": self.because}


@dataclass(frozen=True)
class CandidateMatch:
    """One frame of masked measurement. Numbers only: the policy decides."""
    frame: int
    found: bool
    x: float = 0.0
    y: float = 0.0
    score: float = 0.0
    seed_score: float = 0.0
    second_score: float = 0.0
    transform: str = ""
    clipped: bool = False
    reason: str = ""

    @property
    def margin(self) -> float:
        return float(self.score) - float(self.second_score)

    def to_dict(self) -> dict:
        return {"frame": int(self.frame), "found": bool(self.found),
                "x": round(float(self.x), 2), "y": round(float(self.y), 2),
                "score": round(float(self.score), 4),
                "seed_identity_score": round(float(self.seed_score), 4),
                "second_best_score": round(float(self.second_score), 4),
                "margin": round(self.margin, 4),
                "transform": self.transform,
                "search_window_clipped": bool(self.clipped),
                "reason": self.reason}


@dataclass
class FrameDecision:
    """One frame of the candidate's own point policy."""
    source_frame: int
    status: str
    origin: str
    method: str = ""
    x: Optional[float] = None
    y: Optional[float] = None
    score: Optional[float] = None
    segment: Optional[int] = None
    note: str = ""
    reacquired: bool = False
    gates: Dict[str, bool] = field(default_factory=dict)
    measurement: Dict[str, Any] = field(default_factory=dict)

    @property
    def observed(self) -> bool:
        return self.status in (STATUS_OBSERVED, STATUS_MANUAL)

    def to_point(self) -> dict:
        """Exactly the prediction contract's point fields, and nothing else."""
        return {"source_frame": int(self.source_frame), "status": self.status,
                "origin": self.origin, "method": self.method,
                "x": None if self.x is None else round(float(self.x), 2),
                "y": None if self.y is None else round(float(self.y), 2),
                "score": None if self.score is None else round(float(self.score), 4)}

    def to_raw(self) -> dict:
        return {"frame": int(self.source_frame), "status": self.status,
                "origin": self.origin, "method": self.method,
                "x": self.x, "y": self.y, "score": self.score,
                "segment": self.segment, "reacquired": bool(self.reacquired),
                "gates": dict(self.gates), "measurement": dict(self.measurement),
                "note": self.note}


class ForegroundPolicy:
    """Accept / abstain / lose / re-acquire, frame by frame.

    Pure: it never touches an image. The measurement layer asks it where to look
    (:meth:`request_for`) and hands back the numbers it found (:meth:`step`), so the
    decision rules -- the identity gate, the ambiguity gate, the bounded step, the
    loss, the bounded re-acquisition window and its confirmation -- are testable on
    synthetic numbers alone. That separation is the point: a synthetic test proves
    the *policy*, never a tracker's accuracy.
    """

    def __init__(self, *, seed_frame: int, seed_x: float, seed_y: float,
                 target: str = "clubhead", loss_frames: int = LOSS_FRAMES,
                 search_radius_px: float = SEARCH_RADIUS_PX,
                 reacq_radius_px: float = REACQ_RADIUS_PX,
                 max_step_px: float = MAX_STEP_PX,
                 reacq_max_step_px: float = REACQ_MAX_STEP_PX,
                 min_score: float = MIN_SCORE,
                 seed_identity_min_score: float = SEED_IDENTITY_MIN_SCORE,
                 reacq_min_score: float = REACQ_MIN_SCORE,
                 reacq_confirm_frames: int = REACQ_CONFIRM_FRAMES,
                 reacq_window_frames: int = REACQ_WINDOW_FRAMES,
                 ambiguity_margin: float = AMBIGUITY_MARGIN) -> None:
        self.target = target
        self.seed_frame = int(seed_frame)
        self.seed_x = float(seed_x)
        self.seed_y = float(seed_y)
        self.loss_frames = int(loss_frames)
        self.search_radius_px = float(search_radius_px)
        self.reacq_radius_px = float(reacq_radius_px)
        self.max_step_px = float(max_step_px)
        self.reacq_max_step_px = float(reacq_max_step_px)
        self.min_score = float(min_score)
        self.seed_identity_min_score = float(seed_identity_min_score)
        self.reacq_min_score = float(reacq_min_score)
        self.reacq_confirm_frames = int(reacq_confirm_frames)
        self.reacq_window_frames = int(reacq_window_frames)
        self.ambiguity_margin = float(ambiguity_margin)

        self.decisions: List[FrameDecision] = []
        #: False when the seed patch/mask was refused: nothing is measured at all.
        self.measure = True
        self.refusal_reason = ""
        self.reset = False
        self.lost = False
        self.misses = 0
        #: ELAPSED frames left in the bounded re-acquisition window. Every frame spent
        #: lost consumes one, whether it carried a candidate or not: the window is a
        #: finite number of elapsed source frames, never a hold on a lucky candidate.
        self.reacq_frames_left = 0
        self.loss_frame: Optional[int] = None
        self.last: Optional[Tuple[float, float]] = None
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.reacq_anchor: Optional[Tuple[float, float]] = None
        #: The last measured confirmation position inside the window (the search anchor
        #: for the next confirming frame). Reset whenever a confirmation run breaks.
        self.reacq_last: Optional[Tuple[float, float]] = None
        #: Consecutive measured confirmations seen so far in the current attempt, and
        #: the frames they were measured at (frame numbers only: the earlier ones stay
        #: reported as lost and are never rewritten by a later frame).
        self.reacq_confirmations = 0
        self.confirm_frames: List[int] = []
        self.segment = 0
        self.last_drawn_frame: Optional[int] = None
        self.seed_recorded = False
        self.observations = 0
        self.reacquired_runs = 0

    # -- seed -------------------------------------------------------------

    def seed_decision(self) -> FrameDecision:
        """The seed's own point: an operator placement, one per track."""
        self.seed_recorded = True
        self.last = (self.seed_x, self.seed_y)
        self.last_drawn_frame = self.seed_frame
        return self._record(FrameDecision(
            self.seed_frame, STATUS_MANUAL, ORIGIN_OPERATOR, method=METHOD_SEED,
            x=self.seed_x, y=self.seed_y, score=None, segment=self.segment,
            note=("seed (operator placement): the one operator anchor of this track; "
                  "never scored as an automatic observation")))

    def refuse_seed(self, reason: str) -> None:
        """The seed carried no usable evidence: measure nothing, ever."""
        self.measure = False
        self.refusal_reason = reason

    # -- requests ---------------------------------------------------------

    def _search_origin(self) -> Optional[Tuple[float, float]]:
        """Where the SEARCH is centred.

        While lost this is ALWAYS the last observed physical position before the loss.
        An unconfirmed candidate is never allowed to become the next search centre:
        chaining those would let the window walk away from the target one confirmation
        at a time, which is exactly the unbounded behaviour this candidate refuses.
        """
        if self.lost:
            return self.reacq_anchor
        return self.last

    def _pending_origin(self) -> Optional[Tuple[float, float]]:
        """The anchor the NEXT confirmation must be compatible with.

        Inside a confirmation run that is the previous confirming candidate's position
        (consecutive compatibility); with no run open it is the same physical anchor the
        search uses.
        """
        if self.lost and self.reacq_confirmations:
            return self.reacq_last
        return self._search_origin()

    def request_for(self, frame: int) -> Optional[SearchRequest]:
        """Where to look this frame, or None when nothing may be looked for."""
        frame = int(frame)
        if not self.measure or self.reset or frame <= self.seed_frame:
            return None
        if self.lost:
            if self.reacq_frames_left <= 0:
                return None
            anchor = self.reacq_anchor
            if anchor is None:
                return None
            return SearchRequest(frame, anchor[0], anchor[1], self.reacq_radius_px, True,
                                 "bounded re-acquisition around the last OBSERVED "
                                 "physical position (no velocity extrapolation, and "
                                 "never re-centred on an unconfirmed candidate)")
        if self.last is None:
            return None
        step = math.hypot(self.velocity[0], self.velocity[1])
        factor = (self.max_step_px / step) if step > self.max_step_px > 0 else 1.0
        predicted = (self.last[0] + VELOCITY_WEIGHT * factor * self.velocity[0],
                     self.last[1] + VELOCITY_WEIGHT * factor * self.velocity[1])
        return SearchRequest(frame, predicted[0], predicted[1], self.search_radius_px,
                             False,
                             "local search around the motion-extrapolated position "
                             "(the extrapolation is a search centre only)")

    # -- decisions --------------------------------------------------------

    def step(self, frame: int, observation: Optional[CandidateMatch] = None,
             *, cut: bool = False) -> FrameDecision:
        frame = int(frame)
        if cut:
            return self._structural_cut(frame)
        if frame < self.seed_frame:
            return self._record(FrameDecision(
                frame, STATUS_MISSING, ORIGIN_DERIVED,
                note=("before the seed: nothing is tracked ahead of the operator's own "
                      "seed, and nothing is inferred")))
        if frame == self.seed_frame:
            return self.seed_decision() if not self.seed_recorded else self.decisions[0]
        if not self.measure:
            return self._record(FrameDecision(
                frame, STATUS_MISSING, ORIGIN_DERIVED,
                note=(f"no measurement: the seed was refused ({self.refusal_reason}), so "
                      f"this target is never tracked and no point is invented")))
        if self.reset:
            return self._record(FrameDecision(
                frame, STATUS_MISSING, ORIGIN_DERIVED,
                note=("the source was structurally cut: the tracker was reset at the cut "
                      "and nothing is measured or re-acquired across it, so this frame "
                      "carries no point")))
        if self.lost and self.reacq_frames_left <= 0:
            return self._record(FrameDecision(
                frame, STATUS_LOST, ORIGIN_DERIVED,
                note=("lost, and the bounded re-acquisition window is over: nothing is "
                      "drawn, no coordinate is predicted and no stale marker is kept")))

        required = self.reacq_min_score if self.lost else self.min_score
        bound = self.reacq_max_step_px if self.lost else self.max_step_px
        measurement: Dict[str, Any] = {}
        gates: Dict[str, bool] = {}
        accepted = False
        if observation is not None:
            measurement = observation.to_dict()
            if observation.found:
                search_origin = self._search_origin()
                pending_origin = self._pending_origin()
                global_step = (math.hypot(observation.x - search_origin[0],
                                          observation.y - search_origin[1])
                               if search_origin is not None else 0.0)
                pending_step = (math.hypot(observation.x - pending_origin[0],
                                           observation.y - pending_origin[1])
                                if pending_origin is not None else 0.0)
                measurement["step_from_last_physical_px"] = round(global_step, 2)
                measurement["step_from_previous_confirmation_px"] = round(pending_step,
                                                                         2)
                gates = {"score": observation.score >= required,
                         "seed_identity":
                             observation.seed_score >= self.seed_identity_min_score,
                         "unambiguous": observation.margin >= self.ambiguity_margin,
                         #: consecutive compatibility with the previous confirming
                         #: candidate (the pending step)...
                         "step": pending_step <= bound,
                         #: ...AND the bounded distance from the last OBSERVED physical
                         #: anchor, so a chain of individually-modest confirmations can
                         #: never walk the re-acquisition away from the target.
                         "global_step": global_step <= bound}
                accepted = all(gates.values())
        if not accepted:
            if observation is not None and not observation.found:
                reason = observation.reason or "no_masked_candidate"
            else:
                failed = [name for name, ok in gates.items() if not ok]
                reason = ("refused: " + ", ".join(failed)) if failed else "no_candidate"
            return self._miss(frame, reason, gates, measurement)
        if self.lost:
            return self._confirming(frame, observation, gates, measurement)
        return self._tracking(frame, observation, gates, measurement)

    def wants_appearance_update(self, observation: CandidateMatch) -> bool:
        """A guarded appearance refresh: never while re-acquiring, never on a weak
        match, and never without the seed identity still holding."""
        return ((not self.lost)
                and observation.score >= APPEARANCE_UPDATE_MIN_SCORE
                and observation.seed_score >= self.seed_identity_min_score)

    # -- internals --------------------------------------------------------

    def _record(self, decision: FrameDecision) -> FrameDecision:
        self.decisions.append(decision)
        return decision

    def _structural_cut(self, frame: int) -> FrameDecision:
        """A cut resets the source: state is dropped and nothing is re-acquired."""
        self.reset = True
        self.lost = False
        self.misses = 0
        self.reacq_frames_left = 0
        self._break_confirmation()
        return self._record(FrameDecision(
            frame, STATUS_MISSING, ORIGIN_DERIVED,
            note=("structural cut at this frame: the tracker is reset, no state crosses "
                  "the cut and no re-acquisition is attempted on the far side of it")))

    def _break_confirmation(self) -> int:
        """Abandon a confirmation run. Returns how many confirmations it held.

        Nothing already reported is rewritten: the frames of a broken run stay exactly
        as they were reported (as loss/null), which is what keeps the run
        prefix-invariant.
        """
        held = self.reacq_confirmations
        self.reacq_confirmations = 0
        self.confirm_frames = []
        self.reacq_last = None
        return held

    def _miss(self, frame: int, reason: str, gates: Dict[str, bool],
              measurement: Dict[str, Any]) -> FrameDecision:
        if self.lost:
            broken = self._break_confirmation()
            self.reacq_frames_left = max(0, self.reacq_frames_left - 1)
            note = (f"lost; re-acquisition refused this frame ({reason}); "
                    f"{self.reacq_frames_left} frame(s) of the bounded re-acquisition "
                    f"window remain")
            if broken:
                note += (f"; the confirmation run of {broken} frame(s) was broken by "
                         f"this refusal and those frames stay lost, with no "
                         f"coordinates and no rewrite")
            if self.reacq_frames_left == 0:
                note += ("; the window is now over, so this target stays lost and "
                         "nothing further is reported for it")
            return self._record(FrameDecision(frame, STATUS_LOST, ORIGIN_DERIVED,
                                              note=note, gates=gates,
                                              measurement=measurement))
        self.misses += 1
        decision = self._record(FrameDecision(
            frame, STATUS_MISSING, ORIGIN_DERIVED,
            note=(f"short gap: {self.misses} consecutive frame(s) with no accepted "
                  f"candidate (loss_frames={self.loss_frames}); reason: {reason}"),
            gates=gates, measurement=measurement))
        if self.misses >= self.loss_frames:
            # CAUSAL DECLARATION. The loss starts at THIS frame; the earlier short gaps
            # of the run stay reported exactly as they were reported (missing, no
            # coordinates) and are not rewritten. Rewriting them would make the report
            # depend on frames that arrive later, which is what prefix invariance
            # forbids.
            self.lost = True
            self.loss_frame = frame
            self.reacq_frames_left = self.reacq_window_frames
            self.reacq_anchor = self.last
            self.reacq_last = None
            self.reacq_confirmations = 0
            self.confirm_frames = []
            self.velocity = (0.0, 0.0)
            decision.status = STATUS_LOST
            decision.note = (f"tracker refused {self.misses} consecutive frames "
                             f"(>= loss_frames={self.loss_frames}): the target is "
                             f"declared lost from THIS frame, and a bounded "
                             f"re-acquisition window of {self.reacq_window_frames} "
                             f"elapsed frame(s) opens around the last physical "
                             f"position. The earlier {self.misses - 1} short gap "
                             f"frame(s) of this run stay reported as they were (missing, "
                             f"no coordinates) and are not rewritten by this decision")
        return decision

    def _open_segment(self, frame: int, *, fresh: bool) -> int:
        if fresh or self.last_drawn_frame is None or frame != self.last_drawn_frame + 1:
            self.segment += 1
        self.last_drawn_frame = frame
        return self.segment

    def _tracking(self, frame: int, observation: CandidateMatch,
                  gates: Dict[str, bool], measurement: Dict[str, Any]) -> FrameDecision:
        self.misses = 0
        x, y = float(observation.x), float(observation.y)
        segment = self._open_segment(frame, fresh=False)
        if self.last is not None:
            self.velocity = (x - self.last[0], y - self.last[1])
        decision = FrameDecision(
            frame, STATUS_OBSERVED, ORIGIN_AUTOMATIC, method=METHOD_FOREGROUND,
            x=x, y=y, score=observation.score, segment=segment,
            note=(f"masked foreground match (transform {observation.transform}"
                  + (", search window clamped at a frame edge"
                     if observation.clipped else "")
                  + f"): masked score {observation.score:.3f}, seed identity "
                    f"{observation.seed_score:.3f}, second peak "
                    f"{observation.second_score:.3f}"),
            gates=gates, measurement=measurement)
        self.last = (x, y)
        self.observations += 1
        return self._record(decision)

    def _confirming(self, frame: int, observation: CandidateMatch,
                    gates: Dict[str, bool], measurement: Dict[str, Any]) -> FrameDecision:
        """One measured candidate inside the re-acquisition window.

        CAUSAL RULE. A frame is reported as re-acquired only when the CURRENT frame is
        the one that completes ``reacq_confirm_frames`` consecutive measured
        confirmations. An earlier candidate frame is NEVER backfilled: it was already
        reported as lost with no coordinates, and a later frame that confirms does not
        rewrite it. That is what keeps the run prefix-invariant (appending frames can
        never change what an earlier frame reported) and keeps the reported latency
        honest, because the confirmation delay is counted rather than hidden.

        The window is a finite number of ELAPSED source frames: every frame spent lost
        consumes one, so a partial confirmation run cannot hold the window open.
        """
        self.reacq_confirmations += 1
        self.reacq_last = (float(observation.x), float(observation.y))
        self.confirm_frames.append(frame)
        if self.reacq_confirmations < self.reacq_confirm_frames:
            self.reacq_frames_left = max(0, self.reacq_frames_left - 1)
            return self._record(FrameDecision(
                frame, STATUS_LOST, ORIGIN_DERIVED,
                note=(f"re-acquisition candidate accepted by every gate at "
                      f"({observation.x:.0f}, {observation.y:.0f}) but this is "
                      f"confirmation {self.reacq_confirmations}/"
                      f"{self.reacq_confirm_frames} only: this frame stays LOST and "
                      f"carries no point, and it will NOT be backfilled if a later "
                      f"frame completes the run"),
                gates=gates, measurement=measurement))
        self.reacquired_runs += 1
        segment = self._open_segment(frame, fresh=True)
        decision = FrameDecision(
            frame, STATUS_OBSERVED, ORIGIN_AUTOMATIC, method=METHOD_REACQUIRED,
            x=float(observation.x), y=float(observation.y),
            score=float(observation.score), segment=segment, reacquired=True,
            gates=gates, measurement=measurement,
            note=(f"re-acquired AT THIS FRAME after {self.reacq_confirmations} "
                  f"consecutive measured confirmations (source frames "
                  f"{self.confirm_frames}); declared loss at source frame "
                  f"{self.loss_frame}; masked score {observation.score:.3f}, seed "
                  f"identity {observation.seed_score:.3f}, second peak "
                  f"{observation.second_score:.3f}. The earlier candidate frame(s) of "
                  f"this run stay lost with no coordinates and are not backfilled; "
                  f"segment {segment} begins at this confirmed frame, so no trail "
                  f"bridges the loss"))
        self.last = (float(observation.x), float(observation.y))
        #: No velocity is carried across a re-acquisition: the next frame searches
        #: around the confirmed position itself, not around a jump that was never
        #: observed frame to frame.
        self.velocity = (0.0, 0.0)
        self.observations += 1
        self.lost = False
        self.reacq_frames_left = 0
        self.misses = 0
        self._break_confirmation()
        self.last_drawn_frame = frame
        return self._record(decision)

    # -- reporting --------------------------------------------------------

    def summary(self) -> dict:
        counts: Dict[str, int] = {}
        for decision in self.decisions:
            counts[decision.status] = counts.get(decision.status, 0) + 1
        return {"target": self.target,
                "seed_frame": self.seed_frame,
                "measure": bool(self.measure),
                "refusal_reason": self.refusal_reason,
                "status_counts": dict(sorted(counts.items())),
                "observed_frames": sum(1 for d in self.decisions if d.observed),
                "automatic_observed_frames": sum(1 for d in self.decisions
                                                 if d.origin == ORIGIN_AUTOMATIC),
                "lost_frames": counts.get(STATUS_LOST, 0),
                "missing_frames": counts.get(STATUS_MISSING, 0),
                "segments": max((d.segment for d in self.decisions
                                 if d.segment is not None), default=-1) + 1,
                "reacquired_runs": self.reacquired_runs,
                "reacquired_frames": sum(1 for d in self.decisions if d.reacquired),
                "reacq_confirmations_required": self.reacq_confirm_frames,
                "structural_cut_reset": bool(self.reset),
                "terminal_lost": bool(self.lost and self.reacq_frames_left <= 0),
                "profile": PROFILE_NAME}


# --------------------------------------------------------------------------
# measurement layer: masked foreground matching (cv2/numpy imported lazily)

def _cv():
    import cv2  # noqa: PLC0415 - lazy so the policy layer stays stdlib-only
    return cv2


def _np():
    import numpy as np  # noqa: PLC0415
    return np


def _grey(image):
    cv2 = _cv()
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype("float32")
    return image.astype("float32")


def _patch(grey, x: float, y: float, size: int = TEMPLATE_SIZE):
    """A centred patch, or None when it would not be fully inside the frame."""
    np = _np()
    half = size // 2
    height, width = grey.shape[:2]
    x0, y0 = int(round(x)) - half, int(round(y)) - half
    if x0 < 0 or y0 < 0 or x0 + size > width or y0 + size > height:
        return None
    return np.ascontiguousarray(grey[y0:y0 + size, x0:x0 + size])


def seed_foreground_mask(patch, *, ring_px: int = MASK_BORDER_RING_PX):
    """Target-centred foreground evidence from a seed patch.

    Returns ``(mask_or_None, stats)``. The rules, all from the seed frame alone:

    * a central prior (falling off from the patch centre) keeps the mask on the
      target instead of the surrounding context;
    * a pixel counts as foreground evidence when its grey differs from the patch's
      own estimated background (the median of the border ring) or when it carries a
      gradient -- a bright head on dark turf qualifies by both routes;
    * the patch must carry contrast at all (``MASK_MIN_CONTRAST``), the mask must
      cover a plausible fraction of the patch (``MASK_FOREGROUND_MIN_FRACTION`` ..
      ``MASK_FOREGROUND_MAX_FRACTION``) and enough pixels (``MASK_MIN_PIXELS``). A
      flat or unstructured patch carries no identity to follow, so it is refused
      rather than allowed to drift.

    ``stats`` carries the numbers behind the decision, so a refusal can be shown
    instead of asserted.
    """
    cv2 = _cv()
    np = _np()
    size = int(patch.shape[0])
    stats: Dict[str, Any] = {"template_size": size, "reason": ""}
    contrast = float(patch.std())
    stats["patch_std"] = round(contrast, 3)
    if contrast < MASK_MIN_CONTRAST:
        stats["reason"] = "seed_patch_is_flat_low_texture"
        return None, stats
    ring = max(1, int(ring_px))
    if size < 3 * ring:
        stats["reason"] = "seed_patch_too_small_for_a_background_estimate"
        return None, stats
    border = np.concatenate([patch[:ring, :].ravel(), patch[-ring:, :].ravel(),
                             patch[:, :ring].ravel(), patch[:, -ring:].ravel()])
    background = float(np.median(border))
    # A robust spread of the SAME ring: a busy surround raises the foreground bar
    # instead of letting background texture into the mask.
    sigma = 1.4826 * float(np.median(np.abs(border - background)))
    threshold = max(MASK_BACKGROUND_DELTA,
                    MASK_BACKGROUND_SIGMA_FACTOR * min(sigma, MASK_BACKGROUND_MAX_SIGMA))
    stats["background_grey"] = round(background, 3)
    stats["background_sigma"] = round(sigma, 3)
    stats["foreground_threshold"] = round(threshold, 3)
    gradient_x = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    half = (size - 1) / 2.0
    ys, xs = np.mgrid[0:size, 0:size].astype("float32")
    dx = (xs - half) / max(half, 1.0)
    dy = (ys - half) / max(half, 1.0)
    prior = np.exp(-MASK_CENTER_PRIOR_FALLOFF * (dx * dx + dy * dy))
    foreground = (((np.abs(patch - background) >= threshold)
                   | (gradient >= MASK_GRADIENT_MIN))
                  & (prior >= MASK_CENTER_PRIOR_MIN))
    mask = foreground.astype("float32")
    pixels = int(mask.sum())
    fraction = pixels / float(size * size)
    stats["mask_pixels"] = pixels
    stats["mask_fraction"] = round(fraction, 4)
    stats["centre_prior_min"] = MASK_CENTER_PRIOR_MIN
    if pixels < MASK_MIN_PIXELS:
        stats["reason"] = "seed_mask_covers_too_few_pixels"
        return None, stats
    if fraction < MASK_FOREGROUND_MIN_FRACTION:
        stats["reason"] = "seed_mask_covers_too_little_of_the_patch"
        return None, stats
    if fraction > MASK_FOREGROUND_MAX_FRACTION:
        stats["reason"] = "seed_mask_covers_almost_the_whole_patch"
        return None, stats
    inside = patch[mask > 0]
    stats["masked_std"] = round(float(inside.std()), 3)
    if float(inside.std()) < MASK_MIN_CONTRAST / 3.0:
        stats["reason"] = "seed_mask_is_flat_within_the_foreground"
        return None, stats
    return mask, stats


def _masked_ncc_map(window, template, mask):
    """Masked zero-mean normalised cross-correlation, as a map over the window.

    ``score(u, v)`` compares the template (weighted by its own mask) with the window
    patch at ``(u, v)``. Implemented as three ``TM_CCORR`` passes over the mask,
    because OpenCV's masked ``matchTemplate`` does not support the zero-mean
    coefficient the identity gate needs.
    """
    cv2 = _cv()
    np = _np()
    count = float(mask.sum())
    if count < MASK_MIN_PIXELS:
        return None
    mean_template = float((template * mask).sum()) / count
    centred = (template - mean_template) * mask
    denominator_template = float(np.sqrt(float((centred * centred).sum())))
    if denominator_template <= EPSILON:
        return None
    window_float = window.astype("float32")
    sum_window = cv2.matchTemplate(window_float, mask, cv2.TM_CCORR)
    sum_window_squared = cv2.matchTemplate(window_float * window_float, mask,
                                           cv2.TM_CCORR)
    cross = cv2.matchTemplate(window_float, centred, cv2.TM_CCORR)
    variance = np.maximum(sum_window_squared - (sum_window * sum_window) / count, 0.0)
    denominator_window = np.sqrt(variance)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = cross / (denominator_window * denominator_template + EPSILON)
    score = np.where(denominator_window < math.sqrt(MIN_MASK_VARIANCE), 0.0, score)
    return np.clip(score, -1.0, 1.0).astype("float32")


def _masked_score(patch, template, mask) -> float:
    """The same metric at one position, computed directly (the map's reference)."""
    np = _np()
    count = float(mask.sum())
    if count < MASK_MIN_PIXELS:
        return 0.0
    mean_template = float((template * mask).sum()) / count
    mean_patch = float((patch * mask).sum()) / count
    centred_template = (template - mean_template) * mask
    centred_patch = (patch - mean_patch) * mask
    denominator = float(np.sqrt(float((centred_template * centred_template).sum())
                                * float((centred_patch * centred_patch).sum())))
    if denominator <= EPSILON:
        return 0.0
    return max(-1.0, min(1.0, float((centred_template * centred_patch).sum())
                         / denominator))


def _transform(patch, mask, angle: float, scale: float):
    """A modest fixed rotation/scale hypothesis. The identity transform is exact.

    The mask is re-binarised after warping. The masked-correlation maths counts masked
    pixels and their sums directly, so it is only correct while the mask is binary:
    a nearest-neighbour warp guarantees that, and this makes it a property of the code
    rather than an assumption about the interpolation.
    """
    cv2 = _cv()
    np = _np()
    if angle == 0.0 and scale == 1.0:
        return patch, mask
    size = int(patch.shape[0])
    centre = ((size - 1) / 2.0, (size - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, float(angle), float(scale))
    warped = cv2.warpAffine(patch, matrix, (size, size), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    warped_mask = cv2.warpAffine(mask, matrix, (size, size), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    warped_mask = (warped_mask > 0.5).astype("float32")
    return warped.astype("float32"), warped_mask


def _second_peak(score_map, location, *, separation: float = PEAK_SEPARATION_PX) -> float:
    """The best competing peak, ignoring everything within ``separation`` px."""
    np = _np()
    row, column = int(location[1]), int(location[0])
    radius = int(math.ceil(separation))
    suppressed = score_map.copy()
    y0, y1 = max(0, row - radius), min(suppressed.shape[0], row + radius + 1)
    x0, x1 = max(0, column - radius), min(suppressed.shape[1], column + radius + 1)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    inside = ((ys - row) ** 2 + (xs - column) ** 2) <= separation * separation
    block = suppressed[y0:y1, x0:x1]
    block[inside] = -1.0
    return float(suppressed.max()) if suppressed.size else 0.0


def search_candidate(image, request: SearchRequest, *, appearance, appearance_mask,
                     seed_patch, seed_mask) -> CandidateMatch:
    """Measure one frame inside the bounded window the policy asked for.

    The window is CLAMPED at the frame edge (never rejected), the match is masked to
    the seed's foreground evidence, and the frozen seed patch is scored at the same
    position so the policy can apply its identity gate. Nothing here decides
    acceptance: that is the policy's job.
    """
    cv2 = _cv()
    grey = _grey(image)
    size = int(appearance.shape[0])
    half = size // 2
    height, width = grey.shape[:2]
    radius = int(round(request.radius_px))
    centre_x, centre_y = int(round(request.anchor_x)), int(round(request.anchor_y))
    raw = (centre_x - half - radius, centre_y - half - radius,
           centre_x + half + radius + 1, centre_y + half + radius + 1)
    x0, y0 = max(0, raw[0]), max(0, raw[1])
    x1, y1 = min(width, raw[2]), min(height, raw[3])
    clipped = (x0, y0, x1, y1) != raw
    if x1 - x0 < size or y1 - y0 < size:
        return CandidateMatch(request.frame, False, clipped=clipped,
                              reason="frame_too_small_for_the_template")
    window = grey[y0:y1, x0:x1]
    if float(window.std()) < MIN_WINDOW_STD:
        return CandidateMatch(request.frame, False, clipped=clipped,
                              reason="search_window_is_flat_low_texture")
    best: Optional[Tuple[float, float, float, Tuple[float, float], Tuple[int, int],
                         Any]] = None
    peaks: List[Tuple[float, Tuple[float, float]]] = []
    for angle, scale in TRANSFORMS:
        template, template_mask = _transform(appearance, appearance_mask, angle, scale)
        if float(template_mask.sum()) < MASK_MIN_PIXELS:
            continue
        score_map = _masked_ncc_map(window, template, template_mask)
        if score_map is None or score_map.size == 0:
            continue
        _, max_value, _, max_location = cv2.minMaxLoc(score_map)
        point = (float(x0 + max_location[0] + half), float(y0 + max_location[1] + half))
        peaks.append((float(max_value), point))
        if best is None or float(max_value) > best[0]:
            best = (float(max_value), angle, scale, point,
                    (int(max_location[0]), int(max_location[1])), score_map)
    if best is None:
        return CandidateMatch(request.frame, False, clipped=clipped,
                              reason="no_masked_template_available")
    score, angle, scale, point, location, score_map = best
    seed_score = _masked_score(
        window[location[1]:location[1] + size, location[0]:location[0] + size],
        seed_patch, seed_mask)
    second = _second_peak(score_map, location)
    for value, other in peaks:
        if math.hypot(other[0] - point[0], other[1] - point[1]) > PEAK_SEPARATION_PX:
            second = max(second, value)
    return CandidateMatch(request.frame, True, x=point[0], y=point[1], score=score,
                          seed_score=seed_score, second_score=second,
                          transform=transform_label(angle, scale), clipped=clipped)


# --------------------------------------------------------------------------
# the run: one forward leg, decoded by the caller

LIMITATIONS = (
    "Experimental masked-foreground template method (`foreground_template`): the seed "
    "patch's central foreground evidence is matched by masked zero-mean normalised "
    "cross-correlation inside a bounded local search. It is still a TEMPLATE method -- "
    "it follows what the seed showed, it is not a detector, and whether it tracks the "
    "clubhead better is a measurement for the primary to make, not a claim made here.",
    "The seed mask is derived once, from the seed frame's native image and native "
    "coordinate alone. It is never re-derived from a later frame, and the "
    "rotation/scale hypotheses are small and fixed (see the profile), so a large or "
    "fast rotation/scale change degrades the identity gate: the frame is then refused "
    "rather than followed.",
    "Opt-in bounded automatic re-acquisition (`foreground_template_reacquired`): after "
    "a declared loss, only a bounded window of ELAPSED frames around the LAST PHYSICAL "
    "position may re-acquire, at a higher threshold, and the frame is reported only "
    "when IT is the one completing the required consecutive measured confirmations. "
    "Earlier candidate frames are never back-filled: they stay lost with no "
    "coordinates, which keeps the confirmation delay in the reported latency and keeps "
    "the run prefix-invariant. No global search is performed, no point past the window "
    "is ever reported, and no re-acquisition crosses a structural cut. Nothing bridges "
    "the loss and no stale marker is left behind.",
    "The motion-extrapolated search centre is a prediction used to bound the search "
    "window; it is never reported as a point. Only measured, gate-passing positions "
    "are reported, every re-acquired frame carries its own method name, and the run "
    "never places an operator anchor of its own.",
)


@dataclass
class ForegroundRun:
    target: str
    seed: dict
    interval_frames: List[int]
    available: bool = False
    reason: str = ""
    decisions: List[FrameDecision] = field(default_factory=list)
    raw_frames: List[dict] = field(default_factory=list)
    seed_stats: Dict[str, Any] = field(default_factory=dict)
    appearance_updates: int = 0
    policy_summary: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=lambda: list(LIMITATIONS))

    def points(self) -> List[dict]:
        return [decision.to_point() for decision in self.decisions]

    def summary(self) -> dict:
        counts: Dict[str, int] = {}
        for decision in self.decisions:
            counts[decision.status] = counts.get(decision.status, 0) + 1
        reasons: Dict[str, int] = {}
        for decision in self.decisions:
            if decision.observed:
                continue
            head = decision.note.split(";")[0][:70]
            if head:
                reasons[head] = reasons.get(head, 0) + 1
        reacquired = [decision.source_frame for decision in self.decisions
                      if decision.reacquired]
        return {"target": self.target,
                "available": bool(self.available),
                "reason": self.reason,
                "seed": dict(self.seed),
                "frames": len(self.decisions),
                "status_counts": dict(sorted(counts.items())),
                "observed_frames": sum(1 for d in self.decisions if d.observed),
                "automatic_observed_frames": sum(1 for d in self.decisions
                                                 if d.origin == ORIGIN_AUTOMATIC),
                "lost_frames": counts.get(STATUS_LOST, 0),
                "missing_frames": counts.get(STATUS_MISSING, 0),
                "reacquired_frames": len(reacquired),
                "reacquired_frame_range": ([min(reacquired), max(reacquired)]
                                           if reacquired else None),
                "methods": sorted({d.method for d in self.decisions if d.method}),
                "appearance_updates": self.appearance_updates,
                "empty_frame_reasons": dict(sorted(reasons.items(),
                                                   key=lambda item: -item[1])),
                "seed_mask_stats": dict(self.seed_stats),
                "policy": dict(self.policy_summary),
                "profile": PROFILE_NAME}


def track_sequence(frames: Iterable[Tuple[int, Any]], *, seed_frame: int, seed_x: float,
                   seed_y: float, interval_frames: Sequence[int],
                   cut_frames: Sequence[int] = (), target: str = "clubhead"
                   ) -> ForegroundRun:
    """Run one forward leg over an ascending stream of ``(frame, image)`` pairs.

    The stream must start at or before ``seed_frame`` and be ascending; frames before
    the seed are decoded and dropped (they carry no measurement), a frame missing from
    the stream is reported unmeasured, and nothing outside ``interval_frames`` is ever
    reported.
    """
    interval = [int(frame) for frame in interval_frames]
    cuts = {int(frame) for frame in cut_frames}
    seed = {"frame": int(seed_frame), "x": float(seed_x), "y": float(seed_y)}
    run = ForegroundRun(target=target, seed=seed, interval_frames=interval)
    policy = ForegroundPolicy(seed_frame=seed_frame, seed_x=seed_x, seed_y=seed_y,
                              target=target)

    stream = iter(frames)
    buffered: Optional[Tuple[int, Any]] = None
    last_seen: Optional[int] = None

    def image_for(frame: int):
        nonlocal buffered, last_seen
        while True:
            if buffered is None:
                try:
                    buffered = next(stream)
                except StopIteration:
                    return None
                if last_seen is not None and int(buffered[0]) <= last_seen:
                    raise ValueError("frames must arrive in ascending order")
                last_seen = int(buffered[0])
            if int(buffered[0]) == frame:
                image = buffered[1]
                buffered = None
                return image
            if int(buffered[0]) > frame:
                return None       # the stream skipped this frame
            buffered = None       # a frame this run does not report

    seed_patch = appearance = appearance_mask = seed_mask = None
    seed_image = image_for(int(seed_frame))
    if seed_image is None:
        policy.refuse_seed("seed_frame_not_decodable")
    else:
        grey = _grey(seed_image)
        seed_patch = _patch(grey, seed_x, seed_y)
        if seed_patch is None:
            policy.refuse_seed("seed_patch_not_fully_inside_frame")
        else:
            mask, stats = seed_foreground_mask(seed_patch)
            run.seed_stats = dict(stats)
            if mask is None:
                policy.refuse_seed(stats.get("reason") or "seed_mask_refused")
            else:
                appearance, appearance_mask, seed_mask = seed_patch, mask, mask
                run.available = True
    run.reason = policy.refusal_reason

    raw_by_frame: Dict[int, dict] = {}
    for frame in interval:
        request_record = None
        observation = None
        image = None
        updated = False
        if frame < int(seed_frame):
            policy.step(frame)
        elif frame in cuts and frame != int(seed_frame):
            policy.step(frame, cut=True)
        else:
            request = policy.request_for(frame) if frame > int(seed_frame) else None
            if request is not None:
                request_record = request.to_dict()
                image = image_for(frame)
                observation = (CandidateMatch(frame, False, reason="frame_not_decodable")
                               if image is None else
                               search_candidate(image, request, appearance=appearance,
                                                appearance_mask=appearance_mask,
                                                seed_patch=seed_patch,
                                                seed_mask=seed_mask))
            decision = policy.step(frame, observation)
            if (image is not None and observation is not None and observation.found
                    and decision.observed and not decision.reacquired
                    and policy.wants_appearance_update(observation)):
                refreshed = _patch(_grey(image), float(observation.x),
                                   float(observation.y))
                if refreshed is not None and float(refreshed.std()) >= MASK_MIN_CONTRAST:
                    appearance = refreshed
                    updated = True
                    run.appearance_updates += 1
        raw_by_frame[frame] = {"frame": frame, "request": request_record,
                               "decision": policy.decisions[-1].to_raw(),
                               "appearance_updated": bool(updated)}
    run.decisions = list(policy.decisions)
    run.raw_frames = [raw_by_frame.get(int(decision.source_frame),
                                       {"frame": int(decision.source_frame)})
                      for decision in run.decisions]
    run.policy_summary = policy.summary()
    return run


def track_foreground(cap, *, seed_frame: int, seed_x: float, seed_y: float,
                     start_frame: int, end_frame: int,
                     cut_frames: Sequence[int] = (), target: str = "clubhead"
                     ) -> ForegroundRun:
    """Decode the interval sequentially and track one forward leg from the seed."""
    cv2 = _cv()

    def decoded():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(seed_frame))
        frame = int(seed_frame)
        while frame <= int(end_frame):
            ok, image = cap.read()
            if not ok:
                return
            yield frame, image
            frame += 1

    return track_sequence(decoded(), seed_frame=seed_frame, seed_x=seed_x,
                          seed_y=seed_y,
                          interval_frames=list(range(int(start_frame),
                                                     int(end_frame) + 1)),
                          cut_frames=cut_frames, target=target)
