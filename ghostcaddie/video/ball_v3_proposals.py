"""High-recall candidate proposals for ball tracker v3.

This module exists because of a measurement, not a preference. Phase 0 of the v3
pass ran the v2 pipeline against the only real ball ground truth in the repository
and attributed every lost frame to the stage that lost it
(``tools/ball_v3_diagnose.py``). On labelled frames inside the v2 detector's
trained scale:

* some generator proposed the true ball on **31.0%** of frames;
* ``ConsensusParams.min_score = 0.25`` then deleted the proposal on **19 of the
  26** frames where one existed -- the true ball's own candidate scored 0.086 to
  0.243, a band sitting immediately below the floor;
* of what survived, the learned detector ranked the true ball **first 71.4%** of
  the time.

So the discriminator was never the bottleneck. **v2 was conservative where it
should have been generous and permissive where it should have been strict.** This
module inverts that: propose generously and let a ranker say no.

Three differences from ``ball_candidates.ensemble_candidates``:

1. **No score floor.** A candidate is not filtered on its own generator's raw
   response, because that response is exactly what does not separate a ball from
   grass glare -- the reason the ensemble is recall-oriented in the first place.
2. **A per-frame budget instead of a per-method cap.** v2 kept 16 peaks per
   method; the ball is often not in one method's top 16 on cluttered grass. v3
   keeps a larger per-method list and then trims to a frame budget, so a method
   that is confidently wrong cannot spend the whole budget.
3. **Spatial dedupe in native pixels.** Five methods firing on one object is one
   proposal with five witnesses, not five proposals; the witness count is kept as
   a feature rather than used as a filter.

Nothing here decides anything, and nothing here is ground truth. Every proposal
keeps the methods that witnessed it and their scores.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from . import ball_candidates as bc
from . import ball_v3_blobs as bblobs


@dataclass(frozen=True)
class ProposalParams:
    """The proposal stage's whole decision surface.

    Defaults are set from the recall sweep in ``docs/ball-tracker-v3-results.md``;
    the gate they were chosen against is ball-proposal rate on the legacy
    human-labelled corpus, which is development-consumed and never reported as
    accuracy.
    """

    #: Peaks kept per generator before the frame budget is applied. v2 used 16.
    per_method_limit: int = 96
    #: Longest side of the working copy the dense generators run on. A 4K clip
    #: halved turns a 12 px ball into 6 px, which is the regime where every
    #: method's response collapses.
    work_max_side: int = 2880
    #: Proposals kept per frame after dedupe. This is a compute budget, not a
    #: confidence statement: the ranker scores all of them.
    max_per_frame: int = 240
    #: Merge proposals closer than this many radii; the survivors keep every
    #: witnessing method.
    dedupe_radii: float = 1.0
    #: Deliberately zero. v2's 0.25 cost a measured 3x in ball recall.
    min_score: float = 0.0


@dataclass
class Proposal:
    """One deduplicated location, in native pixels, with all of its witnesses."""

    frame_index: int
    x: float
    y: float
    radius: float
    best_score: float
    methods: tuple[str, ...]
    method_scores: dict = field(default_factory=dict)
    #: Spread of the witnesses in pixels. Tight agreement across independent
    #: signals is informative; it is a feature, never a gate.
    spread: float = 0.0

    @property
    def witnesses(self) -> int:
        return len(self.methods)


def _dedupe(candidates: Sequence[bc.Candidate2], params: ProposalParams
            ) -> list[Proposal]:
    """Greedy merge by descending score. Highest-scoring member fixes geometry."""
    proposals: list[Proposal] = []
    members: list[list[bc.Candidate2]] = []
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        placed = False
        for proposal, group in zip(proposals, members):
            # The SMALLER radius sets the tolerance. Using the larger one lets a
            # coarse-scale blob 100 px wide swallow the fine-scale proposal that
            # is actually sitting on the ball -- the same class of silent recall
            # loss that ConsensusParams.min_score caused in v2, measured at 3x.
            tolerance = params.dedupe_radii * max(
                2.0, min(candidate.radius, proposal.radius))
            if math.hypot(candidate.x - proposal.x,
                          candidate.y - proposal.y) <= tolerance:
                group.append(candidate)
                placed = True
                break
        if not placed:
            proposals.append(Proposal(
                frame_index=candidate.frame_index, x=candidate.x, y=candidate.y,
                radius=candidate.radius, best_score=candidate.score,
                methods=(candidate.method,),
                method_scores={candidate.method: candidate.score}))
            members.append([candidate])

    for proposal, group in zip(proposals, members):
        scores: dict[str, float] = {}
        for member in group:
            scores[member.method] = max(scores.get(member.method, 0.0), member.score)
        proposal.methods = tuple(sorted(scores))
        proposal.method_scores = scores
        proposal.spread = max(
            (math.hypot(a.x - b.x, a.y - b.y) for a in group for b in group),
            default=0.0)
    return proposals


def propose_frame(candidates: Sequence[bc.Candidate2], params: ProposalParams
                  ) -> list[Proposal]:
    """Dedupe one frame's candidates and trim to the frame budget."""
    usable = [c for c in candidates if c.score >= params.min_score]
    proposals = _dedupe(usable, params)
    # Rank for the budget by witness count first, then by response. A location
    # three independent signals agree on is a better use of a compute slot than a
    # single method's slightly hotter peak -- but neither is a confidence claim,
    # and everything kept is scored by the ranker regardless.
    proposals.sort(key=lambda p: (p.witnesses, p.best_score), reverse=True)
    return proposals[:params.max_per_frame]


#: The v2 five, plus the generator built for the miss they make. Measured ball
#: recall on the legacy labelled corpus, per generator, at work_max_side 2880:
#: local_distinctness 0.583, brightness_cue 0.417, radial_symmetry 0.298,
#: local_contrast 0.131, flow_residual 0.036, frame_difference 0.000.
METHOD_NAMES = tuple(bc.METHOD_NAMES) + ("local_distinctness",)
GENERATORS = dict(bc.GENERATORS,
                  local_distinctness=bblobs.candidates_local_distinctness)


def propose(frames: Sequence[np.ndarray], indices: Sequence[int], *,
            params: Optional[ProposalParams] = None,
            methods: Sequence[str] = METHOD_NAMES
            ) -> tuple[dict[int, list[Proposal]], dict]:
    """Propose ball locations for every frame. Recall first; nothing is decided."""
    if len(frames) != len(indices):
        raise ValueError("frames and indices must be the same length")
    if not frames:
        raise ValueError("at least one frame is required")
    params = params or ProposalParams()
    unknown = [name for name in methods if name not in GENERATORS]
    if unknown:
        raise ValueError(f"unknown candidate methods: {unknown}")
    contexts, info = bc.build_contexts(frames, indices,
                                       max_side=params.work_max_side)
    raw: dict[int, list[bc.Candidate2]] = {index: [] for index in indices}
    for name in methods:
        generator = GENERATORS[name]
        for context in contexts:
            for candidate in generator(context,
                                       limit=params.per_method_limit):
                if candidate.score > params.min_score:
                    raw[context.index].append(candidate)
    per_frame = {index: propose_frame(raw.get(index, []), params)
                 for index in indices}
    info = dict(info)
    info.update({
        "per_method_limit": params.per_method_limit,
        "work_max_side": params.work_max_side,
        "max_per_frame": params.max_per_frame,
        "min_score": params.min_score,
        "mean_raw_candidates": round(
            float(np.mean([len(raw.get(i, [])) for i in indices])), 2),
        "mean_proposals": round(
            float(np.mean([len(per_frame[i]) for i in indices])), 2),
    })
    return per_frame, info


def proposal_recall(per_frame: dict[int, Sequence[Proposal]],
                    truth: dict[int, tuple[float, float, float]],
                    *, tolerance_radii: float = 1.0) -> dict:
    """Does the proposal set contain the ball? The ceiling on everything after it.

    ``truth`` maps frame index to (x, y, radius). A proposal counts as finding the
    ball when it lands within ``tolerance_radii`` ball radii: a prediction further
    away than the ball is wide is a prediction on a different object.
    """
    found = missed = 0
    ranks: list[int] = []
    for index, (gx, gy, radius) in truth.items():
        tolerance = max(6.0, tolerance_radii * radius)
        proposals = per_frame.get(index) or []
        hits = [position for position, p in enumerate(proposals)
                if math.hypot(p.x - gx, p.y - gy) <= tolerance]
        if hits:
            found += 1
            ranks.append(min(hits) + 1)
        else:
            missed += 1
    total = max(1, found + missed)
    return {
        "frames": found + missed,
        "found": found,
        "missed": missed,
        "recall": round(found / total, 4),
        "median_position_in_list": float(np.median(ranks)) if ranks else None,
    }
