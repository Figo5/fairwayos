"""Reference-free temporal association over frozen detector candidates.

Why this exists. Two independent audits established that the golf-ball detector
places a tiny candidate on the visually apparent ball in 40 of 43 moving frames,
at raw confidences as low as 0.01, surrounded by brighter background and
clubhead distractors. So the detector is not the bottleneck -- picking the right
candidate is. Confidence alone cannot do it: the ball is routinely the FAINTEST
candidate in the frame.

What this does. It reads every candidate in every frame and proposes one
coherent path through them, scoring motion coherence rather than confidence. The
proposal is found by dynamic programming over the whole interval, so it is
initialised by the optimiser -- there is no seed, no reference coordinate, no
crop, and no per-frame greedy pick that could lock onto a distractor early.

What it refuses to do. The reference-free proposal is not an accepted ball
observation without independent identity qualification. A frame whose candidates
cannot continue the proposal emits NOTHING. Gaps are gaps: nothing is
interpolated, smoothed or carried forward, and every proposed point is one of
the detector's own candidate objects.

Invalid geometry is rejected at the boundary, not repaired: the decoder audit
found NaN coordinates and negative/inverted boxes reaching NMS, and silently
reordering corners would have hidden that.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

State = Tuple[int, int, Optional[int], Optional[int]]


class InvalidCandidate(ValueError):
    """Raised for geometry that must never reach association or rendering."""


@dataclass(frozen=True)
class Candidate:
    frame: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    candidate_index: Optional[int] = None

    def __post_init__(self):
        for n in ("x1", "y1", "x2", "y2", "score"):
            v = float(getattr(self, n))
            if not math.isfinite(v):
                raise InvalidCandidate(f"{n} is not finite: {v!r}")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise InvalidCandidate(
                f"box is inverted or empty: "
                f"({self.x1}, {self.y1}) -> ({self.x2}, {self.y2}). "
                f"Corners are NOT reordered: an inverted box is a decoder bug.")

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def diag(self) -> float:
        return math.hypot(self.width, self.height)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict:
        d = {"frame": self.frame,
             "xyxy": [self.x1, self.y1, self.x2, self.y2],
             "center_xy": [self.cx, self.cy],
             "wh": [self.width, self.height],
             "score": float(self.score)}
        if self.candidate_index is not None:
            d["candidate_index"] = self.candidate_index
        return d


def parse_candidates(rows: Sequence[dict]) -> Tuple[List[Candidate], int]:
    """Build candidates from raw rows, DROPPING invalid geometry and counting it."""
    kept, rejected = [], 0
    for fallback_index, r in enumerate(rows or []):
        b = r.get("box_xyxy") or r.get("bbox_xyxy")
        if not b or len(b) != 4:
            rejected += 1
            continue
        try:
            kept.append(Candidate(frame=int(r["frame"]), x1=float(b[0]),
                                  y1=float(b[1]), x2=float(b[2]), y2=float(b[3]),
                                  score=float(r.get("score", 0.0)),
                                  candidate_index=r.get("candidate_index", fallback_index)))
        except (InvalidCandidate, TypeError, ValueError):
            rejected += 1
    return kept, rejected


@dataclass(frozen=True)
class AssociationPolicy:
    """Frozen before any evaluation. No value here is fitted to a reference.

    max_step_px_per_frame is a geometric bound, not a tuned threshold: on a
    1280-wide frame, an object that moves more than a quarter of the frame width
    between adjacent frames is not the same object being followed.
    """
    max_step_px_per_frame: float = 320.0
    max_frame_gap: int = 4
    miss_cost: float = 60.0
    accel_weight: float = 1.0
    size_change_weight: float = 40.0
    score_weight: float = 20.0
    ideal_ball_diag_px: float = 10.0
    max_ball_diag_px: float = 35.0
    ball_scale_weight: float = 75.0
    oversize_reject_diag_px: float = 120.0
    min_track_frames: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AssociationResult:
    """Reference-free association output before/after identity acceptance.

    `proposed_by_frame` preserves the coherent path as a research proposal.
    `accepted_by_frame` is empty unless an independent identity qualifier is
    explicitly supplied by a caller outside this reference-free selector.
    """
    proposed_by_frame: Dict[int, Optional[Candidate]]
    accepted_by_frame: Dict[int, Optional[Candidate]]
    acceptance_state: str
    rejection_reasons: Tuple[str, ...]

    @property
    def proposed_frames(self) -> int:
        return sum(c is not None for c in self.proposed_by_frame.values())

    @property
    def accepted_frames(self) -> int:
        return sum(c is not None for c in self.accepted_by_frame.values())


_MISS = None


def _pair_cost(prev: Candidate, cur: Candidate, prev_v, gap: int,
               p: AssociationPolicy) -> Optional[float]:
    """Cost of continuing a path from `prev` to `cur`. None = not allowed."""
    dx, dy = cur.cx - prev.cx, cur.cy - prev.cy
    step = math.hypot(dx, dy) / max(1, gap)
    if step > p.max_step_px_per_frame:
        return None
    if prev.diag > p.oversize_reject_diag_px or cur.diag > p.oversize_reject_diag_px:
        return None
    vx, vy = dx / max(1, gap), dy / max(1, gap)
    cost = 0.0
    if prev_v is not None:
        # acceleration: the dominant term, and the only one that knows the ball
        # is a ballistic object rather than a bright blob
        cost += p.accel_weight * math.hypot(vx - prev_v[0], vy - prev_v[1])
    # a tracked object's apparent size changes smoothly
    denom = max(1e-6, prev.diag + cur.diag)
    cost += p.size_change_weight * (2.0 * abs(cur.diag - prev.diag) / denom)
    cost += p.ball_scale_weight * (
        abs(cur.diag - p.ideal_ball_diag_px) / p.max_ball_diag_px)
    # confidence helps, but cannot outvote coherence. Kept NON-NEGATIVE so that
    # path costs stay comparable: a high score reduces the penalty, it never
    # pays a path to exist.
    cost += p.score_weight * (1.0 - min(1.0, max(0.0, float(cur.score))))
    return cost


def _associate_path(frames: Dict[int, Sequence[Candidate]],
                    policy: Optional[AssociationPolicy] = None
                    ) -> Dict[int, Optional[Candidate]]:
    """Propose one coherent path through all candidates. Gaps stay empty.

    Dynamic programming over (frame, candidate, incoming velocity): each state
    keeps the cheapest way to arrive at that candidate, so the chosen track is
    globally cheapest rather than greedily locked in at the first frame.
    """
    p = policy or AssociationPolicy()
    order = sorted(frames)
    if not order:
        return {}

    # state: (frame_idx, cand_idx, prev_frame_idx, prev_cand_idx) ->
    # (length, cost, prev_state, velocity).  The previous candidate is part of
    # the state because acceleration is second-order: the same current candidate
    # can be reached with different incoming velocities.
    # Compared lexicographically by (-length, cost): a path that explains more
    # frames coherently beats a shorter cheaper one. Without this the optimum is
    # always a single isolated candidate, because every edge costs something.
    best: Dict[State, Tuple[int, float, Optional[State], Optional[tuple]]] = {}
    for fi, f in enumerate(order):
        for ci, c in enumerate(frames[f] or []):
            # a path may start at any candidate: this is the auto-initialisation,
            # chosen by the optimiser and not by a supplied coordinate
            best[(fi, ci, None, None)] = (1, 0.0, None, None)

    for fi, f in enumerate(order):
        for ci, c in enumerate(frames[f] or []):
            incoming = [(s, v) for s, v in best.items() if s[0] == fi and s[1] == ci]
            for fj in range(fi + 1, min(fi + 1 + p.max_frame_gap, len(order))):
                gap = order[fj] - f
                if gap > p.max_frame_gap:
                    break
                for cj, n in enumerate(frames[order[fj]] or []):
                    ns: State = (fj, cj, fi, ci)
                    for state, cur in incoming:
                        step = _pair_cost(c, n, cur[3], gap, p)
                        if step is None:
                            continue
                        cand = (cur[0] + 1, cur[1] + step + p.miss_cost * (gap - 1))
                        have = best.get(ns)
                        if have is None or (-cand[0], cand[1]) < (-have[0], have[1]):
                            v = ((n.cx - c.cx) / gap, (n.cy - c.cy) / gap)
                            best[ns] = (cand[0], cand[1], state, v)

    if not best:
        return {f: None for f in order}

    def path_of(state):
        out = []
        while state is not None:
            out.append(state)
            state = best[state][2]
        return list(reversed(out))

    # longest coherent path first, then lowest mean edge cost
    def rank(s):
        length, cost = best[s][0], best[s][1]
        return (-length, cost / max(1, length - 1))

    end = min(best, key=rank)
    path = path_of(end)
    if len(path) < p.min_track_frames:
        return {f: None for f in order}

    track: Dict[int, Optional[Candidate]] = {f: None for f in order}
    for fi, ci, _, _ in path:
        track[order[fi]] = frames[order[fi]][ci]
    return track


def associate_proposals(frames: Dict[int, Sequence[Candidate]],
                        policy: Optional[AssociationPolicy] = None,
                        *,
                        identity_qualified: bool = False
                        ) -> AssociationResult:
    """Return proposed association separately from accepted observations.

    This selector is reference-free: it has no seed, crop, reviewed coordinate,
    or independent visual identity input. Therefore its coherent path is only a
    proposal by default. Callers that have independently qualified identity may
    opt in with `identity_qualified=True`; this function does not fabricate that
    evidence from length, confidence, size, or motion coherence.
    """
    proposed = _associate_path(frames, policy)
    if identity_qualified:
        return AssociationResult(
            proposed_by_frame=proposed,
            accepted_by_frame=dict(proposed),
            acceptance_state="accepted_identity_qualified",
            rejection_reasons=(),
        )
    accepted = {f: None for f in proposed}
    reasons = ("independent_identity_required",)
    if not any(c is not None for c in proposed.values()):
        reasons = ("no_coherent_proposal",) + reasons
    return AssociationResult(
        proposed_by_frame=proposed,
        accepted_by_frame=accepted,
        acceptance_state="proposal_only",
        rejection_reasons=reasons,
    )


def associate(frames: Dict[int, Sequence[Candidate]],
              policy: Optional[AssociationPolicy] = None,
              *,
              identity_qualified: bool = False
              ) -> Dict[int, Optional[Candidate]]:
    """Return accepted ball observations, abstaining by default.

    Use `associate_proposals` to inspect the unvalidated coherent path. The
    accepted API intentionally emits no ball observations unless independent
    identity qualification is supplied by the caller.
    """
    return associate_proposals(
        frames, policy, identity_qualified=identity_qualified).accepted_by_frame
