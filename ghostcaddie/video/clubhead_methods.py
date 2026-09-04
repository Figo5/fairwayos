"""Independent research-only clubhead candidate methods."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math

class CandidateState(str, Enum):
    OBSERVED = "observed"
    UNAVAILABLE = "unavailable"

@dataclass(frozen=True)
class ClubheadCandidate:
    method: str
    frame_index: int
    point: tuple[float, float] | None
    state: CandidateState
    confidence: float
    warning: str | None = None
    def __post_init__(self):
        if not self.method or isinstance(self.frame_index, bool) or self.frame_index < 0:
            raise ValueError("invalid candidate identity")
        if not isinstance(self.state, CandidateState):
            raise ValueError("invalid candidate state")
        if isinstance(self.confidence, bool) or not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("invalid candidate confidence")
        if self.point is not None and (len(self.point) != 2 or any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) for v in self.point)):
            raise ValueError("invalid candidate point")
        if self.state is CandidateState.OBSERVED and self.point is None:
            raise ValueError("observed candidate needs point")

@dataclass(frozen=True)
class ClubheadMethodComparison:
    candidates: tuple[ClubheadCandidate, ...]
    selected_method: str | None
    research_only: bool = True
    ground_truth: bool = False
    production_eligible: bool = False
    @classmethod
    def from_candidates(cls, candidates):
        items=tuple(sorted(candidates,key=lambda c:(c.method,c.frame_index)))
        if not items: raise ValueError("at least one candidate required")
        best=max((c for c in items if c.state is CandidateState.OBSERVED), key=lambda c:(c.confidence,-len(c.method)), default=None)
        return cls(items, best.method if best else None)
    @property
    def method_names(self): return tuple(sorted({c.method for c in self.candidates}))

def _unavailable(method, frame, warning):
    return ClubheadCandidate(method, frame, None, CandidateState.UNAVAILABLE, 0.0, warning)

def track_candidate(method, frames, seed_frame, seed_point, *, max_step=55.0, min_similarity=.55):
    """Bounded local candidate; no semantic promotion or reacquisition across ambiguity."""
    try:
        import cv2, numpy as np
    except ImportError as exc: raise RuntimeError("OpenCV and NumPy are required") from exc
    if not frames or not 0 <= seed_frame < len(frames): raise ValueError("seed frame outside frames")
    h,w=frames[0].shape[:2]; x,y=seed_point
    if not 0 <= x < w or not 0 <= y < h: raise ValueError("seed point outside image")
    out=[_unavailable(method,i,"before_seed") for i in range(seed_frame)]
    out.append(ClubheadCandidate(method,seed_frame,(float(x),float(y)),CandidateState.OBSERVED,1.0))
    prev=cv2.cvtColor(frames[seed_frame],cv2.COLOR_BGR2GRAY); p=np.array([[[x,y]]],np.float32)
    for i in range(seed_frame+1,len(frames)):
        cur=cv2.cvtColor(frames[i],cv2.COLOR_BGR2GRAY); nxt,st,_=cv2.calcOpticalFlowPyrLK(prev,cur,p,None,winSize=(21,21),maxLevel=2)
        if nxt is None or st is None or not int(st[0][0]): out.append(_unavailable(method,i,"flow_ambiguous")); break
        q=tuple(float(v) for v in nxt[0][0]); step=math.hypot(q[0]-float(p[0][0][0]),q[1]-float(p[0][0][1]))
        back,bst,_=cv2.calcOpticalFlowPyrLK(cur,prev,nxt,None,winSize=(21,21),maxLevel=2)
        fb=math.hypot(float(back[0][0][0])-float(p[0][0][0]),float(back[0][0][1])-float(p[0][0][1])) if back is not None and bst is not None and int(bst[0][0]) else float('inf')
        valid=all(math.isfinite(v) for v in (*q,step,fb)) and step<=max_step and fb<=3.0 and 0<=q[0]<w and 0<=q[1]<h
        if not valid: out.append(_unavailable(method,i,"motion_or_backward_ambiguity")); break
        conf=max(0.0,min(1.0,1-fb/3)); out.append(ClubheadCandidate(method,i,q,CandidateState.OBSERVED,conf)); p=nxt; prev=cur
    return out
