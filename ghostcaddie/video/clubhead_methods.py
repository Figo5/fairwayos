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

# Two supported, genuinely distinct algorithms: Lucas-Kanade point tracking with
# forward/backward flow consistency, and region/template appearance re-detection.
# Alias names must not be presented as distinct algorithms.
SUPPORTED_METHODS = ("lk_point", "region_template")

_UNSET = object()

def _region_template_track(frames, seed_frame, seed_point, *,
                           match_threshold, max_search_radius_px,
                           template_size_px, ambiguity_margin):
    import cv2
    import numpy as np
    out = [_unavailable("region_template", i, "before_seed") for i in range(seed_frame)]
    x, y = float(seed_point[0]), float(seed_point[1])
    out.append(ClubheadCandidate("region_template", seed_frame, (x, y), CandidateState.OBSERVED, 1.0))
    half = template_size_px // 2
    h, w = frames[0].shape[:2]
    xs, ys = int(round(x)), int(round(y))
    if not (half <= ys < h - half and half <= xs < w - half):
        return out + [_unavailable("region_template", j, "seed_template_out_of_bounds") for j in range(seed_frame + 1, len(frames))]
    template = cv2.cvtColor(frames[seed_frame], cv2.COLOR_BGR2GRAY)[ys - half:ys + half, xs - half:xs + half]
    for i in range(seed_frame + 1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        cxr = int(round(x)); cyr = int(round(y))
        x0 = max(0, cxr - max_search_radius_px)
        x1 = min(w, cxr + max_search_radius_px + half * 2)
        y0 = max(0, cyr - max_search_radius_px)
        y1 = min(h, cyr + max_search_radius_px + half * 2)
        region = gray[y0:y1, x0:x1]
        if region.shape[0] < template.shape[0] or region.shape[1] < template.shape[1]:
            out.extend(_unavailable("region_template", j, "search_region_out_of_bounds") for j in range(i, len(frames)))
            break
        scores = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        best_y, best_x = np.unravel_index(int(np.argmax(scores)), scores.shape)
        best = float(scores[best_y, best_x])
        masked = np.array(scores)
        masked[best_y, best_x] = -np.inf
        second = float(np.max(masked)) if masked.size else -np.inf
        cx = x0 + int(best_x) + half
        cy = y0 + int(best_y) + half
        valid = all(math.isfinite(v) for v in (best, float(cx), float(cy))) and 0 <= cx < w and 0 <= cy < h
        if not valid or best < match_threshold:
            out.extend(_unavailable("region_template", j, "low_appearance_match") for j in range(i, len(frames)))
            break
        if best - second < ambiguity_margin:
            out.extend(_unavailable("region_template", j, "appearance_match_ambiguous") for j in range(i, len(frames)))
            break
        conf = max(0.0, min(1.0, best))
        out.append(ClubheadCandidate("region_template", i, (float(cx), float(cy)), CandidateState.OBSERVED, conf))
        x, y = float(cx), float(cy)
    return out

def track_candidate(method, frames, seed_frame, seed_point, *, max_step=55.0, max_backward_error_pixels=3.0,
                    match_threshold=_UNSET, max_search_radius_px=_UNSET, template_size_px=_UNSET,
                    ambiguity_margin=_UNSET):
    """Bounded clubhead candidate tracking.

    ``lk_point``: Lucas-Kanade with forward/backward flow consistency; no
    semantic reacquisition across ambiguity. ``max_backward_error_pixels`` is
    the explicit forward/backward limit; confidence is
    ``1 - backward_error / max_backward_error_pixels``.

    ``region_template``: appearance re-detection. The template is extracted
    around the seed point on the seed frame and re-matched by normalized
    cross-correlation within ``max_search_radius_px`` of the previous position.
    Ambiguous matches (best-vs-second margin below ``ambiguity_margin``) or
    matches below ``match_threshold`` mark every remaining frame UNAVAILABLE —
    never a guessed position. Confidence is the bounded match score.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"unsupported clubhead method {method!r}; supported methods: {SUPPORTED_METHODS}"
        )
    region_kwargs = dict(match_threshold=match_threshold, max_search_radius_px=max_search_radius_px,
                         template_size_px=template_size_px, ambiguity_margin=ambiguity_margin)
    for name, value in region_kwargs.items():
        if method == "lk_point" and value is not _UNSET:
            raise ValueError(f"{name} applies only to region_template")
    if (isinstance(max_backward_error_pixels, bool)
            or not isinstance(max_backward_error_pixels, (int, float))
            or not math.isfinite(max_backward_error_pixels)
            or max_backward_error_pixels <= 0):
        raise ValueError("max_backward_error_pixels must be a finite positive number")
    try:
        import cv2, numpy as np
    except ImportError as exc: raise RuntimeError("OpenCV and NumPy are required") from exc
    if not frames or not 0 <= seed_frame < len(frames): raise ValueError("seed frame outside frames")
    h,w=frames[0].shape[:2]; x,y=seed_point
    if not 0 <= x < w or not 0 <= y < h: raise ValueError("seed point outside image")
    if method == "region_template":
        resolved = {
            "match_threshold": 0.7 if match_threshold is _UNSET else match_threshold,
            "max_search_radius_px": 30 if max_search_radius_px is _UNSET else max_search_radius_px,
            "template_size_px": 24 if template_size_px is _UNSET else template_size_px,
            "ambiguity_margin": 0.1 if ambiguity_margin is _UNSET else ambiguity_margin,
        }
        for name in ("match_threshold", "ambiguity_margin"):
            v = resolved[name]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0.0 < v < 1.0:
                raise ValueError(f"{name} must be a finite number in (0, 1)")
        for name in ("max_search_radius_px", "template_size_px"):
            v = resolved[name]
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be a positive integer")
        return _region_template_track(frames, seed_frame, seed_point, **resolved)
    out=[_unavailable(method,i,"before_seed") for i in range(seed_frame)]
    out.append(ClubheadCandidate(method,seed_frame,(float(x),float(y)),CandidateState.OBSERVED,1.0))
    prev=cv2.cvtColor(frames[seed_frame],cv2.COLOR_BGR2GRAY); p=np.array([[[x,y]]],np.float32)
    fb_tol=float(max_backward_error_pixels)
    for i in range(seed_frame+1,len(frames)):
        cur=cv2.cvtColor(frames[i],cv2.COLOR_BGR2GRAY); nxt,st,_=cv2.calcOpticalFlowPyrLK(prev,cur,p,None,winSize=(21,21),maxLevel=2)
        if nxt is None or st is None or not int(st[0][0]):
            out.extend(_unavailable(method,j,"flow_ambiguous") for j in range(i, len(frames)))
            break
        q=tuple(float(v) for v in nxt[0][0]); step=math.hypot(q[0]-float(p[0][0][0]),q[1]-float(p[0][0][1]))
        back,bst,_=cv2.calcOpticalFlowPyrLK(cur,prev,nxt,None,winSize=(21,21),maxLevel=2)
        fb=math.hypot(float(back[0][0][0])-float(p[0][0][0]),float(back[0][0][1])-float(p[0][0][1])) if back is not None and bst is not None and int(bst[0][0]) else float('inf')
        valid=all(math.isfinite(v) for v in (*q,step,fb)) and step<=max_step and fb<=fb_tol and 0<=q[0]<w and 0<=q[1]<h
        if not valid:
            out.extend(_unavailable(method,j,"motion_or_backward_ambiguity") for j in range(i, len(frames)))
            break
        conf=max(0.0,min(1.0,1-fb/fb_tol)); out.append(ClubheadCandidate(method,i,q,CandidateState.OBSERVED,conf)); p=nxt; prev=cur
    return out