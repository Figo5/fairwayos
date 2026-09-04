"""Research-only hybrid seeded point tracker.

Combines local template matching with forward/backward Lucas-Kanade checks.
Ambiguous or inconsistent evidence terminates the track; gaps are unavailable.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional, Tuple

@dataclass(frozen=True)
class HybridPoint:
    frame_index: int
    point: Optional[Tuple[float,float]]
    state: str
    confidence: float
    uncertainty: Optional[float]
    warning: Optional[str]=None

class HybridSeededTracker:
    def __init__(self, *, search_radius=28, template_radius=8, fb_error=2.5,
                 max_step=60.0, min_match=0.55, max_prediction_frames=0):
        if search_radius<=0 or template_radius<=1 or fb_error<=0 or max_step<=0 or not 0<=min_match<=1 or max_prediction_frames<0:
            raise ValueError('invalid hybrid tracker limits')
        self.search_radius=int(search_radius); self.template_radius=int(template_radius)
        self.fb_error=float(fb_error); self.max_step=float(max_step); self.min_match=float(min_match)
        self.max_prediction_frames=int(max_prediction_frames)

    def track(self, frames, seed_frame_index:int, seed_point:Tuple[float,float]):
        if not frames or seed_frame_index<0 or seed_frame_index>=len(frames): raise ValueError('seed frame outside frames')
        first=frames[seed_frame_index]; h,w=first.shape[:2]; x,y=seed_point
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (x,y)) or not(0<=x<w and 0<=y<h): raise ValueError('seed point outside image')
        try:
            import cv2, numpy as np
        except ImportError as exc: raise RuntimeError('OpenCV and NumPy are required') from exc
        def gray(f): return cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) if len(f.shape)==3 else f
        prev=gray(first); point=np.array([[[x,y]]],dtype=np.float32)
        r=[HybridPoint(i,None,'unavailable',0.0,None,'before_seed') for i in range(seed_frame_index)]
        r.append(HybridPoint(seed_frame_index,(float(x),float(y)),'observed',1.0,0.0))
        for i in range(seed_frame_index+1,len(frames)):
            cur=gray(frames[i]); nxt,st,err=cv2.calcOpticalFlowPyrLK(prev,cur,point,None,winSize=(21,21),maxLevel=3)
            valid=nxt is not None and st is not None and int(st[0][0])==1
            warning=None; confidence=0.0; candidate=None
            if valid:
                candidate=(float(nxt[0][0][0]),float(nxt[0][0][1])); step=math.hypot(candidate[0]-float(point[0][0][0]),candidate[1]-float(point[0][0][1]))
                back,bst,_=cv2.calcOpticalFlowPyrLK(cur,prev,nxt,None,winSize=(21,21),maxLevel=3)
                fb=math.hypot(float(back[0][0][0])-float(point[0][0][0]),float(back[0][0][1])-float(point[0][0][1])) if back is not None and bst is not None and int(bst[0][0])==1 else float('inf')
                valid=all(math.isfinite(v) for v in (*candidate,step,fb)) and step<=self.max_step and fb<=self.fb_error and self.template_score(prev,cur,point[0][0],candidate,cv2)>=self.min_match
                if valid: confidence=max(0.0,min(1.0,1.0-fb/self.fb_error));
                else: warning='flow_or_template_inconsistent'
            if valid:
                r.append(HybridPoint(i,candidate,'observed',confidence, self.fb_error*(1-confidence))); point=np.array([[[candidate[0],candidate[1]]]],dtype=np.float32)
            else:
                r.append(HybridPoint(i,None,'unavailable',0.0,None,warning or 'track_terminated')); break
            prev=cur
        return r

    def template_score(self, prev, cur, old, new, cv2):
        ox,oy=map(int,old); nx,ny=map(int,new); r=self.template_radius
        if ox-r<0 or oy-r<0 or ox+r>=prev.shape[1] or oy+r>=prev.shape[0] or nx-r<0 or ny-r<0 or nx+r>=cur.shape[1] or ny+r>=cur.shape[0]: return 0.0
        a=prev[oy-r:oy+r+1,ox-r:ox+r+1]; b=cur[ny-r:ny+r+1,nx-r:nx+r+1]
        score=cv2.matchTemplate(b,a,cv2.TM_CCOEFF_NORMED)[0][0] if a.shape==b.shape else 0.0
        return float(score) if math.isfinite(float(score)) else 0.0
