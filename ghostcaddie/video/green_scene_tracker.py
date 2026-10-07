"""Experimental, operator-initialized ball observations in a green camera view.

White component association is a clip-development method, not a trained or
validated general golf-ball detector. Missing components stay missing. Camera
motion is a ground-only image similarity estimate, with explicit quality gates.
"""
import cv2
import numpy as np

def pin(hsv):
 m=cv2.inRange(hsv,np.array([21,110,90]),np.array([37,255,255]))
 k,l,stats,cents=cv2.connectedComponentsWithStats(m)
 cand=[]
 for idx,st in enumerate(stats[1:],1):
  x,y,w,h,a=map(int,st)
  if h>35 and h/w>3 and a>30:
   ys,xs=np.nonzero(l==idx);bottom=ys.max();xbase=float(np.median(xs[ys>=bottom-4]));cand.append((h,(xbase,float(bottom))))
 return max(cand)[1] if cand else None

def candidates(hsv):
 m=cv2.inRange(hsv,np.array([0,0,160]),np.array([179,100,255]))
 k,l,stats,cents=cv2.connectedComponentsWithStats(m);rows=[]
 for st,c in zip(stats[1:],cents[1:]):
  x,y,w,h,a=map(int,st)
  if 3<=a<=3000 and max(w,h)<80 and .5<w/h<2.2 and a/(w*h)>.4:
   rows.append({'point':c.tolist(),'size':[w,h],'area':a})
 return rows

def affine(prev,curr,ball,pinpoint):
 pg=cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY);cg=cv2.cvtColor(curr,cv2.COLOR_BGR2GRAY)
 hsv=cv2.cvtColor(prev,cv2.COLOR_BGR2HSV)
 mask=cv2.inRange(hsv,np.array([20,18,35]),np.array([95,230,215]));mask[:170]=0
 for pt in [ball,pinpoint]:
  if pt:cv2.circle(mask,tuple(map(round,pt)),65,0,-1)
 p=cv2.goodFeaturesToTrack(pg,1000,.001,6,mask=mask)
 if p is None or len(p)<12:return None,{'reason':'few_features','features':0 if p is None else len(p)}
 q,ok,e=cv2.calcOpticalFlowPyrLK(pg,cg,p,None,winSize=(41,41),maxLevel=3)
 back,bok,be=cv2.calcOpticalFlowPyrLK(cg,pg,q,None,winSize=(41,41),maxLevel=3)
 good=(ok.ravel()>0)&(bok.ravel()>0)&(np.linalg.norm(back-p,axis=2).ravel()<1)&(e.ravel()<25)
 pp=p[good].reshape(-1,2);qq=q[good].reshape(-1,2)
 if len(pp)<12:return None,{'reason':'few_matches','matches':len(pp)}
 M,inside=cv2.estimateAffinePartial2D(pp,qq,method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=1500)
 if M is None:return None,{'reason':'fit_failed'}
 use=inside.ravel()>0;scale=float(np.linalg.norm(M[:,0]));err=np.linalg.norm(pp[use]@M[:,:2].T+M[:,2]-qq[use],axis=1);span=np.ptp(pp[use],axis=0)
 quality={'inliers':int(use.sum()),'matches':len(pp),'scale':scale,'rms':float(np.sqrt(np.mean(err**2))),'span':span.tolist()}
 if use.sum()<12 or use.mean()<.65 or not .90<scale<1.12 or quality['rms']>1.0 or min(span)<80:return None,dict(quality,reason='quality_gate')
 return M,quality


def track_ground_anchor(previous_gray, current_gray, point):
    """Track an operator-selected ground anchor with forward/backward LK.

    On failure return no point. The caller must explicitly initialize a new
    anchor; a previous coordinate is never returned as a current observation.
    """
    if point is None:
        return None, {'reason': 'anchor_missing'}
    if previous_gray.shape != current_gray.shape or previous_gray.ndim != 2:
        raise ValueError("anchor images must be matching grayscale frames")
    initial = np.asarray(point, np.float32).reshape(1, 1, 2)
    if not np.isfinite(initial).all():
        raise ValueError("anchor point must be finite")
    q, ok, error = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, initial, None, winSize=(41, 41), maxLevel=3)
    back, bok, _ = cv2.calcOpticalFlowPyrLK(current_gray, previous_gray, q, None, winSize=(41, 41), maxLevel=3)
    quality = {'forward_backward_px': float(np.linalg.norm(back - initial)), 'photometric_error': float(error[0, 0])}
    height, width = current_gray.shape
    x, y = q.ravel()
    if (not ok[0, 0] or not bok[0, 0] or not np.isfinite(q).all()
            or not all(np.isfinite(v) for v in quality.values())
            or quality['forward_backward_px'] > 1.2 or quality['photometric_error'] > 25
            or not (0 <= x < width and 0 <= y < height)):
        return None, dict(quality, reason='quality_gate')
    return q.ravel().tolist(), quality
