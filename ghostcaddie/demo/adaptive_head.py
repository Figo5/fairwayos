"""Seed-supported mask geometry, isolated from the frozen first head experiment.

A placement may touch the edge of a head rather than its centroid. Seed admission
therefore measures distance to actual compact-core pixels. Core thickness comes
from the original seed neighborhood, not the thickest unrelated masked object.
No later coordinates, correction, reference labels or target class enter here.
"""
import math
from .segmented_head import POINT_POLICY
from .foreground_tracker import ForegroundPolicy, CandidateMatch

PROFILE = {
    'seed_neighborhood_radius_px':3,'seed_support_distance_px':3,
    'core_min_distance_px':1.8,'core_seed_peak_fraction':.75,
    'core_min_area_px':3,'core_max_area_px':180,'core_max_dimension_px':24,
    'min_elongation':1.25,'min_seed_area_ratio':.25,'max_seed_area_ratio':4,
    'threshold_frozen_after_seed':True,'exported_confidence':None,
    'research_only':True,'production_eligible':False,
    'inactive_probability_thresholds':['min_score','seed_identity_min_score','reacq_min_score','ambiguity_margin'],
}


def regions(mask, threshold, seed_point=None):
    import cv2
    import numpy as np
    dt=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,3)
    core=(dt>=threshold).astype(np.uint8)
    count,labels,stats,centers=cv2.connectedComponentsWithStats(core,8)
    result=[]
    for index in range(1,count):
        x,y,w,h,area=map(int,stats[index])
        if not PROFILE['core_min_area_px']<=area<=PROFILE['core_max_area_px']:continue
        if max(w,h)>PROFILE['core_max_dimension_px']:continue
        pixels=np.argwhere(labels==index)[:,::-1].astype(np.float32)
        axes=cv2.minAreaRect(pixels)[1]
        elongation=max(axes)/max(min(axes),.5)
        item={'xy':centers[index].tolist(),'area':area,'bbox':[x,y,w,h],
              'elongation':float(elongation),'core_threshold_px':threshold}
        if seed_point is not None:
            item['seed_support_distance_px']=float(np.linalg.norm(pixels-np.asarray(seed_point),axis=1).min())
        result.append(item)
    return result


def seed_threshold(mask, point):
    import cv2
    import numpy as np
    dt=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,3)
    x,y=map(round,point);radius=PROFILE['seed_neighborhood_radius_px']
    local=dt[max(0,y-radius):min(dt.shape[0],y+radius+1),max(0,x-radius):min(dt.shape[1],x+radius+1)]
    peak=float(local.max()) if local.size else 0
    return max(PROFILE['core_min_distance_px'],PROFILE['core_seed_peak_fraction']*peak)


class SeedSupportedHeadPolicy:
    def __init__(self,seed,origin,cut_frames,point_policy=None):
        self.seed=dict(seed);self.origin=list(origin);self.cuts=set(cut_frames)
        self.policy_settings={**POINT_POLICY,**(point_policy or {})}
        self.policy=ForegroundPolicy(seed_frame=seed['frame'],seed_x=seed['x'],seed_y=seed['y'],**self.policy_settings)
        self.policy.seed_decision();self.last_frame=seed['frame']-1
        self.threshold=None;self.seed_region=None;self.seed_valid=None

    def step(self,frame,mask):
        if frame!=self.last_frame+1:raise ValueError('Masks must arrive once in contiguous forward order')
        self.last_frame=frame;admitted=[]
        if frame==self.seed['frame']:
            point=[self.seed['x']-self.origin[0],self.seed['y']-self.origin[1]]
            self.threshold=seed_threshold(mask,point)
            parts=regions(mask,self.threshold,point)
            choices=[p for p in parts if p['seed_support_distance_px']<=PROFILE['seed_support_distance_px'] and p['elongation']>=PROFILE['min_elongation']]
            self.seed_region=choices[0] if len(choices)==1 else None
            self.seed_valid=self.seed_region is not None
            if not self.seed_valid:
                self.policy.measure=False;self.policy.refusal_reason='no unique nonround compact region supported by initial placement'
            decision=self.policy.decisions[0]
        else:
            parts=regions(mask,self.threshold)
            request=self.policy.request_for(frame)
            if request and self.seed_region:
                for p in parts:
                    xy=[p['xy'][0]+self.origin[0],p['xy'][1]+self.origin[1]]
                    ratio=p['area']/self.seed_region['area']
                    if (p['elongation']>=PROFILE['min_elongation'] and
                        PROFILE['min_seed_area_ratio']<=ratio<=PROFILE['max_seed_area_ratio'] and
                        math.dist(xy,[request.anchor_x,request.anchor_y])<=request.radius_px):
                        admitted.append({**p,'native_xy':xy})
            candidate=None
            if len(admitted)==1:
                xy=admitted[0]['native_xy'];candidate=CandidateMatch(frame,True,x=xy[0],y=xy[1],score=1,seed_score=1,second_score=0,transform='seed-supported learned mask region')
            decision=self.policy.step(frame,candidate,cut=frame in self.cuts)
        point=decision.to_point()
        if point['origin']=='automatic':
            point['method']='sam2_head_region_reacquired' if decision.method.endswith('reacquired') else 'sam2_head_region'
            point['score']=None
        return point,{'frame':frame,'parts':parts,'mask_area':int(mask.sum()),
                      'threshold_from_seed':self.threshold,'admitted':admitted,
                      'decision':decision.to_raw(),'score_note':'Internal binary admission indicators, not confidence.'}
