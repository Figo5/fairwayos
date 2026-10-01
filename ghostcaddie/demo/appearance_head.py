"""Causal mask-core appearance experiment. No corrections or reference coordinates."""
import math
import copy
from .adaptive_head import PROFILE as GEOMETRY,seed_threshold
from .segmented_head import POINT_POLICY
from .foreground_tracker import ForegroundPolicy,CandidateMatch
from .head_identity import describe_core,compare_core_identity,MIN_CORE_SAMPLES,THRESHOLDS,BALL_CONFUSION

PROFILE={
    'geometry':dict(GEOMETRY),'identity_thresholds':THRESHOLDS,
    'ball_confusion':BALL_CONFUSION,'min_core_samples':MIN_CORE_SAMPLES,
    'round_bright_fraction_refusal':.5,'min_nonround_elongation':1.25,
    'admission':'preserve nonround geometry; only when no geometry candidate exists use round-core original-seed appearance',
    'original_seed_descriptor_frozen':True,'template_refresh':False,
    'exported_confidence':None,'physical_identity_established':False,
    'research_only':True,'production_eligible':False,
}


def measured_cores(mask,image,threshold,seed_point=None):
    import cv2
    import numpy as np
    if mask.shape!=image.shape[:2]:raise ValueError('Image and mask dimensions differ')
    dt=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,3)
    _,labels,stats,centers=cv2.connectedComponentsWithStats((dt>=threshold).astype(np.uint8),8)
    result=[]
    for index in range(1,len(stats)):
        x,y,w,h,area=map(int,stats[index])
        if not GEOMETRY['core_min_area_px']<=area<=GEOMETRY['core_max_area_px']:continue
        if max(w,h)>GEOMETRY['core_max_dimension_px']:continue
        local=labels[y:y+h,x:x+w]==index
        pixels=np.argwhere(local)[:,::-1].astype(np.float32)+[x,y]
        axes=cv2.minAreaRect(pixels.astype(np.float32))[1]
        elongation=float(max(axes)/max(min(axes),.5))
        try:appearance=describe_core(image[y:y+h,x:x+w],local)
        except ValueError as exc:appearance={'unavailable':str(exc)}
        region={'xy':centers[index].tolist(),'area':area,'bbox':[x,y,w,h],
                'elongation':elongation,'appearance':appearance}
        if seed_point is not None:
            region['seed_support_distance_px']=float(np.linalg.norm(pixels-np.asarray(seed_point),axis=1).min())
        result.append(region)
    return result


def round_bright(region):
    appearance=region['appearance']
    return (region['elongation']<PROFILE['min_nonround_elongation'] and
            appearance.get('bright_low_sat_fraction',0)>=PROFILE['round_bright_fraction_refusal'])


class AppearanceHeadPolicy:
    def __init__(self,seed,origin,cut_frames,stroke_type,point_policy=None):
        if stroke_type not in THRESHOLDS:raise ValueError('Unsupported stroke family')
        self.seed=dict(seed);self.origin=list(origin);self.cuts=set(cut_frames);self.stroke_type=stroke_type
        if point_policy and set(point_policy)-set(POINT_POLICY):raise ValueError('Unknown point policy setting')
        self.policy_settings={**POINT_POLICY,**(point_policy or {})}
        self.policy=ForegroundPolicy(seed_frame=seed['frame'],seed_x=seed['x'],seed_y=seed['y'],**self.policy_settings)
        self.policy.seed_decision();self.last_frame=seed['frame']-1
        self.threshold=None;self.seed_region=None;self.seed_valid=None;self.seed_descriptor=None;self.seed_mode=None

    def step(self,frame,mask,image):
        if frame!=self.last_frame+1:raise ValueError('Frames must arrive once in contiguous forward order')
        self.last_frame=frame;admitted=[];checks=[]
        if frame==self.seed['frame']:
            point=[self.seed['x']-self.origin[0],self.seed['y']-self.origin[1]]
            self.threshold=seed_threshold(mask,point)
            parts=measured_cores(mask,image,self.threshold,point)
            supported=[p for p in parts if p['seed_support_distance_px']<=GEOMETRY['seed_support_distance_px']]
            geometric=[p for p in supported if p['elongation']>=GEOMETRY['min_elongation']]
            choices=geometric or [p for p in supported if 'unavailable' not in p['appearance'] and not round_bright(p)]
            self.seed_region=choices[0] if len(choices)==1 else None
            self.seed_valid=self.seed_region is not None
            if self.seed_valid:
                self.seed_mode='geometry' if geometric else 'appearance_round'
                if 'unavailable' not in self.seed_region['appearance']:
                    self.seed_descriptor=copy.deepcopy(self.seed_region['appearance'])
            else:
                self.policy.measure=False;self.policy.refusal_reason='no unique seed-supported core with usable non-ball appearance'
            decision=self.policy.decisions[0]
        else:
            parts=measured_cores(mask,image,self.threshold)
            request=self.policy.request_for(frame)
            if request and self.seed_region:
                geometric=[];round_matches=[]
                for p in parts:
                    xy=[p['xy'][0]+self.origin[0],p['xy'][1]+self.origin[1]]
                    ratio=p['area']/self.seed_region['area']
                    if (not GEOMETRY['min_seed_area_ratio']<=ratio<=GEOMETRY['max_seed_area_ratio'] or
                        math.dist(xy,[request.anchor_x,request.anchor_y])>request.radius_px):continue
                    if self.seed_mode=='geometry' and p['elongation']>=GEOMETRY['min_elongation']:
                        geometric.append({**p,'native_xy':xy,'admission_gate':'geometry'});continue
                    if 'unavailable' in p['appearance']:
                        checks.append({'region_xy':xy,'accepted':False,'reason':p['appearance']['unavailable']});continue
                    if self.seed_descriptor is None:continue
                    match=compare_core_identity(self.seed_descriptor,p['appearance'],self.stroke_type)
                    if round_bright(p):match={**match,'accepted':False,'refusal_reasons':match['refusal_reasons']+['round_bright_core_ambiguous_with_ball']}
                    checks.append({'region_xy':xy,**match})
                    if match['accepted']:round_matches.append({**p,'native_xy':xy,'admission_gate':'original_seed_appearance'})
                admitted=geometric or round_matches
            candidate=None
            if len(admitted)==1:
                xy=admitted[0]['native_xy'];candidate=CandidateMatch(frame,True,x=xy[0],y=xy[1],score=1,seed_score=1,second_score=0,transform=admitted[0]['admission_gate']+' binary mask-core gate')
            decision=self.policy.step(frame,candidate,cut=frame in self.cuts)
        point=decision.to_point()
        if point['origin']=='automatic':
            point['method']='sam2_head_region_reacquired' if decision.method.endswith('reacquired') else 'sam2_head_region'
            point['score']=None
        return point,{'frame':frame,'parts':parts,'identity_checks':checks,'admitted':admitted,
                      'mask_area':int(mask.sum()),'threshold_from_seed':self.threshold,
                      'decision':decision.to_raw(),'score_note':'Binary gate only; no model confidence or established physical identity.'}
