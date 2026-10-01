"""Research head-region measurements from a learned object mask, in native pixels.

The mask model identifies a prompted region, not a clubhead class. Splitting thick
regions prevents a connected shaft from pulling the measured head point upward.
No reference labels, image coordinates from later frames, or predictions enter here.
"""
import math

PROFILE = {
    'core_min_distance_px': 1.8, 'core_peak_fraction': .75,
    'core_min_area_px': 3, 'core_max_area_px': 180,
    'core_max_dimension_px': 24, 'seed_max_distance_px': 3,
    'seed_min_elongation': 1.25, 'max_seed_area_ratio': 3,
    'candidate_min_elongation': 1.25,
    'min_seed_area_ratio': 1/3, 'connectivity': 8,
    'distance_transform_mask_size': 3,
    'score_kind': 'binary region-admission indicator, not model confidence',
    'research_only': True, 'production_eligible': False,
}


def compact_regions(mask):
    import cv2
    import numpy as np
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    threshold = max(PROFILE['core_min_distance_px'],
                    PROFILE['core_peak_fraction'] * float(distance.max()))
    core = (distance >= threshold).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(core, 8)
    regions = []
    for index in range(1, count):
        x,y,w,h,area = map(int,stats[index])
        if not PROFILE['core_min_area_px'] <= area <= PROFILE['core_max_area_px']:
            continue
        if max(w,h) > PROFILE['core_max_dimension_px']:
            continue
        pixels = np.argwhere(labels == index)[:,::-1].astype(np.float32)
        axes = cv2.minAreaRect(pixels)[1]
        elongation = max(axes)/max(min(axes), .5)
        regions.append({'xy': centroids[index].tolist(), 'area':area,
                        'bbox':[x,y,w,h], 'elongation':float(elongation),
                        'core_threshold_px':threshold})
    return regions


def select_seed(regions, point):
    close = [region for region in regions
             if math.dist(region['xy'],point) <= PROFILE['seed_max_distance_px']
             and region['elongation'] >= PROFILE['seed_min_elongation']]
    return close[0] if len(close) == 1 else None


def admitted_regions(regions, seed, origin, anchor, radius):
    if seed is None:
        return []
    admitted = []
    for region in regions:
        point = [region['xy'][0]+origin[0], region['xy'][1]+origin[1]]
        ratio = region['area']/seed['area']
        if (region['elongation'] >= PROFILE['candidate_min_elongation'] and
                math.dist(point,anchor) <= radius and
                PROFILE['min_seed_area_ratio'] <= ratio <= PROFILE['max_seed_area_ratio']):
            admitted.append({**region,'native_xy':point})
    return admitted


POINT_POLICY = {
    'loss_frames':3, 'search_radius_px':24, 'reacq_radius_px':65,
    'max_step_px':30, 'reacq_max_step_px':65, 'min_score':.55,
    'seed_identity_min_score':.45, 'reacq_min_score':.70,
    'ambiguity_margin':.06, 'reacq_confirm_frames':2, 'reacq_window_frames':24,
}
PROFILE['point_policy'] = POINT_POLICY
PROFILE['inactive_probability_thresholds'] = [
    'min_score','seed_identity_min_score','reacq_min_score','ambiguity_margin']
PROFILE['seed_mask_source'] = 'first forward video mask at the original seed frame'
PROFILE['exported_confidence'] = None


class MaskHeadPolicy:
    """Causal geometry policy shared by inference and verified mask replay.

The immediate prompt preview is not used for seed admission: SAM2's preflight
may refine it. The first forward mask is still the original seeded frame, without
any future frame or correction. Internal ones are boolean admission indicators;
exported points carry no confidence value.
    """
    def __init__(self, seed, origin, cut_frames, *, point_policy=None):
        from .foreground_tracker import ForegroundPolicy
        self.seed = dict(seed)
        self.origin = list(origin)
        self.cut_frames = set(cut_frames)
        self.policy_settings = dict(POINT_POLICY)
        if point_policy is not None:
            unknown = set(point_policy)-set(POINT_POLICY)
            if unknown:
                raise ValueError('Unknown point-policy settings: '+str(sorted(unknown)))
            self.policy_settings.update(point_policy)
        self.policy = ForegroundPolicy(seed_frame=seed['frame'],seed_x=seed['x'],
                                       seed_y=seed['y'],**self.policy_settings)
        self.policy.seed_decision()
        self.seed_region = None
        self.seed_valid = None
        self.last_frame = seed['frame']-1

    def step(self, frame, mask):
        from .foreground_tracker import CandidateMatch
        if frame != self.last_frame+1:
            raise ValueError('Masks must arrive once in contiguous forward order')
        self.last_frame = frame
        regions = compact_regions(mask)
        if frame == self.seed['frame']:
            self.seed_region = select_seed(regions,[self.seed['x']-self.origin[0],
                                                    self.seed['y']-self.origin[1]])
            self.seed_valid = self.seed_region is not None
            if not self.seed_valid:
                self.policy.measure = False
                self.policy.refusal_reason = 'initial forward mask has no unique nonround head region at the seed'
            decision = self.policy.decisions[0]
            admitted = []
        else:
            request = self.policy.request_for(frame)
            admitted = admitted_regions(regions,self.seed_region,self.origin,
                [request.anchor_x,request.anchor_y],request.radius_px) if request else []
            candidate = None
            if len(admitted) == 1:
                p = admitted[0]['native_xy']
                candidate = CandidateMatch(frame,True,x=p[0],y=p[1],score=1,
                    seed_score=1,second_score=0,transform='learned mask geometry gate')
            decision = self.policy.step(frame,candidate,cut=frame in self.cut_frames)
        point = decision.to_point()
        if point['origin'] == 'automatic':
            point['method'] = ('sam2_head_region_reacquired' if decision.method.endswith('reacquired')
                               else 'sam2_head_region')
            point['score'] = None
        return point, {'frame':frame,'mask_area':int(mask.sum()),'parts':regions,
            'admitted':admitted,'decision':decision.to_raw(),
            'score_note':'Internal binary admission indicators are not confidence; exported confidence is null.'}
