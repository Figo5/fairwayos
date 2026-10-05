"""Current-frame context refusal for swing masks; no coordinate generation.

Developed after visual inspection of v2: small mask cores can be grass, shaft,
or hands despite favorable sparse scores. This policy requires visible current
MoveNet wrists, separation from the grip, a compact core, and local contrast.
The pose crop is operator supplied and may use interpolated crop keyframes;
only the current frame's measured wrist coordinates enter the admission gate.
Neither pose nor mask appearance establishes physical clubhead identity.
"""
import copy
import math
from .swing_head import SwingHeadPolicy, PROFILE as HEAD_PROFILE, _finite_number

MIN_LOCAL_CONTRAST = 15.0
MIN_BACKGROUND_SAMPLES = 16
MAX_CORE_ELONGATION = 4.0
WRIST_SCORE_MIN = .3
WRIST_SEPARATION_AT_720 = 35.0
PROFILE = {**copy.deepcopy(HEAD_PROFILE), 'version': 'swing_context_v3',
           'context_required': True, 'min_local_contrast_gray': MIN_LOCAL_CONTRAST,
           'contrast_ring_margin_px': 6, 'min_background_samples': MIN_BACKGROUND_SAMPLES,
           'max_core_elongation': MAX_CORE_ELONGATION, 'wrist_score_min': WRIST_SCORE_MIN,
           'wrist_separation_at_720_px': WRIST_SEPARATION_AT_720,
           'context': 'Current MoveNet wrists on operator body ROI; no future pose or target coordinate.',
           'development_after_v2_visual_inspection': True}


class ContextSwingHeadPolicy(SwingHeadPolicy):
    def step(self, frame, mask, image, *, body_frame=None):
        if body_frame is not None and (not isinstance(body_frame, dict) or
                type(body_frame.get('source_frame')) is not int or body_frame['source_frame'] != frame):
            raise ValueError('Body context must belong to exactly the current source frame')
        self.current_wrists = []
        if body_frame and body_frame.get('available') is True:
            for joint in body_frame.get('keypoints', []):
                if joint.get('name') not in ('wr_l', 'wr_r'):
                    continue
                if not all(_finite_number(joint.get(k)) for k in ('x', 'y', 'score')):
                    raise ValueError('Wrist evidence must be finite native measurements')
                if not 0 <= joint['score'] <= 1:
                    raise ValueError('Wrist score must be in [0,1]')
                if joint.get('visible') is True and joint['score'] >= WRIST_SCORE_MIN:
                    self.current_wrists.append({k: joint[k] for k in ('name', 'x', 'y', 'score')})
        point, raw = super().step(frame, mask, image)
        raw['current_wrist_evidence'] = copy.deepcopy(self.current_wrists)
        raw['context_profile'] = PROFILE
        return point, raw

    def _extra_refusals(self, part, image):
        reasons = []
        contrast = part['local_contrast']
        if (contrast['background_samples'] < MIN_BACKGROUND_SAMPLES or
                contrast['absolute_median_gray_difference'] is None or
                contrast['absolute_median_gray_difference'] < MIN_LOCAL_CONTRAST):
            reasons.append('insufficient_current_local_contrast')
        if part['elongation'] > MAX_CORE_ELONGATION:
            reasons.append('thin_core_ambiguous_with_shaft')
        native_xy = [part['xy'][i] + self.origin[i] for i in range(2)]
        radius = WRIST_SEPARATION_AT_720 * image.shape[0] / 720
        distance = min((math.dist(native_xy, [w['x'], w['y']]) for w in self.current_wrists), default=None)
        part['wrist_context'] = {'nearest_visible_wrist_distance_px': distance,
                                 'minimum_separation_px': radius}
        if distance is None:
            reasons.append('no_visible_current_wrist_evidence')
        elif distance <= radius:
            reasons.append('candidate_near_current_grip')
        return reasons
