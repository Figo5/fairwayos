"""Appearance-identity admission for seed-supported mask cores, causal by construction.

The caller supplies the immutable descriptor measured once at the operator-seed
frame and the descriptor of the current frame's compact core. Only the current
frame is compared against the original seed; no template refresh, no future
frames, no coordinates, no clip identity and no target point/score leave here.

Geometry, position and area stay outside this module. This module never claims
physical clubhead identity: metallic reflections, motion blur, shaft texture and
white heads make appearance ambiguous, so it only returns a binary admission
under fixed stroke-family thresholds.
"""
import cv2
import numpy as np

MIN_CORE_SAMPLES = 8
BRIGHT_VALUE_MIN = 190
LOW_SATURATION_MAX = 60
DARK_PIXEL_GRAY_MAX = 80
DARK_SEED_LAB_L_MAX = 130.0
# Below this texture level both cores are essentially flat and dispersion says nothing.
DISPERSION_TEXTURE_FLOOR = 1.0
DISPERSION_DENOMINATOR_FLOOR = 1.0

THRESHOLDS = {
    'putt': {'lab_median_distance_max': 45.0, 'dispersion_ratio_min': .25,
             'dispersion_ratio_max': 4.0},
    'full_swing': {'lab_median_distance_max': 60.0, 'dispersion_ratio_min': .25,
                   'dispersion_ratio_max': 4.0},
}
BALL_CONFUSION = {'seed_lab_l_max': DARK_SEED_LAB_L_MAX, 'candidate_lab_l_gain_min': 45.0,
                  'candidate_bright_low_sat_fraction_min': .5}
DESCRIPTOR_KEYS = ('sample_count', 'lab_median', 'gray_dispersion',
                   'bright_low_sat_fraction', 'dark_fraction')
SUBJECT = 'appearance of native core pixels only, not physical clubhead identity'


def describe_core(image_bgr, core_mask):
    """Measure the immutable descriptor of one core mask over native BGR pixels."""
    if not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8:
        raise ValueError('image_bgr must be a uint8 numpy array')
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError('image_bgr must have shape (height,width,3)')
    if not isinstance(core_mask, np.ndarray) or core_mask.dtype != np.bool_:
        raise ValueError('core_mask must be a boolean numpy array')
    if core_mask.ndim != 2 or core_mask.shape != image_bgr.shape[:2]:
        raise ValueError('core_mask must be 2-D and match the image height/width')
    count = int(core_mask.sum())
    if count == 0:
        raise ValueError('core_mask has no pixels to describe')
    if count < MIN_CORE_SAMPLES:
        raise ValueError(f'core_mask has {count} pixels, below the {MIN_CORE_SAMPLES} minimum')
    pixels = image_bgr[core_mask].reshape(-1, 1, 3)
    lab = cv2.cvtColor(pixels, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY).reshape(-1).astype(np.float64)
    hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
    median_lab = np.median(lab, axis=0)
    dispersion = 1.4826 * float(np.median(np.abs(gray - np.median(gray))))
    descriptor = {
        'sample_count': count,
        'lab_median': [float(v) for v in median_lab],
        'gray_dispersion': float(dispersion),
        'bright_low_sat_fraction': float(np.mean((hsv[:, 2] >= BRIGHT_VALUE_MIN) &
                                                 (hsv[:, 1] <= LOW_SATURATION_MAX))),
        'dark_fraction': float(np.mean(gray <= DARK_PIXEL_GRAY_MAX)),
        'core_pixel_source': 'native_mask_core',
        'subject': SUBJECT,
    }
    if not all(np.isfinite(v) for v in descriptor['lab_median'] + [descriptor['gray_dispersion'],
                                                                  descriptor['bright_low_sat_fraction'],
                                                                  descriptor['dark_fraction']]):
        raise ValueError('core descriptor produced non-finite features')
    return descriptor


def _checked_descriptor(descriptor, name):
    if not isinstance(descriptor, dict):
        raise ValueError(f'{name} must be a descriptor dict')
    missing = [key for key in DESCRIPTOR_KEYS if key not in descriptor]
    if missing:
        raise ValueError(f'{name} is missing descriptor keys {missing}')
    count = descriptor['sample_count']
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError(f'{name} sample_count must be a non-negative integer')
    lab = descriptor['lab_median']
    values = list(lab) + [descriptor['gray_dispersion'], descriptor['bright_low_sat_fraction'],
                          descriptor['dark_fraction']]
    if len(lab) != 3 or not all(isinstance(v, (int, float, np.number)) and
                                not isinstance(v, bool) and np.isfinite(v) for v in values):
        raise ValueError(f'{name} features must be finite numbers with a 3-element lab_median')
    if descriptor['gray_dispersion'] < 0:
        raise ValueError(f'{name} gray_dispersion must be non-negative')
    for key in ('bright_low_sat_fraction', 'dark_fraction'):
        if not 0 <= descriptor[key] <= 1:
            raise ValueError(f'{name} {key} must be a fraction in [0,1]')
    return descriptor


def compare_core_identity(seed_descriptor, current_descriptor, stroke_type):
    """Admit or refuse one current core against the immutable original seed descriptor."""
    if stroke_type not in THRESHOLDS:
        raise ValueError(f'stroke_type must be one of {sorted(THRESHOLDS)}')
    seed = _checked_descriptor(seed_descriptor, 'seed_descriptor')
    current = _checked_descriptor(current_descriptor, 'current_descriptor')
    bounds = THRESHOLDS[stroke_type]
    refusal = []
    if seed['sample_count'] < MIN_CORE_SAMPLES:
        refusal.append('seed_core_sample_count_below_minimum')
    if current['sample_count'] < MIN_CORE_SAMPLES:
        refusal.append('current_core_sample_count_below_minimum')
    seed_lab, current_lab = np.asarray(seed['lab_median']), np.asarray(current['lab_median'])
    delta = current_lab - seed_lab
    distance = float(np.linalg.norm(delta))
    ratio = float(current['gray_dispersion'] / max(seed['gray_dispersion'],
                                                  DISPERSION_DENOMINATOR_FLOOR))
    textured = max(seed['gray_dispersion'], current['gray_dispersion']) >= DISPERSION_TEXTURE_FLOOR
    bright_delta = float(current['bright_low_sat_fraction'] - seed['bright_low_sat_fraction'])
    dark_delta = float(current['dark_fraction'] - seed['dark_fraction'])
    differences = {'lab_median_delta': [float(v) for v in delta],
                   'lab_median_euclidean': distance,
                   'gray_dispersion_ratio': ratio,
                   'gray_dispersion_compared': textured,
                   'bright_low_sat_fraction_delta': bright_delta,
                   'dark_fraction_delta': dark_delta}
    if distance > bounds['lab_median_distance_max']:
        refusal.append('lab_median_distance_exceeds_stroke_bound')
    if textured and not bounds['dispersion_ratio_min'] <= ratio <= bounds['dispersion_ratio_max']:
        refusal.append('gray_dispersion_ratio_out_of_stroke_bounds')
    if (seed_lab[0] <= BALL_CONFUSION['seed_lab_l_max'] and
            delta[0] >= BALL_CONFUSION['candidate_lab_l_gain_min'] and
            current['bright_low_sat_fraction'] >= BALL_CONFUSION['candidate_bright_low_sat_fraction_min']):
        refusal.append('bright_low_saturation_candidate_against_dark_seed')
    if not all(np.isfinite(v) for v in differences['lab_median_delta'] +
               [distance, ratio, bright_delta, dark_delta]):
        raise ValueError('identity comparison produced non-finite differences')
    return {'accepted': not refusal, 'stroke_type': stroke_type,
            'feature_differences': differences,
            'thresholds': dict(bounds, **BALL_CONFUSION),
            'refusal_reasons': refusal,
            'decision_basis': 'fixed stroke-family appearance thresholds on native core pixels',
            'physical_identity_established': False,
            'subject': SUBJECT}
