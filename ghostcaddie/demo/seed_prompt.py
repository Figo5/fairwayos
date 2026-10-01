"""Derive negative prompt points that separate a physical clubhead from a neighbouring
round white golf ball inside the ORIGINAL seeded frame.

This module reads exactly one already-loaded frame array plus the operator's positive
seed point.  It never reads a later frame, a reference label, a per-clip condition, a
configured ball seed, a model answer or an LLM judgement.  It emits binary pixel
measurements, the actual thresholds used and the code profile -- never a confidence.

Coordinates are absolute native pixels *of the array that was passed in*.  A caller
working on a crop passes the crop and translates the seed itself: nothing here applies
or removes an origin offset.
"""
import math
import numbers
from dataclasses import dataclass

PROFILE = {
    # positive prompt: seed +/- half extent, clamped to the image
    'positive_box_half_px': 7,
    # placement window around the positive seed
    'search_radius_px': 26.0,
    'min_negative_gap_px': 4.0,
    'max_negative_points': 1,
    'max_recorded_rejections': 12,
    # ball size/shape gates (6..12 px diameter on 720p/1080p broadcast)
    'min_ball_radius_px': 2.5,
    'max_ball_radius_px': 7.5,
    'half_extent_tolerance_px': 0.5,
    'min_area_factor': 0.60,
    'max_area_factor': 1.40,
    'max_axis_ratio': 1.60,
    'min_solidity': 0.78,
    'min_circularity': 0.70,
    # colour gates, evaluated against the local turf context, not absolute white
    'max_saturation': 60,
    'min_value': 110,
    'min_value_above_context': 25,
    'context_ring_inner_px': 8.0,
    'context_ring_outer_px': 40.0,
    'min_context_px': 40,
    'profile_kind': 'fixed global thresholds declared here, no per-clip adaptation',
    'score_kind': 'binary measurements only; no confidence value is produced',
    'research_only': True,
    'production_eligible': False,
}


@dataclass(frozen=True)
class HeadPrompts:
    """Positive seed point, derived seed box, derived negative ball centres, metadata."""
    positive_point: tuple
    positive_box: tuple
    negative_points: tuple
    metadata: dict

    def to_dict(self):
        return {'positive_point': list(self.positive_point),
                'positive_box': list(self.positive_box),
                'negative_points': [list(point) for point in self.negative_points],
                'metadata': dict(self.metadata)}


def derive_head_prompts(image_bgr, seed_x, seed_y):
    import cv2
    import numpy as np

    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('image_bgr must be an HxWx3 uint8 BGR frame of native pixels')
    height, width = int(image.shape[0]), int(image.shape[1])
    if min(height, width) < 2 * PROFILE['positive_box_half_px'] + 2:
        raise ValueError('image_bgr is smaller than the declared positive box')
    x = _native_coordinate(seed_x, 'seed_x')
    y = _native_coordinate(seed_y, 'seed_y')
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f'seed ({x},{y}) lies outside the {width}x{height} image')

    box = _clamped_box(x, y, width, height)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    context_value, context_reason, context_px = _local_context_value(value, x, y)
    white = ((saturation <= PROFILE['max_saturation']) &
             (value >= PROFILE['min_value']) &
             ((value.astype(np.int16) - int(context_value)) >= PROFILE['min_value_above_context']))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        white.astype(np.uint8), 8)
    # Raster lookup is rounded and clamped; the original subpixel placement is
    # retained in the positive prompt, distance gates and metadata.
    seed_label = int(labels[min(height-1,round(y)), min(width-1,round(x))])
    admitted, rejected, seed_region_round_white = [], [], False
    for index in range(1, count):
        measured = _measure_region(index, labels, stats, centroids, saturation, value,
                                   (x, y), context_value)
        if index == seed_label:
            measured['contains_positive_seed'] = True
            seed_region_round_white = not measured['failed_shape_gates']
            rejected.append(measured)
            continue
        (rejected if measured['failed'] else admitted).append(measured)
    admitted.sort(key=lambda region: (region['seed_distance_px'],
                                      region['center_native_xy'][1],
                                      region['center_native_xy'][0]))

    failure_reason = None
    if seed_region_round_white and admitted:
        failure_reason = 'positive_seed_inside_round_white_region_prompts_ambiguous'
    elif len(admitted) > PROFILE['max_negative_points']:
        failure_reason = 'more_admissible_candidates_than_max_negative_points'
    negatives = (() if failure_reason else
                 tuple(tuple(region['center_native_xy'])
                       for region in admitted[:PROFILE['max_negative_points']]))
    reasons = [failure_reason] if failure_reason else []
    reasons.append('no_admissible_round_white_candidate' if not admitted
                   else 'round_white_candidate_admitted')
    metadata = {
        'frames_read': 1,
        'frame_source': 'the single seeded frame array passed to this call',
        'future_frame_access': False,
        'image_shape_hw': [height, width],
        'seed_point_native_xy': [x, y],
        'positive_point_native_xy': [x, y],
        'positive_box_native_xywh': list(box),
        'positive_box_half_px': PROFILE['positive_box_half_px'],
        'negative_points_native_xy': [list(point) for point in negatives],
        'context_value': round(context_value, 3),
        'context_source': context_reason,
        'context_pixels': context_px,
        'thresholds': _declared_thresholds(),
        'code_profile': dict(PROFILE),
        'score_kind': PROFILE['score_kind'],
        'confidence': None,
        'candidate_policy': ('exactly one admissible round white region is required; '
                             'a seed sitting in a round white region or a second '
                             'admissible region fails the frame closed'),
        'admitted_regions': admitted,
        'rejected_region_count': len(rejected),
        'rejected_regions': rejected[:PROFILE['max_recorded_rejections']],
        'reasons': reasons,
        'failure_closed': bool(failure_reason),
        'failure_reason': failure_reason,
        'claim': ('pixel-measurement negative prompts for round white blobs; no '
                  'clubhead-semantics claim and no ball ground truth'),
    }
    return HeadPrompts((x, y), box, negatives, metadata)


def _measure_region(index, labels, stats, centroids, saturation, value, seed_xy, context_value):
    import cv2
    import numpy as np

    left = int(stats[index, cv2.CC_STAT_LEFT])
    top = int(stats[index, cv2.CC_STAT_TOP])
    region_w = int(stats[index, cv2.CC_STAT_WIDTH])
    region_h = int(stats[index, cv2.CC_STAT_HEIGHT])
    area = int(stats[index, cv2.CC_STAT_AREA])
    component = labels == index
    rows, columns = np.nonzero(component)
    points = np.column_stack((columns, rows)).astype(np.float32)
    _, (rect_w, rect_h), _ = cv2.minAreaRect(points)
    half_extent = max(rect_w, rect_h) / 2.0
    axis_ratio = max(rect_w, rect_h) / max(min(rect_w, rect_h), 0.5)
    patch = component[top:top + region_h, left:left + region_w].astype(np.uint8)
    contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    circularity = (4.0 * math.pi * area / perimeter ** 2) if perimeter > 0 else 0.0
    solidity = (area / hull_area) if hull_area > 0 else 0.0
    center = (int(round(float(centroids[index][0]))), int(round(float(centroids[index][1]))))
    distance = math.hypot(center[0] - seed_xy[0], center[1] - seed_xy[1])
    mean_saturation = float(saturation[rows, columns].mean())
    mean_value = float(value[rows, columns].mean())
    tolerance = PROFILE['half_extent_tolerance_px']
    min_area = PROFILE['min_area_factor'] * math.pi * PROFILE['min_ball_radius_px'] ** 2
    max_area = PROFILE['max_area_factor'] * math.pi * PROFILE['max_ball_radius_px'] ** 2
    shape_gates = [
        (area < min_area, 'area_below_min'),
        (area > max_area, 'area_above_max'),
        (half_extent < PROFILE['min_ball_radius_px'] - tolerance,
         'half_extent_below_min_ball_radius'),
        (half_extent > PROFILE['max_ball_radius_px'] + tolerance,
         'half_extent_above_max_ball_radius'),
        (axis_ratio > PROFILE['max_axis_ratio'], 'axis_ratio_above_max'),
        (solidity < PROFILE['min_solidity'], 'solidity_below_min'),
        (circularity < PROFILE['min_circularity'], 'circularity_below_min'),
        (mean_saturation > PROFILE['max_saturation'], 'mean_saturation_above_max'),
        (mean_value - context_value < PROFILE['min_value_above_context'],
         'brightness_not_above_local_context'),
    ]
    placement_gates = [
        (distance < PROFILE['min_negative_gap_px'], 'distance_below_min_negative_gap'),
        (distance > PROFILE['search_radius_px'], 'distance_above_search_radius'),
    ]
    # shape/colour gates ask whether the region is a round white blob at all; the
    # placement gates ask whether it may serve as a negative prompt for this seed.
    shape_failed = [name for condition, name in shape_gates if condition]
    placement_failed = [name for condition, name in placement_gates if condition]
    failed = shape_failed + placement_failed
    return {'center_native_xy': list(center), 'area_px': area,
            'bbox_native_xywh': [left, top, region_w, region_h],
            'half_extent_px': round(half_extent, 3),
            'equivalent_radius_px': round(math.sqrt(area / math.pi), 3),
            'axis_ratio': round(axis_ratio, 3), 'solidity': round(solidity, 3),
            'circularity': round(circularity, 3),
            'mean_saturation': round(mean_saturation, 3),
            'mean_value': round(mean_value, 3),
            'context_value': round(context_value, 3),
            'seed_distance_px': round(distance, 3),
            'contains_positive_seed': False,
            'failed_gates': failed, 'failed_shape_gates': shape_failed,
            'failed_placement_gates': placement_failed, 'failed': bool(failed)}


def _local_context_value(value, seed_x, seed_y):
    import numpy as np

    height, width = int(value.shape[0]), int(value.shape[1])
    rows, columns = np.ogrid[0:height, 0:width]
    distance = np.hypot(columns - seed_x, rows - seed_y)
    ring = ((distance >= PROFILE['context_ring_inner_px']) &
            (distance <= PROFILE['context_ring_outer_px']))
    pixels = value[ring]
    if pixels.size >= PROFILE['min_context_px']:
        return float(np.median(pixels)), 'local_ring_median_value', int(pixels.size)
    return float(np.median(value)), 'global_median_value_ring_too_small', int(pixels.size)


def _declared_thresholds():
    return {name: PROFILE[name] for name in sorted(PROFILE) if isinstance(PROFILE[name], (int, float))
            and not isinstance(PROFILE[name], bool)}


def _clamped_box(seed_x, seed_y, width, height):
    half = PROFILE['positive_box_half_px']
    left = max(0, seed_x - half)
    top = max(0, seed_y - half)
    right = min(width - 1, seed_x + half)
    bottom = min(height - 1, seed_y + half)
    return (left, top, right - left + 1, bottom - top + 1)


def _native_coordinate(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f'{name} must be a finite native pixel coordinate, got {value!r}')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be a finite native pixel coordinate, got {value!r}')
    return result
