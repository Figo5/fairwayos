"""Frame-bound evaluation of visible objects; independent labels are required.

Native coordinates only. Detection availability is separate from correctness.
Wrong-object locks require explicit confuser regions, not just a large error.
"""
from __future__ import annotations
import math
import re
from statistics import median

OBJECTS = ('ball', 'clubhead', 'body', 'cup')
STATES = {'visible', 'absent', 'ambiguous', 'unreviewed'}


def _number(x):
    return type(x) in (int, float) and math.isfinite(x)


def _point(p, width, height):
    return (isinstance(p, list) and len(p) == 2 and all(_number(x) for x in p)
            and 0 <= p[0] < width and 0 <= p[1] < height)


def _box(b, width, height):
    return (isinstance(b, list) and len(b) == 4 and all(_number(x) for x in b)
            and 0 <= b[0] < b[2] <= width and 0 <= b[1] < b[3] <= height)


def validate_labels(labels, source_sha256, native_pts):
    if labels.get('schema_version') != 'fairwayos-four-object-labels.v1':
        raise ValueError('unsupported labels schema')
    if labels.get('source_sha256') != source_sha256 or not re.fullmatch(r'[0-9a-f]{64}', source_sha256):
        raise ValueError('labels belong to a different source')
    if not native_pts or any(not _number(t) for t in native_pts) or any(b <= a for a, b in zip(native_pts, native_pts[1:])):
        raise ValueError('invalid native frame timeline')
    width, height = labels['width'], labels['height']
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError('invalid native dimensions')
    seen = set()
    for frame in labels['frames']:
        n = frame['frame']
        if type(n) is not int or n in seen or not 0 <= n < len(native_pts):
            raise ValueError('duplicate or out-of-range frame')
        seen.add(n)
        t = frame['timestamp_seconds']
        if not _number(t) or abs(t - native_pts[n]) > 1e-6:
            raise ValueError('label timestamp does not match native PTS')
        if set(frame['objects']) != set(OBJECTS):
            raise ValueError('each frame must declare all four visibility states')
        for name, obj in frame['objects'].items():
            if obj.get('visibility') not in STATES:
                raise ValueError('invalid visibility')
            state = obj['visibility']
            if state in {'visible', 'absent'}:
                if obj.get('label_source') not in {'human_ground_truth', 'human_confirmed'}:
                    raise ValueError('independent scoring requires human reference labels')
            if state == 'visible':
                if name == 'body':
                    if not _box(obj.get('bbox'), width, height):
                        raise ValueError('visible body requires target-person bounding box')
                    if not obj.get('anchors') or not all(_point(p, width, height) for p in obj['anchors'].values()):
                        raise ValueError('visible body requires named native anchors')
                elif not _point(obj.get('point'), width, height):
                    raise ValueError('visible point object requires native center')
                if not _number(obj.get('tolerance_px')) or obj['tolerance_px'] <= 0:
                    raise ValueError('matching tolerance must be declared before evaluation')
            elif any(obj.get(k) is not None for k in ('point', 'bbox', 'anchors')):
                raise ValueError('non-visible states cannot assert a target position')
            if not all(_box(b, width, height) for b in obj.get('confusers', [])):
                raise ValueError('invalid confuser region')
    return labels


def _iou(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union > 0 else 0


def _distribution(values):
    ordered = sorted(values)
    rank = (len(values)-1)*.9 if values else 0
    low, high = math.floor(rank), math.ceil(rank)
    p90 = ordered[low] + (ordered[high]-ordered[low])*(rank-low) if values else None
    return {'samples': len(values), 'median_px': median(values) if values else None,
            'max_px': max(values) if values else None, 'p90_px': p90}


def evaluate_four_objects(labels, observations, *, source_sha256, native_pts, body_iou_min=.5):
    validate_labels(labels, source_sha256, native_pts)
    if not _number(body_iou_min) or not 0 < body_iou_min <= 1:
        raise ValueError('invalid body IoU threshold')
    predicted = {}
    for row in observations:
        n = row['frame']
        if type(n) is not int or n in predicted or not 0 <= n < len(native_pts):
            raise ValueError('duplicate or out-of-range observation frame')
        if not _number(row.get('timestamp_seconds')) or abs(row['timestamp_seconds']-native_pts[n]) > 1e-6:
            raise ValueError('observation PTS mismatch')
        predicted[n] = row
    results = {}
    for name in OBJECTS:
        counts = dict(visible=0, absent=0, matched=0, misses=0, localization_errors=0,
                      wrong_locks=0, false_on_absent=0, withheld_ambiguous=0, unreviewed=0,
                      predicted_on_scored=0)
        errors, anchors = [], {}
        for frame in labels['frames']:
            obj = frame['objects'][name];state = obj['visibility'];row = predicted.get(frame['frame'], {})
            pred = row.get('target' if name == 'cup' else name)
            # Withheld/inferred predictions cannot earn measurement credit.
            if not isinstance(pred, dict) or pred.get('state') != 'observed' or pred.get('provenance') != 'current_pixels':
                pred = None
            if pred is not None:
                if name == 'body':
                    if not isinstance(pred, dict) or not _box(pred.get('bbox'), labels['width'], labels['height']):
                        raise ValueError('invalid predicted body')
                    point = [(pred['bbox'][0]+pred['bbox'][2])/2, (pred['bbox'][1]+pred['bbox'][3])/2]
                else:
                    point = pred.get('point') if isinstance(pred, dict) else pred
                    if not _point(point, labels['width'], labels['height']):
                        raise ValueError('invalid predicted native center')
            if state == 'unreviewed':counts['unreviewed'] += 1;continue
            if state == 'ambiguous':counts['withheld_ambiguous'] += 1;continue
            counts[state] += 1
            if pred is not None:counts['predicted_on_scored'] += 1
            if state == 'absent':
                if pred is not None:counts['false_on_absent'] += 1
                continue
            if name == 'body':
                for anchor in obj['anchors']:
                    stats = anchors.setdefault(anchor, {'visible':0, 'observed':0, 'within_tolerance':0, 'target_person_within_tolerance':0, 'errors':[]})
                    stats['visible'] += 1
            if pred is None:counts['misses'] += 1;continue
            if name == 'body':
                matched = _iou(pred['bbox'], obj['bbox']) >= body_iou_min
                for anchor, truth in obj['anchors'].items():
                    observed = pred.get('keypoints', {}).get(anchor)
                    stats = anchors.setdefault(anchor, {'visible':0, 'observed':0, 'within_tolerance':0, 'target_person_within_tolerance':0, 'errors':[]})
                    if observed:
                        p = observed['point']
                        if not _point(p, labels['width'], labels['height']):raise ValueError('invalid anchor')
                        error = math.dist(p, truth);stats['observed'] += 1;stats['errors'].append(error)
                        stats['within_tolerance'] += int(error <= obj['tolerance_px'])
                        stats['target_person_within_tolerance'] += int(matched and error <= obj['tolerance_px'])
            else:
                error = math.dist(point, obj['point']);errors.append(error);matched = error <= obj['tolerance_px']
            if matched:counts['matched'] += 1
            else:
                if any(b[0] <= point[0] <= b[2] and b[1] <= point[1] <= b[3] for b in obj.get('confusers', [])):
                    counts['wrong_locks'] += 1
                else:counts['localization_errors'] += 1
        for stats in anchors.values():stats['center_error'] = _distribution(stats.pop('errors'))
        scored = counts['visible']+counts['absent']
        results[name] = {'counts':counts, 'scored_frames':scored,
            'precision':counts['matched']/counts['predicted_on_scored'] if counts['predicted_on_scored'] else None,
            'visible_recall':counts['matched']/counts['visible'] if counts['visible'] else None,
            'false_positive_rate_on_absent':counts['false_on_absent']/counts['absent'] if counts['absent'] else None,
            'wrong_lock_rate_on_visible':counts['wrong_locks']/counts['visible'] if counts['visible'] else None,
            'center_error_all_visible_predictions':_distribution(errors), 'body_anchors':anchors}
    return {'source_sha256':source_sha256, 'independent_scored_frames':len({f['frame'] for f in labels['frames'] if any(o['visibility'] in {'visible','absent'} for o in f['objects'].values())}),
            'metrics_available':any(v['scored_frames'] for v in results.values()), 'objects':results,
            'body_identity_metric':'target-person box IoU; named anchor error reported separately',
            'body_iou_min':body_iou_min, 'wrong_lock_metric':'explicit annotated confuser regions only',
            'physical_mph_verified':False}


def validate_source_groups(clips, *, consumed_groups=()):
    """Prevent sibling/session and duplicate-byte leakage before training."""
    groups, digests, ids = {}, {}, set()
    for clip in clips:
        ident, group, digest, split = (clip[k] for k in ('id', 'source_group', 'source_sha256', 'split'))
        if ident in ids or not all(isinstance(v, str) and v for v in (ident, group, digest, split)):
            raise ValueError('invalid or duplicate source identity')
        ids.add(ident)
        if not re.fullmatch(r'[0-9a-f]{64}', digest):raise ValueError('invalid source hash')
        if groups.setdefault(group, split) != split or digests.setdefault(digest, split) != split:
            raise ValueError('source session or identical bytes cross partitions')
        if split in {'test', 'final_test', 'holdout'} and group in consumed_groups:
            raise ValueError('consumed source group cannot be independent holdout')
    return {'groups':groups, 'source_count':len(clips), 'group_isolation_verified':True}
