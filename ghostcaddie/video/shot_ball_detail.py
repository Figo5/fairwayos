"""Current native pixels propose small near-player balls; no operator input.

This supplements a detector, never treats a white component as a ball by itself.
The temporal gate publishes only actual component samples, including samples
confirmed retrospectively. It never fills a missing frame.
"""
import cv2
import numpy as np


def near_player_components(frame, body, components, green_fraction, *, scale=1., head=None):
    """Locate bounded native searches from a fresh analysis-coordinate pose."""
    if not body or body.get('provenance') != 'current_pixels':
        return []
    joints = body['keypoints']
    required = ('left_hip', 'right_hip', 'left_ankle', 'right_ankle')
    if any(k not in joints or joints[k]['confidence'] < .85 for k in required):
        return []
    if not any(joints.get(k, {}).get('confidence', 0) >= .75
               for k in ('left_wrist', 'right_wrist')):
        return []
    bbox = np.asarray(body['bbox'], float) / scale
    bh = bbox[3] - bbox[1]
    if bh < 80 or body.get('crowded_acquisition') or body.get('track_id') is None:
        return []
    ankles = [np.asarray(joints[k]['point'], float) / scale
              for k in ('left_ankle', 'right_ankle')]
    hip = np.mean([np.asarray(joints[k]['point'], float) / scale
                   for k in ('left_hip', 'right_hip')], axis=0)
    height, width = frame.shape[:2]
    x0, x1 = max(0, int(bbox[0] - bh * .35)), min(width, int(bbox[2] + bh * .35))
    y0 = max(0, int(max(hip[1], min(a[1] for a in ankles) - bh * .12)))
    y1 = min(height, int(bbox[3] + bh * .15))
    roi = frame[y0:y1, x0:x1]
    if not roi.size:
        return []
    candidates = []
    for component in components(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)):
        point = np.asarray(component['point'], float) + [x0, y0]
        w, h = component['size']
        diameter = max(w, h)
        # A tiny lobe needs enough native pixels to distinguish it from a glyph.
        if not (3 <= min(w, h) and diameter <= bh * .06 and .7 <= w / h <= 1.5):
            continue
        if any(np.linalg.norm(point - ankle) <= bh * .085 for ankle in ankles):
            continue
        if any(np.linalg.norm(point - np.asarray(joints[k]['point']) / scale) <= bh * .075
               for k in ('left_wrist', 'right_wrist') if k in joints):
            continue
        if head and np.linalg.norm(point - np.asarray(head['point']) / scale) <= max(4, diameter * .6):
            continue
        x, y = point
        pad = max(20, round(diameter * 2))
        patch = frame[max(0, round(y) - pad):min(height, round(y) + pad + 1),
                      max(0, round(x) - pad):min(width, round(x) + pad + 1)]
        if green_fraction(patch) < .65:
            continue
        # A detached ball needs turf clearance toward the closest shoe. An
        # attached bright toe/logo has a dark shoe corridor despite green around it.
        ankle = min(ankles, key=lambda a: np.linalg.norm(point - a))
        direction = ankle - point
        length = np.linalg.norm(direction)
        samples = point + np.linspace(diameter * .65, min(diameter * 2, length * .5), 12)[:, None] * direction / length
        pixels = frame[np.clip(np.round(samples[:, 1]).astype(int), 0, height - 1),
                       np.clip(np.round(samples[:, 0]).astype(int), 0, width - 1)]
        if green_fraction(pixels.reshape(1, -1, 3)) < .75:
            continue
        candidates.append(dict(component, point=point.tolist(), hip=hip.tolist(),
                               body_height=bh, body_id=body.get('track_id')))
    # Too many round rivals is an ambiguous search, not a reason to pick four.
    return candidates if len(candidates) <= 4 else []


def aligned_neural_score(component, boxes):
    """A high score must localize this particular native component tightly."""
    point = np.asarray(component['point']);diameter = max(component['size'])
    scores = []
    for box in boxes:
        if box['class_id'] != 32 or box['score'] < .85:
            continue
        bounds = np.asarray(box['box'], float)
        size = bounds[2:] - bounds[:2]
        if min(size) <= 0 or not .6 <= max(size) / diameter <= 1.7:
            continue
        if np.linalg.norm((bounds[:2] + bounds[2:]) * .5 - point) > max(2, diameter * .35):
            continue
        if not (np.all(bounds[:2] <= point) and np.all(point <= bounds[2:])):
            continue
        scores.append(box['score'])
    return max(scores) if scores else None


class NativeBallSearch:
    def __init__(self, model, components, green_fraction):
        self.model, self.components, self.green_fraction = model, components, green_fraction

    def __call__(self, frame, body, *, scale=1., head=None, sample=False):
        candidates = near_player_components(frame, body, self.components,
                                            self.green_fraction, scale=scale, head=head)
        height, width = frame.shape[:2]
        for candidate in candidates:
            candidate['neural_score'] = None
            candidate['neural_checked'] = sample
            if not sample:
                continue
            x, y = candidate['point'];radius = max(40, round(max(candidate['size']) * 6))
            x0, y0 = max(0, round(x) - radius), max(0, round(y) - radius)
            crop = frame[y0:min(height, round(y) + radius + 1),
                         x0:min(width, round(x) + radius + 1)]
            result = self.model.predict(crop, imgsz=640, classes=[32], conf=.25,
                                        device='cpu', verbose=False)[0]
            boxes = [{'class_id': int(b.cls.item()), 'score': float(b.conf.item()),
                      'box': (b.xyxy[0].cpu().numpy() + [x0, y0, x0, y0]).tolist()}
                     for b in result.boxes]
            candidate['neural_score'] = aligned_neural_score(candidate, boxes)
        return candidates


def confirmed_detail_chains(rows, *, sample_step=6, analysis_scale=1.):
    """One unambiguous consecutive current component with three neural checks.

    Existing current ground registration, when present, locates searches.
    Otherwise the near-player ball must be approximately static in the image.
    Actual output coordinates are always the current white pixels.
    A missing/ambiguous sample splits the chain; scores never bridge the gap.
    """
    confirmed = []
    chain = []
    key = None

    def transform(point, row):
        matrix = row.get('previous_to_current')
        if matrix is None:
            return np.asarray(point, float)
        matrix = np.asarray(matrix, float).copy()
        matrix[:, 2] /= analysis_scale
        return matrix[:, :2] @ point + matrix[:, 2]

    def finish():
        anchors = [r['frame'] for r, c in chain if c.get('neural_score') is not None]
        windows = []
        for a, b in zip(anchors, anchors[2:]):
            if b - a > sample_step * 3:
                continue
            samples = [(r, c) for r, c in chain if a <= r['frame'] <= b]
            anchor = samples[0][1]
            predicted = np.asarray(anchor['point'], float)
            valid = True
            for r, c in samples[1:]:
                predicted = transform(predicted, r)
                if (np.linalg.norm(predicted-c['point']) > max(3, max(anchor['size']) * .5)
                        or not .8 <= max(c['size']) / max(anchor['size']) <= 1.25):
                    valid = False
                    break
            if valid:
                windows.append((a, b))
        for r, c in chain:
            window = next(((a, b) for a, b in windows if a <= r['frame'] <= b), None)
            if window is not None:
                a, b = window
                checks = [n for n in anchors if a <= n <= b]
                confirmed.append((r, dict(c, confirmation_frames=checks)))

    for row in rows:
        candidates = row.get('detail_ball_candidates', [])
        body = row.get('body')
        next_key = (row.get('body_scene'), body.get('track_id')) if body else None
        candidate = candidates[0] if len(candidates) == 1 else None
        if candidate and (body is None or candidate.get('body_id') != body.get('track_id')):
            candidate = None
        if candidate and candidate.get('neural_checked') and candidate.get('neural_score') is None:
            candidate = None
        continues = candidate is not None and next_key is not None and next_key == key
        if continues and chain:
            prev_row, prev = chain[-1]
            residual = np.linalg.norm(np.asarray(candidate['point']) - transform(prev['point'], row))
            ratio = max(candidate['size']) / max(prev['size'])
            continues = (row['frame'] == prev_row['frame'] + 1
                         and residual <= max(4, candidate['body_height'] * .025)
                         and .95 <= candidate['body_height'] / prev['body_height'] <= 1.05
                         and .65 <= ratio <= 1.5)
        if not continues:
            finish();chain = [];key = next_key
        if candidate is not None and next_key is not None:
            chain.append((row, candidate))
    finish()
    return confirmed
