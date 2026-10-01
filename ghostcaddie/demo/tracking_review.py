"""Read-only review of saved research predictions, bound to registered media bytes."""
import json
import math
from pathlib import Path

from .errors import DemoConfigError
from . import target_benchmark as bench

MAX_RESULT_BYTES = 16 * 1024 * 1024


def load_tracking_review(path, records):
    """Validate saved point contracts without reference labels or an accuracy claim."""
    path = Path(path)
    if path.stat().st_size > MAX_RESULT_BYTES:
        raise DemoConfigError('Tracking results exceed 16 MiB')
    try:
        payload = json.loads(path.read_text())
        known = {r.sha256: r for r in records}
        clips = []
        seen_hashes = set()
        seeds = {}
        for block in payload['clips']:
            record = known.get(block['sha256'])
            if record is None or block['sha256'] in seen_hashes:
                raise ValueError('Results must bind once to registered source hashes')
            seen_hashes.add(block['sha256'])
            targets = block['targets']
            if not targets or set(targets) - {'ball', 'clubhead'}:
                raise ValueError('Unknown tracking target')
            for target, result in targets.items():
                seed = result['seed']
                if seed is not None:
                    if (isinstance(seed['frame'], bool) or not isinstance(seed['frame'], int)
                            or not 0 <= seed['frame'] < record.frame_count):
                        raise ValueError('Seed outside source frames')
                    for key, extent in [('x', record.width), ('y', record.height)]:
                        value = seed[key]
                        if (isinstance(value, bool) or not isinstance(value, (int, float))
                                or not math.isfinite(value) or not 0 <= value <= extent):
                            raise ValueError('Seed outside native source pixels')
                seeds[(block['sha256'], target)] = seed
            clips.append({'id': block['id'], 'sha256': record.sha256,
                          'width': record.width, 'height': record.height,
                          'window': {'start_frame': 0, 'end_frame': record.frame_count - 1},
                          'cut_frames': [], 'targets': targets})
        validated = bench.validate_predictions(payload, {
            'clips': clips, 'clip_index': {c['id']: c for c in clips}})
        reviews = {}
        for clip in validated['clips']:
            reviews[clip['sha256']] = {}
            for target, result in clip['targets'].items():
                seed = seeds[(clip['sha256'], target)]
                points = result['points']
                if any(p['origin'] == 'automatic' and (seed is None or p['source_frame'] <= seed['frame']) for p in points):
                    raise ValueError('Automatic observations require an earlier seed')
                frames = [p['source_frame'] for p in points if seed is not None and p['source_frame'] > seed['frame']]
                automatic = {p['source_frame'] for p in points if p['origin'] == 'automatic'}
                gaps = []
                if frames:
                    for frame in range(min(frames), max(frames) + 1):
                        if frame in automatic:
                            continue
                        if gaps and gaps[-1]['end_frame'] == frame - 1:
                            gaps[-1]['end_frame'] = frame
                        else:
                            gaps.append({'start_frame': frame, 'end_frame': frame})
                reviews[clip['sha256']][target] = {
                    'variant': validated['variant'], 'target': target,
                    'source_sha256': clip['sha256'],
                    'start_frame': min(frames) if frames else None,
                    'end_frame': max(frames) if frames else None,
                    'automatic_frames': len(automatic), 'gaps': gaps,
                    'accuracy_claim': False,
                    'note': 'Saved automatic positions only. Gaps can be occlusion or tracking loss; this does not measure accuracy.'}
        return reviews
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
        raise DemoConfigError(f'Invalid tracking results: {exc}') from exc
