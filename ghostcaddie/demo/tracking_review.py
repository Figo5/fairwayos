"""Read-only review of source-bound saved research predictions."""
from .saved_tracks import load_saved_tracks


def load_tracking_review(path, records):
    return summarize_saved_tracks(load_saved_tracks(path, records))


def summarize_saved_tracks(saved):
    """Summarize the same validated snapshot that supplies preview and export."""
    reviews = {}
    for sha, clip in saved['clips_by_sha'].items():
        reviews[sha] = {}
        for target, result in clip['targets'].items():
            seed, points = result['seed'], result['points']
            frames = [p['source_frame'] for p in points
                      if seed is not None and p['source_frame'] > seed['frame']]
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
            reviews[sha][target] = {
                'variant': saved['variant'], 'target': target, 'source_sha256': sha,
                'start_frame': min(frames) if frames else None,
                'end_frame': max(frames) if frames else None,
                'automatic_frames': len(automatic), 'gaps': gaps,
                'accuracy_claim': False,
                'note': 'Saved automatic positions only. Gaps can be occlusion or tracking loss; this does not measure accuracy.'}
    return reviews
