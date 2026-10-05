"""Opt-in saved research tracks for broadcast rendering; no new measurements."""
from types import SimpleNamespace

from .errors import DemoConfigError

DISCLOSURE = 'SAVED AUTOMATIC RESEARCH · 2D IMAGE ESTIMATES'


def attach_saved_tracks(spec, resolved, predictions_path, expected_sha256=None):
    """Bind immutable saved observations before rendering; ignore assisted paths."""
    from .saved_tracks import load_saved_tracks, build_saved_track
    source = resolved.source
    record = SimpleNamespace(sha256=spec['source_sha256'], width=source.width,
                             height=source.height, frame_count=source.frame_count)
    if not record.frame_count:
        raise DemoConfigError('Saved tracking export requires a known source frame count')
    saved = load_saved_tracks(predictions_path, [record], require_all_registered=False)
    if expected_sha256 and saved['prediction_sha256'] != expected_sha256:
        raise DemoConfigError('Saved tracking file changed; restart review to load it')
    block = saved['clips_by_sha'].get(record.sha256)
    if not block:
        raise DemoConfigError('No saved tracking results match this source video')
    targets = block['targets']
    cuts = set()
    config = getattr(resolved, 'config', None)
    for cut in getattr(config, 'cuts', ()):
        if cut.decision in {'accepted', 'operator_added'}:
            cuts.update(cut.span if cut.kind == 'dissolve' and cut.span else [cut.frame])
    for override in getattr(config, 'overrides', ()):
        if override.get('decision') == 'accept' and 'frame' in override:
            cuts.add(override['frame'])
    applied = 0
    excluded_seeds = {}
    excluded_assistance = {}
    for entry in spec['broadcast']['scenes']:
        scene = entry.get('spec')
        if scene is None:
            continue
        frames = range(entry['window']['start_frame'], entry['window']['end_frame'] + 1)
        imported = {}
        excluded_seeds[entry['id']] = scene.get('seeds', {})
        excluded_assistance[entry['id']] = {'corrections': scene.get('corrections', {}),
                                           'manual_segments': scene.get('manual_segments', {})}
        for name, target in targets.items():
            # A forward track belongs only to the scene containing its operator seed.
            seed = target.get('seed')
            if seed is None or seed['frame'] not in frames:
                continue
            imported[name] = build_saved_track(
                name, target, frames, cut_frames=sorted(cuts)).to_dict()
        scene['saved_tracks'] = imported
        scene['manual_segments'] = {}
        scene['corrections'] = {}
        scene['seeds'] = {}
        # Missing imported targets stay absent; no automatic search or assistance fallback.
        applied += len(imported)
    if not applied:
        raise DemoConfigError('No saved tracking seed lies inside a stroke scene window')
    provenance = {'mode': 'saved_automatic_research',
                  'prediction_sha256': saved['prediction_sha256'],
                  'variant': saved['variant'], 'source_sha256': record.sha256,
                  'model_rerun': False, 'accuracy_claim': False,
                  'assisted_paths_used': False, 'physical_metrics_available': False,
                  'excluded_configured_seeds': excluded_seeds,
                  'excluded_assistance': excluded_assistance, 'cut_frames': sorted(cuts),
                  'note': 'Saved observations only; operator seed retained, gaps drawn empty. '
                          'Config seeds and assistance are preserved in the document but excluded from target tracking in this export.'}
    spec['saved_tracking'] = provenance
    spec['showcase']['disclosure'] = DISCLOSURE
    spec['broadcast']['presentation']['disclosure'] = DISCLOSURE
    return provenance
