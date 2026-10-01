#!/usr/bin/env python3
"""Compose a research seed-capability route from already measured prediction bytes.

Only initial-mask admission chooses the head engine. No reference coordinate,
later tracking outcome or per-clip score selects a route. Refusal keeps the entire
classic head block; it never switches engines halfway through a track.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghostcaddie.demo import target_benchmark as bench


def compose(baseline, candidate, seed_valid, declared_putts):
    baseline_ids = {c['id'] for c in baseline['clips']}
    if set(seed_valid) - set(declared_putts) or set(seed_valid) - baseline_ids:
        raise ValueError('Only declared putts can be routed')
    if any(type(valid) is not bool for valid in seed_valid.values()):
        raise ValueError('Initial seed admission must be a boolean')
    measured = {c['id']: c for c in candidate['clips']}
    if set(measured) != baseline_ids:
        raise ValueError('Prediction clip sets differ')
    result = copy.deepcopy(baseline)
    result['variant'] = 'sam2_seed_capability_fallback'
    routing = {}
    for clip in result['clips']:
        cid = clip['id']
        if measured[cid]['targets']['ball'] != clip['targets']['ball']:
            raise ValueError('Candidate must preserve every classic ball block')
        valid = seed_valid.get(cid, False)
        if valid:
            clip['targets']['clubhead'] = copy.deepcopy(measured[cid]['targets']['clubhead'])
        routing[cid] = {'head_engine': 'sam2_head_region' if valid else 'classic_replay',
                        'initial_seed_admitted': seed_valid.get(cid),
                        'reason': 'initial mask geometry admitted' if valid else
                        ('initial mask refused' if cid in seed_valid else 'not measured by SAM2')}
    return result, routing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve() in [args.baseline.resolve(), args.candidate.resolve()]:
        raise ValueError('Output must preserve measured input files')
    dataset, dataset_sha = bench.load_dataset(args.dataset, verify_hashes=True)
    if args.out.resolve() in {Path(c['path']).resolve() for c in dataset['clips']}:
        raise ValueError('Output must not overwrite a source clip')
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    bench.validate_predictions(baseline, dataset)
    bench.validate_predictions(candidate, dataset)
    measured_report = json.loads(Path(str(args.candidate)+'.report.json').read_text())
    if measured_report['predictions']['sha256'] != bench.sha256_file(args.candidate):
        raise ValueError('Measured predictions differ from their report')
    runtime = measured_report['runtime']
    if runtime['baseline_predictions_sha256'] != bench.sha256_file(args.baseline):
        raise ValueError('Measured candidate uses another baseline')
    raw_path = Path(runtime['raw_path'])
    if runtime['raw_sha256'] != bench.sha256_file(raw_path):
        raise ValueError('Raw inference record changed')
    raw = json.loads(raw_path.read_text())
    clips = {c['id']: c for c in dataset['clips']}
    for cid, measured in raw['clips'].items():
        source = clips[cid]
        expected = {'id':cid, 'path':source['path'], 'sha256':source['sha256'],
                    'seed':source['targets']['clubhead']['seed'],
                    'window':source['window'], 'cut_frames':source['cut_frames']}
        if measured['input'] != expected:
            raise ValueError('Inference inputs differ from the scored collection')
    payload, routing = compose(baseline, candidate,
        {cid:c['seed_valid'] for cid,c in raw['clips'].items()},
        {c['id'] for c in dataset['clips'] if c['stroke_type']=='putt'})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(bench.canonical_json(payload))
    predictions = bench.validate_predictions(payload, dataset)
    report = bench.evaluate(dataset, predictions, dataset_path=str(args.dataset),
        dataset_sha256=dataset_sha, predictions_path=str(args.out),
        predictions_sha256=bench.sha256_file(args.out), runtime={
            'research_only':True, 'production_eligible':False,
            'composer_sha256':bench.sha256_file(Path(__file__)),
            'baseline_predictions_sha256':bench.sha256_file(args.baseline),
            'measured_predictions_sha256':bench.sha256_file(args.candidate),
            'measured_runtime':runtime, 'routing':routing,
            'selection_inputs':['initial seed mask geometry','declared stroke type'],
            'reference_labels_used_for_routing':False,
            'tracker_rerun':False,
            'note':'Whole target blocks replayed. No mid-track fallback, later correction, score-based or clip-ID-based route.'})
    Path(str(args.out)+'.report.json').write_text(bench.canonical_json(report))
    Path(str(args.out)+'.report.md').write_text(bench.render_markdown(report))


if __name__ == '__main__':
    main()
