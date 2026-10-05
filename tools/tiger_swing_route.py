"""Availability-based full-swing research route, frozen before scoring.

Retain classic automatic observations; where classic is empty, allow a currently
observed, causally confirmed swing mask core. No later label, per-clip outcome,
interpolation, future anchor, or presentation correction selects either engine.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghostcaddie.demo import target_benchmark as bench

RULE = "All full swings: classic automatic observation first; otherwise current confirmed swing mask core. Putts and ball blocks unchanged."
SWING_METHODS = ('sam2_swing_core', 'sam2_swing_core_reacquired')


def validate_measured_record(record, measured, measured_path, dataset_sha, baseline_sha):
    from ghostcaddie.demo import swing_context, swing_head
    if record.get('predictions', {}).get('sha256') != bench.sha256_file(measured_path):
        raise ValueError('Executed report does not bind measured prediction bytes')
    if record.get('reference', {}).get('sha256') != dataset_sha:
        raise ValueError('Measured report belongs to a different reference dataset')
    if measured.get('variant') != 'full_swing_mask_core_research' or record['predictions'].get('variant') != measured['variant']:
        raise ValueError('Only the declared full-swing measurement variant is eligible')
    runtime = record.get('runtime')
    if not isinstance(runtime, dict) or runtime.get('profile') != swing_context.PROFILE:
        raise ValueError('Executed report must declare the current context measurement profile')
    if (runtime.get('head_region_sha256') != bench.sha256_file(Path(swing_context.__file__)) or
            runtime.get('swing_measurement_base_sha256') != bench.sha256_file(Path(swing_head.__file__))):
        raise ValueError('Executed policy hashes must match the current context implementation')
    if runtime.get('baseline_predictions_sha256') != baseline_sha:
        raise ValueError('Measured run used a different baseline')
    raw_path = runtime.get('raw_path')
    if not isinstance(raw_path, str) or not Path(raw_path).is_file() or bench.sha256_file(Path(raw_path)) != runtime.get('raw_sha256'):
        raise ValueError('Executed report does not bind raw measurement bytes')


def compose(baseline, measured, cohort):
    result = copy.deepcopy(baseline)
    result['variant'] = 'full_swing_availability_research'
    old = {c['id']: c for c in result['clips']}
    new = {c['id']: c for c in measured['clips']}
    if (len(old) != len(baseline['clips']) or len(new) != len(measured['clips'])
            or len({c['id'] for c in cohort}) != len(cohort)
            or set(old) != set(new) or set(old) != {c['id'] for c in cohort}):
        raise ValueError('Routes must cover the same complete collection')
    decisions = {}
    for source in cohort:
        cid = source['id']
        if old[cid]['sha256'] != new[cid]['sha256']:
            raise ValueError('Source hash changed between routes')
        if source['stroke_type'] not in ('full_swing', 'putt'):
            raise ValueError('Unsupported operator stroke family')
        if source['stroke_type'] != 'full_swing':
            continue
        original = old[cid]['targets']['clubhead']
        learned = new[cid]['targets']['clubhead']
        if original['seed'] is None or original['seed'] != learned['seed']:
            raise ValueError('Routes must use the same original seed')
        if any(p['origin'] == 'operator' and p['source_frame'] != original['seed']['frame']
               for p in original['points'] + learned['points']):
            raise ValueError('Only the original seed may be an operator point')
        if any(p['origin'] == 'derived' and p['x'] is not None
               for p in original['points'] + learned['points']):
            raise ValueError('Derived coordinates are not eligible for selection')
        classic = {p['source_frame']: p for p in original['points']}
        masked = {p['source_frame']: p for p in learned['points']}
        if len(classic) != len(original['points']) or len(masked) != len(learned['points']):
            raise ValueError('Duplicate source frame')
        selected = []
        trace = []
        for frame in sorted(set(classic) | set(masked)):
            prior, current = classic.get(frame), masked.get(frame)
            if prior and prior['origin'] in ('operator', 'automatic'):
                point, engine = prior, 'classic'
            elif current and current['origin'] == 'automatic':
                if current['method'] not in SWING_METHODS or current['score'] is not None:
                    raise ValueError('Only declared null-score swing measurements may fill an empty frame')
                point, engine = current, 'swing_core'
            else:
                point, engine = (prior or current), 'empty'
            selected.append(copy.deepcopy(point))
            trace.append({'source_frame': frame, 'engine': engine})
        original['points'] = selected
        decisions[cid] = trace
    return result, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'baseline', 'baseline-report', 'measured', 'measured-report', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    dataset, sha = bench.load_dataset(args.dataset, verify_hashes=True)
    writes = [args.out, Path(str(args.out) + '.report.json'), Path(str(args.out) + '.report.md'),
              Path(str(args.out) + '.selection.json')]
    protected = {p.resolve() for p in (args.dataset, args.baseline, args.baseline_report, args.measured, args.measured_report)} | {
        Path(c['path']).resolve() for c in dataset['clips']}
    if not args.out.name.endswith('.predictions.json') or any(p.exists() or p.resolve() in protected for p in writes):
        raise ValueError('Choose fresh .predictions.json outputs that preserve all inputs')
    baseline = json.loads(args.baseline.read_text())
    measured = json.loads(args.measured.read_text())
    record = json.loads(args.measured_report.read_text())
    baseline_record = json.loads(args.baseline_report.read_text())
    bench.validate_predictions(baseline, dataset)
    bench.validate_predictions(measured, dataset)
    validate_measured_record(record, measured, args.measured, sha, bench.sha256_file(args.baseline))
    if baseline_record.get('predictions', {}).get('sha256') != bench.sha256_file(args.baseline):
        raise ValueError('Executed baseline report does not bind baseline prediction bytes')
    payload, trace = compose(baseline, measured, [{'id': c['id'], 'stroke_type': c['stroke_type']} for c in dataset['clips']])
    for prior, selected in zip(baseline['clips'], payload['clips']):
        if bench.canonical_json(prior['targets']['ball']) != bench.canonical_json(selected['targets']['ball']):
            raise ValueError('Ball block changed')
    validated = bench.validate_predictions(payload, dataset)
    prediction_text = bench.canonical_json(payload)
    selection_text = bench.canonical_json({'rule': RULE, 'decisions': trace})
    digest = lambda value: hashlib.sha256(value.encode('utf-8')).hexdigest()
    runtime = {'research_only': True, 'production_eligible': False, 'model_rerun': False,
               'selection_rule': RULE, 'selection_uses_reference_labels': False,
               'baseline_sha256': bench.sha256_file(args.baseline), 'measured_sha256': bench.sha256_file(args.measured),
               'baseline_path': str(args.baseline.resolve()), 'baseline_report_path': str(args.baseline_report.resolve()),
               'baseline_report_sha256': bench.sha256_file(args.baseline_report), 'baseline_executed_runtime': baseline_record['runtime'],
               'measured_report_sha256': bench.sha256_file(args.measured_report), 'measured_runtime': record['runtime'],
               'selection_sha256': digest(selection_text), 'tool_sha256': bench.sha256_file(Path(__file__)),
               'benchmark_sha256': bench.sha256_file(Path(bench.__file__)),
               'repository_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
               'repository_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=Path(__file__).resolve().parents[1], text=True).splitlines(),
               'note': 'Availability rule predeclared before this experiment was scored; already inspected development footage, not independent holdout. Classical and mask tracks have independent causal state; selection does not feed back into either tracker.'}
    report = bench.evaluate(dataset, validated, dataset_path=str(args.dataset),
                            dataset_sha256=sha, predictions_path=str(args.out), predictions_sha256=digest(prediction_text), runtime=runtime)
    texts = [prediction_text, bench.canonical_json(report), bench.render_markdown(report), selection_text]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for path, contents in zip(writes, texts):
        with path.open('x', encoding='utf-8') as handle:
            handle.write(contents)


if __name__ == '__main__':
    main()
