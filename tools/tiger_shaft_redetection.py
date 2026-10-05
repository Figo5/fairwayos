#!/usr/bin/env python3
"""Independent current-frame shaft/head research replay and availability selection.

References enter evaluation only. The detector receives the original seed,
native cached pixels, current body measurements, crop geometry and cuts. Its
causality is conditional on the previously supplied operator body ROI schedule.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostcaddie.demo import target_benchmark as bench

METHODS = ('shaft_head', 'shaft_head_reacquired')
RULE = 'Retain every prior selected observation; only current confirmed shaft/head measurements fill empty full-swing frames. Ball blocks and putt heads unchanged.'


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_bound_json(path, expected=None):
    data = path.read_bytes()
    checksum = hashlib.sha256(data).hexdigest()
    if expected is not None and checksum != expected:
        raise ValueError('Input bytes changed: ' + str(path))
    return json.loads(data), checksum


def select_available(baseline, measured, cohort):
    output = copy.deepcopy(baseline)
    old = {c['id']: c for c in output['clips']}
    new = {c['id']: c for c in measured['clips']}
    if (len(old) != len(output['clips']) or len(new) != len(measured['clips']) or
            len({c['id'] for c in cohort}) != len(cohort) or
            set(old) != set(new) or set(old) != {c['id'] for c in cohort}):
        raise ValueError('Routes must cover the same complete collection')
    trace = {}
    for source in cohort:
        cid = source['id']
        if old[cid]['sha256'] != new[cid]['sha256']:
            raise ValueError('Source hashes differ')
        if source['stroke_type'] not in ('full_swing', 'putt'):
            raise ValueError('Unsupported stroke family')
        if source['stroke_type'] == 'putt':
            continue
        prior = old[cid]['targets']['clubhead']
        current = new[cid]['targets']['clubhead']
        if prior['seed'] is None or prior['seed'] != current['seed']:
            raise ValueError('Routes need the same original seed')
        for block in (prior, current):
            for point in block['points']:
                if point['origin'] == 'operator' and point['source_frame'] != prior['seed']['frame']:
                    raise ValueError('Only original operator seeds are allowed')
                if point['origin'] == 'derived' and (point['x'] is not None or point['y'] is not None):
                    raise ValueError('Derived coordinates cannot enter availability selection')
        first = {p['source_frame']: p for p in prior['points']}
        second = {p['source_frame']: p for p in current['points']}
        if len(first) != len(prior['points']) or len(second) != len(current['points']):
            raise ValueError('Duplicate source frame')
        selected, decisions = [], []
        for frame in sorted(set(first) | set(second)):
            a, b = first.get(frame), second.get(frame)
            if a and a['origin'] in ('automatic', 'operator'):
                point, engine = a, 'prior_selected'
            elif b and b['origin'] == 'automatic':
                if frame <= prior['seed']['frame'] or any(prior['seed']['frame'] <= c <= frame for c in source.get('cut_frames', [])):
                    raise ValueError('New fills must follow the original seed and precede cuts')
                if b['method'] not in METHODS or b['score'] is not None:
                    raise ValueError('Only null-score declared shaft measurements may fill a gap')
                point, engine = b, 'current_shaft_head'
            else:
                point, engine = a or b, 'empty'
            selected.append(copy.deepcopy(point))
            decisions.append({'source_frame': frame, 'engine': engine})
        prior['points'] = selected
        trace[cid] = decisions
    output['variant'] = 'selected_shaft_head_research'
    return output, trace


def body_rows(receipt, source, first, last):
    item = receipt['clips'][source['id']]
    path = Path(item['analysis_path'])
    record, _ = read_bound_json(path, item['analysis_sha256'])
    if item['source_sha256'] != source['sha256'] or record['source']['sha256'] != source['sha256']:
        raise ValueError('Body analysis belongs to a different source')
    if (record['source']['width'], record['source']['height']) != (source['width'], source['height']):
        raise ValueError('Body analysis must use native source geometry')
    rows = [r for scene in record['scenes'] for r in scene['body']['per_frame']]
    if any(type(r.get('source_frame')) is not int or r['source_frame'] < 0 for r in rows):
        raise ValueError('Body source frames must be non-negative integers')
    frames = {r['source_frame']: r for r in rows}
    if len(frames) != len(rows) or not set(range(first, last + 1)) <= set(frames):
        raise ValueError('Body evidence must cover each current seeded frame exactly once')
    # This bridge accepts the archived canonical MoveNet output schema. Its
    # letterbox inversion and ROI origin establish native source coordinates.
    for row in rows:
        if 'ROI_xywh' in row or 'wrists' in row:
            raise ValueError('Alternate body fields bypass canonical native provenance')
        if any(row.get(k) not in (None, False, 'none') for k in ('smoothed', 'smoothing', 'temporal_filter')):
            raise ValueError('Only raw current-frame body inference is supported')
        if row.get('available') is not True:
            row['coordinate_space'] = 'native_source_pixels'
            continue
        transform, roi = row.get('transform', {}), row.get('roi_xywh')
        if (not isinstance(roi, list) or len(roi) != 4 or
                transform.get('roi_origin_xy') != roi[:2] or
                transform.get('model_input_wh') != [192, 192] or
                not isinstance(transform.get('letterbox_scale'), (float, int)) or
                not math.isfinite(transform['letterbox_scale']) or transform['letterbox_scale'] <= 0):
            raise ValueError('Body evidence lacks native MoveNet transform provenance')
        for keypoint in row.get('keypoints', []):
            x, y = keypoint.get('x'), keypoint.get('y')
            if (type(x) not in (int,float) or type(y) not in (int,float) or
                    not math.isfinite(x) or not math.isfinite(y) or
                    not roi[0]-40*source['height']/720 <= x <= roi[0]+roi[2]+40*source['height']/720 or
                    not roi[1]-40*source['height']/720 <= y <= roi[1]+roi[3]+40*source['height']/720):
                raise ValueError('Body keypoints disagree with native ROI coordinates')
        row['coordinate_space'] = 'native_source_pixels'
    return frames


def run_measurement(args, dataset, baseline, engine):
    import cv2
    import numpy as np
    if args.frames_receipt is None or args.body_receipt is None:
        raise ValueError('Measurement needs both frame and body receipts')
    frames, frames_sha = read_bound_json(args.frames_receipt)
    body, body_sha = read_bound_json(args.body_receipt)
    ids = {c['id'] for c in dataset['clips']}
    if set(frames['clips']) != ids or set(body['clips']) != ids:
        raise ValueError('Receipts must cover the complete collection')
    if bench.sha256_file(Path(body['model_path'])) != body['model_sha256']:
        raise ValueError('Archived body model provenance changed')
    output = copy.deepcopy(baseline)
    output['variant'] = 'full_swing_shaft_redetection_research'
    targets = {c['id']: c for c in output['clips']}
    measured = {}
    for source in dataset['clips']:
        if source['stroke_type'] != 'full_swing':
            continue
        cid = source['id']
        entry = frames['clips'][cid]
        expected = {k: source[k] for k in ('id', 'path', 'sha256', 'stroke_type', 'window', 'cut_frames')}
        seed = source['targets']['clubhead']['seed']
        expected['seed'] = seed
        if entry['input'] != expected:
            raise ValueError('Frame receipt differs from original seed/source/window')
        crop = entry['crop']
        if (len(crop) != 4 or any(type(v) is not int for v in crop) or
                crop[0] < 0 or crop[1] < 0 or min(crop[2:]) <= 0 or
                crop[0] + crop[2] > source['width'] or crop[1] + crop[3] > source['height']):
            raise ValueError('Crop must lie in native source geometry')
        first, last = seed['frame'], source['window']['end_frame']
        if set(entry['frames']) != {str(f) for f in range(first, last + 1)}:
            raise ValueError('Frame receipt must cover the complete seeded window')
        poses = body_rows(body, source, first, last)
        detector = engine.ShaftHeadPolicy(seed, crop[:2], source['cut_frames'],
                                         (source['width'], source['height']), source['stroke_type'])
        points, raw = [], []
        for frame in range(first, last + 1):
            item = entry['frames'][str(frame)]
            path = Path(item['path'])
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('Cached frame bytes changed')
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if image is None or image.dtype != np.uint8 or image.shape != (crop[3], crop[2], 3):
                raise ValueError('Frame dimensions differ from native crop')
            point, evidence = detector.step(frame, image, body_frame=poses[frame])
            points.append(point); raw.append(evidence)
        block = targets[cid]['targets']['clubhead']
        if any(p['origin'] == 'automatic' for p in block['points'] if p['source_frame'] < first):
            raise ValueError('Automatic observation before seed')
        block['points'] = [p for p in block['points'] if p['source_frame'] < first] + points
        measured[cid] = {'input': expected, 'crop': crop, 'points': points, 'raw': raw}
    runtime = {'frames_receipt_path': str(args.frames_receipt.resolve()),
               'frames_receipt_sha256': frames_sha,
               'body_receipt_path': str(args.body_receipt.resolve()),
               'body_receipt_sha256': body_sha,
               'body_context_provenance': body, 'opencv': cv2.__version__, 'numpy': np.__version__,
               'opencv_build_information': cv2.getBuildInformation(), 'python': sys.version,
               'sam_masks_used_to_propose': False, 'new_neural_inference': False,
               'operator_annotations_used': ['original_seed', 'window', 'cut_frames', 'body_roi_keyframe_schedule'],
               'causality_scope': 'Current pixels/current pose outputs only, conditional on non-causal operator ROI crop assistance.'}
    return output, {'profile': engine.PROFILE, 'clips': measured}, runtime


def run_selection(args, dataset, baseline, engine):
    if args.measured is None or args.measured_report is None:
        raise ValueError('Selection needs measured predictions and their executed report')
    measured = json.loads(args.measured.read_text())
    record = json.loads(args.measured_report.read_text())
    bench.validate_predictions(measured, dataset)
    if record.get('predictions', {}).get('sha256') != bench.sha256_file(args.measured):
        raise ValueError('Report does not bind measured prediction bytes')
    if record.get('reference', {}).get('sha256') != bench.sha256_file(args.dataset):
        raise ValueError('Measured report belongs to a different dataset')
    if measured.get('variant') != 'full_swing_shaft_redetection_research' or record['predictions'].get('variant') != measured['variant']:
        raise ValueError('Only independent shaft measurements may enter this route')
    runtime = record.get('runtime')
    if not isinstance(runtime, dict) or runtime.get('profile') != engine.PROFILE:
        raise ValueError('Measured profile differs from current detector')
    if runtime.get('engine_sha256') != bench.sha256_file(Path(engine.__file__)):
        raise ValueError('Detector code differs from executed measurement')
    for key, path in [('geometry_module_sha256', ROOT/'ghostcaddie/demo/swing_head.py'),
                      ('descriptor_module_sha256', ROOT/'ghostcaddie/demo/head_identity.py'),
                      ('tool_sha256', Path(__file__)), ('benchmark_sha256', Path(bench.__file__))]:
        if runtime.get(key) != bench.sha256_file(path):
            raise ValueError('Executed detector dependency bytes changed: ' + key)
    for stem in ('frames_receipt', 'body_receipt'):
        path = runtime.get(stem + '_path')
        if not isinstance(path, str) or not Path(path).is_file() or bench.sha256_file(Path(path)) != runtime.get(stem + '_sha256'):
            raise ValueError('Executed input receipt bytes changed: ' + stem)
    if runtime.get('baseline_predictions_sha256') != bench.sha256_file(args.baseline):
        raise ValueError('Measured run used a different baseline')
    raw_path = runtime.get('raw_path')
    if not isinstance(raw_path, str) or not Path(raw_path).is_file():
        raise ValueError('Raw evidence changed')
    raw, _ = read_bound_json(Path(raw_path), runtime.get('raw_sha256'))
    frames, _ = read_bound_json(Path(runtime['frames_receipt_path']), runtime['frames_receipt_sha256'])
    poses, _ = read_bound_json(Path(runtime['body_receipt_path']), runtime['body_receipt_sha256'])
    if raw.get('profile') != engine.PROFILE or set(raw.get('clips', {})) != {c['id'] for c in dataset['clips'] if c['stroke_type']=='full_swing'}:
        raise ValueError('Raw evidence does not cover the declared measurement cohort')
    measured_clips = {c['id']:c for c in measured['clips']}
    for source in dataset['clips']:
        if source['stroke_type'] != 'full_swing':
            continue
        cid, seed = source['id'], source['targets']['clubhead']['seed']
        first, last = seed['frame'], source['window']['end_frame']
        entry, evidence = frames['clips'][cid], raw['clips'][cid]
        expected = {k: source[k] for k in ('id','path','sha256','stroke_type','window','cut_frames')}
        expected['seed'] = seed
        if entry['input'] != expected or evidence['input'] != expected or evidence['crop'] != entry['crop']:
            raise ValueError('Raw measurement geometry/source/input differs')
        body_rows(poses, source, first, last)
        if set(entry['frames']) != {str(f) for f in range(first,last+1)}:
            raise ValueError('Executed frame receipt has incomplete coverage')
        for item in entry['frames'].values():
            if bench.sha256_file(Path(item['path'])) != item['sha256']:
                raise ValueError('Executed cached pixels changed')
        points = [p for p in measured_clips[cid]['targets']['clubhead']['points'] if p['source_frame'] >= first]
        if (points != evidence['points'] or len(points) != len(evidence['raw']) or
                [p['source_frame'] for p in points] != list(range(first,last+1))):
            raise ValueError('Measured points disagree with complete raw evidence')
        for point, decision in zip(points, evidence['raw']):
            d = decision['decision']
            if decision['frame'] != point['source_frame'] or d['frame'] != point['source_frame'] or any(d[k] != point[k] for k in ('status','origin','method','x','y','score')):
                raise ValueError('Measured point disagrees with current raw decision')
    output, decisions = select_available(baseline, measured,
        [{'id': c['id'], 'stroke_type': c['stroke_type'], 'cut_frames':c['cut_frames']} for c in dataset['clips']])
    return output, {'selection_rule': RULE, 'decisions': decisions}, {
        'selection_rule': RULE, 'selection_uses_reference_labels': False,
        'measured_path': str(args.measured.resolve()), 'measured_sha256': bench.sha256_file(args.measured),
        'measured_report_sha256': bench.sha256_file(args.measured_report),
        'measured_runtime': runtime, 'proposal_feedback': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'baseline', 'baseline-report', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('frames-receipt', 'body-receipt', 'measured', 'measured-report'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    code_paths = [ROOT/'ghostcaddie/demo/shaft_head.py', Path(__file__), Path(bench.__file__),
                  ROOT/'ghostcaddie/demo/swing_head.py', ROOT/'ghostcaddie/demo/head_identity.py']
    executed_code = {str(path): bench.sha256_file(path) for path in code_paths}
    from ghostcaddie.demo import shaft_head as engine
    if not args.out.name.endswith('.predictions.json'):
        raise ValueError('Output must end in .predictions.json')
    raw_path = args.out.with_name(args.out.name.replace('.predictions.json', '.raw.json'))
    writes = [args.out, raw_path, Path(str(args.out) + '.report.json'), Path(str(args.out) + '.report.md')]
    inputs = [v.resolve() for k, v in vars(args).items() if k != 'out' and v is not None]
    dataset, dataset_sha = bench.load_dataset(args.dataset, verify_hashes=True)
    protected = set(inputs) | {Path(c['path']).resolve() for c in dataset['clips']}
    if any(p.exists() or p.resolve() in protected for p in writes):
        raise ValueError('Choose fresh outputs that preserve all source/evidence files')
    baseline = json.loads(args.baseline.read_text())
    bench.validate_predictions(baseline, dataset)
    baseline_report = json.loads(args.baseline_report.read_text())
    if baseline_report.get('predictions', {}).get('sha256') != bench.sha256_file(args.baseline):
        raise ValueError('Baseline report does not bind baseline prediction bytes')
    if baseline_report.get('reference', {}).get('sha256') != dataset_sha:
        raise ValueError('Baseline report belongs to a different reference dataset')
    if args.measured is not None or args.measured_report is not None:
        if args.frames_receipt is not None or args.body_receipt is not None:
            raise ValueError('Select using measured evidence only; do not mix execution stages')
        payload, raw, runtime = run_selection(args, dataset, baseline, engine)
    else:
        payload, raw, runtime = run_measurement(args, dataset, baseline, engine)
    validated = bench.validate_predictions(payload, dataset)
    if any(bench.sha256_file(Path(path)) != checksum for path, checksum in executed_code.items()):
        raise ValueError('Executed code changed during this run')
    for old, new in zip(baseline['clips'], payload['clips']):
        if old['id'] != new['id'] or old['targets']['ball'] != new['targets']['ball']:
            raise ValueError('Ball blocks changed')
        if next(c for c in dataset['clips'] if c['id'] == old['id'])['stroke_type'] == 'putt' and old != new:
            raise ValueError('Putt clip changed')
    raw_text, prediction_text = bench.canonical_json(raw), bench.canonical_json(payload)
    runtime.update({'profile': engine.PROFILE, 'engine_sha256': bench.sha256_file(Path(engine.__file__)),
        'executed_code_start_sha256': executed_code,
        'qualitative_review_required_before_media_export': True,
        'tool_sha256': bench.sha256_file(Path(__file__)), 'benchmark_sha256': bench.sha256_file(Path(bench.__file__)),
        'geometry_module_sha256': bench.sha256_file(ROOT/'ghostcaddie/demo/swing_head.py'),
        'descriptor_module_sha256': bench.sha256_file(ROOT/'ghostcaddie/demo/head_identity.py'),
        'baseline_predictions_sha256': bench.sha256_file(args.baseline),
        'baseline_report_sha256': bench.sha256_file(args.baseline_report),
        'baseline_executed_runtime': baseline_report['runtime'],
        'raw_path': str(raw_path.resolve()), 'raw_sha256': digest(raw_text),
        'repository_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'repository_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).splitlines(),
        'research_only': True, 'production_eligible': False, 'physical_identity_established': False,
        'exported_confidence': None, 'target_reference_labels_read_by_detector': False,
        'note': 'Previously inspected AI development references, not independent/human ground truth. Sparse metrics do not establish quality at unlabeled frames.'})
    report = bench.evaluate(dataset, validated, dataset_path=str(args.dataset), dataset_sha256=dataset_sha,
                            predictions_path=str(args.out), predictions_sha256=digest(prediction_text), runtime=runtime)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for path, contents in zip(writes, [prediction_text, raw_text, bench.canonical_json(report), bench.render_markdown(report)]):
        with path.open('x', encoding='utf-8') as handle:
            handle.write(contents)


if __name__ == '__main__':
    main()
