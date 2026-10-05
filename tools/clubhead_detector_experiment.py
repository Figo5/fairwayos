#!/usr/bin/env python3
"""Opt-in author-box detector research. Never changes selected Tiger tracks."""
import argparse
import datetime
import os
import shutil
import tempfile
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, obj):
    path = Path(path)
    if path.exists():
        raise ValueError('Preserve existing result: ' + str(path))
    path.write_text(json.dumps(obj, indent=2) + '\n')


def iou(a, b):
    intersect = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    area = lambda x: max(0., x[2]-x[0])*max(0., x[3]-x[1])
    union = area(a)+area(b)-intersect
    return intersect/union if union else 0.


def score(rows, threshold=.5, operating_conf=.25):
    """One author box per frame; extra/negative-frame detections are false positives."""
    truths = sum(r['truth'] is not None for r in rows)
    ranked = []
    for number, row in enumerate(rows):
        for p in row['predictions']:
            ranked.append((p['confidence'], number, p['xyxy']))
    ranked.sort(key=lambda x: -x[0])
    matched = set()
    tp = fp = op_tp = op_fp = 0
    curve = []
    for confidence, number, box in ranked:
        truth = rows[number]['truth']
        correct = truth is not None and number not in matched and iou(box, truth) >= threshold
        if correct:
            matched.add(number); tp += 1
        else:
            fp += 1
        if confidence >= operating_conf:
            op_tp += int(correct); op_fp += int(not correct)
        curve.append((tp/truths if truths else 0., tp/(tp+fp)))
    ap = sum(max((p for r, p in curve if r >= k/100), default=0.) for k in range(101))/101 if truths else None
    negative = [r for r in rows if r['truth'] is None]
    return {'frames': len(rows), 'positive_frames': truths, 'negative_frames': len(negative),
            'true_positives': op_tp, 'false_positives': op_fp, 'false_negatives': truths-op_tp,
            'precision': op_tp/(op_tp+op_fp) if op_tp+op_fp else None,
            'recall': op_tp/truths if truths else None, 'ap': ap,
            'negative_frames_with_detection': sum(any(p['confidence'] >= operating_conf for p in r['predictions']) for r in negative)}


def metrics(rows, settings):
    thresholds = settings['ap_iou_thresholds']
    if not thresholds or thresholds[0] != .5:
        raise ValueError('AP50 requires first IoU threshold 0.5')
    scores = [score(rows, t, settings['operating_conf']) for t in thresholds]
    first = dict(scores[0])
    first['AP50'] = first.pop('ap')
    first['AP50_95'] = sum(s['ap'] for s in scores)/len(scores) if first['positive_frames'] else None
    return first


def validate_test_manifest(training, test):
    from ghostcaddie.demo.clubhead_data import golfer_identity, public_identity
    trained = [e for e in training['accepted'] if e['split'] in ('train', 'val')]
    by_id = {e['video_id']: e for e in trained}
    identities = {golfer_identity(e['notes']) for e in trained}
    urls = {public_identity(e['youtube_id'], e['source']) for e in trained}
    sources = {e['source_sha256'] for e in trained}
    frames = {f['image_sha256'] for e in trained for f in e['frames']}
    for entry in test['accepted']:
        if entry['split'] != 'test':
            if entry != by_id.get(entry['video_id']):
                raise ValueError('Training/validation manifest entry changed')
            continue
        if (entry['video_id'] in by_id or golfer_identity(entry['notes']) in identities
                or public_identity(entry['youtube_id'], entry['source']) in urls
                or entry['source_sha256'] in sources
                or any(f['image_sha256'] in frames for f in entry['frames'])):
            raise ValueError('Test overlaps frozen train/validation content or declared identity')


def validate_test_selection(selection, manifest, manifest_sha, planned_ids):
    actual = {e['video_id'] for e in manifest['accepted'] if e['split'] == 'test'}
    accepted = set(selection['accepted_ids'])
    rejected = set(selection['rejected_ids'])
    if (not actual or actual != accepted or accepted & rejected
            or accepted | rejected != set(planned_ids)
            or selection['manifest_sha256'] != manifest_sha):
        raise ValueError('Test dataset differs from the fixed alignment-reviewed roster')


def train(run):
    import torch
    import ultralytics
    from ultralytics import YOLO
    protocol = json.loads((run/'coordination/TRAINING_PROTOCOL.json').read_text())
    manifest = run/'data/training_dataset/manifest.json'
    if (run/'results/CHECKPOINT_FREEZE.json').exists() or (run/'results/learned_detector').exists():
        raise ValueError('Training already attempted; preserve the fixed run')
    if not torch.backends.mps.is_available():
        raise ValueError('Protocol device mps unavailable')
    weights = Path(__file__).resolve().parents[1]/'yolo11n.pt'
    write(run/'results/TRAINING_START.json', {'protocol_sha256': sha(run/'coordination/TRAINING_PROTOCOL.json'),
        'manifest_sha256': sha(manifest), 'pretrained_sha256': sha(weights), 'torch':torch.__version__, 'ultralytics':ultralytics.__version__})
    keys = ('epochs','imgsz','batch','device','workers','seed','optimizer','lr0','patience','mosaic','mixup','degrees','translate','scale','fliplr')
    YOLO(str(weights)).train(data=str(run/'data/training_dataset/dataset.yaml'), project=str(run/'results'), name='learned_detector',
                            exist_ok=False, plots=False, val=True, cache=False, **{k:protocol[k] for k in keys})
    best = run/'results/learned_detector/weights/best.pt'
    write(run/'results/CHECKPOINT_FREEZE.json', {'checkpoint':str(best), 'checkpoint_sha256':sha(best),
        'training_manifest_sha256':sha(manifest), 'pretrained_sha256':sha(weights),
        'script_sha256':sha(__file__), 'protocol_sha256':sha(run/'coordination/TRAINING_PROTOCOL.json'),
        'evaluation_settings':protocol['evaluation'], 'test_unsealed':False,
        'selection':'best validation mAP50-95 from the single fixed training run'})


def evaluate(run, dataset, split):
    import numpy as np
    from ultralytics import YOLO
    freeze = json.loads((run/'results/CHECKPOINT_FREEZE.json').read_text())
    for path, expected in [(freeze['checkpoint'], freeze['checkpoint_sha256']), (__file__, freeze['script_sha256']),
                           (run/'coordination/TRAINING_PROTOCOL.json',freeze['protocol_sha256']),
                           (run/'data/training_dataset/manifest.json',freeze['training_manifest_sha256'])]:
        if sha(path) != expected:
            raise ValueError('Frozen input changed: '+str(path))
    settings = freeze['evaluation_settings']
    destination = run/'results'/f'{split}.evaluation'
    start = run/'results'/'TEST_EVALUATION_START.json'
    if destination.exists() or (split == 'test' and start.exists()):
        raise ValueError('One-shot evaluation already attempted; preserve outputs')
    dependencies_path = run/'results/EVALUATION_DEPENDENCIES.json'
    dependencies = json.loads(dependencies_path.read_text())
    if dependencies['checkpoint_freeze_sha256'] != sha(run/'results/CHECKPOINT_FREEZE.json'):
        raise ValueError('Evaluation dependencies bind a different checkpoint freeze')
    for path, expected in dependencies['code_hashes'].items():
        if sha(path) != expected:
            raise ValueError('Frozen evaluation dependency changed: '+path)
    manifest = json.loads((dataset/'manifest.json').read_text())
    if split == 'test':
        validate_test_manifest(json.loads((run/'data/training_dataset/manifest.json').read_text()), manifest)
        selection_path = run/'results/TEST_EVALUATION_PLAN.json'
        selection = json.loads(selection_path.read_text())
        validate_test_selection(selection, manifest, sha(dataset/'manifest.json'), dependencies['planned_test_ids'])
        write(start, {'dataset_manifest_sha256':sha(dataset/'manifest.json'), 'checkpoint_sha256':freeze['checkpoint_sha256'],
                      'evaluation_dependencies_sha256':sha(dependencies_path),
                      'test_selection_sha256':sha(selection_path),
                      'evaluation_settings':settings, 'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    model = YOLO(freeze['checkpoint'])
    if model.names != {0: 'clubhead'}:
        raise ValueError('Expected the one-class clubhead checkpoint')
    rows = []
    for entry in manifest['accepted']:
        if entry['split'] != split:
            continue
        for f in entry['frames']:
            path = dataset/'images'/split/f['image_name']
            if sha(path) != f['image_sha256']:
                raise ValueError('Evaluation image changed')
            text = (dataset/'labels'/split/f['label_name']).read_text()
            if text != f['label_text']:
                raise ValueError('Evaluation label changed')
            truth = None
            if text:
                _, x, y, w, h = map(float, text.split());width,height=f['width'],f['height']
                truth=[(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height]
            pred = model.predict(str(path), imgsz=settings['imgsz'], conf=settings['prediction_conf_floor'],
                                 iou=settings['nms_iou'], max_det=settings['max_det'], device='mps', verbose=False)[0]
            detections = [{'xyxy':b.xyxy[0].cpu().tolist(),'confidence':float(b.conf[0].cpu())} for b in pred.boxes]
            rows.append({'clip':entry['video_id'],'frame':f['index'],'image_sha256':f['image_sha256'],'truth':truth,'predictions':detections})
    ids = sorted({r['clip'] for r in rows})
    per = {cid:metrics([r for r in rows if r['clip']==cid],settings) for cid in ids}
    ap = [v['AP50'] for v in per.values() if v['AP50'] is not None]
    boot = np.random.default_rng(17).choice(ap,(4000,len(ap)),replace=True).mean(axis=1) if ap else []
    output={'schema':'fairway.author-box-evaluation.v1','split':split,'checkpoint_sha256':freeze['checkpoint_sha256'],
            'evaluation_dependencies_sha256':sha(dependencies_path),
            'settings':settings,'dataset_manifest_sha256':sha(dataset/'manifest.json'),'pooled':metrics(rows,settings), 'per_clip':per,
            'macro_AP50':float(np.mean(ap)) if ap else None,
            'macro_AP50_clip_bootstrap_95_percent':np.quantile(boot,[.025,.975]).tolist() if len(boot) else None,
            'claims':{'production':False,'independent_human_alignment_verification':False,'physical_core_localization':False,
                      'checkpoint_selected_on_this_split':split=='val', 'statistical_heldout':False},
            'bootstrap_clip_counts':{'included_positive_clips':len(ap),'excluded_no_positive_clips':len(per)-len(ap)},
            'limitations':'Small source-group sample; AI alignment sampling only; bootstrap interval is unstable with few clips; no retuning on test.'}
    staging = Path(tempfile.mkdtemp(prefix=f'.{split}.evaluation-', dir=run/'results'))
    try:
        write(staging/'predictions.json', rows)
        write(staging/'metrics.json', output)
        if destination.exists():
            raise ValueError('Evaluation destination appeared during inference')
        os.replace(staging, destination)
    finally:
        if staging.exists(): shutil.rmtree(staging)
    print(json.dumps(output,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['train','evaluate']);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--dataset',type=Path);parser.add_argument('--split',choices=['val','test'],default='val')
    args=parser.parse_args()
    if args.action=='train': train(args.run)
    else: evaluate(args.run,args.dataset or args.run/'data/training_dataset',args.split)
