#!/usr/bin/env python3
"""Replay a frozen clubhead detector over checksum-bound cached native pixels.

Produces research proposals only. No reference labels, smoothing, gap filling,
seed adaptation, source edits, or selected-track changes occur here.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.clubhead_detector_experiment import sha
from ghostcaddie.demo.tracking import STATUS_OBSERVED


def seed_identity(seed):
    """Bind frame and coordinates; an optional receipt note is descriptive."""
    if not {'frame', 'x', 'y'} <= set(seed) or set(seed) - {'frame', 'x', 'y', 'note'}:
        raise ValueError('Seed requires frame/x/y and permits only a descriptive note')
    return {key: seed[key] for key in ('frame', 'x', 'y')}


def learned_point(frame, center):
    # The shared schema uses a legacy observed_template status for observations.
    # The method records the actual detector; no template matching occurs here.
    return {'source_frame': frame, **center, 'status': STATUS_OBSERVED,
            'origin': 'automatic', 'method': 'learned_bbox_center'}


def observation(frame, seed, center, cut_frames):
    if frame == seed['frame']:
        return {'source_frame': frame, 'x': seed['x'], 'y': seed['y'],
                'status': 'manual_anchor', 'origin': 'operator',
                'method': 'operator_placement', 'score': None}
    if center is not None and not any(seed['frame'] <= cut <= frame for cut in cut_frames):
        return learned_point(frame, center)
    return {'source_frame': frame, 'x': None, 'y': None, 'status': 'missing',
            'origin': 'derived', 'method': '', 'score': None}


def unique_center(detections, crop, threshold):
    eligible=[p for p in detections if p['confidence'] >= threshold]
    if len(eligible)!=1:
        return None
    b=eligible[0]['xyxy']
    return {'x':(b[0]+b[2])/2+crop[0], 'y':(b[1]+b[3])/2+crop[1], 'score':eligible[0]['confidence']}


def probe(args):
    import cv2
    import numpy as np
    from ultralytics import YOLO
    frozen=json.loads(args.freeze.read_text())
    if sha(frozen['checkpoint'])!=frozen['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint bytes changed')
    receipt=json.loads(args.frames_receipt.read_text());baseline=json.loads(args.baseline.read_text())
    clips={c['id']:c for c in baseline['clips']}
    if len(clips)!=len(baseline['clips']) or set(clips)!=set(receipt['clips']):
        raise ValueError('Inputs must cover same complete collection')
    if args.out.exists():
        raise ValueError('Preserve previous results; choose a fresh output directory')
    settings=frozen['evaluation_settings'];model=YOLO(frozen['checkpoint'])
    if model.names != {0: 'clubhead'}:
        raise ValueError('Expected the one-class clubhead checkpoint')
    inputs={str(p):sha(p) for p in (args.freeze,args.frames_receipt,args.baseline,Path(__file__))}
    dependencies=args.freeze.parent/'EVALUATION_DEPENDENCIES.json'
    if dependencies.exists():
        inputs[str(dependencies)]=sha(dependencies)
    output=copy.deepcopy(baseline);output['variant']='learned_bbox_center_research';raw={};counts={}
    for target in output['clips']:
        cid=target['id'];entry=receipt['clips'][cid];source=entry['input'];crop=entry['crop']
        if source['sha256']!=target['sha256'] or sha(source['path'])!=target['sha256']:
            raise ValueError('Original source bytes changed: '+cid)
        block=target['targets']['clubhead'];seed=block['seed']
        if seed_identity(seed)!=seed_identity(source['seed']):
            raise ValueError('Seed binding differs')
        first=seed['frame'];last=source['window']['end_frame']
        if set(entry['frames'])!={str(f) for f in range(first,last+1)}:
            raise ValueError('Receipt must cover each seeded frame')
        points=[];records=[];available=0
        for frame in range(first,last+1):
            item=entry['frames'][str(frame)];data=Path(item['path']).read_bytes()
            if hashlib.sha256(data).hexdigest()!=item['sha256']:
                raise ValueError('Frame bytes changed')
            pixels=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            if pixels is None or pixels.shape!=(crop[3],crop[2],3):
                raise ValueError('Native crop dimensions changed')
            pred=model.predict(pixels,imgsz=settings['imgsz'],conf=settings['prediction_conf_floor'],iou=settings['nms_iou'],
                               max_det=settings['max_det'],device='mps',verbose=False)[0]
            detections=[{'xyxy':b.xyxy[0].cpu().tolist(),'confidence':float(b.conf[0].cpu()),'class_id':int(b.cls[0].cpu())} for b in pred.boxes]
            if any(p['class_id']!=0 for p in detections):
                raise ValueError('Model is not the one-class clubhead checkpoint')
            center=unique_center(detections,crop,settings['operating_conf'])
            structural=any(first<=c<=frame for c in source['cut_frames'])
            point=observation(frame,seed,center,source['cut_frames'])
            if point['origin']=='automatic':
                available+=1
            if frame not in source['cut_frames']:
                points.append(point)
            records.append({'source_frame':frame,'image_sha256':item['sha256'],'crop_xywh':crop,'detections':detections,
                            'decision':'operator_seed_only' if frame==first else ('cut_suppressed' if structural else ('single_bbox_center' if center is not None else 'absent_or_ambiguous'))})
        block.pop('options',None);block['points']=points;raw[cid]=records
        counts[cid]={'frames_inferred':len(records),'single_head_proposals':available,'stroke_type':source['stroke_type']}
        print(cid,counts[cid],flush=True)
    if any(sha(p)!=h for p,h in inputs.items()) or sha(frozen['checkpoint'])!=frozen['checkpoint_sha256']:
        raise ValueError('Executed input changed during replay')
    args.out.mkdir()
    for name,obj in [('candidate.predictions.json',output),('candidate.raw.json',raw)]:
        (args.out/name).write_text(json.dumps(obj,indent=2)+'\n')
    evidence={str(args.out/name):sha(args.out/name) for name in ('candidate.predictions.json','candidate.raw.json')}
    receipt={'input_hashes':inputs,'output_hashes':evidence,'completed_utc':datetime.now(timezone.utc).isoformat(),
            'evaluation_dependencies_sha256':inputs.get(str(dependencies)),
            'dependency_receipt_scope':'Archived evaluation dependencies; live replay source is bound separately in input_hashes',
            'checkpoint_sha256':frozen['checkpoint_sha256'],'settings':settings,'counts':counts,'reference_labels_used':False,
            'selected_tracks_changed':False,'bbox_centers_are_physical_core_measurements':False,'model_confidence_is_calibrated':False,
            'putts':'included as out-of-training-domain diagnostics; no promotion', 'production':False}
    (args.out/'PROBE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('freeze','frames-receipt','baseline','out'):
        p.add_argument('--'+name,type=Path,required=True)
    probe(p.parse_args())
