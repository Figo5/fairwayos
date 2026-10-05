#!/usr/bin/env python3
"""Replay frozen detector proposals with current body/shaft pixel support.

The measurement action reads no target reference dataset. It preserves previous
outputs and binds raw proposals, cached pixels, body rows and executed source.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.clubhead_detector_experiment import sha
from ghostcaddie.demo.proposal_tracking import ProposalHeadPolicy, METHOD, METHOD_REACQUIRED


def select_gaps(baseline, candidate, sources):
    output=copy.deepcopy(baseline)
    old={c['id']:c for c in output['clips']};new={c['id']:c for c in candidate['clips']}
    if set(old)!=set(new) or len(old)!=len(output['clips']) or len(new)!=len(candidate['clips']):
        raise ValueError('Routes must cover the same complete collection')
    if set(old)!={s['id'] for s in sources} or len(sources)!=len(old):
        raise ValueError('Source roster must cover each clip exactly once')
    additions=[]
    for source in sources:
        cid=source['id'];a=old[cid];b=new[cid]
        if a['sha256']!=b['sha256'] or a['sha256']!=source['sha256']:
            raise ValueError('Routes need identical source bytes')
        if a['targets']['ball']!=b['targets']['ball']:
            raise ValueError('Ball blocks must remain unchanged')
        if source['stroke_type']=='putt':
            if a!=b:raise ValueError('Putt heads must remain unchanged')
            continue
        if source['stroke_type']!='full_swing':raise ValueError('Unsupported stroke family')
        prior=a['targets']['clubhead'];current=b['targets']['clubhead']
        if prior['seed']!=current['seed'] or prior['seed'] is None:
            raise ValueError('Routes need identical original seeds')
        for block in (prior,current):
            for point in block['points']:
                if point['origin'] not in ('automatic','operator','derived'):
                    raise ValueError('Undeclared assistance cannot enter the route')
                if point['origin']=='operator' and point['source_frame']!=prior['seed']['frame']:
                    raise ValueError('Only original operator seeds may enter the route')
                if point['origin']=='derived' and (point['x'] is not None or point['y'] is not None):
                    raise ValueError('Derived coordinates cannot enter gap selection')
        before={p['source_frame']:p for p in prior['points']};after={p['source_frame']:p for p in current['points']}
        if len(before)!=len(prior['points']) or len(after)!=len(current['points']):raise ValueError('Duplicate frames')
        selected=[]
        for f in sorted(set(before)|set(after)):
            p=before.get(f);q=after.get(f)
            if p and p['origin'] in ('automatic','operator'):
                selected.append(copy.deepcopy(p));continue
            if q and q['origin']=='automatic':
                if (q['method'] not in (METHOD,METHOD_REACQUIRED) or q['score'] is not None
                        or q['status']!='observed_template'
                        or any(type(q[k]) not in (int,float) or not math.isfinite(q[k]) for k in ('x','y'))
                        or f<=prior['seed']['frame'] or f>source['window']['end_frame']
                        or any(prior['seed']['frame']<=cut<=f for cut in source['cut_frames'])):
                    raise ValueError('Only current confirmed in-window pixel measurements may fill gaps')
                selected.append(copy.deepcopy(q));additions.append({'id':cid,'point':copy.deepcopy(q)})
            else:
                point=p or q
                if point['origin']!='derived' or point['x'] is not None or point['y'] is not None:
                    raise ValueError('Nonoriginal assistance or interpolation cannot enter the route')
                selected.append(copy.deepcopy(point))
        prior['points']=selected
    output['variant']='proposal_support_gap_fill_research'
    return output,additions


def measure(run):
    import cv2
    import numpy as np
    from ghostcaddie.demo.proposal_head import PROFILE
    plan_path=run/'PLAN.json';plan=json.loads(plan_path.read_text())
    for p,h in plan['input_hashes'].items():
        if sha(p)!=h:raise ValueError('Frozen input changed: '+p)
    source_hashes={s['path']:s['sha256'] for s in plan['sources']}
    freeze=json.loads(Path(plan['input_paths']['checkpoint_freeze']).read_text())
    source_hashes[freeze['checkpoint']]=freeze['checkpoint_sha256']
    for p,h in source_hashes.items():
        if sha(p)!=h:raise ValueError('Original source or checkpoint changed: '+p)
    context_path=Path(plan['body_context_path'])
    if sha(context_path)!=plan['body_context_sha256']:raise ValueError('Body context changed')
    context=json.loads(context_path.read_text());frames=json.loads(Path(plan['input_paths']['frames']).read_text())
    proposals=json.loads(Path(plan['input_paths']['raw_detector']).read_text());baseline=json.loads(Path(plan['input_paths']['baseline']).read_text())
    ids={s['id'] for s in plan['sources']}
    if set(context['clips'])!=ids or set(frames['clips'])!=ids or set(proposals)!=ids or {c['id'] for c in baseline['clips']}!=ids:
        raise ValueError('Inputs must cover the complete frozen collection')
    dest=run/'results/replay'
    if dest.exists():raise ValueError('Preserve prior results; choose another run directory')
    files=[Path(__file__),ROOT/'ghostcaddie/demo/proposal_head.py',ROOT/'ghostcaddie/demo/proposal_tracking.py',
           ROOT/'ghostcaddie/demo/shaft_head.py',ROOT/'ghostcaddie/demo/head_identity.py',ROOT/'ghostcaddie/demo/swing_head.py',
           ROOT/'ghostcaddie/demo/tracking.py',ROOT/'tools/clubhead_detector_experiment.py']
    hashes={str(p):sha(p) for p in files};snapshots=run/'coordination/executed_code';snapshots.mkdir()
    for p in files:shutil.copyfile(p,snapshots/p.name)
    started={'started_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(plan_path),'body_context_sha256':sha(context_path),
             'code_hashes':hashes,'profile':PROFILE,'inference_settings_retuned':False,'reference_dataset_read':False,
             'runtime':{'opencv':cv2.__version__,'numpy':np.__version__,'python':sys.version}}
    (run/'results/MEASUREMENT_START.json').write_text(json.dumps(started,indent=2)+'\n')
    staging=Path(tempfile.mkdtemp(prefix='.proposal-replay-',dir=run/'results'))
    output=copy.deepcopy(baseline);output['variant']='proposal_supported_full_swing_research';clips={c['id']:c for c in output['clips']};counts={}
    try:
        with (staging/'measurements.ndjson').open('w') as rawfile:
            for source in plan['sources']:
                cid=source['id'];entry=frames['clips'][cid];crop=entry['crop'];body=context['clips'][cid]
                if source['sha256']!=body['source_sha256'] or entry['input']['sha256']!=source['sha256']:raise ValueError('Source mismatch')
                modelrows={r['source_frame']:r for r in proposals[cid]};poses={r['source_frame']:r for r in body['rows']}
                first=source['seed']['frame'];last=source['window']['end_frame'];expected=set(range(first,last+1))
                if set(modelrows)!=expected or len(modelrows)!=len(proposals[cid]) or set(poses)!=expected or len(poses)!=len(body['rows']):
                    raise ValueError('Every source frame needs one proposal and body row')
                policy=(ProposalHeadPolicy(source['seed'],crop[:2],source['cut_frames'],[source['width'],source['height']])
                        if source['stroke_type']=='full_swing' else None)
                points=[];n=0
                for frame in range(first,last+1):
                    item=entry['frames'][str(frame)];data=Path(item['path']).read_bytes();checksum=hashlib.sha256(data).hexdigest();row=modelrows[frame]
                    if checksum!=item['sha256'] or checksum!=row['image_sha256'] or row['crop_xywh']!=crop:raise ValueError('Proposal/pixel binding changed')
                    if any(d['class_id']!=0 for d in row['detections']):raise ValueError('Expected one clubhead class')
                    if not policy:continue
                    pixels=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
                    if pixels is None or pixels.shape!=(crop[3],crop[2],3):raise ValueError('Native crop geometry changed')
                    point,evidence=policy.step(frame,pixels,poses[frame],detections=row['detections'])
                    if point['origin']=='automatic' and [point['x'],point['y']]!=evidence['selected']['centroid_native']:
                        raise ValueError('Observation differs from current measured core')
                    if frame not in source['cut_frames']:points.append(point)
                    n+=point['origin']=='automatic'
                    rawfile.write(json.dumps({'id':cid,'source_frame':frame,'image_sha256':checksum,'point':point,'evidence':evidence})+'\n')
                if policy:
                    clips[cid]['targets']['clubhead'].pop('options',None);clips[cid]['targets']['clubhead']['points']=points
                counts[cid]={'source_frames_checked':len(expected),'physical_measurement_replayed':policy is not None,'confirmed_observations':n}
                print(cid,counts[cid],flush=True)
        selected,additions=select_gaps(baseline,output,plan['sources'])
        for name,obj in [('candidate.predictions.json',output),('selected.candidate.predictions.json',selected),('ADDITIONS.json',additions)]:
            (staging/name).write_text(json.dumps(obj,indent=2)+'\n')
        if any(sha(p)!=h for p,h in {**plan['input_hashes'],**source_hashes,**hashes,str(context_path):plan['body_context_sha256'],str(plan_path):started['plan_sha256']}.items()):
            raise ValueError('Executed input changed during replay')
        receipt={**started,'completed_utc':datetime.now(timezone.utc).isoformat(),'counts':counts,'addition_count':len(additions),'new_neural_inference':False,
                 'coordinates':'Current measured compact-core centroid, no bbox centers or synthesized endpoints','promotion':False,
                 'output_hashes':{name:sha(staging/name) for name in ['candidate.predictions.json','selected.candidate.predictions.json','ADDITIONS.json','measurements.ndjson']}}
        (staging/'REPLAY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');os.replace(staging,dest)
        print('Gap-fill candidates:',len(additions),flush=True)
    finally:
        if staging.exists():shutil.rmtree(staging)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    measure(p.parse_args().run)
