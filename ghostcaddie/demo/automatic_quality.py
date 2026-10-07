"""Sparse, source-bound evaluation, never read by the video pipeline.

AI visual references are development aids, not independent ground truth.
Unknown labels do not contribute denominators. Explicit negatives, misses,
wrong-object observations and point errors are reported separately.
"""
import argparse
import json
import math
from pathlib import Path


def evaluate(references, predictions):
    if references.get('schema') != 'fairwayos-automatic-visual-reference.v1':
        raise ValueError('Unsupported reference schema')
    if references.get('provenance') != 'ai_visual_reference':
        raise ValueError('Reference provenance must be explicit')
    metrics={};details=[];seen=set()
    for label in references['labels']:
        clip=label['clip'];frame=label['frame'];obj=label['object']
        key=(clip,frame,obj)
        if key in seen:raise ValueError('Duplicate reference')
        seen.add(key)
        if obj not in ('ball','cup') or type(frame) is not int or frame<0:raise ValueError('Invalid object/frame')
        pred=predictions[clip]
        if pred['source_sha256'] != label['source_sha256']:raise ValueError('Source hash mismatch')
        rows=pred['observations']['frames']
        if frame >= len(rows) or rows[frame]['frame'] != frame:raise ValueError('Missing prediction frame')
        row=rows[frame];point=row['ball']['point'] if obj=='ball' and row['ball'] else row['target'] if obj=='cup' else None
        if point is not None and (len(point)!=2 or not all(math.isfinite(v) for v in point)):raise ValueError('Invalid prediction point')
        status=label['status']
        if status not in ('visible','absent','unknown'):raise ValueError('Invalid reference status')
        group=(label['split'],label['slice'],obj)
        m=metrics.setdefault(group,dict(visible=0,absent=0,unknown=0,matched=0,misses=0,wrong_locations=0,false_observations=0,reviewed_exclusion_regions=0,exclusion_hits=0,errors_px=[]))
        m[status]+=1;detail=dict(clip=clip,frame=frame,object=obj,status=status,predicted=point)
        box=label.get('excluded_region')
        if box is not None:
            if len(box)!=4 or not all(math.isfinite(v) for v in box) or not (box[0]<box[2] and box[1]<box[3]):raise ValueError('Invalid exclusion region')
            m['reviewed_exclusion_regions']+=1
            hit=point is not None and box[0]<=point[0]<=box[2] and box[1]<=point[1]<=box[3]
            m['exclusion_hits']+=int(hit);detail['excluded_region_hit']=hit
        if status=='visible':
            reference=label['point'];tol=label['tolerance_px']
            if (len(reference)!=2 or not all(math.isfinite(v) for v in reference)
                    or type(tol) not in (float,int) or not 0<tol<=50):raise ValueError('Invalid reference point/tolerance')
            if point is None:m['misses']+=1
            else:
                error=math.dist(point,reference);m['errors_px'].append(error);detail['error_px']=error
                m['matched' if error<=tol else 'wrong_locations']+=1
        elif status=='absent' and point is not None:m['false_observations']+=1
        details.append(detail)
    report=[]
    for group,m in sorted(metrics.items()):
        errors=m.pop('errors_px')
        report.append(dict(split=group[0],slice=group[1],object=group[2],**m,
            matched_visible_fraction=m['matched']/m['visible'] if m['visible'] else None,
            false_observation_fraction=m['false_observations']/m['absent'] if m['absent'] else None,
            max_point_error_px=max(errors) if errors else None))
    return {'schema':'fairwayos-automatic-quality.v1','reference_provenance':'ai_visual_reference',
            'independent_ground_truth':False,'production_accuracy_claim':False,
            'limitations':['Sparse samples do not establish full-video accuracy or identity-switch/drift rates',
                           'Same-event held-out clips do not measure independent-camera generalization'],
            'metrics':report,'details':details}


def main():
    p=argparse.ArgumentParser();p.add_argument('references');p.add_argument('predictions');p.add_argument('output');a=p.parse_args()
    result=evaluate(json.loads(Path(a.references).read_text()),json.loads(Path(a.predictions).read_text()))
    Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
