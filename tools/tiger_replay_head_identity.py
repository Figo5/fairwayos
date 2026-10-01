#!/usr/bin/env python3
"""Measure heads causally from checksum-verified SAM2 masks, without new prompts.

This separates an expensive mask run from the head measurement policy. It does
not read the model run's old decisions or head coordinates. Only native masks,
initial seeds, current-frame RGB pixels, crop origins, windows and cuts enter the policy. No assisted path or later coordinate enters it.
"""
import argparse
import json
from pathlib import Path
import sys
import subprocess

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ghostcaddie.demo import target_benchmark as bench
from ghostcaddie.demo.appearance_head import AppearanceHeadPolicy as MaskHeadPolicy, PROFILE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--mask-receipt',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--frames-receipt',type=Path,required=True)
    parser.add_argument('--ids',nargs='+')
    args=parser.parse_args()
    if not args.out.name.endswith('.predictions.json'):
        raise ValueError('Output name must end with .predictions.json to keep raw evidence separate')
    import cv2
    import numpy as np
    raw_out=args.out.with_name(args.out.name.replace('.predictions.json','')+'.raw.json')
    writes=[args.out,raw_out,Path(str(args.out)+'.report.json'),Path(str(args.out)+'.report.md')]
    if any(p.exists() for p in writes):raise ValueError('Choose a fresh output name to preserve previous experiments')
    if any(p.resolve() in {x.resolve() for x in [args.baseline,args.dataset,args.mask_receipt,args.frames_receipt]} for p in writes):
        raise ValueError('Output must preserve inputs')
    dataset,dataset_sha=bench.load_dataset(args.dataset,verify_hashes=True)
    if any(p.resolve() in {Path(c['path']).resolve() for c in dataset['clips']} for p in writes):
        raise ValueError('Output must not overwrite a source clip')
    payload=json.loads(args.baseline.read_text())
    bench.validate_predictions(payload,dataset)
    payload['variant']='declared_cohort_mask_core_appearance'
    receipt=json.loads(args.mask_receipt.read_text())
    source_record=Path(receipt['model_record'])
    if bench.sha256_file(source_record)!=receipt['model_record_sha256']:
        raise ValueError('Model inference record changed')
    model=json.loads(source_record.read_text())
    frames=json.loads(args.frames_receipt.read_text())
    if frames['mask_receipt_sha256']!=bench.sha256_file(args.mask_receipt):raise ValueError('Frame receipt belongs to different masks')
    clips={c['id']:c for c in dataset['clips']}
    if set(receipt['clips'])!={c['id'] for c in dataset['clips']}:
        raise ValueError('All-shot replay must cover every declared clip')
    ids=set(args.ids or clips)
    if not ids<=set(clips) or (args.ids and len(ids)!=len(args.ids)):raise ValueError('Duplicate or undeclared measurement IDs')
    if set(frames['clips'])!=set(clips):raise ValueError('Frame receipt must cover the complete collection')
    measured={}
    for cid,entry in receipt['clips'].items():
        if cid not in ids:continue
        source=clips[cid]
        expected={'id':cid,'path':source['path'],'sha256':source['sha256'],
                  'seed':source['targets']['clubhead']['seed'],'stroke_type':source['stroke_type'],
                  'window':source['window'],'cut_frames':source['cut_frames']}
        if (entry['input']!=expected or model['clips'][cid]['input']!=expected
                or entry['crop']!=model['clips'][cid]['crop']):
            raise ValueError('Mask inputs differ from frozen collection')
        frame_entry=frames['clips'][cid]
        if frame_entry['input']!=expected or frame_entry['crop']!=entry['crop']:raise ValueError('Frame inputs differ from mask collection')
        seed=expected['seed'];first=seed['frame'];last=expected['window']['end_frame']
        if set(entry['masks'])!={str(f) for f in range(first,last+1)}:
            raise ValueError('Masks must cover the complete seeded window exactly')
        if set(frame_entry['frames'])!=set(entry['masks']):raise ValueError('Frame/mask coverage differs')
        head=MaskHeadPolicy(seed,entry['crop'][:2],expected['cut_frames'],stroke_type=expected['stroke_type'],point_policy=model['clips'][cid]['actual_policy_settings'])
        points=[];raw=[]
        for frame in range(first,last+1):
            item=entry['masks'][str(frame)];path=Path(item['path'])
            if bench.sha256_file(path)!=item['sha256']:raise ValueError('Mask bytes changed')
            image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape!=(entry['crop'][3],entry['crop'][2]):
                raise ValueError('Mask dimensions differ from native crop')
            if not np.isin(image,[0,255]).all():raise ValueError('Masks must be binary')
            frame_item=frame_entry['frames'][str(frame)];frame_path=Path(frame_item['path'])
            if bench.sha256_file(frame_path)!=frame_item['sha256']:raise ValueError('Frame pixels changed')
            bgr=cv2.imread(str(frame_path))
            if bgr is None or bgr.shape!=(entry['crop'][3],entry['crop'][2],3):raise ValueError('Native frame dimensions differ from mask crop')
            point,decision=head.step(frame,image>0,bgr)
            points.append(point);raw.append(decision)
        block=next(c for c in payload['clips'] if c['id']==cid)['targets']['clubhead']
        if any(p['origin']=='automatic' for p in block['points'] if p['source_frame']<first):
            raise ValueError('Classic block contains an automatic point before its seed')
        block['points']=[p for p in block['points'] if p['source_frame']<first]+points
        measured[cid]={'input':expected,'crop':entry['crop'],
                      'seed_valid':head.seed_valid,'actual_policy_settings':head.policy_settings,'seed_threshold':head.threshold,
                      'seed_mode':head.seed_mode,'seed_descriptor':head.seed_descriptor,'seed_parts':[] if head.seed_region is None else [head.seed_region],
                      'points':points,'raw':raw}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    raw_out=args.out.with_name(args.out.name.replace('.predictions.json','')+'.raw.json')
    raw_payload={'profile':PROFILE,'clips':measured,
                 'mask_receipt_sha256':bench.sha256_file(args.mask_receipt),
                 'model_record_sha256':bench.sha256_file(source_record),'frames_receipt_sha256':bench.sha256_file(args.frames_receipt)}
    raw_out.write_text(bench.canonical_json(raw_payload))
    args.out.write_text(bench.canonical_json(payload))
    runtime={'research_only':True,'production_eligible':False,
        'tool_sha256':bench.sha256_file(Path(__file__)),
        'benchmark_code_sha256':bench.sha256_file(Path(__file__).resolve().parents[1]/'ghostcaddie/demo/target_benchmark.py'),
        'point_defaults_module_sha256':bench.sha256_file(Path(__file__).resolve().parents[1]/'ghostcaddie/demo/segmented_head.py'),
        'repository_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),
        'repository_status':subprocess.check_output(['git','status','--porcelain'],cwd=Path(__file__).resolve().parents[1],text=True).splitlines(),
        'head_region_sha256':bench.sha256_file(Path(__file__).resolve().parents[1]/'ghostcaddie/demo/appearance_head.py'),
        'identity_module_sha256':bench.sha256_file(Path(__file__).resolve().parents[1]/'ghostcaddie/demo/head_identity.py'),
        'geometry_module_sha256':bench.sha256_file(Path(__file__).resolve().parents[1]/'ghostcaddie/demo/adaptive_head.py'),
        'point_policy_sha256':bench.sha256_file(Path(__file__).resolve().parents[1]/'ghostcaddie/demo/foreground_tracker.py'),
        'opencv':cv2.__version__,'numpy':np.__version__,'profile':PROFILE,
        'raw_path':str(raw_out),'raw_sha256':bench.sha256_file(raw_out),
        'mask_receipt_path':str(args.mask_receipt),'mask_receipt_sha256':bench.sha256_file(args.mask_receipt),
        'model_record_path':str(source_record),'model_record_sha256':bench.sha256_file(source_record),
        'model_inference':{k:model[k] for k in ['device','torch','checkpoint_sha256','source_commit','official_source','inference_code_sha256','point_policy_sha256']},
        'baseline_predictions_sha256':bench.sha256_file(args.baseline),
        'baseline_replayed':True,'model_rerun':False,'actual_engine_by_clip':{cid:('SAM2 mask-core appearance' if cid in ids else 'baseline replay') for cid in clips},'frames_receipt_path':str(args.frames_receipt),'frames_receipt_sha256':bench.sha256_file(args.frames_receipt),'physical_identity_established':False,
        'labels_read_by_head_policy':False,'seed_admission_uses_first_forward_mask':True,
        'exported_confidence':None,
        'note':'Masks from archived actual inference reused; head policy recomputed in forward order. Immediate prompt preview never selects capability. Classic ball blocks replayed; declared measured head windows recomputed, outside cohort retains baseline. Core descriptor comes from original seed and current native RGB only.'}
    report=bench.evaluate(dataset,bench.validate_predictions(payload,dataset),
        dataset_path=str(args.dataset),dataset_sha256=dataset_sha,
        predictions_path=str(args.out),predictions_sha256=bench.sha256_file(args.out),runtime=runtime)
    Path(str(args.out)+'.report.json').write_text(bench.canonical_json(report))
    Path(str(args.out)+'.report.md').write_text(bench.render_markdown(report))


if __name__=='__main__':main()
