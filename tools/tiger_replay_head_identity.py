#!/usr/bin/env python3
"""Measure heads causally from checksum-verified SAM2 masks, without new prompts.

This separates an expensive mask run from the head measurement policy. It does
not read the model run's old decisions or head coordinates. Only native masks,
initial seeds, current-frame RGB pixels, crop origins, windows and cuts enter the policy. No assisted path or later coordinate enters it.
"""
import argparse
import hashlib
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
    parser.add_argument('--baseline-report',type=Path,
                        help='Required for swing: executed report binding baseline prediction bytes')
    parser.add_argument('--mask-receipt',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--frames-receipt',type=Path,required=True)
    parser.add_argument('--body-receipt',type=Path,help='Checksum-bound current-frame native MoveNet body measurements')
    parser.add_argument('--ids',nargs='+')
    parser.add_argument('--policy', choices=('appearance', 'swing', 'swing_context'), default='appearance',
                        help='Swing measures the complete declared full-swing cohort only')
    args=parser.parse_args()
    if args.policy.startswith('swing') and args.baseline_report is None:
        raise ValueError('Swing replay requires the executed baseline report')
    policy_class, profile = MaskHeadPolicy, PROFILE
    policy_path = Path(__file__).resolve().parents[1]/'ghostcaddie/demo/appearance_head.py'
    engine_name = 'SAM2 mask-core appearance'
    if args.policy.startswith('swing'):
        from ghostcaddie.demo.swing_head import SwingHeadPolicy, PROFILE as SWING_PROFILE
        policy_class, profile = SwingHeadPolicy, SWING_PROFILE
        policy_path = Path(__file__).resolve().parents[1]/'ghostcaddie/demo/swing_head.py'
        engine_name = 'SAM2 component-adaptive swing cores'
    body_receipt = None
    if args.policy == 'swing_context':
        from ghostcaddie.demo.swing_context import ContextSwingHeadPolicy, PROFILE as CONTEXT_PROFILE
        policy_class, profile = ContextSwingHeadPolicy, CONTEXT_PROFILE
        policy_path = Path(__file__).resolve().parents[1]/'ghostcaddie/demo/swing_context.py'
        engine_name = 'SAM2 swing cores with current wrist and image contrast refusal'
        if args.body_receipt is None:
            raise ValueError('Context policy requires a checksum-bound body receipt')
        body_receipt=json.loads(args.body_receipt.read_text())
    if not args.out.name.endswith('.predictions.json'):
        raise ValueError('Output name must end with .predictions.json to keep raw evidence separate')
    import cv2
    import numpy as np
    raw_out=args.out.with_name(args.out.name.replace('.predictions.json','')+'.raw.json')
    writes=[args.out,raw_out,Path(str(args.out)+'.report.json'),Path(str(args.out)+'.report.md')]
    if any(p.exists() for p in writes):raise ValueError('Choose a fresh output name to preserve previous experiments')
    inputs=[args.baseline,args.dataset,args.mask_receipt,args.frames_receipt]
    if args.body_receipt is not None:inputs.append(args.body_receipt)
    if args.baseline_report is not None:inputs.append(args.baseline_report)
    if any(p.resolve() in {x.resolve() for x in inputs} for p in writes):
        raise ValueError('Output must preserve inputs')
    dataset,dataset_sha=bench.load_dataset(args.dataset,verify_hashes=True)
    if any(p.resolve() in {Path(c['path']).resolve() for c in dataset['clips']} for p in writes):
        raise ValueError('Output must not overwrite a source clip')
    payload=json.loads(args.baseline.read_text())
    bench.validate_predictions(payload,dataset)
    baseline_report=None
    if args.baseline_report is not None:
        baseline_report=json.loads(args.baseline_report.read_text())
        if baseline_report.get('predictions',{}).get('sha256')!=bench.sha256_file(args.baseline):
            raise ValueError('Executed baseline report does not bind baseline prediction bytes')
    payload['variant']=('full_swing_mask_core_research' if args.policy.startswith('swing')
                        else 'declared_cohort_mask_core_appearance')
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
    if args.policy.startswith('swing'):
        cohort={cid for cid,c in clips.items() if c['stroke_type']=='full_swing'}
        if args.ids and ids!=cohort:raise ValueError('Swing policy must measure every declared full swing')
        ids=cohort
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
        head=policy_class(seed,entry['crop'][:2],expected['cut_frames'],stroke_type=expected['stroke_type'],point_policy=model['clips'][cid]['actual_policy_settings'])
        body_frames={}
        if body_receipt is not None:
            item=body_receipt['clips'][cid];body_path=Path(item['analysis_path'])
            if bench.sha256_file(body_path)!=item['analysis_sha256']:
                raise ValueError('Body measurement bytes changed')
            body_record=json.loads(body_path.read_text())
            if body_record['source']['sha256']!=source['sha256'] or item['source_sha256']!=source['sha256']:
                raise ValueError('Body context belongs to a different source')
            if (body_record['source']['width'],body_record['source']['height'])!=(source['width'],source['height']):
                raise ValueError('Body context must use native source geometry')
            rows=[r for scene in body_record['scenes'] for r in scene['body']['per_frame']]
            body_frames={r['source_frame']:r for r in rows}
            if len(body_frames)!=len(rows) or not set(range(first,last+1))<=set(body_frames):
                raise ValueError('Body context must cover each current seeded frame exactly once')
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
            context={'body_frame':body_frames[frame]} if body_receipt is not None else {}
            point,decision=head.step(frame,image>0,bgr,**context)
            points.append(point);raw.append(decision)
        block=next(c for c in payload['clips'] if c['id']==cid)['targets']['clubhead']
        if any(p['origin']=='automatic' for p in block['points'] if p['source_frame']<first):
            raise ValueError('Classic block contains an automatic point before its seed')
        block['points']=[p for p in block['points'] if p['source_frame']<first]+points
        measured[cid]={'input':expected,'crop':entry['crop'],
                      'seed_valid':head.seed_valid,'actual_policy_settings':head.policy_settings,'seed_threshold':getattr(head,'threshold',None),
                      'seed_mode':getattr(head,'seed_mode',None),'seed_descriptor':head.seed_descriptor,'seed_parts':[] if head.seed_region is None else [head.seed_region],
                      'points':points,'raw':raw}
    raw_out=args.out.with_name(args.out.name.replace('.predictions.json','')+'.raw.json')
    raw_payload={'profile':profile,'clips':measured,
                 'mask_receipt_sha256':bench.sha256_file(args.mask_receipt),
                 'model_record_sha256':bench.sha256_file(source_record),'frames_receipt_sha256':bench.sha256_file(args.frames_receipt)}
    validated=bench.validate_predictions(payload,dataset)
    raw_text=bench.canonical_json(raw_payload)
    prediction_text=bench.canonical_json(payload)
    digest=lambda value:hashlib.sha256(value.encode('utf-8')).hexdigest()
    runtime={'research_only':True,'production_eligible':False,
        'tool_sha256':bench.sha256_file(Path(__file__)),
        'benchmark_code_sha256':bench.sha256_file(Path(__file__).resolve().parents[1]/'ghostcaddie/demo/target_benchmark.py'),
        'point_defaults_module_sha256':bench.sha256_file(Path(__file__).resolve().parents[1]/'ghostcaddie/demo/segmented_head.py'),
        'repository_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),
        'repository_status':subprocess.check_output(['git','status','--porcelain'],cwd=Path(__file__).resolve().parents[1],text=True).splitlines(),
        'head_region_sha256':bench.sha256_file(policy_path),
        'swing_measurement_base_sha256':bench.sha256_file(Path(__file__).resolve().parents[1]/'ghostcaddie/demo/swing_head.py'),
        'identity_module_sha256':bench.sha256_file(Path(__file__).resolve().parents[1]/'ghostcaddie/demo/head_identity.py'),
        'geometry_module_sha256':bench.sha256_file(Path(__file__).resolve().parents[1]/'ghostcaddie/demo/adaptive_head.py'),
        'point_policy_sha256':bench.sha256_file(Path(__file__).resolve().parents[1]/'ghostcaddie/demo/foreground_tracker.py'),
        'opencv':cv2.__version__,'numpy':np.__version__,'profile':profile,
        'raw_path':str(raw_out),'raw_sha256':digest(raw_text),
        'mask_receipt_path':str(args.mask_receipt),'mask_receipt_sha256':bench.sha256_file(args.mask_receipt),
        'model_record_path':str(source_record),'model_record_sha256':bench.sha256_file(source_record),
        'model_inference':{k:model[k] for k in ['device','torch','checkpoint_sha256','source_commit','official_source','inference_code_sha256','point_policy_sha256']},
        'baseline_predictions_sha256':bench.sha256_file(args.baseline),
        'baseline_replayed':True,'model_rerun':False,'actual_engine_by_clip':{cid:(engine_name if cid in ids else 'baseline replay') for cid in clips},'frames_receipt_path':str(args.frames_receipt),'frames_receipt_sha256':bench.sha256_file(args.frames_receipt),'physical_identity_established':False,
        'operator_annotations_used':['original_seed','stroke_type','window','cut_frames'],
        'labels_read_by_head_policy':False,'seed_admission_uses_first_forward_mask':True,
        'exported_confidence':None,
        'note':'Masks from archived actual inference reused; head policy recomputed in forward order. Immediate prompt preview never selects capability. Baseline ball blocks replayed unchanged; declared measured head windows recomputed, outside cohort retains baseline. Core descriptor comes from original seed and current native RGB only.'}
    if baseline_report is not None:
        runtime.update({'baseline_report_path':str(args.baseline_report.resolve()),
                        'baseline_report_sha256':bench.sha256_file(args.baseline_report),
                        'baseline_executed_runtime':baseline_report['runtime']})
    if body_receipt is not None:
        runtime.update({'body_receipt_path':str(args.body_receipt.resolve()),
                        'body_receipt_sha256':bench.sha256_file(args.body_receipt),
                        'body_context_provenance':body_receipt,
                        'additional_operator_annotations':['body_roi_keyframes_and_crop_interpolation'],
                        'current_body_measurements_only':True})
    report=bench.evaluate(dataset,validated,
        dataset_path=str(args.dataset),dataset_sha256=dataset_sha,
        predictions_path=str(args.out),predictions_sha256=digest(prediction_text),runtime=runtime)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    texts=[prediction_text,raw_text,bench.canonical_json(report),bench.render_markdown(report)]
    for path,contents in zip(writes,texts):
        with path.open('x',encoding='utf-8') as handle:handle.write(contents)


if __name__=='__main__':main()
