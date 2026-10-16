"""All-shot seed-only SAM2 experiment with automatic negative-ball prompts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

RUN = Path('~/fairway-os-runs/tiger-all-clips-cycle')
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from ghostcaddie.demo.segmented_head import MaskHeadPolicy, PROFILE as REGION_PROFILE
from ghostcaddie.demo.seed_prompt import derive_head_prompts, PROFILE as PROMPT_PROFILE
from ghostcaddie.demo import target_benchmark as bench

PROFILE = {'model': 'SAM2.1 Hiera tiny', 'crop_radius_native_px': 112,
           'seed_box_radius_native_px': 7, 'point_prompt': True,
           'mask_logit_threshold': 0, 'max_candidates': 1,
           'search_radius_px': 24, 'min_score': .55,
           'seed_identity_min_score': .45, 'reacq_min_score': .70,
           'ambiguity_margin': .06,
           'loss_frames': 3, 'recovery_elapsed_frames': 24,
           'confirm_frames': 2, 'local_step_px': 30,
           'recovery_anchor_bound_px': 65, 'research_only': True,
           'production_eligible': False, 'future_labels_used': False,
           'negative_prompts': [], 'corrections': []}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--run-dir', type=Path, default=RUN)
    parser.add_argument('--runtime-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', choices=['mps','cpu'])
    parser.add_argument('--negative-prompts', action='store_true')
    parser.add_argument('--prompt-module',type=Path)
    parser.add_argument('--dataset', type=Path, default=Path('~/fairway-os-runs/tiger-tracking-cycle/labels/tiger_targets.v1.json'))
    parser.add_argument('--baseline', type=Path, default=Path('~/fairway-os-runs/tiger-redetection-cycle/results/previous_selected.predictions.json'))
    parser.add_argument('--out', type=Path, default=RUN/'results/all_sam2.predictions.json')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    runtime_dir = (args.runtime_dir or Path('~/fairway-os-runs/tiger-redetection-cycle/runtime')).resolve()
    checkpoint = (args.checkpoint or Path('~/fairway-os-runs/tiger-redetection-cycle/models/sam2.1_hiera_tiny.pt')).resolve()
    if args.out.resolve() == args.baseline.resolve():
        raise ValueError('Output must not overwrite the baseline')
    if args.out.resolve() == args.dataset.resolve():
        raise ValueError('Output must not overwrite the reference dataset')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw_out = args.out.with_name(args.out.name.replace('.predictions.json', '') + '.raw.json')
    storage_name = args.out.stem.replace('.predictions', '')
    frames_root = run/'sam2_head_frames'/storage_name
    masks_root = run/'sam2_head_masks'/storage_name
    evidence = run/'inference_evidence'/storage_name
    receipt_path = args.out.with_name(storage_name+'.mask_receipt.json')
    write_paths = [args.out,raw_out,receipt_path,Path(str(args.out)+'.report.json'),Path(str(args.out)+'.report.md')]
    if any(path.resolve() in {args.baseline.resolve(),args.dataset.resolve()} for path in write_paths):
        raise ValueError('Output artifacts must not overwrite reference or baseline inputs')
    for path in write_paths+[frames_root,masks_root,evidence]:
        if path.exists():
            raise ValueError(f'Inference evidence already exists: {path}; choose a fresh output name')
    sys.path[:0] = [str(runtime_dir/'deps'), str(runtime_dir/'sam2')]
    import cv2
    import numpy as np
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    prompt_function=derive_head_prompts;prompt_profile=PROMPT_PROFILE
    prompt_path=REPO/'ghostcaddie/demo/seed_prompt.py'
    if args.prompt_module:
        import importlib.util
        prompt_path=args.prompt_module.resolve()
        spec=importlib.util.spec_from_file_location('frozen_head_prompt',prompt_path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        prompt_function=module.derive_head_prompts;prompt_profile=module.PROFILE
    reference, reference_sha = bench.load_dataset(args.dataset, verify_hashes=True)
    if any(path.resolve() in {Path(c['path']).resolve() for c in reference['clips']} for path in write_paths):
        raise ValueError('Output must not overwrite a source clip')
    baseline = bench.validate_predictions(json.loads(args.baseline.read_text()), reference)
    payload = json.loads(args.baseline.read_text())
    payload['variant'] = 'selected_cohort_sam2_head'
    args.ids = args.ids or [c['id'] for c in reference['clips']]
    if len(set(args.ids)) != len(args.ids):
        raise ValueError('Duplicate clip IDs')
    if not set(args.ids) <= {c['id'] for c in reference['clips']}:
        raise ValueError('SAM2 experiment accepts only declared collection IDs')
    PROFILE['region_profile'] = dict(REGION_PROFILE)
    PROFILE['inactive_probability_thresholds'] = REGION_PROFILE['inactive_probability_thresholds']
    PROFILE['exported_confidence'] = None
    PROFILE['negative_prompt_profile'] = dict(prompt_profile) if args.negative_prompts else None
    PROFILE['crop_policy'] = 'putt: seed-centered225px; full_swing: native full-height rectangle of width0.95*height centered on seed, clamped'
    PROFILE['negative_prompts'] = 'automatically derived on seed image only' if args.negative_prompts else []
    PROFILE['full_swing_motion_policy'] = {'search_radius_px':120,'max_step_px':180,'reacq_radius_px':240,'reacq_max_step_px':240,'native_scale_basis':'height/720'}
    PROFILE['policy_scope'] = 'putt defaults above; full-swing motion fields override them with height/720 scaling; actual per-clip settings below are authoritative'
    # Explicit inference-only manifest: no target labels are retained or consulted.
    inputs = [{'id':c['id'], 'path':c['path'], 'sha256':c['sha256'],
               'seed':c['targets']['clubhead']['seed'], 'stroke_type':c['stroke_type'], 'window':c['window'],
               'cut_frames':c['cut_frames']} for c in reference['clips'] if c['id'] in args.ids]
    # Model functions receive only inputs above, never reference labels.
    device = args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    predictor = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',
        str(checkpoint), device=device, apply_postprocessing=False)
    metadata = {'profile':PROFILE, 'device':device, 'torch':torch.__version__,
                'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime_dir/'sam2',text=True).strip(),
                'official_source':'https://github.com/facebookresearch/sam2',
                'inference_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'negative_prompt_sha256':hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
                'head_region_sha256':hashlib.sha256((REPO/'ghostcaddie/demo/segmented_head.py').read_bytes()).hexdigest(),
                'point_policy_sha256':hashlib.sha256((REPO/'ghostcaddie/demo/foreground_tracker.py').read_bytes()).hexdigest(),
                'scores_are_binary_gate_indicators_not_model_confidence':True,'clips':{}}
    metadata['benchmark_code_sha256'] = bench.sha256_file(REPO/'ghostcaddie/demo/target_benchmark.py')
    metadata['prompt_image_encoding'] = 'seed crop JPEG quality100, identical to the crop image SAM2 reads'
    metadata['reference_dataset_sha256'] = reference_sha
    metadata['repository_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    metadata['repository_status'] = subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).splitlines()
    metadata['actual_engine_by_clip'] = {c['id']:('SAM2 head' if c['id'] in args.ids else 'classic baseline replay') for c in reference['clips']}
    evidence.mkdir(parents=True)
    for name,path in {'runner.py':Path(__file__),'head_region.py':REPO/'ghostcaddie/demo/segmented_head.py','point_policy.py':REPO/'ghostcaddie/demo/foreground_tracker.py','seed_prompt.py':prompt_path}.items():
        (evidence/name).write_bytes(path.read_bytes())
    metadata['source_snapshot_directory'] = str(evidence)
    for clip in inputs:
        started=time.monotonic()
        if bench.sha256_file(Path(clip['path'])) != clip['sha256']:
            raise ValueError('Source clip checksum changed')
        seed=clip['seed'];lo=int(seed['frame']);hi=int(clip['window']['end_frame'])
        cap=cv2.VideoCapture(clip['path']);width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        radius=PROFILE['crop_radius_native_px'];x0=max(0,round(seed['x'])-radius);y0=max(0,round(seed['y'])-radius)
        x1=min(width,round(seed['x'])+radius+1);y1=min(height,round(seed['y'])+radius+1)
        policy_settings = None
        if clip['stroke_type'] != 'putt':
            crop_width = min(width, round(height*.95))
            x0 = max(0,min(width-crop_width,round(seed['x'])-crop_width//2))
            x1 = x0+crop_width;y0=0;y1=height
            scale = height/720
            policy_settings = {k:PROFILE['full_swing_motion_policy'][k]*scale for k in ['search_radius_px','max_step_px','reacq_radius_px','reacq_max_step_px']}
        directory=frames_root/clip['id']
        directory.mkdir(parents=True)
        # Decode in source order so codec seeking cannot shift a nonzero seed.
        for frame in range(0,hi+1):
            ok,img=cap.read()
            if not ok:
                cap.release()
                raise ValueError(f'Cannot decode requested source frame {frame}')
            if frame < lo:
                continue
            cv2.imwrite(str(directory/f'{frame-lo:05d}.jpg'),img[y0:y1,x0:x1],[cv2.IMWRITE_JPEG_QUALITY,100])
        cap.release()
        if len(list(directory.glob('*.jpg'))) != hi-lo+1:
            raise ValueError('Extracted frame count differs from requested window')
        with torch.inference_mode():
            state=predictor.init_state(str(directory),offload_video_to_cpu=True,offload_state_to_cpu=True)
            px,py=seed['x']-x0,seed['y']-y0;br=PROFILE['seed_box_radius_native_px']
            prompt_metadata = None
            prompt_patch_origin = None
            model_points = [[px,py]];model_labels=[1]
            box=[max(0,px-br),max(0,py-br),min(x1-x0-1,px+br),min(y1-y0-1,py+br)]
            if args.negative_prompts:
                image=cv2.imread(str(directory/'00000.jpg'))
                qx=max(0,round(px)-48);qy=max(0,round(py)-48)
                prompt_patch_origin=[qx,qy]
                local=image[qy:min(image.shape[0],round(py)+49),qx:min(image.shape[1],round(px)+49)]
                prompts=prompt_function(local,px-qx,py-qy)
                prompt_metadata=prompts.to_dict()
                for nx,ny in prompts.negative_points:
                    model_points.append([nx+qx,ny+qy]);model_labels.append(0)
            _,_,initial=predictor.add_new_points_or_box(state,frame_idx=0,obj_id=1,
                points=np.array(model_points,dtype=np.float32),labels=np.array(model_labels,dtype=np.int32),
                box=np.array(box,dtype=np.float32))
            seed_mask=(initial[0,0]>0).cpu().numpy()
            head = MaskHeadPolicy(seed,[x0,y0],clip['cut_frames'],point_policy=policy_settings)
            raw=[];points=[]
            maskdir=masks_root/clip['id']
            maskdir.mkdir(parents=True)
            print(clip['id'],'seed mask area',int(seed_mask.sum()),'preview not used for admission',flush=True)
            for index,ids,logits in predictor.propagate_in_video(state):
                frame=lo+index
                mask=(logits[0,0]>0).cpu().numpy()
                cv2.imwrite(str(maskdir/f'f{frame:04d}.png'),mask.astype(np.uint8)*255)
                point, record = head.step(frame,mask)
                points.append(point)
                raw.append(record)
            metadata['clips'][clip['id']]={'input':clip,'crop':[x0,y0,x1-x0,y1-y0],
                'seed_parts':[] if head.seed_region is None else [head.seed_region],'seed_valid':head.seed_valid,
                'prompt_preview_mask_area':int(seed_mask.sum()),'points':points,'raw':raw,
                'negative_prompt_metadata':prompt_metadata,'prompt_patch_origin_crop':prompt_patch_origin,'model_points_crop':model_points,
                'model_box_crop_xyxy':box,'prompt_metadata_coordinate_space':'local seed-image patch; add prompt_patch_origin_crop to obtain model crop coordinates',
                'model_point_labels':model_labels,'actual_policy_settings':head.policy_settings,
                'seconds':time.monotonic()-started}
        raw_out.write_text(json.dumps(metadata,indent=2)+'\n')
        print(clip['id'],'complete',round(time.monotonic()-started,2),'seconds',flush=True)
        block = next(c for c in payload['clips'] if c['id']==clip['id'])['targets']['clubhead']
        block['points'] = [p for p in block['points'] if p['source_frame'] < lo] + points

    receipt = {'model_record':str(raw_out.resolve()),'model_record_sha256':bench.sha256_file(raw_out),'clips':{}}
    for cid,record in metadata['clips'].items():
        first=record['input']['seed']['frame'];last=record['input']['window']['end_frame']
        receipt['clips'][cid]={'input':record['input'],'crop':record['crop'],'masks':{str(f):{'path':str((masks_root/cid/f'f{f:04d}.png').resolve()),'sha256':bench.sha256_file(masks_root/cid/f'f{f:04d}.png')} for f in range(first,last+1)}}
    receipt_path.write_text(bench.canonical_json(receipt))
    predicted = bench.validate_predictions(payload,reference)
    args.out.write_text(bench.canonical_json(payload))
    runtime = {'interpreter':sys.executable,'python':sys.version.split()[0],
               'model':metadata['profile']['model'],'device':device,
               'negative_prompt_sha256':metadata['negative_prompt_sha256'],
               'tool_sha256':metadata['inference_code_sha256'],
               'point_policy_sha256':metadata['point_policy_sha256'],
               'opencv':cv2.__version__,'numpy':np.__version__,
               'head_region_sha256':metadata['head_region_sha256'],
               'checkpoint_sha256':metadata['checkpoint_sha256'],
               'model_source_commit':metadata['source_commit'],
               'profile':PROFILE,'raw_path':str(raw_out),'raw_sha256':bench.sha256_file(raw_out),'reference_inputs_used_by_model':['seed','window','cut_frames'],
               'labels_read_by_model':False,'baseline_replayed':True,
               'actual_engine_by_clip':metadata['actual_engine_by_clip'],
               'mask_receipt_path':str(receipt_path),'mask_receipt_sha256':bench.sha256_file(receipt_path),
               'baseline_predictions_sha256':bench.sha256_file(args.baseline),
               'note':'Classic target options retained for baseline comparison; head inference uses the declared learned-mask profile. Legacy observed_template is an automatic-observation status, not a claim of template matching. Scores are binary region gates, not confidence.'}
    report = bench.evaluate(reference,predicted,dataset_path=str(args.dataset),
        dataset_sha256=reference_sha,predictions_path=str(args.out),
        predictions_sha256=bench.sha256_file(args.out),runtime=runtime)
    Path(str(args.out)+'.report.json').write_text(bench.canonical_json(report))
    Path(str(args.out)+'.report.md').write_text(bench.render_markdown(report))
    for split in ['development','evaluation']:
        m=report['splits'][split]['targets']['clubhead']
        print(split,{k:m[k] for k in ['correct_within_tolerance','incorrect_visible_count','miss_count']},flush=True)

if __name__=='__main__':main()
