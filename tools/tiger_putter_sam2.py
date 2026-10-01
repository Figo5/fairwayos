"""Bounded seed-only learned-mask probe; source references never feed the model."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import shutil

RUN = Path('/Users/giofiore/fairway-os-runs/tiger-redetection-cycle')
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from ghostcaddie.demo.segmented_head import MaskHeadPolicy, PROFILE as REGION_PROFILE
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
    parser.add_argument('--dataset', type=Path, default=Path('/Users/giofiore/fairway-os-runs/tiger-tracking-cycle/labels/tiger_targets.v1.json'))
    parser.add_argument('--baseline', type=Path, default=RUN/'results/previous_selected.predictions.json')
    parser.add_argument('--out', type=Path, default=RUN/'results/sam2_head.predictions.json')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    runtime_dir = (args.runtime_dir or run/'runtime').resolve()
    checkpoint = (args.checkpoint or run/'models/sam2.1_hiera_tiny.pt').resolve()
    if args.out.resolve() == args.baseline.resolve():
        raise ValueError('Output must not overwrite the baseline')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw_out = args.out.with_name(args.out.name.replace('.predictions.json', '') + '.raw.json')
    sys.path[:0] = [str(runtime_dir/'deps'), str(runtime_dir/'sam2')]
    import cv2
    import numpy as np
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    reference, reference_sha = bench.load_dataset(args.dataset, verify_hashes=True)
    if args.out.resolve() in {Path(c['path']).resolve() for c in reference['clips']}:
        raise ValueError('Output must not overwrite a source clip')
    baseline = bench.validate_predictions(json.loads(args.baseline.read_text()), reference)
    payload = json.loads(args.baseline.read_text())
    payload['variant'] = 'sam2_head_region'
    args.ids = args.ids or [c['id'] for c in reference['clips'] if c['stroke_type']=='putt']
    if len(set(args.ids)) != len(args.ids):
        raise ValueError('Duplicate clip IDs')
    if not set(args.ids) <= {c['id'] for c in reference['clips'] if c['stroke_type']=='putt'}:
        raise ValueError('SAM2 head experiment accepts only declared putt IDs')
    PROFILE['region_profile'] = dict(REGION_PROFILE)
    PROFILE['inactive_probability_thresholds'] = REGION_PROFILE['inactive_probability_thresholds']
    PROFILE['exported_confidence'] = None
    # Explicit inference-only manifest: no target labels are retained or consulted.
    inputs = [{'id':c['id'], 'path':c['path'], 'sha256':c['sha256'],
               'seed':c['targets']['clubhead']['seed'], 'window':c['window'],
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
                'head_region_sha256':hashlib.sha256((REPO/'ghostcaddie/demo/segmented_head.py').read_bytes()).hexdigest(),
                'point_policy_sha256':hashlib.sha256((REPO/'ghostcaddie/demo/foreground_tracker.py').read_bytes()).hexdigest(),
                'scores_are_binary_gate_indicators_not_model_confidence':True,'clips':{}}
    for clip in inputs:
        started=time.monotonic()
        assert hashlib.sha256(Path(clip['path']).read_bytes()).hexdigest()==clip['sha256']
        seed=clip['seed'];lo=int(seed['frame']);hi=int(clip['window']['end_frame'])
        cap=cv2.VideoCapture(clip['path']);width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        radius=PROFILE['crop_radius_native_px'];x0=max(0,round(seed['x'])-radius);y0=max(0,round(seed['y'])-radius)
        x1=min(width,round(seed['x'])+radius+1);y1=min(height,round(seed['y'])+radius+1)
        directory=run/'sam2_head_frames'/clip['id']
        if directory.exists():shutil.rmtree(directory)
        directory.mkdir(parents=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES,lo)
        for frame in range(lo,hi+1):
            ok,img=cap.read();assert ok
            cv2.imwrite(str(directory/f'{frame-lo:05d}.jpg'),img[y0:y1,x0:x1],[cv2.IMWRITE_JPEG_QUALITY,100])
        cap.release()
        assert len(list(directory.glob('*.jpg'))) == hi-lo+1
        with torch.inference_mode():
            state=predictor.init_state(str(directory),offload_video_to_cpu=True,offload_state_to_cpu=True)
            px,py=seed['x']-x0,seed['y']-y0;br=PROFILE['seed_box_radius_native_px']
            _,_,initial=predictor.add_new_points_or_box(state,frame_idx=0,obj_id=1,
                points=np.array([[px,py]],dtype=np.float32),labels=np.array([1],dtype=np.int32),
                box=np.array([px-br,py-br,px+br,py+br],dtype=np.float32))
            seed_mask=(initial[0,0]>0).cpu().numpy()
            head = MaskHeadPolicy(seed,[x0,y0],clip['cut_frames'])
            raw=[];points=[]
            maskdir=run/'sam2_head_masks'/clip['id']
            if maskdir.exists():shutil.rmtree(maskdir)
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
                'seconds':time.monotonic()-started}
        raw_out.write_text(json.dumps(metadata,indent=2)+'\n')
        print(clip['id'],'complete',round(time.monotonic()-started,2),'seconds',flush=True)
        block = next(c for c in payload['clips'] if c['id']==clip['id'])['targets']['clubhead']
        block['points'] = [p for p in block['points'] if p['source_frame'] < lo] + points

    predicted = bench.validate_predictions(payload,reference)
    args.out.write_text(bench.canonical_json(payload))
    runtime = {'interpreter':sys.executable,'python':sys.version.split()[0],
               'model':metadata['profile']['model'],'device':device,
               'tool_sha256':metadata['inference_code_sha256'],
               'point_policy_sha256':metadata['point_policy_sha256'],
               'opencv':cv2.__version__,'numpy':np.__version__,
               'head_region_sha256':metadata['head_region_sha256'],
               'checkpoint_sha256':metadata['checkpoint_sha256'],
               'model_source_commit':metadata['source_commit'],
               'profile':PROFILE,'raw_path':str(raw_out),'raw_sha256':bench.sha256_file(raw_out),'reference_inputs_used_by_model':['seed','window','cut_frames'],
               'labels_read_by_model':False,'baseline_replayed':True,
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
