"""Runner provenance and seed-only inputs using a fake mask model, no weights."""
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class FakeMask:
    def __init__(self, array): self.array = array
    def __getitem__(self, key): return self
    def __gt__(self, value): return self
    def cpu(self): return self
    def numpy(self): return self.array


class RunnerTests(unittest.TestCase):
    def test_native_seed_manifest_provenance_and_preseed_replay(self):
        spec = importlib.util.spec_from_file_location('all_clip_sam2_research_runner', ROOT/'tools/tiger_clubhead_sam2.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run/'models').mkdir()
            (run/'models/sam2.1_hiera_tiny.pt').write_bytes(b'fake fixture, not model weights')
            source = run/'source.avi'
            writer = cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),30,(140,100))
            for _ in range(3): writer.write(np.zeros((100,140,3),np.uint8))
            writer.release()
            seed = {'frame':1,'x':50.25,'y':70.5}
            clip = {'id':'putt','stroke_type':'putt','path':str(source),
                    'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                    'window':{'start_frame':0,'end_frame':2},'cut_frames':[],
                    'targets':{'clubhead':{'seed':seed,'labels':['never pass to model']}}}
            base = {'clips':[{'id':'putt','targets':{'ball':{'points':['classic']},
                     'clubhead':{'points':[{'source_frame':0,'sentinel':'preseed unchanged'}]}}}]}
            baseline = run/'base.json';baseline.write_text(json.dumps(base))
            out = run/'learned.predictions.json'
            mask = np.zeros((100,140),np.uint8);mask[66:75,42:59]=1
            predictor = mock.Mock()
            def init(path, **kwargs):
                self.assertEqual(len(list(Path(path).glob('*.jpg'))),2)
                return {'frames':path}
            predictor.init_state.side_effect = init
            # A refused immediate preview must not veto a valid first forward mask.
            predictor.add_new_points_or_box.return_value = (0,[1],FakeMask(np.zeros_like(mask)))
            predictor.propagate_in_video.return_value = [(0,[1],FakeMask(mask)),(1,[1],FakeMask(mask))]
            torch = ModuleType('torch');torch.__version__='fake-test-runtime'
            torch.backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:False))
            torch.inference_mode=contextlib.nullcontext
            sam = ModuleType('sam2');build=ModuleType('sam2.build_sam')
            build.build_sam2_video_predictor=mock.Mock(return_value=predictor)
            counts = {'correct_within_tolerance':0,'incorrect_visible_count':0,'miss_count':0}
            def evaluate(*a, **kwargs):
                return {'runtime':kwargs['runtime'], 'splits':{s:{'targets':{'clubhead':counts}} for s in ['development','evaluation']}}
            previous_path=list(sys.path)
            try:
                with mock.patch.dict(sys.modules,{'torch':torch,'sam2':sam,'sam2.build_sam':build}), \
                     mock.patch.object(runner.bench,'load_dataset',return_value=({'clips':[clip]},'frozen-reference-hash')), \
                     mock.patch.object(runner.bench,'validate_predictions',side_effect=lambda payload,refs:payload), \
                     mock.patch.object(runner.bench,'evaluate',side_effect=evaluate), \
                     mock.patch.object(runner.bench,'render_markdown',return_value='fixture report'), \
                     mock.patch.object(runner.subprocess,'check_output',return_value='fake-source-revision\n'), \
                     mock.patch.object(sys,'argv',['runner','--run-dir',str(run),'--baseline',str(baseline),'--out',str(out),'--checkpoint',str(run/'models/sam2.1_hiera_tiny.pt'),'--runtime-dir',str(run/'runtime'),'--negative-prompts']):
                    runner.main()
            finally:
                sys.path[:]=previous_path
            predicted=json.loads(out.read_text())
            self.assertEqual(predicted['clips'][0]['targets']['ball'],base['clips'][0]['targets']['ball'])
            self.assertEqual(predicted['clips'][0]['targets']['clubhead']['points'][0],base['clips'][0]['targets']['clubhead']['points'][0])
            raw=json.loads((run/'learned.raw.json').read_text())
            self.assertNotIn('labels',str(raw['clips']['putt']['input']))
            self.assertTrue(raw['clips']['putt']['seed_valid'])
            self.assertEqual(raw['clips']['putt']['prompt_preview_mask_area'],0)
            self.assertEqual(raw['head_region_sha256'],hashlib.sha256((ROOT/'ghostcaddie/demo/segmented_head.py').read_bytes()).hexdigest())
            self.assertEqual(len(raw['clips']['putt']['raw']),2)
            predictor.add_new_points_or_box.assert_called_once()
            prompt=predictor.add_new_points_or_box.call_args.kwargs
            self.assertEqual(prompt['frame_idx'],0)
            self.assertEqual(prompt['points'].tolist(),[[50.25,70.5]])
            self.assertIsNotNone(raw['clips']['putt']['prompt_patch_origin_crop'])
            receipt=json.loads((run/'learned.mask_receipt.json').read_text())
            self.assertEqual(receipt['model_record_sha256'],hashlib.sha256((run/'learned.raw.json').read_bytes()).hexdigest())
            self.assertEqual(set(receipt['clips']['putt']['masks']),{'1','2'})
            # Reusing an evidence name must fail before overwriting prior masks.
            with mock.patch.object(sys,'argv',['runner','--run-dir',str(run),'--baseline',str(baseline),'--out',str(out)]):
                with self.assertRaisesRegex(ValueError,'evidence already exists'):runner.main()
            report=json.loads(Path(str(out)+'.report.json').read_text())
            self.assertEqual(report['runtime']['raw_sha256'],hashlib.sha256((run/'learned.raw.json').read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
