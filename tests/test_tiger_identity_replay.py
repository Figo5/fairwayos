"""Replay integration with native mask/RGB fixtures; no model or reference coordinates."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class ReplayTests(unittest.TestCase):
    def exercise(self, damage=None):
        spec = importlib.util.spec_from_file_location('identity_replay', ROOT / 'tools/tiger_replay_head_identity.py')
        runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
            def save(name, value):
                p = run / name; p.write_text(json.dumps(value)); return p
            source = run / 'original.mp4'; source.write_bytes(b'immutable source fixture')
            seed = {'frame': 1, 'x': 50.25, 'y': 70.5}
            clip = {'id': 'putt', 'path': str(source), 'sha256': sha(source), 'stroke_type': 'putt',
                    'window': {'start_frame': 0, 'end_frame': 3}, 'cut_frames': [],
                    'targets': {'clubhead': {'seed': seed, 'labels': ['not passed to policy']}}}
            inp = {k: clip[k] for k in ['id', 'path', 'sha256', 'stroke_type', 'window', 'cut_frames']}
            inp['seed'] = seed
            crop = [40, 60, 30, 30]
            masks = {}; frames = {}
            for f in range(1, 4):
                mask = np.zeros((30, 30), np.uint8); mask[6:15, 4:19] = 255
                m = run / f'mask{f}.png'; cv2.imwrite(str(m), mask)
                rgb = run / f'frame{f}.jpg'; cv2.imwrite(str(rgb), np.full((30, 30, 3), 120, np.uint8))
                masks[str(f)] = {'path': str(m), 'sha256': sha(m)}
                frames[str(f)] = {'path': str(rgb), 'sha256': sha(rgb)}
            model = save('model.json', {'clips': {'putt': {'input': inp, 'crop': crop, 'actual_policy_settings': {}}},
                **{k: 'fixture' for k in ['device', 'torch', 'checkpoint_sha256', 'source_commit', 'official_source', 'inference_code_sha256', 'point_policy_sha256']}})
            receipt = save('masks.json', {'model_record': str(model), 'model_record_sha256': sha(model),
                'clips': {'putt': {'input': inp, 'crop': crop, 'masks': masks}}})
            frame_receipt = save('frames.json', {'mask_receipt_sha256': sha(receipt),
                'clips': {'putt': {'input': inp, 'crop': crop, 'frames': frames}}})
            base = {'clips': [{'id': 'putt', 'targets': {'ball': {'sentinel': 'unchanged ball'},
                'clubhead': {'points': [{'source_frame': 0, 'origin': 'derived', 'sentinel': 'unchanged preseed'}]}}}]}
            baseline = save('baseline.json', base); dataset = save('dataset.json', {})
            if damage == 'rgb': Path(frames['2']['path']).write_bytes(b'changed pixels')
            if damage == 'mask': Path(masks['2']['path']).write_bytes(b'changed mask')
            if damage == 'model': model.write_text('{}')
            out = run / ('unsafe.json' if damage == 'name' else 'result.predictions.json')
            with mock.patch.object(runner.bench, 'load_dataset', return_value=({'clips': [clip]}, 'fixture-reference')),\
                 mock.patch.object(runner.bench, 'validate_predictions', side_effect=lambda p, d: p),\
                 mock.patch.object(runner.bench, 'evaluate', side_effect=lambda *a, **k: {'runtime': k['runtime']}),\
                 mock.patch.object(runner.bench, 'render_markdown', return_value='fixture report'),\
                 mock.patch.object(runner.subprocess, 'check_output', return_value='fixture-revision'),\
                 mock.patch('sys.argv', ['replay', '--dataset', str(dataset), '--baseline', str(baseline),
                         '--mask-receipt', str(receipt), '--frames-receipt', str(frame_receipt), '--out', str(out)]):
                if damage:
                    with self.assertRaises(ValueError): runner.main()
                    self.assertFalse(out.exists())
                    self.assertFalse((run / 'result.raw.json').exists())
                else:
                    runner.main(); result = json.loads(out.read_text())['clips'][0]['targets']
                    self.assertEqual(result['ball'], base['clips'][0]['targets']['ball'])
                    self.assertEqual(result['clubhead']['points'][0], base['clips'][0]['targets']['clubhead']['points'][0])
                    raw = json.loads((run / 'result.raw.json').read_text())
                    self.assertEqual(raw['clips']['putt']['input']['seed'], seed)
                    self.assertNotIn('labels', raw['clips']['putt']['input'])
                    self.assertEqual(raw['clips']['putt']['points'][0]['origin'], 'operator')
                    self.assertTrue(raw['clips']['putt']['seed_valid'])
                    self.assertEqual(raw['clips']['putt']['points'][1]['origin'], 'automatic')
                    runtime = json.loads(Path(str(out) + '.report.json').read_text())['runtime']
                    self.assertFalse(runtime['model_rerun'])
                    self.assertEqual(runtime['frames_receipt_sha256'], sha(frame_receipt))
                    with self.assertRaisesRegex(ValueError, 'fresh output'): runner.main()
            self.assertEqual(json.loads(baseline.read_text()), base)
            self.assertEqual(source.read_bytes(), b'immutable source fixture')

    def test_forward_replay_preserves_fractional_seed_preseed_and_balls(self):
        self.exercise()

    def test_tampered_evidence_and_unsafe_output_names_fail_before_writes(self):
        for damage in ['rgb', 'mask', 'model', 'name']:
            with self.subTest(damage=damage): self.exercise(damage)
