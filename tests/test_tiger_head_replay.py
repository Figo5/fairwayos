"""Execution contract tests; synthetic empty collection is not tracker evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import tiger_replay_head_identity as replay


class HeadReplayExecutionTests(unittest.TestCase):
    def execute(self, root, fail=False, tamper=False):
        root=Path(root)
        paths={k:root/(k+'.json') for k in ['dataset','baseline','baseline-report','mask-receipt','frames-receipt']}
        paths['dataset'].write_text('{}');paths['baseline'].write_text(json.dumps({'clips':[]}))
        model=root/'model.json';model.write_text(json.dumps({'clips':{},**{k:'fixture' for k in ['device','torch','checkpoint_sha256','source_commit','official_source','inference_code_sha256','point_policy_sha256']}}))
        paths['mask-receipt'].write_text(json.dumps({'clips':{},'model_record':str(model),'model_record_sha256':replay.bench.sha256_file(model)}))
        paths['frames-receipt'].write_text(json.dumps({'clips':{},'mask_receipt_sha256':replay.bench.sha256_file(paths['mask-receipt'])}))
        paths['baseline-report'].write_text(json.dumps({'predictions':{'sha256':'f'*64 if tamper else replay.bench.sha256_file(paths['baseline'])},'runtime':{}}))
        output=root/'measured.predictions.json'
        argv=['replay']+[v for k,p in paths.items() for v in ['--'+k,str(p)]]+['--out',str(output)]
        with patch('sys.argv',argv), patch.object(replay.bench,'load_dataset',return_value=({'clips':[]},'d'*64)), \
                patch.object(replay.bench,'validate_predictions',side_effect=lambda p,d:p), \
                patch.object(replay.bench,'evaluate',side_effect=ValueError('evaluation failed') if fail else lambda *a,**k:k), \
                patch.object(replay.bench,'render_markdown',return_value='fixture report'):
            replay.main()
        return output

    def test_evaluation_failure_writes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,'evaluation failed'):self.execute(root,fail=True)
            self.assertFalse(list(Path(root).glob('measured*')))

    def test_tampered_baseline_report_writes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,'bind'):self.execute(root,tamper=True)
            self.assertFalse(list(Path(root).glob('measured*')))

    def test_default_appearance_report_pins_exact_written_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            output=self.execute(root);report=json.loads(Path(str(output)+'.report.json').read_text())
            self.assertEqual(report['predictions_sha256'],replay.bench.sha256_file(output))
            self.assertEqual(report['runtime']['raw_sha256'],replay.bench.sha256_file(Path(root)/'measured.raw.json'))
            self.assertEqual(json.loads(output.read_text())['variant'],'declared_cohort_mask_core_appearance')


if __name__=='__main__':unittest.main()
