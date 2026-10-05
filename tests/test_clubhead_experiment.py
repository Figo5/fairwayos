import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('clubhead_experiment',Path(__file__).resolve().parents[1]/'tools/clubhead_detector_experiment.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class MatchingTests(unittest.TestCase):
    def test_duplicates_and_negative_frames_count_false_positive(self):
        b=[0,0,10,10]
        rows=[{'truth':b,'predictions':[{'confidence':.9,'xyxy':b},{'confidence':.8,'xyxy':b}]},
              {'truth':None,'predictions':[{'confidence':.7,'xyxy':b}]}]
        r=m.score(rows)
        self.assertEqual((r['true_positives'],r['false_positives'],r['false_negatives']),(1,2,0))
        self.assertEqual(r['negative_frames_with_detection'],1)
        self.assertEqual(r['ap'],1.)
    def test_wrong_location_is_both_false_positive_and_miss(self):
        r=m.score([{'truth':[0,0,10,10],'predictions':[{'confidence':.8,'xyxy':[20,20,30,30]}]}])
        self.assertEqual((r['true_positives'],r['false_positives'],r['false_negatives']),(0,1,1))
        self.assertEqual(r['ap'],0.)
    def test_confidence_floor_does_not_fill_operating_miss(self):
        b=[0,0,10,10];r=m.score([{'truth':b,'predictions':[{'confidence':.1,'xyxy':b}]}])
        self.assertEqual((r['true_positives'],r['false_positives'],r['false_negatives']),(0,0,1))
        self.assertEqual(r['ap'],1.)
    def test_no_truth_ap_undefined(self):
        self.assertIsNone(m.score([{'truth':None,'predictions':[]}])['ap'])

class EvaluationContractTests(unittest.TestCase):
    def test_ap50_and_average_iou_ap(self):
        b=[0,0,10,10];rows=[{'truth':b,'predictions':[{'confidence':.8,'xyxy':[0,0,8,10]}]}]
        r=m.metrics(rows,{'ap_iou_thresholds':[.5,.75,.85],'operating_conf':.25})
        self.assertEqual(r['AP50'],1.)
        self.assertAlmostEqual(r['AP50_95'],2/3)
        with self.assertRaises(ValueError):m.metrics(rows,{'ap_iou_thresholds':[.6],'operating_conf':.25})
    def test_test_content_overlap_is_fatal_even_under_different_public_url(self):
        a={'video_id':'swing_001','split':'train','source':'yt','youtube_id':'one','notes':'Golfer One',
           'source_sha256':'a'*64,'frames':[{'image_sha256':'b'*64}]}
        b={'video_id':'swing_002','split':'test','source':'yt','youtube_id':'two','notes':'Golfer Two',
           'source_sha256':'c'*64,'frames':[{'image_sha256':'b'*64}]}
        with self.assertRaises(ValueError):m.validate_test_manifest({'accepted':[a]},{'accepted':[b]})
        b['frames'][0]['image_sha256']='d'*64
        m.validate_test_manifest({'accepted':[a]},{'accepted':[b]})
        b['source_sha256']=a['source_sha256']
        with self.assertRaises(ValueError):m.validate_test_manifest({'accepted':[a]},{'accepted':[b]})
    def test_declared_identity_and_url_aliases_are_not_holdout(self):
        a={'video_id':'swing_001','split':'train','source':'yt','youtube_id':'ABC','notes':'Tiger Woods1',
           'source_sha256':'a'*64,'frames':[{'image_sha256':'b'*64}]}
        b={'video_id':'swing_002','split':'test','source':'yt','youtube_id':'DEF','notes':'tiger_woods',
           'source_sha256':'c'*64,'frames':[{'image_sha256':'d'*64}]}
        with self.assertRaises(ValueError):m.validate_test_manifest({'accepted':[a]},{'accepted':[b]})
        b['notes']='another person';b['youtube_id']='https://youtu.be/ABC?t=4'
        with self.assertRaises(ValueError):m.validate_test_manifest({'accepted':[a]},{'accepted':[b]})
    def test_one_shot_guard_runs_before_model_instantiation(self):
        import tempfile,json
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as name:
            r=Path(name);(r/'results').mkdir();(r/'data/training_dataset').mkdir(parents=True);(r/'coordination').mkdir()
            checkpoint=r/'weights.pt';checkpoint.write_bytes(b'fake model bytes')
            protocol=r/'coordination/TRAINING_PROTOCOL.json';protocol.write_text('{}')
            manifest=r/'data/training_dataset/manifest.json';manifest.write_text('{}')
            freeze={'checkpoint':str(checkpoint),'checkpoint_sha256':m.sha(checkpoint),'script_sha256':m.sha(m.__file__),
                    'protocol_sha256':m.sha(protocol),'training_manifest_sha256':m.sha(manifest),'evaluation_settings':{}}
            (r/'results/CHECKPOINT_FREEZE.json').write_text(json.dumps(freeze));(r/'results/TEST_EVALUATION_START.json').write_text('{}')
            with patch('ultralytics.YOLO') as model:
                with self.assertRaisesRegex(ValueError,'One-shot'):m.evaluate(r,r/'data/training_dataset','test')
                model.assert_not_called()
    def test_test_selection_cannot_drop_source_or_be_empty(self):
        manifest={'accepted':[{'video_id':'a','split':'test'}]};selection={'accepted_ids':['a'],'rejected_ids':['b'],'manifest_sha256':'hash'}
        m.validate_test_selection(selection,manifest,'hash',['a','b'])
        for selection,manifest in [({'accepted_ids':['a'],'rejected_ids':[],'manifest_sha256':'hash'},manifest),
                                   ({'accepted_ids':[],'rejected_ids':['a','b'],'manifest_sha256':'hash'},{'accepted':[]}),
                                   ({'accepted_ids':['a'],'rejected_ids':['b'],'manifest_sha256':'changed'},manifest)]:
            with self.assertRaises(ValueError):m.validate_test_selection(selection,manifest,'hash',['a','b'])
