import copy
import unittest
from ghostcaddie.demo.automatic_quality import evaluate

class QualityContractTests(unittest.TestCase):
    def inputs(self):
        ref={'schema':'fairwayos-automatic-visual-reference.v1','provenance':'ai_visual_reference','labels':[]}
        for frame,status in enumerate(['visible','visible','absent','unknown']):
            ref['labels'].append(dict(clip='a',source_sha256='source',frame=frame,object='ball',split='development',slice='sky',status=status,point=[10,10],tolerance_px=2))
        pred={'a':{'source_sha256':'source','observations':{'frames':[dict(frame=n,ball={'point':[10,10]} if n in (0,2,3) else None,target=None) for n in range(4)]}}}
        return ref,pred
    def test_misses_false_observations_and_unknown_have_separate_denominators(self):
        r,p=self.inputs();m=evaluate(r,p)['metrics'][0]
        self.assertEqual((m['visible'],m['absent'],m['unknown']),(2,1,1))
        self.assertEqual((m['matched'],m['misses'],m['false_observations']),(1,1,1))
        self.assertEqual(m['matched_visible_fraction'],.5)
    def test_wrong_object_is_not_a_successful_detection(self):
        r,p=self.inputs();p['a']['observations']['frames'][0]['ball']['point']=[40,40]
        self.assertEqual(evaluate(r,p)['metrics'][0]['wrong_locations'],1)
    def test_source_mismatch_and_missing_frames_fail(self):
        r,p=self.inputs();p['a']['source_sha256']='different'
        with self.assertRaises(ValueError):evaluate(r,p)
        r,p=self.inputs();p['a']['observations']['frames'].pop()
        with self.assertRaises(ValueError):evaluate(r,p)
    def test_unknown_global_visibility_can_have_a_known_wrong_object_region(self):
        r,p=self.inputs();r['labels'][3]['excluded_region']=[8,8,12,12]
        m=evaluate(r,p)['metrics'][0]
        self.assertEqual(m['reviewed_exclusion_regions'],1)
        self.assertEqual(m['exclusion_hits'],1)
        self.assertEqual(m['unknown'],1)
    def test_duplicate_labels_fail(self):
        r,p=self.inputs();r['labels'].append(copy.deepcopy(r['labels'][0]))
        with self.assertRaises(ValueError):evaluate(r,p)
