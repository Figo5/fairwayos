import inspect
import json
import unittest
import numpy as np
from ghostcaddie.demo.head_identity import (BALL_CONFUSION, MIN_CORE_SAMPLES, THRESHOLDS,
                                            compare_core_identity, describe_core)


def image(bgr,size=(60,60)):
    frame=np.zeros((size[0],size[1],3),np.uint8);frame[:]=bgr
    return frame


def core(size=(60,60),box=(20,20,40,40)):
    mask=np.zeros(size,np.bool_);x0,y0,x1,y1=box;mask[y0:y1,x0:x1]=True
    return mask


class HeadIdentityTests(unittest.TestCase):
    def test_similar_dark_round_head_is_accepted_against_dark_seed(self):
        dark=image((60,60,62))
        seed=describe_core(dark,core())
        frame=image((60,60,62));frame[20:40,20:40]=63
        current=describe_core(frame,core())
        result=compare_core_identity(seed,current,'putt')
        self.assertTrue(result['accepted'])
        self.assertEqual(result['refusal_reasons'],[])
        self.assertLess(result['feature_differences']['lab_median_euclidean'],
                        THRESHOLDS['putt']['lab_median_distance_max'])

    def test_nearby_white_ball_is_refused_for_dark_seed(self):
        seed=describe_core(image((60,60,62)),core())
        ball=image((243,245,247))
        current=describe_core(ball,core())
        result=compare_core_identity(seed,current,'putt')
        self.assertFalse(result['accepted'])
        self.assertGreater(current['bright_low_sat_fraction'],.9)
        self.assertIn('lab_median_distance_exceeds_stroke_bound',result['refusal_reasons'])
        self.assertIn('bright_low_saturation_candidate_against_dark_seed',result['refusal_reasons'])
        self.assertGreater(result['feature_differences']['lab_median_delta'][0],
                           BALL_CONFUSION['candidate_lab_l_gain_min'])

    def test_moderate_shading_is_tolerated_for_both_stroke_families(self):
        seed=describe_core(image((60,60,62)),core())
        for stroke in ('putt','full_swing'):
            with self.subTest(stroke=stroke):
                current=describe_core(image((85,85,88)),core())
                result=compare_core_identity(seed,current,stroke)
                self.assertTrue(result['accepted'],result['refusal_reasons'])

    def test_markedly_different_turf_appearance_is_refused(self):
        seed=describe_core(image((60,60,62)),core())
        turf=describe_core(image((70,150,60)),core())
        result=compare_core_identity(seed,turf,'full_swing')
        self.assertFalse(result['accepted'])
        self.assertIn('lab_median_distance_exceeds_stroke_bound',result['refusal_reasons'])
        self.assertGreater(result['feature_differences']['lab_median_euclidean'],
                           THRESHOLDS['full_swing']['lab_median_distance_max'])

    def test_minimum_sample_count_blocks_partial_descriptors(self):
        seed=describe_core(image((60,60,62)),core())
        self.assertEqual(seed['sample_count'],400)
        short=dict(seed,sample_count=MIN_CORE_SAMPLES-1)
        result=compare_core_identity(short,seed,'putt')
        self.assertFalse(result['accepted'])
        self.assertIn('seed_core_sample_count_below_minimum',result['refusal_reasons'])
        result=compare_core_identity(seed,short,'putt')
        self.assertIn('current_core_sample_count_below_minimum',result['refusal_reasons'])

    def test_empty_mismatched_and_insufficient_cores_are_refused(self):
        dark=image((60,60,62))
        with self.assertRaisesRegex(ValueError,'no pixels'):
            describe_core(dark,np.zeros((60,60),np.bool_))
        with self.assertRaisesRegex(ValueError,'below the 8 minimum'):
            describe_core(dark,core((60,60),(0,0,3,2)))
        with self.assertRaisesRegex(ValueError,'match the image'):
            describe_core(dark,np.ones((30,30),np.bool_))
        with self.assertRaisesRegex(ValueError,'boolean numpy array'):
            describe_core(dark,np.ones((60,60),np.uint8))
        with self.assertRaisesRegex(ValueError,'uint8 numpy array'):
            describe_core(dark.astype(np.float32),core())
        with self.assertRaisesRegex(ValueError,'shape \\(height,width,3\\)'):
            describe_core(np.zeros((60,60),np.uint8),core())

    def test_undeclared_stroke_family_and_malformed_descriptors_are_refused(self):
        seed=describe_core(image((60,60,62)),core())
        with self.assertRaisesRegex(ValueError,'stroke_type'):
            compare_core_identity(seed,seed,'chip')
        with self.assertRaisesRegex(ValueError,'missing descriptor keys'):
            compare_core_identity({k:v for k,v in seed.items() if k!='dark_fraction'},seed,'putt')
        with self.assertRaisesRegex(ValueError,'finite numbers'):
            compare_core_identity({**seed,'lab_median':[float('nan'),128.,128.]},seed,'putt')
        with self.assertRaisesRegex(ValueError,'non-negative integer'):
            compare_core_identity({**seed,'sample_count':True},seed,'putt')
        with self.assertRaisesRegex(ValueError,'fraction in \\[0,1\\]'):
            compare_core_identity({**seed,'dark_fraction':1.4},seed,'putt')

    def test_stroke_specific_thresholds_change_the_decision(self):
        seed=describe_core(image((60,60,62)),core())
        current=describe_core(image((110,110,112)),core())
        putt=compare_core_identity(seed,current,'putt')
        swing=compare_core_identity(seed,current,'full_swing')
        distance=putt['feature_differences']['lab_median_euclidean']
        self.assertGreater(distance,THRESHOLDS['putt']['lab_median_distance_max'])
        self.assertLess(distance,THRESHOLDS['full_swing']['lab_median_distance_max'])
        self.assertTrue(swing['accepted'],swing['refusal_reasons'])
        self.assertFalse(putt['accepted'])
        self.assertEqual(putt['thresholds']['lab_median_distance_max'],45.0)
        self.assertEqual(swing['thresholds']['lab_median_distance_max'],60.0)

    def test_textured_candidate_with_similar_color_is_refused(self):
        seed=describe_core(image((60,60,62)),core())
        frame=image((60,60,62));frame[20:30,20:40]=10;frame[30:40,20:40]=110
        current=describe_core(frame,core())
        result=compare_core_identity(seed,current,'putt')
        self.assertTrue(result['feature_differences']['gray_dispersion_compared'])
        self.assertIn('gray_dispersion_ratio_out_of_stroke_bounds',result['refusal_reasons'])
        self.assertLess(result['feature_differences']['lab_median_euclidean'],45.0)

    def test_flat_cores_are_not_refused_for_absent_texture(self):
        seed=describe_core(image((60,60,62)),core())
        current=describe_core(image((70,70,72)),core())
        result=compare_core_identity(seed,current,'putt')
        self.assertFalse(result['feature_differences']['gray_dispersion_compared'])
        self.assertTrue(result['accepted'],result['refusal_reasons'])

    def test_inputs_and_descriptors_are_not_mutated(self):
        dark=image((60,60,62));mask=core()
        image_before,mask_before=dark.copy(),mask.copy()
        seed=describe_core(dark,mask)
        current=describe_core(image((70,70,72)),mask)
        seed_before,current_before=json.loads(json.dumps(seed)),json.loads(json.dumps(current))
        compare_core_identity(seed,current,'putt')
        self.assertEqual(dark.tobytes(),image_before.tobytes())
        self.assertEqual(mask.tobytes(),mask_before.tobytes())
        self.assertEqual(seed,seed_before)
        self.assertEqual(current,current_before)

    def test_output_is_deterministic_finite_json_without_scores(self):
        seed=describe_core(image((60,60,62)),core())
        current=describe_core(image((85,85,88)),core())
        first=compare_core_identity(seed,current,'putt')
        self.assertEqual(first,compare_core_identity(seed,current,'putt'))
        payload=json.dumps({'seed':seed,'current':current,'result':first},allow_nan=False)
        self.assertIn('native_mask_core',payload)
        numbers=[v for d in (seed,current,first['feature_differences'],first['thresholds'])
                 for v in _numbers(d)]
        self.assertTrue(all(np.isfinite(v) for v in numbers))
        for key in _keys(seed)+_keys(first):
            self.assertNotRegex(key,'score|confidence|probability')
        self.assertNotIn('x',_keys(first));self.assertNotIn('y',_keys(first))
        self.assertFalse(first['physical_identity_established'])
        self.assertIsInstance(seed['sample_count'],int)

    def test_comparison_signature_has_no_frame_or_reference_inputs(self):
        self.assertEqual(list(inspect.signature(compare_core_identity).parameters),
                         ['seed_descriptor','current_descriptor','stroke_type'])
        self.assertEqual(list(inspect.signature(describe_core).parameters),
                         ['image_bgr','core_mask'])


def _numbers(value):
    if isinstance(value,dict):
        return [n for v in value.values() for n in _numbers(v)]
    if isinstance(value,list):
        return [n for v in value for n in _numbers(v)]
    if isinstance(value,bool) or not isinstance(value,(int,float)):
        return []
    return [float(value)]


def _keys(value):
    return [str(k) for k in value]


if __name__=='__main__':
    unittest.main()
