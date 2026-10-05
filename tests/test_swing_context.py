import copy
import unittest
import numpy as np
from ghostcaddie.demo.swing_context import ContextSwingHeadPolicy
from tests.test_swing_head import fixture


def body(frame, xy=(0, 0), score=.7):
    return {'source_frame': frame, 'available': True, 'keypoints':
            [{'name': 'wr_l', 'x': xy[0], 'y': xy[1], 'score': score, 'visible': True}]}


class SwingContextTests(unittest.TestCase):
    def head(self):
        return ContextSwingHeadPolicy({'frame': 0, 'x': 140., 'y': 250.}, [100, 200], [])

    def test_current_contrast_and_distant_wrist_admit_exact_native_core(self):
        mask, image = fixture()
        h = self.head(); h.step(0, mask, image, body_frame=body(0))
        p, raw = h.step(1, mask, image, body_frame=body(1))
        self.assertEqual((p['x'], p['y']), (140, 250))
        self.assertIsNone(p['score'])
        self.assertGreater(raw['parts'][0]['local_contrast']['absolute_median_gray_difference'], 15)

    def test_missing_or_nearby_wrists_refuse_current_point(self):
        mask, image = fixture()
        for current in [None, body(1, (140, 250)), body(1, score=.1)]:
            h=self.head(); h.step(0, mask, image, body_frame=body(0))
            p, raw=h.step(1, mask, image, body_frame=current)
            self.assertIsNone(p['x']); self.assertFalse(raw['identity_checks'][0]['accepted'])

    def test_low_contrast_original_seed_disables_only_measured_recovery(self):
        mask, image=fixture(); image[:]=(60, 60, 62)
        h=self.head(); p,_=h.step(0, mask, image, body_frame=body(0))
        self.assertEqual(p['origin'], 'operator'); self.assertFalse(h.seed_valid)
        for f in range(1, 5):self.assertIsNone(h.step(f, mask, fixture()[1], body_frame=body(f))[0]['x'])

    def test_current_low_contrast_or_thin_shaft_refused(self):
        mask,image=fixture(); h=self.head(); h.step(0,mask,image,body_frame=body(0))
        flat=image.copy();flat[:]=(60,60,62)
        self.assertIsNone(h.step(1,mask,flat,body_frame=body(1))[0]['x'])
        part={'local_contrast':{'background_samples':50,'absolute_median_gray_difference':50},'elongation':5,'xy':[40,50]}
        self.assertIn('thin_core_ambiguous_with_shaft',h._extra_refusals(part,image))

    def test_future_wrong_or_nonfinite_body_evidence_rejected(self):
        mask,image=fixture()
        for current in [body(1), {**body(0),'source_frame':True}, body(0, (float('nan'), 2)), body(0,score=2)]:
            h=self.head()
            with self.assertRaises(ValueError):h.step(0,mask,image,body_frame=current)
            self.assertEqual(h.last_frame,-1)

    def test_future_append_preserves_prefix_and_frozen_descriptor(self):
        mask,image=fixture();a=self.head();b=self.head()
        first=[a.step(f,mask,image,body_frame=body(f))[0] for f in range(5)]
        second=[b.step(f,mask,image,body_frame=body(f))[0] for f in range(5)]
        frozen=copy.deepcopy(b.seed_descriptor)
        for f in range(5,10):b.step(f,mask,image,body_frame=body(f,(140,250)))
        self.assertEqual(first,second);self.assertEqual(frozen,b.seed_descriptor)


if __name__=='__main__':unittest.main()
