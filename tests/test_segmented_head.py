"""Geometry contracts for learned masks; these are not real-video accuracy tests."""
import unittest
import cv2
import numpy as np
from ghostcaddie.demo.segmented_head import compact_regions, select_seed, admitted_regions, MaskHeadPolicy


class SegmentedHeadTests(unittest.TestCase):
    def test_connected_long_shaft_does_not_pull_head_centroid(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[70:79, 35:52] = 1
        mask[5:71, 42:44] = 1
        regions = compact_regions(mask)
        seed = select_seed(regions, [43, 74])
        self.assertIsNotNone(seed)
        self.assertLess(abs(seed['xy'][0]-43), 1)
        self.assertLess(abs(seed['xy'][1]-74), 1)
        self.assertLess(seed['bbox'][3], 10)

    def test_round_ball_is_not_a_valid_seed(self):
        mask = np.zeros((80, 80), np.uint8)
        cv2.circle(mask, (30, 30), 6, 1, -1)
        regions = compact_regions(mask)
        self.assertTrue(regions)
        self.assertIsNone(select_seed(regions, [30, 30]))

    def test_separate_head_and_ball_are_not_averaged(self):
        mask = np.zeros((80, 80), np.uint8)
        mask[30:39, 20:37] = 1
        cv2.circle(mask, (48, 34), 5, 1, -1)
        mask[34, 36:44] = 1
        regions = compact_regions(mask)
        self.assertEqual(len(regions), 2)
        seed = select_seed(regions, [28, 34])
        self.assertIsNotNone(seed)
        self.assertLess(abs(seed['xy'][0]-28), 1)
        self.assertIsNone(select_seed(regions, [48, 34]))

    def test_ambiguous_seed_and_no_mask_refuse(self):
        a = {'xy':[10,10], 'area':10, 'elongation':2}
        b = {'xy':[12,10], 'area':10, 'elongation':2}
        self.assertIsNone(select_seed([a,b], [11,10]))
        self.assertEqual(compact_regions(np.zeros((20,20), np.uint8)), [])
        self.assertEqual(admitted_regions([a], None, [100,200], [110,210], 24), [])

    def test_native_origin_area_and_radius_admission(self):
        seed = {'xy':[10,10], 'area':12, 'elongation':2}
        regions = [seed, {**seed, 'xy':[50,10]}, {**seed, 'area':37}]
        admitted = admitted_regions(regions, seed, [100,200], [110,210], 24)
        self.assertEqual(len(admitted), 1)
        self.assertEqual(admitted[0]['native_xy'], [110,210])
        self.assertEqual(seed['xy'], [10,10])
        self.assertEqual(admitted_regions([seed], seed, [100,200], [110,210], 0)[0]['native_xy'], [110,210])

    def test_later_round_region_is_also_refused(self):
        seed={'xy':[10,10],'area':12,'elongation':2}
        round_region={**seed,'elongation':1}
        self.assertEqual(admitted_regions([round_region],seed,[0,0],[10,10],24),[])

    def test_first_forward_seed_recovery_is_causal_and_confidence_null(self):
        mask=np.zeros((100,100),np.uint8);mask[70:79,35:52]=1
        blank=np.zeros_like(mask)
        seed={'frame':0,'x':43,'y':74}
        head=MaskHeadPolicy(seed,[0,0],[7])
        head.step(0,mask)
        self.assertTrue(head.seed_valid)
        point,_=head.step(1,mask)
        self.assertEqual(point['origin'],'automatic')
        self.assertIsNone(point['score'])
        for frame in [2,3,4]:head.step(frame,blank)
        early,_=head.step(5,mask)
        recovered,_=head.step(6,mask)
        self.assertIsNone(early['x'])
        self.assertEqual(recovered['method'],'sam2_head_region_reacquired')
        self.assertIsNone(recovered['score'])
        cut,_=head.step(7,mask)
        later,_=head.step(8,mask)
        self.assertIsNone(cut['x'])
        self.assertIsNone(later['x'])
        self.assertIsNone(early['x'])
        with self.assertRaises(ValueError):head.step(8,mask)

    def test_refused_seed_never_acquires_from_later_frames(self):
        mask=np.zeros((80,80),np.uint8);cv2.circle(mask,(30,30),6,1,-1)
        head=MaskHeadPolicy({'frame':0,'x':30,'y':30},[0,0],[])
        head.step(0,mask)
        self.assertFalse(head.seed_valid)
        mask[:]=0;mask[26:35,22:39]=1
        for frame in range(1,30):
            point,_=head.step(frame,mask)
            self.assertIsNone(point['x'])
            self.assertNotEqual(point['origin'],'automatic')


if __name__ == '__main__':
    unittest.main()
