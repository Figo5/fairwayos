import unittest
import cv2
import numpy as np
from ghostcaddie.demo.adaptive_head import SeedSupportedHeadPolicy,seed_threshold


class AdaptiveHeadTests(unittest.TestCase):
    def test_edge_placement_is_supported_without_moving_the_seed(self):
        mask=np.zeros((90,90),np.uint8);mask[50:59,30:47]=1
        seed={'frame':0,'x':45,'y':54}
        head=SeedSupportedHeadPolicy(seed,[0,0],[])
        anchor,_=head.step(0,mask);point,_=head.step(1,mask)
        self.assertTrue(head.seed_valid)
        self.assertEqual(anchor['x'],45)
        self.assertLess(abs(point['x']-38),1)
        self.assertIsNone(point['score'])

    def test_unrelated_thicker_ball_cannot_raise_seed_threshold(self):
        mask=np.zeros((90,90),np.uint8);mask[50:57,30:45]=1
        threshold=seed_threshold(mask,[37,53])
        cv2.circle(mask,(65,53),10,1,-1)
        self.assertEqual(seed_threshold(mask,[37,53]),threshold)
        head=SeedSupportedHeadPolicy({'frame':0,'x':37,'y':53},[0,0],[])
        head.step(0,mask)
        self.assertTrue(head.seed_valid)
        point,_=head.step(1,mask)
        self.assertLess(abs(point['x']-37),1)

    def test_round_seed_refuses_all_later_frames(self):
        mask=np.zeros((90,90),np.uint8);cv2.circle(mask,(40,50),6,1,-1)
        head=SeedSupportedHeadPolicy({'frame':0,'x':40,'y':50},[0,0],[])
        head.step(0,mask);self.assertFalse(head.seed_valid)
        mask[:]=0;mask[46:55,32:49]=1
        for frame in range(1,5):self.assertIsNone(head.step(frame,mask)[0]['x'])

    def test_recovery_needs_current_confirmation_and_cut_is_terminal(self):
        mask=np.zeros((90,90),np.uint8);mask[46:55,32:49]=1;blank=np.zeros_like(mask)
        # Native seed must be translated once, by the caller's explicit origin.
        head=SeedSupportedHeadPolicy({'frame':0,'x':140,'y':250},[100,200],[7])
        head.step(0,mask);head.step(1,mask)
        for frame in [2,3,4]:head.step(frame,blank)
        early,_=head.step(5,mask);recovered,_=head.step(6,mask)
        self.assertIsNone(early['x']);self.assertEqual(recovered['x'],140)
        self.assertEqual(recovered['method'],'sam2_head_region_reacquired')
        self.assertIsNone(head.step(7,mask)[0]['x']);self.assertIsNone(head.step(8,mask)[0]['x'])
        self.assertIsNone(early['x'])


if __name__=='__main__':unittest.main()
