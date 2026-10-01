import copy
import unittest
import cv2
import numpy as np
from ghostcaddie.demo.appearance_head import AppearanceHeadPolicy


def fixture(round_head=True,color=100):
    mask=np.zeros((110,110),np.uint8)
    if round_head:cv2.circle(mask,(50,60),6,1,-1)
    else:mask[56:65,42:59]=1
    image=np.full((110,110,3),[50,120,60],np.uint8);image[mask>0]=color
    return mask>0,image


def policy(cuts=()):
    return AppearanceHeadPolicy({'frame':0,'x':150.,'y':260.},[100,200],cuts,'putt')


class AppearanceHeadTests(unittest.TestCase):
    def test_round_dark_head_is_supported_without_moving_seed(self):
        mask,image=fixture();head=policy();anchor,_=head.step(0,mask,image);point,_=head.step(1,mask,image)
        self.assertTrue(head.seed_valid);self.assertEqual(head.seed_mode,'appearance_round')
        self.assertEqual((anchor['x'],anchor['y']),(150,260));self.assertEqual(anchor['origin'],'operator')
        self.assertEqual((point['x'],point['y']),(150,260));self.assertIsNone(point['score'])

    def test_round_white_seed_is_ambiguous_and_never_measured(self):
        mask,image=fixture(color=240);head=policy();head.step(0,mask,image)
        self.assertFalse(head.seed_valid)
        for f in range(1,5):self.assertIsNone(head.step(f,mask,image)[0]['x'])

    def test_ball_cannot_replace_dark_round_head(self):
        mask,image=fixture();head=policy();head.step(0,mask,image);white=image.copy();white[mask]=240
        for f in range(1,5):
            point,raw=head.step(f,mask,white);self.assertIsNone(point['x'])
            self.assertTrue(raw['identity_checks']);self.assertFalse(raw['identity_checks'][0]['accepted'])

    def test_geometric_path_is_preserved_when_color_is_not_a_reliable_identity(self):
        mask,image=fixture(False,60);head=policy();head.step(0,mask,image);white=image.copy();white[mask]=220
        point,_=head.step(1,mask,white)
        self.assertEqual(head.seed_mode,'geometry');self.assertEqual(point['origin'],'automatic')

    def test_current_confirmations_are_required_no_backfill_and_cut_is_terminal(self):
        mask,image=fixture();blank=np.zeros_like(mask);head=policy([7]);head.step(0,mask,image);head.step(1,mask,image)
        for f in [2,3,4]:head.step(f,blank,image)
        early,_=head.step(5,mask,image);recovered,_=head.step(6,mask,image)
        self.assertIsNone(early['x']);self.assertEqual(recovered['method'],'sam2_head_region_reacquired')
        self.assertEqual(recovered['x'],150);self.assertIsNone(head.step(7,mask,image)[0]['x']);self.assertIsNone(head.step(8,mask,image)[0]['x'])
        self.assertIsNone(early['x'])

    def test_earlier_raw_descriptor_cannot_mutate_original_seed_identity(self):
        mask,image=fixture();head=policy();_,raw=head.step(0,mask,image);before=copy.deepcopy(head.seed_descriptor)
        raw['parts'][0]['appearance']['lab_median'][0]=255
        self.assertEqual(head.seed_descriptor,before)
        self.assertEqual(head.step(1,mask,image)[0]['origin'],'automatic')

    def test_reordered_frames_and_invalid_family_are_refused(self):
        mask,image=fixture();head=policy();head.step(0,mask,image)
        with self.assertRaises(ValueError):head.step(2,mask,image)
        with self.assertRaises(ValueError):AppearanceHeadPolicy({'frame':0,'x':50,'y':60},[0,0],[],'unknown')

if __name__=='__main__':unittest.main()
