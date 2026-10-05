import unittest
from unittest.mock import patch
import numpy as np

from ghostcaddie.demo.proposal_tracking import ProposalHeadPolicy, UnseededConfirmation, METHOD, METHOD_REACQUIRED


class ProposalChronologyTests(unittest.TestCase):
    def test_malformed_detections_rejected_on_seed_missing_body_and_after_cut(self):
        image=np.zeros((200,200,3),dtype=np.uint8)
        bad=[{'xyxy':[0,0,20,20],'confidence':float('nan'),'class_id':0}]
        p=ProposalHeadPolicy({'frame':0,'x':20,'y':30},[0,0],[2],[200,200])
        for frame in (0,1,2,3):
            with self.subTest(frame=frame):
                with self.assertRaises(ValueError):
                    p.step(frame,image,None,detections=bad)
                p.step(frame,image,None,detections=[])

    def test_unseeded_confirmation_has_no_operator_anchor_or_backfill(self):
        p=UnseededConfirmation(720)
        self.assertEqual(p.step(0,{'centroid_native':[20,30]})['status'],'missing')
        measured={'centroid_native':[21.5,31.5]}
        self.assertEqual(p.step(1,measured)['x'],21.5)
        self.assertEqual(p.step(2,None)['status'],'missing')
        self.assertEqual(p.step(3,measured)['status'],'missing')
        self.assertEqual(p.step(4,measured)['method'],METHOD_REACQUIRED)

    def test_cut_is_terminal_and_large_jump_requires_new_confirmation(self):
        p=UnseededConfirmation(720,cut_frames=[5]);a={'centroid_native':[10,10]};b={'centroid_native':[600,600]}
        p.step(0,a);self.assertEqual(p.step(1,a)['method'],METHOD)
        self.assertEqual(p.step(2,b)['status'],'missing')
        self.assertEqual(p.step(3,b)['method'],METHOD_REACQUIRED)
        p.step(4,b)
        self.assertEqual(p.step(5,b)['status'],'missing')
        self.assertEqual(p.step(6,b)['status'],'missing')
        with self.assertRaises(ValueError):p.step(6,b)

    def test_seed_and_missing_current_body_fail_closed(self):
        p=ProposalHeadPolicy({'frame':0,'x':20,'y':30},[0,0],[],[200,200])
        image=np.zeros((200,200,3),dtype=np.uint8)
        seed,raw=p.step(0,image,None,detections=[])
        self.assertEqual((seed['origin'],seed['x'],seed['score']),('operator',20.,None))
        missing,raw=p.step(1,image,None,detections=[])
        self.assertEqual(missing['status'],'missing')
        self.assertEqual(raw['body_fail_closed'],'body_frame_absent')
        with self.assertRaises(ValueError):p.step(1,image,None,detections=[])

    def test_seeded_confirmation_exports_core_and_truthful_method(self):
        p=ProposalHeadPolicy({'frame':0,'x':20,'y':30},[0,0],[],[200,200]);image=np.zeros((200,200,3),dtype=np.uint8)
        p.step(0,image,None,detections=[])
        body={'source_frame':1,'coordinate_space':'native_source_pixels','available':True,'roi_xywh':[50,50,30,50],
              'keypoints':[{'name':'wr_l','x':70,'y':80,'visible':True,'score':.9}]}
        evidence={'selected':{'centroid_native':[100.25,150.75]}}
        with patch('ghostcaddie.demo.proposal_head.measure_proposals',return_value=evidence):
            a,_=p.step(1,image,body,detections=[])
            body['source_frame']=2;b,raw=p.step(2,image,body,detections=[])
        self.assertEqual(a['status'],'missing')
        self.assertEqual((b['x'],b['y'],b['method'],b['score']),(100.25,150.75,METHOD,None))
        self.assertEqual(raw['decision']['method'],METHOD)
