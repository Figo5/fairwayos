"""Synthetic CV contracts, not golf footage accuracy evidence."""
import copy
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from ghostcaddie.demo import shaft_head as h


def body(frame, wrist=(100, 300), roi=(40, 320, 120, 300)):
    return {'source_frame': frame, 'coordinate_space':'native_source_pixels', 'available': True, 'roi_xywh': list(roi),
            'keypoints': [{'name': n, 'x': wrist[0], 'y': wrist[1], 'score': .8,
                          'visible': True} for n in ('wr_l', 'wr_r')]}


def image(head=True, shaft=True, bright=False, second=False, scale=1):
    im = np.full((720, 400, 3), 170, np.uint8)
    color = (255,)*3 if bright else (30,)*3
    if shaft: cv2.line(im, (100,300), (265,300), (30,)*3, 2)
    if head:
        if bright: cv2.circle(im, (265,300), 7, color, -1)
        else: cv2.ellipse(im, (265,300), (8,5), 0, 0, 360, color, -1)
    if second:
        cv2.line(im, (100,300), (230,200), (30,)*3, 2)
        cv2.ellipse(im, (230,200), (8,5), 0, 0, 360, (30,)*3, -1)
    return cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) if scale != 1 else im


def policy(seed=None, origin=(0,0), size=(400,720), cuts=()):
    return h.ShaftHeadPolicy(seed or {'frame':0,'x':265.,'y':300.}, origin, cuts, size)


class ShaftHeadTests(unittest.TestCase):
    def track(self, im, b=None):
        p=policy(); p.step(0, im)
        first=p.step(1,im, b or body(1))
        second=p.step(2,im, {**(b or body(2)), 'source_frame':2})
        return first, second

    def test_real_extraction_two_confirmations_exact_supported_centroid(self):
        (p1,r1),(p2,r2)=self.track(image())
        self.assertIsNone(p1['x']); self.assertEqual(p2['origin'],'automatic')
        selected=r2['selected']
        core=next(m for m in r2['measurements'] if m['support_sha256']==selected['support_sha256'])
        centroid=np.mean(core['support_pixels_native'],axis=0)
        np.testing.assert_allclose([p2['x'],p2['y']],centroid,atol=.005)
        self.assertNotEqual([p2['x'],p2['y']],core['distal'])
        self.assertIsNone(p2['score']); self.assertEqual(len(core['support_sha256']),64)
        self.assertEqual(r2['segment_count'],2);self.assertEqual(len(r2['clusters']),1)

    def test_thin_shaft_blob_without_shaft_wrong_wrist_and_bright_ball_refuse(self):
        cases=[(image(head=False),None),(image(shaft=False),None),
               (image(),body(1,wrist=(20,600))), (image(bright=True),None)]
        for im,b in cases:
            with self.subTest(b=b):self.assertIsNone(self.track(im,b)[1][0]['x'])

    def test_two_heads_ambiguous_and_body_occlusion_refused(self):
        first,second=self.track(image(second=True))
        self.assertIsNone(second[0]['x'])
        self.assertEqual(second[1]['refusal_reason'],'ambiguous_multiple_core_clusters')
        self.assertIsNone(self.track(image(),body(1,roi=(200,200,120,200)))[1][0]['x'])

    def test_native_crop_offset_and_720_1080_scale(self):
        for scale in (1,1.5):
            im=image(scale=scale)
            p=policy({'frame':0,'x':365*scale,'y':300*scale},
                     origin=(int(100*scale),0),size=(int(500*scale),int(720*scale)))
            p.step(0,im)
            for f in (1,2):
                b=body(f,(200*scale,300*scale),(140*scale,320*scale,120*scale,300*scale))
                q,r=p.step(f,im,b)
            self.assertEqual(q['origin'],'automatic')
            np.testing.assert_allclose([q['x']/scale,q['y']/scale],[365,300],atol=1)
            self.assertEqual(p.scale_s,scale)

    def test_gap_requires_current_confirmation_no_backfill_prefix_invariant(self):
        im=image();a=policy();b=policy();prefix=[]
        for f in range(3):
            q,_=a.step(f,im,body(f));prefix.append(q)
            self.assertEqual(q,b.step(f,im,body(f))[0])
        missing=a.step(3,np.full_like(im,170),body(3))[0]
        pending=a.step(4,im,body(4))[0]
        q,_=a.step(5,im,body(5))
        self.assertIsNone(missing['x']);self.assertIsNone(pending['x'])
        self.assertEqual(q['method'],h.METHOD_REACQUIRED)
        self.assertEqual(prefix[-1]['source_frame'],2)
        self.assertEqual(prefix[-1]['x'],265)

    def test_far_stale_confirmation_does_not_permanently_block_recovery(self):
        im=image();p=policy();p.step(0,im)
        def evidence(x):return {'selected':{'centroid_native':[x,300.]}}
        for f,x in [(1,20.),(2,20.),(3,350.),(4,350.)]:
            with patch.object(h,'_detect',return_value=evidence(x)):
                q,r=p.step(f,im,body(f))
            if f==3:self.assertIsNone(q['x'])
        self.assertEqual(q['x'],350);self.assertEqual(q['method'],h.METHOD_REACQUIRED)

    def test_ambiguity_resets_pending_and_segment_cap_refuses_whole_frame(self):
        im=image();p=policy();p.step(0,im)
        self.assertIsNone(p.step(1,im,body(1))[0]['x'])
        self.assertIsNone(p.step(2,image(second=True),body(2))[0]['x'])
        self.assertIsNone(p.step(3,im,body(3))[0]['x'])
        self.assertEqual(p.step(4,im,body(4))[0]['origin'],'automatic')
        lines=np.array([[[n,0,n+40,100]] for n in range(4097)],dtype=np.int32)
        with patch.object(cv2,'HoughLinesP',return_value=lines):
            q,r=p.step(5,im,body(5))
        self.assertIsNone(q['x']);self.assertTrue(r['cap_exceeded']);self.assertEqual(r['segment_count'],4097)

    def test_missing_current_pose_wrists_and_invalid_scores_fail_closed(self):
        for b in (None,{**body(1),'available':False},body(1)):
            p=policy();p.step(0,image())
            if b is not None and b.get('available'):[k.update(score=.1) for k in b['keypoints']]
            self.assertIsNone(p.step(1,image(),b)[0]['x'])
        for field,value in [('score',2),('score',True),('visible',1),('score',float('nan'))]:
            p=policy();p.step(0,image());b=body(1);[k.update({field:value}) for k in b['keypoints']]
            self.assertIsNone(p.step(1,image(),b)[0]['x'])

    def test_input_validation_does_not_advance_frame(self):
        im=image()
        for b in ({**body(1),'source_frame':2},{**body(1),'source_frame':True},
                  {**body(1),'normalized':True},body(1,wrist=(float('nan'),2)),
                  body(1,wrist=(400,2)),body(1,roi=(300,200,200,200))):
            p=policy();p.step(0,im)
            with self.assertRaises(ValueError):p.step(1,im,b)
            self.assertEqual(p.last_frame,0)
        for bad in (np.zeros((0,400,3),np.uint8),im.astype(float),im[:,:,0],np.zeros((721,400,3),np.uint8)):
            p=policy()
            with self.assertRaises(ValueError):p.step(0,bad)
            self.assertEqual(p.last_frame,-1)
        for f in (True,1,-1,1.5):
            with self.assertRaises(ValueError):policy().step(f,im)

    def test_terminal_cut_including_seed_and_original_operator_seed(self):
        im=image();p=policy(cuts=(2,));q,_=p.step(0,im)
        self.assertEqual((q['x'],q['y']),(265,300));self.assertEqual(q['origin'],'operator')
        p.step(1,im,body(1))
        for f in (2,3,4):self.assertIsNone(p.step(f,im,body(f))[0]['x'])
        p=policy(cuts=(0,));self.assertIsNone(p.step(0,im)[0]['x'])
        self.assertIsNone(p.step(1,im,body(1))[0]['x'])

    def test_filled_region_edge_refused_as_shaft_but_dark_bright_ribbons_pass(self):
        gray=np.full((720,400),170,np.uint8)
        cv2.rectangle(gray,(100,300),(265,400),30,-1)
        self.assertFalse(h._shaft_ribbon(gray,(0,0),(100,300),(265,300),1)['accepted'])
        for val in (30,255):
            gray=np.full((720,400),170,np.uint8)
            cv2.line(gray,(100,300),(265,300),val,2)
            self.assertTrue(h._shaft_ribbon(gray,(0,0),(100,298),(265,298),1)['accepted'])
        gray=np.full((720,400),170,np.uint8)
        cv2.line(gray,(100,300),(230,200),30,2)
        self.assertTrue(h._shaft_ribbon(gray,(0,0),(100,300),(230,200),1)['accepted'])

    def test_shaft_compression_ringing_keeps_actual_bounded_contrast_support(self):
        gray=np.full((720,400),100,np.uint8)
        gray[299:302,100:266]=60
        gray[300,100:266]=170
        ribbon=h._shaft_ribbon(gray,(0,0),(100,300),(265,300),1)
        self.assertTrue(ribbon['accepted']);self.assertEqual(ribbon['sign_counts']['bright'],len(ribbon['profiles']))
        gray[299:320,100:266]=30
        self.assertFalse(h._shaft_ribbon(gray,(0,0),(100,300),(265,300),1)['accepted'])

    def test_ribbon_veto_does_not_remove_competing_head_clusters(self):
        with patch.object(h,'_shaft_ribbon',return_value={'accepted':True}):
            q,r=self.track(image(second=True))[1]
        self.assertIsNone(q['x']);self.assertEqual(len(r['clusters']),2)

    def test_one_current_reliable_wrist_suffices_without_inventing_other(self):
        im=image();p=policy();p.step(0,im)
        for f in (1,2):
            b=body(f);b['keypoints'][1]['visible']=False
            q,r=p.step(f,im,b)
        self.assertEqual(q['origin'],'automatic')
        self.assertEqual(set(r['body']['wrists']),{'wr_l'})

    def test_clipped_background_and_crop_edge_refused(self):
        im=image(head=False)
        cv2.rectangle(im,(255,270),(270,340),(30,)*3,-1)
        self.assertIsNone(self.track(im)[1][0]['x'])
        p=policy(size=(280,720));cropped=image()[:,:280]
        p.step(0,cropped)
        for f in (1,2):q,r=p.step(f,cropped,body(f))
        self.assertIsNone(q['x'])
        self.assertTrue(any(v.get('measurement_refusal')=='patch_clipped_by_crop' for v in r['lines']))

    def test_unmarked_coordinate_space_refused_and_first_acquisition_named(self):
        p=policy();im=image();p.step(0,im)
        b=body(1);del b['coordinate_space']
        with self.assertRaises(ValueError):p.step(1,im,b)
        self.assertEqual(p.last_frame,0)
        p.step(1,np.full_like(im,170),body(1))
        p.step(2,im,body(2));q,_=p.step(3,im,body(3))
        self.assertEqual(q['method'],h.METHOD_SHAFT)

    def test_invalid_constructor_types_bounds_and_stroke(self):
        for kwargs in ({'seed':{'frame':True,'x':265,'y':300}}, {'size':(400.,720)},
                       {'origin':(0.,0)}, {'cuts':(True,)}, {'seed':{'frame':0,'x':400,'y':300}},
                       {'seed':{'frame':0,'x':float('nan'),'y':300}}):
            with self.assertRaises(ValueError):policy(**kwargs)
        with self.assertRaises(ValueError):h.ShaftHeadPolicy({'frame':0,'x':0,'y':0},(0,0),[],(400,720),'putt')

    def test_dedup_keeps_measured_representative_not_transitive_average(self):
        cores=[{'centroid_native':[x,0], 'area':a,'mask':'dark'} for x,a in [(0,30),(7,20),(14,10)]]
        clusters=h._cluster_cores(cores,8)
        self.assertEqual(len(clusters),2);self.assertEqual(clusters[0]['selected']['centroid_native'],[0,0])
        self.assertEqual(h._border_pixels(np.ones((2,2),np.uint8),4),[1,1,1,1])

if __name__=='__main__':unittest.main()
