import copy
import unittest
from unittest import mock
import cv2
import numpy as np
from ghostcaddie.video.shot_people import PersonIdentity,person_score,supported_club,publish_people
from ghostcaddie.video.shot_metrics import analyze_motion


def body(x=150,track=1):
    points={'left_shoulder':[x-10,70],'right_shoulder':[x+10,70],
            'left_hip':[x-10,135],'right_hip':[x+10,135],
            'left_wrist':[x-2,155],'right_wrist':[x+2,155], 'nose':[x,55],
            'left_ankle':[x-10,235],'right_ankle':[x+10,235]}
    return {'bbox':[x-35,45,x+35,245],'track_id':track,'appearance':[0,1,0],
            'confidence':.95,'keypoints':{n:{'point':p,'confidence':.9} for n,p in points.items()}}


def row(n, *, b=None, c=None,t=None,scene=0):
    return {'frame':n,'scene':scene,'body_scene':scene,'timestamp_seconds':t if t is not None else n*.04,
            'body':copy.deepcopy(b),'clubhead':copy.deepcopy(c),'ball':None,'target':None,'previous_to_current':None}


class EvidenceTests(unittest.TestCase):
    def fixture(self,head=True,shaft=True):
        f=np.full((300,400,3),(60,125,70),np.uint8)
        if shaft:cv2.line(f,(150,155),(195,235),(15,15,15),2)
        if head:cv2.ellipse(f,(200,239),(7,4),0,0,360,(15,15,15),-1)
        return f,[{'box':[191,230,209,245],'score':.9}]

    def test_learned_box_alone_cannot_publish(self):
        f,p=self.fixture(False,False)
        self.assertIsNone(supported_club(f,body(),p))

    def test_head_without_current_shaft_is_not_a_club(self):
        f,p=self.fixture(True,False)
        self.assertIsNone(supported_club(f,body(),p))

    def test_shaft_without_independent_wider_core_is_not_a_head(self):
        f,p=self.fixture(False,True)
        self.assertIsNone(supported_club(f,body(),p))

    def test_independent_shaft_head_and_hands_publish_current_pixels(self):
        f,p=self.fixture();c=supported_club(f,body(),p)
        self.assertIsNotNone(c)
        self.assertEqual(c['provenance'],'current_pixels')
        self.assertLess(np.linalg.norm(np.array(c['point'])-[200,239]),4)

    def test_missing_all_current_hands_blocks_head(self):
        f,p=self.fixture();b=body();del b['keypoints']['left_wrist'];del b['keypoints']['right_wrist']
        self.assertIsNone(supported_club(f,b,p))

    def test_primary_person_ambiguity_and_cut_reset(self):
        identity=PersonIdentity()
        self.assertIsNone(identity.select([body(180),body(220)],(400,300)))
        first=identity.select([body(200)],(400,300));self.assertEqual(first['track_id'],1)
        second=identity.select([body(200)],(400,300),cut=True);self.assertEqual(second['track_id'],2)

    def test_no_golfer_identity_from_confidence_alone(self):
        b=body();b['keypoints']['left_wrist']['point']=[70,155]
        self.assertIsNone(person_score(b,(400,300)))

    def test_gap_breaks_publication_support(self):
        c={'point':[200,239],'body_track_id':1}
        rows=[row(0,b=body(),c=c),row(1,b=body(),c=c),row(2,b=body()),row(3,b=body(),c=c),row(4,b=body(),c=c)]
        publish_people(rows)
        self.assertTrue(all(r['clubhead'] is None for r in rows))
        self.assertTrue(all(r['body'] is not None for r in rows))

    def test_identity_and_cut_prevent_support_aggregation(self):
        rows=[row(n,b=body(track=1 if n<2 else 2),c={'point':[200,239],'body_track_id':1 if n<2 else 2},scene=0 if n<2 else 1) for n in range(4)]
        publish_people(rows)
        self.assertTrue(all(r['body'] is None and r['clubhead'] is None for r in rows))


class MetricsTests(unittest.TestCase):
    def test_native_detail_in_unresolved_view_uses_pts_and_respects_identity_and_channel(self):
        rows=[row(0,b=body(),scene=None,t=10),row(1,b=body(),scene=None,t=10.08)]
        for n,r in enumerate(rows):
            r['body_scene']=0;r['view']='unresolved';r['ball_space']='image'
            r['ball']={'point':[200+n*4,239],'context':'near_player_native_detail','body_track_id':1}
        m=analyze_motion(rows,source_sha256='a'*64)
        self.assertAlmostEqual(m['motion']['ball_image']['median'],50)
        self.assertEqual(m['motion']['ball_ground_residual']['samples'],0)
        rows[1]['body']['track_id']=2;rows[1]['ball']['body_track_id']=2
        self.assertEqual(analyze_motion(rows,source_sha256='a'*64)['motion']['ball_image']['samples'],0)
        rows[1]['body']['track_id']=1;rows[1]['ball'].pop('context')
        self.assertEqual(analyze_motion(rows,source_sha256='a'*64)['motion']['ball_image']['samples'],0)

    def test_actual_pts_and_identity_are_required(self):
        c={'point':[200,239],'body_track_id':1}
        rows=[row(0,b=body(),c=c,t=10),row(1,b=body(),c=dict(c,point=[204,239]),t=10.08)]
        m=analyze_motion(rows,source_sha256='a'*64)
        self.assertAlmostEqual(m['motion']['clubhead_image']['median'],50)
        rows[1]['body']['track_id']=2
        self.assertEqual(analyze_motion(rows,source_sha256='a'*64)['motion']['clubhead_image']['samples'],0)

    def test_gaps_and_nonpositive_time_never_get_velocities(self):
        c={'point':[200,239],'body_track_id':1}
        for n,t in [(2,.08),(1,0),(1,.5)]:
            rows=[row(0,b=body(),c=c),row(n,b=body(),c=c,t=t)]
            self.assertEqual(analyze_motion(rows,source_sha256='a'*64)['motion']['clubhead_image']['samples'],0)

    def test_hip_translation_is_subtracted_and_physical_units_are_absent(self):
        c={'point':[200,239],'body_track_id':1}
        rows=[row(0,b=body(),c=c),row(1,b=body(x=154),c=dict(c,point=[204,239]))]
        m=analyze_motion(rows,source_sha256='a'*64)
        self.assertEqual(m['motion']['clubhead_body_relative']['median'],0)
        self.assertTrue(all(v is None for v in m['physical'].values()))
        self.assertIsNone(m['impact']);self.assertIsNone(m['swing_tempo'])

    def test_camera_failure_does_not_emit_ground_residual(self):
        rows=[row(0),row(1)]
        for r in rows:r['ball']={'point':[200,239]};r['target']=[210,245]
        m=analyze_motion(rows,source_sha256='a'*64)
        self.assertEqual(m['motion']['ball_ground_residual']['samples'],0)
        self.assertEqual(m['motion']['cup_ground_residual']['samples'],0)

class ReviewRegressionTests(unittest.TestCase):
    def test_thick_shaft_without_lobe_is_not_a_head(self):
        for thickness in (1,2,3,4,6):
            with self.subTest(thickness=thickness):
                f=np.full((300,400,3),(60,125,70),np.uint8)
                cv2.line(f,(150,155),(200,239),(15,15,15),thickness)
                p=[{'box':[190,229,210,249],'score':.95}]
                self.assertIsNone(supported_club(f,body(),p))

    def test_distinct_tracks_do_not_get_pooled_body_analytics(self):
        c={'point':[200,239],'body_track_id':1}
        rows=[row(0,b=body(),c=c),row(1,b=body(),c=dict(c,point=[204,239])),
              row(2,b=body(track=2),c=dict(c,body_track_id=2)),
              row(3,b=body(track=2),c=dict(c,point=[220,239],body_track_id=2))]
        m=analyze_motion(rows,source_sha256='a'*64)
        self.assertTrue(m['body_aggregate_withheld'])
        self.assertIsNone(m['motion']['clubhead_image']['median'])
        self.assertEqual(len(m['segments']),2)
        self.assertAlmostEqual(m['segments'][0]['motion']['clubhead_image']['median'],100)
        self.assertAlmostEqual(m['segments'][1]['motion']['clubhead_image']['median'],500)

    def test_arm_bbox_change_does_not_delete_torso_relative_motion(self):
        c={'point':[200,239],'body_track_id':1}
        b=body();b['bbox'][1]=0
        rows=[row(0,b=body(),c=c),row(1,b=b,c=dict(c,point=[204,239]))]
        m=analyze_motion(rows,source_sha256='a'*64)
        self.assertEqual(m['motion']['clubhead_body_relative']['samples'],1)

    def test_numpy_target_does_not_crash(self):
        r=row(0,b=body(),c={'point':[200,239],'body_track_id':1})
        r['ball']={'point':[201,239]};r['target']=np.array([210,240])
        self.assertEqual(analyze_motion([r],source_sha256='a'*64)['simultaneous_four_layer_frames'],1)

    def test_person_veto_misses_and_reacquisition(self):
        p=PersonIdentity();first=p.select([body()],(400,300))
        self.assertEqual(p.select([body(x=151)],(400,300))['track_id'],first['track_id'])
        wrong=body();wrong['appearance']=[1,0,0]
        for _ in range(3):self.assertIsNone(p.select([wrong],(400,300)))
        self.assertNotEqual(p.select([body()],(400,300))['track_id'],first['track_id'])

class PrimaryPersonRegressionTests(unittest.TestCase):
    def test_cropped_upper_body_cannot_inherit_primary_track(self):
        identity=PersonIdentity();identity.select([body()],(400,300))
        operator=body();del operator['keypoints']['nose'];del operator['keypoints']['left_ankle'];del operator['keypoints']['right_ankle']
        self.assertIsNone(identity.select([operator],(400,300)))
        self.assertIsNone(identity.select([operator],(400,300),cut=True))

    def test_image_edge_clipping_withholds_current_person(self):
        identity=PersonIdentity();identity.select([body()],(400,300))
        b=body();b['bbox'][0]=0
        self.assertIsNone(identity.select([b],(400,300)))

class RearViewContinuityTests(unittest.TestCase):
    def test_face_turn_does_not_replace_current_body_pose(self):
        p=PersonIdentity();a=p.select([body()],(400,300));rear=body();del rear['keypoints']['nose']
        b=p.select([rear],(400,300))
        self.assertIsNotNone(b);self.assertEqual(a['track_id'],b['track_id'])
        self.assertNotIn('nose',b['keypoints'])
        self.assertIsNone(PersonIdentity().select([rear],(400,300)))

class VisibleHandAndReflectiveShaftTests(unittest.TestCase):
    def test_one_actual_confident_hand_suffices_without_inventing_the_other(self):
        f,p=EvidenceTests().fixture();b=body();del b['keypoints']['left_wrist']
        c=supported_club(f,b,p);self.assertIsNotNone(c);self.assertEqual(c['visible_hand_count'],1)

    def test_weak_current_wrists_cannot_publish_a_head(self):
        f,p=EvidenceTests().fixture();b=body()
        for n in ('left_wrist','right_wrist'):b['keypoints'][n]['confidence']=.6
        self.assertIsNone(supported_club(f,b,p))

    def test_bright_shaft_has_measurable_current_pixel_width(self):
        from ghostcaddie.video.shot_people import shaft_pixel_width
        f=np.full((300,400),100,np.uint8);cv2.line(f,(150,155),(195,235),240,2)
        self.assertTrue(np.isfinite(shaft_pixel_width(f,np.array([150.,155]),np.array([195.,235]))))

class CrowdAcquisitionRegressionTests(unittest.TestCase):
    def test_standing_folded_hands_are_not_setup(self):
        b=body()
        for n in ('left_wrist','right_wrist'):b['keypoints'][n]['point'][1]=132
        self.assertIsNone(person_score(b,(400,300)))

    def test_crowded_person_cannot_acquire_but_primary_can_continue(self):
        b=body();b['crowded_acquisition']=True
        self.assertIsNone(PersonIdentity().select([b],(400,300)))
        p=PersonIdentity();initial=p.select([body()],(400,300))
        self.assertEqual(p.select([b],(400,300))['track_id'],initial['track_id'])
