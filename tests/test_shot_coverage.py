import unittest
import cv2
import numpy as np
from ghostcaddie.video.shot_people import reflective_head,shaft_search_club,person_score,rear_subject_score,PersonIdentity,fuse_club_candidates,shaft_pixel_width
from tests.test_shot_people_metrics import body

class ReflectiveEvidenceTests(unittest.TestCase):
    def fixture(self):
        return np.full((300,400,3),(60,125,70),np.uint8)

    def test_actual_transverse_metal_pixels(self):
        f=self.fixture();cv2.line(f,(193,240),(205,240),(230,230,230),2)
        core=reflective_head(f,np.array([199,240]),200,[0,80],1)
        self.assertIsNotNone(core)
        self.assertEqual(core['core_polarity'],'bright')
        for x,y in core['support_pixels']:self.assertGreater(f[y,x,0],200)

    def test_round_ball_and_connected_ball_blade_are_withheld(self):
        for blade in (False,True):
            f=self.fixture();cv2.circle(f,(200,240),5,(230,230,230),-1)
            if blade:cv2.line(f,(200,240),(214,240),(230,230,230),2)
            self.assertIsNone(reflective_head(f,np.array([202,240]),200,[0,80],1))

    def test_bright_shaft_only_and_two_lobes_are_withheld(self):
        f=self.fixture();cv2.line(f,(200,233),(200,247),(230,230,230),3)
        self.assertIsNone(reflective_head(f,np.array([200,240]),200,[0,80],1))
        f=self.fixture()
        for y in (237,245):cv2.line(f,(195,y),(207,y),(230,230,230),1)
        self.assertIsNone(reflective_head(f,np.array([201,240]),200,[0,80],1))

    def test_geometric_search_still_requires_current_hands_and_shaft(self):
        f=self.fixture();cv2.line(f,(190,240),(208,240),(230,230,230),2)
        self.assertIsNone(shaft_search_club(f,body()))
        b=body();b['keypoints'].pop('left_wrist');b['keypoints'].pop('right_wrist')
        self.assertIsNone(shaft_search_club(f,b))

    def test_small_full_view_box_is_not_promoted_by_size_alone(self):
        b=body();b['bbox']=[115,45,185,95]
        self.assertIsNone(person_score(b,(400,300)))

    def test_rear_subject_needs_complete_unambiguous_current_pose(self):
        b=body();b['keypoints'].pop('nose')
        for n,p in {'left_ear':[140,55],'right_ear':[160,55],
                    'left_wrist':[140,125],'right_wrist':[160,125],
                    'left_knee':[135,190],'right_knee':[165,190],
                    'left_ankle':[135,235],'right_ankle':[165,235]}.items():
            b['keypoints'][n]={'point':p,'confidence':.95}
        self.assertIsNone(rear_subject_score(b,(400,300)))
        self.assertIsNone(PersonIdentity().select([b],(400,300)))
        b['crowded_acquisition']=True
        self.assertIsNone(rear_subject_score(b,(400,300)))
        self.assertIsNone(PersonIdentity().select([b],(400,300)))

    def test_ambiguous_branch_cannot_be_overridden_by_other_source(self):
        head={'point':[200,240]}
        self.assertIsNone(fuse_club_candidates(None,head,200,learned_ambiguous=True))
        self.assertIsNone(fuse_club_candidates(head,None,200,geometric_ambiguous=True))
        self.assertIsNone(fuse_club_candidates(head,{'point':[260,240]},200))

    def test_filled_region_edge_is_not_a_shaft_cross_section(self):
        gray=np.full((300,400),120,np.uint8);gray[:,200:]=20
        self.assertTrue(np.isinf(shaft_pixel_width(gray,np.array([200.,150.]),np.array([200.,240.]))))
