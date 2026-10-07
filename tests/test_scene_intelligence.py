import unittest
from dataclasses import replace
from ghostcaddie.video.scene_intelligence import ObservedGroundTrail, StrategyBinding, build_strategy_layer
from ghostcaddie.video.calibration import VideoCalibration
from ghostcaddie.geometry import Point2D, CoordinateSystem
from ghostcaddie.models import ClubProfile, PlayerProfile, CourseModel, ShotEvent

I=((1,0,0),(0,1,0),(0,0,1))
class GroundTrailTests(unittest.TestCase):
    def test_camera_zoom_pan_reprojects_stationary_ball_without_false_motion(self):
        trail=ObservedGroundTrail()
        trail.update(10,'green',(100,100))
        result=trail.update(11,'green',(210,205),((2,0,10),(0,2,5),(0,0,1)))
        self.assertEqual(result,((210,205),(210,205)))

    def test_detection_gaps_bad_registration_and_cuts_do_not_join(self):
        for frame,scene,ball,matrix in [(2,'green',None,I),(2,'green',(2,2),None),(3,'green',(3,3),I),(2,'cut',(2,2),I)]:
            with self.subTest(frame=frame,scene=scene,ball=ball,matrix=matrix):
                t=ObservedGroundTrail();t.update(1,'green',(1,1))
                result=t.update(frame,scene,ball,matrix)
                self.assertLessEqual(len(result),1)

    def test_age_limit_preserves_only_recent_observations(self):
        t=ObservedGroundTrail(2)
        for n in range(4):r=t.update(n,'green',(n,n),I)
        self.assertEqual(r,((2,2),(3,3)))

class StrategyTests(unittest.TestCase):
    def setUp(self):
        self.player=PlayerProfile('fixture',{'7i':ClubProfile('7i',150,6,8,sample_size=30)})
        self.course=CourseModel('synthetic',3,CoordinateSystem(),
            [[Point2D(0,-20),Point2D(160,-20),Point2D(160,20),Point2D(0,20)]],
            [[Point2D(140,-15),Point2D(160,-15),Point2D(160,15),Point2D(140,15)]],Point2D(150,0))
        self.shot=ShotEvent('fixture','fixture','fixture',1,1,Point2D(0,0),Point2D(150,0),Point2D(151,2),'tee','7i',150,{'speed_mph':0,'direction_deg':0},'2026-10-06')
        self.calibration=VideoCalibration(1000,800,'pixels','yards',((0,0),(1000,0),(1000,800),(0,800)),((0,-40),(200,-40),(200,40),(0,40)))
        self.binding=StrategyBinding('a'*64,'green',10,'synthetic')

    def test_existing_engine_target_is_projected_with_explicit_scenario_label(self):
        layer=build_strategy_layer(self.shot,self.course,self.player,self.calibration,self.binding)
        self.assertEqual(layer.label,'SCENARIO TARGET')
        pixel=layer.target_in_frame('a'*64,'green',I,reference_frame=10)
        world=self.calibration.to_engine(pixel)
        self.assertAlmostEqual(world.x,layer.pipeline.recommendation.recommended_target.x)
        self.assertAlmostEqual(world.y,layer.pipeline.recommendation.recommended_target.y)
        self.assertIsNone(layer.target_in_frame('a'*64,'green',I,reference_frame=11))
        self.assertIsNone(layer.target_in_frame('b'*64,'green',I,reference_frame=10))
        self.assertIsNone(layer.target_in_frame('a'*64,'cut',I,reference_frame=10))
        self.assertIsNone(layer.target_in_frame('a'*64,'green',None,reference_frame=10))

    def test_missing_inputs_pixels_as_yards_and_putting_are_rejected(self):
        with self.assertRaises(ValueError):build_strategy_layer(self.shot,self.course,None,self.calibration,self.binding)
        with self.assertRaises(ValueError):build_strategy_layer(self.shot,self.course,self.player,replace(self.calibration,engine_units='reference_image_pixels'),self.binding)
        with self.assertRaises(ValueError):build_strategy_layer(replace(self.shot,lie='green'),self.course,self.player,self.calibration,self.binding)


class GroundAnchorTests(unittest.TestCase):
    def test_missing_anchor_is_not_held_and_blank_patch_is_rejected(self):
        import numpy as np
        from ghostcaddie.video.green_scene_tracker import track_ground_anchor
        blank=np.zeros((160,200),np.uint8)
        point,quality=track_ground_anchor(blank,blank,None)
        self.assertIsNone(point)
        point,quality=track_ground_anchor(blank,blank,(100,80))
        self.assertIsNone(point)

    def test_ground_patch_tracks_camera_pan(self):
        import cv2
        import numpy as np
        from ghostcaddie.video.green_scene_tracker import track_ground_anchor
        rng=np.random.default_rng(42)
        previous=cv2.GaussianBlur(rng.integers(0,255,(160,200),dtype=np.uint8),(3,3),0)
        current=cv2.warpAffine(previous,np.float32([[1,0,4],[0,1,3]]),(200,160))
        point,quality=track_ground_anchor(previous,current,(100,80))
        self.assertIsNotNone(point)
        self.assertAlmostEqual(point[0],104,delta=.2)
        self.assertAlmostEqual(point[1],83,delta=.2)

if __name__=='__main__':unittest.main()
