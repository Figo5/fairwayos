import unittest
from tools.tiger_learned_head_probe import unique_center, seed_identity, learned_point, observation
class NativeBoxTests(unittest.TestCase):
    def test_native_crop_offset_not_model_input_coordinate(self):
        p=unique_center([{'confidence':.7,'xyxy':[10,20,30,40]}],[266,0,684,720],.25)
        self.assertEqual(p,{'x':286.,'y':30.,'score':.7})
    def test_competing_boxes_are_withheld_without_top_score_shortcut(self):
        self.assertIsNone(unique_center([{'confidence':.9,'xyxy':[0,0,10,10]}, {'confidence':.3,'xyxy':[20,20,30,30]}],[0,0,100,100],.25))
    def test_below_threshold_does_not_fill_gap(self):
        self.assertIsNone(unique_center([{'confidence':.24,'xyxy':[0,0,10,10]}],[0,0,100,100],.25))
    def test_receipt_note_does_not_change_seed_but_coordinates_do(self):
        original={'frame':0,'x':10,'y':20}
        self.assertEqual(seed_identity(original),seed_identity({**original,'note':'Address'}))
        self.assertNotEqual(seed_identity(original),seed_identity({**original,'x':11}))
        with self.assertRaises(ValueError):
            seed_identity({**original,'scale':2})
        with self.assertRaises(ValueError):
            seed_identity({'frame':0,'x':10})
    def test_detector_observation_uses_shared_status_and_actual_method(self):
        from ghostcaddie.demo.tracking import AUTOMATIC_STATUSES
        p=learned_point(7,{'x':10,'y':20,'score':.6})
        self.assertIn(p['status'],AUTOMATIC_STATUSES)
        self.assertEqual(p['method'],'learned_bbox_center')
        self.assertEqual(p['source_frame'],7)
        self.assertEqual(p['score'],.6)
    def test_seed_is_operator_anchor_even_with_conflicting_detector(self):
        p=observation(7,{'frame':7,'x':10,'y':20},{'x':100,'y':200,'score':.99},[])
        self.assertEqual((p['x'],p['y']),(10,20))
        self.assertEqual(p['origin'],'operator')
        self.assertIsNone(p['score'])
    def test_cut_suppresses_present_and_future_proposals_without_backfill(self):
        seed={'frame':0,'x':10,'y':20};center={'x':11,'y':21,'score':.8}
        self.assertEqual(observation(3,seed,center,[4])['origin'],'automatic')
        for frame in [4,5,10]:
            self.assertEqual(observation(frame,seed,center,[4])['status'],'missing')
        self.assertEqual(observation(3,seed,None,[4])['status'],'missing')
