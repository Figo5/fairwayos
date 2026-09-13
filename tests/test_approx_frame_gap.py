import unittest
from approx_analytics import Point, build_metrics

class SourceGapTests(unittest.TestCase):
    def test_missing_source_frame_resets_velocity(self):
        rows = [{'source_frame':i, 'ball':Point(x,0), 'head':Point(x,0)} for i,x in [(1,0),(3,10),(4,12)]]
        values=build_metrics(rows)
        self.assertIsNone(values[1]['ball_disp_px'])
        self.assertIsNone(values[1]['head_disp_px'])
        self.assertEqual(values[2]['ball_disp_px'],2)
