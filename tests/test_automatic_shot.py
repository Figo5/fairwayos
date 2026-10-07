"""Automatic initialization must come from image evidence, never clip labels."""
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from ghostcaddie.video.automatic_shot import (ball_component, detect_cup, observe_video,
    supported_ball_proposals, probe_source, verify_output)
from ghostcaddie.video.shot_render import render_video


def green_frame(ball=True):
    frame=np.full((240,320,3),(60,125,70),np.uint8)
    if ball:cv2.circle(frame,(210,165),9,(245,245,245),-1)
    return frame


class AutomaticEvidenceTests(unittest.TestCase):
    def test_a_missing_ball_is_not_a_prediction_observation(self):
        self.assertIsNone(ball_component(green_frame(False),(210,165),18))

    def test_invalid_previous_size_withholds_association(self):
        for size in [None, True, 0, -1, float('nan'), float('inf')]:
            with self.subTest(size=size):
                self.assertIsNone(ball_component(green_frame(), (210,165), size))

    def test_competing_current_components_are_withheld(self):
        frame=green_frame(False)
        cv2.circle(frame,(199,165),5,(245,245,245),-1)
        cv2.circle(frame,(221,165),5,(245,245,245),-1)
        self.assertIsNone(ball_component(frame,(207,165),11))

    def test_person_overlap_and_off_green_neural_proposals_are_rejected(self):
        box={'class_id':32,'score':.95,'box':[201,156,219,174]}
        self.assertEqual(len(supported_ball_proposals(green_frame(),[box])),1)
        person={'class_id':0,'score':.9,'box':[190,120,240,220]}
        self.assertEqual(supported_ball_proposals(green_frame(),[box,person]),[])
        partial_person={'class_id':0,'score':.9,'box':[215,120,250,220]}
        self.assertEqual(supported_ball_proposals(green_frame(),[box,partial_person]),[])
        self.assertEqual(supported_ball_proposals(np.zeros((240,320,3),np.uint8),[box]),[])

    def test_flag_without_visible_cup_is_not_a_target(self):
        frame=green_frame(False)
        cv2.rectangle(frame,(100,20),(104,140),(0,210,230),-1)
        self.assertIsNone(detect_cup(frame))
        cv2.ellipse(frame,(102,142),(17,5),0,0,360,(215,215,215),-1)
        cup=detect_cup(frame)
        self.assertIsNotNone(cup)
        self.assertAlmostEqual(cup['point'][0],102,delta=3)
        self.assertAlmostEqual(cup['point'][1],142,delta=3)

    def test_broad_yellow_clothing_and_pale_foot_are_not_a_flagstick(self):
        frame=green_frame(False)
        cv2.rectangle(frame,(100,20),(125,140),(0,210,230),-1)
        cv2.ellipse(frame,(112,142),(17,5),0,0,360,(215,215,215),-1)
        self.assertIsNone(detect_cup(frame))

    def test_cup_proposal_without_temporal_track_is_withheld(self):
        with mock.patch('ghostcaddie.video.automatic_shot.detect_cup',return_value={'point':[100.,100.],'score':100}), mock.patch('ghostcaddie.video.automatic_shot.track_ground_anchor',return_value=(None,{'reason':'not_supported'})):
            r=self._observe(lambda frame:[{'class_id':32,'score':.95,'box':[201,156,219,174]}])
        self.assertTrue(all(row['target'] is None for row in r['frames']))
        self.assertFalse(any(i['object']=='cup' for i in r['initializations']))

    def _observe(self, detector):
        frames=[(n,green_frame()) for n in range(12)]
        spec={'width':320,'height':240,'frames':12}
        with tempfile.TemporaryDirectory() as d, mock.patch('ghostcaddie.video.automatic_shot.decoded_frames',return_value=iter(frames)):
            return observe_video('filename_has_no_coordinates.mp4',spec,Path(d),detector,sample_step=3)

    def test_single_isolated_neural_detection_does_not_start_a_track(self):
        calls=[0]
        def detector(frame):
            calls[0]+=1
            return [{'class_id':32,'score':.9,'box':[201,156,219,174]}] if calls[0]==1 else []
        r=self._observe(detector)
        self.assertEqual(r['operator_seed_count'],0)
        self.assertFalse(r['initializations'])
        self.assertTrue(all(row['ball'] is None for row in r['frames']))

    def test_two_similarly_likely_balls_are_withheld(self):
        def detector(frame):return [{'class_id':32,'score':.95,'box':[201,156,219,174]}, {'class_id':32,'score':.94,'box':[110,156,128,174]}]
        r=self._observe(detector)
        self.assertFalse(r['initializations'])

    def test_repeated_neural_evidence_initializes_without_any_user_seed(self):
        r=self._observe(lambda frame:[{'class_id':32,'score':.95,'box':[201,156,219,174]}])
        self.assertEqual(r['operator_seed_count'],0)
        self.assertTrue(r['initializations'])
        self.assertTrue(all(row['ball'] is not None for row in r['frames']))
        self.assertEqual(r['initializations'][0]['method'],'automatic_neural_proposal')


class NativeTimingTests(unittest.TestCase):
    def test_variable_frame_cadence_survives_render_without_retiming(self):
        import av
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'variable.mp4';out=Path(d)/'out.mp4'
            tb=Fraction(1,1000);pts=[0,33,80,120,153,210]
            with av.open(str(source),'w',options={'video_track_timescale':'1000'}) as c:
                stream=c.add_stream('libx264',rate=30);stream.width=320;stream.height=240;stream.pix_fmt='yuv420p';stream.codec_context.time_base=stream.time_base=tb
                for p in pts:
                    f=av.VideoFrame.from_ndarray(green_frame(False),format='bgr24');f.pts=p;f.time_base=tb
                    for packet in stream.encode(f):c.mux(packet)
                for packet in stream.encode():c.mux(packet)
            spec=probe_source(source)
            rows=[{'frame':n,'scene':None,'ball':None,'target':None,'previous_to_current':None} for n in range(len(pts))]
            render_video(source,spec,{'frames':rows},out)
            check=verify_output(source,out)
            self.assertEqual(check['source_frames'],6)
            self.assertEqual(check['max_pts_delta_seconds'],0)
            self.assertEqual(probe_source(out)['pts'],[p/1000 for p in pts])

if __name__=='__main__':unittest.main()
