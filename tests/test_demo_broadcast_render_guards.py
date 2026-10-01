"""Regressions found by independent review of real Masters renders."""
import unittest
from unittest.mock import patch
from ghostcaddie.demo import broadcast_config as bc
from ghostcaddie.demo.errors import DemoConfigError
from tests.test_demo_broadcast import base_config, base_scene
from tests import test_demo_broadcast as fixtures
from tests.demo_fixtures import TempDirTestCase


class BroadcastRenderGuardTests(unittest.TestCase):
    def setUp(self):
        from ghostcaddie.demo import worker
        self.worker = worker

    def test_putting_badges_cover_every_marked_phase(self):
        marks = {'address': 0, 'takeaway': 26, 'downswing': 37, 'impact': 40, 'finish': 60}
        for frame, expected in [(0,'ADDRESS'),(30,'BACKSTROKE'),(38,'FORWARD STROKE'),
                                (45,'CONTACT'),(60,'FINISH')]:
            with self.subTest(frame=frame):
                self.assertEqual(self.worker._broadcast_badge('putt', frame, marks), expected)

    def test_metric_gate_survives_roundtrip_and_refuses_bad_bounds(self):
        cfg=base_config(scenes=[base_scene(metric_hide_frames=[[24,32]])])
        parsed=bc.parse_broadcast(cfg)
        self.assertEqual(bc.parse_broadcast(parsed.to_dict()).scenes[0].metric_hide_frames, ((24,32),))
        for ranges in [[[9,12]],[[32,24]],[[24,32],[30,36]],[[True,30]]]:
            with self.subTest(ranges=ranges), self.assertRaises(DemoConfigError):
                bc.parse_broadcast(base_config(scenes=[base_scene(metric_hide_frames=ranges)]))

    def test_metric_gate_does_not_hold_a_body_value_through_contact(self):
        records=[{'source_frame': f,'available': True,'value': 10 if f<5 else 20}
                 for f in range(12)]
        def measurements(items, metrics, **kw):
            return {'hip_tilt': {r['source_frame']:r['value'] if r['available'] else None for r in items}}
        with patch.object(self.worker,'_frame_metric_series',side_effect=measurements):
            series=self.worker._broadcast_card_display(records, ['hip_tilt'], handedness='right',
                    wrist_reference=None,min_run=3,hidden=[[4,6]])
        for frame in range(4,7):
            self.assertIsNone(series['display']['hip_tilt'][frame]['value'])
            self.assertFalse(series['display']['hip_tilt'][frame]['held'])
        self.assertEqual(series['display']['hip_tilt'][7]['value'],20)
        self.assertEqual(series['records']['hip_tilt']['operator_gated_frames'],3)
        self.assertTrue(records[5]['available'])  # Skeleton measurement remains untouched.
        gated=self.worker._broadcast_metric_record(records[5],[[4,6]])
        cards=self.worker._showcase_card_payload(['hip_tilt','tempo'],gated,
               {'tempo':{'state':'supported','value':3.667}},handedness='right')
        self.assertFalse(cards[0]['available'])
        self.assertEqual(cards[1]['value'],'3.67:1')

    def test_text_scrim_adds_contrast_without_changing_default_renderer(self):
        import numpy as np
        from ghostcaddie.demo import render_showcase as rs,showcase as sc
        picture=np.full((1080,1920,3),255,dtype=np.uint8)
        layout=sc.plan_layout(1920,1080,(),card_position='top')
        chrome={'brand':'FAIRWAY OS','disclosure':'OPERATOR ASSISTED'}
        default,_=rs.compose_chrome(picture,layout=layout,chrome=chrome)
        scrim,_=rs.compose_chrome(picture,layout=layout,chrome={**chrome,'text_scrim':True})
        y=layout.brand.y-3;x=layout.brand.x+20
        self.assertGreater(default[y,x].mean(),240)
        self.assertLess(scrim[y,x].mean(),140)


class PuttCardIntegrationTests(TempDirTestCase):
    _clip = fixtures.BroadcastEndToEndTests._clip
    def test_putt_render_shows_published_tempo_and_measure_only_writes_no_video(self):
        import json,sys
        from ghostcaddie.demo.broadcast import run_broadcast
        clip=self._clip()
        cfg=base_config(scenes=[base_scene(stroke_type='putt',window={'start_frame':0,'end_frame':40},
              keyframes={'address':0,'backstroke':10,'forward_stroke':30,'contact':35,'finish':39},
              roi=[10,10,60,90],targets={},seeds={})],presentation={'metrics':['shoulder_tilt','hip_tilt','tempo']})
        path=self.tmp/'putt.json';path.write_text(json.dumps(cfg))
        out=self.tmp/'putt-out';r=run_broadcast(str(clip),str(path),str(out),cv_python=sys.executable)
        self.assertIn('tempo',r.render['scenes'][0]['metrics'])
        self.assertEqual(r.scenes[0]['phases']['tempo']['backswing_downswing_ratio'],4.0)
        out2=self.tmp/'measure-only';run_broadcast(str(clip),str(path),str(out2),cv_python=sys.executable,render=False)
        self.assertFalse((out2/'broadcast_annotated.mp4').exists())
