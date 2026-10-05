"""Source-bound saved observations must reach the renderer without assistance."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ghostcaddie.demo.broadcast_saved import attach_saved_tracks, DISCLOSURE
from ghostcaddie.demo.errors import DemoConfigError
from ghostcaddie.demo.saved_tracks import load_saved_tracks
from ghostcaddie.demo.tracking import trail_window, track_from_dict


class SavedBroadcastTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'saved.json'
        self.sha = 'a' * 64
        self.record = SimpleNamespace(sha256=self.sha, width=100, height=80, frame_count=10)
        self.resolved = SimpleNamespace(source=self.record)
        def point(f, status, origin, x=None, y=None, method=''):
            return dict(source_frame=f,status=status,origin=origin,x=x,y=y,method=method,score=None)
        self.payload = dict(schema_version='fairwayos-target-predictions.v1',variant='research', clips=[
            dict(id='shot',sha256=self.sha,targets={'clubhead':dict(seed=dict(frame=2,x=50,y=60),points=[
                point(2,'manual_anchor','operator',50,60,'operator_placement'),
                point(3,'observed_template','automatic',51,60,'sam2_head_region'),
                point(4,'lost','derived'),
                point(6,'observed_template','automatic',55,60,'sam2_head_region_reacquired')])})])
        self.write()
        self.scene = dict(seeds={'clubhead':dict(frame=2,x=70,y=65)},
                          corrections={'clubhead':[dict(frame=4,x=70,y=65)]},
                          manual_segments={'clubhead':{'replace_automatic':True}},
                          body={'model_path':'pose'},phases={'keyframes':{'contact':3}})
        self.spec = dict(source_sha256=self.sha,showcase={'disclosure':'assisted'},broadcast={
            'presentation':{'disclosure':'assisted'},'scenes':[
                dict(id='stroke',window=dict(start_frame=0,end_frame=9),spec=self.scene),
                dict(id='reaction',window=dict(start_frame=10,end_frame=11),spec=None)]})

    def write(self): self.path.write_text(json.dumps(self.payload))

    def test_import_keeps_gaps_and_reacquisition_excludes_assistance(self):
        before = copy.deepcopy(self.payload)
        result = attach_saved_tracks(self.spec,self.resolved,self.path)
        track = track_from_dict('clubhead',self.scene['saved_tracks']['clubhead'])
        by = {p.source_frame:p for p in track.points}
        self.assertEqual(by[2].origin,'operator')
        self.assertEqual((by[3].x,by[6].x),(51,55))
        self.assertIsNone(by[6].score)
        self.assertFalse(by[4].drawn); self.assertFalse(by[5].drawn)
        self.assertEqual([p.source_frame for p in trail_window(track.points,6)],[6])
        self.assertEqual(track.mode,'saved_automatic_research')
        for key in ['manual_segments','corrections','seeds']: self.assertEqual(self.scene[key],{})
        self.assertEqual(self.scene['body'],{'model_path':'pose'})
        self.assertEqual(self.scene['phases']['keyframes'],{'contact':3})
        self.assertEqual(self.spec['showcase']['disclosure'],DISCLOSURE)
        self.assertEqual(result['excluded_configured_seeds']['stroke']['clubhead']['x'],70)
        self.assertFalse(result['model_rerun']);self.assertFalse(result['accuracy_claim'])
        self.assertEqual(json.loads(self.path.read_text()),before)

    def test_wrong_source_missing_seed_window_and_changed_snapshot_refuse(self):
        self.spec['source_sha256'] = 'b'*64
        with self.assertRaises(DemoConfigError): attach_saved_tracks(self.spec,self.resolved,self.path)
        self.spec['source_sha256'] = self.sha
        with self.assertRaises(DemoConfigError): attach_saved_tracks(self.spec,self.resolved,self.path,'c'*64)
        self.spec['broadcast']['scenes'][0]['window']['start_frame'] = 3
        with self.assertRaises(DemoConfigError): attach_saved_tracks(self.spec,self.resolved,self.path)

    def test_single_seed_is_not_carried_to_a_later_camera_scene(self):
        later=copy.deepcopy(self.spec['broadcast']['scenes'][0]);later['id']='later';later['window']=dict(start_frame=7,end_frame=9)
        self.spec['broadcast']['scenes'][0]['window']['end_frame']=6
        self.spec['broadcast']['scenes'].append(later)
        attach_saved_tracks(self.spec,self.resolved,self.path)
        self.assertEqual(later['spec']['saved_tracks'],{})
        frames=self.scene['saved_tracks']['clubhead']['points']
        self.assertEqual([p['source_frame'] for p in frames],list(range(7)))

    def test_digest_is_of_exact_file_bytes_and_invalid_utf8_is_a_config_error(self):
        raw=json.dumps(self.payload,indent=2).replace('\n','\r\n').encode()
        self.path.write_bytes(raw)
        loaded=load_saved_tracks(self.path,[self.record])
        self.assertEqual(loaded['prediction_sha256'],hashlib.sha256(raw).hexdigest())
        self.path.write_bytes(b'\xff')
        with self.assertRaises(DemoConfigError):load_saved_tracks(self.path,[self.record])

    def test_seed_anchor_disagreement_and_invented_sam2_confidence_refuse(self):
        target=self.payload['clips'][0]['targets']['clubhead']
        target['points'][0]['x']=51;self.write()
        with self.assertRaises(DemoConfigError):load_saved_tracks(self.path,[self.record])
        target['points'][0]['x']=50;target['points'][1]['score']=.99;self.write()
        with self.assertRaises(DemoConfigError):load_saved_tracks(self.path,[self.record])

    def test_export_passes_accepted_cuts_and_overrides_to_the_importer(self):
        self.resolved.config=SimpleNamespace(cuts=[SimpleNamespace(
            decision='accepted',kind='hard_cut',span=None,frame=5)],overrides=[{'decision':'accept','frame':8}])
        result=attach_saved_tracks(self.spec,self.resolved,self.path)
        self.assertEqual(result['cut_frames'],[5,8])
        by={p['source_frame']:p for p in self.scene['saved_tracks']['clubhead']['points']}
        self.assertEqual(by[6]['status'],'missing')
        self.assertIsNone(by[6]['x'])

    def test_saved_report_origin_and_limitations_match_imported_points(self):
        from ghostcaddie.demo.broadcast import _limitations, _tracking_origin_summary
        attach_saved_tracks(self.spec,self.resolved,self.path)
        summary = _tracking_origin_summary(self.scene['saved_tracks'], saved=True)
        self.assertEqual(summary['clubhead']['origin'],'saved_automatic_research')
        self.assertIsNone(summary['clubhead']['is_a_detection'])
        self.assertIn('no saved seed',summary['ball']['reason'])
        limitations = _limitations(self.resolved,{},self.spec)
        self.assertIn('saved automatic research', limitations[0])
        self.assertNotIn('not automatic perception',' '.join(limitations))
        self.assertNotIn('markers are operator-anchored',' '.join(limitations))

    def test_worker_consumes_import_without_running_classic_or_manual_tracker(self):
        try: from ghostcaddie.demo import worker
        except ImportError: self.skipTest('CV worker unavailable')
        attach_saved_tracks(self.spec,self.resolved,self.path)
        self.scene.update(interval=dict(start_frame=0,end_frame=9),native_fps=30,targets={},interpolation={})
        with patch.object(worker,'body_pass',return_value={'per_frame':[]}), \
             patch.object(worker,'measure_target',side_effect=AssertionError('classic tracker ran')), \
             patch('ghostcaddie.demo.phases.resolve_phases',return_value={}) as phases:
            _,objects,serialized,_,raw=worker._measure(None,self.scene)
        phases.assert_called_once()
        self.assertEqual(phases.call_args.kwargs['tracks'],{})
        self.assertEqual(set(objects),{'clubhead'})
        self.assertEqual(serialized['clubhead']['points'][4]['status'],'lost')
        self.assertEqual(raw,{'clubhead':{}})
