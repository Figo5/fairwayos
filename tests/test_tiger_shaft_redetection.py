import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import tiger_shaft_redetection as route
from tests.test_tiger_swing_route import payload, point


class ShaftSelectionTests(unittest.TestCase):
    def setUp(self):
        self.old=payload();self.new=copy.deepcopy(self.old)
        for c in self.new['clips']:
            c['targets']['clubhead']['points']=[point(0,'operator','operator_placement'),point(1,method='shaft_head',x=45),point(2,method='shaft_head_reacquired',x=46)]
        self.cohort=[{'id':'swing','stroke_type':'full_swing'},{'id':'putt','stroke_type':'putt'}]

    def test_prior_positions_win_only_current_measured_gap_filled(self):
        out,trace=route.select_available(self.old,self.new,self.cohort)
        ps=out['clips'][0]['targets']['clubhead']['points']
        self.assertEqual(ps[:2],self.old['clips'][0]['targets']['clubhead']['points'][:2])
        self.assertEqual(ps[2],self.new['clips'][0]['targets']['clubhead']['points'][2])
        self.assertEqual(trace['swing'][2]['engine'],'current_shaft_head')

    def test_balls_putts_and_inputs_preserved(self):
        old,new=copy.deepcopy(self.old),copy.deepcopy(self.new)
        out,_=route.select_available(self.old,self.new,self.cohort)
        self.assertEqual(out['clips'][1],old['clips'][1])
        for a,b in zip(old['clips'],out['clips']):self.assertEqual(a['targets']['ball'],b['targets']['ball'])
        self.assertEqual(self.old,old);self.assertEqual(self.new,new)

    def test_label_change_and_future_append_preserve_selected_prefix(self):
        out,trace=route.select_available(self.old,self.new,self.cohort)
        changed=[{**c,'labels':[{'frame':2,'x':-9999}]} for c in self.cohort]
        self.assertEqual(route.select_available(self.old,self.new,changed),(out,trace))
        self.new['clips'][0]['targets']['clubhead']['points'].append(point(3,method='shaft_head',x=47))
        later,_=route.select_available(self.old,self.new,self.cohort)
        self.assertEqual(later['clips'][0]['targets']['clubhead']['points'][:3],out['clips'][0]['targets']['clubhead']['points'])

    def test_later_anchor_derived_coordinates_wrong_method_or_score_refused(self):
        for change in [{'origin':'operator'},{'origin':'derived','x':None},{'method':'template'},{'score':.99}]:
            new=copy.deepcopy(self.new);new['clips'][0]['targets']['clubhead']['points'][2].update(change)
            with self.assertRaises(ValueError):route.select_available(self.old,new,self.cohort)

    def test_duplicate_collection_frames_wrong_source_or_seed_refused(self):
        for kind in ['collection','frame','sha','seed']:
            new=copy.deepcopy(self.new)
            if kind=='collection':new['clips'].append(new['clips'][0])
            elif kind=='frame':new['clips'][0]['targets']['clubhead']['points'].append(new['clips'][0]['targets']['clubhead']['points'][1])
            elif kind=='sha':new['clips'][0]['sha256']='f'*64
            else:new['clips'][0]['targets']['clubhead']['seed']['x']=999
            with self.assertRaises(ValueError):route.select_available(self.old,new,self.cohort)


class BodyEvidenceTests(unittest.TestCase):
    def test_content_source_geometry_and_current_frame_coverage_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'body.json';source={'id':'a','sha256':'a'*64,'width':720,'height':720}
            valid={'source':{'sha256':'a'*64,'width':720,'height':720},'scenes':[{'body':{'per_frame':[{'source_frame':0},{'source_frame':1}]}}]}
            def receipt(record):
                path.write_text(json.dumps(record));return {'clips':{'a':{'analysis_path':str(path),'analysis_sha256':route.bench.sha256_file(path),'source_sha256':source['sha256']}}}
            record=receipt(valid);self.assertEqual(set(route.body_rows(record,source,0,1)),{0,1})
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'bytes'):route.body_rows(record,source,0,1)
            for kind in ['source','geometry','duplicate','missing','bool']:
                bad=copy.deepcopy(valid)
                if kind=='source':bad['source']['sha256']='b'*64
                elif kind=='geometry':bad['source']['height']=100
                elif kind=='duplicate':bad['scenes'][0]['body']['per_frame'].append({'source_frame':0})
                elif kind=='missing':bad['scenes'][0]['body']['per_frame'].pop()
                else:bad['scenes'][0]['body']['per_frame'][0]['source_frame']=True
                with self.assertRaises(ValueError):route.body_rows(receipt(bad),source,0,1)

    def test_available_native_transform_and_alternate_fields_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'body.json';source={'id':'a','sha256':'a'*64,'width':720,'height':720}
            row={'source_frame':0,'available':True,'roi_xywh':[200,200,120,300],
                 'transform':{'roi_origin_xy':[200,200],'model_input_wh':[192,192],'letterbox_scale':.64},
                 'keypoints':[{'name':'wr_l','x':250.,'y':300.,'score':.8,'visible':True}]}
            def check(r):
                record={'source':{'sha256':'a'*64,'width':720,'height':720},'scenes':[{'body':{'per_frame':[r]}}]}
                path.write_text(json.dumps(record));receipt={'clips':{'a':{'analysis_path':str(path),'analysis_sha256':route.bench.sha256_file(path),'source_sha256':source['sha256']}}}
                return route.body_rows(receipt,source,0,0)
            self.assertEqual(check(row)[0]['coordinate_space'],'native_source_pixels')
            for field,value in [('roi_origin_xy',[0,0]),('model_input_wh',[256,256]),('letterbox_scale',0),('letterbox_scale',float('nan'))]:
                changed=copy.deepcopy(row);changed['transform'][field]=value
                with self.assertRaises(ValueError):check(changed)
            changed=copy.deepcopy(row);changed['keypoints'][0]['x']=.8
            with self.assertRaises(ValueError):check(changed)
            for field,value in [('ROI_xywh',[0,0,720,720]),('wrists',{}),('smoothed',True)]:
                with self.assertRaises(ValueError):check({**row,field:value})

    def test_new_fill_at_or_after_cut_refused(self):
        old=payload();new=copy.deepcopy(old)
        new['clips'][0]['targets']['clubhead']['points'][2]=point(2,method='shaft_head',x=46)
        cohort=[{'id':'swing','stroke_type':'full_swing','cut_frames':[2]},{'id':'putt','stroke_type':'putt','cut_frames':[]}]
        with self.assertRaisesRegex(ValueError,'precede cuts'):route.select_available(old,new,cohort)


class ShaftReplayExecutionTests(unittest.TestCase):
    def execute(self, directory, fail=False, tamper=False):
        root=Path(directory);paths={k:root/(k+'.json') for k in ['dataset','baseline','baseline-report','frames-receipt','body-receipt']}
        paths['dataset'].write_text('{}');paths['baseline'].write_text(json.dumps({'clips':[]}))
        model=root/'fixture-model.bin';model.write_bytes(b'fixture, never executed')
        paths['frames-receipt'].write_text(json.dumps({'clips':{}}))
        paths['body-receipt'].write_text(json.dumps({'clips':{},'model_path':str(model),'model_sha256':route.bench.sha256_file(model)}))
        paths['baseline-report'].write_text(json.dumps({'predictions':{'sha256':'f'*64 if tamper else route.bench.sha256_file(paths['baseline'])},'reference':{'sha256':'d'*64},'runtime':{}}))
        output=root/'shaft.predictions.json';argv=['replay']+[v for k,p in paths.items() for v in ['--'+k,str(p)]]+['--out',str(output)]
        with patch('sys.argv',argv),patch.object(route.bench,'load_dataset',return_value=({'clips':[]},'d'*64)), \
                patch.object(route.bench,'validate_predictions',side_effect=lambda p,d:p), \
                patch.object(route.bench,'evaluate',side_effect=ValueError('evaluation failed') if fail else lambda *a,**k:k), \
                patch.object(route.bench,'render_markdown',return_value='fixture report'):
            route.main()
        return output

    def test_failed_evaluation_publishes_no_candidate(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,'evaluation failed'):self.execute(root,fail=True)
            self.assertFalse(list(Path(root).glob('shaft*')))

    def test_tampered_baseline_refused_before_writes(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,'bind'):self.execute(root,tamper=True)
            self.assertFalse(list(Path(root).glob('shaft*')))

    def test_report_binds_exact_prediction_and_raw_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            output=self.execute(root);report=json.loads(Path(str(output)+'.report.json').read_text())
            self.assertEqual(report['predictions_sha256'],route.bench.sha256_file(output))
            self.assertEqual(report['runtime']['raw_sha256'],route.bench.sha256_file(Path(root)/'shaft.raw.json'))
            self.assertFalse(report['runtime']['sam_masks_used_to_propose'])


class MeasurementBindingTests(unittest.TestCase):
    def fixture(self, root):
        from argparse import Namespace
        from ghostcaddie.demo import shaft_head as engine
        baseline=payload();measured=copy.deepcopy(baseline)
        measured['variant']='full_swing_shaft_redetection_research'
        measured['clips'][0]['targets']['clubhead']['points'][2]=point(2,method='shaft_head',x=46)
        seed=baseline['clips'][0]['targets']['clubhead']['seed']
        source={'id':'swing','path':'/fixture.mp4','sha256':baseline['clips'][0]['sha256'],
                'stroke_type':'full_swing','width':720,'height':720,'window':{'end_frame':2},
                'cut_frames':[],'targets':{'clubhead':{'seed':seed}}}
        dataset={'clips':[source,{'id':'putt','stroke_type':'putt','cut_frames':[]}]}
        args=Namespace(dataset=root/'dataset.json',baseline=root/'baseline.json',
                       measured=root/'measured.json',measured_report=root/'report.json')
        for path,value in [(args.dataset,{}),(args.baseline,baseline),(args.measured,measured)]:path.write_text(json.dumps(value))
        frame_items={}
        for f in range(3):
            path=root/('pixel'+str(f));path.write_bytes(('native fixture '+str(f)).encode())
            frame_items[str(f)]={'path':str(path),'sha256':route.bench.sha256_file(path)}
        expected={k:source[k] for k in ('id','path','sha256','stroke_type','window','cut_frames')};expected['seed']=seed
        frames=root/'frames.json';frames.write_text(json.dumps({'clips':{'swing':{'input':expected,'crop':[0,0,720,720],'frames':frame_items}}}))
        analysis=root/'body.json';analysis.write_text(json.dumps({'source':{'sha256':source['sha256'],'width':720,'height':720},'scenes':[{'body':{'per_frame':[{'source_frame':f,'available':False} for f in range(3)]}}]}))
        bodies=root/'bodies.json';bodies.write_text(json.dumps({'clips':{'swing':{'analysis_path':str(analysis),'analysis_sha256':route.bench.sha256_file(analysis),'source_sha256':source['sha256']}}}))
        ps=measured['clips'][0]['targets']['clubhead']['points']
        raw=root/'raw.json';raw.write_text(json.dumps({'profile':engine.PROFILE,'clips':{'swing':{'input':expected,'crop':[0,0,720,720],'points':ps,'raw':[{'frame':p['source_frame'],'decision':{'frame':p['source_frame'],**{k:p[k] for k in ('status','origin','method','x','y','score')}}} for p in ps]}}}))
        runtime={'profile':engine.PROFILE,'engine_sha256':route.bench.sha256_file(Path(engine.__file__)),
                 'baseline_predictions_sha256':route.bench.sha256_file(args.baseline),'raw_path':str(raw),'raw_sha256':route.bench.sha256_file(raw)}
        for key,path in [('geometry_module',route.ROOT/'ghostcaddie/demo/swing_head.py'),('descriptor_module',route.ROOT/'ghostcaddie/demo/head_identity.py'),('tool',Path(route.__file__)),('benchmark',Path(route.bench.__file__))]:runtime[key+'_sha256']=route.bench.sha256_file(path)
        for key,path in [('frames_receipt',frames),('body_receipt',bodies)]:runtime[key+'_path']=str(path);runtime[key+'_sha256']=route.bench.sha256_file(path)
        report={'predictions':{'sha256':route.bench.sha256_file(args.measured),'variant':measured['variant']},'reference':{'sha256':route.bench.sha256_file(args.dataset)},'runtime':runtime}
        args.measured_report.write_text(json.dumps(report))
        return args,dataset,baseline,engine,report

    def test_profile_code_receipt_raw_and_decisions_bound(self):
        for kind in ('valid','profile','engine','dependency','receipt','raw','decision','point','geometry','pixels','body'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);args,dataset,baseline,engine,report=self.fixture(root)
                rt=report['runtime']
                if kind=='profile':rt['profile']={}
                elif kind=='engine':rt['engine_sha256']='f'*64
                elif kind=='dependency':rt['geometry_module_sha256']='f'*64
                elif kind=='receipt':(root/'frames.json').write_text('{}')
                elif kind=='raw':(root/'raw.json').write_text('{}')
                elif kind in ('decision','point','geometry'):
                    raw=json.loads((root/'raw.json').read_text());entry=raw['clips']['swing']
                    if kind=='decision':entry['raw'][2]['decision']['x']=999
                    elif kind=='point':entry['points'][2]['x']=999
                    else:entry['crop'][0]=1
                    (root/'raw.json').write_text(json.dumps(raw));rt['raw_sha256']=route.bench.sha256_file(root/'raw.json')
                elif kind=='pixels':(root/'pixel2').write_bytes(b'tampered')
                elif kind=='body':(root/'body.json').write_text('{}')
                args.measured_report.write_text(json.dumps(report))
                with patch.object(route.bench,'validate_predictions',side_effect=lambda p,d:p):
                    if kind=='valid':
                        out,raw,runtime=route.run_selection(args,dataset,baseline,engine)
                        self.assertEqual(out['clips'][0]['targets']['clubhead']['points'][2]['x'],46)
                    else:
                        with self.assertRaises(ValueError):route.run_selection(args,dataset,baseline,engine)


if __name__=='__main__':unittest.main()
