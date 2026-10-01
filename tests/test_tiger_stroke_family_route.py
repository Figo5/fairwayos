import copy
import unittest
import hashlib
import json
import tempfile
from pathlib import Path
from unittest import mock
from tools import tiger_stroke_family_route as route
from tools.tiger_stroke_family_route import compose

class FamilyRouteTests(unittest.TestCase):
    def test_only_declared_family_selects_heads_and_ball_is_never_changed(self):
        base={'clips':[{'id':c,'targets':{'clubhead':{'points':['classic']},'ball':{'points':['ball']}}} for c in ['p','s']]}
        learned=copy.deepcopy(base)
        learned['variant']='declared_cohort_mask_core_appearance'
        for c in learned['clips']:c['targets']['clubhead']['points']=['learned']
        original=copy.deepcopy(base)
        cohort=[{'id':'p','stroke_type':'putt'},{'id':'s','stroke_type':'full_swing'}]
        result,engines=compose(base,learned,cohort)
        self.assertEqual(result['clips'][0]['targets']['clubhead']['points'],['learned'])
        self.assertEqual(result['clips'][1]['targets']['clubhead']['points'],['classic'])
        self.assertEqual(base,original)
        self.assertEqual(result['clips'][0]['targets']['ball'],base['clips'][0]['targets']['ball'])
        self.assertEqual(engines['s'],'classic head baseline replay')
        self.assertEqual(engines['p'],'learned head policy: declared_cohort_mask_core_appearance')
        # Scores or outcome-like metadata cannot change the family decision.
        learned['clips'][1]['score']=1;learned['clips'][0]['score']=0
        self.assertEqual(compose(base,learned,cohort)[0]['clips'][1]['targets']['clubhead']['points'],['classic'])
        learned['clips'][0]['targets']['clubhead']['points'].append('later edit')
        self.assertEqual(result['clips'][0]['targets']['clubhead']['points'],['learned'])

    def test_partial_cohort_and_unknown_family_are_refused(self):
        p={'clips':[{'id':'x','targets':{}}]}
        with self.assertRaises(ValueError):compose(p,p,[])
        with self.assertRaises(ValueError):compose(p,p,[{'id':'x','stroke_type':'unknown'}])

    def test_executed_report_binds_engine_and_profile_before_any_write(self):
        for case in ['valid','changed_report_hash','missing_engine','report_as_output']:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                baseline={'clips':[{'id':cid,'targets':{'clubhead':{'points':['classic']},'ball':{'points':['ball']}}} for cid in ['p','s']]}
                learned=copy.deepcopy(baseline);learned['variant']='declared_cohort_mask_core_appearance'
                learned['clips'][0]['targets']['clubhead']['points']=['appearance']
                paths={name:root/(name+'.json') for name in ['baseline','learned','dataset','report','out']}
                paths['baseline'].write_text(json.dumps(baseline));paths['learned'].write_text(json.dumps(learned));paths['dataset'].write_text('{}')
                report={'predictions':{'sha256':hashlib.sha256(paths['learned'].read_bytes()).hexdigest()},
                    'runtime':{'actual_engine_by_clip':{'p':'SAM2 mask-core appearance','s':'SAM2 mask-core appearance'},'profile':{'immutable_seed':True}}}
                if case=='changed_report_hash':report['predictions']['sha256']='changed'
                if case=='missing_engine':del report['runtime']['actual_engine_by_clip']['s']
                paths['report'].write_text(json.dumps(report));before=paths['report'].read_bytes()
                out=paths['report'] if case=='report_as_output' else paths['out']
                data={'clips':[{'id':cid,'stroke_type':stroke,'path':str(root/(cid+'.mp4'))} for cid,stroke in [('p','putt'),('s','full_swing')]]}
                argv=['route','--dataset',str(paths['dataset']),'--baseline',str(paths['baseline']),'--learned',str(paths['learned']),'--learned-report',str(paths['report']),'--out',str(out)]
                with mock.patch('sys.argv',argv),mock.patch.object(route.bench,'load_dataset',return_value=(data,'fixture-reference')),mock.patch.object(route.bench,'validate_predictions',side_effect=lambda p,d:p),mock.patch.object(route.bench,'evaluate',side_effect=lambda *a,**kw:{'runtime':kw['runtime']}),mock.patch.object(route.bench,'render_markdown',return_value='fixture'):
                    if case=='valid':
                        route.main();result=json.loads(Path(str(out)+'.report.json').read_text())['runtime']
                        self.assertEqual(result['actual_engine_by_clip'],{'p':'SAM2 mask-core appearance','s':'classic head baseline replay'})
                        self.assertEqual(result['learned_executed_report']['runtime']['profile'],{'immutable_seed':True})
                        self.assertEqual(result['learned_executed_report']['sha256'],hashlib.sha256(before).hexdigest())
                    else:
                        with self.assertRaises(ValueError):route.main()
                        self.assertFalse(paths['out'].exists())
                        self.assertFalse(Path(str(out)+'.report.json').exists())
                self.assertEqual(paths['report'].read_bytes(),before)

if __name__=='__main__':unittest.main()
