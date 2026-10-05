import copy
import unittest
from tools.tiger_proposal_support import select_gaps


def inputs():
    def point(f,x=None,method=''):
        return {'source_frame':f,'x':x,'y':x,'origin':'automatic' if x is not None else 'derived',
                'status':'observed_template' if x is not None else 'missing','score':None,'method':method}
    seed={'frame':0,'x':10,'y':10};anchor={**point(0,10),'origin':'operator','status':'manual_anchor'}
    head={'seed':seed,'points':[anchor,point(1,11,'shaft_head'),point(2),point(3)]}
    base={'clips':[{'id':'a','sha256':'abc','targets':{'ball':{'points':[point(1,5,'template')]},'clubhead':head}}]}
    new=copy.deepcopy(base);new['clips'][0]['targets']['clubhead']['points'][1]=point(1,999,'learned_shaft_core')
    new['clips'][0]['targets']['clubhead']['points'][2]=point(2,12.125,'learned_shaft_core')
    roster=[{'id':'a','sha256':'abc','stroke_type':'full_swing','window':{'end_frame':3},'cut_frames':[3]}]
    return base,new,roster


class SelectionContracts(unittest.TestCase):
    def test_existing_observations_and_balls_survive_exactly_and_inputs_unchanged(self):
        a,b,r=inputs();before=copy.deepcopy(a);out,added=select_gaps(a,b,r)
        self.assertEqual(out['clips'][0]['targets']['clubhead']['points'][:2],a['clips'][0]['targets']['clubhead']['points'][:2])
        self.assertEqual(out['clips'][0]['targets']['ball'],a['clips'][0]['targets']['ball'])
        self.assertEqual(added[0]['point']['x'],12.125)
        self.assertEqual(a,before)

    def test_cut_and_network_centers_cannot_fill(self):
        for changed in ['cut','method','score','operator','status','finite','old_assistance']:
            with self.subTest(changed=changed):
                a,b,r=inputs();p=b['clips'][0]['targets']['clubhead']['points'][2]
                if changed=='cut':r[0]['cut_frames']=[2]
                if changed=='method':p['method']='learned_bbox_center'
                if changed=='score':p['score']=.99
                if changed=='operator':p['origin']='operator';p['status']='manual_anchor'
                if changed=='status':p['status']='missing'
                if changed=='finite':p['x']=float('nan')
                if changed=='old_assistance':a['clips'][0]['targets']['clubhead']['points'][2]['origin']='interpolated'
                with self.assertRaises(ValueError):select_gaps(a,b,r)

    def test_putt_is_preserved_without_new_engine(self):
        a,b,r=inputs();r[0]['stroke_type']='putt'
        with self.assertRaises(ValueError):select_gaps(a,b,r)
        out,added=select_gaps(a,copy.deepcopy(a),r)
        self.assertEqual(out['clips'],a['clips']);self.assertEqual(added,[])

    def test_cross_source_and_duplicate_predictions_refused(self):
        a,b,r=inputs();b['clips'][0]['sha256']='other'
        with self.assertRaises(ValueError):select_gaps(a,b,r)
        a,b,r=inputs();b['clips'][0]['targets']['clubhead']['points'].append(b['clips'][0]['targets']['clubhead']['points'][0])
        with self.assertRaises(ValueError):select_gaps(a,b,r)
