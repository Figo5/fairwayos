import copy
import unittest
from tools.tiger_stroke_family_route import compose

class FamilyRouteTests(unittest.TestCase):
    def test_only_declared_family_selects_heads_and_ball_is_never_changed(self):
        base={'clips':[{'id':c,'targets':{'clubhead':{'points':['classic']},'ball':{'points':['ball']}}} for c in ['p','s']]}
        learned=copy.deepcopy(base)
        for c in learned['clips']:c['targets']['clubhead']['points']=['learned']
        original=copy.deepcopy(base)
        cohort=[{'id':'p','stroke_type':'putt'},{'id':'s','stroke_type':'full_swing'}]
        result,engines=compose(base,learned,cohort)
        self.assertEqual(result['clips'][0]['targets']['clubhead']['points'],['learned'])
        self.assertEqual(result['clips'][1]['targets']['clubhead']['points'],['classic'])
        self.assertEqual(base,original)
        self.assertEqual(result['clips'][0]['targets']['ball'],base['clips'][0]['targets']['ball'])
        self.assertEqual(engines['s'],'classic head baseline replay')
        # Scores or outcome-like metadata cannot change the family decision.
        learned['clips'][1]['score']=1;learned['clips'][0]['score']=0
        self.assertEqual(compose(base,learned,cohort)[0]['clips'][1]['targets']['clubhead']['points'],['classic'])
        learned['clips'][0]['targets']['clubhead']['points'].append('later edit')
        self.assertEqual(result['clips'][0]['targets']['clubhead']['points'],['learned'])

    def test_partial_cohort_and_unknown_family_are_refused(self):
        p={'clips':[{'id':'x','targets':{}}]}
        with self.assertRaises(ValueError):compose(p,p,[])
        with self.assertRaises(ValueError):compose(p,p,[{'id':'x','stroke_type':'unknown'}])

if __name__=='__main__':unittest.main()
