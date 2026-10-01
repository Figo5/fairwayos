import copy
import unittest
from tools.tiger_seed_fallback import compose


class SeedFallbackTests(unittest.TestCase):
    def setUp(self):
        self.base = {'clips':[{'id':cid,'targets':{
            'ball':{'points':['classic ball']},
            'clubhead':{'points':['classic head']}}} for cid in ['putt_a','putt_b','swing']]}
        self.candidate = copy.deepcopy(self.base)
        for c in self.candidate['clips']:
            c['targets']['clubhead'] = {'points':['measured head', 'lost', 'recovered']}

    def test_seed_only_route_preserves_full_blocks_and_inputs(self):
        before = copy.deepcopy(self.base)
        result, routing = compose(self.base, self.candidate,
            {'putt_a':True,'putt_b':False}, {'putt_a','putt_b'})
        self.assertEqual(self.base, before)
        self.assertEqual(result['clips'][0]['targets']['clubhead'],
                         self.candidate['clips'][0]['targets']['clubhead'])
        for i in [1,2]:
            self.assertEqual(result['clips'][i], self.base['clips'][i])
        for i in range(3):
            self.assertEqual(result['clips'][i]['targets']['ball'], self.base['clips'][i]['targets']['ball'])
        self.assertEqual(routing['putt_a']['head_engine'], 'sam2_head_region')
        self.assertEqual(routing['putt_b']['head_engine'], 'classic_replay')

    def test_undeclared_stroke_changed_ball_and_non_boolean_refuse(self):
        with self.assertRaises(ValueError):
            compose(self.base,self.candidate,{'swing':True},{'putt_a'})
        with self.assertRaises(ValueError):
            compose(self.base,self.candidate,{'putt_a':1},{'putt_a'})
        self.candidate['clips'][0]['targets']['ball'] = {'points':['different']}
        with self.assertRaises(ValueError):
            compose(self.base,self.candidate,{'putt_a':True},{'putt_a'})


if __name__ == '__main__':
    unittest.main()
