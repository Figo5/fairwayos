import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ghostcaddie.demo.errors import DemoConfigError
from ghostcaddie.demo.tracking_review import load_tracking_review


class TrackingReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'results.json'
        self.record = SimpleNamespace(sha256='a' * 64, frame_count=10, width=100, height=80)
        def point(frame, status, origin, x=None, y=None, method=''):
            return {'source_frame': frame, 'status': status, 'origin': origin,
                    'x': x, 'y': y, 'score': None, 'method': method}
        self.payload = {'schema_version': 'fairwayos-target-predictions.v1', 'variant': 'research',
                        'clips': [{'id': 'source', 'sha256': self.record.sha256, 'targets': {
                            'clubhead': {'seed': {'frame': 2, 'x': 50, 'y': 60}, 'points': [
                                point(2, 'manual_anchor', 'operator', 50, 60, 'operator_placement'),
                                point(3, 'observed_template', 'automatic', 51, 60, 'sam2_head_region'),
                                point(4, 'missing', 'derived'),
                                point(6, 'interpolated', 'derived', 55, 60),
                                point(7, 'observed_template', 'automatic', 56, 60, 'sam2_head_region_reacquired'),
                                point(8, 'lost', 'derived')]}}}]}

    def load(self):
        self.path.write_text(json.dumps(self.payload))
        return load_tracking_review(self.path, [self.record])

    def test_gaps_include_absent_and_interpolated_frames_without_counting_seed(self):
        before = copy.deepcopy(self.payload)
        result = self.load()[self.record.sha256]['clubhead']
        self.assertEqual(result['gaps'], [{'start_frame': 4, 'end_frame': 6}, {'start_frame': 8, 'end_frame': 8}])
        self.assertEqual((result['start_frame'], result['end_frame'], result['automatic_frames']), (3, 8, 2))
        self.assertFalse(result['accuracy_claim'])
        self.assertEqual(self.payload, before)
        self.assertNotIn('seed', result)
        self.assertNotIn('points', result)

    def test_results_for_another_source_are_refused(self):
        self.payload['clips'][0]['sha256'] = 'b' * 64
        with self.assertRaises(DemoConfigError): self.load()

    def test_duplicate_frame_and_hash_are_refused(self):
        self.payload['clips'][0]['targets']['clubhead']['points'].append(copy.deepcopy(
            self.payload['clips'][0]['targets']['clubhead']['points'][1]))
        with self.assertRaises(DemoConfigError): self.load()
        self.payload['clips'][0]['targets']['clubhead']['points'].pop()
        self.payload['clips'].append(copy.deepcopy(self.payload['clips'][0]))
        with self.assertRaises(DemoConfigError): self.load()

    def test_an_assisted_or_preseed_position_cannot_claim_automatic_tracking(self):
        p = self.payload['clips'][0]['targets']['clubhead']['points'][1]
        p['method'] = 'manual_path'
        with self.assertRaises(DemoConfigError): self.load()
        p['method'] = 'sam2_head_region'; p['source_frame'] = 1
        with self.assertRaises(DemoConfigError): self.load()

    def test_out_of_bounds_and_nonfinite_source_pixels_are_refused(self):
        p = self.payload['clips'][0]['targets']['clubhead']['points'][1]
        for field, bad in [('x', 101), ('y', float('nan')), ('source_frame', 10)]:
            original = p[field]; p[field] = bad
            with self.assertRaises(DemoConfigError): self.load()
            p[field] = original

    def test_unseeded_results_have_no_review_range(self):
        target = self.payload['clips'][0]['targets']['clubhead']
        target['seed'] = None; target['points'] = []
        result = self.load()[self.record.sha256]['clubhead']
        self.assertIsNone(result['start_frame']); self.assertEqual(result['gaps'], [])

    def test_corrupt_seed_and_target_are_refused(self):
        target = self.payload['clips'][0]['targets']['clubhead']
        target['seed']['frame'] = True
        with self.assertRaises(DemoConfigError): self.load()
        target['seed']['frame'] = 2
        self.payload['clips'][0]['targets']['shaft'] = target
        with self.assertRaises(DemoConfigError): self.load()
