import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import tiger_swing_route as route
from tools.tiger_swing_route import compose


def point(frame, origin='automatic', method='template', x=40):
    return {'source_frame': frame, 'x': None if origin == 'derived' else x,
            'y': None if origin == 'derived' else 30, 'origin': origin,
            'status': 'missing' if origin == 'derived' else 'manual_anchor' if origin == 'operator' else 'observed_template',
            'method': method, 'score': .8 if method == 'template' and origin == 'automatic' else None}


def payload():
    return {'schema_version': 'fairwayos-target-predictions.v1', 'variant': 'fixture',
            'clips': [{'id': cid, 'sha256': str(i)*64, 'targets': {
                name: {'seed': {'frame': 0, 'x': 40, 'y': 30}, 'points': [point(0, 'operator', 'operator_placement'), point(1), point(2, 'derived', '')]}
                for name in ['ball', 'clubhead']}} for i, cid in enumerate(['swing', 'putt'], 1)]}


class SwingRouteTests(unittest.TestCase):
    def setUp(self):
        self.old = payload()
        self.new = copy.deepcopy(self.old)
        for clip in self.new['clips']:
            clip['targets']['clubhead']['points'] = [point(0, 'operator', 'operator_placement'),
                                                     point(1, method='sam2_swing_core', x=45),
                                                     point(2, method='sam2_swing_core_reacquired', x=46)]
        self.cohort = [{'id': 'swing', 'stroke_type': 'full_swing'}, {'id': 'putt', 'stroke_type': 'putt'}]

    def test_availability_rule_preserves_classic_and_fills_only_current_observation(self):
        out, trace = compose(self.old, self.new, self.cohort)
        points = out['clips'][0]['targets']['clubhead']['points']
        self.assertEqual(points[0], self.old['clips'][0]['targets']['clubhead']['points'][0])
        self.assertEqual(points[1], self.old['clips'][0]['targets']['clubhead']['points'][1])
        self.assertEqual(points[2], self.new['clips'][0]['targets']['clubhead']['points'][2])
        self.assertEqual(trace['swing'][2]['engine'], 'swing_core')

    def test_putts_balls_and_inputs_unchanged(self):
        old, new = copy.deepcopy(self.old), copy.deepcopy(self.new)
        out, trace = compose(self.old, self.new, self.cohort)
        self.assertEqual(out['clips'][1], self.old['clips'][1])
        self.assertNotIn('putt', trace)
        for selected, original in zip(out['clips'], self.old['clips']):
            self.assertEqual(selected['targets']['ball'], original['targets']['ball'])
        self.assertEqual(self.old, old)
        self.assertEqual(self.new, new)

    def test_reference_outcomes_never_select_engine(self):
        out, trace = compose(self.old, self.new, self.cohort)
        cohort = [{**c, 'labels': [{'frame': 2, 'state': 'visible', 'x': -9999}]} for c in self.cohort]
        other, other_trace = compose(self.old, self.new, cohort)
        self.assertEqual((out, trace), (other, other_trace))

    def test_future_points_do_not_backfill_or_change_prefix(self):
        out, _ = compose(self.old, self.new, self.cohort)
        self.new['clips'][0]['targets']['clubhead']['points'].append(point(3, method='sam2_swing_core'))
        later, _ = compose(self.old, self.new, self.cohort)
        self.assertEqual(later['clips'][0]['targets']['clubhead']['points'][:3], out['clips'][0]['targets']['clubhead']['points'])

    def test_unknown_method_or_learned_confidence_refused(self):
        for change in [{'method': 'template'}, {'score': .99}]:
            new = copy.deepcopy(self.new)
            new['clips'][0]['targets']['clubhead']['points'][2].update(change)
            with self.assertRaises(ValueError):
                compose(self.old, new, self.cohort)

    def test_interpolation_and_later_operator_anchors_refused(self):
        for bad in [point(2, 'operator', 'operator_placement'), {**point(2), 'origin': 'derived', 'status': 'interpolated'}]:
            new = copy.deepcopy(self.new)
            new['clips'][0]['targets']['clubhead']['points'][2] = bad
            with self.assertRaises(ValueError):
                compose(self.old, new, self.cohort)

    def test_source_seed_and_collection_mismatch_refused(self):
        for field in ('sha256', 'seed', 'duplicate'):
            new = copy.deepcopy(self.new)
            if field == 'sha256': new['clips'][0]['sha256'] = 'f'*64
            if field == 'seed': new['clips'][0]['targets']['clubhead']['seed']['x'] = 44
            if field == 'duplicate': new['clips'].append(new['clips'][0])
            with self.assertRaises(ValueError):
                compose(self.old, new, self.cohort)


class SwingRouteExecutionTests(unittest.TestCase):
    setUp = SwingRouteTests.setUp

    def execute(self, directory, *, tamper=None, evaluation_failure=False):
        from ghostcaddie.demo import swing_context, swing_head
        root = Path(directory)
        paths = {name: root / (name + '.json') for name in
                 ['dataset', 'baseline', 'baseline-report', 'measured', 'measured-report']}
        paths['dataset'].write_text('{}')
        paths['baseline'].write_text(json.dumps(self.old))
        self.new['variant'] = 'full_swing_mask_core_research'
        paths['measured'].write_text(json.dumps(self.new))
        raw_path = root / 'measurement.raw.json'
        raw_path.write_text('{}')
        for stem in ['baseline', 'measured']:
            sha = route.bench.sha256_file(paths[stem])
            runtime = {'profile': swing_context.PROFILE, 'raw_path': str(raw_path), 'raw_sha256': route.bench.sha256_file(raw_path),
                       'head_region_sha256': route.bench.sha256_file(Path(swing_context.__file__)),
                       'swing_measurement_base_sha256': route.bench.sha256_file(Path(swing_head.__file__)),
                       'baseline_predictions_sha256': route.bench.sha256_file(paths['baseline'])}
            paths[stem + '-report'].write_text(json.dumps({'predictions': {'sha256': 'f'*64 if tamper == stem else sha, 'variant': self.new['variant']}, 'reference': {'sha256': 'd'*64}, 'runtime': runtime}))
        output = root / 'selected.predictions.json'
        dataset = {'clips': [{**c, 'path': str(root / (c['id'] + '.mp4'))} for c in self.cohort]}
        argv = ['route'] + [item for key, path in paths.items() for item in ['--' + key, str(path)]] + ['--out', str(output)]
        with patch('sys.argv', argv), patch.object(route.bench, 'load_dataset', return_value=(dataset, 'd'*64)), \
                patch.object(route.bench, 'validate_predictions', side_effect=lambda p, d: p), \
                patch.object(route.bench, 'evaluate', side_effect=ValueError('evaluation failed') if evaluation_failure else lambda *a, **k: k), \
                patch.object(route.bench, 'render_markdown', return_value='report'):
            route.main()
        return output

    def test_report_tampering_refused_before_any_output(self):
        for stem in ['baseline', 'measured']:
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, 'bind'):
                    self.execute(directory, tamper=stem)
                self.assertFalse(list(Path(directory).glob('selected*')))

    def test_failed_evaluation_publishes_no_candidate_files(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'evaluation failed'):
                self.execute(directory, evaluation_failure=True)
            self.assertFalse(list(Path(directory).glob('selected*')))

    def test_executed_report_pins_written_prediction_and_selection_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.execute(directory)
            report = json.loads(Path(str(output) + '.report.json').read_text())
            self.assertEqual(report['predictions_sha256'], route.bench.sha256_file(output))
            self.assertEqual(report['runtime']['selection_sha256'], route.bench.sha256_file(Path(str(output) + '.selection.json')))
            self.assertIn('baseline_executed_runtime', report['runtime'])

    def test_wrong_dataset_variant_profile_or_raw_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.execute(directory)
            root = Path(directory)
            measured = json.loads((root / 'measured.json').read_text())
            report = json.loads((root / 'measured-report.json').read_text())
            changes = [('reference', 'sha256', 'b'*64), ('predictions', 'variant', 'wrong'),
                       ('runtime', 'profile', {}), ('runtime', 'raw_sha256', '0'*64),
                       ('runtime', 'baseline_predictions_sha256', '0'*64), ('runtime', 'head_region_sha256', '0'*64)]
            for section, key, value in changes:
                modified = copy.deepcopy(report)
                modified[section][key] = value
                with self.assertRaises(ValueError):
                    route.validate_measured_record(modified, measured, root / 'measured.json', 'd'*64, route.bench.sha256_file(root/'baseline.json'))


if __name__ == '__main__':
    unittest.main()
