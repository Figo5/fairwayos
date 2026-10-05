"""Policy behavior, separate from the synthetic pixel validator tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.tiger_ball_evidence import ProposalGate, filter_clip, METHOD_REACQUIRED


def proposal(frame, x=20, y=20):
    return {"source_frame": frame, "x": x, "y": y, "origin": "automatic",
            "status": "observed_template", "method": "template", "score": .9}


class BallProposalPolicyTests(unittest.TestCase):
    def test_refusal_hides_coordinates_and_recovery_never_backfills(self):
        gate = ProposalGate(True, 10)
        self.assertEqual(gate.step(proposal(1), True)["origin"], "automatic")
        rejected = gate.step(proposal(2), False)
        confirming = gate.step(proposal(3), True)
        recovered = gate.step(proposal(4), True)
        self.assertIsNone(rejected["x"])
        self.assertIsNone(confirming["x"])
        self.assertEqual(recovered["method"], METHOD_REACQUIRED)
        self.assertIsNone(recovered["score"])
        self.assertIsNone(confirming["x"])

    def test_confirmation_requires_consecutive_compatible_measured_proposals(self):
        gate = ProposalGate(True, 5)
        gate.step(proposal(1), False)
        self.assertIsNone(gate.step(proposal(2), True)["x"])
        self.assertIsNone(gate.step(proposal(4), True)["x"])
        self.assertIsNone(gate.step(proposal(5, x=40), True)["x"])
        self.assertEqual(gate.step(proposal(6, x=41), True)["x"], 41)

    def test_invalid_seed_and_cut_are_terminal(self):
        for gate in (ProposalGate(False, 10), ProposalGate(True, 10)):
            for f in range(1, 8):
                self.assertIsNone(gate.step(proposal(f), True, cut=f == 1)["x"])

    def test_missing_proposals_reset_confirmation(self):
        gate = ProposalGate(True, 10)
        gate.step(proposal(1), False)
        gate.step(proposal(2), True)
        missing = {**proposal(3), "x": None, "y": None, "origin": "derived", "status": "lost"}
        self.assertIsNone(gate.step(missing, False)["x"])
        self.assertIsNone(gate.step(proposal(4), True)["x"])
        self.assertEqual(gate.step(proposal(5), True)["method"], METHOD_REACQUIRED)

    def test_prefix_invariance_and_no_argument_mutation(self):
        def run(flags):
            gate = ProposalGate(True, 10)
            return [gate.step(proposal(f), flag) for f, flag in enumerate(flags, 1)]
        flags = [True, False, True, True, False, True, True]
        self.assertEqual(run(flags)[:4], run(flags[:4]))
        p = proposal(1)
        original = copy.deepcopy(p)
        ProposalGate(True, 10).step(p, False)
        self.assertEqual(p, original)

    def test_unsorted_proposals_refused(self):
        gate = ProposalGate(True, 10)
        gate.step(proposal(3), True)
        with self.assertRaises(ValueError):
            gate.step(proposal(3), True)

    def test_first_proposal_gap_requires_two_confirmations(self):
        gate = ProposalGate(True, 10, seed_frame=1)
        self.assertIsNone(gate.step(proposal(3), True)["x"])
        self.assertEqual(gate.step(proposal(4), True)["method"], METHOD_REACQUIRED)

    def test_sequential_pixels_seed_only_and_cut_without_proposal(self):
        import numpy as np
        images = [np.full((50, 50, 3), f, dtype=np.uint8) for f in range(7)]
        class Capture:
            def __init__(self): self.index = 0
            def isOpened(self): return True
            def read(self):
                image = images[self.index]
                self.index += 1
                return True, image
            def release(self): pass
        seed = {"frame": 1, "x": 20, "y": 20}
        anchor = {**proposal(1), "origin": "operator", "status": "manual_anchor", "score": None}
        block = {"seed": seed, "points": [anchor, proposal(2), proposal(4), proposal(5)]}
        original = copy.deepcopy(block)
        build_calls, checks = [], []
        def build(image, x, y):
            build_calls.append((int(image[0, 0, 0]), x, y))
            return {"valid": True}
        def check(image, x, y, desc):
            checks.append(int(image[0, 0, 0]))
            return {"accepted": True}
        source = {"path": "synthetic", "width": 50, "height": 50,
                  "window": {"start_frame": 0, "end_frame": 6}, "cut_frames": [3]}
        with patch("cv2.VideoCapture", return_value=Capture()):
            result, audit = filter_clip(source, block, build_seed=build, check_evidence=check)
        self.assertEqual(build_calls, [(1, 20, 20)])
        self.assertEqual(checks, [2])
        self.assertEqual(result["points"][0], anchor)
        self.assertEqual(result["points"][1]["origin"], "automatic")
        self.assertIsNone(result["points"][2]["x"])
        self.assertIsNone(result["points"][3]["x"])
        self.assertEqual(block, original)
        self.assertTrue(audit["seed_valid"])

    def test_assisted_proposals_are_refused(self):
        source = {"window": {"start_frame": 0, "end_frame": 9}, "cut_frames": []}
        block = {"seed": {"frame": 1, "x": 20, "y": 20},
                 "points": [{**proposal(3), "origin": "operator", "status": "manual_anchor"}]}
        with self.assertRaises(ValueError):
            filter_clip(source, block, build_seed=None, check_evidence=None)
        block["points"] = [{**proposal(3), "origin": "derived", "status": "interpolated"}]
        with self.assertRaises(ValueError):
            filter_clip(source, block, build_seed=None, check_evidence=None)

    def test_invalid_source_contract_refused_before_pixels(self):
        source = {"window": {"start_frame": 0, "end_frame": 9}, "cut_frames": []}
        base = {"seed": {"frame": 1, "x": 20, "y": 20}, "points": []}
        invalid = [
            (source, {**base, "seed": {**base["seed"], "frame": 10}}),
            ({**source, "cut_frames": [1]}, base),
            (source, {**base, "points": [proposal(0)]}),
            (source, {**base, "points": [proposal(10)]}),
            (source, {**base, "points": [proposal(2), proposal(2)]}),
            ({**source, "cut_frames": [2]}, {**base, "points": [proposal(2)]}),
        ]
        for src, block in invalid:
            with self.subTest(src=src, block=block), self.assertRaises(ValueError):
                filter_clip(src, block, build_seed=None, check_evidence=None)

    def test_new_methods_pass_existing_native_prediction_contract(self):
        from ghostcaddie.demo import target_benchmark as bench
        point = proposal(2)
        for method in ("ball_evidence_template", METHOD_REACQUIRED):
            q = {**point, "method": method, "score": None}
            normalized = bench._prediction_point(q, "point", {"start_frame": 0, "end_frame": 9},
                                                 [], 1, 50, 50)
            self.assertEqual(normalized["method"], method)
            self.assertEqual(normalized["origin"], "automatic")


class BallEvidenceRunnerTests(unittest.TestCase):
    def setUp(self):
        from ghostcaddie.demo import target_benchmark as bench
        from tests.test_tiger_target_benchmark import clip_payload, dataset_payload, seed, manual, automatic
        from tools import tiger_ball_evidence as runner
        self.bench, self.runner = bench, runner
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'synthetic.mp4'
        self.source.write_bytes(b'Source bytes for hash binding; measurement stubbed, never decoded')
        self.reference = self.root / 'reference.json'
        self.baseline = self.root / 'baseline.predictions.json'
        self.baseline_report = self.root / 'baseline.report.json'
        self.out = self.root / 'candidate.predictions.json'
        clip = clip_payload(path=str(self.source), sha256=bench.sha256_file(self.source),
                            frame_count=6, window={'start_frame': 0, 'end_frame': 5},
                            targets={name: {'seed': seed(0), 'options': {}, 'labels': []}
                                     for name in ('ball', 'clubhead')})
        self.reference.write_text(bench.canonical_json(dataset_payload([clip])))
        self.payload = {'schema_version': bench.PREDICTIONS_SCHEMA, 'variant': 'baseline',
                        'clips': [{'id': clip['id'], 'sha256': clip['sha256'],
                                   'targets': {name: {'seed': {'frame': 0, 'x': 50.0, 'y': 40.0},
                                                      'points': [manual(0), automatic(1)]}
                                               for name in ('ball', 'clubhead')}}]}
        self.baseline.write_text(bench.canonical_json(self.payload))
        self.baseline_report.write_text(json.dumps({'predictions': {'sha256': bench.sha256_file(self.baseline)},
                                                   'runtime': {'fixture': True}}))
        self.argv = ['runner', '--dataset', str(self.reference), '--baseline', str(self.baseline),
                     '--baseline-report', str(self.baseline_report), '--out', str(self.out)]

    def run_main(self):
        with patch('sys.argv', self.argv), patch.object(
                self.runner, 'filter_clip', side_effect=lambda source, block, **kwargs: (copy.deepcopy(block), {})):
            self.runner.main()

    def test_success_pins_bytes_and_preserves_complete_head_block(self):
        self.run_main()
        report = json.loads(Path(str(self.out) + '.report.json').read_text())
        self.assertEqual(report['predictions']['sha256'], self.bench.sha256_file(self.out))
        self.assertEqual(report['runtime']['benchmark_sha256'], self.bench.sha256_file(self.bench.__file__))
        self.assertEqual(report['runtime']['baseline_report_sha256'], self.bench.sha256_file(self.baseline_report))
        self.assertFalse(report['runtime']['target_coordinate_or_visibility_labels_read_by_policy'])
        new = json.loads(self.out.read_text())
        self.assertEqual(self.bench.canonical_json(new['clips'][0]['targets']['clubhead']),
                         self.bench.canonical_json(self.payload['clips'][0]['targets']['clubhead']))
        with self.assertRaisesRegex(ValueError, 'fresh outputs'):
            self.run_main()

    def test_mismatched_baseline_report_writes_nothing(self):
        self.baseline_report.write_text(json.dumps({'predictions': {'sha256': 'f' * 64}, 'runtime': {}}))
        with self.assertRaisesRegex(ValueError, 'Baseline report'):
            self.run_main()
        self.assertFalse(self.out.exists())
        self.assertFalse((self.root / 'candidate.raw.json').exists())

    def test_evaluator_failure_leaves_no_partial_prediction(self):
        with patch.object(self.bench, 'evaluate', side_effect=ValueError('evaluation refused')):
            with self.assertRaisesRegex(ValueError, 'evaluation refused'):
                self.run_main()
        self.assertFalse(self.out.exists())
        self.assertFalse((self.root / 'candidate.raw.json').exists())

    def test_preexisting_raw_file_is_preserved(self):
        raw = self.root / 'candidate.raw.json'
        raw.write_text('preserve me')
        with self.assertRaisesRegex(ValueError, 'fresh outputs'):
            self.run_main()
        self.assertEqual(raw.read_text(), 'preserve me')
        self.assertFalse(self.out.exists())

    def test_refuses_output_matching_source_before_measurement(self):
        source = self.root / 'source.predictions.json'
        self.source.rename(source)
        reference = json.loads(self.reference.read_text())
        reference['clips'][0]['path'] = str(source)
        self.reference.write_text(json.dumps(reference))
        self.argv[-1] = str(source)
        with self.assertRaisesRegex(ValueError, 'fresh outputs'):
            self.run_main()
        self.assertTrue(source.read_bytes().startswith(b'Source bytes'))


if __name__ == "__main__":
    unittest.main()
