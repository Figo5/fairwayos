"""Small-ball searches must retain actual pixels, rivals and temporal gaps."""
import copy
import unittest

import cv2
import numpy as np

from ghostcaddie.video.automatic_shot import white_ball_components, green_fraction
from ghostcaddie.video.shot_ball_detail import (
    near_player_components, aligned_neural_score, confirmed_detail_chains)


def body():
    points = {'left_hip': [105., 130.], 'right_hip': [125., 130.],
              'left_ankle': [102., 210.], 'right_ankle': [129., 210.],
              'right_wrist': [135., 155.]}
    return {'bbox': [80., 20., 150., 220.], 'provenance': 'current_pixels',
            'track_id': 1, 'keypoints': {k: {'point': p, 'confidence': .95}
                                       for k, p in points.items()}}


def green():
    return np.full((260, 300, 3), (60, 125, 70), np.uint8)


def candidates(frame, person=None, **kwargs):
    return near_player_components(frame, body() if person is None else person,
                                  white_ball_components, green_fraction, **kwargs)


def chain(length=20, anchors=(0, 6, 12)):
    return [{'frame': n, 'body_scene': 0, 'body': body(),
             'detail_ball_candidates': [{'point': [190., 211.], 'hip': [115., 130.],
                'size': [9, 9], 'area': 49, 'body_id': 1, 'body_height': 200.,
                'neural_score': .95 if n in anchors else None}]}
            for n in range(length)]


class DetailEvidenceTests(unittest.TestCase):
    def test_real_native_round_component_is_only_a_search_candidate(self):
        frame = green();cv2.circle(frame, (190, 211), 4, (245, 245, 245), -1)
        found = candidates(frame)
        self.assertEqual(len(found), 1)
        np.testing.assert_allclose(found[0]['point'], [190, 211], atol=.1)
        self.assertNotIn('neural_score', found[0])

    def test_missing_or_historical_pose_and_weak_wrist_do_not_search(self):
        frame = green();cv2.circle(frame, (190, 211), 4, (245, 245, 245), -1)
        for change in ('provenance', 'wrist', 'ankle'):
            b = body()
            if change == 'provenance':b['provenance'] = 'previous_frame'
            elif change == 'wrist':b['keypoints']['right_wrist']['confidence'] = .5
            else:del b['keypoints']['left_ankle']
            self.assertEqual(candidates(frame, b), [])
        self.assertEqual(near_player_components(frame, None, white_ball_components, green_fraction), [])

    def test_shoe_bright_pixel_flat_cup_glyph_and_current_head_are_withheld(self):
        for kind in ('shoe', 'flat', 'glyph', 'head'):
            frame = green()
            if kind == 'shoe':cv2.circle(frame, (129, 210), 4, (245, 245, 245), -1)
            elif kind == 'flat':cv2.ellipse(frame, (190, 211), (10, 2), 0, 0, 360, (245, 245, 245), -1)
            elif kind == 'glyph':cv2.rectangle(frame, (180, 209), (205, 212), (245, 245, 245), -1)
            else:cv2.circle(frame, (190, 211), 4, (245, 245, 245), -1)
            self.assertEqual(candidates(frame, head={'point': [190., 211.]} if kind == 'head' else None), [])

    def test_native_coordinates_are_mapped_from_scaled_current_pose(self):
        frame = green();cv2.circle(frame, (190, 211), 4, (245, 245, 245), -1)
        b = body();b['bbox'] = [v * .5 for v in b['bbox']]
        for j in b['keypoints'].values():j['point'] = [v * .5 for v in j['point']]
        np.testing.assert_allclose(candidates(frame, b, scale=.5)[0]['point'], [190, 211], atol=.1)

    def test_neural_evidence_needs_high_score_tight_position_and_size(self):
        component = {'point': [190., 211.], 'size': [9, 9]}
        box = {'class_id': 32, 'score': .95, 'box': [185., 206., 195., 216.]}
        self.assertEqual(aligned_neural_score(component, [box]), .95)
        for change in ('weak', 'offcenter', 'oversize', 'wrongclass'):
            b = copy.deepcopy(box)
            if change == 'weak':b['score'] = .84
            elif change == 'offcenter':b['box'] = [195., 206., 205., 216.]
            elif change == 'oversize':b['box'] = [170., 191., 210., 231.]
            else:b['class_id'] = 0
            self.assertIsNone(aligned_neural_score(component, [b]))

    def test_three_checks_publish_current_samples_only_inside_confirmed_window(self):
        rows = chain();before = copy.deepcopy(rows)
        observed = confirmed_detail_chains(rows)
        self.assertEqual([r['frame'] for r, _ in observed], list(range(13)))
        self.assertEqual(rows, before)
        self.assertEqual(confirmed_detail_chains(chain(anchors=(0, 6))), [])
        self.assertEqual(confirmed_detail_chains(chain(25, anchors=(0, 12, 24))), [])

    def test_rivals_missing_components_identity_and_cut_split_confirmation(self):
        for change in ('rival', 'missing', 'identity', 'cut'):
            rows = chain()
            if change == 'rival':rows[8]['detail_ball_candidates'] *= 2
            elif change == 'missing':rows[8]['detail_ball_candidates'] = []
            elif change == 'identity':
                for r in rows[8:]:r['body']['track_id'] = 2
            else:
                for r in rows[8:]:r['body_scene'] = 1
            self.assertEqual(confirmed_detail_chains(rows), [])

    def test_camera_translation_is_search_only_and_large_jump_is_not_filled(self):
        rows = chain()
        for r in rows:
            c = r['detail_ball_candidates'][0];c['point'][0] += r['frame'] * 5;c['hip'][0] += r['frame'] * 5
            r['previous_to_current'] = [[1, 0, 5], [0, 1, 0]] if r['frame'] else None
        observed = confirmed_detail_chains(rows)
        self.assertEqual(observed[-1][1]['point'][0], 250.)
        rows[8]['detail_ball_candidates'][0]['point'][0] += 30
        self.assertEqual(confirmed_detail_chains(rows), [])

    def test_explicit_neural_miss_splits_support_and_unknown_identity_withholds(self):
        rows = chain(20, anchors=(0, 6, 18))
        rows[12]['detail_ball_candidates'][0]['neural_checked'] = True
        self.assertEqual(confirmed_detail_chains(rows), [])
        rows = chain()
        rows[8]['detail_ball_candidates'][0]['body_id'] = 2
        self.assertEqual(confirmed_detail_chains(rows), [])

    def test_current_shoe_connection_and_hand_glint_are_not_detached_balls(self):
        frame = green()
        cv2.line(frame, (129, 210), (155, 210), (15, 15, 15), 12)
        cv2.circle(frame, (155, 210), 4, (245, 245, 245), -1)
        self.assertEqual(candidates(frame), [])
        frame = green();cv2.circle(frame, (190, 211), 4, (245, 245, 245), -1)
        b = body();b['keypoints']['right_wrist']['point'] = [190., 211.]
        self.assertEqual(candidates(frame, b), [])

    def test_body_scale_jump_splits_neural_lineage(self):
        rows = chain()
        rows[8]['detail_ball_candidates'][0]['body_height'] *= 1.2
        self.assertEqual(confirmed_detail_chains(rows), [])

    def test_slow_object_handoff_cannot_accumulate_drift_and_static_ball_ignores_hips(self):
        rows = chain()
        for r in rows:r['detail_ball_candidates'][0]['point'][0] += r['frame']
        self.assertEqual(confirmed_detail_chains(rows), [])
        rows = chain()
        for r in rows:r['detail_ball_candidates'][0]['hip'][0] += r['frame'] * 5
        self.assertEqual(len(confirmed_detail_chains(rows)), 13)
