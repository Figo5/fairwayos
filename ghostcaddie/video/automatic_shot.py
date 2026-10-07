"""Upload-only shot intelligence: neural ball proposals and scene tracking.

No seed files, shot frame numbers, selected coordinates or reference clips are
inputs. YOLO detects sports-ball candidates on green views; temporal evidence
and image components establish a track. A visible cup/flagstick can supply an
observed pin target, never a claimed optimal aim. Unsupported views stay blank.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from .green_scene_tracker import affine, candidates, track_ground_anchor
from .source_binding import bind_source, sha256_file
from .temporal_objects import sky_candidates, cup_candidates, unique_chains, cup_publication_runs
from .shot_ball_detail import NativeBallSearch, confirmed_detail_chains

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def green_fraction(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([24, 20, 35]), np.array([95, 230, 230]))
    return float(np.mean(mask > 0))


def detect_cup(frame):
    """Find the pale cup beside the foot of a yellow flagstick.

    A flag alone or a shaft tip is insufficient. This first detector supports
    yellow flagsticks, and leaves removed/other-colour poles unresolved.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, np.array([15, 125, 70]), np.array([45, 255, 255]))
    white = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([179, 65, 255]))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(yellow)
    choices = []
    height, width = frame.shape[:2]
    for index, stat in enumerate(stats[1:], 1):
        _, _, w, h, area = map(int, stat)
        if h < max(45, height * .15) or h / w < 8 or area < 25:
            continue
        ys, xs = np.nonzero(labels == index)
        bottom = int(ys.max())
        base = xs[ys >= bottom - 4]
        x = float(np.median(base))
        pole_width = max(2., float(np.ptp(base) + 1))
        if pole_width > width * .035:
            continue
        radius = max(18, int(pole_width * 5))
        low, high = max(0, int(bottom - 3 * pole_width)), min(height, int(bottom + max(40, 3 * pole_width)))
        left, right = max(0, int(x - radius)), min(width, int(x + radius))
        roi = white[low:high, left:right]
        _, parts, shapes, centers = cv2.connectedComponentsWithStats(roi)
        pixels = []
        for j, shape in enumerate(shapes[1:], 1):
            _, _, cw, ch, count = map(int, shape)
            cx, cy = centers[j - 1]
            if (count >= 3 and cw / ch >= 1.25 and abs(left + cx - x) < radius
                    and abs(low + cy - bottom) < max(38, pole_width * 3)):
                yy, xx = np.nonzero(parts == j)
                pixels.extend(zip(xx + left, yy + low))
        if not pixels:
            continue
        y = float(np.median(np.array(pixels)[:, 1]))
        choices.append({'point': [x, y], 'shaft_height': h, 'cup_white_pixels': len(pixels),
                        'score': h + len(pixels) * .1, 'method': 'yellow_shaft_and_visible_cup'})
    return max(choices, key=lambda row: row['score']) if choices else None


def supported_ball_proposals(frame, boxes):
    """Require a neural ball proposal, a green surround and no person overlap."""
    height, width = frame.shape[:2]
    people = [b['box'] for b in boxes if b['class_id'] == 0 and b['score'] >= .35]
    accepted = []
    for b in boxes:
        if b['class_id'] != 32 or b['score'] < .55:
            continue
        x0, y0, x1, y1 = b['box']
        x, y = (x0 + x1) / 2, (y0 + y1) / 2
        diameter = max(x1 - x0, y1 - y0)
        if diameter < 5 or diameter > min(width, height) * .16:
            continue
        if any(min(x1, px1) > max(x0, px0) and min(y1, py1) > max(y0, py0)
               for px0, py0, px1, py1 in people):
            continue
        pad = max(15, int(diameter * 1.6))
        patch = frame[max(0, int(y) - pad):min(height, int(y) + pad),
                      max(0, int(x) - pad):min(width, int(x) + pad)]
        if patch.size and green_fraction(patch) >= .55:
            accepted.append(dict(b, point=[x, y]))
    return accepted



def white_ball_components(hsv):
    # Bright low-chroma foreground, rather than the desaturated green itself.
    mask=cv2.inRange(hsv,np.array([0,0,170]),np.array([179,80,255]))
    gray=cv2.cvtColor(cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR),cv2.COLOR_BGR2GRAY)
    background=cv2.morphologyEx(gray,cv2.MORPH_OPEN,np.ones((101,101),np.uint8))
    mask[cv2.subtract(gray,background)<25]=0
    _,_,stats,centers=cv2.connectedComponentsWithStats(mask)
    rows=[]
    for stat,center in zip(stats[1:],centers[1:]):
        x,y,w,h,area=map(int,stat)
        if 3<=area<=4000 and max(w,h)<100 and .6<w/h<1.8 and area/(w*h)>.4:
            rows.append({'point':center.tolist(),'size':[w,h],'area':area})
    return rows

def ball_component(frame, predicted, previous_size, target=None, *, proposal=False):
    """Associate current white pixels. A prediction is only a search hint."""
    if (predicted is None or isinstance(previous_size, (bool, type(None)))
            or not np.isfinite(previous_size) or previous_size <= 0):
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    eligible = []
    radius = max(7, previous_size * .4) if proposal else max(25, previous_size * 2)
    for row in white_ball_components(hsv):
        point = np.array(row['point'])
        dist = float(np.linalg.norm(point - predicted))
        size = max(row['size'])
        if (dist <= radius and .35 < size / previous_size < 2.2
                and (target is None or np.linalg.norm(point - target) > max(12, size))):
            cost = dist / radius + .25 * abs(np.log(size / previous_size))
            eligible.append((cost, row))
    eligible.sort(key=lambda item: item[0])
    if not eligible:
        return None
    if len(eligible) > 1 and eligible[1][0] - eligible[0][0] < .25:
        return None
    return eligible[0][1]


class NeuralBallDetector:
    def __init__(self, model_path):
        from ultralytics import YOLO
        if not Path(model_path).is_file():
            raise RuntimeError('Local ball detector weights are unavailable')
        self.model_path = Path(model_path)
        self.model = YOLO(str(model_path))

    def __call__(self, frame):
        result = self.model.predict(frame, imgsz=1280, classes=[0, 32], conf=.25,
                                    device='cpu', verbose=False)[0]
        return [{'class_id': int(b.cls.item()), 'score': float(b.conf.item()),
                 'box': b.xyxy[0].tolist()} for b in result.boxes]


def probe_source(source, ffprobe='ffprobe'):
    doc = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-select_streams', 'v:0',
        '-show_streams', '-show_frames', '-show_entries',
        'stream=width,height,avg_frame_rate,r_frame_rate,time_base,sample_aspect_ratio:frame=best_effort_timestamp,best_effort_timestamp_time,pkt_duration,pkt_duration_time',
        '-of', 'json', str(source)], timeout=90))
    stream = doc['streams'][0]
    pts = [float(f['best_effort_timestamp_time']) for f in doc['frames']]
    if len(pts) < 2 or any(b <= a for a, b in zip(pts, pts[1:])):
        raise RuntimeError('Video presentation timestamps are missing or unordered')
    if len(pts) > 6000 or pts[-1] - pts[0] > 180:
        raise RuntimeError('Automatic shot clips must be at most 180 seconds and 6000 frames')
    if stream.get('sample_aspect_ratio') not in (None, 'N/A', '1:1'):
        raise RuntimeError('Non-square source pixels are not supported in this automatic prototype')
    return dict(stream, pts=pts, frame_pts=[int(f['best_effort_timestamp']) for f in doc['frames']],
                frame_durations=[int(f.get('pkt_duration') or 0) for f in doc['frames']], frames=len(pts))


def decoded_frames(source, width, height, ffmpeg='ffmpeg'):
    cmd = [ffmpeg, '-v', 'error', '-threads', '2', '-i', str(source), '-map', '0:v:0',
           '-an', '-fps_mode', 'passthrough', '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    size = width * height * 3
    try:
        n = 0
        while True:
            data = proc.stdout.read(size)
            if not data:
                break
            if len(data) != size:
                raise RuntimeError('Truncated decoded video frame')
            yield n, np.frombuffer(data, np.uint8).reshape(height, width, 3).copy()
            n += 1
        if proc.wait(timeout=20):
            raise RuntimeError('Source decoder failed: ' + proc.stderr.read().decode()[-300:])
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)
        proc.stderr.close()



def track_flag_cup(segment, rows, initializations, read, indices, scene):
    # A cup needs repeated scene evidence, independently of the ball.
    cups = [r for r in segment if r.get('cup_proposal')]
    confirmed = [r for r in cups if any(0 < abs(other['frame'] - r['frame']) <= 2
                 and np.linalg.norm(np.array(other['cup_proposal']['point']) - r['cup_proposal']['point']) < 30
                 for other in cups)]
    if confirmed:
        cup_seed = max(confirmed, key=lambda r: r['cup_proposal']['score'])
        cn, point = cup_seed['frame'], cup_seed['cup_proposal']['point']
        cup_seed['target'] = point
        initializations.append({'object': 'cup', 'scene': scene, 'frame': cn,
            'point': point, 'method': 'automatic_visible_cup_and_flagstick'})
        for direction, ordered in [(-1, [n for n in reversed(indices) if n < cn]),
                                   (1, [n for n in indices if n > cn])]:
            prev_gray = cv2.cvtColor(read(cn), cv2.COLOR_BGR2GRAY)
            tracked = point
            for n in ordered:
                gray = cv2.cvtColor(read(n), cv2.COLOR_BGR2GRAY)
                tracked, quality = track_ground_anchor(prev_gray, gray, tracked)
                rows[n]['target'], rows[n]['target_quality'] = tracked, quality
                prev_gray = gray
                if tracked is None:
                    break
        # A one-frame pole/cuff proposal is not a stable cup target.
        # Repeated detector proposals alone do not establish a track.
        if sum(r['target'] is not None for r in segment) < 5:
            for row in segment:
                if row.get('target_method') != 'current_visible_flat_cup_rim':
                    row['target'] = None
            initializations.pop()

def head_ball_overlap(club, ball, *, scale=1.):
    separation=float(np.linalg.norm(np.array(club['point'])-ball['point']))
    limit=max(3,max(ball['size'])*.5+1)
    if club.get('search_resolution')=='native_source_pixels':
        limit=max(limit,4*scale,.75*max(ball['size']))
    return separation<=limit


def observe_video(source, spec, cache, detector, *, ffmpeg='ffmpeg', sample_step=6, people_detector=None,
                  semantic_client=None, source_sha256=None):
    """Analyze a clip without operator seeds. Cache is temporary and lossless."""
    if people_detector is not None:
        people_detector.identity.reset(new_clip=True)
    width, height = int(spec['width']), int(spec['height'])
    scale = min(1., 1280 / width, 720 / height)
    analysis_size = (round(width * scale), round(height * scale))
    rows = []
    semantic_images = {}
    detail_search = (NativeBallSearch(detector.model, white_ball_components, green_fraction)
                     if people_detector is not None and hasattr(detector, 'model') else None)
    scene = -1
    was_green = False
    was_sky = False
    people = []
    prev_thumbnail = None
    body_scene = 0
    for n, native in decoded_frames(source, width, height, ffmpeg):
        frame = cv2.resize(native, analysis_size) if scale != 1 else native
        fraction = green_fraction(frame)
        green = fraction >= .62
        sky_proposals = sky_candidates(frame) if not green else []
        hsv_sky = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sky = not green and float(np.mean((hsv_sky[:,:,1] < 90) & (hsv_sky[:,:,2] > 150))) >= .55
        thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36))
        cut = prev_thumbnail is not None and float(np.mean(cv2.absdiff(thumb, prev_thumbnail))) > 35
        if cut:
            body_scene += 1
        if (green or sky) and (cut or green != was_green or sky != was_sky):
            scene += 1
        row = {'frame': n, 'scene': scene if green or sky else None, 'view': 'ground' if green else ('sky' if sky else 'unresolved'), 'green_fraction': fraction,
               'ball': None, 'target': None, 'previous_to_current': None,
               'camera': {'reason': 'no_ground_registration'}, 'ball_proposals': [],
               'sky_proposals': sky_proposals, 'rim_proposals': []}
        row.update(body=None, clubhead=None, body_scene=body_scene,
                   timestamp_seconds=spec.get('pts', [])[n] if spec.get('pts') else None)
        if people_detector is not None:
            # All views, including golfer shots that are not majority green.
            # Skip only empty-view sampling; no carried pose is ever published.
            active = people_detector.identity.previous is not None
            if active or n % sample_step == 0 or cut:
                row['body'], row['clubhead'] = people_detector(frame, cut=cut, detail_frame=native)
        if detail_search is not None:
            row['detail_ball_candidate_space'] = 'native_source_pixels'
            # Extra native head proposals must not change the existing ball
            # search's exclusion region. Final publication checks both layers.
            ball_exclusion_head = (getattr(people_detector, 'baseline_head', row['clubhead'])
                                   if row['body'] is not None else None)
            row['detail_ball_candidates'] = detail_search(native, row['body'], scale=scale,
                head=ball_exclusion_head, sample=n % sample_step == 0 or cut)
            candidates = row['detail_ball_candidates']
            if semantic_client is not None and len(candidates) == 1 and candidates[0]['neural_checked']:
                from .shot_semantics import component_images
                semantic_images[n] = component_images(native, candidates[0],
                    semantic_client.directory/'evidence', source_sha256=source_sha256,
                    frame_number=n, body_height=candidates[0]['body_height'])
        if green:
            path = cache / f'{n:06d}.png'
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError('Unable to cache analysis frame')
            row['cup_proposal'] = detect_cup(frame)
            if n % sample_step == 0 or not was_green or cut:
                boxes = detector(frame)
                people = [b['box'] for b in boxes if b['class_id'] == 0 and b['score'] >= .35]
                row['ball_proposals'] = supported_ball_proposals(frame, boxes)
            row['rim_proposals'] = cup_candidates(frame, people)
        rows.append(row)
        prev_thumbnail, was_green, was_sky = thumb, green, sky
    if len(rows) != spec['frames']:
        raise RuntimeError('Decoded frame count disagrees with probed source')
    initializations = []
    for chain in unique_chains(rows, 'sky_proposals', min_support=12, radius=35, moving=True):
        if len(chain) < 12:
            continue
        for row, point in chain:
            row['ball'] = point
            row['ball_method'] = 'current_sky_component_with_temporal_support'
            row['ball_space'] = 'image'
        first, point = chain[0]
        initializations.append({'object':'ball','scene':first['scene'],'frame':first['frame'],
            'point':point['point'][:], 'method':'automatic_temporal_sky_component'})
    scenes = sorted({r['scene'] for r in rows if r['scene'] is not None})
    for scene in scenes:
        segment = [r for r in rows if r['scene'] == scene and r['view'] == 'ground']
        if not segment:
            continue
        read = lambda n: cv2.imread(str(cache / f'{n:06d}.png'))
        indices = [r['frame'] for r in segment]
        # Cup inference runs independently, including scenes without a ball.
        track_flag_cup(segment, rows, initializations, read, indices, scene)
        # Fit each actual source transition once; backtracking inverts it.
        previous = None
        for row in segment:
            current = read(row['frame'])
            if previous is not None:
                matrix, quality = affine(previous, current, None, None)
                row['previous_to_current'] = None if matrix is None else matrix.tolist()
                row['camera'] = quality
            previous = current
        indices = [r['frame'] for r in segment]
        proposals = [(r, b) for r in segment for b in r['ball_proposals']]
        # Two neural samples must agree plausibly; a single isolated detection
        # does not start a trajectory, and multiple similarly likely balls are
        # intentionally withheld rather than arbitrarily selecting one.
        stable = []
        for row, box in proposals:
            peers = [(rr, bb) for rr, bb in proposals if 0 < abs(rr['frame'] - row['frame']) <= sample_step * 2]
            agrees = any(np.linalg.norm(np.array(box['point']) - bb['point']) < max(80, analysis_size[0] * .15)
                         for rr, bb in peers)
            rivals = [b for b in row['ball_proposals'] if b is not box and b['score'] >= box['score'] - .12]
            if agrees and not rivals:
                stable.append((row, box))
        if not stable:
            continue
        seed_ball = None
        for seed_row, seed_box in sorted(stable, key=lambda item: item[1]['score'], reverse=True):
            seed_frame = seed_row['frame']
            seed_image = read(seed_frame)
            seed_ball = ball_component(seed_image, seed_box['point'], max(seed_box['box'][2] - seed_box['box'][0], seed_box['box'][3] - seed_box['box'][1]), proposal=True)
            if seed_ball is not None:
                break
        if seed_ball is None:
            continue
        seed_row['ball'] = seed_ball
        seed_row['ball_method'] = 'neural_detector_and_current_white_component'
        initializations.append({'object': 'ball', 'scene': scene, 'frame': seed_frame,
            'point': seed_ball['point'], 'method': 'automatic_neural_proposal', 'score': seed_box['score']})
        for direction, ordered in [(-1, [n for n in reversed(indices) if n < seed_frame]),
                                   (1, [n for n in indices if n > seed_frame])]:
            last = seed_ball
            search = np.array(last['point'])
            gaps = 0
            for n in ordered:
                r = rows[n]
                m = rows[n + 1]['previous_to_current'] if direction == -1 else r['previous_to_current']
                if m is not None:
                    h = np.vstack([m, [0, 0, 1]])
                    if direction == -1:
                        h = np.linalg.inv(h)
                    search = h[:2, :2] @ search + h[:2, 2]
                current = read(n)
                found = ball_component(current, search, max(last['size'])) if gaps <= 3 else None
                if found is None and gaps > 3:
                    anchors=[box for rr,box in stable if rr['frame']==n]
                    if len(anchors)==1:
                        box=anchors[0]
                        found=ball_component(current,box['point'],max(box['box'][2]-box['box'][0],box['box'][3]-box['box'][1]),proposal=True)
                        if found:
                            initializations.append({'object':'ball','scene':scene,'frame':n,'point':found['point'][:],
                                'method':'automatic_neural_reacquisition','score':box['score']})
                r['ball'] = found
                r['ball_method'] = 'automatic_neural_initialized_component_association' if found else 'not_resolved'
                if found:
                    last, search, gaps = found, np.array(found['point']), 0
                else:
                    gaps += 1
    cup_chains = unique_chains(rows, 'rim_proposals', min_support=18, radius=12)
    for run in cup_publication_runs(rows, cup_chains):
        for row, point in run:
            row['target'] = point['point'][:]
            row['target_method'] = 'current_visible_flat_cup_rim'
        first, point = run[0]  # First actual published support, never chain lead-in.
        initializations.append({'object':'cup','scene':first['scene'],'frame':first['frame'],
            'point':point['point'][:], 'method':'automatic_visible_flat_cup_rim'})
    from .shot_people import publish_people
    # First apply exactly the existing overlap and body/head publication gates.
    for row in rows:
        if row.get('clubhead') and row.get('ball'):
            if head_ball_overlap(row['clubhead'],row['ball'],scale=scale):
                row['clubhead'] = None
        if row.get('clubhead') and row.get('target') and np.linalg.norm(np.array(row['clubhead']['point'])-row['target'])<=6:
            row['clubhead'] = None
    publish_people(rows)
    heads_before_detail = {r['frame']: r['clubhead']['point'][:] for r in rows if r.get('clubhead')}
    head_removals = []
    # Native detail evidence supplements missing balls; it cannot overwrite the
    # established whole-view ball, target or camera channels. Confirmation is
    # bounded to actual consecutive components between three neural samples.
    previous_detail = None
    for row, component in confirmed_detail_chains(rows, sample_step=sample_step, analysis_scale=scale):
        if row['ball'] is not None:
            previous_detail = None
            continue
        point = np.asarray(component['point']) * scale
        size = np.asarray(component['size']) * scale
        neighbors = rows[max(0, row['frame']-sample_step):row['frame']+sample_step+1]
        if any(r.get('ball') and r['ball'].get('context') != 'near_player_native_detail'
               and r['body_scene'] == row['body_scene']
               and np.linalg.norm(point-r['ball']['point']) > max(8, max(size)*2)
               for r in neighbors):
            previous_detail = None
            continue
        if row.get('target') and np.linalg.norm(point - row['target']) <= max(12, max(size)):
            previous_detail = None
            continue
        row['ball'] = {'point': point.tolist(), 'size': size.tolist(),
                       'area': component['area'], 'area_unit': 'native_pixels_squared',
                       'provenance': 'current_pixels', 'body_track_id': component['body_id'],
                       'context': 'near_player_native_detail',
                       'confirmation_frames': component['confirmation_frames'],
                       'publication': 'retrospective_three_neural_checks_on_consecutive_components'}
        row['ball_method'] = 'current_native_pose_crop_neural_confirmed_round_component'
        row['ball_space'] = 'image'
        detail_key = (row['body_scene'], component['body_id'], row['frame'])
        if previous_detail != (*detail_key[:2], row['frame'] - 1):
            initializations.append({'object': 'ball', 'scene': row['scene'],
                'body_scene': row['body_scene'], 'frame': row['frame'],
                'point': point.tolist(), 'method': 'automatic_native_pose_crop_neural_chain'})
        previous_detail = detail_key
    # Return native coordinates; the renderer never has to infer scale.
    for row in rows:
        if row.get('clubhead') and row.get('ball'):
            separation = np.linalg.norm(np.array(row['clubhead']['point'])-row['ball']['point'])
            if head_ball_overlap(row['clubhead'],row['ball'],scale=scale):
                if row['ball'].get('context') == 'near_player_native_detail':
                    head_removals.append({'frame':row['frame'],'head_point_analysis':row['clubhead']['point'][:],
                                          'ball_method':row['ball_method'],'separation_analysis_px':float(separation)})
                row['clubhead'] = None  # A ball's dark shadow cannot become a head core.
        if row.get('clubhead') and row.get('target') and np.linalg.norm(np.array(row['clubhead']['point'])-row['target'])<=6:
            row['clubhead']=None  # Unresolved head/cup overlap cannot become a club.
    publish_people(rows)
    direct_vetoes = {v['frame'] for v in head_removals}
    for row in rows:
        if row['frame'] in heads_before_detail and row.get('clubhead') is None and row['frame'] not in direct_vetoes:
            head_removals.append({'frame':row['frame'], 'head_point_analysis':heads_before_detail[row['frame']],
                                 'reason':'consecutive_head_support_broken_after_detail_ball_overlap'})
    baseline_only = copy.deepcopy(rows)
    for baseline_row in baseline_only:
        if (baseline_row.get('clubhead') or {}).get('search_resolution') == 'native_source_pixels':
            baseline_row['clubhead'] = None
    publish_people(baseline_only)
    baseline_supported = {r['frame'] for r in baseline_only if r.get('clubhead')}
    head_native_joining_frames = [r['frame'] for r in rows if r.get('clubhead')
        and r['clubhead'].get('search_resolution') != 'native_source_pixels'
        and r['frame'] not in baseline_supported]
    semantic_diagnostics = {'enabled': semantic_client is not None, 'published_frames': [],
                            'independent_ground_truth': False, 'provides_coordinates': False}
    if semantic_client is not None:
        from .shot_semantics import semantic_chains
        prospective = semantic_chains(rows, sample_step=sample_step, analysis_scale=scale)
        anchors = sorted({n for r,c in prospective for n in c['confirmation_frames']})
        decisions = semantic_client.classify([semantic_images[n] for n in anchors if n in semantic_images])
        semantic_diagnostics.update(config=semantic_client.config, decisions=decisions,
                                    prospective_anchor_frames=anchors)
        previous_semantic = None
        for row, c in semantic_chains(rows, decisions=decisions, sample_step=sample_step, analysis_scale=scale):
            point = np.asarray(c['point'])*scale;size = np.asarray(c['size'])*scale
            proposed = {'point': point.tolist(), 'size': size.tolist()}
            neighbors = rows[max(0,row['frame']-sample_step):row['frame']+sample_step+1]
            conflict = any(r.get('ball') and r['body_scene']==row['body_scene']
                and r['ball'].get('context') != 'near_player_ai_semantics'
                and np.linalg.norm(point-r['ball']['point']) > max(8,max(size)*2) for r in neighbors)
            # A semantic ball cannot evict an established ball, cup or head.
            if (row['ball'] is not None or conflict
                    or (row.get('target') and np.linalg.norm(point-row['target'])<=max(12,max(size)))
                    or (row.get('clubhead') and head_ball_overlap(row['clubhead'],proposed,scale=scale))):
                previous_semantic = None;continue
            proposed.update(area=c['area'], area_unit='native_pixels_squared',
                provenance='current_pixels', body_track_id=c['body_id'], context='near_player_ai_semantics',
                confirmation_frames=c['confirmation_frames'],
                confirmation_receipt_keys=[decisions[n]['receipt_key'] for n in c['confirmation_frames']],
                semantic_model=semantic_client.config['model'], prompt_sha256=semantic_client.config['prompt_sha256'],
                publication='three_semantic_checks_on_consecutive_current_components')
            row['ball'] = proposed;row['ball_space'] = 'image'
            row['ball_method'] = 'current_native_pose_crop_ai_classified_round_component'
            semantic_diagnostics['published_frames'].append(row['frame'])
            key = (row['body_scene'], row['view'], c['body_id'], row['frame'])
            if previous_semantic != (*key[:3],row['frame']-1):
                initializations.append({'object':'ball','scene':row['scene'],'body_scene':row['body_scene'],
                    'frame':row['frame'],'point':point.tolist(),'method':'automatic_native_semantic_ball_chain'})
            previous_semantic = key
    detail_diagnostics = {'enabled':detail_search is not None, 'coordinate_space':'native_source_pixels',
                          'frames':[{'frame':r['frame'],'candidates':r.get('detail_ball_candidates', [])}
                                    for r in rows if r.get('detail_ball_candidates')]}
    for row in rows:
        row.pop('detail_ball_candidates', None)
        row.pop('detail_ball_candidate_space', None)
    for row in rows:
        if row.get('body'):
            body = row['body']
            body['bbox'] = [v / scale for v in body['bbox']]
            for value in body['keypoints'].values():
                value['point'] = [v / scale for v in value['point']]
        if row.get('clubhead'):
            club = row['clubhead']
            club['point'] = [v / scale for v in club['point']]
            club['shaft'] = [[v / scale for v in point] for point in club['shaft']]
            if 'proposal_box' in club:
                club['proposal_box'] = [v / scale for v in club['proposal_box']]
            if 'support_pixels' in club:
                club['support_pixels'] = [[v / scale for v in point] for point in club['support_pixels']]
            if 'shaft_width_px' in club:club['shaft_width_px']/=scale
            club.setdefault('core_area_unit','analysis_pixels_squared')
        if row['ball']:
            row['ball']['point'] = [v / scale for v in row['ball']['point']]
            row['ball']['size'] = [v / scale for v in row['ball']['size']]
        if row['target']:
            row['target'] = [v / scale for v in row['target']]
        if row['previous_to_current']:
            row['previous_to_current'][0][2] /= scale
            row['previous_to_current'][1][2] /= scale
    for item in initializations:
        item['point'] = [v / scale for v in item['point']]
    return {'frames': rows, 'initializations': initializations, 'operator_seed_count': 0,
            'head_analysis_frames_supported_only_via_native_joining': head_native_joining_frames,
            'detail_ball_head_overlap_vetoes': head_removals,
            'detail_ball_diagnostics': detail_diagnostics,
            'semantic_ball_diagnostics': semantic_diagnostics,
            'method': 'local_yolo11_sports_ball_native_detail_and_temporal_scene_vision', 'analysis_scale': scale,
            'limitations': ['Sports-ball model plus local association, not a validated universal golf detector',
                'Cup inference requires a repeated pale rim or visible cup-plus-flagstick; textured airborne views may remain unresolved',
                'Camera-aware image trajectory only; no physical speed, airborne 3D or optimized aim']}


def verify_output(source, output, *, ffmpeg='ffmpeg', ffprobe='ffprobe'):
    before, after = probe_source(source, ffprobe), probe_source(output, ffprobe)
    problems = []
    if before['frames'] != after['frames']:
        problems.append('frame_count_changed')
    delta = max(abs(a-b) for a, b in zip(before['pts'], after['pts']))
    if delta > max(1e-6, float(Fraction(before['time_base']))):
        problems.append('frame_presentation_times_changed')
    streams = lambda p: json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-show_streams', '-of', 'json', str(p)], timeout=30))['streams']
    audio_before = [s for s in streams(source) if s['codec_type'] == 'audio']
    audio_after = [s for s in streams(output) if s['codec_type'] == 'audio']
    audio = {'present': bool(audio_before), 'bit_exact': None}
    if audio_before:
        def digest(p):
            return subprocess.check_output([ffmpeg, '-v', 'error', '-i', str(p), '-map', '0:a:0', '-c', 'copy', '-f', 'md5', '-'], timeout=60).decode().strip()
        audio['bit_exact'] = bool(audio_after) and digest(source) == digest(output)
        audio['source_start_time'] = float(audio_before[0].get('start_time') or 0)
        audio['output_start_time'] = float(audio_after[0].get('start_time') or 0) if audio_after else None
        if not audio['bit_exact'] or abs(audio['source_start_time']-audio['output_start_time']) > .001:
            problems.append('audio_changed')
    result = {'ok': not problems, 'problems': problems, 'source_frames': before['frames'],
              'output_frames': after['frames'], 'max_pts_delta_seconds': delta, 'audio': audio}
    if problems:
        raise RuntimeError('Output verification failed: ' + ', '.join(problems))
    return result


def analyze_shot(source, outdir, *, expected_sha256=None, detector=None, model_path=None, people_detector=None,
                 ffmpeg='ffmpeg', ffprobe='ffprobe'):
    """The video is the only end-user input; provider configuration is server-owned."""
    from .shot_render import render_video
    source = Path(source).resolve()
    digest = expected_sha256 or sha256_file(source)
    binding = bind_source(source, digest)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if source == (outdir/'annotated.mp4').resolve():
        raise ValueError('Output directory must not overwrite the input video')
    spec = probe_source(source, ffprobe)
    model_path = Path(model_path or os.environ.get('FAIRWAY_SHOT_MODEL') or ROOT/'yolo11n.pt')
    if detector is None:
        detector = NeuralBallDetector(model_path)
    from .shot_people import BodyClubDetector
    from .shot_metrics import analyze_motion
    if people_detector is None:
        people_detector = BodyClubDetector(
            Path(os.environ.get('FAIRWAY_BODY_MODEL') or ROOT/'yolo11n-pose.pt'),
            Path(os.environ.get('FAIRWAY_CLUB_MODEL') or ROOT/'models/research/clubhead-db-best.pt'))
    people_model_hashes = {str(path):sha256_file(path) for path in (people_detector.pose_path, people_detector.head_path)}
    semantic_client = None
    if os.environ.get('FAIRWAYOS_BALL_VISION') == 'claude-cli':
        from .shot_semantics import SemanticBallClient
        semantic_client = SemanticBallClient(outdir/'semantic_ball')
    with tempfile.TemporaryDirectory(prefix='scene-cache-', dir=outdir) as cache:
        observations = observe_video(source, spec, Path(cache), detector, ffmpeg=ffmpeg, people_detector=people_detector,
                                     semantic_client=semantic_client, source_sha256=digest)
    analytics = analyze_motion(observations['frames'], source_sha256=digest)
    write_json(outdir/'analytics.json', analytics)
    write_json(outdir/'observations.json', observations)
    review = outdir/'review'
    review.mkdir(exist_ok=True)
    bind_source(source, digest)
    output = render_video(source, spec, observations, outdir/'annotated.mp4', ffmpeg=ffmpeg, review_dir=review)
    bind_source(source, digest)
    verification = verify_output(source, output, ffmpeg=ffmpeg, ffprobe=ffprobe)
    if people_model_hashes != {path:sha256_file(path) for path in people_model_hashes}:
        raise RuntimeError('Body/club model changed during analysis')
    write_json(outdir/'verification.json', verification)
    rows = observations['frames']
    ball_count = sum(r['ball'] is not None for r in rows)
    cup_count = sum(r['target'] is not None for r in rows)
    body_count = sum(r.get('body') is not None for r in rows)
    club_count = sum(r.get('clubhead') is not None for r in rows)
    club_resolutions = {resolution:sum(r.get('clubhead') is not None and
        r['clubhead'].get('search_resolution','analysis_pixels') == resolution for r in rows)
        for resolution in ('analysis_pixels','native_source_pixels')}
    doc = {'schema_version':'fairwayos-shot-intelligence.v1', 'analysis':'shot_intelligence',
           'automatic_no_seed':True, 'operator_seed_count':0,
           'source':{'sha256':digest, 'frames':spec['frames'], 'width':int(spec['width']), 'height':int(spec['height'])},
           'layers':{
               'ball':{'state':'partial' if ball_count else 'unavailable', 'method':observations['method'], 'frames_with_observation':ball_count, 'limitations':observations['limitations'][:1]},
               'cup':{'state':'partial' if cup_count else 'unavailable', 'method':'automatic_cup_detection_and_bidirectional_LK', 'frames_with_observation':cup_count, 'limitations':observations['limitations'][1:2]},
               'body':{'state':'partial' if body_count else 'unavailable', 'method':'current_yolo11_pose_with_primary_person_association', 'frames_with_observation':body_count, 'limitations':['Automatic primary golfer candidate; generic pose and person identity are not independently validated']},
               'clubhead':{'state':'partial' if club_count else 'unavailable', 'method':'current_hand_shaft_and_independent_dark_or_reflective_core_with_native_semantic_fallback', 'frames_with_observation':club_count, 'frames_by_search_resolution':club_resolutions, 'analysis_pixels_published_only_via_native_run_joining':len(observations['head_analysis_frames_supported_only_via_native_joining']), 'limitations':['Research neural and geometric search proposals; independent shaft/core gates, ambiguity veto; blurred, occluded or inseparable ball/head pixels remain unresolved','Native fallback needs a pixel-supported semantic proposal; head center accuracy is unmeasured']}},
           'analytics':analytics,
           'initializations':observations['initializations'],
           'physical_metrics':{'mph':None, 'carry_yards':None, 'reason':'Reliable physical calibration or measured data unavailable'},
           'strategy':{'target':'automatically detected visible cup' if cup_count else None, 'optimized_aim':None},
           'note':'Automatic image observations; target is the detected cup. No physical speed or optimized-aim claim.',
           'outputs':{'video':str(output), 'observations':str(outdir/'observations.json'), 'verification':str(outdir/'verification.json'), 'analytics':str(outdir/'analytics.json')},
           'verification':verification, 'model':{'weights_sha256':sha256_file(model_path) if model_path.is_file() else None, 'initialization':'automatic neural image proposals; no user coordinates'},
           'body_club_models':{'pose_sha256':sha256_file(people_detector.pose_path), 'club_proposal_sha256':sha256_file(people_detector.head_path), 'club_model_role':'local research search proposal; never sufficient to publish a head'},
           'semantic_ball':observations['semantic_ball_diagnostics'],
           'source_binding':binding}
    write_json(outdir/'result.json', doc)
    return doc


def main():
    parser = argparse.ArgumentParser(description='Automatic FairwayOS ball-and-target clip')
    parser.add_argument('video')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--expected-sha256')
    args = parser.parse_args()
    result = analyze_shot(args.video, args.outdir, expected_sha256=args.expected_sha256)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
