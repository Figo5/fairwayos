"""Native video renderer for automatic ball-and-cup observations."""
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import math
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .scene_intelligence import ObservedGroundTrail


@lru_cache(32)
def font(size, bold=False):
    options = [Path('/System/Library/Fonts/Supplemental') / ('Arial Bold.ttf' if bold else 'Arial.ttf'),
               Path('/usr/share/fonts/truetype/dejavu') / ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')]
    for path in options:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


@lru_cache(32)
def card(kind):
    if kind == 'brand':
        image = Image.new('RGBA', (319, 98))
        d = ImageDraw.Draw(image)
        d.rounded_rectangle((0, 0, 318, 97), 12, fill=(9, 26, 22, 218))
        d.line((17, 19, 17, 40), fill=(152, 230, 197), width=3)
        d.text((29, 13), 'FAIRWAYOS', font=font(24, True), fill=(247, 249, 240))
        d.text((17, 47), 'SHOT INTELLIGENCE', font=font(16, True), fill=(236, 239, 226))
        d.text((17, 73), 'BALL · CLUB · BODY · CUP', font=font(11), fill=(169, 198, 181))
    elif kind == 'target':
        image = Image.new('RGBA', (173, 46))
        d = ImageDraw.Draw(image)
        d.rounded_rectangle((0, 0, 172, 45), 9, fill=(13, 29, 24, 220))
        d.ellipse((10, 12, 18, 20), fill=(236, 205, 119))
        d.text((27, 7), 'CUP TARGET', font=font(14, True), fill=(252, 238, 199))
        d.text((27, 26), 'Detected cup', font=font(11), fill=(194, 204, 179))
    else:
        image = Image.new('RGBA', (340, 73))
        d = ImageDraw.Draw(image)
        d.rounded_rectangle((0, 0, 339, 72), 11, fill=(9, 26, 22, 217))
        title, subtitle = ('AIRBORNE BALL', 'Observed image location · no 3D estimate') if kind == 'air' else (('OBSERVED BALL', 'Native detail observation · video motion only') if kind == 'detail' else (('OBSERVED BALL', 'Ball marker + camera-aware trail') if kind == 'ball' else ('FOLLOWING THE SHOT', 'Observations appear when resolved')))
        d.text((17, 13), title, font=font(18, True), fill=(236, 242, 227))
        d.text((17, 42), subtitle, font=font(13), fill=(174, 210, 188))
    return np.array(image)


@lru_cache(64)
def body_card(angle_text, club_text):
    image = Image.new('RGBA', (252, 100))
    d = ImageDraw.Draw(image)
    d.rounded_rectangle((0, 0, 251, 99), 11, fill=(9, 26, 22, 217))
    d.text((15, 12), 'SWING OBSERVATIONS', font=font(15, True), fill=(235, 242, 229))
    d.text((15, 39), angle_text, font=font(13), fill=(194, 215, 199))
    d.text((15, 61), club_text, font=font(13), fill=(161, 212, 245))
    d.text((15, 82), '2D video · physical speed unverified', font=font(10), fill=(169, 188, 176))
    return np.array(image)


def blend(frame, rgba, x, y, scale, mask):
    rgba = cv2.resize(rgba, (max(1, round(rgba.shape[1]*scale)), max(1, round(rgba.shape[0]*scale))))
    h, w = rgba.shape[:2]
    x, y = max(0, min(frame.shape[1]-w, round(x))), max(0, min(frame.shape[0]-h, round(y)))
    alpha = rgba[:, :, 3:4] / 255
    roi = frame[y:y+h, x:x+w]
    roi[:] = np.clip(roi*(1-alpha) + rgba[:, :, :3][:, :, ::-1]*alpha, 0, 255).astype(np.uint8)
    mask[y:y+h, x:x+w] = 255


def compose(frame, row, trail):
    height, width = frame.shape[:2]
    scale = min(height / 720, width / 1266)
    out, layer = frame.copy(), frame.copy()
    mask = np.zeros((height, width), np.uint8)
    point = row['ball']['point'] if row['ball'] else None
    m = row['previous_to_current']
    matrix = None if m is None else [*m, [0, 0, 1]]
    points = trail.update(row['frame'], str(row['scene']), point, matrix)
    if len(points) > 1:
        coords = np.round(points).astype(np.int32)
        cv2.polylines(layer, [coords], False, (31, 48, 30), max(2, round(6*scale)), cv2.LINE_AA)
        cv2.polylines(layer, [coords], False, (156, 229, 169), max(1, round(2*scale)), cv2.LINE_AA)
        cv2.polylines(mask, [coords], False, 255, max(3, round(10*scale)), cv2.LINE_AA)
    if point:
        center = tuple(map(round, point))
        radius = max(round(12*scale), round(max(row['ball']['size'])*.7 + 5*scale))
        cv2.circle(layer, center, radius+1, (24, 44, 25), max(2, round(3*scale)), cv2.LINE_AA)
        cv2.circle(layer, center, radius, (163, 236, 185), max(1, round(2*scale)), cv2.LINE_AA)
        cv2.circle(mask, center, radius+5, 255, -1)
    target = row['target']
    body, club = row.get('body'), row.get('clubhead')
    if body:
        from .shot_people import centre
        shoulder = centre(body, ('left_shoulder', 'right_shoulder'))
        hip = centre(body, ('left_hip', 'right_hip'))
        hands = centre(body, ('left_wrist', 'right_wrist'))
        # Two torso anchors and hands, rather than a full-body skeleton.
        for p in (shoulder, hip, hands):
            if p is not None:
                c = tuple(map(round, p))
                cv2.circle(layer, c, max(3,round(4*scale)), (202, 227, 212), 1, cv2.LINE_AA)
                cv2.circle(mask, c, max(4,round(7*scale)), 255, -1)
    history = getattr(trail, 'club_history', [])
    identity = (row.get('body_scene'), club.get('body_track_id')) if club else None
    if (not club or identity != getattr(trail, 'club_identity', None)
            or row['frame'] != getattr(trail, 'club_frame', -2)+1):
        history = []
    if club:
        history = (history + [club['point']])[-12:]
        c = tuple(map(round, club['point']))
        if len(history)>1:
            coords = np.round(history).astype(np.int32)
            cv2.polylines(layer, [coords], False, (241, 193, 130), max(1,round(2*scale)), cv2.LINE_AA)
            cv2.polylines(mask, [coords], False, 255, max(3,round(6*scale)), cv2.LINE_AA)
        cv2.circle(layer, c, max(6,round(9*scale)), (241, 193, 130), max(1,round(2*scale)), cv2.LINE_AA)
        cv2.circle(mask, c, max(8,round(13*scale)), 255, -1)
    trail.club_history, trail.club_identity, trail.club_frame = history, identity, row['frame']
    if target:
        x, y = map(round, target)
        rx, ry = max(8, round(45*scale)), max(3, round(14*scale))
        cv2.ellipse(layer, (x, y), (rx+1, ry+1), 0, 0, 360, (19, 37, 24), max(2, round(4*scale)), cv2.LINE_AA)
        cv2.ellipse(layer, (x, y), (rx, ry), 0, 0, 360, (119, 206, 240), max(1, round(2*scale)), cv2.LINE_AA)
        cv2.ellipse(mask, (x, y), (rx+6, ry+6), 0, 0, 360, 255, -1)
    out = cv2.addWeighted(layer, .84, out, .16, 0)
    if target:
        bx = min(width - round(197*scale), x + round(55*scale))
        by = max(round(130*scale), min(height-round(151*scale), y-round(28*scale)))
        blend(out, card('target'), bx, by, scale, mask)
    blend(out, card('brand'), 25*scale, 24*scale, scale, mask)
    if body:
        geometry = row.get('body_geometry', {})
        angle = geometry.get('torso_axis_2d_degrees')
        angle_text = f'Torso axis in image: {round(angle)}°' if angle is not None else 'Body landmarks observed'
        speed = row.get('motion', {}).get('clubhead_image_px_s')
        club_text = f'Club image motion: {round(speed)} px/s' if speed is not None else ('Clubhead observed' if club else 'Clubhead unresolved')
        blend(out, body_card(angle_text, club_text), width-277*scale, 24*scale, scale, mask)
    ball_kind = ('detail' if row.get('ball', {}).get('context') in ('near_player_native_detail','near_player_ai_semantics')
                 else ('air' if row.get('ball_space') == 'image' else 'ball')) if point else 'following'
    blend(out, card(ball_kind), 25*scale, height-98*scale, scale, mask)
    cv2.putText(out, 'AUTOMATIC VIDEO ANALYSIS', (max(0, width-round(307*scale)), height-round(29*scale)),
                cv2.FONT_HERSHEY_SIMPLEX, max(.25, .44*scale), (238, 245, 225), 1, cv2.LINE_AA)
    mask[max(0, height-round(48*scale)):height-round(12*scale), max(0, width-round(318*scale)):] = 255
    return out, mask


def render_video(source, spec, observations, dest, *, ffmpeg='ffmpeg', review_dir=None):
    """Encode every native frame at its own PTS, including variable cadence."""
    import av
    from .automatic_shot import decoded_frames
    dest = Path(dest)
    tmp = dest.with_name(dest.stem+'.video.mp4')
    width, height = int(spec['width']), int(spec['height'])
    time_base = Fraction(spec['time_base'])
    rate = Fraction(spec['avg_frame_rate'])
    pts = spec['frame_pts']
    durations = spec['frame_durations']
    trail = ObservedGroundTrail(max(2, round(float(rate)*3)))
    duration_by_pts = dict(zip(pts, durations))
    frames_written = 0
    with av.open(str(tmp), 'w', options={'video_track_timescale': str(time_base.denominator)}) as output:
        stream = output.add_stream('libx264', rate=rate)
        stream.width, stream.height = width+(width%2), height+(height%2)
        stream.pix_fmt = 'yuv420p'
        stream.time_base = stream.codec_context.time_base = time_base
        stream.codec_context.thread_count = 2
        stream.codec_context.colorspace = stream.codec_context.color_primaries = stream.codec_context.color_trc = 1
        stream.codec_context.color_range = 1
        stream.options = {'crf':'18', 'preset':'fast'}
        for n, frame in decoded_frames(source, width, height, ffmpeg):
            out, mask = compose(frame, observations['frames'][n], trail)
            if review_dir and (n % max(1, round(float(rate))) == 0 or n == len(pts)-1):
                cv2.imwrite(str(Path(review_dir)/f'frame-{n:06d}.png'), out)
                cv2.imwrite(str(Path(review_dir)/f'mask-{n:06d}.png'), mask)
            if width%2 or height%2:
                out = cv2.copyMakeBorder(out, 0, height%2, 0, width%2, cv2.BORDER_CONSTANT)
            rgb = av.VideoFrame.from_ndarray(out, format='bgr24')
            yuv = rgb.reformat(format='yuv420p', src_colorspace='ITU709', dst_colorspace='ITU709',
                               src_color_range='JPEG', dst_color_range='MPEG')
            yuv.colorspace, yuv.color_range = 1, 1
            yuv.pts, yuv.time_base = pts[n], time_base
            yuv.duration = durations[n]
            for packet in stream.encode(yuv):
                if packet.pts in duration_by_pts:
                    packet.duration = duration_by_pts[packet.pts]
                output.mux(packet)
            frames_written += 1
        for packet in stream.encode():
            if packet.pts in duration_by_pts:
                packet.duration = duration_by_pts[packet.pts]
            output.mux(packet)
    if frames_written != len(pts):
        raise RuntimeError('Renderer did not preserve the decoded frame count')
    subprocess.run([ffmpeg, '-v', 'error', '-y', '-copyts', '-i', str(tmp), '-i', str(source),
        '-map', '0:v:0', '-map', '1:a:0?', '-c', 'copy', '-video_track_timescale', str(time_base.denominator),
        '-movflags', '+faststart', str(dest)], check=True, timeout=120)
    tmp.unlink()
    return dest
