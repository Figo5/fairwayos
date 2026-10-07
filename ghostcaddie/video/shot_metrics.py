"""Video-time, pixel-space analytics derived only from observed consecutive rows."""
import math
from statistics import median
import numpy as np
from .shot_people import centre, torso_length


def distribution(values, unit):
    return {'unit':unit,'samples':len(values),'median':float(median(values)) if values else None,
            'p90':float(np.percentile(values,90)) if values else None}


def angle(a,b,c,minimum_length=2):
    first=np.array(a)-b;second=np.array(c)-b
    norm=np.linalg.norm(first)*np.linalg.norm(second)
    if np.linalg.norm(first)<minimum_length or np.linalg.norm(second)<minimum_length:return None
    return float(np.degrees(np.arccos(np.clip(np.dot(first,second)/norm,-1,1))))


def body_geometry(body):
    if not body:return {}
    kp=body['keypoints'];result={}
    for side in ('left','right'):
        names=[side+'_'+part for part in ('hip','knee','ankle')]
        if all(n in kp for n in names):
            value=angle(*(kp[n]['point'] for n in names),minimum_length=max(3,torso_length(body)*.05))
            if value is not None:result[side+'_knee_angle_2d_degrees']=value
    sh=centre(body,('left_shoulder','right_shoulder'));hip=centre(body,('left_hip','right_hip'))
    if sh is not None and hip is not None and np.linalg.norm(sh-hip)>20:
        result['torso_axis_2d_degrees']=float(np.degrees(np.arctan2(sh[0]-hip[0],hip[1]-sh[1])))
    return result


def analyze_motion(rows, *, source_sha256):
    names=('ball_image','ball_ground_residual','clubhead_image','clubhead_body_relative','hands_body_relative','cup_ground_residual','clubhead_torso_relative')
    series={n:[] for n in names}
    club_names=[n for n in names if n.startswith('clubhead')]
    club_resolution_series={}
    ball_context_series={}
    groups={};geometry={};joint_frames=0;previous=None;last_key=None;scale_epoch=0
    for row in rows:
        row['motion']={};body=row.get('body');club=row.get('clubhead')
        row['body_geometry']=body_geometry(body)
        key=None
        if body:
            identity=(row['body_scene'],body['track_id'])
            prior_body=previous.get('body') if previous else None
            changed_scale=prior_body and identity==last_key and not .95<torso_length(body)/max(1,torso_length(prior_body))<1.05
            if identity!=last_key or changed_scale:scale_epoch+=1
            key=(*identity,scale_epoch);last_key=identity
        elif row['scene'] is not None:
            key=('scene',row['scene'])
        group=groups.setdefault(key,{'frames':[],'motion':{n:[] for n in names},'geometry':{},'club_resolutions':{},'ball_contexts':{}}) if key is not None else None
        if group is not None:
            group['frames'].append(row['frame'])
            for name,value in row['body_geometry'].items():
                geometry.setdefault(name,[]).append(value);group['geometry'].setdefault(name,[]).append(value)
        if all(x is not None for x in (row.get('ball'),body,club,row.get('target'))):joint_frames+=1
        t=row.get('timestamp_seconds');pt=previous.get('timestamp_seconds') if previous else None
        dt=t-pt if t is not None and pt is not None else 0
        if previous is None or row['frame']!=previous['frame']+1 or not 0<dt<=.12:
            previous=row;continue
        def velocity(name,point,old):
            value=float(np.linalg.norm(np.array(point)-old)/dt)
            series[name].append(value);row['motion'][name+'_px_s']=value
            if group is not None:group['motion'][name].append(value)
            if name.startswith('clubhead'):club_sample(name,value)
            if name.startswith('ball_'):
                context=row['ball'].get('context','whole_view')
                ball_context_series.setdefault(context,{n:[] for n in names if n.startswith('ball_')})[name].append(value)
                if group is not None:
                    group['ball_contexts'].setdefault(context,{n:[] for n in names if n.startswith('ball_')})[name].append(value)
        def club_sample(name,value):
            resolution=club.get('search_resolution','analysis_pixels')
            club_resolution_series.setdefault(resolution,{n:[] for n in club_names})[name].append(value)
            if group is not None:
                group['club_resolutions'].setdefault(resolution,{n:[] for n in club_names})[name].append(value)
        same_scene=row['scene'] is not None and row['scene']==previous['scene']
        oldbody=previous.get('body');oldclub=previous.get('clubhead')
        same_body=body and oldbody and body['track_id']==oldbody['track_id'] and row['body_scene']==previous['body_scene']
        detail_contexts=('near_player_native_detail','near_player_ai_semantics')
        detail_pair=(same_body and row.get('ball') and previous.get('ball')
                     and row['ball'].get('context')==previous['ball'].get('context')
                     and row['ball'].get('context') in detail_contexts
                     and row['ball'].get('body_track_id')==previous['ball'].get('body_track_id')==body['track_id']
                     and row.get('view')==previous.get('view')
                     and .95 <= torso_length(body)/max(1,torso_length(oldbody)) <= 1.05)
        has_detail=any(r.get('ball') and r['ball'].get('context') in detail_contexts
                       for r in (row,previous))
        ball_pair_valid=detail_pair if has_detail else same_scene
        if row.get('ball') and previous.get('ball') and ball_pair_valid and row.get('ball_space','ground')==previous.get('ball_space','ground'):
            current=row['ball']['point'];old=previous['ball']['point']
            velocity('ball_image',current,old)
            if row.get('previous_to_current') is not None and row.get('ball_space','ground')=='ground':
                m=np.array(row['previous_to_current']);velocity('ball_ground_residual',current,m[:,:2]@old+m[:,2])
        if row.get('target') is not None and previous.get('target') is not None and same_scene and row.get('previous_to_current') is not None:
            m=np.array(row['previous_to_current']);velocity('cup_ground_residual',row['target'],m[:,:2]@previous['target']+m[:,2])
        if same_body:
            hip=centre(body,('left_hip','right_hip'));oldhip=centre(oldbody,('left_hip','right_hip'))
            bh=torso_length(body);oldbh=torso_length(oldbody)
            stable_scale=oldbh>1 and bh>1 and .95<bh/oldbh<1.05
            if (club and oldclub and club['body_track_id']==oldclub['body_track_id']
                    and club.get('search_resolution')==oldclub.get('search_resolution')):
                velocity('clubhead_image',club['point'],oldclub['point'])
                if stable_scale and hip is not None and oldhip is not None:
                    velocity('clubhead_body_relative',np.array(club['point'])-hip,np.array(oldclub['point'])-oldhip)
                    value=float(np.linalg.norm((np.array(club['point'])-hip)/bh-(np.array(oldclub['point'])-oldhip)/oldbh)/dt)
                    series['clubhead_torso_relative'].append(value)
                    club_sample('clubhead_torso_relative',value)
                    if group is not None:group['motion']['clubhead_torso_relative'].append(value)
            hands=centre(body,('left_wrist','right_wrist'));oldhands=centre(oldbody,('left_wrist','right_wrist'))
            if stable_scale and all(p is not None for p in (hip,oldhip,hands,oldhands)):
                velocity('hands_body_relative',hands-hip,oldhands-oldhip)
        previous=row
    units=lambda name:'torso-lengths/video-second' if name=='clubhead_torso_relative' else 'source-pixels/video-second'
    segments=[{'key':list(key),'first_frame':min(g['frames']),'last_frame':max(g['frames']),
               'motion':{n:distribution([] if (n.startswith('clubhead') and len(g['club_resolutions'])>1) or (n.startswith('ball_') and len(g['ball_contexts'])>1) else v,units(n)) for n,v in g['motion'].items()},
               'ball_motion_by_context':{r:{n:distribution(v,units(n)) for n,v in values.items()} for r,values in g['ball_contexts'].items()},
               'ball_mixed_context_summary_withheld':len(g['ball_contexts'])>1,
               'club_motion_by_search_resolution':{r:{n:distribution(v,units(n)) for n,v in values.items()} for r,values in g['club_resolutions'].items()},
               'club_mixed_resolution_summary_withheld':len(g['club_resolutions'])>1,
               'body_geometry':{n:distribution(v,'projected image degrees') for n,v in g['geometry'].items()}} for key,g in groups.items()]
    multiple_people_scales=sum(bool(g['geometry']) for g in groups.values())>1
    return {'schema_version':'four-layer-video-analytics.v1','source_sha256':source_sha256,
            'time_basis':'source presentation timestamps; broadcast replay speed may differ from real time',
            'motion':{name:distribution([] if (multiple_people_scales and name.startswith(('clubhead','hands'))) or (name.startswith('clubhead') and len(club_resolution_series)>1) or (name.startswith('ball_') and len(ball_context_series)>1) else values,units(name)) for name,values in series.items()},
            'ball_motion_by_context':{r:{n:distribution(v,units(n)) for n,v in values.items()} for r,values in ball_context_series.items()},
            'ball_mixed_context_summary_withheld':len(ball_context_series)>1,
            'club_motion_by_search_resolution':{r:{n:distribution([] if multiple_people_scales else v,units(n)) for n,v in values.items()} for r,values in club_resolution_series.items()},
            'body_geometry':{} if multiple_people_scales else {name:distribution(values,'projected image degrees') for name,values in geometry.items()},
            'segments':segments,'body_aggregate_withheld':multiple_people_scales,
            'simultaneous_four_layer_frames':joint_frames,
            'impact':None,'swing_tempo':None,'physical':{'ball_mph':None,'clubhead_mph':None,'carry_yards':None},
            'limitations':['Generic pose and local research clubhead evidence; coverage does not prove accuracy',
                           'Pixel motion includes perspective, detector jitter and any uncompensated camera motion',
                           'Body-relative motion subtracts hip translation, not a calibrated camera or 3D rotation',
                           'Body distributions are separated by scene, person track and torso-scale changes; no mixed-person summary',
                           'Club distributions also separate search resolutions; mixed-resolution summaries are withheld',
                           'Ball distributions separate whole-view, native neural and AI semantic evidence; mixed summaries are withheld',
                           'Cup residual measures tracking/registration jitter, not physical cup movement',
                           'No inferred contact, swing tempo, physical speed or 3D body rotation']}
