"""Current-frame body and club evidence for the video-only local prototype.

Pose confidence is not identity accuracy. A rejected research head detector is
only a search proposal: publication requires independent pixels and a visible
shaft attached to current hands. Gaps and ambiguous people remain unresolved.
"""
from pathlib import Path
import copy
import math

import cv2
import numpy as np

NAMES = ('nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder',
         'left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip',
         'left_knee','right_knee','left_ankle','right_ankle')


def centre(body, names):
    points = [body['keypoints'][name]['point'] for name in names if name in body['keypoints']]
    return np.mean(points, axis=0) if len(points) == len(names) else None


def torso_length(body):
    sh=centre(body,('left_shoulder','right_shoulder'))
    hip=centre(body,('left_hip','right_hip'))
    return float(np.linalg.norm(sh-hip)) if sh is not None and hip is not None else 0.


def iou(a, b):
    x0,y0=np.maximum(a[:2],b[:2]);x1,y1=np.minimum(a[2:],b[2:])
    inter=max(0,x1-x0)*max(0,y1-y0)
    return inter/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def person_score(body, image_size):
    """Prominence plus current hand/posture evidence; no names/shirt colours."""
    w,h=image_size;b=body['bbox'];bh=b[3]-b[1]
    hands=centre(body,('left_wrist','right_wrist'))
    shoulders=centre(body,('left_shoulder','right_shoulder'))
    hips=centre(body,('left_hip','right_hip'))
    # Small players are eligible only after a fresh detail-view pose; full-view
    # small boxes (often caddies) retain the baseline admission floor.
    detail=body.get('method')=='current_yolo11_pose_detail'
    if bh < h*(.08 if detail else .18) or any(p is None for p in (hands,shoulders,hips)):
        return None
    if detail and bh<h*.18 and not (min(body['keypoints'][n]['confidence'] for n in ('left_wrist','right_wrist'))>=.7
                                   and max(body['keypoints'][n]['confidence'] for n in ('left_wrist','right_wrist'))>=.85):
        return None
    if b[0] < w*.025 or b[2] > w*.975 or hips[1] <= shoulders[1]:
        return None
    gap=np.linalg.norm(np.array(body['keypoints']['left_wrist']['point'])-body['keypoints']['right_wrist']['point'])/bh
    if gap>.13 or hands[1] < shoulders[1]+bh*.12 or hands[1]>hips[1]+bh*.32:
        return None
    lean=abs(math.degrees(math.atan2(shoulders[0]-hips[0],hips[1]-shoulders[1])))
    if lean<12 and hands[1]-hips[1]<bh*.04:
        return None  # Folded hands on an upright spectator are not setup evidence.
    central=1-min(1,abs((b[0]+b[2])/2-w/2)/(w*.5))
    return float(.35*central+.35*(1-gap/.13)+.2*body['confidence']+.1*min(1,bh/h))


class PersonIdentity:
    def __init__(self):
        self.previous=None;self.next_id=1;self.misses=0

    def reset(self, *, new_clip=False):
        self.previous=None;self.misses=0
        if new_clip:self.next_id=1

    def select(self, people, image_size, *, cut=False):
        if cut:self.reset()
        ranked=[]
        for body in people:
            w,h=image_size;b=body['bbox'];kp=body['keypoints']
            # Cropped foreground crew/caddies cannot inherit the player's track.
            # Need current head plus a foot, not a confident upper-body box alone.
            head=kp.get('nose')
            feet=[kp[n] for n in ('left_ankle','right_ankle') if n in kp]
            rear=rear_subject_score(body,(w,h)) if self.previous is None else None
            if ((self.previous is None and (head is None or head['confidence']<.6) and rear is None)
                    or not any(p['confidence']>=.6 for p in feet)
                    or b[0]<=w*.01 or b[2]>=w*.99 or b[1]<=h*.01 or b[3]>=h*.99):
                continue
            if self.previous is not None:
                overlap=iou(body['bbox'],self.previous['bbox'])
                ph=self.previous['bbox'][3]-self.previous['bbox'][1]
                bh=body['bbox'][3]-body['bbox'][1]
                if overlap<.35 or not .7<bh/ph<1.4:continue
                # Appearance only vetoes a spatial match; it cannot acquire one.
                if np.linalg.norm(np.array(body['appearance'])-self.previous['appearance'])>.45:continue
                score=overlap
            else:
                if body.get('crowded_acquisition'):
                    continue
                score=person_score(body,image_size)
                if score is None:score=rear
                if score is None:continue
                if rear is not None and person_score(body,image_size) is None:body['acquisition']='complete_rear_subject_pose'
            ranked.append((score,body))
        ranked.sort(key=lambda pair:pair[0],reverse=True)
        if not ranked or (len(ranked)>1 and ranked[0][0]-ranked[1][0]<.12):
            self.misses+=1
            if self.misses>=3:self.reset()
            return None
        body=ranked[0][1]
        if self.previous is None:
            body['track_id']=self.next_id;self.next_id+=1
        else:
            body['track_id']=self.previous['track_id']
            if self.previous.get('acquisition')=='complete_rear_subject_pose' and body['keypoints'].get('nose',{}).get('confidence',0)<.6:
                body['acquisition']='complete_rear_subject_pose'
        self.previous=body;self.misses=0
        return body


def rear_subject_score(body,image_size):
    """A prominent complete rear subject, without inferring a face or grip.

    This is generic primary-subject association, not named-player recognition.
    Hidden hand landmarks are withheld from both rendering and analytics.
    """
    w,h=image_size;kp=body['keypoints'];b=body['bbox'];bh=b[3]-b[1]
    if bh<h*.3 or body.get('crowded_acquisition') or kp.get('nose',{}).get('confidence',0)>=.2:return None
    needed=('left_ear','right_ear','left_shoulder','right_shoulder','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_wrist','right_wrist')
    if not all(n in kp for n in needed):return None
    if min(kp[n]['confidence'] for n in ('left_ear','right_ear','left_wrist','right_wrist'))<.6:return None
    if min(kp[n]['confidence'] for n in needed if not n.endswith(('ear','wrist')))<.9:return None
    sh=centre(body,('left_shoulder','right_shoulder'));hip=centre(body,('left_hip','right_hip'))
    ears=centre(body,('left_ear','right_ear'));hands=centre(body,('left_wrist','right_wrist'))
    feet=centre(body,('left_ankle','right_ankle'))
    if not ears[1]<sh[1]<hands[1]<hip[1]<feet[1]:return None
    if np.linalg.norm(np.array(kp['left_wrist']['point'])-kp['right_wrist']['point'])>bh*.15:return None
    stance=abs(kp['left_ankle']['point'][0]-kp['right_ankle']['point'][0])/bh
    if not .12<=stance<=.4:return None
    # Ear/leg confidence alone also admits an upright rear spectator. Without
    # a projected golf posture, this view needs an independent action/club
    # classifier. Do not promote the upright development case as a golfer.
    lean=abs(math.degrees(math.atan2(sh[0]-hip[0],hip[1]-sh[1])))
    if lean<12:return None
    return float(.6+.2*(1-min(1,abs((b[0]+b[2])/2-w/2)/(w*.5)))+.1*body['confidence'])


def compact_head(frame, box, endpoint, body_height, shaft_direction=None, shaft_width=0, *, ambiguity_px_scale=1., diagnostics=None):
    """Resolve an actual dark head core, not the learned box centre/white ball."""
    h,w=frame.shape[:2];x0,y0,x1,y1=map(float,box)
    x0,y0=max(0,math.floor(x0)),max(0,math.floor(y0));x1,y1=min(w,math.ceil(x1)),min(h,math.ceil(y1))
    if x1<=x0 or y1<=y0:return None
    pad=max(5,round(body_height*.025))
    surround=frame[max(0,y0-pad):min(h,y1+pad),max(0,x0-pad):min(w,x1+pad)]
    gray=cv2.cvtColor(frame[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
    background=float(np.median(cv2.cvtColor(surround,cv2.COLOR_BGR2GRAY)))
    mask=(gray<background-18).astype(np.uint8)*255
    # Remove thin shaft pixels before locating the independent head lobe.
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    count,labels,stats,centres=cv2.connectedComponentsWithStats(mask)
    candidates=[]
    for i in range(1,count):
        bx,by,bw,bh,area=stats[i];point=np.array([x0+centres[i,0],y0+centres[i,1]])
        if not 3<=area<=body_height**2*.012 or max(bw,bh)>body_height*.15:continue
        if max(bw,bh)/max(1,min(bw,bh))>8 or area/(bw*bh)<.25:continue
        # The physical lobe must be wider than an unresolved shaft edge.
        if min(bw,bh)<2 or np.linalg.norm(point-endpoint)>max(8,body_height*.045):continue
        if shaft_direction is not None:
            direction=np.array(shaft_direction,float)
            norm=np.linalg.norm(direction)
            if norm<1:continue
            perpendicular=np.array([-direction[1],direction[0]])/norm
            yy,xx=np.nonzero(labels==i)
            transverse=np.c_[xx,yy]@perpendicular
            if np.percentile(transverse,90)-np.percentile(transverse,10)<max(2.5,body_height*.008,shaft_width*1.8):continue
        candidates.append((np.linalg.norm(point-endpoint),point.tolist(),int(area)))
    candidates.sort(key=lambda item:item[0])
    if not candidates:return None
    if len(candidates)>1 and candidates[1][0]-candidates[0][0]<3*ambiguity_px_scale:
        if diagnostics is not None:diagnostics['core_ambiguous']=True
        return None
    return {'point':candidates[0][1],'dark_core_area':candidates[0][2]}


def shaft_pixel_width(gray,p,q):
    direction=q-p;length=np.linalg.norm(direction)
    if length<1:return float('inf')
    perp=np.array([-direction[1],direction[0]])/length
    widths=[]
    for fraction in (.3,.4,.5,.6,.7):
        midpoint=p+(q-p)*fraction
        offsets=np.arange(-8,9);samples=np.round(midpoint+offsets[:,None]*perp).astype(int)
        samples[:,0]=np.clip(samples[:,0],0,gray.shape[1]-1);samples[:,1]=np.clip(samples[:,1],0,gray.shape[0]-1)
        values=gray[samples[:,1],samples[:,0]].astype(float)
        dark=abs(values-np.median(values))>18
        # Current line must have an independently measurable thin cross-section.
        if not dark.any():continue
        nearest=int(np.argmin(np.where(dark,abs(offsets),999)))
        if abs(offsets[nearest])>2:continue
        lo=hi=nearest
        while lo>0 and dark[lo-1]:lo-=1
        while hi<len(dark)-1 and dark[hi+1]:hi+=1
        if lo==0 or hi==len(dark)-1:continue
        widths.append(hi-lo+1)
    return float(np.median(widths)) if len(widths)>=3 else float('inf')


def reflective_head(frame, endpoint, body_height, direction, shaft_width, *, diagnostics=None):
    """Current elongated metal lobe across a shaft; round bright balls vetoed."""
    if not np.isfinite(shaft_width):return None
    radius=max(8,round(body_height*.045));h,w=frame.shape[:2]
    x,y=np.round(endpoint).astype(int)
    x0,y0=max(0,x-radius),max(0,y-radius);x1,y1=min(w,x+radius+1),min(h,y+radius+1)
    if x0==0 or y0==0 or x1==w or y1==h:return None
    patch=frame[y0:y1,x0:x1];gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
    background=float(np.median(np.r_[gray[0],gray[-1],gray[:,0],gray[:,-1]]))
    mask=((gray>background+20)&(hsv[:,:,1]<70)).astype(np.uint8)
    count,labels,stats,centres=cv2.connectedComponentsWithStats(mask)
    axis=np.array(direction,float);axis/=max(1,np.linalg.norm(axis));perp=np.array([-axis[1],axis[0]])
    found=[]
    for i in range(1,count):
        bx,by,bw,bh,area=stats[i]
        if not 6<=area<=body_height**2*.004 or max(bw,bh)>body_height*.08:continue
        if bx==0 or by==0 or bx+bw>=mask.shape[1] or by+bh>=mask.shape[0]:continue
        yy,xx=np.nonzero(labels==i);pixels=np.c_[xx+x0,yy+y0]
        # A ball touching a blade can form an elongated connected component.
        # Its thick round lobe must not be laundered into the head centroid.
        core_radius=float(cv2.distanceTransform((labels==i).astype(np.uint8),cv2.DIST_L2,5).max())
        if core_radius>max(2.,body_height*.006):continue
        along=np.percentile(pixels@axis,90)-np.percentile(pixels@axis,10)
        transverse=np.percentile(pixels@perp,90)-np.percentile(pixels@perp,10)
        # Reject round balls and bright shaft fragments, including thick ones.
        if transverse<max(4,body_height*.01,shaft_width*1.8) or transverse<max(1,along)*1.7:continue
        point=[x0+float(centres[i,0]),y0+float(centres[i,1])]
        if np.linalg.norm(np.array(point)-endpoint)>max(6,body_height*.035):continue
        found.append({'point':point,'head_core_area':int(area),'core_polarity':'bright',
                      'support_pixels':pixels.tolist()})
    if len(found)!=1:
        if len(found)>1 and diagnostics is not None:diagnostics['core_ambiguous']=True
        return None
    return found[0]


def shaft_search_club(frame,body, *, diagnostics=None, ambiguity_px_scale=1.):
    """Proposal-free putter search, independently verified by a thin shaft."""
    if body is None:return None
    wrists=[v for n,v in body['keypoints'].items() if n.endswith('wrist') and v['confidence']>=.75]
    if not wrists:return None
    hands=np.mean([v['point'] for v in wrists],axis=0);bh=body['bbox'][3]-body['bbox'][1]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);edges=cv2.Canny(gray,45,120)
    rad=round(bh*1.35);hx,hy=np.round(hands).astype(int);h,w=gray.shape
    roi=np.zeros_like(edges);roi[max(0,hy-rad):min(h,hy+rad),max(0,hx-rad):min(w,hx+rad)]=edges[max(0,hy-rad):min(h,hy+rad),max(0,hx-rad):min(w,hx+rad)]
    lines=cv2.HoughLinesP(roi,1,np.pi/360,threshold=max(15,round(bh*.08)),minLineLength=max(20,round(bh*.2)),maxLineGap=max(3,round(bh*.035)))
    if lines is None:return None
    from ghostcaddie.demo.shaft_head import _shaft_ribbon
    found=[]
    for line in lines.reshape(-1,4):
        p,q=np.array(line[:2],float),np.array(line[2:],float)
        if np.linalg.norm(q-hands)<np.linalg.norm(p-hands):p,q=q,p
        if np.linalg.norm(p-hands)>bh*.14 or not .25*bh<=np.linalg.norm(q-hands)<=1.35*bh:continue
        width=shaft_pixel_width(gray,p,q)
        if width>max(3,bh*.015):continue
        core=reflective_head(frame,q,bh,q-p,width,diagnostics=diagnostics)
        if core is None and q[1]>hands[1]+bh*.2:
            # The unlearned dark path is limited to a turf-backed head below
            # the grip. It cannot search arbitrary sleeve/tree endpoints.
            pad=max(8,bh*.045);box=[q[0]-pad,q[1]-pad,q[0]+pad,q[1]+pad]
            candidate=compact_head(frame,box,q,bh,q-p,width,ambiguity_px_scale=ambiguity_px_scale,diagnostics=diagnostics)
            if candidate:
                x,y=np.round(candidate['point']).astype(int);radius=max(8,round(bh*.035))
                patch=frame[max(0,y-radius):min(h,y+radius+1),max(0,x-radius):min(w,x+radius+1)]
                hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
                turf=float(np.mean((hsv[:,:,0]>=24)&(hsv[:,:,0]<=95)&(hsv[:,:,1]>=20)&(hsv[:,:,1]<=230)))
                # A green shadow is not independent head appearance.
                cx,cy=np.round(candidate['point']).astype(int)
                pigment=cv2.cvtColor(frame[max(0,cy-1):min(h,cy+2),max(0,cx-1):min(w,cx+2)],cv2.COLOR_BGR2HSV)
                green_core=float(np.mean((pigment[:,:,0]>=24)&(pigment[:,:,0]<=95)&(pigment[:,:,1]>=70)))
                if turf>=.65 and green_core<.5:core=dict(candidate,head_core_area=candidate['dark_core_area'],core_polarity='dark')
        if core is None:continue
        if anatomical_overlap(core['point'],body):continue
        ribbon=_shaft_ribbon(gray,(0,0),p.tolist(),q.tolist(),max(.5,bh/400))
        if not ribbon['accepted']:continue
        found.append(dict(core,shaft=[p.tolist(),q.tolist()],body_track_id=body['track_id'],
            visible_hand_count=len(wrists),method='current_hand_thin_shaft_and_'+core['core_polarity']+'_lobe',
            provenance='current_pixels',model_role='automatic_pose_with_geometric_search',
            shaft_width_px=width))
    if not found:return None
    if any(np.linalg.norm(np.array(c['point'])-found[0]['point'])>max(6*ambiguity_px_scale,bh*.025) for c in found[1:]):
        if diagnostics is not None:diagnostics['ambiguous']=True
        return None
    return max(found,key=lambda c:c['head_core_area'])


def anatomical_overlap(point,body):
    bh=body['bbox'][3]-body['bbox'][1]
    return any(np.linalg.norm(np.array(point)-v['point'])<max(4,bh*.035) for n,v in body['keypoints'].items()
               if n in ('nose','left_eye','right_eye','left_ear','right_ear','left_ankle','right_ankle'))


def fuse_club_candidates(learned,geometric,body_height,*,learned_ambiguous=False,geometric_ambiguous=False,ambiguity_px_scale=1.):
    if learned_ambiguous or geometric_ambiguous:return None
    if learned and geometric and np.linalg.norm(np.array(learned['point'])-geometric['point'])>max(8*ambiguity_px_scale,body_height*.04):return None
    return learned or geometric


def supported_club(frame, body, proposals, *, diagnostics=None, ambiguity_px_scale=1.):
    """Require hand-attached current shaft + independently resolved distal core."""
    if body is None:return None
    wrists=[body['keypoints'][n] for n in ('left_wrist','right_wrist')
            if n in body['keypoints'] and body['keypoints'][n]['confidence']>=.75]
    hands=np.mean([p['point'] for p in wrists],axis=0) if wrists else None
    if hands is None:return None
    bh=body['bbox'][3]-body['bbox'][1]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,45,120)
    dilated=cv2.dilate(edges,np.ones((3,3),np.uint8))
    # Restrict shafts to the selected body's reach; no global line ranking.
    hh,ww=gray.shape;rad=round(bh*1.35);hx,hy=(int(round(v)) for v in hands)
    roi=np.zeros_like(edges);roi[max(0,hy-rad):min(hh,hy+rad),max(0,hx-rad):min(ww,hx+rad)]=edges[max(0,hy-rad):min(hh,hy+rad),max(0,hx-rad):min(ww,hx+rad)]
    lines=cv2.HoughLinesP(roi,1,np.pi/360,threshold=max(15,round(bh*.08)),minLineLength=max(20,round(bh*.2)),maxLineGap=max(3,round(bh*.035)))
    if lines is None:return None
    candidates=[]
    for proposal in proposals:
        if proposal['score']<.35:continue
        box=proposal['box'];middle=np.array([(box[0]+box[2])/2,(box[1]+box[3])/2])
        if not .25*bh <= np.linalg.norm(middle-hands) <= 1.35*bh:continue
        if max(box[2]-box[0],box[3]-box[1])>bh*.18:continue
        supports=[]
        for line in lines.reshape(-1,4):
            p,q=np.array(line[:2],float),np.array(line[2:],float)
            if np.linalg.norm(q-hands)<np.linalg.norm(p-hands):p,q=q,p
            length=np.linalg.norm(q-p)
            if np.linalg.norm(p-hands)>bh*.14 or np.linalg.norm(q-middle)>max(12,bh*.08):continue
            # Observed line points must cover the route, not just its endpoints.
            samples=np.linspace(p,q,max(2,round(length))).round().astype(int)
            ratio=float(np.mean(dilated[samples[:,1],samples[:,0]]>0))
            if ratio<.8:continue
            shaft_width=shaft_pixel_width(gray,p,q)
            if shaft_width>max(3,bh*.015):continue
            core=compact_head(frame,box,q,bh,q-p,shaft_width,ambiguity_px_scale=ambiguity_px_scale,diagnostics=diagnostics)
            if core is None or anatomical_overlap(core['point'],body):continue
            supports.append((ratio*length,p.tolist(),q.tolist(),core))
        if not supports:continue
        _,p,q,core=max(supports,key=lambda item:item[0])
        candidates.append(dict(core,score=proposal['score'],shaft=[p,q],body_track_id=body['track_id'],visible_hand_count=len(wrists),
                               method='current_hand_shaft_and_dark_head_core',provenance='current_pixels',
                               proposal_box=box,model_role='research_search_proposal_only'))
    # Duplicate neural boxes cannot vote twice; spatially distinct rivals veto.
    candidates.sort(key=lambda item:item['score'],reverse=True)
    if not candidates:return None
    if any(np.linalg.norm(np.array(c['point'])-candidates[0]['point'])>max(8*ambiguity_px_scale,bh*.04) for c in candidates[1:]):
        if diagnostics is not None:diagnostics['ambiguous']=True
        return None
    return candidates[0]


def publish_people(rows, *, min_body_support=3, min_club_support=3):
    """Retrospective consecutive support, never fills an unsupported frame."""
    for key,minimum in [('body',min_body_support),('clubhead',min_club_support)]:
        run=[]
        def finish():
            if len(run)<minimum:
                for row in run:row[key]=None
            run.clear()
        for row in rows:
            value=row.get(key)
            if key=='clubhead' and row.get('body') is None:
                row[key]=value=None
            identity=value.get('track_id') if key=='body' and value else value.get('body_track_id') if value else None
            prior=run[-1].get(key) if run else None
            prior_id=prior.get('track_id') if key=='body' and prior else prior.get('body_track_id') if prior else None
            discontinuity=run and (row['frame']!=run[-1]['frame']+1 or row['body_scene']!=run[-1]['body_scene'] or identity!=prior_id)
            if key=='clubhead' and value and run:
                reference=torso_length(row['body'])
                discontinuity=discontinuity or np.linalg.norm(np.array(value['point'])-prior['point'])>reference*1.5
            if value is None or discontinuity:finish()
            if value is not None:run.append(row)
        finish()
    for row in rows:
        if row.get('body') is None:row['clubhead']=None
        for key in ('body','clubhead'):
            if row.get(key):row[key]['publication']='retrospective_min3_consecutive'


def round_bright_head_conflict(frame, point, body_height, *, diagnostics=None):
    """Withhold a possible ball shadow even when ball semantics are unresolved.

    This is negative current-pixel evidence only, never a ball observation.
    Long chrome strips and crop-boundary white backgrounds are excluded.
    """
    height,width=frame.shape[:2];x,y=np.round(point).astype(int)
    radius=max(16,math.ceil(body_height*.06*1.25)+2)
    x0,y0=max(0,x-radius),max(0,y-radius)
    patch=frame[y0:min(height,y+radius+1),x0:min(width,x+radius+1)]
    if not patch.size:return True
    hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
    mask=((hsv[:,:,1]<=90)&(hsv[:,:,2]>=170)).astype(np.uint8)
    count,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
    nearest=None;conflict=False
    for i in range(1,count):
        bx,by,bw,bh,area=stats[i];diameter=max(bw,bh)
        if bx==0 or by==0 or bx+bw==mask.shape[1] or by+bh==mask.shape[0]:continue
        if not (min(bw,bh)>=3 and diameter<=body_height*.06 and .6<=bw/bh<=1.8 and area/(bw*bh)>=.5):continue
        center=np.array([x0+centers[i,0],y0+centers[i,1]])
        distance=float(np.linalg.norm(np.array(point)-center));threshold=max(4,diameter*.75)
        observation={'point':center.tolist(),'size':[int(bw),int(bh)],'distance_native_px':distance,
            'distance_over_diameter':distance/diameter,'veto_threshold_native_px':threshold,
            'peak_value':int(hsv[:,:,2][labels==i].max()),'coordinate_space':'native_source_pixels',
            'role':'negative round-bright evidence only; not a ball observation'}
        if nearest is None or distance/diameter<nearest['distance_over_diameter']:nearest=observation
        conflict=conflict or distance<=threshold
    if diagnostics is not None:diagnostics.update(nearest=nearest,conflict=bool(conflict))
    return bool(conflict)


def native_club_detail(frame, body, native, model):
    """Search original pixels; return coordinates in the caller's analysis space.

    The current pose only bounds a search. Publication still needs a fresh
    hand-attached shaft and distal head core, with ambiguity withholding.
    """
    h,w=frame.shape[:2];nh,nw=native.shape[:2]
    fx,fy=w/nw,h/nh
    # Exact uniform mapping keeps the export's single-scale round trip exact.
    if fx>=1 or fy>=1 or not math.isclose(fx,fy,rel_tol=1e-9):return None
    current=copy.deepcopy(body)
    current['bbox']=[v/(fx if i%2==0 else fy) for i,v in enumerate(body['bbox'])]
    for value in current['keypoints'].values():
        value['point']=[value['point'][0]/fx,value['point'][1]/fy]
    wrists=[v for k,v in current['keypoints'].items() if k.endswith('wrist') and v['confidence']>=.75]
    if not wrists:return None
    hands=np.mean([v['point'] for v in wrists],axis=0)
    bh=current['bbox'][3]-current['bbox'][1];rad=round(bh*1.35)
    hx,hy=np.round(hands).astype(int)
    x0,y0=max(0,hx-rad),max(0,hy-rad)
    x1,y1=min(nw,hx+rad+1),min(nh,hy+rad+1)
    crop=native[y0:y1,x0:x1]
    if not crop.size:return None
    prediction=model.predict(crop,imgsz=1280,conf=.35,device='cpu',verbose=False)[0]
    proposals=[]
    for b in prediction.boxes:
        box=b.xyxy[0].cpu().numpy()
        if ((x0>0 and box[0]<=2) or (y0>0 and box[1]<=2)
                or (x1<nw and box[2]>=crop.shape[1]-2)
                or (y1<nh and box[3]>=crop.shape[0]-2)):continue
        proposals.append({'box':(box+[x0,y0,x0,y0]).tolist(),'score':float(b.conf.item())})
    ld,gd={},{}
    learned=supported_club(native,current,proposals,diagnostics=ld,ambiguity_px_scale=1/fx)
    geometric=shaft_search_club(native,current,diagnostics=gd,ambiguity_px_scale=1/fx)
    # Native geometric-only dark lobes can resolve shoe edges beside a shaft.
    # Require the independently pixel-supported semantic search proposal here;
    # geometry may veto it, but cannot rescue a missing learned head.
    if learned is None or ld.get('core_ambiguous') or gd.get('core_ambiguous'):return None
    head=fuse_club_candidates(learned,geometric,bh,
        learned_ambiguous=ld.get('ambiguous',False),geometric_ambiguous=gd.get('ambiguous',False),ambiguity_px_scale=1/fx)
    if head is None:return None
    bright={}
    if round_bright_head_conflict(native,head['point'],bh,diagnostics=bright):return None
    head=copy.deepcopy(head)
    head['point']=[head['point'][0]*fx,head['point'][1]*fy]
    head['shaft']=[[x*fx,y*fy] for x,y in head['shaft']]
    if 'proposal_box' in head:
        head['proposal_box']=[v*(fx if i%2==0 else fy) for i,v in enumerate(head['proposal_box'])]
    if 'support_pixels' in head:
        head['support_pixels']=[[x*fx,y*fy] for x,y in head['support_pixels']]
    if 'shaft_width_px' in head:head['shaft_width_px']*=fx
    head.update(search_resolution='native_source_pixels',
                core_area_unit='native_source_pixels_squared',
                search_scale_native_per_analysis=[1/fx,1/fy],round_bright_check=bright)
    return head


class BodyClubDetector:
    def __init__(self, pose_path, head_path):
        from ultralytics import YOLO
        for path in (pose_path,head_path):
            if not Path(path).is_file():raise RuntimeError('Local body/club research weights unavailable')
        self.pose_path,self.head_path=Path(pose_path),Path(head_path)
        self.pose=YOLO(str(pose_path));self.head=YOLO(str(head_path));self.identity=PersonIdentity()
        self.baseline_head=None

    def _people(self, frame, result, *, origin=(0,0), factors=(1,1), detail=False):
        people=[];h,w=frame.shape[:2];ox,oy=origin;fx,fy=factors
        all_boxes=[[x0*fx+ox,y0*fy+oy,x1*fx+ox,y1*fy+oy] for x0,y0,x1,y1 in result.boxes.xyxy.tolist()] if result.boxes is not None else []
        if result.keypoints is not None and result.keypoints.conf is not None:
            for index,box in enumerate(result.boxes):
                bbox=all_boxes[index];kp=result.keypoints.data[index].tolist()
                points={NAMES[i]:{'point':[x*fx+ox,y*fy+oy],'confidence':score} for i,(x,y,score) in enumerate(kp)
                        if score>=.5 and 0<x*fx+ox<w and 0<y*fy+oy<h}
                if not all(n in points for n in ('left_shoulder','right_shoulder','left_hip','right_hip')):continue
                shoulder=np.mean([points[n]['point'] for n in ('left_shoulder','right_shoulder')],axis=0)
                hip=np.mean([points[n]['point'] for n in ('left_hip','right_hip')],axis=0)
                if np.linalg.norm(shoulder-hip)<max(12,(bbox[3]-bbox[1])*.07):continue
                x0,y0=np.minimum(shoulder,hip).astype(int);x1,y1=np.maximum(shoulder,hip).astype(int)
                x0=max(0,x0-10);x1=min(w,max(x0+1,x1+10));y0=max(0,y0);y1=min(h,max(y0+1,y1))
                patch=cv2.cvtColor(frame[y0:y1,x0:x1],cv2.COLOR_BGR2HSV)
                # Saturated hue plus a neutral-value channel supports grey shirts.
                saturation=(patch[:,:,1]>=40).astype(np.uint8)*255
                hist=cv2.calcHist([patch],[0],saturation,[12],[0,180]).ravel()
                neutral=cv2.calcHist([patch],[2],255-saturation,[4],[0,256]).ravel()
                hist=np.r_[hist,neutral];hist=hist/max(1,hist.sum())
                bh=bbox[3]-bbox[1];cx=(bbox[0]+bbox[2])/2
                crowded=any(other_index!=index and other[3]-other[1]>=bh*.5
                            and abs((other[0]+other[2])/2-cx)<bh*.8
                            and min(other[3],bbox[3])-max(other[1],bbox[1])>bh*.4
                            for other_index,other in enumerate(all_boxes))
                people.append({'bbox':bbox,'keypoints':points,'confidence':float(box.conf.item()),'crowded_acquisition':crowded,
                               'appearance':hist.tolist(),'method':'current_yolo11_pose_detail' if detail else 'current_yolo11_pose','provenance':'current_pixels'})
        return people,all_boxes

    def __call__(self, frame, *, cut=False, detail_frame=None):
        self.baseline_head=None
        if cut:self.identity.reset()
        result=self.pose.predict(frame,imgsz=960,conf=.4,device='cpu',verbose=False)[0]
        people,all_boxes=self._people(frame,result);h,w=frame.shape[:2]
        # Detail crops only propose fresh poses. A previous box provides a search
        # region, never coordinates to publish. No detail pose is carried forward.
        need_detail=(self.identity.previous is None and not any(person_score(p,(w,h)) is not None for p in people)) or (self.identity.previous is not None and self.identity.previous.get('method')=='current_yolo11_pose_detail')
        if need_detail and detail_frame is not None:
            nh,nw=detail_frame.shape[:2];fx,fy=w/nw,h/nh
            if self.identity.previous is not None and not cut:
                b=self.identity.previous['bbox'];pad=(b[3]-b[1])*.6
                regions=[(max(0,int((b[0]-pad)/fx)),max(0,int((b[1]-pad)/fy)),min(nw,int((b[2]+pad)/fx)),min(nh,int((b[3]+pad)/fy)))]
            else:
                regions=[(x,y,min(nw,x+int(nw*.6)),min(nh,y+int(nh*.6))) for x,y in [(0,0),(int(nw*.4),0),(0,int(nh*.4)),(int(nw*.4),int(nh*.4))]]
            for x0,y0,x1,y1 in regions:
                crop=detail_frame[y0:y1,x0:x1]
                if not crop.size:continue
                pred=self.pose.predict(crop,imgsz=960,conf=.4,device='cpu',verbose=False)[0]
                # Withhold boundary-clipped tile people; complete current bodies
                # can be found in overlapping tiles without inventing limbs.
                if pred.boxes is not None:
                    valid=[i for i,b in enumerate(pred.boxes.xyxy.tolist()) if b[0]>2 and b[1]>2 and b[2]<crop.shape[1]-2 and b[3]<crop.shape[0]-2]
                    pred=pred[valid]
                extra,boxes=self._people(frame,pred,origin=(x0*fx,y0*fy),factors=(fx,fy),detail=True)
                for p in extra:
                    matches=[i for i,old in enumerate(people) if iou(p['bbox'],old['bbox'])>.5]
                    if not matches:people.append(p)
                    elif person_score(people[matches[0]],(w,h)) is None:people[matches[0]]=p
                for box in boxes:
                    if not any(iou(box,old)>.8 for old in all_boxes):all_boxes.append(box)
            for p in people:
                b=p['bbox'];bh=b[3]-b[1];cx=(b[0]+b[2])/2
                # One maximum-overlap box is self. A heavily overlapping second
                # person must still be a crowd veto, never another self match.
                own=max(range(len(all_boxes)),key=lambda i:iou(all_boxes[i],b)) if all_boxes else -1
                p['crowded_acquisition']=any(i!=own and other[3]-other[1]>=bh*.5 and abs((other[0]+other[2])/2-cx)<bh*.8 and min(other[3],b[3])-max(other[1],b[1])>bh*.4 for i,other in enumerate(all_boxes))
        body=self.identity.select(people,(w,h),cut=cut)
        if body is None:return None,None
        # In a rear torso-occluded grip the pose model can assign confident
        # wrists behind the golfer. Do not publish or measure those hidden hands.
        if body.get('acquisition')=='complete_rear_subject_pose' or (self.identity.previous is not None and self.identity.previous.get('acquisition')=='complete_rear_subject_pose'):
            body['acquisition']='complete_rear_subject_pose'
        kp=body['keypoints']
        if body.get('acquisition')=='complete_rear_subject_pose':
            body['withheld_keypoints']=['left_wrist','right_wrist']
            for name in body['withheld_keypoints']:kp.pop(name,None)
        return body,self._club(frame,body,detail_frame)

    def _club(self, frame, body, detail_frame):
        result=self.head.predict(frame,imgsz=1280,conf=.35,device='cpu',verbose=False)[0]
        proposals=[{'box':box.xyxy[0].tolist(),'score':float(box.conf.item())} for box in result.boxes]
        ld,gd={},{}
        learned=supported_club(frame,body,proposals,diagnostics=ld)
        geometric=shaft_search_club(frame,body,diagnostics=gd)
        self.baseline_head=fuse_club_candidates(learned,geometric,body['bbox'][3]-body['bbox'][1],
            learned_ambiguous=ld.get('ambiguous',False),geometric_ambiguous=gd.get('ambiguous',False))
        head=self.baseline_head
        baseline_withheld=(ld.get('ambiguous') or gd.get('ambiguous') or ld.get('core_ambiguous')
                           or gd.get('core_ambiguous') or learned is not None or geometric is not None)
        if head is None and detail_frame is not None and not baseline_withheld:
            head=native_club_detail(frame,body,detail_frame,self.head)
        return head
