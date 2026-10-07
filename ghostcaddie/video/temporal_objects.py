"""Image evidence for tiny sky objects and flag-free cup rims.

These are conservative appearance/temporal candidates, not a trained golf
classifier. Every emitted point is a current image component; predictions and
LK estimates only associate components and never count as observations.
"""
import cv2
import numpy as np


def sky_candidates(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sky = (hsv[:, :, 1] < 90) & (hsv[:, :, 2] > 150)
    if sky.mean() < .55:
        return []
    contrast = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, np.ones((11, 11), np.uint8))
    mask = ((contrast >= 24) & (hsv[:,:,1] < 110)).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    height, width = gray.shape
    result = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        cx, cy = centers[i]
        if not (2 <= area <= 100 and 1 <= w <= 16 and 1 <= h <= 16 and .4 <= w/h <= 2.5):
            continue
        if min(cx, width-cx) < width*.04 or min(cy, height-cy) < height*.04:
            continue
        pad = 14
        local = sky[max(0,y-pad):y+h+pad, max(0,x-pad):x+w+pad]
        if local.mean() < .96:
            continue
        result.append({'point':[float(cx),float(cy)], 'size':[int(w),int(h)],
                       'area':int(area), 'contrast':float(contrast[labels==i].mean())})
    return result


def cup_candidates(frame, people=()):
    """Flat pale candidates, including broad ambiguity blockers.

    Broad shapes cannot establish an appearance-only cup, but remain in
    temporal competition. Removing them early could falsely promote a
    narrow tee marker. The flagstick branch supplies separate evidence.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    white = cv2.inRange(hsv, np.array([0,0,170]), np.array([179,65,255]))
    # Join a fragmented rim by at most two pixels, never a large virtual cup.
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((2,3),np.uint8))
    _, _, stats, centers = cv2.connectedComponentsWithStats(white)
    height, width = gray.shape
    result = []
    for stat, center in zip(stats[1:], centers[1:]):
        x,y,w,h,area = map(int,stat);cx,cy=map(float,center)
        if not (8 <= w <= min(240,width*.19) and 1 <= h <= 35 and 2.0 <= w/h <= 35 and area >= 10):
            continue
        if cy < height*.24 or cy > height*.94 or cx < width*.06 or cx > width*.94:
            continue
        if any(min(x+w,px1+5)>max(x,px0-5) and min(y+h,py1+8)>max(y,py0-5) for px0,py0,px1,py1 in people):
            continue
        pad=max(12,min(w,60))
        patch=hsv[max(0,y-pad):min(height,y+h+pad),max(0,x-pad):min(width,x+w+pad)]
        green=cv2.inRange(patch,np.array([24,20,35]),np.array([95,230,230]))
        if (green>0).mean() < .80:
            continue
        # Long bright scratches with almost no area are not reliable rims.
        if area/(w*h) < .20 or float(gray[y:y+h,x:x+w].max()) - float(np.median(gray[max(0,y-pad):min(height,y+h+pad), max(0,x-pad):min(width,x+w+pad)])) < 35:
            continue
        result.append({'point':[cx,cy], 'size':[w,h], 'score':float(w),
                       'method':'visible_flat_cup_rim', 'publishable_cup':w <= 45})
    return result


def unique_chains(rows, key, *, min_support, radius, moving=False):
    """One unambiguous component chain per span, reset at scenes and misses."""
    chains=[];active=[];previous_scene=object()
    for row in rows:
        if row['scene'] != previous_scene or (active and row['frame'] != active[0][-1][0]['frame']+1):
            active=[]
        previous_scene=row['scene']
        proposals=row.get(key,[])
        new=[];used=set()
        for chain in active:
            last=chain[-1][1]
            distances=sorted((float(np.linalg.norm(np.array(p['point'])-last['point'])),i)
                             for i,p in enumerate(proposals))
            if not distances or distances[0][0]>radius:
                continue
            if len(distances)>1 and distances[1][0]-distances[0][0]<3:
                continue
            i=distances[0][1]
            if i in used:
                continue
            p=proposals[i]
            ratio=max(p['size'])/max(last['size'])
            recent=chain[-20:]
            median_size=float(np.median([max(q['size']) for _,q in recent]))
            aspect=(p['size'][0]/p['size'][1])/(last['size'][0]/last['size'][1])
            if not (.4 <= ratio <= 2.5 and .5 <= max(p['size'])/median_size <= 2 and .5 <= aspect <= 2):
                continue
            chain.append((row,p));used.add(i);new.append(chain)
        for i,p in enumerate(proposals):
            if i not in used:
                chain=[(row,p)];chains.append(chain);new.append(chain)
        active=new
    eligible=[]
    for chain in chains:
        if len(chain)<min_support:
            continue
        points=np.array([p['point'] for _,p in chain])
        if moving and np.linalg.norm(np.ptp(points,axis=0)) < 4:
            continue
        if moving and len(points)>2 and np.percentile(np.linalg.norm(np.diff(points,n=2,axis=0),axis=1),90)>12:
            continue
        eligible.append(chain)
    # Multiple sustained objects in the same frames cannot be identified as
    # the ball/cup by appearance alone. Fail closed for the overlap.
    ambiguous=set()
    for i,a in enumerate(eligible):
        af={r['frame'] for r,_ in a}
        for b in eligible[i+1:]:
            overlap=af & {r['frame'] for r,_ in b}
            ambiguous.update(overlap)
    # Recheck contiguous support after removing ambiguous frames.
    result=[]
    for chain in eligible:
        pieces=[];piece=[]
        for row,p in chain:
            if row['frame'] in ambiguous or (piece and row['frame'] != piece[-1][0]['frame']+1):
                if piece:pieces.append(piece)
                piece=[]
            if row['frame'] not in ambiguous:piece.append((row,p))
        if piece:pieces.append(piece)
        for part in pieces:
            if len(part)<min_support:continue
            points=np.array([p['point'] for _,p in part])
            if moving and (np.linalg.norm(np.ptp(points,axis=0))<4 or np.percentile(np.linalg.norm(np.diff(points,n=2,axis=0),axis=1),90)>12):continue
            result.append(part)
    return result


def cup_publication_runs(rows, chains, *, min_support=18):
    """Consecutive registered cup evidence, with raw broad-rival veto.

    A broad rival need not survive temporal association to remain ambiguous.
    This conservative whole-view veto can also withhold real cups. Appearance
    alone still cannot distinguish every solo pale marker from a cup.
    """
    published=[]
    for chain in chains:
        if not chain or not all(p.get('publishable_cup',False) for _,p in chain):
            continue
        first,last=chain[0][0],chain[-1][0]
        if any(not p.get('publishable_cup',False)
               for row in rows
               if row['scene']==first['scene'] and first['frame']-12 <= row['frame'] <= last['frame']+12
               for p in row.get('rim_proposals',[])):
            continue
        def usable(row,point):
            if row.get('target') is not None:
                return False  # Preserve stronger flag-supported cup evidence.
            ball=row.get('ball')
            if ball:
                radius=(max(ball['size'])+point['size'][0])/2+3
                if np.linalg.norm(np.array(ball['point'])-point['point']) <= radius:
                    return False  # A ball/blur overlapping a rim is ambiguous.
            return True
        run=[]
        def finish():
            if len(run)>=min_support:
                published.append(run[:])
            run.clear()
        for (before,p),(current,q) in zip(chain,chain[1:]):
            matrix=current.get('previous_to_current')
            valid=current['frame']==before['frame']+1 and usable(before,p) and usable(current,q)
            if matrix is None:
                valid=False
            else:
                matrix=np.asarray(matrix,float)
                valid=valid and matrix.shape==(2,3) and np.isfinite(matrix).all()
                if valid:
                    residual=np.linalg.norm(np.array(q['point'])-(matrix[:,:2] @ p['point']+matrix[:,2]))
                    valid=residual<=2.5
            if valid:
                if not run:
                    run.append((before,p))
                run.append((current,q))
            else:
                finish()
        finish()
    return published
