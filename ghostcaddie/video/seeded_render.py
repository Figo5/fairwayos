"""Bounded research-only seeded point renderer."""
from __future__ import annotations
import hashlib, json, pathlib
import cv2
from .seeded_tracking import SeedPoint, SeededPointTracker

FLAGS={"research_only":True,"ground_truth":False,"production_eligible":False}

def render_seeded_clip(video, out_dir, *, seed_frame_index, ball, clubhead, window=180):
    out=pathlib.Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); start=max(0,seed_frame_index-window); cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    frames=[]; source_indices=[]
    for i in range(min(window*2+1,total-start)):
        ok,frame=cap.read()
        if not ok: break
        frames.append(frame); source_indices.append(start+i)
    cap.release()
    if seed_frame_index not in source_indices: raise ValueError("seed frame not decoded")
    local=source_indices.index(seed_frame_index)
    tracker=SeededPointTracker(max_prediction_frames=2,max_step=80.0)
    tracks={label:tracker.track(frames,SeedPoint(local,point,label)) for label,point in (("ball",ball),("clubhead",clubhead))}
    temp=out/'intermediate.mp4'; h,w=frames[0].shape[:2]; writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    colors={"ball":(0,180,255),"clubhead":(255,100,0)}; trails={"ball":[],"clubhead":[]}
    for i,frame in enumerate(frames):
        image=frame.copy()
        for label in ("ball","clubhead"):
            item=tracks[label][i]
            if item.point is not None:
                p=tuple(round(v) for v in item.point); trails[label].append(p)
                for a,b in zip(trails[label],trails[label][1:]): cv2.line(image,a,b,colors[label],2)
                cv2.circle(image,p,6,colors[label],-1); cv2.circle(image,p,10,colors[label],2)
                text=f"{label} {item.state} c={item.confidence:.2f}"
            else: text=f"{label} {item.state}"
            cv2.putText(image,text,(8,20+22*(label=="clubhead")),cv2.FONT_HERSHEY_SIMPLEX,.45,colors[label],1,cv2.LINE_AA)
        cv2.putText(image,f"Seeded research tracking | source frame {source_indices[i]}",(8,h-30),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1,cv2.LINE_AA)
        cv2.putText(image,"POSE/analytics/impact/landing: UNAVAILABLE",(8,h-10),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,200,255),1,cv2.LINE_AA)
        writer.write(image)
    writer.release()
    frames_count=len(frames)
    diagnostics={"schema_version":"seeded-tracking.v1",**FLAGS,"coordinate_space":"pixels","seed_frame_index":seed_frame_index,"source_frame_start":start,"source_frame_count":frames_count,"pose":None,"analytics":None,"impact":None,"landing":None,"calibration":None,"tracks":{k:{"seed":list(vals[local].point),"observed":sum(x.state=="observed" for x in vals),"predicted":sum(x.state=="predicted" for x in vals),"unavailable":sum(x.state=="unavailable" for x in vals),"terminated":any(x.warning=="track_terminated" for x in vals)} for k,vals in tracks.items()}}
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2,sort_keys=True)+'\n')
    prov={"schema_version":"seeded-tracking-provenance.v1",**FLAGS,"source_file":pathlib.Path(video).name,"seed_frame_index":seed_frame_index,"seed_points":{"ball":list(ball),"clubhead":list(clubhead)},"frame_range":[start,start+frames_count-1],"tracker":"OpenCV Lucas-Kanade pyramidal optical flow","artifact":"annotated_video.mp4","sha256_intermediate":hashlib.sha256(temp.read_bytes()).hexdigest()}
    (out/'provenance.json').write_text(json.dumps(prov,indent=2,sort_keys=True)+'\n')
    return temp, diagnostics
