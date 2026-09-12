"""Render the ACTUAL automatic outcome: body only, ball and clubhead absent.

No seeds, no annotations, no reference coordinates. Layers that produced no
automatic detection draw nothing and say why on screen.

Reproducible: .venv/bin/python3 tools/demo/render_rory_no_seed.py
"""
import hashlib, json, os, shutil, subprocess
import cv2

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "out", "rory_no_seed")
SRC = "/tmp/fairway-three-target-sources/downloads/6083133193001.mp4"
F = cv2.FONT_HERSHEY_SIMPLEX
BODY_C = (255, 190, 40)
SK = [('sh_l','sh_r'),('sh_l','hip_l'),('sh_r','hip_r'),('hip_l','hip_r'),
      ('hip_l','kn_l'),('kn_l','ank_l'),('hip_r','kn_r'),('kn_r','ank_r')]
PTS = {'sh_l','sh_r','hip_l','hip_r','kn_l','kn_r','ank_l','ank_r'}


def main():
    d = json.load(open(os.path.join(OUT, "raw_results.json")))
    lo, hi = d["interval_native_frames"]
    body = {r["source_frame"]: r for r in d["results"]["body"]["records"]}
    cap = cv2.VideoCapture(SRC); frames, i = {}, 0
    while True:
        ok, fr = cap.read()
        if not ok or i > hi: break
        if i >= lo: frames[i] = fr
        i += 1
    cap.release()

    seq = os.path.join(OUT, "_seq"); shutil.rmtree(seq, ignore_errors=True)
    os.makedirs(seq)
    n = 0
    for f in range(lo, hi + 1):
        img = frames[f].copy(); H, W = img.shape[:2]
        b = body.get(f); ok_b = bool(b and b.get("visible_keypoint_count", 0) > 0)
        if ok_b:
            kp = {k["name"]: k for k in b["keypoints"]}
            for a, c in SK:
                A, C = kp.get(a), kp.get(c)
                if A and C and A["visible"] and C["visible"]:
                    cv2.line(img, (int(A["x"]), int(A["y"])),
                             (int(C["x"]), int(C["y"])), BODY_C, 3, cv2.LINE_AA)
            for k in b["keypoints"]:
                if k["visible"] and k["name"] in PTS:
                    cv2.circle(img, (int(k["x"]), int(k["y"])), 4, BODY_C, -1,
                               cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (W, 34), (14, 14, 14), -1)
        cv2.putText(img, "FULLY AUTOMATIC run - no seed, no annotation, no operator "
                         "click", (16, 23), F, 0.6, (235, 235, 235), 2, cv2.LINE_AA)
        cv2.rectangle(img, (0, H - 60), (W, H), (14, 14, 14), -1)
        cv2.putText(img, f"body: {'tracked' if ok_b else 'none'}", (16, H - 38), F,
                    0.55, (110, 235, 140) if ok_b else (120, 120, 235), 2, cv2.LINE_AA)
        cv2.putText(img, "clubhead: NO AUTOMATIC DETECTOR", (210, H - 38), F, 0.55,
                    (120, 120, 235), 2, cv2.LINE_AA)
        cv2.putText(img, "ball: NO AUTOMATIC DETECTOR", (610, H - 38), F, 0.55,
                    (120, 120, 235), 2, cv2.LINE_AA)
        cv2.putText(img, f"native f{f}  |  no speed or distance is derivable: this "
                         f"clip's capture rate is unknown and there is no spatial "
                         f"calibration", (16, H - 14), F, 0.45, (175, 175, 175), 1,
                    cv2.LINE_AA)
        cv2.imwrite(os.path.join(seq, f"{n:05d}.png"), img); n += 1

    final = os.path.join(OUT, "rory_automatic_actual.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "30000/1001",
                    "-i", os.path.join(seq, "%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "19", "-r", "30000/1001",
                    "-movflags", "+faststart", final], check=True)
    shutil.rmtree(seq, ignore_errors=True)
    h = hashlib.sha256()
    with open(final, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""): h.update(c)
    print(json.dumps({"video": final, "sha256": h.hexdigest(), "frames": n,
                      "body_frames": sum(1 for f in range(lo, hi+1)
                                         if body.get(f, {}).get("visible_keypoint_count", 0) > 0),
                      "clubhead_frames": 0, "ball_frames": 0}, indent=1))


if __name__ == "__main__":
    main()
