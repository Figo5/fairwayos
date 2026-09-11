"""Render the bounded Rory segment with all three layers, cleared on loss.

Reproducible:  .venv/bin/python3 tools/demo/render_rory_three_target.py
"""
import json, os, subprocess, sys
import cv2, numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "out", "rory_three_target")
SRC = "/tmp/fairway-three-target-sources/downloads/6083133193001.mp4"
F = cv2.FONT_HERSHEY_SIMPLEX
SK = [('sh_l','sh_r'),('sh_l','el_l'),('el_l','wr_l'),('sh_r','el_r'),('el_r','wr_r'),
      ('sh_l','hip_l'),('sh_r','hip_r'),('hip_l','hip_r'),('hip_l','kn_l'),
      ('kn_l','ank_l'),('hip_r','kn_r'),('kn_r','ank_r')]

def main():
    d = json.load(open(os.path.join(OUT, "raw_results.json")))
    lo, hi = d["interval_native_frames"]
    body = {r["source_frame"]: r for r in d["results"]["body"]["records"]}
    club = {r["source_frame"]: r for r in d["results"]["clubhead"]["records"]}
    ball = {r["source_frame"]: r for r in d["results"]["ball"]["records"]}

    cap = cv2.VideoCapture(SRC); frames = {}; i = 0
    while True:
        ok, fr = cap.read()
        if not ok or i > hi: break
        if i >= lo: frames[i] = fr
        i += 1
    cap.release()

    out = []
    for f in range(lo, hi + 1):
        img = frames[f].copy(); H, W = img.shape[:2]
        b = body.get(f)
        if b and b.get("visible_keypoint_count", 0) > 0:
            kp = {k["name"]: k for k in b["keypoints"]}
            for a, c in SK:
                A, C = kp.get(a), kp.get(c)
                if A and C and A["visible"] and C["visible"]:
                    cv2.line(img, (int(A["x"]), int(A["y"])), (int(C["x"]), int(C["y"])),
                             (255, 190, 40), 3)
            for k in b["keypoints"]:
                if k["visible"]:
                    cv2.circle(img, (int(k["x"]), int(k["y"])), 4, (255, 190, 40), -1)
        c = club.get(f)
        if c and c.get("visible"):
            mp = c.get("mask_path")
            drew = False
            if mp and os.path.exists(mp):
                m = np.load(mp)
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, (60, 255, 140), 2); drew = True
            if not drew:
                x0, y0, x1, y1 = c["bbox_xyxy"]
                cv2.rectangle(img, (x0, y0), (x1, y1), (60, 255, 140), 2)
        bl = ball.get(f)
        if bl and bl.get("visible") and bl.get("point_xy"):
            x, y = [int(v) for v in bl["point_xy"]]
            cv2.circle(img, (x, y), 13, (60, 200, 255), 2)
            cv2.line(img, (x - 20, y), (x - 15, y), (60, 200, 255), 2)
            cv2.line(img, (x + 15, y), (x + 20, y), (60, 200, 255), 2)

        # status strip: state per layer, cleared on loss (no stale markers)
        cv2.rectangle(img, (0, H - 52), (W, H), (16, 14, 14), -1)
        def lab(name, rec, key):
            if rec is None: return f"{name}: not analysed", (120, 120, 235)
            ok_ = rec.get("visible") if key == "v" else rec.get("visible_keypoint_count", 0) > 0
            return ((f"{name}: tracked", (110, 235, 140)) if ok_
                    else (f"{name}: unavailable", (120, 120, 235)))
        x = 18
        for name, rec, key in (("body", b, "kp"), ("clubhead", c, "v"), ("ball", bl, "v")):
            t, col = lab(name, rec, key)
            cv2.putText(img, t, (x, H - 18), F, 0.62, col, 2, cv2.LINE_AA)
            x += int(len(t) * 12) + 28
        cv2.putText(img, f"Rory McIlroy, official PGA TOUR  |  native frame {f}  "
                         f"t={f*1001/30000:.3f}s  |  research only, pseudo-label, "
                         f"assisted seeds, NOT ground truth", (18, 28), F, 0.56,
                    (190, 190, 190), 1, cv2.LINE_AA)
        out.append(img)

    raw = os.path.join(OUT, "_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), 30000 / 1001, (W, H))
    for im in out: vw.write(im)
    # labelled slow replay
    for im in out: vw.write(im); vw.write(im); vw.write(im)
    vw.release()
    final = os.path.join(OUT, "rory_three_target_demo.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "19", "-movflags", "+faststart",
                    final], check=True)
    os.remove(raw)
    import hashlib
    h = hashlib.sha256()
    with open(final, "rb") as fh:
        for c2 in iter(lambda: fh.read(1 << 20), b""): h.update(c2)
    print(json.dumps({"video": final, "sha256": h.hexdigest(),
                      "frames": len(out) * 4}, indent=1))

if __name__ == "__main__":
    main()
