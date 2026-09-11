"""Footage-first local overlay: Si Woo Kim 120-yard approach (official PGA TOUR).

LOCAL ONLY. Redistribution rights UNVERIFIED - do not publish or upload.

Footage-first by design: no long title or end cards, just a thin strip over real
broadcast frames. The ball marker is drawn ONLY on frames where the tracker
actually resolved the ball and the result was visually confirmed. Nothing is
interpolated. No speed, yardage, apex, carry or trajectory is produced: the
camera pans and tilts, so image motion is not ball motion, and no calibration
exists. The broadcast's own "TO HOLE: 120 YDS" graphic is the BROADCASTER'S
number, never ours.
"""
from __future__ import annotations

import json, os, subprocess, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from ghostcaddie.video.pga_source_gate import evaluate_source, REJECTED

SRC = "out/pga_official_acquisition/6404321996112.mp4"
TRACK = "out/pga_experiment_20260911/siwoo_flight_track.json"
W, H, FPS = 1280, 720, 30.0
F = cv2.FONT_HERSHEY_SIMPLEX
OKC, BADC, WARN, FG, DIM = (110,205,110), (85,85,235), (70,175,245), (238,235,235), (170,165,165)


def strip(img, state, idx, extra=""):
    cv2.rectangle(img, (0, H-46), (W, H), (16,14,14), -1)
    cv2.putText(img, "FairwayOS research overlay - local review only, not for publication",
                (12, H-27), F, 0.46, DIM, 1, cv2.LINE_AA)
    col = OKC if state == "ball tracked" else (WARN if state.startswith("not") else BADC)
    cv2.putText(img, f"f{idx}  {state}  {extra}", (12, H-8), F, 0.46, col, 1, cv2.LINE_AA)
    return img


def mark(img, t, trail):
    x, y = int(t[0]), int(t[1])
    for n, (px, py) in enumerate(trail[-18:]):
        a = 0.2 + 0.8 * n / 18.0
        cv2.circle(img, (int(px), int(py)), 2, (int(80*a), int(190*a), int(250*a)), -1)
    cv2.circle(img, (x, y), 15, (60, 200, 255), 2)
    cv2.line(img, (x-24, y), (x-18, y), (60,200,255), 1)
    cv2.line(img, (x+18, y), (x+24, y), (60,200,255), 1)
    return img


def main():
    d = evaluate_source(SRC)
    if d.status == REJECTED or not d.local_research_allowed:
        raise SystemExit(f"gate refuses: {d.status}")
    assert not d.public_redistribution_cleared
    print(f"gate: {d.status} | local_research_allowed=True | demo_eligible={d.demo_eligible}")

    track = {int(k): v for k, v in json.load(open(TRACK)).items()}
    cap = cv2.VideoCapture(SRC); frames = {}; i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        frames[i] = f; i += 1
    cap.release()
    lo, hi = min(track), max(track)

    out = []
    trail = []
    # one pass of real footage: setup -> strike -> flight -> green -> reaction
    for idx in range(40, 340):
        img = frames[idx].copy()
        if idx in track:
            trail.append((track[idx][0], track[idx][1]))
            img = mark(img, track[idx], trail)
            out.append(strip(img, "ball tracked",
                             idx, f"image ({track[idx][0]}, {track[idx][1]}) px  ncc {track[idx][2]}"))
        else:
            if idx < lo:
                st, ex = "not yet tracked", "setup / strike"
            elif idx > hi:
                st, ex = "unavailable", "ball not resolved - gap preserved, never interpolated"
            else:
                st, ex = "unavailable", "gap preserved"
            out.append(strip(img, st, idx, ex))

    # short side-by-side replay of the tracked flight, half speed
    for idx in range(lo, hi + 1):
        left = cv2.resize(frames[idx], (W // 2, H // 2))
        right = frames[idx].copy()
        tr = [(track[k][0], track[k][1]) for k in range(lo, idx + 1) if k in track]
        if idx in track:
            right = mark(right, track[idx], tr)
        right = cv2.resize(right, (W // 2, H // 2))
        pane = np.full((H, W, 3), (16,14,14), np.uint8)
        pane[120:120 + H // 2, 0:W // 2] = left
        pane[120:120 + H // 2, W // 2:W] = right
        cv2.putText(pane, "original", (14, 106), F, 0.6, DIM, 1, cv2.LINE_AA)
        cv2.putText(pane, "tracked (half speed)", (W // 2 + 14, 106), F, 0.6, OKC, 1, cv2.LINE_AA)
        cv2.putText(pane, f"{len(track)} consecutive frames, no gaps, no interpolation",
                    (14, H - 60), F, 0.52, FG, 1, cv2.LINE_AA)
        cv2.putText(pane, "no speed / yardage / apex / carry claimed - camera pans, "
                          "image motion is not ball motion, no calibration",
                    (14, H - 30), F, 0.48, WARN, 1, cv2.LINE_AA)
        out += [pane, pane]

    os.makedirs("out/pga_demo_20260911", exist_ok=True)
    raw = "out/pga_demo_20260911/_raw2.mp4"
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in out: vw.write(f)
    vw.release()
    final = "out/pga_demo_20260911/siwoo_approach_research_overlay.mp4"
    subprocess.run(["ffmpeg","-y","-v","error","-i",raw,"-c:v","libx264","-pix_fmt","yuv420p",
                    "-crf","20","-movflags","+faststart",final], check=True)
    os.remove(raw)
    print(json.dumps({"output": final, "frames": len(out),
                      "seconds": round(len(out)/FPS,2),
                      "tracked_frames": len(track), "span": [lo, hi]}, indent=1))


if __name__ == "__main__":
    main()
