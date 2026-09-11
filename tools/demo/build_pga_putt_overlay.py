"""Watchable local research overlay for the official PGA TOUR Scheffler putt.

LOCAL ONLY. Public redistribution rights are UNVERIFIED: do not publish this
render or upload it anywhere. User authorization to analyse locally is recorded
separately from rights and is not a rights determination.

Nothing is fabricated. The ball marker is drawn only on frames where the tracker
actually resolved a candidate that was visually confirmed; gaps stay gaps. No
speed, distance, break or trajectory number is produced: the camera pans, so
image-space pixels per frame is NOT ball speed, and no calibration exists.
"""
from __future__ import annotations

import json, os, subprocess, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from ghostcaddie.video.pga_source_gate import evaluate_source, REJECTED

SRC = "out/pga_official_acquisition/6404323161112.mp4"
TRACK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                     "out", "pga_experiment_20260911", "scheffler_track.json")
W, H, FPS = 1280, 720, 30.0
F = cv2.FONT_HERSHEY_SIMPLEX
OK, BAD, WARN, FG, DIM = (110,205,110), (85,85,235), (70,175,245), (238,235,235), (165,160,160)


def card(lines, secs):
    img = np.full((H, W, 3), (20,18,18), np.uint8)
    y = 90
    for s, sc, col, gap in lines:
        if s: cv2.putText(img, s, (60, y), F, sc, col, 2 if sc >= 0.85 else 1, cv2.LINE_AA)
        y += gap
    return [img] * int(round(secs * FPS))


def strip(img, note, state, frame_idx, extra=""):
    cv2.rectangle(img, (0, H-64), (W, H), (18,16,16), -1)
    cv2.putText(img, note, (14, H-40), F, 0.5, DIM, 1, cv2.LINE_AA)
    cv2.putText(img, f"source frame {frame_idx}   ball: {state}   {extra}",
                (14, H-16), F, 0.52, OK if state == "resolved" else BAD, 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (W, 34), (18,16,16), -1)
    cv2.putText(img, "FairwayOS research overlay - RESEARCH ONLY - local review, "
                     "not for publication", (14, 23), F, 0.54, WARN, 1, cv2.LINE_AA)
    return img


def main():
    d = evaluate_source(SRC)
    if d.status == REJECTED or not d.local_research_allowed:
        raise SystemExit(f"source gate refuses: {d.status}")
    if d.public_redistribution_cleared:
        raise SystemExit("unexpected: this build assumes redistribution is NOT cleared")
    print(f"gate: {d.status} | local_research_allowed=True | demo_eligible={d.demo_eligible}")

    track = {int(k): v for k, v in json.load(open(os.path.normpath(TRACK))).items()}
    cap = cv2.VideoCapture(SRC)
    frames = {}
    i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        frames[i] = f
        i += 1
    cap.release()

    out = []
    out += card([
        ("FairwayOS - PGA TOUR research overlay", 1.15, FG, 62),
        ("Scottie Scheffler, 24-foot birdie putt, hole 4", 0.72, FG, 34),
        ("2026 TOUR Championship, final round (official PGA TOUR video)", 0.62, DIM, 46),
        ("", 0, FG, 6),
        ("Source acquired by the user with explicit authorization.", 0.60, FG, 30),
        ("Authorization to analyse locally is NOT a rights determination.", 0.60, WARN, 30),
        ("Public redistribution rights are UNVERIFIED. Do not publish.", 0.60, BAD, 44),
        ("", 0, FG, 6),
        ("Ball marker is drawn ONLY where the tracker actually resolved the ball.", 0.58, FG, 28),
        ("Gaps stay gaps. No speed, distance, break or trajectory is claimed:", 0.58, FG, 28),
        ("the camera pans, so image pixels per frame is not ball speed, and no", 0.58, FG, 28),
        ("calibration exists.", 0.58, FG, 40),
        ("research_only=true   ground_truth=false   production_eligible=false", 0.60, WARN, 30),
    ], 7.0)

    def render(idx, scale_note, trail):
        img = frames[idx].copy()
        t = track.get(idx)
        if t:
            x, y = int(t[0]), int(t[1])
            for n, (px, py) in enumerate(trail[-12:]):
                a = 0.25 + 0.75 * n / 12.0
                cv2.circle(img, (int(px), int(py)), 2, (int(90*a), int(200*a), int(255*a)), -1)
            cv2.circle(img, (x, y), 16, (60, 200, 255), 2)
            cv2.line(img, (x-26, y), (x-19, y), (60,200,255), 1)
            cv2.line(img, (x+19, y), (x+26, y), (60,200,255), 1)
            return strip(img, scale_note, "resolved", idx, f"image position ({t[0]:.0f}, {t[1]:.0f}) px")
        return strip(img, scale_note, "unavailable", idx, "no confirmed candidate - gap preserved")

    # context (no marker yet)
    for idx in range(60, 196):
        out.append(strip(frames[idx].copy(),
                         "Original broadcast, unmodified pixels. Overlay begins at the stroke.",
                         "not tracked yet", idx))
    # the putt, real time
    trail = []
    for idx in range(196, 250):
        if idx in track: trail.append((track[idx][0], track[idx][1]))
        out.append(render(idx, "Real time. Ball marker only where actually resolved.", trail))
    # slow motion replay
    out += card([("Replay - 1/4 speed", 1.0, FG, 56),
                 ("Same frames, each held 4x. No interpolation, no added frames.", 0.60, DIM, 34),
                 ("The track has a real gap where the ball was not resolved.", 0.60, WARN, 34)], 3.0)
    trail = []
    for idx in range(200, 242):
        if idx in track: trail.append((track[idx][0], track[idx][1]))
        f4 = render(idx, "Replay 1/4 speed - frames repeated, never interpolated.", trail)
        out += [f4] * 4

    res = sum(1 for i in range(196, 250) if i in track)
    out += card([
        ("What this shows, and what it does not", 1.05, FG, 62),
        ("", 0, FG, 6),
        (f"Ball resolved on {res} of 54 frames across the tracked window.", 0.64, OK, 34),
        ("The track ends at the cup: after that the ball is no longer resolvable.", 0.64, FG, 34),
        ("That the putt was HOLED is the broadcaster's caption, not our measurement.", 0.62, WARN, 44),
        ("", 0, FG, 6),
        ("NOT CLAIMED: ball speed, putt distance, break, green slope, trajectory,", 0.60, BAD, 28),
        ("strokes gained, or any calibrated figure. No ground-truth labels exist,", 0.60, BAD, 28),
        ("so accuracy here is unmeasured, not good.", 0.60, BAD, 40),
        ("Clubhead and impact remain unavailable and are not shown.", 0.60, DIM, 34),
        ("Local research artifact. Redistribution rights unverified - do not publish.", 0.60, WARN, 30),
    ], 7.0)

    os.makedirs("out/pga_demo_20260911", exist_ok=True)
    raw = "out/pga_demo_20260911/_raw.mp4"
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in out: vw.write(f)
    vw.release()
    final = "out/pga_demo_20260911/scheffler_putt_research_overlay.mp4"
    subprocess.run(["ffmpeg","-y","-v","error","-i",raw,"-c:v","libx264","-pix_fmt","yuv420p",
                    "-crf","20","-movflags","+faststart",final], check=True)
    os.remove(raw)
    print(json.dumps({"output": final, "frames": len(out),
                      "seconds": round(len(out)/FPS, 2),
                      "ball_resolved_frames": res}, indent=1))


if __name__ == "__main__":
    main()
