"""Build the watchable FairwayOS research demo (research-only, local).

Composes, per sampled frame: the ORIGINAL source frame on the left and the
ANNOTATED overlay on the right, with a state panel that reports the honest
per-layer state read from the artifact's diagnostics.json.

Every number rendered is read from the artifact. Nothing is inferred, smoothed,
interpolated or invented. Layers that are unavailable are drawn as unavailable.
Clubhead candidates that the artifact REJECTED are drawn as rejected and are
never presented as clubhead identity.

Usage:
    python tools/demo/build_fairwayos_demo.py --out out/fairwayos_demo_<date>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from ghostcaddie.video.pga_source_gate import require_demo_eligible

ROOT = "out/research_training_gauntlet"
SRC = f"{ROOT}/pexels_6573485/source.mp4"
ACCEPTED = f"{ROOT}/fairwayos_unified_pexels_6573485"           # ball unavailable 121/121
REJECTED = f"{ROOT}/fairwayos_unified_pexels_6573485_pre_ball_plausibility"

W, H = 1920, 1080
PANE_W, PANE_H = 940, 529          # each video pane
FONT = cv2.FONT_HERSHEY_SIMPLEX

# cv2 draws BGR, not RGB.
BG = (20, 18, 18)
FG = (238, 235, 235)
DIM = (158, 150, 150)
OK = (110, 205, 110)      # green
WARN = (70, 175, 245)     # amber
BAD = (85, 85, 235)       # red
RULE = (64, 58, 58)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def text(img, s, xy, scale=0.62, color=FG, thick=1):
    cv2.putText(img, s, xy, FONT, scale, color, thick, cv2.LINE_AA)


def load_source_frames(path, wanted):
    """Return {source_frame_index: BGR frame} for the indices we need."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    want, got, i = set(wanted), {}, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = fr
        i += 1
        if len(got) == len(want):
            break
    cap.release()
    missing = want - set(got)
    if missing:
        raise SystemExit(f"source frames missing: {sorted(missing)[:5]}")
    return got


def card(lines, seconds, fps):
    """A full-frame title/closing card held for `seconds`."""
    img = np.full((H, W, 3), BG, np.uint8)
    y = 150
    for s, scale, color, gap in lines:
        if s:
            text(img, s, (110, y), scale, color, 2 if scale >= 0.9 else 1)
        y += gap
    return [img] * int(round(seconds * fps))


def state_color(state):
    return {"observed": OK, "unavailable": BAD, "predicted": WARN,
            "rejected": BAD, "candidate": WARN}.get(state, DIM)


def compose(orig, over, obs, seg, meta):
    """One demo frame: original | overlay + honest state panel."""
    img = np.full((H, W, 3), BG, np.uint8)

    o = cv2.resize(orig, (PANE_W, PANE_H))
    a = cv2.resize(over, (PANE_W, PANE_H))
    img[70:70 + PANE_H, 10:10 + PANE_W] = o
    img[70:70 + PANE_H, 970:970 + PANE_W] = a

    cv2.rectangle(img, (10, 70), (10 + PANE_W, 70 + PANE_H), RULE, 1)
    cv2.rectangle(img, (970, 70), (970 + PANE_W, 70 + PANE_H), RULE, 1)
    text(img, "ORIGINAL SOURCE  (no overlay, unmodified pixels)", (14, 56), 0.62, DIM)
    text(img, seg["overlay_label"], (974, 56), 0.62, seg["overlay_color"])

    # header
    text(img, "FairwayOS research demo - RESEARCH ONLY - not ground truth, "
              "not production", (14, 30), 0.66, WARN)

    y0 = 70 + PANE_H + 42
    cv2.line(img, (10, y0 - 24), (W - 10, y0 - 24), RULE, 1)

    # frame provenance
    text(img, f"source frame {obs['frame_index']:>3d}/{meta['src_frames'] - 1}"
              f"   t={obs['timestamp_seconds']:.3f}s"
              f"   sampled every {meta['step']} source frames"
              f"   source {meta['src_fps']:.0f} fps -> render {meta['render_fps']:.0f} fps",
         (14, y0), 0.60, FG)

    # layer states
    y = y0 + 40
    for label, st, detail in seg["rows"](obs):
        text(img, label, (14, y), 0.62, DIM)
        text(img, st.upper(), (210, y), 0.62, state_color(st), 2)
        text(img, detail, (430, y), 0.56, FG)
        y += 32

    # what the viewer is looking at (fills the lower band instead of dead space)
    y = y0 + 166
    cv2.line(img, (14, y - 26), (W - 10, y - 26), RULE, 1)
    text(img, seg["heading"], (14, y), 0.64, seg["overlay_color"], 2)
    y += 32
    for ln in seg["explain"]:
        emph = ln.startswith("!")
        text(img, ln[1:].lstrip() if emph else ln, (14, y), 0.58, WARN if emph else FG)
        y += 28

    # caveat strip
    text(img, seg["caveat"], (14, H - 60), 0.58, WARN)
    text(img, "research_only=true   ground_truth=false   production_eligible=false"
              "   |  no impact, trajectory, landing, calibration or analytics claimed",
         (14, H - 26), 0.56, DIM)
    return img


def rows_accepted(obs):
    b, c, g = obs["ball"], obs["clubhead"], obs["golfer"]
    p = obs["pose"]["state"]
    yield ("pose / body", p,
           f"golfer bbox conf {g.get('confidence', 0):.3f}, "
           f"{len(g.get('keypoints', []))} keypoints")
    bd = (f"tracker: {b.get('tracker_warning') or 'n/a'}"
          if b["state"] == "unavailable"
          else f"conf {b.get('confidence', 0):.3f}")
    yield ("golf ball", b["state"], bd)
    cp = c.get("candidate_point")
    cd = f"REJECTED candidate {('(%.0f,%.0f)' % tuple(cp)) if cp else '(none)'} - {c.get('rejection')}"
    yield ("clubhead", c["state"], cd)
    yield ("impact / contact", "unavailable",
           "SwingNet bracket is candidate-only; exact contact never established")


def rows_rejected(obs):
    b, c, g = obs["ball"], obs["clubhead"], obs["golfer"]
    yield ("pose / body", obs["pose"]["state"],
           f"golfer bbox conf {g.get('confidence', 0):.3f}")
    ro = b.get("rendered_overlay") or {}
    yield ("golf ball (WITHDRAWN)", "rejected",
           f"was rendered conf {b.get('confidence', 0):.3f}, "
           f"tracer {ro.get('tracer_points', 0)} pts - followed background texture")
    yield ("clubhead", c["state"], f"{c.get('rejection')}")
    yield ("verdict", "rejected",
           "this track did NOT follow a golf ball; withdrawn 2026-09-01")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=15.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # PGA-ONLY GATE (2026-09-11 reset): fails closed. This builder was written
    # against Pexels amateur footage, which the user rejected as the product
    # demo; the gate now refuses it rather than letting it drift back in.
    require_demo_eligible(SRC)

    acc = json.load(open(f"{ACCEPTED}/diagnostics.json"))
    rej = json.load(open(f"{REJECTED}/diagnostics.json"))
    prov = json.load(open(f"{ACCEPTED}/provenance.json"))
    media = prov["media"]

    obs_a, obs_r = acc["observations"], rej["observations"]
    idx = [o["frame_index"] for o in obs_a]
    step = idx[1] - idx[0]
    meta = {"src_frames": media["frame_count"], "src_fps": media["fps"],
            "render_fps": args.fps, "step": step}

    src = load_source_frames(SRC, idx)
    src_sha = sha256(SRC)

    frames = []
    frames += card([
        ("FairwayOS - research demo", 1.25, FG, 74),
        ("What the local perception stack actually does, and what it refuses to claim.", 0.72, DIM, 62),
        ("", 0, FG, 10),
        (f"Source: Pexels 6573485 (source page marked free to use)", 0.68, FG, 38),
        (f"  {prov['source']['url']}", 0.58, DIM, 38),
        (f"  sha256 {src_sha}", 0.52, DIM, 50),
        (f"  {media['width']}x{media['height']} H.264, {media['fps']:.0f} fps, "
         f"{media['frame_count']} frames", 0.58, DIM, 60),
        ("Every value shown is read from the artifact's diagnostics.json.", 0.62, FG, 36),
        ("Nothing is interpolated, smoothed or inferred. Gaps stay gaps.", 0.62, FG, 60),
        ("research_only = true    ground_truth = false    production_eligible = false", 0.66, WARN, 40),
    ], 5.0, args.fps)

    frames += card([
        ("Part 1 of 2 - current accepted capability", 1.0, FG, 70),
        ("Artifact: fairwayos_unified_pexels_6573485", 0.62, DIM, 54),
        ("", 0, FG, 8),
        ("Pose / body ....... observed on 121 of 121 sampled frames", 0.70, OK, 44),
        ("Golf ball ......... UNAVAILABLE on 121 of 121", 0.70, BAD, 44),
        ("Clubhead .......... UNAVAILABLE on 121 of 121", 0.70, BAD, 44),
        ("                    (120 candidate points exist - all REJECTED)", 0.60, DIM, 56),
        ("", 0, FG, 8),
        ("Timing: source 30 fps, every 2nd frame sampled -> 121 samples at 15 fps.", 0.60, FG, 36),
        ("Duplicate-frame detection: NOT ASSESSED by this generator.", 0.60, WARN, 36),
        ("Blur, camera motion and occlusion: NOT ASSESSED.", 0.60, WARN, 36),
    ], 5.5, args.fps)

    seg_a = {"overlay_label": "OVERLAY  (accepted artifact rendering)",
             "overlay_color": OK, "rows": rows_accepted,
             "heading": "PART 1 - what the stack actually establishes on this clip",
             "explain": [
                 "The golfer is found and followed on every sampled frame by a generic COCO pose model.",
                 "That is body/pose tracking only. It carries no golf-specific meaning.",
                 "The ball is never localised: the tracker reports no valid candidate on all 121 frames.",
                 "Clubhead candidate pixels exist, but every one is rejected for insufficient temporal",
                 "support and low confidence, so no clubhead track is asserted at any point.",
                 "! Accuracy here is UNMEASURED, not good: no ground-truth labels exist for this clip.",
             ],
             "caveat": "Clubhead candidates are REJECTED candidates. They do not "
                       "establish clubhead identity, contact or impact."}
    for i, o in enumerate(obs_a, 1):
        frames.append(compose(src[o["frame_index"]],
                              cv2.imread(f"{ACCEPTED}/annotated_frames/frame_{i:06d}.jpg"),
                              o, seg_a, meta))

    frames += card([
        ("Part 2 of 2 - a failure we are keeping on the record", 1.0, FG, 70),
        ("Artifact: fairwayos_unified_pexels_6573485_pre_ball_plausibility", 0.60, DIM, 54),
        ("", 0, FG, 8),
        ("An earlier run reported a golf ball on 117 of 121 frames", 0.70, WARN, 44),
        ("and drew a 40-point tracer with a zoom inset. It looked convincing.", 0.70, WARN, 44),
        ("", 0, FG, 8),
        ("It was wrong. The track followed background texture - the flagstick", 0.70, BAD, 44),
        ("base - not a golf ball. It was withdrawn on 2026-09-01.", 0.70, BAD, 56),
        ("", 0, FG, 8),
        ("It is shown here, unedited, because a demo that hides its", 0.62, FG, 36),
        ("false positives is not evidence of anything.", 0.62, FG, 36),
    ], 6.0, args.fps)

    seg_r = {"overlay_label": "WITHDRAWN OVERLAY  (rejected 2026-09-01)",
             "overlay_color": BAD, "rows": rows_rejected,
             "heading": "PART 2 - the same clip, an earlier run that looked like it worked",
             "explain": [
                 "The right pane is a real earlier output: a ball marker, a growing tracer and a zoom inset,",
                 "reported on 117 of 121 frames. It is the kind of overlay a demo would normally lead with.",
                 "It is a false positive. The track sat on background texture near the flagstick base and",
                 "moved with it. No golf ball was ever localised. The plausibility gate withdrew it.",
                 "! Shown unedited, in full, because the failure is the evidence. A demo that shows only",
                 "! its successes cannot tell you which of them are real.",
             ],
             "caveat": "This ball track is FALSE. Retained as rejected evidence, "
                       "never as a detection result."}
    for i, o in enumerate(obs_r, 1):
        frames.append(compose(src[o["frame_index"]],
                              cv2.imread(f"{REJECTED}/annotated_frames/frame_{i:06d}.jpg"),
                              o, seg_r, meta))

    frames += card([
        ("What this demo does and does not show", 1.0, FG, 74),
        ("", 0, FG, 6),
        ("WORKS:  a generic COCO pose model finds and follows the golfer", 0.68, OK, 42),
        ("        reliably on this clip (121/121 sampled frames).", 0.68, OK, 50),
        ("FAILS:  golf-ball localisation. Unavailable on every frame, and the", 0.68, BAD, 42),
        ("        one track that did fire was a false positive.", 0.68, BAD, 50),
        ("FAILS:  clubhead. Unavailable on every frame; candidates rejected", 0.68, BAD, 42),
        ("        for insufficient temporal support and low confidence.", 0.68, BAD, 56),
        ("NEVER CLAIMED: impact, contact, trajectory, landing, calibration,", 0.66, WARN, 40),
        ("course coordinates, ShotEvent, analytics, recommendations, any speed", 0.66, WARN, 40),
        ("or spin figure, and any precision/recall number. No ground-truth", 0.66, WARN, 40),
        ("labels exist for this clip, so accuracy is unmeasured, not good.", 0.66, WARN, 52),
        ("Selection: both segments are the complete 121-frame runs, uncut.", 0.62, DIM, 36),
    ], 7.5, args.fps)

    raw = os.path.join(args.out, "frames_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()

    final = os.path.join(args.out, "fairwayos_research_demo.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    "-movflags", "+faststart", final], check=True)
    os.remove(raw)

    json.dump({
        "schema": "fairwayos-research-demo/v1",
        "research_only": True, "ground_truth": False, "production_eligible": False,
        "source": {"path": SRC, "sha256": src_sha, **prov["source"],
                   "rights": "Pexels source page marked free to use; local copy only"},
        "artifacts_reused": {
            "accepted_capability": ACCEPTED,
            "retained_rejected_evidence": REJECTED,
        },
        "transform": {
            "sampled_source_frames": [idx[0], idx[-1]], "sample_step": step,
            "samples": len(idx), "render_fps": args.fps,
            "duplicate_frame_detection": "not_assessed_by_this_generator",
            "slow_motion_timing": "encoded fps is not proof of capture fps",
        },
        "layer_totals": {
            "pose_observed": sum(1 for o in obs_a if o["pose"]["state"] == "observed"),
            "ball_unavailable": sum(1 for o in obs_a if o["ball"]["state"] == "unavailable"),
            "clubhead_unavailable": sum(1 for o in obs_a if o["clubhead"]["state"] == "unavailable"),
            "clubhead_rejected_candidates": sum(1 for o in obs_a if o["clubhead"].get("candidate_point")),
            "rejected_run_ball_rendered": sum(1 for o in obs_r if o["ball"]["state"] == "observed"),
        },
        "frames_written": len(frames),
        "output": final,
    }, open(os.path.join(args.out, "demo_provenance.json"), "w"), indent=1)
    print(json.dumps({"output": final, "frames": len(frames)}, indent=1))


if __name__ == "__main__":
    main()
