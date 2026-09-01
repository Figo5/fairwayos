#!/usr/bin/env python3
"""Render a local, research-only PT/ONNX/generic comparison overlay."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "out/research_training_gauntlet/mmu_candidate/source.mp4"
DIAG = ROOT / "out/research_training_gauntlet/ai_demo_mmu_full_fixed2/diagnostics.json"
OUT = ROOT / "out/research_model_comparison"
RAW = OUT / "comparison_raw.mp4"
VIDEO = OUT / "comparison_overlay_h264_yuv420p.mp4"
REPORT = OUT / "comparison_diagnostics.json"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    observations = json.loads(DIAG.read_text())["observations"]
    by_frame = {int(o["frame_index"]): o for o in observations}
    frames = sorted(f for f, o in by_frame.items() if o.get("ball", {}).get("state") == "observed")
    if not frames:
        raise RuntimeError("no observed local output frames")
    start, end = max(0, min(frames) - 12), max(frames) + 12
    cap = cv2.VideoCapture(str(SOURCE))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer = cv2.VideoWriter(str(RAW), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    rendered = 0
    while rendered <= end - start:
        ok, frame = cap.read()
        if not ok: break
        fi = start + rendered
        o = by_frame.get(fi, {})
        ball = o.get("ball", {})
        # PT and ONNX are explicitly unavailable: no marker geometry is drawn.
        candidates = {
            "PT": {"state": "unavailable"},
            "ONNX": {"state": "unavailable"},
            "GENERIC": {"state": "unavailable"},
        }
        if ball.get("state") == "observed" and ball.get("point"):
            candidates["GENERIC"] = {"state": "candidate", "x": ball["point"]["x"], "y": ball["point"]["y"], "confidence": ball.get("confidence", 0.0)}
        # Diagnostic bars: top=research boundary, bottom lanes by backend.
        cv2.rectangle(frame, (0, 0), (width, 6), (210, 80, 20), -1)
        lane = {"PT": (90, 90, 90), "ONNX": (210, 180, 40), "GENERIC": (190, 50, 190)}
        for i, label in enumerate(("PT", "ONNX", "GENERIC")):
            y = height - 54 + i * 16
            active = candidates[label]["state"] == "candidate"
            cv2.rectangle(frame, (0, y), (width, y + 10), lane[label] if active else (65, 65, 65), -1)
            text = f"{label}: {'CANDIDATE' if active else 'UNAVAILABLE'}"
            if active: text += f"  conf {candidates[label]['confidence']:.2f}"
            cv2.putText(frame, text, (8, y + 9), cv2.FONT_HERSHEY_SIMPLEX, .38, (255,255,255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (0, height - 6), (width, height), (30, 30, 210), -1)
        cv2.putText(frame, f"FRAME {fi+1}  TIME {fi/fps:.3f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame, "RESEARCH ONLY | NOT GOLF-BALL IDENTITY | IDENTITY UNAVAILABLE", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, .40, (255,255,255), 1, cv2.LINE_AA)
        if candidates["GENERIC"]["state"] == "candidate":
            x, y = int(round(candidates["GENERIC"]["x"])), int(round(candidates["GENERIC"]["y"]))
            # Suppressed marker style: dashed-ish square plus cross, labelled generic.
            cv2.rectangle(frame, (x-11,y-11), (x+11,y+11), lane["GENERIC"], 2)
            cv2.drawMarker(frame, (x,y), lane["GENERIC"], cv2.MARKER_CROSS, 18, 1)
            cv2.putText(frame, "GENERIC CANDIDATE", (min(max(4,x+14), width-160), max(60,y-12)), cv2.FONT_HERSHEY_SIMPLEX, .36, lane["GENERIC"], 1, cv2.LINE_AA)
        writer.write(frame); rendered += 1
    cap.release(); writer.release()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(RAW), "-vf", "scale=in_range=pc:out_range=tv,format=yuv420p", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(VIDEO)], check=True)
    report = {"schema_version":"research-model-comparison-overlay.v1", "status":"research_only", "source":"research_training_gauntlet/mmu_candidate/source.mp4", "model_output":"research_training_gauntlet/ai_demo_mmu_full_fixed2/diagnostics.json", "frames_rendered": rendered, "frame_range":[start,start+rendered-1], "backends":{"PT":{"state":"unavailable"},"ONNX":{"state":"unavailable"},"GENERIC":{"state":"local_candidate_only","source_model":"local_golf_ball"}}, "identity":"unavailable", "ground_truth":False, "production_eligible":False, "warnings":["generic candidate is not golf-ball identity","PT and ONNX outputs unavailable","no ground truth asserted"]}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"video":str(VIDEO),"report":str(REPORT),"frames":rendered,"range":[start,start+rendered-1]}))

if __name__ == "__main__": main()
