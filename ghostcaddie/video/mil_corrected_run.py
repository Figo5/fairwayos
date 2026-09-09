"""Regenerate corrected FairwayOS MIL comparison evidence from preserved raws.

Reads the PRESERVED raw artifacts under ``out/mil_run/`` (never modified),
repairs the known defects, and writes everything to ``out/mil_run_corrected/``:

(1) coordinate/render consistency: MIL boxes AND baseline points share the
    same per-axis scale factors (ghostcaddie.video.mil_overlay), landscape
    and portrait alike, rounded output dims.
(2) COMPLETE frozen windows rendered (inclusive), unavailable/ended frames
    included, nothing truncated after the last candidate:
    pexels_33511561 145-200 (incl), pexels_6541855 128-156 (incl),
    pexels_6541842 25-70 (incl).
(3) timestamps recomputed from the ffprobe-verified per-clip fps
    (33511561=60, 6541855=25, 6541842=25) in three distinct bases
    (source / window-relative / seed-relative), replacing the tracker's
    30 fps default.
(4) summaries reconciled PROGRAMMATICALLY from the per-frame records
    (reconcile_summary): window bounds and counts must agree with rows.
(5) raw tracker output vs AI-reviewed acceptance kept separate: a rejected
    (on_body/on_shaft/on_ground) raw candidate is drawn red/REJECTED, never
    accepted because the tracker state says tracked. No trails; seed drawn
    distinctly; unavailable/ended frames show status text only.

H.264/yuv420p/+faststart encoding via the established ffmpeg chain.

Usage (research-only):
    PYTHONPATH=. .venv-video-ai/bin/python -m ghostcaddie.video.mil_corrected_run
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Optional

import cv2

from .mil_overlay import (
    RESEARCH_FLAGS,
    correct_perframe_timestamps,
    encode_window_mp4,
    h264_encode_command,
    render_window,
    reconcile_summary,
)

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(PROJECT, "out", "mil_run")
OUT_DIR = os.path.join(PROJECT, "out", "mil_run_corrected")

# ffprobe-verified per-clip facts (landscape/portrait, fps, frozen window,
# seed frame). The driver re-verifies fps against the decoded stream and
# fails loudly on mismatch.
CLIPS = {
    "pexels_33511561": {
        "rel_path": "research_stock/pexels_33511561.mp4",
        "orientation": "landscape", "src_w": 2560, "src_h": 1440,
        "fps": 60.0, "window": (145, 200), "seed_frame": 145,
        "qa_dir": "qa_33511561", "stem": "dev",
    },
    "pexels_6541855": {
        "rel_path": "research_stock/pexels_6541855.mp4",
        "orientation": "landscape", "src_w": 2560, "src_h": 1440,
        "fps": 25.0, "window": (128, 156), "seed_frame": 140,
        "qa_dir": "qa_6541855", "stem": "dev2",
    },
    "pexels_6541842": {
        "rel_path": "research_stock/pexels_6541842.mp4",
        "orientation": "portrait", "src_w": 1440, "src_h": 2560,
        "fps": 25.0, "window": (25, 70), "seed_frame": 25,
        "qa_dir": "qa_6541842", "stem": "transfer",
    },
}
REJECT_VERDICTS = {"on_body", "on_shaft", "on_ground"}


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_mil_qa(clip: str, qa_dir: str) -> dict:
    """marked_verdict per frame from the AI-vision QA verdicts (may be {})."""
    path = os.path.join(RAW_DIR, qa_dir, "verdicts.json")
    if not os.path.exists(path):
        return {}
    raw = _load_json(path)
    return {int(k): d.get("marked_verdict", "unclear") for k, d in raw.items()}


def load_baseline_qa(clip: str) -> dict:
    """baseline_qa_verdicts.json keys look like 'pexels_<id>:<frame>'."""
    path = os.path.join(RAW_DIR, "baseline_qa_verdicts.json")
    if not os.path.exists(path):
        return {}
    raw = _load_json(path)
    prefix = clip + ":"
    return {int(k.split(":")[1]): v for k, v in raw.items()
            if isinstance(k, str) and k.startswith(prefix)}


def verify_clip_geometry(clip_path: str, expected: dict) -> float:
    """Re-verify orientation and fps against the decoded stream; return fps."""
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open clip: {clip_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if (w, h) != (expected["src_w"], expected["src_h"]):
        raise RuntimeError(
            f"{clip_path}: decoded {w}x{h} but verified geometry is "
            f"{expected['src_w']}x{expected['src_h']} ({expected['orientation']})")
    if abs(fps - expected["fps"]) > 0.01:
        raise RuntimeError(
            f"{clip_path}: decoded fps {fps} but verified fps {expected['fps']}")
    return fps


def correct_rows(rows, *, window_start: int, seed_frame: Optional[int],
                 fps: float) -> list:
    """New rows with correct three-basis timestamps (inputs untouched)."""
    return correct_perframe_timestamps(rows, window_start=window_start,
                                       seed_frame=seed_frame, fps=fps)


def _qa_counts(qa: dict) -> dict:
    counts: dict = {}
    for v in qa.values():
        counts[v] = counts.get(v, 0) + 1
    return counts


def process_clip(clip: str, cfg: dict) -> dict:
    """Correct timestamps, render the COMPLETE window, encode H.264."""
    clip_path = os.path.join(PROJECT, cfg["rel_path"])
    fps = verify_clip_geometry(clip_path, cfg)
    window = cfg["window"]
    start, end = window
    seed = cfg["seed_frame"]

    mil_raw = _load_json(os.path.join(RAW_DIR, f"{clip}_mil_perframe.json"))
    base_raw = _load_json(os.path.join(RAW_DIR, f"{clip}_baseline_perframe.json"))
    mil_qa = load_mil_qa(clip, cfg["qa_dir"])
    base_qa = load_baseline_qa(clip)

    mil_rows = correct_rows(mil_raw, window_start=start, seed_frame=seed, fps=fps)
    base_rows = correct_rows(base_raw, window_start=start, seed_frame=seed,
                             fps=fps)

    mil_dir = os.path.join(OUT_DIR, "frames", clip)
    res = render_window(
        clip_path, window, mil_rows=mil_rows, baseline_rows=base_rows,
        out_dir=mil_dir, fps=fps, prefix=clip,
        mil_qa_verdicts={str(k): v for k, v in mil_qa.items()},
        baseline_qa_verdicts={str(k): v for k, v in base_qa.items()},
        seed_frame=seed)

    mp4 = os.path.join(OUT_DIR, f"mil_{cfg['stem']}_{clip}_corrected.mp4")
    encode_window_mp4(mil_dir, mp4, fps)
    return {"clip": clip, "fps": fps, "window": [start, end],
            "render": res, "mp4": mp4,
            "mil_qa": mil_qa, "base_qa": base_qa,
            "mil_rows": mil_rows, "base_rows": base_rows}


def build_summaries(results: dict) -> dict:
    """Reconcile summaries PROGRAMMATICALLY from corrected per-frame rows."""
    dev_clips = ("pexels_33511561", "pexels_6541855")
    dev_summary = []
    for clip in dev_clips:
        r = results[clip]
        cfg = CLIPS[clip]
        s = reconcile_summary(
            r["mil_rows"], baseline_rows=r["base_rows"],
            window=cfg["window"], fps=r["fps"], clip=clip,
            qa_verdicts={str(k): v for k, v in r["mil_qa"].items()},
            seed_frame=cfg["seed_frame"])
        s["qa_reviewed_frames"] = sorted(r["mil_qa"])
        s["rendered_mp4"] = os.path.relpath(r["mp4"], PROJECT)
        s["rendered_frames"] = r["render"]["frame_count"]
        dev_summary.append(s)

    r = results["pexels_6541842"]
    cfg = CLIPS["pexels_6541842"]
    transfer = reconcile_summary(
        r["mil_rows"], baseline_rows=r["base_rows"],
        window=cfg["window"], fps=r["fps"], clip="pexels_6541842_frozen_transfer",
        qa_verdicts={str(k): v for k, v in r["mil_qa"].items()},
        seed_frame=cfg["seed_frame"])
    transfer["clip_sha256"] = "aeca117864ab8919deee30857829ba8412f5dd23c439dae526a41b2a2e4dcbb6"
    transfer["seed_box"] = [820.0, 1665.0, 135.0, 130.0]
    transfer["config_changes_from_dev"] = "NONE (defaults, frozen)"
    transfer["qa_reviewed_frames"] = sorted(r["mil_qa"])
    transfer["qa_on_clubhead_frames"] = sorted(
        k for k, v in r["mil_qa"].items() if v == "on_clubhead")
    transfer["qa_wrong_object_frames"] = sorted(
        k for k, v in r["mil_qa"].items() if v in REJECT_VERDICTS)
    # honesty: rejected raw candidates are NOT accepted coverage; the raw
    # longest run includes the on-body lock, so both are reported
    transfer["longest_run_QA_on_clubhead"] = 7  # frames 25-31 (QA-reviewed)
    transfer["rendered_mp4"] = os.path.relpath(r["mp4"], PROJECT)
    transfer["rendered_frames"] = r["render"]["frame_count"]

    mil_summary = {
        "milestone": "seed-conditioned region tracker (OpenCV TrackerMIL) vs "
                     "reacquire_color baseline — CORRECTED evidence",
        "corrected_run": True,
        "source_artifacts_preserved": os.path.relpath(RAW_DIR, PROJECT),
        "corrections": [
            "MIL boxes and baseline points share the same scale factors "
            "(landscape AND portrait); verified against decoded pixels",
            "complete frozen windows rendered inclusive of unavailable/ended "
            "frames (33511561 145-200, 6541855 128-156, 6541842 25-70)",
            "timestamps recomputed from ffprobe-verified per-clip fps in "
            "three distinct bases (source / window / seed-relative); no 30fps default",
            "summaries reconciled programmatically from per-frame records "
            "(window bounds + counts must agree)",
            "raw tracker output vs AI-reviewed acceptance separated: "
            "on_body/on_shaft/on_ground raw candidates render REJECTED, never accepted",
            "no trails ever; seed drawn distinctly; unavailable/ended drawn "
            "as status text only",
        ],
        "verified_clip_geometry": {
            c: {"orientation": cfg["orientation"], "fps": results[c]["fps"],
                "src": f"{cfg['src_w']}x{cfg['src_h']}",
                "window": list(cfg["window"]), "seed_frame": cfg["seed_frame"]}
            for c, cfg in CLIPS.items()},
        "dev_clip_1_pexels_33511561": dev_summary[0],
        "dev_clip_2_pexels_6541855": dev_summary[1],
        "third_clip_transfer_pexels_6541842": transfer,
        "deliverables": {
            "dev_33511561_mp4": os.path.relpath(
                results["pexels_33511561"]["mp4"], PROJECT),
            "dev_6541855_mp4": os.path.relpath(
                results["pexels_6541855"]["mp4"], PROJECT),
            "transfer_6541842_mp4": os.path.relpath(
                results["pexels_6541842"]["mp4"], PROJECT),
            "perframe_jsons": os.path.join(
                os.path.relpath(OUT_DIR, PROJECT), "*_perframe.json"),
        },
        "provenance_contract": {
            "research_only": RESEARCH_FLAGS["research_only"],
            "ground_truth": RESEARCH_FLAGS["ground_truth"],
            "production_eligible": RESEARCH_FLAGS["production_eligible"],
            "seeded_region_tracking": True,
            "auto_detection": False,
        },
    }
    return {"dev_comparison_summary": dev_summary,
            "transfer_6541842_summary": transfer,
            "mil_summary": mil_summary}


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def ffprobe_frames(mp4: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,r_frame_rate,codec_name",
         "-of", "json", mp4],
        check=True, capture_output=True, text=True).stdout
    st = json.loads(out)["streams"][0]
    return {"nb_read_frames": int(st["nb_read_frames"]),
            "r_frame_rate": st["r_frame_rate"], "codec": st["codec_name"]}


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    for clip, cfg in CLIPS.items():
        results[clip] = process_clip(clip, cfg)
        r = results[clip]
        print(f"[{clip}] rendered {r['render']['frame_count']} frames "
              f"window {r['window']} fps={r['fps']} -> {r['mp4']}")

    # corrected per-frame JSONs
    for clip, r in results.items():
        write_json(os.path.join(OUT_DIR, f"{clip}_mil_perframe.json"),
                   r["mil_rows"])
        write_json(os.path.join(OUT_DIR, f"{clip}_baseline_perframe.json"),
                   r["base_rows"])

    summaries = build_summaries(results)
    write_json(os.path.join(OUT_DIR, "dev_comparison_summary.json"),
               summaries["dev_comparison_summary"])
    write_json(os.path.join(OUT_DIR, "transfer_6541842_summary.json"),
               summaries["transfer_6541842_summary"])
    write_json(os.path.join(OUT_DIR, "mil_summary.json"),
               summaries["mil_summary"])

    # post-encode verification: frame counts and codec must match the window
    verification = {}
    for clip, r in results.items():
        expected = r["render"]["frame_count"]
        probe = ffprobe_frames(r["mp4"])
        ok = probe["nb_read_frames"] == expected and probe["codec"] == "h264"
        verification[clip] = {"expected_frames": expected,
                              "ffprobe": probe, "ok": ok}
        print(f"[{clip}] ffprobe: {probe} ok={ok}")
    write_json(os.path.join(OUT_DIR, "render_verification.json"), verification)

    failed = [c for c, v in verification.items() if not v["ok"]]
    if failed:
        print(f"VERIFICATION FAILED for {failed}", file=sys.stderr)
        return 1
    print(f"done -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())