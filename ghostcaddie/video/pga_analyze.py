"""Single-invocation local PGA research analyzer CLI.

Usage:
    python -m ghostcaddie.video.pga_analyze INPUT --output-dir OUT \
        [--assist-json PATH] [--source-frame-start N] [--source-frame-end N] \
        [--crop x,y,w,h] [--sample-step N] [--seed-club x,y@frame]

Creates exactly: annotated_video.mp4, contact_sheet.jpg (derived from that
MP4), diagnostics.json, provenance.json, report.md.

LOCAL ONLY: no network, no cloud, no production pipeline. AI-assisted
initialization (assist JSON with crop/seed/reference) is recorded and
visibly labeled ``assisted``; it is never reported as automatic. Without an
assist JSON the run is labeled ``automatic`` with explicit
initialization-source values. If no reliable crop/seed exists, outputs
record ``unavailable`` rather than guessing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ghostcaddie.video.pga_research_analyzer import (
    DEFAULTS,
    MotionBookkeeper,
    SegmentTracker,
    TrailAccumulator,
    build_diagnostics,
    build_provenance,
    consecutive_duplicate_pairs,
    detect_motion_cuts,
    detect_split_screen,
    evaluate_against_references,
    identify_golfer,
    make_contact_sheet,
    render_frame,
    run_ball_track,
    run_club_track,
    layer_initialisation,
    automatic_tee_seed,
    static_dark_mask,
    write_video,
    RESEARCH_FLAGS,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m ghostcaddie.video.pga_analyze",
        description="Local research-only PGA swing analyzer (annotated video + honest diagnostics).",
    )
    ap.add_argument("input", help="local input video path")
    ap.add_argument("--output-dir", required=True, help="output directory (created)")
    ap.add_argument("--assist-json", default=None,
                    help="AI-assisted initialization JSON: crop, ball_tee, club_seed "
                         "{x,y,source_frame}, references (evaluation-only)")
    ap.add_argument("--source-frame-start", type=int, default=None,
                    help="first source frame to sample (default: auto scan)")
    ap.add_argument("--source-frame-end", type=int, default=None,
                    help="last source frame (inclusive) to sample")
    ap.add_argument("--crop", default=None, help="crop x,y,w,h in source pixels (overrides assist)")
    ap.add_argument("--sample-step", type=int, default=None, help="sample every Nth frame (default 2)")
    ap.add_argument("--seed-club", default=None, help="club seed x,y@source_frame (overrides assist)")
    ap.add_argument("--seed-ball-tee", default=None, help="tee ball position x,y (overrides assist)")
    ap.add_argument("--max-frames", type=int, default=200, help="cap on sampled frames (default 200)")
    ap.add_argument("--pose-model", default=None, help="path to yolo11n-pose.pt override")
    return ap.parse_args(argv)


def _load_assist(path: Optional[str]) -> dict:
    if not path:
        return {}
    with open(path) as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("assist JSON must be an object")
    return data


def _resolve_crop(assist: dict, crop_arg: Optional[str], frame_shape) -> Tuple[Optional[Dict[str, int]], str]:
    """Crop selection precedence: --crop > assist.crop > auto split-screen heuristic."""
    h, w = frame_shape[:2]
    if crop_arg:
        parts = [int(v) for v in crop_arg.split(",")]
        if len(parts) != 4:
            raise ValueError("--crop must be x,y,w,h")
        x, y, cw, ch = parts
        return {"x": x, "y": y, "w": cw, "h": ch}, "cli_crop"
    if assist.get("crop"):
        c = assist["crop"]
        return {"x": int(c["x"]), "y": int(c["y"]), "w": int(c["w"]), "h": int(c["h"])}, "assist_json"
    # automatic heuristic: detect split screen; if found use the live panel,
    # else the full frame. Recorded honestly; may be overridden by the caller.
    return None, "unresolved"


def _scan_golfer_present(crop_img: np.ndarray, pose_model, min_conf: float) -> bool:
    res = pose_model(crop_img, verbose=False, conf=max(0.3, min_conf - 0.2))[0]
    if res.boxes is None or not len(res.boxes):
        return False
    for i, cls in enumerate(res.boxes.cls.tolist()):
        if int(cls) == 0 and float(res.boxes.conf[i]) >= min_conf - 0.2:
            return True
    return False


def run(args: argparse.Namespace) -> dict:
    os.makedirs(args.output_dir, exist_ok=True)
    assist = _load_assist(args.assist_json)
    assist_used = bool(assist)

    pose_path = args.pose_model or _find_pose_model()
    if pose_path is None:
        raise SystemExit("yolo11n-pose.pt not found locally; pass --pose-model")
    from ultralytics import YOLO  # local only
    # Observed load outcome: discovery/hashing alone is never reported as loaded.
    pose_load_state, pose_load_error = "not_attempted", None
    try:
        pose_model = YOLO(pose_path)
        pose_load_state = "loaded"
    except Exception as exc:  # preserve the error state instead of crashing silently
        pose_load_state, pose_load_error = "load_failed", f"{type(exc).__name__}: {exc}"
        _write_minimal(
            args,
            {"schema": "ghostcaddie-pga-research-diagnostics/v1",
             "status": "unavailable", "reason": "pose_model_load_failed",
             "pose_model_path": pose_path, "error": pose_load_error,
             **RESEARCH_FLAGS},
            pose_model_path=pose_path,
            pose_load_state=pose_load_state,
            pose_load_error=pose_load_error,
        )
        return {"status": "unavailable", "reason": "pose_model_load_failed"}

    params = dict(DEFAULTS)
    if args.sample_step:
        params["sample_step"] = int(args.sample_step)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.input}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = args.source_frame_start if args.source_frame_start is not None else 0
    end = args.source_frame_end if args.source_frame_end is not None else total - 1
    step = params["sample_step"]
    frames: Dict[int, np.ndarray] = {}
    f = start
    while f <= end and len(frames) < args.max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if ok:
            frames[f] = fr
        f += step
    cap.release()
    if not frames:
        raise SystemExit("no frames sampled")
    source_frames = sorted(frames)
    any_frame = frames[source_frames[0]]

    # ---- crop resolution ----
    crop, crop_source = _resolve_crop(assist, args.crop, any_frame)
    split_info: Dict = {"split_screen_detected": False, "live_panel": None, "seam_x": None}
    if crop is None:
        sample_keys = source_frames[: max(3, min(8, len(source_frames)))]
        split_info = detect_split_screen([frames[k] for k in sample_keys])
        if split_info["split_screen_detected"] and split_info["live_panel"] == "left":
            crop = {"x": 0, "y": 0, "w": int(split_info["seam_x"]), "h": any_frame.shape[0]}
            crop_source = "auto_split_screen_live_panel"
        elif split_info["split_screen_detected"] and split_info["live_panel"] == "right":
            crop = {"x": int(split_info["seam_x"]), "y": 0,
                    "w": any_frame.shape[1] - int(split_info["seam_x"]), "h": any_frame.shape[0]}
            crop_source = "auto_split_screen_live_panel"
        else:
            crop = {"x": 0, "y": 0, "w": any_frame.shape[1], "h": any_frame.shape[0]}
            crop_source = "auto_full_frame"

    if crop["w"] < DEFAULTS["min_crop_width"] or crop["h"] < DEFAULTS["min_crop_height"]:
        diag_min = {
            "schema": "ghostcaddie-pga-research-diagnostics/v1",
            "status": "unavailable",
            "reason": "insufficient_resolution",
            "crop": crop,
            **RESEARCH_FLAGS,
        }
        _write_minimal(args, diag_min, mode="unavailable", pose_model_path=pose_path,
                       pose_load_state=pose_load_state, pose_load_error=pose_load_error)
        return {"status": "unavailable", "reason": "insufficient_resolution"}

    crops = [frames[f][crop["y"]:crop["y"] + crop["h"], crop["x"]:crop["x"] + crop["w"]].copy()
             for f in source_frames]

    # ---- flags on the sampled sequence ----
    dup_pairs = consecutive_duplicate_pairs(crops, source_frames, params["duplicate_diff_thr"])
    duplicate_frames = {pair[1]: pair[0] for pair in dup_pairs}
    cut_frames = detect_motion_cuts(crops, source_frames)
    seg_tracker = SegmentTracker()
    segment_ids: List[int] = []
    last_f = None
    for f in source_frames:
        if last_f is not None and f in cut_frames:
            seg_tracker.on_cut()
        segment_ids.append(seg_tracker.current())
        last_f = f

    # ---- pose over sampled frames (identity-gated) ----
    pose_rows: List[dict] = []
    for idx, f in enumerate(source_frames):
        img = crops[idx]
        res = pose_model(img, verbose=False, conf=0.05, iou=0.45)[0]
        persons = []
        if res.keypoints is not None and len(res.keypoints) > 0:
            confs = res.boxes.conf.tolist()
            for i in range(len(confs)):
                xy = res.keypoints.xy[i].cpu().numpy()
                kc = (res.keypoints.conf[i].cpu().numpy()
                      if res.keypoints.conf is not None else np.zeros(len(xy)))
                x1, y1, x2, y2 = map(float, res.boxes.xyxy[i])
                persons.append({"conf": float(confs[i]), "bbox": (x1, y1, x2, y2),
                                "kps": xy, "kpc": kc})
        persons.sort(key=lambda p: -p["conf"])
        gated, reason = identify_golfer(persons, crop["w"],
                                        params["pose_person_min_conf"],
                                        params["pose_bbox_center_corridor"])
        if gated is not None:
            pose_rows.append({
                "person": gated,
                "identity_gate": "single_person_crop",
                "gate_reason": reason,
                "bbox_center_x": (gated["bbox"][0] + gated["bbox"][2]) / 2.0,
                "keypoints": [[float(a), float(b)] for a, b in gated["kps"]],
                "keypoint_conf": [float(v) for v in gated["kpc"]],
            })
        else:
            pose_rows.append({"person": None, "identity_gate": "failed:" + reason,
                              "gate_reason": reason, "bbox_center_x": None,
                              "keypoints": None, "keypoint_conf": None})

    # ---- initialization (tee / club seed) ----
    ball_tee = _resolve_tee(args.seed_ball_tee, assist, crops, source_frames, pose_rows)
    club_seed = _resolve_club_seed(args.seed_club, assist, source_frames)
    modes = {
        "pose": "automatic",
        "ball": "automatic" if (ball_tee and not assist_used and not args.seed_ball_tee) else
                ("assisted" if ball_tee else "unavailable"),
        "clubhead": "automatic" if (club_seed and not assist_used and not args.seed_club) else
                    ("assisted" if club_seed else "unavailable"),
        "crop": crop_source if not (assist_used or args.crop) else
                ("assisted" if crop_source in ("assist_json", "cli_crop") else crop_source),
    }

    layers = layer_initialisation(
        ball_tee, club_seed,
        ball_tee_source=("cli" if args.seed_ball_tee else
                         "assisted" if assist.get("ball_tee") else
                         "automatic_unconfirmed"),
        club_seed_source=("cli" if args.seed_club else
                          "assisted" if assist.get("club_seed") else
                          "automatic_unconfirmed"),
    )
    # Layers degrade INDEPENDENTLY. A missing clubhead seed must not discard
    # pose and ball as well -- clubhead is falsified at this resolution, so
    # gating everything behind it means PGA footage yields nothing at all.
    if not any(v["can_run"] for v in layers.values()):
        diag = {
            "schema": "ghostcaddie-pga-research-diagnostics/v1",
            "status": "unavailable",
            "reason": "no layer could initialise: " + "; ".join(
                v["reason"] for v in layers.values() if v["reason"]),
            "layers": layers,
            "modes": modes,
            "crop": crop,
            "crop_source": crop_source,
            "flags": {"split_screen": split_info},
            **RESEARCH_FLAGS,
        }
        _write_minimal(args, diag, mode="unavailable", pose_model_path=pose_path,
                       pose_load_state=pose_load_state, pose_load_error=pose_load_error)
        return {"status": "unavailable", "reason": diag["reason"]}

    # ---- trackers ----
    book_ball = MotionBookkeeper(expected_step=step)
    book_club = MotionBookkeeper(expected_step=step)
    def _all_unavailable(book, reason):
        """A layer that could not initialise still reports, frame by frame."""
        rows = []
        for f in source_frames:
            obs = book.observe(f, None, None, 0.0, "unavailable", reason)
            rows.append({"source_frame": f, "state": "unavailable", "x": None,
                         "y": None, "confidence": 0.0, "source": reason,
                         "segment_id": 1, "_book": obs})
        return rows

    if layers["ball"]["can_run"]:
        ball_rows, ball_meta = run_ball_track(crops, source_frames, book_ball,
                                              duplicate_frames, ball_tee, params,
                                              segment_ids)
    else:
        ball_rows, ball_meta = _all_unavailable(book_ball, "seed_unavailable"), {}

    if layers["clubhead"]["can_run"]:
        static_dark = static_dark_mask(crops)
        club_rows = run_club_track(crops, source_frames, book_club, duplicate_frames,
                                   static_dark, (club_seed["x"], club_seed["y"]),
                                   int(club_seed["source_frame"]), pose_rows, params,
                                   segment_ids)
    else:
        club_rows = _all_unavailable(book_club, "seed_unavailable")

    # ---- flag results / reference agreement (evaluation-only) ----
    flag_results: Dict[str, object] = {
        "split_screen": split_info,
        "camera_cuts": {"cut_source_frames": cut_frames,
                        "segments_total": seg_tracker.cuts + 1},
        "duplicate_frames": {"pairs": [list(p) for p in dup_pairs],
                             "count": len(dup_pairs),
                             "excluded_from_success_counts": True},
        "target_disappearance": {
            "ball": {"latched": ball_meta.get("latched", False)},
            "clubhead": {"tracker_dead": any(
                r["source"] == "tracker_dead_no_reinit" for r in club_rows)},
        },
        "segments": {"identity": "per-segment golfer identity; resets across cuts",
                     "segments_total": seg_tracker.cuts + 1},
    }
    slow_note = {
        "note": "sampled cadence timing may not be real-time; source may be slow motion; "
                "no real-time speed claims; px/frame only",
        "status": "flagged",
    }
    flag_results["slow_motion"] = slow_note

    refs = assist.get("references", {}) if assist else {}
    ball_agreement, ball_wrong = ([], [])
    club_agreement, club_wrong = ([], [])
    if refs:
        if refs.get("ball"):
            ball_agreement, ball_wrong = evaluate_against_references(book_ball, refs["ball"])
        if refs.get("clubhead"):
            club_agreement, club_wrong = evaluate_against_references(book_club, refs["clubhead"])
    flag_results["ball_wrong_object"] = ball_wrong
    flag_results["club_wrong_object"] = club_wrong

    render_fps = src_fps / step
    diag = build_diagnostics(
        source_frames, src_fps, step, render_fps, book_ball, book_club,
        flag_results, crop, f"left-annotated view; right panel: honest states + clean inset",
        extra={
            "status": "complete",
            "modes": modes,
            "crop_source": crop_source,
            "assist_json": os.path.relpath(args.assist_json) if args.assist_json else None,
            "initialization": {
                "ball_tee_xy": list(ball_tee) if ball_tee else None,
                "ball_tee_source": layers["ball"]["seed_source"],
                "club_seed_xy": ([club_seed["x"], club_seed["y"]]
                                 if club_seed else None),
                "club_seed_source_frame": (int(club_seed["source_frame"])
                                           if club_seed else None),
                "club_seed_source": layers["clubhead"]["seed_source"],
                "layers": layers,
            },
            "ball_track_meta": ball_meta,
            "reference_agreement": {
                "evidence_class": "ai_assisted_native_inspection (never ground truth)",
                "ball": ball_agreement,
                "clubhead": club_agreement,
            },
            "parameters_frozen": True,
            "per_clip_threshold_adjustments": "none (same frozen parameters for every clip)",
        },
    )
    with open(os.path.join(args.output_dir, "diagnostics.json"), "w") as fh:
        json.dump(diag, fh, indent=1)

    # ---- render annotated video ----
    ball_trail_obj = TrailAccumulator()
    club_trail_obj = TrailAccumulator()
    out_frames = []
    ball_rows_by_frame = {r["source_frame"]: r for r in ball_rows}
    club_rows_by_frame = {r["source_frame"]: r for r in club_rows}
    for idx, f in enumerate(source_frames):
        rb = ball_rows_by_frame[f]
        rc = club_rows_by_frame[f]
        if rb["state"] == "detected":
            ball_trail_obj.append(f, rb["x"], rb["y"])
        else:
            ball_trail_obj.break_gap()
        if rc["state"] == "detected":
            club_trail_obj.append(f, rc["x"], rc["y"])
        else:
            club_trail_obj.break_gap()
        canvas, _ = render_frame(
            frames[f], crop, rb, rc, pose_rows[idx],
            ball_trail_obj.points(), club_trail_obj.points(),
            f, mode_label="research", segment_id=segment_ids[idx],
            scale_note="no calibration - no physical units",
        )
        out_frames.append(canvas)
    video_path = os.path.join(args.output_dir, "annotated_video.mp4")
    write_video(out_frames, render_fps, video_path)
    make_contact_sheet(video_path, os.path.join(args.output_dir, "contact_sheet.jpg"))

    # ---- provenance ----
    import hashlib
    sha = hashlib.sha256()
    with open(args.input, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    prov = build_provenance(
        args.input, os.path.relpath(args.output_dir), modes,
        assist_json=args.assist_json,
        pose_model_path=pose_path,
        pose_load_state=pose_load_state,
        pose_load_error=pose_load_error,
        extra={
            "source_sha256": sha.hexdigest(),
            "source_frames_total": src_w,
            "sampled_frames": len(source_frames),
            "crop": crop,
            "layers": layers,
            "render_fps": render_fps,
            "outputs": ["annotated_video.mp4", "contact_sheet.jpg", "diagnostics.json",
                        "provenance.json", "report.md"],
        },
    )
    with open(os.path.join(args.output_dir, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=1)

    # ---- report ----
    report = _build_report(diag, prov, args, ball_rows, club_rows)
    with open(os.path.join(args.output_dir, "report.md"), "w") as fh:
        fh.write(report)
    return {"status": "complete", "video": video_path, "diagnostics": diag}


def _find_pose_model() -> Optional[str]:
    cands = [
        "out/research_training_gauntlet/yolo11n-pose.pt",
        "yolo11n-pose.pt",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "yolo11n-pose.pt"),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def _resolve_tee(seed_arg, assist, crops, source_frames, pose_rows):
    if seed_arg:
        x, y = [float(v) for v in seed_arg.split(",")]
        return (x, y)
    if assist.get("ball_tee"):
        return (float(assist["ball_tee"][0]), float(assist["ball_tee"][1]))
    # automatic: deterministic persistence locator first (bounded lower ROI).
    # A ball at rest is round AND stays put across frames; a single-frame bright
    # blob scan cannot tell a ball from a shoe highlight or a cloud.
    cand = automatic_tee_seed(crops[: min(12, len(crops))], roi_fraction=0.45,
                              ball_color="white")
    if cand is not None:
        return (float(cand[0]), float(cand[1]))
    # fallback: single-frame bright round blob
    for img in crops[: min(12, len(crops))]:
        from ghostcaddie.video.pga_research_analyzer import bright_blobs
        for b in bright_blobs(img, DEFAULTS["ball_tee_v_thr"], DEFAULTS["ball_tee_s_thr"]):
            if (25 <= b["area"] <= 400 and 6 <= b["w"] <= 26 and 6 <= b["h"] <= 26):
                return (b["x"], b["y"])
    return None


def _resolve_club_seed(seed_arg, assist, source_frames):
    if seed_arg:
        body = seed_arg.split("@")
        x, y = [float(v) for v in body[0].split(",")]
        return {"x": x, "y": y, "source_frame": int(body[1]) if len(body) > 1 else source_frames[0]}
    if assist.get("club_seed"):
        cs = assist["club_seed"]
        return {"x": float(cs["x"]), "y": float(cs["y"]),
                "source_frame": int(cs.get("source_frame", source_frames[0]))}
    return None  # conservative: no guessing for club seed in automatic mode


def _write_minimal(args, diag, mode="unavailable", pose_model_path=None,
                   pose_load_state="not_attempted", pose_load_error=None):
    with open(os.path.join(args.output_dir, "diagnostics.json"), "w") as fh:
        json.dump(diag, fh, indent=1)
    prov = build_provenance(args.input, os.path.relpath(args.output_dir),
                            modes={"pose": "unavailable", "ball": "unavailable",
                                   "clubhead": "unavailable", "crop": "unresolved"},
                            assist_json=args.assist_json,
                            pose_model_path=pose_model_path,
                            pose_load_state=pose_load_state,
                            pose_load_error=pose_load_error,
                            extra={"status": "unavailable", "reason": diag.get("reason")})
    with open(os.path.join(args.output_dir, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=1)
    with open(os.path.join(args.output_dir, "report.md"), "w") as fh:
        fh.write(_build_report(diag, prov, args, [], [], unavailable=True))


def _render_model_route(prov) -> List[str]:
    """Render the model route from the provenance payload (no second source)."""
    route = prov.get("model_route")
    if not isinstance(route, dict):
        return [f"- {route}"]
    rt = route.get("runtime", {})
    libs = route.get("libraries", {})
    out = [
        f"- Runtime: python {rt.get('python')} on {rt.get('platform')}",
        "- Libraries actually imported: "
        + (", ".join(f"{k} {v}" for k, v in sorted(libs.items())) or "none recorded"),
    ]
    for m in route.get("local_models", []):
        role, state = m.get("role"), m.get("state")
        if not m.get("discovered"):
            out.append(f"- Local model ({role}): unavailable "
                       f"(no model file found; no hash)")
            continue
        line = (f"- Local model ({role}): `{m.get('path')}` "
                f"sha256 `{m.get('sha256')}` [{state}]")
        if state == "load_failed":
            line += f" - load attempted and FAILED: {m.get('load_error')}"
        elif state == "not_attempted":
            line += " - discovered and hashed only; no load was attempted"
        out.append(line)
    remote = route.get("remote_models") or []
    out.append("- Remote models: "
               + (", ".join(str(r) for r in remote) if remote else
                  "none; this artifact was produced locally with no network calls"))
    out.append(f"- {route.get('note')}")
    return out


def _build_report(diag, prov, args, ball_rows, club_rows, unavailable=False) -> str:
    n = diag.get("frames_total", 0)
    b = diag.get("ball", {})
    c = diag.get("clubhead", {})
    flags = diag.get("flags", {})
    lines = [
        "# PGA Research Analyzer Report",
        "",
        f"- Source: `{prov.get('source_path')}` (sha256 `{prov.get('source_sha256', 'n/a')}`)",
        f"- Output dir: `{prov.get('output_dir')}`",
        f"- Modes: {prov.get('modes')}",
        f"- Assist JSON: `{prov.get('assist_json') or 'none'}`",
        f"- Flags: research_only=true, ground_truth=false, production_eligible=false",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"python -m ghostcaddie.video.pga_analyze {args.input} --output-dir {args.output_dir}"
        + (f" --assist-json {args.assist_json}" if args.assist_json else "")
        + (f" --source-frame-start {args.source_frame_start}" if args.source_frame_start is not None else "")
        + (f" --source-frame-end {args.source_frame_end}" if args.source_frame_end is not None else "")
        + (f" --crop {args.crop}" if args.crop else "")
        + (f" --seed-club {args.seed_club}" if args.seed_club else "")
        + (f" --seed-ball-tee {args.seed_ball_tee}" if args.seed_ball_tee else "")
        + (f" --sample-step {args.sample_step}" if args.sample_step else ""),
        "```",
        "",
        "## Coverage",
        "",
    ]
    if unavailable:
        lines += [
            f"- Status: **unavailable** — {diag.get('reason')}",
            "- No tracking claims. Initialization did not meet the reliability bar; "
            "emit-unavailable-instead-of-guessing policy applied.",
            "",
        ]
    else:
        lines += [
            f"- Sampled frames: {n} (source f{diag.get('source_frames', [0, 0])[0]}"
            f"–f{diag.get('source_frames', [0, 0])[-1]}, step {diag.get('sample_step')}, "
            f"render {diag.get('render_fps')} fps)",
            f"- Ball: detected {b.get('coverage_detected', 0)}, duplicates excluded "
            f"{b.get('duplicates_excluded', 0)}, longest consecutive run {b.get('longest_consecutive_run_frames')} frames",
            f"- Clubhead: detected {c.get('coverage_detected', 0)}, duplicates excluded "
            f"{c.get('duplicates_excluded', 0)}, longest consecutive run {c.get('longest_consecutive_run')} frames",
            f"- Duplicate pairs: {diag.get('duplicate_pairs')}",
            "",
            "## Flags",
            "",
            f"- Split screen / frozen panel: {flags.get('split_screen')}",
            f"- Camera cuts: {flags.get('camera_cuts')}",
            f"- Slow-motion timing uncertainty: {diag.get('slow_motion_timing_uncertainty', {}).get('status')}",
            f"- Target disappearance: {flags.get('target_disappearance')}",
            "",
            "## Wrong-object matches",
            "",
            f"- Ball: {len(b.get('wrong_object_matches', []))} "
            f"({json.dumps(b.get('wrong_object_matches', []))[:400]})",
            f"- Clubhead: {len(c.get('wrong_object_matches', []))} "
            f"({json.dumps(c.get('wrong_object_matches', []))[:400]})",
            "",
            "## Limitations",
            "",
            "- Research-only demonstration; references (if used) are AI-assisted native reads, "
            "never ground truth.",
            "- Physical units (mph/RPM) unavailable: pixel space only, slow-motion timing uncertainty.",
            "- No interpolation: gaps stay gaps; trails clear on gaps.",
            "- Same frozen parameters used for every clip; no per-clip threshold tuning.",
            "",
        ]
    lines += [
        "## Honest states",
        "",
        "- Ball states: " + json.dumps(diag.get("ball", {}).get("counts", {})),
        "- Clubhead states: " + json.dumps(diag.get("clubhead", {}).get("counts", {})),
        "",
        "## Model route",
        "",
    ]
    lines += _render_model_route(prov)
    lines += [""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    result = run(args)
    print(json.dumps({"status": result.get("status"),
                      "output_dir": args.output_dir,
                      "reason": result.get("reason")}, indent=1))


if __name__ == "__main__":
    main()