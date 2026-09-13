#!/usr/bin/env python3
"""Research-only pixel-space analytics side-panel renderer.

Uses accepted final native_point fields from refinement results only. It never falls
back to coarse points, rejected refinements, or bridged gaps.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FPS_NUM = 30000
FPS_DEN = 1001
FPS = FPS_NUM / FPS_DEN
PANEL_WIDTH = 520
GOLF_BALL_DIAMETER_M = 0.04267


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    confidence: float | None = None
    semantic: bool = False


def _finite(v: Any, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number")
    return float(v)


def parse_native_point(entry: dict[str, Any] | None, label: str) -> Point | None:
    """Return accepted final native_point or None.

    Rejected refinements are treated as unavailable even if an internal attempt
    had a native_point. A historic inconclusive measurement can still have a
    final accepted semantic fallback, but only when the final top-level fields
    explicitly say so. Coarse detections and attempts are intentionally not
    accepted by this function.
    """
    if not isinstance(entry, dict):
        return None
    if "rejected" in entry:
        return None
    semantic = entry.get("semantic_only") is True
    if "inconclusive" in entry:
        semantic_check = entry.get("semantic_check")
        if not semantic or not isinstance(semantic_check, dict) or semantic_check.get("supported") is not True:
            return None
    elif semantic:
        return None
    raw = entry.get("native_point")
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError(f"{label}.native_point must have at least x,y")
    x = _finite(raw[0], f"{label}.native_point[0]")
    y = _finite(raw[1], f"{label}.native_point[1]")
    conf = None
    if len(raw) >= 3 and raw[2] is not None:
        conf = _finite(raw[2], f"{label}.native_point[2]")
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"{label}.native_point confidence must be in [0,1]")
    return Point(x, y, conf, semantic=semantic)


def displacement(cur: Point | None, prev: Point | None) -> float | None:
    if cur is None or prev is None:
        return None
    return math.hypot(cur.x - prev.x, cur.y - prev.y)


def speed_px_per_display_second(disp_px: float | None, fps: float = FPS) -> float | None:
    if disp_px is None:
        return None
    fps = _finite(fps, "fps")
    if fps <= 0:
        raise ValueError("fps must be > 0")
    return disp_px * fps


def scale_from_ball_diameter(ball_diameter_px: float | None) -> dict[str, Any] | None:
    if ball_diameter_px is None:
        return None
    px = _finite(ball_diameter_px, "ball_diameter_px")
    if px <= 0:
        raise ValueError("ball_diameter_px must be > 0")
    return {
        "m_per_px": GOLF_BALL_DIAMETER_M / px,
        "assumption": "at-ball-depth projected scale from assumed 42.67mm golf ball diameter; not true 3D",
    }


def projected_speed_mps(
    disp_px: float | None,
    *,
    m_per_px: float | None = None,
    action_time_scale: float | None = None,
    fps: float = FPS,
) -> float | None:
    """Return projected speed only with explicit m/px and action-time scale."""
    if disp_px is None or m_per_px is None or action_time_scale is None:
        return None
    mpp = _finite(m_per_px, "m_per_px")
    scale = _finite(action_time_scale, "action_time_scale")
    fps = _finite(fps, "fps")
    if mpp <= 0 or scale <= 0 or fps <= 0:
        raise ValueError("m_per_px, action_time_scale and fps must be > 0")
    return disp_px * mpp * fps * scale


def load_tracks(vision_json: Path) -> list[dict[str, Any]]:
    data = json.loads(vision_json.read_text())
    frames = data.get("frames")
    if not isinstance(frames, dict):
        raise ValueError("vision JSON must contain frames object")
    out = []
    for frame_key in sorted(frames, key=lambda s: int(s)):
        entry = frames[frame_key]
        if not isinstance(entry, dict):
            raise ValueError(f"frame {frame_key} must be object")
        out.append({
            "source_frame": int(frame_key),
            "ball": parse_native_point(entry.get("ball_refinement"), f"frames.{frame_key}.ball_refinement"),
            "head": parse_native_point(entry.get("clubhead_refinement"), f"frames.{frame_key}.clubhead_refinement"),
        })
    return out


def build_metrics(tracks: list[dict[str, Any]], *, m_per_px: float | None = None, action_time_scale: float | None = None) -> list[dict[str, Any]]:
    prev_ball = None
    prev_head = None
    rows = []
    previous_source = None
    for display_idx, row in enumerate(tracks, start=1):
        if previous_source is not None and row['source_frame'] != previous_source + 1:
            prev_ball = prev_head = None
        previous_source = row['source_frame']
        ball = row["ball"]
        head = row["head"]
        bd = displacement(ball, prev_ball)
        hd = displacement(head, prev_head)
        rows.append({
            "display_frame": display_idx,
            "source_frame": row["source_frame"],
            "ball": ball,
            "head": head,
            "ball_disp_px": bd,
            "head_disp_px": hd,
            "ball_speed_px_per_s": speed_px_per_display_second(bd),
            "head_speed_px_per_s": speed_px_per_display_second(hd),
            "ball_projected_mps": projected_speed_mps(bd, m_per_px=m_per_px, action_time_scale=action_time_scale),
            "head_projected_mps": projected_speed_mps(hd, m_per_px=m_per_px, action_time_scale=action_time_scale),
        })
        # No bridging nulls: only set previous when current exists; reset on gaps.
        prev_ball = ball if ball is not None else None
        prev_head = head if head is not None else None
    return rows


def _fmt(v: float | None, unit: str = "") -> str:
    if v is None:
        return "UNAVAILABLE"
    return f"{v:.2f}{unit}"


def _fmt_conf(v: float | None) -> str:
    return "UNAVAILABLE" if v is None else f"{v:.2f}"


def render_video(input_mp4: Path, vision_json: Path, output_mp4: Path, *, m_per_px: float | None = None, action_time_scale: float | None = None, ball_diameter_px: float | None = None) -> dict[str, Any]:
    import cv2
    import numpy as np

    if input_mp4.resolve() == output_mp4.resolve():
        raise ValueError("input and output paths must differ")

    scale_info = scale_from_ball_diameter(ball_diameter_px)
    if scale_info and m_per_px is None:
        m_per_px = scale_info["m_per_px"]
    metrics = build_metrics(load_tracks(vision_json), m_per_px=m_per_px, action_time_scale=action_time_scale)

    cap = cv2.VideoCapture(str(input_mp4))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {input_mp4}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if width != 1280 or height != 720:
        raise ValueError(f"expected 1280x720 source, got {width}x{height}")
    if not math.isclose(source_fps, FPS, rel_tol=0, abs_tol=0.01):
        raise ValueError(f"expected source FPS 30000/1001 ({FPS:.3f}), got {source_fps:.3f}")
    if len(metrics) != frames:
        raise ValueError(f"vision frames ({len(metrics)}) do not match video frames ({frames})")

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_mp4.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width + PANEL_WIDTH, height))
    if not writer.isOpened():
        raise RuntimeError("cannot open video writer")

    for idx in range(frames):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"decode failed at frame {idx}")
        row = metrics[idx]
        panel = np.zeros((height, PANEL_WIDTH, 3), dtype=np.uint8)
        panel[:] = (24, 24, 24)
        y = 34
        def text(s, color=(235,235,235), scale=0.58, thick=1):
            nonlocal y
            cv2.putText(panel, s, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
            y += 28
        text("RESEARCH ESTIMATE - PIXEL SPACE", (120, 220, 255), 0.62, 2)
        text("Not calibrated mph / no carry", (180, 180, 180))
        text(f"Display frame {row['display_frame']:02d} / source {row['source_frame']}")
        text(f"FPS: 30000/1001 ({FPS:.3f})")
        y += 10
        text(f"Ball accepted native_point{' (semantic)' if row['ball'] and row['ball'].semantic else ''}", (255, 230, 120), 0.56, 2)
        if row["ball"]:
            text(f"pos: ({row['ball'].x:.1f}, {row['ball'].y:.1f}) conf {_fmt_conf(row['ball'].confidence)}")
        else:
            text("pos: UNAVAILABLE")
        text(f"d/frame: {_fmt(row['ball_disp_px'], ' px')}")
        text(f"speed: {_fmt(row['ball_speed_px_per_s'], ' px/display-s')}")
        y += 10
        text(f"Clubhead accepted native_point{' (semantic)' if row['head'] and row['head'].semantic else ''}", (120, 255, 160), 0.56, 2)
        if row["head"]:
            text(f"pos: ({row['head'].x:.1f}, {row['head'].y:.1f}) conf {_fmt_conf(row['head'].confidence)}")
        else:
            text("pos: UNAVAILABLE")
        text(f"d/frame: {_fmt(row['head_disp_px'], ' px')}")
        text(f"speed: {_fmt(row['head_speed_px_per_s'], ' px/display-s')}")
        y += 10
        if m_per_px is not None and action_time_scale is not None:
            text("Projected physical approx", (255, 180, 120), 0.56, 2)
            text(f"ball: {_fmt(row['ball_projected_mps'], ' m/s')}")
            text(f"head: {_fmt(row['head_projected_mps'], ' m/s')}")
            text("user-supplied scale/time only", (180,180,180), 0.48)
            if scale_info:
                text("at-ball-depth; not true 3D", (180,180,180), 0.48)
        else:
            text("Physical speed: UNAVAILABLE", (200, 200, 200))
            text("requires explicit m/px + action-time", (170,170,170), 0.48)
        y = height - 55
        cv2.putText(panel, "apparent image motion includes jitter/camera; not physical ball motion", (18, y - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170,170,170), 1, cv2.LINE_AA)
        cv2.putText(panel, "uses final native_point only; no coarse/rejected/attempts; no gap bridging", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170,170,170), 1, cv2.LINE_AA)
        writer.write(np.concatenate([frame, panel], axis=1))
    cap.release()
    writer.release()

    # Normalize final encoding and exact rate.
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", str(tmp),
        "-r", "30000/1001", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_mp4),
    ]
    subprocess.run(cmd, check=True)
    tmp.unlink(missing_ok=True)
    return {"frames": frames, "width": width + PANEL_WIDTH, "height": height, "fps": "30000/1001", "metrics": metrics}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--vision", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--m-per-px", type=float)
    p.add_argument("--action-time-scale", type=float)
    p.add_argument("--ball-diameter-px", type=float)
    args = p.parse_args()
    info = render_video(args.input, args.vision, args.output, m_per_px=args.m_per_px, action_time_scale=args.action_time_scale, ball_diameter_px=args.ball_diameter_px)
    print(json.dumps({k: v for k, v in info.items() if k != "metrics"}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
