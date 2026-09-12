"""Fresh AI-vision frame decisions for bounded local golf-video smoke runs.

This module deliberately does not read prior demo decisions, evaluation references,
or seed coordinates. It decodes requested native frames, asks one bounded Hermes
child process to inspect those clean stills, validates the returned strict JSON,
and can render only the fresh observations it received.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Iterable, Mapping, Sequence


TARGETS = ("ball", "clubhead")
WITHHELD = ["saved demo decisions", "evaluation references", "annotation seeds"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ffprobe_video(path: Path) -> dict:
    out = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,codec_name,pix_fmt",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "r_frame_rate": s.get("r_frame_rate"),
        "avg_frame_rate": s.get("avg_frame_rate"),
        "nb_frames": int(s["nb_frames"]) if str(s.get("nb_frames", "")).isdigit() else None,
        "duration": float(s["duration"]) if s.get("duration") else None,
        "codec_name": s.get("codec_name"),
        "pix_fmt": s.get("pix_fmt"),
    }


def validate_frames(frames: Sequence[int], nb_frames: int | None = None) -> list[int]:
    vals = [int(f) for f in frames]
    if not vals:
        raise ValueError("at least one frame is required")
    if vals != sorted(vals) or len(vals) != len(set(vals)):
        raise ValueError("frames must be unique and sorted native indices")
    if min(vals) < 0:
        raise ValueError("frames must be non-negative")
    if nb_frames is not None and max(vals) >= nb_frames:
        raise ValueError(f"frame {max(vals)} outside source frame count {nb_frames}")
    return vals


def decode_frames(video: Path, frames: Sequence[int], outdir: Path) -> dict[int, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}
    for frame in frames:
        p = outdir / f"native_{frame:06d}.jpg"
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-i", str(video),
            "-vf", f"select=eq(n\\,{frame})", "-vsync", "0", "-frames:v", "1", str(p),
        ], check=True, capture_output=True, text=True)
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeError(f"failed to decode source frame {frame}")
        paths[frame] = p
    return paths


def build_prompt(video: Path, source_sha256: str, meta: Mapping, frame_paths: Mapping[int, Path]) -> str:
    listing = "\n".join(f"- source_frame {f}: {p}" for f, p in frame_paths.items())
    frames = list(frame_paths)
    return f"""You are doing a bounded blind golf-video visual inspection.

Use your vision tool on each local image path below, one frame at a time. Do not use or ask for annotation seeds, previous demo decisions, reference coordinates, hidden reports, or temporal copying. If a target is not visually separable, emit null/visible false. Inspect all listed frames before answering.

Source path: {video}
Source SHA-256: {source_sha256}
Geometry: {meta.get('width')}x{meta.get('height')}
Native frames requested: {frames}
Images:
{listing}

Return ONLY one strict JSON object with this shape:
{{
  "frames": [
    {{
      "source_frame": 3058,
      "ball": {{"visible": false, "point_xy": null, "confidence": 0.0, "uncertainty": "why/null or why visible"}},
      "clubhead": {{"visible": false, "bbox_xyxy": null, "point_xy": null, "confidence": 0.0, "uncertainty": "why/null or why visible"}}
    }}
  ],
  "provenance": {{"provider": "openai-codex", "model": "gpt-5.5", "method": "Hermes child vision tool over clean decoded native frames"}}
}}

Rules: coordinates are original image pixels; ball point is center; clubhead box is head-only, not shaft or hands; confidence is 0..1; use null for unsupported coordinates; no mph/carry/impact/landing claims."""


def build_hermes_command(query_file: Path, max_turns: int = 8) -> list[str]:
    query = query_file.read_text()
    return [
        "hermes", "--provider", "openai-codex", "-m", "gpt-5.5", "--reasoning", "medium",
        "chat", "--max-turns", str(max_turns), "--query", query,
    ]


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Hermes chat prints banners and session trailers. Prefer complete JSON lines,
    # then fall back to balanced-brace objects that contain the required key.
    for line in reversed(text.splitlines()):
        s = line.strip().strip("│╭╰─ ")
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if "frames" in obj:
                return obj
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for start in reversed(starts):
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:end + 1]
                    try:
                        obj = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if "frames" in obj:
                        return obj
                    break
    raise json.JSONDecodeError("No strict JSON object with frames found", text, 0)


def _bounded_conf(v) -> float:
    try:
        x = float(v)
    except Exception:
        return 0.0
    if not math.isfinite(x):
        return 0.0
    return max(0.0, min(1.0, x))


def _point(raw, width: int, height: int):
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
        return None
    try:
        x, y = float(raw[0]), float(raw[1])
    except Exception:
        return None
    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= width and 0 <= y <= height):
        return None
    return [x, y]


def _box(raw, width: int, height: int):
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in raw]
    except Exception:
        return None
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        return None
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        return None
    return [x0, y0, x1, y1]


def normalize_frame_decision(raw: Mapping, *, source_sha256: str, width: int, height: int) -> dict:
    if len(source_sha256) != 64:
        raise ValueError("source_sha256 must be full hex sha256")
    out = {
        "source_frame": int(raw["source_frame"]),
        "source_sha256": source_sha256,
        "pseudo_label": True,
        "research_only": True,
        "ground_truth": False,
        "production_eligible": False,
    }
    for target in TARGETS:
        r = raw.get(target) or {}
        visible = bool(r.get("visible"))
        pt = _point(r.get("point_xy"), width, height)
        box = _box(r.get("bbox_xyxy"), width, height)
        if target == "ball" and pt is None:
            visible = False
        if target == "clubhead" and pt is None and box is None:
            visible = False
        out[target] = {
            "visible": visible,
            "point_xy": pt if visible else None,
            "bbox_xyxy": box if visible else None,
            "confidence": _bounded_conf(r.get("confidence")) if visible else 0.0,
            "uncertainty": str(r.get("uncertainty") or ("visible" if visible else "not visually separable")),
            "pseudo_label": True,
            "ground_truth": False,
            "research_only": True,
            "production_eligible": False,
        }
    return out


def run_child_inference(query_file: Path, outdir: Path, max_turns: int) -> tuple[dict, dict]:
    cmd = build_hermes_command(query_file, max_turns=max_turns)
    proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[2]), capture_output=True, text=True, timeout=900)
    (outdir / "hermes_stdout.log").write_text(proc.stdout or "")
    (outdir / "hermes_stderr.log").write_text(proc.stderr or "")
    routing = {"command": cmd[:7] + ["chat", "--max-turns", str(max_turns), "--query", "<prompt omitted; see prompt.md>"], "returncode": proc.returncode}
    if proc.returncode != 0:
        raise RuntimeError(f"Hermes child failed rc={proc.returncode}; stderr tail={(proc.stderr or '')[-500:]}")
    return _extract_json(proc.stdout), routing


def render_video(video: Path, decisions: Sequence[Mapping], meta: Mapping, outdir: Path) -> Path:
    import cv2

    frames = [int(d["source_frame"]) for d in decisions]
    decoded = decode_frames(video, frames, outdir / "render_frames_clean")
    annotated = outdir / "render_frames_marked"
    annotated.mkdir(parents=True, exist_ok=True)
    w, h = int(meta["width"]), int(meta["height"])
    for idx, d in enumerate(decisions):
        f = int(d["source_frame"])
        img = cv2.imread(str(decoded[f]))
        if img is None:
            raise RuntimeError(f"failed to load decoded frame {f}")
        for target, color in (("ball", (60, 220, 255)), ("clubhead", (80, 255, 120))):
            r = d[target]
            if r["visible"] and r.get("point_xy"):
                x, y = [int(round(v)) for v in r["point_xy"]]
                cv2.circle(img, (x, y), 14 if target == "ball" else 9, color, 2)
            if r["visible"] and r.get("bbox_xyxy"):
                x0, y0, x1, y1 = [int(round(v)) for v in r["bbox_xyxy"]]
                cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.rectangle(img, (0, h - 54), (w, h), (16, 14, 14), -1)
        cv2.putText(img, f"fresh GPT5.5 vision pseudo-labels | native source f{f} | no seeds/references | research only", (16, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
        p = annotated / f"seq_{idx:06d}.jpg"
        cv2.imwrite(str(p), img)
    final = outdir / "fresh_ai_vision_overlay.mp4"
    fps_exact = str(meta.get("r_frame_rate") or "30/1")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", fps_exact, "-i", str(annotated / "seq_%06d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-movflags", "+faststart", str(final)], check=True, capture_output=True, text=True)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(final), "-f", "null", "-"], check=True, capture_output=True, text=True)
    return final


def run_smoke_pipeline(video: Path, frames: Sequence[int], outdir: Path, *, render: bool = True, max_turns: int = 8) -> dict:
    video = video.resolve()
    if not video.exists() or not video.is_file():
        raise FileNotFoundError(video)
    outdir.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_file(video)
    meta = ffprobe_video(video)
    frame_list = validate_frames(frames, meta.get("nb_frames"))
    decoded = decode_frames(video, frame_list, outdir / "input_frames")
    query_file = outdir / "prompt.md"
    query_file.write_text(build_prompt(video, source_sha, meta, decoded))
    raw, routing = run_child_inference(query_file, outdir, max_turns=max_turns)
    by_frame = {int(r["source_frame"]): r for r in raw.get("frames", [])}
    if sorted(by_frame) != frame_list:
        raise RuntimeError(f"child returned frames {sorted(by_frame)}, expected {frame_list}")
    decisions = [normalize_frame_decision(by_frame[f], source_sha256=source_sha, width=meta["width"], height=meta["height"]) for f in frame_list]
    doc = {
        "source": {"path": str(video), "sha256": source_sha, **meta},
        "interval_native_frames": [frame_list[0], frame_list[-1]],
        "requested_frames": frame_list,
        "inference": {"provenance": {**(raw.get("provenance") or {}), "input_policy": {"withheld": WITHHELD, "images": {str(k): str(v) for k, v in decoded.items()}}, "routing": routing}},
        "decisions": decisions,
        "metric_speed": {"available": False, "reason": "no calibration and no capture-action time; image observations only"},
        "research_only": True,
        "pseudo_label": True,
        "ground_truth": False,
        "production_eligible": False,
    }
    result_json = outdir / "fresh_ai_vision_results.json"
    result_json.write_text(json.dumps(doc, indent=2, sort_keys=True))
    outputs = {"decisions_json": str(result_json), "prompt": str(query_file), "stdout_log": str(outdir / "hermes_stdout.log"), "stderr_log": str(outdir / "hermes_stderr.log")}
    if render:
        final = render_video(video, decisions, meta, outdir)
        outputs["video"] = str(final)
        outputs["video_sha256"] = sha256_file(final)
    return {"ok": True, "outputs": outputs, "routing": routing}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bounded local video -> fresh Hermes GPT5.5 vision result")
    ap.add_argument("video", type=Path)
    ap.add_argument("--frames", required=True, help="comma-separated native frame indices, e.g. 3058,3068,3074")
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/fairway-reusable-vision-smoke"))
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--max-turns", type=int, default=8)
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    frames = [int(x.strip()) for x in args.frames.split(",") if x.strip()]
    result = run_smoke_pipeline(args.video, frames, args.outdir, render=not args.no_render, max_turns=args.max_turns)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
