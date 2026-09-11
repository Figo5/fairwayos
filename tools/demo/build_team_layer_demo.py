"""Integrate completed team layer handoffs into a local Si Woo demo.

Fails closed unless body, clubhead, and ball READY handoffs satisfy the strict
research-only contract. If any layer is missing, writes a checkpoint report with
a rerunnable command and does not fabricate a video.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ghostcaddie.video.team_layer_integration import LayerContractError, load_completed_layers, render_timeline_states

SOURCE_SHA256 = "cefbdf25400f5821893747e2b4a60ca5a11f990920ab3a9c8bba32a8c2d3deae"
DEFAULT_SOURCE = Path("out/pga_official_acquisition/6404321996112.mp4")
DEFAULT_LAYER_ROOT = Path("/tmp/fairway-team")
DEFAULT_OUTDIR = Path("out/pga_demo_20260911")
EXPECTED_LAYERS = ("body", "clubhead", "ball")
# Clean native review f210-f240 found no hard cut at f216 or f224; layer loss is
# handled by each observation state instead of a synthetic cut reset.
CUTS: Tuple[int, ...] = ()
FRAME_START = 40
FRAME_END_EXCLUSIVE = 340
FPS = 30000 / 1001
W, H = 1280, 720
COLORS = {"body": (255, 165, 60), "clubhead": (80, 255, 80), "ball": (0, 220, 255)}
DIM = (175, 175, 175)
FG = (238, 235, 235)
PANEL_BG = (16, 14, 14)
WARN = (70, 175, 245)


def _cv2():
    import cv2
    return cv2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ffprobe(path: Path) -> dict:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,codec_name,pix_fmt",
        "-of", "json", str(path),
    ], text=True)
    return json.loads(out)["streams"][0]


def load_source_frames(path: Path, frame_range: Iterable[int]) -> Dict[int, object]:
    cv2 = _cv2()
    wanted = set(int(i) for i in frame_range)
    cap = cv2.VideoCapture(str(path))
    frames = {}
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frames[i] = frame
        i += 1
    cap.release()
    missing = sorted(wanted - set(frames))
    if missing:
        raise RuntimeError(f"source decode missing requested frames: {missing[:5]}")
    return frames


def observation_map(layers: Mapping[str, Mapping]) -> Dict[str, list]:
    return {name: list(layer["observations"]) for name, layer in layers.items()}


def draw_text(img, text: str, org: Tuple[int, int], scale=0.44, color=FG, thick=1) -> None:
    cv2 = _cv2()
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_geometry(img, layer: str, state: Mapping, histories: Mapping[str, List[dict]]) -> None:
    cv2 = _cv2()
    if not state.get("visible"):
        return
    geom = state.get("geometry")
    if not geom:
        return
    color = COLORS[layer]
    for prev in histories.get(layer, [])[-18:]:
        pg = prev.get("geometry") if isinstance(prev, Mapping) else None
        if isinstance(pg, Mapping) and pg.get("type") == "point":
            cv2.circle(img, (int(round(pg["x"])), int(round(pg["y"]))), 2, color, -1)
    if geom.get("type") == "point":
        x, y = int(round(geom["x"])), int(round(geom["y"]))
        radius = 12 if layer == "ball" else 9
        cv2.circle(img, (x, y), radius, color, 2)
        cv2.line(img, (x - radius - 7, y), (x - radius - 2, y), color, 1)
        cv2.line(img, (x + radius + 2, y), (x + radius + 7, y), color, 1)
        draw_text(img, f"{layer.upper()} {state.get('confidence') if state.get('confidence') is not None else 'n/a'}", (x + 14, max(20, y - 12)), 0.42, color)
    elif geom.get("type") in {"box", "box_keypoints"}:
        if geom.get("type") == "box_keypoints":
            box = geom["box"]
        else:
            box = geom
        x1, y1, x2, y2 = [int(round(box[k])) for k in ("x1", "y1", "x2", "y2")]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if geom.get("type") == "box_keypoints":
            for point in geom.get("keypoints", []):
                px, py = int(round(point["x"])), int(round(point["y"]))
                if x1 - 20 <= px <= x2 + 20 and y1 - 20 <= py <= y2 + 20:
                    cv2.circle(img, (px, py), 3, color, -1)
        draw_text(img, f"{layer.upper()} {state.get('confidence') if state.get('confidence') is not None else 'n/a'}", (x1, max(18, y1 - 8)), 0.45, color)
    elif geom.get("type") == "keypoints":
        for p in geom.get("points", []):
            cv2.circle(img, (int(round(p["x"])), int(round(p["y"]))), 3, color, -1)
        draw_text(img, layer.upper(), (16, 210 + 20 * list(COLORS).index(layer)), 0.45, color)


def render_frame(source, frame_index: int, states: Mapping[str, Mapping], histories: Mapping[str, List[dict]]):
    cv2 = _cv2()
    img = source.copy()
    for layer in EXPECTED_LAYERS:
        draw_geometry(img, layer, states[layer], histories)
    inset = cv2.resize(source, (256, 144), interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (8, 8), (272, 176), PANEL_BG, -1)
    img[24:168, 16:272] = inset
    draw_text(img, "clean source inset", (18, 20), 0.42, DIM)
    cv2.rectangle(img, (0, H - 88), (W, H), PANEL_BG, -1)
    draw_text(img, "FairwayOS local research | validated team layers only | pseudo-labels, not ground truth", (12, H - 65), 0.48, FG)
    draw_text(img, f"native src f{frame_index} | exact source SHA bound | no speed/carry/apex/landing/impact/calibration/analytics", (12, H - 43), 0.42, WARN)
    y = H - 21
    x = 12
    for layer in EXPECTED_LAYERS:
        st = states[layer]
        text = f"{layer}: {'visible' if st.get('visible') else st.get('state', 'unavailable')}"
        draw_text(img, text[:52], (x, y), 0.40, COLORS[layer] if st.get("visible") else DIM)
        x += 410
    return img


def write_video(frames: List[object], final: Path) -> None:
    cv2 = _cv2()
    raw = final.with_name(final.stem + "_raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for frame in frames:
        vw.write(frame)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(final)], check=True)
    raw.unlink(missing_ok=True)


def write_report(path: Path, report: dict) -> None:
    lines = [
        "FairwayOS team integration report",
        "",
        f"status: {report['status']}",
        f"source: {report['source']}",
        f"source_sha256: {report['source_sha256']}",
        f"ffprobe: {json.dumps(report.get('ffprobe', {}), sort_keys=True)}",
        f"layer_root: {report['layer_root']}",
        f"accepted_layers: {json.dumps(report.get('accepted_layers', {}), sort_keys=True)}",
        f"blocked_reason: {report.get('blocked_reason')}",
        f"cut_resolution: {report.get('cut_resolution')}",
        f"cut_review_artifact: {report.get('cut_review_artifact')}",
        f"rerun_command: {report['rerun_command']}",
        "",
        "Research/provenance limits: pseudo_label=true, ground_truth=false, research_only=true, production_eligible=false; local ignored media only; no public upload/deploy/push.",
    ]
    if report.get("output"):
        lines.extend([
            "",
            f"output: {report['output']}",
            f"output_sha256: {report['output_sha256']}",
            f"rendered_frames: {report['rendered_frames']}",
            f"decode_verified: {report.get('decode_verified')}",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--layer-root", type=Path, default=DEFAULT_LAYER_ROOT)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--allow-awaiting", action="store_true", help="write checkpoint instead of failing nonzero when layers are missing")
    args = ap.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    report_path = args.outdir / "team_integration_report.md"
    runner_python = Path(".venv-video-ai/bin/python") if Path(".venv-video-ai/bin/python").exists() else Path(sys.executable)
    rerun = f"{runner_python} tools/demo/build_team_layer_demo.py --source {args.source} --layer-root {args.layer_root} --outdir {args.outdir}"
    actual_sha = sha256(args.source)
    probe = ffprobe(args.source)
    report = {
        "status": "blocked",
        "source": str(args.source),
        "source_sha256": actual_sha,
        "ffprobe": probe,
        "layer_root": str(args.layer_root),
        "rerun_command": rerun,
        "accepted_layers": {},
        "cut_resolution": "native f212-f224 tile inspected: no hard cut at f215/f216/f217; visible pan continuity through f224. Cut reset remains conservative at f216 until layer producers converge.",
        "cut_review_artifact": "out/pga_demo_20260911/cut_review/f212_224_tile.jpg",
    }
    if actual_sha != SOURCE_SHA256:
        report["blocked_reason"] = "source_sha256_mismatch"
        write_report(report_path, report)
        raise SystemExit("source_sha256_mismatch")
    try:
        layers = load_completed_layers(args.layer_root, EXPECTED_LAYERS, SOURCE_SHA256, frame_size=(W, H))
    except LayerContractError as exc:
        report["status"] = "awaiting_layers"
        report["blocked_reason"] = str(exc)
        write_report(report_path, report)
        print(json.dumps({"status": "awaiting_layers", "report": str(report_path), "reason": str(exc), "rerun_command": rerun}, indent=2))
        return 0 if args.allow_awaiting else 2
    report["accepted_layers"] = {name: {"observations": len(layer["observations"])} for name, layer in layers.items()}
    frames = load_source_frames(args.source, range(FRAME_START, FRAME_END_EXCLUSIVE))
    timeline = render_timeline_states(observation_map(layers), range(FRAME_START, FRAME_END_EXCLUSIVE), cuts=CUTS)
    histories = {layer: [] for layer in EXPECTED_LAYERS}
    rendered = []
    for frame_index in range(FRAME_START, FRAME_END_EXCLUSIVE):
        states = timeline[frame_index]
        for layer in EXPECTED_LAYERS:
            if states[layer].get("visible"):
                histories[layer].append(states[layer])
            else:
                histories[layer] = []
        rendered.append(render_frame(frames[frame_index], frame_index, states, histories))
    final = args.outdir / "siwoo_team_layers_research_overlay.mp4"
    write_video(rendered, final)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(final), "-f", "null", "-"], check=True)
    report.update({
        "status": "rendered",
        "blocked_reason": None,
        "output": str(final),
        "output_sha256": sha256(final),
        "rendered_frames": len(rendered),
        "decode_verified": True,
    })
    write_report(report_path, report)
    (args.outdir / "team_integration_diagnostics.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "rendered", "output": str(final), "sha256": report["output_sha256"], "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
