"""CLI entry points for demo mode: ``demo-video`` and ``demo-setup``.

`ghostcaddie.cli` registers these two subcommands and dispatches to them, but the
definitions live here so the canonical CLI surface stays readable. Both commands
are stdlib-only: the CV work is dispatched into another interpreter, and the
picker window is a subprocess for exactly that reason.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .errors import DemoError

DEMO_COMMANDS = ("demo-video", "demo-setup", "broadcast-showcase")


def register(sub) -> None:
    """Attach the demo subcommands to an argparse subparser collection."""
    video = sub.add_parser(
        "demo-video",
        help="Operator-assisted demo of one selected clip: seeded ball/club "
             "tracking, 2D analytics, phase cards, annotated video. Local only; "
             "experimental and image-space by construction.")
    video.add_argument("video", type=Path, help="Local video file. Never uploaded.")
    video.add_argument("--config", required=True, type=Path,
                       help="Demo config JSON (see docs/demo-mode.md).")
    video.add_argument("--out", required=True, type=Path,
                       help="Output directory for the demo artifact set.")
    video.add_argument("--cv-python", default=None,
                       help="Interpreter that has cv2+numpy. Default: "
                            "$FAIRWAY_DEMO_PYTHON, then the canonical body "
                            "interpreter, then this one.")
    video.add_argument("--body-model", default=None,
                       help="Explicit MoveNet TFLite path. Default: the canonical "
                            "model search path. Never downloaded.")
    video.add_argument("--no-render", action="store_true",
                       help="Skip the annotated video (cards and JSON still written).")
    video.add_argument("--no-summary", action="store_true",
                       help="Skip demo_summary.md.")
    video.add_argument("--showcase", action="store_true",
                       help="Write the opt-in presentation artifact set "
                            "(showcase_* names, chrome drawn over the picture) instead "
                            "of the demo set. Also enabled by showcase.enabled in the "
                            "config.")
    video.add_argument("--no-showcase", action="store_true",
                       help="Force the demo renderer even when showcase.enabled is "
                            "true in the config.")
    video.add_argument("--redact-path", action="store_true",
                       help="Record only file names, not full paths.")
    video.add_argument("--json", action="store_true",
                       help="Print demo_analysis.json to stdout instead of a summary.")

    setup = sub.add_parser(
        "demo-setup",
        help="Interactive local picker: interval, ROI, ball/club seeds, phase "
             "keyframes, corrections -> config JSON. No Python editing needed.")
    setup.add_argument("video", type=Path, help="Local video file. Never uploaded.")
    setup.add_argument("--config", type=Path, default=None,
                       help="Existing config to edit.")
    setup.add_argument("--out", type=Path, default=None,
                       help="Where to save the config (default: next to the clip).")
    setup.add_argument("--start", type=int, default=None,
                       help="Start the picker on this source frame.")
    setup.add_argument("--cv-python", default=None,
                       help="Interpreter that has cv2 (needs a GUI backend).")
    setup.add_argument("--viewport", default=None,
                       help="Maximum WINDOW size, e.g. 1300x860. The HUD gutter is "
                            "inside it, so the picture gets what is left.")

    _register_broadcast(sub)


def _register_broadcast(sub) -> None:
    """The BROADCAST SHOWCASE workflow: several scene windows in one whole shot.

    Kept as a command group rather than a flag on ``demo-video`` because it answers a
    different question: ``demo-video`` renders ONE interval, the broadcast workflow
    renders the whole shot and draws only inside the windows the operator declares.
    """
    bc = sub.add_parser(
        "broadcast-showcase",
        help="Whole-shot broadcast showcase: one or more golfer-visible scene "
             "windows inside the clip, source cadence and source audio preserved. "
             "Local only; experimental and image-space by construction.")
    bc_sub = bc.add_subparsers(dest="broadcast_command", required=True)

    run_p = bc_sub.add_parser(
        "run", help="Render one clip from a broadcast config.")
    run_p.add_argument("clip", type=Path, help="Local video file. Never uploaded.")
    run_p.add_argument("--config", required=True, type=Path,
                       help="Broadcast config JSON "
                            "(schema fairwayos-broadcast-config.v1).")
    run_p.add_argument("--out", required=True, type=Path,
                       help="Output directory for the broadcast artifact set.")
    run_p.add_argument("--cv-python", default=None,
                       help="Interpreter that has cv2+numpy.")
    run_p.add_argument("--body-model", default=None,
                       help="Explicit MoveNet TFLite path. Never downloaded.")
    run_p.add_argument("--no-render", action="store_true",
                       help="Measure only; write no video.")
    run_p.add_argument("--redact-path", action="store_true",
                       help="Record only file names, not full paths.")
    run_p.add_argument("--force", action="store_true",
                       help="Replace an existing finished render in --out.")

    val_p = bc_sub.add_parser(
        "validate", help="Parse and bind a broadcast config to a clip; render "
                         "nothing. This is the fast check for a config an operator "
                         "has just written.")
    val_p.add_argument("clip", type=Path, help="Local video file. Never uploaded.")
    val_p.add_argument("--config", required=True, type=Path,
                       help="Broadcast config JSON.")
    val_p.add_argument("--json", action="store_true",
                       help="Print the bound result as JSON.")

    sc_p = bc_sub.add_parser(
        "scenes", help="Discover cut and dissolve CANDIDATES in a local clip. "
                       "Suggestions only: nothing is applied to a config.")
    sc_p.add_argument("clip", type=Path, help="Local video file. Never uploaded.")
    sc_p.add_argument("--out", type=Path, default=None,
                      help="Where to write the discovery JSON "
                           "(default: stdout).")
    sc_p.add_argument("--config", type=Path, default=None,
                      help="Existing broadcast config: its saved decisions are merged "
                           "in, so a rejected proposal cannot come back.")
    sc_p.add_argument("--scores", action="store_true",
                      help="Include the per-frame score table in the output.")

    ba_p = bc_sub.add_parser(
        "batch", help="Render one broadcast config per clip from a batch manifest; "
                      "one bounded subprocess per clip, one row per clip in the "
                      "combined report.")
    ba_p.add_argument("--manifest", required=True, type=Path,
                      help="Batch manifest JSON (a list of entries, or the baseline "
                           "manifest.json with its 'clips').")
    ba_p.add_argument("--out", required=True, type=Path,
                      help="Root output directory for the batch.")
    ba_p.add_argument("--cv-python", default=None)
    ba_p.add_argument("--body-model", default=None)
    ba_p.add_argument("--force", action="store_true",
                      help="Replace finished renders.")
    ba_p.add_argument("--timeout", type=int, default=3600,
                      help="Per-clip wall-clock bound in seconds (default 3600).")

    ui_p = bc_sub.add_parser(
        "ui", help="Local browser review UI: import a clip, drag the golfer region "
                   "over real frames, mark phases, place seeds, save a config and "
                   "render. Loopback only.")
    ui_p.add_argument("--host", default="127.0.0.1",
                      help="Loopback host. Any other value is refused.")
    ui_p.add_argument("--port", type=int, default=8765)
    ui_p.add_argument("--out-root", default=None,
                      help="Where UI renders are written (default: the run dir).")
    ui_p.add_argument("--config-dir", default=None,
                      help="Where UI configs are written (default: the run dir).")
    ui_p.add_argument("--extra-root", action="append", default=[],
                      help="Additional allow-listed media root (repeatable).")
    ui_p.add_argument("--cv-python", default=None)
    ui_p.add_argument("--body-model", default=None)
    ui_p.add_argument("--open", action="store_true",
                      help="Open the page in the default browser.")


def _run_demo_video(args) -> None:
    from .pipeline import run_demo
    from .report import render_terminal_summary

    if getattr(args, "showcase", False) and getattr(args, "no_showcase", False):
        raise SystemExit("demo-video: --showcase and --no-showcase are contradictory; "
                         "nothing was run.")
    showcase: Optional[bool] = None
    if getattr(args, "showcase", False):
        showcase = True
    elif getattr(args, "no_showcase", False):
        showcase = False
    try:
        result = run_demo(
            str(args.video), str(args.config), str(args.out),
            cv_python=args.cv_python, body_model_path=args.body_model,
            render=not args.no_render, summary=not args.no_summary,
            redact_path=args.redact_path, showcase=showcase)
    except DemoError as exc:
        raise SystemExit(f"demo-video failed: {exc}") from exc
    # Config warnings are printed as well as persisted: an operator who is about to
    # trust a config written for another clip should not have to open the report.
    for warning in result.warnings:
        print(f"demo-video: CONFIG WARNING: {warning}", file=sys.stderr)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str))
        return
    print(render_terminal_summary(result))


def _run_demo_setup(args) -> None:
    from .paths import resolve_cv_interpreter, run_setup_process

    try:
        interpreter = resolve_cv_interpreter(args.cv_python)
    except DemoError as exc:
        raise SystemExit(f"demo-setup failed: {exc}") from exc
    passthrough: List[str] = [str(args.video)]
    if args.config is not None:
        passthrough += ["--config", str(args.config)]
    if args.out is not None:
        passthrough += ["--out", str(args.out)]
    if args.start is not None:
        passthrough += ["--start", str(args.start)]
    if getattr(args, "viewport", None):
        passthrough += ["--viewport", str(args.viewport)]
    print(f"demo-setup: picker interpreter {interpreter['path']} "
          f"({interpreter['how']}, python {interpreter.get('python_version') or '?'})")
    code = run_setup_process(interpreter["path"], passthrough)
    if code == 3:
        print("demo-setup: the session ended with unsaved changes. Nothing was "
              "written; run it again and press s to save.", file=sys.stderr)
    raise SystemExit(code)


def _run_broadcast_showcase(args) -> None:
    from .broadcast_config import load_broadcast, validate_broadcast
    from .probe import probe_source

    command = getattr(args, "broadcast_command", None)
    if command == "validate":
        try:
            config = load_broadcast(str(args.config))
            probe = probe_source(str(args.clip))
            resolved = validate_broadcast(config, probe)
        except DemoError as exc:
            raise SystemExit(f"broadcast-showcase validate failed: {exc}") from exc
        summary = {
            "ok": True,
            "config": str(args.config),
            "clip": str(args.clip),
            "clip_sha256": None,
            "frame_count": probe.frame_count,
            "fps": probe.fps,
            "interval": {"start_frame": resolved.start_frame,
                         "end_frame": resolved.end_frame},
            "scenes": [{"id": s.scene.scene_id, "kind": s.scene.kind,
                        "stroke_type": s.scene.stroke_type,
                        "start_frame": s.start_frame, "end_frame": s.end_frame,
                        "phase_vocabulary": list(s.scene.phase_vocabulary)}
                       for s in resolved.scenes],
            "warnings": resolved.warnings,
        }
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
            return
        print(f"broadcast-showcase validate: OK  ({len(resolved.scenes)} scene(s), "
              f"clip {probe.width}x{probe.height} @ {probe.fps:.6f} fps, "
              f"{probe.frame_count} frames)")
        for scene in resolved.scenes:
            print(f"  scene {scene.scene.scene_id} [{scene.scene.kind}/"
                  f"{scene.scene.stroke_type}] frames {scene.start_frame}.."
                  f"{scene.end_frame} inclusive")
        for warning in resolved.warnings:
            print(f"  warning: {warning}")
        return
    if command == "run":
        from .broadcast import run_broadcast

        try:
            result = run_broadcast(
                str(args.clip), str(args.config), str(args.out),
                cv_python=args.cv_python, body_model_path=args.body_model,
                render=not args.no_render, redact_path=args.redact_path,
                force=args.force)
        except DemoError as exc:
            raise SystemExit(f"broadcast-showcase run failed: {exc}") from exc
        for warning in result.warnings:
            print(f"broadcast-showcase: WARNING: {warning}", file=sys.stderr)
        render = result.render or {}
        check = (result.verification or {}).get("render_ffprobe") or {}
        print(f"broadcast-showcase: {result.artifacts.get('output_dir')}")
        print(f"  video:   {result.artifacts.get('annotated')}")
        print(f"  report:  {result.artifacts.get('report')}")
        print(f"  frames:  {render.get('frames')} "
              f"({render.get('overlay_frames')} in-window, "
              f"{render.get('empty_frames')} bare) @ {render.get('fps_rational')}")
        print(f"  audio:   {render.get('mux_note') or 'not muxed'}")
        print(f"  verify:  {'PASS' if check.get('ok') else 'FAIL ' + str(check)}")
        return
    if command == "scenes":
        from .broadcast_config import load_broadcast
        from .discovery import discover

        existing = []
        try:
            if args.config is not None:
                existing = load_broadcast(str(args.config)).cuts
        except DemoError as exc:
            raise SystemExit(f"broadcast-showcase scenes failed: {exc}") from exc
        try:
            record = discover(str(args.clip), existing=existing)
        except DemoError as exc:
            raise SystemExit(f"broadcast-showcase scenes failed: {exc}") from exc
        if not args.scores:
            record["scores"].pop("per_frame", None)
        if args.out is not None:
            out = Path(args.out).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
            print(f"broadcast-showcase scenes: {len(record['candidates'])} candidate(s) "
                  f"-> {out}")
        else:
            print(json.dumps(record, indent=2, sort_keys=True))
        return
    if command == "batch":
        from .batch import run_batch

        try:
            record = run_batch(str(args.manifest), str(args.out), force=args.force,
                               timeout=args.timeout, cv_python=args.cv_python,
                               body_model=args.body_model)
        except DemoError as exc:
            raise SystemExit(f"broadcast-showcase batch failed: {exc}") from exc
        counts = record["counts"]
        print(f"broadcast-showcase batch: {counts['succeeded']}/{counts['total']} "
              f"succeeded ({counts['failed']} failed, {counts['refused']} refused)")
        for item in record["results"]:
            line = f"  {item['id']}: {item['status']}"
            if item.get("reason"):
                line += f" — {item['reason']}"
            print(line)
        print(f"  report: {Path(args.out).expanduser() / 'batch_report.md'}")
        if counts["failed"] or counts["refused"]:
            # Non-zero on any clip that did not succeed: a batch is not a success
            # because some of its clips were.
            raise SystemExit(1)
        return
    if command == "ui":
        from .broadcast_ui import serve

        try:
            serve(host=args.host, port=args.port, out_root=args.out_root,
                  config_dir=args.config_dir, extra_roots=args.extra_root,
                  cv_python=args.cv_python, body_model=args.body_model,
                  open_browser=args.open)
        except DemoError as exc:
            raise SystemExit(f"broadcast-showcase ui failed: {exc}") from exc
        return
    raise SystemExit(f"unknown broadcast-showcase command: {command!r}")


def dispatch(args, argv: Optional[List[str]] = None) -> None:
    if args.command == "demo-video":
        _run_demo_video(args)
        return
    if args.command == "demo-setup":
        _run_demo_setup(args)
        return
    if args.command == "broadcast-showcase":
        _run_broadcast_showcase(args)
        return
    raise SystemExit(f"unknown demo command: {args.command!r}")


def run(argv: Optional[List[str]] = None) -> int:
    """Stand-alone entry point: ``python -m ghostcaddie.demo.cli demo-video ...``."""
    parser = argparse.ArgumentParser(prog="ghostcaddie demo")
    sub = parser.add_subparsers(dest="command", required=True)
    register(sub)
    args = parser.parse_args(argv)
    dispatch(args, argv)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(run())
