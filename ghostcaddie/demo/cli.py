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

DEMO_COMMANDS = ("demo-video", "demo-setup")


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


def dispatch(args, argv: Optional[List[str]] = None) -> None:
    if args.command == "demo-video":
        _run_demo_video(args)
        return
    if args.command == "demo-setup":
        _run_demo_setup(args)
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
