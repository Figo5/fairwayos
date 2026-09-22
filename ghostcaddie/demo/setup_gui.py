"""The interactive seed/keyframe picker window (cv2 highgui).

Run as ``<cv interpreter> -m ghostcaddie.demo.setup_gui VIDEO [--config IN] [--out OUT]``.
`demo-setup` launches this out of process in an interpreter that has cv2, so the CLI
itself stays stdlib-only.

Usability rules this file exists to honour:

* **Fit to the screen, map back exactly.** Native 4K footage does not fit a laptop
  display, so the frame is shown through an explicit viewport transform and every
  click/drag is converted back to native pixels (see ``picker.viewport_transform``).
  The old behaviour -- native size with ``WINDOW_AUTOSIZE`` -- put the controls off
  screen on the very clip this tool is for.
* **The HUD lives in its own gutter.** The picture is not overlapped by status text,
  so the operator can see the ball they are clicking on.
* **One decode per navigation.** The current frame is cached; idle redraws at ~20 ms
  do not re-seek and re-decode, which made the old loop sluggish over long clips.
* **Quitting is possible.** A dirty session needs a second ``q``/Escape, and the
  second one really does quit (``PickerState.request_quit``).
* **A save that cannot validate says why** in the HUD instead of writing a config the
  pipeline will refuse.
* **ASCII only in the window.** cv2 Hershey fonts render middle dots as ``??``.

`draw_picker_view` is a pure function of state + frame + transform, so the picker's
appearance can be rendered to a PNG in a test without a display.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .picker import (HELP_LINES, PHASE_KEYS_UPPER, PickerState, native_to_viewport,
                     roi_to_native, viewport_to_native)
from .probe import probe_source

WINDOW = "FairwayOS demo-setup (h = help, s = save, q = quit)"
DEFAULT_VIEWPORT = (1300, 860)
HUD_W = 470
TEAL = (172, 196, 52)
BALL = (70, 235, 255)
CLUB = (255, 96, 208)
TEXT = (238, 238, 238)
MUTED = (170, 170, 170)
AMBER = (90, 205, 255)
PANEL = (22, 22, 22)


def _text(img, text, org, scale=0.42, color=TEXT, thick=1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                cv2.LINE_AA)


def draw_picker_view(frame: np.ndarray, state: PickerState, transform: dict, *,
                     mouse: Optional[Tuple[float, float]] = None,
                     drag: Optional[List[Tuple[int, int]]] = None,
                     show_help: bool = True, show_numbers: bool = True) -> np.ndarray:
    """Compose the picker window: picture at left, HUD gutter at right.

    ``mouse`` and ``drag`` are in VIEWPORT pixels (that is what cv2 reports); all
    state coordinates are native, and the drawing maps them out through the same
    transform that maps clicks back in.
    """
    scale = float(transform["scale"]) or 1.0
    view = cv2.resize(frame, (int(transform["view_w"]), int(transform["view_h"])),
                      interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST)
    canvas_h = max(int(transform["viewport_h"]),
                   int(transform["view_h"]) + 2 * int(transform["offset_y"]))
    canvas_w = int(transform["view_w"]) + HUD_W
    canvas = np.full((canvas_h, canvas_w, 3), 24, dtype=np.uint8)
    ox = int(transform["offset_x"])
    oy = int(transform["offset_y"])
    canvas[oy:oy + view.shape[0], ox:ox + view.shape[1]] = view

    if state.roi:
        x0, y0 = native_to_viewport((state.roi[0], state.roi[1]), transform)
        x1, y1 = native_to_viewport((state.roi[0] + state.roi[2],
                                     state.roi[1] + state.roi[3]), transform)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 200, 255), 2)
        _text(canvas, "ROI (operator input)", (x0 + 4, max(14, y0 - 6)), 0.45,
              (0, 200, 255))
    for name, color in (("ball", BALL), ("clubhead", CLUB)):
        seed = state.seeds.get(name)
        if seed and seed[0] == state.frame:
            px, py = native_to_viewport((seed[1], seed[2]), transform)
            cv2.circle(canvas, (px, py), 10, color, 2)
            _text(canvas, f"{name} seed", (px + 12, py - 8), 0.45, color)
        for (frame, x, y) in state.corrections.get(name, []):
            if frame == state.frame:
                px, py = native_to_viewport((x, y), transform)
                cv2.circle(canvas, (px, py), 8, TEXT, 2)
                _text(canvas, f"{name} correction", (px + 10, py - 6), 0.42, TEXT)
    for phase, frame in state.keyframes.items():
        if frame == state.frame:
            # pinned INSIDE the picture height: with a centred viewport, canvas_h can
            # be taller than the picture and this label used to fall off the bottom
            _text(canvas, f"phase mark: {phase}",
                  (ox + 12, oy + int(transform["view_h"]) - 16), 0.55, TEAL, 2)
    if drag and len(drag) > 1:
        cv2.rectangle(canvas, (int(drag[0][0]), int(drag[0][1])),
                      (int(drag[-1][0]), int(drag[-1][1])), AMBER, 1)
    if mouse is not None and viewport_to_native(mouse, transform) is not None:
        cv2.drawMarker(canvas, (int(mouse[0]), int(mouse[1])), AMBER,
                       cv2.MARKER_CROSS, 22, 1)
        if show_numbers:
            native = viewport_to_native(mouse, transform)
            if native:
                _text(canvas, f"native ({int(native[0])},{int(native[1])})",
                      (int(mouse[0]) + 12, int(mouse[1]) + 18), 0.45, AMBER)

    hud_x = int(transform["view_w"])
    canvas[:, hud_x:hud_x + HUD_W] = (canvas[:, hud_x:hud_x + HUD_W] * 0.25).astype(
        canvas.dtype)
    y = 22
    for line in state.status_lines(show_help=show_help):
        if y > canvas_h - 8:
            break
        color = MUTED if line.startswith(("-", "  ")) else TEXT
        if line.startswith("demo-setup"):
            color = TEAL
        _text(canvas, line, (hud_x + 8, y), 0.40, color)
        y += 16
    _text(canvas, f"view scale {scale:.3f}", (hud_x + 8, canvas_h - 10), 0.38, MUTED)
    return canvas


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="ghostcaddie demo-setup (picker)")
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", type=Path, default=None,
                        help="existing config to edit")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to save the config (default: next to the clip)")
    parser.add_argument("--start", type=int, default=None,
                        help="start on this source frame")
    parser.add_argument("--viewport", default=f"{DEFAULT_VIEWPORT[0]}x{DEFAULT_VIEWPORT[1]}",
                        help="maximum picture viewport, e.g. 1600x950")
    args = parser.parse_args(argv)

    src = Path(args.video).expanduser()
    if not src.is_file():
        print(f"demo-setup: source video not found: {src}", file=sys.stderr)
        return 1
    probe = probe_source(str(src))
    existing = None
    if args.config is not None:
        if not Path(args.config).is_file():
            print(f"demo-setup: config not found: {args.config}", file=sys.stderr)
            return 1
        from .config import load_config
        existing = load_config(str(args.config))
    state = PickerState(str(src), probe, existing=existing,
                        out_path=str(args.out) if args.out else None)
    if args.start is not None:
        state.goto(args.start)

    try:
        vw, vh = (int(v) for v in str(args.viewport).lower().split("x"))
    except ValueError:
        print("demo-setup: --viewport must look like 1600x950", file=sys.stderr)
        return 1
    transform = viewport_transform_for(probe, vw, vh)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"demo-setup: OpenCV could not open {src}", file=sys.stderr)
        return 1
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    except cv2.error as exc:
        print(f"demo-setup: no GUI backend available ({exc}). Write the config JSON "
              f"by hand instead: see docs/demo-mode.md.", file=sys.stderr)
        return 2

    state_flag = {"help": True, "numbers": True}
    mouse: Optional[Tuple[float, float]] = None
    drag: List[Tuple[int, int]] = []
    cached: dict = {"frame": None, "image": None}
    quit_requested = False

    def current_frame_image():
        """Decode only when the current frame changed (one decode per navigation)."""
        if cached["frame"] != state.frame or cached["image"] is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, state.frame)
            ok, img = cap.read()
            cached["frame"] = state.frame if ok else None
            cached["image"] = img if ok else None
        return cached["image"]

    def on_mouse(event, x, y, _flags, _param) -> None:
        nonlocal mouse
        mouse = (float(x), float(y))
        if event == cv2.EVENT_LBUTTONDOWN:
            drag.clear()
            drag.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and drag:
            drag[:] = [drag[0], (x, y)]
        elif event == cv2.EVENT_LBUTTONUP and drag:
            (x0, y0), (x1, y1) = drag[0], (x, y)
            drag.clear()
            if abs(x1 - x0) > 8 and abs(y1 - y0) > 8:
                native = roi_to_native((min(x0, x1), min(y0, y1),
                                        abs(x1 - x0), abs(y1 - y0)), transform)
                state.set_roi(*native)

    try:
        cv2.setMouseCallback(WINDOW, on_mouse)
    except cv2.error:
        pass  # headless-ish build: keyboard-only still works

    print("\n".join(HELP_LINES))
    print(f"demo-setup: viewport {vw}x{vh} -> scale {transform['scale']:.4f} "
          f"(clicks are mapped back to native pixels)")
    while not quit_requested:
        frame = current_frame_image()
        if frame is None:
            state.step(1)
            continue
        canvas = draw_picker_view(frame, state, transform, mouse=mouse, drag=drag,
                                  show_help=state_flag["help"],
                                  show_numbers=state_flag["numbers"])
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue
        native = viewport_to_native(mouse, transform) if mouse else None
        if key == ord("q"):
            quit_requested = state.request_quit()
        elif key == 27:
            quit_requested = state.request_quit()
        elif key in (ord("h"), ord("?")):
            state_flag["help"] = not state_flag["help"]
        elif key == ord("H"):
            state_flag["numbers"] = not state_flag["numbers"]
        elif key in (ord("s"), ord("w")):
            path = state.save()
            print(f"demo-setup: {'saved ' + path if path else 'config not saved'}")
        elif key in (81, 2, ord("a")):
            state.step(-1)
        elif key in (83, 3, ord("d")):
            state.step(1)
        elif key == ord("n"):
            state.step(-10)
        elif key == ord("N"):
            state.step(10)
        elif key == ord("["):
            state.jump_seconds(-1.0)
        elif key == ord("]"):
            state.jump_seconds(1.0)
        elif key == ord("i"):
            state.set_interval_start()
        elif key == ord("o"):
            state.set_interval_end()
        elif key == ord("g"):
            state.clear_interval()
        elif key == ord("x"):
            state.clear_roi()
        elif key == ord("1") and native:
            state.place_seed("ball", *native)
        elif key == ord("2") and native:
            state.place_seed("clubhead", *native)
        elif key == ord("3") and native:
            state.place_correction("ball", *native)
        elif key == ord("4") and native:
            state.place_correction("clubhead", *native)
        elif key == ord("5"):
            state.clear_corrections("ball")
        elif key == ord("6"):
            state.clear_corrections("clubhead")
        elif key == ord("c") or key == ord("C"):
            state.clear_keyframes()
        elif 65 <= key <= 90 and chr(key) in PHASE_KEYS_UPPER:
            # Upper-case only, which is what the help and the docs say: lower-case
            # a/d/i are navigation, so accepting them as phase marks would make a
            # navigation key silently change the phase set.
            # Out-of-order marks are accepted while editing and reported; the config
            # validation is what refuses them, so the session never crashes mid-edit.
            state.mark_phase(PHASE_KEYS_UPPER[chr(key)])
    cap.release()
    cv2.destroyAllWindows()
    payload = state.to_config_dict()
    print(f"demo-setup: finished. config: {state.out_path} "
          f"({'saved' if not state.dirty else 'NOT SAVED - press s next time'})")
    print(json.dumps({k: payload[k] for k in ("interval", "roi", "seeds", "keyframes")},
                     indent=2))
    return 0 if not state.dirty else 3


def viewport_transform_for(probe, window_w: int, window_h: int) -> dict:
    """Map a WHOLE-WINDOW bound onto a picture viewport.

    The HUD gutter is part of the window, so the picture gets what is left. Without
    this, ``--viewport 1600x950`` produced a ~2070 px window that does not fit a
    common laptop display.
    """
    from .picker import viewport_transform
    max_w = max(360, int(window_w) - HUD_W - 8)
    max_h = max(240, int(window_h) - 8)
    return viewport_transform(probe.width, probe.height, max_w, max_h)


if __name__ == "__main__":
    raise SystemExit(main())
