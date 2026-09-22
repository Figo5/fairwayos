"""Regenerate the small non-blank FrameViews fixture, deterministically.

The fixture is synthetic on purpose: it must be small enough to live in the
repository, and it must not be a crop of any real (uncleared) media. It is
textured rather than blank so the shared-pyramid crop path and the direct crop
path actually have something to disagree about -- on a blank image every crop is
the same crop and the test proves nothing.

Run from the repository root:

    .venv-video-ai/bin/python3 tests/fixtures/make_frameviews_fixture.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def build(seed: int = 20260921) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height, width = 240, 320
    frame = np.zeros((height, width, 3), np.uint8)
    # turf: two-tone gradient plus mown-stripe banding
    columns = np.linspace(0, 1, width)[None, :]
    rows = np.linspace(0, 1, height)[:, None]
    frame[..., 0] = np.clip(28 + 30 * rows + 12 * np.sin(columns * 14 * np.pi), 0, 255)
    frame[..., 1] = np.clip(105 + 40 * columns - 20 * rows, 0, 255)
    frame[..., 2] = np.clip(38 + 18 * rows, 0, 255)
    noise = rng.normal(0, 9, (height, width, 3))
    frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    frame = cv2.GaussianBlur(frame, (0, 0), 0.9)
    # a large light object (a ball-sized disc), a small one, and a dark one
    cv2.circle(frame, (150, 120), 40, (198, 205, 190), -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (150, 120), 40, (150, 160, 148), 2, lineType=cv2.LINE_AA)
    cv2.circle(frame, (52, 190), 6, (243, 246, 238), -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (268, 62), 9, (46, 44, 40), -1, lineType=cv2.LINE_AA)
    cv2.putText(frame, "fixture", (222, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (235, 238, 230), 1, cv2.LINE_AA)
    return frame


def main() -> int:
    target = Path(__file__).resolve().parent / "frameviews_turf.png"
    frame = build()
    cv2.imwrite(str(target), frame)
    print(f"wrote {target} ({target.stat().st_size} bytes, "
          f"{frame.shape[1]}x{frame.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
