"""Context and motion representation for a v3 candidate.

The v2 detector saw one 64x64 crop scaled to about eight ball radii. Inside that
window a golf ball and a shirt highlight are the same picture, which is why the
v2 pass named saturation as its failure. The Phase 0 measurement refined that: on
in-regime balls an oracle crop still beat the best same-frame distractor 62.5% of
the time with a median margin of +0.023 -- a real signal, but far too thin to rank
on. Outside the trained scale it inverted completely (8.3%, margin -0.369), which
is a *scale* gap rather than a discrimination one.

So a v3 candidate is described by three registered views of the same place, at two
scales, plus what moved:

* **tight** -- about 8 ball radii, the ball's own appearance at native detail;
* **context** -- about 32 ball radii, which is what "is there a golfer's shoe
  next to this, is this sitting on the fairway, is there a row of range balls"
  actually needs to be visible;
* **motion** -- the camera-compensated frame difference over the context window.
  A ball that has been struck moves unlike the grass it flies over; a shirt
  highlight moves exactly like the golfer.

All three are resampled to the same 64x64 grid and stacked into one 7-channel
tensor, so a single small CNN reads them together and the scales stay registered.
Two scales in one tensor is also what closes the scale gap: the tight window is
defined in ball radii, so a 6 px ball and a 70 px ball arrive at the network
looking like the same size object, which is the property the v2 fixed 64 px crop
did not have.
"""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

#: Grid every view is resampled onto.
GRID = 64
#: Tight window, in ball radii. Eight radii keeps the ball about a quarter of the
#: window wide, which leaves room for the contact shadow and a little background.
TIGHT_RADII = 8.0
#: Context window, in ball radii.
CONTEXT_RADII = 32.0
#: Smallest windows in pixels, so a 2 px ball does not produce a 16 px crop that
#: upsampling turns into pure interpolation noise.
MIN_TIGHT_PX = 24
MIN_CONTEXT_PX = 96
#: 7 = tight BGR + context BGR + motion.
CHANNELS = 7


def _window(frame: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    """A square window centred on (x, y), edge-replicated past the frame border.

    Two choices worth stating. Clamping the window into the frame would silently
    move the candidate off the centre of its own crop, and the network would then
    be reading a position it was not asked about -- so the window is padded
    instead. And the padding **replicates** rather than reflects: reflection at a
    border three pixels from a ball produces a mirrored copy of that ball inside
    the crop, which is a fabricated second object of exactly the kind this
    pipeline exists to not hallucinate. Replication smears, which is honest about
    having no data there.
    """
    size = max(2, int(round(size)))
    half = size // 2
    left, top = int(round(x)) - half, int(round(y)) - half
    height, width = frame.shape[:2]
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, left + size - width)
    pad_bottom = max(0, top + size - height)
    if pad_left or pad_top or pad_right or pad_bottom:
        frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right,
                                   cv2.BORDER_REPLICATE)
        left += pad_left
        top += pad_top
    return frame[top:top + size, left:left + size]


def _resample(patch: np.ndarray) -> np.ndarray:
    interpolation = (cv2.INTER_AREA if patch.shape[0] > GRID else cv2.INTER_LINEAR)
    return cv2.resize(patch, (GRID, GRID), interpolation=interpolation)


def compensate(previous_gray: np.ndarray, shift: tuple[float, float]) -> np.ndarray:
    """Shift the previous frame by the estimated camera motion, in native pixels."""
    dx, dy = shift
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return previous_gray
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(previous_gray, matrix,
                          (previous_gray.shape[1], previous_gray.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def motion_residual(gray: np.ndarray, compensated_previous: Optional[np.ndarray]
                    ) -> np.ndarray:
    """What moved, once the camera's own motion has been taken out."""
    if compensated_previous is None:
        return np.zeros_like(gray, dtype=np.uint8)
    return cv2.absdiff(gray, compensated_previous)


def candidate_stack(frame: np.ndarray, residual: Optional[np.ndarray],
                    x: float, y: float, radius: float) -> np.ndarray:
    """The 7-channel (H, W, C) description of one candidate. uint8 throughout."""
    radius = max(1.0, float(radius))
    tight_px = max(MIN_TIGHT_PX, int(round(TIGHT_RADII * radius)))
    context_px = max(MIN_CONTEXT_PX, int(round(CONTEXT_RADII * radius)))
    tight = _resample(_window(frame, x, y, tight_px))
    context = _resample(_window(frame, x, y, context_px))
    if residual is None:
        motion = np.zeros((GRID, GRID), dtype=np.uint8)
    else:
        motion = _resample(_window(residual, x, y, context_px))
        if motion.ndim == 3:
            motion = motion[..., 0]
    return np.dstack([tight, context, motion[..., None]]).astype(np.uint8)


def stack_batch(frame: np.ndarray, residual: Optional[np.ndarray],
                positions) -> np.ndarray:
    """(N, GRID, GRID, CHANNELS) for many candidates on one frame."""
    if not len(positions):
        return np.zeros((0, GRID, GRID, CHANNELS), dtype=np.uint8)
    return np.stack([candidate_stack(frame, residual, x, y, radius)
                     for x, y, radius in positions])


def geometry_features(x: float, y: float, radius: float, width: int, height: int
                      ) -> list[float]:
    """Scale and position, normalised so the network is not handed raw pixels.

    Absolute pixel coordinates do not transfer between a 1080x1920 phone clip and
    a 3840x2160 shot of the same swing; a fraction of the frame does.
    """
    short_side = max(1, min(width, height))
    return [
        x / max(1, width),
        y / max(1, height),
        radius / short_side,
        math.log(max(1.0, radius)) / 5.0,
    ]
