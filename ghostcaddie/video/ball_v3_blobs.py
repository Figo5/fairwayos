"""A proposal generator built for the miss the v2 ensemble actually makes.

Phase 0 measured the v2 ensemble proposing the true ball on 31.0% of real
labelled frames, and raising its per-method cap from 16 to 96 moved that to 33.3%
and then stopped -- the candidate count per frame saturates at about 55 whatever
the cap is. So the ball is not being ranked out of a long list; **most of the
time no generator responds at it at all.**

Looking at the frames explains why. The labelled balls in this corpus are a white
ball on bright grass, a white ball on sand-coloured dry grass, a white ball
against a dark clubhead, a **grey ball in shadow**, a **yellow ball on
yellow-green grass**, and an **orange ball on a green mat**. Two of the five v2
generators key on brightness, and the radial-symmetry transform runs on a
grayscale gradient field where a yellow ball on yellow grass has almost no edge.
A ball is not reliably *bright*. It is reliably **locally distinct**: a small
compact region whose colour differs from the turf immediately around it.

So this generator asks that question directly, and asks it in a colour space
where it is answerable:

* work in CIE Lab, where distance approximates perceived colour difference, so a
  grey ball on green grass and a white ball on green grass are both far from
  their surround even though only one is brighter;
* at each scale, compare each pixel to the *local* background -- a box mean over
  roughly three ball radii -- rather than to a global threshold, so a shadowed
  ball on shadowed grass still stands out;
* normalise by local colour variation, so uniform turf with a small distinct
  object scores high and busy foliage does not;
* multiply by a compactness term from a difference-of-Gaussians at the same
  scale, in **both polarities**, so a dark ball is found as readily as a bright
  one and a long thin highlight on a club shaft is not;
* keep many local maxima. This stage is a recall stage. The ranker says no.

This does not replace the v2 ensemble; it joins it. Independent signals that fail
in different ways are the entire reason the ensemble design exists.
"""
from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from .ball_candidates import Candidate2, FrameContext, _peaks, _suppress
from . import ball_candidates as bc

#: Ball radius as a fraction of the frame's short side. Spans a 3 px ball on a 4K
#: wide tee shot up to a macro ball filling a tenth of the frame.
SCALES = (0.0030, 0.0050, 0.0080, 0.0130, 0.0210, 0.0340, 0.0550, 0.0900)
#: Background window, in ball radii. Three radii is wide enough to be turf rather
#: than ball, and narrow enough to stay local under a shadow edge.
BACKGROUND_RADII = 3.0
#: Local colour variation below this (in Lab units) is treated as this value, so
#: perfectly flat turf does not divide the score to infinity.
MIN_LOCAL_VARIATION = 2.0
#: Every scale is computed on a pyramid level where the ball is about this many
#: pixels in radius, so all scales cost the same as the finest.
TARGET_RADIUS_PX = 6.0


def _box(image: np.ndarray, radius: int) -> np.ndarray:
    size = max(3, int(radius) * 2 + 1)
    return cv2.blur(image, (size, size))


def _colour_distance(lab: np.ndarray, radius: int) -> np.ndarray:
    """How far is each pixel from the colour of the turf immediately around it."""
    background = _box(lab, int(max(1, round(radius * BACKGROUND_RADII))))
    difference = lab - background
    return np.sqrt(np.sum(difference * difference, axis=2))


def _response(lab: np.ndarray, radius: int) -> np.ndarray:
    """Locally distinct AND compact, both judged on colour rather than lightness.

    The first version of this multiplied the colour-distinctness map by a
    difference-of-Gaussians taken on the **L** channel, and that quietly
    reintroduced the brightness assumption the whole generator exists to avoid:
    an orange ball on a green mat and a yellow ball on yellow-green grass differ
    from their surround in *chroma*, not in lightness, so the lightness DoG was
    near zero exactly where the colour term was strongest. Measured cost:
    ``pexels_6573621`` -- a high-contrast orange ball, plainly visible -- scored
    0.0 recall.

    Both terms are now taken from the same Lab colour-distance map:

    * **distinctness** -- the distance, divided by how much distance there
      normally is nearby, so uniform turf holding one odd object scores high and
      busy foliage does not;
    * **compactness** -- a difference-of-Gaussians on that same distance map, so
      a small round region of odd colour beats a large patch or a long thin one.
    """
    distance = _colour_distance(lab, radius)
    variation = _box(distance, int(max(1, round(radius * BACKGROUND_RADII))))
    distinctness = distance / np.maximum(variation, MIN_LOCAL_VARIATION)
    sigma = max(0.8, radius / math.sqrt(2.0))
    small = cv2.GaussianBlur(distance, (0, 0), sigma)
    large = cv2.GaussianBlur(distance, (0, 0), sigma * 1.6)
    compactness = np.abs(small - large)
    return distinctness * compactness


def _pyramid_level(colour: np.ndarray, radius: float) -> tuple[np.ndarray, float]:
    """Downsample so the ball is always about ``TARGET_RADIUS_PX`` across.

    Scale space done the textbook way, and it matters twice. Running every scale
    at full resolution means an 80 px blur on a 2880 px frame, which dominated the
    runtime (6.6 s/frame measured). It is also worse numerically: a large-radius
    Gaussian on a full-resolution frame is the same computation as a small one on
    a downsampled frame, only slower and more quantised.

    This is the single-level entry point; ``_levels_for`` reuses one halving
    pyramid across all eight scales so a small level is never built from the
    largest image.
    """
    levels = bc.ImagePyramid(colour)
    factor = (TARGET_RADIUS_PX / radius) if radius > TARGET_RADIUS_PX else 1.0
    return levels.at(factor)


def candidates_local_distinctness(context: FrameContext, *,
                                  scales: Sequence[float] = SCALES,
                                  peaks_per_scale: int = 24,
                                  limit: int = 96,
                                  pyramid: bool = False) -> list[Candidate2]:
    """Small, compact, locally-distinct regions at several ball scales.

    Peaks are kept **per scale** and are not made to compete on score across
    scales, because a cross-scale score comparison is exactly what let one busy
    region spend the whole budget.

    ``pyramid=False`` reproduces the original path, where every scale resized the
    full working copy directly.
    """
    colour = context.colour
    if colour is None or colour.size == 0:
        return []
    scale = float(context.scale or 1.0)
    native_short = context.min_side or min(colour.shape[0], colour.shape[1])
    levels = bc.ImagePyramid(colour, mode="shared" if pyramid else "direct")

    out: list[Candidate2] = []
    for fraction in scales:
        native_radius = fraction * native_short
        working_radius = native_radius / max(scale, 1e-6)
        if working_radius < 1.0:
            continue
        factor = (TARGET_RADIUS_PX / working_radius
                  if working_radius > TARGET_RADIUS_PX else 1.0)
        level, level_scale = levels.at(factor)
        if min(level.shape[:2]) < 16:
            continue
        radius_px = max(1, int(round(working_radius * level_scale)))
        if radius_px * 2 >= min(level.shape[:2]):
            continue
        lab = cv2.cvtColor(level, cv2.COLOR_BGR2LAB).astype(np.float32)
        response = _response(lab, radius_px)
        response = cv2.GaussianBlur(response, (0, 0), max(0.8, radius_px * 0.5))
        top = _robust_top(response)
        if top <= 1e-6:
            continue
        # level pixels -> working pixels -> native pixels
        to_native = 1.0 / max(level_scale, 1e-6) * scale
        for x, y, value in _peaks(response, radius_px, peaks_per_scale):
            out.append(Candidate2(
                frame_index=context.index,
                x=x * to_native, y=y * to_native,
                radius=native_radius,
                # Clipped at 1.0: "far above the 99.5th percentile" and "at it"
                # are the same proposal-worthy fact, and this is an ordering
                # strength for a budget, never a probability.
                score=float(min(1.0, value / top)),
                method="local_distinctness",
                cues={"scale_fraction": fraction, "level_radius_px": radius_px,
                      "level_scale": round(level_scale, 5)}))
    return _suppress(out, limit)


def _robust_top(response: np.ndarray) -> float:
    """A high percentile, not the maximum.

    Normalising by the frame maximum was measurably wrong: on
    ``pexels_6573621`` a chain-link fence and a road sign produced a response so
    much larger than everything else that the orange ball on the mat -- plainly
    visible, high contrast -- normalised to a score below the 96th proposal and
    was dropped. One hot region must not be allowed to set the scale for the whole
    frame.
    """
    sample = response[response > 0]
    if sample.size == 0:
        return 0.0
    return float(np.percentile(sample, 99.5))
