"""Independent candidate generators for tiny golf balls.

The point of this module is *independence*. The previous single-tracker approach
failed because one method's opinion was treated as evidence. Here each generator
looks at the frame through a different physical signal -- radial symmetry, blob
contrast at ball scale, compensated frame difference, optical-flow residual,
local brightness/chroma separation -- and none of them is allowed to influence
another. A frame where four of them agree is a very different object from a frame
where one of them is excited, and the consensus stage downstream is what turns
that difference into confidence.

Nothing here decides anything. A candidate is a proposal with a method name and
its supporting cues attached, and every downstream stage keeps the method tag, so
a reviewer can always ask *who* claimed this ball.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import cv2
import numpy as np

#: Ball radius as a fraction of ``min(width, height)``. Spans the corpus: a 3 px
#: ball on a 4K wide tee shot (0.0007 of the short side would be sub-pixel, so the
#: smallest scale here is deliberately a compromise) up to a macro ball.
DEFAULT_SCALES = (0.0035, 0.006, 0.010, 0.016, 0.026, 0.042, 0.070, 0.110)

#: The dense methods run on a downscaled working copy so a 4K clip costs about the
#: same as a 1080p one. Coordinates are mapped back to native pixels, and the
#: scale is recorded, because "we detected this after downscaling by 2" is part
#: of what a result means.
WORK_MAX_SIDE = 1920

METHOD_NAMES = ("radial_symmetry", "local_contrast", "frame_difference",
                "flow_residual", "brightness_cue")


@dataclass(frozen=True)
class Candidate2:
    """One proposal from one method on one frame, in native pixels."""

    frame_index: int
    x: float
    y: float
    radius: float
    score: float
    method: str
    cues: dict = field(default_factory=dict)

    def key(self) -> tuple[str, int, int, int]:
        return (self.method, self.frame_index, int(round(self.x)), int(round(self.y)))


class FrameContext:
    """Everything the generators are allowed to see about one frame pair.

    Camera motion is **deferrable**. Only ``frame_difference`` and
    ``flow_residual`` read it, and a phase correlation on a 2880-px working copy
    measured 0.028 s/frame -- more than the entire ``local_distinctness``
    generator. When neither of those is requested the FFT should not run at all,
    so ``motion_fn`` is called at most once, on first access, and the answer is
    cached in place.
    """

    __slots__ = ("index", "gray", "colour", "previous_gray", "scale",
                 "min_side", "_motion", "_motion_fn")

    def __init__(self, index: int, gray: np.ndarray, colour: np.ndarray,
                 previous_gray: Optional[np.ndarray] = None,
                 motion: tuple[float, float] = (0.0, 0.0), scale: float = 1.0,
                 min_side: int = 0,
                 motion_fn: Optional[Any] = None) -> None:
        self.index = index
        self.gray = gray
        self.colour = colour
        self.previous_gray = previous_gray
        self.scale = scale
        self.min_side = min_side
        self._motion = motion
        self._motion_fn = motion_fn

    @property
    def motion(self) -> tuple[float, float]:
        if self._motion_fn is not None:
            self._motion = self._motion_fn()
            self._motion_fn = None
        return self._motion


def _work_scale(width: int, height: int, max_side: int = WORK_MAX_SIDE) -> float:
    return min(1.0, max_side / float(max(width, height)))


def _working(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return frame
    size = (max(16, int(round(frame.shape[1] * scale))),
            max(16, int(round(frame.shape[0] * scale))))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


#: Smallest side a pyramid level may have. Below this a level is all
#: interpolation and no signal, and `_peaks` has nothing to localise.
PYRAMID_MIN_SIDE = 48
#: Target ball radius, in level pixels, for scale-space generators. Chosen so the
#: DoG/box kernels stay small: a Gaussian with sigma 118 on a full-resolution
#: 2880-px frame is the same computation as sigma 4 on a 1/30-scale level, only
#: slower and more quantised. This is the textbook scale-space argument, and
#: `ball_v3_blobs` already relies on it.
PYRAMID_TARGET_PX = 8.0


class ImagePyramid:
    """Downsampled views of one image, for the scale-space generators.

    Three modes, because the generators did not start out the same shape and a
    comparison has to reproduce what each one actually did:

    ``shared``  successive halving, with the small levels built from an
                already-small source and cached. This is the new path.
    ``direct``  resize straight from the full image on every call, which is what
                ``local_distinctness`` used to do -- correct, but eight resizes
                of the largest image per frame.
    ``off``     always the full image at scale 1.0, which is what
                ``local_contrast``, ``brightness_cue`` and ``radial_symmetry``
                used to do (they had no pyramid at all).
    """

    MODES = ("shared", "direct", "off")

    __slots__ = ("image", "_levels", "_mode")

    def __init__(self, image: np.ndarray, mode: str = "shared") -> None:
        if mode not in self.MODES:
            raise ValueError(f"pyramid mode must be one of {self.MODES}")
        self.image = image
        self._levels: dict[float, np.ndarray] = {1.0: image}
        self._mode = mode

    def _shrink(self, source: np.ndarray, factor: float) -> np.ndarray:
        height = max(PYRAMID_MIN_SIDE, int(round(source.shape[0] * factor)))
        width = max(PYRAMID_MIN_SIDE, int(round(source.shape[1] * factor)))
        if (width, height) == (source.shape[1], source.shape[0]):
            return source
        return cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)

    def _clamp(self, factor: float) -> float:
        height, width = self.image.shape[:2]
        floor = PYRAMID_MIN_SIDE / float(max(1, min(height, width)))
        return max(min(1.0, float(factor)), min(1.0, floor))

    def at(self, factor: float) -> tuple[np.ndarray, float]:
        """A level near ``factor``, and its true scale measured from its size."""
        if self._mode == "off":
            return self.image, 1.0
        width = self.image.shape[1]
        factor = self._clamp(factor)
        if self._mode == "direct":
            level = self._shrink(self.image, factor)
            return level, level.shape[1] / float(width)
        if factor not in self._levels:
            if factor > 0.5:
                self._levels[factor] = self._shrink(self.image, factor)
            else:
                # walk down by halving from the smallest level still at or above
                # the target, so the expensive step is never taken from full res
                scale = min((s for s in self._levels if s >= factor),
                            key=lambda s: s)
                while scale / 2.0 >= factor:
                    half = scale / 2.0
                    if half not in self._levels:
                        self._levels[half] = self._shrink(self._levels[scale], 0.5)
                    scale = half
                self._levels[factor] = self._shrink(self._levels[scale], factor / scale)
        level = self._levels[factor]
        return level, level.shape[1] / float(width)


def pyramid_factor_for(radius_px: float, target: float = PYRAMID_TARGET_PX
                       ) -> float:
    """How far to downsample so ``radius_px`` becomes about ``target``."""
    if radius_px <= target:
        return 1.0
    return float(target) / float(radius_px)


def _peaks(response: np.ndarray, radius: int, limit: int) -> list[tuple[int, int, float]]:
    """Local maxima of a response map, suppressed within ``radius``.

    ``np.maximum`` rather than copy-then-mask: it is one pass instead of three,
    and it is exactly the same values. This function runs eight times per
    generator per frame, so the difference is not cosmetic.
    """
    if response.size == 0:
        return []
    working = np.maximum(response, 0)
    kernel = np.ones((max(3, radius * 2 + 1), max(3, radius * 2 + 1)), np.uint8)
    local_max = cv2.dilate(working, kernel)
    peak_mask = (working >= local_max) & (working > 0)
    ys, xs = np.nonzero(peak_mask)
    if len(xs) == 0:
        return []
    values = working[ys, xs]
    order = np.argsort(values)[::-1][:limit]
    return [(int(xs[i]), int(ys[i]), float(values[i])) for i in order]


def _normalise(value: float, half: float, full: float) -> float:
    """A saturating 0..1 score, with the two constants stated where they are used."""
    if value <= half:
        return 0.0
    if value >= full:
        return 1.0
    return float((value - half) / max(1e-6, full - half))


# ---------------------------------------------------------------------------
# 1. radial symmetry
# ---------------------------------------------------------------------------

def _radial_votes(gx: np.ndarray, gy: np.ndarray, magnitude: np.ndarray,
                  strong: np.ndarray):
    """The part of the vote field that does not depend on the radius.

    Every scale gathered the same strong-gradient pixels, the same magnitudes and
    the same unit vectors, eight times per frame. Hoisting them is exact -- the
    values are unchanged -- and the gathers were a measurable share of the
    transform.
    """
    ys, xs = np.nonzero(strong)
    if len(xs) == 0:
        return None
    gathered = magnitude[ys, xs]
    denominator = gathered + 1e-6
    mean_magnitude = float(gathered.mean()) or 1.0
    return {
        "x": xs, "y": ys,
        "ux": gx[ys, xs] / denominator,
        "uy": gy[ys, xs] / denominator,
        "strength": np.clip(gathered / mean_magnitude, 0.0, 3.0),
    }


def _radial_symmetry_response(votes, shape: tuple[int, int], radius: float,
                              level_scale: float = 1.0) -> np.ndarray:
    """Fast radial symmetry transform, in both polarities.

    Each strong-gradient pixel votes for the centre it would imply at ``radius``,
    in both directions, weighted by its gradient strength. This is the real
    transform; the first version accumulated shifted copies with ``np.roll`` on a
    coarse lattice, which localised a 4 px disc about 20 px off -- useless as a
    candidate generator, and it still scored 1.0, so the error was invisible.

    ``level_scale`` maps the vote positions onto a coarser grid. The radius is in
    full working pixels and the *whole* vote position is scaled -- scaling only
    the radius term would mix two coordinate systems and put the centres in the
    wrong place, which is a defect that would look like a recall loss rather than
    a bug.

    Accumulation uses ``np.bincount`` on a small accumulator and ``np.add.at`` on
    a large one: ``bincount`` zeroes ``minlength`` elements per call, so on a
    full-resolution frame its allocation cost exceeds the scatter it replaces.
    """
    height, width = shape
    response = np.zeros((height, width), dtype=np.float32)
    if votes is None:
        return response
    xs, ys = votes["x"], votes["y"]
    flat_size = height * width
    for sign in (-1.0, 1.0):
        vote_x = np.rint((xs + sign * radius * votes["ux"]) * level_scale
                         ).astype(np.int64)
        vote_y = np.rint((ys + sign * radius * votes["uy"]) * level_scale
                         ).astype(np.int64)
        inside = ((vote_x >= 0) & (vote_x < width)
                  & (vote_y >= 0) & (vote_y < height))
        indices = vote_y[inside] * width + vote_x[inside]
        weights = votes["strength"][inside]
        if flat_size <= 1_000_000:
            counts = np.bincount(indices, weights=weights, minlength=flat_size)
            response += counts.reshape(height, width).astype(np.float32)
        else:
            np.add.at(response.reshape(-1), indices, weights)
    return cv2.GaussianBlur(response, (0, 0),
                            max(1.0, radius * level_scale * 0.5))


def candidates_radial_symmetry(context: FrameContext, *, scales: Sequence[float] = DEFAULT_SCALES,
                               peaks_per_scale: int = 24, limit: int = 16,
                               pyramid: bool = False
                               ) -> list[Candidate2]:
    """Gradient votes for their own centre, in both senses.

    A ball lighter than its background and a ball darker than its background both
    accumulate, which matters because half the corpus is a white ball on grass and
    the other half is a yellow ball against dark turf.

    ``pyramid`` runs the coarse scales on a downsampled copy of the vote field:
    blurring a 2880-px accumulator with sigma 83 is the same computation as
    blurring a 1/10-scale one with sigma 8, only slower. ``pyramid=False`` is the
    original full-resolution path, kept so a previous pass's numbers can still be
    reproduced.
    """
    gray = context.gray
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    # "strong" is a *local* comparison: a frame-wide percentile would be a
    # scene-dependent threshold in disguise, which is the specific mistake the v1
    # pass documented.
    neighbourhood = cv2.GaussianBlur(magnitude, (0, 0), 4.0)
    strong = magnitude > np.maximum(2.0 * neighbourhood, 1e-3)
    height, width = gray.shape[:2]
    min_side = min(height, width)
    votes = _radial_votes(gx, gy, magnitude, strong)
    levels = ImagePyramid(gray, mode="shared" if pyramid else "off")

    found: list[Candidate2] = []
    for fraction in scales:
        radius = max(1, int(round(fraction * min_side)))
        if radius < 1 or radius > min_side // 3:
            continue
        # The vote field is built at level resolution by scaling the vote
        # positions: the same transform, evaluated on a coarser grid.
        _level, level_scale = levels.at(pyramid_factor_for(radius))
        level_height, level_width = _level.shape[:2]
        response = _radial_symmetry_response(
            votes, (level_height, level_width), float(radius), level_scale)
        if not (response > 0).any():
            continue
        to_working = 1.0 / max(level_scale, 1e-9)
        for x, y, value in _peaks(response, max(1, int(round(radius * level_scale))),
                                  peaks_per_scale):
            if not (radius <= x * to_working < width - radius
                    and radius <= y * to_working < height - radius):
                continue
            found.append(Candidate2(context.index,
                                    (x + 0.5) * to_working * context.scale,
                                    (y + 0.5) * to_working * context.scale,
                                    radius * context.scale,
                                    _normalise(value, 2.5, 9.0),
                                    "radial_symmetry",
                                    {"accumulator": round(value, 4),
                                     "radius_working_px": radius,
                                     "level_scale": round(level_scale, 5)}))
    found.sort(key=lambda candidate: candidate.score, reverse=True)
    return _suppress(found, limit)


# ---------------------------------------------------------------------------
# 2. local contrast (blob) at ball scale
# ---------------------------------------------------------------------------

#: Target DoG sigma, in level pixels, for ``local_contrast``. A Gaussian with
#: sigma 119 on a full-resolution 2880-px frame is the same filter as sigma 4 on a
#: 1/30-scale level. Four is chosen rather than something smaller so the two
#: finest scales -- the ones a real ball actually lands on -- are never
#: downsampled at all.
PYRAMID_TARGET_SIGMA = 4.0


def candidates_local_contrast(context: FrameContext, *,
                              scales: Sequence[float] = DEFAULT_SCALES,
                              limit: int = 16, pyramid: bool = False
                              ) -> list[Candidate2]:
    """Difference-of-Gaussian response: "there is a small disc here, lighter or
    darker than its surroundings", at the scales a ball actually occupies.

    ``pyramid=False`` reproduces the original full-resolution path, where the
    coarsest scale alone blurred a 2880-px frame with a ~700-tap separable
    kernel.
    """
    gray = cv2.GaussianBlur(context.gray, (0, 0), 0.6)
    height, width = gray.shape[:2]
    min_side = min(height, width)
    levels = ImagePyramid(gray, mode="shared" if pyramid else "off")
    found: list[Candidate2] = []
    for fraction in scales:
        sigma = max(0.8, fraction * min_side / math.sqrt(2.0))
        _level, level_scale = levels.at(
            PYRAMID_TARGET_SIGMA / sigma if sigma > PYRAMID_TARGET_SIGMA else 1.0)
        level = _level
        sigma_level = max(0.8, sigma * level_scale)
        small = cv2.GaussianBlur(level, (0, 0), sigma_level)
        large = cv2.GaussianBlur(level, (0, 0), sigma_level * 1.6)
        response = small - large
        magnitude = np.abs(response)
        # normalise per scale so one scale's units do not dominate the merge
        deviation = float(np.median(magnitude)) or 1e-3
        normalised = magnitude / (6.0 * deviation)
        to_working = 1.0 / max(level_scale, 1e-9)
        for x, y, value in _peaks(normalised, max(1, int(round(sigma_level))), 12):
            working_x, working_y = (x + 0.5) * to_working, (y + 0.5) * to_working
            if not (2 <= working_x < width - 2 and 2 <= working_y < height - 2):
                continue
            polarity = 1.0 if response[y, x] > 0 else -1.0
            found.append(Candidate2(context.index,
                                    working_x * context.scale,
                                    working_y * context.scale,
                                    sigma * math.sqrt(2.0) * context.scale,
                                    _normalise(value, 18.0, 40.0),
                                    "local_contrast",
                                    {"dog_response": round(float(response[y, x]), 3),
                                     "polarity": polarity,
                                     "sigma_working_px": round(sigma, 3),
                                     "level_scale": round(level_scale, 5)}))
    found.sort(key=lambda candidate: candidate.score, reverse=True)
    return _suppress(found, limit)


# ---------------------------------------------------------------------------
# 3. compensated frame difference
# ---------------------------------------------------------------------------

def _shift_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def candidates_frame_difference(context: FrameContext, *, limit: int = 16
                                ) -> list[Candidate2]:
    """Motion energy after removing the camera's own motion.

    Without compensation a pan lights up the whole frame edge; with it, what is
    left is the thing that moved independently -- which is what a struck ball is.
    """
    if context.previous_gray is None:
        return []
    # phaseCorrelate reports how far the *scene* moved between the frames, so the
    # previous frame is shifted by +motion to line up with the current one.
    previous = _shift_image(context.previous_gray, context.motion[0],
                            context.motion[1])
    difference = np.abs(context.gray - previous)
    blurred = cv2.GaussianBlur(difference, (0, 0), 1.0)
    deviation = float(np.median(np.abs(blurred)))
    # The scale has an absolute floor. Without it, a perfectly static pair of
    # frames divides near-zero noise by near-zero deviation and manufactures
    # confident candidates out of sensor grain -- which is how a tracker ends up
    # "detecting" a ball in an empty frame.
    scale = max(8.0 * deviation, 1.5)
    normalised = blurred / scale
    height, width = blurred.shape[:2]
    found = []
    for x, y, value in _peaks(normalised, 2, 24):
        if not (3 <= x < width - 3 and 3 <= y < height - 3):
            continue
        patch = normalised[max(0, y - 6):y + 7, max(0, x - 6):x + 7]
        if patch.size == 0:
            continue
        area = float((patch > 1.0).sum())
        radius = max(1.5, math.sqrt(max(area, 1.0) / math.pi))
        found.append(Candidate2(context.index,
                                (x + 0.5) * context.scale, (y + 0.5) * context.scale,
                                radius * context.scale,
                                _normalise(value, 1.0, 3.5), "frame_difference",
                                {"normalised_difference": round(value, 3),
                                 "area_working_px": round(area, 1),
                                 "camera_motion_working_px": [round(context.motion[0], 2),
                                                              round(context.motion[1], 2)]}))
    found.sort(key=lambda candidate: candidate.score, reverse=True)
    return _suppress(found, limit)


# ---------------------------------------------------------------------------
# 4. optical flow residual
# ---------------------------------------------------------------------------

def candidates_flow_residual(context: FrameContext, *, limit: int = 16
                             ) -> list[Candidate2]:
    """Where local motion disagrees with the global motion.

    A ball that is stationary relative to the ground produces nothing here, which
    is correct and is why this method cannot be the only one: an at-rest ball is
    exactly the case it is blind to.
    """
    if context.previous_gray is None:
        return []
    flow = cv2.calcOpticalFlowFarneback(context.previous_gray, context.gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    residual_x = flow[..., 0] - context.motion[0]
    residual_y = flow[..., 1] - context.motion[1]
    magnitude = np.sqrt(residual_x ** 2 + residual_y ** 2)
    magnitude = cv2.GaussianBlur(magnitude, (0, 0), 1.2)
    deviation = float(np.median(np.abs(magnitude)))
    # same absolute floor as frame differencing: sub-pixel flow noise on a static
    # scene must not become a candidate
    scale = max(5.0 * deviation, 0.35)
    normalised = magnitude / scale
    height, width = magnitude.shape[:2]
    found = []
    for x, y, value in _peaks(normalised, 3, 20):
        if not (4 <= x < width - 4 and 4 <= y < height - 4):
            continue
        patch = magnitude[max(0, y - 6):y + 7, max(0, x - 6):x + 7]
        area = float((patch > 1.5).sum()) if patch.size else 0.0
        radius = max(1.5, math.sqrt(max(area, 1.0) / math.pi))
        found.append(Candidate2(context.index,
                                (x + 0.5) * context.scale, (y + 0.5) * context.scale,
                                radius * context.scale,
                                _normalise(value, 1.5, 4.0), "flow_residual",
                                {"residual_working_px_per_frame":
                                     round(float(magnitude[y, x]), 3),
                                 "global_motion_working_px_per_frame":
                                     [round(context.motion[0], 2),
                                      round(context.motion[1], 2)]}))
    found.sort(key=lambda candidate: candidate.score, reverse=True)
    return _suppress(found, limit)


# ---------------------------------------------------------------------------
# 5. brightness / chroma cue
# ---------------------------------------------------------------------------

#: Target box-filter radius, in level pixels, for ``brightness_cue``. This
#: generator ran four box filters per scale on a full-resolution float32 colour
#: copy -- sixteen multi-channel passes over a 52 MB array, plus a second copy of
#: the whole frame made before the loop even started. A box filter commutes with
#: INTER_AREA downsampling for the mean it computes, so the same measurement is
#: available on a small level.
PYRAMID_TARGET_RADIUS = 5.0


def candidates_brightness_cue(context: FrameContext, *,
                              scales: Sequence[float] = DEFAULT_SCALES,
                              limit: int = 16, pyramid: bool = False
                              ) -> list[Candidate2]:
    """A locally brighter (or darker) small disc with a different colour cast.

    Included because it is the cue that catches a yellow range ball on grass,
    which is barely brighter than the turf but a long way from it in chroma.

    ``pyramid=False`` reproduces the original full-resolution path.
    """
    gray = context.gray
    height, width = gray.shape[:2]
    min_side = min(height, width)
    colour_levels = ImagePyramid(context.colour, mode="shared" if pyramid else "off")
    gray_levels = ImagePyramid(gray, mode="shared" if pyramid else "off")
    found: list[Candidate2] = []
    for fraction in scales:
        radius = max(1.0, fraction * min_side)
        factor = (PYRAMID_TARGET_RADIUS / radius
                  if radius > PYRAMID_TARGET_RADIUS else 1.0)
        _colour, level_scale = colour_levels.at(factor)
        _gray, _ = gray_levels.at(factor)
        level_colour = _colour.astype(np.float32)
        level_gray = _gray
        radius_level = max(1.0, radius * level_scale)
        kernel = max(3, int(round(radius_level * 2)) | 1)
        inner = cv2.blur(level_gray, (kernel, kernel))
        outer = cv2.blur(level_gray, (kernel * 3, kernel * 3))
        difference = inner - outer
        chroma_inner = cv2.blur(level_colour, (kernel, kernel))
        chroma_outer = cv2.blur(level_colour, (kernel * 3, kernel * 3))
        chroma_delta = np.linalg.norm(chroma_inner - chroma_outer, axis=2)
        combined = np.abs(difference) / 12.0 + chroma_delta / 22.0
        to_working = 1.0 / max(level_scale, 1e-9)
        # truncation, not rounding: this is what the pre-optimisation path did,
        # and rounding here changed the candidate set on the legacy corpus
        # (104.7 -> 112.5 proposals per frame at the same recall).
        for x, y, value in _peaks(combined, max(1, int(radius_level)), 12):
            working_x, working_y = (x + 0.5) * to_working, (y + 0.5) * to_working
            if not (3 <= working_x < width - 3 and 3 <= working_y < height - 3):
                continue
            found.append(Candidate2(context.index,
                                    working_x * context.scale,
                                    working_y * context.scale,
                                    radius * context.scale,
                                    _normalise(value, 10.0, 24.0), "brightness_cue",
                                    {"brightness_delta": round(float(difference[y, x]), 2),
                                     "chroma_delta": round(float(chroma_delta[y, x]), 2),
                                     "level_scale": round(level_scale, 5)}))
    found.sort(key=lambda candidate: candidate.score, reverse=True)
    return _suppress(found, limit)


# ---------------------------------------------------------------------------
# ensemble
# ---------------------------------------------------------------------------

def _suppress(candidates: Sequence[Candidate2], limit: int,
              overlap: float = 0.6) -> list[Candidate2]:
    """Non-maximum suppression inside one method, by circle overlap."""
    kept: list[Candidate2] = []
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        clash = False
        for other in kept:
            distance = math.hypot(candidate.x - other.x, candidate.y - other.y)
            if distance < overlap * max(candidate.radius, other.radius):
                clash = True
                break
        if not clash:
            kept.append(candidate)
        if len(kept) >= limit:
            break
    return kept


GENERATORS = {
    "radial_symmetry": candidates_radial_symmetry,
    "local_contrast": candidates_local_contrast,
    "frame_difference": candidates_frame_difference,
    "flow_residual": candidates_flow_residual,
    "brightness_cue": candidates_brightness_cue,
}


def estimate_motion(previous_gray: np.ndarray, gray: np.ndarray) -> tuple[float, float]:
    """Global translation between two frames, by phase correlation."""
    window = cv2.createHanningWindow((gray.shape[1], gray.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(previous_gray, gray, window)
    return float(dx), float(dy)


def build_contexts(frames: Sequence[np.ndarray], indices: Sequence[int], *,
                   max_side: int = WORK_MAX_SIDE, motion: str = "eager"
                   ) -> tuple[list[FrameContext], dict]:
    """One context per frame, with the camera motion each generator needs.

    ``motion="lazy"`` defers the phase correlation to first access. Only
    ``frame_difference`` and ``flow_residual`` read it, so a run of the
    colour-based generators should not pay for an FFT per frame. The default
    stays eager so that every existing caller -- including the tests that assert
    on ``median_camera_motion_fraction`` -- behaves exactly as before.
    """
    if motion not in ("eager", "lazy"):
        raise ValueError(f"motion must be 'eager' or 'lazy', not {motion!r}")
    if not frames:
        raise ValueError("at least one frame is required")
    height, width = frames[0].shape[:2]
    scale = _work_scale(width, height, max_side)
    contexts: list[FrameContext] = []
    previous_gray: Optional[np.ndarray] = None
    motions = []
    for frame, index in zip(frames, indices):
        working = _working(frame, scale)
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if previous_gray is None:
            motion_value: tuple[float, float] = (0.0, 0.0)
            motion_fn = None
        elif motion == "eager":
            motion_value = estimate_motion(previous_gray, gray)
            motion_fn = None
            motions.append(motion_value)
        else:
            motion_value = (0.0, 0.0)
            motion_fn = _deferred_motion(previous_gray, gray)
        contexts.append(FrameContext(index=index, gray=gray, colour=working,
                                     previous_gray=previous_gray,
                                     motion=motion_value, motion_fn=motion_fn,
                                     scale=1.0 / scale if scale > 0 else 1.0,
                                     min_side=min(height, width)))
        previous_gray = gray
    motion_fraction = 0.0
    if motions:
        median = float(np.median([math.hypot(dx, dy) for dx, dy in motions]))
        motion_fraction = median / float(min(height, width))
    return contexts, {
        "work_scale": scale,
        "native_size": [width, height],
        "motion_mode": motion,
        "median_camera_motion_fraction": (round(motion_fraction, 5)
                                          if motion == "eager" else None),
    }


def _deferred_motion(previous_gray: np.ndarray, gray: np.ndarray):
    """A one-shot camera-motion estimate, run only if someone asks for it."""
    def estimate() -> tuple[float, float]:
        return estimate_motion(previous_gray, gray)
    return estimate


def ensemble_candidates(contexts: Sequence[FrameContext], *,
                        methods: Sequence[str] = METHOD_NAMES,
                        per_method_limit: int = 16) -> dict[int, list[Candidate2]]:
    """Run every requested generator independently and group by frame."""
    unknown = [name for name in methods if name not in GENERATORS]
    if unknown:
        raise ValueError(f"unknown candidate methods: {unknown}")
    per_frame: dict[int, list[Candidate2]] = {context.index: [] for context in contexts}
    for name in methods:
        generator = GENERATORS[name]
        for context in contexts:
            for candidate in generator(context, limit=per_method_limit):
                per_frame[context.index].append(candidate)
    for index, candidates in per_frame.items():
        # A candidate that did not clear its own method's floor is not a
        # candidate; passing it on would let noise accumulate agreement.
        per_frame[index] = [candidate for candidate in candidates
                            if candidate.score > 0.01]
        per_frame[index].sort(key=lambda candidate: candidate.score, reverse=True)
    return per_frame
