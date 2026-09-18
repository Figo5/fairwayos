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


@dataclass
class FrameContext:
    """Everything the generators are allowed to see about one frame pair."""

    index: int
    gray: np.ndarray                     # working-scale grayscale, float32
    colour: np.ndarray                   # working-scale BGR, uint8
    previous_gray: Optional[np.ndarray] = None
    motion: tuple[float, float] = (0.0, 0.0)
    scale: float = 1.0                   # working -> native multiplier
    min_side: int = 0                    # native short side


def _work_scale(width: int, height: int, max_side: int = WORK_MAX_SIDE) -> float:
    return min(1.0, max_side / float(max(width, height)))


def _working(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return frame
    size = (max(16, int(round(frame.shape[1] * scale))),
            max(16, int(round(frame.shape[0] * scale))))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _peaks(response: np.ndarray, radius: int, limit: int) -> list[tuple[int, int, float]]:
    """Local maxima of a response map, suppressed within ``radius``."""
    if response.size == 0:
        return []
    working = response.copy()
    working[working < 0] = 0
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

def _radial_symmetry_response(gx: np.ndarray, gy: np.ndarray, magnitude: np.ndarray,
                              strong: np.ndarray, radius: int) -> np.ndarray:
    """Fast radial symmetry transform, in both polarities.

    Each strong-gradient pixel votes for the centre it would imply at ``radius``,
    in both directions, weighted by its gradient strength. This is the real
    transform; the first version accumulated shifted copies with ``np.roll`` on a
    coarse lattice, which localised a 4 px disc about 20 px off -- useless as a
    candidate generator, and it still scored 1.0, so the error was invisible.
    """
    height, width = gx.shape[:2]
    response = np.zeros((height, width), dtype=np.float32)
    ys, xs = np.nonzero(strong)
    if len(xs) == 0:
        return response
    denominator = magnitude[ys, xs] + 1e-6
    unit_x = gx[ys, xs] / denominator
    unit_y = gy[ys, xs] / denominator
    mean_magnitude = float(magnitude[ys, xs].mean()) or 1.0
    strength = np.clip(magnitude[ys, xs] / mean_magnitude, 0.0, 3.0)
    for sign in (-1.0, 1.0):
        vote_x = np.rint(xs + sign * radius * unit_x).astype(np.int64)
        vote_y = np.rint(ys + sign * radius * unit_y).astype(np.int64)
        inside = ((vote_x >= 0) & (vote_x < width)
                  & (vote_y >= 0) & (vote_y < height))
        np.add.at(response, (vote_y[inside], vote_x[inside]), strength[inside])
    return cv2.GaussianBlur(response, (0, 0), max(1.0, radius * 0.5))


def candidates_radial_symmetry(context: FrameContext, *, scales: Sequence[float] = DEFAULT_SCALES,
                               peaks_per_scale: int = 24, limit: int = 16
                               ) -> list[Candidate2]:
    """Gradient votes for their own centre, in both senses.

    A ball lighter than its background and a ball darker than its background both
    accumulate, which matters because half the corpus is a white ball on grass and
    the other half is a yellow ball against dark turf.
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

    found: list[Candidate2] = []
    for fraction in scales:
        radius = max(1, int(round(fraction * min_side)))
        if radius < 1 or radius > min_side // 3:
            continue
        response = _radial_symmetry_response(gx, gy, magnitude, strong, radius)
        for x, y, value in _peaks(response, max(1, radius), peaks_per_scale):
            if not (radius <= x < width - radius and radius <= y < height - radius):
                continue
            found.append(Candidate2(context.index,
                                    (x + 0.5) * context.scale, (y + 0.5) * context.scale,
                                    radius * context.scale,
                                    _normalise(value, 2.5, 9.0),
                                    "radial_symmetry",
                                    {"accumulator": round(value, 4),
                                     "radius_working_px": radius}))
    found.sort(key=lambda candidate: candidate.score, reverse=True)
    return _suppress(found, limit)


# ---------------------------------------------------------------------------
# 2. local contrast (blob) at ball scale
# ---------------------------------------------------------------------------

def candidates_local_contrast(context: FrameContext, *,
                              scales: Sequence[float] = DEFAULT_SCALES,
                              limit: int = 16) -> list[Candidate2]:
    """Difference-of-Gaussian response: "there is a small disc here, lighter or
    darker than its surroundings", at the scales a ball actually occupies."""
    gray = cv2.GaussianBlur(context.gray, (0, 0), 0.6)
    height, width = gray.shape[:2]
    min_side = min(height, width)
    found: list[Candidate2] = []
    for fraction in scales:
        sigma = max(0.8, fraction * min_side / math.sqrt(2.0))
        small = cv2.GaussianBlur(gray, (0, 0), sigma)
        large = cv2.GaussianBlur(gray, (0, 0), sigma * 1.6)
        response = small - large
        magnitude = np.abs(response)
        # normalise per scale so one scale's units do not dominate the merge
        deviation = float(np.median(np.abs(magnitude))) or 1e-3
        normalised = magnitude / (6.0 * deviation)
        for x, y, value in _peaks(normalised, max(1, int(round(sigma))), 12):
            if not (2 <= x < width - 2 and 2 <= y < height - 2):
                continue
            polarity = 1.0 if response[y, x] > 0 else -1.0
            found.append(Candidate2(context.index,
                                    (x + 0.5) * context.scale, (y + 0.5) * context.scale,
                                    sigma * math.sqrt(2.0) * context.scale,
                                    _normalise(value, 18.0, 40.0),
                                    "local_contrast",
                                    {"dog_response": round(float(response[y, x]), 3),
                                     "polarity": polarity,
                                     "sigma_working_px": round(sigma, 3)}))
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

def candidates_brightness_cue(context: FrameContext, *,
                              scales: Sequence[float] = DEFAULT_SCALES,
                              limit: int = 16) -> list[Candidate2]:
    """A locally brighter (or darker) small disc with a different colour cast.

    Included because it is the cue that catches a yellow range ball on grass,
    which is barely brighter than the turf but a long way from it in chroma.
    """
    colour = context.colour.astype(np.float32)
    gray = context.gray
    height, width = gray.shape[:2]
    min_side = min(height, width)
    found: list[Candidate2] = []
    for fraction in scales:
        radius = max(1.0, fraction * min_side)
        kernel = max(3, int(round(radius * 2)) | 1)
        inner = cv2.blur(gray, (kernel, kernel))
        outer = cv2.blur(gray, (kernel * 3, kernel * 3))
        difference = inner - outer
        chroma_inner = cv2.blur(colour, (kernel, kernel))
        chroma_outer = cv2.blur(colour, (kernel * 3, kernel * 3))
        chroma_delta = np.linalg.norm(chroma_inner - chroma_outer, axis=2)
        combined = np.abs(difference) / 12.0 + chroma_delta / 22.0
        for x, y, value in _peaks(combined, max(1, int(radius)), 12):
            if not (3 <= x < width - 3 and 3 <= y < height - 3):
                continue
            found.append(Candidate2(context.index,
                                    (x + 0.5) * context.scale, (y + 0.5) * context.scale,
                                    radius * context.scale,
                                    _normalise(value, 10.0, 24.0), "brightness_cue",
                                    {"brightness_delta": round(float(difference[y, x]), 2),
                                     "chroma_delta": round(float(chroma_delta[y, x]), 2)}))
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


def build_contexts(frames: Sequence[np.ndarray], indices: Sequence[int],
                   *, max_side: int = WORK_MAX_SIDE) -> tuple[list[FrameContext], dict]:
    """One context per frame, with the camera motion each generator needs."""
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
        motion = (0.0, 0.0)
        if previous_gray is not None:
            motion = estimate_motion(previous_gray, gray)
            motions.append(motion)
        contexts.append(FrameContext(index=index, gray=gray, colour=working,
                                     previous_gray=previous_gray, motion=motion,
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
        "median_camera_motion_fraction": round(motion_fraction, 5),
    }


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
