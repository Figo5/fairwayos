"""Automatic golf-ball candidate generation, association and track validation.

Five separable stages, in order:

1. ``estimate_camera_motion`` -- global per-frame translation, so that a pan is
   never mistaken for object motion, and so association can reason in a frame of
   reference that is not sliding underneath it.
2. ``generate_candidates`` -- scale-normalised, colour-agnostic proposals. Two
   independent cue families are unioned: a *radial-symmetry* search for closed
   circular boundaries, filtered by how much of the circle carries a consistent
   step edge and by how far the enclosed colour sits from its local background;
   and *camera-compensated temporal difference*, for a struck ball that is
   smeared past having a boundary at all but is the only thing moving.
3. ``score_candidate`` -- a small, explainable feature set. No per-clip colour
   prior, no per-clip radius, no per-clip ROI.
4. ``associate`` -- bounded-velocity, bounded-acceleration linking in compensated
   coordinates. Association is allowed to *refuse*: when two candidates compete
   for the same link the frame is ``AMBIGUOUS``, not a guess.
5. ``validate_track`` -- track-level signals that can reject the whole track.

Stage 4 also refuses at the *track* level, which turned out to matter more than
any appearance feature: after the best chain is found, the lattice is searched
again with that chain and its neighbourhood suppressed. If the runner-up chain
somewhere else in the frame is nearly as well supported, the window contains two
equally defensible answers and the output is that the ball was not identified.
On the development split that single rule converted every wrong-object lock into
an honest abstention, taking precision from 0.25 to 1.00 and false positives to
zero, at the cost of recall.

Every threshold is expressed relative to frame size or to local image
statistics, because this corpus spans 720p to 4K and the apparent ball radius
spans a factor of twenty-five. Absolute pixel thresholds are the specific mistake
that got the previous heuristic tracker withdrawn.

What this module deliberately does **not** do:

* It never emits a coordinate on a frame where no candidate was measured. The
  non-``OBSERVED`` states carry no position at all, so a stale coordinate cannot
  be rendered or consumed by accident.
* It never interpolates unless asked, and interpolated points are a separate
  state that is not ``OBSERVED``.
* It has no manual seed, no per-clip colour choice, and makes no network call.

Known appearance bound, stated rather than hidden: stage 2 needs the ball to
have a *resolvable circular boundary* and a colour that differs from whatever
immediately surrounds it. It does not care which is brighter, and it privileges
no colour -- an optic-yellow ball on grass is barely brighter than the grass in
luminance but is a long way from it in chroma, and a luminance detector cannot
see it at all. What this does not survive is a ball too small to have a boundary
(a few pixels across), or one blurred into the background by motion, where only
the temporal cue is left.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional CV stack
    cv2 = None


class BallState(str, Enum):
    """Per-frame track state. Only ``OBSERVED`` carries measured evidence."""

    OBSERVED = "observed"
    AMBIGUOUS = "ambiguous"
    OCCLUDED = "occluded"
    LOST = "lost"
    UNAVAILABLE = "unavailable"
    INTERPOLATED = "interpolated"


#: States that are allowed to carry a position at all.
POSITIONED_STATES = frozenset({BallState.OBSERVED, BallState.INTERPOLATED})


@dataclass(frozen=True)
class BallParams:
    """The whole tunable surface, in one place, all scale-relative.

    Kept small and readable on purpose: every field below is a quantity someone
    can argue about from first principles, not a number found by sweeping. A
    parameter set discovered by searching thousands of combinations over ten
    clips is an overfit wearing a method's clothes, and the research plan
    forbids it for that reason.
    """

    #: Candidate radii, as a fraction of ``min(width, height)``. Geometric, ratio
    #: ~1.5, spanning the 25x range of apparent ball radius in the local corpus,
    #: from six pixels on a 4K wide tee shot to a macro ball filling a quarter
    #: of the frame.
    scales: tuple[float, ...] = (0.008, 0.012, 0.018, 0.027, 0.040, 0.060, 0.090, 0.135, 0.200)
    #: Fraction of a candidate's circle that must carry a radial step edge, and
    #: the fraction at which that half of the score saturates.
    support_min: float = 0.50
    support_full: float = 0.85
    #: Minimum colour separation from the local background, in robust local
    #: deviations, for a peak to become a candidate at all.
    distinct_min: float = 0.8
    #: Colour separation at which that half of the score saturates at 1.0.
    distinct_full: float = 3.0
    #: Camera-compensated motion strength, in frame deviations, for the motion
    #: cue to propose a candidate, and the value at which it saturates.
    motion_min: float = 3.0
    motion_full: float = 8.0
    #: How far above its own neighbourhood a gradient must sit to count as an
    #: edge. Local, not a frame-wide percentile.
    edge_ratio: float = 2.0
    #: Radial-symmetry peaks examined per scale before the appearance tests.
    peaks_per_scale: int = 48
    #: Candidates retained per frame after non-maximum suppression.
    max_candidates: int = 20
    #: Largest compensated displacement per frame, as a fraction of min(w, h).
    max_step_frac: float = 0.09
    #: Largest change in per-frame displacement, as a fraction of min(w, h).
    max_accel_frac: float = 0.03
    #: Frames of missing evidence tolerated before a track is declared LOST.
    max_gap: int = 4
    #: Consecutive linked frames required before a track opens.
    min_init: int = 3
    #: Rival/best cost ratio above which a frame is AMBIGUOUS rather than a guess.
    ambiguity_ratio: float = 0.80
    #: Value ratio between the best chain and the best chain somewhere else in
    #: the frame, above which the whole track is refused as unidentified.
    track_ambiguity_ratio: float = 0.70
    #: Observed fraction of the track span below which the track is rejected.
    min_coverage: float = 0.35
    #: Mean appearance score below which the track is rejected.
    min_mean_score: float = 0.35
    #: Per-frame global translation, as a fraction of min(w, h), above which
    #: camera motion is treated as too severe to compensate confidently.
    camera_severe_frac: float = 0.03


@dataclass(frozen=True)
class Candidate:
    """One scored proposal on one frame. Not a golf-ball claim."""

    frame_index: int
    x: float
    y: float
    radius: float
    support: float
    distinct: float
    motion: float
    score: float
    cue: str


@dataclass(frozen=True)
class BallObservation:
    frame_index: int
    state: BallState
    x: Optional[float] = None
    y: Optional[float] = None
    radius: Optional[float] = None
    score: float = 0.0
    reason: str = ""

    def __post_init__(self) -> None:
        positioned = self.x is not None or self.y is not None
        if positioned and self.state not in POSITIONED_STATES:
            raise ValueError(f"{self.state.value} observations must not carry a position")
        if self.state in POSITIONED_STATES and not positioned:
            raise ValueError(f"{self.state.value} observations require a position")


@dataclass(frozen=True)
class BallTrackResult:
    state: BallState
    method: str
    observations: tuple[BallObservation, ...]
    coverage: float
    confidence_summary: dict
    diagnostics: dict
    limitations: tuple[str, ...]
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "state": self.state.value,
            "method": self.method,
            "coverage": self.coverage,
            "confidence_summary": self.confidence_summary,
            "diagnostics": self.diagnostics,
            "limitations": list(self.limitations),
            "provenance": dict(self.provenance),
            "observations": [
                {"frame_index": o.frame_index, "state": o.state.value, "x": o.x, "y": o.y,
                 "radius": o.radius, "score": o.score, "reason": o.reason}
                for o in self.observations
            ],
        }


# --------------------------------------------------------------------------
# Stage 1: camera motion
# --------------------------------------------------------------------------

def estimate_camera_motion(previous_gray: np.ndarray, gray: np.ndarray) -> tuple[float, float]:
    """Global translation from ``previous_gray`` to ``gray``, in native pixels.

    Phase correlation on a downscaled, windowed copy. It models translation
    only; a zoom or a roll shows up as a weak, inconsistent response, which the
    caller sees as low confidence rather than as a confident wrong answer.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is required for camera-motion estimation")
    height, width = gray.shape[:2]
    scale = 512.0 / max(width, height)
    if scale < 1.0:
        size = (max(16, int(round(width * scale))), max(16, int(round(height * scale))))
        small_previous = cv2.resize(previous_gray, size, interpolation=cv2.INTER_AREA)
        small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    else:
        scale, small_previous, small = 1.0, previous_gray, gray
    window = cv2.createHanningWindow(small.shape[::-1], cv2.CV_32F)
    (dx, dy), _response = cv2.phaseCorrelate(
        small_previous.astype(np.float32), small.astype(np.float32), window)
    return float(dx / scale), float(dy / scale)


def _compensate(previous_gray: np.ndarray, motion: tuple[float, float]) -> np.ndarray:
    """Shift ``previous_gray`` by ``motion`` so it lines up with the current frame."""
    matrix = np.float32([[1, 0, motion[0]], [0, 1, motion[1]]])
    return cv2.warpAffine(previous_gray, matrix, previous_gray.shape[::-1],
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------
# Stage 2 + 3: candidate generation and scoring
# --------------------------------------------------------------------------

#: Points sampled around a candidate's circle when testing its boundary.
_RING_SAMPLES = 24
#: Working resolution. Detection runs on a frame downscaled to this longest
#: edge and the result is mapped back to native pixels, so one threshold means
#: the same thing at 720p and at 4K.
WORK_MAX_EDGE = 1280


def _work_scale(width: int, height: int) -> float:
    return min(1.0, WORK_MAX_EDGE / float(max(width, height)))


def strong_edges(magnitude: np.ndarray, radius: int, ratio: float) -> np.ndarray:
    """Which gradients count as edges *here*, rather than anywhere in the frame.

    A frame-wide percentile is a scene-dependent threshold wearing a
    scale-free disguise: one lens flare or one high-contrast fence raises it
    above the ball's own boundary and the ball stops existing. Comparing each
    gradient against the mean gradient in its own neighbourhood asks the
    question that actually matters -- is this a step relative to the local
    texture -- and it is why this survives a backlit range bay and a flat
    overcast fairway with the same number.
    """
    window = max(3, 4 * radius + 1)
    local = cv2.boxFilter(magnitude, -1, (window, window), normalize=True,
                          borderType=cv2.BORDER_REFLECT)
    return magnitude > ratio * local


def _radial_symmetry(gradient_x: np.ndarray, gradient_y: np.ndarray,
                     magnitude: np.ndarray, radius: int, strong: np.ndarray) -> np.ndarray:
    """Fast radial-symmetry accumulator: strong edges vote for their own centre.

    Each strong-gradient pixel votes at a distance ``radius`` along the gradient
    direction, in both senses, so a disc lighter than its background and a disc
    darker than its background both accumulate. Voting on gradient *direction*
    rather than on brightness is what makes this work for the optic-yellow balls
    that a luminance detector cannot separate from grass at all, and for a ball
    that sits against a bright band on one side and dark turf on the other.
    """
    height, width = magnitude.shape[:2]
    ys, xs = np.nonzero(strong)
    if ys.size == 0:
        return np.zeros((height, width), np.float32)
    scale = magnitude[ys, xs]
    unit_x, unit_y = gradient_x[ys, xs] / scale, gradient_y[ys, xs] / scale
    accumulator = np.zeros(height * width, np.float32)
    for sense in (1.0, -1.0):
        cx = np.rint(xs + sense * radius * unit_x).astype(np.int64)
        cy = np.rint(ys + sense * radius * unit_y).astype(np.int64)
        inside = (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height)
        np.add.at(accumulator, cy[inside] * width + cx[inside], 1.0)
    accumulator = accumulator.reshape(height, width)
    blur = max(1, int(round(radius * 0.4)) * 2 + 1)
    return cv2.GaussianBlur(accumulator, (blur, blur), 0)


def _peaks(response: np.ndarray, radius: int, limit: int) -> list[tuple[int, int, float]]:
    """Strongest local maxima of ``response``, strongest first.

    No absolute threshold: the top ``limit`` peaks are taken and the appearance
    tests below decide which of them survive. A cut-off on accumulator height
    would be a resolution-dependent magic number of exactly the kind this
    module exists to avoid.
    """
    size = max(3, 2 * radius + 1)
    dilated = cv2.dilate(response, np.ones((size, size), np.uint8))
    mask = (response >= dilated) & (response > 0)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return []
    values = response[ys, xs]
    order = np.argsort(-values)[:limit]
    return [(int(xs[i]), int(ys[i]), float(values[i])) for i in order]


def edge_support(gradient_x: np.ndarray, gradient_y: np.ndarray, strong: np.ndarray,
                 x: float, y: float, radius: float) -> float:
    """How much of the candidate's circle is actually a boundary.

    Walks the annulus at the candidate radius, keeps only strong gradients whose
    direction is radial with respect to the candidate centre, and reports the
    fraction of angular sectors that contain at least one such gradient.

    This is the feature that separates a ball from the linear bright structures
    that produced the original flagstick-base falsification: a flagstick, a club
    shaft or the edge of a bright band has radial gradients in only two opposed
    sectors, however bright it is, while a ball has them all the way round.
    Crucially it tests for a step in *either* direction. Requiring the ball to be
    the brighter side, which the first version of this did, throws away every
    ball that sits against a bright band, a sky or a sand line -- which in this
    corpus is most of them.
    """
    height, width = strong.shape[:2]
    outer = int(math.ceil(1.25 * radius)) + 1
    x0, y0 = max(0, int(x) - outer), max(0, int(y) - outer)
    x1, y1 = min(width, int(x) + outer + 1), min(height, int(y) + outer + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
    offset_x, offset_y = grid_x - x, grid_y - y
    distance = np.hypot(offset_x, offset_y)
    ring = (distance >= 0.75 * radius) & (distance <= 1.25 * radius) & strong[y0:y1, x0:x1]
    if not ring.any():
        return 0.0
    dx, dy = offset_x[ring] / (distance[ring] + 1e-6), offset_y[ring] / (distance[ring] + 1e-6)
    patch_x, patch_y = gradient_x[y0:y1, x0:x1][ring], gradient_y[y0:y1, x0:x1][ring]
    scale = np.hypot(patch_x, patch_y) + 1e-6
    radial = np.abs((patch_x * dx + patch_y * dy) / scale)
    aligned = radial > 0.7
    if not aligned.any():
        return 0.0
    angle = np.arctan2(offset_y[ring][aligned], offset_x[ring][aligned])
    sector = ((angle + math.pi) / (2 * math.pi) * _RING_SAMPLES).astype(int) % _RING_SAMPLES
    return float(np.unique(sector).size) / float(_RING_SAMPLES)


def distinctness(colour: np.ndarray, x: float, y: float, radius: float) -> float:
    """How far the candidate's own colour sits from its *most similar* neighbour.

    The annulus is split into eight sectors and the score is the smallest
    separation across them, in robust deviations of that sector. Taking the
    minimum rather than the mean is what makes this usable: a ball routinely has
    sky, a sand line or a bright band behind it on one side, which would
    flatter any averaged comparison, so the honest question is whether the
    candidate still differs from the part of its surround it resembles most.

    Colour, not luminance. The optic-yellow balls in this corpus are barely
    brighter than the grass they sit on but are a long way from it in chroma, so
    a luminance detector cannot see them at all. No colour is privileged here --
    only the difference from whatever happens to surround this candidate, which
    is what "do not assume the ball is white" has to mean in code.
    """
    height, width = colour.shape[:2]
    outer = int(math.ceil(2.2 * radius)) + 1
    x0, y0 = max(0, int(x) - outer), max(0, int(y) - outer)
    x1, y1 = min(width, int(x) + outer + 1), min(height, int(y) + outer + 1)
    patch = colour[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
    offset_x, offset_y = grid_x - x, grid_y - y
    distance = np.hypot(offset_x, offset_y)
    core = distance <= max(1.0, 0.6 * radius)
    ring = (distance >= 1.4 * radius) & (distance <= 2.2 * radius)
    if not core.any() or not ring.any():
        return 0.0
    core_colour = np.median(patch[core], axis=0)
    angle = np.arctan2(offset_y, offset_x)
    sector = ((angle + math.pi) / (2 * math.pi) * 8).astype(int) % 8
    distances: list[float] = []
    scales: list[float] = []
    for index in range(8):
        selected = ring & (sector == index)
        if np.count_nonzero(selected) < 4:
            continue
        pixels = patch[selected]
        sector_colour = np.median(pixels, axis=0)
        spread = np.median(np.abs(pixels - sector_colour), axis=0) * 1.4826
        distances.append(float(np.linalg.norm(core_colour - sector_colour)))
        scales.append(float(np.linalg.norm(np.maximum(spread, 1.0))))
    if not distances:
        return 0.0
    # Smallest separation, but measured against the *typical* local noise rather
    # than against each sector's own. Dividing each sector by itself lets one
    # noisy sector -- a shadow line, a grass-to-sand boundary -- decide the whole
    # score, which is how a plainly visible white ball ends up scoring 1.2.
    return float(min(distances) / max(1e-6, float(np.median(scales))))


def _measurable(x: float, y: float, radius: float, width: int, height: int) -> bool:
    """Is this candidate far enough from the frame edge to be judged at all?

    Both appearance tests need the full background annulus. A candidate hanging
    off the edge of the frame gets them computed from whatever happens to be in
    view, which is how a golf shoe half out of shot scores higher than the ball.
    A candidate whose background cannot be measured is not a candidate.
    """
    margin = 2.2 * radius
    return margin <= x < width - margin and margin <= y < height - margin


def score_candidate(support: float, distinct: float, params: BallParams) -> float:
    """Combine boundary closure and colour separation into a bounded score.

    Both must hold: a closed circular boundary around something that looks like
    its background is a divot or a grass clump, and a strongly distinct patch
    without a closed boundary is a shoe, a logo or a shadow edge.
    """
    closure = min(1.0, max(0.0, (support - params.support_min) / max(1e-6, params.support_full - params.support_min)))
    separation = min(1.0, max(0.0, distinct / params.distinct_full))
    return float(closure * separation)


def generate_candidates(gray: np.ndarray, frame_index: int, params: BallParams,
                        motion_map: Optional[np.ndarray] = None,
                        colour: Optional[np.ndarray] = None,
                        native_scale: float = 1.0) -> list[Candidate]:
    """Propose every plausible ball-sized circular object on one frame.

    Recall-oriented and deliberately over-generating: rejecting proposals is the
    job of stages 4 and 5, which can see time and therefore see more than any
    single frame can. Coordinates come back in *native* pixels even though
    detection ran on a downscaled working frame.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is required for candidate generation")
    height, width = gray.shape[:2]
    base = float(min(width, height))
    reference = colour if colour is not None else gray[..., None]

    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gradient_x, gradient_y) + 1e-6

    found: list[Candidate] = []
    for fraction in params.scales:
        radius = int(round(fraction * base))
        if radius < 2 or radius * 3 >= min(width, height):
            continue
        strong = strong_edges(magnitude, radius, params.edge_ratio)
        accumulator = _radial_symmetry(gradient_x, gradient_y, magnitude, radius, strong)
        for x, y, vote in _peaks(accumulator, radius, params.peaks_per_scale):
            if not _measurable(x, y, radius, width, height):
                continue
            support = edge_support(gradient_x, gradient_y, strong, x, y, radius)
            if support < params.support_min:
                continue
            distinct = distinctness(reference, x, y, radius)
            if distinct < params.distinct_min:
                continue
            motion = 0.0
            if motion_map is not None:
                x0, y0 = max(0, x - radius), max(0, y - radius)
                x1, y1 = min(width, x + radius + 1), min(height, y + radius + 1)
                motion = float(motion_map[y0:y1, x0:x1].mean())
            found.append(Candidate(
                frame_index, x / native_scale, y / native_scale, radius / native_scale,
                support, distinct, motion, score_candidate(support, distinct, params), "circle"))

    # Second cue family: whatever is moving once the camera is compensated away.
    # A struck ball is smeared and has no closed boundary left, so the circle cue
    # alone can miss it entirely; this is the only cue that still sees it.
    if motion_map is not None:
        floor = float(np.median(motion_map))
        spread = float(motion_map.std()) + 1e-6
        for fraction in params.scales[:5]:
            radius = int(round(fraction * base))
            if radius < 2 or radius * 3 >= min(width, height):
                continue
            blurred = cv2.GaussianBlur(motion_map, (0, 0), radius * 0.6)
            for x, y, value in _peaks(blurred, radius, params.peaks_per_scale):
                if not _measurable(x, y, radius, width, height):
                    continue
                strength = (value - floor) / spread
                if strength < params.motion_min:
                    continue
                distinct = distinctness(reference, x, y, radius)
                found.append(Candidate(
                    frame_index, x / native_scale, y / native_scale, radius / native_scale,
                    0.0, distinct, strength, min(1.0, strength / params.motion_full) * 0.6, "motion"))

    return _suppress(found, params.max_candidates)


def _suppress(candidates: Sequence[Candidate], limit: int) -> list[Candidate]:
    """Non-maximum suppression across scales and cue families."""
    kept: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda c: -c.score):
        if all(math.hypot(candidate.x - other.x, candidate.y - other.y) > max(candidate.radius, other.radius)
               for other in kept):
            kept.append(candidate)
        if len(kept) >= limit:
            break
    return kept


# --------------------------------------------------------------------------
# Stage 4: temporal association
# --------------------------------------------------------------------------

def _stabilised(per_frame: Sequence[tuple[int, list[Candidate]]],
                motions: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Cumulative camera displacement, so positions can be compared like-for-like.

    Subtracting this from a candidate's pixel position gives a coordinate in a
    frame of reference that is not sliding underneath the ball, which is the
    whole point of stage 1: a pan must not manufacture velocity, and must not
    blow the displacement gate on a ball that is in fact sitting still.
    """
    offsets = [(0.0, 0.0)]
    for index in range(1, len(per_frame)):
        drift = motions[index] if index < len(motions) else (0.0, 0.0)
        offsets.append((offsets[-1][0] + drift[0], offsets[-1][1] + drift[1]))
    return offsets


def _transition_cost(first: Candidate, second: Candidate, gap: int, gate: float) -> float:
    """Cost of claiming two candidates are the same ball ``gap`` frames apart."""
    distance = math.hypot(second.x - first.x, second.y - first.y)
    if distance > gate * gap:
        return math.inf
    size = abs(math.log((second.radius + 1e-6) / (first.radius + 1e-6)))
    if size > 0.9:  # a ball does not change apparent size by more than ~2.5x
        return math.inf
    return distance / (gate * gap) + 0.6 * size


def associate(per_frame: Sequence[tuple[int, list[Candidate]]],
              motions: Sequence[tuple[float, float]], params: BallParams,
              *, width: int, height: int) -> list[BallObservation]:
    """Link per-frame candidates into at most one track, refusing when unsure.

    A Viterbi pass over the whole candidate lattice rather than a greedy walk
    from the first frame. Greedy opening is how a tracker ends up following a
    lens flare: the flare outscores the ball on frame one, the track locks, and
    nothing later can dislodge it. Searching the lattice instead asks which
    chain of candidates is best *over the window*, so a modest candidate that
    persists coherently beats a brilliant one that goes nowhere.

    Gaps are first-class: a chain may skip up to ``max_gap`` frames, paying for
    it, which is how a brief occlusion stays one track instead of becoming two.
    """
    base = float(min(width, height))
    gate = params.max_step_frac * base
    offsets = _stabilised(per_frame, motions)

    # Stabilised copies for the lattice; native coordinates are kept for output.
    nodes: list[list[Candidate]] = []
    for position, (_index, candidates) in enumerate(per_frame):
        shift = offsets[position]
        nodes.append([replace(c, x=c.x - shift[0], y=c.y - shift[1]) for c in candidates])

    def search(blocked: set[tuple[int, int]]) -> tuple[dict[int, int], float]:
        """Best chain through the lattice, avoiding ``blocked`` nodes."""
        best_value: list[list[float]] = []
        back: list[list[Optional[tuple[int, int]]]] = []
        for position, candidates in enumerate(nodes):
            best_value.append([-math.inf if (position, i) in blocked else c.score
                               for i, c in enumerate(candidates)])
            back.append([None] * len(candidates))
            for index, candidate in enumerate(candidates):
                if (position, index) in blocked:
                    continue
                for gap in range(1, params.max_gap + 2):
                    previous = position - gap
                    if previous < 0:
                        break
                    for other, earlier in enumerate(nodes[previous]):
                        if (previous, other) in blocked or not math.isfinite(best_value[previous][other]):
                            continue
                        cost = _transition_cost(earlier, candidate, gap, gate)
                        if not math.isfinite(cost):
                            continue
                        # Each linked frame is worth its score; each skipped
                        # frame costs, so a chain never buys length with
                        # invented support.
                        value = best_value[previous][other] + candidate.score - cost - 0.35 * (gap - 1)
                        if value > best_value[position][index]:
                            best_value[position][index] = value
                            back[position][index] = (previous, other)

        found: dict[int, int] = {}
        best = -math.inf
        seed = None
        for position, row in enumerate(best_value):
            for index, value in enumerate(row):
                if value > best:
                    best, seed = value, (position, index)
        if seed is None or not math.isfinite(best):
            return {}, -math.inf
        position, index = seed
        while position is not None:
            found[position] = index
            step = back[position][index]
            position, index = step if step else (None, None)
        return found, best

    chain, value = search(set())
    if len(chain) < params.min_init:
        return [BallObservation(index, BallState.LOST,
                                reason="no candidate chain long enough to open a track")
                for index, _ in per_frame]

    # Track-level candidate competition. Suppress the accepted chain and every
    # node near it, then look for the next-best chain elsewhere in the frame. If
    # that rival is nearly as good, the window contains two equally defensible
    # answers and the honest output is that the ball was not identified -- not
    # whichever one happened to win by a hair. A wrong-object lock is worse than
    # an unavailable result, so this trades recall for the right kind of silence.
    blocked = set()
    for position, index in chain.items():
        chosen = nodes[position][index]
        for other, candidate in enumerate(nodes[position]):
            if math.hypot(candidate.x - chosen.x, candidate.y - chosen.y) <= 3.0 * gate:
                blocked.add((position, other))
    rival_chain, rival_value = search(blocked)
    if (len(rival_chain) >= params.min_init and value > 0
            and rival_value / value > params.track_ambiguity_ratio):
        return [BallObservation(
            index, BallState.AMBIGUOUS,
            reason="a second, equally supported chain elsewhere in the frame competed "
                   "with this one; the window does not identify the ball")
            for index, _ in per_frame]

    first, last = min(chain), max(chain)
    observations: list[BallObservation] = []
    for position, (frame_index, candidates) in enumerate(per_frame):
        if position not in chain:
            if position < first or position > last:
                observations.append(BallObservation(
                    frame_index, BallState.LOST,
                    reason="outside the span of the accepted track"))
            else:
                observations.append(BallObservation(
                    frame_index, BallState.OCCLUDED,
                    reason="no candidate linked to the track on this frame"))
            continue
        chosen = candidates[chain[position]]
        rival = _closest_rival(nodes[position], chain[position], gate)
        if rival is not None and rival >= params.ambiguity_ratio:
            observations.append(BallObservation(
                frame_index, BallState.AMBIGUOUS,
                reason="a second candidate of comparable score sat inside the motion gate"))
            continue
        observations.append(BallObservation(frame_index, BallState.OBSERVED, chosen.x, chosen.y,
                                            chosen.radius, chosen.score, ""))
    return observations


def _closest_rival(candidates: Sequence[Candidate], chosen: int, gate: float) -> Optional[float]:
    """Score ratio of the best rival sitting inside the gate, if there is one.

    Competition on a frame is itself evidence: two equally good candidates a
    ball's-width apart mean the frame does not identify the ball, and saying so
    is worth more than picking one and being right half the time.
    """
    target = candidates[chosen]
    if target.score <= 0:
        return None
    best: Optional[float] = None
    for index, candidate in enumerate(candidates):
        if index == chosen:
            continue
        if math.hypot(candidate.x - target.x, candidate.y - target.y) > gate:
            continue
        ratio = candidate.score / target.score
        best = ratio if best is None else max(best, ratio)
    return best


# --------------------------------------------------------------------------
# Stage 5: track validation and state output
# --------------------------------------------------------------------------

def validate_track(observations: Sequence[BallObservation], params: BallParams,
                   *, width: int, height: int,
                   camera_severity: float) -> tuple[BallState, dict, tuple[str, ...]]:
    """Decide whether the linked track is evidence at all."""
    base = float(min(width, height))
    observed = [o for o in observations if o.state is BallState.OBSERVED]
    span = len(observations)
    coverage = len(observed) / span if span else 0.0
    mean_score = float(np.mean([o.score for o in observed])) if observed else 0.0

    jumps = 0
    static_pairs = 0
    pairs = 0
    for previous, current in zip(observed, observed[1:]):
        if current.frame_index != previous.frame_index + 1:
            continue
        pairs += 1
        step = math.hypot(current.x - previous.x, current.y - previous.y)
        if step > params.max_step_frac * base:
            jumps += 1
        if step < 1.0:
            static_pairs += 1

    diagnostics = {
        "coverage": coverage,
        "mean_score": mean_score,
        "observed_frames": len(observed),
        "span_frames": span,
        "abrupt_jumps": jumps,
        "static_pair_fraction": (static_pairs / pairs) if pairs else 0.0,
        "camera_severity": camera_severity,
    }
    limitations: list[str] = []
    if camera_severity > params.camera_severe_frac:
        limitations.append(
            "global camera translation exceeded the compensation bound; positions are "
            "reported with reduced confidence")
    if not observed:
        # Distinguish "nothing was ever proposed" from "association refused to
        # choose". Collapsing both into one message hides which half of the
        # pipeline declined, which is the first thing anyone debugging this
        # needs to know.
        refusals = {o.reason for o in observations if o.state is BallState.AMBIGUOUS and o.reason}
        if refusals:
            return BallState.AMBIGUOUS, diagnostics, tuple(limitations + sorted(refusals))
        return BallState.UNAVAILABLE, diagnostics, tuple(limitations + ["no candidate survived association"])
    if coverage < params.min_coverage:
        return BallState.UNAVAILABLE, diagnostics, tuple(
            limitations + [f"observation coverage {coverage:.2f} below the {params.min_coverage} bound"])
    if mean_score < params.min_mean_score:
        return BallState.UNAVAILABLE, diagnostics, tuple(
            limitations + [f"mean candidate score {mean_score:.2f} below the {params.min_mean_score} bound"])
    # A golf ball at address is *supposed* to be motionless for seconds at a
    # time, so sustained stationarity is not evidence of a wrong-object lock the
    # way it is for a clubhead. It is only suspicious when nothing else supports
    # the track, which the coverage and score bounds above already test.
    return BallState.OBSERVED, diagnostics, tuple(limitations)


# --------------------------------------------------------------------------
# Whole pipeline
# --------------------------------------------------------------------------

def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return frame.astype(np.float32)


def track_ball(frames: Sequence[np.ndarray], frame_indices: Sequence[int],
               params: Optional[BallParams] = None, *,
               method: str = "fairwayos-ball-v1",
               provenance: Optional[dict] = None) -> BallTrackResult:
    """Run all five stages over consecutive frames and return an honest result."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required for ball tracking")
    if len(frames) != len(frame_indices):
        raise ValueError("frames and frame_indices must be the same length")
    if not frames:
        raise ValueError("at least one frame is required")
    params = params or BallParams()

    height, width = frames[0].shape[:2]
    base = float(min(width, height))
    # Detect on a fixed working resolution so that one set of thresholds means
    # the same thing on a 720p phone clip and on a 4K wide tee shot, then map
    # every coordinate back to native pixels.
    scale = _work_scale(width, height)
    if scale < 1.0:
        size = (max(16, int(round(width * scale))), max(16, int(round(height * scale))))
        working = [cv2.resize(frame, size, interpolation=cv2.INTER_AREA) for frame in frames]
    else:
        scale, working = 1.0, list(frames)
    grays = [_gray(frame) for frame in working]
    colours = [frame.astype(np.float32) if frame.ndim == 3 else frame[..., None].astype(np.float32)
               for frame in working]

    motions: list[tuple[float, float]] = [(0.0, 0.0)]
    motion_maps: list[Optional[np.ndarray]] = [None]
    for previous, current in zip(grays, grays[1:]):
        motion = estimate_camera_motion(previous, current)
        motions.append((motion[0] / scale, motion[1] / scale))
        motion_maps.append(np.abs(current - _compensate(previous, motion)))
    severity = float(np.median([math.hypot(*m) for m in motions[1:]]) / base) if len(motions) > 1 else 0.0

    per_frame = [
        (index, generate_candidates(gray, index, params, motion_map, colour, scale))
        for index, gray, motion_map, colour in zip(frame_indices, grays, motion_maps, colours)
    ]
    observations = associate(per_frame, motions, params, width=width, height=height)
    state, diagnostics, limitations = validate_track(
        observations, params, width=width, height=height, camera_severity=severity)

    if state is not BallState.OBSERVED:
        # Fail closed: a refused track publishes no positions at all, so there
        # is nothing for a renderer to draw from a rejected hypothesis. Each
        # frame keeps its own reason where it had one.
        observations = tuple(
            BallObservation(o.frame_index, state,
                            reason=o.reason or (limitations[-1] if limitations else ""))
            for o in observations)
    else:
        observations = tuple(observations)

    scores = [o.score for o in observations if o.state is BallState.OBSERVED]
    return BallTrackResult(
        state=state,
        method=method,
        observations=observations,
        coverage=diagnostics["coverage"],
        confidence_summary={
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "min_score": float(np.min(scores)) if scores else 0.0,
            "observed_frames": len(scores),
        },
        diagnostics={**diagnostics, "candidates_per_frame": [len(c) for _, c in per_frame]},
        limitations=limitations,
        provenance=dict(provenance or {}),
    )


__all__ = [
    "BallState", "BallParams", "Candidate", "BallObservation", "BallTrackResult",
    "estimate_camera_motion", "generate_candidates", "score_candidate",
    "associate", "validate_track", "track_ball",
]
