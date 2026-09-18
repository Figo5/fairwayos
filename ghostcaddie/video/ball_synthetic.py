"""Synthetic golf-ball ground truth on real golf backgrounds.

The problem this solves: there are no human labels, and an agent must not invent
them. Inserting a ball into real footage is different -- the position is known
*because the generator put it there*. That is legitimate supervision, and it is
labelled ``SYNTHETIC_GROUND_TRUTH`` everywhere it appears, never "ground truth"
without qualification.

What "realistic" means here, and why it matters: a perfect white circle pasted on
grass is a trivial detection problem and teaches a model nothing about the real
failure mode. A golf ball on a course is a few pixels across, shaded, partially
blurred by its own motion, often partly hidden, sitting next to grass highlights
and clothing that are brighter than it is. So the generator renders a shaded
sphere with a specular highlight and a contact shadow, applies sub-pixel
placement, directional motion blur bounded by the frame's own velocity, JPEG-like
degradation, and occlusion by a real patch taken from the same frame.

Insertion is confined to TRAIN clips. This module refuses a clip that is not in
the training split unless it is told explicitly that the caller is generating a
sealed-evaluation set, so a background from validation or final test cannot leak
into training by accident.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np

ORIGIN_SYNTHETIC = "SYNTHETIC_GROUND_TRUTH"

#: Ball surface colours seen on a course, as BGR. Yellow range balls are
#: deliberately included: the v1 corpus' one success was an optic-yellow ball and
#: that is not representative of white-ball footage.
BALL_COLOURS = {
    "white": (242, 242, 242),
    "off_white": (225, 226, 214),
    "dirty_white": (208, 208, 196),
    "yellow": (70, 220, 214),
    "range_orange": (90, 180, 235),
}

VISIBILITY_STATES = ("VISIBLE", "PARTIAL", "OCCLUDED")


@dataclass(frozen=True)
class BallAppearance:
    colour: str = "white"
    radius_px: float = 6.0
    brightness: float = 1.0
    shading: float = 0.35
    specular: float = 0.55
    blur_px: float = 0.0
    jpeg_quality: int = 0            # 0 = leave the frame's compression alone
    noise: float = 2.0
    shadow: float = 0.35
    dimple: float = 0.05

    def bgr(self) -> tuple[int, int, int]:
        base = BALL_COLOURS[self.colour]
        return tuple(int(min(255, max(0, channel * self.brightness))) for channel in base)


@dataclass(frozen=True)
class BallSample:
    """One inserted ball and everything known about it."""

    frame_index: int
    x: float
    y: float
    radius_px: float
    visibility: str
    occlusion_fraction: float
    blur_px: float
    colour: str
    seed: int
    origin: str = ORIGIN_SYNTHETIC
    #: Verification of the insertion itself: where the frame actually changed
    #: versus where the label claims the ball is. A synthetic label is only worth
    #: as much as this check.
    changed_pixels: int = 0
    centroid_error_px: Optional[float] = None
    verified: bool = False
    unverified_reason: str = ""

    def as_label(self) -> dict:
        return {
            "state": "VISIBLE" if self.visibility == "VISIBLE" else (
                "OCCLUDED" if self.visibility == "OCCLUDED" else "AMBIGUOUS"),
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "radius": round(self.radius_px, 3),
            "origin": self.origin,
            "visibility": self.visibility,
            "occlusion_fraction": round(self.occlusion_fraction, 3),
            "blur_px": round(self.blur_px, 2),
            "colour": self.colour,
            "seed": self.seed,
            "insertion_verified": self.verified,
            "insertion_centroid_error_px": self.centroid_error_px,
            "insertion_unverified_reason": self.unverified_reason or None,
        }


@dataclass
class TrajectorySpec:
    kind: str = "linear"           # stationary | linear | accelerating | entrance | exit
    vx: float = 0.0                # px per frame
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    occlusion_from: float = -1.0   # fraction of the sequence, -1 = never
    occlusion_to: float = -1.0
    occlusion_fraction: float = 0.0


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render_ball_patch(appearance: BallAppearance, *, supersample: int = 4) -> np.ndarray:
    """Render one shaded golf ball as a BGRA patch with a soft, anti-aliased edge.

    Rendered at ``supersample`` times the target size and box-filtered down, so a
    ball of radius 3 px still has a plausible gradient instead of a hard pixel
    block -- which is what a nearest-neighbour circle would give and what makes a
    synthetic dataset detectable by its own artefacts.
    """
    radius = max(0.75, float(appearance.radius_px))
    # enough room for blur, shadow and the highlight
    pad = 2.0 + appearance.blur_px * 1.2 + radius * 0.6
    size = int(math.ceil((radius + pad) * 2))
    scale = supersample
    big = size * scale
    centre = big / 2.0
    radius_big = radius * scale

    ys, xs = np.mgrid[0:big, 0:big].astype(np.float32)
    dx, dy = xs - centre, ys - centre
    distance = np.sqrt(dx * dx + dy * dy)
    alpha = np.clip((radius_big - distance) / max(1.0, 1.2 * scale), 0.0, 1.0)

    # diffuse shading: light from the upper left, so the ball is not flat
    normal_z = np.sqrt(np.clip(1.0 - (distance / max(radius_big, 1e-3)) ** 2, 0.0, 1.0))
    light = np.clip(0.55 + 0.45 * (-dx - dy) / max(radius_big, 1e-3) * 0.5, 0.0, 1.0)
    shade = (1.0 - appearance.shading) + appearance.shading * light * (0.6 + 0.4 * normal_z)
    base = np.array(appearance.bgr(), dtype=np.float32)
    patch = np.zeros((big, big, 3), dtype=np.float32) + base[None, None, :] * shade[..., None]

    # specular highlight, offset up-left, small relative to the ball
    highlight_radius = max(0.6, radius_big * 0.28) * scale / supersample
    hx, hy = centre - radius_big * 0.35, centre - radius_big * 0.4
    highlight = np.exp(-(((xs - hx) ** 2 + (ys - hy) ** 2) / (2 * highlight_radius ** 2)))
    specular = np.clip(highlight * appearance.specular * 255.0, 0, 255)
    patch = np.clip(patch + specular[..., None], 0, 255)

    if appearance.dimple > 0:
        # a faint dimple texture; without it every synthetic ball is a smooth disc
        rng = np.random.default_rng(1234)
        texture = rng.normal(0.0, 1.0, size=(big, big)).astype(np.float32)
        texture = cv2.GaussianBlur(texture, (0, 0), max(1.0, radius_big * 0.18))
        patch = np.clip(patch * (1.0 + texture[..., None] * appearance.dimple * 6.0),
                        0, 255)

    if appearance.blur_px > 0.05:
        length = int(max(3, round(appearance.blur_px * supersample * 1.5)))
        if length % 2 == 0:
            length += 1
        kernel = np.zeros((length, length), dtype=np.float32)
        cv2.line(kernel, (0, length // 2), (length - 1, length // 2), 1.0,
                 thickness=max(1, int(round(radius_big * 0.6))))
        kernel /= max(kernel.sum(), 1e-6)
        patch = cv2.filter2D(patch, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (size, size), interpolation=cv2.INTER_AREA)
    return np.dstack([patch, alpha * 255.0]).astype(np.uint8)


def _composite(frame: np.ndarray, patch: np.ndarray, centre_x: float,
               centre_y: float) -> np.ndarray:
    """Alpha-composite a patch at a *sub-pixel* position, clipped to the frame."""
    height, width = frame.shape[:2]
    size = patch.shape[0]
    left = int(round(centre_x - size / 2.0))
    top = int(round(centre_y - size / 2.0))
    sub_x = left - (centre_x - size / 2.0)
    sub_y = top - (centre_y - size / 2.0)
    # shift by the fractional part instead of rounding it away
    matrix = np.float32([[1, 0, sub_x], [0, 1, sub_y]])
    patch = cv2.warpAffine(patch, matrix, (size, size), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)

    x0, x1 = max(0, left), min(width, left + size)
    y0, y1 = max(0, top), min(height, top + size)
    if x0 >= x1 or y0 >= y1:
        return frame
    patch_x0, patch_y0 = x0 - left, y0 - top
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    region = patch[patch_y0:patch_y0 + (y1 - y0), patch_x0:patch_x0 + (x1 - x0)]
    alpha = (region[..., 3:4] / 255.0)
    blended = roi * (1.0 - alpha) + region[..., :3].astype(np.float32) * alpha
    out = frame.copy()
    out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def _occlude(frame: np.ndarray, centre_x: float, centre_y: float, radius: float,
             fraction: float, rng: random.Random) -> np.ndarray:
    """Hide part of the ball behind a real patch taken from the same frame.

    A synthetic flat occluder would be another artefact to learn. A real patch --
    turf, a trouser leg, a gloved hand -- has the right texture and the right
    colour for the scene, which is the point.
    """
    if fraction <= 0.01:
        return frame
    height, width = frame.shape[:2]
    size = int(max(6, round(radius * 3)))
    for _ in range(8):
        sx = rng.randint(0, max(0, width - size - 1))
        sy = rng.randint(0, max(0, height - size - 1))
        if abs(sx + size / 2 - centre_x) < radius * 3 and \
                abs(sy + size / 2 - centre_y) < radius * 3:
            continue                                    # do not draw the ball's own pixels back
        donor = frame[sy:sy + size, sx:sx + size].copy()
        cover = max(2, int(round(radius * 2.4)))
        cover = min(cover, size)
        y0 = int(round(centre_y - cover / 2 - radius * 0.6))
        x0 = int(round(centre_x - cover / 2 - radius * 0.6))
        y0 = max(0, min(height - cover - 1, y0))
        x0 = max(0, min(width - cover - 1, x0))
        donor_piece = donor[:cover, :cover]
        mask = np.zeros((cover, cover), dtype=np.uint8)
        # cover a fraction of the ball: a slab whose thickness scales with it
        thickness = int(max(1, round(cover * (0.35 + 0.4 * fraction))))
        mask[:thickness, :] = 255
        mask = cv2.GaussianBlur(mask, (0, 0), 1.2)
        out = frame.copy()
        target = out[y0:y0 + cover, x0:x0 + cover].astype(np.float32)
        mask_f = (mask[..., None] / 255.0).astype(np.float32)
        out[y0:y0 + cover, x0:x0 + cover] = np.clip(
            target * (1 - mask_f) + donor_piece.astype(np.float32) * mask_f,
            0, 255).astype(np.uint8)
        return out
    return frame


def _degrade(frame: np.ndarray, quality: int, noise: float, rng: random.Random,
             *, around: Optional[tuple[float, float, float]] = None) -> np.ndarray:
    """Degrade the ball's own neighbourhood, not the whole frame.

    The first version added Gaussian grain and a fresh JPEG pass to every pixel of
    every frame. That is both unrealistic (the source clip already has one
    compression history) and destructive to measurement: a whole-frame diff is no
    longer evidence about where the ball is, because every pixel changed. Local
    degradation keeps the insertion position measurable while still presenting the
    detector with grain and compression artefacts.
    """
    out = frame
    if around is not None and (noise > 0.01 or quality):
        cx, cy, radius = around
        window = int(max(8, radius * 6))
        x0 = max(0, int(cx) - window)
        y0 = max(0, int(cy) - window)
        x1 = min(frame.shape[1], int(cx) + window)
        y1 = min(frame.shape[0], int(cy) + window)
        if x1 > x0 and y1 > y0:
            patch = frame[y0:y1, x0:x1].astype(np.float32).copy()
            if noise > 0.01:
                patch += np.random.default_rng(rng.randint(0, 2 ** 31)).normal(
                    0.0, noise, patch.shape).astype(np.float32)
            patch = np.clip(patch, 0, 255).astype(np.uint8)
            if quality:
                ok, encoded = cv2.imencode(".jpg", patch,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if ok:
                    patch = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            out = frame.copy()
            out[y0:y1, x0:x1] = patch
    return out


def insert_ball(frame: np.ndarray, sample: BallSample,
                appearance: BallAppearance, *, rng: Optional[random.Random] = None
                ) -> np.ndarray:
    """Insert one ball into one frame. Returns a new frame; the input is untouched."""
    rng = rng or random.Random(sample.seed)
    out = frame
    if appearance.shadow > 0.01 and sample.visibility != "OCCLUDED":
        # a soft contact shadow breaks the closed-circular-boundary signature
        height, width = frame.shape[:2]
        shadow = frame.copy()
        radius = max(2, int(round(sample.radius_px * 1.5)))
        cv2.ellipse(shadow, (int(sample.x), int(sample.y + sample.radius_px * 1.1)),
                    (radius, max(1, radius // 2)), 0, 0, 360, (40, 40, 40), -1)
        shadow = cv2.GaussianBlur(shadow, (0, 0), max(1.0, sample.radius_px * 0.5))
        alpha = 0.35 * appearance.shadow
        out = np.clip(out.astype(np.float32) * (1 - alpha)
                      + shadow.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    patch = render_ball_patch(appearance)
    out = _composite(out, patch, sample.x, sample.y)
    out = _occlude(out, sample.x, sample.y, sample.radius_px,
                   sample.occlusion_fraction, rng)
    return _degrade(out, appearance.jpeg_quality, appearance.noise, rng,
                    around=(sample.x, sample.y, appearance.radius_px))


def classify_insertion(*, margin_px: float, error_px: Optional[float], radius_px: float,
                       changed_pixels: int) -> tuple[bool, str]:
    """Decide whether an insertion is measurable supervision, and say why not.

    Extracted from the sequence loop so the rule can be tested directly instead of
    only through luck about where a ball happened to land.
    """
    if margin_px <= 2.0 * radius_px:
        return False, ("ball sits within two radii of the frame edge, so the changed "
                       "region is clipped and its centroid is not the ball centre")
    tolerance = max(2.0, radius_px)
    if error_px is None or error_px > tolerance:
        return False, (f"changed-pixel centroid is {error_px} px from the label "
                       f"(tolerance {tolerance:.1f} px, {changed_pixels} px changed): "
                       "the render is not measurable here")
    return True, ""


def measure_insertion(before: np.ndarray, after: np.ndarray, x: float, y: float,
                      radius: float, *, threshold: int = 25,
                      search_multiple: float = 6.0) -> dict:
    """Where the changed pixels actually are, in a window around the label.

    This is the honest check on the generator: the label says where the ball was
    placed, and this says where the frame actually changed. If those disagree, the
    supervision is wrong no matter how good the render looks.
    """
    height, width = before.shape[:2]
    window = int(max(8, radius * search_multiple))
    x0 = max(0, int(x) - window)
    y0 = max(0, int(y) - window)
    x1 = min(width, int(x) + window)
    y1 = min(height, int(y) + window)
    region_before = before[y0:y1, x0:x1].astype(np.int16)
    region_after = after[y0:y1, x0:x1].astype(np.int16)
    delta = np.abs(region_after - region_before).sum(axis=2)
    mask = delta > threshold
    count = int(mask.sum())
    if count == 0:
        return {"changed_pixels": 0, "centroid": None, "error_px": None,
                "window": [x0, y0, x1, y1]}
    ys, xs = np.nonzero(mask)
    centroid = (float(xs.mean()) + x0, float(ys.mean()) + y0)
    return {
        "changed_pixels": count,
        "centroid": [round(centroid[0], 2), round(centroid[1], 2)],
        "error_px": round(float(math.hypot(centroid[0] - x, centroid[1] - y)), 2),
        "window": [x0, y0, x1, y1],
    }


# ---------------------------------------------------------------------------
# trajectories
# ---------------------------------------------------------------------------

def sample_trajectory(spec: TrajectorySpec, count: int, width: int, height: int,
                      radius: float, rng: random.Random,
                      start: Optional[tuple[float, float]] = None
                      ) -> list[tuple[float, float]]:
    """Sub-pixel positions for a sequence."""
    kind = spec.kind
    if kind == "stationary":
        base_x = rng.uniform(radius * 2, width - radius * 2)
        base_y = rng.uniform(radius * 2, height - radius * 2)
        return [(base_x + rng.uniform(-0.4, 0.4), base_y + rng.uniform(-0.4, 0.4))
                for _ in range(count)]
    if start is not None:
        start_x, start_y = start
    elif kind == "entrance":
        start_x, start_y = -radius * 2.0, rng.uniform(0.2, 0.8) * height
    elif kind == "exit":
        start_x, start_y = rng.uniform(0.1, 0.5) * width, rng.uniform(0.2, 0.8) * height
    else:
        start_x, start_y = rng.uniform(0.25, 0.75) * width, rng.uniform(0.3, 0.9) * height

    positions = []
    x, y = start_x, start_y
    vx, vy = spec.vx * rng.uniform(0.8, 1.25), spec.vy * rng.uniform(0.8, 1.25)
    for _ in range(count):
        positions.append((x, y))
        vx += spec.ax
        vy += spec.ay
        x += vx
        y += vy
    return positions


def trajectory_for(radius_px: float, min_side: int, rng: random.Random,
                   count: int = 12, kind: Optional[str] = None) -> TrajectorySpec:
    """Pick a plausible trajectory expressed in *ball radii per frame*.

    Speed as a fraction of the frame is what produced the first version's
    nonsense: at 4K, "25% of the frame per frame" is 540 px, so the ball crossed
    the whole shot in two frames and nearly every labelled frame was off-screen.
    A struck ball does move fast -- but what makes it hard is that it moves fast
    *relative to its own size*, so the speed range belongs in radii.
    """
    if kind is None:
        kind = rng.choice(["stationary", "stationary", "slow", "linear", "linear",
                           "accelerating", "fast", "entrance", "exit"])
    if kind == "stationary":
        return TrajectorySpec(kind="stationary")
    if kind == "slow":
        speed = rng.uniform(0.05, 0.4) * radius_px
    elif kind in ("linear", "entrance", "exit"):
        speed = rng.uniform(0.4, 2.0) * radius_px
    elif kind == "fast":
        speed = rng.uniform(2.0, 6.0) * radius_px
    elif kind == "accelerating":
        speed = rng.uniform(0.2, 1.0) * radius_px
    else:
        raise ValueError(f"unknown trajectory kind {kind!r}")
    angle = rng.uniform(0, 2 * math.pi)
    spec = TrajectorySpec(kind=kind, vx=speed * math.cos(angle),
                          vy=speed * math.sin(angle) * 0.6)
    if kind == "accelerating":
        # accelerate to roughly 3x over the sequence, not to 20x
        spec.ax = spec.vx * (2.0 / max(1, count))
        spec.ay = spec.vy * (2.0 / max(1, count))
    return spec


def in_frame_start(spec: TrajectorySpec, count: int, width: int, height: int,
                   radius: float) -> tuple[float, float]:
    """Start position chosen so a non-entrance/exit trajectory stays in frame."""
    travel_x = spec.vx * count + 0.5 * spec.ax * count * count
    travel_y = spec.vy * count + 0.5 * spec.ay * count * count
    margin = radius * 1.5 + 2.0

    def place(travel: float, limit: int) -> float:
        if abs(travel) >= limit - 2 * margin:
            return limit / 2.0                          # it will cross; start centred
        if travel >= 0:
            return (limit + margin * 2 - travel) / 2.0
        return (limit - 2 * margin - travel) / 2.0

    return place(travel_x, width), place(travel_y, height)


def appearance_for(min_side: int, rng: random.Random, *, blur_px: float = 0.0
                   ) -> BallAppearance:
    """Sample a ball that is hard for the right reasons.

    Radius is drawn in *native* pixels and deliberately skews small: a ball
    between 3 and 20 px across is the real problem, and a macro ball is the easy
    case the previous method was measured on.
    """
    radius = float(np.clip(rng.lognormvariate(math.log(max(2.5, min_side * 0.004)), 0.55),
                           2.0, max(12.0, min_side * 0.08)))
    colour = rng.choices(list(BALL_COLOURS),
                         weights=[0.55, 0.2, 0.1, 0.12, 0.03])[0]
    return BallAppearance(
        colour=colour,
        radius_px=radius,
        brightness=rng.uniform(0.72, 1.12),
        shading=rng.uniform(0.2, 0.55),
        specular=rng.uniform(0.25, 0.8),
        blur_px=blur_px,
        jpeg_quality=rng.choice([0, 0, 0, 45, 32]),
        noise=rng.uniform(0.5, 3.5),
        shadow=rng.uniform(0.0, 0.5),
        dimple=rng.uniform(0.0, 0.09),
    )


# ---------------------------------------------------------------------------
# sequences and crops
# ---------------------------------------------------------------------------

def synthesize_sequence(frames: Sequence[np.ndarray], *, seed: int,
                        forced_kind: Optional[str] = None,
                        occlusion: bool = True, blur: bool = True
                        ) -> tuple[list[np.ndarray], list[BallSample]]:
    """Insert one ball tracking through a real sequence.

    Returns the modified frames and the exact per-frame labels. Frames before the
    ball enters and after it leaves are returned untouched and labelled
    ``OCCLUDED``/offscreen, so a tracker is never trained to see it where it
    cannot be.
    """
    if not frames:
        raise ValueError("at least one frame is required")
    rng = random.Random(seed)
    height, width = frames[0].shape[:2]
    min_side = min(width, height)
    appearance = appearance_for(min_side, rng)
    spec = trajectory_for(appearance.radius_px, min_side, rng, count=len(frames),
                          kind=forced_kind)
    start = None
    if spec.kind not in ("entrance", "exit"):
        start = in_frame_start(spec, len(frames), width, height, appearance.radius_px)
    positions = sample_trajectory(spec, len(frames), width, height,
                                  appearance.radius_px, rng, start=start)

    occlusion_from = occlusion_to = -1.0
    occlusion_fraction = 0.0
    if occlusion and rng.random() < 0.45:
        occlusion_from = rng.uniform(0.1, 0.5)
        occlusion_to = min(1.0, occlusion_from + rng.uniform(0.1, 0.4))
        occlusion_fraction = rng.uniform(0.3, 0.85)

    out_frames, samples = [], []
    speed = 0.0
    for index, (frame, (x, y)) in enumerate(zip(frames, positions)):
        progress = index / max(1, len(frames) - 1)
        if index > 0:
            speed = math.hypot(x - positions[index - 1][0], y - positions[index - 1][1])
        # A ball at rest has no motion blur; anything else blurs in proportion to
        # how fast it moved relative to its own size, capped so a 3 px ball does
        # not become a 30 px smear.
        blur_px = 0.0 if (spec.kind == "stationary" or not blur) else min(
            2.5 * appearance.radius_px, speed * 0.5)
        local_appearance = BallAppearance(**{**asdict(appearance), "blur_px": blur_px})

        occluded_fraction = 0.0
        if occlusion_from >= 0 and occlusion_from <= progress <= occlusion_to:
            occluded_fraction = occlusion_fraction
        inside = (-appearance.radius_px <= x <= width + appearance.radius_px
                  and -appearance.radius_px <= y <= height + appearance.radius_px)
        if not inside:
            visibility = "OCCLUDED"                     # off-screen: nothing to see
        elif occluded_fraction >= 0.75:
            visibility = "OCCLUDED"
        elif occluded_fraction > 0.02:
            visibility = "PARTIAL"
        else:
            visibility = "VISIBLE"

        sample = BallSample(frame_index=index, x=x, y=y,
                            radius_px=appearance.radius_px, visibility=visibility,
                            occlusion_fraction=occluded_fraction, blur_px=blur_px,
                            colour=appearance.colour, seed=seed)
        if visibility == "OCCLUDED" and not inside:
            out_frames.append(frame)
            samples.append(sample)
            continue
        inserted = insert_ball(frame, sample, local_appearance, rng=rng)
        out_frames.append(inserted)

        # Verify the insertion against the frame it came from. A synthetic label
        # whose ball is not actually there -- because the render vanished into a
        # blown-out region, or the patch fell off the frame edge -- is worse than
        # no label at all, so it is measured and marked rather than assumed.
        margin = min(x, y, width - x, height - y)
        measured = measure_insertion(frame, inserted, x, y, appearance.radius_px)
        if visibility == "OCCLUDED":
            sample = replace(sample, changed_pixels=measured["changed_pixels"],
                             centroid_error_px=measured["error_px"], verified=True)
        else:
            verified, reason = classify_insertion(
                margin_px=margin, error_px=measured["error_px"],
                radius_px=appearance.radius_px,
                changed_pixels=measured["changed_pixels"])
            sample = replace(sample, changed_pixels=measured["changed_pixels"],
                             centroid_error_px=measured["error_px"],
                             verified=verified, unverified_reason=reason)
        samples.append(sample)
    return out_frames, samples


def extract_crop(frame: np.ndarray, x: float, y: float, size: int) -> Optional[np.ndarray]:
    """Native-scale crop centred on (x, y), or None when it would leave the frame."""
    half = size // 2
    left, top = int(round(x)) - half, int(round(y)) - half
    if left < 0 or top < 0 or left + size > frame.shape[1] or top + size > frame.shape[0]:
        return None
    return frame[top:top + size, left:left + size].copy()


def crop_label(x: float, y: float, size: int) -> tuple[float, float]:
    """Where the ball sits inside its crop, in crop pixels."""
    half = size // 2
    return x - (int(round(x)) - half), y - (int(round(y)) - half)


# ---------------------------------------------------------------------------
# dataset assembly
# ---------------------------------------------------------------------------

@dataclass
class CropRecord:
    path: str
    label_x: float
    label_y: float
    radius: float
    positive: bool
    origin: str
    source: str
    clip_id: str
    frame_index: int
    meta: dict = field(default_factory=dict)


def _read_clip_window(media: Path, start: int, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(media))
    if not capture.isOpened():
        raise SystemExit(f"cannot decode {media}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start)))
    frames = []
    for _ in range(count):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def generate_synthetic_crops(*, train_clips: Sequence[Mapping[str, Any]],
                             media_root: Path, out_dir: Path, crop_size: int = 64,
                             sequences: int = 60, sequence_length: int = 12,
                             seed: int = 20260918, max_frames_per_clip: Optional[int] = None
                             ) -> dict:
    """Write cropped training images with exact labels, plus the full metadata.

    Positives come from the inserted ball. Negatives are crops from the same
    frames, well away from the insertion *and* silent in the classical candidate
    ensemble, which is the best available proxy for "no ball here" in the absence
    of human labels. That residual risk -- a real ball inside a negative crop --
    is recorded in the metadata rather than hidden.
    """
    rng = random.Random(seed)
    images = out_dir / "crops"
    images.mkdir(parents=True, exist_ok=True)
    records: list[CropRecord] = []
    manifest = {
        "origin": ORIGIN_SYNTHETIC,
        "crop_size": crop_size,
        "sequences": 0,
        "positives": 0,
        "negatives": 0,
        "unverified_skipped": 0,
        "clips": [],
        "seed": seed,
        "note": "Insertion coordinates are known because the generator placed "
                "them. This is synthetic supervision on real backgrounds, not "
                "human ground truth.",
    }
    for clip in train_clips:
        media = media_root / Path(clip["media_path"]).name
        if not media.exists():
            continue
        length = clip.get("frames") or 0
        used = 0
        for sequence_index in range(max(1, sequences // max(1, len(train_clips)))):
            if max_frames_per_clip and used >= max_frames_per_clip:
                break
            start = rng.randint(0, max(0, length - sequence_length - 1)) if length \
                else rng.randint(0, 200)
            frames = _read_clip_window(media, start, sequence_length)
            if len(frames) < 4:
                continue
            modified, samples = synthesize_sequence(frames, seed=rng.randint(0, 2 ** 31))
            for frame, sample in zip(modified, samples):
                if sample.visibility == "OCCLUDED":
                    continue
                if not sample.verified:
                    # The frame barely changed where the label says the ball is:
                    # the render is invisible at this texture and exposure. A
                    # detector cannot be expected to find it, so it is not a
                    # positive example, and calling it one would be supervision
                    # that contradicts the pixels.
                    manifest["unverified_skipped"] += 1
                    continue
                crop = extract_crop(frame, sample.x, sample.y, crop_size)
                if crop is None:
                    manifest["unverified_skipped"] += 1
                    continue
                lx, ly = crop_label(sample.x, sample.y, crop_size)
                name = f"pos_{clip['clip_id']}_{start + sample.frame_index}_{sample.seed}.jpg"
                cv2.imwrite(str(images / name), crop,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                records.append(CropRecord(name, lx, ly, sample.radius_px, True,
                                          ORIGIN_SYNTHETIC, "insertion",
                                          clip["clip_id"], start + sample.frame_index,
                                          {"visibility": sample.visibility,
                                           "blur_px": sample.blur_px,
                                           "colour": sample.colour,
                                           "insertion_centroid_error_px":
                                               sample.centroid_error_px,
                                           "changed_pixels": sample.changed_pixels}))
                manifest["positives"] += 1
                used += 1

                for _ in range(2):
                    nx = sample.x + rng.choice([-1, 1]) * rng.uniform(6, 20) * max(
                        1.0, sample.radius_px)
                    ny = sample.y + rng.choice([-1, 1]) * rng.uniform(6, 20) * max(
                        1.0, sample.radius_px)
                    negative = extract_crop(frame, nx, ny, crop_size)
                    if negative is None:
                        continue
                    name = (f"neg_{clip['clip_id']}_{start + sample.frame_index}_"
                            f"{int(nx)}_{int(ny)}.jpg")
                    cv2.imwrite(str(images / name), negative,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    records.append(CropRecord(name, crop_size / 2, crop_size / 2, 0.0,
                                              False, ORIGIN_SYNTHETIC,
                                              "background_far_from_insertion",
                                              clip["clip_id"],
                                              start + sample.frame_index,
                                              {"distance_px": round(math.hypot(
                                                  nx - sample.x, ny - sample.y), 1)}))
                    manifest["negatives"] += 1
            manifest["sequences"] += 1
        manifest["clips"].append({"clip_id": clip["clip_id"],
                                  "frames_used": used})

    with (out_dir / "crops.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def assert_training_clips(split_manifest: Mapping[str, Any],
                          clip_ids: Iterable[str]) -> None:
    """Refuse to synthesize on anything but training clips, unless told otherwise.

    Synthetic supervision built on a validation or final-test background would
    leak those frames into training through the back door, which is precisely the
    kind of accident a frozen split exists to prevent.
    """
    allowed = {clip["clip_id"] for clip in split_manifest.get("clips", [])
               if clip.get("split") == "train"}
    offending = sorted(set(clip_ids) - allowed)
    if offending:
        raise ValueError(
            f"refusing to generate synthetic training data from non-train clips: "
            f"{offending}. Evaluation sets must stay sealed; pass an explicit "
            "sealed-evaluation flag if this really is an evaluation set.")


def load_crop_records(path: Path) -> list[dict]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records
