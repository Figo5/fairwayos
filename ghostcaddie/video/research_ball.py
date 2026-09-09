"""Isolated, heuristic ball-candidate tracking for perception research only.

This module intentionally consumes RGB arrays and never imports the production
observation contract, renderer, analytics, or generic sports-ball models. Its
outputs are candidates, not golf-ball evidence; production promotion is
explicitly impossible by contract.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple
import math

try:
    import numpy as np
except ImportError:  # pragma: no cover - depends on the optional research env.
    np = None

try:
    import cv2
except ImportError:  # pragma: no cover - optional video acceleration.
    cv2 = None


Point = Tuple[float, float]


@dataclass(frozen=True)
class BallTrackItem:
    frame_index: int
    center: Optional[Point]
    confidence: float
    provenance: str
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BallCandidate:
    """A ranked visual candidate; this is never a validated golf-ball label."""
    center: Point
    confidence: float
    scale: float
    provenance: str = "research_candidate"
    cues: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BallTrackResult:
    track_id: str
    items: Tuple[BallTrackItem, ...]
    longest_gap: int
    provenance: str = "research_candidate"
    production_eligible: bool = False


class ResearchBallTracker:
    """Find bright compact pixel blobs and link them across short gaps.

    This is deliberately a low-level research adapter. It does not claim that a
    candidate is a golf ball and cannot be used by production analytics.
    """

    def __init__(self, min_confidence: float = 0.8, max_gap_frames: int = 2,
                 max_step_pixels: float = 24.0, min_pixels: int = 3,
                 max_component_fraction: float = 0.02, max_aspect_ratio: float = 3.0,
                 exclude_bottom_fraction: float = 0.18):
        if np is None:
            raise RuntimeError("numpy is required for the research ball adapter")
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        if isinstance(max_gap_frames, bool) or not isinstance(max_gap_frames, int):
            raise ValueError("max_gap_frames must be a non-negative integer")
        if isinstance(min_pixels, bool) or not isinstance(min_pixels, int):
            raise ValueError("min_pixels must be a positive integer")
        if isinstance(max_step_pixels, bool) or not isinstance(max_step_pixels, (int, float)):
            raise ValueError("max_step_pixels must be a finite positive number")
        if not math.isfinite(max_step_pixels):
            raise ValueError("max_step_pixels must be a finite positive number")
        if max_gap_frames < 0 or max_step_pixels <= 0 or min_pixels < 1:
            raise ValueError("tracking bounds must be non-negative, max_step_pixels positive, and min_pixels positive")
        if not 0 < max_component_fraction <= 1 or max_aspect_ratio < 1:
            raise ValueError("component bounds must be positive")
        if not 0 <= exclude_bottom_fraction < 1:
            raise ValueError("exclude_bottom_fraction must be in [0, 1)")
        self.min_confidence = float(min_confidence)
        self.max_gap_frames = int(max_gap_frames)
        self.max_step_pixels = float(max_step_pixels)
        self.min_pixels = int(min_pixels)
        self.max_component_fraction = float(max_component_fraction)
        self.max_aspect_ratio = float(max_aspect_ratio)
        self.exclude_bottom_fraction = float(exclude_bottom_fraction)

    def track(self, frames: Sequence[object], frame_indices: Optional[Iterable[int]] = None,
              contexts: Optional[Sequence[Optional[object]]] = None) -> BallTrackResult:
        images = list(frames)
        if not images:
            raise ValueError("at least one frame is required")
        indices = list(frame_indices) if frame_indices is not None else list(range(len(images)))
        if len(indices) != len(images) or any(not isinstance(i, int) or i < 0 for i in indices):
            raise ValueError("frame_indices must match frames and contain non-negative integers")
        if any(a >= b for a, b in zip(indices, indices[1:])):
            raise ValueError("frame_indices must be strictly increasing")
        if contexts is not None and len(contexts) != len(images):
            raise ValueError("contexts must match frames when provided")

        items = []
        previous = None
        previous_image = None
        last_observed = None
        longest_gap = 0
        for position, (index, image) in enumerate(zip(indices, images)):
            if last_observed is not None:
                longest_gap = max(longest_gap, index - last_observed - 1)
            context = contexts[position] if contexts is not None else None
            frame_delta = index - last_observed if last_observed is not None else 1
            step_limit = self.max_step_pixels * frame_delta
            if last_observed is not None and frame_delta - 1 > self.max_gap_frames:
                # Do not relink a candidate after an occlusion longer than the
                # configured research tracking bound. Start a fresh candidate
                # track on a later frame instead of fabricating continuity.
                candidate = None
                warning = "continuity_break"
                previous = None
                last_observed = None
            else:
                candidate = self._candidate(
                    image, previous_image, context, previous,
                    max_step_pixels=step_limit,
                )
                if candidate is not None:
                    center, confidence = candidate
                    if confidence < self.min_confidence:
                        candidate = None
                        warning = "low_confidence"
                    elif previous is not None and self._distance(previous, center) > step_limit:
                        candidate = None
                        warning = "continuity_break"
                    else:
                        warning = None
                else:
                    warning = "roi_unavailable" if isinstance(context, dict) and context.get("roi") is not None else "no_candidate"
            previous_image = image
            if candidate is None:
                gap = index - last_observed - 1 if last_observed is not None else 0
                longest_gap = max(longest_gap, gap)
                warnings = [warning or "unavailable"]
                if last_observed is not None and gap <= self.max_gap_frames:
                    warnings.append("gap")
                items.append(BallTrackItem(index, None, 0.0, "unavailable", tuple(sorted(set(warnings)))))
                continue

            center, confidence = candidate
            provenance = "candidate" if previous is None else "tracked"
            items.append(BallTrackItem(index, center, confidence, provenance))
            previous = center
            last_observed = index

        return BallTrackResult("ball-0", tuple(items), longest_gap)

    def extract_candidates(self, image: object, previous_image: Optional[object] = None,
                           context: Optional[object] = None, roi=None) -> Tuple[BallCandidate, ...]:
        """Extract ranked multi-scale visual candidates inside contextual ROIs.

        ``context`` may contain ``roi``, ``green_bbox`` and ``golfer_bbox`` as
        ``(x1, y1, x2, y2)`` pixel boxes. Temporal change is a ranking cue only.
        Empty/invalid ROIs return no candidates, allowing callers to preserve an
        explicit unavailable state.
        """
        rgb = self._validate_image(image)
        previous_rgb = self._validate_image(previous_image) if previous_image is not None else None
        if previous_rgb is not None and previous_rgb.shape != rgb.shape:
            previous_rgb = None
        context = context if isinstance(context, dict) else {}
        requested_roi = roi if roi is not None else context.get("roi")
        region = self._clip_bbox(requested_roi, rgb.shape[1], rgb.shape[0])
        if requested_roi is not None and region is None:
            return ()
        region = region or (0, 0, rgb.shape[1], rgb.shape[0])
        luminance = rgb.mean(axis=2)
        spread = rgb.max(axis=2) - rgb.min(axis=2)
        motion = None
        if previous_rgb is not None:
            motion = np.abs(luminance - previous_rgb.mean(axis=2))
        candidates = []
        if cv2 is not None:
            gray = np.clip(luminance, 0, 255).astype(np.uint8)
            max_y = int(round(gray.shape[0] * (1.0 - self.exclude_bottom_fraction)))
            gray[max_y:, :] = 0
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=18,
                param1=100, param2=18, minRadius=8, maxRadius=45,
            )
            if circles is not None:
                for cx, cy, radius in circles[0]:
                    cx, cy, radius = float(cx), float(cy), float(radius)
                    if not self._inside((cx, cy), region) or cy >= max_y:
                        continue
                    half = max(2, int(round(radius * 0.7)))
                    bx0, by0 = max(0, int(cx - half)), max(0, int(cy - half))
                    bx1, by1 = min(rgb.shape[1], int(cx + half) + 1), min(rgb.shape[0], int(cy + half) + 1)
                    patch = rgb[by0:by1, bx0:bx1]
                    if patch.size == 0:
                        continue
                    hsv = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_RGB2HSV)
                    white = float(np.mean((hsv[..., 1] < 75) & (hsv[..., 2] > 145)))
                    brightness = float(np.mean(hsv[..., 2])) / 255.0
                    confidence = min(1.0, 0.55 * white + 0.45 * brightness)
                    if confidence >= 0.38:
                        candidates.append(BallCandidate(
                            (cx, cy), confidence, max(0.75, radius / 3.0),
                            "research_candidate", ("circle_proposal",),
                        ))
        # Different local windows represent small and larger apparent objects.
        for scale in (0.75, 1.0, 1.5):
            radius = max(1, int(round(3 * scale)))
            local = self._local_mean(luminance, radius=radius)
            static_mask = ((luminance >= max(120.0, 150.0 - 15.0 * (scale - 0.75))) &
                           (luminance >= local + 30.0) & (spread <= 45.0))
            temporal_mask = np.zeros_like(static_mask)
            if motion is not None:
                baseline = float(np.median(motion))
                temporal_mask = motion >= max(12.0, baseline + 8.0)
                if cv2 is not None:
                    kernel = np.ones((2, 2), dtype=np.uint8)
                    temporal_mask = cv2.morphologyEx(
                        temporal_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel
                    ).astype(bool)
            mask = static_mask | (temporal_mask & (spread <= 65.0) & (luminance >= 105.0))
            x1, y1, x2, y2 = region
            height, width = mask.shape
            allowed = np.zeros_like(mask)
            allowed[y1:y2, x1:x2] = True
            if self.exclude_bottom_fraction:
                allowed[int(round(height * (1.0 - self.exclude_bottom_fraction))):, :] = False
            mask &= allowed
            max_pixels = max(self.min_pixels, int(mask.size * self.max_component_fraction))
            for points, component_width, component_height, centroid, area in self._components(mask):
                    pixel_count = area if area else len(points)
                    if pixel_count < self.min_pixels or pixel_count > max_pixels:
                        continue
                    aspect_ratio = max(component_width, component_height) / min(component_width, component_height)
                    if aspect_ratio > self.max_aspect_ratio:
                        continue
                    if points:
                        ys = np.asarray([p[0] for p in points], dtype=int)
                        xs = np.asarray([p[1] for p in points], dtype=int)
                        values = luminance[ys, xs]
                        contrast = float(np.mean(values - local[ys, xs]))
                        center = (float(np.mean(xs)), float(np.mean(ys)))
                    else:
                        cx, cy = centroid
                        pixel_y, pixel_x = int(cy), int(cx)
                        values = np.asarray([luminance[pixel_y, pixel_x]])
                        contrast = float(values[0] - local[pixel_y, pixel_x])
                        center = (float(cx), float(cy))
                    brightness = float(values.mean()) / 255.0
                    compactness = min(1.0, pixel_count / (9.0 * scale))
                    confidence = compactness * brightness * min(1.0, max(contrast, 12.0) / 60.0)
                    cues = []
                    if motion is not None:
                        if points:
                            changes = float(np.mean(motion[ys, xs]))
                        else:
                            pixel_y, pixel_x = int(cy), int(cx)
                            changes = float(motion[pixel_y, pixel_x])
                        baseline = float(np.median(motion))
                        if changes > max(8.0, baseline + 4.0):
                            confidence *= 1.2
                            cues.append("temporal_difference")
                        else:
                            # Prefer residual motion over static highlights when
                            # temporal evidence is available, without rejecting
                            # the static candidate outright.
                            confidence *= 0.65
                    green = self._clip_bbox(context.get("green_bbox"), width, height)
                    golfer = self._clip_bbox(context.get("golfer_bbox"), width, height)
                    if green and self._inside(center, green):
                        confidence *= 1.35
                        cues.append("green_context")
                    if golfer and self._inside(center, golfer):
                        confidence *= 0.45
                        cues.append("golfer_context")
                    if requested_roi is not None:
                        cues.append("roi")
                    candidates.append(BallCandidate(center, min(1.0, confidence), scale,
                                                    "research_candidate", tuple(cues)))
        candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
        deduped = []
        bins = {}
        for candidate in candidates:
            cell = (int(candidate.center[0] // 2), int(candidate.center[1] // 2))
            nearby = []
            for bx in range(cell[0] - 1, cell[0] + 2):
                for by in range(cell[1] - 1, cell[1] + 2):
                    nearby.extend(bins.get((bx, by), ()))
            if not any(self._distance(candidate.center, prior.center) < 2.0 for prior in nearby):
                deduped.append(candidate)
                bins.setdefault(cell, []).append(candidate)
        return tuple(deduped)

    @staticmethod
    def _components(mask):
        """Yield connected components, using OpenCV when available.

        Centroid convention: coordinates are the mean of the member pixels'
        integer indices in ``(x, col)``/``(y, row)`` order — OpenCV's
        ``connectedComponentsWithStats`` convention (verified identical on
        opencv-python-headless 4.9, 4.12, and 5.0). A 4x4 block spanning
        columns 12..15, rows 24..27 centers at (13.5, 25.5), not a +0.5
        pixel-center offset. Renderers convert these float centers to draw
        pixels with ``int(round(...))``; no rounding is applied here.
        """
        if cv2 is not None:
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8
            )
            for label in range(1, count):
                yield ((),
                       int(stats[label, cv2.CC_STAT_WIDTH]),
                       int(stats[label, cv2.CC_STAT_HEIGHT]),
                       (float(centroids[label][0]), float(centroids[label][1])),
                       int(stats[label, cv2.CC_STAT_AREA]))
            return
        visited = np.zeros(mask.shape, dtype=bool)
        height, width = mask.shape
        for y in range(height):
            for x in range(width):
                if not mask[y, x] or visited[y, x]:
                    continue
                points = ResearchBallTracker._component(mask, visited, y, x)
                component_height = max(p[0] for p in points) - min(p[0] for p in points) + 1
                component_width = max(p[1] for p in points) - min(p[1] for p in points) + 1
                yield points, component_width, component_height, None, len(points)

    def _candidate(self, image: object, previous_image: Optional[object] = None,
                   context=None, previous_center: Optional[Point] = None,
                   max_step_pixels: Optional[float] = None) -> Optional[Tuple[Point, float]]:
        """Return one *candidate* using contrast, color, and compactness cues.

        The detector is intentionally conservative about language: these are
        bright/neutral compact regions, not validated golf-ball detections. A
        local luminance baseline makes small, dim objects detectable while
        rejecting broad bright overlays and flat backgrounds.
        """
        candidates = self.extract_candidates(image, previous_image, context)
        if not candidates:
            return None
        max_step_pixels = self.max_step_pixels if max_step_pixels is None else max_step_pixels
        circle_candidates = [candidate for candidate in candidates
                             if "circle_proposal" in candidate.cues]
        if circle_candidates and previous_center is None:
            candidates = circle_candidates
        if previous_center is not None:
            nearby = [candidate for candidate in candidates
                      if self._distance(previous_center, candidate.center) <= max_step_pixels]
            if nearby:
                moving = [candidate for candidate in nearby
                          if "temporal_difference" in candidate.cues]
                if moving:
                    nearby = moving
                candidates = sorted(
                    nearby,
                    key=lambda candidate: (
                        self._distance(previous_center, candidate.center) / max_step_pixels * 0.55
                        + (1.0 - candidate.confidence) * 0.45
                    ),
                )
        return candidates[0].center, candidates[0].confidence

    @staticmethod
    def _clip_bbox(bbox, width, height):
        if bbox is None:
            return None
        try:
            x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
        except (TypeError, ValueError):
            return None
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    @staticmethod
    def _inside(point, bbox):
        x, y = point
        return bbox[0] <= x < bbox[2] and bbox[1] <= y < bbox[3]

    @staticmethod
    def _validate_image(image: object):
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3 or not np.issubdtype(array.dtype, np.number):
            raise ValueError("frames must be HxWx3 numeric RGB arrays")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("frames must have positive dimensions")
        return array.astype(np.float32, copy=False)

    @staticmethod
    def _local_mean(image, radius: int):
        """Compute an edge-padded box mean without per-pixel Python loops."""
        padded = np.pad(image, radius, mode="edge").astype(np.float64, copy=False)
        integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
        integral = np.cumsum(np.cumsum(integral, axis=0), axis=1)
        height, width = image.shape
        y0, y1 = 0, height
        x0, x1 = 0, width
        total = (
            integral[y1 + 2 * radius, x1 + 2 * radius]
            - integral[y0, x1 + 2 * radius]
            - integral[y1 + 2 * radius, x0]
            + integral[y0, x0]
        )
        # The scalar expression above is not the sliding window; use indexed
        # rectangle corners for the full image in one vectorized operation.
        top = integral[:height, 2 * radius + 1:2 * radius + 1 + width]
        left = integral[2 * radius + 1:2 * radius + 1 + height, :width]
        corners = integral[:height, :width]
        bottom_right = integral[2 * radius + 1:2 * radius + 1 + height,
                                2 * radius + 1:2 * radius + 1 + width]
        return (bottom_right - top - left + corners) / float((2 * radius + 1) ** 2)

    @staticmethod
    def _component(mask, visited, start_y, start_x):
        height, width = mask.shape
        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        points = []
        while stack:
            cy, cx = stack.pop()
            points.append((cy, cx))
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        return points

    @staticmethod
    def _distance(first: Point, second: Point) -> float:
        dx = first[0] - second[0]
        dy = first[1] - second[1]
        return (dx * dx + dy * dy) ** 0.5


@dataclass(frozen=True)
class SeededBallTrackItem:
    frame_index: int
    center: Optional[Point]
    confidence: float
    provenance: str
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SeededBallTrackResult:
    track_id: str
    items: Tuple[SeededBallTrackItem, ...]
    longest_gap: int
    provenance: str = "research_candidate"
    production_eligible: bool = False
    ground_truth: bool = False


class SeededBallTracker:
    """ROI-seeded appearance/geometry tracker for one human-seeded ball.

    Input convention: all frames and seed coordinates are RGB
    (HxWx3, channel order R-G-B) in the same pixel space as the ROI and
    seed point. OpenCV is REQUIRED: the appearance model converts patches
    with ``cv2.COLOR_RGB2HSV`` (not BGR2HSV), and without OpenCV the HSV
    half of the model would silently misinterpret RGB as HSV, so
    construction/usage fails closed with a RuntimeError.

    The seed (frame index, point) is supplied explicitly by a human reviewer
    together with a hard native ROI. The appearance model — a color template
    and mean HSV sampled once at the seed — is fixed for the whole track and
    is never re-tuned from later frames. Each subsequent frame is searched
    within the ROI around the predicted position for a compact blob whose
    appearance matches the seed model; geometry (compactness, bounded
    aspect, bounded step) and appearance must both agree. Any ambiguity —
    more than one plausible match, appearance drift, motion beyond the
    bound, or the ball leaving the ROI or vanishing — fails closed to an
    explicit ``unavailable`` item. This is a research candidate tracker:
    it never claims ground truth and is not production eligible.
    """

    def __init__(self, *, roi: Tuple[int, int, int, int],
                 max_step_pixels: float = 60.0, max_gap_frames: int = 2,
                 search_radius: float = 6.0, max_aspect_ratio: float = 3.0,
                 min_pixels: int = 4, max_component_fraction: float = 0.02,
                 appearance_tolerance: float = 0.35):
        if np is None:
            raise RuntimeError("numpy is required for the research ball adapter")
        if cv2 is None:
            raise RuntimeError(
                "SeededBallTracker requires OpenCV for its HSV appearance model "
                "(cv2.COLOR_RGB2HSV on RGB frames); install opencv-python-headless "
                "instead of silently treating RGB values as pseudo-HSV"
            )
        region = self._clip_roi(roi)
        if region is None:
            raise ValueError("roi must be a non-degenerate (x1, y1, x2, y2) box")
        if isinstance(max_step_pixels, bool) or not isinstance(max_step_pixels, (int, float)) \
                or not math.isfinite(max_step_pixels) or max_step_pixels <= 0:
            raise ValueError("max_step_pixels must be a finite positive number")
        if isinstance(max_gap_frames, bool) or not isinstance(max_gap_frames, int) or max_gap_frames < 0:
            raise ValueError("max_gap_frames must be a non-negative integer")
        if not 0 < search_radius <= max_step_pixels or not max_aspect_ratio >= 1 \
                or min_pixels < 1 or not 0 < max_component_fraction <= 1 \
                or not 0 < appearance_tolerance <= 1:
            raise ValueError("invalid tracker bounds")
        self.roi = region
        self.max_step_pixels = float(max_step_pixels)
        self.max_gap_frames = int(max_gap_frames)
        self.search_radius = float(search_radius)
        self.max_aspect_ratio = float(max_aspect_ratio)
        self.min_pixels = int(min_pixels)
        self.max_component_fraction = float(max_component_fraction)
        self.appearance_tolerance = float(appearance_tolerance)

    @staticmethod
    def _clip_roi(roi):
        try:
            x1, y1, x2, y2 = (int(round(float(v))) for v in roi)
        except (TypeError, ValueError):
            return None
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    def track(self, frames, *, seed_frame_index: int,
              seed_point: Point) -> SeededBallTrackResult:
        images = [np.asarray(frame) for frame in frames]
        if not images:
            raise ValueError("at least one frame is required")
        if not isinstance(seed_frame_index, int) or isinstance(seed_frame_index, bool) \
                or not 0 <= seed_frame_index < len(images):
            raise ValueError("seed_frame_index must be an index into frames")
        seed_rgb = self._validate_image(images[seed_frame_index])
        x, y = seed_point
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (x, y)):
            raise ValueError("seed_point must be finite")
        rx1, ry1, rx2, ry2 = self.roi
        if not (rx1 <= x < rx2 and ry1 <= y < ry2 and 0 <= x < seed_rgb.shape[1] and 0 <= y < seed_rgb.shape[0]):
            raise ValueError("seed_point must lie inside the required roi")
        template = self._sample_appearance(seed_rgb, float(x), float(y))
        seed_area = self._seed_area(seed_rgb, float(x), float(y), template)
        template["seed_area"] = seed_area
        items = [SeededBallTrackItem(seed_frame_index, (float(x), float(y)), 1.0, "seeded")]
        previous = (float(x), float(y))
        last_observed = seed_frame_index
        longest_gap = 0
        for index in range(seed_frame_index + 1, len(images)):
            gap = index - last_observed - 1
            longest_gap = max(longest_gap, gap)
            if gap > self.max_gap_frames:
                # Fail closed: never relink the seeded track across an
                # occlusion longer than the configured bound.
                items.append(SeededBallTrackItem(index, None, 0.0, "unavailable", ("occlusion_exceeded",)))
                continue
            rgb = self._validate_image(images[index])
            match, reason = self._match(rgb, previous, template, frame_delta=index - last_observed)
            if match is None:
                warning = reason if reason in ("ambiguous",) else (
                    "occlusion_exceeded" if index - last_observed - 1 > self.max_gap_frames
                    else "appearance_or_motion_unavailable"
                )
                items.append(SeededBallTrackItem(index, None, 0.0, "unavailable", (warning,)))
                continue
            center, confidence = match
            items.append(SeededBallTrackItem(index, center, confidence, "tracked"))
            previous = center
            last_observed = index
        return SeededBallTrackResult("ball-seed-0", tuple(items), longest_gap)

    def _sample_appearance(self, rgb, x: float, y: float) -> dict:
        """Sample a fixed template once from the seed neighborhood."""
        radius = 3
        y0, y1 = max(0, int(y) - radius), min(rgb.shape[0], int(y) + radius + 1)
        x0, x1 = max(0, int(x) - radius), min(rgb.shape[1], int(x) + radius + 1)
        patch = rgb[y0:y1, x0:x1]
        hsv = self._to_hsv(patch)
        mask = np.ones(patch.shape[:2], dtype=bool)
        return {
            "template": patch,
            "mean_rgb": patch.reshape(-1, 3).mean(axis=0),
            "mean_hsv": hsv.reshape(-1, 3).mean(axis=0),
            "mask": mask,
        }

    def _seed_area(self, rgb, x: float, y: float, template: dict) -> int:
        """Measure the seeded object's apparent pixel area once, at seed time.

        Flood-fills from the seed point under the same fixed appearance mask
        used for candidate extraction (seed-HSV plus seed-RGB tolerance) so
        the component-size cap can admit a ball-sized blob even when the
        close-up ball exceeds a fixed ROI fraction. Bounded to avoid runaway
        regions on pathological seeds.
        """
        rx1, ry1, rx2, ry2 = self.roi
        x0 = max(rx1, 0, int(x) - 96)
        x1 = min(rx2, rgb.shape[1], int(x) + 97)
        y0 = max(ry1, 0, int(y) - 96)
        y1 = min(ry2, rgb.shape[0], int(y) + 97)
        region = rgb[y0:y1, x0:x1]
        if region.size == 0:
            return 0
        color_distance = np.linalg.norm(
            region - template["mean_rgb"].reshape(1, 1, 3), axis=2
        ) / (math.sqrt(3) * 255.0)
        region_hsv = self._to_hsv(region)
        hue_delta = np.abs(region_hsv[..., 0] - template["mean_hsv"][0])
        hue_delta = np.minimum(hue_delta, 180.0 - hue_delta)
        mask = ((hue_delta <= 10.0) &
                (np.abs(region_hsv[..., 1] - template["mean_hsv"][1]) <= 60.0) &
                (color_distance <= self.appearance_tolerance))
        if cv2 is None:  # pragma: no cover - constructor already fails closed
            raise RuntimeError("SeededBallTracker requires OpenCV")
        seed_y, seed_x = int(y) - y0, int(x) - x0
        if not (0 <= seed_y < mask.shape[0] and 0 <= seed_x < mask.shape[1]) or not mask[seed_y, seed_x]:
            return 0
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        label = int(labels[seed_y, seed_x])
        return int(np.count_nonzero(labels == label)) if label > 0 else 0

    def _match(self, rgb, previous: Point, template: dict,
               frame_delta: int) -> Tuple[Optional[Tuple[Point, float]], Optional[str]]:
        height, width = rgb.shape[:2]
        step_limit = self.max_step_pixels * max(1, frame_delta)
        rx1, ry1, rx2, ry2 = self.roi
        cx, cy = previous
        x0 = max(rx1, 0, int(cx - self.search_radius - step_limit))
        x1 = min(rx2, width, int(cx + self.search_radius + step_limit) + 1)
        y0 = max(ry1, 0, int(cy - self.search_radius - step_limit))
        y1 = min(ry2, height, int(cy + self.search_radius + step_limit) + 1)
        if x0 >= x1 or y0 >= y1:
            return None, None
        region = rgb[y0:y1, x0:x1]
        template_rgb = template["mean_rgb"]
        color_distance = np.linalg.norm(
            region - template_rgb.reshape(1, 1, 3), axis=2
        ) / (math.sqrt(3) * 255.0)
        # Candidate component extraction in the fixed seed-HSV space rather
        # than a per-pixel RGB color-distance mask. In the real-clip failure
        # (pexels_6573644 frames 72-75) the ball RGB (121, 117, 39) is within
        # the tolerance of sunlit grass RGB (95, 126, 62) (~0.08 << 0.35), so
        # the old mask merged the ball into one background-sized component
        # and every frame failed closed. Hue and saturation are the
        # discriminative, illumination-robust channels (ball hue ~27-28 vs
        # grass ~50 here); they remain the FIXED seed appearance model —
        # sampled once at the seed, never re-tuned — so all fail-closed
        # contracts (ambiguity, drift, bounds) are unchanged.
        seed_hsv = template["mean_hsv"]
        region_hsv = self._to_hsv(region)
        hue_delta = np.abs(region_hsv[..., 0] - seed_hsv[0])
        hue_delta = np.minimum(hue_delta, 180.0 - hue_delta)
        hsv_mask = (hue_delta <= 10.0) & (np.abs(region_hsv[..., 1] - seed_hsv[1]) <= 60.0)
        # Both masks must agree so a component never grows through a region
        # that violates the fixed seed RGB template.
        mask = hsv_mask & (color_distance <= self.appearance_tolerance)
        # Size the component cap from the fixed ROI rather than the
        # (motion-dependent) current search window, so the bound is stable
        # frame to frame and does not silently reject a ball-sized blob.
        # The cap must also admit the seeded ball itself: scale it from the
        # seed-template neighborhood's apparent object size (sampled once at
        # seed time), because a large close-up ball (real clip: ~12.5k px at
        # 1920x1080) is a legitimate match that a fixed fraction of the ROI
        # (850x300 -> 5101 px) would reject.
        roi_area = (rx2 - rx1) * (ry2 - ry1)
        seed_area = template["seed_area"]
        max_pixels = max(self.min_pixels, int(roi_area * self.max_component_fraction) + 1,
                         int(seed_area * 1.5) + 1)
        matches = []
        for points, comp_w, comp_h, centroid, area in ResearchBallTracker._components(mask):
            pixel_count = area if area else len(points)
            if pixel_count < self.min_pixels or pixel_count > max_pixels:
                continue
            if max(comp_w, comp_h) / max(1, min(comp_w, comp_h)) > self.max_aspect_ratio:
                continue
            if points:
                ys = np.asarray([p[0] for p in points], dtype=int)
                xs = np.asarray([p[1] for p in points], dtype=int)
            else:
                ys = np.asarray([int(centroid[1])])
                xs = np.asarray([int(centroid[0])])
            center = (float(xs.mean()) + x0, float(ys.mean()) + y0)
            if math.hypot(center[0] - cx, center[1] - cy) > self.search_radius + step_limit:
                continue
            half = 2
            by0, by1 = max(0, int(center[1]) - half), min(height, int(center[1]) + half + 1)
            bx0, bx1 = max(0, int(center[0]) - half), min(width, int(center[0]) + half + 1)
            patch = rgb[by0:by1, bx0:bx1]
            if patch.size == 0:
                continue
            appearance = self._appearance_distance(patch, template)
            if appearance > self.appearance_tolerance:
                continue
            compactness = min(1.0, pixel_count / 12.0)
            confidence = compactness * (1.0 - appearance)
            matches.append((center, float(min(1.0, confidence)), pixel_count))
        if not matches:
            return None, None
        if len(matches) > 1 and seed_area > 0:
            # Scale consistency: the seed defines not only appearance but the
            # object's apparent size. Small grass-highlight specks match the
            # seed appearance exactly but are orders of magnitude smaller
            # than the seeded disc; keep only components within a fixed
            # (0.5x..2.0x) seed-area band. This is a deterministic seed-model
            # constraint, not a threshold loosening: if two ball-scale blobs
            # remain, the track still fails closed as 'ambiguous'.
            lower = max(self.min_pixels, seed_area // 2)
            upper = max_pixels
            matches = [m for m in matches if lower <= m[2] <= upper]
        if not matches:
            return None, None
        if len(matches) > 1:
            # Ambiguity: multiple plausible appearance-consistent blobs in the
            # search window. Fail closed rather than guess.
            return None, "ambiguous"
        center, confidence, _area = matches[0]
        return (center, confidence), None

    def _appearance_distance(self, patch, template) -> float:
        mean_rgb = patch.reshape(-1, 3).mean(axis=0)
        rgb_distance = float(np.linalg.norm(mean_rgb - template["mean_rgb"]) / (math.sqrt(3) * 255.0))
        hsv = self._to_hsv(patch)
        hsv_distance = float(np.linalg.norm(hsv.reshape(-1, 3).mean(axis=0) - template["mean_hsv"]) / (math.sqrt(3) * 255.0))
        return min(1.0, 0.5 * rgb_distance + 0.5 * hsv_distance)

    @staticmethod
    def _to_hsv(patch):
        if cv2 is None:
            # Unreachable when constructed normally (the constructor fails
            # closed); kept as defense-in-depth against silent pseudo-HSV.
            raise RuntimeError(
                "SeededBallTracker requires OpenCV for its HSV appearance model "
                "(cv2.COLOR_RGB2HSV on RGB frames)"
            )
        return cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

    @staticmethod
    def _validate_image(image):
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3 or not np.issubdtype(array.dtype, np.number):
            raise ValueError("frames must be HxWx3 numeric RGB arrays")
        if not np.issubdtype(array.dtype, np.uint8):
            # Fail closed: _to_hsv casts to uint8, so float frames (e.g. [0,1]
            # normalized arrays) would silently collapse HSV values and corrupt
            # the appearance model. All callers pass uint8 frames.
            raise ValueError("frames must be uint8 RGB arrays; float frames would be silently collapsed by the uint8 HSV cast")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("frames must have positive dimensions")
        return array.astype(np.float32, copy=False)


def _round_blob_centers(frame, roi, *, min_area=60, min_circularity=0.5):
    """Return [(cx, cy, area, circularity)] of yellow round-ish blobs in a
    uint8 BGR frame within the ROI (as decoded by cv2.VideoCapture)."""
    if cv2 is None:
        raise RuntimeError("OpenCV required")
    x0, y0, x1, y1 = roi
    img = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (15, 90, 90), (42, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        per = cv2.arcLength(c, True)
        circ = (4 * np.pi * area) / (per * per) if per > 0 else 0
        if circ < min_circularity:
            continue
        M = cv2.moments(c)
        if M["m00"] > 0:
            out.append((M["m10"] / M["m00"] + x0, M["m01"] / M["m00"] + y0, area, circ))
    return out


def find_static_ball(frames, *, roi, min_persistent_frames=3, tolerance_px=6.0):
    """Deterministically locate a static ball by round-blob persistence.

    A real at-rest ball appears as a small, round, sharply-edged yellow blob in
    the SAME position across several consecutive frames; background/grass masses
    are non-circular or transient. Returns the mean center of the largest
    persistent round blob, or None when no defensible candidate exists.

    This is a RESEARCH-CANDIDATE seed locator, not a detection claim: it returns
    a candidate center that an AI/human reviewer must confirm before tracking.
    Args: frames are uint8 BGR arrays (as decoded by cv2.VideoCapture); roi is
    (x1,y1,x2,y2). Returns (cx, cy) or None.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV required")
    frames = list(frames)
    if not frames:
        raise ValueError("at least one frame required")
    region = SeededBallTracker._clip_roi(roi)
    if region is None:
        raise ValueError("roi must be a non-degenerate (x1,y1,x2,y2) box")
    if min_persistent_frames < 1:
        raise ValueError("min_persistent_frames must be >= 1")

    per_frame = []
    for f in frames:
        blobs = _round_blob_centers(f, region)
        if not blobs:
            per_frame.append(None)
            continue
        blobs.sort(key=lambda t: -t[2])
        per_frame.append((blobs[0][0], blobs[0][1]))

    best = None
    best_count = 0
    for start in range(len(per_frame)):
        if per_frame[start] is None:
            continue
        cx0, cy0 = per_frame[start]
        xs, ys = [cx0], [cy0]
        for j in range(start + 1, len(per_frame)):
            if per_frame[j] is None:
                break
            cx, cy = per_frame[j]
            if abs(cx - cx0) <= tolerance_px and abs(cy - cy0) <= tolerance_px:
                xs.append(cx); ys.append(cy)
            else:
                break
        if len(xs) >= best_count:
            best_count = len(xs)
            best = (float(np.mean(xs)), float(np.mean(ys)))
    if best_count < min_persistent_frames:
        return None
    return best
