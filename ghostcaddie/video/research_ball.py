"""Isolated, heuristic ball-candidate tracking for perception research only.

This module intentionally consumes RGB arrays and never imports the production
observation contract, renderer, analytics, or generic sports-ball models. Its
outputs are candidates, not golf-ball evidence; production promotion is
explicitly impossible by contract.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

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
                        values = np.asarray([luminance[cy, cx]])
                        contrast = float(values[0] - local[cy, cx])
                        center = (float(cx), float(cy))
                    brightness = float(values.mean()) / 255.0
                    compactness = min(1.0, pixel_count / (9.0 * scale))
                    confidence = compactness * brightness * min(1.0, max(contrast, 12.0) / 60.0)
                    cues = []
                    if motion is not None:
                        changes = float(np.mean(motion[ys, xs])) if points else float(motion[cy, cx])
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
        """Yield connected components, using OpenCV when available."""
        if cv2 is not None:
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8
            )
            for label in range(1, count):
                yield ((),
                       int(stats[label, cv2.CC_STAT_WIDTH]),
                       int(stats[label, cv2.CC_STAT_HEIGHT]),
                       (int(round(centroids[label][0])), int(round(centroids[label][1]))),
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
