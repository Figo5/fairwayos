"""Background (camera) motion between two frames, from static scene features only.

Estimated on every frame pair, whether or not a tracked target was found in them:
background motion is a property of the frames, not of the tracker's luck. The fit
is reported with the evidence a reviewer needs to judge it -- how many points
supported it, how far it misses points it was NOT fitted to, and how much of the
frame those points covered -- and it fails explicitly rather than falling back to
identity, because "the camera did not move" is a claim, not a default.

Affine only. A homography fits this footage too, but affine is the smaller model
and nothing here has shown the extra freedom is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_FEATURES = 12
MAX_FEATURES = 400


STATUSES = ("ok", "insufficient_features", "unstable")


@dataclass(frozen=True)
class Transform:
    """One frame pair's background motion, with its own failure state.

    A matrix that is not a finite 2x3 affine is refused at construction: a 3x3
    homography, a wrong-shaped array or a NaN entry would otherwise surface later
    as a raw numpy error or as a NaN displacement that looks like a measurement.
    """

    matrix: np.ndarray | None
    tracked: int
    inliers: int
    heldout_residual_px: float | None
    coverage: float
    status: str          # "ok" | "insufficient_features" | "unstable"

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")
        if self.matrix is None:
            if self.status == "ok":
                raise ValueError("an ok transform must carry a matrix")
            return
        matrix = np.asarray(self.matrix)
        if matrix.shape != (2, 3):
            raise ValueError(f"affine matrix must be 2x3, got {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError("affine matrix must be finite")


def _failed(status: str, tracked: int = 0) -> Transform:
    return Transform(None, tracked, 0, None, 0.0, status)


def _coverage(points: np.ndarray, shape: tuple[int, int]) -> float:
    """Fraction of the frame's four quadrants that contributed a point."""
    height, width = shape
    quadrants = {(int(x >= width / 2), int(y >= height / 2)) for x, y in points}
    return len(quadrants) / 4.0


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    padded = np.hstack([points, np.ones((len(points), 1), dtype=np.float32)])
    return padded @ matrix.T


def estimate_transform(previous, current, feature_mask=None) -> Transform:
    """Background motion from `previous` to `current`, or an explicit failure."""
    features = cv2.goodFeaturesToTrack(previous, maxCorners=MAX_FEATURES,
                                       qualityLevel=0.01, minDistance=8,
                                       mask=feature_mask)
    if features is None or len(features) < MIN_FEATURES:
        return _failed("insufficient_features", 0 if features is None else len(features))

    moved, tracked_ok, _ = cv2.calcOpticalFlowPyrLK(previous, current, features, None)
    if moved is None:
        return _failed("insufficient_features", len(features))
    keep = tracked_ok.ravel() == 1
    source = features[keep].reshape(-1, 2).astype(np.float32)
    target = moved[keep].reshape(-1, 2).astype(np.float32)
    if len(source) < MIN_FEATURES:
        return _failed("insufficient_features", len(source))

    matrix, inlier_mask = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC)
    if matrix is None:
        return _failed("unstable", len(source))
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0

    # Held-out residual: fit on half the points, measure on the half it never saw.
    # A fit reported only on its own points cannot show it generalises.
    # Contiguous copies: OpenCV refuses strided views of a point set.
    fit_source = np.ascontiguousarray(source[::2])
    fit_target = np.ascontiguousarray(target[::2])
    held_source = np.ascontiguousarray(source[1::2])
    held_target = np.ascontiguousarray(target[1::2])
    fit_matrix = None
    if len(fit_source) >= 3:
        fit_matrix, _ = cv2.estimateAffinePartial2D(fit_source, fit_target, method=cv2.RANSAC)
    residual = None
    if fit_matrix is not None and len(held_source) >= 3:
        predicted = _apply(fit_matrix, held_source)
        residual = float(np.median(np.linalg.norm(predicted - held_target, axis=1)))

    inlier_points = source if inlier_mask is None else source[inlier_mask.ravel() == 1]
    return Transform(matrix=matrix, tracked=len(source), inliers=inliers,
                     heldout_residual_px=residual,
                     coverage=_coverage(inlier_points, previous.shape[:2]),
                     status="ok")


def background_shift(transform: Transform, x: float, y: float) -> tuple[float, float]:
    """How far the camera alone moved the scene point at (x, y), in pixels."""
    if transform.matrix is None:
        raise ValueError(f"no transform available ({transform.status}); background motion "
                         f"is unknown for this pair, and identity is not a safe default")
    moved = _apply(transform.matrix, np.array([[x, y]], dtype=np.float32))[0]
    return float(moved[0] - x), float(moved[1] - y)


def corrected_displacement(first: tuple[float, float], second: tuple[float, float],
                           transform: Transform) -> float:
    """Distance the target moved relative to the scene, with camera motion removed."""
    if transform.matrix is None:
        raise ValueError(f"no transform available ({transform.status}); a displacement "
                         f"cannot be camera-corrected for this pair")
    expected = _apply(transform.matrix, np.array([first], dtype=np.float32))[0]
    return float(np.hypot(second[0] - expected[0], second[1] - expected[1]))
