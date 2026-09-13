"""Background-motion contract: what a camera transform must report before it is used.

All fixtures are synthetic images with a known applied motion, so the expected
answer is arithmetic rather than a judgement about footage.
"""

import cv2
import numpy as np
import pytest

import camera


def _textured(seed=0, size=(360, 640)):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, size, dtype=np.uint8)
    return cv2.GaussianBlur(noise, (5, 5), 0)


def _shifted(image, dx, dy):
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]),
                          borderMode=cv2.BORDER_REFLECT)


def test_a_known_translation_is_recovered():
    first = _textured()
    second = _shifted(first, 6.0, -3.0)

    transform = camera.estimate_transform(first, second)

    assert transform.status == "ok"
    dx, dy = camera.background_shift(transform, 320.0, 180.0)
    assert dx == pytest.approx(6.0, abs=0.3)
    assert dy == pytest.approx(-3.0, abs=0.3)


def test_the_transform_does_not_depend_on_any_tracked_object():
    # The same pair, estimated with no object information at all. Background motion
    # must be available on every pair, including frames where a target is missing.
    first = _textured(seed=1)
    second = _shifted(first, -4.0, 2.0)

    transform = camera.estimate_transform(first, second)

    assert transform.status == "ok"
    assert transform.inliers >= 8
    dx, dy = camera.background_shift(transform, 100.0, 100.0)
    assert (dx, dy) == pytest.approx((-4.0, 2.0), abs=0.3)


def test_a_held_out_residual_is_reported_and_small_for_a_rigid_shift():
    transform = camera.estimate_transform(_textured(seed=2), _shifted(_textured(seed=2), 5, 5))

    assert transform.heldout_residual_px is not None
    assert transform.heldout_residual_px < 1.0


def test_featureless_frames_fail_explicitly_rather_than_returning_identity():
    blank = np.full((360, 640), 128, dtype=np.uint8)

    transform = camera.estimate_transform(blank, blank)

    assert transform.status == "insufficient_features"
    assert transform.matrix is None
    with pytest.raises(ValueError, match="no transform"):
        camera.background_shift(transform, 10.0, 10.0)


def test_coverage_is_reported_so_a_one_corner_fit_is_visible():
    first = _textured(seed=3)
    second = _shifted(first, 3.0, 0.0)

    full = camera.estimate_transform(first, second)
    assert full.coverage == pytest.approx(1.0)

    # Features restricted to one quadrant: the fit may still succeed, but it is
    # extrapolating across the rest of the frame and must say so.
    mask = np.zeros(first.shape, dtype=np.uint8)
    mask[:180, :320] = 255
    corner = camera.estimate_transform(first, second, feature_mask=mask)
    assert corner.coverage <= 0.25


def test_a_minority_moving_actor_is_rejected_as_outliers_even_unmasked():
    # Worth pinning down: with the actor on a third of the frame, the robust fit
    # already ignores it, so a mask is not what saves this case.
    background = _textured(seed=4)
    first, second = background.copy(), _shifted(background, 2.0, 0.0)
    actor = _textured(seed=9, size=(360, 213))
    first[:, :213] = actor
    second[:, :213] = _shifted(actor, -14.0, 0.0)

    unmasked = camera.estimate_transform(first, second)

    assert camera.background_shift(unmasked, 500.0, 180.0)[0] == pytest.approx(2.0, abs=0.3)


def test_a_dominant_moving_actor_defeats_the_fit_unless_it_is_masked_out():
    # The case a mask is actually for: the moving region is the majority, so the
    # robust fit locks onto the actor and calls it the background.
    background = _textured(seed=4)
    first, second = background.copy(), _shifted(background, 2.0, 0.0)
    actor = _textured(seed=9, size=(360, 430))
    first[:, :430] = actor
    second[:, :430] = _shifted(actor, -14.0, 0.0)

    unmasked = camera.estimate_transform(first, second)
    assert abs(camera.background_shift(unmasked, 600.0, 180.0)[0] - 2.0) > 1.0

    mask = np.full(first.shape, 255, dtype=np.uint8)
    mask[:, :430] = 0
    masked = camera.estimate_transform(first, second, feature_mask=mask)
    assert camera.background_shift(masked, 600.0, 180.0)[0] == pytest.approx(2.0, abs=0.3)
    # And the fit says it only saw part of the frame.
    assert masked.coverage <= 0.5


def test_correcting_a_displacement_removes_the_camera_part():
    first = _textured(seed=5)
    second = _shifted(first, 4.0, 0.0)
    transform = camera.estimate_transform(first, second)

    # A target that moved 10px right in the image while the camera moved 4px right
    # has travelled 6px relative to the scene.
    corrected = camera.corrected_displacement((100.0, 100.0), (110.0, 100.0), transform)

    assert corrected == pytest.approx(6.0, abs=0.3)


def test_correction_is_refused_when_the_transform_failed():
    blank = np.full((360, 640), 128, dtype=np.uint8)
    failed = camera.estimate_transform(blank, blank)

    with pytest.raises(ValueError, match="no transform"):
        camera.corrected_displacement((0.0, 0.0), (1.0, 1.0), failed)
