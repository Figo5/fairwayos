import pytest

from analytics import build_report


def test_build_report_requires_metadata_dict():
    with pytest.raises(ValueError, match="metadata"):
        build_report([], None)


def test_report_withholds_physical_metrics_without_required_evidence():
    report = build_report([], {})

    assert "Ball speed: unavailable" in report
    assert "missing physical calibration" in report
    assert "missing reliable action time" in report
    assert "Clubhead speed: unavailable" in report
    assert "Carry distance: unavailable" in report
    assert "missing observed landing" in report


def test_report_includes_bounded_qualitative_observation_notes():
    report = build_report(
        [
            {
                "frame": 12,
                "label": "clubhead",
                "visible": True,
                "confidence": 0.82,
                "note": "clubhead briefly visible near impact",
            }
        ],
        {"calibration": False, "action_time": False},
    )

    assert "Swing mechanics: qualitative observations only" in report
    assert "frame 12" in report
    assert "clubhead briefly visible near impact" in report
    assert "not definitive biomechanics" in report


def _blank_keypoints():
    return [[0.0, 0.0, 0.0] for _ in range(17)]


def test_report_summarizes_supported_2d_elbow_and_knee_angles_from_movenet_keypoints():
    keypoints = _blank_keypoints()
    keypoints[5] = [0.0, 0.0, 0.95]  # left shoulder
    keypoints[7] = [1.0, 0.0, 0.95]  # left elbow
    keypoints[9] = [1.0, 1.0, 0.95]  # left wrist: 90-degree elbow
    keypoints[12] = [0.0, 0.0, 0.95]  # right hip
    keypoints[14] = [1.0, 0.0, 0.95]  # right knee
    keypoints[16] = [1.0, 1.0, 0.95]  # right ankle: 90-degree knee

    report = build_report(
        [{"frame": 3, "time_s": 0.1, "keypoints": keypoints, "keypoint_space": "pixel"}], {}
    )

    assert "2D body angle summaries (image-space; not 3D biomechanics)" in report
    assert "Left elbow: frame 3 90.0°" in report
    assert "Right knee: frame 3 90.0°" in report


def test_normalized_keypoint_angles_require_dimensions_for_image_space_aspect_ratio():
    keypoints = _blank_keypoints()
    # Interior points: normalized 1.0 scales to exactly width, which is one past
    # the last pixel. These are scaled-in equivalents holding the same 29.4 angle.
    keypoints[5] = [0.1, 0.1, 1.0]
    keypoints[7] = [0.1, 0.9, 1.0]
    keypoints[9] = [0.9, 0.1, 1.0]

    report = build_report(
        [
            {
                "frame": 4,
                "keypoints": keypoints,
                "keypoint_space": "normalized",
                "image_width": 1280,
                "image_height": 720,
            }
        ],
        {},
    )

    assert "Left elbow: frame 4 29.4°" in report


def test_keypoints_without_explicit_coordinate_space_are_not_interpreted():
    keypoints = _blank_keypoints()
    # Interior points: normalized 1.0 scales to exactly width, which is one past
    # the last pixel. These are scaled-in equivalents holding the same 29.4 angle.
    keypoints[5] = [0.1, 0.1, 1.0]
    keypoints[7] = [0.1, 0.9, 1.0]
    keypoints[9] = [0.9, 0.1, 1.0]

    report = build_report([{"frame": 4, "keypoints": keypoints}], {})

    assert "No supported 2D body angles from confident visible keypoints." in report
    assert "45.0°" not in report


def test_pixel_keypoint_angles_require_explicit_pixel_space_declaration():
    keypoints = _blank_keypoints()
    keypoints[5] = [0.0, 0.0, 1.0]
    keypoints[7] = [0.0, 1280.0, 1.0]
    keypoints[9] = [720.0, 0.0, 1.0]

    report = build_report(
        [{"frame": 4, "keypoints": keypoints, "keypoint_space": "pixel"}], {}
    )

    assert "Left elbow: frame 4 29.4°" in report


def test_report_rejects_low_confidence_and_nonfinite_keypoints_without_fake_numbers():
    keypoints = _blank_keypoints()
    keypoints[5] = [0.0, 0.0, 0.95]
    keypoints[7] = [float("nan"), 0.0, 0.95]
    keypoints[9] = [1.0, 1.0, 0.95]
    keypoints[11] = [0.0, 0.0, 0.95]
    keypoints[13] = [1.0, 0.0, 0.49]
    keypoints[15] = [1.0, 1.0, 0.95]

    report = build_report([{"frame": 9, "keypoints": keypoints, "keypoint_space": "pixel"}], {})

    assert "No supported 2D body angles from confident visible keypoints." in report
    assert "nan" not in report.lower()
    assert "0.0°" not in report


def test_report_formats_only_verified_phase_event_timing_metadata():
    report = build_report(
        [],
        {
            "phase_events": [
                {"label": "address", "frame": 10, "time_s": 0.333, "verified": True},
                {"label": "impact", "frame": 25, "time_s": 0.833, "verified": True},
                {"label": "follow-through", "frame": 40, "time_s": 1.333, "verified": False},
            ]
        },
    )

    assert "Verified phase/event timing (supplied metadata only; no phase detection)" in report
    assert "address: frame 10, t=0.333s" in report
    assert "address → impact: 15 frames, 0.500s" in report
    assert "follow-through" not in report


def test_report_rejects_bool_numeric_frame_confidence_above_one_and_truthy_verified_string():
    keypoints = _blank_keypoints()
    keypoints[5] = [0.0, 0.0, 1.0]
    keypoints[7] = [1.0, 0.0, 1.01]
    keypoints[9] = [1.0, 1.0, 1.0]

    report = build_report(
        [
            {
                "frame": True,
                "keypoints": keypoints,
                "keypoint_space": "pixel",
                "confidence": True,
                "note": "bool confidence must not be rendered as numeric confidence",
            }
        ],
        {
            "phase_events": [
                {"label": "impact", "frame": True, "time_s": 0.5, "verified": True},
                {"label": "finish", "frame": 30, "time_s": 1.0, "verified": "yes"},
            ]
        },
    )

    assert "No supported 2D body angles from confident visible keypoints." in report
    assert "frame True" not in report
    assert "confidence True" not in report
    assert "Verified phase/event timing" not in report


def _metadata():
    return {"calibration": False, "action_time": False,
            "valid_target": False, "landing": False}


def _keypoints(**overrides):
    points = [[0.0, 0.0, 0.0] for _ in range(17)]
    points[5] = [100.0, 100.0, 0.99]   # left shoulder
    points[7] = [200.0, 100.0, 0.99]   # left elbow
    points[9] = [200.0, 200.0, 0.99]   # left wrist
    observation = {"frame": 3, "keypoints": points, "keypoint_space": "native_pixel"}
    observation.update(overrides)
    return observation


def test_native_pixel_angles_are_reported_when_points_fit_the_stated_frame():
    report = build_report([_keypoints(image_width=640, image_height=480)], _metadata())

    assert "Left elbow: frame 3" in report


def test_impossible_native_pixel_points_are_withheld_not_turned_into_angles():
    # x=1000 cannot exist in a 640-wide frame. An angle from it looks plausible.
    far = _keypoints(image_width=640, image_height=480)
    far["keypoints"][7] = [200.0, 1000.0, 0.99]

    report = build_report([far], _metadata())

    assert "Left elbow" not in report
    assert "No supported 2D body angles" in report


def test_a_point_exactly_on_the_far_edge_is_outside_the_frame():
    edge = _keypoints(image_width=640, image_height=480)
    edge["keypoints"][7] = [200.0, 640.0, 0.99]   # valid x is 0..639

    assert "Left elbow" not in build_report([edge], _metadata())


def test_bounds_are_only_enforced_when_the_observation_states_them():
    # Without dimensions there is nothing to check against, so the contract's
    # pixel space still passes unvalidated rather than silently dropping data.
    unbounded = _keypoints()
    unbounded["keypoints"][7] = [200.0, 1000.0, 0.99]

    assert "Left elbow: frame 3" in build_report([unbounded], _metadata())


def test_unusable_dimensions_do_not_enable_bounds_checking():
    for bad in ({"image_width": 0, "image_height": 480},
                {"image_width": -640, "image_height": 480},
                {"image_width": "640", "image_height": 480}):
        assert "Left elbow: frame 3" in build_report([_keypoints(**bad)], _metadata())


def test_normalized_keypoints_are_scaled_and_bounds_checked():
    points = [[0.0, 0.0, 0.0] for _ in range(17)]
    points[5] = [0.2, 0.2, 0.99]
    points[7] = [0.4, 0.2, 0.99]
    points[9] = [0.4, 0.4, 0.99]
    good = {"frame": 1, "keypoints": points, "keypoint_space": "normalized",
            "image_width": 640, "image_height": 480}
    assert "Left elbow: frame 1" in build_report([good], _metadata())

    outside = {**good, "keypoints": [list(p) for p in points]}
    outside["keypoints"][7] = [0.4, 1.5, 0.99]   # scales to x=960 in a 640 frame
    assert "Left elbow" not in build_report([outside], _metadata())
