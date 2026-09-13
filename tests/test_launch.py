"""Launch-evidence contract: what evidence must exist before a number is shown.

Fixtures are labelled synthetic tracks, not recordings. Each states the single
property it exercises. Nothing here asserts anything about real golf speeds, and
no test supplies a capture frame rate chosen to make an answer look plausible.
"""

import pytest

import launch


def _obs(*frames_xy, kind="measured"):
    return [launch.Observation(f, x, y, kind) for f, x, y in frames_xy]


# --- evidence selection ----------------------------------------------------

def test_only_consecutive_frames_form_an_interval():
    # A gap is missing evidence: the frames either side are not one frame apart,
    # so their displacement is not one frame's travel.
    obs = _obs((10, 100.0, 200.0), (11, 100.0, 188.0), (13, 100.0, 164.0))

    usable, rejected = launch.intervals(obs)

    assert [(i.first.frame, i.second.frame) for i in usable] == [(10, 11)]
    assert any("11 -> 13" in r and "not consecutive" in r for r in rejected)


def test_an_identical_coordinate_pair_is_refused_rather_than_scored_zero():
    # The same coordinate twice is equally consistent with a stationary target and
    # with the model repeating its previous answer; it supports no speed either way.
    obs = _obs((10, 100.0, 200.0), (11, 100.0, 200.0))

    usable, rejected = launch.intervals(obs)

    assert usable == []
    assert any("identical coordinate" in r for r in rejected)


def test_a_frame_cannot_be_used_twice():
    with pytest.raises(ValueError, match="frame 10"):
        launch.intervals(_obs((10, 100.0, 200.0), (10, 101.0, 199.0)))


@pytest.mark.parametrize("bad", [True, False, float("nan"), float("inf"), "12", None])
def test_non_numeric_or_non_finite_coordinates_are_refused(bad):
    with pytest.raises(ValueError):
        launch.Observation(10, bad, 200.0, "measured")


def test_an_unknown_observation_kind_is_refused():
    with pytest.raises(ValueError, match="kind"):
        launch.Observation(10, 1.0, 2.0, "eyeballed")


def test_an_interval_is_semantic_if_either_end_is():
    obs = [launch.Observation(10, 100.0, 200.0, "measured"),
           launch.Observation(11, 100.0, 188.0, "semantic")]

    usable, _ = launch.intervals(obs)

    assert usable[0].kind == "semantic"


# --- scale -----------------------------------------------------------------

def test_the_scale_interval_comes_from_the_apparent_size_interval():
    scale = launch.scale_mm_per_px(9.0, 11.0)

    # 42.67mm across 11px is the smallest mm/px; across 9px the largest.
    assert scale.low_mm_per_px == pytest.approx(42.67 / 11.0)
    assert scale.high_mm_per_px == pytest.approx(42.67 / 9.0)
    assert "not a calibrated silhouette" in scale.caveat


@pytest.mark.parametrize("low,high", [(0.0, 11.0), (-9.0, 11.0), (11.0, 9.0),
                                      (True, 11.0), (float("inf"), 11.0)])
def test_a_scale_that_is_not_a_positive_ordered_interval_is_refused(low, high):
    with pytest.raises(ValueError):
        launch.scale_mm_per_px(low, high)


# --- timing ----------------------------------------------------------------

def test_a_timing_scenario_must_carry_a_positive_capture_rate():
    for bad in (0.0, -240.0, True, float("nan")):
        with pytest.raises(ValueError):
            launch.TimingScenario("bad", bad)


def test_a_scenario_is_conditional_unless_it_was_established_by_evidence():
    assumed = launch.TimingScenario("if captured at 240 fps", 240.0)
    established = launch.TimingScenario(
        "measured capture rate", 240.0,
        evidence=launch.Evidence("measured", "declared by the capture device"))

    assert assumed.established is False
    assert assumed.evidence.status == "assumed"
    assert established.established is True


# --- estimate --------------------------------------------------------------

def _rising_ball(kind="measured"):
    """Four consecutive frames, ~11.5px per frame. Pixel motion only; no claim."""
    return _obs((556, 749.0, 612.5), (557, 749.5, 601.0), (558, 750.0, 589.5),
                (559, 750.5, 578.0), kind=kind)


def test_without_timing_no_physical_speed_is_produced():
    result = launch.estimate(_rising_ball(), scale=launch.scale_mm_per_px(9.0, 11.0))

    assert result["physical_speed"]["scenarios"] == []
    assert result["physical_speed"]["reliability"]["level"] == "Unsupported"
    assert "capture frame rate is not established" in \
        result["physical_speed"]["reliability"]["reasons"]
    # Image motion is measured and is reported regardless: it needs no timing.
    assert result["image_motion"]["low"] == pytest.approx(11.51, abs=0.02)
    assert result["image_motion"]["reliability"]["level"] == "Moderate"


def test_without_scale_no_physical_speed_is_produced():
    result = launch.estimate(_rising_ball(),
                             scenarios=[launch.TimingScenario("if 240 fps", 240.0)])

    assert result["physical_speed"]["scenarios"] == []
    assert "image scale is not established" in \
        result["physical_speed"]["reliability"]["reasons"]


def test_a_scenario_range_spans_both_the_scale_and_the_measured_spread():
    scale = launch.scale_mm_per_px(9.0, 11.0)
    result = launch.estimate(_rising_ball(), scale=scale,
                             scenarios=[launch.TimingScenario("if 240 fps", 240.0)])

    scenario = result["physical_speed"]["scenarios"][0]
    low = result["image_motion"]["low"] * scale.low_mm_per_px / 1000.0 * 240.0
    high = result["image_motion"]["high"] * scale.high_mm_per_px / 1000.0 * 240.0
    assert scenario["low_mps"] == pytest.approx(low)
    assert scenario["high_mps"] == pytest.approx(high)
    assert scenario["conditional"] is True
    assert scenario["frames_used"] == [[556, 557], [557, 558], [558, 559]]


def test_every_reported_number_names_the_frames_it_came_from():
    result = launch.estimate(_rising_ball(), scale=launch.scale_mm_per_px(9.0, 11.0),
                             scenarios=[launch.TimingScenario("if 240 fps", 240.0)])

    assert result["evidence"]["frames"] == [556, 557, 558, 559]
    assert result["evidence"]["intervals_used"] == 3
    assert result["evidence"]["kinds"] == {"measured": 4}


# --- reliability -----------------------------------------------------------

def test_reliability_is_unsupported_without_usable_intervals():
    result = launch.estimate([], scale=launch.scale_mm_per_px(9.0, 11.0),
                             scenarios=[launch.TimingScenario("if 240 fps", 240.0)])

    assert result["image_motion"]["reliability"]["level"] == "Unsupported"
    assert result["physical_speed"]["reliability"]["level"] == "Unsupported"
    assert "no usable consecutive interval" in result["image_motion"]["reliability"]["reasons"]


def test_an_assumed_capture_rate_leaves_physical_speed_unsupported():
    # An assumed multiplier is an input, not a finding. However consistent the
    # frames are, it can never lift the physical speed above Unsupported, and it
    # must never be labelled Moderate.
    result = launch.estimate(_rising_ball(), scale=launch.scale_mm_per_px(9.0, 11.0),
                             scenarios=[launch.TimingScenario("if 240 fps", 240.0)])

    physical = result["physical_speed"]
    assert physical["reliability"]["level"] == "Unsupported"
    assert "capture frame rate is assumed, not established" in physical["reliability"]["reasons"]
    # The scenario is still offered, explicitly as a condition rather than a finding.
    assert physical["scenarios"][0]["conditional"] is True
    # And the measured image motion keeps its own, separate reliability.
    assert result["image_motion"]["reliability"]["level"] == "Moderate"


def test_an_established_capture_rate_is_the_only_route_to_a_supported_speed():
    established = launch.TimingScenario(
        "measured capture rate", 240.0,
        evidence=launch.Evidence("measured", "declared by the capture device"))

    result = launch.estimate(_rising_ball(), scale=launch.scale_mm_per_px(9.0, 11.0),
                             scenarios=[established])

    assert result["physical_speed"]["reliability"]["level"] == "Low"
    assert any("scale is conditional" in r
               for r in result["physical_speed"]["reliability"]["reasons"])


def test_semantic_evidence_caps_image_motion_reliability_at_low():
    result = launch.estimate(_rising_ball(kind="semantic"),
                             scale=launch.scale_mm_per_px(9.0, 11.0),
                             scenarios=[launch.TimingScenario("if 240 fps", 240.0)])

    reliability = result["image_motion"]["reliability"]
    assert reliability["level"] == "Low"
    assert any("semantic" in r for r in reliability["reasons"])


def test_inconsistent_intervals_lower_reliability_and_say_by_how_much():
    jumpy = _obs((556, 749.0, 612.0), (557, 749.0, 600.0), (558, 749.0, 596.0))

    result = launch.estimate(jumpy, scale=launch.scale_mm_per_px(9.0, 11.0),
                             scenarios=[launch.TimingScenario("if 240 fps", 240.0)])

    reliability = result["image_motion"]["reliability"]
    assert reliability["level"] == "Low"
    assert any("spread" in r for r in reliability["reasons"])


def test_two_frames_alone_are_low_not_moderate():
    result = launch.estimate(_obs((556, 749.0, 612.5), (557, 749.5, 601.0)),
                             scale=launch.scale_mm_per_px(9.0, 11.0),
                             scenarios=[launch.TimingScenario("if 240 fps", 240.0)])

    reliability = result["image_motion"]["reliability"]
    assert reliability["level"] == "Low"
    assert any("1 usable interval" in r for r in reliability["reasons"])


def test_identity_is_unverified_so_image_motion_stops_at_moderate():
    # Four clean consecutive measured frames with no spread at all: still not High,
    # because nothing in the pipeline verifies the points are the same object.
    result = launch.estimate(_rising_ball(), scale=launch.scale_mm_per_px(9.0, 11.0))

    reliability = result["image_motion"]["reliability"]
    assert reliability["level"] == "Moderate"
    assert any("identity" in r for r in reliability["reasons"])


# --- report ----------------------------------------------------------------

def test_the_report_carries_inputs_formula_and_every_rejection():
    obs = _rising_ball() + _obs((565, 751.0, 520.0))
    result = launch.estimate(obs, scale=launch.scale_mm_per_px(9.0, 11.0),
                             scenarios=[launch.TimingScenario("if 240 fps", 240.0)])
    document, text = launch.report(result, target="ball")

    assert document["inputs"]["scale"]["low_mm_per_px"] > 0
    assert "(px/frame) x (mm/px) / 1000 x capture_fps" in document["formula"]
    assert any("not consecutive" in r for r in document["rejected"])
    assert "if 240 fps" in text and "conditional" in text.lower()
    assert "carry" not in text.lower()


def test_the_report_states_the_blocker_when_timing_is_missing():
    result = launch.estimate(_rising_ball(), scale=launch.scale_mm_per_px(9.0, 11.0))
    document, text = launch.report(result, target="ball")

    assert document["physical_speed"]["scenarios"] == []
    assert "capture frame rate is not established" in text
    assert "m/s" not in text and "mph" not in text


# --- golfer-facing panel ---------------------------------------------------

def test_the_panel_shows_no_coordinates_or_confidences():
    result = launch.estimate(_rising_ball(), scale=launch.scale_mm_per_px(9.0, 11.0),
                             scenarios=[launch.TimingScenario("if 240 fps", 240.0)])

    lines = launch.panel_lines({"ball": result})

    body = " ".join(lines)
    assert "749" not in body and "612" not in body      # no native coordinates
    assert "conf" not in body.lower()
    assert "Ball speed" in body and "if 240 fps" in body
    # The measured half is Moderate; the physical half stays Unsupported.
    assert "Moderate" in body and "Unsupported" in body


def test_the_panel_says_unsupported_rather_than_showing_a_number():
    result = launch.estimate([], scale=None, scenarios=[])

    lines = launch.panel_lines({"ball": result})

    body = " ".join(lines)
    assert "Ball speed: Unsupported" in body
    assert "m/s" not in body and "mph" not in body


def test_the_panel_reports_each_target_independently():
    # Both are Unsupported as physical speeds while timing is unidentified, but for
    # different reasons, and only the tracked one carries measured movement.
    ok = launch.estimate(_rising_ball(), scale=launch.scale_mm_per_px(9.0, 11.0),
                         scenarios=[launch.TimingScenario("if 240 fps", 240.0)])
    none = launch.estimate([], scale=None, scenarios=[])

    lines = launch.panel_lines({"ball": ok, "club": none})
    ball = lines[lines.index("Ball speed: Unsupported"):lines.index("Club speed: Unsupported")]
    club = lines[lines.index("Club speed: Unsupported"):]

    assert any("capture frame rate is assumed" in line for line in ball)
    assert any("measured movement" in line for line in ball)
    assert any("no usable consecutive interval" in line for line in club)
    assert not any("measured movement" in line for line in club)


def test_the_panel_shows_a_phase_only_when_evidence_supplies_one():
    result = launch.estimate(_rising_ball(), scale=launch.scale_mm_per_px(9.0, 11.0),
                             scenarios=[launch.TimingScenario("if 240 fps", 240.0)])

    assert not any("Phase" in line for line in launch.panel_lines({"ball": result}))
    assert any("Phase: ball departure" in line
               for line in launch.panel_lines({"ball": result},
                                              phase="ball departure (frames 556-559)"))


# --- regressions from parent review ----------------------------------------

def _parent_repro():
    return [launch.Observation(i, i * 5.0, 0.0, "measured") for i in range(3)]


def test_an_assumed_timing_flag_cannot_produce_moderate_or_high():
    # Parent repro. Wholly assumed time and a highlight-based scale: the speed is
    # Unsupported, whatever the frames look like.
    result = launch.estimate(_parent_repro(), scale=launch.scale_mm_per_px(9.0, 11.0),
                             scenarios=[launch.TimingScenario("unverified", 300.0)])

    assert result["physical_speed"]["reliability"]["level"] == "Unsupported"


@pytest.mark.parametrize("bogus", [True, False, "false", "measured", 1, None])
def test_timing_evidence_must_be_an_evidence_record_not_a_flag(bogus):
    # A caller assertion is not evidence: no bool and no string can promote timing.
    with pytest.raises(ValueError, match="evidence"):
        launch.TimingScenario("unverified", 300.0, evidence=bogus)


@pytest.mark.parametrize("bogus", [True, "scale", 4.2, [9, 11]])
def test_a_scale_that_is_not_a_scale_interval_is_refused(bogus):
    with pytest.raises(ValueError, match="scale"):
        launch.estimate(_parent_repro(), scale=bogus)


@pytest.mark.parametrize("bogus", [True, "false", "measured", 1])
def test_identity_evidence_must_be_an_evidence_record(bogus):
    with pytest.raises(ValueError, match="identity"):
        launch.estimate(_parent_repro(), identity=bogus)


def test_evidence_status_must_be_known_and_measured_evidence_must_name_its_source():
    with pytest.raises(ValueError, match="status"):
        launch.Evidence("probably", "vibes")
    with pytest.raises(ValueError, match="source"):
        launch.Evidence("measured", "")


def test_measured_timing_with_an_assumed_scale_is_low_not_high():
    measured_timing = launch.TimingScenario(
        "capture rate from camera metadata", 300.0,
        evidence=launch.Evidence("measured", "declared by the capture device"))

    result = launch.estimate(_parent_repro(), scale=launch.scale_mm_per_px(9.0, 11.0),
                             scenarios=[measured_timing])

    reliability = result["physical_speed"]["reliability"]
    assert reliability["level"] == "Low"
    assert any("scale" in r for r in reliability["reasons"])
    assert any("camera motion" in r for r in reliability["reasons"])


def test_a_fully_evidenced_chain_can_still_reach_high():
    # The grades must not be permanently pinned down: real calibration, real timing,
    # real camera support and verified identity together do reach High.
    scale = launch.scale_mm_per_px(9.0, 11.0,
                                   evidence=launch.Evidence("measured", "surveyed target"))
    result = launch.estimate(
        _parent_repro(), scale=scale,
        scenarios=[launch.TimingScenario("measured", 300.0,
                                         evidence=launch.Evidence("measured", "device"))],
        camera=launch.Evidence("measured", "affine background transform, held-out 0.4px"),
        identity=launch.Evidence("measured", "per-frame identity adjudication"))

    assert result["physical_speed"]["reliability"]["level"] == "High"


def test_uncorrected_displacement_is_never_described_as_corrected():
    result = launch.estimate(_parent_repro(), scale=launch.scale_mm_per_px(9.0, 11.0),
                             scenarios=[launch.TimingScenario("if 300 fps", 300.0)])
    document, text = launch.report(result, target="ball")

    assert document["camera_support"]["status"] == "absent"
    assert "not removed" in text
    assert "corrected" not in text.replace("uncorrected", "")


def test_the_formula_states_the_unit_conversion_it_actually_performs():
    scale = launch.scale_mm_per_px(9.0, 11.0)
    result = launch.estimate(_parent_repro(), scale=scale,
                             scenarios=[launch.TimingScenario("if 300 fps", 300.0)])
    row = result["physical_speed"]["scenarios"][0]

    assert "1000" in launch.FORMULA and "m/s" in launch.FORMULA
    expected = result["image_motion"]["low"] * scale.low_mm_per_px / 1000.0 * 300.0
    assert row["low_mps"] == pytest.approx(expected)


def test_scenario_rows_carry_their_own_grade_separate_from_measurement():
    result = launch.estimate(_parent_repro(), scale=launch.scale_mm_per_px(9.0, 11.0),
                             scenarios=[launch.TimingScenario("if 300 fps", 300.0)])

    row = result["physical_speed"]["scenarios"][0]
    assert row["grade"] == "conditional scenario, not a measurement"
    assert row["conditional"] is True


# --- deriving the moving window from the observations ----------------------

def test_a_stationary_then_moving_track_separates_without_naming_a_frame():
    # Six frames: three at rest with sub-pixel jitter, three travelling. The split
    # comes from the displacements themselves; no frame number appears in the code.
    obs = _obs((100, 10.0, 10.0), (101, 10.3, 10.1), (102, 10.1, 10.4),
               (103, 22.0, 10.0), (104, 34.0, 10.2), (105, 46.0, 10.1))

    split = launch.split_by_motion(obs)

    assert split["separable"] is True
    assert [i.frames for i in split["moving"]] == [[102, 103], [103, 104], [104, 105]]
    assert all(i.px < 1.0 for i in split["stationary"])


def test_a_track_with_one_motion_state_is_not_split():
    # All intervals alike: there is no evidence of two states, so none is invented.
    obs = _obs((100, 10.0, 10.0), (101, 22.0, 10.0), (102, 34.0, 10.0), (103, 46.0, 10.0))

    split = launch.split_by_motion(obs)

    assert split["separable"] is False
    assert "no clear separation" in split["reason"]
    assert split["moving"] == []


def test_too_few_intervals_to_separate_says_so():
    split = launch.split_by_motion(_obs((100, 10.0, 10.0), (101, 22.0, 10.0)))

    assert split["separable"] is False
    assert "at least" in split["reason"]


def test_the_moving_window_must_be_consecutive():
    # Two bursts of motion separated by rest: only the longest consecutive run is
    # taken, because a window assembled from separate bursts is not one event.
    obs = _obs((100, 10.0, 10.0), (101, 10.2, 10.0), (102, 40.0, 10.0),
               (103, 40.2, 10.0), (104, 70.0, 10.0), (105, 100.0, 10.0))

    split = launch.split_by_motion(obs)

    assert [i.frames for i in split["moving"]] == [[103, 104], [104, 105]]
