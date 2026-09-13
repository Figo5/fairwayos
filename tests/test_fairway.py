import subprocess
from pathlib import Path

import pytest

import fairway
import vision


def _stats(**overrides):
    stats = {"frames": 30, "body_frames": 0, "errors": 0, "keypoints": [],
             "image_width": 1280, "image_height": 720,
             "ball_unconfirmed": 0, "clubhead_unconfirmed": 0,
             "ball_failed": 0, "clubhead_failed": 0,
             "ball_inconclusive": 0, "clubhead_inconclusive": 0,
             "ball_semantic": 0, "clubhead_semantic": 0,
             "ball": [], "ball_conf": [], "clubhead": [], "clubhead_conf": []}
    stats.update(overrides)
    return stats


def _row(stats, label):
    return next(o for o in fairway.summarize(stats)[0] if o["label"] == label)


def test_continuity_run_breaks_on_gaps_and_jumps():
    assert fairway.longest_continuous_run([]) == 0
    assert fairway.longest_continuous_run([(1, 10, 10), (2, 20, 14), (3, 31, 19)]) == 3
    assert fairway.longest_continuous_run([(1, 10, 10), (5, 20, 14)]) == 1   # frame gap
    assert fairway.longest_continuous_run([(1, 10, 10), (2, 900, 14)]) == 1  # teleport


def test_repeated_coordinate_warning_counts_the_known_vision_failure():
    # A model that has stopped looking repeats one coordinate across frames.
    assert fairway.repeated_coordinates([(1, 5, 5), (2, 5, 5), (3, 5, 5)]) == 3
    assert fairway.repeated_coordinates([(1, 5, 5), (2, 6, 7)]) == 1
    assert fairway.repeated_coordinates([]) == 0

    note = _row(_stats(clubhead=[(i, 5, 5) for i in range(6)],
                       clubhead_conf=[0.9] * 6), "clubhead")["note"]
    assert "one identical coordinate recurs in 6 frames" in note


def test_targets_never_reported_claim_nothing():
    for label in ("ball", "clubhead"):
        row = _row(_stats(), label)
        assert row["visible"] is False
        assert "not reported by the vision pass" in row["note"]
    assert _row(_stats(), "body")["visible"] is False


def test_failed_vision_frames_are_disclosed():
    assert _row(_stats(errors=4), "vision")["note"].startswith("4 of 30 frames")
    assert not [o for o in fairway.summarize(_stats())[0] if o["label"] == "vision"]


def test_a_long_continuous_run_still_does_not_assert_a_valid_target():
    # A false lock onto a fixed background object produces a perfect run, so
    # continuity must never be what sets valid_target.
    locked = _stats(clubhead=[(i, 500, 500) for i in range(30)], clubhead_conf=[0.99] * 30)
    assert fairway.longest_continuous_run(locked["clubhead"]) == 30

    flags = fairway.summarize(locked)[1]
    assert flags == {"valid_target": False, "calibration": False,
                     "action_time": False, "landing": False}
    assert fairway.summarize(_stats())[1] == flags


def test_no_phase_events_are_invented():
    assert "phase_events" not in fairway.summarize(_stats())[1]


def test_keypoints_are_passed_through_with_an_explicit_coordinate_space():
    triples = [[float(i), float(i * 2), 0.9] for i in range(17)]
    rows = [o for o in fairway.summarize(_stats(body_frames=1, keypoints=[(7, triples)]))[0]
            if "keypoints" in o]

    assert len(rows) == 1
    assert rows[0]["keypoint_space"] == "native_pixel"
    assert rows[0]["frame"] == 7
    assert rows[0]["keypoints"] == triples
    # Dimensions must travel with the row so analytics can reject impossible points.
    assert (rows[0]["image_width"], rows[0]["image_height"]) == (1280, 720)


def test_body_note_does_not_treat_confidence_as_occlusion_evidence():
    note = _row(_stats(body_frames=5, keypoints=[]), "body")["note"]
    assert "not evidence that the joint was occluded" in note


def test_keypoint_rows_without_dimensions_withhold_angles_instead_of_crashing():
    triples = [[float(i), float(i), 0.9] for i in range(17)]
    stats = _stats(body_frames=1, keypoints=[(7, triples)])
    del stats["image_width"], stats["image_height"]

    row = next(o for o in fairway.summarize(stats)[0] if "keypoints" in o)

    # No coordinate space means analytics cannot validate, so it withholds angles.
    assert "keypoint_space" not in row
    assert row["keypoints"] == triples


def test_unusable_dimensions_also_withhold_rather_than_being_trusted():
    triples = [[float(i), float(i), 0.9] for i in range(17)]
    for bad in ({"image_width": 0, "image_height": 720},
                {"image_width": -1280, "image_height": 720}):
        stats = _stats(body_frames=1, keypoints=[(7, triples)], **bad)
        row = next(o for o in fairway.summarize(stats)[0] if "keypoints" in o)
        assert "keypoint_space" not in row


SOURCE = Path("/tmp/fairway-three-target-sources/downloads/6083133193001.mp4")


@pytest.mark.skipif(not SOURCE.is_file(), reason="source video not present")
def test_source_rate_is_the_exact_rational_not_a_rounded_float():
    rate = fairway.source_frame_rate(SOURCE)

    # OpenCV's float rounds to 2997/100 when encoded; the source is 30000/1001.
    assert rate == "30000/1001"
    assert rate != "2997/100"


@pytest.mark.parametrize("reported", ["N/A", "", "0/0", "30000/0", "30000", "abc/def"])
def test_a_video_without_a_usable_rational_rate_is_refused(monkeypatch, reported):
    # "N/A" is ffprobe's unknown answer and contains a slash, so a slash alone
    # is not enough to call it a rate.
    monkeypatch.setattr(fairway.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, reported + "\n", ""))
    with pytest.raises(SystemExit, match="no rational frame rate"):
        fairway.source_frame_rate(Path("x.mp4"))


def test_crop_geometry_matches_the_accepted_experiment():
    # 192 window centred on the coarse point, origin deliberately unclamped.
    assert fairway.crop_origin(819, 427) == (723, 331)
    assert fairway.crop_origin(10, 10) == (-86, -86)      # near an edge: stays negative

    # The accepted transform: native = origin + model/2 at 2x upscale.
    assert fairway.to_native(191.0, 256.0, 723, 331) == (818.5, 459.0)


def test_edge_crop_is_black_padded_rather_than_recentred():
    import numpy as np
    frame = np.full((720, 1280, 3), 255, dtype=np.uint8)

    crop = fairway.build_crop(frame, -86, -86)

    assert crop.shape == (192, 192, 3)
    assert not crop[:86, :86].any()          # outside the frame stays black
    assert crop[86:, 86:].all()              # real pixels land at the right offset


def test_a_wholly_offscreen_crop_is_all_padding_not_a_crash():
    import numpy as np
    frame = np.full((720, 1280, 3), 255, dtype=np.uint8)

    assert not fairway.build_crop(frame, -500, -500).any()


def test_unconfirmed_coarse_proposals_are_disclosed():
    note = _row(_stats(ball=[(i, 10 + i, 10) for i in range(4)], ball_conf=[0.9] * 4,
                       ball_unconfirmed=3), "ball")["note"]

    assert "3 coarse proposal(s) were left unmarked" in note
    assert "codex" in note and "hermes" in note   # both backends named


def _turf(bright_at=None, size=200, level=125):
    """Flat turf-like field, optionally with one small bright ball-like blob."""
    import numpy as np
    gray = np.full((size, size), level, dtype=np.uint8)
    gray += np.random.default_rng(0).integers(-6, 7, gray.shape, dtype=np.int8).astype(np.uint8)
    if bright_at:
        import cv2
        cv2.circle(gray, bright_at, 5, 248, -1)
    return gray


def test_a_point_on_a_ball_like_blob_is_supported():
    gray = _turf(bright_at=(100, 100))

    assert fairway.is_supported(gray, 100, 100, "ball")
    assert fairway.local_contrast(gray, 100, 100) > fairway.BALL_SUPPORT_MIN


def test_a_point_on_bare_turf_is_not_supported():
    # Null control: nothing bright anywhere, so no position can be backed.
    gray = _turf()

    assert not fairway.is_supported(gray, 100, 100, "ball")
    assert not fairway.is_supported(gray, 40, 160, "ball")


def test_a_point_near_but_off_the_blob_is_not_supported():
    # The generic form of a refined point landing beside the ball rather than on it.
    gray = _turf(bright_at=(100, 130))

    assert fairway.is_supported(gray, 100, 130, "ball")
    assert not fairway.is_supported(gray, 100, 100, "ball")   # ~30px above the blob


def test_an_off_frame_point_is_unmeasurable_not_measured_at_zero():
    # A zero would read back as a real measurement on flat turf. None says the
    # point was never measured, which is what vision_raw.json must record.
    gray = _turf(bright_at=(100, 100))

    assert fairway.local_contrast(gray, -5, 100) is None
    assert fairway.local_contrast(gray, 100, 999) is None
    assert fairway.support_value(gray, -5, 100, "ball") is None
    assert not fairway.is_supported(gray, -5, 100, "ball")


def test_a_uniformly_bright_field_has_no_blob_to_support_a_point():
    # Bright pixels alone are not evidence: support is contrast against surroundings.
    gray = _turf(level=250)

    assert not fairway.is_supported(gray, 100, 100, "ball")


def _turf_with(shape, level=150):
    """Turf-like field carrying one drawn shape, for head support controls."""
    import cv2
    import numpy as np
    gray = np.full((200, 200), level, dtype=np.uint8)
    gray += np.random.default_rng(1).integers(-6, 7, gray.shape, dtype=np.int8).astype(np.uint8)
    if shape == "head":                      # compact dark mass
        cv2.circle(gray, (100, 100), 14, 40, -1)
    elif shape == "shaft":                   # thin dark line through the point
        cv2.line(gray, (60, 60), (140, 140), 40, 4)
    return gray


def test_a_point_on_a_compact_dark_head_is_supported():
    gray = _turf_with("head")

    assert fairway.is_supported(gray, 100, 100, "clubhead")
    darkness, fill, background, decides = fairway.head_support(gray, 100, 100)
    assert darkness >= fairway.HEAD_DARK_MARGIN and fill >= fairway.HEAD_FILL_MIN
    assert decides and background > fairway.HEAD_DARK_MARGIN


def test_a_point_on_the_thin_shaft_is_rejected_though_it_is_dark():
    # The shaft is as dark as the head; only its thickness tells them apart.
    gray = _turf_with("shaft")
    darkness, fill = fairway.head_support(gray, 100, 100)[:2]

    assert darkness >= fairway.HEAD_DARK_MARGIN    # dark enough on its own
    assert fill < fairway.HEAD_FILL_MIN            # but nowhere near thick enough
    # The measurement did decide here: the shaft pixels could have passed it.
    assert fairway.measurement_decides(gray, 100, 100, "clubhead")
    assert not fairway.is_supported(gray, 100, 100, "clubhead")


def test_a_point_in_the_air_above_the_head_is_rejected():
    # The generic form of the rejected transfer frames: marker high, on turf.
    gray = _turf_with("head")

    assert not fairway.is_supported(gray, 100, 60, "clubhead")


def test_bare_turf_supports_no_head():
    assert not fairway.is_supported(_turf_with(None), 100, 100, "clubhead")


def test_head_and_ball_signatures_do_not_accept_each_other():
    bright = _turf(bright_at=(100, 100))
    dark = _turf_with("head")

    assert fairway.is_supported(bright, 100, 100, "ball")
    assert not fairway.is_supported(bright, 100, 100, "clubhead")
    assert fairway.is_supported(dark, 100, 100, "clubhead")
    assert not fairway.is_supported(dark, 100, 100, "ball")


def test_head_support_off_frame_is_unmeasurable_not_zero():
    assert fairway.head_support(_turf_with("head"), -3, 100) is None
    assert fairway.support_value(_turf_with("head"), -3, 100, "clubhead") is None
    assert not fairway.is_supported(_turf_with("head"), -3, 100, "clubhead")


def test_both_vision_clis_are_checked_before_the_expensive_pass(monkeypatch):
    # The refine CLI runs once per proposal AFTER the coarse pass, so a missing
    # refine CLI used to surface only once every frame had already been paid for.
    monkeypatch.setattr(fairway.body, "load_model", lambda model=None: "interpreter")
    for missing in (vision.CLI, vision.REFINE_CLI):
        monkeypatch.setattr(fairway.shutil, "which",
                            lambda name, missing=missing: None if name == missing else "/bin/x")
        with pytest.raises(SystemExit) as caught:
            fairway.check_prerequisites()
        assert missing in str(caught.value)


def test_prerequisites_return_the_loaded_interpreter_so_it_is_loaded_once(monkeypatch):
    loads = []
    monkeypatch.setattr(fairway.body, "load_model",
                        lambda model=None: loads.append(model) or "interpreter")
    monkeypatch.setattr(fairway.shutil, "which", lambda name: "/bin/x")

    assert fairway.check_prerequisites(Path("/w.tflite")) == "interpreter"
    assert loads == [Path("/w.tflite")]


class _Done:
    """A finished subprocess, for the paths that only inspect returncode/stdout."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _one_ball_proposal():
    import numpy as np
    frame = np.zeros((200, 200, 3), dtype=np.uint8)      # flat black: supports nothing
    gray = np.zeros((200, 200), dtype=np.uint8)
    return [frame], [gray], [{"golfer": None, "ball": (100, 100, 0.9), "clubhead": None}]


def test_a_refine_backend_outage_is_not_recorded_as_an_image_rejection(monkeypatch, tmp_path):
    # Every attempt errors, so no image measurement was ever made. Recording that
    # as "not backed by the image" would report a broken backend as evidence.
    frames, grays, results = _one_ball_proposal()
    monkeypatch.setattr(fairway.vision, "refine",
                        lambda crop, message, target: {"error": "refine CLI exited 1"})

    record = fairway.refine_targets(frames, grays, results, 7, tmp_path, 1, "ball")[7]

    assert "rejected" not in record and "native_point" not in record
    assert "did not run to an answer" in record["failed"]
    assert all("error" in attempt for attempt in record["attempts"])


def test_an_attempt_that_was_measured_and_failed_stays_an_image_rejection(monkeypatch,
                                                                         tmp_path):
    frames, grays, results = _one_ball_proposal()
    answers = [{"error": "refine CLI timed out after 120s"}, {"point": (10, 10, 0.9)}]
    messages = []

    def fake_refine(crop, message, target):
        messages.append(message)
        return answers.pop(0)

    monkeypatch.setattr(fairway.vision, "refine", fake_refine)
    record = fairway.refine_targets(frames, grays, results, 7, tmp_path, 1, "ball")[7]

    assert "failed" not in record
    assert "was backed by the image" in record["rejected"]
    # Each attempt has its own message file, so one cannot read back another's answer.
    assert len(set(messages)) == len(messages) == 2


def test_a_failed_crop_pass_is_disclosed_without_claiming_the_image_refused_it():
    note = _row(_stats(ball_failed=3), "ball")["note"]

    assert "3 coarse proposal(s)" in note and "did not run to an answer" in note
    assert "nothing here says the image refused them" in note


def test_a_short_interval_is_refused_rather_than_silently_shrinking_the_report():
    class Capture:
        def __init__(self, available):
            self.left = available

        def isOpened(self):
            return True

        def grab(self):
            return True

        def read(self):
            import numpy as np
            if self.left <= 0:
                return False, None
            self.left -= 1
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self):
            pass

    import cv2 as real_cv2
    original = real_cv2.VideoCapture
    try:
        real_cv2.VideoCapture = lambda path: Capture(12)
        with pytest.raises(SystemExit) as caught:
            fairway.read_interval(Path("clip.mp4"), 900, 40)
        assert "only 12 of 40" in str(caught.value) and "--frames 12" in str(caught.value)

        real_cv2.VideoCapture = lambda path: Capture(40)
        assert len(fairway.read_interval(Path("clip.mp4"), 900, 40)) == 40
    finally:
        real_cv2.VideoCapture = original


def test_encode_caps_the_sequence_at_this_runs_frames(monkeypatch, tmp_path):
    # An earlier, longer run's leftovers in the same directory are consecutive
    # files, which ffmpeg would otherwise keep reading into this video.
    seen = {}
    monkeypatch.setattr(fairway.subprocess, "run",
                        lambda cmd, **kwargs: seen.setdefault("cmd", cmd) and _Done())
    monkeypatch.setattr(fairway, "source_frame_rate", lambda video: "30000/1001")

    fairway.encode(tmp_path, tmp_path / "annotated.mp4", "30000/1001", 240, 48)

    cmd = seen["cmd"]
    assert cmd[cmd.index("-start_number") + 1] == "240"
    assert cmd[cmd.index("-frames:v") + 1] == "48"


def test_a_dropped_frame_write_stops_the_run_instead_of_shortening_the_video(monkeypatch,
                                                                            tmp_path):
    monkeypatch.setattr(fairway.cv2, "imwrite", lambda *args: False)

    with pytest.raises(SystemExit) as caught:
        fairway.write_image(tmp_path / "a00000.png", None)
    assert "output is incomplete" in str(caught.value)


def test_an_ffmpeg_without_libx264_is_refused_before_the_expensive_pass(monkeypatch):
    monkeypatch.setattr(fairway.body, "load_model", lambda model=None: "interpreter")
    monkeypatch.setattr(fairway.shutil, "which", lambda name: "/bin/x")
    monkeypatch.setattr(fairway.subprocess, "run",
                        lambda *args, **kwargs: _Done(stdout="V..... libvpx\n"))

    with pytest.raises(SystemExit) as caught:
        fairway.check_prerequisites()
    assert "libx264" in str(caught.value)


def test_a_head_against_a_background_as_dark_as_itself_is_undecided_not_refused():
    # The measurement is relative: where the surround is as dark as the head, no
    # point could pass it whatever is there, so a failure is not a refusal. This
    # is the frozen Furyk 267-287 regime, where the head crosses dark trousers.
    import cv2
    import numpy as np
    dark = np.full((200, 200), 30, dtype=np.uint8)      # head on dark clothing
    cv2.circle(dark, (100, 100), 14, 12, -1)
    on_turf = _turf_with("head")                        # the same head on turf

    assert not fairway.is_supported(dark, 100, 100, "clubhead")
    assert not fairway.measurement_decides(dark, 100, 100, "clubhead")
    # Bare bright turf still decides: nothing dark there is a real negative.
    assert fairway.measurement_decides(_turf_with(None), 100, 100, "clubhead")
    # An empty patch of the same dark cloth measures the same, which is the point:
    # present and absent are indistinguishable there.
    assert fairway.head_support(dark, 100, 100)[1] == fairway.head_support(dark, 40, 40)[1]

    assert fairway.is_supported(on_turf, 100, 100, "clubhead")
    assert fairway.measurement_decides(on_turf, 100, 100, "clubhead")


def test_an_undecidable_rejection_is_recorded_as_undecided_not_as_an_image_refusal(
        monkeypatch, tmp_path):
    import numpy as np
    frame = np.full((200, 200, 3), 30, dtype=np.uint8)   # uniformly dark surroundings
    gray = np.full((200, 200), 30, dtype=np.uint8)
    results = [{"golfer": None, "ball": None, "clubhead": (100, 100, 0.9)}]
    monkeypatch.setattr(fairway.vision, "refine",
                        lambda crop, message, target: {"point": (192, 192, 0.9)})
    monkeypatch.setattr(fairway.vision, "verify_head",
                        lambda crop, message, x, y: {"supported": False,
                                                     "verdict": "unsupported"})

    record = fairway.refine_targets([frame], [gray], results, 9, tmp_path, 1, "clubhead")[9]

    assert "rejected" not in record and "native_point" not in record
    assert "cannot tell a clubhead that is present from one that is not" \
        in record["inconclusive"]
    assert all(attempt["decides"] is False for attempt in record["attempts"])


def test_a_decidable_rejection_is_still_an_image_refusal(monkeypatch, tmp_path):
    # Bright surroundings with nothing dark in them: the measurement could have
    # passed, and did not. That is a refusal of the point and evidence of absence,
    # and must not be softened into "undecided".
    import numpy as np
    frame = np.full((200, 200, 3), 150, dtype=np.uint8)
    gray = frame[:, :, 0].copy()
    results = [{"golfer": None, "ball": None, "clubhead": (100, 100, 0.9)}]
    monkeypatch.setattr(fairway.vision, "refine",
                        lambda crop, message, target: {"point": (192, 192, 0.9)})

    record = fairway.refine_targets([frame], [gray], results, 9, tmp_path, 1, "clubhead")[9]

    assert "inconclusive" not in record
    assert "was backed by the image" in record["rejected"]


def test_an_undecided_target_is_disclosed_as_neither_present_nor_absent():
    note = _row(_stats(clubhead_inconclusive=4), "clubhead")["note"]

    assert "4 coarse proposal(s)" in note and "could not decide" in note
    assert "not evidence that the clubhead was absent" in note
    assert "not evidence that it was present either" in note


def _undecided_head(monkeypatch, tmp_path, verify):
    """One clubhead proposal on uniformly dark pixels, where the gate cannot decide."""
    import numpy as np
    frame = np.full((200, 200, 3), 30, dtype=np.uint8)
    gray = np.full((200, 200), 30, dtype=np.uint8)
    results = [{"golfer": None, "ball": None, "clubhead": (100, 100, 0.9)}]
    monkeypatch.setattr(fairway.vision, "refine",
                        lambda crop, message, target: {"point": (192, 192, 0.9)})
    monkeypatch.setattr(fairway.vision, "verify_head", verify)
    return fairway.refine_targets([frame], [gray], results, 9, tmp_path, 1, "clubhead")[9]


def test_the_semantic_fallback_confirms_the_point_the_crop_pass_already_returned(
        monkeypatch, tmp_path):
    asked = {}

    def verify(crop, message, x, y):
        asked["point"] = (x, y)
        return {"supported": True, "verdict": "supported"}

    record = _undecided_head(monkeypatch, tmp_path, verify)

    # It is asked about the refine pass's own candidate, never for a position.
    assert asked["point"] == tuple(record["attempts"][0]["model_point"])
    assert record["native_point"] == record["attempts"][0]["native_point"]
    assert record["semantic_only"] is True
    # And the frame is still recorded as one the measurement could not decide.
    assert "inconclusive" in record and "image_support" not in record


def test_the_semantic_fallback_fails_closed_on_no_error_and_junk(monkeypatch, tmp_path):
    for answer in ({"supported": False, "verdict": "unsupported"},
                   {"supported": False, "verdict": "uncertain"},
                   {"error": "head verification timed out after 120s"},
                   {"error": "head verification returned nothing"},
                   {}):
        record = _undecided_head(monkeypatch, tmp_path, lambda c, m, x, y, a=answer: a)

        assert "native_point" not in record and "semantic_only" not in record
        assert record["semantic_check"] == answer


def test_the_semantic_fallback_never_runs_where_the_measurement_decided(monkeypatch,
                                                                       tmp_path):
    # Bright surroundings: the measurement decided and refused. A fallback here
    # would be overriding evidence, not filling a gap.
    import numpy as np
    frame = np.full((200, 200, 3), 150, dtype=np.uint8)
    gray = frame[:, :, 0].copy()
    results = [{"golfer": None, "ball": None, "clubhead": (100, 100, 0.9)}]
    monkeypatch.setattr(fairway.vision, "refine",
                        lambda crop, message, target: {"point": (192, 192, 0.9)})
    monkeypatch.setattr(fairway.vision, "verify_head", _must_not_run)

    record = fairway.refine_targets([frame], [gray], results, 9, tmp_path, 1, "clubhead")[9]
    assert "rejected" in record and "semantic_check" not in record


def test_the_semantic_fallback_never_runs_for_the_ball(monkeypatch, tmp_path):
    import numpy as np
    frame = np.full((200, 200, 3), 30, dtype=np.uint8)
    gray = np.full((200, 200), 30, dtype=np.uint8)
    results = [{"golfer": None, "ball": (100, 100, 0.9), "clubhead": None}]
    monkeypatch.setattr(fairway.vision, "refine",
                        lambda crop, message, target: {"point": (192, 192, 0.9)})
    monkeypatch.setattr(fairway.vision, "verify_head", _must_not_run)

    record = fairway.refine_targets([frame], [gray], results, 9, tmp_path, 1, "ball")[9]
    assert "semantic_check" not in record


def _must_not_run(*args, **kwargs):
    raise AssertionError("the semantic fallback ran where it must not")


def test_semantically_confirmed_markers_are_disclosed_as_judgement_not_measurement():
    note = _row(_stats(clubhead_semantic=6, clubhead=[(1, 2, 3)],
                       clubhead_conf=[0.9]), "clubhead")["note"]

    assert "6 of the marked clubhead positions rest on no image measurement" in note
    assert "one model's judgement of a crop, not a measurement" in note
    assert "labelled semantic in the video" in note
