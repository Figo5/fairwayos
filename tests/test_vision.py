import subprocess
from pathlib import Path

import pytest

import vision


def test_json_is_extracted_from_a_chatty_answer_and_refused_when_absent():
    assert vision._extract_json('here you go {"ball": null} hope that helps') == {"ball": None}
    assert vision._extract_json("I cannot see a ball.") is None
    assert vision._extract_json('{"ball": ') is None   # truncated
    assert vision._extract_json('["a", "b"]') is None  # not an object


def test_points_outside_the_frame_or_non_numeric_are_refused():
    assert vision._point({"x": 810, "y": 372, "conf": 0.86}, 1280, 720) == (810, 372, 0.86)
    assert vision._point({"x": 1280, "y": 372, "conf": 0.9}, 1280, 720) is None  # off right
    assert vision._point({"x": -1, "y": 10, "conf": 0.9}, 1280, 720) is None     # off left
    assert vision._point({"x": "810", "y": 10, "conf": 0.9}, 1280, 720) is None  # numeric str
    assert vision._point({"x": True, "y": 10, "conf": 0.9}, 1280, 720) is None   # bool
    assert vision._point({"x": 10, "y": 10}, 1280, 720) is None                  # no conf
    assert vision._point(None, 1280, 720) is None


def test_confidence_must_be_a_real_number_in_range():
    assert vision._confidence(0.0) == 0.0 and vision._confidence(1.0) == 1.0
    assert vision._confidence(1.01) is None
    assert vision._confidence(-0.1) is None
    assert vision._confidence(float("nan")) is None
    assert vision._confidence(float("inf")) is None
    assert vision._confidence("0.9") is None
    assert vision._confidence(True) is None   # bool is not a confidence

    # A point carrying any of those is refused outright.
    assert vision._point({"x": 10, "y": 10, "conf": float("nan")}, 1280, 720) is None
    assert vision._point({"x": 10, "y": 10, "conf": 1.5}, 1280, 720) is None


def test_prompt_geometry_comes_from_the_frame_not_a_hardcoded_size():
    prompt = vision.PROMPT_TEMPLATE.format(width=1920, height=1080, max_x=1919, max_y=1079)

    assert "1920x1080" in prompt and "0..1919" in prompt and "0..1079" in prompt
    assert "1280" not in prompt and "720" not in prompt


def test_boxes_must_be_ordered_and_inside_the_frame():
    good = {"x1": 475, "y1": 180, "x2": 657, "y2": 646, "conf": 0.94}
    assert vision._box(good, 1280, 720) == (475, 180, 657, 646, 0.94)
    assert vision._box({**good, "x1": 657, "x2": 475}, 1280, 720) is None   # inverted
    assert vision._box({**good, "x2": 1281}, 1280, 720) is None             # past edge
    assert vision._box({**good, "x1": 10, "x2": 10}, 1280, 720) is None     # zero width
    assert vision._box({**good, "conf": "high"}, 1280, 720) is None
    assert vision._box({**good, "y1": False}, 1280, 720) is None            # bool


def test_a_missing_cli_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(vision.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no hermes")))
    result = vision.read_frame(Path("frame.png"), 1280, 720)

    assert "could not run" in result["error"]
    assert "ball" not in result  # a failed call contributes no observation


def test_an_unreadable_image_is_reported_before_any_cli_call(monkeypatch):
    monkeypatch.setattr(vision.subprocess, "run",
                        lambda *a, **k: pytest.fail("CLI must not run without an image"))
    result = vision.read_frame(Path("does-not-exist.png"))

    assert "cannot read image" in result["error"]


def test_the_image_path_sent_to_the_cli_is_always_absolute(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, '{"golfer":null,"ball":null,"clubhead":null}', "")

    monkeypatch.setattr(vision.subprocess, "run", fake_run)
    vision.read_frame(Path("relative/frame.png"), 1280, 720)

    # The CLI resets its working directory, so a relative path fails there.
    assert Path(seen["cmd"][seen["cmd"].index("--image") + 1]).is_absolute()
    assert "--provider" in seen["cmd"] and "--max-turns" in seen["cmd"]
