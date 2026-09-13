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


class _Done:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _verify(monkeypatch, tmp_path, answer=None, returncode=0, raises=None):
    """Run verify_head against a canned CLI answer written where the CLI writes it."""
    message = tmp_path / "verify.txt"

    def fake_run(cmd, **kwargs):
        if raises is not None:
            raise raises
        if answer is not None:
            Path(cmd[cmd.index("-o") + 1]).write_text(answer)
        return _Done(returncode=returncode, stderr="boom")

    monkeypatch.setattr(vision.subprocess, "run", fake_run)
    return vision.verify_head(tmp_path / "crop.png", message, 194, 201)


def test_only_a_plain_supported_answer_confirms_a_point(monkeypatch, tmp_path):
    answer = "supported, the point lies within the dark compact club head."
    assert _verify(monkeypatch, tmp_path, answer) == {
        "supported": True, "verdict": "supported", "answer": answer}


def test_every_other_answer_leaves_the_point_unsupported(monkeypatch, tmp_path):
    for answer in ("unsupported, the point is on turf/background, not the club head.",
                   "uncertain, the crop is too blurred to tell.",
                   "Supported? I cannot say.",          # not the bare word
                   "the point looks like the head to me",
                   "{\"supported\": true}"):            # right sentiment, wrong schema
        result = _verify(monkeypatch, tmp_path, answer)
        assert result["supported"] is False, answer


def test_a_failed_or_empty_verification_is_an_error_not_a_yes(monkeypatch, tmp_path):
    assert "error" in _verify(monkeypatch, tmp_path, "supported, sure.", returncode=1)
    assert "error" in _verify(monkeypatch, tmp_path, "   \n")
    assert "error" in _verify(monkeypatch, tmp_path,
                              raises=subprocess.TimeoutExpired(cmd="codex", timeout=120))
    assert "error" in _verify(monkeypatch, tmp_path, raises=OSError("no codex"))


def test_the_verification_prompt_is_the_frozen_one_and_carries_the_candidate(monkeypatch,
                                                                            tmp_path):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"], seen["prompt"] = cmd, kwargs["input"]
        Path(cmd[cmd.index("-o") + 1]).write_text("uncertain, no.")
        return _Done()

    monkeypatch.setattr(vision.subprocess, "run", fake_run)
    vision.verify_head(tmp_path / "crop.png", tmp_path / "v.txt", 194.4, 201.6)

    # The bounded review that accepted this fallback used exactly this backend.
    assert seen["cmd"][:3] == ["codex", "exec", "--skip-git-repo-check"]
    assert "read-only" in seen["cmd"] and 'model_reasoning_effort="medium"' in seen["cmd"]
    assert seen["prompt"].endswith("Candidate point: (194, 201)")
    assert "Do not provide corrected coordinates" in seen["prompt"]
