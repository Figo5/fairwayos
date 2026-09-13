from pathlib import Path

import numpy as np
import pytest

import body


class FakeInterpreter:
    """Returns one fixed normalized keypoint so the inverse transform is testable."""

    def __init__(self, ky, kx, score=0.9):
        self.raw = np.zeros((1, 1, 17, 3), dtype=np.float32)
        self.raw[0, 0, :, 0] = ky
        self.raw[0, 0, :, 1] = kx
        self.raw[0, 0, :, 2] = score
        self.tensor_side = None

    def get_input_details(self):
        return [{"index": 0}]

    def get_output_details(self):
        return [{"index": 1}]

    def set_tensor(self, index, value):
        self.tensor_side = value.shape[1:3]

    def invoke(self):
        pass

    def get_tensor(self, index):
        return self.raw


def test_edge_clipped_crop_is_padded_to_square_not_stretched():
    # A box at the left edge clips to 220x320, which must be padded, never resized
    # to a square directly: that would change the golfer's proportions.
    crop = np.zeros((320, 220, 3), dtype=np.uint8)
    padded, pad_x, pad_y = body.pad_to_square(crop)

    assert padded.shape[0] == padded.shape[1] == 320
    assert (pad_x, pad_y) == (50, 0)


def test_already_square_crop_is_left_alone():
    padded, pad_x, pad_y = body.pad_to_square(np.zeros((200, 200, 3), dtype=np.uint8))

    assert padded.shape[:2] == (200, 200)
    assert (pad_x, pad_y) == (0, 0)


def test_padding_is_undone_so_keypoints_land_in_true_frame_pixels():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    box = (0, 0, 100, 300, 0.9)                 # clips to x 0..220, y 0..320
    x1, y1, x2, y2 = body.square_crop(box, 1280, 720)
    assert (x2 - x1, y2 - y1) == (220, 320)     # deliberately non-square after clipping

    # An off-centre point exposes the stretch: padded space is 320 wide with 50px
    # of padding each side, so kx=0.25 is 30px into the real crop, not 55px.
    points = body.keypoints(FakeInterpreter(ky=0.25, kx=0.25), frame, box)
    assert points[0][:2] == (x1 + 30, y1 + 80)


def test_keypoints_outside_the_frame_are_scored_zero_not_drawn():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    box = (1180, 420, 1280, 720, 0.9)

    # kx=1.0 maps past the right edge of the frame: impossible geometry.
    points = body.keypoints(FakeInterpreter(ky=0.5, kx=1.0), frame, box)
    x, _, score = points[0]
    assert x >= 1280 and score == 0.0

    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert body.draw(canvas, points) == 0
    assert not canvas.any()  # nothing was drawn at an impossible coordinate


def test_a_degenerate_box_is_still_widened_by_the_margin():
    # square_crop's margin means a zero-area box never produces an empty crop;
    # the empty-crop guard in keypoints() is defensive, not a reachable path here.
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert body.square_crop((10, 10, 10, 10, 0.9), 1280, 720) == (0, 0, 30, 30)
    assert len(body.keypoints(FakeInterpreter(0.5, 0.5), frame, (10, 10, 10, 10, 0.9))) == 17


def test_model_path_prefers_the_flag_then_the_environment(monkeypatch):
    monkeypatch.setenv("FAIRWAY_BODY_MODEL", "/env/weights.tflite")
    assert body.model_path("/flag/weights.tflite") == Path("/flag/weights.tflite")
    assert body.model_path() == Path("/env/weights.tflite")

    monkeypatch.delenv("FAIRWAY_BODY_MODEL")
    assert body.model_path() == body.DEFAULT_MODEL_PATH


def test_absent_weights_are_reported_as_a_missing_file_with_both_overrides(tmp_path):
    # Reported before the runtime import, so the message is about the file rather
    # than whatever LiteRT says when handed a path that is not there.
    with pytest.raises(FileNotFoundError) as caught:
        body.load_model(tmp_path / "not-here.tflite")

    message = str(caught.value)
    assert "not-here.tflite" in message
    assert "--body-model" in message and "FAIRWAY_BODY_MODEL" in message
