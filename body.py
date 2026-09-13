"""Body keypoints from the local MoveNet SinglePose TFLite model via LiteRT.

MoveNet is single-pose: it needs a crop holding one person, which the per-frame
vision pass supplies. No weights are loaded by unpickling. Keypoints below the
score threshold are dropped rather than drawn at a guessed position.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

# Weights are never fetched: the resolved path must already be a local file.
# --body-model wins over FAIRWAY_BODY_MODEL, which wins over this default.
DEFAULT_MODEL_PATH = Path("/tmp/fairway-safe-body-runtime/movenet-singlepose-lightning-v3.tflite")
INPUT_SIZE = 192
MIN_SCORE = 0.3
# COCO-17 limbs, as MoveNet orders its keypoints.
EDGES = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
         (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))


def model_path(override: Path | str | None = None) -> Path:
    """Where the MoveNet weights are, by explicit path, environment, or default."""
    return Path(override or os.environ.get("FAIRWAY_BODY_MODEL") or DEFAULT_MODEL_PATH)


def load_model(path: Path | str | None = None):
    """Load the local TFLite weights. Existence is checked before the runtime
    import so a missing file is reported as a missing file, not as a runtime error."""
    resolved = model_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"no MoveNet TFLite weights at {resolved}. Nothing is downloaded: point "
            f"--body-model or FAIRWAY_BODY_MODEL at a local .tflite file")

    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(resolved))
    interpreter.allocate_tensors()
    return interpreter


def square_crop(box: tuple[int, int, int, int, float], width: int, height: int,
                margin: int = 20) -> tuple[int, int, int, int]:
    """The square region we want around a person box, clipped to the frame.

    At a frame edge the clip makes this rectangular, so it is NOT square on its
    own. `keypoints` pads it back to square before inference rather than
    stretching it, which is what keeps body proportions intact.
    """
    x1, y1, x2, y2 = box[:4]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = max(x2 - x1, y2 - y1) // 2 + margin
    return (max(cx - half, 0), max(cy - half, 0),
            min(cx + half, width), min(cy + half, height))


def pad_to_square(crop: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Letterbox a crop to a square. Returns the square and the padding offsets."""
    crop_h, crop_w = crop.shape[:2]
    side = max(crop_h, crop_w)
    pad_x, pad_y = (side - crop_w) // 2, (side - crop_h) // 2
    padded = cv2.copyMakeBorder(crop, pad_y, side - crop_h - pad_y,
                                pad_x, side - crop_w - pad_x,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded, pad_x, pad_y


def keypoints(interpreter, frame: np.ndarray,
              box: tuple[int, int, int, int, float]) -> list[tuple[int, int, float]]:
    """Return one (x, y, score) per MoveNet keypoint, in full-frame pixels.

    The crop is letterboxed to a square before the 192x192 resize, so a golfer at
    a frame edge is padded rather than stretched. The padding is subtracted again
    when mapping keypoints back, so coordinates stay in true frame pixels. A point
    landing outside the frame is impossible geometry and is scored 0.0 so it is
    neither drawn nor reported.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = square_crop(box, width, height)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return []

    padded, pad_x, pad_y = pad_to_square(crop)
    side = padded.shape[0]
    tensor = cv2.resize(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB), (INPUT_SIZE, INPUT_SIZE))
    input_details = interpreter.get_input_details()[0]
    interpreter.set_tensor(input_details["index"], tensor.astype(np.float32)[None])
    interpreter.invoke()
    raw = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0, 0]

    points = []
    for ky, kx, score in raw:
        x = int(x1 + kx * side - pad_x)
        y = int(y1 + ky * side - pad_y)
        inside = 0 <= x < width and 0 <= y < height
        points.append((x, y, float(score) if inside else 0.0))
    return points


def draw(canvas: np.ndarray, points: list[tuple[int, int, float]]) -> int:
    """Draw only keypoints the model actually scored. Returns how many were drawn."""
    drawn = 0
    for index_a, index_b in EDGES:
        if points[index_a][2] >= MIN_SCORE and points[index_b][2] >= MIN_SCORE:
            cv2.line(canvas, points[index_a][:2], points[index_b][:2], (0, 255, 0), 2)
    for x, y, score in points:
        if score >= MIN_SCORE:
            cv2.circle(canvas, (x, y), 4, (0, 255, 0), -1)
            drawn += 1
    return drawn
