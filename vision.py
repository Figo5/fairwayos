"""Fresh per-frame vision acquisition through the local Hermes CLI.

One subprocess per frame image, bounded timeout and turns, strict JSON out. The
prompt carries no coordinates and no prior answers, so nothing is seeded: every
number returned is the model's own reading of that one image. A frame that
fails, times out, or answers with anything but parseable in-frame JSON yields no
observation.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import cv2

CLI = "hermes"
MODEL = "gpt-5.5"
PROVIDER = "openai-codex"
REASONING = "low"
MAX_TURNS = 1
TIMEOUT_SECONDS = 120

PROMPT_TEMPLATE = """This is a single {width}x{height} frame from a golf broadcast. Report ONLY what you can actually see in this image.

Return strict JSON, nothing else:
{{"golfer": {{"x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>, "conf": <0.0-1.0>}} or null,
 "ball": {{"x": <int>, "y": <int>, "conf": <0.0-1.0>}} or null,
 "clubhead": {{"x": <int>, "y": <int>, "conf": <0.0-1.0>}} or null}}

Rules:
- Coordinates are pixels, origin top-left, x in 0..{max_x}, y in 0..{max_y}.
- "golfer" is the tight bounding box of the single player swinging or addressing the ball. Not a spectator, caddie, or camera operator. If several people are visible, pick only the player holding the club.
- "ball" is the center of the small white golf ball only. Not a shoe, hat, logo, or sign.
- "clubhead" is the center of the metal head at the far end of the golf club shaft, away from the hands.
- If you cannot clearly see an object in THIS image, return null for it. Do not guess a plausible location.
"""


def _number(value: Any) -> float | None:
    """Accept only a real finite number. Rejects bools and numeric strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _confidence(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and 0.0 <= number <= 1.0 else None


def _extract_json(text: str) -> dict | None:
    """Pull the JSON object out of a CLI answer, or give up."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _point(value: Any, width: int, height: int) -> tuple[int, int, float] | None:
    """Accept a point only if every field is a real number inside the frame."""
    if not isinstance(value, dict):
        return None
    x, y = _number(value.get("x")), _number(value.get("y"))
    # The coarse schema names it "conf", the refine schema "confidence".
    raw_conf = value["conf"] if "conf" in value else value.get("confidence")
    conf = _confidence(raw_conf)
    if x is None or y is None or conf is None:
        return None
    if not (0 <= x < width and 0 <= y < height):
        return None
    return int(x), int(y), conf


def _box(value: Any, width: int, height: int) -> tuple[int, int, int, int, float] | None:
    if not isinstance(value, dict):
        return None
    x1, y1 = _number(value.get("x1")), _number(value.get("y1"))
    x2, y2 = _number(value.get("x2")), _number(value.get("y2"))
    conf = _confidence(value.get("conf"))
    if None in (x1, y1, x2, y2) or conf is None:
        return None
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return None
    return int(x1), int(y1), int(x2), int(y2), conf


def _run_cli(path: Path, prompt: str) -> dict:
    """Run the vision CLI on one image. Never raises."""
    try:
        done = subprocess.run(
            [CLI, "chat", "--image", str(path), "--query-file", "-",
             "--oneshot", "-Q", "-m", MODEL, "--provider", PROVIDER,
             "--reasoning", REASONING, "--max-turns", str(MAX_TURNS)],
            input=prompt, capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"vision CLI timed out after {TIMEOUT_SECONDS}s"}
    except OSError as exc:
        return {"error": f"vision CLI could not run: {exc}"}

    if done.returncode != 0:
        return {"error": f"vision CLI exited {done.returncode}",
                "stderr": done.stderr[-300:], "raw": done.stdout[-300:]}

    parsed = _extract_json(done.stdout)
    if parsed is None:
        return {"error": "vision CLI returned no parseable JSON", "raw": done.stdout[-300:]}
    return {"parsed": parsed}


def read_frame(image: Path, width: int | None = None,
               height: int | None = None) -> dict:
    """Ask the vision CLI about one frame image. Never raises; reports failure instead.

    The CLI resets its working directory, so the image path is always made
    absolute. Frame geometry comes from the image itself unless supplied, so the
    prompt never states dimensions the image does not have.
    """
    path = Path(image).resolve()
    if width is None or height is None:
        loaded = cv2.imread(str(path))
        if loaded is None:
            return {"error": f"cannot read image {path}"}
        height, width = loaded.shape[:2]

    prompt = PROMPT_TEMPLATE.format(width=width, height=height,
                                    max_x=width - 1, max_y=height - 1)
    answer = _run_cli(path, prompt)
    if "error" in answer:
        return answer
    parsed = answer["parsed"]

    return {
        "golfer": _box(parsed.get("golfer"), width, height),
        "ball": _point(parsed.get("ball"), width, height),
        "clubhead": _point(parsed.get("clubhead"), width, height),
    }


# --- Ball refinement -------------------------------------------------------
# A second look at a small crop around the coarse proposal. This runs on a
# DIFFERENT backend from the coarse pass: the Codex CLI, which is the backend the
# bounded four-frame refinement experiment was accepted on. Swapping it for
# another CLI would not inherit that acceptance, so the backend is recorded
# alongside every refined point.
REFINE_CLI = "codex"
REFINE_MODEL = "gpt-5.5"
CROP_SIZE = 192
CROP_UPSCALE = 2
MODEL_INPUT_SIZE = CROP_SIZE * CROP_UPSCALE

_REFINE_SUBJECT = {
    "ball": "Identify the physical golf ball center only.",
    # The shaft is long, thin and also dark; the head is the compact mass at its
    # end. Naming that difference is what stops the point landing on the shaft.
    "clubhead": ("Identify the center of the physical club HEAD only: the solid "
                 "compact mass at the end of the shaft. Do NOT mark the thin "
                 "shaft, and do not mark a point above or beside the head."),
}
_REFINE_ABSENT = {"ball": "physical ball", "clubhead": "physical club head"}


def refine_schema(target: str) -> str:
    return (
        'Return only strict JSON: {"visible": true, "x": number, "y": number, '
        '"confidence": number, "notes": string} or {"visible": false, "x": null, '
        '"y": null, "confidence": 0, "notes": string}. Coordinates x,y must be in '
        f'the attached {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE} image local pixel '
        f'coordinates, origin top-left. {_REFINE_SUBJECT[target]} '
        f'If no {_REFINE_ABSENT[target]} is visible, return visible false. '
        'Do not use outside information.'
    )


def backends() -> dict:
    """Which backend produced which stage. Recorded with the outputs."""
    return {
        "coarse": {"cli": CLI, "model": MODEL, "provider": PROVIDER,
                   "reasoning": REASONING},
        "refine": {"cli": REFINE_CLI, "model": REFINE_MODEL,
                   "reasoning": "medium (Codex CLI default)"},
    }


def refine(crop_image: Path, message_file: Path, target: str) -> dict:
    """Locate one target inside an upscaled crop, in crop-local pixels.

    Returns {"point": (x, y, conf)} when the model says it sees a ball,
    {"point": None} when it says it does not, or {"error": ...}. The prompt
    states no prior coordinate: the crop is the only input, so this is a fresh
    reading rather than a correction applied to the coarse answer.
    """
    path = Path(crop_image).resolve()
    # The CLI writes its answer here. Any file already at this path is an earlier
    # run's answer, and reading that back would be exactly the seeding this
    # pipeline refuses, so it goes before the call rather than after.
    message_file.unlink(missing_ok=True)
    try:
        done = subprocess.run(
            [REFINE_CLI, "exec", "--skip-git-repo-check", "-m", REFINE_MODEL,
             "--image", str(path), "--output-last-message", str(message_file),
             refine_schema(target)],
            capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"refine CLI timed out after {TIMEOUT_SECONDS}s"}
    except OSError as exc:
        return {"error": f"refine CLI could not run: {exc}"}

    if done.returncode != 0:
        return {"error": f"refine CLI exited {done.returncode}",
                "stderr": done.stderr[-300:]}

    text = message_file.read_text() if message_file.is_file() else done.stdout
    parsed = _extract_json(text)
    if parsed is None:
        return {"error": "refine CLI returned no parseable JSON", "raw": text[-300:]}

    if parsed.get("visible") is not True:
        return {"point": None}
    point = _point(parsed, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    if point is None:
        return {"error": "refine CLI claimed visible with an unusable point",
                "raw": text[-300:]}
    return {"point": point}


# --- Semantic fallback for an undecided head -------------------------------
# Frozen from the bounded five-case verification the parent reviewed
# (/tmp/fairway-semantic-head-check): same CLI, model, reasoning effort, sandbox
# and prompt text, so the acceptance that run earned is the acceptance this uses.
# It answers one yes/no about a candidate point that the refine pass already
# returned. It is never asked for a position and cannot supply one, so it cannot
# introduce a location of its own. Anything but a plain "supported" is a no.
VERIFY_PROMPT = """You are verifying a proposed golf clubhead point in a single image crop. The attached image is a clean unmarked crop. Candidate point coordinates are in this crop's image pixel coordinate system, with origin at top-left. Decide whether the candidate point lies on the physical compact golf club head itself. Do not count shaft, trouser/body, turf, background, shadow, or empty pixels as the club head. Be strict. Return exactly one of these words, then one short reason: supported, unsupported, uncertain. Do not provide corrected coordinates or alternative labels.

Candidate point: ({x}, {y})"""


def verify_head(crop_image: Path, message_file: Path, model_x: float,
                model_y: float) -> dict:
    """Ask whether a candidate point lies on the physical club head. Fail-closed.

    Returns {"supported": bool, "verdict": str} on a readable answer, or
    {"error": ...}. Only the exact word "supported" is a yes: an unreadable
    answer, an unexpected word, a non-zero exit or a timeout all leave the point
    unsupported, because a fallback that guesses on failure would be inventing
    exactly what the measurement could not establish.
    """
    path = Path(crop_image).resolve()
    message_file.unlink(missing_ok=True)
    prompt = VERIFY_PROMPT.format(x=int(model_x), y=int(model_y))
    try:
        done = subprocess.run(
            [REFINE_CLI, "exec", "--skip-git-repo-check", "--sandbox", "read-only",
             "-m", REFINE_MODEL, "-c", 'model_reasoning_effort="medium"',
             "--image", str(path), "-o", str(message_file), "-"],
            input=prompt, capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"head verification timed out after {TIMEOUT_SECONDS}s"}
    except OSError as exc:
        return {"error": f"head verification could not run: {exc}"}

    if done.returncode != 0:
        return {"error": f"head verification exited {done.returncode}",
                "stderr": done.stderr[-300:]}

    text = message_file.read_text() if message_file.is_file() else done.stdout
    words = text.strip().lower().replace(",", " ").split()
    if not words:
        return {"error": "head verification returned nothing"}
    verdict = words[0]
    return {"supported": verdict == "supported", "verdict": verdict,
            "answer": text.strip()[:200]}
