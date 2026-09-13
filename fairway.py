"""Fairway CLI: annotate a local golf video interval with body, ball and clubhead.

Every marker comes from inference run on that exact frame: the golfer box, ball
and clubhead from a fresh per-frame vision call, the body keypoints from MoveNet
on the golfer crop. Nothing is interpolated, carried forward, or seeded with a
prior answer. A target the frame's inference did not report gets no marker.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

import body
import vision
from analytics import build_report

BALL_COLOR = (0, 215, 255)
CLUBHEAD_COLOR = (0, 0, 255)
# Consecutive-frame observations further apart than this are not the same object.
CONTINUITY_PX = 120


def source_frame_rate(video: Path) -> str:
    """The source's exact rational frame rate, as ffprobe reports it.

    OpenCV only offers a float, and encoding from that yields 2997/100, which is
    not the source's 30000/1001. The output must carry the exact rational.
    """
    try:
        done = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise SystemExit(f"ffprobe did not answer within 60s for {video}")
    rate = done.stdout.strip()
    # ffprobe says "N/A" when it does not know, which contains a slash but is not
    # a rate, so both parts are parsed as positive integers before it is trusted.
    numerator, _, denominator = rate.partition("/")
    if (done.returncode != 0 or not numerator.isdigit() or not denominator.isdigit()
            or int(numerator) <= 0 or int(denominator) <= 0):
        raise SystemExit(f"ffprobe reported no rational frame rate for {video}: {rate!r}")
    return rate


def encode(frame_dir: Path, out_video: Path, rate: str, start: int, count: int) -> None:
    """Encode this run's annotated frames at the exact rational rate, then verify it stuck.

    ffmpeg's image sequence reader keeps consuming consecutive files, so a longer
    earlier run's leftovers in this directory would be encoded into this video as
    if they belonged to it. -frames:v stops it at the frames this run produced.
    """
    try:
        done = subprocess.run(
            ["ffmpeg", "-y", "-framerate", rate, "-start_number", str(start),
             "-i", str(frame_dir / "a%05d.png"), "-frames:v", str(count),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", rate, str(out_video)],
            capture_output=True, text=True, timeout=900,
        )
    except subprocess.TimeoutExpired:
        raise SystemExit(f"ffmpeg did not finish within 900s writing {out_video}")
    if done.returncode != 0:
        raise SystemExit(f"ffmpeg failed: {done.stderr[-400:]}")

    written = source_frame_rate(out_video)
    if written != rate:
        raise SystemExit(f"output rate {written} does not match source rate {rate}")


def write_image(path: Path, image) -> None:
    """Write a PNG, refusing to continue if it did not land.

    A dropped frame (full disk, unwritable volume) would otherwise leave a gap the
    encoder stops at, giving a short video under a report claiming every frame.
    """
    if not cv2.imwrite(str(path), image):
        raise SystemExit(f"could not write {path}; output is incomplete")


def read_interval(video: Path, start: int, count: int) -> list:
    """Decode a continuous interval, skipping forward frame by frame (no keyframe seek).

    Timing is not taken from here: OpenCV only offers a float rate, so the output
    is timed from ffprobe's exact rational instead.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    try:
        for _ in range(start):
            if not cap.grab():
                raise SystemExit(f"video ends before frame {start}")
        frames = []
        for _ in range(count):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        # A short interval would silently become the report's denominator, so the
        # count the user asked for is either decoded in full or refused.
        if 0 < len(frames) < count:
            raise SystemExit(
                f"only {len(frames)} of {count} frames decode from {start}; the clip ends "
                f"there. Re-run with --frames {len(frames)} or an earlier --start-frame")
        return frames
    finally:
        cap.release()


def crop_origin(x: float, y: float, size: int = vision.CROP_SIZE) -> tuple[int, int]:
    """Top-left of a size x size window centred on a point, deliberately unclamped.

    The origin is not shifted back inside the frame: keeping it lets the inverse
    transform stay a plain subtraction, and a point near an edge is black-padded
    instead of silently re-centred onto different pixels.
    """
    return math.floor(x - size / 2), math.floor(y - size / 2)


def build_crop(frame, x0: int, y0: int, size: int = vision.CROP_SIZE):
    """Cut a size x size crop at an unclamped origin, black-padding outside pixels."""
    import numpy as np

    height, width = frame.shape[:2]
    crop = np.zeros((size, size, 3), dtype=frame.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(width, x0 + size), min(height, y0 + size)
    if sx1 > sx0 and sy1 > sy0:
        crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return crop


def to_native(model_x: float, model_y: float, x0: int, y0: int,
              upscale: int = vision.CROP_UPSCALE) -> tuple[float, float]:
    """Map a point in the upscaled crop back to native frame pixels."""
    return x0 + model_x / upscale, y0 + model_y / upscale


# These gate acceptance of a refined point: they never move one, so they cannot
# invent a location. What they measure is strictly local brightness against the
# immediate surroundings -- a bright peak near the point for the ball, a dark
# region filling the point's neighbourhood for the club head. That rejects a point
# on flat turf and a point on the thin shaft. It is NOT target recognition: any
# other bright object (a white shoe, a broadcast graphic, a specular highlight) or
# any other large dark region (a trouser leg, a shadow edge) satisfies the same
# measurement, so a gated point is evidence of local contrast, not of identity.
REFINE_ATTEMPTS = 2
BALL_SUPPORT_MIN = 60.0
# One margin, used twice: the fill fraction counts pixels this far below the
# background, so a non-zero fill already implies the point is that dark. A
# separate darkness threshold at the same value would never be the failing half.
HEAD_FILL_MIN = 0.35
HEAD_DARK_MARGIN = 40.0
SUPPORT_INNER = 8
SUPPORT_OUTER = 20
HEAD_INNER = 9
HEAD_OUTER = 26


def _windows(gray, x: int, y: int, inner: int, outer: int):
    """Centre and surrounding windows at a point, or None if the point is off-frame.

    Near a frame border both windows are truncated, so the surround sits to one
    side of the point rather than around it and the centre covers fewer pixels.
    The measurement is still made on real pixels, but it is a weaker one there.
    """
    height, width = gray.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        return None, None
    surround = gray[max(0, y - outer):y + outer + 1, max(0, x - outer):x + outer + 1]
    centre = gray[max(0, y - inner):y + inner + 1, max(0, x - inner):x + inner + 1]
    return centre, surround


def local_contrast(gray, x: int, y: int, inner: int = SUPPORT_INNER,
                   outer: int = SUPPORT_OUTER) -> float | None:
    """Peak brightness within `inner` px of a point minus the median around it.

    None when the point is off-frame: there is nothing to measure there, and a
    zero would be indistinguishable from a real measurement on flat turf.
    """
    import numpy as np

    centre, surround = _windows(gray, x, y, inner, outer)
    if centre is None:
        return None
    return float(centre.max()) - float(np.median(surround))


def head_support(gray, x: int, y: int) -> tuple[float, float, float, bool] | None:
    """Darkness, dark-fill fraction, background level, and whether the test decides.

    The fill fraction is what separates the head from the shaft: a thin dark line
    crossing the window darkens only a sliver of it, a head fills most of it. It
    puts no ceiling on the dark region, so a larger dark mass passes too. None
    when the point is off-frame and nothing was measured.

    The test is relative, so where the background is itself darker than the margin
    it asks for, no pixel value at all could satisfy it: it then reports the
    background rather than the head, and its failure is not evidence of absence.
    That is the last value, and it is arithmetic, not a judgement about the scene --
    on bright turf with nothing dark in sight the test still decides, and correctly
    finds no head. It is True for every point that passes, since a passing pixel is
    HEAD_DARK_MARGIN below the background and pixel values are not negative.
    """
    import numpy as np

    centre, surround = _windows(gray, x, y, HEAD_INNER, HEAD_OUTER)
    if centre is None:
        return None
    background = float(np.median(surround))
    darkness = background - float(centre.min())
    fill = float((centre < background - HEAD_DARK_MARGIN).mean())
    return darkness, fill, background, background > HEAD_DARK_MARGIN


def is_supported(gray, x: float, y: float, target: str) -> bool:
    """Whether the image measurement backs a claimed position for this target.

    An unmeasurable point is not supported: absent evidence is not evidence.
    """
    ix, iy = int(round(x)), int(round(y))
    if target == "ball":
        contrast = local_contrast(gray, ix, iy)
        return contrast is not None and contrast >= BALL_SUPPORT_MIN
    support = head_support(gray, ix, iy)
    return support is not None and support[1] >= HEAD_FILL_MIN


def support_value(gray, x: float, y: float, target: str) -> float | list | None:
    """The support measurement recorded alongside a point; None if unmeasurable.

    For the head that is [darkness, dark fill fraction, background level]: the
    background is recorded because it is what decides whether the other two mean
    anything at this point.
    """
    ix, iy = int(round(x)), int(round(y))
    if target == "ball":
        contrast = local_contrast(gray, ix, iy)
        return None if contrast is None else round(contrast, 1)
    support = head_support(gray, ix, iy)
    return None if support is None else [round(support[0], 1), round(support[1], 3),
                                         round(support[2], 1)]


def measurement_decides(gray, x: float, y: float, target: str) -> bool:
    """Whether this target's test could have passed at this point at all.

    A test that could not have passed has not refused the point: it had nothing to
    say about it. Only the head test has a known blind spot of this kind, so the
    ball test counts as deciding wherever it can be measured; claiming a blind spot
    it has not been shown to have would be invention of the same sort.
    """
    ix, iy = int(round(x)), int(round(y))
    if target == "ball":
        return local_contrast(gray, ix, iy) is not None
    support = head_support(gray, ix, iy)
    return support is not None and support[3]


def longest_continuous_run(points: list[tuple[int, int, int]]) -> int:
    """Longest run of consecutive frames whose observation stayed physically close."""
    best = run = 0
    previous = None
    for frame, x, y in points:
        if previous and frame == previous[0] + 1 and abs(x - previous[1]) <= CONTINUITY_PX \
                and abs(y - previous[2]) <= CONTINUITY_PX:
            run += 1
        else:
            run = 1
        best = max(best, run)
        previous = (frame, x, y)
    return best


def repeated_coordinates(points: list[tuple[int, int, int]]) -> int:
    """Largest count of one identical (x, y) across frames.

    Recurring identical coordinates are equally consistent with a genuinely
    stationary target and with a model repeating an earlier answer instead of
    looking. This count distinguishes neither; it only marks the frames a
    reviewer should check.
    """
    if not points:
        return 0
    return Counter((x, y) for _, x, y in points).most_common(1)[0][1]


def _target_note(label: str, points: list[tuple[int, int, int]], frames: int,
                 confidences: list[float]) -> dict:
    if not points:
        return {"label": label, "visible": False,
                "note": f"not reported by the vision pass in any of {frames} frames"}

    run = longest_continuous_run(points)
    repeated = repeated_coordinates(points)
    note = (f"observed in {len(points)} of {frames} frames "
            f"(first {points[0][0]}, last {points[-1][0]}); longest continuous run "
            f"{run} frames; peak confidence {max(confidences):.2f} somewhere in that set")
    if repeated > 1:
        note += (f"; one identical coordinate recurs in {repeated} frames. That is equally "
                 f"consistent with a stationary target and with the model repeating an "
                 f"earlier answer, and this count proves neither; the frames need review")
    return {"label": label, "visible": True, "note": note}


def summarize(stats: dict) -> tuple[list[dict], dict]:
    frames = stats["frames"]
    observations = []

    if stats["body_frames"]:
        observations.append({
            "label": "body",
            "visible": True,
            "note": (f"MoveNet keypoints drawn on the vision-supplied golfer crop in "
                     f"{stats['body_frames']} of {frames} frames; keypoints scoring below "
                     f"{body.MIN_SCORE} are not drawn. A low score is not evidence that the "
                     f"joint was occluded, and a high one is not evidence that it was not: "
                     f"the model reports confidence, never occlusion"),
        })
    else:
        observations.append({"label": "body", "visible": False,
                             "note": f"no golfer crop was reported in any of {frames} frames"})

    for target, signature in (
            ("ball", f"a brightness peak at least {BALL_SUPPORT_MIN:.0f} levels above the "
                     f"median of its surroundings, somewhere within {SUPPORT_INNER}px of "
                     f"the marked point. Bare turf does not do that; a white shoe, a "
                     f"specular highlight or a broadcast graphic does, so this measurement "
                     f"is local contrast and not ball recognition"),
            ("clubhead", f"a dark region covering at least {HEAD_FILL_MIN:.0%} of the "
                         f"{2 * HEAD_INNER + 1}px window around the marked point. The thin "
                         f"shaft cannot fill that much and is refused; there is no upper "
                         f"bound, so a larger dark mass such as a trouser leg or a shadow "
                         f"edge passes the same measurement, which is therefore local "
                         f"darkness and not head recognition. Being relative, it decides "
                         f"nothing where the head crosses something as dark as itself: "
                         f"those frames are counted separately below rather than as "
                         f"refusals")):
        row = _target_note(target, stats[target], frames, stats[f"{target}_conf"])
        row["note"] += (
            f". {target.capitalize()} positions come from a second pass on a "
            f"{vision.CROP_SIZE}px crop around each coarse proposal, run on a different "
            f"backend ({vision.REFINE_CLI}/{vision.REFINE_MODEL}) from the coarse pass "
            f"({vision.CLI}/{vision.MODEL}); both are recorded in vision_raw.json. A "
            f"refined point is drawn only where an image measurement backs it: {signature}. "
            f"The measurement only accepts or rejects, it never moves a point, so it cannot "
            f"invent a position, and it is taken in a window around the point rather than at "
            f"the point, so a marker can sit a little off the feature that passed it")
        if stats[f"{target}_unconfirmed"]:
            row["note"] += (
                f". {stats[f'{target}_unconfirmed']} coarse proposal(s) were left unmarked "
                f"on the evidence: the crop pass reported nothing visible, or the point it "
                f"reported did not pass the measurement above")
        if stats[f"{target}_inconclusive"]:
            row["note"] += (
                f". A further {stats[f'{target}_inconclusive']} coarse proposal(s) were "
                f"left unmarked where this measurement could not decide: at the points the "
                f"crop pass returned, nothing in the neighbourhood stood far enough from "
                f"the background for the measurement to pass whatever was there, so it "
                f"cannot tell a {target} that is present from one that is not. Those "
                f"frames are not evidence that the {target} was absent, and they are not "
                f"evidence that it was present either")
        if stats[f"{target}_failed"]:
            row["note"] += (
                f". A further {stats[f'{target}_failed']} coarse proposal(s) were left "
                f"unmarked because the crop pass itself did not run to an answer (backend "
                f"error or timeout, recorded per attempt in vision_raw.json). No image "
                f"measurement was made for those, so nothing here says the image refused "
                f"them")
        observations.append(row)

    # Dimensions travel with every keypoint row so analytics can reject
    # coordinates that are impossible for the frame they claim to come from.
    # Without them the row is emitted with no coordinate space, so analytics
    # withholds the angles rather than trusting unvalidatable pixels.
    width, height = stats.get("image_width"), stats.get("image_height")
    dimensioned = isinstance(width, int) and isinstance(height, int) \
        and width > 0 and height > 0
    for frame, triples in stats["keypoints"]:
        row = {"frame": frame, "keypoints": triples}
        if dimensioned:
            row.update(keypoint_space="native_pixel",
                       image_width=width, image_height=height)
        observations.append(row)

    if stats["errors"]:
        observations.append({
            "label": "vision", "visible": False,
            "note": (f"{stats['errors']} of {frames} frames returned no usable vision result "
                     f"and contributed no observation"),
        })

    metadata = {
        # Not derived from continuity. Consecutive close observations are equally
        # consistent with a false lock onto a fixed background object, so a run
        # length cannot establish that the thing tracked is the target. Nothing in
        # this pipeline verifies target identity, so this stays False on evidence.
        "valid_target": False,
        # No object of known physical size was measured in frame, so pixels cannot
        # be converted to distance.
        "calibration": False,
        # The clip is broadcast slow motion of an undeclared factor, so frame count
        # is not a real-time clock.
        "action_time": False,
        # The interval contains no observed landing.
        "landing": False,
        # No phase detection exists here, so no phase_events are supplied.
    }
    return observations, metadata


def check_prerequisites(model: Path | None = None):
    """Load the body model and check the tools, before the expensive vision pass.

    The vision pass costs a subprocess per frame, so discovering a missing body
    runtime or CLI afterwards wastes all of it. The loaded interpreter is returned
    so the run does not pay for a second load.
    """
    try:
        interpreter = body.load_model(model)
    except ImportError as exc:
        raise SystemExit(
            f"body runtime unavailable ({exc}). Install this project's pinned "
            f"dependencies and run with that interpreter (see README.md)")
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot load {body.model_path(model)}: {exc}")

    for cli in (vision.CLI, vision.REFINE_CLI):
        if shutil.which(cli) is None:
            raise SystemExit(f"vision CLI {cli!r} is not on PATH")

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} is required for exact-rate output but is not on PATH")

    # A minimal ffmpeg build can lack libx264, which would only surface after every
    # frame had been paid for.
    try:
        encoders = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error",
                                   "-encoders"], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        raise SystemExit("ffmpeg did not answer within 60s when listing encoders")
    if "libx264" not in encoders.stdout:
        raise SystemExit("this ffmpeg has no libx264 encoder, which the annotated output "
                         "needs. Install an ffmpeg built with libx264")
    return interpreter


def refine_targets(frames: list, grays: list, results: list, start: int, crop_dir: Path,
                   workers: int, target: str) -> dict:
    """Run the crop second pass for every frame whose coarse pass proposed this target."""
    import cv2 as _cv2

    jobs = []
    for offset, result in enumerate(results):
        if "error" in result or not result[target]:
            continue
        index = start + offset
        x0, y0 = crop_origin(result[target][0], result[target][1])
        crop = build_crop(frames[offset], x0, y0)
        model_input = _cv2.resize(crop, (vision.MODEL_INPUT_SIZE, vision.MODEL_INPUT_SIZE),
                                  interpolation=_cv2.INTER_NEAREST)
        path = crop_dir / f"{target}{index:05d}.png"
        _cv2.imwrite(str(path), model_input)
        jobs.append((index, x0, y0, path, grays[offset]))

    def one(job):
        index, x0, y0, path, gray = job
        record = {"crop_origin": [x0, y0], "crop_image": str(path), "attempts": []}
        # Each attempt is a fresh sample; an unsupported point is retried once
        # rather than corrected, because correcting it would be inventing a place.
        # Each attempt writes its own message file so no attempt can read back the
        # answer another attempt, or an earlier run, left at that path.
        for attempt_number in range(REFINE_ATTEMPTS):
            answer = vision.refine(
                path, crop_dir / f"{target}{index:05d}_{attempt_number}.txt", target)
            attempt = dict(answer)
            if answer.get("point"):
                model_x, model_y, conf = answer["point"]
                native_x, native_y = to_native(model_x, model_y, x0, y0)
                attempt.update(model_point=[model_x, model_y],
                               native_point=[native_x, native_y, conf],
                               image_support=support_value(gray, native_x, native_y, target),
                               decides=measurement_decides(gray, native_x, native_y, target),
                               supported=is_supported(gray, native_x, native_y, target))
            record["attempts"].append(attempt)
            if attempt.get("supported"):
                record["native_point"] = attempt["native_point"]
                record["image_support"] = attempt["image_support"]
                break
        else:
            # Distinguish "the image refused the point" from "the crop pass never
            # produced one to test". Claiming the former for the latter would
            # report a backend outage as image evidence.
            measured = [attempt for attempt in record["attempts"] if "decides" in attempt]
            if all("error" in attempt for attempt in record["attempts"]):
                record["failed"] = (
                    f"the {target} crop pass did not run to an answer in "
                    f"{REFINE_ATTEMPTS} attempts, so no image measurement was made: "
                    f"{record['attempts'][-1]['error']}")
            elif measured and not any(attempt["decides"] for attempt in measured):
                # The measurement could not have passed at these points whatever was
                # there, so it did not refuse them: it had nothing to say. This
                # recovers no position and marks nothing; it only stops a blind spot
                # being reported as evidence of absence.
                record["inconclusive"] = (
                    f"no {target} point could be decided: at every point the crop pass "
                    f"returned, nothing in the neighbourhood was far enough from the "
                    f"background for this measurement to pass, so it cannot tell a "
                    f"{target} that is present from one that is not")
            else:
                record["rejected"] = (
                    f"no refined {target} point was backed by the image "
                    f"in {REFINE_ATTEMPTS} attempts")
        return index, record

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, jobs))


def run(video: Path, out_dir: Path, start: int, count: int, workers: int,
        model: Path | None = None) -> dict:
    interpreter = check_prerequisites(model)
    rate = source_frame_rate(video)
    frames = read_interval(video, start, count)
    if not frames:
        raise SystemExit(f"no frames decoded at {start}; nothing to annotate")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_dir = out_dir / "frames"
        frame_dir.mkdir(exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"cannot create output directories under {out_dir}: {exc}")

    paths = []
    for offset, frame in enumerate(frames):
        path = frame_dir / f"f{start + offset:05d}.png"
        write_image(path, frame)
        paths.append(path)

    height, width = frames[0].shape[:2]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda p: vision.read_frame(p, width, height), paths))

    # Second pass: re-read the ball on a small crop around each coarse proposal.
    # The coarse point is preserved either way; it is a proposal, not a result.
    crop_dir = out_dir / "crops"
    crop_dir.mkdir(exist_ok=True)
    # One grayscale conversion per frame, shared by both target passes.
    grays = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    refinements = {target: refine_targets(frames, grays, results, start, crop_dir,
                                          workers, target)
                   for target in ("ball", "clubhead")}

    (out_dir / "vision_raw.json").write_text(json.dumps({
        "backends": vision.backends(),
        "crop_policy": (f"{vision.CROP_SIZE}x{vision.CROP_SIZE} centred on the coarse "
                        f"ball point, black-padded outside the frame, "
                        f"{vision.CROP_UPSCALE}x nearest-neighbour upscale"),
        "frames": {str(start + i): {
            "coarse": r,
            "ball_refinement": refinements["ball"].get(start + i),
            "clubhead_refinement": refinements["clubhead"].get(start + i),
        } for i, r in enumerate(results)},
    }, indent=2, default=list) + "\n")

    annotated_dir = out_dir / "annotated_frames"
    annotated_dir.mkdir(exist_ok=True)

    stats = {"frames": len(frames), "body_frames": 0, "errors": 0, "keypoints": [],
             "image_width": width, "image_height": height,
             "ball_unconfirmed": 0, "clubhead_unconfirmed": 0,
             "ball_failed": 0, "clubhead_failed": 0,
             "ball_inconclusive": 0, "clubhead_inconclusive": 0,
             "ball": [], "ball_conf": [], "clubhead": [], "clubhead_conf": [],
             "rate": rate, "annotated_dir": annotated_dir}
    for offset, (frame, result) in enumerate(zip(frames, results)):
        index = start + offset
        canvas = frame.copy()
        if "error" in result:
            stats["errors"] += 1
        else:
            if result["golfer"]:
                points = body.keypoints(interpreter, frame, result["golfer"])
                if points and body.draw(canvas, points):
                    stats["body_frames"] += 1
                    # Native image pixels, as [y, x, confidence] per the contract.
                    stats["keypoints"].append(
                        (index, [[float(y), float(x), float(score)]
                                 for x, y, score in points]))
            for target, colour in (("ball", BALL_COLOR), ("clubhead", CLUBHEAD_COLOR)):
                refinement = refinements[target].get(index) or {}
                if refinement.get("native_point"):
                    native_x, native_y, conf = refinement["native_point"]
                    x, y = int(round(native_x)), int(round(native_y))
                    stats[target].append((index, x, y))
                    stats[f"{target}_conf"].append(conf)
                    cv2.circle(canvas, (x, y), 16, colour, 2)
                    cv2.putText(canvas, f"{target} {conf:.2f}", (x - 30, y - 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
                elif result[target]:
                    # The coarse pass proposed this target and the crop pass
                    # did not back it. A proposal alone does not earn a marker.
                    # Which reason applies is recorded separately: a crop pass
                    # that never ran to an answer tested nothing.
                    key = next((name for name in ("failed", "inconclusive")
                                if refinement.get(name)), "unconfirmed")
                    stats[f"{target}_{key}"] += 1
        cv2.putText(canvas, f"frame {index}", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        write_image(annotated_dir / f"a{index:05d}.png", canvas)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate a golf video interval.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--body-model", type=Path, default=None,
                        help="local MoveNet TFLite weights; overrides "
                             "FAIRWAY_BODY_MODEL. Nothing is downloaded.")
    args = parser.parse_args()

    if args.start_frame < 0:
        parser.error("--start-frame cannot be negative")
    if args.frames < 1:
        parser.error("--frames must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if not args.video.is_file():
        parser.error(f"{args.video} is not a file")
    if args.out_dir.exists() and not args.out_dir.is_dir():
        parser.error(f"--out-dir {args.out_dir} exists and is not a directory")

    stats = run(args.video, args.out_dir, args.start_frame, args.frames, args.workers,
                args.body_model)
    observations, metadata = summarize(stats)
    # Written before the encode: the inference is already paid for, so a muxing
    # failure must not take the report and the raw records down with it.
    (args.out_dir / "analytics.txt").write_text(build_report(observations, metadata) + "\n")
    print(f"wrote {args.out_dir}/analytics.txt\nwrote {args.out_dir}/vision_raw.json")
    encode(stats["annotated_dir"], args.out_dir / "annotated.mp4", stats["rate"],
           args.start_frame, stats["frames"])
    print(f"wrote {args.out_dir}/annotated.mp4")


if __name__ == "__main__":
    main()
