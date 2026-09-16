"""Out-of-process worker for the canonical pipeline: decode, identify, pose, render.

Run as ``<interpreter> -m ghostcaddie.video.canonical_worker`` with a JSON spec on
stdin; one JSON document comes back on stdout. This is the only canonical module
that imports cv2, numpy or a model runtime, which is what keeps the orchestrator
(and therefore the whole output contract) testable on a bare interpreter.

The golfer-identity problem is the reason this file exists in this shape. MoveNet
is a SINGLE-pose model: run on a whole wide frame it returns one skeleton with no
guarantee it belongs to the player. The audit's objection is specific -- on a wide
shot with a spectator, a caddie or a camera operator in frame, the returned
skeleton may be the wrong person -- so the gate answers that specific question,
locally and offline:

  1. propose: whole-frame MoveNet on every sampled frame. A candidate exists only
     if a full-torso pose is found on most of them; the median box is the subject.
  2. challenge: blank the subject out of a few probe frames and run MoveNet again.
     A second person that survives on most probes, away from the subject, means
     the frame holds more than one body and the proposal is not self-evidently
     the player.
  3. tie-break, only if challenged: compare camera-compensated motion density
     inside each body. A golfer swings; a bystander stands. Without a decisive
     margin the answer is AMBIGUOUS.
  4. measure: re-run MoveNet on the padded subject crop, so the final pose is not
     the 192-pixel whole-frame downscale.

An operator ROI skips steps 1-3. It is user-supplied input, recorded as such in
provenance, and is never described as automatic perception.

This is a verification gate built on a single-pose model, not a person detector.
Its limits are stated on every record it produces rather than papered over.
"""
from __future__ import annotations

import json
import math
import os
import sys

# Keypoint order of MoveNet SinglePose Lightning v3.
KP_NAMES = ["nose", "eye_l", "eye_r", "ear_l", "ear_r", "sh_l", "sh_r", "el_l",
            "el_r", "wr_l", "wr_r", "hip_l", "hip_r", "kn_l", "kn_r", "ank_l", "ank_r"]

#: Rendered skeleton: torso and limbs only. Face keypoints are omitted because
#: they are the least stable and add clutter without adding information.
SKELETON_EDGES = [("sh_l", "sh_r"), ("sh_l", "el_l"), ("el_l", "wr_l"),
                  ("sh_r", "el_r"), ("el_r", "wr_r"), ("sh_l", "hip_l"),
                  ("sh_r", "hip_r"), ("hip_l", "hip_r"), ("hip_l", "kn_l"),
                  ("kn_l", "ank_l"), ("hip_r", "kn_r"), ("kn_r", "ank_r")]

TORSO = ("sh_l", "sh_r", "hip_l", "hip_r")

#: Only these are drawn. Face keypoints are the least stable of the seventeen and
#: add a cluster of dots that carries no information about a swing.
RENDERED_KEYPOINTS = frozenset(n for edge in SKELETON_EDGES for n in edge)

SCORE_THRESHOLD = 0.30          # MoveNet keypoint confidence for "visible"
MIN_VISIBLE_KEYPOINTS = 8       # of 17, before a frame counts as a usable pose
#: A pose is the subject only if it fills a real share of the moving region that
#: defined the subject, and is centred inside it. Measured against the SUBJECT
#: box, not the padded crop, so it does not move when the padding changes.
MIN_POSE_SUBJECT_AREA_FRACTION = 0.25
ANALYSIS_WIDTH = 320            # motion/jitter analysis resolution
MOTION_DIFF_THRESHOLD = 18      # 8-bit intensity delta that counts as motion
JITTER_FRACTION_OF_WIDTH = 0.03  # camera shift above this = camera-dominated
ROI_PAD = 0.25                  # subject box -> crop, padded by this on each side

MIN_PRIMARY_COVERAGE = 0.60     # full-torso proposals needed before a subject exists
PROBE_FRAMES = 5                # evenly spaced frames used to challenge the proposal
SECOND_PERSON_MIN_HITS = 3      # of PROBE_FRAMES, before a second body is believed
SECOND_PERSON_MAX_IOU = 0.20    # above this it is the subject bleeding through
SECOND_PERSON_AGREEMENT = 0.30  # probe hits must agree with their own median box
MOTION_DENSITY_RATIO = 3.0      # subject must out-move a rival by this to win

SUPPORTED_COVERAGE = 0.80
PARTIAL_COVERAGE = 0.40

IDENTITY_LIMITATION = (
    "Golfer identity comes from a single-pose model plus a second-person "
    "challenge, not from a person detector. MoveNet proposes one body per frame; "
    "the proposal is accepted as the golfer only when re-running the model with "
    "that body blanked out finds no other consistent person. Bodies that overlap "
    "the subject, and people too small or too occluded for MoveNet to resolve at "
    "all, cannot be challenged this way and would not be detected.")

OPERATOR_ROI_LIMITATION = (
    "The region of interest was supplied by the operator. It is user input, not "
    "automatic perception, and any pose found inside it is attributed to the "
    "golfer on the operator's authority.")

POSE_LIMITATION = (
    "MoveNet SinglePose Lightning returns exactly one skeleton per crop. Poses "
    "are 2D image-space only: no 3D, no swing phase, no club relationship and no "
    "kinematics are inferred.")


# --------------------------------------------------------------------------

def _versions():
    import cv2, numpy as np
    v = {"cv2": cv2.__version__, "numpy": np.__version__}
    try:
        import ai_edge_litert
        v["ai_edge_litert"] = getattr(ai_edge_litert, "__version__", "unknown")
    except Exception as exc:
        v["ai_edge_litert"] = f"unavailable: {type(exc).__name__}"
    return v


def sample_frames(video, native_fps, sample_fps, start_s, end_s, max_frames):
    """Decode a bounded, evenly spaced set of frames, keeping native indices."""
    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open the source: {video}")
    try:
        step = max(1, int(round(float(native_fps) / float(sample_fps))))
        first = int(math.floor(start_s * native_fps))
        last = int(math.ceil(end_s * native_fps))
        out, idx = [], 0
        while len(out) < max_frames:
            ok = cap.grab()
            if not ok:
                break
            if idx >= first and idx <= last and (idx - first) % step == 0:
                ok, img = cap.retrieve()
                if not ok or img is None:
                    break
                out.append((idx, img))
            idx += 1
            if idx > last:
                break
        return out, step
    finally:
        cap.release()


def camera_shifts(frames, native_width):
    """Per-pair global camera translation in NATIVE pixels, via phase correlation."""
    import cv2, numpy as np
    shifts = []
    prev = None
    scale = ANALYSIS_WIDTH / float(native_width)
    for _sf, img in frames:
        small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None,
                           fx=scale, fy=scale)
        cur = small.astype(np.float32)
        if prev is not None:
            (dx, dy), _resp = cv2.phaseCorrelate(prev, cur)
            shifts.append((dx / scale, dy / scale))
        prev = cur
    return shifts


def motion_accumulator(frames, shifts, native_width):
    """Camera-compensated inter-frame motion energy, and its scale factor.

    Global camera translation is removed first, so a pan does not read as if the
    whole scene were moving. What survives is motion relative to the scene.
    """
    import cv2, numpy as np
    scale = ANALYSIS_WIDTH / float(native_width)
    grays = [cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale)
             for _sf, img in frames]
    if len(grays) < 2:
        return None, scale
    accum = np.zeros(grays[0].shape, np.float32)
    for i in range(len(grays) - 1):
        dx, dy = shifts[i]
        m = np.float32([[1, 0, dx * scale], [0, 1, dy * scale]])
        warped = cv2.warpAffine(grays[i], m, (grays[i].shape[1], grays[i].shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        accum += (cv2.absdiff(warped, grays[i + 1]) > MOTION_DIFF_THRESHOLD).astype(np.float32)
    return accum, scale


def motion_density(accum, scale, box_xywh):
    """Mean motion energy per pixel inside a native-coordinate box."""
    if accum is None:
        return 0.0
    x, y, w, h = box_xywh
    x0, y0 = int(x * scale), int(y * scale)
    x1, y1 = int((x + w) * scale), int((y + h) * scale)
    patch = accum[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
    return float(patch.mean()) if patch.size else 0.0


def pose_box(points):
    """Bounding box of a confident full-torso pose, or None. [x0,y0,x1,y1]."""
    vis = [p for p in points if p["visible"]]
    if len(vis) < MIN_VISIBLE_KEYPOINTS:
        return None
    have = {p["name"] for p in vis}
    if any(t not in have for t in TORSO):
        return None
    xs = [p["x"] for p in vis]
    ys = [p["y"] for p in vis]
    return [min(xs), min(ys), max(xs), max(ys)]


def iou(a, b):
    """Intersection over union of two [x0,y0,x1,y1] boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def to_xywh(box):
    return [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])]


def challenge_subject(net, frames, subject_box, frame_wh):
    """Look for a SECOND person once the subject is blanked out of the frame.

    Returns (second_box or None, hits, probes). A hallucinated pose on erased
    pixels is the expected false positive here, so a second body is believed only
    when it survives most probes, sits away from the subject, and the surviving
    detections agree with each other.
    """
    import numpy as np
    w, h = frame_wh
    if len(frames) == 0:
        return None, 0, 0
    idx = sorted({int(round(i)) for i in
                  np.linspace(0, len(frames) - 1, min(PROBE_FRAMES, len(frames)))})
    x0, y0, x1, y1 = [int(v) for v in subject_box]
    pad = int(0.10 * max(x1 - x0, y1 - y0))
    hits = []
    for i in idx:
        img = frames[i][1].copy()
        ys, ye = max(0, y0 - pad), min(h, y1 + pad)
        xs, xe = max(0, x0 - pad), min(w, x1 + pad)
        img[ys:ye, xs:xe] = int(img.mean())
        box = pose_box(net.infer(img, (0, 0))[0])
        if box is not None and iou(box, subject_box) <= SECOND_PERSON_MAX_IOU:
            hits.append(box)
    if len(hits) < SECOND_PERSON_MIN_HITS:
        return None, len(hits), len(idx)
    median = list(np.median(np.array(hits), axis=0))
    agreeing = [b for b in hits if iou(b, median) >= SECOND_PERSON_AGREEMENT]
    if len(agreeing) < SECOND_PERSON_MIN_HITS:
        return None, len(agreeing), len(idx)
    return median, len(agreeing), len(idx)


def pad_to_crop(bbox, frame_w, frame_h, pad=ROI_PAD):
    """Pad a subject box into the crop fed to the pose model.

    The subject box comes from inter-frame motion, so it covers what MOVED: a
    swing shows arms, torso and club but can clip the feet and the follow-through.
    Padding recovers those without inflating the box so far that a background
    person would pass the subject-area check.
    """
    x, y, w, h = bbox
    dx, dy = w * pad, h * pad
    x0 = int(max(0, math.floor(x - dx)))
    y0 = int(max(0, math.floor(y - dy)))
    x1 = int(min(frame_w, math.ceil(x + w + dx)))
    y1 = int(min(frame_h, math.ceil(y + h + dy)))
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def establish_identity(frames, spec, net):
    """Decide whether a pose may be attributed to the golfer, and say why if not."""
    import numpy as np
    _sf, first = frames[0]
    h, w = first.shape[:2]
    shifts = camera_shifts(frames, w)
    mags = sorted(math.hypot(dx, dy) for dx, dy in shifts)
    jitter = (mags[len(mags) // 2] if mags else 0.0)

    base = {"jitter_px_median": round(jitter, 2),
            "jitter_px_max": round(max(mags), 2) if mags else 0.0,
            "jitter_fraction_of_width": round(jitter / w, 4) if w else 0.0,
            "camera_jitter_high": bool(jitter > JITTER_FRACTION_OF_WIDTH * w),
            "score_threshold": SCORE_THRESHOLD,
            "subject_box_xywh": None, "roi_xywh": None,
            "second_person": None}

    def stop(state, *why):
        return {**base, "layer_state": state, "identity_method": "none",
                "method": "none", "limitations": list(why) + [IDENTITY_LIMITATION]}

    if spec.get("roi"):
        x, y, rw, rh = [int(v) for v in spec["roi"]]
        if rw <= 0 or rh <= 0 or x < 0 or y < 0 or x + rw > w or y + rh > h:
            return {**base, "layer_state": "failed", "identity_method": "operator_roi",
                    "method": "none",
                    "limitations": [f"operator ROI {list(spec['roi'])} does not fit "
                                    f"inside the {w}x{h} frame"]}
        return {**base, "identity_method": "operator_roi",
                "subject_box_xywh": [x, y, rw, rh],
                "roi_xywh": pad_to_crop([x, y, rw, rh], w, h, pad=0.10),
                "limitations": [OPERATOR_ROI_LIMITATION, POSE_LIMITATION]}

    if net is None:
        return stop("unavailable", "the body model is not available, so no person "
                                   "can be proposed or challenged")

    # 1. propose
    boxes = [b for b in (pose_box(net.infer(img, (0, 0))[0]) for _sf, img in frames)
             if b is not None]
    coverage = len(boxes) / float(len(frames))
    base["primary_proposal_coverage"] = round(coverage, 4)
    if coverage < MIN_PRIMARY_COVERAGE:
        return stop("unavailable" if not boxes else "ambiguous",
                    f"a full-torso pose was proposed on only {len(boxes)} of "
                    f"{len(frames)} sampled frames ({coverage:.0%}); there is no "
                    f"body stable enough to attribute to the golfer. Supply --roi "
                    f"to analyse this clip.")
    subject = list(np.median(np.array(boxes), axis=0))
    base["subject_box_xywh"] = to_xywh(subject)

    # 2. challenge
    second, hits, probes = challenge_subject(net, frames, subject, (w, h))
    base["second_person"] = {"found": second is not None,
                             "box_xywh": to_xywh(second) if second else None,
                             "probe_hits": hits, "probes": probes}
    limitations = [IDENTITY_LIMITATION, POSE_LIMITATION]
    if second is None:
        method = "movenet_proposal_unchallenged"
    else:
        # 3. tie-break on motion: a golfer swings, a bystander stands
        accum, scale = motion_accumulator(frames, shifts, w)
        d_subject = motion_density(accum, scale, to_xywh(subject))
        d_second = motion_density(accum, scale, to_xywh(second))
        base["second_person"].update(
            {"subject_motion_density": round(d_subject, 4),
             "second_motion_density": round(d_second, 4)})
        if base["camera_jitter_high"]:
            return stop("ambiguous",
                        f"a second person was found on {hits}/{probes} probe frames, "
                        f"and the camera moves a median of {jitter:.1f} px between "
                        f"sampled frames ({jitter / w:.1%} of frame width), so subject "
                        f"motion cannot be told from camera motion. There is no safe "
                        f"way to say which body is the golfer. Supply --roi to "
                        f"analyse this clip.")
        if d_second <= 0 or d_subject / d_second < MOTION_DENSITY_RATIO:
            return stop("ambiguous",
                        f"a second person was found on {hits}/{probes} probe frames "
                        f"and does not move decisively less than the proposed subject "
                        f"(motion density {d_subject:.2f} vs {d_second:.2f}; a factor "
                        f"of {MOTION_DENSITY_RATIO:g} is required). Which body is the "
                        f"golfer cannot be established. Supply --roi to analyse this "
                        f"clip.")
        method = "movenet_proposal_motion_disambiguated"
        limitations.append(
            f"a second person is present; the golfer was chosen because that body "
            f"moves {d_subject / d_second:.1f}x more than the other. That is a motion "
            f"argument, not an identification of the player.")

    if base["camera_jitter_high"]:
        limitations.append(
            f"the camera moves a median of {jitter:.1f} px between sampled frames "
            f"({jitter / w:.1%} of frame width); keypoints inherit that movement and "
            f"frame-to-frame changes in the 2D metrics are not purely the golfer's.")
    return {**base, "identity_method": method,
            "roi_xywh": pad_to_crop(base["subject_box_xywh"], w, h),
            "limitations": limitations}


# --------------------------------------------------------------------------

class MoveNet:
    """MoveNet SinglePose Lightning over an explicit crop. No pickle, no torch."""

    def __init__(self, model_path):
        from ai_edge_litert.interpreter import Interpreter
        self.itp = Interpreter(model_path=model_path, num_threads=2)
        self.itp.allocate_tensors()
        self.inp = self.itp.get_input_details()[0]
        self.out = self.itp.get_output_details()[0]
        _, self.ih, self.iw, _ = self.inp["shape"]

    def infer(self, crop, roi_xy):
        """Keypoints in NATIVE frame coordinates, with the letterbox recorded."""
        import cv2, numpy as np
        h, w = crop.shape[:2]
        scale = min(self.iw / w, self.ih / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        canvas = np.zeros((self.ih, self.iw, 3), np.uint8)
        ox, oy = (self.iw - nw) // 2, (self.ih - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(crop, (nw, nh))
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        # MoveNet float input takes raw [0,255] float32, NOT [0,1]: dividing by
        # 255 silently collapses every keypoint score to ~0.03.
        tensor = (rgb[None, ...] if self.inp["dtype"] == np.uint8
                  else rgb.astype(np.float32)[None, ...])
        self.itp.set_tensor(self.inp["index"], tensor)
        self.itp.invoke()
        kp = self.itp.get_tensor(self.out["index"])[0][0]     # (17,3) y,x,score
        rx, ry = roi_xy
        pts = []
        for i, name in enumerate(KP_NAMES):
            y, x, s = float(kp[i][0]), float(kp[i][1]), float(kp[i][2])
            cx = (x * self.iw - ox) / scale + rx
            cy = (y * self.ih - oy) / scale + ry
            pts.append({"name": name, "x": round(cx, 2), "y": round(cy, 2),
                        "score": round(s, 4), "visible": bool(s >= SCORE_THRESHOLD)})
        return pts, {"letterbox_scale": round(scale, 6), "offset_xy": [int(ox), int(oy)],
                     "model_input_wh": [int(self.iw), int(self.ih)],
                     "roi_origin_xy": [int(rx), int(ry)]}


def attribute(points, subject_box):
    """Is this pose confidently the identified subject? Reason if not.

    Three independent ways to fail, all stated: too few confident keypoints, an
    incomplete torso, or a pose that is not plausibly the body that defined the
    subject box (too small, or centred outside it).
    """
    vis = [p for p in points if p["visible"]]
    if len(vis) < MIN_VISIBLE_KEYPOINTS:
        return False, f"only {len(vis)} of 17 keypoints above {SCORE_THRESHOLD}"
    have = {p["name"] for p in vis}
    missing = [n for n in TORSO if n not in have]
    if missing:
        return False, f"incomplete torso: {'+'.join(missing)} below confidence"
    xs = [p["x"] for p in vis]
    ys = [p["y"] for p in vis]
    area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    sx, sy, sw, sh = subject_box
    subject_area = float(sw * sh) or 1.0
    if area / subject_area < MIN_POSE_SUBJECT_AREA_FRACTION:
        return False, (f"pose covers {area / subject_area:.1%} of the identified "
                       f"subject region; too small to be that subject")
    cx, cy = (max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0
    if not (sx <= cx <= sx + sw and sy <= cy <= sy + sh):
        return False, ("pose centre falls outside the identified subject region; "
                       "this is a different body")
    return True, ""


# --------------------------------------------------------------------------

#: Annotated output is downscaled to this width. A 4K annotated copy is a huge
#: file whose overlay text renders at a few pixels tall; the analysis itself is
#: unaffected, because pose runs on the native crop before this.
RENDER_MAX_WIDTH = 1280
STATUS_BAR_H = 30
CALM = (235, 235, 235)
ACCENT = (120, 220, 120)
MUTED = (155, 155, 155)


def draw(img, points, attributed, body_state, scale=1.0):
    """Overlay onto a COPY. Only supported information; never a stale marker.

    ``points`` are in native coordinates; ``scale`` maps them onto ``img``, which
    may have been downscaled for output. Nothing is drawn for a frame that was
    not attributed to the golfer, and nothing is ever carried over from a
    previous frame: an absent layer leaves the picture alone.
    """
    import cv2
    out = img.copy()
    h, w = out.shape[:2]
    if attributed and points:
        by = {p["name"]: p for p in points if p["visible"]}
        thick = max(1, round(w / 640))
        for a, b in SKELETON_EDGES:
            if a in by and b in by:
                cv2.line(out, (int(by[a]["x"] * scale), int(by[a]["y"] * scale)),
                         (int(by[b]["x"] * scale), int(by[b]["y"] * scale)),
                         ACCENT, thick, cv2.LINE_AA)
        for name, p in by.items():
            if name not in RENDERED_KEYPOINTS:
                continue
            cv2.circle(out, (int(p["x"] * scale), int(p["y"] * scale)),
                       thick + 1, CALM, -1, cv2.LINE_AA)
    bar = out[0:STATUS_BAR_H, 0:w].copy()
    cv2.rectangle(out, (0, 0), (w, STATUS_BAR_H), (20, 20, 20), -1)
    cv2.addWeighted(bar, 0.25, out[0:STATUS_BAR_H, 0:w], 0.75, 0,
                    out[0:STATUS_BAR_H, 0:w])
    label = f"BODY {body_state.upper()}"
    if not attributed:
        label += " - no confident golfer pose"
    cv2.putText(out, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CALM, 1, cv2.LINE_AA)
    right = "BALL UNAVAILABLE   CLUBHEAD UNAVAILABLE"
    (tw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(out, right, (max(10, w - tw - 10), 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, MUTED, 1, cv2.LINE_AA)
    return out


def write_video(path, images, fps):
    """Encode the annotated frames. Returns (written, reason) -- never a lie."""
    import cv2
    if not images:
        return False, "no frames were rendered, so no annotated video was produced"
    h, w = images[0].shape[:2]
    for fourcc in ("avc1", "mp4v"):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), float(fps), (w, h))
        if vw.isOpened():
            for im in images:
                vw.write(im)
            vw.release()
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                return True, f"encoded {len(images)} frames at {fps:g} fps ({fourcc})"
        else:
            vw.release()
    if os.path.exists(path):
        os.remove(path)
    return False, ("OpenCV could not open a video writer for this output path with "
                   "any available codec; no annotated video was produced")


def run(spec):
    import cv2
    frames, step = sample_frames(spec["video"], spec["native_fps"],
                                 spec["sample_fps"], spec["start_seconds"],
                                 spec["end_seconds"], spec["max_frames"])
    if not frames:
        raise RuntimeError("no frames were decoded from the requested interval")
    fh, fw = frames[0][1].shape[:2]

    # The model is loaded BEFORE identity, because identity is established with
    # it: the gate proposes a body with MoveNet and then challenges that proposal.
    net, model_error = None, ""
    if spec.get("model_found"):
        try:
            net = MoveNet(spec["model_path"])
        except Exception as exc:
            model_error = f"{type(exc).__name__}: {exc}"
    ident = establish_identity(frames, spec, net)

    pose_frames, attributed_frames, per_frame = [], [], []
    if net is not None and ident.get("roi_xywh") and "layer_state" not in ident:
        # 4. measure: re-run on the padded subject crop, not the whole-frame
        # downscale, so the pose that reaches the metrics is the best one available
        x, y, rw, rh = ident["roi_xywh"]
        for sf, img in frames:
            pts, transform = net.infer(img[y:y + rh, x:x + rw], (x, y))
            ok, why = attribute(pts, ident["subject_box_xywh"])
            rec = {"source_frame": int(sf), "source_sha256": spec["source_sha256"],
                   "native_wh": [fw, fh], "roi_xywh": ident["roi_xywh"],
                   "subject_box_xywh": ident["subject_box_xywh"],
                   "transform": transform, "keypoints": pts,
                   "visible_keypoint_count": sum(1 for p in pts if p["visible"]),
                   "attributed_to_golfer": ok, "rejection_reason": why}
            pose_frames.append((sf, pts, ok))
            per_frame.append(rec)
            if ok:
                attributed_frames.append(rec)

    if "layer_state" in ident:
        state = ident["layer_state"]
    elif model_error:
        state = "failed"
    elif not spec.get("model_found"):
        state = "unavailable"
    else:
        cov = len(attributed_frames) / float(len(frames))
        state = ("supported" if cov >= SUPPORTED_COVERAGE else
                 "partial" if cov >= PARTIAL_COVERAGE else "ambiguous")
        if state == "ambiguous":
            ident["limitations"] = list(ident.get("limitations", [])) + [
                f"a confident full-torso pose was attributed on only "
                f"{len(attributed_frames)} of {len(frames)} sampled frames "
                f"({cov:.0%}); that is too little to treat the pose as the golfer"]
        elif state == "partial":
            ident["limitations"] = list(ident.get("limitations", [])) + [
                f"pose was attributed on {len(attributed_frames)} of {len(frames)} "
                f"sampled frames ({cov:.0%}); metrics use only those frames"]
    ident["layer_state"] = state
    ident["method"] = ("movenet_singlepose_lightning_v3_on_identified_subject"
                       if attributed_frames else "none")
    if model_error:
        ident.setdefault("limitations", []).append(
            f"the MoveNet runtime could not be initialised: {model_error}")
    elif not spec.get("model_found"):
        ident.setdefault("limitations", []).append(
            "the MoveNet model file was not found; body analysis needs it. "
            "Run `python -m ghostcaddie models` to see where it is looked for.")

    render = {"written": False, "reason": "rendering was not requested"}
    if spec.get("render"):
        show = state not in ("ambiguous", "unavailable", "failed")
        by_frame = {sf: (pts, ok) for sf, pts, ok in pose_frames}
        rs = min(1.0, RENDER_MAX_WIDTH / float(fw))
        images = []
        for sf, img in frames:
            pts, ok = by_frame.get(sf, (None, False))
            out_img = (cv2.resize(img, (int(fw * rs), int(fh * rs)),
                                  interpolation=cv2.INTER_AREA) if rs < 1.0 else img)
            images.append(draw(out_img, pts, bool(show and ok), state, scale=rs))
        ok, reason = write_video(spec["annotated_path"], images, spec["sample_fps"])
        render = {"written": ok, "reason": reason, "overlays_drawn": bool(show),
                  "frames": len(images), "output_wh": [images[0].shape[1],
                                                       images[0].shape[0]] if images else None}

    debug_dir = spec.get("debug_dir")
    if debug_dir and ident.get("roi_xywh"):
        try:
            dbg = cv2.resize(frames[0][1], None, fx=0.5, fy=0.5) if fw > 1600 else frames[0][1].copy()
            ds = 0.5 if fw > 1600 else 1.0
            x, y, rw, rh = [int(v * ds) for v in ident["roi_xywh"]]
            cv2.rectangle(dbg, (x, y), (x + rw, y + rh), (0, 200, 255), 2)
            if ident.get("subject_box_xywh"):
                bx, by_, bw, bh = [int(v * ds) for v in ident["subject_box_xywh"]]
                cv2.rectangle(dbg, (bx, by_), (bx + bw, by_ + bh), (120, 255, 120), 2)
            sec = (ident.get("second_person") or {}).get("box_xywh")
            if sec:
                bx, by_, bw, bh = [int(v * ds) for v in sec]
                cv2.rectangle(dbg, (bx, by_), (bx + bw, by_ + bh), (0, 80, 255), 2)
            cv2.imwrite(os.path.join(debug_dir, "identity_roi.png"), dbg)
        except Exception:
            pass    # debug art is never allowed to fail a run

    return {
        "sampled_frame_count": len(frames),
        "frame_range": [int(frames[0][0]), int(frames[-1][0])],
        "decimation_step": step,
        "identity": ident,
        "attributed_frames": attributed_frames,
        "render": render,
        "diagnostics": {
            "identity": ident,
            "per_frame": per_frame,
            "render": render,
            "sampling": {"native_fps": spec["native_fps"],
                         "sample_fps": spec["sample_fps"],
                         "decimation_step": step,
                         "frames": [int(sf) for sf, _ in frames]},
            "thresholds": {
                "score_threshold": SCORE_THRESHOLD,
                "min_visible_keypoints": MIN_VISIBLE_KEYPOINTS,
                "min_pose_subject_area_fraction": MIN_POSE_SUBJECT_AREA_FRACTION,
                "roi_pad": ROI_PAD,
                "min_primary_coverage": MIN_PRIMARY_COVERAGE,
                "probe_frames": PROBE_FRAMES,
                "second_person_min_hits": SECOND_PERSON_MIN_HITS,
                "motion_density_ratio": MOTION_DENSITY_RATIO,
                "jitter_fraction_of_width": JITTER_FRACTION_OF_WIDTH,
                "supported_coverage": SUPPORTED_COVERAGE,
                "partial_coverage": PARTIAL_COVERAGE},
        },
        "environment": {"python": sys.version.split()[0], "versions": _versions()},
    }


def main():
    spec = json.loads(sys.stdin.read())
    json.dump(run(spec), sys.stdout)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
