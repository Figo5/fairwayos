"""Model adapters that PROBE real capability at runtime.

Nothing here is a hardcoded "model_present": every adapter answers by actually
looking — importing its runtime, checking its model file, hashing it. An adapter
that cannot run says why, naming the missing dependency or path, and refuses to
produce output rather than inventing one.

Body route note: the Ultralytics path is a pickle surface
(ultralytics/utils/patches.py injects weights_only=False) and stays unsafe. It is
NOT the body route any more. Body now runs on MoveNet SinglePose Lightning as a
TFLite flatbuffer through ai-edge-litert: no pickle, no torch, no checkpoint
conversion, so the old blocker does not apply to it.
"""
from __future__ import annotations

import hashlib
import importlib
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence


class AdapterUnavailable(RuntimeError):
    """Raised when an adapter is asked to run without real capability."""


@dataclass(frozen=True)
class Capability:
    target: str
    available: bool          # can actually run right now
    safe: bool               # runtime has no unsafe deserialisation surface
    reason: str
    runtime: str = ""
    runtime_present: bool = False
    model_present: bool = False
    model_path: str = ""
    model_sha256: str = ""
    demonstrated: str = ""   # what has actually been shown to work, if anything


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _importable(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


class _Adapter:
    target = "unset"
    runtime_module = ""

    def __init__(self, model_path: Optional[str] = None,
                 _force_runtime_missing: bool = False):
        self.model_path = model_path or self.default_model
        self._force_runtime_missing = _force_runtime_missing

    def _probe(self, safe: bool, safe_reason: str, demonstrated: str = "") -> Capability:
        runtime_ok = (not self._force_runtime_missing
                      and _importable(self.runtime_module))
        model_ok = bool(self.model_path) and os.path.isfile(self.model_path)
        sha = _sha256(self.model_path) if model_ok else ""
        if not model_ok:
            reason = f"model file not found: {self.model_path}"
        elif not runtime_ok:
            reason = (f"runtime module {self.runtime_module!r} is not importable "
                      f"in this interpreter; install it in an isolated env")
        else:
            reason = safe_reason
        return Capability(target=self.target, available=(runtime_ok and model_ok and safe),
                          safe=safe, reason=reason, runtime=self.runtime_module,
                          runtime_present=runtime_ok, model_present=model_ok,
                          model_path=self.model_path or "", model_sha256=sha,
                          demonstrated=demonstrated)

    def capability(self) -> Capability:
        raise NotImplementedError

    def run_frames(self, frames: Sequence, source_sha256: Optional[str],
                   seed: Optional[dict] = None) -> List[dict]:
        cap = self.capability()
        if not source_sha256:
            raise ValueError("source_sha256 is required: output must be source-bound")
        if not cap.available:
            raise AdapterUnavailable(f"{self.target}: {cap.reason}")
        try:
            return self._run(frames, source_sha256, cap, seed=seed)
        except TypeError:
            return self._run(frames, source_sha256, cap)

    def _run(self, frames, source_sha256, cap) -> List[dict]:
        raise NotImplementedError


class BodyMoveNetAdapter(_Adapter):
    """MoveNet SinglePose Lightning, TFLite flatbuffer. No pickle surface."""
    target = "body"
    runtime_module = "ai_edge_litert"
    default_model = "/tmp/fairway-safe-body-runtime/movenet-singlepose-lightning-v3.tflite"
    KP = ["nose","eye_l","eye_r","ear_l","ear_r","sh_l","sh_r","el_l","el_r",
          "wr_l","wr_r","hip_l","hip_r","kn_l","kn_r","ank_l","ank_r"]

    def capability(self) -> Capability:
        return self._probe(
            safe=True,
            safe_reason="MoveNet TFLite flatbuffer via ai-edge-litert: no pickle, "
                        "no torch, no checkpoint conversion. The Ultralytics "
                        "weights_only=False blocker does not apply to this route.",
            demonstrated="safe-body-runtime worker ran f310-360 on the Tommy source; "
                         "independent validation of that run is recorded separately")

    def _run(self, frames, source_sha256, cap) -> List[dict]:
        import numpy as np
        from ai_edge_litert.interpreter import Interpreter
        itp = Interpreter(model_path=cap.model_path, num_threads=2)
        itp.allocate_tensors()
        inp = itp.get_input_details()[0]
        out = itp.get_output_details()[0]
        _, ih, iw, _ = inp["shape"]
        recs = []
        for source_frame, img in frames:
            h, w = img.shape[:2]
            # explicit letterbox transform, recorded so it can be inverted
            scale = min(iw / w, ih / h)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            import cv2
            resized = cv2.resize(img, (nw, nh))
            canvas = np.zeros((ih, iw, 3), np.uint8)
            ox, oy = (iw - nw) // 2, (ih - nh) // 2
            canvas[oy:oy+nh, ox:ox+nw] = resized
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            # MoveNet float input takes raw [0,255] float32, NOT [0,1]. Dividing
            # by 255 silently collapses every keypoint score to ~0.03.
            tensor = (rgb.astype(np.uint8)[None, ...] if inp["dtype"] == np.uint8
                      else rgb.astype(np.float32)[None, ...])
            itp.set_tensor(inp["index"], tensor)
            itp.invoke()
            kp = itp.get_tensor(out["index"])[0][0]      # (17,3) y,x,score
            pts = []
            for i, name in enumerate(self.KP):
                y, x, s = float(kp[i][0]), float(kp[i][1]), float(kp[i][2])
                nx, ny = x * iw - ox, y * ih - oy        # undo letterbox
                pts.append({"name": name,
                            "x": round(nx / scale, 2), "y": round(ny / scale, 2),
                            "score": round(s, 4), "visible": bool(s >= 0.30)})
            recs.append({
                "source_frame": int(source_frame), "source_sha256": source_sha256,
                "native_wh": [int(w), int(h)],
                "transform": {"letterbox_scale": round(scale, 6),
                              "offset_xy": [int(ox), int(oy)],
                              "model_input_wh": [int(iw), int(ih)]},
                "keypoints": pts,
                "visible_keypoint_count": sum(1 for p in pts if p["visible"]),
                "initialization": "automatic",
                "assistance": "none; single-person model, no human seed",
                "limitations": "single-person pose only; not golf-specific; no swing "
                               "phase, joint angle, club relationship or kinematics",
                "research_only": True, "pseudo_label": True,
                "ground_truth": False, "production_eligible": False,
            })
        return recs


class UltralyticsBodyAdapter(_Adapter):
    """Legacy body route. Permanently unsafe; retained so the reason is visible."""
    target = "body_legacy_ultralytics"
    runtime_module = "ultralytics"
    default_model = "/Users/giofiore/ghostcaddie-tour/yolo11n-pose.pt"

    def capability(self) -> Capability:
        c = self._probe(safe=False, safe_reason="")
        return Capability(**{**c.__dict__, "available": False, "safe": False,
            "reason": "UNSAFE: ultralytics/utils/patches.py injects "
                      "weights_only=False, so YOLO() performs an unrestricted "
                      "pickle load. Superseded by the MoveNet TFLite body route."})


class ClubheadSam2Adapter(_Adapter):
    target = "clubhead"
    runtime_module = "torch"
    default_model = "/tmp/fairway-sam-head/checkpoints/sam2.1_hiera_tiny.pt"

    def capability(self) -> Capability:
        return self._probe(
            safe=True,
            safe_reason="NOT EXECUTED by this adapter yet, so no load-path safety "
                        "is claimed here. It requires a reviewed source-specific "
                        "seed box; the seed path is not wired into the worker.",
            demonstrated="57 accepted frames f310-366 on the Tommy source, "
                         "independently audited; requires a reviewed seed box")

    def _run(self, frames, source_sha256, cap, seed=None):
        """Box-prompted SAM2.1 video segmentation from a reviewed seed.

        Loads ONLY through sam2/build_sam.py, which hardcodes
        torch.load(..., weights_only=True) with no fallback, and installs a guard
        that raises if the unused Hiera weights_path load is ever entered.
        """
        if seed is None or not seed.get("box_xyxy"):
            raise AdapterUnavailable(
                "clubhead requires a reviewed seed box bound to this source; "
                "it is not an unattended automatic detector")
        import os, sys, tempfile
        import numpy as np
        import cv2
        sys.path.insert(0, "/tmp/fairway-sam-head/pkgs")
        sys.path.insert(0, "/tmp/fairway-sam-head/sam2-src")
        import torch
        import sam2.modeling.backbones.hieradet as hd
        _orig = hd.Hiera.__init__
        def _guarded(self_, *a, **k):
            if k.get("weights_path") is not None:
                raise RuntimeError("unsafe backbone weights_path load is disabled")
            return _orig(self_, *a, **k)
        hd.Hiera.__init__ = _guarded
        from sam2.build_sam import build_sam2_video_predictor

        tmp = tempfile.mkdtemp(prefix="sam2frames_")
        order = []
        for i, (sf, img) in enumerate(frames):
            cv2.imwrite(os.path.join(tmp, f"{i:05d}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            order.append(int(sf))
        seed_idx = seed_index(order, seed["frame"])
        pred = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_t.yaml", cap.model_path,
            device=torch.device("cpu"), apply_postprocessing=False)
        st = pred.init_state(video_path=tmp, offload_video_to_cpu=True,
                             offload_state_to_cpu=True, async_loading_frames=False)
        pred.add_new_points_or_box(inference_state=st, frame_idx=seed_idx, obj_id=1,
                                   box=np.array(seed["box_xyxy"], dtype=np.float32))
        recs = []
        for idx, _ids, logits in pred.propagate_in_video(
                st, start_frame_idx=seed_idx, max_frame_num_to_track=len(order),
                reverse=False):
            m = (logits[0].detach().cpu() > 0).numpy()
            if m.ndim == 3:
                m = m[0]
            area = int(m.sum())
            sf = order[int(idx)]
            rec = {"source_frame": sf, "source_sha256": source_sha256,
                   "area_px": area, "visible": area > 0, "bbox_xyxy": None,
                   "mask_rle_area": area,
                   "initialization": "assisted",
                   "assistance": ("reviewed seed box on native frame "
                                  f"{seed['frame']}; AI-reviewed, not ground truth"),
                   "limitations": ("object-box/mask semantics only; the mask centroid "
                                   "is NOT a material point, NO trajectory, speed or 3D "
                                   "is implied"),
                   "research_only": True, "pseudo_label": True,
                   "ground_truth": False, "production_eligible": False}
            if area > 0:
                ys, xs = np.where(m)
                rec["bbox_xyxy"] = [int(xs.min()), int(ys.min()),
                                    int(xs.max() + 1), int(ys.max() + 1)]
                np.save(os.path.join(tmp, f"mask_{sf:06d}.npy"), m)
                rec["mask_path"] = os.path.join(tmp, f"mask_{sf:06d}.npy")
            recs.append(rec)
        recs.sort(key=lambda r: r["source_frame"])
        return recs


def ball_records(order, tracks_xy, occlusion, expected_dist, sx, sy,
                 seed_frame, source_sha256):
    """Build ball records from RAW model output, losing nothing.

    Kept deliberately pure (plain numbers in, dicts out) so the retention and
    seed/prediction contract is testable without torch or a checkpoint.

    Retention: the model's raw coordinate and BOTH gate logits are recorded on
    every frame, accepted or not, so a rejection can be diagnosed afterwards.
    Rendering is unaffected: point_xy stays None on a rejected frame, so no
    stale marker can be drawn.

    Seeds are not predictions: the frame carrying the reviewed seed is labelled
    state="seed"/is_seed=True and is excluded from predictive_frames().
    """
    import math
    sig = lambda z: 1 / (1 + math.exp(-z))
    recs = []
    for i, sf in enumerate(order):
        o, e = float(occlusion[i]), float(expected_dist[i])
        score = (1 - sig(o)) * (1 - sig(e))
        v = score > 0.5
        raw = [round(float(tracks_xy[i][0]) / sx, 2),
               round(float(tracks_xy[i][1]) / sy, 2)]
        is_seed = int(sf) == int(seed_frame)
        recs.append({
            "source_frame": int(sf), "source_sha256": source_sha256,
            "visible": v,
            "point_xy": list(raw) if v else None,
            "raw_point_xy": raw,
            "occlusion_logit": round(o, 4),
            "expected_dist_logit": round(e, 4),
            "visibility_score": round(score, 6),
            "is_seed": is_seed,
            "state": "seed" if is_seed else ("observed" if v else "unavailable"),
            "initialization": "assisted",
            "assistance": (f"reviewed seed point on native frame {seed_frame}; "
                           "AI-reviewed, not ground truth"),
            "limitations": ("image-space point track only; NO speed, carry, 3D, "
                            "launch angle or landing is implied. Gaps are gaps: "
                            "non-visible frames are never interpolated. "
                            "raw_point_xy is ungated model output kept for "
                            "diagnosis, NOT an accepted observation."),
            "research_only": True, "pseudo_label": True,
            "ground_truth": False, "production_eligible": False})
    return recs


def predictive_frames(records):
    """Accepted frames the model actually predicted: the seed never counts."""
    return sum(1 for r in records if r.get("visible") and not r.get("is_seed"))


def seed_index(order, seed_frame):
    """Position of the seeded frame in the decoded order. Refuses if absent.

    The previous `order.index(f) if f in order else 0` silently initialised the
    model on the FIRST decoded frame whenever the operator's chosen frame was
    not in the window, while still reporting the chosen frame as the seed. A
    seed that cannot be honoured must fail loudly, not quietly become a
    different seed.
    """
    try:
        return list(order).index(int(seed_frame))
    except ValueError:
        rng = f"{order[0]}..{order[-1]}" if len(order) else "empty"
        raise AdapterUnavailable(
            f"seed frame {seed_frame} is not in the decoded window ({rng}); "
            f"refusing to fall back to another frame") from None


def seed_box_from_point(point_xy, uncertainty_radius_px):
    """Square prompt box around a reviewed seed POINT, sized by its own stated
    uncertainty. This invents no new coordinate: it is the reviewed point and
    the radius the reviewer published with it.
    """
    r = float(uncertainty_radius_px)
    if r <= 0:
        raise ValueError("a seed point needs a stated uncertainty radius to become "
                         "a box; refusing to invent one")
    x, y = float(point_xy[0]), float(point_xy[1])
    return [x - r, y - r, x + r, y + r]


# ponytail: plausibility ceiling only, never a coverage knob. The reviewed seed
# box is 14x14 px (196 px^2); anything past ~10x that on a 1280x720 broadcast
# frame is turf/sky/player, not a golf ball. It can only REJECT an emission.
BALL_MAX_PLAUSIBLE_AREA_PX = 2000


def sam2_ball_record(source_frame, mask, source_sha256, seed_frame,
                     max_area_px=BALL_MAX_PLAUSIBLE_AREA_PX):
    """One ball record from a SAM2 mask. Loss and bad locks stay visible.

    An empty mask is a gap, never a carried-forward point. A mask far too large
    for a ball is reported as a failed lock ("implausible_area") rather than
    being drawn: a memory tracker that latches onto turf must show up as a
    failure in the raw record, not as a confident ball marker.
    """
    import numpy as np
    m = np.asarray(mask, bool)
    area = int(m.sum())
    is_seed = int(source_frame) == int(seed_frame)
    rec = {"source_frame": int(source_frame), "source_sha256": source_sha256,
           "area_px": area, "visible": False, "point_xy": None, "bbox_xyxy": None,
           "raw_centroid_xy": None,
           "is_seed": is_seed, "state": "unavailable",
           "initialization": "assisted",
           "assistance": (f"reviewed seed point on native frame {seed_frame}, widened "
                          "to a box by its own published uncertainty radius; "
                          "AI-reviewed, not ground truth"),
           "limitations": ("segmentation mask centroid in image space only; NO speed, "
                           "carry, 3D, launch angle or landing is implied. Gaps are "
                           "gaps: a frame with no mask is never interpolated."),
           "research_only": True, "pseudo_label": True,
           "ground_truth": False, "production_eligible": False}
    if area == 0:
        return rec
    ys, xs = np.where(m)
    # image-moment centroid in pixel-index coordinates, the same convention the
    # body keypoints and the renderer already use
    rec["raw_centroid_xy"] = [round(float(xs.mean()), 2), round(float(ys.mean()), 2)]
    if area > max_area_px:
        rec["state"] = "implausible_area"
        rec["reason"] = (f"mask area {area} px^2 exceeds the {max_area_px} px^2 "
                         "plausibility ceiling for a golf ball; reported as a failed "
                         "lock, not rendered")
        return rec
    rec["visible"] = True
    rec["state"] = "seed" if is_seed else "observed"
    rec["point_xy"] = list(rec["raw_centroid_xy"])
    rec["bbox_xyxy"] = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
    return rec


def _sam2_propagate(frames, model_path, seed_box, seed_frame):
    """Box-prompted SAM2.1 video propagation. Yields (source_frame, bool mask).

    Loads ONLY through sam2/build_sam.py, which hardcodes
    torch.load(..., weights_only=True) with no fallback, and installs a guard
    that raises if the unused Hiera weights_path load is ever entered.
    """
    import os, sys, tempfile
    import numpy as np
    import cv2
    sys.path.insert(0, "/tmp/fairway-sam-head/pkgs")
    sys.path.insert(0, "/tmp/fairway-sam-head/sam2-src")
    import torch
    import sam2.modeling.backbones.hieradet as hd
    _orig = hd.Hiera.__init__
    def _guarded(self_, *a, **k):
        if k.get("weights_path") is not None:
            raise RuntimeError("unsafe backbone weights_path load is disabled")
        return _orig(self_, *a, **k)
    hd.Hiera.__init__ = _guarded
    from sam2.build_sam import build_sam2_video_predictor

    tmp = tempfile.mkdtemp(prefix="sam2frames_")
    order = []
    for i, (sf, img) in enumerate(frames):
        cv2.imwrite(os.path.join(tmp, f"{i:05d}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        order.append(int(sf))
    seed_idx = seed_index(order, seed_frame)
    pred = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_t.yaml", model_path,
        device=torch.device("cpu"), apply_postprocessing=False)
    st = pred.init_state(video_path=tmp, offload_video_to_cpu=True,
                         offload_state_to_cpu=True, async_loading_frames=False)
    pred.add_new_points_or_box(inference_state=st, frame_idx=seed_idx, obj_id=1,
                               box=np.array(seed_box, dtype=np.float32))
    for idx, _ids, logits in pred.propagate_in_video(
            st, start_frame_idx=seed_idx, max_frame_num_to_track=len(order),
            reverse=False):
        m = (logits[0].detach().cpu() > 0).numpy()
        if m.ndim == 3:
            m = m[0]
        yield order[int(idx)], m, tmp


class BallSam2Adapter(_Adapter):
    """Pivot ball route: video object segmentation, not point tracking.

    The BootsTAPIR point-track route is retained (BallTapirAdapter) and is
    treated as falsified on this source: its RAW coordinates drift from the
    sparse AI reference points, so it is not a thresholding problem. This route
    is a different model class on the already-audited safe SAM2 load path.
    """
    target = "ball_sam2"
    runtime_module = "torch"
    default_model = "/tmp/fairway-sam-head/checkpoints/sam2.1_hiera_tiny.pt"

    def capability(self) -> Capability:
        return self._probe(
            safe=True,
            safe_reason="SAM2.1 tiny through sam2/build_sam.py, which hardcodes "
                        "weights_only=True with no fallback; the unused Hiera "
                        "weights_path load is guarded. Requires a reviewed "
                        "source-bound seed; not an unattended detector.",
            demonstrated="the same loader and propagation produced real clubhead "
                         "masks on this source; it is UNPROVEN on the ball, which "
                         "is far smaller and motion-blurred")

    def _run(self, frames, source_sha256, cap, seed=None):
        if seed is None or not (seed.get("point_xy") or seed.get("box_xyxy")):
            raise AdapterUnavailable(
                "ball_sam2 requires a reviewed seed bound to this source hash")
        box = seed.get("box_xyxy") or seed_box_from_point(
            seed["point_xy"], seed.get("uncertainty_radius_px", 0))
        recs = [sam2_ball_record(sf, m, source_sha256, int(seed["frame"]))
                for sf, m, _tmp in _sam2_propagate(frames, cap.model_path, box,
                                                   int(seed["frame"]))]
        recs.sort(key=lambda r: r["source_frame"])
        return recs


class BallTapirAdapter(_Adapter):
    target = "ball"
    runtime_module = "torch"
    default_model = "/tmp/fairway-learned/tracker/weights/bootstapir_checkpoint_v2.pt"

    def capability(self) -> Capability:
        return self._probe(
            safe=True,
            safe_reason="NOT EXECUTED by this adapter yet, so no load-path safety "
                        "is claimed here. It requires a source-specific reviewed "
                        "seed bound to the source hash.",
            demonstrated="did NOT transfer on a hash-bound cross-clip rerun "
                         "(1/51 Morikawa, 0/71 Gotterup); needs a source-specific "
                         "reviewed seed and is unproven on new footage")

    def _run(self, frames, source_sha256, cap, seed=None):
        """Point-tracked ball from a reviewed seed, via the repaired safe loader.

        The checkpoint is loaded with weights_only=True; there is no unrestricted
        fallback. Loss is preserved: a frame the model marks occluded/uncertain is
        reported not-visible, never interpolated.
        """
        if seed is None or not seed.get("point_xy"):
            raise AdapterUnavailable(
                "ball requires a reviewed seed point bound to this source hash")
        import sys
        import numpy as np
        import cv2
        sys.path.insert(0, "/tmp/fairway-learned/tracker")
        import inspect
        import torch
        from tapnet.torch import tapir_model

        order = [int(sf) for sf, _ in frames]
        seed_idx = seed_index(order, seed["frame"])
        H, W = frames[0][1].shape[:2]
        rw, rh = 512, 320
        vid = np.stack([cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (rw, rh))
                        for _, im in frames]).astype(np.float32)
        vid = torch.tensor(vid)[None] / 255.0 * 2 - 1
        sx, sy = rw / W, rh / H
        q = torch.tensor([[[float(seed_idx),
                            float(seed["point_xy"][1]) * sy,
                            float(seed["point_xy"][0]) * sx]]], dtype=torch.float32)
        # same safe-load contract as the audited tracker CLI: weights_only=True
        # with NO unrestricted fallback
        if "weights_only" not in inspect.signature(torch.load).parameters:
            raise AdapterUnavailable(
                "this torch has no weights_only support; refusing to load")
        try:
            sd = torch.load(cap.model_path, map_location="cpu", weights_only=True)
        except Exception as e:
            raise AdapterUnavailable(
                f"safe checkpoint load failed ({type(e).__name__}); aborting "
                f"without an unrestricted pickle fallback") from e
        model = tapir_model.TAPIR(pyramid_level=1)
        model.load_state_dict(sd)
        model = model.eval()
        with torch.no_grad():
            out = model(vid, q)
        tr = out["tracks"][0][0].detach().cpu().numpy()          # T,2 (x,y resized)
        occ = out["occlusion"][0][0].detach().cpu().numpy()
        exp = out["expected_dist"][0][0].detach().cpu().numpy()
        recs = ball_records(order=order, tracks_xy=[(float(p[0]), float(p[1])) for p in tr],
                            occlusion=[float(z) for z in occ],
                            expected_dist=[float(z) for z in exp],
                            sx=sx, sy=sy, seed_frame=int(seed["frame"]),
                            source_sha256=source_sha256)
        return recs


def all_adapters():
    return {a.target: a for a in (BodyMoveNetAdapter(), ClubheadSam2Adapter(),
                                  BallTapirAdapter(), BallSam2Adapter())}
