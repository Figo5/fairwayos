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

    def run_frames(self, frames: Sequence, source_sha256: Optional[str]) -> List[dict]:
        cap = self.capability()
        if not source_sha256:
            raise ValueError("source_sha256 is required: output must be source-bound")
        if not cap.available:
            raise AdapterUnavailable(f"{self.target}: {cap.reason}")
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
            safe_reason="SAM2.1 tiny: sam2/build_sam.py hardcodes "
                        "torch.load(..., weights_only=True) with no fallback; the "
                        "unused Hiera weights_path load is guarded at call time.",
            demonstrated="57 accepted frames f310-366 on the Tommy source, "
                         "independently audited; requires a reviewed seed box")

    def _run(self, frames, source_sha256, cap):
        raise AdapterUnavailable(
            "clubhead requires a reviewed seed box for the specific source; "
            "it is not an unattended automatic detector. Provide a reviewed seed.")


class BallTapirAdapter(_Adapter):
    target = "ball"
    runtime_module = "torch"
    default_model = "/tmp/fairway-learned/tracker/weights/bootstapir_checkpoint_v2.pt"

    def capability(self) -> Capability:
        return self._probe(
            safe=True,
            safe_reason="BootsTAPIR loads with weights_only=True via the repaired "
                        "safe loader.",
            demonstrated="did NOT transfer on a hash-bound cross-clip rerun "
                         "(1/51 Morikawa, 0/71 Gotterup); needs a source-specific "
                         "reviewed seed and is unproven on new footage")

    def _run(self, frames, source_sha256, cap):
        raise AdapterUnavailable(
            "ball requires a source-specific reviewed seed bound to the source "
            "hash; unattended operation is not demonstrated on new sources.")


def all_adapters():
    return {a.target: a for a in (BodyMoveNetAdapter(), ClubheadSam2Adapter(),
                                  BallTapirAdapter())}
