"""Fresh AI-vision frame decisions for bounded local golf-video smoke runs.

This module deliberately does not read prior demo decisions, evaluation references,
or seed coordinates. It decodes requested native frames, asks one bounded Hermes
child process to inspect those clean stills, validates the returned strict JSON,
and can render only the fresh observations it received.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Mapping, Sequence


TARGETS = ("ball", "clubhead")
MAX_BATCH_FRAMES = 32
DEFAULT_MAX_TOTAL_FRAMES = 256
MAX_CONCURRENCY = 4
BODY_SUPPORTED_KEYPOINTS = ("sh_l", "sh_r", "hip_l", "hip_r", "kn_l", "kn_r", "ank_l", "ank_r")
WITHHELD = ["saved demo decisions", "evaluation references", "annotation seeds"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_LIVE_PGIDS: set[int] = set()
_LIVE_PGID_LOCK = threading.Lock()


def terminate_live_children(sig: int = signal.SIGTERM) -> int:
    """Signal every child process group this module currently has in flight.

    Cancelling futures does not interrupt a thread blocked in Popen.communicate, so aborting a
    parallel batched run has to reach the OS: without this the executor still waits for the
    slow siblings of a batch that already failed.
    """
    with _LIVE_PGID_LOCK:
        pgids = sorted(_LIVE_PGIDS)
    signalled = 0
    for pgid in pgids:
        try:
            os.killpg(pgid, sig)
            signalled += 1
        except OSError:
            pass
    return signalled


def _run_bounded_process(cmd: Sequence[str], *, cwd: Path | None = None, timeout_s: float = 60) -> subprocess.CompletedProcess:
    popen_kwargs = {
        "cwd": str(cwd) if cwd is not None else None,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }
    if hasattr(os, "setsid"):
        popen_kwargs["preexec_fn"] = os.setsid
    proc = subprocess.Popen(list(cmd), **popen_kwargs)
    pgid = None
    if "preexec_fn" in popen_kwargs:  # own session, so its pgid is safe to signal
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            pgid = None
    if pgid is not None:
        with _LIVE_PGID_LOCK:
            _LIVE_PGIDS.add(pgid)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        if hasattr(os, "killpg"):
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    proc.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        try:
            proc.communicate(timeout=1)
        except Exception:
            pass
        raise exc
    finally:
        if pgid is not None:
            with _LIVE_PGID_LOCK:
                _LIVE_PGIDS.discard(pgid)
    return subprocess.CompletedProcess(list(cmd), proc.returncode, stdout, stderr)


def _run_checked(cmd: Sequence[str], *, cwd: Path | None = None, timeout_s: float = 60) -> subprocess.CompletedProcess:
    proc = _run_bounded_process(cmd, cwd=cwd, timeout_s=timeout_s)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout, stderr=proc.stderr)
    return proc


def ffprobe_video(path: Path) -> dict:
    out = _run_checked([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,codec_name,pix_fmt",
        "-of", "json", str(path),
    ], timeout_s=30).stdout
    s = json.loads(out)["streams"][0]
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "r_frame_rate": s.get("r_frame_rate"),
        "avg_frame_rate": s.get("avg_frame_rate"),
        "nb_frames": int(s["nb_frames"]) if str(s.get("nb_frames", "")).isdigit() else None,
        "duration": float(s["duration"]) if s.get("duration") else None,
        "codec_name": s.get("codec_name"),
        "pix_fmt": s.get("pix_fmt"),
    }


def validate_frames(frames: Sequence[int], nb_frames: int | None = None) -> list[int]:
    vals = list(frames)
    if not vals:
        raise ValueError("at least one frame is required")
    if len(vals) > 32:
        raise ValueError("at most 32 frames are allowed for smoke runs")
    if any(type(f) is not int for f in vals):
        raise ValueError("frames must be strict integer native indices")
    if vals != sorted(vals) or len(vals) != len(set(vals)):
        raise ValueError("frames must be unique and sorted native indices")
    if min(vals) < 0:
        raise ValueError("frames must be non-negative")
    if nb_frames is not None and max(vals) >= nb_frames:
        raise ValueError(f"frame {max(vals)} outside source frame count {nb_frames}")
    return vals


def decode_frames(video: Path, frames: Sequence[int], outdir: Path) -> dict[int, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}
    for frame in frames:
        p = outdir / f"native_{frame:06d}.jpg"
        _run_checked([
            "ffmpeg", "-y", "-v", "error", "-i", str(video),
            "-vf", f"select=eq(n\\,{frame})", "-vsync", "0", "-frames:v", "1", str(p),
        ], timeout_s=45)
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeError(f"failed to decode source frame {frame}")
        paths[frame] = p
    return paths


def build_prompt(video: Path, source_sha256: str, meta: Mapping, frame_paths: Mapping[int, Path], *, job_nonce: str | None = None) -> str:
    listing = "\n".join(f"- source_frame {f}: {p}" for f, p in frame_paths.items())
    frames = list(frame_paths)
    nonce = job_nonce or uuid.uuid4().hex
    return f"""You are doing a bounded blind golf-video visual inspection.

Use your vision tool on each local image path below, one frame at a time. Do not use or ask for annotation seeds, previous demo decisions, reference coordinates, hidden reports, or temporal copying. If a target is not visually separable, emit null/visible false. Inspect all listed frames before answering.

Source path: {video}
Source SHA-256: {source_sha256}
Geometry: {meta.get('width')}x{meta.get('height')}
Native frames requested: {frames}
Job nonce: {nonce}
Images:
{listing}

Return ONLY one strict JSON object with this shape. The job_nonce value must exactly match the Job nonce above:
{{
  "job_nonce": "{nonce}",
  "frames": [
    {{
      "source_frame": <one requested native frame index>,
      "ball": {{"visible": false, "point_xy": null, "confidence": 0.0, "uncertainty": "why/null or why visible"}},
      "clubhead": {{"visible": false, "bbox_xyxy": null, "point_xy": null, "confidence": 0.0, "uncertainty": "why/null or why visible"}}
    }}
  ],
  "provenance": {{"provider": "openai-codex", "model": "gpt-5.5", "method": "Hermes child vision tool over clean decoded native frames"}}
}}

Rules: coordinates are original image pixels; ball point is center; clubhead box is head-only, not shaft or hands; confidence is 0..1; use null for unsupported coordinates; no mph/carry/impact/landing claims."""


def build_hermes_command(query_file: Path, max_turns: int = 8) -> list[str]:
    query = query_file.read_text()
    return [
        "hermes", "--provider", "openai-codex", "-m", "gpt-5.5", "--reasoning", "medium",
        "chat", "--max-turns", str(max_turns), "--query", query,
    ]


def _extract_json(text: str, *, expected_nonce: str | None = None) -> dict:
    text = text.strip()
    decoder = json.JSONDecoder()

    def acceptable(obj: object) -> bool:
        return isinstance(obj, dict) and "frames" in obj and (expected_nonce is None or obj.get("job_nonce") == expected_nonce)

    try:
        obj = json.loads(text)
        if acceptable(obj):
            return obj
    except json.JSONDecodeError:
        pass

    found_nonce_mismatch = False
    for start, ch in reversed(list(enumerate(text))):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if acceptable(obj):
            return obj
        if isinstance(obj, dict) and "frames" in obj and expected_nonce is not None:
            found_nonce_mismatch = True
    msg = "No strict JSON object with frames found"
    if found_nonce_mismatch:
        msg = "No strict JSON object with matching job_nonce found"
    raise json.JSONDecodeError(msg, text, 0)

def _bounded_conf(v) -> float:
    if type(v) not in (int, float):
        raise ValueError("confidence must be a finite number in 0..1")
    x = float(v)
    if not math.isfinite(x) or not (0.0 <= x <= 1.0):
        raise ValueError("confidence must be a finite number in 0..1")
    return x


def _point(raw, width: int, height: int):
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
        return None
    if any(type(v) not in (int, float) for v in raw):
        return None
    x, y = float(raw[0]), float(raw[1])
    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
        return None
    return [x, y]


def _box(raw, width: int, height: int):
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 4:
        return None
    if any(type(v) not in (int, float) for v in raw):
        return None
    x0, y0, x1, y1 = [float(v) for v in raw]
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        return None
    if not (0 <= x0 < x1 < width and 0 <= y0 < y1 < height):
        return None
    return [x0, y0, x1, y1]


def normalize_frame_decision(raw: Mapping, *, source_sha256: str, width: int, height: int) -> dict:
    if len(source_sha256) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in source_sha256):
        raise ValueError("source_sha256 must be full hex sha256")
    source_frame = raw["source_frame"]
    if type(source_frame) is not int:
        raise ValueError("source_frame must be a strict integer native index")
    out = {
        "source_frame": source_frame,
        "source_sha256": source_sha256,
        "pseudo_label": True,
        "research_only": True,
        "ground_truth": False,
        "production_eligible": False,
    }
    for target in TARGETS:
        r = raw.get(target) or {}
        visible = r.get("visible")
        if type(visible) is not bool:
            raise ValueError(f"{target}.visible must be boolean")
        pt = _point(r.get("point_xy"), width, height)
        box = _box(r.get("bbox_xyxy"), width, height)
        if visible:
            if target == "ball" and pt is None:
                raise ValueError("ball.point_xy must be in-bounds when visible")
            if target == "clubhead" and pt is None and box is None:
                raise ValueError("clubhead point_xy or bbox_xyxy must be in-bounds when visible")
            confidence = _bounded_conf(r.get("confidence"))
        else:
            confidence = 0.0
        out[target] = {
            "visible": visible,
            "point_xy": pt if visible else None,
            "bbox_xyxy": box if visible else None,
            "confidence": confidence,
            "uncertainty": str(r.get("uncertainty") or ("visible" if visible else "not visually separable")),
            "pseudo_label": True,
            "ground_truth": False,
            "research_only": True,
            "production_eligible": False,
        }
    return out


def normalize_body_pose_records(raw_records: Sequence[Mapping], *, requested_frames: Sequence[int], source_sha256: str, width: int, height: int) -> list[dict]:
    by_frame = {r["source_frame"]: r for r in raw_records if type(r.get("source_frame")) is int}
    decisions: list[dict] = []
    for frame in requested_frames:
        raw = by_frame.get(int(frame))
        supported = {}
        min_score = 1.0
        if raw and raw.get("source_sha256") == source_sha256:
            for kp in raw.get("keypoints", []):
                name = kp.get("name")
                if name not in BODY_SUPPORTED_KEYPOINTS or type(kp.get("visible")) is not bool or not kp.get("visible"):
                    continue
                pt = _point([kp.get("x"), kp.get("y")], width, height)
                if pt is None:
                    continue
                score = _bounded_conf(kp.get("score"))
                supported[name] = {"x": pt[0], "y": pt[1], "score": score, "visible": True}
                min_score = min(min_score, score)
        state = "observed" if len(supported) >= 2 else "unavailable"
        decisions.append({
            "source_frame": int(frame),
            "source_sha256": source_sha256,
            "state": state,
            "visible": state == "observed",
            "keypoints": supported,
            "confidence": round(min_score if supported else 0.0, 4),
            "anchor": "torso_hips_legs",
            "method": "MoveNet SinglePose Lightning, TFLite via ai-edge-litert",
            "automatic": True,
            "unsupported_not_rendered": ["wr_l", "wr_r", "el_l", "el_r", "nose", "eye_l", "eye_r", "ear_l", "ear_r"],
            "pseudo_label": True,
            "research_only": True,
            "ground_truth": False,
            "production_eligible": False,
        })
    return decisions


def run_body_pose_inference(video: Path, frames: Sequence[int], source_sha256: str, outdir: Path, *, registry=None, timeout_s: float = 300) -> dict:
    from ghostcaddie.upload.runtimes import RuntimeRegistry

    started = time.perf_counter()
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        reg = registry or RuntimeRegistry.default()
        spec = reg.require("body")
        req = {"video": str(video), "source_sha256": source_sha256, "sampling_step": 1, "max_frames": len(frames), "frame_start": min(frames), "frame_end": max(frames), "seed": None}
        req_path = outdir / "body_request.json"
        req_path.write_text(json.dumps(req, indent=2, sort_keys=True))
        proc = _run_bounded_process([spec.interpreter, "-m", "ghostcaddie.upload.worker", "body", str(req_path)], cwd=Path(__file__).resolve().parents[2], timeout_s=timeout_s)
        (outdir / "body_stdout.log").write_text(proc.stdout or "")
        (outdir / "body_stderr.log").write_text(proc.stderr or "")
        try:
            parsed = json.loads((proc.stdout or "").strip().splitlines()[-1])
        except Exception as exc:
            parsed = {"ok": False, "error": f"unparseable worker output: {exc}"}
        runtime_s = round(time.perf_counter() - started, 3)
        if proc.returncode != 0 or not parsed.get("ok"):
            return {"state": "unavailable", "automatic": True, "observations": 0, "blocker": parsed.get("error", f"worker rc={proc.returncode}"), "runtime_seconds": runtime_s, "records": [], "logs": {"stdout": str(outdir / "body_stdout.log"), "stderr": str(outdir / "body_stderr.log")}}
        return {"state": "observed", "automatic": True, "observations": len(parsed.get("records", [])), "runtime_seconds": runtime_s, "records": parsed.get("records", []), "worker": {"interpreter": spec.interpreter, "returncode": proc.returncode}, "logs": {"stdout": str(outdir / "body_stdout.log"), "stderr": str(outdir / "body_stderr.log")}}
    except Exception as exc:
        return {"state": "unavailable", "automatic": True, "observations": 0, "blocker": str(exc), "runtime_seconds": round(time.perf_counter() - started, 3), "records": []}


def run_child_inference(query_file: Path, outdir: Path, max_turns: int, *, job_nonce: str | None = None, timeout_s: float = 900) -> tuple[dict, dict]:
    cmd = build_hermes_command(query_file, max_turns=max_turns)
    outdir.mkdir(parents=True, exist_ok=True)
    proc = _run_bounded_process(cmd, cwd=outdir, timeout_s=timeout_s)
    (outdir / "hermes_stdout.log").write_text(proc.stdout or "")
    (outdir / "hermes_stderr.log").write_text(proc.stderr or "")
    routing = {"command": cmd[:7] + ["chat", "--max-turns", str(max_turns), "--query", "<prompt omitted; see prompt.md>"], "returncode": proc.returncode, "cwd": str(outdir), "isolation_note": "child runs in per-job cwd; prompt-only withholding is not a sandbox"}
    if proc.returncode != 0:
        raise RuntimeError(f"Hermes child failed rc={proc.returncode}; stderr tail={(proc.stderr or '')[-500:]}")
    return _extract_json(proc.stdout, expected_nonce=job_nonce), routing


def render_video(video: Path, decisions: Sequence[Mapping], meta: Mapping, outdir: Path) -> Path:
    import cv2

    frames = [int(d["source_frame"]) for d in decisions]
    decoded = decode_frames(video, frames, outdir / "render_frames_clean")
    annotated = outdir / "render_frames_marked"
    shutil.rmtree(annotated, ignore_errors=True)  # a shorter run must not inherit a longer run's seq_*.jpg
    annotated.mkdir(parents=True, exist_ok=True)
    w, h = int(meta["width"]), int(meta["height"])
    for idx, d in enumerate(decisions):
        f = int(d["source_frame"])
        img = cv2.imread(str(decoded[f]))
        if img is None:
            raise RuntimeError(f"failed to load decoded frame {f}")
        for target, color in (("ball", (60, 220, 255)), ("clubhead", (80, 255, 120))):
            r = d[target]
            if r["visible"] and r.get("point_xy"):
                x, y = [int(round(v)) for v in r["point_xy"]]
                cv2.circle(img, (x, y), 14 if target == "ball" else 9, color, 2)
            if r["visible"] and r.get("bbox_xyxy"):
                x0, y0, x1, y1 = [int(round(v)) for v in r["bbox_xyxy"]]
                cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        body = d.get("body") or {}
        kps = body.get("keypoints") or {}
        for a, b in (("sh_l", "sh_r"), ("sh_l", "hip_l"), ("sh_r", "hip_r"), ("hip_l", "hip_r"), ("hip_l", "kn_l"), ("hip_r", "kn_r"), ("kn_l", "ank_l"), ("kn_r", "ank_r")):
            if a in kps and b in kps:
                pa, pb = kps[a], kps[b]
                cv2.line(img, (int(round(pa["x"])), int(round(pa["y"]))), (int(round(pb["x"])), int(round(pb["y"]))), (210, 130, 255), 2)
        for kp in kps.values():
            cv2.circle(img, (int(round(kp["x"])), int(round(kp["y"]))), 4, (210, 130, 255), 1)
        cv2.rectangle(img, (0, h - 54), (w, h), (16, 14, 14), -1)
        timing = "native-timed contiguous" if all((b - a) == 1 for a, b in zip(frames, frames[1:])) else "sampled non-temporal preview; no speed"
        cv2.putText(img, f"fresh GPT5.5 pseudo-labels | sampled {idx+1}/{len(frames)} source f{f} | {timing}", (16, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
        p = annotated / f"seq_{idx:06d}.jpg"
        cv2.imwrite(str(p), img)
    final = outdir / "fresh_ai_vision_overlay.mp4"
    is_contiguous = all((b - a) == 1 for a, b in zip(frames, frames[1:]))
    fps_exact = str(meta.get("r_frame_rate") or "30/1") if is_contiguous else "1"
    _run_checked(["ffmpeg", "-y", "-v", "error", "-framerate", fps_exact, "-i", str(annotated / "seq_%06d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-movflags", "+faststart", str(final)], timeout_s=120)
    _run_checked(["ffmpeg", "-v", "error", "-i", str(final), "-f", "null", "-"], timeout_s=60)
    return final


def run_smoke_pipeline(video: Path, frames: Sequence[int], outdir: Path, *, render: bool = True, max_turns: int = 8, budget_s: float | None = None) -> dict:
    """Run one bounded <=32-frame fresh job. budget_s caps this job's child and body timeouts so a
    job started near a caller's deadline cannot outlive it."""
    job_started = time.perf_counter()

    def remaining(cap: float) -> float:
        if budget_s is None:
            return cap
        return max(1.0, min(cap, budget_s - (time.perf_counter() - job_started)))

    video = video.resolve()
    if not video.exists() or not video.is_file():
        raise FileNotFoundError(video)
    outdir.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_file(video)
    meta = ffprobe_video(video)
    frame_list = validate_frames(frames, meta.get("nb_frames"))
    decoded = decode_frames(video, frame_list, outdir / "input_frames")
    query_file = outdir / "prompt.md"
    job_nonce = uuid.uuid4().hex
    query_file.write_text(build_prompt(video, source_sha, meta, decoded, job_nonce=job_nonce))
    raw, routing = run_child_inference(query_file, outdir, max_turns=max_turns, job_nonce=job_nonce, timeout_s=remaining(900.0))
    returned_frames = []
    for r in raw.get("frames", []):
        source_frame = r.get("source_frame")
        if type(source_frame) is not int:
            raise RuntimeError("child returned non-integer source_frame")
        returned_frames.append(source_frame)
    if returned_frames != frame_list or len(returned_frames) != len(set(returned_frames)):
        raise RuntimeError(f"child returned frames {returned_frames}, expected {frame_list}")
    by_frame = {r["source_frame"]: r for r in raw.get("frames", [])}
    decisions = [normalize_frame_decision(by_frame[f], source_sha256=source_sha, width=meta["width"], height=meta["height"]) for f in frame_list]
    body_status = run_body_pose_inference(video, frame_list, source_sha, outdir / "body_pose", timeout_s=remaining(300.0))
    body_decisions = normalize_body_pose_records(body_status.get("records", []), requested_frames=frame_list, source_sha256=source_sha, width=meta["width"], height=meta["height"])
    for decision, body in zip(decisions, body_decisions):
        decision["body"] = body
    doc = {
        "source": {"path": str(video), "sha256": source_sha, **meta},
        "interval_native_frames": [frame_list[0], frame_list[-1]],
        "requested_frames": frame_list,
        "inference": {"provenance": {**(raw.get("provenance") or {}), "input_policy": {"withheld": WITHHELD, "images": {str(k): str(v) for k, v in decoded.items()}}, "routing": routing}, "body_pose": {k: v for k, v in body_status.items() if k != "records"}},
        "decisions": decisions,
        "sampled_frame_preview": {"available": True, "frame_indices": frame_list, "temporal_interpretation": "contiguous native FPS only" if all((b - a) == 1 for a, b in zip(frame_list, frame_list[1:])) else "sampled non-temporal preview; no speed interpretation"},
        "metric_speed": {"available": False, "reason": "no calibration and no capture-action time; image observations only"},
        "research_only": True,
        "pseudo_label": True,
        "ground_truth": False,
        "production_eligible": False,
    }
    result_json = outdir / "fresh_ai_vision_results.json"
    result_json.write_text(json.dumps(doc, indent=2, sort_keys=True))
    outputs = {"decisions_json": str(result_json), "prompt": str(query_file), "stdout_log": str(outdir / "hermes_stdout.log"), "stderr_log": str(outdir / "hermes_stderr.log")}
    if render:
        final = render_video(video, decisions, meta, outdir)
        outputs["video"] = str(final)
        outputs["video_sha256"] = sha256_file(final)
    return {"ok": True, "outputs": outputs, "routing": routing}


def parse_interval(text: str) -> tuple[int, int]:
    """Parse a contiguous native interval "start:end" / "start-end" (both inclusive)."""
    parts = text.replace("-", ":").split(":")
    if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
        raise ValueError(f"interval must be START:END native indices, got {text!r}")
    return int(parts[0]), int(parts[1])


def partition_interval(start: int, end: int, *, batch_size: int = MAX_BATCH_FRAMES, max_total_frames: int = DEFAULT_MAX_TOTAL_FRAMES) -> list[list[int]]:
    """Split an inclusive native interval into bounded batches the <=32-frame job accepts."""
    if type(start) is not int or type(end) is not int:
        raise ValueError("interval bounds must be strict integer native indices")
    if start < 0:
        raise ValueError("interval bounds must be non-negative")
    if end < start:
        raise ValueError("interval end must not precede start")
    if type(batch_size) is not int or not (1 <= batch_size <= MAX_BATCH_FRAMES):
        raise ValueError(f"batch_size must be 1..{MAX_BATCH_FRAMES}")
    total = end - start + 1
    if total > max_total_frames:
        raise ValueError(f"interval of {total} frames exceeds max_total_frames={max_total_frames}")
    return [list(range(s, min(s + batch_size, end + 1))) for s in range(start, end + 1, batch_size)]


def reusable_batch_document(path: Path, *, frames: Sequence[int], source_sha256: str) -> dict | None:
    """Return a previously written batch document only if it is bound to this source and interval."""
    want = [int(f) for f in frames]
    try:
        doc = json.loads(Path(path).read_text())
    except Exception:
        return None
    if not isinstance(doc, dict) or (doc.get("source") or {}).get("sha256") != source_sha256:
        return None
    if doc.get("requested_frames") != want:
        return None
    decisions = doc.get("decisions") or []
    if [d.get("source_frame") for d in decisions] != want:
        return None
    if any(d.get("source_sha256") != source_sha256 for d in decisions):
        return None
    return doc


def merge_batch_documents(docs: Sequence[Mapping], *, requested_frames: Sequence[int], source_sha256: str) -> list[dict]:
    """Concatenate per-batch decisions in native order; reject drift, gaps and duplicates."""
    want = [int(f) for f in requested_frames]
    seen: dict[int, dict] = {}
    for doc in docs:
        if (doc.get("source") or {}).get("sha256") != source_sha256:
            raise ValueError("batch document was produced from a different source sha256")
        for decision in doc.get("decisions") or []:
            frame = decision.get("source_frame")
            if type(frame) is not int:
                raise ValueError("batch decision has a non-integer source_frame")
            if decision.get("source_sha256") != source_sha256:
                raise ValueError(f"decision for frame {frame} carries a foreign source sha256")
            if frame in seen:
                raise ValueError(f"duplicate decision for native frame {frame}")
            seen[frame] = dict(decision)
    missing = [f for f in want if f not in seen]
    if missing:
        raise ValueError(f"missing decisions for native frames {missing}")
    extra = sorted(set(seen) - set(want))
    if extra:
        raise ValueError(f"unrequested decisions for native frames {extra}")
    return [seen[f] for f in want]


def run_batched_pipeline(video: Path, start: int, end: int, outdir: Path, *, batch_size: int = MAX_BATCH_FRAMES,
                         max_total_frames: int = DEFAULT_MAX_TOTAL_FRAMES, concurrency: int = 1,
                         deadline_seconds: float = 3600.0, resume: bool = False, render: bool = True,
                         max_turns: int = 8) -> dict:
    """Track a contiguous native interval longer than one job by running bounded <=32-frame jobs.

    Each batch is an ordinary fresh job: its own decode, its own child inference, its own body pass,
    and its own untouched raw outputs on disk. Batching only partitions and concatenates; it never
    reinterprets a prediction from a neighbouring batch.
    """
    video = Path(video).resolve()
    if not video.exists() or not video.is_file():
        raise FileNotFoundError(video)
    if type(concurrency) is not int or not (1 <= concurrency <= MAX_CONCURRENCY):
        raise ValueError(f"concurrency must be 1..{MAX_CONCURRENCY}")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_file(video)
    meta = ffprobe_video(video)
    nb_frames = meta.get("nb_frames")
    if nb_frames is not None and end >= nb_frames:
        raise ValueError(f"frame {end} outside source frame count {nb_frames}")
    batches = partition_interval(start, end, batch_size=batch_size, max_total_frames=max_total_frames)
    requested = list(range(start, end + 1))
    started = time.perf_counter()
    records = [{"index": i, "frames": [b[0], b[-1]], "count": len(b), "status": "pending",
                "outdir": str(outdir / f"batch_{i:03d}_{b[0]}_{b[-1]}")} for i, b in enumerate(batches)]
    docs: list[dict | None] = [None] * len(batches)

    abort = threading.Event()
    failures: list[str] = []

    def budget_left() -> float:
        return deadline_seconds - (time.perf_counter() - started)

    def work(i: int) -> None:
        if abort.is_set():
            records[i]["status"] = "cancelled"
            return
        remaining = budget_left()
        if remaining <= 0:
            raise TimeoutError(f"no time left against deadline_seconds={deadline_seconds} before batch {i}")
        frames = batches[i]
        bdir = Path(records[i]["outdir"])
        result_json = bdir / "fresh_ai_vision_results.json"
        if resume:
            doc = reusable_batch_document(result_json, frames=frames, source_sha256=source_sha)
            if doc is not None:
                docs[i], records[i]["status"] = doc, "reused"
                return
        run_smoke_pipeline(video, frames, bdir, render=False, max_turns=max_turns, budget_s=remaining)
        docs[i] = json.loads(result_json.read_text())
        records[i]["status"] = "fresh"

    def record_failure(i: int, exc: BaseException) -> None:
        records[i]["status"], records[i]["error"] = "failed", str(exc)
        failures.append(f"batch {i} frames {records[i]['frames']}: {exc}")

    if concurrency == 1:
        for i in range(len(batches)):
            try:
                work(i)
            except Exception as exc:
                record_failure(i, exc)
                break
    else:
        pool = ThreadPoolExecutor(max_workers=concurrency)
        try:
            futures = {pool.submit(work, i): i for i in range(len(batches))}
            for future in as_completed(futures):  # completion order, so a fast failure is seen at once
                try:
                    future.result()
                except Exception as exc:
                    record_failure(futures[future], exc)
                    abort.set()
                    for pending in futures:
                        pending.cancel()
                    terminate_live_children()  # unblock siblings still waiting on a child process
                    break
        finally:
            # wait=True would re-introduce the very wait the abort just cancelled
            pool.shutdown(wait=False, cancel_futures=True)
    for record in records:
        if record["status"] == "pending":
            record["status"] = "cancelled" if failures else "skipped"

    elapsed = time.perf_counter() - started
    runtime_s = round(elapsed, 3)
    overrun = None
    if elapsed > deadline_seconds:
        overrun = TimeoutError(
            f"batched run exceeded deadline_seconds={deadline_seconds} (elapsed {runtime_s}s); failing closed")
    status_doc = {"source": {"path": str(video), "sha256": source_sha}, "interval_native_frames": [start, end],
                  "batch_size": batch_size, "concurrency": concurrency, "resume": resume,
                  "deadline_seconds": deadline_seconds, "deadline_exceeded": overrun is not None,
                  "runtime_seconds": runtime_s, "batches": records}
    (outdir / "batch_status.json").write_text(json.dumps(status_doc, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError(
            "bounded batched run failed closed; no merged document written. "
            f"Completed batches are preserved under {outdir} and can be reused with resume=True. "
            f"Status: {outdir / 'batch_status.json'}. Failures: " + "; ".join(failures))
    if overrun is not None:
        # every batch succeeded, but a run that ran past its bound must not report success
        raise overrun

    decisions = merge_batch_documents([d for d in docs if d is not None], requested_frames=requested, source_sha256=source_sha)
    doc = {
        "source": {"path": str(video), "sha256": source_sha, **meta},
        "interval_native_frames": [start, end],
        "requested_frames": requested,
        "batching": {"batch_size": batch_size, "batch_count": len(batches), "concurrency": concurrency,
                     "max_total_frames": max_total_frames, "deadline_seconds": deadline_seconds,
                     "resume": resume, "runtime_seconds": runtime_s, "batches": records,
                     "note": "per-batch raw outputs are kept unmodified; merging concatenates decisions only"},
        "inference": {"per_batch": [{"index": i, "frames": records[i]["frames"], "status": records[i]["status"],
                                     "inference": (docs[i] or {}).get("inference")} for i in range(len(batches))]},
        "decisions": decisions,
        "sampled_frame_preview": {"available": True, "frame_indices": requested,
                                  "temporal_interpretation": "contiguous native FPS only"},
        "metric_speed": {"available": False, "reason": "no calibration and no capture-action time; image observations only"},
        "research_only": True,
        "pseudo_label": True,
        "ground_truth": False,
        "production_eligible": False,
    }
    result_json = outdir / "fresh_ai_vision_results.json"
    result_json.write_text(json.dumps(doc, indent=2, sort_keys=True))
    outputs = {"decisions_json": str(result_json), "batch_status": str(outdir / "batch_status.json")}
    if render:
        final = render_video(video, decisions, meta, outdir)
        outputs["video"] = str(final)
        outputs["video_sha256"] = sha256_file(final)
    return {"ok": True, "outputs": outputs, "batching": status_doc}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bounded local video -> fresh Hermes GPT5.5 vision result")
    ap.add_argument("video", type=Path)
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--frames", help="comma-separated native frame indices, e.g. 3058,3068,3074 (max 32)")
    sel.add_argument("--interval", help="contiguous inclusive native interval START:END, batched into bounded jobs")
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/fairway-reusable-vision-smoke"))
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=MAX_BATCH_FRAMES, help=f"frames per job, 1..{MAX_BATCH_FRAMES} (--interval only)")
    ap.add_argument("--max-total-frames", type=int, default=DEFAULT_MAX_TOTAL_FRAMES, help="hard cap on interval length")
    ap.add_argument("--concurrency", type=int, default=1, help=f"parallel jobs, 1..{MAX_CONCURRENCY}; sequential by default")
    ap.add_argument("--deadline-seconds", type=float, default=3600.0, help="wall-clock bound for the whole batched run")
    ap.add_argument("--resume", action="store_true", help="reuse completed batches whose source sha256 and frames match; fresh inference otherwise")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.interval:
        start, end = parse_interval(args.interval)
        result = run_batched_pipeline(
            args.video, start, end, args.outdir, batch_size=args.batch_size,
            max_total_frames=args.max_total_frames, concurrency=args.concurrency,
            deadline_seconds=args.deadline_seconds, resume=args.resume,
            render=not args.no_render, max_turns=args.max_turns)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    frames = [int(x.strip()) for x in args.frames.split(",") if x.strip()]
    result = run_smoke_pipeline(args.video, frames, args.outdir, render=not args.no_render, max_turns=args.max_turns)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
