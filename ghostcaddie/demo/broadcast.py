"""The broadcast showcase orchestrator: several scenes, one whole-shot render.

This module is **stdlib-only**, like :mod:`ghostcaddie.demo.pipeline`: it validates the
config, binds the source, writes the spec, runs the CV worker in another interpreter
and verifies what came back. It never imports cv2 or numpy.

The shape of a run
------------------
1. **Parse** the broadcast config (:mod:`ghostcaddie.demo.broadcast_config`).
2. **Bind** it to the real clip: hash, probe, interval, windows. A clip whose digest
   does not match ``shot.expected_source_sha256`` is refused, not warned about.
3. **One spec per scene**, built through the *existing* demo spec builder, with
   ``interval = the scene window``. That is what makes a scene an ordinary demo run:
   its ROI, seeds, corrections, anchors, interpolation limits, target options and
   phases are validated and measured by the code that already does it.
4. **One render** of the whole clip, at the source's own rational cadence, with
   overlays drawn only inside the windows, then the clip's own audio muxed back on
   with ``-c:a copy``.
5. **Verify**: output frame count equals the source's, the rational cadence matches,
   the audio stream and its packet md5 match the source, and moov precedes mdat. A
   failure here is a failure of the run, stated in the artifacts, never a silent pass.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import DEMO_TOOL_VERSION
from .broadcast_config import (BROADCAST_CONFIG_SCHEMA, BroadcastConfig,
                               ResolvedBroadcast, validate_broadcast)
from .config import ResolvedConfig, effective_roi
from .errors import (DemoConfigError, DemoError, DemoRenderError, DemoSourceError)
from .paths import resolve_cv_interpreter, run_demo_worker, sha256_of
from .probe import probe_timestamps

BROADCAST_ANALYSIS_SCHEMA = "fairwayos-broadcast-analysis.v1"
BROADCAST_PROVENANCE_SCHEMA = "fairwayos-broadcast-provenance.v1"
BROADCAST_REPORT_NAME = "broadcast_report.md"

#: Output directories a broadcast run must never write into. These hold the baseline
#: research media and the original showcase, both of which are evidence: a render that
#: overwrote them would destroy the thing it is being compared against.
PROTECTED_OUTPUT_PREFIXES = (
    "/Users/giofiore/fairway-os-demo/pga-showcase",
    "/Users/giofiore/fairway-os-runs/tiger-masters-2019-2026-10-01",
)

#: The repository this code lives in. No generated media is ever written inside it,
#: so the whole tree is protected, not just paths that happen to contain `.git`.
REPO_ROOT = Path(__file__).resolve().parents[2]

# The parent-source manifest used to be a path to one operator's run directory baked in
# here. It is not any more: `shot.parent_source_manifest` names one explicitly, and
# otherwise the manifest is looked for beside the clip itself (see
# :func:`_manifest_candidates`). A digest of the parent source is re-stated only when a
# manifest that actually contains this clip was read, so provenance cannot attribute a
# clip to a manifest it did not verify.


def _utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------- output paths

def broadcast_out_paths(out_dir) -> Dict[str, Any]:
    root = Path(out_dir)
    return {
        "root": root,
        "annotated": root / "broadcast_annotated.mp4",
        "video_only": root / ".broadcast_video_only.mp4",
        "report": root / BROADCAST_REPORT_NAME,
        "analysis": root / "broadcast_analysis.json",
        "provenance": root / "broadcast_provenance.json",
        "config_copy": root / "broadcast_config.json",
        "contact_sheet": root / "broadcast_contact_sheet.png",
        "keyframes_dir": root / "keyframes",
        "debug_dir": root / "broadcast_debug",
    }


def guard_output_dir(out_dir, *, repo_root: Optional[str] = None,
                     force: bool = False) -> Path:
    """Refuse an output directory that would damage evidence, or a finished run.

    The check is on the realpath, so a symlink into a protected tree is caught, and it
    applies to the directory itself *and* to everything under it. The git work tree is
    protected in full -- not only paths that happen to contain ``.git`` -- because the
    repository policy is that no generated media is ever committed.
    """
    root = Path(out_dir).expanduser()
    real = Path(os.path.realpath(str(root)))
    for protected in PROTECTED_OUTPUT_PREFIXES:
        if str(real) == protected or str(real).startswith(protected + os.sep):
            raise DemoConfigError(
                f"refusing to write broadcast output into {real}: it is inside the "
                f"preserved tree {protected}. The baseline clips and the original "
                f"showcase are evidence and are never written to.")
    repo = Path(os.path.realpath(repo_root)) if repo_root else REPO_ROOT
    repo = Path(repo).resolve()
    if real == repo or str(real).startswith(str(repo) + os.sep):
        raise DemoConfigError(
            f"refusing to write broadcast output into {real}: it is inside the git "
            f"work tree {repo}. Generated media is never written into the repository; "
            f"choose an output directory outside it (the run directory is the usual "
            f"place).")
    finished = broadcast_out_paths(root)["annotated"]
    if finished.exists() and not force:
        raise DemoConfigError(
            f"{finished} already exists. A broadcast run does not overwrite a finished "
            f"render by accident; pass --force to replace it, or choose another --out.")
    return root


# ---------------------------------------------------------------- the spec

def _scene_spec(scene, resolved: ResolvedBroadcast, *, out_dir: str,
                paths: Dict[str, Any], body_model, redact_path: bool) -> dict:
    """A complete demo spec for ONE scene, with ``interval = the scene window``."""
    from .pipeline import build_spec

    cfg = scene.scene.demo
    frame_times = None
    timestamps = getattr(resolved.source, "timestamps", None)
    if timestamps:
        frame_times = [float(t) for t in
                       timestamps[scene.start_frame:scene.end_frame + 1]]
        if len(frame_times) != scene.frame_count:
            frame_times = None
    scene_resolved = ResolvedConfig(
        config=cfg, source=resolved.source, start_frame=scene.start_frame,
        end_frame=scene.end_frame, frame_count=scene.frame_count,
        warnings=list(scene.warnings), notes=[], frame_times=frame_times)
    spec = build_spec(scene_resolved, sha256_of(resolved.source.path), out_dir, paths,
                      body_model=body_model, render_enabled=False,
                      redact_path=redact_path, showcase_enabled=False)
    spec["roi_effective"] = effective_roi(cfg, scene.start_frame)
    return spec


def build_broadcast_spec(resolved: ResolvedBroadcast, *, out_dir: str,
                         paths: Dict[str, Any], body_model, redact_path: bool,
                         render_enabled: bool) -> dict:
    """The whole-run spec: the render contract plus one measurement spec per scene."""
    from .config import SOURCE_TIMING_DEFAULT_CAVEAT
    from .pipeline import build_spec

    cfg = resolved.config
    presentation = cfg.presentation.to_dict()
    render = dict(cfg.render)
    render.setdefault("max_width", 1920)
    render.setdefault("crf", 18)
    render.setdefault("preset", "medium")
    showcase_block = {
        "enabled": True,
        "brand": presentation["brand"], "brand_sub": presentation["brand_sub"],
        "disclosure": presentation["disclosure"], "metrics": presentation["metrics"],
        "card_debounce_frames": presentation["card_debounce_frames"],
        "card_row_max_width": presentation["card_row_max_width"],
        "card_position": presentation["card_position"],
        "ball_trail_points": presentation["ball_trail_points"],
        "club_trail_points": presentation["club_trail_points"],
        "impact_accent": presentation["impact_accent"],
        "impact_accent_span_frames": presentation["impact_accent_span_frames"],
    }
    timing_block = dict(cfg.timing)
    timing_block.setdefault("nominal_fps_trusted", False)
    timing_block.setdefault("accept_variable_cadence", False)
    timing_block.setdefault("source_cadence_caveat", SOURCE_TIMING_DEFAULT_CAVEAT)

    source = resolved.source
    scene_entries: List[dict] = []
    for scene in resolved.scenes:
        entry = {
            "id": scene.scene.scene_id, "label": scene.scene.label,
            "kind": scene.scene.kind, "stroke_type": scene.scene.stroke_type,
            "window": {"start_frame": scene.start_frame,
                       "end_frame": scene.end_frame,
                       "frame_count": scene.frame_count,
                       "note": (scene.scene.window.note or "")},
            "keyframes": dict(scene.scene.keyframes),
            "vocabulary": list(scene.scene.phase_vocabulary),
            "contact_frame": scene.scene.contact_frame,
            "metrics": list(scene.scene.metrics) if scene.scene.metrics else None,
            "metric_hide_frames": [list(pair) for pair in scene.scene.metric_hide_frames],
            "handedness": scene.scene.demo.handedness,
            "spec": None,
        }
        if scene.scene.is_stroke:
            entry["spec"] = _scene_spec(scene, resolved, out_dir=out_dir, paths=paths,
                                        body_model=body_model,
                                        redact_path=redact_path)
        scene_entries.append(entry)

    audio = _audio_contract(source)
    return {
        "video": source.path,
        "source_sha256": sha256_of(source.path),
        "native_fps": float(source.fps),
        "native_fps_rational": _fps_rational(source),
        "native_wh": [int(source.width), int(source.height)],
        "pixel_aspect": float(source.pixel_aspect_ratio),
        "handedness": "right",
        "interval": {"start_frame": resolved.start_frame,
                     "end_frame": resolved.end_frame},
        "roi": None,
        "roi_keyframes": [],
        "roi_effective": None,
        "roi_author": "",
        "seeds": {},
        "corrections": {},
        "manual_segments": {},
        "targets": {},
        "interpolation": {"enabled": True, "max_gap_frames": 2},
        "phases": {"keyframes": {}, "allow_approx": False, "fps": float(source.fps),
                   "frame_times": None, "start_frame": resolved.start_frame},
        "body": {"model_path": getattr(body_model, "path", ""),
                 "model_found": bool(getattr(body_model, "found", False))},
        "render": {"enabled": bool(render_enabled),
                   "path": str(paths["video_only"]),
                   "max_width": render.get("max_width"),
                   "crf": render.get("crf"), "preset": render.get("preset"),
                   "style": "broadcast", "stride": 1},
        "showcase": showcase_block,
        "showcase_enabled": True,
        "broadcast": {
            "outside_windows": cfg.outside_windows,
            "final_path": str(paths["annotated"]),
            "audio": audio,
            "scenes": scene_entries,
            "presentation": presentation,
            "timing": timing_block,
            "scene_windows": [{"id": s.scene.scene_id,
                               "start_frame": s.start_frame,
                               "end_frame": s.end_frame,
                               "kind": s.scene.kind,
                               "stroke_type": s.scene.stroke_type}
                              for s in resolved.scenes],
            "cut_candidates": [c.to_dict() for c in cfg.cuts],
            "cut_overrides": list(cfg.overrides),
            "spacing_records": list(cfg.spacing_records),
        },
        "keyframes_dir": str(paths["keyframes_dir"]),
        "contact_sheet": str(paths["contact_sheet"]),
        "debug_dir": str(paths["debug_dir"]),
        "redact_path": bool(redact_path),
        "tool_version": DEMO_TOOL_VERSION,
    }


def _fps_rational(source) -> str:
    """The source's own frame-rate rational, taken from ffprobe's text, not a float."""
    raw = getattr(source, "raw_stream", None) or {}
    text = str(raw.get("r_frame_rate") or raw.get("avg_frame_rate") or "").strip()
    if text and text not in ("0/0", "N/A") and "/" in text:
        try:
            if Fraction(text) > 0:
                return text
        except (ValueError, ZeroDivisionError):
            pass
    value = float(source.fps or 0.0)
    if value <= 0:
        raise DemoSourceError("the source's frame rate could not be determined")
    frac = Fraction(value).limit_denominator(1000000)
    return f"{frac.numerator}/{frac.denominator}"


def _audio_contract(source) -> dict:
    raw = getattr(source, "raw_stream", None) or {}
    audio_raw = {}
    # The probe keeps the video stream; the audio stream is read here with one
    # ffprobe call so the contract can state whether there is audio to preserve.
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=codec_name,sample_rate,channels,time_base,duration",
             "-of", "json", source.path], capture_output=True, text=True, timeout=60)
        payload = json.loads(p.stdout or "{}")
        streams = payload.get("streams") or []
        audio_raw = streams[0] if streams else {}
    except Exception:  # pragma: no cover - ffprobe missing
        audio_raw = {}
    return {"mode": "copy", "source": source.path,
            "present": bool(audio_raw),
            "codec_name": audio_raw.get("codec_name", ""),
            "sample_rate": audio_raw.get("sample_rate", ""),
            "channels": audio_raw.get("channels", ""),
            "time_base": audio_raw.get("time_base", ""),
            "duration": audio_raw.get("duration", "")}


# ---------------------------------------------------------------- verification

def _ffprobe_json(args: Sequence[str]) -> dict:
    p = subprocess.run(["ffprobe", "-v", "error", *args, "-of", "json"],
                       capture_output=True, text=True, timeout=120)
    try:
        return json.loads(p.stdout or "{}")
    except json.JSONDecodeError:
        return {}


def _midx_md5(ffmpeg: str, path: str) -> str:
    try:
        p = subprocess.run([ffmpeg, "-v", "error", "-i", path, "-map", "0:a:0",
                            "-c", "copy", "-f", "md5", "-"], capture_output=True,
                           text=True, timeout=600)
        text = (p.stdout or "").strip()
        return text.splitlines()[-1].strip() if text else ""
    except Exception:  # pragma: no cover - ffmpeg missing
        return ""


def verify_broadcast_render(path: str, *, expected_frames: int, expected_rational: str,
                            source_path: str, source_audio: Mapping[str, Any],
                            ffmpeg: str = "ffmpeg") -> dict:
    """Prove the output keeps the source's frames, cadence and audio.

    Each check is recorded with its own result, so a partial pass cannot be read as a
    full one. ``problems`` is what makes the run fail.
    """
    result: Dict[str, Any] = {"checked": False, "problems": []}
    if not os.path.isfile(path):
        result["reason"] = "the output file does not exist"
        return result
    streams = _ffprobe_json(["-select_streams", "v:0", "-show_entries",
                             "stream=codec_name,pix_fmt,width,height,nb_frames,"
                             "r_frame_rate,avg_frame_rate", "-count_frames",
                             "-show_entries", "stream=nb_read_frames",
                             path]).get("streams") or []
    if not streams:
        result["reason"] = "the output has no video stream"
        return result
    video = streams[0]
    result.update({
        "checked": True, "codec": video.get("codec_name"),
        "pix_fmt": video.get("pix_fmt"), "width": video.get("width"),
        "height": video.get("height"),
        "r_frame_rate": video.get("r_frame_rate"),
        "avg_frame_rate": video.get("avg_frame_rate"),
        "nb_frames_header": _int_or_none(video.get("nb_frames")),
        "nb_frames_counted": _int_or_none(video.get("nb_read_frames")),
    })
    counted = result["nb_frames_counted"] or result["nb_frames_header"]
    if counted != int(expected_frames):
        result["problems"].append(
            f"the output holds {counted} frames, the source has {expected_frames}")
    for key in ("r_frame_rate", "avg_frame_rate"):
        got = str(video.get(key) or "")
        if not _same_rational(got, expected_rational):
            result["problems"].append(
                f"the output {key} is {got!r}, the source's is "
                f"{expected_rational!r}")
    if video.get("codec_name") != "h264":
        result["problems"].append(f"codec is {video.get('codec_name')!r}, not h264")
    if video.get("pix_fmt") != "yuv420p":
        result["problems"].append(f"pix_fmt is {video.get('pix_fmt')!r}, not yuv420p")

    # --- audio: present, same shape, bit-exact copy
    audio: Dict[str, Any] = {"checked": True}
    if source_audio.get("present"):
        out_streams = _ffprobe_json(["-select_streams", "a:0", "-show_entries",
                                     "stream=codec_name,sample_rate,channels",
                                     path]).get("streams") or []
        if not out_streams:
            audio["present"] = False
            result["problems"].append(
                "the source has audio and the output has no audio stream")
        else:
            stream = out_streams[0]
            audio.update({"present": True, "codec_name": stream.get("codec_name"),
                          "sample_rate": stream.get("sample_rate"),
                          "channels": stream.get("channels")})
            for key in ("codec_name", "sample_rate", "channels"):
                expected = str(source_audio.get(key) or "")
                if expected and str(stream.get(key) or "") != expected:
                    result["problems"].append(
                        f"audio {key} is {stream.get(key)!r}, the source's is "
                        f"{expected!r}")
            source_md5 = _midx_md5(ffmpeg, source_path)
            output_md5 = _midx_md5(ffmpeg, path)
            audio["source_audio_md5"] = source_md5
            audio["output_audio_md5"] = output_md5
            audio["bit_exact"] = bool(source_md5) and source_md5 == output_md5
            if not audio["bit_exact"]:
                result["problems"].append(
                    "the output audio packets do not match the source's (the audio "
                    "was re-encoded or trimmed rather than copied)")
    else:
        audio["present"] = False
        audio["reason"] = "the source has no audio stream"
    result["audio"] = audio

    # --- faststart
    with open(path, "rb") as fh:
        head = fh.read(1 << 20)
    mdat_pos = head.find(b"mdat")
    moov_pos = head.find(b"moov")
    faststart = moov_pos >= 0 and (mdat_pos < 0 or moov_pos < mdat_pos)
    result["faststart"] = faststart
    if not faststart:
        result["problems"].append("moov atom is not before mdat (not faststart)")

    # --- decoded presentation timestamps: EVERY decoded frame of both files must sit at
    # the same source-timeline position. A prefix check is not enough -- an offset
    # introduced at frame 200, or a duplicated frame at the tail, passes a five-frame
    # prefix and fails here. Both lists are read with the same reader the source probe
    # uses (the decoded frames' own presentation timestamps, in presentation order,
    # never sorted, JSON so side-data cannot collide with the values), one bounded
    # ffprobe call per file. No pixels are decoded and nothing is re-timed.
    source_ts = probe_timestamps(source_path)
    output_ts = probe_timestamps(path)
    source_pts = source_ts.get("timestamps") or []
    output_pts = output_ts.get("timestamps") or []
    interval = 1.0 / float(Fraction(expected_rational)) if expected_rational else 0.0
    stamps: Dict[str, Any] = {
        "checked": bool(source_pts) and bool(output_pts),
        "basis": "decoded frames' own presentation timestamps (ffprobe, presentation "
                 "order, never sorted)",
        "source_frames": len(source_pts), "output_frames": len(output_pts),
        "compared_frames": min(len(source_pts), len(output_pts)),
        "source_note": source_ts.get("timestamp_note", ""),
        "output_note": output_ts.get("timestamp_note", ""),
    }
    if not source_pts or not output_pts:
        stamps["reason"] = (
            "the frame timestamps of "
            + ("the source and the output" if not source_pts and not output_pts
               else "the output" if not output_pts else "the source")
            + " could not be read, so the two files' frame timelines were not compared")
        result["problems"].append(stamps["reason"])
    else:
        if len(source_pts) != len(output_pts):
            stamps["frame_count_mismatch"] = True
            result["problems"].append(
                f"the two files decode a different number of timestamped frames "
                f"({len(source_pts)} source, {len(output_pts)} output), so a frame index "
                f"does not mean the same thing in both")
        deltas = [abs(float(a) - float(b)) for a, b in zip(source_pts, output_pts)]
        worst = max(deltas)
        stamps.update({
            "source_first_pts": round(source_pts[0], 6),
            "source_last_pts": round(source_pts[-1], 6),
            "output_first_pts": round(output_pts[0], 6),
            "output_last_pts": round(output_pts[-1], 6),
            "max_delta_seconds": round(worst, 6),
            "mean_delta_seconds": round(sum(deltas) / len(deltas), 6),
            "half_frame_interval": round(0.5 * interval, 6),
            "frames_within_half_a_frame": sum(
                1 for delta in deltas if not interval or delta <= 0.5 * interval),
            "output_strictly_increasing": all(
                b > a for a, b in zip(output_pts, output_pts[1:])),
        })
        if interval and worst > 0.5 * interval:
            result["problems"].append(
                f"the output's frames sit up to {worst:.6f}s away from the source's "
                f"across all {stamps['compared_frames']} compared frames, more than half "
                f"a frame interval ({0.5 * interval:.6f}s)")
        if len(output_pts) > 1 and not stamps["output_strictly_increasing"]:
            result["problems"].append(
                "the output's decoded frame timestamps are not strictly increasing, so "
                "at least one frame was duplicated or re-timed rather than carried "
                "through")
    result["timestamps"] = stamps

    # --- duration: the output has to hold the frames it claims, for as long as those
    # frames are, and for as long as the source it was cut from. Duration is read from
    # the container and the video stream; the expectation is
    # ``expected_frames / source cadence``, computed from the RATIONAL so it cannot
    # drift by a rounding. A truncated tail and an appended tail both fail here.
    source_durations = _durations(source_path)
    output_durations = _durations(path)
    expected_duration = (float(expected_frames) / float(Fraction(expected_rational))
                         if expected_frames and expected_rational else 0.0)
    # Half a frame, with a 1 ms floor: a mux rounds the container's duration header to
    # its own time base (measured at ~0.3 ms on this footage), while a dropped or
    # appended frame is a whole frame -- so half a frame separates the two without
    # demanding that the footage be re-timed to make a header agree.
    tolerance = max(0.001, 0.5 * interval) if interval else 0.05
    duration: Dict[str, Any] = {
        "checked": False, "basis": "ffprobe container and first video-stream duration",
        "source_seconds": source_durations, "output_seconds": output_durations,
        "expected_seconds_from_frames_and_cadence": (round(expected_duration, 6)
                                                     if expected_duration else None),
        "tolerance_seconds": round(tolerance, 6),
    }
    measured = (output_durations["video"] if output_durations["video"] is not None
                else output_durations["container"])
    if measured is None:
        duration["reason"] = "the output's duration could not be read at all"
        result["problems"].append(duration["reason"])
    else:
        duration["checked"] = True
        duration["measured_seconds"] = round(measured, 6)
        if expected_duration:
            delta = abs(measured - expected_duration)
            duration["delta_vs_expected_seconds"] = round(delta, 6)
            if delta > tolerance:
                result["problems"].append(
                    f"the output runs {measured:.6f}s, but {expected_frames} frames at "
                    f"{expected_rational} is {expected_duration:.6f}s (a difference of "
                    f"{delta:.6f}s, tolerance {tolerance:.6f}s)")
        source_measured = (source_durations["video"]
                           if source_durations["video"] is not None
                           else source_durations["container"])
        if source_measured is not None:
            delta = abs(measured - source_measured)
            duration["delta_vs_source_seconds"] = round(delta, 6)
            if delta > tolerance:
                result["problems"].append(
                    f"the output runs {measured:.6f}s, the source {source_measured:.6f}s "
                    f"(a difference of {delta:.6f}s, tolerance {tolerance:.6f}s)")
    result["duration"] = duration

    result["ok"] = not result["problems"]
    return result


def _durations(path: str) -> Dict[str, Any]:
    """Container and first video-stream durations in seconds, when ffprobe has them."""
    payload = _ffprobe_json(["-show_entries",
                             "format=duration:stream=codec_type,duration", path])
    format_block = payload.get("format") or {}
    video = next((s for s in (payload.get("streams") or [])
                  if s.get("codec_type") == "video"), {})
    return {"container": _float_or_none(format_block.get("duration")),
            "video": _float_or_none(video.get("duration"))}


def _float_or_none(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _int_or_none(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _same_rational(got: str, expected: str) -> bool:
    if got == expected:
        return True
    try:
        a, b = Fraction(got), Fraction(expected)
    except (ValueError, ZeroDivisionError):
        return False
    if b == 0:
        return False
    return abs(float(a - b) / float(b)) <= 1e-6


# ---------------------------------------------------------------- result

@dataclass
class BroadcastResult:
    source: Dict[str, Any]
    interval: Dict[str, Any]
    config: Dict[str, Any]
    scenes: List[Dict[str, Any]]
    render: Dict[str, Any]
    verification: Dict[str, Any]
    artifacts: Dict[str, Any]
    limitations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)
    created_utc: str = ""

    def to_dict(self) -> dict:
        return {"schema_version": BROADCAST_ANALYSIS_SCHEMA,
                "created_utc": self.created_utc, "source": self.source,
                "interval": self.interval, "config": self.config,
                "scenes": self.scenes, "render": self.render,
                "verification": self.verification, "artifacts": self.artifacts,
                "limitations": self.limitations, "warnings": self.warnings,
                "notes": self.notes, "debug": self.debug}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def _atomic_move(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    shutil.move(src, tmp)
    os.replace(tmp, dst)


# ---------------------------------------------------------------- the run

def run_broadcast(clip: str, config_path: str, out_dir: str, *,
                  cv_python: Optional[str] = None, body_model_path: Optional[str] = None,
                  render: Optional[bool] = None, redact_path: bool = False,
                  force: bool = False, repo_root: Optional[str] = None,
                  probe_fn: Optional[Callable[[str], Any]] = None,
                  worker_runner: Optional[Callable[[dict, str], dict]] = None,
                  interpreter_resolver: Optional[Callable[[Optional[str]], dict]] = None,
                  guard_outputs: bool = True, saved_tracks: Optional[str] = None,
                  saved_tracks_sha256: Optional[str] = None,
                  ) -> BroadcastResult:
    """Run one broadcast showcase. Returns the result; raises on a failed run."""
    from .broadcast_config import load_broadcast
    from .probe import probe_source

    if saved_tracks_sha256 and not saved_tracks:
        raise DemoConfigError("--saved-tracks-sha256 requires --saved-tracks")

    src = Path(clip).expanduser()
    if not src.is_file():
        raise DemoSourceError(f"source video does not exist or is not a file: {clip}")
    config_file = Path(config_path).expanduser()
    if not config_file.is_file():
        raise DemoConfigError(f"config file does not exist: {config_path}")

    config: BroadcastConfig = load_broadcast(str(config_file))
    config_sha = sha256_of(str(config_file))
    source_sha = sha256_of(str(src))
    if config.expected_source_sha256 and config.expected_source_sha256 != source_sha:
        raise DemoSourceError(
            "the clip is not the file this config was written for: "
            f"shot.expected_source_sha256 is {config.expected_source_sha256}, the "
            f"clip's sha256 is {source_sha}. Refusing to render a config against the "
            f"wrong footage.")

    probe = (probe_fn or probe_source)(str(src))
    resolved = validate_broadcast(config, probe)

    root = (guard_output_dir(out_dir, repo_root=repo_root, force=force)
            if guard_outputs else Path(out_dir).expanduser())
    paths = broadcast_out_paths(root)

    from ghostcaddie.video import model_paths
    resolved_model = model_paths.resolve("body", body_model_path)

    render_enabled = True if render is None else bool(render)
    spec = build_broadcast_spec(resolved, out_dir=str(root), paths=paths,
                                body_model=resolved_model, redact_path=redact_path,
                                render_enabled=render_enabled)

    saved_provenance = None
    if saved_tracks:
        from .broadcast_saved import attach_saved_tracks
        saved_provenance = attach_saved_tracks(spec, resolved, saved_tracks, saved_tracks_sha256)
    root.mkdir(parents=True, exist_ok=True)
    paths["debug_dir"].mkdir(parents=True, exist_ok=True)
    paths["keyframes_dir"].mkdir(parents=True, exist_ok=True)

    interpreter = (interpreter_resolver or resolve_cv_interpreter)(cv_python)
    runner = worker_runner or run_demo_worker
    worker = runner(spec, interpreter["path"])

    render_info = dict(worker.get("render") or {})
    verification: Dict[str, Any] = {"render_ffprobe": {"checked": False,
                                                       "reason": "rendering off"}}
    if render_enabled:
        if not render_info.get("written"):
            verification["render_ffprobe"] = {
                "checked": False,
                "reason": render_info.get("reason") or "the worker wrote no video"}
        else:
            verification["render_ffprobe"] = verify_broadcast_render(
                str(paths["annotated"]), expected_frames=resolved.frame_count,
                expected_rational=spec["native_fps_rational"],
                source_path=str(src),
                source_audio=(spec["broadcast"] or {}).get("audio") or {})
            render_info["bytes"] = os.path.getsize(str(paths["annotated"]))
            verification["audio_mux"] = render_info.get("audio_verification") or {}
        render_info["style"] = "broadcast"
        render_info["frames"] = int(render_info.get("frames") or 0)

    scenes = list(worker.get("scenes") or [])
    card_rows = {row.get("scene_id"): row
                 for row in ((render_info.get("scenes") or []))}
    for scene in scenes:
        row = card_rows.get(scene.get("id"))
        if row:
            scene["cards"] = row
        scene["targets_origin"] = _tracking_origin_summary(scene.get("targets") or {}, saved=bool(saved_provenance))
    limitations = _limitations(resolved, worker, spec)
    artifacts = _artifact_record(paths, spec, worker)

    result = BroadcastResult(
        source={
            "display_name": probe.display_name,
            "path": (Path(probe.path).name if redact_path else probe.path),
            "sha256": source_sha,
            "width": probe.width, "height": probe.height,
            "fps": probe.fps, "fps_rational": spec["native_fps_rational"],
            "nominal_fps": probe.nominal_fps,
            "frame_count": probe.frame_count,
            "duration_seconds": probe.duration_seconds,
            "pixel_aspect_ratio": probe.pixel_aspect_ratio,
            "container_format": probe.container_format, "codec": probe.codec,
            "variable_cadence": probe.variable_cadence,
            "cadence_note": probe.cadence_note,
            "audio": (spec["broadcast"] or {}).get("audio") or {},
            "description": config.source_description,
        },
        interval={"start_frame": resolved.start_frame,
                  "end_frame": resolved.end_frame,
                  "frame_count": resolved.frame_count,
                  "seconds_source": {"start": resolved.start_frame / float(probe.fps),
                                     "end": resolved.end_frame / float(probe.fps)},
                  "seconds_note": ("seconds are derived for display only; every frame "
                                   "index in this run is an integer source frame")},
        config={"path": (Path(config_file).name if redact_path else str(config_file)),
                "sha256": config_sha,
                "schema_version": BROADCAST_CONFIG_SCHEMA,
                "out_dir": (str(root.name) if redact_path else str(root)),
                "outside_windows": config.outside_windows,
                "presentation": spec["broadcast"]["presentation"],
                "tracking_export": saved_provenance,
                "cut_candidates": [c.to_dict() for c in config.cuts],
                "cut_overrides": list(config.overrides),
                "anchor_spacing": list(config.spacing_records)},
        scenes=scenes, render=render_info, verification=verification,
        artifacts=artifacts, limitations=limitations,
        warnings=list(resolved.warnings), notes=[],
        debug={"scene_windows": (spec.get("broadcast") or {}).get("scene_windows"),
               "environment": worker.get("environment") or {},
               "spec_ref": "broadcast_debug/spec.json",
               "spec": _sanitized_spec(spec),
               "worker": "broadcast"},
        created_utc=_utcnow())

    payload = result.to_dict()
    _write_json(paths["analysis"], payload)
    _write_json(paths["config_copy"], _config_copy_payload(config, resolved, spec,
                                                           result))
    _write_json(paths["provenance"], _provenance(result, resolved, probe, spec,
                                                 config_sha, source_sha, redact_path,
                                                 config=config,
                                                 config_dir=config_file.parent))
    _write_debug_files(paths["debug_dir"], worker, spec)

    paths["report"].write_text(render_broadcast_report(result), encoding="utf-8")

    # The video-only intermediate is an implementation detail of the mux; it is
    # removed on success so a finished directory has one video in it, as documented.
    if render_info.get("muxed") and os.path.isfile(str(paths["video_only"])):
        os.remove(str(paths["video_only"]))

    if render_enabled and not (verification["render_ffprobe"].get("checked")
                               and verification["render_ffprobe"].get("ok")):
        raise DemoRenderError(
            "the broadcast render failed verification: "
            + (render_info.get("reason")
               or verification["render_ffprobe"].get("reason")
               or "; ".join(verification["render_ffprobe"].get("problems") or [])
               or json.dumps({k: v for k, v in render_info.items()
                              if k in ("written", "frames", "muxed", "mux_note")},
                             sort_keys=True))
            + ". The report, analysis and provenance were written with this failure "
              "stated; nothing is claimed about the video.")
    return result


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tracking_origin_summary(tracks: Mapping[str, Any], *, saved: bool = False) -> Dict[str, Any]:
    """Where each target's points came from, counted.

    ``absent`` is a real answer: a target with no seed and no anchor was never tracked,
    and saying so is the difference between "nothing was found" and "nothing was
    looked for".
    """
    out: Dict[str, Any] = {}
    for name in ("ball", "clubhead"):
        payload = tracks.get(name)
        if not payload:
            out[name] = {"origin": "absent",
                         "reason": ("no saved seed inside this scene window" if saved else
                                    "no seed, correction or manual anchor for this target in this scene")}
            continue
        counts: Dict[str, int] = {}
        for point in (payload.get("points") or []):
            status = str(point.get("status") or "unknown")
            counts[status] = counts.get(status, 0) + 1
        observed = sum(counts.get(k, 0) for k in
                       ("manual_anchor", "observed_template", "observed_template_lk"))
        out[name] = {
            "origin": ("saved_automatic_research" if payload.get("mode") == "saved_automatic_research" else
                       "operator_anchor" if counts.get("manual_anchor") else
                       "template_from_operator_seed" if observed else "absent"),
            "points_by_status": counts,
            "observed": observed,
            "total_points": len(payload.get("points") or []),
            "coverage": payload.get("coverage"),
            "is_a_detection": None if saved else False,
            "measurement_note": ("saved automatic research observations from a prior run; no model rerun"
                                 if saved else "operator placements or template matches"),
        }
    return out


def _repo_state() -> dict:
    """The repo commit and whether the tree is dirty, for provenance."""
    import subprocess as _sp

    out = {"commit": "", "dirty": None, "note": ""}
    try:
        commit = _sp.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                         capture_output=True, text=True, timeout=30)
        out["commit"] = (commit.stdout or "").strip()
        status = _sp.run(["git", "status", "--porcelain"], cwd=str(REPO_ROOT),
                         capture_output=True, text=True, timeout=30)
        out["dirty"] = bool((status.stdout or "").strip())
    except Exception as exc:  # pragma: no cover - git may be absent
        out["note"] = f"git state unavailable: {type(exc).__name__}"
    return out


#: Per-clip start/end field spellings that the manifests this cycle writes use for the
#: same two facts (both are frame indices in the PARENT source, not in the trimmed
#: clip). The baseline manifest spells them ``start_frame``/``end_frame_exclusive``;
#: a manifest of clips trimmed from a second source spells them
#: ``source_start_frame``/``source_end_frame_exclusive``. Both are read, and which one
#: was used is recorded, so neither manifest has to be rewritten to be usable.
_MANIFEST_START_KEYS = ("start_frame", "source_start_frame")
_MANIFEST_END_EXCLUSIVE_KEYS = ("end_frame_exclusive", "source_end_frame_exclusive")


def _manifest_candidates(clip_path: str, hint: str,
                         config_dir: Optional[Path]) -> List[Tuple[str, Path]]:
    """Where a parent-source record for this clip may live, in preference order.

    The operator's ``shot.parent_source_manifest`` wins when it is given. Otherwise
    the manifest is looked for **beside the clip**: ``<parent>/manifest.json`` (the
    baseline layout) and ``<parent>/<clip folder>_manifest.json`` (the layout of a
    second batch). Both come from the footage in front of us, so nothing about one
    operator's disk is baked into this source file, and a clip that has no manifest
    anywhere is reported as having none rather than quietly attributed to someone
    else's.
    """
    candidates: List[Tuple[str, Path]] = []
    if hint:
        path = Path(hint).expanduser()
        if not path.is_absolute() and config_dir is not None:
            path = Path(config_dir) / path
        candidates.append(("shot.parent_source_manifest", path))
    if clip_path:
        clip_dir = Path(clip_path).expanduser().absolute().parent
        candidates.append(("beside the clip", clip_dir / "manifest.json"))
        candidates.append(("beside the clip",
                           clip_dir.parent / "manifest.json"))
        candidates.append(("beside the clip",
                           clip_dir.parent / f"{clip_dir.name}_manifest.json"))
    seen: set = set()
    unique: List[Tuple[str, Path]] = []
    for label, path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append((label, path))
    return unique


def _first_present(entry: Mapping[str, Any],
                   names: Sequence[str]) -> Tuple[Any, str]:
    """The first of ``names`` that the entry actually carries, with the name used."""
    for name in names:
        if name in entry:
            return entry[name], name
    return None, ""


def _parent_source_record(clip_sha256: str, *, manifest_hint: str = "",
                          clip_path: str = "",
                          config_dir: Optional[Path] = None) -> dict:
    """The manifest's own record of this clip, when one is named or found beside it.

    This is where the parent (pre-trim) source digest and the native trim frames live.
    They are re-stated rather than re-derived, and an absence is reported as an
    absence: the run does not guess a manifest, and it does not assert a parent source
    it could not read.
    """
    out: Dict[str, Any] = {"manifest": "", "resolved_from": "", "found": False}
    text: Optional[str] = None
    tried: List[str] = []
    for label, path in _manifest_candidates(clip_path, manifest_hint, config_dir):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            tried.append(f"{path}: {type(exc).__name__}")
            continue
        out["manifest"] = str(path)
        out["resolved_from"] = label
        break
    if text is None:
        out["note"] = (
            "no parent-source manifest was named in shot.parent_source_manifest and "
            "none was found beside the clip, so the parent source digest and the "
            "native trim frames are not re-stated"
            + (f" (tried: {'; '.join(tried[:3])})" if tried else ""))
        out["tried"] = tried
        return out
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        out["note"] = f"the manifest at {out['manifest']} is not valid JSON: {exc}"
        return out
    if not isinstance(payload, Mapping):
        out["note"] = f"the manifest at {out['manifest']} is not a JSON object"
        return out
    out["sha256"] = _sha256_text(text)
    clips = payload.get("clips")
    if not isinstance(clips, (list, tuple)):
        out["note"] = f"the manifest at {out['manifest']} lists no clips"
        return out
    for entry in clips:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("clip_sha256")) != clip_sha256:
            continue
        start, start_key = _first_present(entry, _MANIFEST_START_KEYS)
        end, end_key = _first_present(entry, _MANIFEST_END_EXCLUSIVE_KEYS)
        out.update({
            "found": True, "clip_id": entry.get("id"), "label": entry.get("label"),
            "manifest_clip_path": entry.get("clip"),
            "native_trim_frames": {
                "start_frame": start, "end_frame_exclusive": end,
                "start_key": start_key, "end_key": end_key},
        })
        # The manifest's own account of the parent source is re-stated ONLY here, beside
        # a matched clip: a parent digest next to a clip the manifest does not list
        # would read as this clip's parent source, which is exactly the claim a lookup
        # that found nothing must not make.
        out["parent_source"] = payload.get("source")
        out["parent_source_sha256"] = payload.get("source_sha256")
        out["trim_method"] = payload.get("trim_method")
        out["body_model_sha256"] = payload.get("body_model_sha256")
        if "body_state" in entry:
            out["baseline_body_state"] = entry["body_state"]
        if "golfer_window_end" in entry:
            out["golfer_window_end"] = entry["golfer_window_end"]
        break
    else:
        out["note"] = (f"this clip's sha256 is not among the "
                       f"{len(clips)} clip entries in the manifest at "
                       f"{out['manifest']}, so nothing is re-stated from it")
    return out


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    """SHA-256 over canonical JSON: sorted keys, no whitespace variance."""
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                   default=str))


def _artifact_record(paths: Dict[str, Any], spec: dict, worker: dict) -> dict:
    out = {"output_dir": str(paths["root"]),
           "annotated": str(paths["annotated"]),
           "report": str(paths["report"]),
           "analysis": str(paths["analysis"]),
           "provenance": str(paths["provenance"]),
           "config_copy": str(paths["config_copy"]),
           "keyframes_dir": str(paths["keyframes_dir"]),
           "debug_dir": str(paths["debug_dir"]),
           "still_count": int((worker.get("keyframes") or {}).get("count", 0)),
           "contact_sheet": str(paths["contact_sheet"])}
    return out


def _config_copy_payload(config: BroadcastConfig, resolved: ResolvedBroadcast,
                         spec: dict, result: BroadcastResult) -> dict:
    payload = config.to_dict()
    payload["_resolved"] = {
        "created_utc": result.created_utc,
        "tool_version": DEMO_TOOL_VERSION,
        "schema_version": BROADCAST_CONFIG_SCHEMA,
        "source": {"path": resolved.source.path,
                   "sha256": sha256_of(resolved.source.path),
                   "fps": resolved.source.fps,
                   "fps_rational": spec["native_fps_rational"],
                   "width": resolved.source.width,
                   "height": resolved.source.height},
        "frame_bounds": {
            "interval": {"start_frame": resolved.start_frame,
                         "end_frame": resolved.end_frame,
                         "end_inclusive": True},
            "scenes": [{"id": s.scene.scene_id, "start_frame": s.start_frame,
                        "end_frame": s.end_frame, "end_inclusive": True}
                       for s in resolved.scenes]},
        "window_hashes": [
            {"id": s.scene.scene_id,
             "window_sha256": _canonical_hash({"clip_sha256": sha256_of(
                 resolved.source.path), "start_frame": s.start_frame,
                 "end_frame": s.end_frame})}
            for s in resolved.scenes],
        "note": ("keys beginning with an underscore are ignored by the schema, so this "
                 "file can be passed straight back to broadcast-showcase run"),
    }
    return payload


def _provenance(result: BroadcastResult, resolved: ResolvedBroadcast, probe,
                spec: dict, config_sha: str, source_sha: str,
                redact_path: bool, config: Optional[BroadcastConfig] = None,
                config_dir: Optional[Path] = None) -> dict:
    interp = spec.get("body", {})
    parent = _parent_source_record(
        source_sha,
        manifest_hint=(config.parent_source_manifest if config else ""),
        clip_path=getattr(probe, "path", "") or "",
        config_dir=config_dir)
    scene_hashes = [
        {"id": s.scene.scene_id,
         "window_sha256": _canonical_hash({"clip_sha256": source_sha,
                                           "start_frame": s.start_frame,
                                           "end_frame": s.end_frame})}
        for s in resolved.scenes]
    return {
        "schema_version": BROADCAST_PROVENANCE_SCHEMA,
        "created_utc": result.created_utc,
        "tool_version": DEMO_TOOL_VERSION,
        "repo": _repo_state(),
        "source": {"path": (Path(probe.path).name if redact_path else probe.path),
                   "sha256": source_sha,
                   "fps_rational": spec.get("native_fps_rational"),
                   "frame_count": probe.frame_count,
                   "width": probe.width, "height": probe.height,
                   "variable_cadence": probe.variable_cadence,
                   "irregular_timestamps": bool(
                       getattr(probe, "irregular_timestamps", False)),
                   "timestamp_note": getattr(probe, "timestamp_note", "")},
        # The pre-trim parent source and the native trim frames are re-stated from the
        # manifest that was named (or found beside the clip), never re-derived.
        "parent_source": parent,
        "shot": {
            "id": (config.shot_id if config else ""),
            "label": (config.shot_label if config else ""),
            "stroke_type": (config.stroke_type if config else ""),
            "expected_source_sha256":
                (config.expected_source_sha256 if config else ""),
            "expected_sha_matches": bool(
                config and config.expected_source_sha256
                and config.expected_source_sha256 == source_sha),
            "parent_source_manifest":
                (config.parent_source_manifest if config else ""),
            "source_notes": (config.source_notes if config else ""),
            "expected_fps": (config.expected_fps if config else None),
            "expected_frames": (config.expected_frames if config else None),
            "note": ("the shot facts above are the operator's own declarations, "
                     "recorded verbatim; only expected_sha_matches is a check this run "
                     "performed"),
        },
        "config": {"sha256": config_sha},
        "saved_tracking": spec.get("saved_tracking"),
        "frame_bounds": {
            "interval": [resolved.start_frame, resolved.end_frame],
            "end_inclusive": True,
            "semantics": ("zero-based frames of the clip named above; both ends of a "
                          "scene window are inclusive; frames are never round-tripped "
                          "through seconds"),
            "scenes": [[s.scene.scene_id, s.start_frame, s.end_frame]
                       for s in resolved.scenes],
            "window_hashes": scene_hashes},
        "body_model": {"path": (spec.get("body") or {}).get("model_path", ""),
                       "found": (spec.get("body") or {}).get("model_found"),
                       "sha256_baseline": parent.get("body_model_sha256")},
        "render": {"output_to_source_frames": "identity",
                   "stride": 1, "holds": [],
                   "fps_rational": spec.get("native_fps_rational"),
                   "audio_mode": (spec.get("broadcast") or {}).get("audio", {}).get(
                       "mode"),
                   "verification": result.verification},
        "targets_origin": {scene.get("id"): scene.get("targets_origin")
                           for scene in result.scenes},
        "operator_inputs": {"used_for_target_tracking": not bool(spec.get("saved_tracking")),
            "scenes": [{"id": s.scene.scene_id, "kind": s.scene.kind,
                        "stroke_type": s.scene.stroke_type,
                        "window": [s.start_frame, s.end_frame],
                        "keyframes": dict(s.scene.keyframes),
                        "phase_vocabulary": list(s.scene.phase_vocabulary),
                        "roi": list(s.scene.demo.roi) if s.scene.demo.roi else None,
                        "roi_keyframes": [k.to_dict()
                                          for k in s.scene.demo.roi_keyframes],
                        "seeds": {k: v.to_dict()
                                  for k, v in s.scene.demo.seeds.items()},
                        "corrections": {k: [p.to_dict() for p in v] for k, v in
                                        s.scene.demo.corrections.items()}}
                       for s in resolved.scenes],
            "anchor_spacing": list(resolved.config.spacing_records),
            "cut_decisions": _cut_decisions(resolved.config),
            "note": ("every ROI, seed, anchor and phase mark above is an operator "
                     "input. None of them is a detection.")},
        "verification": result.verification,
        "environment": "recorded in broadcast_debug/environment.json",
    }


def _cut_decisions(config: BroadcastConfig) -> dict:
    """Cut decisions with who decided, and an honest note when nobody is named."""
    decided = []
    unnamed = 0
    for cut in config.cuts:
        if cut.decision in ("proposed", ""):
            continue
        if not cut.decided_by:
            unnamed += 1
        decided.append({"frame": cut.frame, "kind": cut.kind,
                        "decision": cut.decision,
                        "decided_by": cut.decided_by or "not recorded in the config",
                        "decided_at": cut.decided_at, "note": cut.note,
                        "method": cut.method})
    for item in config.overrides:
        if not item.get("decided_by"):
            unnamed += 1
        decided.append({"frame": item.get("frame"),
                        "decision": item.get("decision"),
                        "decided_by": item.get("decided_by")
                        or "not recorded in the config",
                        "decided_at": item.get("decided_at", ""),
                        "note": item.get("note", ""), "method": "operator override"})
    return {"decisions": decided, "unnamed_decisions": unnamed}


def _sanitized_spec(spec: dict) -> dict:
    """The worker spec with each scene's phase marks rewritten in its own vocabulary.

    The worker is *instructed* in canonical slots (that is how the existing phase
    resolver is reused), but the instruction is also evidence a reviewer might read,
    so it is published in the same words the operator used. A putt's spec therefore
    carries ``contact`` and not ``impact``, and the translation is lossless because
    the slot mapping is a fixed function of the stroke type.
    """
    import copy

    out = copy.deepcopy(spec)
    broadcast = out.get("broadcast") or {}
    for entry in (broadcast.get("scenes") or []):
        scene_spec = entry.get("spec")
        if not scene_spec:
            continue
        phases = scene_spec.get("phases") or {}
        phases["keyframes"] = dict(entry.get("keyframes") or {})
        phases["keyframe_semantics"] = ("operator marks in this scene's own vocabulary; "
                                        "the analytics map them onto their internal slots")
        scene_spec["phases"] = phases
    return out


def _write_debug_files(debug_dir: Path, worker: dict, spec: dict) -> None:
    _write_json(debug_dir / "environment.json", worker.get("environment") or {})
    _write_json(debug_dir / "spec.json", _sanitized_spec(spec))
    windows = (spec.get("broadcast") or {}).get("scene_windows") or []
    _write_json(debug_dir / "scene_windows.json", {"scenes": windows})
    per_scene = []
    for scene in (worker.get("scenes") or []):
        per_scene.append({
            "id": scene.get("id"), "kind": scene.get("kind"),
            "stroke_type": scene.get("stroke_type"),
            "window": scene.get("window"),
            "body_state": (scene.get("body") or {}).get("state"),
            "body_coverage": (scene.get("body") or {}).get("coverage"),
            "targets": {name: (payload or {}).get("coverage")
                        for name, payload in (scene.get("targets") or {}).items()},
            "phases": {name: {"frame": (info or {}).get("frame"),
                              "label": (info or {}).get("label")}
                       for name, info in
                       ((scene.get("phases") or {}).get("phases") or {}).items()},
        })
    _write_json(debug_dir / "scenes.json", {"scenes": per_scene})


def _limitations(resolved: ResolvedBroadcast, worker: dict, spec: dict) -> List[str]:
    from .result import DISCLOSURE_LONG

    out = list(DISCLOSURE_LONG)
    if spec.get("saved_tracking"):
        out[0] = ("Targets are saved automatic research observations from a prior operator-seeded run. "
                  "The source region and phase inputs remain operator-assisted. Importing these "
                  "positions does not establish physical identity or general tracking accuracy.")
    if spec.get("saved_tracking"):
        out.append("Ball and clubhead markers use saved automatic research observations from a prior run "
                   "and its original operator seed; no model rerun or accuracy claim. Gaps stay empty; "
                   "the imported positions are not verified ball-flight measurements.")
    else:
        out.append(
            "The ball and clubhead markers are operator-anchored or seeded image-space "
            "points. They are not detections, and no broadcast tracer graphic was used as "
            "one.")
    render = worker.get("render") or {}
    if render.get("empty_frames"):
        out.append(
            f"{render['empty_frames']} frames lie outside every scene window and carry "
            f"only the source picture"
            + (" and the brand mark." if resolved.config.outside_windows == "brand_only"
               else "."))
    if render.get("muxed"):
        out.append("The clip's own audio stream was copied bit-exactly; the overlay "
                   "render changes no audio.")
    else:
        out.append("This render carries no audio stream (the source had none, or "
                   "muxing was switched off); that is recorded, not hidden.")
    return out


# ---------------------------------------------------------------- report

def render_broadcast_report(result: BroadcastResult) -> str:
    """The combined per-clip report: one row per scene, plus what was checked."""
    from .showcase_report import BANNED_CLAIMS

    src = result.source
    lines: List[str] = []
    lines.append("# Broadcast showcase report")
    lines.append("")
    lines.append(f"Created {result.created_utc} · tool {DEMO_TOOL_VERSION} · "
                 f"schema `{BROADCAST_CONFIG_SCHEMA}`")
    lines.append("")
    lines.append("## What this is")
    lines.append("")
    lines.append("One whole shot, rendered end to end at the source's own cadence with "
                 "the source's own audio. Overlays and metric cards exist **only** "
                 "inside the scene windows below; every other frame is the source "
                 "picture. At a window boundary everything is cleared: body, trails, "
                 "cards, phase badge and the `hand_rise` reference frame.")
    lines.append("")
    saved = result.config.get("tracking_export")
    if saved:
        lines.append("Saved automatic research positions, with the original operator seed. "
                     "Missing/lost frames stay empty. Assisted paths are excluded; body and phase "
                     "configuration remain operator-assisted. No model rerun or accuracy claim.")
        lines.append(f"Prediction sha256: `{saved['prediction_sha256']}` · variant `{saved['variant']}`")
        lines.append("")
    lines.append("## Source")
    lines.append("")
    lines.append(f"* file: `{src['display_name']}`")
    lines.append(f"* sha256: `{src['sha256']}`")
    lines.append(f"* {src['width']}x{src['height']}, {src['fps_rational']} fps "
                 f"({src['fps']:.6f}), {src['frame_count']} frames")
    lines.append(f"* audio: "
                 + (f"{src['audio'].get('codec_name')} "
                    f"{src['audio'].get('sample_rate')}Hz "
                    f"{src['audio'].get('channels')}ch"
                    if src['audio'].get('present') else "none"))
    if src.get("description"):
        lines.append(f"* operator's description: {src['description']}")
    lines.append("")
    lines.append("## Playback contract")
    lines.append("")
    render = result.render
    lines.append(f"* output `{render.get('wh')}`, cadence `{render.get('fps_rational')}`"
                 f", {render.get('frames')} frames")
    lines.append("* frame *i* of the output is frame *i* of the source (identity "
                 "mapping); no stride, no hold, no stretch, no crop.")
    lines.append(f"* frames inside a window: {render.get('overlay_frames')}; frames "
                 f"outside every window: {render.get('empty_frames')}")
    lines.append(f"* audio: {render.get('mux_note') or 'not muxed'}")
    verification = result.verification.get("render_ffprobe") or {}
    lines.append(f"* verification: {'PASS' if verification.get('ok') else 'FAIL'} "
                 + ("; ".join(verification.get("problems") or []) or
                    verification.get("reason") or ""))
    stamps = verification.get("timestamps") or {}
    if stamps.get("checked"):
        lines.append(
            f"* frame timeline: all {stamps.get('compared_frames')} decoded frames of "
            f"the output compared against the source's own presentation timestamps "
            f"(source {stamps.get('source_frames')}, output {stamps.get('output_frames')} "
            f"frames); largest difference {stamps.get('max_delta_seconds')}s, half a "
            f"frame interval being {stamps.get('half_frame_interval')}s")
    elif stamps:
        lines.append(f"* frame timeline: NOT compared — "
                     f"{stamps.get('reason') or 'no usable frame timestamps'}")
    durations = verification.get("duration") or {}
    if durations.get("checked"):
        source_seconds = durations.get("source_seconds") or {}
        source_measured = (source_seconds.get("video")
                           if source_seconds.get("video") is not None
                           else source_seconds.get("container"))
        lines.append(
            f"* duration: output {durations.get('measured_seconds')}s vs source "
            f"{source_measured}s, expected "
            f"{durations.get('expected_seconds_from_frames_and_cadence')}s from the frame "
            f"count and the source's rational cadence (tolerance "
            f"{durations.get('tolerance_seconds')}s)")
    elif durations:
        lines.append(f"* duration: NOT checked — "
                     f"{durations.get('reason') or 'no readable duration'}")
    lines.append("")
    lines.append("## Scenes")
    lines.append("")
    lines.append("| id | kind | stroke | window (inclusive) | frames | body | phases "
                 "marked | targets |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for scene in result.scenes:
        window = scene.get("window") or {}
        body = (scene.get("body") or {}).get("state") or "not measured"
        coverage = (scene.get("body") or {}).get("coverage") or {}
        if coverage:
            body = f"{body} ({coverage.get('frames_with_pose')}/"\
                   f"{coverage.get('frames_total')} poses)"
        phases = (scene.get("phases") or {}).get("phases") or {}
        marked = ", ".join(name for name, info in phases.items()
                           if (info or {}).get("frame") is not None) or "none"
        targets = ", ".join(f"{name}" for name in (scene.get("targets") or {})) or "none"
        lines.append(
            f"| `{scene.get('id')}` | {scene.get('kind')} | "
            f"{scene.get('stroke_type')} | "
            f"{window.get('start_frame')}..{window.get('end_frame')} | "
            f"{window.get('frame_count')} | {body} | {marked} | {targets} |")
    lines.append("")
    for scene in result.scenes:
        lines.append(f"### Scene `{scene.get('id')}`")
        lines.append("")
        phases = (scene.get("phases") or {}).get("phases") or {}
        for name, info in phases.items():
            if (info or {}).get("frame") is None:
                continue
            lines.append(f"* **{name}** frame {info['frame']} "
                         f"({info.get('label')}, {info.get('method')})")
        tempo = (scene.get("phases") or {}).get("tempo") or {}
        if tempo.get("state") == "supported":
            lines.append(f"* tempo: {tempo.get('label', 'tempo')} "
                         f"{tempo.get('backswing_frames')}f back : "
                         f"{tempo.get('downswing_frames')}f through "
                         f"({tempo.get('backswing_downswing_ratio')}:1, source frames)")
        elif tempo:
            lines.append(f"* tempo: unavailable — {tempo.get('reason', '')}")
        cards = scene.get("cards") or {}
        if cards:
            lines.append(f"* cards: {', '.join(cards.get('metrics') or []) or 'none'}"
                         + (f"; omitted {', '.join(cards.get('omitted_metrics') or [])}"
                            if cards.get("omitted_metrics") else ""))
            held = int(cards.get("held_value_frames") or 0)
            if held:
                lines.append(
                    f"* card stability: a card held its previous measured value on "
                    f"{held} frame(s) across a gap, disclosed with the debounce bound "
                    f"of {cards.get('card_debounce_frames')} frames. A held value is "
                    f"marked as held on the video and is NEVER drawn on a still.")
            if not cards.get("card_row"):
                lines.append(f"* card row omitted: "
                             f"{cards.get('row_reason') or 'no card'}")
            else:
                lines.append(f"* card row position: {cards.get('card_position')} — "
                             f"{cards.get('card_position_reason')}")
        origin = scene.get("targets_origin") or {}
        for name, info in origin.items():
            lines.append(f"* {name}: origin `{info.get('origin')}`"
                         + (f", {info.get('observed')} observed of "
                            f"{info.get('total_points')}"
                            if info.get("observed") is not None else "")
                         + (f" — {info.get('reason')}" if info.get("reason") else ""))
        lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for item in result.limitations:
        lines.append(f"* {item}")
    lines.append("")
    lines.append("## Claims this report does not make")
    lines.append("")
    for claim in BANNED_CLAIMS:
        lines.append(f"* {claim}")
    lines.append("")
    return "\n".join(lines)
