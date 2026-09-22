"""Orchestration of one demo run: config -> worker -> verification -> artifacts.

This module is stdlib-only, exactly like the canonical orchestrator, and for the
same reason: the whole output contract (config validation, statuses, analytics,
report wording, render contract) must be testable without cv2 or a model file.
Decoding, pose inference, tracking and rendering happen in ``demo.worker``, which
is dispatched into an interpreter that has those dependencies.

Failures are stated, never smoothed over:

* a source that cannot be probed, hashed or bound -> the run stops before any
  model or decoder is touched;
* a config that is out of bounds, fractional, non-unique or contradictory -> the
  run stops, and nothing is clipped or repaired;
* a render that fails, or an output whose decoded properties do not match what was
  promised (H.264, yuv420p, faststart, native cadence, expected frame count) ->
  the artifacts are still written so the reason is readable, and the run exits as
  a failure.

Two verifications run on every successful worker result: phase resolution is
re-derived here from the worker's own records and compared (a mismatch is an
internal error, not a warning), and the rendered video is re-probed with ffprobe.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING)

from . import DEMO_ANALYSIS_SCHEMA, DEMO_PROVENANCE_SCHEMA, DEMO_TOOL_VERSION
from .analytics import (body_metrics, missingness_report, targets_summary)
from .config import (PHASE_ORDER, ResolvedConfig, load_config,
                     roi_envelope_for, validate_config)
from .errors import (DemoConfigError, DemoError, DemoRenderError, DemoSourceError)
from .paths import (candidate_interpreters, out_paths, resolve_cv_interpreter,
                    resolve_out_paths, run_demo_worker, showcase_still_path,
                    sha256_of)
from .phases import resolve_phases
from .probe import SourceProbe, probe_source
from .report import (render_report, render_summary, render_terminal_summary)
from .result import DISCLOSURE_LONG, DemoResult
from .tracking import resolve_track

if TYPE_CHECKING:  # pragma: no cover
    from .tracking import TargetTrack

MAX_ATOM_SCAN = 1 << 22


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------
# output verification

def _atom_order(path: str) -> List[str]:
    """Top-level MP4 atom types in file order, so faststart can be checked."""
    order: List[str] = []
    try:
        size = os.path.getsize(path)
    except OSError:
        return order
    with open(path, "rb") as fh:
        offset = 0
        while offset < min(size, MAX_ATOM_SCAN):
            fh.seek(offset)
            header = fh.read(8)
            if len(header) < 8:
                break
            atom_size = int.from_bytes(header[:4], "big")
            atom_type = header[4:8].decode("ascii", "replace")
            order.append(atom_type)
            if atom_size == 1:
                extended = fh.read(8)
                if len(extended) < 8:
                    break
                atom_size = int.from_bytes(extended, "big")
            if atom_size <= 0:
                break
            offset += atom_size
    return order


def verify_render(path: str, *, expected_frames: int, expected_fps: float,
                  expected_wh: Sequence[int]) -> dict:
    """ffprobe the rendered file and compare it with what was promised."""
    import subprocess
    result: Dict[str, Any] = {"path": path, "checked": True, "ok": True,
                              "problems": []}
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        result.update(ok=False, problems=["rendered file missing or empty"])
        return result
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
           "-show_format", path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        result.update(ok=False, problems=[f"ffprobe could not verify the render: {exc}"])
        return result
    if proc.returncode != 0:
        result.update(ok=False,
                      problems=[f"ffprobe rejected the rendered file: "
                                f"{(proc.stderr or '').strip()[:200]}"])
        return result
    payload = json.loads(proc.stdout)
    stream = next((s for s in payload.get("streams") or []
                   if s.get("codec_type") == "video"), None)
    if stream is None:
        result.update(ok=False, problems=["rendered file has no video stream"])
        return result
    fps_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    try:
        num, _, den = fps_text.partition("/")
        actual_fps = float(num) / float(den or 1)
    except (TypeError, ValueError, ZeroDivisionError):
        actual_fps = 0.0
    frames = stream.get("nb_frames")
    try:
        frame_count = int(str(frames))
    except (TypeError, ValueError):
        frame_count = None
    result.update(
        codec=stream.get("codec_name"), pix_fmt=stream.get("pix_fmt"),
        width=stream.get("width"), height=stream.get("height"),
        fps=round(actual_fps, 4), frames=frame_count,
        duration=payload.get("format", {}).get("duration"),
        atoms=_atom_order(path))
    problems = result["problems"]
    if stream.get("codec_name") != "h264":
        problems.append(f"codec is {stream.get('codec_name')}, not h264")
    if stream.get("pix_fmt") != "yuv420p":
        problems.append(f"pixel format is {stream.get('pix_fmt')}, not yuv420p")
    if abs(actual_fps - expected_fps) > 0.01:
        problems.append(f"playback cadence is {actual_fps:g} fps, expected "
                        f"{expected_fps:g} fps")
    if frame_count is not None and frame_count != expected_frames:
        problems.append(f"the rendered file holds {frame_count} frames, expected "
                        f"{expected_frames}")
    if [int(stream.get("width") or 0), int(stream.get("height") or 0)] != \
            [int(v) for v in expected_wh]:
        problems.append(f"rendered size is {stream.get('width')}x"
                        f"{stream.get('height')}, expected "
                        f"{expected_wh[0]}x{expected_wh[1]}")
    atoms = result["atoms"]
    if "moov" in atoms and "mdat" in atoms:
        result["faststart"] = atoms.index("moov") < atoms.index("mdat")
        if not result["faststart"]:
            problems.append("moov atom is not before mdat (not faststart)")
    else:
        result["faststart"] = False
        problems.append("could not locate both moov and mdat atoms")
    result["ok"] = not problems
    return result


# --------------------------------------------------------------------------
# the run

def build_spec(resolved: ResolvedConfig, source_sha256: str, out_dir: str,
               paths: Dict[str, Path], *, body_model, render_enabled: bool,
               redact_path: bool, showcase_enabled: bool = False) -> dict:
    cfg = resolved.config
    return {
        "video": resolved.source.path,
        "source_sha256": source_sha256,
        "native_fps": float(resolved.source.fps),
        "native_wh": [resolved.source.width, resolved.source.height],
        "pixel_aspect": float(resolved.source.pixel_aspect_ratio),
        "handedness": cfg.handedness,
        "interval": {"start_frame": resolved.start_frame,
                     "end_frame": resolved.end_frame},
        "roi": list(cfg.roi) if cfg.roi else None,
        # The moving crop: the boxes themselves, the region they move through (used
        # for coverage measurement) and the operator's own note about who placed them.
        "roi_keyframes": [k.to_dict() for k in cfg.roi_keyframes],
        "roi_effective": roi_envelope_for(cfg),
        "roi_author": (cfg.roi_keyframes[0].note if cfg.roi_keyframes else ""),
        "seeds": {name: point.to_dict() for name, point in cfg.seeds.items()},
        "corrections": {name: [p.to_dict() for p in points]
                        for name, points in cfg.corrections.items()},
        # Opt-in operator-anchored presentation segments. Omitted blocks mean the
        # classic behaviour, so an older config behaves exactly as it did.
        "manual_segments": {name: block.to_dict()
                            for name, block in cfg.manual_segments.items()},
        "targets": {name: {"options": options.to_dict(), "mode": options.mode}
                    for name, options in cfg.targets.items()},
        "interpolation": cfg.interpolation.to_dict(),
        "phases": {"keyframes": dict(cfg.keyframes),
                   "allow_approx": bool(cfg.allow_approx_phases),
                   "fps": float(resolved.source.fps),
                   "frame_times": list(resolved.frame_times)
                   if resolved.frame_times else None,
                   "start_frame": int(resolved.start_frame)},
        "body": {"model_path": getattr(body_model, "path", ""),
                 "model_found": bool(getattr(body_model, "found", False))},
        "render": {"enabled": bool(render_enabled), "path": str(paths["annotated"]),
                   "max_width": cfg.render.max_width, "crf": cfg.render.crf,
                   "preset": cfg.render.preset,
                   # The presentation style is decided here, once, so the worker draws
                   # the style the orchestrator wrote the output contract for.
                   "style": ("showcase" if showcase_enabled else "demo"),
                   "stride": int(cfg.render.stride)},
        "showcase": cfg.showcase.to_dict() if showcase_enabled else None,
        "showcase_enabled": bool(showcase_enabled),
        "keyframes_dir": str(paths["keyframes_dir"]),
        "contact_sheet": str(paths["contact_sheet"]),
        "debug_dir": str(paths["debug_dir"]),
        "redact_path": bool(redact_path),
        "tool_version": DEMO_TOOL_VERSION,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def _redact(path: str, redact: bool) -> str:
    return Path(path).name if redact else path


def run_demo(video: str, config_path: str, out_dir: str, *,
             cv_python: Optional[str] = None, body_model_path: Optional[str] = None,
             render: Optional[bool] = None, summary: bool = True,
             redact_path: bool = False, showcase: Optional[bool] = None,
             probe_fn: Optional[Callable[[str], SourceProbe]] = None,
             worker_runner: Optional[Callable[[dict, str], dict]] = None,
             interpreter_resolver: Optional[Callable[[Optional[str]], dict]] = None
             ) -> DemoResult:
    """Run the demo on one local clip. Returns the result; raises on any failure.

    ``showcase`` is a tri-state: ``None`` follows the config's ``showcase.enabled``,
    ``True`` forces the showcase presentation on (with the block's defaults when the
    config has none), ``False`` forces the demo renderer.
    """
    from ghostcaddie.video.source_binding import bind_source, sha256_file

    src = Path(video).expanduser()
    if not src.is_file():
        raise DemoSourceError(f"source video does not exist or is not a file: {video}")
    if not os.access(src, os.R_OK):
        raise DemoSourceError(f"source video is not readable: {video}")

    config_file = Path(config_path).expanduser()
    if not config_file.is_file():
        raise DemoConfigError(f"config file does not exist: {config_path}")
    config = load_config(str(config_file))
    cfg = config
    config_sha = sha256_file(config_file)

    showcase_enabled = (bool(cfg.showcase.enabled) if showcase is None
                        else bool(showcase))

    probe = (probe_fn or probe_source)(str(src))
    resolved = validate_config(config, probe)

    source_sha = sha256_file(src)
    binding = bind_source(src, source_sha)

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    paths = resolve_out_paths(out, showcase=showcase_enabled)
    paths["debug_dir"].mkdir(parents=True, exist_ok=True)
    paths["keyframes_dir"].mkdir(parents=True, exist_ok=True)

    from ghostcaddie.video import model_paths
    resolved_model = model_paths.resolve("body", body_model_path)

    render_enabled = config.render.render if render is None else bool(render)
    spec = build_spec(resolved, source_sha, str(out), paths, body_model=resolved_model,
                      render_enabled=render_enabled, redact_path=redact_path,
                      showcase_enabled=showcase_enabled)

    interpreter = (interpreter_resolver or resolve_cv_interpreter)(cv_python)
    runner = worker_runner or run_demo_worker
    worker = runner(spec, interpreter["path"])

    # --- derived analytics (pure, stdlib)
    body_records = worker.get("body", {}).get("per_frame") or []
    frames_possible = resolved.frame_count
    tracks: Dict[str, "TargetTrack"] = {}
    for name in ("ball", "clubhead"):
        payload = (worker.get("targets") or {}).get(name)
        if payload is None:
            continue
        tracks[name] = _track_from_payload(name, payload, resolved)

    body_2d = body_metrics(body_records, frames_possible=frames_possible,
                           pixel_aspect=probe.pixel_aspect_ratio,
                           handedness=config.handedness)
    target_summary = targets_summary(tracks, frames_possible=frames_possible)
    missingness = missingness_report(body_records, tracks)

    phase_payload = _resolve_phases_for_result(worker, resolved, tracks, body_records)
    consistency = _phase_consistency(worker.get("phases"), phase_payload)
    if consistency.get("checked") and not consistency.get("consistent"):
        raise DemoError(
            "internal inconsistency: the phase resolution re-derived by the "
            "orchestrator differs from the one the worker rendered with ("
            + "; ".join(consistency.get("differences") or [])
            + "); refusing to publish a report that may disagree with the video")

    render_info = dict(worker.get("render") or {})
    keyframes_info = dict(worker.get("keyframes") or {})
    verification: Dict[str, Any] = {"render_ffprobe": {"checked": False,
                                                       "reason": "rendering was not requested"}}
    if render_enabled:
        if not render_info.get("written"):
            verification["render_ffprobe"] = {
                "checked": False,
                "reason": render_info.get("reason", "the worker did not write a video")}
        else:
            verification["render_ffprobe"] = verify_render(
                str(paths["annotated"]), expected_frames=int(render_info.get("frames") or 0),
                expected_fps=float(probe.fps),
                expected_wh=[int(v) for v in render_info.get("wh") or [0, 0]])
            render_info["pix_fmt"] = verification["render_ffprobe"].get("pix_fmt") or ""
            render_info["faststart"] = bool(verification["render_ffprobe"].get("faststart"))
            render_info["codec"] = verification["render_ffprobe"].get("codec") or ""
            render_info["duration"] = verification["render_ffprobe"].get("duration")
            render_info["bytes"] = os.path.getsize(paths["annotated"])

    limitations = _limitations(resolved, worker, target_summary, body_2d,
                               verification)

    result = DemoResult(
        source={
            "display_name": probe.display_name,
            "path": _redact(probe.path, redact_path),
            "sha256": source_sha,
            "width": probe.width, "height": probe.height,
            "fps": probe.fps, "nominal_fps": probe.nominal_fps,
            "duration_seconds": probe.duration_seconds,
            "frame_count": probe.frame_count,
            "pixel_aspect_ratio": probe.pixel_aspect_ratio,
            "container_format": probe.container_format, "codec": probe.codec,
            "variable_cadence": probe.variable_cadence,
            "cadence_note": probe.cadence_note,
            "timestamp_note": probe.timestamp_note,
            "irregular_timestamps": probe.irregular_timestamps,
            "timestamp_stats": probe.timestamp_stats,
            "timing_caveat": cfg.timing.source_cadence_caveat,
            # The operator's own description of what the clip is. Recorded rather than
            # generated: the tool cannot identify a player or a rights status.
            "description": cfg.source_description,
        },
        interval={
            "start_frame": resolved.start_frame, "end_frame": resolved.end_frame,
            "frame_count": resolved.frame_count,
            "start_seconds": cfg.start_seconds, "end_seconds": cfg.end_seconds,
            "endpoint_semantics": resolved.effective()["endpoint_semantics"],
            "seconds_basis": resolved.seconds_basis,
            "source_timestamps_available": resolved.frame_times is not None,
        },
        config={
            "path": _redact(str(config_file), redact_path),
            "sha256": config_sha,
            "out_dir": _redact(str(out), redact_path),
            "roi_xywh": list(cfg.roi) if cfg.roi else None,
            "seeds": {k: v.to_dict() for k, v in cfg.seeds.items()},
            "corrections": {k: [p.to_dict() for p in v]
                            for k, v in cfg.corrections.items()},
            "keyframes": dict(cfg.keyframes),
            "interpolation": config.interpolation.to_dict(),
            "timing": cfg.timing.to_dict(),
            "render": cfg.render.to_dict(),
            "handedness": cfg.handedness,
            "allow_approx_phases": cfg.allow_approx_phases,
            "effective": resolved.effective(),
            "seconds_basis": resolved.seconds_basis,
        },
        body={
            "state": worker.get("body", {}).get("state", "unavailable"),
            "method": worker.get("body", {}).get("method", "none"),
            "coverage": {
                "frames_with_pose": sum(1 for r in body_records if r.get("available")),
                "frames_total": len(body_records),
            },
            "identity": worker.get("body", {}).get("identity", {}),
            "roi_xywh": worker.get("body", {}).get("roi_xywh"),
            "model": resolved_model.to_dict(),
            "interpreter": interpreter["path"],
            "limitations": list(worker.get("body", {}).get("limitations") or []),
            "per_frame": body_records,
        },
        analytics={"body_2d": body_2d, "targets": target_summary,
                   "missingness": missingness},
        phases=phase_payload,
        targets={name: payload for name, payload in
                 (worker.get("targets") or {}).items()},
        render=render_info,
        keyframes=keyframes_info,
        presentation=_presentation_record(paths, cfg, render_info, keyframes_info,
                                        showcase_enabled),
        limitations=limitations,
        warnings=list(resolved.warnings),
        notes=list(resolved.notes),
        debug={
            "raw_measurements": worker.get("raw_measurements") or {},
            "environment": worker.get("environment") or {},
            "spec": spec,
            "verification": verification,
            "fps_source": "average frame rate reported by ffprobe",
            "phase_resolution_consistent": bool(consistency.get("consistent")),
            "phase_resolution_check": consistency,
        },
        created_utc=_utcnow())

    analysis = result.to_dict()
    analysis["debug_refs"] = {"debug_dir": _redact(str(paths["debug_dir"]), redact_path)}
    _write_json(paths["analysis"], analysis)
    if showcase_enabled:
        from .showcase_report import (render_linkedin_draft, render_showcase_report,
                                      showcase_config_payload)
        paths["report"].write_text(render_showcase_report(result), encoding="utf-8")
        paths["linkedin"].write_text(render_linkedin_draft(result), encoding="utf-8")
        # The run's own config, with the resolution it was bound to, written under an
        # underscore key so the file still parses as a valid config and can be re-run.
        _write_json(paths["config_copy"],
                    showcase_config_payload(config, resolved, probe, source_sha,
                                            spec, result))
    else:
        paths["report"].write_text(render_report(result), encoding="utf-8")
        if summary:
            paths["summary"].write_text(render_summary(result), encoding="utf-8")

    provenance = _provenance(result, resolved, probe, binding, interpreter, spec,
                             config_sha, config_file, source_sha, paths,
                             verification, redact_path)
    _write_json(paths["provenance"], provenance)
    _write_debug_files(paths["debug_dir"], result, worker, spec)

    if render_enabled and not (verification["render_ffprobe"].get("checked")
                               and verification["render_ffprobe"].get("ok")):
        raise DemoRenderError(
            "the annotated video is not a verified H.264/yuv420p/faststart render at "
            "source cadence: "
            + (render_info.get("reason")
               or "; ".join(verification["render_ffprobe"].get("problems") or [])
               or "unknown reason")
            + ". The report, analysis and provenance were written with this failure "
              "stated; no video is claimed.")
    return result


def _presentation_record(paths: Dict[str, Path], cfg, render_info: dict,
                         keyframes_info: dict, showcase_enabled: bool) -> dict:
    """What the run produced and how it was presented, in one record.

    The report, the analysis JSON and the provenance all read this instead of each
    deriving the artifact list, so the three cannot disagree about which files exist
    or which style the video carries.
    """
    from .showcase import METRICS

    out_dir = paths["keyframes_dir"] if showcase_enabled else paths["keyframes_dir"].parent
    names = ([paths[role].name for role in ("annotated", "report", "analysis",
                                            "provenance", "config_copy", "linkedin",
                                            "contact_sheet")]
             if showcase_enabled else
             [paths[role].name for role in ("annotated", "report", "analysis",
                                            "provenance", "summary", "contact_sheet")])
    record: Dict[str, Any] = {
        "style": "showcase" if showcase_enabled else "demo",
        "renderer": ("ghostcaddie.demo.render_showcase" if showcase_enabled
                     else "ghostcaddie.demo.render"),
        "output_dir": str(out_dir),
        "output_names": names,
        "output_paths": {role: str(path) for role, path in paths.items()
                         if not hasattr(path, "is_dir") or not path.is_dir()},
    }
    if not showcase_enabled:
        record["note"] = ("the demo renderer puts its chrome in a side column or an "
                          "external gutter beside the picture; nothing in this run "
                          "draws over the footage.")
        return record

    options = cfg.showcase
    metrics = list(options.metrics)
    record.update({
        "brand": {"title": options.brand, "subtitle": options.brand_sub},
        "disclosure": options.disclosure,
        "phase_badge_rule": ("the badge names the span the frame is IN: the last "
                             "operator phase marker at or before the frame. Before "
                             "the first marker it is hidden rather than showing the "
                             "next phase early."),
        "cards": {
            "video_metrics": metrics,
            "still_metrics": metrics[:3],
            "definitions": {name: {"label": METRICS[name]["label"],
                                   "unit": METRICS[name]["unit"],
                                   "space": METRICS[name]["space"],
                                   "source": METRICS[name]["source"],
                                   "description": METRICS[name]["description"]}
                            for name in metrics if name in METRICS},
            "placeholder": "-- (drawn dimmed; no value is held or invented)",
        },
        "chrome": {
            "inside_picture": bool(render_info.get("chrome_inside_picture", True)),
            "added_width_px": int(render_info.get("added_width_px") or 0),
            "layout": render_info.get("layout") or {},
            "layout_problems": render_info.get("layout_problems") or [],
            "measured_coverage": render_info.get("chrome_coverage") or {},
            "text_engine": render_info.get("text_engine") or {},
        },
        "stills": [{"phase": card.get("phase"), "path": card.get("path"),
                    "frame": card.get("frame"), "label": card.get("label"),
                    "metrics": card.get("metrics"),
                    "unavailable_metrics": card.get("unavailable_metrics")}
                   for card in (keyframes_info.get("cards") or [])],
        "stills_missing": [{"phase": item.get("phase"), "path": item.get("path"),
                            "reason": item.get("reason")}
                           for item in (keyframes_info.get("missing") or [])],
        "contact_sheet": keyframes_info.get("contact_sheet") or "",
        "contact_sheet_panels": keyframes_info.get("contact_sheet_panels") or [],
        "impact_accent_frames": render_info.get("impact_accent_frames") or [],
        "trail_points": render_info.get("trail_points") or {},
        "notes": [
            "The picture fills the frame: no gutter, letterbox or sidebar was added.",
            "Cards are translucent chrome drawn over the footage; the measured overlap "
            "with the golfer region and with the drawn ball/club points is recorded "
            "under chrome.measured_coverage rather than described as zero.",
            "Card values are 2D image-space quantities; 2D projected shoulder and hip "
            "tilt are labelled as 2D tilts and are not 3D turn measurements.",
        ],
    })
    return record


def _track_from_payload(name: str, payload: dict, resolved: ResolvedConfig):
    """Rebuild the pure track object from the worker's JSON, so analytics and the
    report work on the same records the video was drawn from."""
    from .tracking import track_from_dict
    return track_from_dict(name, payload)


def _resolve_phases_for_result(worker: dict, resolved: ResolvedConfig,
                               tracks: Dict[str, "TargetTrack"],
                               body_records: Sequence[dict]) -> dict:
    from .analytics import body_angles_for_record
    from .phases import frame_seconds_factory
    config = resolved.config
    return resolve_phases(
        config_keyframes=dict(config.keyframes),
        interval_frames=list(resolved.frames),
        body_records=body_records, tracks=tracks,
        fps=float(resolved.source.fps), allow_approx=bool(config.allow_approx_phases),
        body_angles_fn=(lambda record: body_angles_for_record(
            record, pixel_aspect=float(resolved.source.pixel_aspect_ratio),
            handedness=config.handedness)),
        frame_seconds=frame_seconds_factory(resolved.frame_times,
                                            resolved.start_frame))


def _phase_consistency(worker_phases: Optional[dict], derived: dict) -> dict:
    """Compare the facts that matter, so the report cannot contradict the video.

    Angles are not compared numerically: the worker's payload rounds coordinates to
    two decimals when it serialises, so an exact float comparison would be a
    rounding test rather than a consistency test.
    """
    if worker_phases is None:
        return {"checked": False, "reason": "worker returned no phase resolution"}
    differences: List[str] = []

    def facts(payload: dict) -> dict:
        phases = payload.get("phases") or {}
        return {name: (info.get("label"), info.get("frame"))
                for name, info in phases.items()}

    worker_facts, derived_facts = facts(worker_phases), facts(derived)
    for name in sorted(set(worker_facts) | set(derived_facts)):
        if worker_facts.get(name) != derived_facts.get(name):
            differences.append(f"{name}: worker {worker_facts.get(name)} vs "
                               f"orchestrator {derived_facts.get(name)}")
    w_tempo, d_tempo = worker_phases.get("tempo") or {}, derived.get("tempo") or {}
    for key in ("state", "backswing_frames", "downswing_frames", "label",
                "backswing_start_marker"):
        if w_tempo.get(key) != d_tempo.get(key):
            differences.append(f"tempo.{key}: worker {w_tempo.get(key)} vs "
                               f"orchestrator {d_tempo.get(key)}")
    return {"checked": True, "consistent": not differences, "differences": differences}


def _limitations(resolved: ResolvedConfig, worker: dict, target_summary: dict,
                 body_2d: dict, verification: dict) -> List[str]:
    out: List[str] = []
    probe = resolved.source
    cfg = resolved.config
    if cfg.timing.accept_variable_cadence and probe.variable_cadence:
        out.append("The source has a variable container cadence and the operator "
                   "accepted it explicitly: frame times are container-cadence only.")
    if not cfg.roi:
        out.append("No operator ROI was supplied, so no golfer pose was attributed "
                   "and no 2D body quantity is reported on this run.")
    body_state = worker.get("body", {}).get("state", "unavailable")
    if body_state in ("ambiguous", "unavailable", "failed"):
        out.append(f"The body layer is {body_state.upper()} on this clip: the pose "
                   f"was not confidently the golfer often enough, so no 2D body "
                   f"quantity is reported rather than an uncertain one.")
    for name, metrics in target_summary.items():
        if not metrics.get("available"):
            out.append(f"{name} is UNAVAILABLE on this run: "
                       f"{metrics.get('reason', 'no operator seed was supplied')}.")
            continue
        if metrics.get("interpolated_frames"):
            out.append(f"{metrics['interpolated_frames']} {name} frames are derived by "
                       f"interpolation across bounded gaps; they are drawn dashed and "
                       f"are excluded from observed quantities.")
        if metrics.get("lost_frames"):
            out.append(f"{name} was lost for {metrics['lost_frames']} frames after the "
                       f"tracker missed it repeatedly; those frames stay missing and "
                       f"nothing bridges them.")
    if not cfg.keyframes and not cfg.allow_approx_phases:
        out.append("No phase keyframes were supplied, so every phase card is "
                   "UNAVAILABLE and no tempo is reported.")
    unsupported = body_2d.get("unavailable") or []
    if unsupported:
        out.append(f"{len(unsupported)} of the 2D body quantities did not reach the "
                   f"reporting threshold on this clip: "
                   + "; ".join(f"{m['name']} ({m['reason']})" for m in unsupported[:4])
                   + ".")
    render = worker.get("render") or {}
    if render.get("written"):
        if render.get("dimensions_trimmed_to_even"):
            out.append("The rendered frame size was trimmed to even dimensions, which "
                       "H.264 yuv420p requires; the analysis coordinates are native.")
        if render.get("scale") not in (None, 1.0):
            wh = render.get("wh") or [0, 0]
            picture = render.get("picture_wh") or wh
            gutter = render.get("panel_external_gutter_px") or 0
            out.append(f"The annotated video was scaled to {picture[0]}x{picture[1]} for a "
                       f"manageable file size"
                       + (f", with a {gutter}px analytics gutter beside the picture so no "
                          f"chrome sits over the swing" if gutter else "")
                       + "; every measurement is in native source pixels.")
        if verification.get("render_ffprobe", {}).get("checked"):
            out.append("The rendered video was re-probed independently: "
                       + (f"{verification['render_ffprobe'].get('codec')} / "
                          f"{verification['render_ffprobe'].get('pix_fmt')} / "
                          f"faststart "
                          f"{verification['render_ffprobe'].get('faststart')} / "
                          f"{verification['render_ffprobe'].get('frames')} frames."
                          if verification["render_ffprobe"].get("ok") else
                          "verification reported problems: "
                          + "; ".join(verification["render_ffprobe"].get("problems") or [])))
    out.append("This run is a single selected clip. Nothing here generalises to other "
               "clips, and no tracker on this path is described as a detector.")
    return out


def _provenance(result: DemoResult, resolved: ResolvedConfig, probe: SourceProbe,
                binding: dict, interpreter: dict, spec: dict, config_sha: str,
                config_file: Path, source_sha: str, paths: Dict[str, Path],
                verification: dict, redact_path: bool) -> dict:
    outputs = []
    showcase = bool(result.presentation.get("style") == "showcase")
    # ``provenance`` is deliberately absent from both lists: this file is written
    # AFTER these hashes are taken, so including it would record the previous run's
    # hash (or none at all) as if it described this one.
    roles = (("annotated", "report", "analysis", "config_copy", "linkedin",
              "contact_sheet") if showcase else
             ("annotated", "report", "analysis", "summary", "contact_sheet"))
    for role in roles:
        path = paths.get(role)
        if path is None or not os.path.isfile(path):
            continue
        outputs.append({"role": role, "path": _redact(str(path), redact_path),
                        "bytes": os.path.getsize(path),
                        "sha256": sha256_of(path)})
    for card in (result.keyframes.get("cards") or []):
        if card.get("written"):
            path = Path(card["path"])
            outputs.append({"role": (f"still:{card['phase']}" if showcase
                                     else f"keyframe:{card['phase']}"),
                            "path": _redact(str(path), redact_path),
                            "bytes": os.path.getsize(path) if path.is_file() else 0,
                            "sha256": sha256_of(path) if path.is_file() else ""})
    if showcase:
        for item in (result.keyframes.get("missing") or []):
            path = Path(str(item.get("path")))
            if not item.get("written") or not path.is_file():
                continue
            outputs.append({"role": f"still-unavailable:{item.get('phase')}",
                            "path": _redact(str(path), redact_path),
                            "bytes": os.path.getsize(path),
                            "sha256": sha256_of(path)})
    cfg = resolved.config
    return {
        "schema_version": DEMO_PROVENANCE_SCHEMA,
        "created_utc": result.created_utc,
        "tool": {"name": "fairwayos-demo", "version": DEMO_TOOL_VERSION,
                 "analysis_schema": DEMO_ANALYSIS_SCHEMA},
        "disclosure": result.disclosure(),
        "source": {"path": _redact(probe.path, redact_path),
                   "display_name": probe.display_name,
                   "sha256": source_sha,
                   "description": cfg.source_description,
                   "probe": probe.to_dict()},
        "source_binding": binding,
        "config": {
            "path": _redact(str(config_file), redact_path),
            "sha256": config_sha,
            "schema_version": cfg.schema_version,
            "effective_resolved_config": resolved.effective(),
            "warnings": list(resolved.warnings),
            "notes": list(resolved.notes),
        },
        "manual_input_records": {
            "origin": "operator",
            "roi_xywh": list(cfg.roi) if cfg.roi else None,
            "roi_keyframes": [{"index": i, **key.to_dict(),
                               "origin": "operator_crop_geometry",
                               "author": (key.note or "operator"),
                               "method": ("manual ROI boxes across a camera pan, "
                                          "interpolated linearly in position and size")}
                              for i, key in enumerate(cfg.roi_keyframes)],
            "roi_effective_xywh": roi_envelope_for(cfg),
            "seeds": [{"target": name, **point.to_dict()}
                      for name, point in sorted(cfg.seeds.items())],
            "keyframes": [{"phase": phase, "frame": frame}
                          for phase, frame in sorted(cfg.keyframes.items(),
                                                     key=lambda kv: kv[1])],
            "corrections": [{"target": name, **point.to_dict()}
                            for name in sorted(cfg.corrections)
                            for point in cfg.corrections[name]],
            "manual_anchor_segments": [{"target": name, **block.to_dict()}
                                       for name, block in
                                       sorted(cfg.manual_segments.items())],
            "interpolation": cfg.interpolation.to_dict(),
            "statement": ("Every entry above is operator input. No point in this run "
                          "was produced by an automatic detector, and no operator "
                          "input is described as a detection."),
        },
        "processing": {
            "interpreter": interpreter,
            "interpreter_candidates": candidate_interpreters(interpreter.get("path")),
            "worker_module": "ghostcaddie.demo.worker",
            "environment": result.debug.get("environment") or {},
            "decoding": "cv2.VideoCapture, every frame of the bounded interval",
            "passes": ["pose+track", "render (streamed to ffmpeg)"],
            "bounded": {"max_frames": cfg.max_frames,
                        "frames_processed": resolved.frame_count},
        },
        "reuse": {
            "canonical_modules_used": ["ghostcaddie.video.source_binding (hash binding)",
                                       "ghostcaddie.video.model_paths (body model "
                                       "resolution)",
                                       "ghostcaddie.video.canonical_worker.MoveNet + "
                                       "attribute (pose)",
                                       "ghostcaddie.upload.runtimes.probe_interpreter "
                                       "(interpreter probe)"],
            "canonical_files_modified": False,
            "canonical_behavior_changed": False,
            "note": ("Demo mode reuses canonical utilities read-only. The canonical "
                     "workflow, its schemas, its layer states and its tests are "
                     "untouched; the demo never writes canonical output names."),
        },
        "network": {
            "calls": "none",
            "basis": ("no network client is imported by this path, proxy variables are "
                      "stripped and FAIRWAYOS_OFFLINE=1 is set for the worker"),
            "external_ai_service": "none",
            "downloads": "none",
        },
        "outputs": outputs,
        "presentation": result.presentation,
        "spec": spec,
        "verification": verification,
        "render": result.render,
    }


def _write_debug_files(debug_dir: Path, result: DemoResult, worker: dict,
                       spec: dict) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    _write_json(debug_dir / "body_frames.json",
                {"per_frame": result.body.get("per_frame") or [],
                 "state": result.body.get("state"),
                 "coverage": result.body.get("coverage")})
    for name, payload in (worker.get("targets") or {}).items():
        _write_json(debug_dir / f"target_{name}.json",
                    {"track": payload,
                     "raw_measurements": (worker.get("raw_measurements") or {}).get(name)
                     or {}})
    _write_json(debug_dir / "phase_resolution.json", result.phases)
    _write_json(debug_dir / "render.json", result.render)
    _write_json(debug_dir / "spec.json", spec)
    _write_json(debug_dir / "verification.json",
                {"render": result.debug.get("verification", {}),
                 "phase_resolution_consistent":
                     result.debug.get("phase_resolution_consistent"),
                 "environment": result.debug.get("environment")})
    _write_json(debug_dir / "analytics.json",
                {"body_2d": result.analytics.get("body_2d"),
                 "targets": result.analytics.get("targets"),
                 "missingness": result.analytics.get("missingness")})

