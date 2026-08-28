"""Best-effort, explicitly gated automatic perception for arbitrary YouTube clips.

This boundary never invents observations.  Detector implementations are optional
and loaded dynamically; the core package has no ML dependency or model claims.
"""
from __future__ import annotations

import importlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .automatic_perception import evaluate_sequence_gates
from .extraction import extract_frames, generate_contact_sheet
from .annotations import render_annotated_video
from .prepare import prepare_video
from .youtube import (DownloadError, DownloadLimits, DownloaderUnavailable,
                      YtDlpDownloader, parse_youtube_url)

DEFAULT_YTDLP = "/Users/giofiore/ghostcaddie-tour/.venv-video-modern/bin/yt-dlp"
DEFAULT_NODE = "/usr/local/bin/node"
AUTO_FORMAT = "worstvideo[height<=480][ext=mp4]/worstvideo[height<=480]/worst[ext=mp4]/worst"


class DetectorUnavailable(RuntimeError):
    """Automatic detector is not installed or cannot be loaded."""


@dataclass(frozen=True)
class AutoTryConfig:
    url: str
    out: Path
    calibration: Any = None
    course: Any = None
    player: Any = None
    project_root: Optional[Path] = None
    segment_start: float = 0.0
    segment_duration: Optional[float] = None
    render_video: bool = False
    fallback_human: bool = False
    sample_fps: float = 2.0
    max_frames: Optional[int] = None
    yt_dlp: Optional[str] = DEFAULT_YTDLP
    js_runtime: Optional[str] = DEFAULT_NODE


def validate_segment(url: str, start: float = 0.0, duration: Optional[float] = None):
    source = parse_youtube_url(url)
    if isinstance(start, bool) or not isinstance(start, (int, float)) or not math.isfinite(start) or start < 0:
        raise DownloadError("segment start must be non-negative", "invalid_segment")
    if duration is not None and (isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0):
        raise DownloadError("segment duration must be positive", "invalid_segment")
    return source.video_id, float(start), None if duration is None else float(duration)


def load_installed_detector() -> Any:
    """Load only an explicitly configured detector module, never search blindly."""
    spec = os.environ.get("GHOSTCADDIE_AUTO_DETECTOR", "").strip()
    if not spec:
        raise DetectorUnavailable("no automatic detector is configured")
    try:
        module = importlib.import_module(spec)
        detector = getattr(module, "detector", None)
        if detector is None:
            factory = getattr(module, "create_detector", None)
            detector = factory() if callable(factory) else module
        if not callable(getattr(detector, "detect", None)):
            raise DetectorUnavailable("configured detector has no detect(frame) method")
        return detector
    except DetectorUnavailable:
        raise
    except Exception as exc:
        raise DetectorUnavailable("configured automatic detector could not be loaded") from exc


def _reasoned_observations(raw):
    if raw is None:
        return [], {}
    if hasattr(raw, "items") and hasattr(raw, "to_dict"):
        items = [item.to_dict() for item in raw.items]
        return items, {"validated_observations": raw}
    if isinstance(raw, dict):
        return list(raw.get("observations") or []), raw
    if isinstance(raw, list):
        return raw, {}
    return [], {}


def _blocking_reasons(observations, detector_result, *, validated=False):
    reasons = []
    if not observations:
        reasons.append("detector_unavailable")
        return reasons
    cuts = detector_result.get("cut_frames", []) if isinstance(detector_result, dict) else []
    if cuts:
        reasons.append("cut")
    if any(item.get("golfer_count", 0) > 1 or item.get("person_count", 0) > 1 for item in observations if isinstance(item, dict)):
        reasons.append("multiple_golfers")
    fields = ("ball", "club", "clubhead", "contact")
    for field in fields:
        if not any(isinstance(item, dict) and item.get(field) is not None for item in observations):
            reasons.append(field + "_unavailable")
    confidences = [item.get("confidence") for item in observations if isinstance(item, dict) and isinstance(item.get("confidence"), (int, float))]
    if confidences and min(confidences) < 0.5:
        reasons.append("low_confidence")
    if not validated and not any(isinstance(item, dict) and item.get("pose") is not None for item in observations):
        reasons.append("pose_unavailable")
    return reasons


def _write_json(path: Path, payload: Any):
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n")


def _download_failure(exc: DownloadError) -> str:
    """Map internal downloader codes to stable, sanitized user categories."""
    mapping = {
        "downloader_unavailable": "downloader_not_found",
        "download_timeout": "timeout",
        "size_limit_exceeded": "size_limit",
        "disk_limit_exceeded": "size_limit",
        "invalid_media": "decode_failure",
        "protected_content": "video_unavailable",
        "unavailable": "video_unavailable",
    }
    if exc.code in mapping:
        return mapping[exc.code]
    text = str(exc).lower()
    if "ejs" in text:
        return "ejs_missing"
    if "javascript" in text or "js runtime" in text:
        return "javascript_runtime_missing"
    if "format" in text:
        return "format_unavailable"
    if exc.code in {"download_failed", "malformed_metadata"}:
        return "network_failure"
    return "video_unavailable"


def auto_try(config: AutoTryConfig, *, downloader=None, detector=None, analytics_runner=None) -> dict:
    """Run bounded ingestion and best-effort perception, returning safe diagnostics."""
    out = Path(config.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ("recommendation.json", "normalized_shot.json", "overlay.svg"):
        (out / name).unlink(missing_ok=True)
    try:
        _, start, duration = validate_segment(config.url, config.segment_start, config.segment_duration)
    except DownloadError as exc:
        payload = {"schema_version": "youtube-auto-try.v1", "status": "blocked", "ingestion_status": "failed",
                   "perception_status": "not_run", "coordinate_space": "pixels", "observations": [],
                   "blocking_reasons": [exc.code], "warnings": [str(exc)]}
        _write_json(out / "diagnostics.json", payload)
        return payload
    ingest = out / "ingest"
    try:
        downloader = downloader or YtDlpDownloader(
            config.yt_dlp or DEFAULT_YTDLP,
            js_runtime=config.js_runtime,
            limits=DownloadLimits(max_segment_seconds=20.0),
            format_selector=AUTO_FORMAT,
        )
        result = downloader.download(config.url, str(ingest), segment_start=start, segment_duration=duration)
    except DownloaderUnavailable as exc:
        category = "javascript_runtime_missing" if config.js_runtime and not Path(config.js_runtime).is_file() else "downloader_not_found"
        payload = {"schema_version": "youtube-auto-try.v1", "status": "blocked", "ingestion_status": "failed",
                   "perception_status": "not_run", "coordinate_space": "pixels", "observations": [],
                   "blocking_reasons": [category], "warnings": [str(exc)]}
        _write_json(out / "diagnostics.json", payload)
        return payload
    except DownloadError as exc:
        category = _download_failure(exc)
        payload = {"schema_version": "youtube-auto-try.v1", "status": "blocked", "ingestion_status": "failed",
                   "perception_status": "not_run", "coordinate_space": "pixels", "observations": [],
                   "blocking_reasons": [category], "warnings": [str(exc)]}
        _write_json(out / "diagnostics.json", payload)
        return payload

    try:
        frames = extract_frames(result.path, str(out / "frames"), sample_fps=config.sample_fps, max_frames=config.max_frames)
    except Exception:
        payload = {"schema_version": "youtube-auto-try.v1", "status": "blocked", "ingestion_status": "complete",
                   "perception_status": "not_run", "coordinate_space": "pixels", "observations": [],
                   "blocking_reasons": ["frame_extraction_failed"],
                   "warnings": ["frame extraction produced no usable evidence"]}
        _write_json(out / "diagnostics.json", payload)
        return payload
    refs = ["frames/frame_manifest.json"]
    if frames.frames:
        generate_contact_sheet(frames.output_directory, str(out / "contact_sheet.jpg"), columns=min(4, len(frames.frames)))
        refs.append("contact_sheet.jpg")
    annotated = out / "annotated_frames"; annotated.mkdir(parents=True, exist_ok=True)
    for frame in frames.frames:
        # Copying is intentional: without observations, no visual mark is claimed.
        shutil.copy2(Path(frames.output_directory) / frame.filename, annotated / frame.filename)
        refs.append("annotated_frames/" + frame.filename)
    if config.render_video and frames.frames:
        render_annotated_video(annotated, out / "annotated_video.mp4", frame_rate=config.sample_fps)
        refs.append("annotated_video.mp4")
    if config.fallback_human:
        prepare_video(result.path, str(out), sample_fps=config.sample_fps, max_frames=config.max_frames)
        refs.extend(["annotation_workspace.html", "video-human-annotations.v1.json"])

    try:
        detector = detector or load_installed_detector()
        detected = detector.detect([str(Path(frames.output_directory) / f.filename) for f in frames.frames])
    except DetectorUnavailable as exc:
        detected = None
        detector_warning = str(exc)
    except Exception:
        detected = None
        detector_warning = "automatic detector failed without usable evidence"
    observations, details = _reasoned_observations(detected)
    reasons = _blocking_reasons(observations, details,
                                validated="validated_observations" in details)
    if config.calibration is None:
        reasons.append("calibration_unavailable")
    payload = {"schema_version": "youtube-auto-try.v1", "status": "complete" if not reasons else "blocked",
               "ingestion_status": "complete", "perception_status": "complete" if observations else "unavailable",
               "coordinate_space": "course" if not reasons else "pixels", "observations": observations,
               "blocking_reasons": sorted(set(reasons)), "artifact_references": refs,
               "detector": "configured" if detected is not None else "unavailable"}
    if detected is None:
        payload["warnings"] = [detector_warning]
    if config.fallback_human:
        payload["fallback_mode"] = "human-annotation-preparation"
        payload["warnings"] = payload.get("warnings", []) + ["explicit fallback-human selected"]
    # A recommendation is deliberately only possible through a caller-supplied,
    # already-gated reconstruction result; raw detector dictionaries never enter
    # analytics and this module never synthesizes one.
    if not reasons and analytics_runner is not None and details.get("validated_observations") is not None:
        try:
            analytics = analytics_runner(details["validated_observations"], config.calibration,
                                         config.course, config.player)
        except Exception as exc:
            payload["status"] = "blocked"
            payload["blocking_reasons"] = ["analytics_unavailable"]
            payload["warnings"] = ["gated analytics failed without producing a recommendation"]
        else:
            if isinstance(analytics, dict):
                payload.update(analytics)
                payload["coordinate_space"] = "course"
    # A recommendation is deliberately impossible without the explicit runner.
    if not reasons and isinstance(details.get("analytics"), dict):
        payload["analytics"] = details["analytics"]
    if not reasons and details.get("shot_event") is not None:
        payload["normalized_shot"] = details["shot_event"]
    _write_json(out / "diagnostics.json", payload)
    return payload
