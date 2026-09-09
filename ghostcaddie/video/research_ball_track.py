"""Research-only ``research-ball-track`` runner around ``SeededBallTracker``.

This module renders human-seeded candidate tracks for visual research review.
It never produces ground truth, production observations, or analytics: every
artifact keeps ``research_only=True``, ``ground_truth=False``, and
``production_eligible=False``. Source-frame indices are preserved everywhere
diagnostics or provenance are serialized; overlay rendering uses a local
zero-based index space internally (mapped back at every boundary).
"""

from dataclasses import replace
from pathlib import Path
import json
import shutil
import subprocess

import cv2
import numpy as np

from .research_ball import SeededBallTracker, SeededBallTrackItem, SeededBallTrackResult
from .research_overlay import build_seeded_ball_overlay_filter

RESEARCH_FLAGS = {
    "research_only": True,
    "ground_truth": False,
    "production_eligible": False,
}
SLOW_VIEW_FACTOR = 4
ARTIFACT_SCHEMA = "research-ball-track.v1"


def validate_seed_and_range(*, start_frame: int, end_frame: int, seed_frame: int,
                            seed_x: float, seed_y: float, roi) -> tuple:
    """Validate CLI-supplied source-frame/seed/ROI bounds; return the ROI box."""
    for name, value in (("start_frame", start_frame), ("end_frame", end_frame),
                        ("seed_frame", seed_frame)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if end_frame < start_frame:
        raise ValueError("end_frame must be greater than or equal to start_frame")
    if not start_frame <= seed_frame <= end_frame:
        raise ValueError("seed_frame must lie inside the source frame range")
    for name, value in (("seed_x", seed_x), ("seed_y", seed_y)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{name} must be a non-negative number")
    if len(roi) != 4:
        raise ValueError("roi must be (x1, y1, x2, y2)")
    x1, y1, x2, y2 = (int(round(float(v))) for v in roi)
    if not (x2 > x1 and y2 > y1):
        raise ValueError("roi must be a non-degenerate (x1, y1, x2, y2) box")
    if not (x1 <= seed_x < x2 and y1 <= seed_y < y2):
        raise ValueError("seed point must lie inside the roi")
    return (x1, y1, x2, y2)


def _ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to render research MP4 artifacts")
    return ffmpeg


def _encode(ffmpeg: str, source: Path, output: Path, *, fps: float,
            extra_filters: str = "", output_fps: float = None) -> None:
    args = [ffmpeg, "-y", "-v", "error", "-i", str(source)]
    filters = "scale=in_range=pc:out_range=tv,format=yuv420p"
    if extra_filters:
        filters = extra_filters + "," + filters
    if output_fps is not None:
        args.extend(["-vf", filters, "-r", f"{output_fps:.6f}"])
    else:
        args.extend(["-vf", filters])
    args.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 str(output)])
    subprocess.run(args, check=True)


def run_research_ball_track(video: str, out_dir: str, *, start_frame: int, end_frame: int,
                            seed_frame: int, seed_x: float, seed_y: float, roi,
                            slow_view_factor: int = SLOW_VIEW_FACTOR) -> dict:
    """Track the seeded ball over [start_frame, end_frame] and write research artifacts."""
    region = validate_seed_and_range(
        start_frame=start_frame, end_frame=end_frame, seed_frame=seed_frame,
        seed_x=seed_x, seed_y=seed_y, roi=roi)
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_path = Path(video).expanduser().resolve()
    if not source_path.is_file():
        raise ValueError(f"video does not exist: {source_path.name}")

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise ValueError("video could not be opened")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    if end_frame >= max(total, 0):
        cap.release()
        raise ValueError(f"end_frame {end_frame} exceeds decoded frame count {max(total, 0)}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    source_indices = []
    for position in range(start_frame, end_frame + 1):
        ok, frame_bgr = cap.read()
        if not ok:
            cap.release()
            raise ValueError(f"failed to decode source frame {position}")
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        source_indices.append(position)
    cap.release()
    height, width = frames[0].shape[:2]

    # Tracker parameters are the module defaults; no per-frame reseeding.
    tracker = SeededBallTracker(roi=region)
    result = tracker.track(
        frames, seed_frame_index=seed_frame - start_frame,
        seed_point=(float(seed_x), float(seed_y)))
    if result.production_eligible or result.ground_truth:
        raise RuntimeError("seeded tracker contract violation")

    # Tracker item indices are already local to the decoded frame sequence;
    # only diagnostics/provenance map them back to source indices.
    local_result = SeededBallTrackResult(
        result.track_id,
        result.items,
        result.longest_gap, result.provenance,
        production_eligible=False, ground_truth=False)
    overlay_filter = build_seeded_ball_overlay_filter(
        local_result, width=width, height=height)

    items_payload = [{
        "frame_index": start_frame + item.frame_index,
        "center": None if item.center is None else [float(item.center[0]), float(item.center[1])],
        "confidence": float(item.confidence),
        "provenance": item.provenance,
        "warnings": list(item.warnings),
    } for item in result.items]
    # The tracker only tracks from the seed forward; earlier frames in the
    # requested window stay explicitly unavailable (never back-seeded).
    pre_seed = [{
        "frame_index": index,
        "center": None, "confidence": 0.0, "provenance": "unavailable",
        "warnings": ["pre_seed"],
    } for index in range(start_frame, seed_frame)]
    items_payload = pre_seed + items_payload
    tracked_count = sum(item["provenance"] == "tracked" for item in items_payload)
    unavailable_count = sum(item["provenance"] == "unavailable" for item in items_payload)

    diagnostics = {
        "schema_version": ARTIFACT_SCHEMA,
        **RESEARCH_FLAGS,
        "status": "research_candidate",
        "coordinate_space": "pixels",
        "tracker": "SeededBallTracker",
        "seed_frame_index": seed_frame,
        "seed": {
            "frame_index": seed_frame,
            "point": [float(seed_x), float(seed_y)],
        },
        "roi": list(region),
        "source_frame_start": start_frame,
        "source_frame_end": end_frame,
        "source_frame_count": len(frames),
        "track": {
            "track_id": result.track_id,
            "items": items_payload,
            "longest_gap": result.longest_gap,
            "tracked_count": tracked_count,
            "unavailable_count": unavailable_count,
        },
        "pose": None, "analytics": None, "impact": None, "landing": None,
        "calibration": None,
        "warnings": [
            "research candidate candidates only; not a validated golf ball",
            "no per-frame reseeding; fixed seed appearance model",
        ],
    }
    (out / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")

    ffmpeg = _ffmpeg()
    source_rate_path = out / "annotated_source_rate.mp4"
    slow_view_path = out / "annotated_slow_view.mp4"
    crop_path = out / "seed_crop.mp4"
    contact_path = out / "contact_sheet.jpg"
    raw_segment = out / ".intermediate_raw.mp4"
    intermediate = out / ".intermediate_annotated.mp4"
    # Trim to the requested source-frame window, then apply the overlay in a
    # second pass so the filter's enable indices align with local frame numbers.
    subprocess.run([
        ffmpeg, "-y", "-v", "error", "-i", str(source_path),
        "-vf", f"trim=start_frame={start_frame}:end_frame={end_frame + 1},setpts=PTS-STARTPTS",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(raw_segment),
    ], check=True)
    _encode(ffmpeg, raw_segment, intermediate, fps=fps,
            extra_filters=overlay_filter)
    _encode(ffmpeg, intermediate, source_rate_path, fps=fps)
    _encode(ffmpeg, source_rate_path, slow_view_path, fps=fps,
            extra_filters=f"setpts={SLOW_VIEW_FACTOR}*PTS",
            output_fps=fps / SLOW_VIEW_FACTOR)
    cx1, cy1, cx2, cy2 = region
    crop_width = min(cx2 - cx1, width)
    crop_height = min(cy2 - cy1, height)
    _encode(ffmpeg, source_rate_path, crop_path, fps=fps,
            extra_filters=f"crop={crop_width}:{crop_height}:{cx1}:{cy1}")
    subprocess.run([
        ffmpeg, "-y", "-v", "error", "-i", str(source_rate_path),
        "-vf", f"scale=320:-1,tile=4x2", "-frames:v", "1", str(contact_path),
    ], check=True)
    intermediate.unlink(missing_ok=True)
    raw_segment.unlink(missing_ok=True)

    provenance = {
        "schema_version": ARTIFACT_SCHEMA + "-provenance",
        **RESEARCH_FLAGS,
        "tracker": "SeededBallTracker",
        "tracker_module": "ghostcaddie.video.research_ball",
        "seed_frame_index": seed_frame,
        "source_file": source_path.name,
        "source_sha256_first_frames": None,
        "seed": {"frame_index": seed_frame, "point": [float(seed_x), float(seed_y)]},
        "roi": list(region),
        "source_frame_range": [start_frame, end_frame],
        "decoded_frame_indices": source_indices,
        "slow_view_factor": SLOW_VIEW_FACTOR,
        "artifacts": {
            "annotated_source_rate": "annotated_source_rate.mp4",
            "annotated_slow_view": "annotated_slow_view.mp4",
            "seed_crop": "seed_crop.mp4",
            "contact_sheet": "contact_sheet.jpg",
            "diagnostics": "diagnostics.json",
            "readme": "README.md",
        },
    }
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    readme = (
        "# Seeded ball research track\n\n"
        "RESEARCH ONLY — NOT A GOLF-BALL DETECTOR — NOT GROUND TRUTH — NOT PRODUCTION ELIGIBLE.\n\n"
        f"- Source: {source_path.name}\n"
        f"- Source frame range: [{start_frame}, {end_frame}] (seed frame {seed_frame})\n"
        f"- Seed point: ({seed_x:g}, {seed_y:g}); ROI {region}\n"
        "- Annotated source-rate video: annotated_source_rate.mp4\n"
        f"- Annotated slow view ({SLOW_VIEW_FACTOR}x slower): annotated_slow_view.mp4\n"
        "- Native ROI crop: seed_crop.mp4; contact sheet: contact_sheet.jpg\n"
        "- Diagnostics: diagnostics.json; provenance: provenance.json\n\n"
        "Markers: green box = human seed, yellow box = tracked candidate,\n"
        "red tick on the bottom bar = unavailable frame. Gaps are never bridged.\n"
    )
    (out / "README.md").write_text(readme)

    return diagnostics