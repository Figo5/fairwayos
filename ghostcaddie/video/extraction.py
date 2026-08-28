"""Deterministic, streaming ffmpeg frame and contact-sheet artifacts."""

import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .errors import VideoExtractionError, VideoPathError

MANIFEST_NAME = "frame_manifest.json"


@dataclass(frozen=True)
class FrameRecord:
    frame_index: int
    timestamp_seconds: Optional[float]
    filename: str

    def to_dict(self):
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "filename": self.filename,
        }


@dataclass(frozen=True)
class FrameExtractionResult:
    output_directory: str
    frames: List[FrameRecord]
    manifest_path: str

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def to_dict(self):
        return {
            "schema_version": "video-diagnostics.v1",
            "frame_count": self.frame_count,
            "frames": [frame.to_dict() for frame in self.frames],
        }


@dataclass(frozen=True)
class ContactSheetResult:
    output_path: str
    tile_count: int
    columns: int
    rows: int
    width: int
    height: int


def _positive(value, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise VideoExtractionError(f"{name} must be positive")


def _source_path(source: str) -> Path:
    """Validate a video source; absolute paths are intentionally allowed."""
    path = Path(source).expanduser()
    if not path.is_file():
        raise VideoPathError(f"source video does not exist: {source}")
    resolved = path.resolve()
    if not os.access(resolved, os.R_OK):
        raise VideoPathError(f"source video is not readable: {source}")
    return resolved


def _output_directory(path_text: str, source: Path) -> Path:
    path = Path(path_text).expanduser()
    resolved = path.resolve()
    if resolved == source or resolved == source.parent and path.suffix:
        raise VideoExtractionError("output path must be a directory separate from the source video")
    if resolved.exists() and not resolved.is_dir():
        raise VideoExtractionError("output path must be a directory")
    try:
        source.relative_to(resolved)
    except ValueError:
        pass
    else:
        raise VideoExtractionError("output directory cannot contain the source video")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _run_ffmpeg(args: List[str], cwd: Optional[Path] = None) -> None:
    try:
        completed = subprocess.run(args, cwd=str(cwd) if cwd else None, capture_output=True, text=True, check=False)
    except (OSError, TypeError) as exc:
        raise VideoExtractionError(f"unable to execute ffmpeg: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()
        raise VideoExtractionError(f"ffmpeg failed{': ' + detail if detail else ''}")


def validate_video_source(source: str) -> Path:
    """Return a resolved regular readable video path, including absolute paths."""
    return _source_path(source)


def extract_frames(source: str, output_directory: str, *, sample_fps: Optional[float] = None,
                   max_frames: Optional[int] = None, ffmpeg: str = "ffmpeg") -> FrameExtractionResult:
    """Extract numbered JPEGs without buffering video data in Python."""
    _positive(sample_fps, "sample_fps")
    _positive(max_frames, "max_frames")
    if sample_fps is None and max_frames is None:
        raise VideoExtractionError("sample_fps or max_frames is required")
    if max_frames is not None and (isinstance(max_frames, bool) or int(max_frames) != max_frames):
        raise VideoExtractionError("max_frames must be a positive integer")
    source_path = _source_path(source)
    output_path = _output_directory(output_directory, source_path)
    # Remove only our own numbered artifacts so reruns cannot inherit stale frames.
    for old_frame in output_path.glob("frame_*.jpg"):
        old_frame.unlink()
    pattern = output_path / "frame_%06d.jpg"
    args = [ffmpeg, "-v", "error", "-i", str(source_path)]
    if sample_fps is not None:
        args.extend(["-vf", f"fps={sample_fps:g}"])
    args.extend(["-vsync", "0"])
    if max_frames is not None:
        args.extend(["-frames:v", str(int(max_frames))])
    args.append(str(pattern))
    _run_ffmpeg(args)

    files = sorted(output_path.glob("frame_*.jpg"))
    if not files:
        raise VideoExtractionError("ffmpeg produced no frames")
    count = len(files)
    records = [FrameRecord(i, (i - 1) / float(sample_fps) if sample_fps else None,
                           f"frame_{i:06d}.jpg") for i in range(1, count + 1)]
    manifest = output_path / MANIFEST_NAME
    frame_payload = [r.to_dict() for r in records]
    manifest.write_text(json.dumps({
        "schema_version": "video-diagnostics.v1",
        "frame_count": count,
        "artifact_references": [r.filename for r in records],
        "frames": frame_payload,
        "frame_observations": frame_payload,
    }, sort_keys=True, indent=2) + "\n")
    return FrameExtractionResult(str(output_path), records, str(manifest))


def generate_contact_sheet(frames_directory: str, output_path: str, *, columns: int = 4,
                           frame_width: int = 320, frame_height: int = 180,
                           ffmpeg: str = "ffmpeg") -> ContactSheetResult:
    """Tile extracted numbered frames with ffmpeg's deterministic tile filter."""
    if isinstance(columns, bool) or not isinstance(columns, int) or columns <= 0:
        raise VideoExtractionError("columns must be a positive integer")
    for value, name in ((frame_width, "frame_width"), (frame_height, "frame_height")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise VideoExtractionError(f"{name} must be a positive integer")
    frames_path = Path(frames_directory).expanduser().resolve()
    if not frames_path.is_dir():
        raise VideoExtractionError("frames directory does not exist")
    files = sorted(frames_path.glob("frame_*.jpg"))
    if not files:
        raise VideoExtractionError("frames directory contains no extracted frames")
    output = Path(output_path).expanduser().resolve()
    try:
        output.relative_to(frames_path)
    except ValueError:
        pass
    else:
        raise VideoExtractionError("contact sheet output path is unsafe")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = len(files)
    rows = (count + columns - 1) // columns
    args = [ffmpeg, "-v", "error", "-framerate", "1", "-start_number", "1",
            "-i", "frame_%06d.jpg", "-frames:v", str(count), "-vf",
            f"scale={frame_width}:{frame_height},tile={columns}x{rows}", "-q:v", "2", str(output)]
    _run_ffmpeg(args, cwd=frames_path)
    return ContactSheetResult(str(output), count, columns, rows, columns * frame_width, rows * frame_height)
