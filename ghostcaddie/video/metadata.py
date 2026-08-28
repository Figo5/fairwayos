"""ffprobe-only video inspection; this module never decodes video frames."""

import json
import subprocess
from fractions import Fraction
from typing import Any, Dict, Optional

from .contracts import VideoMetadata
from .errors import VideoMetadataError, VideoProbeError


def _required(mapping: Dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping or mapping[key] in (None, ""):
        raise VideoMetadataError(f"missing {section} metadata: {key}")
    return mapping[key]


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise VideoMetadataError(f"invalid {name} metadata") from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise VideoMetadataError(f"non-finite {name} metadata")
    return result


def _rate(stream: Dict[str, Any]) -> float:
    raw = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    if not raw or raw == "0/0":
        raise VideoMetadataError("missing or invalid frame rate metadata")
    try:
        result = float(Fraction(str(raw)))
    except (ValueError, ZeroDivisionError):
        result = _number(raw, "frame_rate")
    if result <= 0 or result != result or result in (float("inf"), float("-inf")):
        raise VideoMetadataError("invalid frame rate metadata")
    return result


def parse_ffprobe_metadata(payload: Dict[str, Any], source_identifier: Optional[str] = None) -> VideoMetadata:
    if not isinstance(payload, dict):
        raise VideoMetadataError("ffprobe output must be an object")
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise VideoMetadataError("missing ffprobe streams metadata")
    video = next((item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"), None)
    if video is None:
        raise VideoMetadataError("ffprobe output has no video stream")
    format_info = payload.get("format")
    if not isinstance(format_info, dict):
        raise VideoMetadataError("missing ffprobe format metadata")
    frame_count = video.get("nb_frames")
    if frame_count in (None, "", "N/A"):
        frame_count_value = None
    else:
        try:
            frame_count_value = int(frame_count)
        except (TypeError, ValueError) as exc:
            raise VideoMetadataError("invalid frame_count metadata") from exc
    try:
        return VideoMetadata(
            container_format=str(_required(format_info, "format_name", "format")),
            codec=str(_required(video, "codec_name", "video stream")),
            width=int(_required(video, "width", "video stream")),
            height=int(_required(video, "height", "video stream")),
            frame_rate=_rate(video),
            duration_seconds=_number(_required(format_info, "duration", "format"), "duration"),
            frame_count=frame_count_value,
            source_identifier=source_identifier,
        )
    except VideoMetadataError:
        raise
    except (TypeError, ValueError) as exc:
        raise VideoMetadataError("invalid video metadata") from exc


def inspect_video(source: str, ffprobe: str = "ffprobe") -> VideoMetadata:
    """Inspect *source* through ffprobe JSON output without loading frames."""
    try:
        completed = subprocess.run(
            [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", source],
            capture_output=True, text=True, check=False,
        )
    except (OSError, TypeError) as exc:
        raise VideoProbeError(f"unable to execute ffprobe: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()
        raise VideoProbeError(f"ffprobe failed{': ' + detail if detail else ''}")
    try:
        payload = json.loads(completed.stdout)
    except (TypeError, ValueError) as exc:
        raise VideoProbeError("ffprobe returned invalid JSON") from exc
    return parse_ffprobe_metadata(payload, source_identifier=source)
