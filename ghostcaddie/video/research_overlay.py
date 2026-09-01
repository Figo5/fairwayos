"""Dependency-free text overlays for research video diagnostics."""

from pathlib import Path
import math


def artifact_source_label(path: Path, artifact_root: Path) -> str:
    """Return a stable, non-local source label relative to an artifact root."""
    try:
        return path.resolve().relative_to(artifact_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("source path must be inside the artifact root") from exc


def is_clipped(x: float, y: float, radius: float, width: int, height: int) -> bool:
    """Return whether a circular target intersects the source image boundary."""
    return (x - radius < 0 or y - radius < 0 or
            x + radius >= width or y + radius >= height)


def overlay_lines(*, frame_index: int, fps: float, state: str,
                  confidence: float, uncertainty_px: float,
                  clipped: bool = False) -> tuple[str, ...]:
    """Return human-readable research labels for a zero-based frame index."""
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    if fps <= 0:
        raise ValueError("fps must be positive")
    return (
        f"FRAME {frame_index + 1}  TIME {frame_index / fps:.3f}s",
        "SHARED RESEARCH ADAPTER  track_id: ball-0",
        "NOT A GOLF-BALL DETECTOR | RESEARCH ONLY",
        f"state: {state}  confidence: {confidence:.2f}  uncertainty: +/-{uncertainty_px:.1f}px"
        + ("  CLIPPED TARGET" if clipped else ""),
        "WARNING: false positives possible; no production analytics",
        "VALIDATED BALL IDENTITY: UNAVAILABLE",
    )


def build_research_ffmpeg_filter(items, *, fps: float, width: int, height: int,
                                visually_aligned: bool = True,
                                rejection_reason: str = "visual_alignment_rejected") -> str:
    """Build a dependency-free, pixel-space research candidate overlay.

    ``items`` are already-produced candidate coordinates from a local research
    adapter.  This function only renders them; it never promotes them to golf
    ball evidence or to a production observation.  Each marker is enabled for
    exactly its source frame, while trail segments remain visible thereafter.
    """
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError("fps must be a finite positive number")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer")
    if not isinstance(visually_aligned, bool):
        raise ValueError("visually_aligned must be boolean")
    if not isinstance(rejection_reason, str) or not rejection_reason:
        raise ValueError("rejection_reason must be a non-empty string")
    normalized = []
    previous = None
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("research overlay items must be mappings")
        frame = item.get("frame")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError("research overlay frame must be a non-negative integer")
        if previous is not None and frame <= previous:
            raise ValueError("research overlay frames must be strictly increasing")
        values = []
        for name in ("x", "y", "radius", "uncertainty_px"):
            value = item.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("research overlay coordinates must be finite numbers")
            values.append(float(value))
        x, y, radius, uncertainty = values
        if radius <= 0 or uncertainty < 0 or not (0 <= x < width and 0 <= y < height):
            raise ValueError("research overlay candidate is outside image bounds")
        normalized.append((frame, x, y, radius, uncertainty))
        previous = frame
    if not normalized and visually_aligned:
        raise ValueError("at least one research overlay item is required")

    # Geometric filters are intentionally used instead of drawtext/drawline:
    # minimal FFmpeg builds may omit those filters. The top/bottom bars are a
    # visual legend; the full disclaimer remains in the JSON sidecar.
    filters = [
        f"drawbox=x=0:y=0:w={width}:h=4:color={'blue' if rejection_reason.startswith('object_consistency_') else 'yellow'}:t=fill",
        f"drawbox=x=0:y={height-4}:w={width}:h=4:color=red:t=fill",
    ]
    if not visually_aligned:
        return ",".join(filters)
    for index, (frame, x, y, radius, uncertainty) in enumerate(normalized):
        enable = f"enable='eq(n\\,{frame})'"
        filters.append(
            f"drawbox=x={x-radius:g}:y={y-radius:g}:w={2*radius:g}:h={2*radius:g}:color=yellow:t=2:{enable}"
        )
        filters.append(
            f"drawbox=x={x-uncertainty:g}:y={y-uncertainty:g}:w={2*uncertainty:g}:h={2*uncertainty:g}:color=orange:t=1:{enable}"
        )
        if index:
            _, px, py, _, _ = normalized[index - 1]
            trail_enable = f"enable='gte(n\\,{frame})'"
            # A dotted trail is portable across FFmpeg builds that omit
            # drawline; it remains a candidate path, not a trajectory claim.
            filters.append(
                f"drawbox=x={px-2:g}:y={py-2:g}:w=4:h=4:color=yellow@0.75:t=fill:{trail_enable}"
            )
    return ",".join(filters)
