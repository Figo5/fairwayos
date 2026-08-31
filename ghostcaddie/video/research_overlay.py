"""Dependency-free text overlays for research video diagnostics."""

from pathlib import Path


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
    )
