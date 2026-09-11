"""Hash-bound source gate for FairwayOS research runs.

A run must prove that the media bytes, the caller's expected source hash, and any
seed/handoff file all describe the same source before inference or rendering can
consume coordinates. This module imports only standard-library code so callers can
fail closed before model checkpoints, media decoders, or optional CV stacks load.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Union

PathLike = Union[str, Path]

_IMPORTS_AT_MODULE_LEVEL = ["hashlib", "json", "pathlib", "typing"]


class SourceBindingError(RuntimeError):
    """Raised when a research run is not bound to the source bytes it claims."""


def sha256_file(path: PathLike) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_seed_source_sha256(seeds_path: Path) -> tuple[Optional[str], Optional[str], int]:
    try:
        data = json.loads(seeds_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exact parser text is not contract
        raise SourceBindingError(f"invalid seed JSON at {seeds_path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise SourceBindingError(f"seed file must contain a JSON object: {seeds_path}")
    seeds = data.get("seeds", [])
    seed_count = len(seeds) if isinstance(seeds, list) else 0
    source_sha = data.get("source_sha256")
    clip_label = data.get("clip")
    return (source_sha if isinstance(source_sha, str) else None,
            clip_label if isinstance(clip_label, str) else None,
            seed_count)


def bind_source(video: Optional[PathLike], expected_sha256: Optional[str],
                seeds_path: Optional[PathLike] = None) -> dict:
    """Return a binding record only when all supplied source hashes agree.

    ``expected_sha256`` is mandatory. If ``seeds_path`` is supplied, the seed file
    must also carry ``source_sha256`` matching the actual video bytes. Clip names,
    run directory names, and player names are deliberately ignored as proof.
    """
    if not video:
        raise SourceBindingError("video is required; there is no default source")
    if not expected_sha256:
        raise SourceBindingError("expected_sha256 is required before consuming media")
    video_path = Path(video).expanduser()
    if not video_path.exists():
        raise SourceBindingError(f"video not found: {video_path}")
    actual = sha256_file(video_path)
    if actual != expected_sha256:
        raise SourceBindingError(
            "video sha256 does not match expected source bytes; refusing a "
            f"name-bound or transplanted run (actual={actual}, expected={expected_sha256})")

    seed_count = 0
    seed_file = None
    clip_label = None
    if seeds_path is not None:
        seed_path = Path(seeds_path).expanduser()
        if not seed_path.exists():
            raise SourceBindingError(f"seed file not found: {seed_path}")
        seed_sha, clip_label, seed_count = _load_seed_source_sha256(seed_path)
        seed_file = str(seed_path.resolve())
        if not seed_sha:
            detail = f"; clip label {clip_label!r} is not byte provenance" if clip_label else ""
            raise SourceBindingError(f"seed file has no source_sha256{detail}")
        if seed_sha != actual:
            raise SourceBindingError(
                f"seed source_sha256 disagrees with video bytes (seed={seed_sha}, video={actual}); "
                "refusing to transplant coordinates across clips")

    return {
        "hash_bound": True,
        "video": str(video_path.resolve()),
        "source_sha256": actual,
        "seeds_file": seed_file,
        "seed_count": seed_count,
        "clip_label": clip_label,
        "research_only": True,
        "pseudo_label": True,
        "ground_truth": False,
        "production_eligible": False,
    }
