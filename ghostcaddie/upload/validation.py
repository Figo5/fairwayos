"""Upload validation: path safety, real limits, decoded facts.

Every limit is checked against the FILE, never against caller-supplied metadata.
Size is checked before any decode so a hostile file cannot force a large read.
Remote URLs are refused outright: this tool takes local files, which removes the
SSRF surface rather than trying to filter it.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Optional

_URL = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


class UploadRejected(ValueError):
    """Raised when an upload cannot be accepted. Always carries a reason."""


@dataclass(frozen=True)
class VideoLimits:
    max_bytes: int = 512 * 1024 * 1024      # 512 MB
    max_seconds: float = 180.0
    min_width: int = 320
    min_height: int = 240
    max_width: int = 7680
    max_height: int = 4320
    decode_timeout_seconds: float = 60.0


@dataclass(frozen=True)
class ValidatedVideo:
    path: str
    sha256: str
    size_bytes: int
    width: int
    height: int
    frames: int
    fps_num: int
    fps_den: int
    duration_seconds: float

    @property
    def fps(self) -> float:
        return self.fps_num / float(self.fps_den or 1)


def safe_join(root: str, name: str) -> str:
    """Join `name` under `root`, refusing traversal or absolute escape."""
    if not name or name in (".", ".."):
        raise UploadRejected("empty or invalid filename")
    if os.path.isabs(name) or _URL.match(name):
        raise UploadRejected(f"filename must be a plain name, got {name!r}")
    root_real = os.path.realpath(root)
    target = os.path.realpath(os.path.join(root_real, name))
    if target != root_real and not target.startswith(root_real + os.sep):
        raise UploadRejected(f"path escapes the upload directory: {name!r}")
    return target


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_upload(path: str, limits: Optional[VideoLimits] = None,
                    workdir: Optional[str] = None) -> ValidatedVideo:
    limits = limits or VideoLimits()
    if _URL.match(str(path)):
        raise UploadRejected(
            "remote URLs are not accepted; this tool analyses local files only")
    if not os.path.isfile(path):
        raise UploadRejected(f"file not found: {path}")

    size = os.path.getsize(path)
    if size <= 0:
        raise UploadRejected("file is empty")
    if size > limits.max_bytes:                     # checked BEFORE decode
        raise UploadRejected(
            f"file size {size} exceeds limit {limits.max_bytes}")

    import cv2
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise UploadRejected("file could not be opened as video")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        ok, _ = cap.read()                          # prove at least one real frame
    finally:
        cap.release()

    if not ok or w <= 0 or h <= 0:
        raise UploadRejected("file could not be decoded as video")
    if w < limits.min_width or h < limits.min_height:
        raise UploadRejected(
            f"decoded dimensions {w}x{h} below minimum "
            f"{limits.min_width}x{limits.min_height}")
    if w > limits.max_width or h > limits.max_height:
        raise UploadRejected(f"decoded dimensions {w}x{h} above maximum")
    if fps <= 0:
        raise UploadRejected("decoded frame rate is unavailable")
    duration = frames / fps if frames > 0 else 0.0
    if duration <= 0:
        raise UploadRejected("decoded duration is unavailable")
    if duration > limits.max_seconds:
        raise UploadRejected(
            f"duration {duration:.2f}s exceeds limit {limits.max_seconds}s")

    num, den = (round(fps * 1000), 1000)
    return ValidatedVideo(path=os.path.realpath(path), sha256=sha256_file(path),
                          size_bytes=size, width=w, height=h, frames=frames,
                          fps_num=num, fps_den=den, duration_seconds=duration)
