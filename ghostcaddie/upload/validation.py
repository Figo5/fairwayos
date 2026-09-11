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


def resolve_import_path(name: str, import_dir: str) -> str:
    """Resolve a submitted name INSIDE the import directory.

    Finding 1: the service previously accepted any absolute path readable by the
    service user. A readable video outside the root was accepted and its absolute
    path, hash and metadata returned. Submission is now confined to an explicit
    import directory, checked after realpath so symlinks cannot escape.
    """
    if not name:
        raise UploadRejected("no filename given")
    if _URL.match(str(name)):
        raise UploadRejected("remote URLs are not accepted")
    root = os.path.realpath(import_dir)
    cand = name if os.path.isabs(name) else os.path.join(root, name)
    real = os.path.realpath(cand)          # resolves symlinks BEFORE the check
    if real != root and not real.startswith(root + os.sep):
        raise UploadRejected(
            f"{name!r} is outside the import directory. Put the file in "
            f"{root} and submit its name, or upload it through the browser.")
    if not os.path.isfile(real):
        raise UploadRejected(f"file not found in the import directory: {name}")
    return real


def probe_video(path: str, timeout: float = 60.0) -> dict:
    """Decode-probe OUT OF PROCESS with a hard wall-clock kill.

    Finding 3: decode_timeout_seconds was declared but never enforced; a hostile
    or corrupt file could hang the serving thread inside cv2 before the bounded
    worker path was ever reached.
    """
    import subprocess, sys, json as _json
    code = (
        "import sys,json,cv2\n"
        "p=sys.argv[1]\n"
        "c=cv2.VideoCapture(p)\n"
        "ok=c.isOpened()\n"
        "w=int(c.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))\n"
        "n=int(c.get(cv2.CAP_PROP_FRAME_COUNT)); f=float(c.get(cv2.CAP_PROP_FPS) or 0)\n"
        "r,_=c.read()\n"
        "c.release()\n"
        "print(json.dumps({'opened':bool(ok),'read':bool(r),'width':w,'height':h,"
        "'frames':n,'fps':f}))\n"
    )
    try:
        pr = subprocess.run([sys.executable, "-c", code, path],
                            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise UploadRejected(
            f"decode probe timed out after {timeout}s; refusing the file")
    if pr.returncode != 0:
        raise UploadRejected("file could not be probed as video")
    try:
        return _json.loads((pr.stdout or "").strip().splitlines()[-1])
    except Exception:
        raise UploadRejected("decode probe returned unparsable output")


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

    info = probe_video(path, timeout=limits.decode_timeout_seconds)
    ok = info.get("opened") and info.get("read")
    w, h = int(info.get("width", 0)), int(info.get("height", 0))
    frames = int(info.get("frames", 0))
    fps = float(info.get("fps", 0) or 0)

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
