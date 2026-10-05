"""Local browser review UI for the broadcast workflow (stdlib only, loopback only).

What it is for
--------------
The operator needs to see a clip, drag the golfer box over a real frame, mark the
stroke's own phases, place the ball/club seeds on frames where the target is actually
visible, save a config and render it. No JSON editing, no memorised hotkeys.

Security choices, all deliberate
--------------------------------
* Binds **127.0.0.1** and refuses any other ``--host``. There is no remote endpoint.
* Every mutating request needs the **per-launch CSRF token** in ``X-Broadcast-Token``,
  and the ``Host`` header must be a loopback name. A cross-site form post cannot set a
  custom header, so it cannot reach a mutation.
* The client never sends a filesystem path to read or write. Media and results are
  addressed by **server-issued ids** from a registry built at launch from allow-listed
  roots; a path that escapes its root (``..``, symlink) is refused after realpath.
* Bodies are bounded (``Content-Length`` required, <= 64 KiB) and validated with the
  same parsers the CLI uses. An invalid config is **not written**.
* Frames are decoded server-side by frame index with ffmpeg's ``select=eq(n\\,F)``, so a
  preview is the exact source frame -- the browser's ``<video>.currentTime`` is never
  used to decide a frame number.
* Renders run as bounded argv subprocesses (no shell), one at a time.
"""
from __future__ import annotations

import json
import mimetypes
import os
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

from .broadcast import PROTECTED_OUTPUT_PREFIXES, broadcast_out_paths
from .broadcast_config import BROADCAST_CONFIG_SCHEMA
from .broadcast_ui_page import PAGE
from .errors import DemoConfigError

MAX_JSON_BYTES = 64 * 1024
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024
RENDER_TIMEOUT_SECONDS = 3600
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
CLIP_SUFFIXES = (".mp4", ".mov", ".m4v", ".mkv", ".webm")
FRAME_MAX_WIDTH = 1280

RUN_ROOT = Path("/Users/giofiore/fairway-os-runs/tiger-showcase-cycle")
BASELINE_CLIPS = Path(
    "/Users/giofiore/fairway-os-runs/tiger-masters-2019-2026-10-01/clips")


# ---------------------------------------------------------------- registry

@dataclass
class ClipRecord:
    clip_id: str
    path: str
    name: str
    root: str
    sha256: str = ""
    width: int = 0
    height: int = 0
    fps_rational: str = ""
    frame_count: int = 0
    has_audio: bool = False

    def to_dict(self) -> dict:
        return {"id": self.clip_id, "name": self.name, "path": self.path,
                "root": self.root, "sha256": self.sha256, "width": self.width,
                "height": self.height, "fps_rational": self.fps_rational,
                "frame_count": self.frame_count, "has_audio": self.has_audio}


def _sha256_file(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ffprobe_json(args: Sequence[str]) -> dict:
    try:
        p = subprocess.run(["ffprobe", "-v", "error", *args, "-of", "json"],
                           capture_output=True, text=True, timeout=120)
        return json.loads(p.stdout or "{}")
    except Exception:
        return {}


def probe_clip(path: str) -> dict:
    payload = _ffprobe_json(["-show_streams", "-show_format", path])
    streams = payload.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    return {"width": int(video.get("width") or 0),
            "height": int(video.get("height") or 0),
            "fps_rational": str(video.get("r_frame_rate") or ""),
            "frame_count": int(video.get("nb_frames") or 0),
            "has_audio": bool(audio)}


def decode_frame_png(clip: str, frame: int, *, max_width: int = FRAME_MAX_WIDTH,
                     ffmpeg: str = "ffmpeg") -> bytes:
    """One exact source frame, decoded server-side by index.

    ``select=eq(n\\,F)`` selects the F-th DECODED frame, which is the only frame index
    the configs use. That is why the browser never derives a frame from playback time:
    a seek and a decoded index are not the same thing on long-GOP H.264.
    """
    if frame < 0:
        raise DemoConfigError(f"frame index must be non-negative, got {frame}")
    scale = f"scale='min({int(max_width)},iw)':-2"
    cmd = [ffmpeg, "-v", "error", "-i", clip, "-vf",
           f"select='eq(n\\,{int(frame)})',{scale}", "-frames:v", "1",
           "-f", "image2pipe", "-vcodec", "png", "-"]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0 or not proc.stdout:
        raise DemoConfigError(
            f"frame {frame} could not be decoded from "
            f"{os.path.basename(clip)}: {(proc.stderr or b'').decode()[:200]}")
    return proc.stdout


class Registry:
    """Server-issued ids for clips and results. The client never sends a path."""

    def __init__(self, roots: Sequence[Path]) -> None:
        self.roots = [Path(os.path.realpath(str(r))) for r in roots]
        self.clips: Dict[str, ClipRecord] = {}
        self.results: Dict[str, dict] = {}
        self.jobs: Dict[str, dict] = {}
        self.tracking_review: Dict[str, dict] = {}
        self.saved_tracks: dict = {}
        self.lock = threading.Lock()
        self.scan()

    def scan(self) -> None:
        for root in self.roots:
            if not root.is_dir():
                continue
            for path in sorted(root.iterdir()):
                if path.suffix.lower() in CLIP_SUFFIXES and path.is_file():
                    self.register_clip(str(path))

    def _allowed(self, path: str) -> bool:
        real = Path(os.path.realpath(path))
        return any(real == root or root in real.parents for root in self.roots)

    def register_clip(self, path: str) -> ClipRecord:
        real = os.path.realpath(path)
        if not self._allowed(real):
            raise DemoConfigError(
                f"{os.path.basename(path)} is outside the allow-listed roots "
                f"({[str(r) for r in self.roots]}); the UI does not read arbitrary "
                f"paths")
        if not os.path.isfile(real):
            raise DemoConfigError(f"not a file: {os.path.basename(path)}")
        with self.lock:
            for record in self.clips.values():
                if record.path == real:
                    return record
            facts = probe_clip(real)
            clip_id = f"clip{len(self.clips) + 1:02d}"
            record = ClipRecord(
                clip_id=clip_id, path=real, name=os.path.basename(real),
                root=str(next((r for r in self.roots
                               if Path(real) == r or r in Path(real).parents), "")),
                sha256=_sha256_file(real), **facts)
            self.clips[clip_id] = record
            return record

    def clip(self, clip_id: str) -> ClipRecord:
        record = self.clips.get(clip_id)
        if record is None:
            raise DemoConfigError(f"unknown clip id {clip_id!r}")
        return record

    def new_job(self, *, clip_id: str, config_path: str, out_dir: str) -> str:
        job_id = uuid.uuid4().hex[:12]
        self.jobs[job_id] = {
            "id": job_id, "clip_id": clip_id, "config": config_path,
            "out_dir": out_dir, "state": "running", "started": time.time(),
            "finished": None, "exit_code": None, "log": "", "result": None}
        return job_id


# ---------------------------------------------------------------- server

def viewport_to_native(rect: Mapping[str, float], natural: Mapping[str, float],
                       client_x: float, client_y: float) -> Optional[dict]:
    """Displayed CSS pixels -> NATIVE source pixels, or ``None`` outside the picture.

    The reference implementation of the mapping the page runs in JavaScript: the
    image is scaled to FIT its box and centred, so a click in the letterbox is
    refused rather than silently mapped to a corner pixel -- which is exactly what the
    desktop picker does. ``tests/fixtures/viewport_vectors.json`` holds the shared
    vectors both implementations are checked against.
    """
    if natural["w"] <= 0 or natural["h"] <= 0:
        return None
    scale = min(rect["width"] / natural["w"], rect["height"] / natural["h"])
    if scale <= 0:
        return None
    drawn_w = natural["w"] * scale
    drawn_h = natural["h"] * scale
    off_x = rect["left"] + (rect["width"] - drawn_w) / 2.0
    off_y = rect["top"] + (rect["height"] - drawn_h) / 2.0
    x = (client_x - off_x) / scale
    y = (client_y - off_y) / scale
    if x < 0 or y < 0 or x > natural["w"] or y > natural["h"]:
        return None
    return {"x": round(x), "y": round(y), "scale": scale}


def make_handler(registry: Registry, token: str, config_dir: Path,
                 render_config: dict) -> type:
    class Handler(BaseHTTPRequestHandler):
        server_version = "FairwayOS-BroadcastUI/1.0"

        # --- helpers -------------------------------------------------
        def log_message(self, format, *args):  # keep the console readable
            pass

        def _send(self, code: int, body: bytes, content_type: str,
                  extra: Optional[dict] = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code: int, payload: dict) -> None:
            self._send(code, (json.dumps(payload, default=str) + "\n").encode(),
                       "application/json; charset=utf-8")

        def _error(self, code: int, message: str) -> None:
            self._json(code, {"ok": False, "error": message})

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").split(":")[0].strip("[]")
            return host in LOOPBACK_HOSTS

        def _token_ok(self) -> bool:
            return self.headers.get("X-Broadcast-Token") == token

        def _read_json(self) -> dict:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise DemoConfigError("Content-Length is required on this request")
            try:
                length = int(raw_length)
            except ValueError:
                raise DemoConfigError("Content-Length must be an integer")
            if length < 0 or length > MAX_JSON_BYTES:
                raise DemoConfigError(
                    f"body must be at most {MAX_JSON_BYTES} bytes, got {length}")
            body = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DemoConfigError(f"body is not valid JSON: {exc}")
            if not isinstance(payload, dict):
                raise DemoConfigError("body must be a JSON object")
            return payload

        def _guard(self) -> bool:
            if not self._host_ok():
                self._error(403, "the Host header is not a loopback name")
                return False
            if not self._token_ok():
                self._error(403, "missing or wrong X-Broadcast-Token")
                return False
            return True

        # --- routes --------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802
            if not self._host_ok():
                self._error(403, "the Host header is not a loopback name")
                return
            parsed = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            try:
                if parsed.path == "/":
                    self._send(200, PAGE.replace("__TOKEN__", token).encode(),
                               "text/html; charset=utf-8")
                    return
                if parsed.path == "/api/clips":
                    self._json(200, {"ok": True,
                                     "clips": [c.to_dict()
                                               for c in registry.clips.values()]})
                    return
                if parsed.path == "/api/tracking":
                    record = registry.clip(str(query.get("clip", "")))
                    target = query.get("target", "clubhead")
                    if target not in {"ball", "clubhead"}:
                        raise DemoConfigError("unknown tracking target")
                    self._json(200, {"ok": True, "tracking": registry.tracking_review.get(
                        record.sha256, {}).get(target)})
                    return
                if parsed.path == "/api/tracking/points":
                    record = registry.clip(str(query.get("clip", "")))
                    target = query.get("target", "clubhead")
                    if target not in {"ball", "clubhead"}:
                        raise DemoConfigError("unknown tracking target")
                    block = registry.saved_tracks.get("clips_by_sha", {}).get(record.sha256)
                    result = (block or {}).get("targets", {}).get(target)
                    self._json(200, {"ok": True, "source_sha256": record.sha256,
                                     "target": target, "result": result,
                                     "available": bool(block),
                                     "prediction_sha256": registry.saved_tracks.get("prediction_sha256")})
                    return
                if parsed.path == "/api/library":
                    self._json(200, {"ok": True, "roots": _library(registry)})
                    return
                if parsed.path == "/api/configs":
                    self._json(200, {"ok": True,
                                     "configs": _config_list(config_dir, registry)})
                    return
                if parsed.path == "/api/config":
                    name = str(query.get("name", ""))
                    if name:
                        body = _read_config_file(config_dir, name)
                        self._send(200, body, "application/json")
                        return
                    job_config = registry.jobs.get(str(query.get("id", "")))
                    if job_config is None or not job_config.get("config"):
                        self._error(404, "no config for that id")
                        return
                    self._send(200, Path(job_config["config"]).read_bytes(),
                               "application/json")
                    return
                if parsed.path == "/api/frame":
                    record = registry.clip(str(query.get("clip", "")))
                    raw_frame = str(query.get("frame", "0"))
                    if not raw_frame.lstrip("-").isdigit():
                        self._error(400, "frame must be a whole number")
                        return
                    png = decode_frame_png(record.path, int(raw_frame))
                    self._send(200, png, "image/png",
                               {"X-Source-Frame": raw_frame,
                                "X-Source-Width": str(record.width),
                                "X-Source-Height": str(record.height)})
                    return
                if parsed.path == "/api/job":
                    job = registry.jobs.get(str(query.get("id", "")))
                    if job is None:
                        self._error(404, "unknown job id")
                        return
                    self._json(200, {"ok": True, "job": _job_public(job)})
                    return
                if parsed.path == "/api/artifact":
                    payload = _artifact_body(registry, str(query.get("id", "")),
                                             str(query.get("name", "")))
                    if payload is None:
                        self._error(404, "unknown artifact")
                        return
                    body, ctype = payload
                    self._send(200, body, ctype,
                               {"Content-Disposition":
                                f'attachment; filename="{os.path.basename(query["name"])}"'})
                    return
                if parsed.path == "/api/config":
                    job_config = registry.jobs.get(str(query.get("id", "")))
                    if job_config is None or not job_config.get("config"):
                        self._error(404, "no config for that id")
                        return
                    body = Path(job_config["config"]).read_bytes()
                    self._send(200, body, "application/json")
                    return
            except DemoConfigError as exc:
                self._error(400, str(exc))
                return
            except Exception as exc:  # pragma: no cover - defensive
                self._error(500, f"{type(exc).__name__}: {exc}")
                return
            self._error(404, "no such endpoint")

        def do_HEAD(self) -> None:  # noqa: N802
            self.do_GET()

        def do_POST(self) -> None:  # noqa: N802
            if not self._guard():
                return
            parsed = urlparse(self.path)
            try:
                payload = self._read_json()
                if parsed.path == "/api/validate":
                    self._json(200, _validate(registry, payload))
                    return
                if parsed.path == "/api/save":
                    self._json(200, _save_config(config_dir, payload))
                    return
                if parsed.path == "/api/render":
                    self._json(200, _start_render(registry, config_dir,
                                                  render_config, payload))
                    return
                if parsed.path == "/api/scenes":
                    from .discovery import discover

                    record = registry.clip(str(payload.get("clip", "")))
                    found = discover(record.path)
                    self._json(200, {"ok": True,
                                     "candidates": found["candidates"],
                                     "scores": found["scores"]})
                    return
                if parsed.path == "/api/import":
                    self._json(200, {"ok": True, "clip": _import_clip(registry, payload)
                                     .to_dict()})
                    return
            except DemoConfigError as exc:
                self._error(400, str(exc))
                return
            except Exception as exc:  # pragma: no cover - defensive
                self._error(500, f"{type(exc).__name__}: {exc}")
                return
            self._error(404, "no such endpoint")

    return Handler


def _import_clip(registry: Registry, payload: dict) -> ClipRecord:
    """Register a clip from an allow-listed root, by relative name or by path.

    Two ways in, both confined: ``{root, name}`` where ``name`` may be a subpath
    (``new-clips/09_hole03_tee.mp4``) and ``{path}`` as an absolute path that must
    already lie inside one of the allow-listed roots. The confinement check is on the
    realpath in :meth:`Registry.register_clip`, so ``..`` and symlinks are refused
    either way, and the client still never sends a path it can write to.
    """
    root = str(payload.get("root") or "")
    name = str(payload.get("name") or "")
    path = str(payload.get("path") or "")
    if path and not (root or name):
        candidate = os.path.realpath(path)
    elif root and name:
        if os.path.isabs(name):
            raise DemoConfigError(
                "name must be relative to the folder it was listed under, not an "
                "absolute path; use `path` to import an absolute one")
        candidate = os.path.realpath(os.path.join(root, name))
    else:
        raise DemoConfigError(
            "import needs either {root, name} or {path}")
    return registry.register_clip(candidate)


def _library(registry: Registry, *, max_depth: int = 3,
             max_files: int = 400) -> List[dict]:
    """What can be imported, per allow-listed root. Names only, never an open path.

    The scan is recursive (bounded depth and count) because a batch of new clips lives
    in its own subfolder -- ``<root>/new-clips/09_hole03_tee.mp4`` -- and a
    non-recursive listing of the root would show the operator none of them. What is
    returned is a path RELATIVE to the root; the client sends that relative name back
    and the server joins and confines it, so the page still never holds a path.
    """
    out: List[dict] = []
    for root in registry.roots:
        if not root.is_dir():
            continue
        media: List[dict] = []
        others: List[str] = []
        for path in sorted(root.rglob("*")):
            try:
                relative = path.relative_to(root)
            except ValueError:  # pragma: no cover - rglob is rooted at root
                continue
            if len(relative.parts) > max_depth or path.name.startswith("."):
                continue
            if not path.is_file():
                continue
            if path.suffix.lower() in CLIP_SUFFIXES:
                media.append({"name": relative.as_posix(),
                              "bytes": path.stat().st_size})
            elif len(others) < 50:
                others.append(relative.as_posix())
            if len(media) >= max_files:
                break
        out.append({"root": str(root), "files": media,
                    "media": [item["name"] for item in media if "/" not in item["name"]],
                    "others": others})
    return out


def _clip_for_config(registry: Registry, config: Mapping[str, Any]
                     ) -> Tuple[Optional[ClipRecord], str]:
    """Which registered clip a config was written for, and how that was decided.

    The digest is authoritative when the config carries one; the clip path hint is
    the fallback (exact realpath first, then file name). ``(None, reason)`` means the
    config's footage is not in the library, which is a refusal, not a guess: loading
    its scenes against whatever clip happened to be selected is how a Hole 12 config
    ended up editing the Hole 7 clip.
    """
    shot = config.get("shot") or {}
    expected = str(shot.get("expected_source_sha256") or "")
    if expected:
        for record in registry.clips.values():
            if record.sha256 == expected:
                return record, "expected_source_sha256"
        return None, ("no clip in the library has the sha256 this config expects "
                      f"({expected[:12]}...)")
    hint = str(config.get("clip_path_hint") or "")
    if hint:
        real = os.path.realpath(hint)
        for record in registry.clips.values():
            if record.path == real:
                return record, "clip_path_hint"
        for record in registry.clips.values():
            if record.name == os.path.basename(real):
                return record, "clip_path_hint (by file name)"
        return None, f"the clip this config names is not in the library ({real})"
    return None, ("the config names neither expected_source_sha256 nor clip_path_hint, "
                  "so which clip it belongs to cannot be decided")


def _config_list(config_dir: Path,
                 registry: Optional[Registry] = None) -> List[dict]:
    """The configs in the config directory that are BROADCAST configs.

    A directory that holds a cut scan, a batch manifest or a demo config as well will
    not show them here: an entry that the loader would refuse is not a loadable config,
    and offering it invites the operator to pick something that cannot work.
    """
    from .broadcast_config import parse_broadcast

    config_dir = Path(os.path.realpath(str(config_dir)))
    if not config_dir.is_dir():
        return []
    out = []
    for path in sorted(config_dir.glob("*.json")):
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        if parsed.get("schema_version") != BROADCAST_CONFIG_SCHEMA:
            continue
        entry: Dict[str, Any] = {"name": path.name, "scenes": 0, "loadable": False}
        try:
            config = parse_broadcast(parsed)
        except Exception as exc:
            entry["error"] = str(exc)[:200]
            out.append(entry)
            continue
        shot = parsed.get("shot") or {}
        entry.update({"loadable": True, "shot_id": shot.get("id") or "",
                      "stroke_type": shot.get("stroke_type") or "",
                      "scenes": len(config.scenes),
                      "clip_path_hint": config.clip_path_hint,
                      "expected_source_sha256": config.expected_source_sha256})
        if registry is not None:
            clip, how = _clip_for_config(registry, parsed)
            entry["clip_id"] = clip.clip_id if clip else None
            entry["clip_name"] = clip.name if clip else None
            entry["clip_resolved_by"] = how
        out.append(entry)
    return out


def _read_config_file(config_dir: Path, name: str) -> bytes:
    if name != os.path.basename(name) or name.startswith("."):
        raise DemoConfigError("config name must be a plain file name")
    config_dir = Path(os.path.realpath(str(config_dir)))
    candidate = os.path.realpath(str(config_dir / name))
    if not candidate.startswith(str(config_dir) + os.sep):
        raise DemoConfigError("config name escapes the config directory")
    if not os.path.isfile(candidate):
        raise DemoConfigError(f"no such config: {name}")
    from .broadcast_config import parse_broadcast

    try:
        payload = json.loads(Path(candidate).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DemoConfigError(f"{name} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != \
            BROADCAST_CONFIG_SCHEMA:
        raise DemoConfigError(
            f"{name} is not a broadcast config (schema_version must be "
            f"{BROADCAST_CONFIG_SCHEMA!r}); the review UI loads broadcast configs only")
    try:
        parse_broadcast(payload)
    except DemoConfigError as exc:
        raise DemoConfigError(f"{name} would not load: {exc}") from exc
    return Path(candidate).read_bytes()


def _job_public(job: dict) -> dict:
    return {k: v for k, v in job.items() if k != "process"}


def _artifact_body(registry: Registry, job_id: str,
                   name: str) -> Optional[Tuple[bytes, str]]:
    job = registry.jobs.get(job_id)
    if job is None:
        return None
    base = os.path.realpath(str(job["out_dir"]))
    candidate = os.path.realpath(os.path.join(base, name))
    if not (candidate == base or candidate.startswith(base + os.sep)):
        raise DemoConfigError("artifact name escapes its own output directory")
    if not os.path.isfile(candidate):
        return None
    ctype = mimetypes.guess_type(candidate)[0] or "application/octet-stream"
    return Path(candidate).read_bytes(), ctype


def _validate(registry: Registry, payload: dict) -> dict:
    from .broadcast_config import parse_broadcast, validate_broadcast
    from .probe import probe_source

    clip = registry.clip(str(payload.get("clip", "")))
    config = payload.get("config")
    if not isinstance(config, dict):
        raise DemoConfigError("config must be a JSON object")
    mismatch = _clip_match_error(registry, clip, config)
    if mismatch:
        raise DemoConfigError(mismatch)
    parsed = parse_broadcast(config)
    probe = probe_source(clip.path)
    resolved = validate_broadcast(parsed, probe)
    return {"ok": True,
            "clip": clip.to_dict(),
            "clip_resolved_by": _clip_for_config(registry, config)[1],
            "scenes": [{"id": s.scene.scene_id, "kind": s.scene.kind,
                        "stroke_type": s.scene.stroke_type,
                        "start_frame": s.start_frame, "end_frame": s.end_frame}
                       for s in resolved.scenes],
            "warnings": resolved.warnings}


def _write_config_payload(config_dir: Path, name: str, config: Mapping[str, Any]) -> Path:
    """Validate, then write the operator's own document back.

    The file that is written is the payload that was checked, not a re-rendering of
    the parsed model. A save through the model would restate only the keys the model
    holds, so an assistance mark the schema tolerates but the dataclass does not carry
    (an operator's note on a segment, a comment under an ``_`` key) would disappear
    from a config the operator only opened and saved. Nothing is invented either: the
    payload was accepted by the same parser the CLI uses before this is called.
    """
    from .broadcast_config import parse_broadcast

    name = str(name or "").strip()
    if not name or name != os.path.basename(name) or name.startswith("."):
        raise DemoConfigError(
            "name must be a plain file name with no directory part")
    if not name.endswith(".json"):
        name += ".json"
    parse_broadcast(config)
    config_dir = Path(os.path.realpath(str(config_dir)))
    config_dir.mkdir(parents=True, exist_ok=True)
    target = config_dir / name
    target.write_text(json.dumps(config, indent=2, sort_keys=False) + "\n",
                      encoding="utf-8")
    return target


def _save_config(config_dir: Path, payload: dict) -> dict:
    name = str(payload.get("name") or "").strip()
    config = payload.get("config")
    if not isinstance(config, dict):
        raise DemoConfigError("config must be a JSON object")
    # Validate BEFORE writing: an invalid config is not written, exactly like the picker.
    target = _write_config_payload(config_dir, name, config)
    return {"ok": True, "path": str(target), "name": target.name}


def _clip_match_error(registry: Registry, clip: ClipRecord,
                      config: Mapping[str, Any]) -> Optional[str]:
    """The reason this config must not be used against this clip, if there is one.

    The UI already switches the selected clip to the one a config was written for; this
    is the second line of defence, and it is what makes a mismatch a refusal instead of
    a render that silently measures the wrong footage (the renderer refuses too, but by
    then the operator has waited for a job).
    """
    resolved, how = _clip_for_config(registry, config)
    if resolved is None:
        return (f"this config's clip could not be matched to the library ({how}); "
                f"import the clip it was written for first")
    if resolved.clip_id != clip.clip_id:
        return (f"this config was written for {resolved.name} ({how}), not for the "
                f"selected {clip.name}; select {resolved.name} or import that clip")
    return None


def _start_render(registry: Registry, config_dir: Path, render_config: dict,
                  payload: dict) -> dict:
    from .broadcast_config import parse_broadcast, validate_broadcast
    from .probe import probe_source

    clip = registry.clip(str(payload.get("clip", "")))
    config = payload.get("config")
    keep_config = bool(payload.get("save_config", False))
    name = str(payload.get("name") or f"ui_{clip.clip_id}.json")
    if not isinstance(config, dict):
        raise DemoConfigError("config must be a JSON object")
    mismatch = _clip_match_error(registry, clip, config)
    if mismatch:
        raise DemoConfigError(mismatch)
    mode = payload.get("tracking_mode", "reviewed")
    if mode not in {"reviewed", "saved_automatic"}:
        raise DemoConfigError("Unknown tracking export mode")
    job_config = dict(render_config)
    job_config["use_saved_tracks"] = mode == "saved_automatic"
    if job_config["use_saved_tracks"]:
        from .saved_tracks import load_saved_tracks
        path = render_config.get("tracking_predictions")
        if not path:
            raise DemoConfigError("No saved tracking results were loaded at startup")
        current = load_saved_tracks(path, [clip], require_all_registered=False)
        if current["prediction_sha256"] != registry.saved_tracks.get("prediction_sha256"):
            raise DemoConfigError("Saved tracking file changed; restart review to load it")
        if clip.sha256 not in current["clips_by_sha"]:
            raise DemoConfigError("No saved tracking results match this clip")
    parsed = parse_broadcast(config)
    resolved = validate_broadcast(parsed, probe_source(clip.path))

    if job_config["use_saved_tracks"]:
        from types import SimpleNamespace
        from .broadcast import build_broadcast_spec, broadcast_out_paths
        from .broadcast_saved import attach_saved_tracks
        check_paths = broadcast_out_paths(Path(render_config["out_root"]))
        check_spec = build_broadcast_spec(resolved, out_dir=render_config["out_root"],
                                         paths=check_paths, body_model=SimpleNamespace(path="", found=False),
                                         redact_path=False, render_enabled=True)
        attach_saved_tracks(check_spec, resolved, render_config["tracking_predictions"],
                            registry.saved_tracks.get("prediction_sha256"))
        job_config["tracking_predictions_sha256"] = registry.saved_tracks["prediction_sha256"]

    config_path = Path(os.path.realpath(str(config_dir))) / name
    if keep_config or not config_path.exists():
        # The operator's own document is written back, not a re-rendering of the model
        # (see _write_config_payload): a render must not be the step that loses an
        # assistance mark from the config it was given.
        config_path = _write_config_payload(config_dir, name, config)

    out_root = Path(os.path.realpath(str(render_config["out_root"])))
    render_key = uuid.uuid4().hex
    out_dir = out_root / f"{parsed.shot_id or clip.clip_id}_{mode}_{render_key[:12]}"
    for protected in PROTECTED_OUTPUT_PREFIXES:
        if str(out_dir).startswith(protected):
            raise DemoConfigError(
                f"refusing to write into the preserved tree {protected}")

    # Pin the submitted document for this job; later saves cannot change a running export.
    snapshot_dir = Path(config_dir) / ".render-snapshots" / render_key
    snapshot_config = json.loads(json.dumps(config))
    manifest = (snapshot_config.get("shot") or {}).get("parent_source_manifest")
    if manifest and not Path(manifest).is_absolute():
        snapshot_config["shot"]["parent_source_manifest"] = str((Path(config_dir) / manifest).resolve())
    execution_path = _write_config_payload(snapshot_dir, name, snapshot_config)

    job_id = registry.new_job(clip_id=clip.clip_id, config_path=str(execution_path),
                              out_dir=str(out_dir))
    thread = threading.Thread(target=_run_job, args=(registry, job_id, clip.path,
                                                     str(execution_path), str(out_dir),
                                                     job_config),
                              daemon=True)
    thread.start()
    return {"ok": True, "job": _job_public(registry.jobs[job_id]),
            "config_path": str(config_path),
            "frames": resolved.frame_count}


def _run_job(registry: Registry, job_id: str, clip: str, config_path: str,
             out_dir: str, render_config: dict) -> None:
    job = registry.jobs[job_id]
    cmd = [render_config["python"], "-m", "ghostcaddie", "broadcast-showcase", "run",
           clip, "--config", config_path, "--out", out_dir, "--force"]
    if render_config.get("use_saved_tracks"):
        cmd += ["--saved-tracks", render_config["tracking_predictions"],
                "--saved-tracks-sha256", render_config["tracking_predictions_sha256"]]
    if render_config.get("cv_python"):
        cmd += ["--cv-python", render_config["cv_python"]]
    if render_config.get("body_model"):
        cmd += ["--body-model", render_config["body_model"]]
    env = dict(os.environ)
    repo_root = render_config.get("repo_root") or ""
    if repo_root:
        env["PYTHONPATH"] = os.pathsep.join(
            [repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    job["command"] = " ".join(shlex.quote(part) for part in cmd)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              cwd=repo_root or None, timeout=RENDER_TIMEOUT_SECONDS)
        job["exit_code"] = proc.returncode
        job["log"] = ((proc.stdout or "") + (proc.stderr or ""))[-8000:]
        job["state"] = "succeeded" if proc.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        job["exit_code"] = 124
        job["state"] = "failed"
        job["log"] = f"the render did not finish within {RENDER_TIMEOUT_SECONDS}s"
    except Exception as exc:  # pragma: no cover - defensive
        job["exit_code"] = 1
        job["state"] = "failed"
        job["log"] = f"{type(exc).__name__}: {exc}"
    job["finished"] = time.time()
    job["seconds"] = round(job["finished"] - job["started"], 2)
    if job["state"] == "succeeded":
        paths = broadcast_out_paths(out_dir)
        job["result"] = {
            "out_dir": out_dir,
            "artifacts": {name: str(paths[key])
                          for key, name in (("annotated", "broadcast_annotated.mp4"),
                                            ("report", "broadcast_report.md"),
                                            ("analysis", "broadcast_analysis.json"),
                                            ("provenance",
                                             "broadcast_provenance.json"),
                                            ("config_copy", "broadcast_config.json"))
                          if paths[key].is_file()}}
        if paths["analysis"].is_file():
            try:
                analysis = json.loads(paths["analysis"].read_text(encoding="utf-8"))
                job["result"]["verification"] = analysis.get("verification")
                job["result"]["tracking_export"] = (analysis.get("config") or {}).get("tracking_export")
                job["result"]["render"] = analysis.get("render")
                job["result"]["scenes"] = [
                    {"id": s.get("id"), "kind": s.get("kind"),
                     "stroke_type": s.get("stroke_type"), "window": s.get("window"),
                     "body_state": (s.get("body") or {}).get("state"),
                     "body_coverage": (s.get("body") or {}).get("coverage"),
                     "phases": list(((s.get("phases") or {}).get("phases")
                                     or {}).keys()),
                     "measured": s.get("measured")}
                    for s in (analysis.get("scenes") or [])]
            except Exception:  # pragma: no cover - defensive
                pass


def serve(*, host: str = "127.0.0.1", port: int = 8765, out_root: Optional[str] = None,
          config_dir: Optional[str] = None, extra_roots: Sequence[str] = (),
          cv_python: Optional[str] = None, body_model: Optional[str] = None,
          tracking_predictions: Optional[str] = None,
          open_browser: bool = False) -> None:
    """Launch the review UI. Refuses a non-loopback host."""
    if host not in LOOPBACK_HOSTS:
        raise DemoConfigError(
            f"the review UI binds loopback only; refusing host {host!r}")
    roots = [BASELINE_CLIPS, Path(out_root or RUN_ROOT), Path(config_dir
                                                              or RUN_ROOT / "configs")]
    roots += [Path(p) for p in extra_roots]
    registry = Registry(roots)
    if tracking_predictions:
        from .saved_tracks import load_saved_tracks
        from .tracking_review import summarize_saved_tracks
        registry.saved_tracks = load_saved_tracks(tracking_predictions, registry.clips.values())
        registry.tracking_review = summarize_saved_tracks(registry.saved_tracks)
    token = uuid.uuid4().hex
    render_config = {
        "python": os.environ.get("FAIRWAY_UI_PYTHON") or _this_python(),
        "cv_python": cv_python or os.environ.get("FAIRWAY_BODY_PYTHON") or "",
        "body_model": body_model or os.environ.get("FAIRWAY_BODY_MODEL") or "",
        "tracking_predictions": str(Path(tracking_predictions).resolve()) if tracking_predictions else None,
        "out_root": str(out_root or RUN_ROOT / "showcases"),
        "repo_root": str(Path(__file__).resolve().parents[2]),
    }
    Path(render_config["out_root"]).mkdir(parents=True, exist_ok=True)
    handler = make_handler(registry, token, Path(config_dir or RUN_ROOT / "configs"),
                           render_config)
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"broadcast-showcase ui: {url}")
    print(f"  clips found: {len(registry.clips)} "
          f"({', '.join(c.name for c in registry.clips.values()) or 'none'})")
    print(f"  configs dir: {config_dir or RUN_ROOT / 'configs'}")
    print(f"  renders go to: {render_config['out_root']}")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def _this_python() -> str:
    import sys

    return sys.executable

