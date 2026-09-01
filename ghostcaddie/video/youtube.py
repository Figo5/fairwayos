"""Strict, opt-in YouTube ingestion boundary (milestones Y0/Y1).

This module only turns an allowlisted public YouTube URL into a bounded local
file. It deliberately does not add a CLI command or invoke perception.
"""

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence
from urllib.parse import parse_qs, unquote, urlsplit

from .extraction import validate_video_source
from .metadata import inspect_video

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
_TRACKING = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "feature"}


class DownloadError(ValueError):
    """A stable, sanitized URL or download-boundary failure."""

    def __init__(self, message: str, code: str):
        self.code = code
        super().__init__(message)


class DownloaderUnavailable(DownloadError):
    """No explicitly configured executable is available; use manual download."""

    def __init__(self, message: str = "configured yt-dlp is unavailable; download manually and use the local-video workflow"):
        super().__init__(message, "downloader_unavailable")


@dataclass(frozen=True)
class YouTubeSource:
    platform: str
    video_id: str
    # Kept as an intentionally empty in-memory compatibility slot. It is never
    # populated or serialized, so provenance cannot leak the submitted URL.
    normalized_url: Optional[str] = None

    def __post_init__(self) -> None:
        if self.platform != "youtube" or not _VIDEO_ID.fullmatch(self.video_id):
            raise DownloadError("invalid YouTube source", "invalid_video_id")

    def to_dict(self) -> Dict[str, str]:
        return {"platform": self.platform, "video_id": self.video_id}


@dataclass(frozen=True)
class DownloadLimits:
    max_duration_seconds: float = 3600.0
    max_segment_seconds: float = 30.0
    max_download_bytes: int = 2 * 1024 * 1024 * 1024
    max_disk_bytes: int = 3 * 1024 * 1024 * 1024
    timeout_seconds: float = 900.0


@dataclass(frozen=True)
class YouTubeDownloadResult:
    source: YouTubeSource
    path: str
    downloader: str = "yt-dlp"
    status: str = "downloaded"

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source.to_dict(), "downloader": self.downloader, "status": self.status}


def parse_youtube_url(url: str) -> YouTubeSource:
    """Parse exactly one public HTTPS YouTube video URL."""
    if not isinstance(url, str) or any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise DownloadError("URL contains unsafe characters", "unsupported_youtube_url_form")
    if url.startswith("@url:"):
        raise DownloadError("URL placeholder must be resolved before download", "unresolved_url_placeholder")
    try:
        parts = urlsplit(url)
    except ValueError:
        raise DownloadError("invalid URL", "unsupported_youtube_url_form")
    if parts.scheme.lower() != "https":
        raise DownloadError("only HTTPS YouTube URLs are supported", "unsupported_url_scheme")
    try:
        if parts.username is not None or parts.password is not None or parts.port is not None:
            raise DownloadError("URL credentials and ports are not allowed", "unsupported_youtube_url_form")
    except ValueError as exc:
        raise DownloadError("URL credentials and ports are not allowed", "unsupported_youtube_url_form") from exc
    try:
        host = (parts.hostname or "").lower().rstrip(".")
    except ValueError as exc:
        raise DownloadError("URL host is not allowlisted", "unsupported_url_host") from exc
    if host not in _HOSTS:
        raise DownloadError("URL host is not allowlisted", "unsupported_url_host")
    if parts.fragment:
        raise DownloadError("URL fragments are not supported", "unsupported_youtube_url_form")

    query = parse_qs(parts.query, keep_blank_values=True)
    if any(key.lower() in {"list", "playlist", "channel", "search_query"} for key in query):
        raise DownloadError("playlists are not allowed", "playlist_not_allowed")
    if any(key not in {"v"} and key not in _TRACKING for key in query):
        raise DownloadError("unsupported YouTube URL parameters", "unsupported_youtube_url_form")

    if host == "youtu.be":
        if query.get("v") or parts.path.count("/") != 1:
            raise DownloadError("unsupported YouTube URL form", "unsupported_youtube_url_form")
        raw_id = parts.path.lstrip("/")
    else:
        if parts.path != "/watch" or len(query.get("v", [])) != 1:
            raise DownloadError("only YouTube watch URLs are supported", "unsupported_youtube_url_form")
        raw_id = query["v"][0]
    video_id = unquote(raw_id)
    if not _VIDEO_ID.fullmatch(video_id):
        raise DownloadError("video ID must be a canonical 11-character ID", "invalid_video_id")
    return YouTubeSource("youtube", video_id)


def configured_yt_dlp(executable: Optional[str]) -> str:
    """Validate only an explicitly configured executable; never search PATH."""
    if not executable or not isinstance(executable, str):
        raise DownloaderUnavailable()
    path = Path(executable).expanduser()
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or not os.access(str(resolved), os.X_OK):
            raise DownloaderUnavailable()
        return str(resolved)
    except (OSError, RuntimeError):
        raise DownloaderUnavailable()


def _safe_error(stderr: str) -> str:
    text = re.sub(r"https?://\S+", "[url]", stderr or "")
    text = re.sub(r"(?i)(authorization|cookie|password|token)\s*[:=]\s*\S+", r"\1=[redacted]", text)
    return text[:240].strip() or "downloader reported an error"


class YtDlpDownloader:
    """Bounded subprocess adapter for an explicitly configured yt-dlp."""

    def __init__(self, executable: Optional[str], *, limits: DownloadLimits = DownloadLimits(),
                 runner: Callable[..., Any] = subprocess.run, js_runtime: Optional[str] = None,
                 format_selector: str = "worst"):
        self.executable = configured_yt_dlp(executable)
        self.limits = limits
        self.runner = runner
        self.js_runtime = None
        if js_runtime is not None:
            self.js_runtime = configured_yt_dlp(js_runtime)
        if not isinstance(format_selector, str) or not re.fullmatch(r"[A-Za-z0-9_.,+\[\]=*/<>:-]+", format_selector):
            raise DownloadError("download format selector is unsafe", "unsafe_format")
        self.format_selector = format_selector

    def _run(self, argv: Sequence[str], cwd: Path) -> Any:
        try:
            return self.runner(list(argv), cwd=str(cwd), capture_output=True, text=True,
                               check=False, timeout=self.limits.timeout_seconds, shell=False)
        except subprocess.TimeoutExpired as exc:
            raise DownloadError("downloader timed out", "download_timeout") from exc
        except (OSError, TypeError) as exc:
            raise DownloaderUnavailable() from exc

    def download(self, url: str, output_directory: str, *, segment_start: float = 0.0,
                 segment_duration: Optional[float] = None) -> YouTubeDownloadResult:
        source = parse_youtube_url(url)
        if isinstance(segment_start, bool) or not isinstance(segment_start, (int, float)) or not math.isfinite(segment_start) or segment_start < 0:
            raise DownloadError("segment start must be non-negative", "invalid_segment")
        if segment_duration is not None and (isinstance(segment_duration, bool) or
                not isinstance(segment_duration, (int, float)) or not math.isfinite(segment_duration) or segment_duration <= 0):
            raise DownloadError("segment duration must be positive", "invalid_segment")
        duration = self.limits.max_segment_seconds if segment_duration is None else float(segment_duration)
        if duration > self.limits.max_segment_seconds:
            raise DownloadError("segment duration exceeds configured limit", "segment_limit_exceeded")
        segment_end = float(segment_start) + duration
        requested_output = Path(output_directory).expanduser()
        if not requested_output.is_absolute():
            requested_output = Path.cwd() / requested_output
        for ancestor in (requested_output,) + tuple(requested_output.parents):
            try:
                if ancestor.exists() and ancestor.is_symlink() and ancestor != Path("/var"):
                    raise DownloadError("output directory contains a symlinked ancestor", "unsafe_output")
            except OSError as exc:
                raise DownloadError("output directory is unsafe", "unsafe_output") from exc
        output = requested_output.resolve()
        if output.exists() and not output.is_dir():
            raise DownloadError("output directory is unsafe", "unsafe_output")
        output.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".youtube-", dir=str(output)))
        try:
            metadata_args = [self.executable, "--no-playlist", "--dump-single-json", "--skip-download", url]
            probe = self._run(metadata_args, staging)
            if probe.returncode != 0:
                raise DownloadError("video is unavailable: " + _safe_error(probe.stderr), "unavailable")
            try:
                metadata = json.loads(probe.stdout)
            except (TypeError, ValueError) as exc:
                raise DownloadError("downloader returned invalid metadata", "malformed_metadata") from exc
            if not isinstance(metadata, dict):
                raise DownloadError("downloader returned invalid metadata", "malformed_metadata")
            returned_id = metadata.get("id")
            if not isinstance(returned_id, str) or not _VIDEO_ID.fullmatch(returned_id) or returned_id != source.video_id:
                raise DownloadError("downloader returned invalid video identity", "malformed_metadata")
            if metadata.get("is_live") or metadata.get("live_status") in {"is_live", "is_upcoming", "post_live"}:
                raise DownloadError("live or upcoming videos are not allowed", "live_not_allowed")
            if metadata.get("is_private"):
                raise DownloadError("video is unavailable", "protected_content")
            availability = str(metadata.get("availability", "")).lower()
            detail = " ".join(str(metadata.get(key, "")) for key in ("format", "error", "availability")).lower()
            if availability in {"private", "unavailable", "needs_auth"}:
                raise DownloadError("video is unavailable", "protected_content" if availability == "private" else "unavailable")
            if any(word in detail for word in ("login", "authentication", "sign in", "protected", "drm", "age-restricted")):
                raise DownloadError("video requires protected access", "protected_content")
            duration = metadata.get("duration")
            if duration is not None:
                if isinstance(duration, bool):
                    raise DownloadError("downloader returned invalid duration", "malformed_metadata")
                try:
                    parsed_duration = float(duration)
                except (TypeError, ValueError) as exc:
                    raise DownloadError("downloader returned invalid duration", "malformed_metadata") from exc
                if not math.isfinite(parsed_duration) or parsed_duration <= 0:
                    raise DownloadError("downloader returned invalid duration", "malformed_metadata")
                if parsed_duration > self.limits.max_duration_seconds:
                    raise DownloadError("video duration exceeds configured limit", "duration_limit_exceeded")
            estimated_key = "filesize" if "filesize" in metadata else "filesize_approx"
            estimated = metadata.get(estimated_key)
            if estimated is not None:
                if isinstance(estimated, bool):
                    raise DownloadError("downloader returned invalid filesize", "malformed_metadata")
                try:
                    parsed_size = float(estimated)
                except (TypeError, ValueError) as exc:
                    raise DownloadError("downloader returned invalid filesize", "malformed_metadata") from exc
                if not math.isfinite(parsed_size) or parsed_size < 0 or not parsed_size.is_integer():
                    raise DownloadError("downloader returned invalid filesize", "malformed_metadata")
                if int(parsed_size) > self.limits.max_download_bytes:
                    raise DownloadError("estimated download size exceeds configured limit", "size_limit_exceeded")

            template = staging / "source.download.part.%(ext)s"
            args = [self.executable, "--no-playlist", "--no-progress"]
            if self.js_runtime is not None:
                args.extend(["--js-runtimes", "node:" + self.js_runtime])
            args.extend(["--download-sections",
                    "*" + format(float(segment_start), "g") + "-" + format(segment_end, "g"), "--format",
                    self.format_selector, "-o", str(template), url])
            completed = self._run(args, staging)
            if completed.returncode != 0:
                raise DownloadError("download failed: " + _safe_error(completed.stderr), "download_failed")
            candidates = list(staging.glob("source.download.part.*"))
            if len(candidates) != 1:
                raise DownloadError("downloader produced an unexpected output", "unsafe_output")
            part = candidates[0]
            if part.is_symlink() or not part.is_file():
                raise DownloadError("downloaded output is not a regular file", "unsafe_output")
            if part.stat().st_size > self.limits.max_download_bytes:
                raise DownloadError("download exceeds configured size limit", "size_limit_exceeded")
            if sum(p.stat().st_size for p in staging.rglob("*") if p.is_file() and not p.is_symlink()) > self.limits.max_disk_bytes:
                raise DownloadError("staging disk limit exceeded", "disk_limit_exceeded")
            try:
                validate_video_source(str(part))
                inspect_video(str(part))
            except Exception as exc:
                # Do not expose local paths or ffprobe/downloader internals.
                raise DownloadError("downloaded video failed local verification", "invalid_media") from exc
            final = output / "source.mp4"
            if final.exists() or final.is_symlink():
                raise DownloadError("output file already exists", "unsafe_output")
            os.replace(str(part), str(final))
            provenance = output / "source_metadata.json"
            metadata_tmp = staging / "source_metadata.json.part"
            metadata_tmp.write_text(json.dumps({
                "schema_version": "youtube-source.v1",
                "source": source.to_dict(),
                "downloader": "yt-dlp",
                "status": "downloaded",
            }, sort_keys=True, indent=2) + "\n")
            os.replace(str(metadata_tmp), str(provenance))
            return YouTubeDownloadResult(source, str(final))
        finally:
            shutil.rmtree(str(staging), ignore_errors=True)
