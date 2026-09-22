"""ffprobe-only source facts for the demo path (stdlib; never decodes frames).

The demo renderer plays frames back at the source's cadence, and every frame index
in the config is a *source* frame index, so the probe has to answer three questions
exactly:

* how big is the frame (bounds for the ROI, seeds and corrections),
* how many frames are there (bounds for the interval and for keyframes),
* is the cadence *constant*, and what is it (a nominal fps is not a measured
  action time, and a variable-rate source must not be timed as if it were).

Nothing here downloads, and nothing here opens a socket: one local ffprobe call.
"""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .errors import DemoSourceError

FFPROBE_TIMEOUT_SECONDS = 60
#: r_frame_rate and avg_frame_rate disagreeing by more than this share means the
#: container cadence is not one constant rate.
VARIABLE_CADENCE_TOLERANCE = 0.01


@dataclass
class SourceProbe:
    path: str
    display_name: str
    width: int
    height: int
    nominal_fps: float              # r_frame_rate: the container's declared rate
    average_fps: float              # avg_frame_rate: the rate the frames imply
    duration_seconds: float
    frame_count: Optional[int]      # None when the container does not report it
    pixel_aspect_ratio: float       # SAR, applied before any angle is computed
    container_format: str
    codec: str
    variable_cadence: bool
    cadence_note: str
    #: Per-frame PRESENTATION timestamps (seconds) of the decoded video stream, in
    #: presentation order, when they could be read. Equal r_frame_rate and
    #: avg_frame_rate do NOT prove a constant cadence -- a clip with irregular
    #: timestamps can average to exactly the nominal rate -- so the timestamps
    #: themselves are what the demo checks and reports. They are read from decoded
    #: FRAME headers, never from packets: with B-frames the packets are stored in
    #: decode order, which makes a constant-cadence clip look irregular.
    timestamps: Optional[List[float]] = None
    timestamp_stats: dict = field(default_factory=dict)
    irregular_timestamps: bool = False
    timestamp_note: str = ""
    raw_stream: dict = field(default_factory=dict)

    @property
    def fps(self) -> float:
        """The cadence used for frame<->second conversions (average, per ffprobe)."""
        return self.average_fps or self.nominal_fps

    @property
    def last_frame_index(self) -> Optional[int]:
        if self.frame_count:
            return int(self.frame_count) - 1
        return None

    def to_dict(self) -> dict:
        return {"path": self.path, "display_name": self.display_name,
                "width": self.width, "height": self.height,
                "nominal_fps": self.nominal_fps, "average_fps": self.average_fps,
                "duration_seconds": self.duration_seconds,
                "frame_count": self.frame_count,
                "pixel_aspect_ratio": self.pixel_aspect_ratio,
                "container_format": self.container_format, "codec": self.codec,
                "variable_cadence": self.variable_cadence,
                "cadence_note": self.cadence_note,
                "irregular_timestamps": self.irregular_timestamps,
                "timestamp_stats": self.timestamp_stats,
                "timestamp_note": self.timestamp_note,
                "timestamps_available": self.timestamps is not None}


def _rate(text) -> float:
    if text in (None, "", "0/0", "N/A"):
        return 0.0
    try:
        value = float(Fraction(str(text)))
    except (ValueError, ZeroDivisionError):
        return 0.0
    return value if math.isfinite(value) and value > 0 else 0.0


def _ratio(text) -> float:
    if text in (None, "", "N/A", "0:1"):
        return 1.0
    try:
        num, _, den = str(text).partition(":")
        num_f, den_f = float(num or 1), float(den or 1)
    except ValueError:
        return 1.0
    if den_f <= 0 or not math.isfinite(num_f / den_f) or num_f <= 0:
        return 1.0
    return num_f / den_f


def probe_source(video: str, *, ffprobe: str = "ffprobe") -> SourceProbe:
    """One ffprobe call, parsed strictly. Raises DemoSourceError on any surprise."""
    src = Path(video).expanduser()
    if not src.is_file():
        raise DemoSourceError(f"source video does not exist or is not a file: {video}")
    cmd = [ffprobe, "-v", "error", "-print_format", "json", "-show_streams",
           "-show_format", str(src)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=FFPROBE_TIMEOUT_SECONDS)
    except FileNotFoundError as exc:
        raise DemoSourceError(
            "ffprobe was not found on PATH; the demo path needs it to read source "
            "geometry and cadence (no decoding happens here)") from exc
    except subprocess.TimeoutExpired as exc:
        raise DemoSourceError(f"ffprobe exceeded {FFPROBE_TIMEOUT_SECONDS}s") from exc
    if p.returncode != 0:
        raise DemoSourceError(f"ffprobe failed (rc={p.returncode}): "
                              f"{(p.stderr or '').strip()[:400]}")
    try:
        payload = json.loads(p.stdout)
    except json.JSONDecodeError as exc:
        raise DemoSourceError("ffprobe produced unparsable JSON") from exc

    streams = payload.get("streams") or []
    video_stream = next((s for s in streams
                         if isinstance(s, dict) and s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise DemoSourceError(f"no video stream in {src.name}")
    fmt = payload.get("format") or {}

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise DemoSourceError(f"ffprobe reported no frame size for {src.name}")
    nominal = _rate(video_stream.get("r_frame_rate"))
    average = _rate(video_stream.get("avg_frame_rate"))
    if not nominal and not average:
        raise DemoSourceError(
            f"ffprobe reported no usable frame rate for {src.name}; the demo "
            "renderer refuses to invent a playback cadence")
    duration = 0.0
    for raw in (fmt.get("duration"), video_stream.get("duration")):
        if raw in (None, "", "N/A"):
            continue
        try:
            value = float(str(raw))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            duration = value
            break
    frames_raw = video_stream.get("nb_frames")
    frame_count: Optional[int] = None
    if frames_raw not in (None, "", "N/A"):
        try:
            candidate = int(str(frames_raw).strip())
            frame_count = candidate if candidate > 0 else None
        except ValueError:
            frame_count = None
    if not duration and frame_count and (average or nominal):
        duration = frame_count / (average or nominal)
    if duration <= 0:
        raise DemoSourceError(
            f"ffprobe reported no usable duration for {src.name}; the processed "
            "interval cannot be bounded, so the run is refused")

    reference = average or nominal
    variable = False
    if average and nominal and reference > 0:
        variable = abs(nominal - average) / reference > VARIABLE_CADENCE_TOLERANCE
    if variable:
        note = (f"container cadence is not constant: r_frame_rate={nominal:g} vs "
                f"avg_frame_rate={average:g}. Frame timing on a variable-cadence "
                f"source is not a measured action time.")
    else:
        note = (f"container cadence {reference:g} fps. Container cadence is a "
                f"playback rate, not a measured action time: the clip may be a "
                f"replay, a slow-motion capture, or a re-timed copy of another rate.")

    return SourceProbe(
        path=str(src.resolve()), display_name=src.name, width=width, height=height,
        nominal_fps=nominal, average_fps=average, duration_seconds=duration,
        frame_count=frame_count,
        pixel_aspect_ratio=_ratio(video_stream.get("sample_aspect_ratio")),
        container_format=str(fmt.get("format_name") or ""),
        codec=str(video_stream.get("codec_name") or ""),
        variable_cadence=variable, cadence_note=note,
        raw_stream={k: video_stream.get(k) for k in
                    ("width", "height", "r_frame_rate", "avg_frame_rate", "nb_frames",
                     "sample_aspect_ratio", "codec_name", "pix_fmt")},
        **_aligned_timestamps(probe_timestamps(str(src), ffprobe=ffprobe), frame_count))


#: How far a frame interval may deviate from the median before the cadence is
#: called irregular. Equal rate fields are not evidence of a constant cadence.
TIMESTAMP_TOLERANCE = 0.05
#: Above this many frames the timestamps are not held at all: the demo reports
#: that timing falls back to frames/cadence rather than holding a huge list.
MAX_TIMESTAMPS = 40000


def probe_timestamps(video: str, *, ffprobe: str = "ffprobe") -> dict:
    """Per-frame presentation timestamps of the DECODED video stream.

    ``-show_entries frame=best_effort_timestamp_time`` reads the timestamp attached
    to each decoded frame, in presentation order. Packet timestamps
    (``packet=pts_time``) are deliberately not used: ffprobe lists those in
    decode/file order, so with B-frames they go backwards and a perfectly constant
    clip looks irregular -- that mistake is what this function exists to prevent.
    JSON is used rather than CSV because CSV collides with the per-frame side-data
    lines.

    Nothing here decodes pixels, and nothing here sorts: a sorted list would hide
    exactly the reordering the check is for. If the timestamps cannot be read, the
    result says so instead of inventing a cadence.
    """
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
           "frame=best_effort_timestamp_time", "-of", "json", video]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=FFPROBE_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return timestamps_unavailable(f"frame timestamps could not be read: {exc}")
    if proc.returncode != 0:
        return timestamps_unavailable(
            "frame timestamps could not be read: " + (proc.stderr or "").strip()[:200])
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return timestamps_unavailable(
            "ffprobe returned unparsable JSON for the frame timestamps")
    records = payload.get("frames") or []
    values: List[float] = []
    missing = 0
    for record in records:
        raw = record.get("best_effort_timestamp_time") if isinstance(record, dict) else None
        if raw in (None, "", "N/A"):
            missing += 1
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            missing += 1
            continue
        if not math.isfinite(value):
            missing += 1
            continue
        values.append(value)
        if len(values) > MAX_TIMESTAMPS:
            return {"timestamps": None, "timestamp_stats": {"frames": len(values)},
                    "irregular_timestamps": False,
                    "timestamp_note": (f"the stream has more than {MAX_TIMESTAMPS} "
                                       f"decoded frames; per-frame timestamps were not "
                                       f"held, so seconds fall back to frames / cadence")}
    if not records:
        return timestamps_unavailable(
            "ffprobe reported no decoded frames, so the frame timing could not be read")
    if missing:
        return timestamps_unavailable(
            f"{missing} of {len(records)} decoded frames carry no readable presentation "
            f"timestamp, so the frame-to-second mapping is incomplete and is not used")
    return analyze_timestamps(values)


def timestamps_unavailable(note: str) -> dict:
    """The 'no usable frame timestamps' payload, with the reason attached."""
    return {"timestamps": None, "timestamp_stats": {}, "irregular_timestamps": False,
            "timestamp_note": note}


def analyze_timestamps(values: Sequence[float],
                       tolerance: float = TIMESTAMP_TOLERANCE) -> dict:
    """Decide whether per-frame presentation timestamps describe a constant cadence.

    Split out from the probe so the decision can be tested directly: equal
    ``r_frame_rate``/``avg_frame_rate`` fields do NOT prove a constant cadence, and
    a clip whose intervals vary can still average to exactly the nominal rate.

    The values must arrive in presentation order and increase strictly. They are
    never sorted and never reordered, and a non-monotonic list is reported as
    unusable rather than repaired: packet-order timestamps are exactly what a
    silent sort would have turned into a false "constant cadence, CFR" claim.
    """
    if len(values) < 3:
        return {"timestamps": None, "timestamp_stats": {"frames": len(values)},
                "irregular_timestamps": False,
                "timestamp_note": ("too few decoded frame timestamps to check the "
                                   "cadence; seconds fall back to frames / cadence")}
    for index, (a, b) in enumerate(zip(values, values[1:])):
        if not float(b) > float(a):
            return {
                "timestamps": None,
                "timestamp_stats": {"frames": len(values),
                                    "first_out_of_order_index": index,
                                    "first_pts": round(float(values[0]), 6),
                                    "last_pts": round(float(values[-1]), 6)},
                "irregular_timestamps": True,
                "timestamp_note": (
                    f"the decoded frame timestamps are not strictly increasing in "
                    f"presentation order (first break at index {index}: "
                    f"{float(values[index]):.6f}s -> {float(b):.6f}s), so they do not "
                    f"locate the frames and are not used; frame indices are authoritative")}
    deltas = [float(b) - float(a) for a, b in zip(values, values[1:])]
    ordered = sorted(deltas)
    mid = len(ordered) // 2
    median_delta = (ordered[mid] if len(ordered) % 2
                    else (ordered[mid - 1] + ordered[mid]) / 2.0)
    if median_delta <= 0:
        return {"timestamps": None, "timestamp_stats": {"frames": len(values)},
                "irregular_timestamps": True,
                "timestamp_note": "the decoded frame timestamps have a zero median "
                                  "interval, so they locate nothing"}
    worst = max(abs(delta - median_delta) / median_delta for delta in deltas)
    irregular = worst > tolerance
    stats = {"frames": len(values), "first_pts": round(values[0], 6),
             "last_pts": round(values[-1], 6),
             "median_interval": round(median_delta, 6),
             "min_interval": round(min(deltas), 6),
             "max_interval": round(max(deltas), 6),
             "worst_deviation_fraction": round(worst, 5),
             "implied_fps": round(1.0 / median_delta, 4)}
    note = (f"per-frame decoded presentation timestamps: median interval "
            f"{median_delta:.6f}s (implied {1.0 / median_delta:.4f} fps), worst deviation "
            f"{worst:.2%}; the cadence is "
            + ("NOT constant" if irregular else "constant in the decoded frame "
                                               "timestamps"))
    return {"timestamps": [float(v) for v in values], "timestamp_stats": stats,
            "irregular_timestamps": bool(irregular), "timestamp_note": note}


def timestamp_fields(payload: dict) -> dict:
    """Adapt :func:`probe_timestamps` output to ``SourceProbe`` keyword arguments."""
    return {"timestamps": payload.get("timestamps"),
            "timestamp_stats": payload.get("timestamp_stats") or {},
            "irregular_timestamps": bool(payload.get("irregular_timestamps")),
            "timestamp_note": str(payload.get("timestamp_note") or "")}


def _aligned_timestamps(payload: dict, frame_count: Optional[int]) -> dict:
    """Use the frame timestamps only when they line up with the frame indices.

    A frame index is only a frame index if ``timestamps[index]`` is THAT frame's
    timestamp. If ffprobe returns a different number of decodable frames than the
    container's ``nb_frames``, the alignment cannot be trusted, so the timestamps
    are dropped and the demo falls back to frames / cadence with the reason stated
    rather than shifting every second in the report by an unknown offset.
    """
    fields = timestamp_fields(payload)
    values = fields.get("timestamps")
    if values and frame_count and len(values) != int(frame_count):
        fields["timestamps"] = None
        fields["timestamp_note"] = (
            f"ffprobe returned {len(values)} decoded frame timestamps but the "
            f"container reports {int(frame_count)} frames, so the timestamps cannot "
            f"be aligned with the frame indices and are not used; seconds fall back "
            f"to frames / cadence")
        fields["timestamp_stats"] = dict(fields.get("timestamp_stats") or {},
                                         aligned_with_frame_count=False)
    elif values:
        fields["timestamp_stats"] = dict(fields.get("timestamp_stats") or {},
                                         aligned_with_frame_count=True)
    return fields


def frame_times_for_interval(probe: SourceProbe, start_frame: int,
                             end_frame: int) -> Optional[List[float]]:
    """Timestamps for the processed frames, or None when they are unavailable."""
    if not probe.timestamps:
        return None
    if start_frame < 0 or end_frame >= len(probe.timestamps):
        return None
    return [float(value) for value in probe.timestamps[start_frame:end_frame + 1]]


def decoded_frame_span(probe: SourceProbe, *, ffprobe: str = "ffprobe") -> Optional[int]:
    """Fallback frame count from ffprobe's frame count packet scan, or None.

    Only used when the container does not carry ``nb_frames``. It never decodes
    pixels; ``-count_frames`` would, so it is used only at the operator's request
    through ``demo-setup``, not on the analysis path.
    """
    if probe.frame_count:
        return probe.frame_count
    return None


def describe_unavailable_frame_count(probe: SourceProbe) -> List[str]:
    if probe.frame_count:
        return []
    return [f"the container does not report a frame count for {probe.display_name}; "
            f"the processed interval is bounded by duration x cadence "
            f"({probe.duration_seconds:g}s x {probe.fps:g} fps)"]
