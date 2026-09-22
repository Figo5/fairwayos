"""The canonical FairwayOS video-analysis workflow: schema, orchestration, report.

One local video in, one fixed output contract out. This module is deliberately
**stdlib only**: it imports no cv2, no numpy, no model runtime and opens no
socket. All decoding, pose inference and rendering happen in
``ghostcaddie.video.canonical_worker``, dispatched out of process into the
interpreter that actually has those dependencies. That split is what lets the
whole contract -- schema, metric sufficiency, layer states, report -- be tested
on a bare interpreter with no optional model assets present.

What this workflow claims today, and nothing more:

* **body** -- MoveNet SinglePose Lightning over a crop whose identity has been
  established locally, or an explicit operator ROI. If golfer identity cannot be
  established, body is AMBIGUOUS or UNAVAILABLE. Whole-frame single-pose output
  is never silently called "the golfer".
* **ball** -- UNAVAILABLE. Every candidate route is falsified or withdrawn.
* **clubhead** -- UNAVAILABLE, and blocked at the data layer, not the code layer.
* **physical metrics** -- UNAVAILABLE without validated calibration.

See ``docs/canonical-video-pipeline-audit.md`` for the evidence behind each.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import model_paths
from .source_binding import bind_source, sha256_file, SourceBindingError

SCHEMA_VERSION = "fairwayos-analysis.v1"
WORKER_TIMEOUT_SECONDS = 900

# Bounded processing defaults. A canonical run must terminate on any input.
DEFAULT_SAMPLE_FPS = 6.0
DEFAULT_MAX_FRAMES = 180


class LayerState:
    """The only states a perception layer may report."""
    SUPPORTED = "supported"
    PARTIAL = "partial"
    AMBIGUOUS = "ambiguous"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


VALID_LAYER_STATES = frozenset({LayerState.SUPPORTED, LayerState.PARTIAL,
                                LayerState.AMBIGUOUS, LayerState.UNAVAILABLE,
                                LayerState.FAILED})

#: A layer in one of these states carries no usable observations.
EMPTY_LAYER_STATES = frozenset({LayerState.AMBIGUOUS, LayerState.UNAVAILABLE,
                                LayerState.FAILED})


class MetricState:
    SUPPORTED = "supported"
    UNAVAILABLE = "unavailable"


class AnalysisError(RuntimeError):
    """The canonical run could not be completed. Never a silent empty success."""


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

@dataclass
class SourceInfo:
    display_name: str
    path: str
    sha256: str
    duration_seconds: float
    fps: float
    width: int
    height: int
    container_format: str = ""
    codec: str = ""
    frame_range: Sequence[int] = (0, 0)     # inclusive native frame indices processed
    sample_fps: float = 0.0
    sampled_frame_count: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["frame_range"] = list(self.frame_range)
        d["resolution"] = f"{self.width}x{self.height}"
        return d


@dataclass
class Layer:
    name: str
    state: str
    method: str
    provenance: Dict[str, Any] = field(default_factory=dict)
    coverage: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    diagnostics_ref: str = ""

    def __post_init__(self):
        if self.state not in VALID_LAYER_STATES:
            raise AnalysisError(f"{self.name}: {self.state!r} is not a layer state; "
                                f"valid: {sorted(VALID_LAYER_STATES)}")

    @property
    def has_observations(self) -> bool:
        return self.state not in EMPTY_LAYER_STATES

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Metric:
    name: str
    state: str
    reason: str = ""
    unit: str = ""
    space: str = "2d_image"
    median: Optional[float] = None
    p10: Optional[float] = None
    p90: Optional[float] = None
    frames_used: int = 0
    frames_possible: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnalysisResult:
    source: SourceInfo
    layers: Dict[str, Layer]
    # {"body_2d": {"supported": [...], "unavailable": [...]}, "physical": {...}}
    metrics: Dict[str, Dict[str, List[Metric]]]
    research_status: Dict[str, Any]
    outputs: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    created_utc: str = ""

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "created_utc": self.created_utc or _utcnow(),
            "source": self.source.to_dict(),
            "layers": {k: v.to_dict() for k, v in self.layers.items()},
            "metrics": {group: {kind: [m.to_dict() for m in ms]
                                for kind, ms in buckets.items()}
                        for group, buckets in self.metrics.items()},
            "research_status": dict(self.research_status),
            "outputs": dict(self.outputs),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisResult":
        if d.get("schema_version") != SCHEMA_VERSION:
            raise AnalysisError(f"unsupported schema_version {d.get('schema_version')!r}")
        src = dict(d["source"])
        src.pop("resolution", None)
        return cls(
            source=SourceInfo(**src),
            layers={k: Layer(**v) for k, v in d["layers"].items()},
            metrics={group: {kind: [Metric(**m) for m in ms]
                             for kind, ms in buckets.items()}
                     for group, buckets in d["metrics"].items()},
            research_status=dict(d["research_status"]),
            outputs=dict(d.get("outputs") or {}),
            schema_version=d["schema_version"],
            created_utc=d.get("created_utc", ""))


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------
# the layers that are unavailable, and exactly why
# --------------------------------------------------------------------------

BALL_UNAVAILABLE_REASON = (
    "No ball route qualifies. The heuristic tracker was withdrawn after it was "
    "shown to follow the flagstick base rather than a ball; the BootsTAPIR point "
    "tracker was falsified for transfer under a hash-bound rerun (1/51 and 0/71 "
    "on the corrected clips); SAM2 video segmentation needs a human seed per clip "
    "and is unproven on an object this small and this motion-blurred.")

CLUBHEAD_UNAVAILABLE_REASON = (
    "No clubhead route qualifies, and the blocker is the data, not the code. MIL "
    "region tracking failed transfer with frozen settings, locking to the body for "
    "16 consecutive frames; appearance post-filters were killed by a second "
    "false-lock object; no rights-cleared clubhead detector or annotated corpus "
    "exists to train against; and clubhead-through-impact displacement at the "
    "available frame rates exceeds the tracker design point.")


def ball_layer() -> Layer:
    """The canonical ball layer. There is no configuration that turns this on."""
    return Layer(
        name="ball", state=LayerState.UNAVAILABLE, method="none",
        provenance={"routes_considered": ["heuristic_circle_contrast",
                                          "bootstapir_point_track",
                                          "sam2_video_segmentation"],
                    "routes_admitted": [],
                    "status": "withdrawn/falsified; retained as research evidence only"},
        coverage={"frames_with_observation": 0, "frames_processed": 0, "fraction": 0.0},
        limitations=[BALL_UNAVAILABLE_REASON],
        diagnostics_ref="docs/canonical-video-pipeline-audit.md#3-ball-path--unavailable")


def clubhead_layer() -> Layer:
    return Layer(
        name="clubhead", state=LayerState.UNAVAILABLE, method="none",
        provenance={"routes_considered": ["mil_region_track", "reacquire_color",
                                          "sam2_head_segmentation"],
                    "routes_admitted": [],
                    "status": "transfer FAILED; blocked at the data layer"},
        coverage={"frames_with_observation": 0, "frames_processed": 0, "fraction": 0.0},
        limitations=[CLUBHEAD_UNAVAILABLE_REASON],
        diagnostics_ref="docs/canonical-video-pipeline-audit.md#4-clubhead-path--unavailable")


# --------------------------------------------------------------------------
# 2D image-space body metrics
# --------------------------------------------------------------------------

#: (metric name, required landmark names, unit). Nothing here is 3D and nothing
#: here is clinical: these are angles between image-space pixel coordinates.
BODY_METRIC_SPECS = (
    ("shoulder_tilt", ("sh_l", "sh_r"), "degrees_from_horizontal"),
    ("hip_tilt", ("hip_l", "hip_r"), "degrees_from_horizontal"),
    ("left_knee_flexion", ("hip_l", "kn_l", "ank_l"), "degrees"),
    ("right_knee_flexion", ("hip_r", "kn_r", "ank_r"), "degrees"),
    ("left_elbow_angle", ("sh_l", "el_l", "wr_l"), "degrees"),
    ("right_elbow_angle", ("sh_r", "el_r", "wr_r"), "degrees"),
)

#: A metric needs its landmarks on at least this fraction of the valid body
#: frames, and at least this many frames outright, before it is reported at all.
MIN_METRIC_FRAME_FRACTION = 0.6
MIN_METRIC_FRAMES = 5

BODY_METRIC_DISCLAIMER = (
    "2D image-space estimate from a single uncalibrated camera. It measures the "
    "angle between projected pixel coordinates, not an anatomical joint angle: "
    "it changes with camera position and it is not a clinical or coaching "
    "measurement.")


def _visible(frame: dict) -> Dict[str, dict]:
    return {p["name"]: p for p in frame.get("keypoints", []) if p.get("visible")}


def _tilt_degrees(a: dict, b: dict) -> float:
    """Signed tilt of the a-b segment away from image horizontal, in (-90, 90].

    The segment is a line, not an arrow: left-to-right and right-to-left give the
    same tilt. Without this fold a shoulder line reads as -137 degrees whenever
    the golfer faces the other way, which is a direction, not a tilt.
    """
    deg = math.degrees(math.atan2(b["y"] - a["y"], b["x"] - a["x"]))
    while deg > 90:
        deg -= 180
    while deg <= -90:
        deg += 180
    return deg


def _joint_degrees(a: dict, b: dict, c: dict) -> Optional[float]:
    """Interior angle at b, in degrees. None when a segment has zero length."""
    v1 = (a["x"] - b["x"], a["y"] - b["y"])
    v2 = (c["x"] - b["x"], c["y"] - b["y"])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return None
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def compute_body_metrics(body_frames: Sequence[dict],
                         layer_state: str) -> Dict[str, List[Metric]]:  # noqa: D401
    """Conservative 2D metrics with per-metric landmark sufficiency checks.

    ``body_frames`` must already be filtered to frames whose pose was accepted as
    the golfer. A metric that lacks its landmarks on enough of those frames is
    reported UNAVAILABLE with the count, never estimated from the rest.
    """
    supported: List[Metric] = []
    unavailable: List[Metric] = []
    possible = len(body_frames)

    if layer_state in EMPTY_LAYER_STATES:
        reason = (f"body layer is {layer_state}: no frame is confidently attributed "
                  f"to the golfer, so no metric may be computed across them")
        for name, _lm, unit in BODY_METRIC_SPECS:
            unavailable.append(Metric(name, MetricState.UNAVAILABLE, reason, unit,
                                      frames_possible=possible))
        return {"supported": supported, "unavailable": unavailable}

    for name, landmarks, unit in BODY_METRIC_SPECS:
        values: List[float] = []
        for fr in body_frames:
            vis = _visible(fr)
            if not all(lm in vis for lm in landmarks):
                continue
            pts = [vis[lm] for lm in landmarks]
            v = (_tilt_degrees(*pts) if len(pts) == 2 else _joint_degrees(*pts))
            if v is not None and math.isfinite(v):
                values.append(v)
        need = max(MIN_METRIC_FRAMES, math.ceil(MIN_METRIC_FRAME_FRACTION * possible))
        if len(values) < need:
            unavailable.append(Metric(
                name, MetricState.UNAVAILABLE,
                reason=(f"required landmarks {'+'.join(landmarks)} were confidently "
                        f"visible on only {len(values)} of {possible} attributed "
                        f"frames; {need} are required"),
                unit=unit, frames_used=len(values), frames_possible=possible))
            continue
        values.sort()
        supported.append(Metric(
            name, MetricState.SUPPORTED, reason=BODY_METRIC_DISCLAIMER, unit=unit,
            median=round(statistics.median(values), 2),
            p10=round(_percentile(values, 0.10), 2),
            p90=round(_percentile(values, 0.90), 2),
            frames_used=len(values), frames_possible=possible))
    return {"supported": supported, "unavailable": unavailable}


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise AnalysisError("percentile of an empty sample")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo))


# --------------------------------------------------------------------------
# physical shot metrics
# --------------------------------------------------------------------------

#: (metric, unit, reason it is unavailable without calibration)
PHYSICAL_METRIC_SPECS = (
    ("ball_speed", "m/s", "ball layer unavailable, and no validated physical scale"),
    ("clubhead_speed", "m/s", "clubhead layer unavailable, and no validated physical scale"),
    ("launch_angle", "degrees", "ball layer unavailable, and no validated physical scale"),
    ("carry_distance", "m", "ball layer unavailable, and no validated physical scale"),
    ("apex_height", "m", "ball layer unavailable, and no validated physical scale"),
    # Spin is listed explicitly rather than left absent, so that "spin is not in
    # the output" cannot be misread as an oversight or as a metric that is
    # merely waiting for a flag. It has no route to `supported` at all: nothing
    # on this path observes impact, and no rotation measurement or spin
    # calibration has ever been validated -- restoring a checkpoint, a
    # nominal container FPS or a calibration flag does not change that.
    ("spin_rate", "rpm",
     "no impact is observed (ball and clubhead layers unavailable), and no "
     "validated rotation measurement or spin calibration exists"),
)

NO_ACTION_TIME_NOTE = (
    "Nominal container FPS is not validated action time. A speed derived from it "
    "would inherit any capture/container rate mismatch as a silent scale error, so "
    "no speed is reported from frame indices alone.")


def physical_metrics(calibration: Optional[dict] = None) -> Dict[str, List[Metric]]:
    """Physical metrics stay unavailable unless real calibration was supplied.

    No calibration file has ever been validated against a canonical run, so the
    supported list is empty by construction today; the parameter exists so the
    reason is data-driven rather than a hardcoded string.
    """
    valid = bool(calibration and calibration.get("validated_physical_scale")
                 and calibration.get("validated_action_time"))
    unavailable = []
    for name, unit, reason in PHYSICAL_METRIC_SPECS:
        full = reason if not valid else "calibration supplied but no ball/clubhead observations exist"
        unavailable.append(Metric(name, MetricState.UNAVAILABLE,
                                  reason=f"{full}. {NO_ACTION_TIME_NOTE}",
                                  unit=unit, space="physical"))
    return {"supported": [], "unavailable": unavailable}


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def body_interpreter_path(explicit: Optional[str] = None) -> str:
    """Interpreter that can run MoveNet TFLite. Explicit -> env -> registry."""
    if explicit:
        return explicit
    if os.environ.get("FAIRWAY_BODY_PYTHON"):
        return os.environ["FAIRWAY_BODY_PYTHON"]
    from ghostcaddie.upload.runtimes import RuntimeRegistry
    return RuntimeRegistry.default().specs["body"].interpreter


def _out_paths(out_dir: Path) -> Dict[str, Path]:
    return {"annotated": out_dir / "annotated.mp4",
            "report": out_dir / "report.md",
            "analysis": out_dir / "analysis.json",
            "provenance": out_dir / "provenance.json",
            "diagnostics": out_dir / "diagnostics.json",
            "debug_dir": out_dir / "debug"}


def analyze_video(video: str, out_dir: str, *,
                  sample_fps: float = DEFAULT_SAMPLE_FPS,
                  max_frames: int = DEFAULT_MAX_FRAMES,
                  start_seconds: float = 0.0,
                  end_seconds: Optional[float] = None,
                  roi: Optional[Sequence[int]] = None,
                  body_model: Optional[str] = None,
                  body_interpreter: Optional[str] = None,
                  render: bool = True,
                  expected_sha256: Optional[str] = None,
                  redact_path: bool = False) -> AnalysisResult:
    """Run the canonical local analysis. Local input only; no network at all.

    Raises AnalysisError rather than writing an empty success when the source is
    unusable, the body interpreter cannot be reached, or rendering fails.
    """
    from .metadata import inspect_video

    src = Path(video).expanduser()
    if not src.is_file():
        raise AnalysisError(f"source video does not exist or is not a file: {video}")
    if not os.access(src, os.R_OK):
        raise AnalysisError(f"source video is not readable: {video}")
    if sample_fps <= 0 or not math.isfinite(sample_fps):
        raise AnalysisError("sample_fps must be a positive finite number")
    if max_frames <= 0:
        raise AnalysisError("max_frames must be positive: processing is bounded")

    sha = sha256_file(src)
    # bind_source is the mandatory precondition of every canonical run: media
    # bytes must equal what the caller expects before any model is touched.
    binding = bind_source(src, expected_sha256 or sha)

    meta = inspect_video(str(src))
    end = meta.duration_seconds if end_seconds is None else float(end_seconds)
    if end <= start_seconds:
        raise AnalysisError("end_seconds must be greater than start_seconds")

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    paths = _out_paths(out)
    paths["debug_dir"].mkdir(exist_ok=True)

    resolved_model = model_paths.resolve("body", body_model)
    interpreter = body_interpreter_path(body_interpreter)

    spec = {
        "video": str(src.resolve()),
        "source_sha256": sha,
        "sample_fps": float(sample_fps),
        "max_frames": int(max_frames),
        "start_seconds": float(start_seconds),
        "end_seconds": float(end),
        "roi": list(roi) if roi else None,
        "model_path": resolved_model.path,
        "model_found": resolved_model.found,
        "render": bool(render),
        "annotated_path": str(paths["annotated"]),
        "debug_dir": str(paths["debug_dir"]),
        "native_fps": float(meta.frame_rate),
    }
    worker = _run_body_worker(spec, interpreter)

    body = _body_layer(worker, resolved_model, interpreter, roi)
    layers = {"body": body, "ball": ball_layer(), "clubhead": clubhead_layer()}

    attributed = worker.get("attributed_frames") or []
    body_metrics = compute_body_metrics(attributed, body.state)
    phys = physical_metrics(None)
    metrics = {"body_2d": body_metrics, "physical": phys}

    source_info = SourceInfo(
        display_name=src.name,
        path="" if redact_path else str(src.resolve()),
        sha256=sha,
        duration_seconds=round(meta.duration_seconds, 3),
        fps=round(meta.frame_rate, 4),
        width=meta.width, height=meta.height,
        container_format=meta.container_format, codec=meta.codec,
        frame_range=worker.get("frame_range", [0, 0]),
        sample_fps=float(sample_fps),
        sampled_frame_count=int(worker.get("sampled_frame_count", 0)))

    annotated_ok = bool(worker.get("render", {}).get("written"))
    result = AnalysisResult(
        source=source_info, layers=layers, metrics=metrics,
        research_status={
            "experimental_components_used": [],
            "pseudo_labels_present": False,
            "production_eligible": False,
            "ground_truth": False,
            "note": ("This is a local research workflow. No component on this path "
                     "is production-eligible and no output is ground truth."),
        },
        outputs={
            "annotated_video": str(paths["annotated"]) if annotated_ok else None,
            "annotated_video_written": annotated_ok,
            "annotated_video_reason": worker.get("render", {}).get("reason", ""),
            "report": str(paths["report"]),
            "analysis": str(paths["analysis"]),
            "provenance": str(paths["provenance"]),
            "diagnostics": str(paths["diagnostics"]),
        },
        created_utc=_utcnow())

    paths["analysis"].write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    paths["provenance"].write_text(json.dumps(
        _provenance(result, binding, resolved_model, interpreter, spec, worker),
        indent=2, sort_keys=True) + "\n")
    paths["diagnostics"].write_text(json.dumps(
        worker.get("diagnostics", {}), indent=2, sort_keys=True) + "\n")
    paths["report"].write_text(render_report(result))
    return result


def _run_body_worker(spec: dict, interpreter: str) -> dict:
    """Dispatch decode + identity + pose + render out of process.

    The worker is the only component that imports cv2, numpy or a model runtime.
    Its stdout is one JSON document; a non-zero exit is an AnalysisError, never
    an empty-but-successful result.
    """
    if not interpreter or not os.path.isfile(interpreter) or not os.access(interpreter, os.X_OK):
        raise AnalysisError(
            f"body interpreter is not executable: {interpreter!r}. Point "
            f"$FAIRWAY_BODY_PYTHON at an interpreter that has ai_edge_litert, cv2 "
            f"and numpy (see docs/video-analysis-user-guide.md).")
    env = dict(os.environ)
    # Offline by construction: nothing on this path may reach a network, and a
    # stray proxy variable must not make that look possible.
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        env.pop(var, None)
    env["FAIRWAYOS_OFFLINE"] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [interpreter, "-m", "ghostcaddie.video.canonical_worker"]
    try:
        p = subprocess.run(cmd, input=json.dumps(spec), capture_output=True,
                           text=True, timeout=WORKER_TIMEOUT_SECONDS, env=env)
    except subprocess.TimeoutExpired as exc:
        raise AnalysisError(f"body worker exceeded the {WORKER_TIMEOUT_SECONDS}s "
                            f"bound and was terminated") from exc
    except OSError as exc:
        raise AnalysisError(f"unable to launch the body worker: {exc}") from exc
    if p.returncode != 0:
        raise AnalysisError(f"body worker failed (rc={p.returncode}): "
                            f"{(p.stderr or '').strip()[-600:]}")
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"body worker produced unparsable output: "
                            f"{(p.stdout or '')[:300]}") from exc


def _body_layer(worker: dict, resolved_model, interpreter: str,
                roi: Optional[Sequence[int]]) -> Layer:
    ident = worker.get("identity", {})
    state = ident.get("layer_state", LayerState.FAILED)
    if state not in VALID_LAYER_STATES:
        state = LayerState.FAILED
    processed = int(worker.get("sampled_frame_count", 0))
    attributed = len(worker.get("attributed_frames") or [])
    limitations = list(ident.get("limitations", []))
    if not resolved_model.found:
        limitations.append(resolved_model.reason)
    return Layer(
        name="body", state=state,
        method=ident.get("method", "none"),
        provenance={
            "model": resolved_model.to_dict(),
            "interpreter": interpreter,
            "runtime": "ai_edge_litert (TFLite flatbuffer; no torch, no pickle)",
            "identity_method": ident.get("identity_method", "none"),
            "operator_roi_supplied": bool(roi),
            "roi_xywh": ident.get("roi_xywh"),
            "subject_box_xywh": ident.get("subject_box_xywh"),
            "second_person": ident.get("second_person"),
            "keypoint_score_threshold": ident.get("score_threshold"),
        },
        coverage={"frames_processed": processed,
                  "frames_attributed_to_golfer": attributed,
                  "fraction": round(attributed / processed, 4) if processed else 0.0,
                  "camera_jitter_px_median": ident.get("jitter_px_median")},
        limitations=limitations,
        diagnostics_ref="diagnostics.json#identity")


def _provenance(result: AnalysisResult, binding: dict, resolved_model,
                interpreter: str, spec: dict, worker: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": result.created_utc,
        "source_binding": binding,
        "source": result.source.to_dict(),
        "processing_bounds": {
            "sample_fps": spec["sample_fps"], "max_frames": spec["max_frames"],
            "start_seconds": spec["start_seconds"], "end_seconds": spec["end_seconds"],
            "frame_range": result.source.frame_range},
        "methods": {
            "body": {"model": resolved_model.to_dict(),
                     "identity": worker.get("identity", {}).get("identity_method"),
                     "runtime": "ai_edge_litert"},
            "ball": {"method": None, "reason": BALL_UNAVAILABLE_REASON},
            "clubhead": {"method": None, "reason": CLUBHEAD_UNAVAILABLE_REASON},
        },
        "environment": {
            "orchestrator_python": sys.version.split()[0],
            "worker_interpreter": interpreter,
            "worker_python": worker.get("environment", {}).get("python"),
            "worker_versions": worker.get("environment", {}).get("versions", {}),
            "ffprobe": shutil.which("ffprobe") or "",
            "network_access": "none; the canonical path opens no socket and "
                              "resolves no remote host",
        },
        "model_discovery": model_paths.describe(),
        "research_status": result.research_status,
    }


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def _fmt_metric(m: Metric) -> str:
    if m.state == MetricState.SUPPORTED:
        return (f"- **{m.name}** — median {m.median}° (p10 {m.p10}°, p90 {m.p90}°), "
                f"{m.frames_used}/{m.frames_possible} frames · {m.unit}")
    return f"- **{m.name}** — unavailable: {m.reason}"


def render_report(result: AnalysisResult) -> str:
    s = result.source
    body = result.layers["body"]
    lines = [
        f"# FairwayOS canonical video analysis — {s.display_name}",
        "",
        f"_{result.created_utc} · schema `{result.schema_version}`_",
        "",
        "## VIDEO", "",
        f"- source: `{s.display_name}`" + (f" (`{s.path}`)" if s.path else ""),
        f"- sha256: `{s.sha256}`",
        f"- duration: {s.duration_seconds:g}s · {s.width}×{s.height} · {s.fps:g} fps "
        f"({s.container_format or '?'} / {s.codec or '?'})",
        f"- processed interval: native frames {s.frame_range[0]}–{s.frame_range[1]}, "
        f"sampled at {s.sample_fps:g} fps → {s.sampled_frame_count} frames",
        "",
        "## BODY", "",
        f"- state: **{body.state.upper()}**",
        f"- method: {body.method}",
        f"- identity: {body.provenance.get('identity_method')}",
        f"- coverage: {body.coverage.get('frames_attributed_to_golfer')}"
        f"/{body.coverage.get('frames_processed')} sampled frames attributed to the "
        f"golfer ({body.coverage.get('fraction')})",
        f"- camera jitter (median px between sampled frames): "
        f"{body.coverage.get('camera_jitter_px_median')}",
        "",
        "### Supported 2D image-space metrics", "",
    ]
    sup = result.metrics["body_2d"]["supported"]
    lines += [_fmt_metric(m) for m in sup] if sup else ["- none"]
    lines += ["", "### Unavailable body metrics", ""]
    unav = result.metrics["body_2d"]["unavailable"]
    lines += [_fmt_metric(m) for m in unav] if unav else ["- none"]

    for key, title in (("ball", "BALL"), ("clubhead", "CLUBHEAD")):
        layer = result.layers[key]
        lines += ["", f"## {title}", "",
                  f"- state: **{layer.state.upper()}**",
                  f"- reason: {layer.limitations[0] if layer.limitations else 'n/a'}"]

    # The action-time caveat applies to every physical metric identically. It stays
    # on each record in analysis.json, where a machine reads one metric at a time,
    # and is printed once here, where a person reads the whole section.
    lines += ["", "## PHYSICAL METRICS", ""]
    for m in result.metrics["physical"]["unavailable"]:
        short = m.reason.replace(NO_ACTION_TIME_NOTE, "").strip().rstrip(".")
        lines.append(f"- **{m.name}** ({m.unit}) — unavailable: {short}.")
    lines += ["", f"_{NO_ACTION_TIME_NOTE}_"]

    lines += ["", "## LIMITATIONS", ""]
    for lim in body.limitations:
        lines.append(f"- {lim}")
    lines += [
        "- Single uncalibrated camera: there is no validated physical scale and no "
        "validated action time, so no speed, distance or 3D quantity is reported.",
        "- Ball and clubhead are unavailable by design on this path; no "
        "configuration turns a withdrawn or falsified tracker back on.",
    ]
    exp = result.research_status.get("experimental_components_used") or []
    lines.append(f"- Experimental components used: "
                 f"{', '.join(exp) if exp else 'none'}.")

    lines += ["", "## PROVENANCE", "",
              f"- body model: `{body.provenance.get('model', {}).get('path') or 'not found'}`"
              f" (resolved by {body.provenance.get('model', {}).get('resolved_by') or 'nothing'})",
              f"- body runtime: {body.provenance.get('runtime')}",
              f"- worker interpreter: `{body.provenance.get('interpreter')}`",
              f"- source sha256: `{s.sha256}`",
              f"- ball method: none — {BALL_UNAVAILABLE_REASON}",
              f"- clubhead method: none — {CLUBHEAD_UNAVAILABLE_REASON}",
              "- full detail: `provenance.json`; per-frame detail: `diagnostics.json`",
              ""]
    out = result.outputs
    lines += ["## OUTPUTS", "",
              f"- annotated video: "
              f"{'`annotated.mp4`' if out.get('annotated_video_written') else 'NOT PRODUCED — ' + (out.get('annotated_video_reason') or 'unknown reason')}",
              "- `analysis.json` is the source of truth for this run.", ""]
    return "\n".join(lines)
