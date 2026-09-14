"""Launch evidence -> conditional estimate contract.

Turns accepted tracker observations into a speed statement only when the evidence
supports one, and says exactly what is missing when it does not. Three separate
things must hold before any physical number appears:

1. consecutive-frame observations, so a displacement is one frame's travel;
2. an image scale, which here is conditional on an assumed ball diameter;
3. a capture frame rate, which this footage does not declare.

Nothing infers a capture rate from a plausible golfer speed, and nothing fills a
gap. Where a requirement is missing the estimate is Unsupported and names why.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

BALL_DIAMETER_MM = 42.67          # regulation minimum; an assumption, not a measurement
MPH_PER_MPS = 2.2369362920544
KINDS = ("measured", "semantic")
SPREAD_LIMIT = 0.25
FORMULA = ("speed in m/s = (px/frame) x (mm/px) / 1000 x capture_fps"
           "   [the /1000 converts mm to m]")
EVIDENCE_STATUS = ("measured", "assumed")


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


# Tokens. Only this module hands them out. `_CHECKED` means a factory checked the
# STRUCTURE of what it was given -- shapes, ranges, internal consistency. That is
# not the same as checking the world: a probe dict saying 300 fps is structurally
# fine and still tells us nothing about the recording. `_VERIFIED` is the stronger
# one and is granted only when the evidence is bound to a file whose hash this code
# computed itself. Four grades, weakest first:
#
#   assumed   - supplied as a condition
#   asserted  - the caller says it was measured; nothing here looked
#   checked   - a factory validated the structure of an artifact
#   verified  - additionally bound to a source file this code hashed
_CHECKED = object()
_VERIFIED = object()
GRADES = ("assumed", "asserted", "checked", "verified")


@dataclass(frozen=True)
class Evidence:
    """How a claim came to be believed, and whether anything here checked it.

    `measured` means the caller says the world established it and `source` says
    what. `assumed` means it was supplied as a condition. `validated` is separate
    and stronger: this module checked a concrete artifact and the check passed.
    Grades treat an unvalidated `measured` record as the caller assertion it is.
    """

    status: str
    source: str
    check: str = ""
    token: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.status not in EVIDENCE_STATUS:
            raise ValueError(f"status must be one of {EVIDENCE_STATUS}, got {self.status!r}")
        if self.status == "measured" and not str(self.source).strip():
            raise ValueError("measured evidence must name its source")

    @property
    def is_measured(self) -> bool:
        return self.status == "measured"

    @property
    def grade(self) -> str:
        if not self.is_measured:
            return "assumed"
        if self.token is _VERIFIED:
            return "verified"
        if self.token is _CHECKED:
            return "checked"
        return "asserted"

    @property
    def validated(self) -> bool:
        """Structure was checked. NOT a claim that the fact itself was verified."""
        return self.grade in ("checked", "verified")


def _checked(source: str, check: str) -> Evidence:
    return Evidence("measured", source, check=check, token=_CHECKED)


def bind_to_source(evidence: Evidence, path: Any, sha256: str) -> Evidence:
    """Bind checked evidence to a file whose hash this code computes for itself.

    This is the only step that turns a structural check into a verified one, and it
    is deliberately narrow: it proves the evidence refers to a specific artifact,
    nothing more. It does not make an uploader's claim about that artifact true.
    """
    import hashlib
    from pathlib import Path as _Path

    if not isinstance(evidence, Evidence) or evidence.grade not in ("checked", "verified"):
        raise ValueError("only checked evidence can be bound to a source")
    file = _Path(path)
    if not file.is_file():
        raise ValueError(f"no such source file: {file}")
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(f"{file} hashes to {digest}, not the stated {sha256}")
    # Deliberately still "checked". Hashing proves this evidence refers to that
    # file; it says nothing about whether the claim is true of it. A capture rate
    # asserted alongside a matching hash is still an assertion. Only a factory that
    # DERIVES its value from the artifact can grant "verified", and none of the
    # current ones can derive a capture rate, because the recording does not carry
    # it.
    return Evidence("measured", f"{evidence.source}; bound to {file.name} ({digest[:12]})",
                    check=evidence.check, token=_CHECKED)


def camera_evidence(transform: Any, max_residual_px: float = 1.0,
                    min_coverage: float = 0.5) -> Evidence:
    """Validate a background transform before it may support a physical speed.

    Checks the artifact rather than a description of it: the fit must have
    succeeded, carry a finite 2x3 matrix, generalise to points it was not fitted
    to, and cover enough of the frame not to be extrapolating across it.
    """
    status = getattr(transform, "status", None)
    matrix = getattr(transform, "matrix", None)
    residual = getattr(transform, "heldout_residual_px", None)
    coverage = getattr(transform, "coverage", None)
    if status != "ok" or matrix is None:
        raise ValueError(f"camera transform did not succeed (status {status!r})")
    shape = getattr(matrix, "shape", None)
    if shape != (2, 3):
        raise ValueError(f"camera transform matrix must be 2x3, got {shape}")
    if not all(math.isfinite(float(v)) for row in matrix for v in row):
        raise ValueError("camera transform matrix must be finite")
    if residual is None or _finite(residual, "held-out residual") > max_residual_px:
        raise ValueError(f"camera transform held-out residual {residual} exceeds "
                         f"{max_residual_px}px, so the fit does not generalise")
    if coverage is None or _finite(coverage, "coverage") < min_coverage:
        raise ValueError(f"camera transform covers {coverage} of the frame, under "
                         f"{min_coverage}")
    return _checked(f"affine background transform, held-out residual "
                      f"{float(residual):.2f}px over {float(coverage):.0%} of the frame",
                      check="camera_evidence")


def scale_evidence(reference_mm: float, reference_px: float, source: str) -> Evidence:
    """Validate a scale taken from a reference object actually measured in frame."""
    millimetres = _finite(reference_mm, "reference_mm")
    pixels = _finite(reference_px, "reference_px")
    if millimetres <= 0 or pixels <= 0:
        raise ValueError("reference size in mm and px must both be > 0")
    if not str(source).strip():
        raise ValueError("scale evidence must name the reference object")
    return _checked(f"{source}: {millimetres:g}mm measured as {pixels:g}px",
                      check="scale_evidence")


def timing_evidence(capture_fps: float, probe: dict, key: str = "capture_fps") -> Evidence:
    """Validate a capture rate against a probe of the recording, not against prose."""
    claimed = _finite(capture_fps, "capture_fps")
    if not isinstance(probe, dict) or key not in probe:
        raise ValueError(f"timing evidence needs a probe carrying {key!r}")
    probed = _finite(probe[key], f"probe[{key!r}]")
    if probed <= 0:
        raise ValueError("probed capture rate must be > 0")
    if abs(probed - claimed) > 1e-6:
        raise ValueError(f"claimed capture rate {claimed} does not match the probed "
                         f"{probed}")
    return _checked(f"capture rate {probed:g} fps read from the recording",
                      check="timing_evidence")


def identity_evidence(adjudication: dict, frames: Sequence[int]) -> Evidence:
    """Validate that every frame relied on was actually adjudicated as the target."""
    if not isinstance(adjudication, dict):
        raise ValueError("identity evidence needs a per-frame adjudication mapping")
    missing = [f for f in frames if f not in adjudication]
    if missing:
        raise ValueError(f"frames {missing} were not adjudicated")
    disputed = [f for f in frames if adjudication[f] is not True]
    if disputed:
        raise ValueError(f"frames {disputed} were not adjudicated as the same target")
    return _checked(f"{len(frames)} frames individually adjudicated",
                      check="identity_evidence")


ASSUMED_TIMING = Evidence("assumed", "caller-supplied timing scenario")
ASSUMED_SCALE = Evidence("assumed", "apparent ball size, highlight not silhouette")


def _evidence(value: Any, name: str, allow_none: bool = True) -> Evidence | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, Evidence):
        raise ValueError(f"{name} must be an Evidence record, got {value!r}")
    return value


@dataclass(frozen=True)
class Observation:
    """One accepted tracker point, in native source pixels of a named source frame."""

    frame: int
    x: float
    y: float
    kind: str

    def __post_init__(self) -> None:
        if isinstance(self.frame, bool) or not isinstance(self.frame, int):
            raise ValueError(f"frame must be an int, got {self.frame!r}")
        _finite(self.x, "x")
        _finite(self.y, "y")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")


@dataclass(frozen=True)
class Interval:
    first: Observation
    second: Observation

    @property
    def px(self) -> float:
        return math.hypot(self.second.x - self.first.x, self.second.y - self.first.y)

    @property
    def kind(self) -> str:
        return "measured" if self.first.kind == self.second.kind == "measured" else "semantic"

    @property
    def frames(self) -> list[int]:
        return [self.first.frame, self.second.frame]


@dataclass(frozen=True)
class ScaleInterval:
    low_mm_per_px: float
    high_mm_per_px: float
    px_low: float
    px_high: float
    caveat: str
    evidence: Evidence = ASSUMED_SCALE


@dataclass(frozen=True)
class TimingScenario:
    name: str
    capture_fps: float
    evidence: Evidence = ASSUMED_TIMING

    def __post_init__(self) -> None:
        if _finite(self.capture_fps, "capture_fps") <= 0:
            raise ValueError("capture_fps must be > 0")
        if not isinstance(self.evidence, Evidence):
            raise ValueError(f"timing evidence must be an Evidence record, got "
                             f"{self.evidence!r}")

    @property
    def established(self) -> bool:
        return self.evidence.is_measured


def intervals(observations: Sequence[Observation]) -> tuple[list[Interval], list[str]]:
    """Consecutive-frame pairs that can carry a displacement, and why the rest cannot."""
    seen: set[int] = set()
    for observation in observations:
        if observation.frame in seen:
            raise ValueError(f"frame {observation.frame} appears twice; an observation "
                             f"may be used once")
        seen.add(observation.frame)

    ordered = sorted(observations, key=lambda o: o.frame)
    usable: list[Interval] = []
    rejected: list[str] = []
    for first, second in zip(ordered, ordered[1:]):
        step = second.frame - first.frame
        if step != 1:
            rejected.append(
                f"frames {first.frame} -> {second.frame}: not consecutive ({step - 1} "
                f"frame(s) missing); a displacement across a gap is not one frame's travel")
            continue
        interval = Interval(first, second)
        if interval.px == 0.0:
            rejected.append(
                f"frames {first.frame} -> {second.frame}: identical coordinate, which is "
                f"equally consistent with a stationary target and with a repeated answer, "
                f"so it supports no speed either way")
            continue
        usable.append(interval)
    return usable, rejected


SEPARATION_RATIO = 3.0


def split_by_motion(observations: Sequence[Observation]) -> dict:
    """Split intervals into a resting state and a moving state, from the data alone.

    Two-means on the interval displacements, then a check that the two groups are
    actually far apart. No frame number, no speed prior and no hand-set pixel
    threshold appears here: a track with one motion state is reported as
    inseparable rather than being cut somewhere arbitrary. The moving window is the
    longest CONSECUTIVE run of moving intervals, because a window stitched from
    separate bursts is not one event.
    """
    usable, _ = intervals(observations)
    if len(usable) < 3:
        return {"separable": False, "stationary": [], "moving": [],
                "reason": f"at least 3 usable intervals are needed to separate two "
                          f"motion states, got {len(usable)}"}

    values = sorted(interval.px for interval in usable)
    best = None
    for cut in range(1, len(values)):
        low, high = values[:cut], values[cut:]
        spread = (sum((v - sum(low) / len(low)) ** 2 for v in low)
                  + sum((v - sum(high) / len(high)) ** 2 for v in high))
        if best is None or spread < best[0]:
            best = (spread, sum(low) / len(low), sum(high) / len(high), low[-1])
    _, low_mean, high_mean, boundary = best

    if low_mean <= 0 or high_mean < low_mean * SEPARATION_RATIO:
        return {"separable": False, "stationary": [], "moving": [],
                "reason": (f"no clear separation between motion states: the two groups "
                           f"differ by {high_mean / low_mean:.1f}x, under the "
                           f"{SEPARATION_RATIO:g}x this requires")}

    moving_flags = [interval.px > boundary for interval in usable]
    longest: list[Interval] = []
    current: list[Interval] = []
    for interval, moving in zip(usable, moving_flags):
        current = current + [interval] if moving else []
        if len(current) > len(longest):
            longest = current
    return {"separable": True,
            "stationary": [i for i, m in zip(usable, moving_flags) if not m],
            "moving": longest,
            "reason": (f"resting intervals average {low_mean:.2f}px and moving intervals "
                       f"{high_mean:.2f}px, a {high_mean / low_mean:.1f}x separation")}


def scale_mm_per_px(px_low: float, px_high: float,
                    diameter_mm: float = BALL_DIAMETER_MM,
                    evidence: Evidence | None = None) -> ScaleInterval:
    """Millimetres per pixel, from the interval of apparent ball widths in pixels."""
    low = _finite(px_low, "px_low")
    high = _finite(px_high, "px_high")
    diameter = _finite(diameter_mm, "diameter_mm")
    if low <= 0 or high <= 0 or diameter <= 0:
        raise ValueError("apparent size and diameter must be > 0")
    if low > high:
        raise ValueError("px_low must not exceed px_high")
    return ScaleInterval(
        low_mm_per_px=diameter / high,
        high_mm_per_px=diameter / low,
        px_low=low,
        px_high=high,
        caveat=(f"conditional on an assumed {diameter:.2f}mm ball measured as a bright blob "
                f"{low:g}-{high:g}px across, which is a highlight and not a calibrated "
                f"silhouette; it also holds only at the ball's depth"),
        evidence=_evidence(evidence, "scale evidence") or ASSUMED_SCALE,
    )


def _motion_reliability(usable: list[Interval], spread: float | None,
                        identity: Evidence | None) -> dict:
    """How much the measured image motion itself is worth. No timing involved."""
    if not usable:
        return {"level": "Unsupported", "reasons": ["no usable consecutive interval"]}

    reasons: list[str] = []
    level = "High"
    if any(interval.kind == "semantic" for interval in usable):
        level = "Low"
        reasons.append("at least one interval rests on a semantic confirmation with no "
                       "image measurement behind it")
    if len(usable) < 2:
        level = "Low"
        reasons.append(f"only {len(usable)} usable interval; a single interval cannot show "
                       f"consistency between frames")
    elif spread is not None and spread > SPREAD_LIMIT:
        level = "Low"
        reasons.append(f"interval spread {spread:.0%} exceeds {SPREAD_LIMIT:.0%}, so the "
                       f"frames disagree about how far the target moved")
    if not (identity is not None and identity.is_measured) and level != "Low":
        # Nothing in this pipeline establishes that the points across frames are the
        # same physical object, so the measurement cannot be better than Moderate.
        level = "Moderate"
        reasons.append("target identity across frames is not verified by anything in this "
                       "pipeline, so the displacement may not be one object's travel")
    if not reasons:
        reasons.append("consecutive measured intervals agree within "
                       f"{SPREAD_LIMIT:.0%}")
    return {"level": level, "reasons": reasons}


def _speed_reliability(motion: dict, scale: ScaleInterval | None,
                       scenarios: Sequence[TimingScenario], camera: Evidence | None,
                       identity: Evidence | None, applied: bool = False) -> dict:
    """What a physical speed is worth.

    Missing or assumed timing is an obstruction, not a discount: it makes the speed
    Unsupported however consistent the frames look. Above that floor the grade is
    set by which links of the chain are measured rather than assumed, so a future
    calibrated setup can reach High while no caller assertion ever can.
    """
    blockers: list[str] = []
    if motion["level"] == "Unsupported":
        blockers.append("no usable consecutive interval")
    if scale is None:
        blockers.append("image scale is not established")
    if not scenarios:
        blockers.append("capture frame rate is not established")
    elif any(not scenario.established for scenario in scenarios):
        blockers.append("capture frame rate is assumed, not established")
    # Camera correction is a blocker, not a discount: without it the displacement
    # may be the camera's travel rather than the target's, so there is no measured
    # quantity to put a number on, conditional or otherwise.
    if applied is not True:
        blockers.append("camera correction was not applied to these displacements, so "
                        "they are raw image motion")
    if camera is None:
        blockers.append("camera motion has not been removed from the displacement, so "
                        "the measured travel may be the camera's")
    elif not camera.validated:
        blockers.append("camera correction is asserted but not validated here, so the "
                        "displacement is not established as the target's travel")
    if blockers:
        return {"level": "Unsupported", "reasons": blockers}

    reasons: list[str] = []
    for link, evidence in (("image scale", scale.evidence), ("capture timing",
                           scenarios[0].evidence), ("camera correction", camera),
                           ("target identity", identity)):
        if evidence is None or not evidence.is_measured:
            reasons.append(f"{link} is not measured")
        elif not evidence.validated:
            # The record says measured, but nothing here checked an artifact.
            reasons.append(f"{link} rests on a caller-supplied provenance string, which "
                           f"is a claim about evidence rather than evidence")
    reasons.extend(r for r in motion["reasons"] if r not in reasons)

    links = {"image scale": scale.evidence, "capture timing": scenarios[0].evidence,
             "camera correction": camera, "target identity": identity}
    unbound = [name for name, evidence in links.items()
               if evidence is not None and evidence.grade == "checked"]
    if unbound:
        reasons.append(f"{', '.join(sorted(unbound))} passed a structural check only: "
                       f"nothing here bound it to the recording it describes")

    weak = [r for r in reasons if "not measured" in r or "provenance string" in r
            or "conditional" in r or "not verified" in r or "spread" in r
            or "semantic" in r or "single interval" in r]
    if weak:
        return {"level": "Low", "reasons": reasons}
    if unbound or motion["level"] != "High":
        return {"level": "Moderate",
                "reasons": reasons or ["every link is checked but the measurement itself "
                                       "is not High"]}
    return {"level": "High", "reasons": reasons or ["every link is verified against the "
                                                    "recording"]}


def estimate(observations: Sequence[Observation], scale: ScaleInterval | None = None,
             scenarios: Sequence[TimingScenario] = (),
             camera: Evidence | None = None,
             identity: Evidence | None = None,
             corrections: dict[tuple[int, int], float] | None = None) -> dict:
    """Everything the evidence supports about one target's speed, and nothing else.

    Measured image motion and physical speed are reported separately, because they
    fail for different reasons: image motion needs only consecutive observations,
    while a physical speed additionally needs a scale and a capture frame rate.
    """
    if scale is not None and not isinstance(scale, ScaleInterval):
        raise ValueError(f"scale must be a ScaleInterval, got {scale!r}")
    for scenario in scenarios:
        if not isinstance(scenario, TimingScenario):
            raise ValueError(f"scenario must be a TimingScenario, got {scenario!r}")
    camera = _evidence(camera, "camera evidence")
    identity = _evidence(identity, "identity evidence")
    if corrections is not None and (camera is None or not camera.validated):
        raise ValueError("camera-corrected displacements need validated camera evidence")

    usable, rejected = intervals(observations)
    if corrections is not None:
        for key, value in corrections.items():
            distance = _finite(value, f"correction {key}")
            if distance < 0:
                raise ValueError(f"correction {key} must be a non-negative distance, "
                                 f"got {value!r}")
        missing = [tuple(i.frames) for i in usable
                   if tuple(i.frames) not in corrections]
        if missing:
            raise ValueError(f"camera correction is missing for intervals {missing}; a "
                             f"partly corrected track mixes two different measurements")
    px_values = [corrections[tuple(i.frames)] if corrections is not None else i.px
                 for i in usable]
    px_low = min(px_values) if px_values else None
    px_high = max(px_values) if px_values else None
    spread = None if not px_values or px_high == 0 else (px_high - px_low) / px_high

    motion_reliability = _motion_reliability(usable, spread, identity)
    speed_reliability = _speed_reliability(motion_reliability, scale, scenarios,
                                           camera, identity, corrections is not None)

    rows: list[dict] = []
    # Numbers are emitted only when the displacement itself is established, which
    # needs the camera link validated. Conditional timing makes a row conditional;
    # an unestablished displacement means there is nothing to condition.
    if (usable and scale is not None and camera is not None and camera.validated
            and corrections is not None):
        for scenario in scenarios:
            low = px_low * scale.low_mm_per_px / 1000.0 * scenario.capture_fps
            high = px_high * scale.high_mm_per_px / 1000.0 * scenario.capture_fps
            rows.append({
                "name": scenario.name,
                "capture_fps": scenario.capture_fps,
                "conditional": not scenario.established,
                "low_mps": low, "high_mps": high,
                "low_mph": low * MPH_PER_MPS, "high_mph": high * MPH_PER_MPS,
                "frames_used": [interval.frames for interval in usable],
                "grade": ("measured input" if scenario.established
                          else "conditional scenario, not a measurement"),
            })

    return {
        "evidence": {
            "frames": [o.frame for o in sorted(observations, key=lambda o: o.frame)],
            "intervals_used": len(usable),
            "kinds": dict(Counter(o.kind for o in observations)),
        },
        "image_motion": {
            "low": px_low, "high": px_high, "spread": spread,
            "unit": "px per frame",
            "note": ("image-projected 2D displacement, camera motion removed by an affine "
                     "background transform; motion along the camera axis is still not "
                     "measured") if corrections is not None else
                    ("image-projected 2D displacement; motion along the camera axis is "
                     "not measured, and camera motion is not removed here"),
            "reliability": motion_reliability,
        },
        "scale": None if scale is None else {
            "low_mm_per_px": scale.low_mm_per_px, "high_mm_per_px": scale.high_mm_per_px,
            "apparent_px_low": scale.px_low, "apparent_px_high": scale.px_high,
            "caveat": scale.caveat,
        },
        "physical_speed": {"scenarios": rows, "reliability": speed_reliability},
        "camera_support": {
            "status": ("applied" if corrections is not None else
                       "evidence only, not applied to the displacement"
                       if camera is not None and camera.validated else "absent"),
            "source": None if camera is None else camera.source,
            "note": ("displacement has camera motion removed by an affine background "
                     "transform" if corrections is not None else
                     "displacement is raw image motion; camera motion is not removed"),
        },
        "identity_support": {
            "status": "verified" if identity is not None and identity.is_measured
                      else "unverified",
        },
        "rejected": rejected,
    }


def report(result: dict, target: str) -> tuple[dict, str]:
    """A machine-readable document and the same content as readable text."""
    document = {
        "target": target,
        "formula": FORMULA,
        "inputs": {"scale": result["scale"],
                   "scenarios": [{"name": r["name"], "capture_fps": r["capture_fps"],
                                  "conditional": r["conditional"]}
                                 for r in result["physical_speed"]["scenarios"]]},
        "evidence": result["evidence"],
        "image_motion": result["image_motion"],
        "physical_speed": result["physical_speed"],
        "camera_support": result["camera_support"],
        "identity_support": result["identity_support"],
        "rejected": result["rejected"],
    }

    lines = [f"{target} launch evidence", "=" * (len(target) + 17), ""]
    evidence = result["evidence"]
    lines.append(f"Observations: {len(evidence['frames'])} accepted "
                 f"({', '.join(f'{k} {v}' for k, v in sorted(evidence['kinds'].items())) or 'none'})")
    lines.append(f"Usable consecutive intervals: {evidence['intervals_used']}")
    motion = result["image_motion"]
    if motion["low"] is not None:
        lines.append(f"Measured image motion: {motion['low']:.2f}-{motion['high']:.2f} px "
                     f"per frame ({motion['note']})")
        lines.append(f"Image-motion reliability: {motion['reliability']['level']}")
        for reason in motion["reliability"]["reasons"]:
            lines.append(f"  - {reason}")
    lines.append(f"Camera motion: {result['camera_support']['note']}")
    lines.append("")
    # The formula appears only where a speed was actually produced: printing how a
    # number would be computed, under a heading that produced none, reads as a result.
    if result["physical_speed"]["scenarios"]:
        lines.append(f"Formula: {FORMULA}")
    if result["scale"]:
        scale = result["scale"]
        lines.append(f"Scale input: {scale['low_mm_per_px']:.3f}-{scale['high_mm_per_px']:.3f} "
                     f"mm/px, {scale['caveat']}")
    else:
        lines.append("Scale input: none")
    lines.append("")

    physical = result["physical_speed"]
    if physical["scenarios"]:
        lines.append("Speed, per timing scenario. These are conditions, not findings:")
        for row in physical["scenarios"]:
            label = "conditional" if row["conditional"] else "established"
            lines.append(f"  {row['name']} [{label}]: {row['low_mps']:.1f}-{row['high_mps']:.1f} "
                         f"m/s ({row['low_mph']:.0f}-{row['high_mph']:.0f} mph), from frames "
                         f"{row['frames_used'][0][0]}-{row['frames_used'][-1][1]}")
    else:
        lines.append("Speed: not produced.")
    lines.append("")
    lines.append(f"Physical speed reliability: {physical['reliability']['level']}")
    for reason in physical["reliability"]["reasons"]:
        lines.append(f"  - {reason}")
    if result["rejected"]:
        lines.append("")
        lines.append("Evidence rejected:")
        for reason in result["rejected"]:
            lines.append(f"  - {reason}")
    return document, "\n".join(lines) + "\n"


def _range_text(row: dict) -> str:
    return (f"{row['low_mps']:.0f}-{row['high_mps']:.0f} m/s "
            f"({row['low_mph']:.0f}-{row['high_mph']:.0f} mph)")


def panel_lines(estimates: dict[str, dict], phase: str | None = None) -> list[str]:
    """Golfer-facing panel text: no coordinates, no confidences, no decimals of either."""
    lines: list[str] = []
    if phase:
        # Clip-wide, and worded as such: this panel is repeated on every frame,
        # including address frames, so "Phase:" would assert something false there.
        lines.append(f"Shot summary: {phase}")
    for target, result in estimates.items():
        label = f"{target.capitalize()} speed"
        physical = result["physical_speed"]
        motion = result["image_motion"]
        lines.append(f"{label}: {physical['reliability']['level']}")
        lines.append(f"  why: {physical['reliability']['reasons'][0]}")
        if motion["low"] is not None:
            lines.append(f"  measured movement: {motion['low']:.0f}-{motion['high']:.0f} "
                         f"px per frame ({motion['reliability']['level']})")
        for row in physical["scenarios"]:
            # Shown as a condition the viewer supplies, never as a measurement.
            lines.append(f"  {row['name']}: {_range_text(row)}")
    lines.append("Image-projected 2D motion, not true 3D. Distance is not estimated.")
    return lines
