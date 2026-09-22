"""The demo result object: one run's facts, before any wording is applied.

Keeping the result a plain dataclass in a stdlib module means the report, the
summary, the analysis JSON and the provenance JSON are all rendered from the same
object, so they cannot disagree with each other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import DEMO_ANALYSIS_SCHEMA, DEMO_TOOL_VERSION

DISCLOSURE_SHORT = (
    "Operator-assisted, experimental, 2D image-space demo. Seeded points, frame "
    "selections and corrections are operator input, not automatic detection. No "
    "physical measurement (no speeds, distances, launch angle or spin).")
DISCLOSURE_LONG = [
    "This is an operator-assisted visualisation of one selected clip, not automatic "
    "perception. The ROI, the ball and clubhead seeds, the phase keyframes and every "
    "correction were supplied by an operator; the tracker follows what it was given.",
    "Every quantity is 2D image space from a single uncalibrated camera: angles and "
    "pixel distances change with camera position and lens, and none of them is a "
    "coaching, clinical or physical measurement.",
    "No physical metric is produced. There is no ball speed, no clubhead speed, no "
    "launch angle, no carry, no apex and no spin: there is no validated scale and no "
    "measured action time in this source.",
    "Source timing is container cadence only. The clip may be a replay, a slow-motion "
    "capture or a re-timed copy, so frame counts and seconds are positions in the "
    "source timeline, not measured action time.",
    "This demo path runs locally: no upload, no network call, no downloads and no "
    "remote inference, and nothing here is a claim about how any other clip would "
    "behave.",
]


@dataclass
class DemoResult:
    source: Dict[str, Any]
    interval: Dict[str, Any]
    config: Dict[str, Any]
    body: Dict[str, Any]
    analytics: Dict[str, Any]
    phases: Dict[str, Any]
    targets: Dict[str, Any]
    render: Dict[str, Any]
    keyframes: Dict[str, Any]
    limitations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)
    provenance_refs: Dict[str, Any] = field(default_factory=dict)
    created_utc: str = ""
    schema_version: str = DEMO_ANALYSIS_SCHEMA
    tool_version: str = DEMO_TOOL_VERSION

    def disclosure(self) -> dict:
        return {
            "operator_assisted": True,
            "experimental": True,
            "image_space": True,
            "manual_inputs": ["roi", "seeds", "keyframes", "corrections"],
            "physical_metrics": False,
            "automatic_detection_claims": False,
            "network_calls": "none",
            "external_ai_service": "none",
            "ground_truth": False,
            "production_eligible": False,
            "summary": DISCLOSURE_SHORT,
            "statements": list(DISCLOSURE_LONG),
        }

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "created_utc": self.created_utc,
            "tool": {"name": "fairwayos-demo", "version": self.tool_version,
                     "outputs": ["demo_annotated.mp4", "demo_report.md",
                                 "demo_analysis.json", "demo_provenance.json",
                                 "demo_summary.md", "keyframes/", "debug/"]},
            "mode": self.disclosure(),
            "source": self.source,
            "processing": self.interval,
            "config": self.config,
            "body": self.body,
            "analytics": self.analytics,
            "phases": self.phases,
            "targets": self.targets,
            "render": self.render,
            "keyframes": self.keyframes,
            "limitations": self.limitations,
            "warnings": self.warnings,
            "notes": self.notes,
            "debug": self.debug,
        }

    # -- convenience accessors used by the report ---------------------------
    def body_supported(self) -> List[dict]:
        return list(self.analytics.get("body_2d", {}).get("supported") or [])

    def body_unavailable(self) -> List[dict]:
        return list(self.analytics.get("body_2d", {}).get("unavailable") or [])

    def target(self, name: str) -> dict:
        return dict(self.targets.get(name) or {})

    def phase(self, name: str) -> dict:
        return dict(self.phases.get("phases", {}).get(name) or {})

    def tempo(self) -> dict:
        return dict(self.phases.get("tempo") or {})

    def target_metrics(self, name: str) -> dict:
        return dict(self.analytics.get("targets", {}).get(name) or {})


def metric_line(metric: Dict[str, Any]) -> str:
    """One metric as one line. Rounded numbers only; no false precision."""
    value = metric.get("value")
    used = metric.get("frames_used")
    possible = metric.get("frames_possible")
    if metric.get("state") != "supported" or value is None:
        return f"- **{metric['name']}** — unavailable: {metric.get('reason', '')}"
    p10 = (metric.get("detail") or {}).get("p10")
    p90 = (metric.get("detail") or {}).get("p90")
    spread = ""
    if p10 is not None and p90 is not None:
        spread = f", p10 {p10:.1f} / p90 {p90:.1f}"
    return (f"- **{metric['name']}** — {value:.1f} {metric.get('unit', '')} "
            f"({used}/{possible} frames{spread})")
