"""Operator-assisted demo mode — a local-only, selected-clip visualisation tool.

This namespace is **separate from the canonical workflow** and must not change it.
The canonical path (`python -m ghostcaddie analyze-video`) analyses a clip with no
operator input and reports a strict conditional capability boundary: body where a
golfer identity can be established, ball and clubhead UNAVAILABLE, physical metrics
UNAVAILABLE.

Demo mode is the deliberate opposite trade: it is an *operator-assisted*,
*experimental*, *image-space* visualisation built for carefully selected clips.
An operator supplies an ROI, ball and clubhead seeds, phase keyframes and
corrections; the tool tracks forward from those seeds, measures 2D image-space
quantities only, and discloses the operator input everywhere — in
`demo_report.md`, `demo_analysis.json`, `demo_provenance.json`, the terminal
summary and on the rendered video itself.

What demo mode is not:

* it is not automatic perception — nothing here is described as a detection;
* it is not a measurement of a golf shot — no ball speed, clubhead speed, launch
  angle, carry, apex or spin is produced, and no physical calibration is claimed;
* it is not a network tool — the whole path is local, uses no external AI service
  and downloads nothing.

The orchestrator side of this package is stdlib-only (mirroring
`ghostcaddie.video.canonical`): decoding, pose inference, template tracking,
rendering and the interactive picker live in modules that import cv2/numpy and are
executed out of process in an interpreter that has them.
"""
from __future__ import annotations

DEMO_ANALYSIS_SCHEMA = "fairwayos-demo-analysis.v1"
DEMO_CONFIG_SCHEMA = "fairwayos-demo-config.v1"
DEMO_PROVENANCE_SCHEMA = "fairwayos-demo-provenance.v1"
DEMO_TOOL_VERSION = "fairwayos-demo-0.1.0"

__all__ = ["DEMO_ANALYSIS_SCHEMA", "DEMO_CONFIG_SCHEMA", "DEMO_PROVENANCE_SCHEMA",
           "DEMO_TOOL_VERSION"]
