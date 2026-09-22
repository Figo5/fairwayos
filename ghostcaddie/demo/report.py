"""Wording for the demo outputs. Stdlib only, and deliberately repetitive.

The same disclosure appears in the report, the summary, the analysis JSON, the
provenance and on the rendered video. That redundancy is the point: an operator
handing a clip to someone else cannot accidentally hand over the picture without
the caveats, because the caveats are printed on the picture.

Nothing in here may describe a seeded, corrected or interpolated point as a
detection, and nothing may claim a physical quantity.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from .config import PHASE_ORDER
from .result import DISCLOSURE_LONG, DISCLOSURE_SHORT, DemoResult, metric_line
from .tracking import (STATUS_INTERPOLATED, STATUS_LOST, STATUS_MANUAL,
                       STATUS_MISSING, STATUS_OBSERVED, STATUS_OBSERVED_LK)

#: Visible on every rendered frame. Kept short and ASCII-only on purpose.
VIDEO_TITLE = "FAIRWAY OS  /  DEMO"
VIDEO_SUBTITLE = "OPERATOR-ASSISTED / EXPERIMENTAL / 2D IMAGE-SPACE"
VIDEO_LEGEND = ("body = teal skeleton (operator ROI)   ball = circle (operator-seeded)"
                "   club = diamond (operator-seeded)   dashed = interpolated (derived)")
VIDEO_CAVEAT = "operator-seeded points, not automatic detections - no physical metrics"

PHASE_LABEL_TEXT = {"manual": "OPERATOR-MARKED", "approx": "APPROXIMATE",
                    "unavailable": "UNAVAILABLE"}


def _video_name(result) -> str:
    """The rendered video's own file name, so a showcase run is not reported as the
    demo's file name in the terminal summary."""
    path = ((result.presentation or {}).get("output_paths") or {}).get("annotated")
    return os.path.basename(str(path)) if path else "demo_annotated.mp4"


def _fmt(value: Any, unit: str = "") -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.1f}{(' ' + unit) if unit else ''}"
    return f"{value}{(' ' + unit) if unit else ''}"


def render_report(result: DemoResult) -> str:
    s = result.source
    interval = result.interval
    lines: List[str] = []
    lines += [f"# FairwayOS demo (operator-assisted) — {s['display_name']}", "",
              f"_{result.created_utc} · schema `{result.schema_version}` · "
              f"tool `{result.tool_version}`_", "",
              f"**{DISCLOSURE_SHORT}**", ""]

    # ---------------------------------------------------------------- VIDEO
    lines += ["## VIDEO", "",
              f"- source: `{s['display_name']}`" + (f" (`{s['path']}`)" if s.get("path") else ""),
              f"- sha256: `{s['sha256']}`",
              f"- {s['width']}×{s['height']} · {s['fps']:g} fps (average) · "
              f"{s['duration_seconds']:g}s · {s.get('container_format') or '?'} / "
              f"{s.get('codec') or '?'}",
              f"- processed interval: source frames {interval['start_frame']}–"
              f"{interval['end_frame']} inclusive → {interval['frame_count']} frames, "
              f"every frame at native cadence",
              f"- endpoint semantics: {interval['endpoint_semantics']}",
              f"- seconds basis: {interval.get('seconds_basis', 'frames / cadence')}",
              f"- source cadence: {s['cadence_note']}", ""]
    if s.get("timing_caveat"):
        lines += [f"- source timing caveat (operator): {s['timing_caveat']}", ""]
    if s.get("timestamp_note"):
        lines += [f"- frame timestamps: {s['timestamp_note']}", ""]
    if s.get("irregular_timestamps") or s.get("variable_cadence"):
        lines += ["- **variable-cadence source**: the decoded frame timestamps are not "
                  "evenly spaced, so no frame below is a measured action time and no "
                  "rate is derived from the nominal fps. Frame indices are "
                  "authoritative.", ""]

    # ---------------------------------------------------------------- SETUP
    cfg = result.config
    lines += ["## SETUP", "",
              f"- all operator inputs are recorded in full in `demo_provenance.json` "
              f"(and in `demo_analysis.json`); config sha256 `{cfg.get('sha256', '')}`",
              f"- ROI (source pixels): "
              f"{cfg.get('roi_xywh') if cfg.get('roi_xywh') else 'none — body pose used the whole frame'}",
              ]
    for name, seed in sorted((cfg.get("seeds") or {}).items()):
        lines.append(f"- {name} seed: frame {seed['frame']} at "
                     f"({seed['x']:.0f}, {seed['y']:.0f}) — operator-placed"
                     + (f" ({seed['note']})" if seed.get("note") else ""))
    for name in ("ball", "clubhead"):
        corrections = cfg.get("corrections", {}).get(name) or []
        if corrections:
            frames = ", ".join(str(c["frame"]) for c in corrections)
            lines.append(f"- {name} corrections: {len(corrections)} operator restart "
                         f"point(s) at frame(s) {frames}")
        options = (cfg.get("effective", {}).get("targets") or {}).get(name) or {}
        if options.get("automatic_until_frame") is not None:
            lines.append(f"- {name} automatic matching: stopped by the operator after "
                         f"source frame {options['automatic_until_frame']} "
                         f"(`targets.{name}.automatic_until_frame`); from there on this "
                         f"target is exactly the operator anchors above and nothing else")
    lines.append(f"- interpolation: "
                 f"{'enabled' if cfg.get('interpolation', {}).get('enabled') else 'disabled'}"
                 f", max gap {cfg.get('interpolation', {}).get('max_gap_frames')} frames"
                 f" (dashed, derived, never counted as an observation)")
    lines.append(f"- handedness assumed by the operator: {cfg.get('handedness')} "
                 f"(used only to label lead/trail)")
    render_cfg = cfg.get("render") or {}
    if render_cfg:
        lines.append(f"- render: max width {render_cfg.get('max_width')} px, crf "
                     f"{render_cfg.get('crf')}, x264 preset {render_cfg.get('preset')}, "
                     f"rendered {'yes' if render_cfg.get('render') else 'no'}")
    lines.append(f"- video legend: {VIDEO_LEGEND}")
    lines += [""]

    # ---------------------------------------------------------------- BODY
    body = result.body
    lines += ["## BODY ANALYSIS", "",
              f"- state: **{str(body.get('state', 'unavailable')).upper()}**",
              f"- method: {body.get('method')}",
              f"- coverage: {body.get('coverage', {}).get('frames_with_pose')}/"
              f"{body.get('coverage', {}).get('frames_total')} processed frames carry a "
              f"pose attributed to the golfer region",
              f"- pose model: `{body.get('model', {}).get('path') or 'not found'}` "
              f"({body.get('model', {}).get('resolved_by') or 'nothing'}), runtime "
              f"`{body.get('interpreter')}`",
              ""]
    body_2d = result.analytics.get("body_2d", {})
    lines += ["**2D image-space quantities (degrees / pixels, one uncalibrated camera)**",
              ""]
    supported = result.body_supported()
    lines += [metric_line(m) for m in supported] if supported else ["- none"]
    lines += ["", "**Unavailable body quantities**", ""]
    unavailable = result.body_unavailable()
    lines += [metric_line(m) for m in unavailable] if unavailable else ["- none"]
    conventions = body_2d.get("conventions") or {}
    for text in (conventions.get("tilt"), conventions.get("knee"),
                 conventions.get("sides")):
        if text:
            lines.append(f"_{text}_")
    lines.append("")

    # ---------------------------------------------------------------- PHASES
    lines += ["## SWING PHASES", "",
              "Phase frames are either **operator-marked** or explicitly "
              "**approximate**; a phase that is neither is reported UNAVAILABLE. "
              "Evenly spaced frames are never used as phases.", ""]
    for name in PHASE_ORDER:
        phase = result.phase(name)
        if not phase:
            continue
        label = PHASE_LABEL_TEXT.get(str(phase.get("label")), str(phase.get("label")))
        if phase.get("frame") is None:
            lines.append(f"- **{name}** — UNAVAILABLE: {phase.get('note')}")
            continue
        angles = phase.get("angles_2d") or {}
        bits = []
        if angles.get("shoulder_tilt_deg") is not None:
            bits.append(f"shoulder tilt {angles['shoulder_tilt_deg']:.1f}°")
        if angles.get("hip_tilt_deg") is not None:
            bits.append(f"hip tilt {angles['hip_tilt_deg']:.1f}°")
        lead = angles.get("lead_side")
        knee = (angles.get("knee_interior_deg") or {}).get(lead)
        if knee is not None:
            bits.append(f"{lead} knee interior {knee:.1f}°")
        detail = ((" · " + " · ".join(bits)) if bits
                  else " · no 2D angle was measurable at this frame")
        seconds = phase.get("seconds_source_timeline")
        when = (f"frame {phase['frame']} ({seconds:g}s in the source timeline)"
                if seconds is not None else f"frame {phase['frame']}")
        relative = phase.get("seconds_from_interval_start")
        if relative is not None:
            when += f", {relative:g}s from the interval start"
        lines.append(f"- **{name}** — {when} · {label} · `{phase.get('method')}`"
                     f"{detail}")
    phase_notes = result.phases.get("notes") or []
    if phase_notes:
        lines += ["", "Notes on this phase set:", ""]
        lines += [f"- {note}" for note in phase_notes]
    lines += [""]

    # ---------------------------------------------------------------- TEMPO
    tempo = result.tempo()
    lines += ["## TEMPO", ""]
    if tempo.get("state") != "supported":
        lines += [f"- unavailable: {tempo.get('reason', 'unknown reason')}",
                  f"- {tempo.get('note', '')}", ""]
    else:
        seconds = tempo.get("seconds_source_timeline") or {}
        lines += [
            f"- **backswing** — {tempo['backswing_frames']} source frames "
            f"({_fmt(seconds.get('backswing'), 's')} in the source timeline)",
            f"- **downswing** — {tempo['downswing_frames']} source frames "
            f"({_fmt(seconds.get('downswing'), 's')} in the source timeline)",
            f"- **backswing / downswing ratio** — "
            f"{tempo['backswing_downswing_ratio']:.2f} : 1 "
            f"({tempo['label']} markers; backswing start = "
            f"{tempo['backswing_start_marker']})",
            f"- seconds basis: {tempo.get('seconds_basis', seconds.get('basis'))}",
            f"- definition: {tempo['definition']}",
            f"- {tempo['caveat']}", ""]

    # ---------------------------------------------------------------- TARGETS
    lines += ["## BALL / CLUB", "",
              "Each target is tracked independently from its own operator seed, with "
              "its own tracker settings and its own corrections. A point is either an "
              "operator placement, a template match, or an explicitly derived "
              "interpolation; nothing is bridged across a gap, a loss or a correction.",
              ""]
    for name in ("ball", "clubhead"):
        track = result.target(name)
        metrics = result.target_metrics(name)
        title = name.capitalize() if name != "clubhead" else "Clubhead"
        lines += [f"### {title}", ""]
        if not track or not track.get("available"):
            reason = (track or {}).get("reason", "no operator seed was supplied")
            lines += [f"- state: **UNAVAILABLE** — {reason}", ""]
            continue
        seed = track.get("seed") or {}
        lines += [
            f"- mode: `{track.get('mode')}` · direction `{track.get('direction')}`",
            f"- seed: frame {seed.get('frame')} at "
            f"({float(seed.get('x', 0)):.0f}, {float(seed.get('y', 0)):.0f}) "
            f"({seed.get('source', 'seed')})",
            f"- observed frames: {metrics.get('observed_frames')}"
            f"/{metrics.get('frames_total')} "
            f"({100.0 * float(metrics.get('observed_fraction') or 0):.1f}%), of which "
            f"{metrics.get('automatic_observed_frames')} are automatic template matches "
            f"and {metrics.get('operator_anchor_frames')} are operator anchors",
            f"- interpolated frames (derived, dashed): "
            f"{metrics.get('interpolated_frames')}",
            f"- unobserved frames: {metrics.get('unobserved_frames')} "
            f"({metrics.get('missing_frames')} with no bounded match + "
            f"{metrics.get('lost_frames')} declared lost; these two counts are "
            f"disjoint, not nested)"
            + (f", of which {metrics.get('operator_stop_frames')} are past the "
               f"operator stop frame and already out of scope"
               if metrics.get("operator_stop_frames") else "")
            + (f", and {metrics.get('automatic_stop_frames')} are past the operator's "
               f"automatic_until_frame (only operator anchors exist there)"
               if metrics.get("automatic_stop_frames") else "")
            + f"; longest unobserved run {metrics.get('longest_missing_run')} frames",
            f"- separate trail segments: {metrics.get('segment_count')}",
            f"- image-space path length (observed points only: template matches and "
            f"operator placements, with no loss or correction jump counted): "
            f"{_fmt(metrics.get('path_length_observed_px'), 'px')} "
            f"over {metrics.get('observed_frames')} observed frames",
            f"- largest single-frame observed step: "
            f"{_fmt(metrics.get('max_observed_step_px'), 'px')} — "
            f"{metrics.get('cadence_note')}",
            f"- operator-anchor path length: "
            f"{_fmt(metrics.get('path_length_operator_anchors_px'), 'px')}; "
            f"path length including derived steps: "
            f"{_fmt(metrics.get('path_length_including_interpolated_px'), 'px')} "
            f"(reference only)", ""]
        limits = metrics.get("limitations") or []
        if limits:
            lines += [f"  - {limit}" for limit in limits] + [""]

    # ---------------------------------------------------------------- LIMITS
    lines += ["## LIMITATIONS", ""]
    for statement in DISCLOSURE_LONG:
        lines.append(f"- {statement}")
    for limit in result.limitations:
        lines.append(f"- {limit}")
    unobserved = result.analytics.get("missingness", {})
    reasons = unobserved.get("body_missing_reasons") or {}
    examples = unobserved.get("body_missing_reason_examples") or {}
    lines += ["", "### Missingness, in counts", "",
              f"- body: {unobserved.get('body_frames_with_pose', 0)}/"
              f"{unobserved.get('body_frames_total', 0)} frames carry a pose",
              "- frames with no pose, grouped by reason:"]
    if reasons:
        for key, count in reasons.items():
            example = examples.get(key) or key
            lines.append(f"    - {count} frame(s): {example}")
    else:
        lines.append("    - none")
    lines.append("")
    for name, info in (unobserved.get("targets") or {}).items():
        detail = (f"{info.get('missing_frames')} with no bounded match + "
                  f"{info.get('lost_frames')} declared lost; these two counts are "
                  f"disjoint, not nested")
        extra = []
        if info.get("operator_stop_frames"):
            extra.append(f"{info['operator_stop_frames']} past the operator stop frame")
        if info.get("automatic_stop_frames"):
            extra.append(f"{info['automatic_stop_frames']} past the operator's "
                         f"automatic_until_frame")
        if extra:
            detail += (", of which " + " and ".join(extra)
                       + " (out of scope by the operator's own decision)")
        lines.append(f"- {name}: {info.get('observed_frames')} observed, "
                     f"{info.get('interpolated_frames')} interpolated, "
                     f"{info.get('unobserved_frames')} unobserved ({detail})")

    # ---------------------------------------------------------------- OUTPUTS
    render = result.render
    wh = render.get("wh") or [0, 0]
    if render.get("written"):
        video_line = (f"`{_video_name(result)}` ({render.get('bytes')} bytes, "
                      f"{render.get('frames')} frames, {render.get('fps')} fps, "
                      f"{wh[0]}×{wh[1]}, pix_fmt {render.get('pix_fmt')}, faststart "
                      f"{render.get('faststart')})")
    else:
        video_line = f"NOT PRODUCED — {render.get('reason')}"
    lines += ["", "## OUTPUTS", "",
              f"- annotated video: {video_line}",
              f"- keyframe cards: {len(result.keyframes.get('cards') or [])} written, "
              f"{len(result.keyframes.get('unavailable') or [])} explicitly "
              f"unavailable; contact sheet "
              f"`{result.keyframes.get('contact_sheet') or 'not produced'}`",
              "- `demo_analysis.json` is the machine-readable record of this run.",
              ""]
    if result.warnings:
        lines += ["## CONFIG WARNINGS", ""]
        lines += [f"- {w}" for w in result.warnings] + [""]
    if result.notes:
        lines += ["## NOTES", ""]
        lines += [f"- {n}" for n in result.notes] + [""]
    lines += ["## REPRODUCE", "",
              f"- config: `{result.config.get('path', '')}` "
              f"(sha256 `{result.config.get('sha256', '')}`)",
              f"- command: `python -m ghostcaddie demo-video "
              f"{s['path'] or s['display_name']} --config "
              f"{result.config.get('path', '')} --out {result.config.get('out_dir', '')}`",
              "- demo analysis and rendering run locally: this path makes no network "
              "call, requires no remote inference and downloads nothing, and the media "
              "was not uploaded.",
              ""]
    return "\n".join(lines)


def render_summary(result: DemoResult) -> str:
    """A short, quotable summary: what was built, what it shows, what it does not.

    Written so it can be adapted into a project post without any sentence that the
    artifacts cannot support: no physical quantity, no automatic-detection claim, and
    no claim about the development process (the code of this path is local-only; that
    is an application property, not a statement about how it was written).
    """
    s = result.source
    body = result.body
    tempo = result.tempo()
    lines = [f"# Demo summary — {s['display_name']}", "",
             f"{DISCLOSURE_SHORT}", "",
             "## What this is", "",
             "FairwayOS Demo Mode turns one chosen local golf clip into a presentable "
             "artifact set: an operator draws a region around the golfer, marks the "
             "ball and club, marks the swing phases, and the pipeline re-measures the "
             "body pose inside that region, resolves the phases and tempo, and renders "
             "an annotated video plus keyframe cards and a report. It runs on this "
             "machine, needs no remote inference, and is deliberately separate from "
             "the canonical `analyze-video` pipeline, which asks what the system can "
             "find on its own.", "",
             "## The clip", "",
             f"- clip: {s['width']}×{s['height']}, {s['fps']:g} fps container cadence, "
             f"{s['duration_seconds']:g}s; analysed interval = source frames "
             f"{result.interval['start_frame']}–{result.interval['end_frame']} "
             f"({result.interval['frame_count']} frames, every frame at native "
             f"cadence)"]
    lines.append(f"- body: {str(body.get('state', 'unavailable')).upper()} — "
                 f"{body.get('coverage', {}).get('frames_with_pose')}/"
                 f"{body.get('coverage', {}).get('frames_total')} frames carry a pose "
                 f"attributed to the operator's ROI")
    supported = result.body_supported()
    if supported:
        lead = ", ".join(f"{m['name']} {m['value']:.1f} {m['unit']}"
                         for m in supported[:4] if m.get("value") is not None)
        lines.append(f"- 2D quantities (image space, degrees or pixels): {lead}"
                     " (all medians over the frames that carry them)")
    else:
        lines.append("- 2D quantity table: none reached the reporting threshold")
    for name in ("ball", "clubhead"):
        metrics = result.target_metrics(name)
        track = result.target(name)
        if not track.get("available"):
            lines.append(f"- {name}: unavailable on this clip "
                         f"({(track or {}).get('reason', 'no seed')})")
            continue
        lines.append(f"- {name}: {metrics.get('observed_frames')} observed / "
                     f"{metrics.get('interpolated_frames')} interpolated / "
                     f"{metrics.get('unobserved_frames')} unobserved "
                     f"({metrics.get('missing_frames')} no-match + "
                     f"{metrics.get('lost_frames')} lost), "
                     f"{metrics.get('segment_count')} separate trail segments, "
                     f"image-space path "
                     f"{metrics.get('path_length_observed_px')} px")
    if tempo.get("state") == "supported":
        seconds = tempo.get("seconds_source_timeline") or {}
        lines.append(f"- tempo: backswing {tempo['backswing_frames']} frames"
                     + (f" ({seconds['backswing']:.2f}s)" if seconds.get("backswing")
                        is not None else "")
                     + f" / downswing {tempo['downswing_frames']} frames"
                     + (f" ({seconds['downswing']:.2f}s)" if seconds.get("downswing")
                        is not None else "")
                     + f" = {tempo['backswing_downswing_ratio']:.2f}:1 "
                       f"({tempo['label']} markers, source timeline)")
    else:
        lines.append(f"- tempo: unavailable ({tempo.get('reason', '')})")
    lines += ["- phases: "
              + ", ".join(f"{name}=" + ("unavailable" if result.phase(name).get("frame")
                                        is None else
                                        f"{result.phase(name)['frame']} "
                                        f"({result.phase(name)['label']})")
                          for name in PHASE_ORDER), "",
              "## What it does not claim", ""]
    lines += [f"- {limit}" for limit in DISCLOSURE_LONG]
    lines += ["- The seeds, keyframes and corrections are operator input: nothing in "
              "this artifact was produced by an automatic detector, and the ball and "
              "club tracks are not evidence of tracking capability.", "",
              "## If you post this", "",
              "A short version that stays inside the evidence: the operator-assisted "
              "demo mode measures one chosen swing in 2D image space — body pose "
              "inside an operator-drawn region, phase frames, tempo, and a "
              "hand-seeded ball/club track — and renders it locally. Built with "
              "AI-assisted development on top of this repository's local video "
              "pipeline; the demo analysis itself runs entirely on the machine (no "
              "remote inference). It is a presentation tool, not a perception result: "
              "the seeds, keyframes and corrections are operator input, every quantity "
              "is 2D and uncalibrated, and no physical metric is claimed.", ""]
    return "\n".join(lines)


def render_terminal_summary(result: DemoResult) -> str:
    s = result.source
    body = result.body
    lines = [
        f"source      {s['display_name']}  sha256={s['sha256'][:16]}...",
        f"interval    source frames {result.interval['start_frame']}-"
        f"{result.interval['end_frame']} ({result.interval['frame_count']} frames, "
        f"native cadence)",
        f"body        {str(body.get('state', 'unavailable')).upper()}  "
        f"({body.get('coverage', {}).get('frames_with_pose')}/"
        f"{body.get('coverage', {}).get('frames_total')} frames with a pose)",
    ]
    for name in ("ball", "clubhead"):
        track = result.target(name)
        metrics = result.target_metrics(name)
        if not track.get("available"):
            lines.append(f"{name:<11} UNAVAILABLE  ({(track or {}).get('reason', '')})")
        else:
            lines.append(
                f"{name:<11} {metrics.get('observed_frames')} observed / "
                f"{metrics.get('interpolated_frames')} interpolated / "
                f"{metrics.get('unobserved_frames')} unobserved, "
                f"{metrics.get('segment_count')} segment(s), path "
                f"{metrics.get('path_length_observed_px')}px")
    tempo = result.tempo()
    lines.append(f"tempo       " + (f"{tempo['backswing_frames']}f backswing / "
                                    f"{tempo['downswing_frames']}f downswing = "
                                    f"{tempo['backswing_downswing_ratio']:.2f}:1 "
                                    f"({tempo['label']})"
                                    if tempo.get("state") == "supported"
                                    else f"unavailable ({tempo.get('reason', '')})"))
    render = result.render
    lines.append(f"annotated   " + (f"{_video_name(result)} ({render.get('bytes')} bytes, "
                                    f"{render.get('frames')} frames)"
                                    if render.get("written")
                                    else f"NOT PRODUCED: {render.get('reason')}"))
    lines.append(f"disclosure  operator-assisted · experimental · 2D image-space · "
                 f"no physical metrics")
    lines.append(f"outputs     {result.config.get('out_dir', '')}")
    return "\n".join(lines)


def status_text(status: str) -> str:
    """Short on-video label for one point status. Never a confidence number."""
    return {STATUS_OBSERVED: "auto", STATUS_OBSERVED_LK: "auto+lk",
            STATUS_MANUAL: "manual", STATUS_INTERPOLATED: "interp",
            STATUS_MISSING: "", STATUS_LOST: ""}.get(status, status)
