"""Showcase report, LinkedIn draft and the re-runnable config, from one result.

Stdlib only, and rendered from the same :class:`~ghostcaddie.demo.result.DemoResult`
the video and the analysis JSON come from, so a number in the report cannot be a
different number from the one on the card.

The wording rules that matter here:

* no physical quantity is named anywhere (no speed, carry, launch angle, spin);
* the operator's inputs are named as operator inputs, never as detections;
* a card value that was unavailable in the video is reported as unavailable here;
* the LinkedIn draft is a DRAFT: it says the footage is local, rights are not
  claimed, and it must not be posted without the operator's own review.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .result import DISCLOSURE_SHORT, DemoResult
from .showcase import METRICS, PLACEHOLDER, format_value, phase_markers

#: Words that would turn an image-space demo into a physical claim. The draft and the
#: report are scanned for them by a test, and the generator never emits them.
BANNED_CLAIMS = ("mph", "m/s", "ball speed", "clubhead speed", "swing speed", "carry",
                 "apex", "launch angle", "spin rate", "rpm", "yards", "metres",
                 "meters", "smash factor", "ball flight law")


def _fmt(value: Any, digits: int = 1, unit: str = "") -> str:
    if value is None:
        return "unavailable"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}{unit}"


def render_showcase_report(result: DemoResult) -> str:
    """A presentation-run report: what was shown, with what inputs, and what not to
    conclude from it."""
    presentation = result.presentation or {}
    lines: List[str] = []
    add = lines.append
    add(f"# Showcase report — {result.source.get('display_name', 'clip')}")
    add("")
    add(f"_{DISCLOSURE_SHORT}_")
    add("")
    add(f"- run created (UTC): {result.created_utc}")
    add(f"- tool: fairwayos-demo {result.tool_version}")
    add(f"- presentation: **{presentation.get('style', 'demo')}** renderer "
        f"({presentation.get('renderer', 'n/a')})")
    add(f"- source: `{result.source.get('path')}`")
    add(f"- source sha256: `{result.source.get('sha256')}`")
    source_kind = result.source.get("description") or ""
    if source_kind:
        add(f"- source clip: {source_kind}")
    frame_count = result.source.get("frame_count")
    decoded = (result.interval or {}).get("frame_count")
    add(f"- source raster: {result.source.get('width')}x{result.source.get('height')} "
        f"@ {_fmt(result.source.get('fps'), 3)} fps, "
        + (f"{frame_count} frames" if frame_count
           else "container frame count not reported by the container; "
                f"{decoded if decoded is not None else 'the interval'} frames were "
                "decoded for this run"))
    interval = result.interval
    add(f"- processed interval: source frames {interval.get('start_frame')}.."
        f"{interval.get('end_frame')} ({interval.get('frame_count')} frames, "
        f"{interval.get('seconds_basis')})")
    add("")

    add("## What the video shows")
    add("")
    render = result.render or {}
    if render.get("written"):
        add(f"- file: `{result.presentation.get('output_paths', {}).get('annotated', '')}`"
            f" — {render.get('wh', ['?', '?'])[0]}x{render.get('wh', ['?', '?'])[1]}, "
            f"{render.get('frames')} frames, {render.get('codec') or 'h264'} / "
            f"{render.get('pix_fmt') or 'yuv420p'}, faststart "
            f"{bool(render.get('faststart'))}")
        add(f"- the picture fills the frame at the source's own aspect ratio: "
            f"{render.get('picture_wh', ['?', '?'])[0]}x"
            f"{render.get('picture_wh', ['?', '?'])[1]}, with "
            f"{int(render.get('added_width_px') or 0)}px of added width "
            f"(no gutter, no sidebar, no letterbox)")
        if render.get("dimensions_trimmed_to_even"):
            add("- the encoded size was trimmed to even dimensions, which H.264 "
                "yuv420p requires; every analysis coordinate is native source pixels")
        engine = (render.get("text_engine") or {})
        if engine:
            add(f"- type rendered by: {engine.get('engine')} "
                f"({engine.get('font_path') or 'built-in font'})")
    else:
        add(f"- no video was written: {render.get('reason') or 'unspecified'}")
    disclosure_line = presentation.get("disclosure")
    if disclosure_line:
        add(f"- disclosure drawn on the video: `{disclosure_line}`")
    cards = (presentation.get("cards") or {}).get("video_metrics") or []
    still_cards = (presentation.get("cards") or {}).get("still_metrics") or []
    if cards:
        add(f"- metric cards on the video: {', '.join(cards)}")
        add(f"- metric cards on a still: {', '.join(still_cards)} (at most three: a "
            f"still is read in a feed, not paused on a timeline)")
    add(f"- phase badge: {presentation.get('phase_badge_rule', '')}")
    add("")

    add("## What the chrome covers (measured, not assumed)")
    add("")
    coverage = ((presentation.get("chrome") or {}).get("measured_coverage") or {})
    if coverage:
        add(f"- golfer ROI area in the render: {coverage.get('roi_area_px2')}px²; "
            f"covered by chrome: {coverage.get('roi_covered_px2_total')}px² "
            f"({_fmt((coverage.get('roi_covered_fraction') or 0) * 100, 1, '%')})")
        per_band = coverage.get("roi_covered_px2_by_band") or {}
        if per_band:
            add("- by band: " + ", ".join(f"{name} {px}px²"
                                          for name, px in sorted(per_band.items())))
        add(f"- drawn ball/club points: {coverage.get('drawn_points_total')}; "
            f"under a card or badge: {coverage.get('drawn_points_under_chrome')} "
            f"({_fmt((coverage.get('drawn_points_under_chrome_fraction') or 0) * 100, 1, '%')})")
        add("- the cards are translucent by design; these numbers say what they cover "
            "rather than claiming they cover nothing. Running without "
            "`showcase.enabled` uses the demo renderer, which keeps chrome out of the "
            "picture entirely in an external gutter.")
    else:
        add("- not measured on this run (no showcase chrome was drawn).")
    problems = ((presentation.get("chrome") or {}).get("layout_problems") or [])
    add(f"- layout check: {'no problems reported' if not problems else '; '.join(problems)}")
    add("")

    playback = render.get("playback") or {}
    if playback:
        add("## Playback pace (presentation, not measurement)")
        add("")
        ordinal = {1: "st", 2: "nd", 3: "rd"}.get(int(playback.get("stride") or 1), "th")
        add(f"- stride: every {playback.get('stride')}{ordinal} source frame is encoded, at the "
            f"source's own cadence ({_fmt(playback.get('source_fps_unchanged'), 3)} fps)")
        add(f"- output: {playback.get('output_frames')} frames = "
            f"{_fmt(playback.get('output_seconds'), 2)}s, against "
            f"{_fmt(playback.get('source_interval_seconds'), 2)}s of source timeline "
            f"({playback.get('presentation_speedup')}x presentation speed)")
        holds = playback.get("holds") or []
        repeats = sum(max(0, int(hold.get("output_frames_held") or 0) - 1)
                      for hold in holds)
        add(f"- composition: {playback.get('sampled_source_frames')} sampled source "
            f"frames at stride {playback.get('stride')}"
            + (f", plus {repeats} inserted repeat(s) from {len(holds)} hold(s), so the "
               f"output has {playback.get('output_frames')} frames in total" if repeats
               else ", with no hold inserted")
            + ". The output-to-source frame map is recorded in "
              "showcase_provenance.json (`playback.output_to_source_frames`).")
        add(f"- mapping: {playback.get('rule')}")
        for hold in (playback.get("holds") or []):
            add(f"- hold at source frame {hold.get('source_frame')} "
                f"(`{hold.get('label')}`): output frames {hold.get('output_frames')} — "
                f"{hold.get('output_frames_repeated')} repeat(s) of the frame drawn at "
                f"output {hold.get('output_frame_drawn')}, "
                f"{hold.get('output_frames_held')} output frames in the span "
                f"({hold.get('status')}) — {hold.get('note')}")
        add(f"- {playback.get('note')}")
        add("")

    add("## Phase markers")
    add("")
    add("| phase | frame | label | seconds (source timeline) |")
    add("|---|---|---|---|")
    for name in ("address", "takeaway", "top", "downswing", "impact", "follow_through",
                 "finish"):
        record = result.phase(name)
        frame = record.get("frame")
        add(f"| {name} | {frame if frame is not None else '—'} | "
            f"{record.get('label', 'unavailable')} | "
            f"{_fmt(record.get('seconds_source_timeline'), 3)} |")
    add("")
    tempo = result.tempo()
    if tempo.get("state") == "supported":
        add(f"- tempo: {tempo.get('backswing_frames')} source frames backswing / "
            f"{tempo.get('downswing_frames')} downswing = "
            f"{_fmt(tempo.get('backswing_downswing_ratio'), 2, ':1')} "
            f"(start marker: {tempo.get('backswing_start_marker')}; "
            f"basis: {tempo.get('label')})")
        add(f"- {tempo.get('caveat')}")
    else:
        add(f"- tempo unavailable: {tempo.get('reason')}")
    add("")

    add("## Metric cards, as rendered")
    add("")
    add("These are the values on the timeline cards, read at each phase frame. A value "
        "the body layer did not support is `" + PLACEHOLDER + "` on the card and "
        "`unavailable` here — the same fact in both places.")
    debounce = ((result.render.get("card_values") or {}).get("debounce") or {})
    if any((entry or {}).get("held_count") for entry in debounce.values()):
        add("")
        add("A card holds its previous state until a new state has held for a few "
            "consecutive source frames, so a one-frame resolvability flicker cannot "
            "strobe between a number and `" + PLACEHOLDER + "`. A value shown across "
            "such a flicker is marked `(held)` below and is the last MEASURED value; "
            "the measured per-frame series in showcase_analysis.json is unchanged.")
        for metric, entry in sorted(debounce.items()):
            if not entry or not entry.get("held_count"):
                continue
            add(f"- {metric}: {entry.get('held_count')} held frame(s), state must hold "
                f"for {entry.get('min_run_frames')} consecutive source frames "
                f"(held frames: {_frame_list(entry.get('held_frames'))})")
    add("")
    frame_cards = [metric for metric in cards
                   if (METRICS.get(metric) or {}).get("source") == "frame"]
    clip_cards = [metric for metric in cards if metric not in frame_cards]
    if frame_cards:
        add("| metric | ADDRESS | TOP | IMPACT | FINISH |")
        add("|---|---|---|---|---|")
        for metric in frame_cards:
            add(_metric_row(result, metric))
    if clip_cards:
        add("")
        add("| metric | value for the whole run | basis |")
        add("|---|---|---|")
        for metric in clip_cards:
            add(_clip_metric_row(result, metric))
    add("")
    add("Metric definitions:")
    add("")
    for name in cards:
        spec = METRICS.get(name) or {}
        add(f"- **{spec.get('label', name)}** ({spec.get('space', '')}, "
            f"{spec.get('unit', '')}) — {spec.get('description', '')}")
    add("")

    add("## Body")
    add("")
    body = result.body or {}
    add(f"- state: **{body.get('state')}** (method {body.get('method')})")
    add(f"- pose on {body.get('coverage', {}).get('frames_with_pose')} of "
        f"{body.get('coverage', {}).get('frames_total')} frames inside the operator ROI")
    for supported in result.body_supported():
        add(f"- {supported['name']}: {_fmt(supported.get('value'))} "
            f"{supported.get('unit')} ({supported.get('frames_used')}/"
            f"{supported.get('frames_possible')} frames)")
    for missing in result.body_unavailable():
        add(f"- {missing['name']}: unavailable — {missing.get('reason')}")
    add("")

    add("## Ball and clubhead (operator-seeded, image space)")
    add("")
    for name in ("ball", "clubhead"):
        metrics = result.target_metrics(name)
        if not metrics.get("available"):
            add(f"- **{name}**: unavailable — {metrics.get('reason')}")
            continue
        add(f"- **{name}**: observed on {metrics.get('observed_frames')} of "
            f"{metrics.get('frames_total')} frames "
            f"({metrics.get('operator_anchor_frames')} operator anchor(s), "
            f"{metrics.get('automatic_observed_frames')} automatic observation(s), "
            f"{metrics.get('interpolated_frames')} derived), "
            f"{metrics.get('segment_count')} segment(s), path "
            f"{_fmt(metrics.get('path_length_observed_px'))}px over observed frames only")
        if metrics.get("interpolated_frames") and metrics.get("observed_frames") == \
                metrics.get("operator_anchor_frames"):
            add(f"  - the anchors are not on consecutive frames, so the observed-only "
                f"path is {_fmt(metrics.get('path_length_observed_px'))}px by "
                f"construction; including the derived (interpolated) points the "
                f"presented path is "
                f"{_fmt(metrics.get('path_length_including_interpolated_px'))}px of 2D "
                f"image-space travel, not a speed")
        add(f"  - interpolated {metrics.get('interpolated_frames')}, missing "
            f"{metrics.get('missing_frames')}, lost {metrics.get('lost_frames')}, "
            f"past the operator's stop frame {metrics.get('operator_stop_frames')}")
        if metrics.get("operator_stop_frames"):
            add(f"  - 'past the operator's stop frame' counts frames after "
                f"`targets.{name}.track_until_frame`: the operator declared the target "
                f"out of scope there, so nothing is drawn and nothing is inferred — it "
                f"is not a tracker loss")
    for name in ("ball", "clubhead"):
        for note in (result.target(name).get("limitations") or []):
            add(f"- **{name}**: {note}")
    add("")

    add("## Operator input (this is what makes the video possible)")
    add("")
    for record in (result.config.get("seeds") or {}).items():
        name, point = record
        add(f"- {name} seed: frame {point.get('frame')} at "
            f"({point.get('x')}, {point.get('y')})")
    corrections = result.config.get("corrections") or {}
    if corrections:
        for name, points in corrections.items():
            for point in points:
                add(f"- {name} correction: frame {point.get('frame')} at "
                    f"({point.get('x')}, {point.get('y')})")
    else:
        add("- no corrections were supplied")
    for name, record in sorted((result.render.get("manual_segments") or {}).items()):
        if not record:
            continue
        add(f"- {name} manual anchor segments — operator placements, authored by "
            f"{record.get('author') or 'operator'}"
            + (", REPLACING the automatic track for presentation"
               if record.get("replace_automatic") else
               ", laid over the automatic track"))
        for segment in (record.get("segments") or []):
            add(f"  - {segment.get('name')}: source frames {segment.get('start_frame')}"
                f"..{segment.get('end_frame')}, {segment.get('anchor_count')} operator "
                f"anchor(s) at most {segment.get('max_anchor_spacing_frames')} frame(s) "
                f"apart, {segment.get('interpolated_frames')} interpolated (derived) "
                f"frame(s)")
            for anchor in (segment.get("anchors") or []):
                add(f"    - frame {anchor.get('frame')}: ({anchor.get('x')}, "
                    f"{anchor.get('y')}) {anchor.get('note') or ''}".rstrip())
        hidden = record.get("hidden_frames") or []
        if hidden:
            add(f"  - operator-declared unusable frames ({len(hidden)}): "
                f"{_frame_list(hidden)} — nothing is drawn there and nothing is "
                f"interpolated across them")
        if record.get("replace_automatic"):
            dropped = ((record.get("replaced_automatic_track") or {})
                       .get("automatic_observed_frames_dropped"))
            add(f"  - the automatic track held {dropped} observation(s) on these frames; "
                f"they are still in showcase_analysis.json and the raw measurements, and "
                f"are not drawn")
        if (record.get("trail") or "all") != "all":
            add(f"  - trail rule: {record.get('trail')} (the trail is drawn only inside "
                f"these operator-anchored segments, so it cannot trace camera pan behind "
                f"a stationary target)")
        if record.get("trail_from_frame") is not None:
            add(f"  - trail bound: no trail before source frame "
                f"{record.get('trail_from_frame')} (the marker is still drawn there)")
    add(f"- ROI: {result.config.get('roi_xywh')}")
    keyframes_roi = (result.presentation.get("chrome") or {}).get("roi_keyframes_used")
    roi_keys = result.config.get("roi_keyframes") or []
    if roi_keys:
        add("- moving crop (operator boxes, interpolated between them in position and "
            "size; the crop follows the camera pan and is not a tracked subject):")
        for key in roi_keys:
            add(f"  - frame {key.get('frame')}: {key.get('roi')} "
                f"— {key.get('note') or 'operator box'}")
        add(f"  - region the crop moves through: "
            f"{result.config.get('roi_envelope_xywh') or keyframes_roi}")
    add(f"- wrist reference for hand rise: {result.render.get('wrist_reference')}")
    add(f"- keyframes: {json.dumps(result.config.get('keyframes') or {}, sort_keys=True)}")
    add("- seeds, keyframe marks, corrections, the ROI and the manual anchor segments "
        "are operator input. Nothing in this artifact is described as a detection: the "
        "clubhead presentation is operator-anchored geometry (with linear interpolation "
        "between anchors labelled derived), the ball is an operator-seeded template "
        "track plus the operator's own anchors, the phase frames and the golfer region "
        "are operator marks, and the body landmarks are model-estimated (MoveNet) inside "
        "the operator's crop and reported as estimates.")
    add("")

    add("## What this is not")
    add("")
    for statement in result.disclosure().get("statements") or []:
        add(f"- {statement}")
    for limitation in result.limitations:
        add(f"- {limitation}")
    if result.warnings:
        add("")
        add("Config warnings:")
        add("")
        for warning in result.warnings:
            add(f"- {warning}")
    add("")
    add("## Artifacts")
    add("")
    for role, path in (presentation.get("output_paths") or {}).items():
        add(f"- {role}: `{path}`")
    for card in (result.keyframes.get("cards") or []):
        add(f"- still {card.get('phase')} (frame {card.get('frame')}, "
            f"{card.get('label')}): `{card.get('path')}`")
    for item in (result.keyframes.get("missing") or []):
        add(f"- still {item.get('phase')}: UNAVAILABLE — {item.get('reason')}")
    add(f"- contact sheet: `{result.keyframes.get('contact_sheet')}` "
        f"(panels: {', '.join(result.keyframes.get('contact_sheet_panels') or [])})")
    add("")
    return "\n".join(lines) + "\n"


def _frame_list(frames: Any, limit: int = 24) -> str:
    """A short, honest frame list for the report: the first few, then the count."""
    values = [int(frame) for frame in (frames or [])]
    if not values:
        return "none"
    if len(values) <= limit:
        return ", ".join(str(value) for value in values)
    return (", ".join(str(value) for value in values[:limit])
            + f", ... ({len(values)} frames in total)")


def _display_series(result: DemoResult, metric: str) -> Dict[int, dict]:
    """The card's debounced display series, recomputed from the SAME pure function.

    The report has to print what the CARD showed, including the frames where the card
    held its previous state. Recomputing the series with the same helper on the same
    per-frame records (instead of storing a second copy of a 1,100-frame table in the
    JSON) keeps the two in step by construction: same inputs, same function, same output.
    """
    from .analytics import body_angles_for_record
    from .showcase import DEFAULT_CARD_DEBOUNCE_FRAMES, debounce_series, frame_values

    debounce = ((result.render.get("card_values") or {}).get("debounce") or {})
    min_run = int((debounce.get(metric) or {}).get("min_run_frames")
                  or DEFAULT_CARD_DEBOUNCE_FRAMES)
    wrist = ((result.render or {}).get("wrist_reference") or {}).get("y")
    handedness = result.config.get("handedness", "right")
    values: Dict[int, Optional[float]] = {}
    for record in (result.body.get("per_frame") or []):
        angles = (body_angles_for_record(record, handedness=handedness)
                  if record.get("available") else {})
        values[int(record["source_frame"])] = frame_values(
            angles, handedness=handedness, wrist_reference=wrist).get(metric)
    return debounce_series(values, min_run=min_run)


def _metric_row(result: DemoResult, metric: str) -> str:
    """One row per frame metric: the DISPLAYED value at each of the four phase frames.

    ``(held)`` marks a value the card showed across a bounded flicker of fewer than
    ``card_debounce_frames`` frames: it is the last MEASURED value, and the measurement
    for the frame itself is in showcase_analysis.json.
    """
    label = (METRICS.get(metric) or {}).get("label", metric)
    unit = (METRICS.get(metric) or {}).get("unit", "")
    display = _display_series(result, metric)
    cells: List[str] = []
    for phase in ("address", "top", "impact", "finish"):
        record = result.phase(phase)
        frame = record.get("frame")
        entry = display.get(int(frame)) if frame is not None else None
        value = (entry or {}).get("value")
        if value is None:
            cells.append(PLACEHOLDER)
            continue
        text = format_value(metric, value, unicode_ok=False)
        if (entry or {}).get("held"):
            text += " (held)"
        cells.append(text)
    return f"| {label} ({unit}) | " + " | ".join(cells) + " |"


def _clip_metric_row(result: DemoResult, metric: str) -> str:
    """A clip-level card: one number for the whole run, with its coverage."""
    from .showcase import clip_values

    label = (METRICS.get(metric) or {}).get("label", metric)
    unit = (METRICS.get(metric) or {}).get("unit", "")
    values = clip_values(result.analytics, result.phases,
                         handedness=result.config.get("handedness", "right"))
    entry = values.get(metric) or {}
    value = entry.get("value")
    text = format_value(metric, value, unicode_ok=False) if value is not None \
        else f"{PLACEHOLDER} ({entry.get('reason') or 'unavailable'})"
    return f"| {label} ({unit}) | " + text + " | (whole clip, one value) |"


def render_linkedin_draft(result: DemoResult) -> str:
    """A draft post for internal review. Never posted by this tool."""
    presentation = result.presentation or {}
    cards = (presentation.get("cards") or {}).get("video_metrics") or []
    lines: List[str] = []
    add = lines.append
    add("# LinkedIn post — DRAFT (operator review required before publishing)")
    add("")
    add("Draft status: generated from a local, operator-assisted run. It has not been ")
    add("posted, scheduled or uploaded anywhere, and this tool never publishes.")
    add("")
    add("---")
    add("")
    add("## Draft copy")
    add("")
    add("> Slow-motion, frame by frame, with the numbers on screen.")
    add(">")
    add("> This is an **operator-assisted 2D image-space** analysis of one golf swing "
        "from a camera position: the golfer region, the ball and club points and the "
        "phase frames were marked by hand, and everything drawn is measured in the "
        "image plane.")
    add(">")
    add(f"> On screen: {', '.join(humanise(m) for m in cards) if cards else 'body angles'} "
        "— 2D projections from a single uncalibrated camera, not coaching numbers and "
        "not 3D.")
    add(">")
    add("> No speed, distance, launch angle or spin is measured here, and no physical "
        "claim is made. It is a visualisation of one chosen clip, shown locally.")
    add(">")
    add("> Built as a local pipeline: no upload, no external service, no downloads.")
    add("")
    add("---")
    add("")
    add("## What must be true before this is posted")
    add("")
    add("- The clip is used for local, internal review only. **No rights to the "
        "footage are claimed, and none are implied by this draft**; the poster is "
        "responsible for having the right to publish any clip, music or likeness.")
    add("- The disclosure line visible in the video (`"
        f"{presentation.get('disclosure', 'OPERATOR-ASSISTED / 2D IMAGE-SPACE')}`) "
        "stays in the post or in the first comment.")
    add("- Nothing in the post may be tightened into a physical claim (no ball or "
        "club speed, no carry, no launch angle, no spin).")
    add("- Any numbers quoted must come from `showcase_report.md` for this exact run "
        f"(source sha256 `{result.source.get('sha256', '')[:16]}…`).")
    add("")
    add("## Run facts (for the caption, if useful)")
    add("")
    add(f"- source: {result.source.get('display_name')} "
        f"({result.source.get('width')}x{result.source.get('height')}, "
        f"{result.source.get('fps')} fps container cadence)")
    if result.source.get("description"):
        add(f"- clip: {result.source['description']}")
    add(f"- interval: source frames {result.interval.get('start_frame')}–"
        f"{result.interval.get('end_frame')} "
        f"({result.interval.get('frame_count')} frames)")
    markers = phase_markers(result.phases)
    frames = {name: frame for name, frame in markers.items() if frame is not None}
    if frames:
        add("- phase frames: " + ", ".join(f"{name} f{frame}"
                                           for name, frame in frames.items()))
    tempo = result.tempo()
    if tempo.get("state") == "supported":
        add(f"- tempo: backswing {tempo.get('backswing_frames')} frames vs downswing "
            f"{tempo.get('downswing_frames')} frames "
            f"({_fmt(tempo.get('backswing_downswing_ratio'), 2, ':1')}), source-frame "
            f"counts at container cadence — not measured action time")
    add(f"- body layer: {result.body.get('state')}")
    add("")
    add("## Attachments (local files, not uploaded)")
    add("")
    add(f"- video: `{presentation.get('output_paths', {}).get('annotated', '')}`")
    add(f"- contact sheet: `{presentation.get('output_paths', {}).get('contact_sheet', '')}`")
    add("")
    add("---")
    add("")
    add(DISCLOSURE_SHORT)
    add("")
    return "\n".join(lines) + "\n"


def humanise(metric: str) -> str:
    """A card's metric name as a reader would say it."""
    return {
        "shoulder_tilt": "shoulder tilt (2D)",
        "hip_tilt": "hip tilt (2D)",
        "knee_flexion": "lead knee flexion",
        "head_movement": "head movement (image space)",
        "hand_rise": "hand rise (image space)",
        "sway": "hip-centre sway (image space)",
        "tempo": "tempo (backswing:downswing frames)",
    }.get(metric, metric.replace("_", " "))


def showcase_config_payload(config, resolved, probe, source_sha: str, spec: dict,
                            result: DemoResult) -> dict:
    """The run's config, plus a ``_resolved`` record, written next to the artifacts.

    Keys beginning with ``_`` are ignored by the config schema, so this file still
    loads as a valid config for a re-run while carrying the frames it actually bound
    to. That is deliberately one file rather than two: a re-run from this file cannot
    use numbers from a different resolution.
    """
    payload = config.to_dict()
    payload["_resolved"] = {
        "source": {"path": probe.path, "display_name": probe.display_name,
                   "sha256": source_sha, "width": probe.width, "height": probe.height,
                   "fps": probe.fps, "frame_count": probe.frame_count},
        "source_frames": {"start_frame": resolved.start_frame,
                          "end_frame": resolved.end_frame,
                          "frame_count": resolved.frame_count},
        "seconds_basis": resolved.seconds_basis,
        "endpoint_semantics": resolved.effective().get("endpoint_semantics"),
        "showcase_effective": spec.get("showcase"),
        "presentation_style": (result.presentation or {}).get("style"),
        "tool_version": result.tool_version,
        "created_utc": result.created_utc,
        "note": ("written by fairwayos-demo; keys beginning with an underscore are "
                 "ignored by the config schema, so this file can be passed straight "
                 "back to demo-video for a re-run of exactly these frames"),
    }
    return payload
