# Demo Mode — operator-assisted, selected-clip analysis

Demo Mode is a **local, selected-clip visualization tool** for a carefully chosen golf
swing. It is deliberately *not* the canonical pipeline, and the two answer different
questions.

| | Canonical (`analyze-video`) | Demo (`demo-video`) |
|---|---|---|
| Question | what can this system detect **on its own**? | can an operator **show** a swing clearly, on one chosen clip? |
| Body | conditional 2D pose; `ambiguous`/`unavailable` when the golfer cannot be identified | same body pipeline, but an operator ROI may be supplied to say *which* golfer |
| Ball / clubhead | `UNAVAILABLE` — every route withdrawn or falsified | operator-seeded, operator-corrected 2D image-space tracks, clearly labelled |
| Physical metrics | `UNAVAILABLE` — no validated calibration | still `UNAVAILABLE` — only pixel/image-space quantities are printed |
| Seeding / corrections | never | expected, recorded, and disclosed everywhere |
| Failure mode | declines | declines *and* says what the operator would have to supply |

Demo Mode never upgrades a seeded or manually-corrected result into an automatic
detection. Every artifact it writes carries an operator-input record, and every frame
it renders carries the words **OPERATOR-ASSISTED / EXPERIMENTAL / 2D IMAGE-SPACE**.

The v3 ball research stays closed here. Demo Mode does not train, benchmark, evaluate
a held-out split, or attempt promotion; it does not read sealed research evaluation
media. It uses only clips an operator has chosen and may show locally.

## Quickstart

```bash
# 1. (optional but recommended) pick interval, ROI, seeds and phases by clicking
#    --out CREATES a config; --config EDITS an existing one
.venv/bin/python3 -m ghostcaddie demo-setup CLIP.mp4 --out demo_config.json

# 2. render and report
.venv/bin/python3 -m ghostcaddie demo-video CLIP.mp4 \
    --config demo_config.json --out demo_out/
```

```text
demo_out/
  demo_annotated.mp4     H.264 / yuv420p / +faststart, source playback cadence
  demo_report.md         the seven report sections
  demo_summary.md        short LinkedIn-adaptable summary (optional)
  demo_analysis.json     per-frame body/target status, phases, tempo, coverage
  demo_provenance.json   source and config hashes, resolved config, operator inputs
  keyframes/             one card per marked phase, plus contact_sheet.png (the four
                         primary phases on one 2x2 sheet)
  debug/                 panel layout, coverage, tracker decisions, timing notes
```

Both commands are local-only: no network access, no downloads at setup or run time, no
external vision service. The job is dispatched to an interpreter that has OpenCV
(`.venv/bin/python3` here); the CLI itself imports nothing outside the standard library
and never imports OpenCV, so the command stays usable on a machine without it.

## The picker (`demo-setup`)

`demo-setup` opens a window with the picture on the left and a status gutter on the
right. The gutter never covers the footage, so the point being clicked stays visible.

The frame is shown through an explicit viewport transform. `--viewport` (default
1300x860) bounds the **whole window**: the HUD gutter is part of it, so the picture
gets what is left and the window fits a laptop display. Clicks, drags and the ROI are
converted back to **native source pixels**, and the current scale is printed in the
gutter. A click in the letterbox area is ignored rather than silently mapped to a
corner pixel.

| key | action |
|---|---|
| `d` / `→` / `.` | next frame (`n`/`N`: ±10) |
| `a` / `←` / `,` | previous frame |
| `]` / `[` | ±1 second |
| `i` / `o` | set interval start / end at this frame (end is inclusive here) |
| `g` | clear the interval (whole clip) |
| left-drag | draw the ROI box |
| `x` | clear the ROI |
| `1` / `2` | seed ball / clubhead at the mouse position |
| `3` / `4` | add a ball / clubhead correction at the mouse |
| `5` / `6` | clear ball / clubhead corrections |
| `A` `T` `P` `D` `I` `F` `Z` | mark address / takeaway / top / downswing / impact / follow-through / finish — use the **upper-case** keys, because lower-case `a`/`d`/`i` are navigation |
| `C` | clear all phase marks |
| `h` / `?` | toggle help in the gutter |
| `H` | toggle the native-coordinate readout |
| `s` / `w` | save the config JSON |
| `q` / `Esc` | quit (once if clean, twice if there are unsaved edits) |

The session is dirty-checked: quitting with unsaved edits needs a second `q`, and a
config that fails validation is **not written** — the gutter says why instead. Reopening
an existing config (`--config`) keeps the interval, ROI, seeds, phase marks, per-target
tracker tuning, interpolation limits and render settings, so tuning survives a rerun.
Editing JSON by hand remains fully supported and is the reproducible alternative.

## The rendered video

The picture is never covered by the analytics. Before any frame is drawn the demo
measures two no-go regions in the output's own pixels — the operator's ROI (the
golfer) and the bounding box of every target point it is about to draw (the ball and
club action) — and puts the panel in a side column **only** when the picture leaves a
band twice the panel's width clear of both. Otherwise the output canvas is widened and
the panel sits in an **external gutter** beside the picture, which is what a portrait
golfer in a landscape frame with a flight across the other side actually needs.

Chrome (header band, panel, legend) is drawn **before** the overlays, so a marker or a
trail is never dimmed by the panel it happens to cross. The panel carries only the
phase badge, the three body angles and the tempo; per-frame observed/interpolated/
missing tallies and window bounds live in `demo_report.md` and `debug/`, where they can
be read without competing with the swing. Keyframe cards are rendered from the frame
with the skeleton and markers **only** — no header, panel or legend — plus their own
caption strip with the label, the angles and the disclosure.

## Config reference

`examples/demo_config.template.json` is a template you can parse as-is (a test does),
but **every value in it is illustrative and must be replaced**: the numbers are
generic placeholders, not measurements of any clip, and they describe nothing
generally. Keys beginning with `_` are ignored by the schema, so a config can carry
its own explanation. Everything is optional except `schema_version`; omitted sections
take defaults.

```json
{
  "_comment": "keys starting with _ are ignored, so a config can explain itself",
  "schema_version": "fairwayos-demo-config.v1",
  "clip_path_hint": "CLIP.mp4",
  "interval": {"start_seconds": 0.8, "end_seconds": 7.44, "max_frames": 700},
  "timing": {"source_cadence_caveat": "unknown", "nominal_fps_trusted": false,
             "accept_variable_cadence": false},
  "roi": [640, 60, 760, 1080],
  "seeds": {
    "ball":     {"frame": 24, "x": 700.0, "y": 900.0},
    "clubhead": {"frame": 36, "x": 640.0, "y": 880.0}
  },
  "keyframes": {
    "address": 24, "takeaway": 36, "top": 92, "downswing": 106,
    "impact": 118, "follow_through": 126, "finish": 176
  },
  "targets": {
    "ball":     {"mode": "template", "template_size": 31, "search_radius": 40,
                 "min_score": 0.5, "max_jump_px": 60.0, "loss_frames": 4,
                 "track_until_frame": 130, "automatic_until_frame": 116,
                 "lk_refine": true},
    "clubhead": {"mode": "template", "template_size": 41, "search_radius": 90,
                 "min_score": 0.45, "max_jump_px": 140.0, "loss_frames": 3,
                 "track_until_frame": 130}
  },
  "corrections": {
    "ball":     [{"frame": 118, "x": 860.0, "y": 780.0},
                 {"frame": 121, "x": 960.0, "y": 700.0}],
    "clubhead": [{"frame": 116, "x": 600.0, "y": 880.0}]
  },
  "interpolation": {"enabled": true, "max_gap_frames": 3},
  "render": {"max_width": 1280, "crf": 20, "preset": "medium", "render": true},
  "handedness": "right",
  "allow_approx_phases": false,
  "notes": "illustrative values only - replace them and record your own operator notes"
}
```

**Interval endpoints.** `start_seconds` is inclusive. `end_seconds` is **exclusive**
and may be `null` (run to the end of the clip). Frames are
`start_frame = floor(start_seconds * fps)` and `end_frame = ceil(end_seconds * fps) - 1`,
so `0.8 → 7.44` at 25 fps runs source frames **20..185**. Every frame index in the
config is a zero-based **source** frame, not an index into the render. `max_frames`
is a hard cap and an explicit refusal, never a silent truncation.

**Timing.** The demo reads the **decoded frames' own presentation timestamps**
(`ffprobe -show_entries frame=best_effort_timestamp_time`), not the packets: with
B-frames, packets are stored in decode order and their timestamps go backwards, so a
perfectly constant clip would look variable. A cadence that is not constant is
**refused** unless `timing.accept_variable_cadence` is true; if it is accepted,
provenance records that only frame indices are reliable. `timing.source_cadence_caveat`
defaults to `"unknown"` and should say what the operator actually knows (for example
that the footage is a replay) — the demo will not present slowed footage as measured
action time. Phase and tempo seconds are positions in the **source** timeline;
interval-relative seconds are emitted separately as
`seconds_from_interval_start` and labelled as such.

**Corrections** are manual restart points: `{"frame", "x", "y"}` per target, unique per
frame, ordered, and required to lie inside the interval and the frame. A correction
ends the automatic segment: the trail restarts there and interpolation never bridges
across it.

**Stop frames.** `track_until_frame` puts the target out of scope entirely after that
frame (nothing is drawn or inferred). `automatic_until_frame` only stops the
**matcher**: operator corrections after it are still honoured, which is how a flight
that moves faster than the template search is reported as exactly the operator's own
anchors instead of a false match. Frames between those anchors are counted
separately and never declared lost or interpolated.

**Interpolation** is opt-in and never crosses a loss, a correction or the tracked
range. Interpolated points are labelled `interpolated`, drawn dashed, counted
separately from observations, and never reported as observations.

**Missing input is not an error.** Omit a seed, a phase mark or the whole `roi`
section and the corresponding analysis is reported `unavailable` with a reason.

## What is measured

All geometry is 2D image-space, in source pixels, using the source's pixel aspect
ratio; normalized keypoint x/y are never mixed to produce degrees.

* **body** — shoulder tilt, hip tilt, knee interior angle (named, with its vertex),
  head/nose displacement, body-centre lateral sway, wrist path and wrist height.
  A landmark that is not visible contributes nothing rather than interpolating.
* **swing phases** — address, top, impact, finish, each labelled `operator-marked`
  or `approximate` (derived from the pose track) and never inferred from evenly
  spaced frames.
* **tempo** — backswing and downswing frame counts, their ratio, and per-phase
  seconds from the source timeline. The start marker is the operator's `takeaway`
  when given, otherwise `address`, and which one was used is stated.
* **ball / club** — image-space displacement and observed segment path length in
  pixels. Loss jumps and correction jumps are never counted as motion.

Physical quantities — ball speed, clubhead speed, carry, apex, spin, launch angle —
are **not** produced anywhere in this mode, and the report says so explicitly. Image-space
or approximate values are flagged as such in the report, in `demo_analysis.json`, in
`demo_provenance.json`, and in the video legend.

## Limitations (also printed in every report)

1. **2D only.** Everything is measured in the image plane; no depth, no 3D inference.
2. **Perspective.** A camera not square to the swing plane distorts angles and lengths.
3. **Manual initialization.** Ball and clubhead tracks exist only because an operator
   placed seeds/corrections. They are not detections and not evidence of capability.
4. **Replay / source speed caveat.** Frame timestamps say nothing about real-world
   timing if the footage is slowed or re-timed; tempo in seconds is source-timeline.
5. **No physical calibration.** No distance scale, no club/ball model, so no mph, no
   carry, no spin, no launch angle.
6. **Operator ROI.** The body result is conditional on the region the operator drew.
7. **Tracking is per-clip.** The template/optical-flow settings suit the selected
   clip. Losses are reported as losses; long losses stay missing.
8. **Selected clip, not a corpus.** Nothing here supports a claim about other footage,
   and success on one chosen clip is not accuracy.

## Troubleshooting

* **`ambiguous` / `unavailable` body** — widen or redraw the ROI so one golfer is
  fully inside it; the canonical identity challenge is still applied.
* **Track lost immediately** — check the seed frame against the real ball position
  (`H` shows native coordinates), enlarge `search_radius`, lower `min_score`, or add a
  correction at the next frame where the target is visible.
* **Zero-length path** — the target never moved in image space over observed frames,
  or the seed patch was too flat to match (the debug record says `flat_low_texture`).
* **Refused config** — the message names the field and the reason; demo mode refuses
  rather than truncating, so a wrong frame index cannot silently run anyway.
