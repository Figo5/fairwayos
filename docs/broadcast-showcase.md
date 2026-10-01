# Broadcast showcase — operator workflow

One whole shot, rendered end to end at the source's own cadence with the source's own
audio, where overlays and metric cards exist **only** inside the golfer-visible scene
windows the operator declares. This is a third mode alongside `analyze-video`
(canonical) and `demo-video` (one interval): it answers "show this whole broadcast shot
honestly", not "what can the system detect on its own".

```
canonical analyze-video    what can the system detect with no operator input?
demo-video                 show ONE interval of a chosen clip, polished
broadcast-showcase         show the WHOLE shot, overlays only in declared windows
```

## The four non-negotiables

1. **Outside a window, nothing is drawn.** No skeleton, no ball/club marker, no trail,
   no held card value, no phase badge, no `hand_rise` reference. At most the small
   persistent brand mark stays (`outside_windows: "brand_only"`; `"clean"` draws the
   untouched source picture).
2. **No scene inherits another scene.** Windows may not overlap, a frame belongs to one
   scene, and every per-scene quantity (body pose, tracks, phase marks, card debounce,
   `hand_rise` reference, impact accent) is computed from that scene's frames alone.
3. **Nothing is stretched.** The output has the same frame count, the same **rational**
   frame rate and the same picture as the source. There is no `stride` and no `holds`
   in a broadcast config.
4. **The source audio is copied, not re-encoded** (`-c:a copy`), and the run proves it
   by comparing the audio packet md5 of the output with the source's.

## The config

Schema id: `fairwayos-broadcast-config.v1`; the checked parser and contract live in
`ghostcaddie/demo/broadcast_config.py`. Start from
[`examples/broadcast_config.template.json`](../examples/broadcast_config.template.json)
or the putt variant
[`examples/broadcast_config_putt.template.json`](../examples/broadcast_config_putt.template.json);
**every value in them is a placeholder**.

Frame semantics, stated once: every frame index is a zero-based frame **of the clip you
pass on the command line**, and a scene window is
`{"start_frame": a, "end_frame": b}` with **both ends inclusive**. Frames are never
round-tripped through seconds (at 9832/331 fps, `floor(seconds * fps)` is off by one for
many frames, so a window that travelled through seconds would silently move).

Scene blocks are the demo-config blocks (`roi`, `roi_keyframes`, `seeds`, `targets`,
`corrections`, `manual_segments`, `interpolation`, `handedness`,
`allow_approx_phases`), validated by the demo config parser itself, so
[demo-mode.md](demo-mode.md) documents them.

The `shot` block also carries the operator's own declarations about the footage, and
provenance re-states them verbatim rather than deriving them:

| key | meaning |
|---|---|
| `expected_source_sha256` | the clip this config was written for; the run refuses other footage |
| `parent_source_manifest` | *where the parent (pre-trim) source digest and native trim frames are recorded*. May be absolute or relative to the config file |
| `source_notes` | the operator's own words about the clip |
| `expected_fps`, `expected_frames` | declared expectations, recorded for cross-check |

`parent_source_manifest` is optional. When it is given it wins; when it is not, the
manifest is looked for **beside the clip** — `<clip-folder>/../manifest.json` (the
baseline layout) and `<parent>/<clip-folder>_manifest.json` (a second batch's layout) —
so no path on one operator's disk is baked into the tool. Either spelling of the trim
frames is read (`start_frame`/`end_frame_exclusive` or
`source_start_frame`/`source_end_frame_exclusive`) and the record says which one it
used. A clip with no manifest anywhere is reported as *having none*, and nothing about
its parent is claimed; a manifest that exists but does not list the clip is reported the
same way.

Phases are per stroke type:

| `stroke_type` | phases |
|---|---|
| `full_swing`, `chip`, `pitch`, `bunker`, `unknown` | address, takeaway, top, downswing, impact, follow_through, finish |
| `putt` | address, **backstroke**, **forward_stroke**, **contact**, finish |

A putt has no TOP: naming one on a putt scene is refused, and no putt artifact contains
the strings `TOP`, `DOWNSWING`, `IMPACT` or `SWING ANALYSIS` (a test greps for them).
A putt's tempo is reported as `back : through` from the operator's own three marks, and
is unavailable — not estimated — without all three.

Each scene also declares a `kind`. Only `golfer_stroke` may carry tracking inputs; a
`reaction`, `flight`, `green`, `replay` or `other` window is how the operator says *why*
a stretch has no overlay, and it is refused if it carries a seed, anchor, ROI or phase
mark.

## Command line

```sh
# 1. what does this clip cut to? (suggestions with confidence, nothing applied)
.venv/bin/python -m ghostcaddie broadcast-showcase scenes CLIP.mp4 --out cuts.json

# 2. is my config sound? (parses, binds to the clip, checks every window) - renders nothing
.venv/bin/python -m ghostcaddie broadcast-showcase validate CLIP.mp4 --config C.json

# 3. render one clip
.venv/bin/python -m ghostcaddie broadcast-showcase run CLIP.mp4 --config C.json \
    --out /path/to/showcases/my_shot \
    --cv-python /absolute/path/to/python-with-cv2 \
    --body-model /absolute/path/to/movenet-singlepose-lightning-v3.tflite

# 4. render every clip in a manifest, as bounded subprocesses
.venv/bin/python -m ghostcaddie broadcast-showcase batch --manifest clips.json \
    --out /path/to/showcases/batch_2026_10_01

# 5. the browser review UI (loopback only)
.venv/bin/python -m ghostcaddie broadcast-showcase ui --port 8899 \
    --out-root /path/to/showcases --config-dir /path/to/configs
```

`scenes` and `validate` are stdlib-only and import no OpenCV. `run` and `batch` dispatch
the CV work into the `--cv-python` interpreter, exactly like `demo-video`.

## The browser review UI

`broadcast-showcase ui` serves a local page on **127.0.0.1** and prints the URL. What it
does, all of it real endpoints:

* **import / pick a clip** from the allow-listed roots (the baseline `clips/` directory
  and the run directory by default; `--extra-root` adds more). The selector lists media
  found **recursively** (depth 3, bounded count), so a batch sitting in
  `new-clips/09_hole03_tee.mp4` is selectable rather than invisible, and an explicit
  absolute path inside an allow-listed root can be imported too. Both routes are
  realpath-checked against the roots.
* **preview frames** decoded on the server by exact frame index (`select=eq(n\,F)`), so
  the frame number you mark is the frame number that is rendered. The browser's
  `<video>` time is never used to decide a frame.
* **drag the golfer region** over a real frame; the mapping from the displayed picture
  back to native source pixels is shared with a checked-in vector file
  (`tests/fixtures/viewport_vectors.json`) and a click in the letterbox is refused.
* **mark phases** with buttons that follow the chosen stroke type.
* **place a ball or club seed/correction** on a frame, with an explicit visible-target confirmation for seeds; edit assistance marks or add bounded manual segments.
* **load an existing config** onto its hash-matched clip, retaining crop keyframes, manual marks, cut decisions, timing, presentation and notes on save.
  The list shows **broadcast configs only** (a cut scan or a batch manifest in the same
  folder is not offered), and each entry names the clip it belongs to. Loading resolves
  `expected_source_sha256` (then `clip_path_hint`) against the registry, **selects that
  clip and refreshes its frame bounds before any scene field is loaded**, and refuses
  with *"import that clip first"* when nothing matches — a config can no longer be
  edited, or rendered, against whichever clip happened to be selected.
* **the loaded document IS the state.** Loading deep-clones the file; every edit is made
  in that copy, so `roi_keyframes`, `manual_segments`, `presentation`, `timing`, `cuts`,
  `overrides`, `notes` and any `_`-prefixed comment survive a save unchanged. Save
  validates the document and writes **the operator's own JSON back**, not a re-rendering
  of the parsed model, so a save cannot drop an assistance mark. (A scene's own
  `metric_hide_frames` is preserved by the same mechanism.)
* **edits commit to the selected scene.** Clicking a scene loads its window, phases, ROI
  and seeds into the editors; changing any of them (phase mark, dragged region, seed,
  window, stroke type) writes them straight back into that scene, and *apply fields to
  the selected scene* does it explicitly. *add as a new scene* is the separate action,
  and it says which phase marks were not carried over because they lie outside the new
  window.
* **assistance marks are editable, per target.** The panel lists the selected scene's
  seeds, corrections, ROI keyframes and manual-segment anchors with a *remove* for each;
  the placing modes are explicit — **region** (drag), **seed** (click, with the
  visible-ball confirmation) and **correction** (click, a position the operator records,
  no claim of a detection). *save as ROI keyframe* commits the drawn region at the
  current frame, and *new segment* / *add anchor* create or extend a bounded
  `manual_segments` block (the parser's spacing cap is checked on validate).
* **scan for cuts** and jump the scrubber to a candidate.
* **save** a config (validated first — an invalid config is not written) and **render**
  it, with progress from the real job and downloadable video / report / analysis /
  config when it finishes.

Security: loopback bind only (a non-loopback `--host` is refused), a per-launch CSRF
token required on every mutating request, `Host` must be a loopback name, bodies are
required to carry `Content-Length` and are capped at 64 KiB, explicit import paths are realpath-checked against allowed roots (playback and results use server-issued ids), renders are bounded argv subprocesses with no shell, and outputs
may not be written into the preserved baseline or original-showcase trees.

## What a run writes

```text
out/
  broadcast_annotated.mp4   the whole shot: source cadence, source audio, overlays in windows only
  broadcast_report.md       one row per scene, the playback contract, the verification result
  broadcast_analysis.json   per-scene body/targets/phases, the spec, the environment
  broadcast_provenance.json source + config hashes, frame bounds, operator inputs, cut decisions
  broadcast_config.json     the run's own config - pass it straight back to `run`
  keyframes/                scene-<id>_<phase>.png for every marked phase
  broadcast_contact_sheet.png
  broadcast_debug/          spec.json, scene_windows.json, scenes.json, environment.json, ffmpeg logs
```

## Verification, and what it proves

`verify_broadcast_render` checks, and records each check separately:

* output frame count (from `-count_frames`) equals the source's;
* **every** decoded presentation timestamp matches the source's — the count of compared
  frames, the largest difference, the first/last PTS on both sides, whether the output's
  timestamps are strictly increasing and the half-frame bound are all recorded, so a
  prefix cannot stand in for the whole list;
* **duration**: the output's video-stream (and container) duration against the source's
  and against `expected_frames / rational fps`, with a tolerance of half a frame (1 ms
  floor) — a mux rounds a duration header, a dropped or appended frame is a whole frame;
* output `r_frame_rate` and `avg_frame_rate` equal the source's **rational**
  (within 1e-6 relative), not a rounded float;
* codec `h264`, pix_fmt `yuv420p`;
* an audio stream is present with the same codec, sample rate and channels as the source,
  and its audio-packet md5 **matches the source's** (a bit-exact copy);
* `moov` precedes `mdat` (faststart).

Any failure is a failure of the run: the report says so and nothing is claimed about the
video. A clip whose cadence is not constant is refused outright, because a window cannot
be located reliably in a re-timed stream.

## What it does not do

* No automatic detection of a golfer, a ball or a club. Every marker is an operator
  placement or a seeded tracker following one, and it is labelled that way everywhere.
* No broadcast tracer graphic is ever used as a detection. Ball markers are only drawn
  where an operator confirmed a physically visible ball.
* No physical metric: no ball speed, clubhead speed, launch angle, carry, apex or spin.
* No claim about other footage.

## Troubleshooting

* **`ambiguous` body, no skeleton** — the ROI is too wide. The canonical identity
  challenge has to see one golfer: draw a tight box. On the baseline clips a whole-frame
  ROI gives `ambiguous` while a tight box gives `supported` on the same frames. Some
  clips are `ambiguous` even so (the canonical baseline records the same result for
  them); the run says so rather than inventing a pose.
* **`scene window crosses the accepted cut at frame N`** — split the scene at the cut.
* **`exceeds the broadcast cap`** — an operator anchor gap is too long (max 12 frames,
  and 6 within 12 frames of the contact mark). Add the anchors you actually verified; the
  tool will not fill the gap.
* **`already exists`** — a finished render is not overwritten by accident; pass
  `--force` or choose another `--out`.

## Withholding unreliable values

A golfer scene may set `metric_hide_frames: [[start, end], ...]` with inclusive, ordered, non-overlapping native clip-frame ranges inside its window. These are operator decisions to suppress body-angle **card values** during blur or occlusion; tempo and the confidence-filtered skeleton remain available. Raw measurements remain in the analysis, while gated frames and reasons are recorded separately. Card state resets across a gate so held values cannot leak into it.

`--no-render` runs measurement only and produces no annotated video. Rendering uses a small dark backing behind the brand/disclosure for legibility; this option is confined to the broadcast presentation.
