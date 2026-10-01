# DESIGN — broadcast review UI

Scoped design note for the local review surface
(`ghostcaddie/demo/broadcast_ui.py` serves it; the page itself is one string in
`ghostcaddie/demo/broadcast_ui_page.py`) →
`python -m ghostcaddie broadcast-showcase ui`.
Written to pin the decisions before styling, and kept as the record of what was
decided and why.

**Reading this as:** extending an existing brand (Fairway OS demo/showcase), a dense
*operator tool* for one person working through eight golf clips, the vibe is a
broadcast review bay — quiet, dark, footage-first, leaning on the existing Fairway OS
token set rather than a new system.

## Dials

| dial | value | why |
|---|---|---|
| DESIGN_VARIANCE | **4** | One signature move (below) and otherwise a conventional, legible tool. The operator is working, not browsing. |
| MOTION_INTENSITY | **2** | State changes only: hover, focus, the frame swap. No entrance choreography, no scroll effects. `prefers-reduced-motion` is honoured explicitly in CSS. |
| VISUAL_DENSITY | **6** | The right column is a working panel with commands, scene list and results. Footage keeps the majority of the viewport; the panel scrolls rather than compressing the picture. |

## Signature move

**The window is the interface.** The scrubber is not a progress bar: the current frame
is labelled with whether it is *inside* a scene window (overlays drawn) or outside one
(bare source frame), in the same words the renderer uses. That single line is the whole
product — the operator is deciding where the overlays start and stop, and the tool says
so on every frame instead of hiding it behind a timeline graphic.

Removed after naming it: any separate scene "timeline" widget, any thumbnail strip, any
badge row. One scrubber, one truth line.

## Tokens (reuse before invention)

The existing showcase chrome already carries the brand: teal `#70d4b0` / soft teal
`#3f7f6c`, amber `#ffb030`, near-black card `#16181a`, muted text `#9fb0b2`. The UI
reuses those values rather than introducing a second palette, so a screenshot of the
tool and a frame of the render read as one product.

* roles: `--ink` bg, `--panel` surface, `--panel-2` raised surface, `--line` border,
  `--text` fg, `--muted` muted, `--teal` accent, `--amber` accent-2 (seeds / focus),
  `--ok` / `--bad` states.
* spacing: an 8px base with a 4px half-step, used through one `--gap` variable — no
  arbitrary pixel values in component rules.
* radii: exactly two (3px controls, 4px panels). One flat surface treatment, no
  shadows; separation is by border and surface value.
* type: Arial/Helvetica (system, offline-available on this machine, and the same
  family the renderer already uses for chrome). Sizes: 15px brand, 14px body, 13px
  section titles/heading, 12px labels and secondary, 12px monospace-ish for paths.

## Copy rules

* Buttons name the action in the operator's words: "set start here", "apply fields to
  the selected scene", "add as a new scene", "scan this clip", "save config", "render".
  No "Submit", no "Go". The two scene actions are deliberately separate and separately
  labelled: one updates the scene you are looking at, the other creates a new one.
* Every state message says what happened *and* what to do: an invalid config names the
  field and is **not** written; a render failure prints the tail of the real log; a
  config whose clip is not in the library says so and says to import it, and loads
  nothing rather than the wrong footage.
* A missing value is shown as missing. There is no placeholder metric that looks like
  a measurement.

## Honesty in the controls

* The ball seed asks for confirmation — "Is the ball physically visible at this point
  in frame N?" — before it is recorded. The marker is only ever drawn where an operator
  said they could see the ball.
* The **correction** mode says the opposite in as many words: the point is a position
  the operator recorded, "no claim of a detection". Seeds claim a visible ball;
  corrections claim nothing.
* Phase buttons are generated from the selected stroke type, so a putt offers
  `backstroke / forward_stroke / contact` and never `top`.
* Cut candidates from the scan are shown with their confidence and jump the scrubber
  to the frame; they never edit a window.
* Nothing that is loaded is silently dropped: the page edits the operator's own JSON
  document, so `manual_segments`, `roi_keyframes`, `presentation`, `timing`, `cuts`,
  `overrides` and `notes` are still there after a save. The assistance-mark panel lists
  every mark with a *remove*, so what will be saved is visible before it is saved.

## Accessibility and verification

* Every control is a real `<button>`, `<select>` or `<input>`: reachable by Tab,
  operable by keyboard, with a visible `:focus-visible` ring in amber. Arrow keys step
  the frame when focus is not in a field.
* The canvas is a pointer surface with keyboard equivalents for every action it offers
  (window bounds are numeric inputs; the ROI is drag-only, and the note says so).
* `prefers-reduced-motion` disables transitions.

**Verification status:** checked three ways.

1. **The page's own script, run under node** against a DOM stub
   (`tests/fixtures/broadcast_ui_dom.js` + `broadcast_ui_behaviour.js`, driven by
   `tests/test_demo_broadcast_ui.py`). This exercises the shipped script for real —
   loading a config onto its own clip, a deep-cloned document that keeps
   `manual_segments` / `roi_keyframes` / `presentation` / `cuts` / `timing` / `overrides`
   intact through a save, edits landing on the selected scene only, and add/remove of
   corrections, ROI keyframes and manual anchors.
2. **Real HTTP against a running server** — the token/Host/body/confinement guards,
   frame decode by index, the config list's clip resolution against the twelve actual
   configs, a nested-clip import from `--extra-root`, the save round-trip, and the
   clip-mismatch refusal on both `validate` and `render`.
3. **A code-level review** of the markup/CSS/JS.

Still owed: **the rendered page has not been eyeballed in a browser from this worker**
(there is no browser or screenshot tool in this environment). The 1920px and
320/375px visual check is primary's, who owns the real browser session.
