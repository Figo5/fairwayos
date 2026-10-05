# Review saved tracking and export a showcase

Start `broadcast-showcase ui` with a config directory, an output directory outside
the repository, allowed clip folders via `--extra-root`, and optionally a
`--tracking-predictions` JSON file. Supply the local body runtime and MoveNet model
for pose overlays. The review server stays on loopback; analysis and media remain
local. The saved predictions schema is `fairwayos-target-predictions.v1`.

1. Choose a config and click **load**. Its source hash selects the matching clip.
2. Choose the target to review. Amber positions are automatic observations at the
   exact current source frame, within the stroke scene containing the saved seed.
   Seeds and reviewed assistance are never presented as automatic positions.
3. Use **previous gap** / **next gap** to inspect missing spans. Gaps may be
   occlusion or tracking loss; position coverage is not an accuracy measurement.
4. Choose **Reviewed showcase (assisted)** for the saved presentation with its
   operator assistance, or **Saved automatic targets (research)** to use only
   prior automatic observations and their original operator seed. Automatic mode
   excludes configured seed edits, corrections and assisted target paths. To
   change the measured tracking itself, run a new tracking experiment rather than
   editing old saved observations through this UI.
5. Click **render**. When verification succeeds, play the inline video or download
   the video, report, analysis and config. Each job uses an immutable config
   snapshot and a unique output folder, so later edits and renders cannot change
   its downloads.

Saved mode preserves recorded reacquisition after a gap, but never connects a
trail across that gap. Saved interpolation is omitted. A target can appear only
inside the scene containing its saved seed. This file format has one seed per
target per clip; later camera scenes require separate tracking results. Accepted
cuts cannot be crossed by a stroke scene. Source timing, complete frames and audio
are checked after encoding. A failed verification does not produce a successful
job result.

Results must bind to the exact registered source bytes. A changed prediction file
is refused until the review server restarts and loads it. Adding a clip does not
invent tracking results for it. No saved tracking file is loaded from an arbitrary
path supplied through the HTTP API.

The same import is available from the CLI:

```sh
.venv/bin/python -m ghostcaddie broadcast-showcase run /absolute/path/to/clip.mp4 \
  --config /absolute/path/to/config.broadcast.json \
  --out /absolute/path/to/export \
  --saved-tracks /absolute/path/to/results.predictions.json \
  --cv-python "$FAIRWAY_BODY_PYTHON" --body-model "$FAIRWAY_BODY_MODEL"
```

Omitting `--saved-tracks` keeps the existing assisted workflow. Saved targets are
experimental image observations, not general tracking accuracy or verified
physical identity. Source regions and phase marks remain operator assistance.
There is no calibrated ball speed, carry, launch angle or spin. Broadcast tracer
graphics in source footage remain source content; they are not tracking evidence.
