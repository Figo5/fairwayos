# Ball annotation guide (dataset v2)

How to label golf balls so the labels can be used as evidence about a tracker.
Written after the 2026-09-18 pass, which built the tooling and then declined to
publish machine-reviewed labels — the reason is in §5, and it is the most
important section here.

---

## 1. What a label is

One reviewed frame carries exactly one state:

| state | you must supply | forbidden |
|---|---|---|
| `VISIBLE` | `x`, `y` in native frame pixels; optional `radius` | — |
| `NOT_VISIBLE` | `reason` | any coordinate |
| `OCCLUDED` | `reason` or `explanation` | any coordinate |
| `AMBIGUOUS` | optional `explanation` and a candidate region | any defended coordinate |

A coordinate on a frame whose state is not `VISIBLE` is a fabricated position, and
the validator refuses it. So does the UI: `2`, `3` and `4` never store a
coordinate, even if the pointer was over the canvas.

## 2. Provenance: what you may call ground truth

| origin | meaning | may be called human ground truth? |
|---|---|---|
| `HUMAN_CONFIRMED` | a person read the pixels and confirmed the label; the reviewer must be named | **yes** |
| `AGENT_REVIEWED` | an AI agent read the pixels and recorded the label | **no** — weaker claim |
| `AGENT_PROPOSED` | a detector, carry-over or optical-flow guess | **no** — not a label at all |

Proposals live in a separate namespace. A pending proposal on a frame that also
carries a label is a validation error, and accepting a proposal does **not**
create a label: the reviewer writes the label explicitly. Rejected proposals are
kept, because a rejected guess is exactly the negative evidence a tracker needs.

The v1 corpus records `annotator: "agent-visual"`. That provenance is what it is
and is not upgraded here.

## 3. Running the annotator

```bash
.venv-video-ai/bin/python3 tools/ball_annotate_web.py --clip-id pexels_33511635
# then open http://127.0.0.1:8791/
```

The page shows the clip id, the source SHA-256, the frame index, the state legend
and a progress bar. Controls:

| action | control |
|---|---|
| centre on the ball | click the canvas |
| `VISIBLE` / `NOT_VISIBLE` / `OCCLUDED` / `AMBIGUOUS` | `1` `2` `3` `4` |
| step / jump | `←` `→`, `shift`+arrow for ten frames |
| undo | `u` |
| skip a frame on purpose | `space` |
| next / previous unlabelled frame | `n` / `p` |
| magnifier | move the pointer; the inset never changes label geometry |
| propose from the previous label | `g` (stored as `AGENT_PROPOSED`, never a label) |
| hard-negative tags | select in the list before labelling |

Everything is saved on each action, so closing the page loses nothing and
reopening the same clip resumes where you stopped.

**Zoom never changes geometry.** A click is converted to native frame pixels
before it reaches the store, and the magnifier is a separate view. Source
coordinates are the only coordinates the store accepts.

## 4. Labelling a sequence, not a frame

Label consecutive runs, so continuity, longest-correct-run and identity-switch
metrics stay computable. A run worth labelling covers, where the footage supports
it: pre-appearance frames, address, the ball at rest, early motion, launch and
early flight, temporary occlusion, true loss, and post-loss frames that still
contain distractors.

Negative temporal context is not filler. A tracker that never sees "the ball is
gone, and that bright thing is a shirt" cannot learn when to stop.

Tag what is actually in the frame with the hard-negative vocabulary — `body`,
`clothing`, `shoe`, `club`, `tee`, `flagstick`, `glare`, `background_ball`,
`unknown_bright_object` — and still label the ball honestly. A frame full of
distractors *and* a visible ball is a `VISIBLE` frame with tags.

## 5. Verify every label by rendering it back onto its own pixels

A label that has not been drawn back onto its own frame is not ground truth, it
is a number someone typed. `tools/ball_annotate.py verify` and
`tools/ball_dataset.py qa` both write sheets with the stored label marked on its
frame. Look at them. The v1 pass caught four mislabelled clips this way, one of
them a centre placed on the *tee* rather than the ball.

### Why this pass published no labels

The 2026-09-18 pass was run by an agent, and it declined to promote its own reads
to labels. The evidence, all reproducible from the sheets in
`out/ball_dataset_v2/sheets/` (gitignored):

- A 12-tile magnified sheet was described as a 4x4 grid, and a magnified crop
  that contains a ball was reported as containing none.
- On a control image with a crosshair *drawn by this pass* at a known position,
  the reader reported that no crosshair was visible — on all four controls.
- The same reader on synthetic images was correct and precise (crosshair inside
  the disc, and 200 px to the right, both right).

So the vision path itself works; at real ball scale on real footage it is not
good enough, in either direction. Coordinate reading is unreliable, and so is
accept/reject verification. Labelling anyway would have produced numbers that
look reviewed and are not, which is the failure this whole dataset exists to
avoid. The labels are therefore left to a reviewer at the UI, and the machine's
candidates are kept as pending proposals.

Practical consequence for the next pass: **budget human time for labelling, or
build a verification loop whose correctness you can demonstrate on controls
before trusting it.** A crosshair-in-the-crop control experiment like the one
above is cheap and should gate any automated reviewer.

## 6. Quality control before you trust a batch

```bash
python3 tools/ball_dataset.py validate --show-flags
python3 tools/ball_dataset.py review --clip-id pexels_33511635
python3 tools/ball_dataset.py stats --camera-motion
python3 tools/ball_dataset.py status
```

`validate` fails on: coordinates outside the frame, a missing state, `VISIBLE`
without coordinates, a non-visible state carrying coordinates, an implausible
radius, a frame beyond the clip length, duplicate frame keys, a source-hash
mismatch, split overlap by clip/hash/group, a consumed v1 clip in the v2 final
test, and a proposal sitting on an existing label.

`review` **flags** large frame-to-frame jumps, isolated one-frame visibility,
ambiguous clusters, labels near the match tolerance and pending proposals. It
never rewrites a label: a person decides.

## 7. Rules that are not negotiable

- Never assign coordinates to a non-visible frame.
- Never silently convert an `AMBIGUOUS` frame into a `VISIBLE` one.
- Never promote a proposal to a label without a reviewed coordinate.
- Never move a clip between splits because results look bad.
- Never read final-test labels during development; `--final-test` exists so that
  reaching them is a deliberate act.
- Never commit media, QA output, `out/` or a generated sheet.
