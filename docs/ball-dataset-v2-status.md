# Ball dataset v2 — status

Status date: 2026-09-18. Branch `research/ball-dataset-v2`, base `cba1fdb`.
Plan: [`ball-dataset-v2-plan.md`](ball-dataset-v2-plan.md) (written before anything
was acquired). Annotation rules:
[`ball-annotation-guide.md`](ball-annotation-guide.md).

**FINAL TEST EVALUATED: NO.**

No tracker, baseline, threshold or diagnostic has been run against the v2 final
test, and no v2 label exists yet. Development tooling reads train and validation
by default and needs an explicit `--final-test` to reach anything else.

---

## 1. Sources

| | |
|---|---|
| clips | **56** rights-cleared stock clips (target was 40–60) |
| shoot families (`source_group`) | **26** |
| provider | Pexels |
| dataset id | `fairwayos-ball-v2-2026-09` |
| manifest | `data/ball_dataset_v2/sources.json` |
| acquisition plan | `data/ball_dataset_v2/acquisition_plan.json` |
| re-acquisition tool | `tools/ball_dataset_acquire.py` |
| media on disk | 1.20 GB, all 56 present and hash-verified |
| media committed | **no** (never) |

Total footage: 22,883 frames, 13.9 minutes.

## 2. Licensing

Every clip is Pexels stock video under the **Pexels License**: free to use,
attribution not required, modification allowed; not permitted to sell unaltered
copies, to imply endorsement, or to redistribute on another stock platform. The
licence text was recovered on 2026-09-18 through two independent routes that
agree — a Wayback snapshot
(`https://web.archive.org/web/20260915223912/https://www.pexels.com/license/`)
and a live server-side render of `https://www.pexels.com/license/` — because the
provider's own pages are Cloudflare-gated from this environment.

Each manifest entry records provider, source URL, source identifier, acquisition
date, licence, licence URL, licence-text reference, rights class, original
filename, byte count and SHA-256. A clip whose licence status was not explicit
was excluded rather than guessed; 12 harvested clips were dropped for **content**
reasons (tennis, baseball, soccer, cannabis, baseball equipment, mini-golf,
drone-only scenery) and their media, where already downloaded, was moved to
`out/ball_dataset_v2/media_excluded/` and kept out of the manifest.

## 3. Split (frozen before any tracker work)

`data/ball_dataset_v2/split.json`, `golf-ball-dataset-split.v1`, `status: frozen`,
`seed 20260918`.

| split | clips | groups | frames | footage | resolutions | fps |
|---|---|---|---|---|---|---|
| train | 34 | 11 | 12,313 | 7.7 min | 720p–4K, plus 1440x2560 | 23.98, 24, 25, 29.97, 30, 59.94 |
| validation | 11 | 8 | 5,880 | 3.4 min | 720x1280–4096x2160 | 23.98, 24, 25, 29.97, 30, 60 |
| final test | 11 | 7 | 4,690 | 2.8 min | 426x240–4096x2160 | 25, 29.97, 30 |

Assignment is deterministic: groups ordered by `sha256("20260918:<group>")`,
placed into the most under-filled split by clip count, with one hard rule — a
group shared with the consumed v1 corpus may not enter the final test. The seven
final-test groups (`pexels_1367`, `pexels_3554`, `pexels_4128`, `pexels_4784`,
`pexels_5200`, `pexels_6256`, `pexels_9361`) are disjoint from every v1 family.
The 18 clips belonging to v1 shoot families (6541, 6573, 3351, 3805) sit in train
(16) and validation (2) only.

Blind spot, stated rather than hidden: `pexels_2000000`-style family buckets are
derived from the numeric id prefix, which is deliberately conservative — it may
merge two distinct shoots into one group, which costs balance but never leaks a
golfer across splits. Where a family has one clip, group-disjointness is trivial.

## 4. Labels

| | |
|---|---|
| reviewed labelled frames | **0** |
| state counts | `VISIBLE 0`, `NOT_VISIBLE 0`, `OCCLUDED 0`, `AMBIGUOUS 0` |
| consecutive runs | 0 |
| hard-negative tag counts | none yet (tags are attached at label time) |
| machine candidates recorded | **96 pending proposals** across 8 clips |
| annotation files | `data/ball_dataset_v2/annotations/*.json` (8) |

The 96 proposals are `AGENT_PROPOSED` with `status: pending`. They are not
labels, they are counted separately everywhere, `validate` refuses a pending
proposal that sits on a label, and `review` flags every one of them for a human
to accept or reject. Rejected proposals are kept on purpose: a machine guess that
a reviewer refused is useful negative evidence.

**The label targets in the plan (≥1,500 reviewed frames, ≥500 visible, substantial
absent/occluded/ambiguous coverage) are NOT met.** This is the honest headline of
the pass, and the reason is in §5.

## 5. Known annotation limitations

The pass declined to publish its own machine-reviewed labels. The evidence, all
reproducible from `out/ball_dataset_v2/sheets/` (gitignored):

1. A 12-tile magnified sheet was read as a 4x4 grid, and a magnified crop that
   contains a ball was reported as containing none.
2. On four control images with a crosshair **drawn by this pass** at a known
   position, the reader reported that no crosshair was visible — on all four.
3. On synthetic images the same reader was correct and precise: crosshair inside
   a radius-90 disc, and 200 px to its right, both answered correctly with
   accurate geometry.

So the vision path works; at real ball scale on real footage it is inadequate in
both directions — absolute coordinate reading, and accept/reject verification.
Publishing those reads as labels would have produced numbers that look reviewed
and are not, which is precisely what this dataset exists to prevent. What this
pass did instead: built the reviewed-label pipeline, recorded the machine guesses
as proposals, and documented the control experiment that gates any automated
reviewer.

Consequences for the next pass:

- labelling needs **human** time at `tools/ball_annotate_web.py`, or a
  verification loop that first passes a synthetic control like the one in §5;
- with zero labels there is no camera-motion, ball-size or class-balance
  measurement to report: `stats --camera-motion` returns *"no decodable labelled
  clips in this split"*, which is the truth rather than a zero that looks like a
  measurement;
- the `hard_negative:` vocabulary is enforced by the schema and by tests, but no
  tag is populated until a label exists;
- the v1 corpus (10 clips, 325 agent-reviewed frames) remains the only labelled
  ball ground truth in the repository, and it stays development-consumed.

Rights and provenance limitations: page-level metadata (photographer, exact
title) is Cloudflare-gated, so titles are those recorded at search time and are
marked as such; the authoritative identity is the clip id plus SHA-256.

## 6. Media reproducibility

`python3 tools/ball_dataset.py status`:

```
TRAIN: 34/34 clips present
VALIDATION: 11/11 clips present
FINAL_TEST: 11/11 clips present
```

Media is local, gitignored under `out/ball_dataset_v2/media/`, and addressed by
SHA-256 only. `media_status` re-hashes every file and reports a mismatched file
as *missing with a different hash* — a differently hashed file is a different
clip, not a substitute. Another machine re-acquires the corpus with
`tools/ball_dataset_acquire.py`, which resumes, never overwrites and never adopts
unrecorded bytes.

## 7. Tooling delivered

| tool | purpose |
|---|---|
| `tools/ball_dataset.py` | `manifest`, `split`, `list`, `validate`, `stats`, `status`, `review`, `qa`, `montage`, `proposal-sheet`, `set`, `propose`, `resolve-proposal` |
| `tools/ball_annotate_web.py` | local annotation UI: identity + SHA on screen, step/jump, click-to-centre in source pixels, four states, undo, skip, progress, resume, magnifier that cannot change geometry |
| `tools/ball_dataset_acquire.py` | re-acquire rights-cleared media from the committed plan, verify bytes, probe, record provenance |
| `ghostcaddie/video/ball_dataset.py` | the contract: states, provenance, source binding, split integrity, leakage guards, statistics, media status, review flags |

`cv2`-dependent commands (`montage`, `proposal-sheet`, `qa`, `stats
--camera-motion`) and the UI must be run with an interpreter that has OpenCV
(`.venv-video-ai/bin/python3`); the manifest, split, validation, statistics and
status commands are standard-library only.

## 8. Validation result

```
python3 tools/ball_dataset.py validate --show-flags
OK: 56 clips, 8 annotated, 0 labelled frames, 96 review flags
```

All 96 flags are `pending_proposal` — machine guesses awaiting review, which is
the correct state for this pass. No validation error was raised: no
out-of-frame coordinate, no missing state, no coordinate on a non-visible frame,
no duplicate frame key, no source-hash mismatch, no split overlap by clip, hash
or group, and no consumed v1 clip in the final test.

## 9. Baseline diagnostics

**Not run.** The optional B1 diagnostics need labels to score, and there are
none. Running the old tracker on the v1 corpus would be exactly the retune the
pass forbids, and running anything on the v2 final test is prohibited until the
corpus has labels and an untouched final test means something. This is recorded
as skipped-with-reason, not as a pass.

## 10. What is left before model training

1. **Label the corpus** at `tools/ball_annotate_web.py`, oldest blocker first:
   train, then validation — final test stays untouched until a method exists that
   is already frozen.
2. **Populate the tag vocabulary** while labelling, so the hard-negative
   distribution is measurable rather than asserted (the previous failure was a
   body/clothing lock, and that is the class of frame the half-negatives must
   cover).
3. **Verify labels by rendering them back** onto their own pixels, and re-run
   `validate`/`review` on each batch; the v1 pass caught four mislabelled clips
   exactly this way.
4. **Only then** build evaluate-on-train/validation tooling and run the first
   honest baseline, with the final test still sealed.

The canonical ball layer remains `UNAVAILABLE`, and nothing in this dataset
changes that.
