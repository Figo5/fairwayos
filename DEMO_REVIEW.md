# FairwayOS research demo — review record

Built 2026-09-10 by Claude Opus 5 (`claude-opus-5`) in Claude Code, supervised by
Gio, coordinated by Hermes. Local only. Nothing pushed, published or deployed.

`research_only = true` · `ground_truth = false` · `production_eligible = false`

---

## 1. The artifact

| | |
|---|---|
| Video | `/Users/giofiore/ghostcaddie-tour/out/fairwayos_demo_20260910/fairwayos_research_demo.mp4` |
| sha256 | `e0391cf89bd415a75a96cb4da41161fc90df6a213955eaedf4127a370393b5ba` |
| Contact sheet | `/Users/giofiore/ghostcaddie-tour/out/fairwayos_demo_20260910/demo_contact_sheet.jpg` |
| Provenance | `/Users/giofiore/ghostcaddie-tour/out/fairwayos_demo_20260910/demo_provenance.json` |
| Builder | `/Users/giofiore/ghostcaddie-tour/tools/demo/build_fairwayos_demo.py` |

Verified with ffprobe/ffmpeg, not assumed: 1920x1080, H.264, yuv420p, 15 fps,
601 frames, 40.067 s, full decode with `-v error` produced no output (clean).

## 2. Reproduce exactly

```bash
cd /Users/giofiore/ghostcaddie-tour
.venv/bin/python3 tools/demo/build_fairwayos_demo.py --out out/fairwayos_demo_20260910
ffmpeg -v error -i out/fairwayos_demo_20260910/fairwayos_research_demo.mp4 -f null -
ffprobe -v error -show_entries stream=width,height,r_frame_rate,nb_frames,pix_fmt \
  -show_entries format=duration -of default=nw=1 \
  out/fairwayos_demo_20260910/fairwayos_research_demo.mp4
open out/fairwayos_demo_20260910/fairwayos_research_demo.mp4
```

The builder reads only committed local artifacts and the local source clip. It
makes no network calls and downloads nothing.

## 3. Source and rights

| | |
|---|---|
| Source | Pexels video `6573485`, https://www.pexels.com/video/boy-hitting-a-golf-ball-6573485/ |
| Local path | `/Users/giofiore/ghostcaddie-tour/out/research_training_gauntlet/pexels_6573485/source.mp4` |
| sha256 | `a6e48474045365d1de2d4af76f65da558531684d67da87172cdd15a6dc45e1d6` (matches STATUS.md) |
| Media | 1920x1080 H.264, 30 fps, 310 frames, 10.347 s |
| Rights status | Source page marked free to use. Local copy only. This is NOT a grant of ground truth or production eligibility. |

### Footage deliberately excluded

`out/youtube_fullswing/f3kTTMZlxds.mp4` produces a visually stronger result
(ball 17 / clubhead 47 detected across 89 sampled frames) and was NOT used.
`docs/legally-reproducible-golf-perception-options.md:43` states that public
PGA/YouTube footage is stress-test material only, and no rights record exists for
this clip. Using it would have made the demo look better and the evidence worse.
It remains a local-only research record.

## 4. What is in the video

Every number on screen is read from the artifact's `diagnostics.json` at render
time. Nothing is interpolated, smoothed, inferred or hand-placed.

- **Title card** — source, hash, rights, research flags.
- **Part 1 (121 frames, uncut)** — original source frame on the left, the
  accepted artifact's overlay on the right, plus per-frame state: source frame
  index, timestamp, pose/ball/clubhead/impact states with rejection reasons.
- **Part 2 (121 frames, uncut)** — the same clip against the retained rejected
  ball track, labelled WITHDRAWN, showing the false marker, tracer and zoom
  inset that a normal demo would have led with.
- **Closing card** — what works, what fails, what is never claimed.

## 5. Results shown (from the artifacts, not from this build)

| Layer | Result | Source |
|---|---|---|
| Pose / body | observed 121/121 sampled frames | `fairwayos_unified_pexels_6573485` |
| Golf ball | **unavailable** 121/121 (`no_valid_candidate`) | same |
| Clubhead | **unavailable** 121/121 | same |
| Clubhead candidates | 120 candidate points, **all rejected** (`insufficient_temporal_support;low_confidence`) | same |
| Rejected ball run | ball rendered on 117/121 with a 40-point tracer — **false positive** | `..._pre_ball_plausibility` |

## 6. Acceptance-scope verification (coordinator-requested)

STATUS.md names `fairwayos_unified_pexels_6573485_pre_ball_plausibility/` as the
"Accepted verification output", but the properties STATUS.md describes for the
accepted artifact — "ball marker, tracer, and zoom inset are absent", ball
unavailable on all 121 frames — do not match that directory:

| Directory | ball states | `ball.rendered_overlay` | annotated_video.mp4 sha256 |
|---|---|---|---|
| `fairwayos_unified_pexels_6573485` | unavailable 121/121 | `null` x121 | `eb01781a9a8b3676…` |
| `…_pre_ball_plausibility` | observed 117, predicted 4 | marker+tracer+zoom x121 | `9bc31df7b25a89e4…` |

**The described acceptance matches `fairwayos_unified_pexels_6573485`; the
directory pointer in STATUS.md is stale.** This was NOT silently edited. The demo
uses `fairwayos_unified_pexels_6573485` for current capability and treats
`…_pre_ball_plausibility` as retained rejected evidence, which is the role
STATUS.md itself assigns it. **Hermes should resolve the STATUS.md pointer.**

## 7. Caveats — all disclosed on screen

1. **No ground truth.** No labels exist for this clip. Accuracy, precision,
   recall and false-positive rate are **unmeasured, not good**.
2. **Pose is not golf perception.** A generic COCO pose model locating a person
   carries no golf-specific meaning and is not clubhead, ball or swing evidence.
3. **Clubhead identity does not exist anywhere in this demo.** The 120 candidate
   points are rejected candidates and are rendered as rejected. No frame asserts
   clubhead identity, contact or impact.
4. **Impact is never shown.** The artifact's SwingNet `impact_bracket` is
   `candidate_bracket_only` (frames 180–184) with "exact contact unavailable".
5. **Timing.** Source 30 fps; every 2nd source frame sampled (indices 0,2,…,240)
   → 121 samples rendered at 15 fps. Encoded fps is not proof of capture fps.
6. **Duplicate frames: NOT ASSESSED.** The ai-demo generator that produced these
   artifacts records no duplicate detection. This is stated as "not assessed" on
   screen — it is not a claim that duplicates are absent. (The separate
   `pga_analyze` CLI does detect and exclude duplicates; that is a different
   generator and a different clip.)
7. **Blur, camera motion, occlusion: NOT ASSESSED** (per-frame warnings).
8. **No selective editing.** Both segments are the complete 121-frame runs, in
   order, uncut. The failure segment is included specifically so the demo cannot
   be read as a success reel.
9. **No production claims.** No trajectory, landing, calibration, course
   coordinates, ShotEvent, analytics, recommendation, speed or spin figure.

## 8. Recommended next bounded gate

**Do not** re-run or retune falsified clubhead hypotheses; they stay falsified.

Recommended, in order:

1. **Resolve the STATUS.md acceptance pointer** (section 6). Documentation-only,
   no re-run — but it currently mislabels which artifact is accepted.
2. **Extend the observed-provenance contract** to the other local generators that
   still emit hardcoded route strings (15 artifacts listed in
   `.hermes/fairwayos-loop.md`), each with a RED test first.
3. **Rights-cleared evaluation footage** is the real blocker for anything past
   pose. Everything downstream of ball localisation stays unavailable until
   rights-cleared paired annotations exist. This needs a human decision on
   licensing and is a **stop point**, not something to engineer around.

