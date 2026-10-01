# Tiger target benchmark

The assisted showcase hides how often the native template tracker loses the ball or follows a leg instead of the clubhead. This local runner measures the existing tracker with one visible-target seed per target and no subsequent corrections, against sparse model-reviewed native-frame references. It does not change demo defaults or render video.

The current local dataset and report are at `/Users/giofiore/fairway-os-runs/tiger-tracking-cycle/`. The twelve source clips are bound to `/Users/giofiore/fairway-os-runs/tiger-showcase-cycle/final_batch_manifest.json`. Reference files are local artifacts, not shipped fixtures.

The follow-up comparison is at `/Users/giofiore/fairway-os-runs/tiger-clubhead-cycle/REPORT.md`.
The classic worker now clips the search margin to valid image pixels when the full
target patch fits near an edge. Newly admitted edge matches also need central
contrast or central texture agreement. Score, full-template and maximum-jump guards
still apply. This fixes a deterministic search-window refusal; it does not solve
physical identity or enable automatic recovery in the classic demo.

On the frozen sparse v1 reference, evaluation clubhead agreement rose from 26/64
to 29/64 correct points, with wrong points unchanged at 22. Development results
and all twelve ball tracks stayed identical. Archived original predictions carry
their historical worker source hash; `baseline` always runs the current worker.
The foreground candidate regressed to 21/64 correct evaluation points and remains
an opt-in research CLI experiment. It is not enabled in demo defaults or the editor.
Dense diagnostic v2 labels preserve v1 and add thirty development labels; they
remain AI references, with no human ground-truth or holdout claim.

## Run

```sh
/Users/giofiore/ghostcaddie-tour/.venv-fairway/bin/python tools/tiger_target_benchmark.py \
  --dataset /Users/giofiore/fairway-os-runs/tiger-tracking-cycle/labels/tiger_targets.v1.json \
  --manifest /Users/giofiore/fairway-os-runs/tiger-showcase-cycle/final_batch_manifest.json \
  --variant baseline \
  --out /Users/giofiore/fairway-os-runs/tiger-tracking-cycle/results/baseline.predictions.json
```

Use `--variant lk_agreement` and a different output path for the same tracker with optical-flow agreement enabled. This changes only `lk_refine`; effective options are recorded per target. Use `--validate-only` for schema and source/hash validation without decoding. Each completed run writes predictions, a JSON report and a Markdown report. Runtime records runner, evaluator and actual tracker/config source hashes.

## The clubhead foreground candidate (`--variant clubhead_foreground`)

`--variant clubhead_foreground` leaves the BALL leg byte-identical to `baseline` (same `measure_target` + `resolve_track`, same seeds, same options) and replaces only the CLUBHEAD leg with the isolated experimental tracker in `ghostcaddie/demo/foreground_tracker.py`. That candidate:

- builds a target-centred FOREGROUND MASK from the seed patch alone (native seed frame + native seed coordinate), and matches masked zero-mean normalised cross-correlation instead of the whole 31 px context. A patch with no contrast, no plausible mask, or a seed too close to the frame edge is REFUSED -- the target is then reported missing for every frame rather than followed on weak evidence;
- searches a bounded window around the motion-extrapolated position, CLAMPING the window at a frame edge instead of rejecting it, and refuses a frame whose second-best competing peak is too close to the best (ambiguity abstains: no point, never a stale marker at the last position);
- keeps the frozen seed patch as an identity gate on every accepted point;
- owns its own point policy: `manual_anchor` (the seed only), `observed_template` with method `foreground_template`, and `missing` / `lost`. It does not run `demo.tracking.resolve_track`, so it does not inherit that module's frozen "no automatic point after a loss" rule;
- may RE-ACQUIRE a genuinely lost target inside a bounded window of ELAPSED frames (`reacq_window_frames` frames counted one per source frame spent lost, within `reacq_radius_px` of the LAST PHYSICAL position, at the higher `reacq_min_score`), and only **causally**: a frame is reported as re-acquired only when IT is the one completing `reacq_confirm_frames` consecutive measured confirmations. Earlier candidate frames stay lost with no coordinates and are never back-filled by a later frame that confirms, so the reported recovery latency includes the confirmation delay and the report is prefix-invariant (appending frames cannot rewrite what an earlier frame already said). Re-acquired points carry the method `foreground_template_reacquired` and the new segment begins at the confirmed frame, so no trail bridges the loss. Past the window nothing is reported at all;
- resets at a structural cut and never re-acquires across one, even in a frame where a search would have returned a candidate. (In the current twelve windows every declared cut frame lies OUTSIDE the window, so this policy is exercised by the synthetic tests only.)

The candidate's constants are module-level and fixed for every clip, hole and frame -- it never reads the reference's per-clip options. Its profile, the hash of that profile and the hash of the module are declared in `runtime.candidate_profile`, and per-frame raw metadata (what was searched for, what was measured, which gate failed, whether the appearance model was refreshed) can be written with `--raw-out`. The predictions' `options` block stays the classic tracker's options, because that is what the ball leg of the same run really used.

Synthetic fixtures exercise the candidate's behaviour (mask follow-through past a larger static neighbour, abstention on two indistinguishable candidates, refusal of flat/edge/uninformative seeds, clamped edge windows, bounded re-acquisition with consecutive confirmation, cut resets) and are not performance results for any real clip.

## Clubhead re-detection cycle

The next local cycle is saved at
`/Users/giofiore/fairway-os-runs/tiger-redetection-cycle`. All twelve original trims
and both frozen AI reference files remain unchanged. There are two additional,
explicitly optional research paths; neither changes the editor or classic default.

`--variant putter_redetect` measures fresh compact head regions with relative
contrast and thin-shaft support. It splits connected thick cores, measures each
reconstructed head independently of the shaft, and refuses ambiguous seeds or
candidates. Only declared putts use it; ball and full-swing head tracking remain
classic. It was rejected on these real clips because its recall was too low.

`tools/tiger_putter_sam2.py` uses an externally supplied, official SAM2.1 tiny
checkpoint and checkout. It accepts `--dataset`, `--baseline`, `--out`, `--run-dir`,
`--runtime-dir`, `--checkpoint`, `--device` and optional `--ids` for declared putts.
It downloads nothing. The runtime must provide SAM2, PyTorch, OpenCV, NumPy,
Hydra and iopath. The source video is decoded into a fixed native-pixel crop and
JPEG measurement copies; the input MP4 is never edited. Each head receives only
its original initial point and a box derived from that point, plus the window and
cut metadata. No later prompt, reference coordinate or correction reaches the
model. Every other target block is replayed from the validated classic baseline.

SAM2 produces a generic prompted-object mask, not a clubhead class. The independent
`segmented_head` helper splits thick mask cores and admits a nonround seed within
three pixels. Subsequent native measurements need one nonround, area-compatible region in
the bounded policy search. The reused point policy requires two current-frame
confirmations after loss, never backfills earlier frames, and ends at a camera cut.
Internal binary region admission indicators in the raw decisions are explicitly
**not confidence probabilities**; exported automatic head confidence is null. Stronger appearance/identity checks remain a
production blocker: later admission alone does not prove the region is a head.

`tools/tiger_seed_fallback.py` composes measured results using initial seed admission
only. It requires `--dataset`, `--baseline`, `--candidate`, `--out`; verifies raw
inference hashes and inputs; and retains the entire classic head block when the
initial SAM2 mask is refused. It records the actual engine for each clip. It never
selects a clip by later scores or switches back to classic during a lost track.
The pure candidate is preserved for comparison rather than hiding refusals.

`tools/tiger_replay_head_masks.py` accepts `--dataset`, `--baseline`,
`--mask-receipt`, `--out` to recompute the same head policy from checksum-verified
native masks. It uses the first forward mask at the original seed frame for
capability admission: the immediate prompt preview proved unstable across runs.
It does not read the previous head decisions or add prompts. Exported head
confidence is null, and inactive inherited probability thresholds are labelled
in the profile. Pure SAM2-or-abstain is the primary research route; the optional
classic fallback is a comparison that retains known classic ball drift.

Reports include checkpoint, SAM2 source revision, inference/helper/policy hashes,
effective geometry/policy settings, raw decisions and replay provenance. MPS is
used locally; deterministic behavior on other devices is not asserted. Both paths
remain research-only. See the cycle's `REPORT.md` for measured counts, refusals,
native-frame comparisons and the next improvement priority.

## Scoring contract (unchanged)

- The seed is the sole annotation passed into tracking. Each target has one forward leg. Reference labels are consulted only during scoring; seeds, corrections and interpolated points cannot count as automatic observations.
- Correctness means an automatic native-pixel point within the individual visible reference's tolerance. Misplaced automatic points and missing points are separate counts. Error summaries include all automatic predictions at scored visible labels, including misplaced ones; misses have no pixel error.
- Negative false positives are automatic markers on explicitly occluded/out-of-view labels. Uncertain frames and unlabelled frames are excluded. Labels before/at the seed, outside the configured window, or on a cut remain in the report in separate unscored buckets. Unseeded targets have no accuracy denominator.
- Recovery requires a post-seed negative-to-visible transition with consecutive visible labels. Latency is measured from the first visible frame, with zero meaning immediate recovery. Standard median averages the middle pair when the count is even. Sparse or absent recovery evidence yields null, never an assumed success.
- The reference contract explicitly pins `ground_truth=false`, `human_confirmed=false`, `research_only=true`, `production_eligible=false`, `label_type=ai_visual_reference`. These references support local diagnostics, not production promotion or a human-labelled accuracy claim.
- Four development clips and eight evaluation clips are bookkeeping groups of already inspected showcase footage from the same tournament/broadcast family. Neither group is a statistically unseen holdout. Report putting separately from full swings; address-ball agreement is not evidence of ball-flight tracking.

## Checks

```sh
/Users/giofiore/ghostcaddie-tour/.venv-fairway/bin/python -m pytest \
  tests/test_tiger_target_benchmark.py tests/test_demo_tracking.py tests/test_demo_config.py -q
```

Contracts refuse changed source hashes, missing manifest clips, mismatched seeds, automatic seed observations, invented later operator anchors and forbidden structural predictions. Synthetic fixtures verify scoring and refusal behavior; they are not tracking-performance results.
