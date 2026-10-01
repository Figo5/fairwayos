# Tiger target benchmark

The assisted showcase hides how often the native template tracker loses the ball or follows a leg instead of the clubhead. This local runner measures the existing tracker with one visible-target seed per target and no subsequent corrections, against sparse model-reviewed native-frame references. It does not change demo defaults or render video.

The current local dataset and report are at `/Users/giofiore/fairway-os-runs/tiger-tracking-cycle/`. The twelve source clips are bound to `/Users/giofiore/fairway-os-runs/tiger-showcase-cycle/final_batch_manifest.json`. Reference files are local artifacts, not shipped fixtures.

## Run

```sh
/Users/giofiore/ghostcaddie-tour/.venv-fairway/bin/python tools/tiger_target_benchmark.py \
  --dataset /Users/giofiore/fairway-os-runs/tiger-tracking-cycle/labels/tiger_targets.v1.json \
  --manifest /Users/giofiore/fairway-os-runs/tiger-showcase-cycle/final_batch_manifest.json \
  --variant baseline \
  --out /Users/giofiore/fairway-os-runs/tiger-tracking-cycle/results/baseline.predictions.json
```

Use `--variant lk_agreement` and a different output path for the same tracker with optical-flow agreement enabled. This changes only `lk_refine`; effective options are recorded per target. Use `--validate-only` for schema and source/hash validation without decoding. Each completed run writes predictions, a JSON report and a Markdown report. Runtime records runner, evaluator and actual tracker/config source hashes.

## Scoring contract

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
