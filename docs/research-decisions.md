# Retained research decisions

Research is closed. These existing records explain why the finished canonical
workflow reports body-focused 2D analysis and withholds ball, clubhead and physical
metrics. They are historical evidence, not new validation or runnable product
instructions. References to local media, models and generated outputs require the
original private artifacts and do not establish clean-checkout reproducibility.

- [Canonical audit](canonical-video-pipeline-audit.md) and
  [validation sweep](canonical-validation-sweep.md): source binding, subject
  identity, unsupported metrics, and the body-only output boundary.
- [Ball tracking results](ball-tracking-results.md),
  [v2 results](ball-tracker-v2-results.md), and
  [v3 results](ball-tracker-v3-results.md): the recorded failure-to-generalize
  decisions. V3 was not promoted; its sealed final test was not run. Proposal or
  ranker measurements are not end-to-end tracking accuracy.
- [Clubhead probe](clubhead-tracking-local-research-probe.md) and
  [transfer gap matrix](golf-transfer-gap-matrix.md): why bounded or assisted
  examples did not establish automatic cross-clip tracking.
- [Annotation contract](ball-annotation-guide.md): interpretation of the retained
  text annotations and frozen splits in `data/`.

The corresponding `ghostcaddie/video/ball_*`, MIL, candidate, evaluation and related
research modules and tests remain as historical implementation evidence. The tested
pseudo-clubhead runner now lives in `tools/research/`, not `out/`. Do not restore
research routes as canonical fallbacks or treat demo inputs as evaluation labels.

Three unreferenced presentation/entry-point scripts (`build_fairwayos_demo.py`,
`render_rory_no_seed.py`, `video_to_results.py`) were superseded by canonical/demo
CLI workflows and removed. Experimental tracking implementations remain.

Superseded progress logs, handoffs, acquisition plans and old demo instructions
were removed from the current tree. Their complete versions remain in the
[pre-closeout tree](https://github.com/Figo5/fairwayos/tree/2640b3a4b3b2da579adbe38744d2452ba385a6ff).
The divergent standalone tracker restart is preserved by
`archive/local-tracker-restart-2026-09-22`; it is not part of the finished runtime.
