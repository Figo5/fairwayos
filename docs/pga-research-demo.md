# PGA research demo

`pga-research-demo` is the bounded, local-file research mode. It accepts exactly one
`--video` or `--url` source and supports `--out`, `--max-duration`, `--sample-fps`,
and `--max-frames` bounds (plus optional local model hints). It never calls the
validated shot pipeline or emits events/analytics.

Outputs are research-only: rendered H.264 evidence and a JSON report/diagnostics
with pose/body, ball, and clubhead candidate layers. Each layer is explicitly
`observed`, `unavailable`, `rejected`, or `ambiguous` as applicable; warnings,
confidence, and source/model provenance remain visible. Blocked runs write
`diagnostics.json` with `analytics: null`, `shot_event: null`, and closed
`research_only`, `ground_truth`, and `production_eligible` flags.

Example:

```sh
ghostcaddie pga-research-demo --video ./swing.mp4 --out ./out \
  --max-duration 8 --sample-fps 4 --max-frames 32
```

Media and generated artifacts are local outputs and must not be committed.
