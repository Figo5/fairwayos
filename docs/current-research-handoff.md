# FairwayOS current research handoff

Status date: 2026-09-11. This document is an index and safety boundary for local golf-perception research artifacts. It is not an acceptance report and does not make production claims.

## Hard research flags

All referenced perception records remain:

- `research_only=true`
- `pseudo_label=true`
- `ground_truth=false`
- `production_eligible=false`

Do not call production analytics, calibration, carry/apex/landing, recommendations, or ShotEvent logic from these records. Do not publish or upload derived PGA TOUR renders without express rights clearance.

## Baseline preserved

Baseline and historical repo commits are preserved on branch `pga-research-analyzer`. Existing local `out/` media, model weights, and generated artifacts were not moved into source control by this handoff. The committed PGA analyzer/demo history remains the baseline record; rejected experiments are retained as evidence, not rewritten into wins.

## Source-binding failure and durable fix

Completed work in `/tmp/fairway-tapir-source-repair` found a source-binding defect wider than first reported: Morikawa and Gotterup BootsTAPIR runs were named as separate clips but consumed the same Lipsky video bytes (`61e852fb...c1ef6`). All cross-clip/transfer claims from those runs are invalid and withdrawn.

Durable repo fix added here: `ghostcaddie/video/source_binding.py` plus `tests/test_source_binding.py`. It fails closed before model import unless media bytes match the caller's expected SHA-256 and, when provided, the seed JSON's own `source_sha256`. Clip names/player names are never accepted as byte provenance.

## Reusable repo code retained

- `ghostcaddie/video/team_layer_integration.py`: validates independent body/clubhead/ball layer handoffs, source SHA, research flags, finite bounded geometry, blocked unavailable-only layers, and renderer timeline clearing.
- `ghostcaddie/video/siwoo_layers.py`: local Si Woo helper contracts for independent layer states, cut resets, caddie/body selection risk, exact source-to-display mapping, visibility intervals, body continuation budget, and bbox clamping.
- `tools/demo/build_team_layer_demo.py`: local-only renderer that consumes strict team layer handoffs and writes a checkpoint instead of fabricating missing layers.
- `tools/demo/build_pga_approach_overlay.py`: preexisting local multilayer Si Woo approach overlay work was preserved for parent review rather than overwritten.

## Local work roots audited and classification

Do not delete these `/tmp` roots; they contain useful evidence and rejected history.

- `/tmp/fairway-single-source-build`: first-party single-source Tommy prototype. Useful design: source-bound timeline, independent object states, exact 30000/1001 timing, no missing-ball default, partial occlusion state, no centroid trails. Not copied wholesale because `build_single_source.py` is a local prototype tied to acquired local media and frozen worker paths. Its durable lesson is captured in repo layer contracts/tests and this handoff.
- `/tmp/fairway-parallel-acceptance`: first-party structural acceptance harness. Useful design: source hash checks, finite bounded geometry, seed-vs-predictive counts, demo eligibility defaults false, cross-identity collision detection. Not copied wholesale because it snapshots mutable `/tmp` roots and local artifact paths; selected semantics already overlap with `team_layer_integration.py` and `source_binding.py`.
- `/tmp/fairway-frame-hash-audit`: first-party canonical decoded-frame digest helper. Useful for audits, but not copied as production source because it hardcodes local source paths, a local venv, and OpenCV decoder policy. Keep as an offline audit prototype until generalized.
- `/tmp/fairway-tapir-source-repair`: first-party source-binding repair. Safe reusable subset was ported to `ghostcaddie/video/source_binding.py` with portable tests.
- `/tmp/fairway-learned/tracker`: learned point-tracker prototype, local weights, runs, sheets, and media-derived outputs. Do not copy weights, rendered media, seed files, or run outputs. Loader safety concern remains for body/Ultralytics usage where unrestricted pickle loading was observed in a separate loader trace.
- `/tmp/fairway-sam-head`: SAM2 clubhead worker with third-party SAM2 source checkout, checkpoint-derived outputs, masks, overlays, and local media. Do not vendor `sam2-src`, weights, masks, overlays, or media. The useful result is classified as partial/emerging SAM correction evidence only; it is not a production clubhead tracker.

## Explicit rejected/limited findings

- Appearance post-filters do not salvage the MIL clubhead path. HOG was decisively killed by a second false-lock object; HSV did not establish a positive generalizable case. Do not retune thresholds to revive this.
- Source adequacy remains a bound: clubhead-through-impact at current available frame rates is out of scope for the clips where required displacement exceeds the tracker design point.
- The corrected TAPIR reruns showed Lipsky genuine, Morikawa 1/51, Gotterup 0/71. The learned point tracker does not transfer under the frozen crop/model settings.
- The source-binding failure invalidates previous cross-clip/transfer claims; repaired reruns are development-consumed, not held-out proof.
- Body pose via Ultralytics has an unresolved unrestricted loader concern unless the effective safe-loading path is actually verified. Do not claim safe loading from wrapper intent alone.
- SAM2 head segmentation is partial/emerging correction evidence: f310-f343 full, f344-f360 partial/emerging after independent correction. Partial mask centroids are not head centers, material points, or trajectories.
- The latest single-source Tommy demo is an experimental local artifact, not accepted production: honest two-layer (body + clubhead) only, ball unavailable, no transfer/held-out claim, no safe body-loader claim.

## Reproduction and local asset requirements

Reproduction of media-dependent artifacts requires local ignored assets that are not committed:

- Official PGA TOUR and user-supplied local clips under `out/pga_official_acquisition/` or worker roots.
- Local model weights such as `yolo11n-pose.pt`, SAM2 checkpoints, and learned tracker checkpoints.
- Optional local CV/AI environment (`.venv-video-ai`) with OpenCV/ffmpeg/ultralytics/SAM dependencies as applicable.

Without those local assets, run only portable unit tests and compile checks. CI must not download live media, model weights, or scrape PGA TOUR.

## Evidence hashes to preserve

- Tommy source bytes: `2358a7c9b912b728f1d748a1f7e8aea6e5d7bc9219472ec4e724d503875a0a44`.
- Si Woo source bytes used by team-layer renderer: `cefbdf25400f5821893747e2b4a60ca5a11f990920ab3a9c8bba32a8c2d3deae`.
- Scheffler putt source bytes: `21c8bb54cd550dec2da680351cfd48f9cc61c24ca2ece43e07e30622b26e28f1`.
- Rejected/invalid Lipsky byte reuse in source-binding audit: `61e852fb...c1ef6` (see `/tmp/fairway-tapir-source-repair/source_audit.json` for full value).
- Tommy single-source local video from `/tmp/fairway-single-source-build`: `0356d3ba11456fca2cc2f2f486592feaa9303234e37f976d2dc8edeb7cfe002f`.
- Approved preexisting Si Woo team-layer demo hash recorded in loop checkpoint: `79ace086bb1d1db84aac2007a2c41afc53680ec36c60ad4032b10577f5aaa3bd`.

## Known blockers / next safe work

1. Generalize decoded-frame hash audit into a portable repo tool only after replacing hardcoded paths/venv assumptions with CLI arguments and synthetic tests.
2. If using Ultralytics body pose again, verify the effective safe-loader branch (`ULTRALYTICS_SAFE_LOAD=1` or equivalent) by execution before inference; do not retry unrestricted loading.
3. If integrating SAM masks, carry `observed_full`, `observed_partial_occluded`, and `observed_emerging` separately; never promote partial/emerging masks to full object centers.
4. Any new learned-tracker run must call `bind_source()` before model import/decoder use and must treat seed frame support separately from predictive frames.
