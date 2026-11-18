# Resume FairwayOS — October 7, 2026

Active development is paused at the owner's request. Resume from `main` and the `checkpoint-2026-10-07` GitHub release. The automatic upload flow takes a video without markers or per-video settings and exports a verified annotated MP4 plus video-motion analytics. Tracking remains partial and research-only.

## Start here

First fix target-golfer identity and cup detection. The newer PGA TOUR batch has no cup assertions and Fleetwood's hole 1 lag putt selects the red-bib caddie. Reproduce against the exact frozen inputs and add independent reference labels before comparing fixes. AI diagnostic crops are consumed development evidence, not accuracy labels.

Then improve tiny-ball and putter/blur clubhead proposals, inset/cropped subjects, camera resets and stroke boundaries. The hole 1 drive includes 20 frames of the following approach setup. Avoid pooling motion across strokes, people, views, evidence channels or search resolutions. Unsupported and invisible objects stay absent.

The [four-object evaluator](four-object-evaluation.md) enforces source hashes, native frame/PTS binding, reference provenance and source-group isolation. All newer tournament reference drafts remain unreviewed. Public model pretraining overlap is unknown. Physical mph needs verified measurement/calibration; displayed broadcast speeds remain separately attributed.

## Local recovery map

These paths describe the current development machine, not fresh-clone prerequisites:

| Archive | Location | Role |
|---|---|---|
| Masters automatic v5 | `~/fairway-os-runs/fairwayos-vision-coverage-v5-2026-10-07/` | 23 clips / 9,466 frames; source inference freeze, semantic receipts, media checks, compilation and rejected revisions |
| New PGA TOUR tests | `~/fairway-os-runs/fairwayos-pga-tour-testing-2026-10-07/` | 10 clips / 3,814 frames; lossless trims, unchanged v5 runtime, diagnostics and compilation |
| Earlier stock tests | `~/fairway-os-runs/fairwayos-independent-testing-2026-10-07/` | Six acquired / four inferred; light-fixture false ball, background-patch false cup, cropped-player misses; separate from the PGA request |
| GitHub checkpoint | `~/fairway-os-runs/fairwayos-github-checkpoint-2026-10-07/` | Public release staging and closeout checks |

Checkout: `~/ghostcaddie-tour-legacy/`. Configured worker: `~/ghostcaddie-tour/.venv-fairway/bin/python`. Local galleries use 8797 (Masters), 8800 (stock) and 8801 (new PGA TOUR); the automatic upload service used 8798. These are local processes, not publicly deployed products. Restart them if no longer running.

The public release preserves 33 existing individual exports, both compilations, sanitized observations/analytics, verification and a source-bound manifest. Original downloads, full local archives, model weights, credentials, provider transcripts and caches remain local. Redacting filesystem paths changes neither tracking values nor video bytes.

## Preserve the baseline

PGA TOUR `INPUT_RECEIPT.json`, `coordination/frozen-runtime/`, `BATCH_RESULT.json` and `FINAL_VERIFICATION.json` record the pre-run hashes and unchanged runtime. The Masters batch completed inference under its freeze, followed by a presentation-only observed-ball caption correction with unchanged observations and verified re-exports. Do not rewrite historical receipts after later code changes.

Generic YOLO11 ball/pose and the retained ClubheadDB research checkpoint supply local proposals. Current geometry, visibility, ambiguity and consecutive-observation gates restrict publication. Missing observations are not filled. Keep the clubhead weight's recorded research terms in view before commercial deployment.

Optional `FAIRWAYOS_BALL_VISION=claude-cli` is administrator configuration. Claude Opus 5.5 classifies bounded native crops with no coordinate output. Invalid/uncertain responses and missing providers abstain. The newer PGA TOUR batch made 26 candidate decisions and published 56 associated ball frames; neither is an accuracy score or API-call count. Decisions can vary across repeated requests.

Text-only reviews were completed with retained dispositions. They did not independently annotate the videos. Failed/timed-out reviews remain recorded separately.

## Before a production claim

- Freeze independent visibility/identity/location labels and a held-out source/session split.
- Measure visible recall, precision, wrong-object/person locks, localization, abstentions and runtime for all four objects.
- Exercise a fresh video-only upload, cancellation, retention, original/overlay switching and downloaded-output verification.
- Check native frames/PTS/audio and repeated cloud classification stability.
- Establish applicable footage/model terms and verified evidence for any physical metric.

Curated examples are demonstrations, not an evaluation partition. Do not reuse consumed development footage or AI judgments as independent holdout labels.
