# FairwayOS status — paused October 7, 2026

The automatic video-only workflow, four-object research tracking, native media export, analytics and independent-reference evaluation contracts are saved for resuming development. The owner requested this checkpoint on main and the best existing examples on GitHub. Active tracking improvement is paused.

The product accepts a clip without markers or per-video configuration. Local ball/pose and experimental clubhead proposals use current-image gates; optional Claude Opus 5.5 confirmation sends bounded native crops to the configured cloud provider. It supplies no coordinates. This is a research prototype with partial coverage and known wrong locks, not validated production tracking or a launch monitor.

## Preserved collections

| Collection | Clips | Native frames | Ball / clubhead / body / cup assertions |
|---|---:|---:|---|
| Masters v5 | 23 | 9,466 | 3,781 / 869 / 3,785 / 2,568 |
| New PGA TOUR events | 10 | 3,814 | 860 / 512 / 1,477 / 0 |

Counts are availability, not correctness. The Masters collection has 150 simultaneous four-layer frames. The newer PGA TOUR collection has no cup assertions, a caddie body lock in the lag putt, missing ball/head intervals and a brief next-stroke tail in one drive. Independent accuracy is withheld; AI diagnostics are not reference labels. Physical mph, carry, impact, tempo and 3D metrics remain unavailable/unverified. Separately sourced 142/131/111 mph figures are broadcaster-reported ball speeds.

The [checkpoint release](https://github.com/Figo5/fairwayos/releases/tag/checkpoint-2026-10-07) preserves all 33 annotated clips, both compilations, four standalone showcase examples, sanitized observations/analytics, media verification and source-bound manifests. MP4 bytes are copied unchanged. Original source downloads, complete private run archives, weights, credentials and provider transcripts remain local. All archive entries pass CRC checks and release files have SHA-256 receipts.

## Closeout validation

- Full local suite: **2,458 passed, 2 skipped, 448 subtests passed** in 366.86 seconds; two existing deprecation warnings.
- Source and test compilation passed; staged whitespace checks passed.
- All five existing CLI smoke scenarios passed with their expected artifacts.
- Upload/retention/UI checks: 56 passed. Import-isolation plus archived ball-evidence checks: 53 passed and six subtests.
- Existing collection receipts verify native individual frames/PTS/audio. The PGA TOUR compilation has all 3,814 frames, exact decoded equality to individual exports and maximum PTS delta 0.333 microseconds. The Masters compilation preserves 9,466 frames and fits/pads its picture to 1080p, with a recorded timeline tolerance of approximately two microseconds. Compilation audio is encoded once; compatible individual audio is copied bit-exactly.
- No new inference thresholds, model weights, tracking coordinates or detector improvements were introduced during closeout. Three legacy UI assertions now target `/review`; the demo import-isolation test restores the original module objects to avoid corrupting later mocks. Development requirements include PyAV and Pillow for the rendering contracts.

Local logs and release staging live in `~/fairway-os-runs/fairwayos-github-checkpoint-2026-10-07/`. GitHub Actions runs separately after main is pushed; its result is visible on the repository Actions page.

The initial clean-checkout CI run exposed private-media and development-environment
assumptions in retained tests. Final-test media checks now explicitly skip when
the undistributed files are absent; family isolation still runs against committed
metadata. Synthetic canonical upload tests select their running interpreter, and
the one-shot clubhead guard uses a recording model stand-in without installing or
loading optional weights. These are test portability changes, not tracking fixes.

## When development resumes

Start with cup detection and golfer identity, then independent native-frame labels, held-out source/session evaluation, ball/head proposals through blur/insets/cuts, automatic shot segmentation and measured/calibrated physical analytics. Preserve the original frozen inputs and rejected revisions. Read [the resume guide](docs/resume-fairwayos.md) for the archive map and acceptance criteria.

Earlier workflow status is preserved in [the October 5 record](docs/status-2026-10-05.md). Historical assisted and canonical entry points remain in [the archived guide](docs/legacy-workflows.md).
