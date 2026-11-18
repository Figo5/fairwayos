# Detector proposals with current pixel and shaft support

The frozen learned detector can safely fill one missing full-swing observation in
the current twelve Tiger 2019 Masters development clips. It cannot replace the
selected tracker. The [local results report](~/fairway-os-runs/tiger-proposal-support-cycle/REPORT.md)
and [all sixteen clips](~/fairway-os-runs/tiger-proposal-support-cycle/CLIPS.md)
include the actual replay, independent AI visual reviews and selected exports.

`proposal_head.py` uses every eligible detector box only to restrict the search
for a measured native compact pixel core. It reuses the existing contrast,
current-wrist, Hough segment and thin shaft ribbon checks. The returned coordinate
is the exact support-pixel centroid; box centers, inferred shaft endpoints and
network confidence never become target positions. Multiple independent surviving
cores are withheld. Continuous box containment uses pixel indices plus 0.5;
exported centroids keep the original pixel-index convention. Operator body ROI
rectangles are exclusion context, not person silhouettes or identities.

`ProposalHeadPolicy` retains the original seed, two-current-frame confirmation,
gap reacquisition and terminal cuts. `UnseededConfirmation` permits isolated
research diagnostics without inventing a head seed or backfilling a track. These
gates do not establish semantic clubhead identity. The null observation score
records that limitation rather than presenting a confidence probability.

`tools/tiger_proposal_support.py --run <prepared-run-directory>` checks a frozen
plan, raw model outputs, pixel cache, extracted native body rows, source video
hashes and checkpoint. It snapshots executed dependencies before measurement,
reads no target reference labels and publishes results through atomic staging.
Gap selection retains all original observations, balls and putt heads; only
confirmed full-swing measurements can fill empty frames. Unknown assistance,
scored network centers, nonfinite coordinates and points after cuts are refused.
Every addition requires its own native video review before export. This command
requires prepared local research evidence, not merely an uploaded swing video.

The replay reused 1,078 frozen detector crop predictions across all twelve clips;
seven full-swing windows were remeasured and five putt tracks were preserved. The
standalone full-swing policy yielded just 19 automatic observations. Keeping the
existing selection and adding hole16 frame150 changes sparse head references
from 65 correct / 0 wrong / 27 missing to 66 / 0 / 26 out of 92, and dense
references from 82 / 0 / 27 to 83 / 0 / 26 out of 109. Ball scores remain 108 / 0 / 25
out of 133. These are consumed AI development references, not human-confirmed
accuracy, an independent holdout or all-frame accuracy. Only the changed hole16
video is rerendered; all prior files remain intact.

Actual frozen-model inference on 766 native PNG crops from four additional final
hole shots produced 22 unseeded confirmations on the tee clip and none on the
approach, pitch or putt. Claude Opus 5.5 identified two of those tee points as the
ball (frames 10/11), with another 14 head regions overlapping the ball. The entire
unseeded candidate is rejected for delivery; its raw points stay in the diagnostic
record. It misses the fast swing and uses unvalidated full-swing geometry for
pitch/putt diagnostics. The four selected exports remain body-only. Never turn
geometric refusals, unavailable pose or unlabelled frames into negative labels.

The measurement helper was implemented. Both sessions
reached iteration/provider errors; follow-up work corrected fixtures and hardened
contracts. Review covered code and all 23 proposed point areas.
301 focused tests and 99 subtests pass, including saved-export and UI checks.
The review interface now defaults to source-bound saved results when available
and preserves explicit export choices across target navigation. It waits for
the current lookup before enabling render, so a stale response cannot enable it.

The next data cycle needs frame-aligned head-versus-ball appearance labels,
broadcast background negatives and fast-head/short-stroke positives. The local
review queue contains actual native crops and raw boxes, with training labels
explicitly unpublished. Evaluate future models on fresh frozen source families;
do not retune the consumed author-box test. Repair side-on pose availability and
short-stroke geometry separately before integrating unseeded acquisition into
the existing review/export contract.
