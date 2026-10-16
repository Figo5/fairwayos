# Four-object evaluation

`ghostcaddie.video.four_object_evaluation` evaluates ball, clubhead, target-person body and cup assertions against independent reference labels. It does not run detection or modify the automatic upload workflow.

Labels use schema `fairwayos-four-object-labels.v1`, bind to the exact source SHA-256 and native dimensions, and declare native frame indices and presentation timestamps. Every labelled frame declares all four objects as `visible`, `absent`, `ambiguous` or `unreviewed`. Visible and absent labels require `human_ground_truth` or `human_confirmed` provenance. AI reviews, model predictions and blank drafts cannot supply accuracy labels.

Visible point objects require a native center and a tolerance chosen before scoring. Visible bodies require the target person's box, named anchors and a tolerance. Body identity uses box IoU (default 0.5). Anchor distance, distance-only success and success conditional on target-person identity are reported separately. Missing bodies count as misses for their visible anchors.

Prediction inputs must explicitly assert `state: observed` and `provenance: current_pixels`. Held, interpolated, inferred, bare-coordinate and unprovenanced results receive no measurement credit. The caller must bind the prediction document to the same source and native timeline; original producer observations should be retained alongside any explicit adapter claims.

The scorer reports visible recall, precision, misses, false positives on absent frames, localization errors and wrong-object locks. Wrong locks require a predeclared confuser region; a large distance alone is a localization error. These two error buckets are mutually exclusive. Point and body-anchor errors include median, p90 and maximum over all available visible predictions, including bad matches. Empty denominators produce null metrics.

Source grouping rejects duplicate bytes or sibling source sessions crossing partitions. Consumed groups cannot become independent holdouts. A new clip ID does not establish session independence or absence from model pretraining.

The October 7 new-footage baseline preserves all 2,628 native frame timestamps as unreviewed label drafts. Four videos (814 frames) have model predictions; two are reserved from inference. All accuracy metrics remain withheld. Source-image AI diagnostics are kept separately as debugging evidence. They do not change labels or model weights. Real-world speed in mph remains unverified.

Validation: `python -m pytest tests/test_four_object_evaluation.py tests/test_ball_evaluation.py -q` (42 passed).

## PGA TOUR checkpoint baseline

The subsequent October 7 tournament batch contains ten clips / 3,814 native
frames from Scheffler, McIlroy and Fleetwood. All ten were inferred with zero
manual tracking markers and the unchanged v5 runtime. All reference drafts
remain unreviewed, so independent accuracy remains withheld. Source/player/round
groups are isolated; siblings from the same broadcast are correlated. Cup
assertions are zero and the Fleetwood lag-putt body track selects a caddie.
These AI source-image diagnoses remain separate from scoring labels. See the
[resume guide](resume-fairwayos.md).
