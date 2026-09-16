# Canonical workflow validation sweep

Corpus: `out/demo_improvement_campaign/development/pexels_round4` — 10 clips, all included.

**This is not a benchmark.** The corpus has no ground truth, no held-out split, and was already used during development. These numbers detect drift and show which clips the workflow declines; they support no claim about generalization to unseen footage.

`jitter px` is the median global camera translation between sampled frames, in native pixels, with the fraction of frame width in parentheses. `coverage` is the share of sampled frames carrying a pose attributed to the golfer.

| clip | res | frames | body | identity | 2nd person | coverage | jitter px (frac) | render | metrics | main rejection |
|---|---|---|---|---|---|---|---|---|---|---|
| `pexels_34883764.mp4` | 1080x1920 | 166 | **supported** | movenet_proposal_unchallenged | no | 0.9337 | 3.6 (0.0033) | ok | 6 | pose centre falls outside the identified subject region (4 frames) |
| `pexels_37030686.mp4` | 1080x1920 | 128 | **supported** | movenet_proposal_unchallenged | no | 0.9922 | 3.88 (0.0036) | ok | 6 | incomplete torso: sh_r below confidence (1 frames) |
| `pexels_37080178.mp4` | 1080x1920 | 92 | **supported** | movenet_proposal_unchallenged | no | 1.0 | 4.76 (0.0044) | ok | 6 | none |
| `pexels_37995798.mp4` | 4096x2160 | 84 | **supported** | movenet_proposal_unchallenged | no | 0.9762 | 13.03 (0.0032) | ok | 4 | incomplete torso: sh_r below confidence (1 frames) |
| `pexels_38053415.mp4` | 4096x2160 | 73 | **supported** | movenet_proposal_unchallenged | no | 0.9315 | 0.61 (0.0001) | ok | 6 | only 5 of 17 keypoints above 0.3 (3 frames) |
| `pexels_6541674.mp4` | 2560x1440 | 67 | **supported** | movenet_proposal_unchallenged | no | 1.0 | 5.68 (0.0022) | ok | 4 | none |
| `pexels_6541682.mp4` | 1440x2560 | 100 | **supported** | movenet_proposal_unchallenged | no | 0.99 | 3.22 (0.0022) | ok | 5 | incomplete torso: sh_r below confidence (1 frames) |
| `pexels_6541964.mp4` | 2560x1440 | 70 | **supported** | movenet_proposal_unchallenged | no | 1.0 | 6.23 (0.0024) | ok | 5 | none |
| `pexels_6573534.mp4` | 1920x1080 | 52 | **unavailable** | none | no | 0.0 | 44.92 (0.0234) | ok | 0 | a full-torso pose was proposed on only 0 of 52 sampled frames (0%); there is no body stable enough to attribute to the golfer |
| `pexels_6573621.mp4` | 720x1280 | 78 | **supported** | movenet_proposal_unchallenged | no | 1.0 | 2.21 (0.0031) | ok | 6 | none |

## Clips the workflow declined

- `pexels_6573534.mp4` — **unavailable**: a full-torso pose was proposed on only 0 of 52 sampled frames (0%); there is no body stable enough to attribute to the golfer (camera 44.92 px/frame)

## Highest camera motion

Included, not excluded. Dropping the clips that hurt a number is selection on the metric being reported.

- `pexels_6573534.mp4` — 44.92 px between sampled frames (0.0234 of frame width), body **unavailable**, coverage 0.0
- `pexels_37995798.mp4` — 13.03 px between sampled frames (0.0032 of frame width), body **supported**, coverage 0.9762

## Summary

- body states: 9× supported, 1× unavailable
- renderer produced an annotated video on 10/10 clips
- clips flagged as camera-jitter-dominated: 0
- ball and clubhead are UNAVAILABLE on every clip, by design.
