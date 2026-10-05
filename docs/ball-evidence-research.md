# Seed-bound ball evidence research

`tools/tiger_ball_evidence.py` filters archived classic ball proposals against
current native pixels. It does not feed decisions back into the proposal tracker
or acquire positions that the tracker never proposed. Clubhead blocks are preserved.

The original seed supplies one immutable bright-component threshold and geometry
descriptor. Every later proposal needs one eligible compact component nearby,
with bounded area, dimension and elongation changes. Geometry measures occupied
unit-pixel support rather than just pixel centers. No morphology, coordinate
refinement, confidence estimate or physical identity claim is added. After a
refusal, two consecutive compatible current confirmations are needed; only the
second is emitted with `ball_evidence_template_reacquired`. Earlier gaps remain
empty. Structural cuts are terminal. The normal method is
`ball_evidence_template`; exported scores are null.

```sh
python tools/tiger_ball_evidence.py \
  --dataset /absolute/path/to/ai-reference.json \
  --baseline /absolute/path/to/selected.predictions.json \
  --baseline-report /absolute/path/to/selected.predictions.json.report.json \
  --out /absolute/path/to/new/compact.predictions.json
```

Use a Python environment with OpenCV and NumPy. Outputs must be fresh and outside
source paths. The baseline report must bind the exact proposal bytes. Output
predictions, raw evidence and reports pin source/reference/proposal/code hashes.
Only original seeds, windows, cuts, proposals and current pixels enter the policy;
later target coordinates and visibility labels enter evaluation only. Import the
result with the existing review UI's `--tracking-predictions` or showcase CLI's
`--saved-tracks` option. Body regions and phase marks remain operator assistance.

The October 5 development result is 108 correct / 0 incorrect / 25 missed ball
samples on 133 visible AI-reference samples, with 0 false positives on 5 labeled
negative samples. The previous selected result was 109 / 0 / 24 and 4 false
positives. The extra miss is the confirmation delay at hole-13 putt frame 53 after
missing frames 51–52; observation resumes at 54. Thirty-nine proposed ball frames
are withheld in total; unlabeled frames do not establish accuracy.

The first version scored 105 correct samples. The occupied-pixel geometry fix
recovered three samples after that first result had been inspected. Both versions
were evaluated on the same previously inspected development footage; these
references are not human-confirmed ground truth or an independent holdout.
Sparse and dense references differ for clubheads but contain the same ball labels;
agreement between their ball totals is not independent validation. No production
promotion, full-swing recovery or broadcast-flight/tracer tracking improvement is
established. Test on new, independently annotated footage before promotion.
