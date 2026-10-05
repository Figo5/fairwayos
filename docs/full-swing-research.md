# Current-frame full-swing recovery research

The opt-in `swing_context` policy measures component-adaptive cores from archived
forward SAM2 masks, the original operator seed, current native-size JPEG pixels,
and current measured MoveNet wrists. It generates no intermediate coordinates.
Classic automatic observations take priority; a confirmed mask observation can
fill an empty classic frame. Balls and putt heads remain unchanged. Independent
tracks do not feed selected positions back into either engine.

Each mask component has its own thickness threshold. Geometry measures occupied
unit-pixel support. The frozen seed appearance permits face darkening, but
brightening also requires small chroma change. Context requires local grayscale
contrast, a compact core and separation from current visible wrists. Missing
wrist evidence refuses measurement. Low-contrast original seeds disable the mask
leg without deleting the original operator point or the separate classic track.
Losses need two current confirmations; the first remains empty. Cuts are terminal.
Null exported scores are not confidence. The legacy `observed_template` status
means an automatic observation; the method identifies the actual mask measurement.

Body poses come from checksum-bound prior workflow analyses with operator body
ROI keyframes and interpolated crop geometry. Only current pose coordinates enter
the gate. The crop at a frame can depend on later operator ROI keyframes; temporal
causality is conditional on this fixed assisted crop schedule. No later measured
pose or target output is used. This is not an automatic subject detector.
The radius scales with crop height; all seven measured crops are full source height.
Any seed context refusal disables the mask leg, including absent wrists, proximity
to the grip, excessive elongation, or insufficient local contrast.
Pixels are decoded quality-100 JPEG at native geometry, not lossless source pixels.

```sh
python tools/tiger_replay_head_identity.py --policy swing_context \
  --dataset /absolute/path/to/ai-reference.json \
  --baseline /absolute/path/to/baseline.predictions.json \
  --baseline-report /absolute/path/to/baseline.predictions.json.report.json \
  --mask-receipt /absolute/path/to/MASK_RECEIPT.json \
  --frames-receipt /absolute/path/to/FRAME_RECEIPT.json \
  --body-receipt /absolute/path/to/BODY_RECEIPT.json \
  --out /absolute/path/to/new/measured.predictions.json

python tools/tiger_swing_route.py \
  --dataset /absolute/path/to/ai-reference.json \
  --baseline /absolute/path/to/baseline.predictions.json \
  --baseline-report /absolute/path/to/baseline.predictions.json.report.json \
  --measured /absolute/path/to/new/measured.predictions.json \
  --measured-report /absolute/path/to/new/measured.predictions.json.report.json \
  --out /absolute/path/to/new/selected.predictions.json
```

Outputs must be fresh. Reports bind source, reference, mask, frame, body, baseline,
raw evidence and executed policy bytes. Selection requires the exact current
context profile/code hashes and the same baseline. Validation/evaluation failures
write no candidate files; exclusive writes preserve earlier experiments. Multiple
output files are not an atomic filesystem transaction. All seven declared full
swings receive one policy; target references enter evaluation only. The normal
appearance runner and canonical detection defaults remain available unchanged.

The October 5 selected development result is 23 correct / 0 incorrect / 15 missed
full-swing head samples on 38 visible references, compared with 21 / 0 / 17. Across
all clips it is 64 / 0 / 28 on 92 sparse references, or 81 / 0 / 28 on 109 dense
references. Dense head negatives have 0 false positives on 14 samples. Ball
results remain 108 / 0 / 25 on 133 visible samples and 0 false positives on 5
negative samples. Five putt head blocks and all ball blocks are identical to the
previous selected predictions. Seventy-eight additional current head observations
were inspected visually; that qualitative AI check does not establish their accuracy.

V1 was rejected for a wrong labeled position. V2 recovered three labeled samples
but was rejected after visual review found unlabeled shaft/grip positions. V3
trades some coverage for context refusal and retains two labeled recoveries.
All revisions followed development inspection; none is an independent holdout or
human-confirmed ground truth. Prior experiments and media remain preserved.
Missing SAM masks, motion blur, stale-anchor recovery and transitions between
engines still limit coverage. Physical clubhead identity is not established.
Next work requires current-frame head redetection and fresh independently
annotated footage before production promotion; this is not a finished general
golf tracker or broadcast-flight/tracer detector.
