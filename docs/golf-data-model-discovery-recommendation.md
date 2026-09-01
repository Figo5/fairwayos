# FairwayOS golf data/model discovery recommendation

Date: 2026-08-28

Status: **blocked at legal/reproducibility checkpoint; no implementation authorized**.

Five parallel discovery roles were run: Dataset Scout, License and Provenance Auditor, Model Scout, Benchmark Specialist, and Engineering Specialist. Their structured records were reconciled against the existing FairwayOS contracts and gates.

## Coordinator decision

No reviewed candidate satisfies all of the following simultaneously:

1. downloadable video assets with rights adequate for the intended use;
2. paired frame-level labels for golfer/anchor, ball, clubhead, phase, impact, and landing or trajectory where claimed;
3. explicit subject/privacy permissions;
4. immutable version and artifact hashes;
5. leakage-safe video- or golfer-disjoint splits;
6. checkpoint terms and reproducible local runtime;
7. deterministic held-out evaluation compatible with `automatic-perception.v1`.

Accordingly, no candidate is selected for acquisition, adapter implementation, or training. The single strongest actionable path is a **FairwayOS-owned consented dataset** collected under explicit video, likeness, annotation, and model-use permissions. This is a design recommendation, not a claim that such a dataset currently exists.

## Candidate dispositions

| Candidate | Evidence-supported scope | Disposition | Blocking gaps |
|---|---|---|---|
| GolfDB + SwingNet | Eight-event golf-swing sequencing research; official repository describes external YouTube source videos and a research checkpoint.[16] | Research-only; blocked for production | Source-video rights, separately stated checkpoint terms, complete immutable paired release, and ball/clubhead/landing benchmark package are not established. |
| GolfPose | Golf-swing posture/pose research with project-listed checkpoints and author-controlled dataset access.[3] | Conditional research lead; not approved | Author authorization, raw-data restrictions, source provenance, exact release manifest, hashes, and documented MPS/CPU runtime remain unresolved. |
| ClubheadDB | Clubhead tracking research package; PyPI/repository materials describe a dataset preparation tool.[19] | Blocked before source-video acquisition | Third-party YouTube/Reddit rights, subject permissions, source-video hashes, and an approved checkpoint are not established. No source videos should be acquired. |
| CaddieSet | Research data involving human joint features and ball information.[14] | Research-only/unclear | Repository license scope is not separated for media, annotations, and ball-flight records; no approved checkpoint or complete reproducible video benchmark was verified. |
| MultiSenseGolf | Multimodal wearable/sensor and pose-oriented research data with downloadable archive entries.[17][18] | Auxiliary research only; not a golf-video perception benchmark | Does not provide the required paired visual ball/clubhead/impact/landing package or FairwayOS-compatible held-out video benchmark. |
| UCF Sports Action Dataset — Golf Swing subset | 18 golf-swing RGB sequences with per-frame person boxes and action-localization guidance.[22] | Rejected | Stock-footage source rights, ball/clubhead/phase/impact/landing labels, and golfer-disjoint evaluation are absent. |
| Efficient Golf Ball Detection and Tracking Dataset | Image and image-sequence golf-ball boxes; the official repository links an archive.[23][24] | Rejected | No confirmed released videos, no paired golfer/clubhead/event labels, unclear source-media rights, and no auditable video-level split. |
| Robin-Hood-zjw/golf_swing | GolfDB-derived metadata and one demo MP4.[25] | Rejected; duplicate/reference only | No independent corpus or annotation ontology; underlying YouTube rights and split provenance are unresolved. |
| Zenodo Golf Dataset (Haluza, 2022) | One CC BY 4.0 SPSS tabular file, not video.[26] | Rejected; not vision data | No video or computer-vision annotations. |
| GolfBall_Video_Tracking | Repository-described golf-ball detection experiment.[20][12] | Rejected pending full review | Exact released annotation surface, source-media rights, immutable dataset/checkpoint hashes, and reproducible evaluator are not established. |
| Hosted Roboflow golf-ball/club projects | Hosted object-detection project pages may expose golf-related classes.[21] | Rejected for current milestone | Asset-level source rights, local checkpoint terms, exact export hash, paired video annotations, and leakage-safe evaluation are not established. Hosted inference is not used. |
| Generic YOLO/RTMPose/ByteTrack | Generic person/pose/tracking baselines | Research baseline only | Generic `person`, `pose`, or COCO `sports ball` output cannot become golfer, golf-ball, clubhead, impact, trajectory, landing, or production evidence. |
| FairwayOS-owned consented data | Future production path for all required modalities | Recommended next path when rights are available | Requires collection, annotation, review, hashes, and release/evaluation governance before any training. |

The Creative Commons BY-NC terms reviewed for GolfDB are not a production authorization and do not resolve third-party source-video rights.[2]

## Required next-stage design

### 1. Rights intake before annotation

Create a local-only source manifest for each already downloaded public clip and any future consented clip. Store:

- stable local clip ID;
- source URL and platform ID in the private manifest only;
- acquisition timestamp and bounded downloader configuration;
- source/license classification: `qualitative_stress_test`, `research_only`, `consented_internal`, or `cleared_for_intended_use`;
- source and subject-rights evidence;
- duration, dimensions, FPS, codec, and SHA-256;
- explicit prohibition on redistribution where applicable.

Public PGA/YouTube clips remain qualitative or research-only unless the exact video and subjects have matching rights and annotations. They cannot be ground truth by accessibility alone.

### 2. Existing HITL workspace flow

Use the current offline workspace and explicit human boundary:

1. Run `video-prepare` or the equivalent local preparation path for one clip.
2. Keep extracted frames, contact sheet, and relative frame references beside the blank workspace.
3. Open the generated offline HTML workspace locally; do not add network assets, cookies, cloud calls, or silent writes.
4. Select frames and label only what is visible: golfer box, stable track identity, body/feet anchor, ball, clubhead, phase, impact candidate/interval, and landing.
5. Use explicit `unavailable`, `occluded`, or `ambiguous` states instead of guessing.
6. Preserve annotator ID, annotation version, timestamp, confidence, warnings, excluded frames, and source rights classification.
7. Export only through explicit Save/Submit. Draft documents remain ineligible for analytics.
8. Keep human annotations in `video-human-annotations.v1`; do not rewrite them as automatic observations.

Calibration points may be collected as a separate human-reviewed artifact, but calibration does not become valid merely because four points were clicked. The source and engine coordinate systems, dimensions, point order, and mapping quality must be reviewed independently.

### 3. Annotation schema design gate

Before labeling begins, document and test a versioned annotation schema, preferably a new research-only contract such as `golf-research-annotations.v1`, containing:

- `video`: width, height, FPS, frame count, duration, local clip ID;
- frame-indexed `golfer`: track ID, bounding box, visibility, occlusion, anchor, confidence;
- frame-indexed `ball`: point or box, visibility, occlusion, confidence, explicit unavailable state;
- frame-indexed `clubhead`: point or box, visibility, occlusion, confidence, explicit unavailable state;
- frame-indexed `phase`: canonical FairwayOS phase or `unknown`, confidence, ambiguity;
- `impact`: candidate frame or bounded interval, ambiguity flag, confidence, evidence notes;
- `landing`: point/region only when visibly supported, otherwise unavailable;
- optional calibration artifact with exact four-point pairs and separate review status;
- provenance for every field: `human_ground_truth`, `source_annotation`, `human_confirmed`, `model_prediction`, `pseudo_label`, or `unavailable`;
- annotator, reviewer, schema version, rights classification, and exclusions.

Predictions may be imported only as `model_prediction` or `pseudo_label`; they must never be silently promoted to `human_ground_truth`.

### 4. Annotation protocol and quality control

Use two-pass annotation for clips intended for evaluation:

- Pass A: primary annotator labels all visible fields and marks uncertainty.
- Pass B: independent reviewer labels without seeing Pass A, then adjudicates disagreements.
- Record inter-annotator disagreement separately from model error.
- Define object-specific visibility/occlusion rules before labels are created.
- For impact and landing, permit intervals/regions and `ambiguous` rather than false frame precision.
- Do not require ball or clubhead labels in frames where they are not resolvable; record explicit missingness.
- Freeze the annotation release and hash before model tuning.

### 5. Split strategy

Create the split manifest before any training:

- split by golfer, source video, and event sequence, never by adjacent frames;
- reserve a held-out set that is not used for threshold selection, pseudo-label generation, or checkpoint selection;
- keep broadcast/source identity and duplicate/re-encoded derivatives in the same partition;
- maintain a separate public-footage stress set that cannot affect training or validation metrics;
- record split membership, clip hash, golfer hash/ID policy, and exclusion reasons;
- fail validation if any clip, source sequence, or near-duplicate crosses partitions.

Minimum partitions are `train`, `validation`, and `held_out`; a second independent rights-cleared test set is preferred before any production discussion.

### 6. Training boundary

Training may begin only after all of the following are true:

- annotation schema is documented and validator tests pass;
- source/license classification is approved for the intended research use;
- annotation and media hashes are frozen;
- split manifest is frozen and leakage checks pass;
- the model license and checkpoint terms are recorded;
- the isolated AI environment is pinned separately from the Python 3.9 standard-library analytics environment;
- the experiment configuration specifies seed, input resolution, augmentations, class ontology, sampling, hardware, memory/disk/runtime bounds, and output hash.

Training targets must come only from explicitly labeled `human_ground_truth` or an explicitly documented `source_annotation` class. Pseudo-labels can be used only in a separately named ablation and cannot be used as evaluation truth.

### 7. Evaluation gates

Evaluate separately for:

- golfer detection and anchor error;
- ball detection, visibility-aware precision/recall, and track continuity;
- clubhead detection and continuity;
- phase classification and temporal tolerance;
- impact frame/interval error;
- landing error only where landing is visibly labeled;
- calibration reprojection error only for reviewed calibration artifacts;
- runtime, memory, failure rate, blur/occlusion/cut strata, and confidence calibration.

Report human ground truth, source annotation, pseudo-label, and model prediction as different evidence classes. No metric is valid when training and evaluation share a clip, golfer, near-duplicate, or generated pseudo-label source.

### 8. FairwayOS integration gate

Only a model/evaluator that passes the held-out gates may be wired through `GHOSTCADDIE_AUTO_DETECTOR`. The adapter must:

- emit validated `video-observations.v1`;
- preserve frame/timestamp ordering and pixel coordinates;
- retain field-level confidence, warnings, and automatic provenance;
- leave unsupported values null;
- never infer clubhead, impact, trajectory, landing, calibration, or recommendations from generic evidence;
- keep human annotations on the separate HITL path.

Calibration is applied exactly once. Only after all ball/clubhead/contact/landing/calibration gates pass may the existing reconstruction seam invoke unchanged `run_pipeline()` exactly once. Otherwise output remains partial or blocked, with no recommendation or automatic `ShotEvent`.

## Engineering disposition

No source-code implementation was started from this discovery pass. Future changes should be limited to a research adapter, schema validator, split checker, evaluator, and isolated training runner after a candidate clears the legal/reproducibility checkpoint. Existing CLI commands, human fallback, YouTube ingestion, CI, core analytics, calibration, wind, dispersion, hazards, and session behavior remain unchanged by this discovery pass; the current verified suite is 386 tests.

## Final recommendation

Do not download newly discovered unclear-license assets, do not use hosted inference, do not train on the public clips, and do not lower any production gate. Begin with a documented schema and rights review for local HITL annotation of the already downloaded clips as a research annotation exercise only. The first actionable production-oriented acquisition remains a consented FairwayOS-owned dataset with frame-level ball/clubhead/phase/impact/landing labels and a frozen golfer-disjoint held-out split.

## Sources

[2] https://creativecommons.org/licenses/by-nc/4.0/legalcode.en — CC BY-NC 4.0 legal code
[3] https://github.com/MingHanLee/GolfPose — GolfPose official repository
[12] https://github.com/SeanZ509/GolfBall_Video_Tracking — GolfBall Video Tracking repository
[14] https://raw.githubusercontent.com/damilab/CaddieSet/main/README.md — CaddieSet README
[16] https://raw.githubusercontent.com/wmcnally/golfdb/master/README.md
[17] https://doi.org/10.7910/DVN/LCCLLW
[18] https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/LCCLLW
[19] https://pypi.org/project/clubhead-db
[20] https://huggingface.co/notjulietxd/golf-ball-tracker
[21] https://universe.roboflow.com/golf-gnp/golf-ball-club-detection/dataset/1
[22] https://www.crcv.ucf.edu/data/UCF_Sports_Action.php
[23] https://github.com/rucv/golf_ball
[24] https://arxiv.org/abs/2012.09393
[25] https://github.com/Robin-Hood-zjw/golf_swing
[26] https://zenodo.org/records/6406122
