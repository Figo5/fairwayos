# Golf-Specific Ball and Clubhead Acquisition Plan

Status: **technically ready, data/model blocked**. No candidate currently satisfies all acquisition gates. This plan is intentionally short and does not authorize downloads or perception-pipeline changes.

## Candidate checklist

| Candidate | Intended evidence | Weights | Paired annotations/splits | License terms | Hash/install requirements | Decision |
|---|---|---|---|---|---|---|
| GolfDB + SwingNet | Research-only swing-event sequencing; not ball/clubhead detection | `swingnet_1800.pth.tar` and documented dependencies | Eight event labels are described, but a reproducible paired held-out video/split package was not verified locally | CC BY-NC 4.0; non-commercial research only | Record exact checkpoint SHA-256 and environment lockfile; PyTorch runtime; no production use | Research-only, blocked |
| GolfPose | Conditional golfer/club auxiliary detection and pose | Project-listed detector/pose checkpoints; acquire only after authorization | Project dataset includes images/annotations and published train/validation/test golfer counts, but access requires author authorization and source provenance must be confirmed | Custom EULA allows commercial derived models but prohibits raw-data redistribution; attribution and deletion obligations apply | Written authorization, exact release/hash manifest, isolated MMPose/MMDetection environment, documented dependency versions | Conditional lead, not approved |
| ClubheadDB | Clubhead bounding boxes only | No approved checkpoint located | Metadata and annotations are available, but source-video rights and reproducible source acquisition are unresolved | CC BY-NC 4.0 package declaration; third-party YouTube/Reddit source rights unresolved | Package/asset hashes exist from reconnaissance; no source-video hashes; do not acquire source videos | Blocked before source-video acquisition |
| GolfBall_Video_Tracking / Roboflow projects | Possible golf-ball image evidence | No FairwayOS-approved checkpoint | Reviewed material did not establish paired video annotations, frozen splits, source rights, or a reproducible evaluator | Repository/project labels are insufficient to establish deployment rights or source-media rights | Require written terms, exact model hash, dataset/export hash, deterministic local runtime | Rejected pending full rights/evaluation evidence |
| FairwayOS-owned data | Production path for ball, clubhead, impact, and landing | Train or commission a separately cleared model | Consented high-resolution clips; golfer-level frozen train/validation/test splits; frame-level ball/clubhead/contact/landing labels; calibration points | Explicit video, likeness, annotation, and model-use permissions | Hash every media/annotation/weight artifact; pin Python/CV runtime; record model card and evaluator | Preferred future path |

## Required intake package before any adapter or gate change

1. Written source and subject rights for the actual videos and annotations.
2. Exact downloadable artifact URLs with no cookies, credentials, DRM, proxy, or platform-protection bypass.
3. Media metadata and SHA-256 hashes for every video/annotation/checkpoint artifact.
4. Matching frame-level ball and clubhead labels, plus contact/impact and landing labels where claimed.
5. Frozen golfer-level train/validation/test splits and a deterministic evaluator.
6. Model card covering training data, class definitions, confidence calibration, known failures, and runtime limits.
7. Installation manifest for Python, PyTorch, OpenCV/MMDetection/MMPose or other dependencies.
8. Held-out metrics for ball precision/recall, clubhead precision/recall, temporal continuity, impact frame error, landing error, latency, memory, and failure cases.
9. Explicit separation between pixel evidence and calibrated course analytics.

## Current disposition

No candidate satisfies the complete package. FairwayOS is technically ready to consume a validated adapter, but the golf-specific data/model milestone is blocked. Keep ball, clubhead, contact, impact, trajectory, landing, calibration, automatic `ShotEvent`, and recommendation fields unavailable. Do not modify production gates or invoke `run_pipeline()` until the intake package passes review.
