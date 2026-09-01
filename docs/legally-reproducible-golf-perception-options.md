# Legally Reproducible Golf-Perception Options

**Review date:** 2026-08-28  
**Scope:** research and release triage only; no media, weights, or annotations were downloaded.

## Decision summary

ClubheadDB is accepted as **`blocked_before_source_video_acquisition`**.[1] No ClubheadDB adapter will be built and no ClubheadDB source video will be downloaded. The local status record is `out/clubheaddb_recon/recon_status.json`; the directory is ignored and unpublished.

No candidate currently satisfies all of the following simultaneously: downloadable golf video, matching annotations, reproducible source provenance, and rights adequate for FairwayOS production use.[1][3][7] The automatic-perception gates therefore remain closed.

A license label alone is not treated as proof that a dataset distributor had authority to license embedded third-party video. Creative Commons states that licensors should secure necessary rights before applying a license, and that a license grants only rights the licensor has authority to grant.[2][6][15]

## Classification

### 1. Production-usable

**None at this review.**

A production candidate must have a stable downloadable release, source-level rights or a documented consent/release chain, matching annotations, reproducible hashes and splits, and a checkpoint/model license compatible with the intended FairwayOS deployment.[2][4][6] No reviewed golf-specific candidate met every requirement.

The eventual preferred path remains FairwayOS-owned, consented clips with explicit video and likeness permissions, frozen golfer-level splits, annotations, hashes, and a separately cleared model/checkpoint.[15] That path is not yet an acquired dataset.

### 2. Research-only

| Candidate | Evidence | FairwayOS disposition |
|---|---|---|
| **GolfDB + SwingNet** | The official repository declares CC BY-NC 4.0 and documents trimmed swing videos, eight swing-event labels, downloadable preprocessed clips, annotations, and a pretrained checkpoint.[1][2] | **Research-only and blocked for FairwayOS production.** Non-commercial terms are incompatible with production commercialization, and the required paired labeled validation/video materials were not reproducibly verified in the prior audit. No PCE or impact-accuracy claim is made. |
| **GolfPose** | The official project exposes golf-swing images/annotations and 2D/3D and detector checkpoints.[3] Its EULA permits internal research/model training and commercialization of derived models, while prohibiting raw redistribution; the dataset download requires author authorization by email.[4] | **Strongest conditional lead, not approved.** The EULA is commercially relevant, but email-gated access is not a self-service reproducible release. Before any acquisition, obtain written authorization, verify provenance/subject permissions, record the exact release and hashes, and confirm that the requested FairwayOS deployment is covered. |
| **ClubheadDB** | The package declares CC BY-NC 4.0 and publishes metadata/annotation resources, but source URLs point to third-party YouTube/Reddit media and source-level permissions were not established. | **Accepted blocked status.** Auxiliary clubhead evidence only if a future legal checkpoint clears it; never phase, impact, ball-flight, landing, calibration, analytics, or recommendation evidence. |
| **CaddieSet** | The public repository provides a MIT-licensed repository and a CSV of swing features/ball-flight fields.[13][14] The reviewed public release did not provide a reproducible paired video-and-annotation benchmark suitable for FairwayOS video evaluation. | **Research-only candidate for feature/ball-outcome study, not an automatic-video release.** No production or perception-gate use. |
| **GolfDB Kaggle mirror** | Kaggle metadata labels the mirror CC BY 4.0 and describes approximately 709 MB of videos, but says they are the original GolfDB videos.[7] This conflicts with the official GolfDB repository’s CC BY-NC declaration.[1][2] | **Not accepted as a rights-cleared alternative.** Do not rely on the mirror’s uploader-selected label to override upstream provenance. |

### 3. Qualitative stress-test only

| Candidate | Permitted use in FairwayOS |
|---|---|
| **Current best-effort YouTube workflow** | May accept an accessible clip and create pixel-space artifacts such as frames, tracks, detections, and explicit partial/blocked status. It must not treat the clip as ground truth, infer golf-specific events without matching labels, or produce recommendations from incomplete evidence. |
| **UCF11 “golf swinging” mirror** | The Kaggle description reports a golf-swing action class and broad action-video organization, but not golf phase, impact, ball, clubhead, calibration, or course annotations.[9] It may be used only for generic pipeline robustness if rights are independently cleared for the exact copy. It is not a golf-perception benchmark. |
| **Roboflow golf-swing projects** | The reviewed project pages declare CC BY 4.0 and expose image/object-detection projects, but do not establish a downloadable, rights-cleared video corpus with temporal golf labels.[10][11] They can at most support non-ground-truth image/pixel smoke tests after verifying the exact export terms. |
| **Looking to Learn** | The Figshare record is CC BY 4.0 and describes golf-swing intervention videos, but the downloadable record inspected here was only a small metadata package, not a reproducible annotated video benchmark.[5][6] It cannot validate automatic perception. |

Public PGA/YouTube footage remains stress-test material only.[1] It is never ground truth without matching annotations and source-level rights clearance.[2][6]

### 4. Rejected

| Candidate | Rejection reason |
|---|---|
| **GolfBall_Video_Tracking repository** | The repository describes a custom YOLOv5 golf-ball project and manually annotated frames, but the reviewed materials did not establish a clear dataset license, source-video rights, reproducible release, or benchmark split.[12] |
| **Kaggle Golf-Pose mirror** | The mirror declares CC BY-NC-SA 4.0 and describes a custom golf-pose dataset, but the reviewed metadata does not establish source provenance, matching temporal annotations, or a reproducible rights chain for the media.[8] |
| **Uncleared hosted/model APIs** | No licensed model API has been explicitly approved for FairwayOS. A hosted inference endpoint, API key requirement, or “try in browser” page is not by itself a grant of rights to use its training data, output, or service in FairwayOS. Roboflow pages expose API examples, but this review did not establish a FairwayOS-compatible commercial service agreement.[10][11] |

## Checkpoint/model licensing

- **GolfDB/SwingNet:** research/non-commercial declaration; not a production model path.[1][2]
- **GolfPose checkpoints:** the project lists detector, 2D, and 3D checkpoints, while the dataset EULA permits commercial derived models but prohibits raw-data redistribution.[3][4] This is the only reviewed golf-specific lead with an explicit commercial-derived-model clause, but it still requires authorization and provenance verification.
- **Roboflow-hosted models:** project pages show CC BY 4.0 dataset declarations and hosted inference examples, but no FairwayOS-approved API terms were established.[10][11]
- **Generic permissively licensed pose/detection models:** may be evaluated as engineering components, but they do not supply golf-specific ground truth and cannot open FairwayOS golf-event or analytics gates.[1][3]

## Required gate for the next milestone

Do not change automatic-perception gates until one candidate supplies all of:[1][2][4]

1. explicit rights for the actual video/frames, annotations, and any derived model use;
2. a stable, documented download URL that does not require bypassing authentication, DRM, cookies, proxies, or platform protections;
3. exact media metadata and SHA-256 hashes;
4. matching annotations for the claimed task;
5. frozen train/validation/test splits, preferably golfer-level;
6. a deterministic local evaluator with null/unknown handling;
7. checkpoint terms compatible with the intended deployment; and
8. evidence sufficient to separate pixel observations from golf analytics.

Until then, preserve the existing CI workflow, human fallback, production gates, and best-effort YouTube workflow unchanged. The current verified local suite is 382 tests with optional-dependency skips.[1]

## Sources

[1] https://github.com/wmcnally/golfdb — GolfDB official repository
[2] https://creativecommons.org/licenses/by-nc/4.0/legalcode.en — CC BY-NC 4.0 legal code
[3] https://github.com/MingHanLee/GolfPose — GolfPose official repository
[4] https://raw.githubusercontent.com/MingHanLee/GolfPose/main/LICENSE — GolfPose dataset EULA
[5] https://plos.figshare.com/articles/dataset/Looking_to_Learn_The_Effects_of_Visual_Guidance_on_Observational_Learning_of_the_Golf_Swing/3402664 — Looking to Learn dataset record
[6] https://creativecommons.org/licenses/by/4.0/legalcode.en — CC BY 4.0 legal code
[7] https://www.kaggle.com/api/v1/datasets/view/marcmarais/videos-160 — Kaggle GolfDB mirror metadata
[8] https://www.kaggle.com/api/v1/datasets/view/rakshitgirish/golf-pose — Kaggle Golf-Pose metadata
[9] https://www.kaggle.com/api/v1/datasets/view/pypiahmad/ucf-youtube-action-data-set — Kaggle UCF11 metadata
[10] https://universe.roboflow.com/lvs-rd/golf-swing — Roboflow Golf Swing project
[11] https://universe.roboflow.com/golfrobotjms/jms-golf-ball-tracking — Roboflow JMS golf ball project
[12] https://github.com/SeanZ509/GolfBall_Video_Tracking — GolfBall Video Tracking repository
[13] https://github.com/damilab/CaddieSet — CaddieSet repository
[14] https://raw.githubusercontent.com/damilab/CaddieSet/main/README.md — CaddieSet README
[15] https://creativecommons.org/publicdomain/zero/1.0/legalcode.en — CC0 legal code
