# Public Golf-Perception Dataset Landscape

Date: 2026-08-28

Status: **research and licensing reconnaissance only; no dataset, video, annotation, or checkpoint was downloaded by this report.**

## Scope and acceptance rules

This review looks for a reproducible path containing golf-swing video, temporal event labels, ball or clubhead annotations where available, explicit terms, source URLs, hashes, and fixed train/validation/test splits. A repository license is not assumed to grant rights to third-party videos hosted on YouTube, Reddit, Google Drive, or other services. Candidates with unclear asset-level rights are marked **blocked**.

The report does not alter FairwayOS production gates, analytics, human fallback, or SwingNet status.

## Comparison

| Candidate | License / legal status | Video and resolution/FPS | Labels and objects | Splits / URLs / hashes | Checkpoint | FairwayOS suitability |
|---|---|---|---|---|---|---|
| **Official GolfDB + SwingNet** | Repository README states CC BY-NC 4.0 for the code/repository, but the dataset was assembled from YouTube and the checkpoint has no separately verified asset license. The repository terms therefore do not clear video redistribution or production use. [1][2] | Paper describes 1,400 trimmed HD swings sampled at 720p/30 fps; the reproducible repo points to a Google Drive `videos_160` archive rather than hosting it directly. [2][3] | Eight temporal events: Address, Toe-up, Mid-backswing, Top, Mid-downswing, Impact, Mid-follow-through, Finish. Also swing bounding boxes, club/view metadata; paper describes boxes including ball and clubhead, not separate validated ball/clubhead tracks. [3] | `generate_splits.py` deterministically creates four train/validation pickle splits from `golfDB.mat`; current FairwayOS probe received 404 for the four published split filenames and has no paired `videos_160`. Existing local metadata hashes: `golfDB.pkl` SHA-256 `25067bde45b1a8b11c0f642ae79fbceef0d58423df16d9e72aea90cc55b77d82`; `golfDB.mat` SHA-256 `36523fcef016d93334691ea058d643bcf0d3a4a8c70b733f3b4cc0b8e5e0cf0a`. [2][4] | SwingNet baseline weights are linked from Google Drive; separate checkpoint licensing is unresolved. [2] | **Best event-sequencing research reference, but blocked now.** Do not download or enable until asset rights, paired clips, splits, and checkpoint terms are independently cleared. |
| **CaddieSet** | Public GitHub repository includes an MIT `LICENSE` for the repository contents. It does not state that the underlying swing videos or launch-monitor data are freely redistributable; the public repo currently exposes only the CSV data file. [5][6][7][8] | Paper says 1,757 shots from 8 people, with 924 FACEON and 833 DTL views; the public repository does not provide the claimed videos, resolution/FPS, or a reproducible video download manifest. [5][9] | Eight swing phases are used, with pose/joint-derived features and paired launch-monitor ball-flight values such as distance, carry, direction, spin, and ball speed. These are derived features and scalar shot outcomes, not published per-frame ball/clubhead annotations. [5][8][9] | Public data path is `data/CaddieSet.csv`; current repository tree contains no video directory, split manifest, or dataset hash manifest. Git head observed during review: `3c73d9d40580bb8a5a10711ad1fa10735a205ffe`; this is a source-repository hash, not a video/data-archive hash. [7][8] | No public checkpoint was identified in the repository. The paper used a fine-tuned SwingNet-derived sequence model, which does not clear the underlying checkpoint for FairwayOS. [9] | **Scientifically valuable for future owned-data schema design and ball-outcome linkage; not currently a legally reproducible video benchmark.** Request asset-level permission and a frozen video/annotation release before use. |
| **ClubheadDB** | PyPI metadata and project description state CC BY-NC 4.0. This supports a research-only path, not commercial or production deployment without additional rights review. [10][11] | Project describes over 10,000 down-the-line frames sourced from public YouTube/Reddit posts. Resolution and FPS are not specified in the published description. Video clips are reconstructed locally from source URLs/timestamps rather than delivered as a redistributable archive. [10] | Hand-annotated clubhead locations on 10,180 frames, with null examples represented by zero/empty labels. No event labels, ball annotations, course calibration, or landing labels are advertised. [10] | Video-level golfer split: 47 train, 10 validation, 10 test; metadata and timestamps plus `annotations.parquet` are used by the build tool. PyPI distribution hashes observed: wheel `965c42b33b4f92390dccad1854f5687d6a32f8fedd641cf8781117a702365d61`; sdist `15a1494a2c28855b84f76e9802f018c3387af976e32c19324c7ab2b62521d1d5`. These hash the Python distributions, not the reconstructed videos. [10][11] | No pretrained detector checkpoint was identified in the published project description. | **Best currently actionable research candidate for an isolated clubhead auxiliary track**, subject to verifying the asset-level terms for each source and recording downloaded-source hashes. It cannot establish swing events, impact, ball flight, calibration, or recommendations. |
| **JMS-GOLF-BALL-TRACKING / Roboflow Universe** | Page declares CC BY 4.0 for the dataset/model listing. [13] | 6,392 images are advertised; this is an image dataset, not a swing-video benchmark, and no FPS or clip-level split is supplied on the reviewed page. [13] | One `golfball` class; no temporal event labels, clubhead labels, course calibration, or landing ground truth. [13] | Dataset version 6 URL is public, but a reproducible video/clip manifest and content hashes were not identified. | Roboflow model listing is available, but its hosted/API workflow is not suitable for FairwayOS's local-only, no-cloud-inference boundary. [13] | **Reject for this milestone.** Useful only as a separately reviewed ball-image reference, not as a video/event dataset or local production model. |
| **GolfBall_Video_Tracking** | No explicit license was identified in the reviewed repository page. It includes a demo video and a `.pt` model in the public tree. [12] | A demo video is present, but resolution/FPS, source rights, dataset manifest, and reproducible split are not documented. [12] | Claims manually annotated golf-ball frames, but no auditable annotation release or split manifest was identified. [12] | No acceptable license/hash/split record found. | Includes `yolov5s_V2.pt`, but provenance and license are unclear. [12] | **Reject.** Do not download, use, or publish. |

## Findings

1. **GolfDB remains the best event-label reference, not an available next implementation target.** Its event ontology, official evaluator, and baseline are directly aligned with FairwayOS, but the current missing split/video assets and unresolved checkpoint/video rights keep the gate closed. [1][2][3][4]
2. **CaddieSet is the best scientific match for ball-outcome linkage, but not the best legally reproducible next dataset today.** Its public Git repository contains a CSV of derived features and outcomes, not the underlying videos or a frozen split/archive manifest. [5][7][8][9]
3. **ClubheadDB is the best currently actionable narrow research path.** It has explicit research terms, source metadata/timestamps, hand-labeled clubhead boxes, and golfer-disjoint train/validation/test splits. Its limitations are material: no event labels, no ball labels, no calibration, unspecified source-video resolution/FPS, no verified reconstructed-video hashes yet, and no cleared production checkpoint. [10][11]
4. **No candidate found in this review satisfies all requested fields.** In particular, no legally cleared public release was found that simultaneously provides redistributable golf-swing videos, event labels, per-frame ball and clubhead annotations, course calibration, public hashes, and frozen splits.

## Recommended single next path

**Approve a ClubheadDB-only, research-isolated reconnaissance/evaluation path—subject to one explicit legal and reproducibility checkpoint before any download.** The checkpoint should verify the current CC BY-NC 4.0 terms, confirm that each referenced YouTube/Reddit source may be accessed for local research under the project's constraints, resolve each source URL/timestamp without cookies or bypasses, and produce a local-only manifest containing source identifiers, timestamps, downloaded-file SHA-256 values, video metadata, and the published golfer split. No files should enter Git, and no ClubheadDB output should open FairwayOS phase, impact, ball, calibration, analytics, or recommendation gates.

GolfDB should remain the parallel event-sequencing reference. CaddieSet should remain a candidate for a future permissioned/owned-data collaboration or asset-level release request, not an implementation dependency.

**Implementation status: not started. Awaiting approval.**

## Sources

[1] https://github.com/wmcnally/golfdb
[2] https://raw.githubusercontent.com/wmcnally/golfdb/master/README.md
[3] https://ar5iv.labs.arxiv.org/html/1903.06528
[4] https://github.com/damilab/CaddieSet
[5] https://raw.githubusercontent.com/damilab/CaddieSet/main/README.md
[6] https://raw.githubusercontent.com/damilab/CaddieSet/main/LICENSE
[7] https://github.com/damilab/CaddieSet/tree/main/data
[8] https://raw.githubusercontent.com/damilab/CaddieSet/main/data/CaddieSet.csv
[9] https://arxiv.org/html/2508.20491v1
[10] https://pypi.org/project/clubhead-db
[11] https://pypi.org/pypi/clubhead-db/json
[12] https://github.com/SeanZ509/GolfBall_Video_Tracking
[13] https://universe.roboflow.com/golfrobotjms/jms-golf-ball-tracking
