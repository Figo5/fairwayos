# Automatic shot workflow

The product page accepts one video. Choosing the file starts analysis immediately; there are no object markers, frame selections, player profiles or analysis settings to submit. The result provides an annotated MP4, four-channel coverage and supported video-motion analytics, downloads and a synchronized original-footage comparison.

This local development prototype reproduces the ball-and-cup visual direction of the preceding scene demonstrator without its operator-selected coordinates. It uses local YOLO11 sports-ball proposals, repeated image evidence, current-frame white-component association, green camera registration, temporal sky components, and visible cup-rim evidence with or without a flagstick. The model initializes objects from the uploaded clip; filenames and previous review labels are not pipeline inputs. Cup location is an observed target, not an optimized golf recommendation.

## Run

Use an environment with OpenCV, NumPy and the `shot-intelligence` optional dependencies, FFmpeg/ffprobe on PATH, and three existing local model files: `yolo11n.pt`, `yolo11n-pose.pt`, and `models/research/clubhead-db-best.pt`. The configured Fairway environment already supplies them. See the four-layer section below for model roles and research limits.

```sh
python -m ghostcaddie.upload.server --root /tmp/fairwayos-uploads --port 8790
```

Open `http://127.0.0.1:8790/` and choose a clip. The page submits `analysis=shot_intelligence` automatically. The older diagnostic UI remains at `/review`; it is separate from this user flow. `FAIRWAY_SHOT_PYTHON` optionally selects the worker interpreter and `FAIRWAY_SHOT_MODEL` selects an existing local ball weight file. Missing model weights fail explicitly; processing does not download a model.

For a video-only CLI run:

```sh
python -m ghostcaddie.video.automatic_shot clip.mp4 --outdir result
```

## Evidence and limits

The tested Masters hole-16 input initializes the ball and cup automatically and exports all 714 source frames with unchanged presentation times and bit-exact original AAC payload. A separate variable-cadence test checks preservation of individual frame timestamps. `result.json`, `observations.json` and `verification.json` record source hash, zero operator seeds, initialization methods, coverage and media checks. Upload results retain the existing bounded local media lifetime; download the output to keep it.

Coverage is partial: clear green/sky views and repeated visible cup rims can be supported. Textured backgrounds, tiny near-player balls, occlusion and rapid camera changes may remain unresolved. This is not a validated universal golf detector. Missing or ambiguous objects remain absent; camera-registration failures clear the historical trail. No extrapolated positions are published as observations. Physical mph, carry distance, 3D flight and optimized aim remain unavailable without reliable supporting data. Automatic processing does not imply validated accuracy.

Clips are bounded to 180 seconds and 6000 decoded frames with square pixels. Analysis caches lossless frames temporarily at up to 1280×720; the export preserves native frame dimensions (an odd dimension receives one black padding pixel for H.264), individual presentation timestamps and the first original audio stream. Cancellation terminates the worker process group and cannot publish a late result. Source hashes bind uploads and worker results.

Validation:

```sh
python -m pytest tests/test_automatic_quality.py tests/test_temporal_objects.py tests/test_shot_people_metrics.py tests/test_automatic_shot.py tests/test_upload_automatic_shot.py tests/test_upload_ui_endpoints.py tests/test_upload_security.py tests/test_upload_fresh_ai_job.py tests/test_scene_intelligence.py tests/test_upload_server.py tests/test_upload_ui_canonical_result_mapping.py -q
```

Development artifacts and the real browser-upload result are saved in `~/fairway-os-runs/fairwayos-automatic-shot-2026-10-06/`.


## Temporal extension (2026-10-06)

The automatic worker now considers tiny dark components against bright sky, requires a unique sustained moving chain, and resets at view transitions/cuts. These are image locations, not ground-registered trajectories or 3D flight. Textured tree/crowd intervals can remain unavailable. The model remains generic local YOLO11n, not a newly trained golf classifier.

Flag-free cups use flat pale rim components, person exclusion, repeated image support and a stationary-ground motion check. Cup inference is independent of ball initialization. Ground ball initialization tries ranked repeated neural proposals with tight component alignment; lost tracks require new repeated neural evidence to reacquire. Every component observation uses current pixels.

`ghostcaddie.demo.automatic_quality` evaluates source-bound sparse AI visual references outside the runtime. It reports visible misses, wrong locations and explicit-negative false observations separately; unknown labels are excluded from denominators. Sparse AI references are not independent production accuracy or full-video drift validation.

Current tracking iteration, its frozen baseline/candidate, all 23 native exports and sparse visual-reference evaluation are saved in `~/fairway-os-runs/fairwayos-tracking-v2.3-2026-10-06/`.

The final cup publication gate retains broad pale candidates as ambiguity blockers before rejecting broad appearance-only chains. This prevents a size filter from turning a competing narrow tee marker into a unique cup. It conservatively withholds close, wide flag-free cups. A repeated selection of the same file starts a fresh automatic job; the upload chooser clears its selection after capturing the file.

The reviewed publication branch also vetoes fragmented broad rivals independently of track eligibility, requires 18 consecutive ground-registered support frames, withholds ball/rim overlap, preserves existing flag-supported cup targets, and initializes at the first published support. Failed registration publishes no appearance-only cup. These conservative guards do not replace a golf-specific semantic model or independent-source evaluation.

## Four-layer development extension

The same video-only worker now runs a local generic YOLO11 pose model and a guarded clubhead pass in all views, including golfer views that are not majority green. It associates a prominent primary person using current hand/posture evidence, full-body visibility, spatial continuity and torso appearance. This is a candidate golfer track, not face/name identification. New person acquisition also requires setup-posture evidence and no nearby comparable people; folded-hand spectators in a landing view are withheld. A cropped camera operator found in the first development candidate drove a required current foot and unclipped-bounds guard, with nose support for new acquisition. A continuing spatial/appearance-matched person may turn away; every published body joint still comes from the current frame. Current pose is never carried through a missing frame or replaced with a previous pose.

Local `models/research/clubhead-db-best.pt` supplies search proposals only. Its standalone Tiger route was previously rejected; it is not promoted as a production detector. The author dataset declares CC-BY-NC-4.0; the weights are used here for local research. A head must also have a current shaft connected to a current wrist at confidence ≥0.75 and an independent core wider than the measured shaft. An additional geometric search tests a thin, bounded current shaft stripe and either a turf-backed dark lobe below the hands or a transverse reflective metal lobe. Round bright blobs and round balls connected to a blade are withheld. The worker withholds ball-shadow overlap, ambiguous heads and unsupported/blurred frames. Both body and head publication require at least 3 consecutive actual source observations, confirmed retrospectively. No missing coordinates are filled.

The coverage candidate adds overlapping native-image pose searches when full-view acquisition fails, and fresh cropped pose inference for an existing small subject. Crops propose current poses; previous boxes only locate a search region. Complete, isolated, prominent rear subjects need a projected golf posture as well as ears, torso and legs; an upright rear subject remains unresolved without independent action/club evidence. Hidden rear-grip wrists are withheld from the overlay and hand analytics. These are generic subject-selection heuristics, not player identity or independent joint visibility validation. Operators, spectators, caddies and wrong-person locks remain evaluation risks.

`FAIRWAY_BODY_MODEL` and `FAIRWAY_CLUB_MODEL` select existing local weights; the default pose path is `yolo11n-pose.pt`. Missing weights fail explicitly. All three proposal models are local and no weights are downloaded. With optional cloud AI ball confirmation disabled, no video/frame is sent to another service; see the cloud configuration section below.

Exports add a blue clubhead marker/trail and small shoulder/hip/hand anchors, plus supported projected torso and video-motion readings. A complete skeleton is not drawn. Per-frame and per-segment analytics in `analytics.json` use native source coordinates and actual presentation times. Body summaries are separated by scene/person/torso-scale changes. Hand/club motion relative to hips subtracts image translation; a normalized club series uses projected torso lengths. Neither is calibrated 3D motion. Projected body angles need current confident joints. Cup residual is a tracking/registration jitter proxy. Physical ball/club mph, carry, true impact, swing tempo and 3D body rotation stay null. Address observations do not establish downswing speed.

The upload page shows the four partial layer counts and measurements per observed view automatically, and supplies an analytics download through `/result-analytics?job=...`, under the same retention policy. Choosing the raw clip is still the only analysis action.

Current four-layer development artifacts live in `~/fairway-os-runs/fairwayos-four-layer-v1.4-2026-10-06/`. Candidate D ball/cup outputs remain the rollback baseline; the original four-layer candidate is retained with a rejection receipt. These same-event clips and sparse AI audits are development diagnostics, not an independent accuracy estimate or universal tracking validation.

## Native small-ball development search

The video-only worker also searches native current pixels near confident current hips, ankles and wrists. The pose only bounds the region. Detached round bright components need turf clearance from shoes, absence of current head overlap, tight high-score local sports-ball localization and three neural sample checks on a unique continuous component sequence. An explicit sampled neural miss, a rival, missing current body/component, scene/person change, scale jump or excessive anchor-relative drift splits support. Current ground registration locates a search when available; otherwise the address ball must remain approximately static in the image. No missing coordinate is filled.

Confirmed detail balls supplement missing observations after the old ball/cup/body publication gates. They do not overwrite old ball, target, camera or body rows. Ambiguous adjacency to a different established ball is withheld. They have a separate near-player context, an observed-ball label, and image-motion analytics with source PTS; they are not labelled airborne and do not join a different tracking channel/person/view/scale for a velocity. New ball/head overlap removals and their broken head-support runs are recorded. Diagnostics use explicitly native source pixels and remain separate from published ball fields.

Generic neural confidence and repeated samples are correlated model evidence, not independently verified accuracy. Bright flowers, tiny round impostors, touching ball/head pixels and resolution dependence remain research risks. Sparse same-event AI reviews are consumed diagnostics; physical mph stays null. The frozen rollback is `~/fairway-os-runs/fairwayos-coverage-v2-2026-10-06/`. The next candidate is `~/fairway-os-runs/fairwayos-small-ball-v3-2026-10-06/`; consult its decision and verification receipts for accepted/rejected status.

## Scoped processing retention

The bounded automatic upload worker leases its source during processing, so ordinary polling cannot expire and delete an uploaded input while analysis is reading it. Successful completion starts the normal one-hour review/download window. Explicit media deletion remains immediate. Failure and cancellation release the lease without renewing expired media. Other upload modes retain their existing policy. `tests/test_upload_processing_retention.py` covers expiry, successful renewal, explicit revocation, failure and cancellation.

## Native clubhead detail fallback

When analysis downsizes a source frame, a missing unambiguous baseline head can trigger a fresh original-pixel search near the current hands. The default local ClubheadDB model only proposes boxes; a proposal still needs current hand-attached shaft pixels and a separate distal head core. Geometric-only native proposals are withheld after source-pixel audits exposed shoe-edge errors. Baseline learned/geometric ambiguity or disagreement cannot be rescued by this fallback. Crop-clipped proposals are rejected, and ambiguity margins retain their analysis-equivalent scale.

Native coordinates are converted into the analysis space once for publication checks and returned to source coordinates during export. Core area units record the resolution at which pixels were measured. `frames_by_search_resolution` distinguishes native detail from analysis observations. Ball searches retain the original baseline head exclusions, while final overlap checks give supported balls priority. Existing body, ball, cup and camera observations remain the rollback contract. Three consecutive current observations are still required; missing frames remain absent.

Club motion does not pair measurements found at different search resolutions. Native source detail changes localization and morphology; its head-center error is unmeasured, and address or backswing coverage does not establish downswing speed. The semantic-coverage development receipts are in `~/fairway-os-runs/fairwayos-semantic-coverage-v4-2026-10-07/`. Larger generic ball models and a local golf-ball checkpoint are separate rejected probes, not new default weights. Physical mph remains unavailable without verified calibration or measured data.

A native head core also abstains beside small current round bright pixels. This negative-evidence check operates even when ball semantics cannot publish a ball: a white ball's shadow cannot become a clubhead just because the ball channel is unresolved. It does not publish or carry a ball position. Long chrome strips and white crop-boundary backgrounds are excluded from this veto. Source audits can reveal new errors; rejected revisions and their receipts remain separate from accepted compilation outputs.

### Optional server AI ball confirmation

Set `FAIRWAYOS_BALL_VISION=claude-cli` in the worker/service environment to enable Claude Opus5.5 classification of existing current native round components. Install/configure the administrator's Claude CLI once; no per-video settings or user markers are needed. The default absent configuration retains local tracking. Missing providers/failures abstain.

This mode sends native context/detail crops to the configured cloud provider, with tools disabled, schema-bound responses and bounded call/candidate/time budgets. Model output cannot supply coordinates. Three positive sampled anchors and every intervening current component are required. Old observations remain authoritative; semantic balls cannot evict established heads or cups. Pixel-motion statistics separate evidence channels. Receipts under `semantic_ball/` bind source/frame/component/images/model/prompt; the mode remains development research, without independent accuracy or physical speed validation.
