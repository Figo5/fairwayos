# FairwayOS

Upload a golf clip and receive an annotated video with supported video-motion analytics. The current prototype automatically searches for the **ball, clubhead, golfer and cup**, with no points to mark, frame selections or per-video settings. The Python package is still named `ghostcaddie`.

**Status: paused at the October 7, 2026 development checkpoint.** The upload flow and presentation work. Tracking is partial, wrong-object/person locks remain possible, and independent accuracy has not been established. This is a research prototype, not a finished launch monitor or a production golf tracker.

## What it does

FairwayOS ingests a golf clip and emits an annotated video plus per-clip observations for the ball, clubhead, golfer and cup, with no manual tracking points. The October 7, 2026 checkpoint was evaluated on 33 clips (23 from the 2019 Masters, 10 from newer PGA TOUR events), totalling 13,280 frames. Demo videos derived from copyrighted broadcast footage are not distributed here; the table below describes that evaluation run.

## Actual tracking status

Counts describe frames with published assertions, **not correct detections or validated accuracy**.

| Collection | Clips / frames | Ball | Clubhead | Body | Cup |
|---|---:|---:|---:|---:|---:|
| Tiger 2019 Masters, v5 | 23 / 9,466 | 3,781 | 869 | 3,785 | 2,568 |
| Scheffler, McIlroy, Fleetwood tournament tests | 10 / 3,814 | 860 | 512 | 1,477 | 0 |

- The Masters batch has 150 frames with all four layers simultaneously asserted. That is not a correctness score or full-shot coverage.
- The newer tournament batch misses every cup. Fleetwood's hole 1 lag putt selects the caddie for its ten body assertions. McIlroy's ace and two Fleetwood clips have no ball observations.
- One drive trim includes 20 frames of the next approach setup. Automatic stroke boundaries and broadcast inset handling need improvement.
- Independent reference labels remain unreviewed. AI reviews and Claude classifications are diagnostic evidence, not ground truth. Public footage may overlap unknown model pretraining.
- Analytics use native pixels, source video timestamps and projected 2D body geometry. Camera motion, slow motion, perspective, jitter and wrong identity affect their meaning.
- Physical ball/club mph, carry, spin, launch angle, true impact, swing tempo and 3D biomechanics are unverified or unavailable. Fleetwood's **142 / 131 / 111 mph** figures are separately attributed broadcaster-reported ball speeds, not FairwayOS measurements.

## Run the automatic workflow

Use Python 3.11, FFmpeg and ffprobe:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e '.[shot-intelligence]'
```

Supply three existing local models; processing does not download weights:

- `yolo11n.pt`: generic sports-ball/person proposals.
- `yolo11n-pose.pt`: generic current-frame body pose.
- `models/research/clubhead-db-best.pt`: experimental clubhead search proposals; current-image geometry is also required.

Weights are not distributed. The clubhead checkpoint remains research-only under its recorded dataset terms; see [third-party notices](THIRD_PARTY_NOTICES.md). Missing weights fail explicitly. `FAIRWAY_SHOT_MODEL`, `FAIRWAY_BODY_MODEL` and `FAIRWAY_CLUB_MODEL` can override these automatic-workflow paths. `FAIRWAY_BODY_MODEL` has a different model format in the archived canonical workflow; configure each mode separately.

```sh
# Choose a video on the local page; analysis starts immediately.
.venv/bin/python -m ghostcaddie.upload.server \
  --root /tmp/fairwayos-uploads --port 8790

# Or analyze directly:
.venv/bin/python -m ghostcaddie.video.automatic_shot \
  /absolute/path/to/clip.mp4 --outdir out/shot
```

Open `http://127.0.0.1:8790/`. Inputs are bounded to 180 seconds / 6,000 frames. Outputs: `annotated.mp4`, `result.json`, `observations.json`, `analytics.json` and `verification.json`. Exports preserve native dimensions and presentation timestamps; compatible source audio is copied and checked. Odd dimensions receive padding for H.264. Successful upload processing starts a one-hour review/download window; download results to retain them. `/review` preserves the older assisted diagnostic UI.

### Optional cloud AI ball confirmation

The published checkpoint used `FAIRWAYOS_BALL_VISION=claude-cli`. An administrator configures an authenticated Claude CLI once before starting the server. Claude Opus 5.5 classifies existing native image components and supplies no coordinates. This sends bounded source-image crops to the configured cloud provider. With the variable unset, processing uses the local route and may produce different ball coverage. Provider failures abstain; AI decisions can vary across runs.

```sh
export FAIRWAYOS_BALL_VISION=claude-cli
.venv/bin/python -m ghostcaddie.upload.server \
  --root /tmp/fairwayos-uploads --port 8790
```

The user still provides only a video. AI confirmation does not resolve invisible objects or provide physical calibration. Details: [automatic workflow](docs/automatic-shot-workflow.md).

## Resume here

1. Fix target-golfer identity and cup detection on the recorded caddie/cup failures.
2. Add independent native-frame labels with visible, absent and ambiguous intervals for all four objects. Keep source/player/round groups together and freeze a held-out evaluation set.
3. Improve ball/clubhead proposals in fast blur, wide greens, cropped players, insets and camera cuts. Keep unsupported observations absent.
4. Improve stroke segmentation; measure false locks, localization, visible recall, runtime and AI repeatability before promoting changes.
5. Add physical speeds only with verified measured values or reliable calibration.

[Resume guide and local archive map](docs/resume-fairwayos.md) · [Four-object evaluator](docs/four-object-evaluation.md)

## Validation and older workflows

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q ghostcaddie tests
```

Synthetic tests cover execution, evidence gates, source binding, cancellation, retention, media timing and evaluation contracts. Passing tests do not establish real-video accuracy. Checkpoint results: [FINAL_STATUS.md](FINAL_STATUS.md).

Canonical body-focused analysis, assisted showcases and saved tracking review remain available in the [archived workflow guide](docs/legacy-workflows.md). Automatic research does not expand canonical-mode claims. Synthetic shot analytics and provider fixtures are separate from real video inference.

Source, tests, documentation and safe summaries are in Git. Generated videos are release assets, kept out of Git history. Original downloads, weights, credentials, upload state and detailed private logs remain local. Broadcast rights and model/dataset terms are separate from the source-code license.

[License](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)
