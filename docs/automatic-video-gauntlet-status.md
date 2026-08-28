# Automatic Video Gauntlet Status

Date: 2026-08-28

Status: **bounded ingestion and artifact workflow verified; automatic golf perception remains blocked and unvalidated**.

## Verified workflow

The following commands are available without changing the existing fixture or human workflows:

```bash
python3 -m ghostcaddie video-auto-analyze \
  --video /path/to/golf_video.mp4 \
  --calibration /path/to/calibration.json \
  --course /path/to/course.json \
  --player /path/to/player.json \
  --out out/automatic-analysis
```

`video-auto-analyze` is an alias for the existing guarded `video-automatic-analyze` command. It accepts only validated observation evidence and does not itself claim to be a detector.

For arbitrary supported YouTube ingestion:

```bash
python3 -m ghostcaddie youtube-auto-try \
  --url 'https://youtu.be/VIDEO_ID' \
  --out out/youtube-auto-try \
  --segment-start 0 \
  --segment-duration 20 \
  --yt-dlp /Users/giofiore/ghostcaddie-tour/.venv-video-modern/bin/yt-dlp \
  --render-video
```

The YouTube path enforces HTTPS allowlisting, `--no-playlist`, a 20-second maximum segment, low-resolution selection, download/disk/time limits, `shell=False`, and no cookies, credentials, browser sessions, proxies, DRM bypass, or platform-protection bypass.

## Runtime verification

Isolated AI environment:

- Python 3.11.16
- OpenCV 5.0.0
- PyTorch 2.13.0
- MPS available on Apple M2 Pro
- Ultralytics 8.4.131

Trusted generic smoke weights currently present:

```text
yolo11n.pt
SHA-256: 0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1

yolo11n-pose.pt
SHA-256: 869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0
```

These are generic person/pose weights and are not validated golf detector or clubhead/contact weights.

## Real smoke result

The requested valid URL was processed:

```text
https://youtu.be/PsJvcITOVRc
segment: 0–20 seconds
```

Verified output directory:

```text
out/youtube_auto_try_smoke/
```

Produced:

- downloaded `ingest/source.mp4`;
- 20.019-second, 256x144 source video;
- 40 extracted frames;
- contact sheet;
- 40 annotated-frame artifacts;
- 20-second annotated MP4;
- sanitized `diagnostics.json`.

The contact sheet is viewable, but the source is broadcast footage and remains a stress case only.

Final status:

```json
{
  "status": "blocked",
  "ingestion_status": "complete",
  "perception_status": "unavailable",
  "coordinate_space": "pixels",
  "blocking_reasons": [
    "calibration_unavailable",
    "detector_unavailable"
  ]
}
```

No recommendation, calibration, course-space `ShotEvent`, trajectory, landing, or analytics was fabricated.

## Prior generic smoke limitations

On the existing 854x480 broadcast stress clip:

- person detections: 89/100 frames;
- sports-ball detections: 1/100 frames;
- unique generic person track IDs: 24;
- pose output: 72/100 frames;
- generic person/pose output did not establish golfer identity, feet-anchor validity, golf-ball tracking, clubhead tracking, impact timing, landing, or production reliability.

## Blocking inputs

The gauntlet cannot honestly reach the final supported-video acceptance claim until all of the following exist:

1. an approved automatic adapter that produces validated `video-observations.v1` evidence without human annotation;
2. approved golf-specific detector/pose/ball/clubhead weights, or documented approval for custom weights including source, license, training method, isolation, and hash;
3. at least three consenting real single-shot golf clips, including one held-out clip;
4. four-point calibration for each evaluation clip;
5. ground-truth golfer, anchor, ball, clubhead, phase, impact, and landing annotations;
6. thresholds tuned on designated tuning data and then evaluated on held-out footage.

Until those inputs are supplied and all provisional gates pass, the correct product status is:

> infrastructure and bounded best-effort artifacts ready; automatic golf perception and recommendation generation not validated.

The current implementation intentionally stops rather than simulating success, lowering gates, or labeling generic detections as golf perception.

## Accepted repository quality gate

The public `FairwayOS CI` workflow is accepted as the repository quality gate. It runs on `push` and `pull_request` with Python 3.11, executes the standard-library unittest suite, compiles source and tests, and runs the existing CLI smoke scenarios. The workflow does not install the optional AI/MPS environment, use secrets, or upload videos, weights, annotations, or generated artifacts.

The accepted public workflow run is:

```text
https://github.com/Figo5/fairwayos/actions/runs/33179154929
head: 58a5322ceb6080d31201cc11f96493edfc8872f4
conclusion: success
```

## Accepted GolfDB research-only result

The GolfDB acquisition result is accepted as **research-only and blocked**. The available metadata does not establish a legally cleared, paired labeled evaluation corpus for FairwayOS:

- no confirmed labeled `val_split_*.pkl` files;
- no paired `videos_160` evaluation set;
- no PCE or impact-accuracy claim;
- SwingNet remains disabled for production;
- local downloaded evaluation files remain ignored and unpublished.

The automatic-perception production gates, human fallback, and existing analytics contracts remain unchanged.
