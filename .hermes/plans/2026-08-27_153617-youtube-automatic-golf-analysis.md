# YouTube Ingestion and Automatic Golf Analysis Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add an opt-in, local-only `youtube-analyze` workflow that accepts a validated YouTube video URL, safely downloads one permitted video into a project boundary, routes it through automatic golf perception, applies supplied calibration/course/player context, reuses unchanged GhostCaddie analytics, and produces auditable annotated output without claiming reliability before real-footage evaluation.

**Architecture:** Keep YouTube ingestion, automatic perception, and analytics as separate boundaries. URL parsing and downloader execution produce a sanitized local-video provenance record; the existing FFmpeg metadata/extraction layer consumes only that local file; the future automatic perception adapter emits versioned pixel-space observations; calibration/reconstruction converts supported points exactly once into the existing `ShotEvent`; unchanged `run_pipeline()` produces the recommendation; rendering re-projects only the recommendation into source pixels. The existing fixture, `video-prepare`, `video-import`, `video-human-analyze`, `video-analyze`, `run`, `session`, and `provider-session` commands remain behaviorally unchanged.

**Tech Stack:** FFmpeg/ffprobe; optional locally installed `yt-dlp` or an explicitly configured downloader executable; the automatic golf CV stack from the Automatic Golf Video Perception plan; existing standard-library contracts and analytics. No cookies, credentials, cloud model uploads, shell interpolation, DRM bypass, authentication bypass, or arbitrary URL fetching.

---

## 1. Design principles and hard boundaries

1. YouTube is an input transport, not a trusted data source.
2. Only allowlisted YouTube URL forms are accepted. The command must reject arbitrary hosts, schemes, playlist/channel/search URLs, filesystem URLs, IP addresses, redirect URLs, and wrapper tokens such as an unresolved `@url:` placeholder.
3. Download at most one finite, available, non-private, non-live video, subject to explicit size, duration, frame, and disk budgets.
4. Use subprocess argument arrays only. Never build a shell command string, invoke `shell=True`, evaluate downloader metadata as code, or interpolate URL/path values into a shell expression.
5. Never bypass login, cookies, age restrictions, DRM, rate limits, bot checks, or other platform protections. If the downloader reports that access requires authentication or cannot legally/technically proceed without bypassing a protection, stop with an actionable error.
6. Keep downloaded media and temporary files inside a dedicated permitted output/project boundary, with symlink and traversal checks before and after download.
7. Store only sanitized provenance by default: platform `youtube`, normalized video ID, downloader name/version, selected format summary, and local processing status. Do not store the original URL, title, uploader, description, comments, downloader command line, cookies, headers, or raw logs unless an explicit future privacy option is approved.
8. Treat video pixels, captions, title/description metadata, container metadata, filenames, and downloader output as untrusted input. Do not execute or interpret text from the video.
9. Never upload downloaded media to a model by default. Any cloud model path must be a separate explicit opt-in with a clear consent boundary and separate tests.
10. No automatic result is analytics-ready when required observations, calibration, or confidence gates fail. Emit explicit `null`/`unavailable` evidence and a human-fallback package instead.

## 2. Supported user command

Add a distinct command rather than changing existing video commands:

```bash
python3 -m ghostcaddie youtube-analyze \
  --url 'https://www.youtube.com/watch?v=VIDEO_ID' \
  --calibration calibration.json \
  --course sample_hole.json \
  --player sample_player.json \
  --project-root /path/to/project \
  --out /path/to/output
```

The attached `@url:` form is UI/source notation, not a valid CLI URL. The CLI receives the resolved literal URL and rejects unresolved placeholders or ellipses.

### Required arguments

- `--url`: one supported YouTube watch URL;
- `--calibration`: project-relative four-point calibration JSON for the actual downloaded video dimensions;
- `--course`: project-relative course JSON;
- `--player`: project-relative player JSON;
- `--project-root`: existing project boundary;
- `--out`: output directory, inside the permitted project/output boundary according to the project’s path policy.

### Optional arguments

- `--format` / `--quality`: constrained user-requested quality preset, for example `best-mp4`, `720p`, or `1080p`; never accept arbitrary downloader expressions in the first implementation;
- `--start` and `--duration`: bounded clip window, with non-negative start and duration under the configured maximum;
- `--sample-fps` and `--max-frames`: analysis budget;
- `--max-download-bytes`, `--max-duration-seconds`, and `--max-disk-bytes`: hard safety budgets with conservative defaults;
- `--downloader`: optional explicit executable path or configured downloader name;
- `--device`: `cpu` or explicitly available local accelerator for the automatic perception provider;
- `--detector-model`, `--pose-model`, and `--tracker`: explicit local model/provider selections;
- `--render-video`: request sampled `annotated_video.mp4`.

Do not initially expose raw `yt-dlp` arguments. Expand options only through reviewed, typed flags with bounded values.

### Exit behavior

- Exit `0` only when the video was downloaded, automatic observations were generated, reconstruction and analytics completed, and required artifacts passed integrity checks.
- Exit non-zero for invalid URL, downloader unavailable, unavailable/private/live/protected video, budget exhaustion, malformed metadata, unsafe path, unsupported camera/view, failed calibration, or missing required evidence.
- A partial/fallback package may be written before a non-zero exit, but diagnostics must state `partial` or `failed` and must not contain a recommendation unless analytics actually completed.

## 3. YouTube URL validation and provenance

### Accepted forms for the smallest reliable implementation

Accept only HTTPS URLs whose host is one of:

- `youtube.com`;
- `www.youtube.com`;
- `m.youtube.com`;
- `youtu.be`;
- optionally `youtube-nocookie.com` only if the downloader path is verified to support it.

Accept only:

- `https://www.youtube.com/watch?v=<video_id>`;
- `https://youtu.be/<video_id>`;
- optionally `/shorts/<video_id>` after explicit evaluation.

Normalize host casing and harmless URL encoding, extract the canonical 11-character YouTube ID, and discard nonessential tracking parameters. Reject missing, repeated, malformed, or conflicting IDs; playlist parameters; channel/user/search/live URLs; non-HTTPS schemes; fragments that alter meaning; credentials; ports; and URLs containing control characters.

The parser should return a small immutable record:

```text
YouTubeSource(platform="youtube", video_id="…", normalized_url=None)
```

The normalized URL is retained only in memory or an optional user-visible confirmation, not serialized by default. Reports contain `platform: youtube` and `video_id` only.

### Error categories

Use stable, testable errors such as:

- `unsupported_url_scheme`;
- `unsupported_url_host`;
- `unsupported_youtube_url_form`;
- `missing_video_id`;
- `invalid_video_id`;
- `playlist_not_allowed`;
- `unresolved_url_placeholder`.

## 4. Downloader options

### Option A — optional `yt-dlp` dependency: recommended default

`yt-dlp` is a maintained, feature-rich downloader with YouTube support, format selection, output templates, filtering, rate limits, retries, and download-section controls.[1] It also documents that site support can break as platforms change, so support must be detected at runtime and failures must be surfaced rather than hidden.[2]

Use it only when:

- the executable or Python module is explicitly installed/configured;
- its version is captured in sanitized provenance;
- it is invoked with an argument list;
- `--no-playlist` and a fixed output path/template are used;
- live content is filtered out before download where possible;
- the selected format is bounded and compatible with FFmpeg;
- output is verified after completion;
- stderr is captured only for local diagnostics and sanitized before any report.

Prefer an executable subprocess over importing arbitrary downloader internals initially. Use a dedicated temporary file inside the output boundary, then atomically rename after validating regular-file status, size, and FFprobe readability.

Do not rely on a downloader’s metadata filter alone for security. Re-check final file size, duration, codec/container, and symlink/path invariants locally.

### Option B — user-preinstalled downloader executable

Support an explicit `--downloader /absolute/path/to/yt-dlp` or configured executable only after validating that it is a regular executable and resolving its real path against an approved tool policy. This avoids installing dependencies automatically and is useful for controlled environments, but executable provenance and version handling are more complex.

### Option C — user downloads manually: fallback and safest default

Retain `video-prepare`/`video-auto-analyze --video` as the zero-downloader path. If no approved downloader is available, tell the user to download the video through a permitted method and run the local-file workflow. This is the smallest reliable fallback because it avoids platform access, downloader maintenance, and credential/privacy concerns.

### Recommendation

Implement Option C first as the guaranteed path and Option A as an optional, explicit integration. Do not make downloader installation automatic. The command should fail clearly with a manual-download fallback when no configured `yt-dlp` is available.

## 5. Safe download protocol

1. Parse and validate the URL before invoking any external program.
2. Create a unique output staging directory inside the resolved output boundary; reject symlinked ancestors and output/source collisions.
3. Probe metadata with the downloader without downloading the full video where supported. Request only one video and reject playlists/live content.
4. Enforce expected duration, resolution, and estimated-size budgets before download. If size is unknown, use a streaming byte counter and abort when the hard limit is reached.
5. Pass only fixed, typed arguments. Do not pass user strings as format expressions, output templates, paths, or shell fragments.
6. Capture downloader stdout/stderr separately. Do not write raw logs to reports; sanitize error categories and redact URLs, cookies, authorization headers, local paths, and tokens.
7. Download to a generated filename such as `source.download.part`, never a title-derived filename.
8. Verify the completed file is a regular file within the boundary, below size/duration limits, readable, and accepted by `validate_video_source()` and `inspect_video()`.
9. Atomically rename it to a generated internal filename such as `source.mp4` only after verification.
10. On failure, delete only staging files inside the staging directory and preserve a sanitized failure diagnostic.
11. Never use cookies-from-browser, credential files, proxy credentials, or authentication flags in the initial implementation.

### Rejected states

Reject and explain:

- live/upcoming streams;
- private, deleted, region-blocked, unavailable, age-restricted, or login-required content when access is not already public and permitted;
- DRM or protected streams;
- unsupported audio-only, playlist, HDR/codec, or container outputs;
- duration, estimated-size, resolution, frame-count, or disk-budget violations;
- downloader errors indicating rate limiting, bot checks, or access protections.

The tool must not retry indefinitely or attempt alternate bypass methods.

## 6. Automatic analysis pipeline

After verified download, call the same local path seams as `video-auto-analyze`:

```text
sanitized YouTubeSource
  -> verified local video
  -> inspect_video / ffprobe
  -> deterministic frame extraction
  -> automatic golf detector
  -> temporal tracker
  -> camera-motion/view-quality assessment
  -> phase/contact/flight/landing evidence
  -> validated automatic observations
  -> fixed-camera calibration validation
  -> exactly-once pixel-to-course mapping
  -> existing ShotEvent
  -> unchanged run_pipeline()
  -> diagnostics/recommendation/SVG
  -> pixel-space overlays and optional annotated video
```

Automatic output must use a distinct provenance value or observation schema, for example `video-auto-observations.v1`, not the human annotation schema. Each value records whether it is detected, tracked, model-verified, inferred, or unavailable. Predictions used to bridge a bounded tracking gap must not be serialized as observed detections.

### MVP view gate

The first automatic YouTube release supports only:

- one trimmed shot;
- one golfer;
- fixed or nearly fixed camera;
- visible ground plane sufficient for the supplied calibration;
- sufficient temporal resolution around impact;
- ball and club evidence only where visibly resolvable.

Before inference, compute and report:

- camera-motion score;
- number of persistent golfer tracks;
- resolution/FPS and estimated ball pixel size;
- occlusion and blur indicators;
- whether the course/calibration view is present and dimension-compatible.

If the clip fails the view gate, produce tracking diagnostics and a human-review/video-prepare fallback, but do not run analytics or claim a shot result.

### Failure semantics

- Missing ball after impact: `ball_after_impact = null`, warning `ball_not_resolved`.
- Clubhead blur: contact interval and reduced confidence, or unavailable.
- Landing outside frame: `landing = null`, never a projected landing guess.
- Multiple golfers: reject automatic single-shot analysis unless a deterministic primary-track selection rule is later evaluated; do not choose by arbitrary confidence alone.
- Camera movement: reject MVP analysis when motion exceeds the fixed-camera threshold; later support motion compensation with per-frame homographies.
- Occlusion/low FPS: preserve frame intervals and uncertainty; do not invent sub-frame positions.
- No usable course view/calibration: pixel-space detections may be retained, but course-space `ShotEvent`, expected-strokes, hazards, and recommendation are unavailable.

## 7. Calibration for a YouTube clip

### MVP requirement

The user supplies a project-relative calibration JSON created for the actual downloaded clip. It must contain:

- source image width/height matching FFprobe;
- exactly four finite source image points;
- exactly four paired finite course points;
- source and engine units;
- optional calibration provenance and reprojection quality.

The user can create it by opening the existing offline `video-prepare` workspace on the downloaded/local file, selecting four visible course landmarks, and supplying course coordinates. The workflow must make clear that calibration belongs to this camera/view and cannot be reused blindly for a different crop, zoom, pan, or video.

### No usable course view

If the YouTube clip is a close-up swing, indoor net, simulator view, montage, moving broadcast angle, or otherwise lacks four stable course landmarks:

1. preserve local metadata and pixel-space automatic detections if available;
2. write diagnostics with `calibration_status: unavailable` and the reason;
3. do not construct a course-space `ShotEvent`;
4. do not run expected-strokes, wind, hazard, or decision analysis;
5. generate a human-review package only if a human can reasonably calibrate the view;
6. state that the result is unavailable, not that the shot was poor or successful.

## 8. GhostCaddie integration and outputs

Use an automatic-specific orchestration seam that mirrors the accepted human path but keeps `run_pipeline()` and protected analytics unchanged.

- Load course/player through `ProjectBoundary`.
- Validate calibration dimensions against the downloaded video.
- Convert supported pixel points exactly once through the existing reconstruction boundary.
- Call unchanged `run_pipeline()` exactly once for an accepted automatic `ShotEvent`.
- Keep visual annotations in original pixel coordinates.
- Use inverse calibration only to re-project course-space recommendation graphics for rendering.
- Preserve human fallback as a distinct explicit path; do not merge automatic and human fields silently.

Successful outputs:

```text
out/<safe-video-id>/
├── source.mp4                         # optional retention, user-controlled
├── source_metadata.json               # sanitized YouTube + FFprobe provenance
├── frames/frame_*.jpg
├── frames/frame_manifest.json
├── contact_sheet.jpg
├── observations.json
├── tracks.json
├── event_timeline.json
├── normalized_shot.json
├── recommendation.json
├── overlay.svg
├── diagnostics.json
├── annotated_frames/frame_*.jpg
└── annotated_video.mp4                # only with --render-video
```

`source_metadata.json` contains platform/video ID, downloader version/name, safe format summary, dimensions, duration, FPS, and processing status. It does not contain the original URL by default, local absolute paths, downloader logs, cookies, credentials, title/description, or uploader metadata.

## 9. Security and privacy model

- Local-only processing by default.
- Explicit separate opt-in before any cloud model or remote inference; downloaded media is never sent by default.
- Enforce maximum download bytes, duration, frames, resolution, staging disk use, final output disk use, and subprocess wall time.
- Enforce project-bound path resolution, reject `..`, absolute project resources, symlink escapes, output collisions, and downloaded files outside the staging directory.
- Keep secrets out of CLI arguments, reports, logs, environment dumps, and model provenance.
- Do not read browser cookies or browser profiles.
- Sanitize all downloader errors before writing diagnostics.
- Treat subtitles, thumbnails, titles, descriptions, filenames, and embedded metadata as untrusted strings; never render unsanitized values into HTML/SVG.
- Use deterministic generated filenames and relative artifact references.
- Provide a cleanup option that deletes downloaded source and staging artifacts without touching user input or project resources.
- Document copyright, terms-of-service, and user permission responsibilities without attempting to enforce legal ownership in code.

## 10. Evaluation plan

### URL/download test matrix

Use mocked downloader boundaries and local fixtures; do not make network access part of ordinary unit tests.

Test:

- accepted `youtube.com/watch` and `youtu.be` URLs;
- malformed IDs, unsupported hosts/schemes, credentials, ports, fragments, playlists, channels, search URLs, `youtube.com/live`, unresolved `@url:` placeholders, and arbitrary URLs;
- public finite video metadata;
- unavailable, private, deleted, live/upcoming, protected, age/login-required, rate-limited, and unsupported-format errors;
- downloader missing, non-executable, timeout, nonzero exit, malformed metadata, and sanitized stderr;
- output path traversal, symlink escape, source/output collision, staging cleanup, and atomic rename;
- estimated-size and observed-size limits;
- duration/frame/disk limits;
- deterministic sanitized provenance with no URL/path/cookie leakage.

### Automatic analysis test matrix

- deterministic pixel-space detector fixtures for golfer, anchor, clubhead, ball, and explicit nulls;
- tracker association, bounded gaps, camera-motion flagging, multiple golfers, blur, occlusion, and low FPS;
- fixed-camera acceptance and moving-camera rejection;
- calibration dimension mismatch, degenerate homography, reprojection error, and exactly-once mapping;
- one `run_pipeline()` call on accepted automatic evidence;
- no pipeline call when evidence/calibration gates fail;
- human fallback handoff without overwriting automatic provenance;
- deterministic reports and relative artifact references;
- valid annotated frames and optional MP4 verified with FFprobe.

### Real-footage protocol

Create a consented, locally retained, versioned dataset of real user-provided YouTube URLs or downloaded clips. For evaluation reproducibility, retain the permitted local media and a sanitized source-ID manifest rather than relying on URLs remaining available.

Label per clip:

- golfer box and feet anchor;
- club/clubhead visibility and track;
- ball before/after impact;
- address/top/contact frame interval;
- visible flight and landing/roll only where supported;
- four calibration points and course coordinates;
- camera type/motion, resolution, FPS, blur, occlusion, number of golfers, and view category.

Split by golfer, channel/source, camera, location, and clip—not random frames.

Measure:

- URL/parser acceptance and false acceptance rate;
- download success/failure category rates and budget enforcement;
- detector precision/recall, IoU, anchor pixel error;
- track IDF1/HOTA or equivalent, ID switches, and gap behavior;
- ball pixel error and pre/post-impact recall;
- clubhead recall and endpoint/track error;
- contact-frame error within ±1/±2 frames and phase F1;
- visible-flight trajectory RMSE and landing error only when labeled visible;
- calibration reprojection error and course-coordinate error;
- valid-`ShotEvent` rate, correct-withhold rate, false landing/recommendation rate;
- runtime per video second, peak RAM/VRAM, disk use, and failure distribution.

## 11. Milestones and acceptance criteria

### Y0 — URL and downloader boundary

- Add strict URL parser and sanitized provenance record.
- Add downloader interface with mocked tests.
- Add optional `yt-dlp` executable discovery/version reporting.
- Implement manual-download fallback when unavailable.

Acceptance: URL/path/security tests pass; no network is needed for unit tests; no credentials/cookies are read.

### Y1 — Safe finite YouTube download

- Implement bounded metadata probe and one-video download.
- Enforce live/private/protected/unavailable rejection and size/duration/disk limits.
- Verify output through `validate_video_source()` and `inspect_video()`.
- Add sanitized `source_metadata.json`.

Acceptance: mocked download matrix passes; real public download smoke test is opt-in and documented; downloaded media stays within the boundary.

### Y2 — Route to automatic golf perception

- Wire the YouTube local file into the automatic perception command without duplicating analysis logic.
- Add fixed-camera/view-quality gate and explicit unavailable behavior.
- Preserve automatic observation provenance and human fallback.

Acceptance: one local downloaded fixture reaches the same automatic pipeline as a local file; no model upload; moving/multiple-golfer/low-resolution cases are withheld.

### Y3 — Calibration and unchanged analytics

- Require or validate supplied clip-specific calibration.
- Add automatic observation-to-`ShotEvent` adapter with exactly-once mapping.
- Call unchanged `run_pipeline()` exactly once only after gates pass.

Acceptance: calibration, mapping, pipeline-call, and no-fabrication tests pass; all existing commands and the 219+ suite remain green.

### Y4 — Outputs and re-projection

- Generate diagnostics, normalized shot, recommendation, SVG, annotated frames, and optional annotated MP4.
- Keep raw-pixel detections/trails separate from course-space recommendation overlays.
- Verify artifact paths, JSON schemas, FFmpeg readability, and visual alignment.

Acceptance: a fixed-camera real golf clip produces a viewable annotated result with honest confidence/provenance; unsupported clips produce a useful fallback package without recommendation.

### Y5 — Real-footage release gate

- Run the evaluation protocol on held-out real footage.
- Publish metrics, runtime/hardware, failure categories, and unsupported-view rates.
- Keep the feature experimental until the real-footage gate passes.

Proposed initial gates, subject to baseline measurement:

- zero accepted analytics outputs with invalid schema, unsafe paths, or duplicate coordinate mapping;
- >=95% correct withholding of clips with missing required evidence or unusable calibration;
- false landing claims <=5% on clips where landing is not visible;
- contact within ±2 frames on the fixed-camera high-FPS subset at the agreed target rate;
- ball/club/golfer metrics reported separately by visibility condition rather than averaged into a misleading single score;
- no regressions in all existing commands and the complete test suite.

These are release criteria to validate, not claims about current capability.

## 12. Files likely to change during implementation

Do not change these files during this design-only phase. Likely implementation files are:

- Create: `ghostcaddie/video/youtube_urls.py`
- Create: `ghostcaddie/video/youtube_downloader.py`
- Create: `ghostcaddie/video/youtube_orchestration.py` or extend a provider-neutral orchestration module with a separate entry point
- Create: automatic perception modules from the approved Automatic Golf Video Perception plan
- Modify: `ghostcaddie/cli.py` only to register/dispatch `youtube-analyze`
- Modify: `ghostcaddie/video/__init__.py` only for reviewed public exports
- Add: `tests/test_youtube_urls.py`
- Add: `tests/test_youtube_downloader.py`
- Add: `tests/test_youtube_orchestration.py`
- Add: integration/fixture tests for sanitized artifacts, fallback, calibration, and exactly-once pipeline mapping
- Update: documentation for optional downloader installation, privacy, legal/user responsibility, model setup, and real-footage limitations

Protected from modification: standard-library analytics core, wind/dispersion/hazard math, `run_pipeline()` implementation, fixture behavior, existing human fallback behavior, provider/session commands, and existing output schemas unless a versioned additive contract is approved.

## Sources

[1] [yt-dlp repository and README](https://github.com/yt-dlp/yt-dlp)

[2] [yt-dlp supported sites documentation](https://github.com/yt-dlp/yt-dlp/blob/2026.03.03/supportedsites.md)

[3] [GhostBall-Engine README and architecture](https://raw.githubusercontent.com/footballanalystrohan-glitch/GhostBall-Engine/main/README.md)
