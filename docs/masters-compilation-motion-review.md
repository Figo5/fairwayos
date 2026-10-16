# Masters compilation and motion review

The 16-clip Tiger Woods collection now has a full compilation with the selected FairwayOS overlays, retained broadcast tracers, original clip audio and chapter analytics. The collection is selected highlights, not the complete round. Existing source trims and workflow exports were left intact.

Artifacts are saved at `~/fairway-os-runs/fairwayos-masters-compilation-2026-10-05`:

- `media/fairwayos_tiger_2019_masters_all16.mp4`: 1920×1080, 16 chapters, approximately 4:36.
- `index.html`: chapter player with per-clip coverage, video-motion rates and plots; local preview at `http://127.0.0.1:8780/` while `coordination/serve_preview.py` is running.
- `REPORT.md`, `CLIPS.md`, `ANALYTICS.json`, `analytics.csv`: findings, all individual selected outputs, actual observed frame pairs and summary metrics.
- `INPUT_RECEIPT.json`, `COMPILATION_MANIFEST.json`, `VERIFICATION.json`, `DELIVERY_RECEIPT.json`: source hashes, frame mapping, timing/picture/audio checks and delivery provenance.
- `coordination`: executable preparation/assembly/report/verification scripts, requested model sessions, review disposition and test receipts. Failed/intermediate attempts are identified and preserved.

## Motion API

`ghostcaddie.demo.motion_review.summarize_motion(points, pts_seconds, source_size=(width, height), start_frame=start, end_frame=end, cut_frames=cuts)` is a pure standard-library reducer. Points carry a native source frame, native x/y, origin, status and method. Imported tracking observations retain their existing statuses; saved MoveNet pose estimates use local `STATUS_OBSERVED_POSE`. The caller applies pose confidence/ROI policy; the compilation adapter uses confidence ≥0.5 and discloses the existing operator ROI.

The reducer validates all records and the complete strictly increasing source presentation timeline. It measures only consecutive automatic positions inside the window and excludes cuts, reacquisitions, operator positions, interpolation and gaps. The duration-weighted mean sums only measured pair paths/durations. Medians, p95, the largest observed rate, zero/nonzero step counts and paired coverage remain image measurements. A missing stream has no numeric summary. Equal coordinates do not establish that source frames were repeated.

Units are native pixels per **video** second. Camera pan/zoom, perspective, model identity errors, rounding and encoded slow motion apply. Source height normalization also cannot establish physical speed. An automatic estimate may still depend on an operator seed/crop; its saved provenance is not independent verification.

Physical clubhead/ball speed remains null. No measured speed with legible units was verified in sampled source frames, and no spatial/action-time calibration is available. The API has no flag to enable mph. Club speed is motion just before impact and ball speed just after separation; those quantities need calibrated evidence or a source-bound measurement. See the primary-source links in the artifact report.

Seven full swings and five putts have supported operator phase sets. The adapter recomputes source-timeline tempo from original PTS and binds marks to the copied configuration. Four newer clips have body analytics but no confirmed target tracks or phase sets. Chapter cards show their unavailable values.

Compilation frames retain the original order and source-relative timing within the explicitly documented 1/90000-second output quantization. Every decoded PTS and selected-input hash is verified; image/audio correspondence is sampled and does not certify semantic tracking accuracy. Video uses a new CRF18 encode; audio is resampled to 48k stereo PCM between clips, with silent cards, then encoded AAC once. Existing workflow media remain unchanged.

## Next tracking cycle

Human-review unresolved head/shaft/ball identity around impact and the four newer clips. Refused frames are not automatically negative labels. Reserve separate shot/player/camera conditions before training; reused source-family references are regression evidence, not independent accuracy. Improve impact-window continuity before adding physical-speed claims. Verified mph requires calibrated capture or a source-bound measured value.

DeepSeek V4.1 Flash via OpenCode Go implemented/reviewed the new reducer. Primary checks hardened validation and corrected the separate putt schema. Claude Opus5.5 read part of the collection but hit its weekly limit before completing the review; no Claude approval is claimed.
