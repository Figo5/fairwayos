# Approximate demo analytics

Status: isolated postprocessor prototype; not yet integrated into the tracker CLI.

## Display contract

- Use the final accepted ball/clubhead observation, never a coarse proposal or rejected attempt.
- Label semantic-only head observations separately. Acceptance is a model decision, not proof of identity.
- Displacement is in pixels between consecutive source frames. Clear velocity after any missing observation or missing source frame.
- Pixel speed is per display second and includes camera motion, detector jitter and perspective. It is not physical ball speed.
- Optional physical output is **projected speed under supplied assumptions**, not launch-monitor speed.

## Approximate physical conversion

Let `d` be consecutive-frame image displacement, `s` metres per pixel near the target depth, `f` playback frames per second, and `r` action-rate multiplier (capture cadence divided by playback cadence for simple slow motion). Then:

`projected_m_per_s = d * s * f * r`

This requires independently justified `s` and `r`. Unique decoded frames do not establish capture cadence. If either is unknown, show pixel motion or explicitly named sensitivity scenarios, not an asserted mph value.

An experimental ball-size scale is `s = assumed_ball_diameter_m / apparent_diameter_px`. The USGA minimum diameter is 42.67 mm, not an exact measurement of the pictured ball. Bright highlight width may differ from the silhouette. Quantization, blur, depth changes and camera zoom affect the estimate. Record the diameter interval and the resulting scale interval rather than hiding uncertainty.

Do not apply a tee-depth scale across the whole swing as though clubhead depth were constant. Do not project an airborne ball through a ground-plane homography and call that its actual position. Carry needs launch/landing geometry or a separately validated flight model; unsupported carry remains unavailable.

## Verification

Preserve source/output hashes and native frame mapping. Check actual encoded video beside clean source, not reconstructed coarse overlays. A panel that agrees with a false marker is numerically consistent but not physically correct. Local scale validation must use a separate known dimension; tee-marker spacing is not standardized calibration data.

Sources:
- https://www.usga.org/equipment-standards/equipment-rules-2019/equipment-rules/part-4-rule-4.html
- https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
