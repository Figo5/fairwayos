# Furyk local-scale experiment (development evidence)

Source SHA-256: `26ca428cafb75f0a24edd52bab7b7d19540a9c353c02f8b4515012c3544e0c40`.
Native source frames: 500, 510, 520, 530, 540; 1280x720.

Frozen grayscale thresholds 180/200/220 yielded bright ball-component bounding-box widths of 9–11 pixels. This is a brightness-component proxy, not a validated silhouette diameter. Under the explicit assumption that this width represents a 42.67 mm ball, local scale is 0.00387909–0.00474111 metres per pixel. The USGA diameter is a minimum, not the exact measured diameter of this ball.

At a nominal 9-pixel diameter, ±1 pixel yields scale changes of −10% and +12.5%; this excludes other uncertainty sources. The interval above is a threshold-sensitivity range, NOT a confidence interval.

## Demo use

Show `Conditional local scale: 3.9–4.7 mm/px (tee depth)` as experimental context. Pixel motion times this scale can be labeled `projected distance at tee-depth assumption`; it is not actual 3D travel. Conversion to speed additionally needs action-time assumptions. Keep any illustrative playback-to-action multipliers labeled scenarios, not inferred capture rates. Neither this scale nor a smooth image track establishes carry.

Reproduction artifacts remain local under `/tmp/fairway-ball-scale-experiment/`. Media and evaluation ROI are not tracker initialization. No inferred metric is injected into detections.
