# Third-Party Notices and Research Provenance

FairwayOS source code is distributed under the repository `LICENSE`.
Third-party datasets, source footage, implementations, and model weights are
not automatically covered by that license.

## GolfDB / SwingNet

The GolfDB repository is attributed to its authors at:

- https://github.com/wmcnally/golfdb
- https://arxiv.org/abs/1903.06528

The GolfDB repository README identifies its code/data terms as Creative
Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). The
separately downloaded `swingnet_1800.pth.tar` checkpoint has no independently
verified license in this project record and therefore remains research-only
and not cleared for production or redistribution.

GolfDB clips, annotations, hashes, and acquisition details belong in local
ignored evaluation directories unless their applicable terms explicitly permit
publication. This repository contains no downloaded GolfDB videos, annotations,
or model weights.

## Public PGA TOUR / YouTube material

Publicly accessible PGA TOUR or other platform material is restricted to
bounded, local qualitative stress testing under the applicable source and
platform terms. It is not consented footage, production-cleared data, or
validated ground truth. Downloaded media is not committed or redistributed.

## FairwayOS-owned evaluation data

Consent records, ownership documents, private annotations, and source media
for a future FairwayOS-owned evaluation set are private project records and are
not committed to this public repository. Public files may contain only safe
schemas, synthetic fixtures, redacted manifests, hashes, and reproducible
instructions that do not expose personal data or private paths.

## Architecture inspiration (no code copied)

- Ahmed-El-Zainy, `soccer`: https://github.com/Ahmed-El-Zainy/soccer
  - Reviewed README, `main.py`, requirements, and repository tree.
  - The repository-level `LICENSE` path was not found at the reviewed revision;
    therefore no source code, assets, weights, or media were copied or translated.
  - High-level inspiration is limited to frame-wise detection, bounded ball
    slicing, tracker IDs/history, and optional radar rendering.
- footballanalystrohan-glitch, `GhostBall-Engine`:
  https://github.com/footballanalystrohan-glitch/GhostBall-Engine
  - MIT License, copyright notice `Copyright (c) 2026 Remontada-Analyst`.
  - No code or assets were copied. High-level inspiration is limited to the
    pixel-space/analytical-space boundary, homography concept, guarded
    kinematics, and re-projection structure.
