"""PGA-only source / demo-eligibility gate.

Durable requirement (Gio via Hermes, 2026-09-11 "PGA-ONLY RESET"): FairwayOS
works with PGA TOUR footage only. Amateur, stock, brand-commercial and unknown
footage must never become the product demo. The rejected Pexels experiment is
preserved here as an explicit REJECTED entry so the rejection is durable.

Design rules, in order of importance:

1. FAIL CLOSED. The default answer for anything not in the reviewed registry is
   REVIEW_REQUIRED. Nothing is ever promoted to ELIGIBLE implicitly.
2. NO CLASSIFIER. There is deliberately no pixel or filename heuristic that can
   mark a source PGA. An unknown source is a review task for a human, not a
   detection problem. Filenames are evidence of nothing.
3. CONTENT-BOUND. Decisions key on the file's SHA-256, so renaming, copying or
   re-foldering a clip cannot change its status.
4. TWO TIERS. `local_research_allowed` (may be used for local, ignored
   stress-test work) is separate from `demo_eligible` (may appear in a review
   MP4). Lacking publication rights does not by itself forbid local research,
   and local research never implies publication clearance.

Every entry records the evidence that produced its status, including which
frames were actually inspected. "I looked at the filename" is not evidence.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

ELIGIBLE = "eligible_pga"
REJECTED = "rejected"
REVIEW_REQUIRED = "review_required"


class SourceNotEligible(RuntimeError):
    """Raised when a non-eligible source is offered to a demo/render path."""


# ---------------------------------------------------------------------------
# Reviewed registry. Keys are SHA-256 of the exact local file.
# Statuses were assigned by inspecting ACTUAL FRAMES (see frames_inspected),
# not filenames or directory names.
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, dict] = {
    # ---- REJECTED: amateur stock footage. The 2026-09-10 demo built on this
    # clip was rejected by Gio ("random kid chipping is NOT the goal").
    # Preserved so the rejection is durable and cannot drift back in.
    "a6e48474045365d1de2d4af76f65da558531684d67da87172cdd15a6dc45e1d6": {
        "label": "Pexels 6573485 - amateur junior golfer, stock footage",
        "status": REJECTED,
        "reason": "Amateur stock footage. Not PGA TOUR. Rejected by the user on "
                  "2026-09-10 as the product demo; retained only as rejected history.",
        "source_url": "https://www.pexels.com/video/boy-hitting-a-golf-ball-6573485/",
        "event_evidence": None,
        "player_evidence": None,
        "rights": "Pexels source page marked free to use; still NOT PGA and not a "
                  "product demo source. No redistribution as FairwayOS output.",
        "frames_inspected": "full 121-sample render reviewed 2026-09-10",
        "local_research_allowed": False,
        "public_redistribution_cleared": False,
    },
    # ---- REVIEW REQUIRED: genuine PGA TOUR players, unverified source.
    "d3311ea7470b027e43611fc1251313aa917dcc66f35462b27bf486cdd228b4f0": {
        "label": "YouTube f3kTTMZlxds - DUAL-PANEL side-by-side countdown edit",
        "status": REVIEW_REQUIRED,
        "reason": "Frames show 10 genuine PGA TOUR players via burned-in captions "
                  "(Zalatoris, Spieth, Fitzpatrick, Homa, C. Smith, Schauffele, "
                  "Cantlay, McIlroy, Scheffler, Rahm). BUT structural inspection "
                  "on 2026-09-11 found it is a DUAL-PANEL edit: two 640x720 "
                  "portrait panels side by side with a hard black divider at "
                  "x~638, showing two DIFFERENT players simultaneously (frame 200 "
                  "= Zalatoris left / Spieth right; frame 3400 = Scheffler left / "
                  "McIlroy right), with burned-in ranking captions and a panel "
                  "change roughly every 455 frames. That is a social-media "
                  "comparison edit, not broadcast footage, and it is the opposite "
                  "of the 'dominant uncluttered broadcast footage' the demo "
                  "requires. NO source URL was ever recorded (download.py names "
                  "6n3nFFiS5sE, a different id). Real players on screen is not "
                  "source clearance.",
        "source_url": "https://www.youtube.com/watch?v=f3kTTMZlxds",
        "source_metadata": "oEmbed 2026-09-11: title 'Fantastic Driver Slow "
                           "Motion Swings of World Top 10', channel 'King of Golf' "
                           "(@King_of_Golf) -- a third-party aggregator, NOT PGA TOUR.",
        "event_evidence": None,
        "player_evidence": "Burned-in captions naming 10 PGA TOUR players, "
                           "visually confirmed in extracted frames 2026-09-11",
        "rights": "Third-party re-upload, origin unverified. Local research only; "
                  "no redistribution, no public demo.",
        "frames_inspected": "8-frame contact strip + per-segment caption crops + "
                            "column-variance seam analysis at frame 400, 2026-09-11",
        "local_research_allowed": True,
        "public_redistribution_cleared": False,
    },
    # ---- REVIEW REQUIRED: tournament evidence, player unverified.
    "98f8bb71bff708f8a98a119d5776bdc4bf8e0bcb033a24528121fdca07091ce5": {
        "label": "YouTube zFXYQ8jQbO4 window - Franklin Templeton Shootout tee shot",
        "status": REVIEW_REQUIRED,
        "reason": "Frames show real tournament signage (Franklin Templeton "
                  "Shootout board, GEICO sponsor, tee marker 7) and a continuous "
                  "single-camera broadcast-style shot. BUT the player is NOT "
                  "identifiable from the frames (the directory name claims Zach "
                  "Johnson; directory names are not evidence), the full source "
                  "file is no longer on disk (only this 106-frame window), and "
                  "the Shootout is an unofficial-money event. Duration is far "
                  "below the 20-45s review-MP4 requirement.",
        "source_url": "https://www.youtube.com/watch?v=zFXYQ8jQbO4",
        "event_evidence": "Franklin Templeton Shootout + GEICO signage and tee "
                          "marker 7 visible in frames, confirmed 2026-09-11",
        "player_evidence": "oEmbed 2026-09-11: title names ZACH JOHNSON; channel "
                           "'GolfswingHD'. This is an UPLOADER CLAIM, not "
                           "independent identification, and the channel is a "
                           "third-party swing-footage channel, not PGA TOUR.",
        "rights": "YouTube third-party footage; local research only, no "
                  "redistribution (as already recorded in the prior provenance).",
        "frames_inspected": "5-frame contact strip of the 106-frame window, 2026-09-11",
        "local_research_allowed": True,
        "public_redistribution_cleared": False,
    },
    # ---- REJECTED: brand commercial, no PGA evidence.
    "e82638dd0a99a4b27c1f7ee0ce7ddec9580044a16b4c164e49a771698158cc4e": {
        "label": "YouTube U3AKUznK9us - Titleist brand film",
        "status": REJECTED,
        "reason": "Titleist-watermarked brand/commercial content shot on a beach. "
                  "CORRECTION 2026-09-11: the player IS identified by public "
                  "metadata as Adam Scott, a PGA TOUR player -- my earlier "
                  "'unidentified player' reason was wrong on that point. The "
                  "verdict is unchanged for the correct reason: this is "
                  "brand-owned commercial content, not PGA TOUR broadcast "
                  "footage, and Titleist holds the rights.",
        "source_metadata": "oEmbed 2026-09-11: 'Adam Scott golf swing in slow "
                           "motion 4K', channel 'Titleist' (official brand).",
        "source_url": "https://www.youtube.com/watch?v=U3AKUznK9us",
        "event_evidence": None,
        "player_evidence": None,
        "rights": "Brand-owned commercial footage; no redistribution rights.",
        "frames_inspected": "8-frame contact strip, 2026-09-11",
        "local_research_allowed": False,
        "public_redistribution_cleared": False,
    },
    "1f3109f3ba4433d595c591f18d67dd1f3a3fd64a900f8823f811029b12c7c865": {
        "label": "YouTube YZZOQXXmnTs window - 'Golf Swing HD' channel",
        "status": REJECTED,
        "reason": "Third-party aggregator channel ('GolfswingHD'). CORRECTION "
                  "2026-09-11: the player IS identified by public metadata as "
                  "Justin Thomas, a PGA TOUR player -- my earlier 'unidentified "
                  "player' reason was wrong. Verdict unchanged for the correct "
                  "reason: third-party re-upload of practice-area footage, not "
                  "PGA TOUR broadcast, no redistribution rights.",
        "source_url": "https://www.youtube.com/watch?v=YZZOQXXmnTs",
        "source_metadata": "oEmbed 2026-09-11: 'JUSTIN THOMAS 120fps SLOW MOTION "
                           "DTL GOLF SWING FOOTAGE 1080 HD', channel 'GolfswingHD'.",
        "event_evidence": None,
        "player_evidence": None,
        "rights": "Third-party re-upload; no redistribution rights.",
        "frames_inspected": "3-frame contact strip, 2026-09-11",
        "local_research_allowed": False,
        "public_redistribution_cleared": False,
    },
    "d4e6bc391b13ba30c554e539c70fb583ff78c2429d6d701db7c569fa7e37ec0c": {
        "label": "YouTube rDDuYNSFXF8 window - unidentified player",
        "status": REJECTED,
        "reason": "CORRECTION 2026-09-11: the player IS identified by public "
                  "metadata as Justin Thomas, a PGA TOUR player -- my earlier "
                  "'unidentified player' reason was wrong. Verdict unchanged for "
                  "the correct reason: third-party re-upload on 'GolfswingHD', "
                  "not PGA TOUR broadcast, no redistribution rights.",
        "source_url": "https://www.youtube.com/watch?v=rDDuYNSFXF8",
        "source_metadata": "oEmbed 2026-09-11: 'JUSTIN THOMAS 120fps SLOW MOTION "
                           "DTL IRON GOLF SWING', channel 'GolfswingHD'.",
        "event_evidence": None,
        "player_evidence": None,
        "rights": "Third-party re-upload; no redistribution rights.",
        "frames_inspected": "4-frame contact strip, 2026-09-11",
        "local_research_allowed": False,
        "public_redistribution_cleared": False,
    },
}


def registry() -> Dict[str, dict]:
    """The reviewed registry (copy: callers must not mutate the gate)."""
    return {k: dict(v) for k, v in _REGISTRY.items()}


@dataclass
class Decision:
    status: str
    reason: str
    sha256: Optional[str] = None
    label: Optional[str] = None
    source_url: Optional[str] = None
    event_evidence: Optional[str] = None
    player_evidence: Optional[str] = None
    rights: Optional[str] = None
    frames_inspected: Optional[str] = None
    local_research_allowed: bool = False
    public_redistribution_cleared: bool = False

    @property
    def demo_eligible(self) -> bool:
        # Only an explicitly reviewed ELIGIBLE entry may reach a demo.
        return self.status == ELIGIBLE


def sha256_file(path: str) -> Optional[str]:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def evaluate_sha256(sha: str) -> Decision:
    """Decide on a content hash. Unknown hashes FAIL CLOSED."""
    entry = _REGISTRY.get(sha)
    if entry is None:
        return Decision(
            status=REVIEW_REQUIRED,
            reason="Source is not in the reviewed registry. A human must review "
                   "the actual frames and record event/player/rights evidence "
                   "before it may be used. Unknown is never eligible.",
            sha256=sha,
        )
    return Decision(
        status=entry["status"],
        reason=entry["reason"],
        sha256=sha,
        label=entry.get("label"),
        source_url=entry.get("source_url"),
        event_evidence=entry.get("event_evidence"),
        player_evidence=entry.get("player_evidence"),
        rights=entry.get("rights"),
        frames_inspected=entry.get("frames_inspected"),
        local_research_allowed=bool(entry.get("local_research_allowed", False)),
        public_redistribution_cleared=bool(
            entry.get("public_redistribution_cleared", False)),
    )


def evaluate_source(path: str) -> Decision:
    """Decide on a local file by its CONTENT, never by its name or folder."""
    sha = sha256_file(path)
    if sha is None:
        return Decision(
            status=REVIEW_REQUIRED,
            reason=f"Source is unreadable ({path}); cannot hash, so it cannot be "
                   "reviewed. Fails closed.",
        )
    return evaluate_sha256(sha)


def require_demo_eligible(path: Optional[str] = None,
                          sha256: Optional[str] = None) -> Decision:
    """Gate a demo/render entry point. Raises unless explicitly ELIGIBLE."""
    if path is None and sha256 is None:
        raise SourceNotEligible("no source given; demo eligibility fails closed")
    d = evaluate_sha256(sha256) if sha256 is not None else evaluate_source(path)  # type: ignore[arg-type]
    if not d.demo_eligible:
        raise SourceNotEligible(
            f"PGA-only gate: status={d.status} for {d.label or d.sha256}. "
            f"{d.reason} Demo rendering is blocked. Amateur or unverified "
            f"footage must never be substituted."
        )
    return d
