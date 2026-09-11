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
    # ---- OFFICIAL PGA TOUR broadcast, acquired by the USER, held locally.
    # Authorization and rights are DIFFERENT THINGS and are recorded separately:
    # user_authorization says Gio told us we may analyse this file locally;
    # public_redistribution_cleared says whether anyone may publish anything made
    # from it. The first is true, the second is UNVERIFIED and therefore false.
    "21c8bb54cd550dec2da680351cfd48f9cc61c24ca2ece43e07e30622b26e28f1": {
        "label": "PGA TOUR official - Scheffler 24-foot birdie putt, No. 4, "
                 "2026 TOUR Championship final round (Brightcove 6404323161112)",
        "status": REVIEW_REQUIRED,
        "reason": "Genuine OFFICIAL PGA TOUR competition footage, verified from "
                  "the Brightcove metadata tags (/pgatour/category/competition, "
                  "/pgatour/tournaments/2026/r060) and from the frames themselves "
                  "(TOUR Championship leaderboard bug, 'SCHEFFLER -13 / 4th HOLE / "
                  "FOR BIRDIE & CO-LEAD' lower third). Acquired by HERMES at the user's"
                  " direction (corrected 2026-09-11; not downloaded by the user personally). Local research is "
                  "authorized; redistribution rights are UNVERIFIED, so it stays "
                  "review_required and is never demo-eligible for publication.",
        "source_url": "https://www.pgatour.com/video/competition/6404323161112/"
                      "scottie-scheffler-sinks-foot-birdie-putt-on-no--at-tour-championship",
        "source_metadata": "yt-dlp brightcove:new, id 6404323161112, 1280x720, "
                           "30000/1001 fps, 587 frames, 19.62 s, full decode passes. "
                           "info.json retained alongside the media.",
        "event_evidence": "2026 TOUR Championship, final round, hole 4; burned-in "
                          "leaderboard and lower-third graphics visible in frames.",
        "player_evidence": "Scottie Scheffler - named in the broadcast lower third "
                           "and in the official PGA TOUR video metadata.",
        "rights": "PGA TOUR owns the footage. PUBLIC REDISTRIBUTION RIGHTS ARE "
                  "UNVERIFIED. Do not publish this clip or any render derived from "
                  "it. Local research only.",
        "user_authorization": "Gio explicitly authorized ordinary yt-dlp acquisition "
                              "for LOCAL ANALYSIS; HERMES performed the download at "
                              "his direction (corrected attribution 2026-09-11). "
                              "Authorization to analyse is NOT a rights "
                              "determination and confers no publication right.",
        "frames_inspected": "16-tile contact sheet over all 587 frames plus a "
                            "10-frame strip across the putt, 2026-09-11.",
        "local_research_allowed": True,
        "public_redistribution_cleared": False,
    },
    # ---- OFFICIAL PGA TOUR APPROACH shot. Tee/approach only per the user's
    # 2026-09-11 direction; putting clips are out of scope for demos.
    "cefbdf25400f5821893747e2b4a60ca5a11f990920ab3a9c8bba32a8c2d3deae": {
        "label": "PGA TOUR official - Si Woo Kim 120-yard approach holed for eagle, "
                 "No. 10, 2026 TOUR Championship final round (Brightcove 6404321996112)",
        "status": REVIEW_REQUIRED,
        "reason": "Official PGA TOUR competition footage (Brightcove tags "
                  "/pgatour/category/competition, /pgatour/tournaments/2026/r060). "
                  "APPROACH shot: setup with a 'TO HOLE: 120 YDS' broadcast graphic, "
                  "visible strike, a flight-follow camera pan, ball clearly visible "
                  "against cloud through the flight, then the green and reaction. "
                  "No broadcaster tracer graphic is present in this clip.",
        "source_url": "https://www.pgatour.com/video/competition/6404321996112/"
                      "si-woo-kim-holes-yard-shot-for-eagle-on-no--at-tour-championship",
        "source_metadata": "yt-dlp brightcove:new, 1280x720, 30000/1001 fps, 667 "
                           "frames, 22.27 s, full decode passes. One hard cut at "
                           "f216; f4-215 is the continuous strike-and-flight shot.",
        "event_evidence": "2026 TOUR Championship final round, hole 10; burned-in "
                          "'S.W. KIM -8 / 10th / 431 YDS / DRIVE 310 YDS / TO HOLE: "
                          "120 YDS' graphic and leaderboard visible in frames.",
        "player_evidence": "Si Woo Kim - named in the broadcast graphic and in the "
                           "official PGA TOUR video metadata.",
        "rights": "PGA TOUR owns the footage. PUBLIC REDISTRIBUTION RIGHTS ARE "
                  "UNVERIFIED. Do not publish this clip or any render from it.",
        "user_authorization": "Gio explicitly authorized ordinary yt-dlp acquisition "
                              "for LOCAL ANALYSIS; HERMES performed the download at "
                              "his direction. Authorization is NOT redistribution "
                              "clearance and is NOT a rights determination.",
        "frames_inspected": "16-tile contact sheet over all 667 frames, plus gridded "
                            "strips across the strike (f76-128) and the flight "
                            "(f140-212), 2026-09-11.",
        "local_research_allowed": True,
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


# ===================== official PGA TOUR candidates (not local) =====================
#
# These are genuinely OFFICIAL PGA TOUR competition videos identified from public
# pgatour.com video metadata (2026-09-11). They are the first real broadcast
# candidates found. They are NOT on this machine and this agent MUST NOT acquire
# them, because the PGA TOUR Terms of Use say, verbatim:
#
#   "You shall not use or permit or facilitate others to use PGATOUR.COM by
#    automated electronic processes, robots, spiders, scrapers, webcrawlers, or
#    other computer programs that monitor, copy or download data or other content
#    found on or accessed through PGATOUR.COM"
#
# An automated agent downloading this video is exactly the prohibited act, so the
# acquisition route for ME is closed. The same terms separately say "You may
# download copyrighted material for Your personal use only" -- a route available
# to a PERSON, not to this process -- and "You may not modify, publish, transmit,
# participate in the transfer or sale, create derivative works, or in any way
# exploit, any of the content", which bears directly on whether a rendered
# analysis overlay may be produced or shown at all.
#
# Viewing is not reuse. Personal-use download is not derivative-work permission.
# Local research permission is not public-demo clearance. Keep the three apart.

PGA_TOU_URL = "https://www.pgatour.com/company/terms-of-use"

OFFICIAL_CANDIDATES: Dict[str, dict] = {
    "6404323161112": {
        "title": "Scottie Scheffler sinks 24-foot birdie putt on No. 4 at TOUR Championship",
        "player": "Scottie Scheffler", "duration_s": 20, "hole": "4", "round": "4",
        "event": "2026 TOUR Championship (tournamentId 060)",
        "share_url": "https://www.pgatour.com/video/competition/6404323161112/"
                     "scottie-scheffler-sinks-foot-birdie-putt-on-no--at-tour-championship",
        "poster_evidence": "1920x1080 poster inspected 2026-09-11: tight follow shot "
                           "of the player from behind with a burned-in leaderboard bug. "
                           "Poster framing is a thumbnail choice, not proof of the "
                           "whole clip's framing.",
    },
    "6404324364112": {
        "title": "Justin Rose sinks 32-foot birdie putt on No. 11 at TOUR Championship",
        "player": "Justin Rose", "duration_s": 23, "hole": "11", "round": "4",
        "event": "2026 TOUR Championship (tournamentId 060)",
        "share_url": "https://www.pgatour.com/video/competition/6404324364112/"
                     "justin-rose-sinks-foot-birdie-putt-on-no--at-tour-championship",
        "poster_evidence": "1920x1080 poster inspected 2026-09-11: down-the-line par-3 "
                           "tee shot, TOUR CHAMPIONSHIP final-round leaderboard, Rolex "
                           "bug, AND A BURNED-IN BROADCAST SHOT TRACER with a "
                           "'BALL SPEED 149 MPH' graphic.",
        "hazard": "The broadcast's OWN tracer and ball-speed graphic are burned into "
                  "the picture. Any analysis of this clip must never present the "
                  "broadcaster's tracer as our detection, and must never read the "
                  "broadcast's ball-speed number back out as a measurement of ours.",
    },
    "6404321996112": {
        "title": "Si Woo Kim holes 120-yard shot for eagle on No. 10 at TOUR Championship",
        "player": "Si Woo Kim", "duration_s": 22, "hole": "10", "round": "4",
        "event": "2026 TOUR Championship (tournamentId 060)",
        "share_url": "https://www.pgatour.com/video/competition/6404321996112/"
                     "si-woo-kim-holes-yard-shot-for-eagle-on-no--at-tour-championship",
        "poster_evidence": "1920x1080 poster inspected 2026-09-11: player and caddie "
                           "walking, leaderboard bug. Poster shows no ball-strike "
                           "moment; the approach shot itself is unverified.",
    },
}

for _c in OFFICIAL_CANDIDATES.values():
    _c.update({
        "official_pga_tour": True,
        "locally_available": False,
        "status": REVIEW_REQUIRED,
        "acquisition_by_this_agent": "PROHIBITED",
        "acquisition_blocker": (
            "PGA TOUR Terms of Use forbid automated download of site content, "
            "including video. This agent is an automated process, so it must not "
            "fetch these clips. " + PGA_TOU_URL),
        "human_route": (
            "The Terms permit a PERSON to 'download copyrighted material for Your "
            "personal use only'. That covers local viewing. It does NOT grant "
            "derivative-work or redistribution rights, which the same Terms "
            "restrict explicitly."),
        "rights": "PGA TOUR owns the footage. No redistribution. Archive licensing "
                  "runs through T3Media (sales@t3media.com).",
        "public_redistribution_cleared": False,
        "demo_eligible": False,
    })


def official_candidates() -> Dict[str, dict]:
    """Official PGA TOUR candidates and why this agent cannot acquire them."""
    return {k: dict(v) for k, v in OFFICIAL_CANDIDATES.items()}
