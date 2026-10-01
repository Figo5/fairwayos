"""Scene candidate discovery: where this clip cuts, and how sure the detector is.

**These are suggestions.** A candidate never becomes a scene window on its own: the
operator accepts it (by splitting their own windows there) or rejects it, and the
decision is persisted in the config so re-running discovery cannot resurrect it.

Method, stated plainly
----------------------
The per-frame score comes from ffmpeg's own ``scdet`` filter, run over the decoded
frames: ``scdet`` reports a mean-absolute-frame-difference and a scene score per
frame, which is the same quantity a hand-written HSV-histogram-difference detector
approximates -- but computed by the decoder that already has the frames, with no new
dependency and one implementation instead of two.

* **hard cut** - a single frame whose score stands well above its neighbours and above
  ``cut_threshold``.
* **dissolve** - a *run* of consecutive frames with a modest but sustained score. A
  blend has width: reporting it as two hard cuts, or missing it, would tell the
  operator the wrong thing about where one camera ends.

The detector is deterministic (same clip, same parameters, same JSON) and local.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .broadcast_config import CutCandidate
from .errors import DemoSourceError

SCENE_SCORE_RE = re.compile(r"^lavfi\.scd\.(score|mafd|time)=(.*)$")
FRAME_RE = re.compile(r"^frame:(\d+)")
#: A frame must beat this to be called a hard cut at all.
CUT_THRESHOLD = 0.22
#: A run of frames above this (but below the cut threshold) is a dissolve candidate.
DISSOLVE_THRESHOLD = 0.045
DISSOLVE_MIN_RUN = 3
#: Candidates closer together than this collapse into one (the strongest wins): a cut
#: is one boundary, not a cluster.
MIN_GAP_FRAMES = 3
SAMPLE_STRIDE = 4

FFMPEG_TIMEOUT_SECONDS = 1800


@dataclass
class SceneScores:
    frames: List[int] = field(default_factory=list)
    score: List[float] = field(default_factory=list)
    mafd: List[float] = field(default_factory=list)
    method: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_dict(self, *, summary: bool = True) -> dict:
        out = {"method": self.method, "parameters": self.parameters, "note": self.note,
               "frames_scored": len(self.frames)}
        if not summary:
            out["per_frame"] = [{"frame": f, "score": round(s, 6),
                                 "mafd": round(m, 6)}
                                for f, s, m in zip(self.frames, self.score, self.mafd)]
        return out


def read_scene_scores(clip: str, *, ffmpeg: str = "ffmpeg",
                      timeout: int = FFMPEG_TIMEOUT_SECONDS) -> SceneScores:
    """Per-frame scene scores for a local clip. Raises on any surprise."""
    if not os.path.isfile(clip):
        raise DemoSourceError(f"source video does not exist or is not a file: {clip}")
    cmd = [ffmpeg, "-v", "error", "-i", clip, "-vf",
           "scdet=threshold=0,metadata=mode=print:file=-", "-an", "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise DemoSourceError(
            "ffmpeg was not found on PATH; scene discovery needs it to decode the "
            "clip") from exc
    except subprocess.TimeoutExpired as exc:
        raise DemoSourceError(
            f"scene discovery timed out after {timeout}s on {os.path.basename(clip)}"
        ) from exc
    if proc.returncode != 0:
        raise DemoSourceError(
            f"ffmpeg could not score {os.path.basename(clip)}: "
            f"{(proc.stderr or '').strip()[:300]}")
    scores = SceneScores(
        method="ffmpeg scdet (mean absolute frame difference + scene score) over "
               "decoded frames",
        parameters={"cut_threshold": CUT_THRESHOLD,
                    "dissolve_threshold": DISSOLVE_THRESHOLD,
                    "dissolve_min_run": DISSOLVE_MIN_RUN,
                    "min_gap_frames": MIN_GAP_FRAMES},
        note="scores are relative to this clip's own frames; they are not comparable "
             "across clips and they are not a statement about content")
    current: Dict[str, float] = {}
    pending_frame: Optional[int] = None
    for line in (proc.stdout or "").splitlines():
        match = FRAME_RE.match(line)
        if match:
            if pending_frame is not None and "score" in current:
                scores.frames.append(pending_frame)
                scores.score.append(float(current.get("score", 0.0)))
                scores.mafd.append(float(current.get("mafd", 0.0)))
            pending_frame = int(match.group(1))
            current = {}
            continue
        key = SCENE_SCORE_RE.match(line)
        if key:
            try:
                current[key.group(1)] = float(key.group(2))
            except ValueError:
                continue
    if pending_frame is not None and "score" in current:
        scores.frames.append(pending_frame)
        scores.score.append(float(current.get("score", 0.0)))
        scores.mafd.append(float(current.get("mafd", 0.0)))
    if not scores.frames:
        raise DemoSourceError(
            f"no scene scores came back for {os.path.basename(clip)}; the clip may be "
            f"undecodable or hold a single frame")
    return scores


def _local_peak(index: int, values: Sequence[float]) -> bool:
    """A frame is a peak when it is the largest within SAMPLE_STRIDE frames."""
    lo = max(0, index - SAMPLE_STRIDE)
    hi = min(len(values), index + SAMPLE_STRIDE + 1)
    best = max(values[lo:hi])
    return values[index] >= best


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


#: A hard cut is a SPIKE, not a busy stretch: the frame after a cut normally looks like
#: the cut frame, whereas fast motion keeps producing change. These two ratios are what
#: separate "the camera changed" from "the golfer swung".
SPIKE_RATIO_NEXT = 2.5
SPIKE_RATIO_PREV = 2.5
SPIKE_RATIO_BACKGROUND = 6.0


def detect_cuts(scores: SceneScores, *, cut_threshold: float = CUT_THRESHOLD,
                dissolve_threshold: float = DISSOLVE_THRESHOLD,
                dissolve_min_run: int = DISSOLVE_MIN_RUN,
                min_gap: int = MIN_GAP_FRAMES) -> List[CutCandidate]:
    """Turn per-frame scores into hard-cut and dissolve candidates, in frame order.

    A hard cut has to be an isolated spike: above ``cut_threshold``, the largest score
    in its neighbourhood, several times the clip's own background score, and well above
    the frame that follows it. Without those conditions a fast downswing reports as a
    cut, and a list of twenty cuts in one swing is worse than no list at all.
    """
    values = scores.score
    frames = scores.frames
    background = max(_median(values), 1e-9)
    candidates: List[CutCandidate] = []
    for index, value in enumerate(values):
        if value < cut_threshold or not _local_peak(index, values):
            continue
        following = values[index + 1] if index + 1 < len(values) else 0.0
        previous = values[index - 1] if index > 0 else 0.0
        # A cut has a quiet frame before it AND after it. Requiring only the "after"
        # half reports the trailing edge of a swing as a camera change, which is the
        # single most confusing thing this list could say.
        if following > 0 and value < SPIKE_RATIO_NEXT * following:
            continue
        if previous > 0 and value < SPIKE_RATIO_PREV * previous:
            continue
        if value < SPIKE_RATIO_BACKGROUND * background:
            continue
        # Confidence is monotone in the score and nowhere near saturated: a 43.0 spike
        # and a 0.8 wobble must not both read as 1.0.
        confidence = min(1.0, (value / max(background, 1e-9)) / 60.0)
        candidates.append(CutCandidate(
            frame=int(frames[index]), kind="hard_cut",
            confidence=round(max(0.05, confidence), 4),
            method=("ffmpeg scdet scene score, local maximum above "
                    f"{cut_threshold:g} and an isolated spike"),
            note=(f"score {value:.4f}, next frame {following:.4f}, "
                  f"clip background {background:.4f}"),
            decision="proposed"))

    # --- dissolves: a RUN of modest, sustained change
    run_start: Optional[int] = None
    index = 0
    while index <= len(values):
        value = values[index] if index < len(values) else 0.0
        in_run = (dissolve_threshold <= value < cut_threshold)
        if in_run and run_start is None:
            run_start = index
        elif not in_run and run_start is not None:
            length = index - run_start
            if length >= dissolve_min_run:
                window = values[run_start:index]
                peak = max(window)
                candidates.append(CutCandidate(
                    frame=int(frames[run_start]), kind="dissolve",
                    span=[int(frames[run_start]), int(frames[index - 1])],
                    confidence=round(min(1.0, (peak / cut_threshold) * 0.8), 4),
                    method=f"ffmpeg scdet: {length} consecutive frames of sustained "
                           f"change between {dissolve_threshold:g} and "
                           f"{cut_threshold:g}",
                    note=f"mean score {sum(window) / len(window):.4f}, peak {peak:.4f}",
                    decision="proposed"))
            run_start = None
        index += 1

    candidates.sort(key=lambda c: (c.frame, 0 if c.kind == "hard_cut" else 1))
    merged: List[CutCandidate] = []
    for candidate in candidates:
        if merged and candidate.frame - merged[-1].frame < min_gap:
            if candidate.confidence > merged[-1].confidence:
                merged[-1] = candidate
            continue
        merged.append(candidate)
    return merged


def merge_decisions(candidates: Sequence[CutCandidate],
                    existing: Sequence[Any]) -> List[CutCandidate]:
    """Re-running discovery must not resurrect a decision the operator already made.

    A saved decision is matched to a candidate by frame within
    :data:`MIN_GAP_FRAMES`, because a slightly different score can move a detected
    boundary by a frame or two and that is not a different cut.
    """
    saved: List[tuple] = []
    for item in existing:
        if isinstance(item, CutCandidate):
            saved.append((int(item.frame), item.decision, item.decided_by,
                          item.decided_at, item.note))
        elif isinstance(item, Mapping):
            saved.append((int(item.get("frame") or 0),
                          str(item.get("decision") or "proposed"),
                          str(item.get("decided_by") or ""),
                          str(item.get("decided_at") or ""),
                          str(item.get("note") or "")))
    out: List[CutCandidate] = []
    for candidate in candidates:
        for frame, decision, by, at, note in saved:
            if abs(candidate.frame - frame) <= MIN_GAP_FRAMES:
                candidate.decision = decision
                candidate.decided_by = by
                candidate.decided_at = at
                if note:
                    candidate.note = f"{candidate.note}; operator: {note}"
                break
        out.append(candidate)
    return out


def discover(clip: str, *, ffmpeg: str = "ffmpeg",
             existing: Sequence[Any] = ()) -> dict:
    """The full discovery record for one clip, ready to write as JSON."""
    from .probe import probe_source

    probe = probe_source(clip)
    scores = read_scene_scores(clip, ffmpeg=ffmpeg)
    candidates = merge_decisions(detect_cuts(scores), existing)
    return {
        "schema_version": "fairwayos-broadcast-scenes.v1",
        "clip": clip,
        "clip_name": os.path.basename(clip),
        "clip_sha256": _sha256(clip),
        "width": probe.width, "height": probe.height,
        "fps_rational": str((probe.raw_stream or {}).get("r_frame_rate") or ""),
        "frame_count": probe.frame_count,
        "note": ("These are SUGGESTIONS with a confidence. Nothing here moves a scene "
                 "window by itself: the operator accepts a candidate by splitting "
                 "their own window at it, and the decision is persisted so a re-run "
                 "cannot resurrect it."),
        "scores": scores.to_dict(),
        "candidates": [c.to_dict() for c in candidates],
        "hard_cuts": [c.to_dict() for c in candidates if c.kind == "hard_cut"],
        "dissolves": [c.to_dict() for c in candidates if c.kind == "dissolve"],
    }


def _sha256(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
