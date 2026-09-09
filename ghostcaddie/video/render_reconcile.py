"""Truthful render/reconciliation helpers (pure functions, tested).

Separates raw tracker output from the AI-reviewed display layer so that
post-review suppression never masquerades as an automatic detector gain, and
so trails are cleared across unavailable/rejected/unclear frames.

- ``SourceInterval`` / frame mapping: a tracker's local ``frame_index`` maps to
  a source index as ``source_start + frame_index``.
- ``classify_display(raw_state, ai_verdict)`` -> one of
  ``on_clubhead``/``rejected``/``unclear``/``unavailable``/``unreviewed``.
- ``build_confirmed_segments(frames, state_fn)`` yields the maximal runs of
  CONSECUTIVE on-clubhead frames (a gap or rejected/unclear breaks the segment),
  so a renderer never bridges a gap.
- ``reconcile_counts(observed_source_frames, state_fn, total_interval)`` gives the
  seed / predictive / visually-supported / rejected / unclear / unreviewed /
  unavailable counts.
"""
from __future__ import annotations
from typing import Callable, Iterable, List, Sequence, Tuple

DISPLAY_STATES = (
    "on_clubhead", "rejected", "unclear", "unavailable", "unreviewed",
)

# AI verdicts -> display classification
#   on_clubhead  -> confirmed green
#   on_body / on_shaft -> rejected (off-clubhead) -> red X, NOT green
#   unclear / (anything else) -> unclear
#   unreviewed (no AI verdict) -> unreviewed (never claimed green)
_REJECT = {"on_body", "on_shaft"}
_UNSURE = {"unclear", "unreviewed"}


def source_index_for_frame(source_start: int, frame_index: int) -> int:
    """Map a tracker-local frame_index to the source frame index."""
    if not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError("frame_index must be a non-negative integer")
    return source_start + frame_index


def classify_display(raw_state: str, ai_verdict: str | None) -> str:
    """Classify how a tracker observation should be DISPLAYED after AI review.

    ``raw_state`` is the raw tracker state (``observed`` or ``unavailable``).
    ``ai_verdict`` is the AI-assisted review verdict for an observed frame.
    A raw *unavailable* is always ``unavailable``. An observed frame that was
    never AI-reviewed is ``unreviewed`` (never shown green). Otherwise the
    verdict decides. This layer is AI-REVIEWED DISPLAY, never automatic detection.
    """
    if raw_state != "observed":
        return "unavailable"
    if ai_verdict is None or ai_verdict not in ("on_clubhead", *tuple(_REJECT), "unclear", "unreviewed"):
        return "unreviewed"
    if ai_verdict == "on_clubhead":
        return "on_clubhead"
    if ai_verdict in _REJECT:
        return "rejected"
    return "unclear"


def build_confirmed_segments(
    source_frames: Sequence[int],
    state_at: Callable[[int], str],
) -> List[Tuple[int, int]]:
    """Return maximal [start, end] runs of consecutive ``on_clubhead`` frames.

    Any frame whose display state is not ``on_clubhead`` (unavailable, rejected,
    unclear, unreviewed) breaks the run, so a renderer never draws a green trail
    across a gap or a suppressed point.
    """
    segments: List[Tuple[int, int]] = []
    start = None
    last = None
    for f in source_frames:
        if state_at(f) == "on_clubhead":
            if start is None:
                start = f
            last = f
        else:
            if start is not None and last is not None:
                segments.append((start, last))
                start = None
                last = None
    if start is not None and last is not None:
        segments.append((start, last))
    return segments


def reconcile_counts(
    interval: Sequence[int],
    raw_state_at: Callable[[int], str],
    ai_verdict_at: Callable[[int], str | None],
    seed_source: int,
) -> dict:
    """Compute truthful evidence counts over a frozen source interval.

    Returns a dict with seed / predictive / visually_supported / rejected /
    unclear / unreviewed / unavailable / total_observed.
    """
    displayed = {f: classify_display(raw_state_at(f), ai_verdict_at(f)) for f in interval}
    seed = 1 if displayed.get(seed_source) in ("on_clubhead", "unreviewed") else 0
    observed = [f for f in interval if raw_state_at(f) == "observed"]
    predictive = [f for f in observed if f != seed_source]
    vis_supported = [f for f in observed if displayed[f] == "on_clubhead"]
    rejected = [f for f in observed if displayed[f] == "rejected"]
    unclear = [f for f in observed if displayed[f] == "unclear"]
    unreviewed = [f for f in observed if displayed[f] == "unreviewed"]
    unavailable = [f for f in interval if displayed[f] == "unavailable"]
    return {
        "seed": seed,
        "predictive": len(predictive),
        "visually_supported": len(vis_supported),
        "rejected": len(rejected),
        "unclear": len(unclear),
        "unreviewed": len(unreviewed),
        "unavailable": len(unavailable),
        "total_observed": len(observed),
    }
