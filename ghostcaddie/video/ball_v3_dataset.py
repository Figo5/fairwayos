"""Building listwise training examples: the true ball against its own frame's clutter.

The v2 detector was trained on crops in isolation -- a ball crop here, a
background crop there -- and its negatives were patches sampled 70 to 90 px away
from the insertion, which on turf means *grass*. A model trained that way learns
"ball versus grass", which is not the question. The question is "ball versus the
shoe, the tee, the clubhead, the range ball and the shirt highlight that are in
this same frame", and that comparison never appeared in a v2 training batch.

So an example here is a **whole frame's candidate list**, built like this:

1. take a window of consecutive frames from a *training* clip;
2. insert one synthetic ball through it with ``ball_synthetic.synthesize_sequence``,
   which gives exact per-frame position, radius, visibility, blur and occlusion;
3. run the **real v3 proposal stage on the modified frames**, so the candidate
   list contains the synthetic ball *and* every genuine distractor the generators
   find in that real footage;
4. record which candidate is the ball, and keep a sample of the competitors,
   preferring the ones that scored highest -- those are the hard negatives, and
   they are hard because they came from the same photograph, not because they
   were chosen to be.

Frames where the ball was inserted but no generator proposed it are recorded and
**excluded from the ranking loss**: teaching the model to answer "none of these"
on a frame whose ball it was simply never shown would train it to abstain on hard
frames, which is the opposite of what is wanted. They are counted instead, because
that count is the proposal stage's recall and the ceiling on everything else.

Frames with no ball at all -- before entry, after exit, under full occlusion --
are kept as genuine null examples, because the null slot has to be trained on
something real.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np

from . import ball_v3_context as bctx
from . import ball_v3_proposals as bp

#: Candidates kept per frame. One list is one softmax, so this is the number of
#: competitors the true ball is asked to beat in a single gradient step.
LIST_SIZE = 12
#: Of the kept competitors, this fraction are the highest-scoring proposals (the
#: hard ones); the rest are sampled across the list so the model also sees what
#: easy looks like and the score distribution is not truncated.
HARD_FRACTION = 0.7


@dataclass
class FrameExample:
    """One frame's candidate list, with the answer."""

    clip_id: str
    frame_index: int
    #: (N, 64, 64, 7) uint8
    stacks: np.ndarray
    #: (N, numeric) float32
    numeric: np.ndarray
    #: index of the true ball within the list, or None for "no ball here"
    target: Optional[int]
    #: exact synthetic geometry, for centre-error reporting
    truth: Optional[tuple[float, float, float]]
    #: the proposal-stage outcome for this frame, which is the ceiling
    proposed: bool
    width: int
    height: int


def choose_competitors(count: int, exclude: int, keep: int,
                       rng: np.random.Generator) -> list[int]:
    """Indices of the competitors to keep: mostly the strongest, some sampled.

    ``proposals`` arrive sorted strongest-first, so "the strongest" is a prefix.
    """
    available = [i for i in range(count) if i != exclude]
    if len(available) <= keep:
        return available
    hard = int(round(keep * HARD_FRACTION))
    chosen = available[:hard]
    remainder = available[hard:]
    if remainder and keep > hard:
        extra = rng.choice(len(remainder), size=min(keep - hard, len(remainder)),
                           replace=False)
        chosen = chosen + [remainder[int(i)] for i in extra]
    return chosen


def examples_from_sequence(frames: Sequence[np.ndarray], samples: Sequence[Any],
                           *, clip_id: str, frame_indices: Sequence[int],
                           params: bp.ProposalParams, pose_for,
                           numeric_fn, rng: np.random.Generator,
                           list_size: int = LIST_SIZE,
                           methods=None) -> Iterator[FrameExample]:
    """Propose on the already-modified frames and pair the list with the answer."""
    import cv2

    per_frame, _info = bp.propose(frames, list(range(len(frames))), params=params,
                                  methods=methods or bp.METHOD_NAMES)
    height, width = frames[0].shape[:2]

    previous_gray = None
    for position, (frame, sample) in enumerate(zip(frames, samples)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        residual = None
        if previous_gray is not None:
            # global translation only; the ranker's motion channel wants the
            # camera taken out, not a dense flow field it cannot afford
            shift = _phase_shift(previous_gray, gray)
            residual = bctx.motion_residual(gray, bctx.compensate(previous_gray, shift))
        previous_gray = gray

        proposals = per_frame.get(position) or []
        if not proposals:
            continue

        visible = getattr(sample, "visibility", "VISIBLE") == "VISIBLE"
        verified = getattr(sample, "verified", True)
        truth = None
        target_slot = None
        proposed = False
        if visible and verified:
            gx, gy = float(sample.x), float(sample.y)
            radius = float(sample.radius_px)
            truth = (gx, gy, radius)
            tolerance = max(6.0, radius)
            hits = [i for i, p in enumerate(proposals)
                    if math.hypot(p.x - gx, p.y - gy) <= tolerance]
            if hits:
                proposed = True
                target_slot = hits[0]
        elif visible and not verified:
            # the label says the ball is there and the pixels did not measurably
            # change: not supervision, and not a null example either
            continue

        if truth is not None and not proposed:
            # counted by the caller as a proposal miss; never used as a null
            yield FrameExample(clip_id=clip_id, frame_index=frame_indices[position],
                               stacks=np.zeros((0, bctx.GRID, bctx.GRID,
                                                bctx.CHANNELS), np.uint8),
                               numeric=np.zeros((0, 1), np.float32),
                               target=None, truth=truth, proposed=False,
                               width=width, height=height)
            continue

        keep = choose_competitors(len(proposals),
                                  target_slot if target_slot is not None else -1,
                                  list_size - (1 if target_slot is not None else 0),
                                  rng)
        order = ([target_slot] + keep) if target_slot is not None else keep
        if not order:
            continue
        rng.shuffle(order)
        chosen = [proposals[i] for i in order]
        target = order.index(target_slot) if target_slot is not None else None

        pose = pose_for(frame_indices[position])
        stacks = np.stack([bctx.candidate_stack(frame, residual, p.x, p.y, p.radius)
                           for p in chosen])
        numeric = np.stack([np.asarray(numeric_fn(p, pose, width, height),
                                       dtype=np.float32) for p in chosen])
        yield FrameExample(clip_id=clip_id, frame_index=frame_indices[position],
                           stacks=stacks, numeric=numeric, target=target,
                           truth=truth, proposed=truth is not None,
                           width=width, height=height)


def _phase_shift(previous_gray: np.ndarray, gray: np.ndarray) -> tuple[float, float]:
    import cv2
    a = previous_gray.astype(np.float32)
    b = gray.astype(np.float32)
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), _response = cv2.phaseCorrelate(a, b, window)
    return float(dx), float(dy)


class ExampleWriter:
    """Append examples to a flat uint8 store plus a JSONL index.

    A frame's candidate list is a few hundred kilobytes and there are thousands of
    frames, so the images go into one growing binary file and the index records
    offsets. Nothing here is committed; the store lives under ``out/``.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.images = (self.directory / "stacks.u8").open("wb")
        self.index = (self.directory / "index.jsonl").open("w", encoding="utf-8")
        self.offset = 0
        self.written = 0
        self.missed = 0

    def add(self, example: FrameExample) -> None:
        if not example.proposed and example.truth is not None:
            self.missed += 1
            return
        if example.stacks.size == 0:
            return
        payload = np.ascontiguousarray(example.stacks, dtype=np.uint8).tobytes()
        self.images.write(payload)
        record = {
            "clip_id": example.clip_id,
            "frame_index": example.frame_index,
            "offset": self.offset,
            "count": int(example.stacks.shape[0]),
            "target": example.target,
            "truth": list(example.truth) if example.truth else None,
            "numeric": example.numeric.tolist(),
            "width": example.width,
            "height": example.height,
        }
        self.index.write(json.dumps(record) + "\n")
        self.offset += len(payload)
        self.written += 1

    def close(self) -> dict:
        self.images.close()
        self.index.close()
        return {"lists": self.written, "proposal_misses": self.missed,
                "bytes": self.offset}


class ExampleReader:
    """Read the store back. Memory-mapped: the batches are random-access."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.records = [json.loads(line) for line
                        in (self.directory / "index.jsonl").read_text("utf-8").splitlines()
                        if line.strip()]
        self.blob = np.memmap(self.directory / "stacks.u8", dtype=np.uint8, mode="r")
        self.stride = bctx.GRID * bctx.GRID * bctx.CHANNELS

    def __len__(self) -> int:
        return len(self.records)

    def stacks_for(self, record: dict) -> np.ndarray:
        start = record["offset"]
        size = record["count"] * self.stride
        return np.asarray(self.blob[start:start + size]).reshape(
            record["count"], bctx.GRID, bctx.GRID, bctx.CHANNELS)

    def numeric_for(self, record: dict) -> np.ndarray:
        return np.asarray(record["numeric"], dtype=np.float32)
