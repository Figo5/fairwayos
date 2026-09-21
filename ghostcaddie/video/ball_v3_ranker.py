"""Listwise candidate ranker for ball tracker v3.

v2 asked a network a question it cannot answer from the evidence it was given:
*does this 64x64 crop look like a ball?* Every bright round thing on a golf course
answers yes. v3 asks the question the tracker actually needs answered:

    of the candidates proposed on THIS frame, which one -- if any -- is the ball?

That is a different training objective, not a bigger model. All of a frame's
candidates go through the network together, each producing one scalar; a softmax
runs over the whole list plus a learned **null** slot; and the target is the index
of the true ball, or the null slot on a frame where there is no ball. So:

* the gradient is explicitly *comparative*. Pushing the true ball up necessarily
  pushes the shirt highlight beside it down, which is the thing a per-crop binary
  loss can never express. It is also why the training set must contain the true
  ball and the real distractors **from the same frame**;
* **abstention is learned, not thresholded afterwards.** The null slot competes
  with the candidates, so "none of these is the ball" is a first-class answer with
  a calibrated probability rather than a cutoff bolted on at the end;
* the softmax margin between the best and second candidate is available directly
  as the uncertainty signal the acceptance rule needs.

Input per candidate is the 7-channel two-scale stack from ``ball_v3_context``
(tight + context + camera-compensated motion) plus numeric features: geometry,
body-relative position, and which generators witnessed the proposal.

Weights are saved as a plain state dict and loaded with ``weights_only=True``.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from . import ball_v3_body as bbody
from . import ball_v3_proposals as bp
from . import ball_v3_context as bctx

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    torch = None

    class _Missing:
        class Module:
            pass

    nn = _Missing
    F = None
    TORCH_AVAILABLE = False

#: geometry (4) + body (7) + proposal evidence (2 + one score per method)
#: Always sized by ALL_METHOD_NAMES, not by whichever generators a given run
#: enables: an ablation must remove information, not resize the input layer.
NUMERIC_FEATURES = 4 + len(bbody.FEATURE_NAMES) + 2 + len(bp.ALL_METHOD_NAMES)


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "the v3 ranker needs PyTorch, which is not importable in this "
            "interpreter. Use .venv-video-ai/bin/python3 for anything that "
            "trains or scores; the pose cache is the one pass that uses .venv.")


def device() -> Any:
    _require_torch()
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def numeric_features(proposal, pose, width: int, height: int) -> list[float]:
    """The non-image half of a candidate's description."""
    scores = getattr(proposal, "method_scores", {}) or {}
    return (
        bctx.geometry_features(proposal.x, proposal.y, proposal.radius, width, height)
        + bbody.features(pose, proposal.x, proposal.y, proposal.radius)
        + [len(getattr(proposal, "methods", ())) / float(len(bp.ALL_METHOD_NAMES)),
           min(1.0, float(getattr(proposal, "spread", 0.0))
               / max(1.0, 4.0 * proposal.radius))]
        + [float(scores.get(name, 0.0)) for name in bp.ALL_METHOD_NAMES]
    )


if TORCH_AVAILABLE:

    class CandidateRanker(nn.Module):
        """7-channel two-scale stack + numeric features -> one comparable score."""

        def __init__(self, width: int = 32, numeric: int = NUMERIC_FEATURES) -> None:
            super().__init__()
            self.numeric_size = numeric
            self.encoder = nn.Sequential(
                nn.Conv2d(bctx.CHANNELS, width, 3, stride=2, padding=1),
                nn.BatchNorm2d(width), nn.ReLU(inplace=True),              # 32
                nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
                nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),          # 16
                nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1),
                nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),          # 8
                nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1),
                nn.BatchNorm2d(width * 4), nn.ReLU(inplace=True),          # 4
            )
            # Average and max pooling both: the average carries "what kind of place
            # is this", the maximum carries "there is a small bright thing here",
            # and a tiny ball is invisible to the average alone.
            self.numeric = nn.Sequential(
                nn.Linear(numeric, 64), nn.ReLU(inplace=True),
                nn.Linear(64, 64), nn.ReLU(inplace=True))
            self.head = nn.Sequential(
                nn.Linear(width * 8 + 64, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 1))
            # The null slot: a learned "none of these candidates is the ball"
            # logit that competes in the same softmax. Initialised low so early
            # training does not simply abstain on everything.
            self.null_logit = nn.Parameter(torch.tensor(-2.0))

        def forward(self, images: Any, numeric: Any) -> Any:
            features = self.encoder(images)
            pooled = torch.cat([F.adaptive_avg_pool2d(features, 1).flatten(1),
                                F.adaptive_max_pool2d(features, 1).flatten(1)], dim=1)
            joined = torch.cat([pooled, self.numeric(numeric)], dim=1)
            return self.head(joined).squeeze(-1)


def listwise_loss(scores: Any, target: Optional[int], null_logit: Any) -> Any:
    """Cross-entropy over [candidates..., null].

    ``target`` is the index of the true ball among the candidates, or ``None``
    when the frame holds no ball -- in which case the null slot is the answer.
    Frames where the ball exists but was never proposed are *not* passed here:
    training the model to abstain on a frame whose ball it was simply not shown
    would teach it to abstain on hard frames, which is the opposite of the point.
    """
    _require_torch()
    full = torch.cat([scores, null_logit.reshape(1)])
    index = len(scores) if target is None else int(target)
    return F.cross_entropy(full.unsqueeze(0),
                           torch.tensor([index], device=scores.device))


def rank_frame(model: Any, images: np.ndarray, numeric: np.ndarray, *,
               device: Any, batch_size: int = 256) -> dict:
    """Score one frame's candidates and turn them into a decision with a margin."""
    _require_torch()
    model.eval()
    if len(images) == 0:
        return {"scores": np.zeros(0), "probabilities": np.zeros(0),
                "null_probability": 1.0, "best": None, "margin": 0.0}
    chunks = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            tensor = torch.from_numpy(np.ascontiguousarray(
                batch.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0).to(device)
            numeric_tensor = torch.from_numpy(
                numeric[start:start + batch_size].astype(np.float32)).to(device)
            chunks.append(model(tensor, numeric_tensor).cpu())
        scores = torch.cat(chunks)
        full = torch.cat([scores, model.null_logit.detach().cpu().reshape(1)])
        probabilities = torch.softmax(full, dim=0).numpy()
    candidate_probabilities = probabilities[:-1]
    null_probability = float(probabilities[-1])
    order = np.argsort(-candidate_probabilities)
    best = int(order[0])
    second = (float(candidate_probabilities[order[1]]) if len(order) > 1 else 0.0)
    return {
        "scores": scores.numpy(),
        "probabilities": candidate_probabilities,
        "null_probability": null_probability,
        "best": best,
        "best_probability": float(candidate_probabilities[best]),
        # The signal the v2 pass did not have: how much better is the winner than
        # the runner-up. A near-tie is the case that should become AMBIGUOUS.
        "margin": float(candidate_probabilities[best] - second),
    }


def save_ranker(model: Any, path: Path) -> str:
    _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.detach().cpu()
                for key, value in model.state_dict().items()}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ranker(path: Path, *, device_override: Optional[Any] = None) -> Any:
    _require_torch()
    target = device_override or device()
    state = torch.load(path, map_location="cpu", weights_only=True)
    numeric = state["numeric.0.weight"].shape[1]
    model = CandidateRanker(numeric=int(numeric))
    model.load_state_dict(state)
    return model.to(target).eval()


def weights_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
