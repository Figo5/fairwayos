"""A small local ball detector, trained on synthetic ground truth.

Deliberately small. A golf ball is a handful of pixels; the reason the previous
approach failed was not model capacity, it was supervision and track validation.
So this is a compact convolutional heatmap network (CenterNet-style: predict a
Gaussian blob at the ball centre on a 64x64 crop) that is cheap to train, cheap to
run on the top-K candidates per frame, and easy to audit.

Design choices worth stating:

* **Crops, not whole frames.** Detecting a 3 px ball anywhere in a 4K frame with a
  downscaled full-frame network is hopeless. The classical ensemble proposes, the
  detector decides, and the crop keeps native detail.
* **Heatmap, not a bounding box.** Tiny objects have unreliable boxes; a Gaussian
  centre target is the honest representation at this scale.
* **Presence is learned, not assumed.** Half the value of this model is saying "no"
  to a shirt, a shoe, a clubhead or glare, so negatives are half the training set
  and the score is calibrated as a probability of being a ball.
* **Safe loading.** Weights are stored as a plain state dict and loaded with
  ``weights_only=True``; the file hash is recorded so a result can name the exact
  model it came from.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import cv2
import numpy as np

try:  # torch is optional in this repository; the detector is research-only
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    torch = None

    class _Missing:
        class Module:  # minimal stand-in so the module still imports
            pass

    nn = _Missing
    F = None
    TORCH_AVAILABLE = False

CROP_SIZE = 64
HEATMAP_SIGMA_RATIO = 0.5


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "the learned detector needs PyTorch, which is not importable in this "
            "interpreter. Run these commands with an interpreter that has torch "
            "(for this repository, .venv-video-ai/bin/python3).")


def device() -> Any:
    _require_torch()
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if TORCH_AVAILABLE:

    class BallDetector(nn.Module):
        """64x64 crop -> ball-centre heatmap, presence logit and radius."""

        def __init__(self, width: int = 32) -> None:
            super().__init__()
            self.width = width
            self.encoder = nn.Sequential(
                nn.Conv2d(3, width, 3, stride=2, padding=1), nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),                                   # 32x32
                nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
                nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),        # 16x16
                nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1),
                nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),        # 8x8
                nn.Conv2d(width * 2, width * 4, 3, stride=1, padding=1),
                nn.BatchNorm2d(width * 4), nn.ReLU(inplace=True),        # 8x8
            )
            self.up1 = nn.ConvTranspose2d(width * 4, width * 2, 4, stride=2, padding=1)
            self.up2 = nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1)
            self.up3 = nn.ConvTranspose2d(width, width, 4, stride=2, padding=1)
            self.head = nn.Conv2d(width, 1, 1)
            # Presence is judged from three views of the encoder output: the average
            # (context), the maximum (a small bright thing anywhere in the crop) and
            # the heatmap's own peak. Average pooling alone drowns a 4 px ball in a
            # 64 px crop, which measured as presence 0.18 for the smallest quartile
            # while the same ball at 8-16 px scored 0.62.
            self.presence = nn.Sequential(
                nn.Linear(width * 8, width * 4), nn.ReLU(inplace=True),
                nn.Linear(width * 4, 1))
            self.radius_head = nn.Sequential(
                nn.Linear(width * 8, width * 4), nn.ReLU(inplace=True),
                nn.Linear(width * 4, 1))

        def forward(self, x: Any):
            features = self.encoder(x)
            y = F.relu(self.up1(features))
            y = F.relu(self.up2(y))
            y = F.relu(self.up3(y))
            heatmap = self.head(y)
            if heatmap.shape[-2:] != x.shape[-2:]:
                heatmap = F.interpolate(heatmap, size=(x.shape[2], x.shape[3]),
                                        mode="bilinear", align_corners=False)
            # Average (context) and maximum (something small and bright) pooling of
            # the encoder output. The heatmap's own peak used to feed this head, which
            # coupled the presence decision to an under-trained localisation map and
            # made both worse; the two heads are now independent.
            pooled = torch.cat([F.adaptive_avg_pool2d(features, 1).flatten(1),
                                F.adaptive_max_pool2d(features, 1).flatten(1)], dim=1)
            # contiguous() matters: upsampling's backward pass rejects a view whose
            # memory spans two contiguous subspaces, so a lazily-expanded feature map
            # fails during backprop rather than at the forward pass.
            return heatmap.contiguous(), self.presence(pooled), self.radius_head(pooled)


def _focal_heatmap_loss(logits: Any, target: Any, *, alpha: float = 2.0,
                        beta: float = 4.0) -> Any:
    """CenterNet-style focal loss for a tiny blob on a busy real background.

    Plain pixel-wise BCE on this task has a degenerate optimum: a 3 px blob is a
    handful of positive pixels out of 4096, so predicting "no ball" everywhere
    minimises the loss. Measured behaviour was exactly that -- training loss fell
    while recall collapsed from 0.47 to 0.17 -- so the loss has to weight the few
    positives against the thousands of easy negatives.
    """
    _require_torch()
    probability = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    positive = target >= 0.999
    positive_loss = -((1 - probability) ** alpha) * torch.log(probability) * positive
    negative_loss = -((1 - target) ** beta) * (probability ** alpha) \
        * torch.log(1 - probability) * (~positive)
    count = positive.sum().clamp(min=1.0)
    return (positive_loss.sum() + negative_loss.sum()) / count


def _to_tensor(batch: np.ndarray, device: Any):
    """NHWC uint8 batch -> contiguous NCHW float tensor.

    ``from_numpy`` on a transposed view yields a tensor whose memory spans two
    contiguous subspaces. Convolutions accept that layout, but the backward pass
    rejects it with "view size is not compatible ... Use .reshape() instead", which
    surfaces as an autograd failure rather than a shape error.
    """
    _require_torch()
    array = np.ascontiguousarray(batch.transpose(0, 3, 1, 2)).astype(np.float32) / 255.0
    return torch.from_numpy(array).to(device)


def _gaussian_target(size: int, x: float, y: float, radius: float) -> np.ndarray:
    sigma = max(1.0, radius * HEATMAP_SIGMA_RATIO)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    blob = np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma)))
    return blob.astype(np.float32)


def load_crops(records: Sequence[dict], images_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """Read every crop into memory once: they are 64x64 and there are thousands."""
    images, kept = [], []
    for record in records:
        path = images_dir / record["path"]
        if not path.exists():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        if image.shape[0] != CROP_SIZE or image.shape[1] != CROP_SIZE:
            image = cv2.resize(image, (CROP_SIZE, CROP_SIZE))
        images.append(image)
        kept.append(record)
    if not images:
        raise ValueError(f"no crops could be read from {images_dir}")
    return np.stack(images), kept


def _augment(batch: np.ndarray, labels: list[dict], rng: np.random.Generator,
             pad: int = 6) -> tuple[np.ndarray, list[dict]]:
    """Brightness jitter plus a padded random shift, with the label moved too.

    The shift matters: a detector that only ever sees the ball dead centre learns
    to answer "centre" regardless of input, which is the classic way a tiny-object
    model passes its own training set and fails in the field.
    """
    out = np.zeros_like(batch)
    moved = []
    for image, record in zip(batch, labels):
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        offset_x = int(rng.integers(-pad, pad + 1))
        offset_y = int(rng.integers(-pad, pad + 1))
        left, top = pad + offset_x, pad + offset_y
        crop = padded[top:top + CROP_SIZE, left:left + CROP_SIZE].astype(np.float32)
        gain = float(rng.uniform(0.85, 1.15))
        bias = float(rng.uniform(-12.0, 12.0))
        crop = np.clip(crop * gain + bias, 0, 255)
        out[len(moved)] = crop.astype(np.uint8)
        entry = dict(record)
        if record.get("positive"):
            entry["label_x"] = record["label_x"] - offset_x
            entry["label_y"] = record["label_y"] - offset_y
            if not (0 <= entry["label_x"] < CROP_SIZE and 0 <= entry["label_y"] < CROP_SIZE):
                entry["positive"] = False          # the ball left the crop
        moved.append(entry)
    return out, moved


def train_detector(records: Sequence[dict], images_dir: Path, *, out_dir: Path,
                   epochs: int = 4, batch_size: int = 64, learning_rate: float = 2e-3,
                   seed: int = 20260918, val_fraction: float = 0.15,
                   presence_weight: float = 1.0, heatmap_weight: float = 1.0,
                   radius_weight: float = 0.2, log: bool = True) -> dict:
    """Train on synthetic crops; report real numbers, not a training curve."""
    _require_torch()
    rng = np.random.default_rng(seed)
    images, labels = load_crops(records, images_dir)
    order = rng.permutation(len(labels))
    split = int(len(order) * (1.0 - val_fraction))
    train_idx, val_idx = order[:split], order[split:]

    torch.manual_seed(seed)
    target_device = device()
    model = BallDetector().to(target_device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    best_f1, best_epoch, best_state = -1.0, 0, None
    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(train_idx)
        total, seen = 0.0, 0
        for start in range(0, len(permutation), batch_size):
            chunk = permutation[start:start + batch_size]
            batch, batch_labels = _augment(images[chunk],
                                           [labels[i] for i in chunk], rng)
            images_t = _to_tensor(batch, target_device)
            heatmap_target = np.zeros((len(chunk), 1, CROP_SIZE, CROP_SIZE),
                                      dtype=np.float32)
            # Targets carry the same rank as their heads ((B, 1)). A rank mismatch
            # here broadcasts to (B, B) instead of raising, and the loss then trains
            # on a matrix of cross-example comparisons.
            presence_target = np.zeros((len(chunk), 1), dtype=np.float32)
            radius_target = np.zeros((len(chunk), 1), dtype=np.float32)
            radius_mask = np.zeros((len(chunk), 1), dtype=np.float32)
            for position, record in enumerate(batch_labels):
                if record.get("positive"):
                    heatmap_target[position, 0] = _gaussian_target(
                        CROP_SIZE, record["label_x"], record["label_y"],
                        max(1.0, record.get("radius") or 3.0))
                    presence_target[position, 0] = 1.0
                    radius_target[position, 0] = math.log(
                        max(0.7, float(record.get("radius") or 3.0)))
                    radius_mask[position, 0] = 1.0
            heatmap_t = torch.from_numpy(heatmap_target).to(target_device)
            presence_t = torch.from_numpy(presence_target).to(target_device)
            radius_t = torch.from_numpy(radius_target).to(target_device)
            mask_t = torch.from_numpy(radius_mask).to(target_device)

            heatmap_logits, presence_logits, radius_prediction = model(images_t)
            # focal-ish weighting: negatives are the majority and the failure mode
            weights = torch.where(presence_t > 0.5, torch.ones_like(presence_t),
                                  torch.full_like(presence_t, 1.5))
            presence_loss = F.binary_cross_entropy_with_logits(
                presence_logits, presence_t, weight=weights)
            heatmap_loss = _focal_heatmap_loss(heatmap_logits, heatmap_t)
            radius_loss = (F.smooth_l1_loss(radius_prediction, radius_t,
                                            reduction="none") * mask_t).sum() \
                / mask_t.sum().clamp(min=1.0)
            loss = (presence_weight * presence_loss + heatmap_weight * heatmap_loss
                    + radius_weight * radius_loss)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.item()) * len(chunk)
            seen += len(chunk)

        val = evaluate_detector(model, images, [labels[i] for i in val_idx],
                               device=target_device, batch_size=batch_size)
        history.append({"epoch": epoch + 1, "train_loss": round(total / max(seen, 1), 4),
                        "val": val})
        # Model selection on the synthetic validation slice. The last epoch is not
        # automatically the best one -- recall oscillated between 0.47 and 0.09 across
        # epochs in the first run -- so the best epoch is kept and named.
        if val["f1"] >= best_f1:
            best_f1 = val["f1"]
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        if log:
            print(f"epoch {epoch + 1}/{epochs} loss={total / max(seen, 1):.4f} "
                  f"val_precision={val['precision']:.3f} val_recall={val['recall']:.3f} "
                  f"center_err={val['median_center_error_px']}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "detector_state.pt"
    torch.save(model.state_dict(), weights_path)
    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    report = {
        "origin": "SYNTHETIC_GROUND_TRUTH",
        "architecture": "BallDetector heatmap CNN (width 32, 4 encoder stages, 3 "
                        "transposed-conv stages, presence + radius heads)",
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "crop_size": CROP_SIZE,
        "epochs": epochs,
        "train_crops": int(len(train_idx)),
        "val_crops": int(len(val_idx)),
        "positives": int(sum(1 for record in labels if record.get("positive"))),
        "negatives": int(sum(1 for record in labels if not record.get("positive"))),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "device": str(target_device),
        "weights": weights_path.name,
        "weights_sha256": digest,
        "history": history,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "detector_report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


@torch.no_grad() if TORCH_AVAILABLE else (lambda function: function)
def evaluate_detector(model: Any, images: np.ndarray, labels: Sequence[dict], *,
                      device: Any, batch_size: int = 128,
                      threshold: float = 0.5) -> dict:
    """Precision/recall/centre error at crop level, with exact synthetic labels."""
    _require_torch()
    model.eval()
    positives = [i for i, record in enumerate(labels) if record.get("positive")]
    negatives = [i for i, record in enumerate(labels) if not record.get("positive")]
    if not positives:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "median_center_error_px": None, "positives": 0, "negatives": len(negatives),
                "false_positives": 0, "false_negatives": 0}

    def scores(indices: Sequence[int]) -> dict[int, tuple[float, float, float, float]]:
        """Per index: (presence, centre x, centre y, radius), with sub-pixel centres."""
        results: dict[int, tuple[float, float, float, float]] = {}
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            batch = images[chunk].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            tensor = torch.from_numpy(batch).to(device)
            heatmap, presence_logits, radius_prediction = model(tensor)
            probability = torch.sigmoid(presence_logits).squeeze(1).cpu().numpy()
            heat = heatmap.squeeze(1).cpu().numpy()
            predicted_radius = torch.exp(radius_prediction).squeeze(1).cpu().numpy()
            for position, index in enumerate(chunk):
                cx, cy = _subpixel_peak(heat[position])
                results[int(index)] = (float(probability[position]), cx, cy,
                                       float(predicted_radius[position]))
        return results

    positive_scores = scores(positives)
    negative_index = sorted(negatives)
    negative_scores = scores(negative_index) if negatives else {}

    detected = [index for index in positives
                if positive_scores[index][0] >= threshold]
    errors = []
    for index in detected:
        _, cx, cy, _ = positive_scores[index]
        label = labels[index]
        errors.append(math.hypot(cx - label["label_x"], cy - label["label_y"]))
    false_positives = sum(1 for index in negatives
                          if negative_scores.get(index, (0.0,))[0] >= threshold)
    true_positive = len(detected)
    false_negative = len(positives) - true_positive

    precision = true_positive / max(1, true_positive + false_positives)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    return {
        "positives": len(positives),
        "negatives": len(negatives),
        "true_positives": true_positive,
        "false_positives": false_positives,
        "false_negatives": false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_center_error_px": round(float(np.mean(errors)), 3) if errors else None,
        "median_center_error_px": round(float(np.median(errors)), 3) if errors else None,
        "p90_center_error_px": round(float(np.percentile(errors, 90)), 3) if errors else None,
    }


def save_detector(model: Any, path: Path) -> str:
    """Save a plain state dict and return its hash. Nothing pickled, nothing custom."""
    _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_detector(path: Path, *, device_override: Optional[Any] = None) -> Any:
    """Load with ``weights_only=True``: a state dict is data, not code."""
    _require_torch()
    model = BallDetector()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device_override or device())


def _subpixel_peak(heatmap: np.ndarray, window: int = 2) -> tuple[float, float]:
    """Weighted centroid around the heatmap peak.

    An integer argmax would cap centre accuracy at half a pixel *and* quantise it
    to whole pixels; the labelled centre is fractional, so the estimator has to be
    too, or the measured error is partly the measurement's own fault.
    """
    flat = int(np.argmax(heatmap))
    peak_y, peak_x = divmod(flat, heatmap.shape[1])
    x0 = max(0, peak_x - window)
    x1 = min(heatmap.shape[1], peak_x + window + 1)
    y0 = max(0, peak_y - window)
    y1 = min(heatmap.shape[0], peak_y + window + 1)
    patch = heatmap[y0:y1, x0:x1].astype(np.float64)
    total = patch.sum()
    if total <= 1e-6:
        return float(peak_x), float(peak_y)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return float((patch * xs).sum() / total), float((patch * ys).sum() / total)


@torch.no_grad() if TORCH_AVAILABLE else (lambda function: function)
def score_crops(model: Any, crops: Sequence[np.ndarray], *, device: Any
                ) -> list[dict]:
    """Score prepared 64x64 crops, returning presence, refined centre and radius."""
    _require_torch()
    if not crops:
        return []
    # Inference must not depend on the caller's mode: in train mode batch-norm uses
    # the batch's own statistics, so the same crop would score differently
    # depending on what it was batched with. That is a silent reproducibility
    # failure in a measurement pipeline.
    model.eval()
    batch = np.stack([crop if crop.shape[:2] == (CROP_SIZE, CROP_SIZE)
                      else cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
                      for crop in crops]).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    tensor = torch.from_numpy(batch).to(device)
    heatmap, presence_logits, radius_prediction = model(tensor)
    probability = torch.sigmoid(presence_logits).squeeze(1).cpu().numpy()
    heat = heatmap.squeeze(1).cpu().numpy()
    predicted_radius = torch.exp(radius_prediction).squeeze(1).cpu().numpy()
    results = []
    for position in range(len(crops)):
        cx, cy = _subpixel_peak(heat[position])
        results.append({
            "presence": float(probability[position]),
            "center_in_crop": (cx, cy),
            "radius_px": float(predicted_radius[position]),
            "peak": float(heat[position].max()),
        })
    return results


def crop_at(frame: np.ndarray, x: float, y: float, size: int = CROP_SIZE
            ) -> Optional[np.ndarray]:
    half = size // 2
    left, top = int(round(x)) - half, int(round(y)) - half
    if left < 0 or top < 0 or left + size > frame.shape[1] or top + size > frame.shape[0]:
        return None
    return frame[top:top + size, left:left + size]
