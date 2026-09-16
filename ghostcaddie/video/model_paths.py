"""Explicit, configurable model-file discovery. Never `/tmp`, never a download.

The canonical pipeline must not depend on a cache that happens to still be warm.
`/tmp` was purged on 2026-09-16 and took the SAM2 and BootsTAPIR checkpoints with
it; the body model survived by luck. Resolution order, highest priority first:

  1. an explicit path handed in by the caller (a CLI flag)
  2. the per-model environment variable, e.g. ``FAIRWAY_BODY_MODEL``
  3. ``$FAIRWAY_MODEL_ROOT/<filename>``
  4. each entry of the documented local model search path:
     ``<repo>/models/<filename>`` then ``~/.fairwayos/models/<filename>``
  5. otherwise UNAVAILABLE -- a stated absence, never a silent fetch

Nothing here downloads, copies, or creates anything. `resolve()` always returns a
record; ``found`` is False when the operator has not placed the file, and
``searched`` names every location that was looked at so the message is actionable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where an operator is expected to put model files, in search order.
MODEL_SEARCH_PATH = (REPO_ROOT / "models", Path.home() / ".fairwayos" / "models")

MODEL_ROOT_ENV = "FAIRWAY_MODEL_ROOT"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    env_var: str
    purpose: str
    canonical: bool          # part of the canonical pipeline, or research-only
    note: str = ""


MODELS = {
    "body": ModelSpec(
        "body", "movenet-singlepose-lightning-v3.tflite", "FAIRWAY_BODY_MODEL",
        "MoveNet SinglePose Lightning v3, TFLite flatbuffer (no pickle, no torch)",
        canonical=True),
    "clubhead_sam2": ModelSpec(
        "clubhead_sam2", "sam2.1_hiera_tiny.pt", "FAIRWAY_SAM2_MODEL",
        "SAM2.1 tiny video predictor (research only)", canonical=False,
        note="Research route. The clubhead layer is UNAVAILABLE in the canonical "
             "pipeline; restoring this checkpoint does not change that."),
    "ball_sam2": ModelSpec(
        "ball_sam2", "sam2.1_hiera_tiny.pt", "FAIRWAY_SAM2_MODEL",
        "SAM2.1 tiny video predictor (research only)", canonical=False,
        note="Research route. Unproven on the ball."),
    "ball_tapir": ModelSpec(
        "ball_tapir", "bootstapir_checkpoint_v2.pt", "FAIRWAY_TAPIR_MODEL",
        "BootsTAPIR point tracker (research only)", canonical=False,
        note="Falsified for transfer under a hash-bound rerun (1/51, 0/71). "
             "Restoring this checkpoint does not restore the claim."),
}


@dataclass(frozen=True)
class ResolvedModel:
    key: str
    filename: str
    found: bool
    path: str = ""
    source: str = ""             # which rule supplied it
    reason: str = ""
    searched: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"key": self.key, "filename": self.filename, "found": self.found,
                "path": self.path, "resolved_by": self.source, "reason": self.reason,
                "searched": list(self.searched)}


def _ok(p: Path) -> bool:
    return p.is_file() and os.access(p, os.R_OK)


def resolve(key: str, explicit: Optional[str] = None,
            env: Optional[dict] = None) -> ResolvedModel:
    """Resolve one model file. Returns a record; never raises for absence."""
    if key not in MODELS:
        raise KeyError(f"unknown model key {key!r}; known: {sorted(MODELS)}")
    spec = MODELS[key]
    env = os.environ if env is None else env
    searched: List[str] = []

    def attempt(candidate, source) -> Optional[ResolvedModel]:
        p = Path(candidate).expanduser()
        searched.append(f"{source}: {p}")
        if _ok(p):
            return ResolvedModel(spec.key, spec.filename, True, str(p.resolve()),
                                 source, f"resolved by {source}", searched)
        return None

    if explicit:
        hit = attempt(explicit, "explicit argument")
        if hit:
            return hit
        # An explicit path that is wrong is a configuration error, not an
        # absence: do not quietly fall through to some other file.
        return ResolvedModel(spec.key, spec.filename, False, "", "explicit argument",
                             f"explicit model path is not a readable file: {explicit}",
                             searched)
    if env.get(spec.env_var):
        hit = attempt(env[spec.env_var], f"${spec.env_var}")
        if hit:
            return hit
        return ResolvedModel(spec.key, spec.filename, False, "", f"${spec.env_var}",
                             f"${spec.env_var} is set but is not a readable file: "
                             f"{env[spec.env_var]}", searched)
    if env.get(MODEL_ROOT_ENV):
        hit = attempt(Path(env[MODEL_ROOT_ENV]) / spec.filename, f"${MODEL_ROOT_ENV}")
        if hit:
            return hit
    for root in MODEL_SEARCH_PATH:
        hit = attempt(root / spec.filename, "model search path")
        if hit:
            return hit
    return ResolvedModel(
        spec.key, spec.filename, False, "", "",
        f"{spec.filename} was not found. Place it in one of the documented model "
        f"directories, or set ${spec.env_var}. Nothing is downloaded automatically.",
        searched)


def describe() -> dict:
    """Operator-facing summary of every model the system knows about."""
    return {
        "model_root_env": MODEL_ROOT_ENV,
        "search_path": [str(p) for p in MODEL_SEARCH_PATH],
        "downloads": "never; model files are placed by the operator",
        "models": {k: {**resolve(k).to_dict(), "purpose": s.purpose,
                       "env_var": s.env_var, "canonical": s.canonical,
                       "note": s.note}
                   for k, s in MODELS.items()},
    }
