"""The three perception targets, their runtime safety, and honest outcomes.

A target that cannot run safely is BLOCKED. A target whose model is absent is
UNAVAILABLE. Neither is ever reported as a result, and neither is filled in with
a synthetic stand-in: synthetic data is confined to test fixtures.

Runtime facts recorded here come from earlier traced audits, not from guesses:
  * body      - ultralytics YOLO() reaches torch.load with weights_only=False
                injected by ultralytics/utils/patches.py; unrestricted pickle.
                BLOCKED. Never executed to "unblock" it.
  * clubhead  - SAM2.1 tiny video predictor, sam2/build_sam.py hardcodes
                weights_only=True with no fallback; the unused Hiera
                weights_path load is guarded. Safe on the exact traced path.
  * ball      - BootsTAPIR loads safely, but a hash-bound cross-clip rerun
                showed the method does not transfer (1/51 and 0/71 on the real
                clips). Safe to run, not demonstrated to work on new sources.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional


class TargetName:
    BODY = "body"
    CLUBHEAD = "clubhead"
    BALL = "ball"


class TargetOutcome:
    OBSERVED = "observed"
    UNAVAILABLE = "unavailable"      # could run, found nothing / no source support
    BLOCKED = "blocked"              # must not run: unsafe runtime path
    NOT_RUN = "not_run"


@dataclass(frozen=True)
class RuntimeSafety:
    target: str
    safe_to_run: bool
    reason: str
    model_present: bool = False
    evidence: str = ""


@dataclass
class TargetPlan:
    target: str
    outcome: str
    reason: str
    result: Optional[dict] = None
    initialization: str = "automatic"     # "automatic" | "assisted"
    assisted_disclosure: str = ""


SAM2_CKPT = "/tmp/fairway-sam-head/checkpoints/sam2.1_hiera_tiny.pt"
TAPIR_CKPT = "/tmp/fairway-learned/tracker/weights/bootstapir_checkpoint_v2.pt"


def describe_runtime() -> Dict[str, RuntimeSafety]:
    """Current, traced runtime safety per target. No model is loaded here."""
    return {
        TargetName.BODY: RuntimeSafety(
            target=TargetName.BODY, safe_to_run=False, model_present=True,
            reason="BLOCKED: ultralytics injects weights_only=False "
                   "(ultralytics/utils/patches.py) so YOLO() performs an "
                   "unrestricted pickle load. Not executed.",
            evidence="loader trace 2026-09-11; ULTRALYTICS_SAFE_LOAD unset, "
                     "SAFE_LOAD=False, _SafeLoad.SUPPORTED=True"),
        TargetName.CLUBHEAD: RuntimeSafety(
            target=TargetName.CLUBHEAD,
            safe_to_run=os.path.exists(SAM2_CKPT), model_present=os.path.exists(SAM2_CKPT),
            reason="SAM2.1 tiny: torch.load(..., weights_only=True) hardcoded at "
                   "sam2/build_sam.py with no fallback; unused Hiera weights_path "
                   "load guarded." if os.path.exists(SAM2_CKPT)
                   else "SAM2 checkpoint not present on this machine",
            evidence="independent head audit + loader trace"),
        TargetName.BALL: RuntimeSafety(
            target=TargetName.BALL,
            safe_to_run=os.path.exists(TAPIR_CKPT), model_present=os.path.exists(TAPIR_CKPT),
            reason="BootsTAPIR loads with weights_only=True, but a hash-bound "
                   "cross-clip rerun did NOT transfer (1/51 Morikawa, 0/71 "
                   "Gotterup). Safe to run; not demonstrated on new sources."
                   if os.path.exists(TAPIR_CKPT)
                   else "BootsTAPIR checkpoint not present on this machine",
            evidence="source-binding repair 2026-09-11"),
    }


def plan_targets(runtime: Dict[str, RuntimeSafety]) -> Dict[str, TargetPlan]:
    """Decide, before processing, what each target may honestly produce."""
    plan: Dict[str, TargetPlan] = {}
    for name, rs in runtime.items():
        if not rs.safe_to_run and rs.model_present:
            plan[name] = TargetPlan(name, TargetOutcome.BLOCKED, rs.reason)
        elif not rs.model_present:
            plan[name] = TargetPlan(name, TargetOutcome.UNAVAILABLE, rs.reason)
        else:
            plan[name] = TargetPlan(name, TargetOutcome.NOT_RUN,
                                    "safe runtime available; awaiting a source "
                                    "that supports this target")
    return plan


def is_three_target_success(plan: Dict[str, TargetPlan]) -> bool:
    """True only when all three targets actually produced observations.

    Metadata, plans and blocked/unavailable states never count as success.
    """
    return all(p.outcome == TargetOutcome.OBSERVED and p.result
               for p in (plan.get(TargetName.BODY), plan.get(TargetName.CLUBHEAD),
                         plan.get(TargetName.BALL)) if p is not None) and len(plan) >= 3
