"""Runtime registry: which interpreter can actually run which adapter.

The three adapters do not share one interpreter. Body needs the TFLite runtime;
clubhead and ball need torch. Rather than forcing a global install to unify them,
each target is bound to an interpreter and executed in a SUBPROCESS. That keeps
the torch/pickle-capable interpreter out of the serving process entirely.

Readiness is PROBED by running the candidate interpreter and importing the
modules there, out of process. Nothing here is a hardcoded availability flag.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class InterpreterUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeSpec:
    target: str
    interpreter: str
    modules: List[str]
    purpose: str


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    reason: str
    python_version: str = ""
    missing: tuple = ()


def probe_interpreter(interpreter: str, modules: List[str],
                      timeout: float = 60.0) -> ProbeResult:
    """Import `modules` INSIDE `interpreter`, out of process."""
    if not interpreter or not os.path.isfile(interpreter) or not os.access(interpreter, os.X_OK):
        return ProbeResult(False, f"interpreter not executable: {interpreter}")
    code = (
        "import importlib,json,sys\n"
        f"mods={json.dumps(list(modules))}\n"
        "missing=[]\n"
        "for m in mods:\n"
        "    try: importlib.import_module(m)\n"
        "    except Exception: missing.append(m)\n"
        "print(json.dumps({'v':sys.version.split()[0],'missing':missing}))\n"
    )
    try:
        p = subprocess.run([interpreter, "-c", code], capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ProbeResult(False, f"probe timed out after {timeout}s")
    if p.returncode != 0:
        return ProbeResult(False, f"probe failed: {(p.stderr or '').strip()[:200]}")
    try:
        out = json.loads((p.stdout or "").strip().splitlines()[-1])
    except Exception:
        return ProbeResult(False, f"probe returned unparsable output: {p.stdout[:150]}")
    missing = tuple(out.get("missing", []))
    if missing:
        return ProbeResult(False, "missing modules in that interpreter: "
                                  + ", ".join(missing), out.get("v", ""), missing)
    return ProbeResult(True, "all required modules import in this interpreter",
                       out.get("v", ""))


class RuntimeRegistry:
    def __init__(self, specs: Dict[str, RuntimeSpec]):
        self.specs = specs
        self._cache: Dict[str, ProbeResult] = {}

    @classmethod
    def default(cls) -> "RuntimeRegistry":
        return cls({
            "body": RuntimeSpec(
                "body", os.path.join(REPO, ".venv", "bin", "python3"),
                ["ai_edge_litert", "cv2", "numpy"],
                "MoveNet TFLite: no torch, no pickle surface"),
            "clubhead": RuntimeSpec(
                "clubhead", os.path.join(REPO, ".venv-video-ai", "bin", "python3"),
                ["torch", "cv2", "numpy"],
                "SAM2.1 tiny, weights_only=True with the unused Hiera path guarded"),
            "ball": RuntimeSpec(
                "ball", os.path.join(REPO, ".venv-video-ai", "bin", "python3"),
                ["torch", "cv2", "numpy"],
                "BootsTAPIR, weights_only=True"),
        })

    def probe(self, target: str, refresh: bool = False) -> ProbeResult:
        if refresh or target not in self._cache:
            s = self.specs[target]
            self._cache[target] = probe_interpreter(s.interpreter, s.modules)
        return self._cache[target]

    def readiness(self) -> dict:
        """DEPENDENCY readiness only.

        Finding 6: this previously reported a single "ready" flag that probed
        interpreter modules and nothing else, which read as though the target
        could produce a result. Interpreter readiness, model-file presence and
        actual executability are now reported separately, and executability is
        false whenever the adapter cannot run unattended.
        """
        from ghostcaddie.upload.adapters import all_adapters
        ads = all_adapters()
        out = {}
        for t, sp in self.specs.items():
            r = self.probe(t)
            cap = ads[t].capability() if t in ads else None
            model_present = bool(cap.model_present) if cap else False
            model_sha = (cap.model_sha256 if cap else "") or ""
            needs_seed = t in ("clubhead", "ball")
            can_exec = bool(r.ok and model_present and not needs_seed)
            reason = r.reason if not r.ok else (
                f"model file not found: {cap.model_path}" if not model_present else
                ("interpreter and model are present, but this target cannot run "
                 "unattended: it requires a reviewed source-specific seed that is "
                 "not yet wired into the job path" if needs_seed
                 else "interpreter and model present; target can execute"))
            out[t] = {
                "interpreter_ready": r.ok,
                "model_file_present": model_present,
                "model_sha256": model_sha[:16],
                "requires_reviewed_seed": needs_seed,
                "can_execute_now": can_exec,
                "reason": reason,
                "interpreter": sp.interpreter, "python_version": r.python_version,
                "modules": list(sp.modules), "missing": list(r.missing),
                "purpose": sp.purpose,
            }
        return out

    def require(self, target: str) -> RuntimeSpec:
        r = self.probe(target)
        if not r.ok:
            raise InterpreterUnavailable(f"{target}: {r.reason}")
        return self.specs[target]
