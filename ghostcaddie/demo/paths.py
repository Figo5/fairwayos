"""Interpreter resolution, out-of-process dispatch and the demo output contract.

Demo mode follows the canonical split: the orchestrator that owns the contract is
stdlib-only, and every module that needs cv2/numpy/litert runs in an interpreter
that has them, launched as a subprocess. Nothing is installed, downloaded or
guessed: if no interpreter can import what the demo needs, the run fails with the
list of places that were tried.

Environment overrides, in order:

  1. an explicit ``--cv-python`` flag
  2. ``$FAIRWAY_DEMO_PYTHON``
  3. the canonical body interpreter (``ghostcaddie.video.canonical.body_interpreter_path``,
     which itself honours ``$FAIRWAY_BODY_PYTHON`` and the runtime registry)
  4. the interpreter running the CLI

An explicit path that does not work is an error, never a silent fallback.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .errors import DemoSourceError, DemoUnavailable, DemoWorkerError

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_MODULE = "ghostcaddie.demo.worker"
SETUP_MODULE = "ghostcaddie.demo.setup_gui"

WORKER_TIMEOUT_SECONDS = 3600
DEMO_MODULES = ("cv2", "numpy")

DEMO_PYTHON_ENV = "FAIRWAY_DEMO_PYTHON"


def out_paths(out_dir) -> Dict[str, Path]:
    """The demo output contract, in one place."""
    out = Path(out_dir).expanduser()
    return {
        "annotated": out / "demo_annotated.mp4",
        "report": out / "demo_report.md",
        "analysis": out / "demo_analysis.json",
        "provenance": out / "demo_provenance.json",
        "summary": out / "demo_summary.md",
        "keyframes_dir": out / "keyframes",
        "contact_sheet": out / "keyframes" / "contact_sheet.png",
        "debug_dir": out / "debug",
    }


#: The four phases that must exist as showcase stills, in LinkedIn order.
SHOWCASE_STILL_PHASES = ("address", "top", "impact", "finish")


def showcase_out_paths(out_dir) -> Dict[str, Path]:
    """The showcase output contract: presentation artifacts, same writer.

    Separate names, not a rename of the demo contract, so a showcase run can never
    overwrite a demo artifact set: the demo names and the showcase names are two
    disjoint namespaces in the same directory.
    """
    out = Path(out_dir).expanduser()
    return {
        "annotated": out / "showcase_annotated.mp4",
        "report": out / "showcase_report.md",
        "analysis": out / "showcase_analysis.json",
        "provenance": out / "showcase_provenance.json",
        "config_copy": out / "showcase_config.json",
        "linkedin": out / "linkedin_post_draft.md",
        # A showcase still is named for its phase at the top level
        # (showcase_address.png, ...), so the stills directory IS the output
        # directory: one file per phase, no numbered copies of the same frame.
        "keyframes_dir": out,
        "contact_sheet": out / "showcase_contact_sheet.png",
        "debug_dir": out / "showcase_debug",
    }


def showcase_still_path(out_dir, phase: str) -> Path:
    """The still for one phase: ``showcase_<phase>.png`` in the output directory."""
    return Path(out_dir).expanduser() / f"showcase_{phase}.png"


def resolve_out_paths(out_dir, *, showcase: bool) -> Dict[str, Path]:
    return showcase_out_paths(out_dir) if showcase else out_paths(out_dir)


def candidate_interpreters(explicit: Optional[str] = None
                           ) -> List[Tuple[str, str]]:
    """Every interpreter worth trying, with the rule that produced it."""
    out: List[Tuple[str, str]] = []
    seen = set()

    def add(path: Optional[str], how: str) -> None:
        if not path:
            return
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        out.append((key, how))

    add(explicit, "explicit --cv-python")
    add(os.environ.get(DEMO_PYTHON_ENV), f"${DEMO_PYTHON_ENV}")
    try:
        from ghostcaddie.video.canonical import body_interpreter_path
        add(body_interpreter_path(), "canonical body interpreter")
    except Exception:  # pragma: no cover - canonical resolution is best effort here
        pass
    import sys
    add(sys.executable, "interpreter running this command")
    return out


def probe_interpreter(path: str, modules: Sequence[str] = DEMO_MODULES) -> dict:
    """Import the demo's modules INSIDE ``path``, out of process."""
    from ghostcaddie.upload.runtimes import probe_interpreter as canonical_probe
    result = canonical_probe(path, list(modules))
    return {"ok": bool(result.ok), "reason": result.reason,
            "python_version": result.python_version, "missing": list(result.missing)}


def resolve_cv_interpreter(explicit: Optional[str] = None, *,
                           probe: bool = True) -> dict:
    """Pick the interpreter the CV worker will run in. Never installs anything."""
    attempts: List[dict] = []
    for path, how in candidate_interpreters(explicit):
        if not os.path.isfile(path) or not os.access(path, os.X_OK):
            attempts.append({"path": path, "how": how, "ok": False,
                             "reason": "not an executable file"})
            if how.startswith("explicit"):
                raise DemoUnavailable(
                    f"the --cv-python interpreter is not an executable file: {path}")
            continue
        if not probe:
            return {"path": path, "how": how, "attempts": attempts,
                    "probed": False, "python_version": ""}
        result = probe_interpreter(path)
        attempts.append({"path": path, "how": how, **result})
        if result["ok"]:
            return {"path": path, "how": how, "attempts": attempts, "probed": True,
                    "python_version": result["python_version"],
                    "modules": list(DEMO_MODULES)}
        if how.startswith("explicit"):
            raise DemoUnavailable(
                f"the --cv-python interpreter cannot run the demo worker: "
                f"{result['reason']}")
    raise DemoUnavailable(
        "no interpreter could import the demo's CV dependencies "
        f"({', '.join(DEMO_MODULES)}). Tried: "
        + "; ".join(f"{a['path']} ({a['how']}): {a['reason']}" for a in attempts)
        + ". Point --cv-python or $FAIRWAY_DEMO_PYTHON at an interpreter that has "
          "them; nothing is installed or downloaded automatically.")


def worker_env() -> dict:
    env = dict(os.environ)
    # Offline by construction: no proxy variable may make a network call look
    # possible on this path, and the worker is told it is offline.
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        env.pop(var, None)
    env["FAIRWAYOS_OFFLINE"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_demo_worker(spec: dict, interpreter: str) -> dict:
    """One JSON spec in, one JSON document out. A non-zero exit is an error."""
    if not interpreter or not os.path.isfile(interpreter):
        raise DemoUnavailable(f"worker interpreter is not a file: {interpreter!r}")
    cmd = [interpreter, "-m", WORKER_MODULE]
    try:
        proc = subprocess.run(cmd, input=json.dumps(spec), capture_output=True,
                              text=True, timeout=WORKER_TIMEOUT_SECONDS,
                              env=worker_env())
    except subprocess.TimeoutExpired as exc:
        raise DemoWorkerError(
            f"the demo worker exceeded its {WORKER_TIMEOUT_SECONDS}s bound and was "
            f"terminated") from exc
    except OSError as exc:
        raise DemoWorkerError(f"unable to launch the demo worker: {exc}") from exc
    if proc.returncode != 0:
        raise DemoWorkerError(
            f"demo worker failed (rc={proc.returncode}): "
            f"{(proc.stderr or '').strip()[-1200:]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise DemoWorkerError(
            f"demo worker produced unparsable output: {(proc.stdout or '')[:400]}") from exc


def run_setup_process(interpreter: str, args: Sequence[str]) -> int:
    """The interactive picker runs in the CV interpreter, not in this one."""
    if not interpreter or not os.path.isfile(interpreter):
        raise DemoUnavailable(f"picker interpreter is not a file: {interpreter!r}")
    cmd = [interpreter, "-m", SETUP_MODULE, *[str(a) for a in args]]
    try:
        proc = subprocess.run(cmd, env=worker_env())
    except OSError as exc:
        raise DemoUnavailable(f"unable to launch the setup picker: {exc}") from exc
    return int(proc.returncode)


def sha256_of(path) -> str:
    from ghostcaddie.video.source_binding import sha256_file
    try:
        return sha256_file(Path(path))
    except OSError as exc:  # pragma: no cover - unreadable output artifact
        raise DemoSourceError(f"cannot hash {path}: {exc}") from exc
