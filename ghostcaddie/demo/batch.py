"""The batch runner: one broadcast config per clip, one row per clip in the report.

Design rules, all of them deliberate:

* **Every input appears in the result.** ``len(results) == len(entries)``; a clip that
  failed is reported as failed with its reason, it is not dropped from the report.
* **Each clip is a bounded subprocess.** An argv list (never a shell), a wall-clock
  timeout and its own output directory, so one corrupt clip cannot take the batch down
  and cannot write into another clip's artifacts.
* **The process exits non-zero if any clip did not succeed**, and the combined report
  and JSON are written either way.
* **Output directories are guarded** by the same rule as a single run: never inside the
  preserved baseline tree, the original showcase, or the git work tree, and never over
  a finished render without ``--force``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .broadcast import PROTECTED_OUTPUT_PREFIXES, guard_output_dir
from .errors import DemoConfigError

BATCH_SCHEMA = "fairwayos-broadcast-batch.v1"
DEFAULT_TIMEOUT_SECONDS = 3600


def load_manifest(path: str) -> List[dict]:
    """Read a batch manifest. Accepts a plain entry list or the baseline manifest.

    An entry is ``{"id", "clip", "config", "out"?}``. The baseline manifest's
    ``clips[]`` are accepted as-is (``id`` + ``clip``); an entry without a ``config``
    is reported as refused rather than guessed at, because a clip with no config has
    nothing to render.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise DemoConfigError(f"batch manifest does not exist: {path}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DemoConfigError(f"batch manifest is not valid JSON: {exc}") from exc

    raw: Any
    if isinstance(payload, list):
        raw = payload
    elif isinstance(payload, Mapping) and isinstance(payload.get("clips"), list):
        raw = payload["clips"]
    elif isinstance(payload, Mapping) and isinstance(payload.get("entries"), list):
        raw = payload["entries"]
    else:
        raise DemoConfigError(
            "batch manifest must be a list of entries, or an object with a 'clips' or "
            "'entries' list (the baseline manifest.json is accepted as 'clips')")
    entries: List[dict] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise DemoConfigError(f"manifest entry {index} is not an object")
        entry = {
            "id": str(item.get("id") or f"clip{index:02d}"),
            "clip": str(item.get("clip") or item.get("path") or ""),
            "config": str(item.get("config") or item.get("config_path") or ""),
            "out": str(item.get("out") or item.get("out_dir") or ""),
            "expected_sha256": str(item.get("clip_sha256")
                                   or item.get("expected_source_sha256") or ""),
        }
        entries.append(entry)
    if not entries:
        raise DemoConfigError("batch manifest has no entries")
    return entries


def _sha256_file(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classify(entry: dict) -> Optional[str]:
    """Pre-flight refusals, so a clip that cannot ever succeed says so up front."""
    if not entry["clip"]:
        return "the manifest entry names no clip"
    if not os.path.isfile(entry["clip"]):
        return f"the clip does not exist: {entry['clip']}"
    if not entry["config"]:
        return ("the manifest entry names no broadcast config, so there is nothing to "
                "render (refused rather than guessed)")
    if not os.path.isfile(entry["config"]):
        return f"the config does not exist: {entry['config']}"
    if entry["expected_sha256"]:
        actual = _sha256_file(entry["clip"])
        if actual != entry["expected_sha256"]:
            return (f"the clip's sha256 is {actual}, the manifest expects "
                    f"{entry['expected_sha256']}: this is not the file the run was "
                    f"prepared for")
    return None


def run_batch(manifest_path: str, out_root: str, *,
              force: bool = False, timeout: int = DEFAULT_TIMEOUT_SECONDS,
              cv_python: Optional[str] = None, body_model: Optional[str] = None,
              repo_root: Optional[str] = None,
              command_runner: Optional[Callable[[Sequence[str], int, str], tuple]]
              = None) -> dict:
    """Run one broadcast render per manifest entry. Returns the whole batch record."""
    entries = load_manifest(manifest_path)
    root = guard_output_dir(out_root, repo_root=repo_root, force=True)
    root.mkdir(parents=True, exist_ok=True)
    results: List[dict] = []

    for entry in entries:
        out_dir = entry["out"] or str(root / entry["id"])
        started = time.time()
        refusal = _classify(entry)
        if refusal:
            results.append({"id": entry["id"], "status": "refused", "reason": refusal,
                            "exit_code": None, "seconds": 0.0, "out_dir": out_dir,
                            "clip": entry["clip"], "config": entry["config"]})
            continue
        try:
            guard_output_dir(out_dir, repo_root=repo_root, force=force)
        except DemoConfigError as exc:
            results.append({"id": entry["id"], "status": "refused",
                            "reason": str(exc), "exit_code": None, "seconds": 0.0,
                            "out_dir": out_dir, "clip": entry["clip"],
                            "config": entry["config"]})
            continue

        cmd = [sys.executable, "-m", "ghostcaddie", "broadcast-showcase", "run",
               entry["clip"], "--config", entry["config"], "--out", out_dir]
        if cv_python:
            cmd += ["--cv-python", cv_python]
        if body_model:
            cmd += ["--body-model", body_model]
        if force:
            cmd += ["--force"]
        code, tail = (command_runner(cmd, timeout, str(root))
                      if command_runner else _run_command(cmd, timeout, str(root)))
        elapsed = round(time.time() - started, 2)
        ok = code == 0
        results.append({
            "id": entry["id"], "status": "succeeded" if ok else "failed",
            "reason": "" if ok else _failure_reason(tail),
            "exit_code": int(code), "seconds": elapsed, "out_dir": out_dir,
            "clip": entry["clip"], "config": entry["config"],
            "output_tail": tail[-4000:],
            "artifacts": _artifact_paths(out_dir) if ok else {},
        })

    succeeded = [r for r in results if r["status"] == "succeeded"]
    failed = [r for r in results if r["status"] == "failed"]
    refused = [r for r in results if r["status"] == "refused"]
    record = {
        "schema_version": BATCH_SCHEMA,
        "created_utc": _utcnow(),
        "manifest": str(Path(manifest_path).expanduser()),
        "out_root": str(root),
        "counts": {"total": len(results), "succeeded": len(succeeded),
                   "failed": len(failed), "refused": len(refused)},
        "note": ("Every manifest entry appears exactly once. A clip that failed or was "
                 "refused is reported with its reason; nothing is dropped."),
        "results": results,
    }
    (root / "batch_analysis.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    (root / "batch_report.md").write_text(render_batch_report(record), encoding="utf-8")
    return record


def _run_command(cmd: Sequence[str], timeout: int, cwd: str) -> tuple:
    """Run one bounded render. The child must be able to import this package, so it
    runs from the repository root with that root on ``PYTHONPATH`` -- never from the
    output directory, where the import would fail and look like a clip problem."""
    repo_root = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    try:
        proc = subprocess.run(list(cmd), capture_output=True, text=True,
                              cwd=repo_root, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return 124, f"the render did not finish within {timeout}s and was killed"
    tail = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return proc.returncode, tail


def _failure_reason(tail: str) -> str:
    lines = [line for line in (tail or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if "failed:" in line or "Error" in line or "error" in line:
            return line.strip()
    return lines[-1].strip() if lines else "the render failed with no message"


def _artifact_paths(out_dir: str) -> dict:
    root = Path(out_dir)
    out = {}
    for name in ("broadcast_annotated.mp4", "broadcast_report.md",
                 "broadcast_analysis.json", "broadcast_provenance.json",
                 "broadcast_config.json"):
        path = root / name
        if path.is_file():
            out[name] = str(path)
    return out


def _utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def render_batch_report(record: Mapping[str, Any]) -> str:
    """One row per clip, with the checks that decide whether a clip is done."""
    lines = ["# Broadcast batch report", "",
             f"Created {record.get('created_utc')} · manifest "
             f"`{record.get('manifest')}` · schema `{BATCH_SCHEMA}`", ""]
    counts = record.get("counts") or {}
    lines.append(f"**{counts.get('succeeded')}/{counts.get('total')} succeeded** "
                 f"({counts.get('failed')} failed, {counts.get('refused')} refused)")
    lines.append("")
    lines.append("| id | status | exit | seconds | out | reason |")
    lines.append("|---|---|---|---|---|---|")
    for item in record.get("results") or []:
        lines.append(f"| `{item.get('id')}` | {item.get('status')} | "
                     f"{item.get('exit_code')} | {item.get('seconds')} | "
                     f"{item.get('out_dir')} | {item.get('reason') or ''} |")
    lines.append("")
    for item in record.get("results") or []:
        if item.get("reason"):
            lines.append(f"### `{item.get('id')}` — {item.get('status')}")
            lines.append("")
            lines.append(f"* {item['reason']}")
            if item.get("output_tail"):
                lines.append("")
                lines.append("```")
                lines.append(str(item["output_tail"])[-2000:])
                lines.append("```")
            lines.append("")
    return "\n".join(lines)
