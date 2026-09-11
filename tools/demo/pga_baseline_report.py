"""Measure PGA-footage tracking behaviour on a FIXED evaluation interval and a
FIXED held-out interval, declared BEFORE any tuning.

Reports what the reset brief asks for -- visible coverage, loss structure,
reacquisition and camera drift -- not a bare detection count. Local research
only: the source is REVIEW_REQUIRED, never demo-eligible.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from ghostcaddie.video.pga_source_gate import evaluate_source, REJECTED

SRC = "out/youtube_fullswing/f3kTTMZlxds.mp4"

# Declared BEFORE tuning. Left panel only (x 0..640); the clip is dual-panel.
# Segment boundaries came from camera-cut detection over the whole clip.
INTERVALS = {
    # tuning/inspection is allowed here
    "eval": {"player": "10. WILL ZALATORIS (left panel)",
             "segment": [180, 625], "window": [341, 517]},
    # never tuned on; used only to check that a change generalises.
    # RETAINED even though it turned out degenerate: 88/89 sampled frames are
    # duplicates because this segment is a slow-motion replay. That is a real
    # measurement about the footage and is not deleted to tidy the report.
    "heldout_1_degenerate": {"player": "3. RORY MCILROY (left panel)",
                             "segment": [3340, 3795], "window": [3400, 3576]},
    # Declared AFTER observing heldout_1 was degenerate, and BEFORE any detector
    # tuning (none has occurred). Selection criterion was a property of the
    # FOOTAGE -- duplicate-frame density measured across all nine segments --
    # not detector performance. Cantlay had the lowest duplicate fraction (0.04).
    "heldout_2": {"player": "4. PATRICK CANTLAY (left panel)",
                  "segment": [2890, 3335], "window": [2954, 3130]},
}
CROP = "0,0,640,720"


def run(name, spec, outdir):
    a, b = spec["window"]
    out = os.path.join(outdir, name)
    cmd = [".venv-video-ai/bin/python3", "-m", "ghostcaddie.video.pga_analyze",
           SRC, "--output-dir", out, "--crop", CROP,
           "--source-frame-start", str(a), "--source-frame-end", str(b),
           "--sample-step", "2"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    diag_path = os.path.join(out, "diagnostics.json")
    if not os.path.exists(diag_path):
        return {"interval": name, **spec, "status": "no_diagnostics",
                "stderr": p.stderr[-400:]}
    d = json.load(open(diag_path))
    return {"interval": name, **spec, "status": d.get("status", "complete"),
            "reason": d.get("reason"), "diagnostics": diag_path,
            "summary": summarise(d)}


def summarise(d):
    """Loss structure, not just counts."""
    out = {"frames_total": d.get("frames_total"),
           "duplicate_pairs": len(d.get("duplicate_pairs") or [])}
    for layer in ("ball", "clubhead"):
        L = d.get(layer) or {}
        counts = L.get("counts") or {}
        n = d.get("frames_total") or 0
        det = counts.get("detected", 0)
        out[layer] = {
            "counts": counts,
            "visible_coverage_pct": round(100.0 * det / n, 1) if n else None,
            "longest_consecutive_run": (L.get("longest_consecutive_run_frames")
                                        or L.get("longest_consecutive_run")),
            "wrong_object_matches": len(L.get("wrong_object_matches") or []),
            "duplicates_excluded": L.get("duplicates_excluded"),
        }
    out["flags"] = {k: (v.get("status") if isinstance(v, dict) else v)
                    for k, v in (d.get("flags") or {}).items()}
    out["camera_cuts"] = (d.get("flags") or {}).get("camera_cuts")
    return out


def main():
    d = evaluate_source(SRC)
    if d.status == REJECTED:
        raise SystemExit(f"source REJECTED by the PGA gate: {d.reason}")
    print(f"source gate: {d.status} - {d.label}")
    print("local_research_allowed:", d.local_research_allowed,
          "| demo_eligible:", d.demo_eligible)
    if not d.local_research_allowed:
        raise SystemExit("source is not permitted even for local research")

    outdir = "out/pga_baseline_20260911"
    os.makedirs(outdir, exist_ok=True)
    report = {"schema": "fairwayos-pga-baseline/v1",
              "research_only": True, "ground_truth": False,
              "production_eligible": False, "demo_eligible": d.demo_eligible,
              "source": {"path": SRC, "sha256": d.sha256, "gate_status": d.status,
                         "rights": d.rights},
              "intervals_declared_before_tuning": INTERVALS,
              "panel_note": "dual-panel clip; left panel only via crop " + CROP,
              "results": [run(k, v, outdir) for k, v in INTERVALS.items()]}
    path = os.path.join(outdir, "baseline_report.json")
    json.dump(report, open(path, "w"), indent=1)
    print(json.dumps(report["results"], indent=1)[:2600])
    print("\nwrote", path)


if __name__ == "__main__":
    main()
