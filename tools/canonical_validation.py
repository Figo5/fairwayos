#!/usr/bin/env python3
"""Run the canonical workflow over a local clip directory and summarise it.

This is a REGRESSION/FAILURE-OBSERVATION sweep, not a benchmark. The corpus has
no ground truth, no held-out split, and has already been consumed during
development, so nothing here supports a generalization claim. What it is good for
is noticing that behaviour changed, and seeing which clips the workflow declines.

Every clip in the directory is included. Clips are never dropped for being hard:
excluding the ones that hurt a number is selection on the metric being reported.

    .venv/bin/python3 tools/canonical_validation.py CLIP_DIR --out OUTDIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ghostcaddie.video import canonical     # noqa: E402

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv"}


def summarise(run_dir: Path) -> dict:
    result = canonical.AnalysisResult.from_dict(
        json.loads((run_dir / "analysis.json").read_text()))
    diag = json.loads((run_dir / "diagnostics.json").read_text())
    ident = diag.get("identity", {})
    body = result.layers["body"]
    reasons = {}
    for rec in diag.get("per_frame", []):
        if not rec.get("attributed_to_golfer"):
            reasons[rec["rejection_reason"].split(";")[0]] = \
                reasons.get(rec["rejection_reason"].split(";")[0], 0) + 1
    top = sorted(reasons.items(), key=lambda kv: -kv[1])[:1]
    # A clip declined at the identity gate never reaches per-frame attribution, so
    # its reason lives on the layer, not in the frame records.
    gate_reason = (body.limitations[0] if not top and body.limitations
                   and not body.has_observations else "")
    return {
        "clip": result.source.display_name,
        "sha256": result.source.sha256[:12],
        "resolution": f"{result.source.width}x{result.source.height}",
        "sampled_frames": result.source.sampled_frame_count,
        "body_state": body.state,
        "identity_method": body.provenance.get("identity_method"),
        "second_person_found": bool((ident.get("second_person") or {}).get("found")),
        "pose_coverage": body.coverage.get("fraction"),
        "jitter_px_median": ident.get("jitter_px_median"),
        "jitter_fraction_of_width": ident.get("jitter_fraction_of_width"),
        "camera_jitter_high": ident.get("camera_jitter_high"),
        "renderer_ok": bool(result.outputs.get("annotated_video_written")),
        "supported_metrics": len(result.metrics["body_2d"]["supported"]),
        "notable_failure_mode": (f"{top[0][0]} ({top[0][1]} frames)" if top
                                 else (gate_reason.split(". ")[0] or "none")),
    }


def markdown(rows: list, corpus: str) -> str:
    head = ("| clip | res | frames | body | identity | 2nd person | coverage | "
            "jitter px (frac) | render | metrics | main rejection |")
    sep = "|" + "---|" * 11
    out = [
        "# Canonical workflow validation sweep",
        "",
        f"Corpus: `{corpus}` — {len(rows)} clips, all included.",
        "",
        "**This is not a benchmark.** The corpus has no ground truth, no held-out "
        "split, and was already used during development. These numbers detect drift "
        "and show which clips the workflow declines; they support no claim about "
        "generalization to unseen footage.",
        "",
        "`jitter px` is the median global camera translation between sampled frames, "
        "in native pixels, with the fraction of frame width in parentheses. "
        "`coverage` is the share of sampled frames carrying a pose attributed to the "
        "golfer.",
        "",
        head, sep,
    ]
    for r in rows:
        out.append(
            f"| `{r['clip']}` | {r['resolution']} | {r['sampled_frames']} | "
            f"**{r['body_state']}** | {r['identity_method']} | "
            f"{'yes' if r['second_person_found'] else 'no'} | {r['pose_coverage']} | "
            f"{r['jitter_px_median']} ({r['jitter_fraction_of_width']})"
            f"{' ⚠' if r['camera_jitter_high'] else ''} | "
            f"{'ok' if r['renderer_ok'] else 'FAILED'} | {r['supported_metrics']} | "
            f"{r['notable_failure_mode']} |")
    declined = [r for r in rows if r["body_state"] in
                ("ambiguous", "unavailable", "failed")]
    jittery = sorted(rows, key=lambda r: -(r["jitter_px_median"] or 0))[:2]
    out += ["", "## Clips the workflow declined", ""]
    out += ([f"- `{r['clip']}` — **{r['body_state']}**: {r['notable_failure_mode']} "
             f"(camera {r['jitter_px_median']} px/frame)" for r in declined]
            or ["- none"])
    out += ["", "## Highest camera motion", "",
            "Included, not excluded. Dropping the clips that hurt a number is "
            "selection on the metric being reported.", ""]
    out += [f"- `{r['clip']}` — {r['jitter_px_median']} px between sampled frames "
            f"({r['jitter_fraction_of_width']} of frame width), body "
            f"**{r['body_state']}**, coverage {r['pose_coverage']}" for r in jittery]
    states = {}
    for r in rows:
        states[r["body_state"]] = states.get(r["body_state"], 0) + 1
    out += ["", "## Summary", "",
            "- body states: " + ", ".join(f"{v}× {k}" for k, v in sorted(states.items())),
            f"- renderer produced an annotated video on "
            f"{sum(1 for r in rows if r['renderer_ok'])}/{len(rows)} clips",
            f"- clips flagged as camera-jitter-dominated: "
            f"{sum(1 for r in rows if r['camera_jitter_high'])}",
            "- ball and clubhead are UNAVAILABLE on every clip, by design.", ""]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clips", type=Path, help="Directory of local clips.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reuse", action="store_true",
                    help="Summarise existing run directories instead of rerunning.")
    a = ap.parse_args(argv)

    videos = sorted(p for p in a.clips.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        raise SystemExit(f"no clips found in {a.clips}")
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in videos:
        run_dir = a.out / v.stem
        if not (a.reuse and (run_dir / "analysis.json").is_file()):
            print(f"[run] {v.name}", file=sys.stderr)
            try:
                canonical.analyze_video(str(v), str(run_dir))
            except canonical.AnalysisError as exc:
                rows.append({"clip": v.name, "sha256": "", "resolution": "?",
                             "sampled_frames": 0, "body_state": "failed",
                             "identity_method": "-", "second_person_found": False,
                             "pose_coverage": 0.0, "jitter_px_median": None,
                             "jitter_fraction_of_width": None,
                             "camera_jitter_high": False, "renderer_ok": False,
                             "supported_metrics": 0,
                             "notable_failure_mode": str(exc)[:80]})
                continue
        rows.append(summarise(run_dir))
    (a.out / "validation_summary.json").write_text(
        json.dumps({"corpus": str(a.clips), "not_a_benchmark": True,
                    "ground_truth": False, "held_out": False, "rows": rows},
                   indent=2, sort_keys=True) + "\n")
    report = markdown(rows, str(a.clips))
    (a.out / "validation_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
