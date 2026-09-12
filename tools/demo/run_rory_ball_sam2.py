"""Pivot experiment: run SAM2.1 video segmentation on the RORY ball.

The BootsTAPIR point-track route was falsified on this source by independent
review (raw coordinates drift ~39/101/117 px from the sparse AI reference points
at native f3068/f3080/f3090, so it is not an over-strict visibility gate). This
runs a different model class from the SAME reviewed seed, on the SAME bounded
interval, and scores every emission against the reserved AI reference checks.

Reproducible:  .venv/bin/python3 tools/demo/run_rory_ball_sam2.py

Writes out/rory_ball_sam2/ -- the out/rory_three_target/ baseline is immutable.
"""
import hashlib, json, math, os, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from ghostcaddie.upload.adapters import predictive_frames, seed_box_from_point
from ghostcaddie.upload.runtimes import RuntimeRegistry

SRC = "/tmp/fairway-three-target-sources/downloads/6083133193001.mp4"
SHA = "a699c9970a56029280a00cd3fa3eb13da1c130043aab7bf712d5c628aa9ab7e7"
OUT = os.path.join(REPO, "out", "rory_ball_sam2")
START, END = 3062, 3105           # seed frame onward; f3110 WITHHELD
SEED = {"frame": 3062, "point_xy": [836.0, 550.0], "uncertainty_radius_px": 7}
# reserved AI checks -- NOT seeds, NOT fitting targets, NOT ground truth
CHECKS = {3068: [821, 477], 3080: [805, 358], 3090: [793, 289]}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    assert sha(SRC) == SHA, "SOURCE HASH MISMATCH - refusing"
    os.makedirs(OUT, exist_ok=True)
    spec = RuntimeRegistry.default().require("ball_sam2")
    req = {"video": SRC, "source_sha256": SHA, "sampling_step": 1,
           "max_frames": END - START + 1, "frame_start": START, "frame_end": END,
           "seed": SEED}
    rp = os.path.join(OUT, "_req.json")
    json.dump(req, open(rp, "w"))
    print(f"-- ball_sam2 via {spec.interpreter}  f{START}-{END}", flush=True)
    p = subprocess.run([spec.interpreter, "-m", "ghostcaddie.upload.worker",
                        "ball_sam2", rp], capture_output=True, text=True,
                       cwd=REPO, timeout=3600)
    try:
        out = json.loads((p.stdout or "").strip().splitlines()[-1])
    except Exception:
        out = {"ok": False, "error": (p.stderr or p.stdout or "")[-2000:]}
    os.remove(rp)

    scored = {}
    if out.get("ok"):
        by = {r["source_frame"]: r for r in out["records"]}
        for f, ref in CHECKS.items():
            r = by.get(f)
            scored[f] = {
                "ai_reference_xy": ref, "emitted": bool(r and r.get("visible")),
                "emitted_xy": (r or {}).get("point_xy"),
                "raw_centroid_xy": (r or {}).get("raw_centroid_xy"),
                "state": (r or {}).get("state"),
                "px_from_reference": (
                    round(math.dist(r["point_xy"], ref), 2)
                    if r and r.get("point_xy") else None),
                "raw_px_from_reference": (
                    round(math.dist(r["raw_centroid_xy"], ref), 2)
                    if r and r.get("raw_centroid_xy") else None)}
        n_pred = predictive_frames(out["records"])
        print(f"   predictive frames excluding seed: {n_pred}/{len(out['records'])-1}")
        for f, s in sorted(scored.items()):
            print(f"   check f{f}: state={s['state']} emitted={s['emitted']} "
                  f"px_from_ai_ref={s['px_from_reference']} "
                  f"(raw {s['raw_px_from_reference']})")
    else:
        print("   BLOCKER:", out.get("error", "")[:800])

    json.dump({"source": {"path": SRC, "sha256": SHA},
               "interval_native_frames": [START, END], "withheld": [3110],
               "seed": SEED,
               "seed_prompt_box_xyxy": seed_box_from_point(
                   SEED["point_xy"], SEED["uncertainty_radius_px"]),
               "seed_provenance": "AI-reviewed assisted initialisation "
                                  "(/tmp/fairway-rory-seed-review); not ground truth",
               "method": "SAM2.1 tiny video object segmentation, box prompt derived "
                         "from the reviewed seed point and its published 7 px "
                         "uncertainty radius; no threshold fitted to this run",
               "supersedes": "BootsTAPIR point track, falsified on this source by "
                             "independent raw-coordinate review",
               "reference_checks": scored,
               "reference_check_status": "AI pseudo-labels reserved as checks; NOT "
                                         "ground truth and NOT used to fit anything",
               "predictive_frames_excluding_seed": (
                   predictive_frames(out["records"]) if out.get("ok") else 0),
               "research_only": True, "pseudo_label": True,
               "ground_truth": False, "production_eligible": False,
               "result": out},
              open(os.path.join(OUT, "raw_results.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "raw_results.json"))


if __name__ == "__main__":
    main()
