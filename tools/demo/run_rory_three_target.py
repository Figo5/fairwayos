"""Run body + clubhead + ball on a bounded Rory segment, with reviewed seeds.

Reproducible:  .venv/bin/python3 tools/demo/run_rory_three_target.py

Each target executes in its own interpreter (no torch in this process). Seeds are
AI-reviewed assisted initialisation bound to the source hash; f3110 is WITHHELD.
"""
import hashlib, json, os, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from ghostcaddie.upload.runtimes import RuntimeRegistry

SRC = "/tmp/fairway-three-target-sources/downloads/6083133193001.mp4"
SHA = "a699c9970a56029280a00cd3fa3eb13da1c130043aab7bf712d5c628aa9ab7e7"
OUT = os.path.join(REPO, "out", "rory_three_target")
START, END = 3055, 3105          # contact + follow-through; f3110 WITHHELD
SEEDS = {
    "clubhead": {"frame": 3058, "box_xyxy": [812.0, 581.0, 858.0, 611.0]},
    "ball":     {"frame": 3062, "point_xy": [836.0, 550.0]},
}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def main():
    assert sha(SRC) == SHA, "SOURCE HASH MISMATCH - refusing"
    os.makedirs(OUT, exist_ok=True)
    reg = RuntimeRegistry.default()
    results = {}
    for target in ("body", "clubhead", "ball"):
        spec = reg.require(target)
        req = {"video": SRC, "source_sha256": SHA, "sampling_step": 1,
               "max_frames": END - START + 1, "frame_start": START, "frame_end": END,
               "seed": SEEDS.get(target)}
        rp = os.path.join(OUT, f"_req_{target}.json")
        json.dump(req, open(rp, "w"))
        print(f"-- {target} via {spec.interpreter}", flush=True)
        p = subprocess.run([spec.interpreter, "-m", "ghostcaddie.upload.worker",
                            target, rp], capture_output=True, text=True,
                           cwd=REPO, timeout=1800)
        try:
            out = json.loads((p.stdout or "").strip().splitlines()[-1])
        except Exception:
            out = {"ok": False, "error": (p.stderr or p.stdout or "")[-500:]}
        results[target] = out
        if out.get("ok"):
            obs = sum(1 for r in out["records"]
                      if r.get("visible") or r.get("visible_keypoint_count", 0) > 0)
            print(f"   OK  {obs}/{len(out['records'])} frames with observation")
        else:
            print(f"   BLOCKER: {out.get('error','')[:300]}")
        os.remove(rp)
    json.dump({"source": {"path": SRC, "sha256": SHA},
               "interval_native_frames": [START, END],
               "withheld": [3110],
               "seeds": SEEDS,
               "seed_provenance": "AI-reviewed assisted initialisation "
                                  "(/tmp/fairway-rory-seed-review); not ground truth",
               "research_only": True, "pseudo_label": True,
               "ground_truth": False, "production_eligible": False,
               "results": results},
              open(os.path.join(OUT, "raw_results.json"), "w"), indent=1)
    print("\nwrote", os.path.join(OUT, "raw_results.json"))

if __name__ == "__main__":
    main()
