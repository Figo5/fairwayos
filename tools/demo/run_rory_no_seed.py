"""Run every target with NO seed at all, on the bounded Rory interval.

This is the honest automatic baseline: what the system finds on its own, with no
operator click, no reviewed seed, and no reference coordinate smuggled in as an
initialisation. A target that cannot start unattended must report UNAVAILABLE
here -- that is the correct outcome, not a failure to paper over.

Reproducible:  .venv/bin/python3 tools/demo/run_rory_no_seed.py
Writes out/rory_no_seed/. The approved deliverables are not touched.
"""
import hashlib, json, os, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from ghostcaddie.upload.runtimes import RuntimeRegistry

SRC = "/tmp/fairway-three-target-sources/downloads/6083133193001.mp4"
SHA = "a699c9970a56029280a00cd3fa3eb13da1c130043aab7bf712d5c628aa9ab7e7"
OUT = os.path.join(REPO, "out", "rory_no_seed")
START, END = 3055, 3105


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
    results, summary = {}, {}
    for target in ("body", "clubhead", "ball", "ball_sam2"):
        try:
            spec = reg.require(target)
        except Exception as e:
            results[target] = {"ok": False, "error": f"interpreter: {e}"}
            summary[target] = "unavailable (interpreter)"
            print(f"-- {target}: UNAVAILABLE (interpreter) {e}", flush=True)
            continue
        req = {"video": SRC, "source_sha256": SHA, "sampling_step": 1,
               "max_frames": END - START + 1, "frame_start": START,
               "frame_end": END, "seed": None}          # explicitly NO seed
        rp = os.path.join(OUT, f"_req_{target}.json")
        json.dump(req, open(rp, "w"))
        print(f"-- {target} via {spec.interpreter} (no seed)", flush=True)
        p = subprocess.run([spec.interpreter, "-m", "ghostcaddie.upload.worker",
                            target, rp], capture_output=True, text=True,
                           cwd=REPO, timeout=1800)
        try:
            out = json.loads((p.stdout or "").strip().splitlines()[-1])
        except Exception:
            out = {"ok": False, "error": (p.stderr or p.stdout or "")[-400:]}
        os.remove(rp)
        results[target] = out
        if out.get("ok"):
            obs = sum(1 for r in out["records"]
                      if r.get("visible") or r.get("visible_keypoint_count", 0) > 0)
            summary[target] = f"observed {obs}/{len(out['records'])} frames"
            print(f"   AUTOMATIC: {summary[target]}")
        else:
            summary[target] = f"unavailable: {out.get('error','')[:160]}"
            print(f"   UNAVAILABLE: {out.get('error','')[:200]}")

    doc = {"source": {"path": SRC, "sha256": SHA},
           "interval_native_frames": [START, END],
           "seeds": None,
           "what_this_run_is": "fully automatic: no seed, no operator click, no "
                               "reference coordinate used as initialisation",
           "summary": summary,
           "automatic_targets": [t for t, s in summary.items()
                                 if s.startswith("observed")],
           "unavailable_targets": [t for t, s in summary.items()
                                   if not s.startswith("observed")],
           "research_only": True, "pseudo_label": True,
           "ground_truth": False, "production_eligible": False,
           "results": results}
    json.dump(doc, open(os.path.join(OUT, "raw_results.json"), "w"), indent=1)
    print("\nautomatic:", doc["automatic_targets"])
    print("unavailable:", doc["unavailable_targets"])
    print("wrote", os.path.join(OUT, "raw_results.json"))


if __name__ == "__main__":
    main()
