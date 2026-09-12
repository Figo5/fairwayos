"""Video in, results out. Automatic only: no seed, no annotation, no reference.

  .venv/bin/python3 tools/demo/video_to_results.py <video> [start] [end]

Runs every target that can start unattended, freezes the point observations, and
runs the measurement layer over them. Targets with no automatic detector produce
NO observations and are reported unavailable with the exact missing piece named,
so a reader can tell "we looked and found nothing" from "we cannot look yet".

Image-space speed (px/s) is computed from real frame timestamps. Metric speed is
REFUSED unless a spatial calibration and the real capture rate are both present;
for broadcast clips like this one they are not.
"""
import hashlib, json, os, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from ghostcaddie.upload.observations import (
    MetricSpeedUnavailable, Timebase, image_speed_px_s, metric_speed_mps,
    observations_from_records)
from ghostcaddie.upload.runtimes import RuntimeRegistry

# What each target needs before it can run with NO human assistance.
AUTOMATIC = {
    "body": {"runs_unattended": True,
             "method": "MoveNet SinglePose Lightning, TFLite via ai-edge-litert"},
    "clubhead": {"runs_unattended": False,
                 "missing": "a trained golf CLUBHEAD detector checkpoint. The "
                            "SAM2.1 route is a promptable segmenter: it needs a "
                            "box to start and is not a detector."},
    # CORRECTION to an earlier overstatement of mine: a golf-ball detector DOES
    # exist and DOES fire on this footage. It is not wired in here because it is
    # far too weak to be a tracking layer, and its dataset rights are unresolved.
    "ball": {"runs_unattended": False,
             "missing": "a golf BALL detector good enough to track with. One "
                        "exists (ONNX, declares AGPL-3.0, names={0: golf_ball}) "
                        "and on this interval it hit the STATIONARY ball at "
                        "address on f3055-f3060 and the MOVING ball on f3075 and "
                        "f3079 (both visually confirmed), against 218 predictions "
                        "that were not the ball. 2 moving detections in 51 frames "
                        "is not a track. Its training-dataset rights are also "
                        "unresolved and AGPL-3.0 carries its own obligations. "
                        "BootsTAPIR is a point tracker, not a detector: it needs "
                        "a starting point and drifts off this ball once given "
                        "one."},
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate,nb_frames,width,height", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"playback_fps": float(num) / float(den),
            "playback_fps_exact": s["r_frame_rate"],
            "frames": int(s.get("nb_frames") or 0),
            "width": int(s["width"]), "height": int(s["height"])}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    video = os.path.abspath(sys.argv[1])
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    end = int(sys.argv[3]) if len(sys.argv) > 3 else start + 50
    sha = sha256(video)
    meta = probe(video)
    fps = meta["playback_fps"]
    print(f"source {os.path.basename(video)} sha256 {sha[:16]}...  "
          f"{meta['width']}x{meta['height']} @ {meta['playback_fps_exact']} "
          f"playback  frames {start}-{end}", flush=True)

    reg = RuntimeRegistry.default()
    observations, targets = {}, {}
    for name, spec_info in AUTOMATIC.items():
        if not spec_info["runs_unattended"]:
            targets[name] = {"state": "unavailable",
                             "automatic": False,
                             "observations": 0,
                             "blocker": spec_info["missing"]}
            print(f"-- {name}: UNAVAILABLE - {spec_info['missing']}", flush=True)
            continue
        try:
            spec = reg.require(name)
        except Exception as e:
            targets[name] = {"state": "unavailable", "automatic": True,
                             "observations": 0, "blocker": f"interpreter: {e}"}
            continue
        req = {"video": video, "source_sha256": sha, "sampling_step": 1,
               "max_frames": end - start + 1, "frame_start": start,
               "frame_end": end, "seed": None}
        rp = os.path.join(REPO, "out", f"_auto_req_{name}.json")
        os.makedirs(os.path.dirname(rp), exist_ok=True)
        json.dump(req, open(rp, "w"))
        p = subprocess.run([spec.interpreter, "-m", "ghostcaddie.upload.worker",
                            name, rp], capture_output=True, text=True, cwd=REPO,
                           timeout=1800)
        os.remove(rp)
        try:
            out = json.loads((p.stdout or "").strip().splitlines()[-1])
        except Exception:
            out = {"ok": False, "error": (p.stderr or "")[-300:]}
        if not out.get("ok"):
            targets[name] = {"state": "unavailable", "automatic": True,
                             "observations": 0, "blocker": out.get("error", "")[:200]}
            print(f"-- {name}: UNAVAILABLE - {out.get('error','')[:160]}", flush=True)
            continue
        recs = out["records"]
        if name == "body":
            # the body anchor is the mid-hip, the most stable automatic point
            pts = []
            for r in recs:
                kp = {k["name"]: k for k in r.get("keypoints", [])}
                l, rr = kp.get("hip_l"), kp.get("hip_r")
                if l and rr and l["visible"] and rr["visible"]:
                    pts.append({"source_frame": r["source_frame"], "visible": True,
                                "point_xy": [(l["x"] + rr["x"]) / 2,
                                             (l["y"] + rr["y"]) / 2],
                                "source_sha256": r["source_sha256"],
                                "confidence": round(min(l["score"], rr["score"]), 4)})
            recs = pts
        obs = observations_from_records(recs, target=name, fps=fps,
                                        method=spec_info["method"],
                                        source_sha256=sha)
        observations[name] = obs
        targets[name] = {"state": "observed" if obs else "unavailable",
                         "automatic": True, "observations": len(obs),
                         "method": spec_info["method"],
                         "anchor": "mid-hip" if name == "body" else None}
        print(f"-- {name}: {len(obs)} automatic observations", flush=True)

    # measurement layer over frozen observations
    timebase = Timebase(real_seconds_per_playback_second=None,
                        method=f"file declares playback {meta['playback_fps_exact']} "
                               f"and carries no capture-rate metadata")
    speeds, mph_status = {}, {}
    for name, obs in observations.items():
        rows = []
        for a, b in zip(obs, obs[1:]):
            s = image_speed_px_s(a, b)
            rows.append({"from_frame": a.frame_index, "to_frame": b.frame_index,
                         "distance_px": round(s.distance_px, 3),
                         "dt_seconds": round(s.dt_seconds, 6),
                         "px_per_second": round(s.px_per_second, 2),
                         "frame_gap": s.frame_gap, "spans_gap": s.spans_gap})
        speeds[name] = rows
        try:
            metric_speed_mps(obs[0], obs[1], calibration=None, timebase=timebase)
            mph_status[name] = "unexpectedly produced a metric speed"
        except (MetricSpeedUnavailable, IndexError) as e:
            mph_status[name] = str(e) if isinstance(e, MetricSpeedUnavailable) \
                else "fewer than two observations"

    outdir = os.path.join(REPO, "out", "video_to_results")
    os.makedirs(outdir, exist_ok=True)
    doc = {"source": {"path": video, "sha256": sha, **meta},
           "interval_native_frames": [start, end],
           "mode": "FULLY AUTOMATIC: no seed, no annotation, no reference "
                   "coordinate, no operator click",
           "targets": targets,
           "observations": {k: [o.to_dict() for o in v]
                            for k, v in observations.items()},
           "image_space_speed_px_per_second": speeds,
           "metric_speed": {"available": False, "per_target_reason": mph_status,
                            "requires": ["a spatial calibration (metres per pixel "
                                         "valid for that target's plane/depth)",
                                         "the real capture rate (playback rate is "
                                         "not capture rate)"]},
           "research_only": True, "pseudo_label": True,
           "ground_truth": False, "production_eligible": False}
    p = os.path.join(outdir, "results.json")
    json.dump(doc, open(p, "w"), indent=1)
    print("\nautomatic targets with observations:",
          [k for k, v in targets.items() if v["observations"]])
    print("blocked targets:", {k: v.get("blocker", "")[:80]
                               for k, v in targets.items() if not v["observations"]})
    print("metric speed available:", doc["metric_speed"]["available"])
    print("wrote", p)


if __name__ == "__main__":
    sys.exit(main() or 0)
