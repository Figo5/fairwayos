"""Evaluate sliced vs global ball candidates against REVIEWED native-frame labels.

Protocol is the frozen one in out/pga_experiment_20260911/FROZEN_BASELINE.json.
Reports false positives and localization error, NOT coverage alone.
Labels are evaluation references (ai_assisted_native_inspection), never ground
truth, and are never used to seed or tune the detectors.
"""
from __future__ import annotations

import json
import math
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from ghostcaddie.video.pga_sliced_ball import sliced_candidates, global_candidates
from ghostcaddie.video.pga_source_gate import evaluate_source, REJECTED

SRC = "out/youtube_fullswing/f3kTTMZlxds.mp4"
EXP = "out/pga_experiment_20260911"
MATCH_PX = 12.0


def load_frames(a, b, step):
    cap = cv2.VideoCapture(SRC)
    imgs, i = {}, 0
    while True:
        ok, f = cap.read()
        if not ok or i > b:
            break
        if i >= a and (i - a) % step == 0:
            imgs[i] = f[0:720, 0:640].copy()
        i += 1
    cap.release()
    return imgs


def score(name, detector, imgs, labels):
    hit1 = miss = fp1 = fp_nv = 0
    errs, cand_counts = [], []
    visible = notvis = 0
    per_frame = []
    for f, img in sorted(imgs.items()):
        lab = labels.get(str(f))
        if lab is None:
            continue
        cands = detector(img)
        cand_counts.append(len(cands))
        top = cands[0] if cands else None
        if lab["state"] in ("visible_on_tee", "visible_flight"):
            visible += 1
            lx, ly = lab["xy"]
            d1 = math.hypot(top["x"] - lx, top["y"] - ly) if top else None
            if d1 is not None and d1 <= MATCH_PX:
                hit1 += 1
                errs.append(d1)
            else:
                miss += 1
                if top is not None:
                    fp1 += 1
            per_frame.append({"frame": f, "state": lab["state"],
                              "top1_err_px": None if d1 is None else round(d1, 1),
                              "n_candidates": len(cands)})
        else:
            notvis += 1
            if top is not None:
                fp_nv += 1
            per_frame.append({"frame": f, "state": "not_visible",
                              "top1_present": top is not None,
                              "n_candidates": len(cands)})
    errs.sort()
    return {
        "method": name,
        "frames_scored": visible + notvis,
        "frames_ball_visible": visible,
        "frames_ball_not_visible": notvis,
        "recall_at_1": round(hit1 / visible, 3) if visible else None,
        "hits_at_1": hit1,
        "misses": miss,
        "false_positive_top1_when_visible": fp1,
        "false_positive_when_not_visible": fp_nv,
        "median_localization_error_px": round(errs[len(errs) // 2], 2) if errs else None,
        "max_localization_error_px": round(errs[-1], 2) if errs else None,
        "median_candidates_per_frame": sorted(cand_counts)[len(cand_counts) // 2]
                                       if cand_counts else None,
        "per_frame": per_frame,
    }


def main():
    d = evaluate_source(SRC)
    if d.status == REJECTED or not d.local_research_allowed:
        raise SystemExit(f"source not permitted for local research: {d.status}")
    print(f"source gate: {d.status} | local_research_allowed={d.local_research_allowed} "
          f"| demo_eligible={d.demo_eligible}")

    frozen = json.load(open(f"{EXP}/FROZEN_BASELINE.json"))
    labels_doc = json.load(open(f"{EXP}/reviewed_labels.json"))
    assert labels_doc["IS_GROUND_TRUTH"] is False
    a, b = labels_doc["interval"]
    step = labels_doc["step"]
    imgs = load_frames(a, b, step)

    results = [
        score("global_threshold_baseline", lambda im: global_candidates(im), imgs,
              labels_doc["labels"]),
        score("sliced_local_contrast", lambda im: sliced_candidates(im), imgs,
              labels_doc["labels"]),
    ]
    out = {"schema": "fairwayos-pga-sliced-experiment/v1",
           "research_only": True, "ground_truth": False,
           "production_eligible": False, "demo_eligible": False,
           "frozen_baseline_commit": frozen["git_commit"],
           "detector_config_sha256": frozen["detector_config_sha256"],
           "match_radius_px": MATCH_PX,
           "labels": {"file": f"{EXP}/reviewed_labels.json",
                      "evidence_class": labels_doc["evidence_class"],
                      "is_ground_truth": False},
           "interval": {"window": [a, b], "step": step,
                        "role": "EVAL interval (tuning permitted); held-out and "
                                "reserved intervals untouched by this experiment"},
           "results": results}
    json.dump(out, open(f"{EXP}/sliced_experiment.json", "w"), indent=1)
    for r in results:
        print(f"\n== {r['method']}")
        for k in ("frames_ball_visible", "frames_ball_not_visible", "recall_at_1",
                  "hits_at_1", "misses", "false_positive_top1_when_visible",
                  "false_positive_when_not_visible", "median_localization_error_px",
                  "max_localization_error_px", "median_candidates_per_frame"):
            print(f"   {k:38s} {r[k]}")


if __name__ == "__main__":
    main()
