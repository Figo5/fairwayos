"""Frozen candidate-ranking + abstention evaluation.

Ranking rule and thresholds are FROZEN before the fresh interval is touched.
Reports incorrect tracks and abstention behaviour against reviewed native-frame
labels (evaluation references, never ground truth).
"""
from __future__ import annotations

import json, math, os, sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from ghostcaddie.video.pga_sliced_ball import (
    TEMPORAL_DEFAULTS, sliced_candidates, select_with_abstention)
from ghostcaddie.video.pga_source_gate import evaluate_source, REJECTED

SRC = "out/youtube_fullswing/f3kTTMZlxds.mp4"
MATCH_PX = 12.0


def frames(a, b, step):
    cap = cv2.VideoCapture(SRC); out = {}; i = 0
    while True:
        ok, f = cap.read()
        if not ok or i > b: break
        if i >= a and (i - a) % step == 0: out[i] = f[0:720, 0:640].copy()
        i += 1
    cap.release(); return out


def evaluate(labels, a, b, step, params=None):
    imgs = frames(a, b, step)
    keys = sorted(imgs)
    cands = {k: sliced_candidates(imgs[k]) for k in keys}
    hit = incorrect = abstain_when_visible = 0
    correct_abstain = wrong_emit = 0
    errs = []
    per = []
    for n, k in enumerate(keys):
        L = labels.get(str(k))
        if L is None: continue
        if L["state"] == "ambiguous":
            per.append({"frame": k, "outcome": "excluded_ambiguous"})
            continue   # an unconfirmed frame scores neither way
        prev = cands[keys[n - 1]] if n > 0 else []
        nxt = cands[keys[n + 1]] if n + 1 < len(keys) else []
        sel = select_with_abstention(prev, cands[k], nxt, params)
        visible = L["state"] in ("visible_on_tee", "visible_flight")
        if visible:
            if sel is None:
                abstain_when_visible += 1; per.append({"frame": k, "outcome": "abstained_but_visible"})
            else:
                d = math.hypot(sel["x"] - L["xy"][0], sel["y"] - L["xy"][1])
                if d <= MATCH_PX:
                    hit += 1; errs.append(d); per.append({"frame": k, "outcome": "hit", "err_px": round(d, 2)})
                else:
                    incorrect += 1
                    per.append({"frame": k, "outcome": "incorrect_track",
                                "err_px": round(d, 1), "picked": [round(sel["x"], 1), round(sel["y"], 1)]})
        else:
            if sel is None: correct_abstain += 1; per.append({"frame": k, "outcome": "correct_abstain"})
            else: wrong_emit += 1; per.append({"frame": k, "outcome": "emitted_when_absent",
                                               "picked": [round(sel["x"], 1), round(sel["y"], 1)]})
    vis = hit + incorrect + abstain_when_visible
    nv = correct_abstain + wrong_emit
    errs.sort()
    return {"frames_visible": vis, "frames_not_visible": nv,
            "hits": hit, "incorrect_tracks": incorrect,
            "abstained_when_visible": abstain_when_visible,
            "correct_abstentions": correct_abstain, "emitted_when_absent": wrong_emit,
            "precision_when_emitting": round(hit / (hit + incorrect + wrong_emit), 3)
                                       if (hit + incorrect + wrong_emit) else None,
            "recall_visible": round(hit / vis, 3) if vis else None,
            "median_err_px": round(errs[len(errs) // 2], 2) if errs else None,
            "max_err_px": round(errs[-1], 2) if errs else None,
            "per_frame": per}


def main():
    d = evaluate_source(SRC)
    if d.status == REJECTED or not d.local_research_allowed:
        raise SystemExit("source not permitted for local research")
    print(f"source gate: {d.status} | demo_eligible={d.demo_eligible}")
    which = sys.argv[1] if len(sys.argv) > 1 else "calibration"
    E = "out/pga_experiment_20260911"
    if which == "calibration":
        sets = [("eval", json.load(open(f"{E}/reviewed_labels.json")), 341, 517),
                ("heldout_2", json.load(open(f"{E}/reviewed_labels_heldout2.json")), 2954, 3130)]
    else:
        sets = [("fresh_test_3", json.load(open(f"{E}/reviewed_labels_test3.json")), 3936, 4112)]
    out = {"schema": "fairwayos-pga-ranking/v1", "research_only": True,
           "ground_truth": False, "production_eligible": False, "demo_eligible": False,
           "frozen_params": dict(TEMPORAL_DEFAULTS), "match_radius_px": MATCH_PX,
           "results": {}}
    for name, doc, a, b in sets:
        r = evaluate(doc["labels"], a, b, doc["step"])
        out["results"][name] = r
        print(f"\n== {name}")
        for k in ("frames_visible", "frames_not_visible", "hits", "incorrect_tracks",
                  "abstained_when_visible", "correct_abstentions", "emitted_when_absent",
                  "recall_visible", "precision_when_emitting", "median_err_px", "max_err_px"):
            print(f"   {k:26s} {r[k]}")
    json.dump(out, open(f"{E}/ranking_{which}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
