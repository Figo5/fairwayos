"""Reference-free association over the FROZEN golf-ball candidates.

  .venv/bin/python3 tools/demo/associate_ball_candidates.py

Reads every post-NMS candidate the detector produced (the 0.01 raw floor, as
frozen -- no threshold is changed here), and preserves one coherent path as an
unvalidated proposal. Nothing is seeded: no reference coordinate, no crop, no
operator click. Without independent identity qualification, accepted ball
observations abstain on every frame. Frames whose candidates cannot continue the
proposal emit NOTHING.

The policy is written to disk and hashed BEFORE the association runs, and this
script performs no evaluation against any reference. Selection quality is for an
independent reviewer to judge, not for this script to assert.
"""
import hashlib, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from ghostcaddie.tracking.candidates import (
    AssociationPolicy, associate_proposals, parse_candidates)

RAW = "/tmp/fairway-auto-ball/out_corrected/raw_detections.json"
RAW_SHA = "dcd6e340944eefa32acd7442ebd24e0216e3f17dc2af32ce0af5ed2a00bbaaa7"
OUT = os.path.join(REPO, "out", "ball_association")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    got = sha256_file(RAW)
    if got != RAW_SHA:
        print(f"FROZEN INPUT MISMATCH\n  expected {RAW_SHA}\n  got      {got}")
        return 1
    os.makedirs(OUT, exist_ok=True)

    # ---- freeze the policy BEFORE looking at any result ----
    policy = AssociationPolicy()
    pol_path = os.path.join(OUT, "frozen_policy.json")
    pol_json = json.dumps(
        {"policy": policy.to_dict(),
         "frozen_before_run": True,
         "no_threshold_sweep": "this policy is run once; no value is tuned "
                               "against any reference",
         "detector_threshold_unchanged": "candidates are read at the frozen 0.01 "
                                         "raw floor exactly as produced",
         "initialisation": "none: the path start is chosen by the optimiser, not "
                           "by a coordinate",
         "interpolation": "none: a frame with no selected candidate emits null"},
        indent=1, sort_keys=True)
    open(pol_path, "w").write(pol_json)
    pol_sha = hashlib.sha256(pol_json.encode()).hexdigest()
    print(f"frozen policy sha256 {pol_sha}\n  {policy.to_dict()}", flush=True)

    raw = json.load(open(RAW))
    frames, rejected_total, candidate_total = {}, 0, 0
    for rec in raw["records"]:
        f = int(rec["source_frame"])
        rows = [{"frame": f, "box_xyxy": d["xyxy"], "score": d["confidence"],
                 "candidate_index": i}
                for i, d in enumerate(rec.get("detections", []))]
        kept, rejected = parse_candidates(rows)
        frames[f] = kept
        rejected_total += rejected
        candidate_total += len(kept)

    result = associate_proposals(frames, policy)
    proposed_track = result.proposed_by_frame
    accepted_track = result.accepted_by_frame
    proposed = {f: c for f, c in proposed_track.items() if c is not None}
    accepted = {f: c for f, c in accepted_track.items() if c is not None}
    print(f"candidates read {candidate_total} over {len(frames)} frames "
          f"({rejected_total} invalid boxes rejected)")
    print(f"frames with a proposed candidate: {len(proposed)}/{len(frames)}")
    print(f"frames with an accepted ball observation: {len(accepted)}/{len(frames)}")

    selections = []
    proposal_by_frame = {}
    accepted_by_frame = {}
    for f in sorted(proposed_track):
        cand = proposed_track[f]
        proposal_by_frame[str(f)] = cand.to_dict() if cand is not None else None
        accepted_by_frame[str(f)] = None
        if cand is not None:
            selections.append({
                "source_frame": f,
                "candidate_index": cand.candidate_index,
                "xyxy": cand.to_dict()["xyxy"],
            })

    doc = {
        "input": {"path": RAW, "sha256": got,
                  "source_sha256": raw["source_sha256"],
                  "model_sha256": raw["model_sha256"]},
        "frozen_policy_sha256": pol_sha,
        "policy": policy.to_dict(),
        "method": "exact second-order dynamic-programming proposal over all "
                  "frozen post-NMS candidates; state includes the previous "
                  "candidate so distinct incoming velocities are preserved. "
                  "The coherent path is NOT accepted as ball observations without "
                  "independent identity qualification; motion coherence, size, "
                  "confidence and length are insufficient identity evidence.",
        "initialisation": "automatic: the optimiser chooses where the path "
                          "starts. No reference coordinate, seed or crop is used.",
        "counts": {"frames": len(frames), "candidates": candidate_total,
                   "invalid_boxes_rejected": rejected_total,
                   "frames_with_proposal": len(proposed),
                   "frames_without_proposal": len(frames) - len(proposed),
                   "frames_with_accepted_ball_observation": len(accepted),
                   "frames_without_accepted_ball_observation": len(frames) - len(accepted)},
        "acceptance": {"state": result.acceptance_state,
                       "rejection_reasons": list(result.rejection_reasons),
                       "identity_qualified": False},
        "proposals": selections,
        "proposal_by_frame": proposal_by_frame,
        "accepted_ball_by_frame": accepted_by_frame,
        "selected_by_frame": accepted_by_frame,
        "evaluation": "NONE performed here. This output is for independent "
                      "review; no accuracy or recall is claimed.",
        "research_only": True, "pseudo_label": True,
        "ground_truth": False, "production_eligible": False,
    }
    p = os.path.join(OUT, "selected_track.json")
    json.dump(doc, open(p, "w"), indent=1)
    print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
