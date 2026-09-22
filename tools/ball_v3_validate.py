"""Validate the v3 tracker on real video, and say exactly what each number means.

Two corpora, and they answer different questions:

``legacy``  the human-read annotations in ``data/ball_annotations/``. This corpus
            is **development-consumed**: an earlier pass scored it, and this pass
            used it to choose proposal parameters. It is the only real-labelled
            evidence in the repository, so it is reported as
            *real-labelled development evidence* and never as held-out accuracy.
``v3-validation``
            the 11 validation clips of the v3 evaluation corpus. These have **no
            labels** -- only a note that a playable ball is visible somewhere --
            so no accuracy number is produced for them at all. What is measured
            is behaviour on untouched real video: does it abstain, does it lock
            onto the body, does it accept a track, how long does it take.

The 10 final-test clips of the same corpus are refused by name before any decode,
by ``assert_not_final_test``. Their cached pose entries are never opened either.

Every metric is reported with its definition, its denominator and its sample
size. ``null`` means "not measurable here", never zero.

Ablation semantics, which are different questions and are labelled as such:

* ``--mask motion,body``   **inference masking** on one trained model: does this
  model's output depend on those inputs? It zeroes the named feature group(s)
  only. Association, track-level rules and every other temporal feature are
  retained, so it is a *partial* ablation and never "all temporal information
  removed".
* ``--discriminator PATH`` a **retrained** ablation: can the remaining features
  carry the signal at all? A checkpoint's own persisted mask is applied.
* ``--discriminator none`` the **no-learned-temporal-model reference**: the
  tracker still runs its association, tracklet linking and stated heuristic
  scoring, so this is *not* a frame-independent baseline. It measures the value
  of the learned model, not the value of temporal reasoning.
* ``--framewise``          the **no temporal model, no association** reference:
  each frame is decided from its own ranker top-1. The ranker's motion-residual
  context channels are still computed, so the *input* is not frame-independent
  either; only the decision is.

A cache-hit run's motion/proposal/ranking timings are inherited from the run that
built the cached extraction, and are reported separately from measured runtime.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghostcaddie.video import ball_v3_body as bbody            # noqa: E402
from ghostcaddie.video import ball_v3_proposals as bp          # noqa: E402
from ghostcaddie.video import ball_v3_ranker as brank          # noqa: E402
from ghostcaddie.video import ball_v3_tracker as btrack        # noqa: E402
from ghostcaddie.video import ball_v3_tracklets as btrk        # noqa: E402

ANNOTATIONS = ROOT / "data" / "ball_annotations"
V3_EVAL = ROOT / "data" / "ball_dataset_v3_eval"
BODY_CACHE = ROOT / "out" / "ball_tracker_v3" / "body_cache"

#: The legacy labels carry their own noise floor, stated in data/ball_split.json:
#: "centre precision is roughly +/-15 px or 0.15x ball diameter, whichever is
#: larger". Recovery is reported at the strict radius rule and at this one.
LABEL_NOISE_PX = 15.0
#: The v2 detector's trained regime. A ball larger than this is outside the scale
#: the whole v3 stack was built to cover, so it is reported separately.
IN_REGIME_RADIUS = 40.0


def sealed_clip_ids() -> set[str]:
    payload = json.loads((V3_EVAL / "split.json").read_text(encoding="utf-8"))
    return {c["clip_id"] for c in payload["clips"] if c["split"] == "final_test"}


def assert_not_final_test(clip_ids) -> None:
    clash = set(clip_ids) & sealed_clip_ids()
    if clash:
        raise SystemExit(f"refusing to decode sealed final-test clips: "
                         f"{sorted(clash)}")


# ---------------------------------------------------------------------------
# corpora
# ---------------------------------------------------------------------------

def consecutive_runs(indices, limit: int) -> list[list[int]]:
    runs: list[list[int]] = []
    for index in sorted(indices):
        if runs and index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    out = []
    for run in runs:
        if len(run) < 4:
            continue
        # the tracker needs consecutive frames; a longer run is chunked so one
        # long clip does not dominate the runtime of a pass
        for start in range(0, len(run), limit):
            piece = run[start:start + limit]
            if len(piece) >= 4:
                out.append(piece)
    return out


def legacy_runs(limit: int) -> list[dict]:
    runs = []
    for path in sorted(ANNOTATIONS.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        video = ROOT / payload["video"]
        if not video.exists():
            continue
        clip_id = payload["clip_id"]
        assert_not_final_test([clip_id])
        labels = {int(k): v for k, v in payload["frames"].items()}
        for piece in consecutive_runs(labels, limit):
            runs.append({"clip_id": clip_id, "media": video, "indices": piece,
                         "labels": {i: labels[i] for i in piece},
                         "labelled": True,
                         "family": clip_id})
    return runs


def v3_validation_runs(limit: int) -> list[dict]:
    split = json.loads((V3_EVAL / "split.json").read_text(encoding="utf-8"))
    sources = json.loads((V3_EVAL / "sources.json").read_text(encoding="utf-8"))
    by_id = {c["clip_id"]: c for c in sources["clips"]}
    runs = []
    for record in split["clips"]:
        if record["split"] != "validation":
            continue
        clip_id = record["clip_id"]
        assert_not_final_test([clip_id])
        source = by_id.get(clip_id, {})
        media = ROOT / (source.get("media_path") or "")
        if not media.exists():
            continue
        capture = cv2.VideoCapture(str(media))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if total < limit:
            continue
        first = max(0, total // 3)
        runs.append({"clip_id": clip_id, "media": media,
                     "indices": list(range(first, first + limit)),
                     "labels": {}, "labelled": False,
                     "family": record["source_group"],
                     "holds_playable_ball": record["holds_playable_ball"]})
    return runs


def read_window(video: Path, indices) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
    frames = []
    for _ in indices:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score_run(result, run: dict) -> dict:
    """Turn one tracker result into named, denominator-explicit counts."""
    states = result.states
    labels = run["labels"]
    observed = [s for s in states if s.state == "OBSERVED"]
    counts: dict[str, int] = {}
    for state in states:
        counts[state.state] = counts.get(state.state, 0) + 1

    recovery = {"labelled_visible": 0, "recovered": 0,
                "recovered_within_label_noise": 0, "in_regime": 0,
                "recovered_in_regime": 0}
    false_track = {"labelled_absent": 0, "observed_on_absent": 0,
                   "observed_on_wrong_object": 0, "wrong_object_in_regime": 0}
    for state in states:
        label = labels.get(state.frame_index)
        if label is None:
            continue
        radius = float(label.get("radius") or 5.0)
        in_regime = radius <= IN_REGIME_RADIUS
        visible = bool(label.get("visible")) and not label.get("ambiguous")
        if visible:
            recovery["labelled_visible"] += 1
            if in_regime:
                recovery["in_regime"] += 1
            if state.state != "OBSERVED" or state.x is None:
                continue
            distance = math.hypot(state.x - float(label["x"]),
                                  state.y - float(label["y"]))
            if distance <= max(6.0, radius):
                recovery["recovered"] += 1
                if in_regime:
                    recovery["recovered_in_regime"] += 1
            elif distance <= max(6.0, radius, LABEL_NOISE_PX):
                recovery["recovered_within_label_noise"] += 1
            else:
                false_track["observed_on_wrong_object"] += 1
                if in_regime:
                    false_track["wrong_object_in_regime"] += 1
        elif not label.get("visible"):
            false_track["labelled_absent"] += 1
            if state.state == "OBSERVED":
                false_track["observed_on_absent"] += 1

    # identity switches: an OBSERVED step longer than the link radius is a jump to
    # a different object, not a ball that moved
    switches = 0
    longest_run = current_run = 0
    observed_indices = [s.frame_index for s in observed]
    by_index = {s.frame_index: s for s in observed}
    for state in observed:
        if state.frame_index - 1 in by_index:
            current_run += 1
        else:
            current_run = 1
        longest_run = max(longest_run, current_run)
    for left, right in zip(observed, observed[1:]):
        if right.frame_index != left.frame_index + 1:
            continue
        radius = max(2.0, float(left.radius or 2.0))
        if math.hypot(right.x - left.x, right.y - left.y) > btrk.LINK_RADII * radius:
            switches += 1

    body_locked = sum(1 for s in observed if "BODY_LOCK" in s.diagnostics)
    return {
        "clip_id": run["clip_id"],
        "family": run["family"],
        "frames": len(states),
        "labelled": run["labelled"],
        "accepted": bool(result.accepted),
        "state_counts": counts,
        "observed": len(observed),
        "coverage": round(len(observed) / max(1, len(states)), 4),
        "abstention_rate": round(
            sum(1 for s in states if s.state in ("AMBIGUOUS", "UNAVAILABLE", "LOST"))
            / max(1, len(states)), 4),
        "identity_recovery": recovery,
        "false_track": false_track,
        "switches": switches,
        "observed_runs": len([1 for s, p in zip(observed, [None] + observed[:-1])
                              if p is None
                              or s.frame_index != p.frame_index + 1]),
        "longest_observed_run": longest_run,
        "body_lock_frames": body_locked,
        "body_lock_rate": round(body_locked / max(1, len(observed)), 4)
        if observed else None,
        "club_lock_rate": None,
        "rejection_reasons": list(result.rejection_reasons),
        "track_probability": result.confidence.get("track_probability"),
        "static_fraction": result.confidence.get("static_fraction"),
        "body_inside_fraction": result.confidence.get("body_inside_fraction"),
        "timings": result.timings,
    }


def aggregate(rows: list[dict]) -> dict:
    labelled = [r for r in rows if r["labelled"]]
    visible = sum(r["identity_recovery"]["labelled_visible"] for r in labelled)
    recovered = sum(r["identity_recovery"]["recovered"] for r in labelled)
    in_regime = sum(r["identity_recovery"]["in_regime"] for r in labelled)
    recovered_regime = sum(r["identity_recovery"]["recovered_in_regime"]
                           for r in labelled)
    noise = sum(r["identity_recovery"]["recovered_within_label_noise"]
                for r in labelled)
    absent = sum(r["false_track"]["labelled_absent"] for r in labelled)
    on_absent = sum(r["false_track"]["observed_on_absent"] for r in labelled)
    wrong = sum(r["false_track"]["observed_on_wrong_object"] for r in labelled)
    frames = sum(r["frames"] for r in rows)
    observed = sum(r["observed"] for r in rows)
    # Timings measured in THIS invocation only. A cache-hit run inherits the
    # build run's motion/proposal/ranking seconds; summing those in would report
    # one run's extraction cost once per variant.
    measured = sum(sum(r.get("timings_measured_this_run", {}).values())
                   for r in rows)
    inherited_rows = [r for r in rows if r.get("extraction_reused")]
    inherited = sum(sum(r.get("timings_inherited_extraction_build", {}).values())
                    for r in rows)
    wall = sum(r.get("wall_seconds") or 0.0 for r in rows)
    return {
        "runs": len(rows),
        "frames": frames,
        "accepted_runs": sum(1 for r in rows if r["accepted"]),
        "observed_frames": observed,
        "coverage": round(observed / max(1, frames), 4),
        "identity_recovery": {
            "definition": "OBSERVED and within max(6 px, 1.0 ball radius) of a "
                          "human label that is visible and unambiguous",
            "labelled_visible_frames": visible,
            "recovered": recovered,
            "recall": round(recovered / max(1, visible), 4) if visible else None,
            "within_label_noise": noise,
            "recall_within_label_noise": round(
                (recovered + noise) / max(1, visible), 4) if visible else None,
            "in_regime_frames": in_regime,
            "recovered_in_regime": recovered_regime,
            "recall_in_regime": round(recovered_regime / max(1, in_regime), 4)
            if in_regime else None,
        },
        "false_track": {
            "definition": "OBSERVED on a frame the human label marks absent, or "
                          "OBSERVED further than one ball radius from a visible "
                          "label",
            "labelled_absent_frames": absent,
            "observed_on_absent": on_absent,
            "rate_on_absent": round(on_absent / max(1, absent), 4)
            if absent else None,
            "observed_on_wrong_object": wrong,
            "rate_on_visible": round(wrong / max(1, visible), 4) if visible else None,
        },
        "correct_abstention": {
            "definition": "fraction of labelled ball-absent frames the tracker "
                          "does not call OBSERVED",
            "denominator": absent,
            "value": round(1.0 - on_absent / absent, 4) if absent else None,
        },
        "switches": sum(r["switches"] for r in rows),
        "switches_per_run": round(sum(r["switches"] for r in rows) / max(1, len(rows)), 3),
        "longest_observed_run": max((r["longest_observed_run"] for r in rows),
                                    default=0),
        "body_lock_rate": round(
            sum(r["body_lock_frames"] for r in rows) / max(1, observed), 4)
        if observed else None,
        "static_rejection_runs": sum(
            1 for r in rows if any("static" in reason
                                   for reason in r["rejection_reasons"])),
        "club_lock_rate": None,
        "club_lock_reason": "no clubhead tracker exists on any path; canonical "
                            "CLUBHEAD is UNAVAILABLE and the research clubhead "
                            "path was falsified",
        "runtime": {
            "measured_tracker_seconds": round(measured, 2),
            "measured_seconds_per_frame": round(measured / max(1, frames), 4),
            "tracker_seconds": round(measured, 2),
            "runs_reusing_cached_extraction": len(inherited_rows),
            "runs_with_extraction_built_this_run": len(rows) - len(inherited_rows),
            "inherited_extraction_build_seconds": round(inherited, 2),
            "inherited_note": (
                "motion/proposal/ranking seconds on a cache-hit run are the "
                "timings of the run that BUILT the extraction, inherited from "
                "disk. They are real measurements of the extraction cost, but "
                "not this variant's runtime, so they are reported here and never "
                "added to measured_tracker_seconds"),
            "wall_seconds_this_invocation": round(wall, 2),
            "wall_note": ("sum of per-run wall clock in this invocation, "
                          "including decode and any cache read"),
        },
        "by_family": sorted({r["family"] for r in rows}),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus", choices=("legacy", "v3-validation"))
    parser.add_argument("--frames-per-run", type=int, default=15)
    parser.add_argument("--ranker", type=Path,
                        default=ROOT / "out" / "ball_tracker_v3" / "ranker" / "ranker.pt")
    parser.add_argument("--discriminator", default=None,
                        help="path to a tracklet model, or 'none' for the "
                             "no-temporal-model baseline")
    parser.add_argument("--mask", default=None,
                        help="feature groups to zero at INFERENCE time on the "
                             "loaded model (motion, body, ranking, appearance, "
                             "chroma, scale)")
    parser.add_argument("--only-clips", default=None)
    parser.add_argument("--body-cache", type=Path, default=BODY_CACHE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default=None, help="a name for this run")
    parser.add_argument("--extraction-cache", type=Path, default=None,
                        help="reuse per-frame proposals/contexts/ranker scores "
                             "across ablation variants instead of recomputing "
                             "them. Keyed by source digest, the ordered frame "
                             "indices, ranker weights, proposal and tracker "
                             "params, preprocessing digests, body-input mode and "
                             "selected poses; a mismatched entry is refused")
    parser.add_argument("--no-body-upstream", action="store_true",
                        help="a GENUINELY full no-body path: the ranker is fed no "
                             "poses at all, so its own body inputs are gone too. "
                             "Requires a discriminator whose checkpoint drops the "
                             "body group as well. Distinct from --mask body, which "
                             "only removes the temporal body features")
    parser.add_argument("--framewise", action="store_true",
                        help="no-tracklet reference: decide each frame from its "
                             "own ranker top-1 under the tracker's per-frame "
                             "thresholds. No association, no temporal model -- "
                             "but the ranker's motion-residual context channels "
                             "are still computed, so label it that way")
    args = parser.parse_args(argv)

    runs = (legacy_runs(args.frames_per_run) if args.corpus == "legacy"
            else v3_validation_runs(args.frames_per_run))
    if args.only_clips:
        wanted = set(args.only_clips.split(","))
        runs = [r for r in runs if r["clip_id"] in wanted]
    if not runs:
        print("no runs to score", file=sys.stderr)
        return 2
    assert_not_final_test(r["clip_id"] for r in runs)

    ranker = brank.load_ranker(args.ranker)
    ranker_sha256 = brank.weights_hash(Path(args.ranker))
    discriminator = None
    checkpoint_mask = None
    if args.discriminator and args.discriminator != "none":
        discriminator = btrk.load_discriminator(Path(args.discriminator))
        checkpoint_mask = btrk.discriminator_mask(discriminator)
    mask = (btrk.mask_for_ablation(btrk.groups_for_mask(args.mask))
            if args.mask else None)
    if mask is not None and checkpoint_mask is not None \
            and not np.array_equal(mask, checkpoint_mask):
        print("refusing: --mask conflicts with the mask already stored in "
              f"{args.discriminator}; that checkpoint was trained without "
              f"{btrk.groups_for_mask(args.mask)}", file=sys.stderr)
        return 2
    if args.no_body_upstream:
        # A full no-body run is only a *full* no-body run when the temporal model
        # is also blind to the body group. ``--no-body-upstream`` alone starves
        # the ranker but leaves the tracklet model's body features present and
        # zero, which is a different (and mismeasured) configuration.
        if checkpoint_mask is None:
            print("refusing: --no-body-upstream requires a discriminator whose "
                  "checkpoint drops the body group (nothing couples the flag to "
                  "the model otherwise)", file=sys.stderr)
            return 2
        missing = [name for name in btrk.dropped_feature_names(["body"])
                   if name not in btrk.masked_feature_names(discriminator)]
        if missing:
            print("refusing: --no-body-upstream was given with a checkpoint that "
                  f"still uses body features {missing}", file=sys.stderr)
            return 2
    if mask is not None:
        _install_mask(mask)

    cache = bbody.BodyCache(args.body_cache)
    proposal_params = bp.ProposalParams()
    tracker_params = btrack.TrackerParams()

    rows = []
    started = time.time()
    for run in runs:
        run_started = time.time()
        frames = read_window(run["media"], run["indices"])
        if len(frames) < 4:
            continue
        indices = run["indices"][: len(frames)]
        poses = {}
        if not args.no_body_upstream:
            for index in indices:
                pose = cache.pose(run["clip_id"], index)
                if pose is not None:
                    poses[index] = pose
        extraction, reused = _load_or_build_extraction(
            args.extraction_cache, run, indices, frames, ranker, poses,
            proposal_params, tracker_params, ranker_sha256)
        if args.framewise:
            result = framewise_result(extraction, indices, tracker_params)
        else:
            result = btrack.track_ball_v3(
                frames, indices, ranker=ranker, discriminator=discriminator,
                poses=poses, params=tracker_params,
                proposal_params=proposal_params,
                provenance={"corpus": args.corpus, "clip_id": run["clip_id"],
                            "label": args.label,
                            "inference_mask": btrk.groups_for_mask(args.mask),
                            "no_body_upstream": bool(args.no_body_upstream)},
                extraction=extraction)
        row = score_run(result, {**run, "indices": indices})
        # A run that BUILT its own extraction measured those per-frame seconds
        # itself; only a run that read the extraction off disk inherits the build
        # run's. Classifying by key name alone would call a built extraction
        # "inherited" and report a full pass as if it cost nothing.
        inherited = ({k: v for k, v in result.timings.items()
                      if k in btrack.TIMING_KEYS_FROM_EXTRACTION}
                     if reused else {})
        measured = {k: v for k, v in result.timings.items()
                    if k != "extraction_reused"
                    and (reused or k not in btrack.TIMING_KEYS_FROM_EXTRACTION)}
        row["extraction_reused"] = bool(reused)
        row["timings_measured_this_run"] = measured
        row["timings_inherited_extraction_build"] = inherited
        row["wall_seconds"] = round(time.time() - run_started, 2)
        row["body_poses_available"] = len(poses)
        row["body_available_frames"] = sum(
            1 for index in indices
            if (extraction["frames"].get(str(index)) or {}).get("body_available"))
        rows.append(row)
        print(f"{run['clip_id']:20s} {len(indices):3d}f "
              f"observed={row['observed']:3d} accepted={row['accepted']} "
              f"switches={row['switches']} cached_extraction={reused} "
              f"wall={row['wall_seconds']}s", flush=True)

    payload = {
        "corpus": args.corpus,
        "label": args.label,
        "ranker": str(args.ranker),
        "discriminator": args.discriminator,
        "inference_mask": args.mask,
        "masked_groups": btrk.groups_for_mask(args.mask),
        "masked_features": btrk.dropped_feature_names(
            btrk.groups_for_mask(args.mask)),
        "checkpoint_mask_present": checkpoint_mask is not None,
        "mask_source": ("checkpoint" if checkpoint_mask is not None else
                        ("inference_flag" if mask is not None else "none")),
        "no_body_upstream": bool(args.no_body_upstream),
        "framewise_reference": bool(args.framewise),
        "ablation_scope": (
            "framewise reference: no temporal model and no association. The "
            "ranker's motion-residual context channels are still computed, so "
            "the input is not frame-independent; only the decision is"
            if args.framewise else
            ("tracklet model feature group(s) zeroed at inference; association "
             "and every other temporal feature are retained -- this is a partial "
             "ablation, not all temporal information removed"
             if mask is not None else "full pipeline")),
        "checkpoint_masked_features": btrk.masked_feature_names(discriminator)
        if discriminator is not None else [],
        "frames_per_run": args.frames_per_run,
        "labelled": bool(runs[0]["labelled"]),
        "evidence_class": ("REAL-LABELLED DEVELOPMENT EVIDENCE (development-consumed "
                           "corpus; not held-out)"
                           if runs[0]["labelled"] else
                           "UNLABELLED REAL VIDEO: behaviour only, no accuracy claim"),
        "aggregate": aggregate(rows),
        "per_run": rows,
        "wall_seconds": round(time.time() - started, 1),
        "conditions": {"proposal_params": vars(proposal_params),
                       "tracker_params": vars(tracker_params),
                       "extraction_cache": (None if args.extraction_cache is None
                                            else str(args.extraction_cache)),
                       "extraction_version": btrack.EXTRACTION_VERSION},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=1))
    print(f"\nwrote {args.out}")
    return 0


def _load_or_build_extraction(cache_root: Optional[Path], run: dict,
                              indices, frames, ranker, poses, proposal_params,
                              tracker_params, ranker_sha256) -> tuple[dict, bool]:
    """Reuse per-frame work across variants, or build and store it once.

    An ablation of the *tracklet* model cannot change proposals, context stacks,
    ranker probabilities or body geometry. Recomputing them per variant is how a
    measurement plan turns into a budget problem: the full pipeline cost per
    frame is what makes eight variants unaffordable, not the tracklet model.

    Returns ``(extraction, reused)``. ``reused`` is True only when the dict came
    off disk, and the caller must then report the timings inside it as *inherited
    from the build run*, not as this variant's measured runtime.

    The cache is keyed by :func:`btrack.extraction_fingerprint` -- source digest,
    the ordered frame indices, ranker weights, proposal *and* tracker parameters
    (``top_k`` included), preprocessing/code digests, the body-input mode and the
    selected poses' digest. A stored file whose recorded fingerprint disagrees
    with this call's is **refused**, not reused: with the old key (clip, first
    index, count, body mode, version) a different stride or a different ``top_k``
    could land on the same filename and silently score another configuration.
    """
    fingerprint = btrack.extraction_fingerprint(
        source_sha256=btrack.file_sha256(run["media"]), indices=indices,
        ranker_sha256=ranker_sha256, proposal_params=proposal_params,
        tracker_params=tracker_params,
        body_mode=("body" if poses else "nobody"),
        poses_digest=btrack.pose_digest(poses))
    if cache_root is None:
        return btrack.extract_frames(ranker, frames, indices, poses=poses,
                                     params=tracker_params,
                                     proposal_params=proposal_params), False
    path = cache_root / f"{run['clip_id']}__{fingerprint['fingerprint'][:32]}.json"
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as error:
            raise SystemExit(f"extraction cache {path} is unreadable: {error}")
        recorded = cached.get("fingerprint")
        if recorded != fingerprint["fingerprint"]:
            raise SystemExit(
                f"refusing extraction cache {path}: it records fingerprint "
                f"{recorded} but this call computes "
                f"{fingerprint['fingerprint']}. The cached probabilities were "
                "built from different inputs and must not be reused")
        if int(cached.get("version", -1)) != btrack.EXTRACTION_VERSION \
                or bool(cached.get("body_inputs")) != bool(poses):
            raise SystemExit(
                f"refusing extraction cache {path}: version/body inputs do not "
                "match this call")
        return cached, True
    extraction = btrack.extract_frames(ranker, frames, indices, poses=poses,
                                       params=tracker_params,
                                       proposal_params=proposal_params)
    extraction["fingerprint"] = fingerprint["fingerprint"]
    extraction["fingerprint_components"] = fingerprint["components"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(extraction), encoding="utf-8")
    temporary.replace(path)
    # strip the bookkeeping from the in-memory copy the tracker consumes
    extraction.pop("fingerprint_components", None)
    return extraction, False


def framewise_result(extraction: dict, indices, params) -> Any:
    """A no-tracklet reference: decide each frame from its own ranker output.

    The honest "not a temporal decision" baseline in the strongest form the
    existing rules allow. It applies the two thresholds the tracker itself
    documents as *per-frame* rules -- ``min_probability`` ("the winning candidate
    needs on a frame to be OBSERVED") and ``min_margin`` ("a near-tie becomes
    AMBIGUOUS, never OBSERVED") -- to each frame's top-1 candidate, with no
    association, no tracklet and no tracklet model.

    What it is **not**: a frame-independent input. The extraction it reads is
    built by ``extract_frames``, which hands the ranker context channels derived
    from camera-motion residuals (``FrameViews(frame, residual)``), so the ranker
    still sees motion context. The label therefore reads *no temporal model or
    association; ranker motion-residual context retained*. That also means this
    reference must not be described as "all temporal information removed" -- a
    partial motion-group ablation is not that either.

    The track-level rules (``max_static_fraction``, ``max_body_inside_fraction``,
    ``min_support``, ``min_track_probability``) are deliberately NOT applied:
    they are statements about a track, and inventing per-frame equivalents here
    would be inventing a new rule. No threshold was tuned to produce this
    reference.
    """
    states = []
    for index in indices:
        entry = extraction["frames"].get(str(index)) or {}
        candidates = entry.get("candidates") or []
        state = btrack.FrameState(frame_index=index, state=btrack.UNAVAILABLE)
        state.null_probability = float(entry.get("null_probability", 1.0))
        if not candidates:
            state.state = btrack.LOST
            state.reasons.append("no candidates on this frame")
            states.append(state)
            continue
        best = candidates[0]
        state.probability = float(best["probability"])
        state.margin = float(best.get("margin", 0.0))
        if state.probability < params.min_probability:
            state.reasons.append(
                f"top-1 probability {state.probability:.2f} below "
                f"{params.min_probability:.2f}")
        elif state.margin < params.min_margin:
            state.state = btrack.AMBIGUOUS
            state.reasons.append(
                f"margin {state.margin:.2f} below {params.min_margin:.2f}")
        else:
            state.state = btrack.OBSERVED
            state.x, state.y = float(best["x"]), float(best["y"])
            state.radius = float(best["radius"])
        states.append(state)
    return btrack.TrackResult(
        states=states,
        accepted=any(s.state == btrack.OBSERVED for s in states),
        confidence={"mode": "framewise_reference",
                    "min_probability": params.min_probability,
                    "min_margin": params.min_margin},
        provenance={"method": "framewise reference: no temporal model or "
                              "association; the ranker's motion-residual context "
                              "channels are retained"},
        rejection_reasons=[],
        timings={})


def _install_mask(mask: np.ndarray) -> None:
    """Zero named feature groups at inference time, without retraining.

    This answers "does this model's output depend on these inputs". A *retrained*
    ablation answers a different question -- "can the remaining features carry the
    signal" -- and is loaded from its own weights file instead.
    """
    original = btrk.tracklet_features

    def masked(track, *, window, camera_shift):
        values = original(track, window=window, camera_shift=camera_shift)
        return [float(v) * float(m) for v, m in zip(values, mask)]

    btrk.tracklet_features = masked


if __name__ == "__main__":
    raise SystemExit(main())
