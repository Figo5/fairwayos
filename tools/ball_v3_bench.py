"""Benchmark and profile the v3 ball-proposal pipeline under recorded conditions.

This tool exists because the incoming handoff carried timing numbers whose
conditions are not recoverable: "all six 4.93 s/frame, the v2 five 4.56,
chroma+brightness 1.38, chroma-only 0.50" were measured on a shared machine, at
mixed resolutions, and the published total for one generator set is *smaller*
than the sum of its own per-generator costs. An old contended figure must never
be compared against a fresh warm-cache run, so every number this tool prints is
produced under conditions it also prints.

What it records, for every suite:

* hardware and thread counts (CPU count, OpenCV threads, torch threads, device);
* the exact clips, frame indices and native resolutions that were decoded;
* the generator sets named in full -- never "the reduced one";
* cold (first, decode-inclusive) versus warm (repeat) passes, separately;
* end-to-end seconds per frame **including decode**, plus component timings;
* dispersion over repeated matched samples (median, p25, p75, min, max).

Suites:

``env``        print the recorded machine conditions and stop.
``components`` per-stage attribution on a fixed matched frame subset.
``sets``       A/B/C generator-set comparison with recall, on the dev corpus.
``downstream`` rank inference, context stacks, tracklet linking, temporal of the
               cached body read path, on the existing listwise store.

Ground truth for ``sets`` is the legacy human-read corpus in
``data/ball_annotations/``. That corpus is **development-consumed**: an earlier
pass scored it, and it is used here to choose parameters. Nothing here is
reported as accuracy, and the v3 evaluation corpus is not opened: the final-test
clips are refused by name before any decode.

Usage:
    .venv-video-ai/bin/python3 tools/ball_v3_bench.py env
    .venv-video-ai/bin/python3 tools/ball_v3_bench.py components --repeats 3
    .venv-video-ai/bin/python3 tools/ball_v3_bench.py sets --out OUT.json
    .venv-video-ai/bin/python3 tools/ball_v3_bench.py downstream --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghostcaddie.video import ball_candidates as bc            # noqa: E402
from ghostcaddie.video import ball_v3_proposals as bp          # noqa: E402

ANNOTATIONS = ROOT / "data" / "ball_annotations"
V3_EVAL_SPLIT = ROOT / "data" / "ball_dataset_v3_eval" / "split.json"


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def final_test_clips() -> set[str]:
    """Names of the sealed v3 final-test clips. Metadata read, media never."""
    if not V3_EVAL_SPLIT.exists():
        return set()
    payload = json.loads(V3_EVAL_SPLIT.read_text(encoding="utf-8"))
    return {c["clip_id"] for c in payload["clips"] if c["split"] == "final_test"}


def assert_not_final_test(clip_ids) -> None:
    """Fail closed before a decode. Reading final-test media is out of bounds."""
    clash = set(clip_ids) & final_test_clips()
    if clash:
        raise SystemExit(
            f"refusing to decode sealed final-test clips: {sorted(clash)}")


# ---------------------------------------------------------------------------
# conditions
# ---------------------------------------------------------------------------

def environment() -> dict:
    env = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "cv2_version": cv2.__version__,
        "cv2_threads": cv2.getNumThreads(),
        "numpy_version": np.__version__,
        "opencv_optimized": bool(cv2.useOptimized()),
    }
    try:
        import torch
        env["torch_version"] = torch.__version__
        env["torch_threads"] = torch.get_num_threads()
        env["torch_mps_built"] = bool(torch.backends.mps.is_built())
        env["torch_mps_available"] = bool(torch.backends.mps.is_available())
        env["torch_device"] = ("mps" if torch.backends.mps.is_available()
                               else "cpu")
    except Exception as error:  # pragma: no cover - environment dependent
        env["torch_version"] = f"unavailable: {error}"
    # imported lazily so `env` works in an interpreter without the pose stack
    return env


def set_threads(count: int | None) -> None:
    if count:
        cv2.setNumThreads(count)
        try:
            import torch
            torch.set_num_threads(count)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# the fixed dev frame set -- deterministic, and recorded in the output
# ---------------------------------------------------------------------------

def load_clips(max_radius: float, frames_per_clip: int,
               per_clip: dict[str, list[int]] | None = None) -> list[dict]:
    """Labelled dev frames, in-regime, at native resolution.

    ``per_clip`` pins exact frame indices for a matched comparison; without it the
    first ``frames_per_clip`` labelled frames of each clip are taken, which is what
    ``tools/ball_v3_proposals_eval.py`` does. Both are recorded in the output so a
    later reader can tell which set produced a number.
    """
    clips = []
    for path in sorted(ANNOTATIONS.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        video = ROOT / payload["video"]
        if not video.exists():
            continue
        labels = {int(k): v for k, v in payload["frames"].items()
                  if v.get("visible") and not v.get("ambiguous")
                  and (v.get("radius") or 5.0) <= max_radius}
        if not labels:
            continue
        clip_id = payload["clip_id"]
        if per_clip and clip_id not in per_clip:
            continue
        wanted = sorted(labels) if per_clip else sorted(labels)[:frames_per_clip]
        if per_clip:
            wanted = [i for i in per_clip[clip_id] if i in labels]
            if not wanted:
                continue
        assert_not_final_test([clip_id])
        decoded = read_window(video, wanted[0], wanted[-1])
        indices = [i for i in range(wanted[0], wanted[-1] + 1) if i in decoded]
        if len(indices) < 2:
            continue
        clips.append({
            "clip_id": clip_id,
            "media": str(video),
            "frames": [decoded[i] for i in indices],
            "indices": indices,
            "truth": {i: (float(labels[i]["x"]), float(labels[i]["y"]),
                          float(labels[i].get("radius") or 5.0))
                      for i in indices if i in labels},
            "native": [decoded[indices[0]].shape[1], decoded[indices[0]].shape[0]],
        })
    return clips


def read_window(video: Path, first: int, last: int) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames = {}
    for index in range(first, last + 1):
        ok, frame = capture.read()
        if not ok:
            break
        frames[index] = frame
    capture.release()
    return frames


def decode_only(clip: dict) -> float:
    """Seconds to decode this clip's window from cold. The end-to-end floor."""
    started = time.perf_counter()
    decoded = read_window(Path(clip["media"]), clip["indices"][0],
                          clip["indices"][-1])
    elapsed = time.perf_counter() - started
    if len(decoded) < len(clip["indices"]):
        raise RuntimeError(f"decode short on {clip['clip_id']}")
    return elapsed


# ---------------------------------------------------------------------------
# generator sets -- named in full, never "the reduced one"
# ---------------------------------------------------------------------------

def generator_sets() -> dict[str, tuple[str, ...]]:
    all_six = tuple(bp.ALL_METHOD_NAMES)
    v2_five = tuple(m for m in all_six if m != "local_distinctness")
    return {
        # A -- chroma-only. "chroma" here is local_distinctness: the generator
        # keys on Lab colour distance from the local background.
        "A distinctness only": ("local_distinctness",),
        # B -- the current default pair.
        "B distinctness+brightness": ("local_distinctness", "brightness_cue"),
        # C -- the legacy candidate stack on its own (the v2 five), kept separate
        # from all six so the comparison is unambiguous.
        "C legacy five (v2 stack)": v2_five,
        "C2 all six": all_six,
        # extra attribution rows
        "brightness only": ("brightness_cue",),
        "drop local_contrast": tuple(m for m in all_six if m != "local_contrast"),
    }


# ---------------------------------------------------------------------------
# measurement helpers
# ---------------------------------------------------------------------------

def dispersion(samples: list[float]) -> dict:
    if not samples:
        return {"n": 0}
    ordered = sorted(samples)
    return {
        "n": len(samples),
        "median": round(float(statistics.median(ordered)), 4),
        "p25": round(float(ordered[len(ordered) // 4]), 4),
        "p75": round(float(ordered[(3 * len(ordered)) // 4]), 4),
        "min": round(float(ordered[0]), 4),
        "max": round(float(ordered[-1]), 4),
        "mean": round(float(statistics.fmean(ordered)), 4),
        "std": round(float(statistics.pstdev(ordered)), 4) if len(ordered) > 1 else 0.0,
    }


def proposals_for(clip: dict, params: bp.ProposalParams,
                  methods: tuple[str, ...]) -> tuple[dict, dict]:
    per_frame, info = bp.propose(clip["frames"], clip["indices"], params=params,
                                methods=methods)
    return per_frame, info


def score_clip(clip: dict, per_frame: dict) -> dict:
    """Recall with explicit numerator/denominator and the geometry criterion."""
    result = bp.proposal_recall(per_frame, clip["truth"])
    found_noise = 0
    for index, (gx, gy, radius) in clip["truth"].items():
        tolerance = max(6.0, radius)
        if any(math.hypot(p.x - gx, p.y - gy) <= max(tolerance, 15.0)
               for p in (per_frame.get(index) or [])):
            found_noise += 1
    total = max(1, result["found"] + result["missed"])
    result["recall_within_label_noise"] = round(found_noise / total, 4)
    result["tolerance_rule"] = "within max(6 px, 1.0 * labelled ball radius)"
    result["noise_aware_rule"] = "within max(6 px, 1.0 * radius, 15 px label noise)"
    return result


def diversity(per_frame: dict, indices) -> dict:
    """What the proposal set looks like: count, spatial spread, scale spread."""
    counts, radii, spreads = [], [], []
    for index in indices:
        proposals = per_frame.get(index) or []
        counts.append(len(proposals))
        if proposals:
            radii.append(float(np.median([p.radius for p in proposals])))
            xs = np.array([p.x for p in proposals])
            ys = np.array([p.y for p in proposals])
            spreads.append(float(math.hypot(xs.std(), ys.std())) if len(xs) > 1 else 0.0)
    return {
        "proposals_per_frame_mean": round(float(np.mean(counts)), 2) if counts else 0.0,
        "proposals_per_frame_max": int(max(counts)) if counts else 0,
        "median_scale_spread_px": round(float(np.std(radii)), 2) if radii else None,
        "mean_spatial_spread_px": round(float(np.mean(spreads)), 1) if spreads else None,
    }


# ---------------------------------------------------------------------------
# suites
# ---------------------------------------------------------------------------

def suite_equivalence(clips: list[dict], repeats: int, threads: int | None) -> dict:
    """Did the pyramid change the candidates, and how much?

    Every scale-space generator now has a ``pyramid`` switch, so the same frame
    can be run both ways and the two candidate sets compared directly: how many,
    how far apart, how different the scores, and -- the number that actually
    matters -- whether either one still contains the ball.
    """
    set_threads(threads)
    params = bp.ProposalParams()
    names = tuple(bp.ALL_METHOD_NAMES)
    report: dict[str, dict] = {}
    for name in names:
        generator = bp.GENERATORS[name]
        if name not in ("radial_symmetry", "local_contrast", "brightness_cue",
                        "local_distinctness"):
            report[name] = {"pyramid_switch": False}
            continue
        entry: dict = {"pyramid_switch": True, "modes": {}}
        for mode in (True, False):
            counts, seconds = [], 0.0
            found = total = 0
            agreements, score_deltas = [], []
            for clip in clips:
                assert_not_final_test([clip["clip_id"]])
                contexts, _ = bc.build_contexts(clip["frames"], clip["indices"],
                                                max_side=params.work_max_side,
                                                motion="lazy")
                started = time.perf_counter()
                per_frame = {}
                for context in contexts:
                    per_frame[context.index] = generator(
                        context, limit=params.per_method_limit, pyramid=mode)
                seconds += time.perf_counter() - started
                for index in clip["indices"]:
                    counts.append(len(per_frame[index]))
                    truth = clip["truth"].get(index)
                    if truth is None:
                        continue
                    total += 1
                    gx, gy, radius = truth
                    tolerance = max(6.0, radius)
                    if any(math.hypot(c.x - gx, c.y - gy) <= tolerance
                           for c in per_frame[index]):
                        found += 1
                key = f"{clip['clip_id']}:{mode}"
                entry["modes"].setdefault("_raw", {})[key] = [
                    [(round(c.x, 3), round(c.y, 3), round(c.radius, 3), round(c.score, 6))
                     for c in per_frame[index]] for index in clip["indices"]]
            frames = sum(len(c["indices"]) for c in clips)
            entry["modes"][f"pyramid_{mode}"] = {
                "candidates_per_frame_mean": round(float(np.mean(counts)), 2),
                "candidates_per_frame_max": int(max(counts)),
                "seconds": round(seconds, 2),
                "seconds_per_frame": round(seconds / max(1, frames), 4),
                "ball_found": found,
                "labelled_frames": total,
                "recall": round(found / max(1, total), 4),
            }
        # agreement between the two modes, computed from the stored raw lists
        raw = entry["modes"].pop("_raw", {})
        same_count = positions_matched = positions_total = 0
        worst = 0.0
        for clip in clips:
            a = raw.get(f"{clip['clip_id']}:True")
            b = raw.get(f"{clip['clip_id']}:False")
            if a is None or b is None:
                continue
            for left, right in zip(a, b):
                if len(left) == len(right):
                    same_count += 1
                positions_total += max(len(left), len(right))
                for (x1, y1, _r1, s1) in left:
                    if not right:
                        continue
                    distance, s2 = min((math.hypot(x1 - x2, y1 - y2), s2_)
                                       for x2, y2, _r2, s2_ in right)
                    if distance <= 1.0:
                        positions_matched += 1
                        worst = max(worst, abs(s1 - s2))
        entry["agreement"] = {
            "frames_with_equal_candidate_count": same_count,
            "frames_compared": sum(len(raw.get(f"{c['clip_id']}:True", []))
                                   for c in clips),
            "positions_matched_within_1px": positions_matched,
            "positions_total": positions_total,
            "max_abs_score_delta_at_matched_position": round(worst, 6),
        }
        report[name] = entry
    return report


def suite_sets(clips: list[dict], repeats: int, threads: int | None,
               pyramid: bool = True, only: str | None = None) -> dict:
    set_threads(threads)
    params = bp.ProposalParams(pyramid=pyramid)
    wanted = [name.strip() for name in only.split(",")] if only else None
    results = {}
    for label, methods in generator_sets().items():
        if wanted and label not in wanted:
            continue
        per_clip = {}
        found = missed = 0
        proposals_per_frame, raw_per_frame = [], []
        pass_seconds: list[float] = []   # proposal seconds for one whole pass
        pass_frames: list[int] = []
        warm_seconds: list[float] = []
        warm_frames: list[int] = []
        cold_pass_seconds = 0.0
        decode_seconds = 0.0
        diversity_rows: list[dict] = []
        for clip in clips:
            assert_not_final_test([clip["clip_id"]])
            # Decode is measured once per clip, NOT once per repetition. The
            # comment here previously claimed otherwise, which made the cold
            # end-to-end figure look like it omitted a repeated cost it never had.
            # Decode is charged to the cold pass only.
            decode_seconds += decode_only(clip)
            per_frame: dict = {}
            for repeat_index in range(max(1, repeats)):
                started = time.perf_counter()
                per_frame, info = proposals_for(clip, params, methods)
                elapsed = time.perf_counter() - started
                pass_seconds.append(elapsed)
                pass_frames.append(len(clip["indices"]))
                if repeat_index == 0:
                    cold_pass_seconds += elapsed
                else:
                    warm_seconds.append(elapsed)
                    warm_frames.append(len(clip["indices"]))
                proposals_per_frame.append(info["mean_proposals"])
                raw_per_frame.append(info["mean_raw_candidates"])
            scored = score_clip(clip, per_frame)
            per_clip[clip["clip_id"]] = scored
            found += scored["found"]
            missed += scored["missed"]
            diversity_rows.append(diversity(per_frame, clip["indices"]))
        total = max(1, found + missed)
        frames = sum(pass_frames)
        warm_per_frame = sum(pass_seconds) / max(1, frames)
        cold_frames = sum(len(c["indices"]) for c in clips)
        results[label] = {
            "methods": list(methods),
            "clips": len(clips),
            "labelled_frames": total,
            "found": found,
            "missed": missed,
            "recall": round(found / total, 4),
            "mean_raw_candidates_per_frame": round(float(np.mean(raw_per_frame)), 2),
            "mean_proposals_per_frame": round(float(np.mean(proposals_per_frame)), 2),
            "proposal_seconds_total": round(sum(pass_seconds), 2),
            # The old key was called "..._warm" but averaged cold and warm passes
            # together. Both are now reported under names that say which.
            "proposal_seconds_per_frame_all_passes": round(warm_per_frame, 4),
            "proposal_seconds_per_frame_warm_only": (
                round(sum(warm_seconds) / max(1, sum(warm_frames)), 4)
                if warm_seconds else None),
            "proposal_seconds_per_frame_cold": round(
                cold_pass_seconds / max(1, cold_frames), 4),
            "repeats": max(1, repeats),
            "decode_accounting": ("decode measured once per clip and charged to "
                                  "the cold pass; it is not repeated per "
                                  "repetition"),
            "decode_seconds_total": round(decode_seconds, 2),
            "end_to_end_seconds_per_frame": round(
                (cold_pass_seconds + decode_seconds) / max(1, cold_frames), 4),
            "pass_seconds_per_frame_dispersion": dispersion(
                [s / max(1, f) for s, f in zip(pass_seconds, pass_frames)]),
            "per_clip_recall": {k: v["recall"] for k, v in per_clip.items()},
            "diversity": {
                "proposals_per_frame_mean": round(float(np.mean(
                    [d["proposals_per_frame_mean"] for d in diversity_rows])), 2),
                "proposals_per_frame_max": int(max(
                    d["proposals_per_frame_max"] for d in diversity_rows)),
                "median_scale_spread_px": round(float(np.mean(
                    [d["median_scale_spread_px"] for d in diversity_rows
                     if d["median_scale_spread_px"] is not None])), 2),
                "mean_spatial_spread_px": round(float(np.mean(
                    [d["mean_spatial_spread_px"] for d in diversity_rows
                     if d["mean_spatial_spread_px"] is not None])), 1),
            },
        }
    return results


def suite_components(clips: list[dict], repeats: int, threads: int | None,
                     pyramid: bool = True) -> dict:
    """Per-stage attribution on one matched frame set.

    Cold is the first pass (decode + first-touch caches); warm is every later
    pass. They are reported separately and never mixed.
    """
    set_threads(threads)
    params = bp.ProposalParams(pyramid=pyramid)
    all_six = tuple(bp.ALL_METHOD_NAMES)
    components: dict[str, list[float]] = {}

    def record(name: str, seconds: float) -> None:
        components.setdefault(name, []).append(seconds)

    def call(name: str, context):
        kwargs: dict = {"limit": params.per_method_limit}
        if name in bp.PYRAMID_METHODS:
            kwargs["pyramid"] = params.pyramid
        return bp.GENERATORS[name](context, **kwargs)

    frames_total = sum(len(c["indices"]) for c in clips)
    decode_cold = 0.0
    blend_passes = {"cold": 0, "warm": 0}
    for pass_index in range(max(1, repeats)):
        blend = "warm" if pass_index else "cold"
        blend_passes[blend] += 1
        for clip in clips:
            assert_not_final_test([clip["clip_id"]])
            if pass_index == 0:
                decode_cold += decode_only(clip)
            else:
                decode_only(clip)

            started = time.perf_counter()
            contexts, info = bc.build_contexts(clip["frames"], clip["indices"],
                                               max_side=params.work_max_side)
            record(f"build_contexts[{blend}]", time.perf_counter() - started)

            # split build_contexts into its parts on the same frames
            started = time.perf_counter()
            for frame in clip["frames"]:
                bc._working(frame, bc._work_scale(frame.shape[1], frame.shape[0],
                                                  params.work_max_side))
            record(f"  resize[{blend}]", time.perf_counter() - started)
            started = time.perf_counter()
            grays = [cv2.cvtColor(c.colour, cv2.COLOR_BGR2GRAY).astype(np.float32)
                     for c in contexts]
            record(f"  gray+cast[{blend}]", time.perf_counter() - started)
            started = time.perf_counter()
            for previous, current in zip(grays, grays[1:]):
                bc.estimate_motion(previous, current)
            record(f"  phaseCorrelate[{blend}]", time.perf_counter() - started)

            for name in all_six:
                started = time.perf_counter()
                for context in contexts:
                    call(name, context)
                record(f"generator:{name}[{blend}]", time.perf_counter() - started)

            raw = {index: [] for index in clip["indices"]}
            for name in all_six:
                for context in contexts:
                    for candidate in call(name, context):
                        raw[context.index].append(candidate)
            started = time.perf_counter()
            for index in clip["indices"]:
                bp.propose_frame(raw.get(index, []), params)
            record(f"dedupe+caps[{blend}]", time.perf_counter() - started)

    return {
        "frames": frames_total,
        "decode_cold_seconds": round(decode_cold, 2),
        "blend_passes": dict(blend_passes),
        # Each key holds one sample per clip per pass that carried that blend, so
        # the divisor is frames_total x passes_with_that_blend. The first report
        # divided by frames_total x repeats instead, which halved every value at
        # repeats=2 and made cross-row totals inconsistent.
        "components_seconds_per_frame": {
            name: round(sum(samples) / max(1, frames_total * blend_passes[
                "warm" if "[warm]" in name else "cold"]), 4)
            for name, samples in components.items()},
        "component_denominator_note": (
            "denominator is frames_total x passes_with_that_blend; corrected "
            "values for a repeats=2 run are exactly 2x the previously reported "
            "ones, recomputed from the raw per-clip samples already in this "
            "file rather than re-measured"),
        "component_dispersion": {name: dispersion(v) for name, v in components.items()},
    }


def suite_downstream(args) -> dict:
    """Rank inference, context stacks, tracklet linking, temporal scoring.

    Reads the existing listwise store. Nothing here is trained or tuned; this is
    the cost of the stages after the proposal stage.
    """
    set_threads(args.threads)
    from ghostcaddie.video import ball_v3_context as bctx
    from ghostcaddie.video import ball_v3_dataset as bdata
    from ghostcaddie.video import ball_v3_ranker as brank
    from ghostcaddie.video import ball_v3_tracklets as btrk

    store = Path(args.data)
    reader = bdata.ExampleReader(store)
    records = reader.records[: args.records]
    clips_in_store = {r["clip_id"] for r in reader.records}
    assert_not_final_test(clips_in_store)

    out: dict = {"store": str(store), "records_used": len(records),
                 "records_in_store": len(reader.records)}

    import torch
    device = brank.device()
    model = brank.load_ranker(Path(args.ranker), device_override=device)
    out["device"] = str(device)
    out["weights_sha256"] = brank.weights_hash(Path(args.ranker))

    numeric_all = [reader.numeric_for(r) for r in records]
    total_candidates = sum(len(n) for n in numeric_all)

    # candidate context stacks, both ways, so the shared-pyramid crop can be
    # compared against the per-candidate one on the same geometry
    stacks = []
    started = time.perf_counter()
    for record in records:
        frame = np.zeros((record["height"], record["width"], 3), np.uint8)
        positions = list(zip([float(r[0]) * record["width"] for r in record["numeric"]],
                             [float(r[1]) * record["height"] for r in record["numeric"]],
                             [max(1.0, float(r[2]) * min(record["width"], record["height"]))
                              for r in record["numeric"]]))
        stacks.append(bctx.stack_batch(frame, None, positions))
    stack_seconds = time.perf_counter() - started
    out["context_stack_seconds"] = round(stack_seconds, 3)
    out["context_stack_seconds_per_candidate"] = round(
        stack_seconds / max(1, total_candidates), 6)

    started = time.perf_counter()
    shared = []
    for record in records:
        frame = np.zeros((record["height"], record["width"], 3), np.uint8)
        positions = list(zip([float(r[0]) * record["width"] for r in record["numeric"]],
                             [float(r[1]) * record["height"] for r in record["numeric"]],
                             [max(1.0, float(r[2]) * min(record["width"], record["height"]))
                              for r in record["numeric"]]))
        shared.append(bctx.FrameViews(frame).stack_many(positions))
    shared_seconds = time.perf_counter() - started
    out["frame_views_seconds"] = round(shared_seconds, 3)
    out["frame_views_seconds_per_candidate"] = round(
        shared_seconds / max(1, total_candidates), 6)
    out["frame_views_speedup"] = round(stack_seconds / max(1e-9, shared_seconds), 2)
    deltas = [float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
              for a, b in zip(stacks, shared) if a.shape == b.shape]
    out["frame_views_mean_abs_channel_delta"] = (round(float(np.mean(deltas)), 3)
                                                 if deltas else None)
    out["frame_views_max_abs_channel_delta"] = (
        int(max(float(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())
                for a, b in zip(stacks, shared) if a.shape == b.shape))
        if deltas else None)
    stacks = shared

    # rank inference, warm
    started = time.perf_counter()
    for record, stack in zip(records, stacks):
        brank.rank_frame(model, stack, reader.numeric_for(record), device=device)
    rank_seconds = time.perf_counter() - started
    out["rank_inference_seconds"] = round(rank_seconds, 3)
    out["rank_inference_seconds_per_candidate"] = round(
        rank_seconds / max(1, total_candidates), 6)
    out["rank_inference_seconds_per_frame"] = round(
        rank_seconds / max(1, len(records)), 5)

    # tracklet linking + features on the store's own consecutive runs
    import collections
    by_clip = collections.defaultdict(list)
    for record in records:
        by_clip[record["clip_id"]].append(record)
    nodes_by_frame: dict[int, list] = {}
    for clip_id, group in by_clip.items():
        group.sort(key=lambda r: r["frame_index"])
        for record in group:
            numeric = reader.numeric_for(record)
            nodes: list = []
            for row in numeric:
                nodes.append(btrk.Node(
                    frame_index=record["frame_index"], x=0.0, y=0.0, radius=1.0,
                    probability=float(row[0]), inside_torso=float(row[5]),
                    distance_to_hands=float(row[7]) * 200.0))
            nodes_by_frame[record["frame_index"]] = nodes
    started = time.perf_counter()
    tracks = btrk.build_tracklets(nodes_by_frame, {}, top_k=args.top_k)
    link_seconds = time.perf_counter() - started
    out["tracklet_link_seconds"] = round(link_seconds, 4)
    out["tracklets_built"] = len(tracks)
    started = time.perf_counter()
    for track in tracks:
        btrk.tracklet_features(track, window=len(nodes_by_frame), camera_shift={})
    out["tracklet_feature_seconds"] = round(time.perf_counter() - started, 4)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suite", choices=("env", "components", "equivalence",
                                          "sets", "downstream"))
    parser.add_argument("--repeats", type=int, default=3,
                        help="matched repeats per clip (warm passes beyond the first)")
    parser.add_argument("--max-radius", type=float, default=40.0,
                        help="the v2 detector's trained regime")
    parser.add_argument("--frames-per-clip", type=int, default=12)
    parser.add_argument("--threads", type=int, default=None,
                        help="pin OpenCV/torch threads; recorded in the output")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "out" / "ball_tracker_v3" / "listwise_train")
    parser.add_argument("--ranker", type=Path,
                        default=ROOT / "out" / "ball_tracker_v3" / "ranker" / "ranker.pt")
    parser.add_argument("--records", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--legacy-proposals", action="store_true",
                        help="run the scale-space generators at full resolution "
                             "(the pre-optimisation path) for a before/after")
    parser.add_argument("--only", default=None,
                        help="comma-separated generator-set labels to run")
    args = parser.parse_args(argv)

    env = environment()
    print(json.dumps(env, indent=1), flush=True)
    if args.suite == "env":
        return 0

    payload: dict = {"conditions": env, "suite": args.suite,
                     "max_radius_px": args.max_radius,
                     "ground_truth": "HUMAN_READ_LEGACY_ANNOTATIONS "
                                     "(development-consumed; never accuracy)"}

    if args.suite == "downstream":
        payload["downstream"] = suite_downstream(args)
    else:
        print("decoding labelled windows...", flush=True)
        clips = load_clips(args.max_radius, args.frames_per_clip)
        print(f"{len(clips)} clips, "
              f"{sum(len(c['truth']) for c in clips)} labelled frames", flush=True)
        payload["clips"] = [
            {"clip_id": c["clip_id"], "media": c["media"],
             "native": c["native"], "indices": c["indices"],
             "labelled": len(c["truth"])} for c in clips]
        if args.suite == "components":
            payload["repeats"] = args.repeats
            payload["pyramid"] = not args.legacy_proposals
            payload["components"] = suite_components(
                clips, args.repeats, args.threads,
                pyramid=not args.legacy_proposals)
            print(json.dumps(payload["components"], indent=1), flush=True)
        elif args.suite == "equivalence":
            payload["equivalence"] = suite_equivalence(clips, args.repeats, args.threads)
            print(f"\n{'generator':22s} {'mode':10s} {'cand/f':>8s} "
                  f"{'s/f':>8s} {'recall':>8s} {'found/N':>9s}", flush=True)
            for name, entry in payload["equivalence"].items():
                if not entry.get("pyramid_switch"):
                    continue
                for mode, values in entry["modes"].items():
                    print(f"{name:22s} {mode:10s} "
                          f"{values['candidates_per_frame_mean']:8.1f} "
                          f"{values['seconds_per_frame']:8.4f} "
                          f"{values['recall']:8.3f} "
                          f"{values['ball_found']:4d}/{values['labelled_frames']:<4d}",
                          flush=True)
                print(f"{'':22s} agreement {entry['agreement']}", flush=True)
        else:
            payload["repeats"] = args.repeats
            payload["pyramid"] = not args.legacy_proposals
            payload["generator_sets"] = suite_sets(
                clips, args.repeats, args.threads,
                pyramid=not args.legacy_proposals, only=args.only)
            print(f"\n{'generator set':28s} {'recall':>6s} {'found/N':>9s} "
                  f"{'raw/f':>7s} {'prop/f':>7s} {'s/f prop':>9s} {'s/f e2e':>8s}",
                  flush=True)
            for label, result in payload["generator_sets"].items():
                print(f"{label:28s} {result['recall']:6.3f} "
                      f"{result['found']:4d}/{result['labelled_frames']:<4d} "
                      f"{result['mean_raw_candidates_per_frame']:7.1f} "
                      f"{result['mean_proposals_per_frame']:7.1f} "
                      f"{result['proposal_seconds_per_frame_cold']:9.3f} "
                      f"{result['end_to_end_seconds_per_frame']:8.3f}", flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
