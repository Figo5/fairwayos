#!/usr/bin/env python3
"""Tiger target-tracking benchmark runner (research-only, local).

Runs the EXISTING demo tracker (``ghostcaddie.demo.worker.measure_target`` +
``ghostcaddie.demo.tracking.resolve_track``) over the twelve reference clips in a
``fairwayos-target-reference.v1`` dataset and writes

* normalised predictions ``fairwayos-target-predictions.v1``;
* an evaluation report (JSON + markdown), produced by the stdlib-only evaluator in
  ``ghostcaddie.demo.target_benchmark``.

Four variants exist. ``clubhead_foreground`` and ``putter_redetect`` both leave the BALL
leg identical to ``baseline`` at the same code revision. ``lk_agreement`` enables
optical-flow agreement for both targets.

* ``baseline`` -- the tracker with the reference's own options, no override;
* ``lk_agreement`` -- the same tracker with ``lk_refine=true`` only;
* ``clubhead_foreground`` -- the same classic path for ``ball``, and for the
  ``clubhead`` target an ISOLATED experimental masked-foreground tracker
  (``ghostcaddie.demo.foreground_tracker``) that matches the seed patch's central
  foreground evidence instead of the whole 31 px context, refuses an ambiguous
  frame, and may re-acquire a genuinely lost target inside a bounded window. That
  candidate is NOT driven by the reference's per-clip options: every one of its
  constants is module-level and fixed, and the whole profile plus its source hash is
  declared in the report's ``runtime.candidate_profile``. Its per-frame raw
  metadata can be written with ``--raw-out``.
* ``putter_redetect`` -- a SECOND, unrelated candidate
  (``ghostcaddie.demo.putter_detector``), built after the 31 px masked template was
  found to regress and to follow clothing: instead of following anything, it is an
  UNTRAINED STRUCTURAL detector that re-segments compact bright/metal-like head
  evidence in every frame, checks native size/area/contrast/colour compatibility
  against the operator's seed and rejects thin shaft-only structure, oversized
  clothing and round ball-like distractors by local thin-shaft endpoint geometry. It
  reports its own honest method names (``putter_detector`` /
  ``putter_detector_reacquired``) and is applied ONLY to the ``clubhead`` leg of a
  clip whose declared ``stroke_type`` is ``putt`` (an explicit operator stroke hint,
  recorded in the runtime record). Every ball leg, and the clubhead leg of every
  other clip -- including ``full_swing`` -- runs the SAME classic path with the SAME
  options, unchanged by construction. It is not a semantic model and carries no
  trained weights, and a putt clip whose seed carries no compact-head evidence is
  refused rather than guessed.

What this runner will not do, by construction:

* it feeds the tracker **nothing** but the reference's own ``seed`` per target. The
  reference's coordinate labels, and any corrections or ``manual_segments`` block,
  are never read into the tracker -- they are the thing being scored;
* it does not resize, crop or otherwise mutate a frame: source frames are native
  and the coordinates are native pixels;
* it does not report a point outside the clip window or on a declared cut frame,
  and it does not let an interpolated point bridge a cut: those are structural
  resets, not labels;
* it validates the dataset (including every clip's frozen ``sha256``) before a
  single frame is decoded, and refuses rather than repairing.

Usage:

    /Users/giofiore/ghostcaddie-tour/.venv-fairway/bin/python \
        tools/tiger_target_benchmark.py \
        --dataset /Users/giofiore/fairway-os-runs/tiger-tracking-cycle/labels/tiger_targets.v1.json \
        --out /Users/giofiore/fairway-os-runs/tiger-tracking-cycle/results/baseline.predictions.json \
        --variant baseline

``--validate-only`` needs no cv2 (the contract and the evaluator are stdlib-only).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ghostcaddie.demo import target_benchmark as bench  # noqa: E402  (stdlib-only)

SUPPORTED_VARIANTS = ("baseline", "lk_agreement", "clubhead_foreground",
                      "putter_redetect")
#: What each variant changes and nothing else. ``baseline`` is the tracker with the
#: reference's own options and no override, so it stays byte-for-byte what it was
#: before this ablation existed; ``lk_agreement`` is the SAME tracker, the SAME
#: seeds, the SAME sparse labels, with ``lk_refine=true``: the template match must
#: then also agree with optical flow or the frame is reported unmeasured.
#: ``clubhead_foreground`` keeps ``baseline``'s tracker options for every target --
#: the ball leg runs exactly the classic code -- and swaps only the CLUBHEAD leg for
#: the isolated candidate in ``ghostcaddie.demo.foreground_tracker``, whose own
#: profile is fixed at module level and declared in the report runtime.
#: ``putter_redetect`` swaps the CLUBHEAD leg for a second, unrelated isolated
#: candidate (``ghostcaddie.demo.putter_detector``, an untrained structural detector
#: that re-segments head evidence afresh each frame) and only when the clip's declared
#: ``stroke_type`` is ``putt``: with any other stroke hint, or for the ball of any
#: clip, the classic code and options run exactly as ``baseline``.
VARIANT_OPTION_OVERRIDES: dict = {"baseline": {}, "lk_agreement": {"lk_refine": True},
                                  "clubhead_foreground": {}, "putter_redetect": {}}
VARIANT_NOTES = {
    "baseline": "the tracker with the reference's options; no override",
    "lk_agreement": "the same tracker with lk_refine=true only (template match must "
                    "agree with optical flow); no other option changed",
    "clubhead_foreground": "the same classic tracker and options for ball (byte-"
                           "identical), and the isolated masked-foreground candidate "
                           "for clubhead (fixed module-level profile, declared in "
                           "runtime.candidate_profile)",
    "putter_redetect": "the same classic tracker and options for every ball leg and for "
                       "every clubhead leg whose clip is not a declared putt, and the "
                       "isolated untrained STRUCTURAL putter-head detector for the "
                       "clubhead of a putt clip only (fixed module-level profile, "
                       "declared in runtime.candidate_profile)",
}
#: Which targets each variant replaces with the candidate engine. Only the clubhead
#: is in scope for this finite experiment; the ball baseline must not move.
FOREGROUND_ENGINE_TARGETS = ("clubhead",)
FOREGROUND_VARIANTS = ("clubhead_foreground",)
#: The second candidate arm: same target scope, but gated on the clip's stroke hint.
PUTTER_VARIANTS = ("putter_redetect",)
PUTTER_ENGINE_TARGETS = ("clubhead",)
#: The explicit operator stroke hint a clip must declare for the candidate arm to run
#: at all. It is a hint about the stroke, never a label about the head.
PUTTER_STROKE_TYPE = "putt"
#: Every variant whose candidate profile belongs in the report runtime.
CANDIDATE_VARIANTS = FOREGROUND_VARIANTS + PUTTER_VARIANTS
#: The candidate module each variant may drive, and where its honest method names live.
#: An explicit table, so the report can never name a method its module does not define
#: and ``clubhead_foreground`` keeps declaring exactly what it always declared.
CANDIDATE_MODULE_PATHS = {
    "clubhead_foreground": "ghostcaddie.demo.foreground_tracker",
    "putter_redetect": "ghostcaddie.demo.putter_detector",
}
CANDIDATE_METHOD_ATTRS = {
    "clubhead_foreground": {"observation": "METHOD_FOREGROUND",
                            "reacquired": "METHOD_REACQUIRED",
                            "seed_anchor": "METHOD_SEED"},
    "putter_redetect": {"observation": "METHOD_DETECTOR",
                        "reacquired": "METHOD_REACQUIRED",
                        "seed_anchor": "METHOD_SEED"},
}
RAW_SCHEMA = "fairwayos-target-foreground-raw.v1"


class RunnerError(RuntimeError):
    """The run could not be completed as promised."""


def engine_for(target_name: str, variant: str, *,
               stroke_type: Optional[str] = None) -> str:
    """Which tracker measures ``target_name`` under ``variant``: pure and testable.

    ``classic`` means ``measure_target`` + ``resolve_track`` with the reference's
    options; ``foreground`` and ``putter`` mean one of the two isolated candidates.
    Nothing else is available, and no variant changes what the other legs run. The
    ``putter`` arm is gated on the clip's declared stroke type: without a ``putt`` hint
    the clubhead leg stays on the classic path, so a full-swing clip is unchanged by
    construction rather than by promise.
    """
    if variant not in SUPPORTED_VARIANTS:
        raise RunnerError(f"unsupported variant {variant!r}; this runner implements "
                          f"{list(SUPPORTED_VARIANTS)}")
    if variant in FOREGROUND_VARIANTS and target_name in FOREGROUND_ENGINE_TARGETS:
        return "foreground"
    if (variant in PUTTER_VARIANTS and target_name in PUTTER_ENGINE_TARGETS
            and stroke_type == PUTTER_STROKE_TYPE):
        return "putter"
    return "classic"


def effective_options(block_options: Mapping[str, Any], variant: str) -> dict:
    """The options a variant actually runs with, variant override applied.

    Pure and testable without cv2: the prediction declares this, so the report can
    record what the run really used instead of what the reference happened to say.
    For ``clubhead_foreground`` this is the reference's tracker options unchanged --
    those are what the ball leg runs with -- and the candidate's own fixed profile is
    declared separately in the runtime record, never smuggled in here.
    """
    if variant not in VARIANT_OPTION_OVERRIDES:
        raise RunnerError(f"unsupported variant {variant!r}; this runner implements "
                          f"{list(SUPPORTED_VARIANTS)}")
    options = {key: value for key, value in dict(block_options).items()
               if key != "interpolation"}
    options.update(VARIANT_OPTION_OVERRIDES[variant])
    interpolation = dict(block_options.get("interpolation")
                         or {"enabled": False, "max_gap_frames": 0})
    return {**options, "interpolation": interpolation}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RunnerError(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RunnerError(f"{path} is not valid JSON: {exc}") from exc


def _manifest_check(dataset: Mapping[str, Any], manifest_path: Path) -> dict:
    """Bind the dataset to the showcase batch manifest, id by id and hash by hash.

    The brief scopes this benchmark to the twelve clips of that manifest; this check
    makes the scope a fact of the run rather than a promise.
    """
    manifest = _read_json(manifest_path)
    entries = {entry["id"]: entry for entry in manifest.get("clips") or []}
    problems = []
    for clip in dataset["clips"]:
        entry = entries.get(clip["id"])
        if entry is None:
            problems.append(f"{clip['id']} is not in the manifest")
            continue
        if entry.get("clip_sha256") != clip["sha256"]:
            problems.append(f"{clip['id']} sha256 differs from the manifest")
        declared = Path(entry.get("clip") or "")
        if declared.name and declared.name != Path(clip["path"]).name:
            problems.append(f"{clip['id']} path differs from the manifest")
    missing = sorted(set(entries) - {clip["id"] for clip in dataset["clips"]})
    if missing:
        problems.append(f"dataset omits manifest clip(s): {', '.join(missing)}")
    if problems:
        raise RunnerError("dataset does not match the showcase manifest: "
                          + "; ".join(problems))
    return {"manifest": str(manifest_path), "clips_checked": len(entries),
            "sha256_manifest": bench.sha256_file(manifest_path)}


def _verify_media(cap, clip: Mapping[str, Any]) -> dict:
    """Refuse a clip whose decoded media is not the clip the reference described."""
    import cv2

    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if (width, height) != (clip["width"], clip["height"]):
        raise RunnerError(f"{clip['id']}: decoded {width}x{height} but the reference "
                          f"declares {clip['width']}x{clip['height']}")
    if frames != clip["frame_count"]:
        raise RunnerError(f"{clip['id']}: decoded {frames} frames but the reference "
                          f"declares {clip['frame_count']}")
    declared_fps = clip["fps_rational"]["value"]
    return {"width": width, "height": height, "frame_count": frames,
            "container_fps": round(fps, 6), "declared_fps_rational": declared_fps,
            "container_fps_matches_declared": abs(fps - declared_fps) < 0.01,
            "note": ("the container's nominal frame rate is recorded, not trusted: it "
                     "is not evidence of the real capture cadence")}


def _point_dict(point: Any) -> dict:
    return {"source_frame": int(point.source_frame), "x": point.x, "y": point.y,
            "status": point.status, "origin": point.origin, "method": point.method,
            "score": point.score}


def run_clip(cap, clip: Mapping[str, Any], *, variant: str) -> tuple[dict, dict, dict]:
    """Track every seeded target of one clip.

    The seed is the only reference input the tracker sees; ``corrections`` is always
    empty and no labels or manual segments are read here. The BALL leg always runs
    the classic tracker: only a variant that names the clubhead as a candidate target
    (see :func:`engine_for`) replaces that one leg, and the candidate is handed the
    clip's structural cut frames -- source structure, not labels -- so it cannot
    re-acquire across a cut. The ``putter_redetect`` candidate additionally requires the
    clip's declared ``stroke_type`` to be ``putt``; with any other hint the clubhead leg
    takes the classic path with the classic options.
    """
    from ghostcaddie.demo.config import Point, TargetOptions
    from ghostcaddie.demo.tracking import resolve_track
    from ghostcaddie.demo.worker import measure_target

    start = int(clip["window"]["start_frame"])
    end = int(clip["window"]["end_frame"])
    interval_frames = list(range(start, end + 1))
    targets: dict = {}
    suppressions: dict = {}
    raw_metadata: dict = {}
    for target_name in clip["target_names"]:
        block = clip["targets"][target_name]
        seed = block["seed"]
        if seed is None:
            # No seed: nothing may be tracked, and this runner will not invent one.
            targets[target_name] = {"seed": None, "points": [],
                                    "options": effective_options(block["options"], variant)}
            suppressions[target_name] = {}
            continue
        declared_options = effective_options(block["options"], variant)
        engine = engine_for(target_name, variant,
                            stroke_type=clip.get("stroke_type"))
        if engine in ("foreground", "putter"):
            if engine == "foreground":
                from ghostcaddie.demo.foreground_tracker import track_foreground \
                    as track_leg
            else:
                from ghostcaddie.demo.putter_detector import track_putter as track_leg

            run = track_leg(cap, seed_frame=int(seed["frame"]),
                            seed_x=float(seed["x"]), seed_y=float(seed["y"]),
                            start_frame=start, end_frame=end,
                            cut_frames=clip["cut_frames"], target=target_name)
            points = run.points()
            kept, summary = bench.suppress_structural(points, window=clip["window"],
                                                      cut_frames=clip["cut_frames"])
            targets[target_name] = {"seed": {"frame": int(seed["frame"]),
                                             "x": float(seed["x"]),
                                             "y": float(seed["y"])},
                                    "options": declared_options,
                                    "points": kept}
            suppressions[target_name] = summary
            raw_metadata[target_name] = {"summary": run.summary(),
                                         "limitations": list(run.limitations),
                                         "frames": run.raw_frames}
            continue
        interpolation = declared_options["interpolation"]
        options = TargetOptions(**{key: value for key, value in declared_options.items()
                                   if key != "interpolation"})
        seed_point = Point(frame=int(seed["frame"]), x=float(seed["x"]),
                           y=float(seed["y"]), note=str(seed["note"]))
        raw = measure_target(cap, target_name, options, seed_point, [], start, end)
        track = resolve_track(target_name, options=options,
                              interval_frames=interval_frames, raw=raw,
                              seed=seed_point, corrections=[],
                              interpolation_enabled=bool(interpolation["enabled"]),
                              max_gap_frames=int(interpolation["max_gap_frames"]))
        points = [_point_dict(point) for point in track.points]
        kept, summary = bench.suppress_structural(points, window=clip["window"],
                                                  cut_frames=clip["cut_frames"])
        targets[target_name] = {"seed": {"frame": int(seed["frame"]),
                                         "x": float(seed["x"]),
                                         "y": float(seed["y"])},
                                "options": declared_options,
                                "points": kept}
        suppressions[target_name] = summary
    prediction = {"id": clip["id"], "sha256": clip["sha256"], "targets": targets}
    return prediction, suppressions, raw_metadata


def build_predictions(dataset: Mapping[str, Any], *, variant: str
                      ) -> tuple[dict, dict, dict, dict]:
    """Decode every clip of the dataset and emit the normalised predictions payload."""
    import cv2

    clips = []
    suppressions: dict = {}
    media: dict = {}
    raw_metadata: dict = {}
    for clip in dataset["clips"]:
        cap = cv2.VideoCapture(clip["path"])
        try:
            if not cap.isOpened():
                raise RunnerError(f"{clip['id']}: OpenCV could not open {clip['path']}")
            media[clip["id"]] = _verify_media(cap, clip)
            prediction, suppressed, raw = run_clip(cap, clip, variant=variant)
        finally:
            cap.release()
        clips.append(prediction)
        suppressions[clip["id"]] = suppressed
        if raw:
            raw_metadata[clip["id"]] = raw
    return ({"schema_version": bench.PREDICTIONS_SCHEMA, "variant": variant,
             "clips": clips}, suppressions, media, raw_metadata)


def _runtime_record(variant: str) -> dict:
    import cv2

    from ghostcaddie.demo import config as config_mod
    from ghostcaddie.demo import tracking as tracking_mod
    from ghostcaddie.demo import worker as worker_mod

    tool = Path(__file__).resolve()
    module = Path(bench.__file__).resolve()
    algorithms = {"worker.py": Path(worker_mod.__file__).resolve(),
                  "tracking.py": Path(tracking_mod.__file__).resolve(),
                  "config.py": Path(config_mod.__file__).resolve()}
    candidate: dict = {}
    if variant in CANDIDATE_VARIANTS:
        candidate = _candidate_profile_record(variant, algorithms)
    record = {"interpreter": sys.executable, "python": platform.python_version(),
              "platform": platform.platform(), "cv2": cv2.__version__,
              "tool": str(tool), "tool_sha256": bench.sha256_file(tool),
              "module": str(module), "module_sha256": bench.sha256_file(module),
              "algorithm_code_sha256": {name: {"path": str(path),
                                               "sha256": bench.sha256_file(path)}
                                        for name, path in algorithms.items()},
              "variant": variant,
              "variant_options_override": dict(VARIANT_OPTION_OVERRIDES[variant]),
              "variant_note": VARIANT_NOTES[variant],
              "corrections_fed_to_tracker": [],
              "reference_inputs_used_by_tracker": ["seed"],
              "tracker_scope": ("one forward leg per target from its seed; no backward "
                                "or both-direction leg is supported by this benchmark"),
              "labels_read_by_tracker": False}
    record.update(candidate)
    return record


def _candidate_profile_record(variant: str, algorithms: dict) -> dict:
    """The candidate's own runtime: its source hash, its scope and every constant.

    The report has to be able to say what actually ran. This is where the candidate
    profile goes -- NOT into the prediction's ``options``, which stay the classic
    tracker's options because that is what the ball leg really used. Which candidate
    module runs for which variant, and where its honest method names live, is a table
    (:data:`CANDIDATE_MODULE_PATHS`, :data:`CANDIDATE_METHOD_ATTRS`); the stroke gate
    recorded here is the same one :func:`engine_for` applies.
    """
    import importlib

    candidate_mod = importlib.import_module(CANDIDATE_MODULE_PATHS[variant])
    code = Path(str(candidate_mod.__file__)).resolve()
    algorithms[code.name] = code
    constants = candidate_mod.PROFILE
    stroke = PUTTER_STROKE_TYPE if variant in PUTTER_VARIANTS else None
    targets = list(PUTTER_ENGINE_TARGETS if stroke else FOREGROUND_ENGINE_TARGETS)
    return {
        "candidate_profile": {
            "name": candidate_mod.PROFILE_NAME,
            "variant": variant,
            "targets": targets,
            "applies_when": ({"clip.stroke_type": stroke} if stroke else None),
            "classic_otherwise": (
                "every other target leg, and the clubhead leg of every clip that does "
                "not declare that stroke type, runs the identical classic path with the "
                "identical options" if stroke else None),
            "engine_by_target": {name: engine_for(name, variant, stroke_type=stroke)
                                 for name in bench.TARGET_NAMES},
            "engine_by_target_note": (
                "computed with this candidate's own stroke gate applied; a routing "
                "statement, not a score"),
            "code": {"path": str(code), "sha256": bench.sha256_file(code)},
            "constants": constants,
            "constants_sha256": bench.sha256_text(bench.canonical_json(constants)),
            "methods": {key: getattr(candidate_mod, attr)
                        for key, attr in CANDIDATE_METHOD_ATTRS[variant].items()},
            "driven_by_reference_options": False,
            "note": (
                (f"{candidate_mod.PROFILE_NAME} is an untrained STRUCTURAL detector: no "
                 "trained weights, no object class, and no appearance template to "
                 "follow -- it re-segments compact head evidence in each frame. "
                 if stroke else
                 f"{candidate_mod.PROFILE_NAME} is the masked-foreground candidate. ")
                + "Its constants are module-level and fixed for every clip, hole and "
                  "frame; the reference's options are not read by it. The ball leg of "
                  "this same run is the classic tracker with the options the "
                  "predictions declare."),
        },
        "candidate_raw_metadata": {
            "available": "_raw-out path on this run (schema "
                         f"{RAW_SCHEMA}); a summary per target is in "
                         "candidate_raw_summary",
            "schema_version": RAW_SCHEMA,
        },
    }


def _summarise(results: Mapping[str, Any]) -> str:
    lines = []
    for split in bench.SPLITS:
        block = results["splits"][split]
        lines.append(f"{split}: {len(block['clips'])} clip(s)")
        for name, pooled in block["targets"].items():
            lines.append(
                f"  {name}: evaluated {pooled['evaluated_visible_frames']}, correct "
                f"{pooled['correct_within_tolerance']}, recall "
                f"{pooled['recall_within_tolerance']}, incorrect "
                f"{pooled['incorrect_visible_count']}, miss {pooled['miss_count']}, "
                f"negatives {pooled['negative_labeled_frames']} fp "
                f"{pooled['negative_fp_count']} ({pooled['negative_fp_rate']}), "
                f"longest missing run "
                f"{pooled['longest_missing_run_among_consecutive_visible_labels']}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Research-only target-tracking benchmark over the twelve "
                    "reference clips (sparse AI-visual reference, not ground truth).")
    parser.add_argument("--dataset", required=True, type=Path,
                        help="fairwayos-target-reference.v1 dataset JSON")
    parser.add_argument("--out", required=True, type=Path,
                        help="where to write the predictions JSON")
    parser.add_argument("--variant", default="baseline",
                        help="runner variant to evaluate: 'baseline' (the tracker with "
                             "the reference's options), 'lk_agreement' (the same "
                             "tracker with lk_refine=true only), "
                             "'clubhead_foreground' (the classic tracker for ball and "
                             "the isolated masked-foreground candidate for clubhead) or "
                             "'putter_redetect' (the same classic tracker and options "
                             "everywhere, except the clubhead of a clip whose declared "
                             "stroke_type is 'putt', which runs the isolated untrained "
                             "structural putter-head detector)")
    parser.add_argument("--raw-out", type=Path, default=None,
                        help="where to write the candidate's per-frame raw metadata "
                             "(produced by the clubhead_foreground and putter_redetect "
                             "variants; default: not written)")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="optional showcase batch manifest to bind the scope to "
                             "(clip id, path and sha256 are cross-checked)")
    parser.add_argument("--report-json", type=Path, default=None,
                        help="default: <out>.report.json")
    parser.add_argument("--report-md", type=Path, default=None,
                        help="default: <out>.report.md")
    parser.add_argument("--validate-only", action="store_true",
                        help="validate the dataset and stop (needs no cv2)")
    parser.add_argument("--no-hash-check", action="store_true",
                        help="skip clip file hashing (NOT for benchmarking; validation "
                             "and reporting only)")
    args = parser.parse_args(argv)

    if args.variant not in SUPPORTED_VARIANTS:
        raise RunnerError(f"unsupported variant {args.variant!r}; this runner implements "
                          f"{list(SUPPORTED_VARIANTS)}")
    print(f"variant: {args.variant} — {VARIANT_NOTES[args.variant]}")

    if args.no_hash_check and not args.validate_only:
        raise RunnerError("--no-hash-check is allowed only with --validate-only; tracking requires source hash verification")

    dataset, dataset_sha = bench.load_dataset(args.dataset,
                                              verify_hashes=not args.no_hash_check)
    provenance = dataset["provenance"]
    print(f"reference: {args.dataset} sha256 {dataset_sha[:16]}.. "
          f"({len(dataset['clips'])} clip(s), "
          f"label_type={provenance['label_type']}, ground_truth={provenance['ground_truth']})")
    for clip in dataset["clips"]:
        counts = ", ".join(
            f"{name}: seed={clip['targets'][name]['seed']['frame'] if clip['targets'][name]['seed'] else 'none'} "
            f"labels={len(clip['targets'][name]['labels'])}"
            for name in clip["target_names"])
        print(f"  {clip['id']:24s} {clip['split']:11s} window "
              f"[{clip['window']['start_frame']},{clip['window']['end_frame']}] "
              f"cuts={len(clip['cut_frames'])} {counts}")

    scope: dict = {}
    if args.manifest is not None:
        scope = _manifest_check(dataset, args.manifest)
        print(f"manifest: {args.manifest} — {scope['clips_checked']} clip(s) bound")

    if args.validate_only:
        print("validate-only: dataset accepted; nothing decoded, nothing written")
        return 0

    payload, suppressions, media, raw_metadata = build_predictions(dataset,
                                                                  variant=args.variant)
    predictions_text = bench.canonical_json(payload)
    predictions_sha = bench.sha256_text(predictions_text)
    validated_predictions = bench.validate_predictions(payload, dataset)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(predictions_text, encoding="utf-8")
    print(f"predictions: {args.out} sha256 {predictions_sha[:16]}..")

    if raw_metadata:
        raw_payload = {"schema_version": RAW_SCHEMA, "variant": args.variant,
                       "predictions_path": str(args.out),
                       "predictions_sha256": predictions_sha,
                       "note": ("per-frame raw metadata of the experimental foreground "
                                "candidate: what was searched for, what was measured, "
                                "which gates passed and why a frame carried no point. "
                                "It is raw evidence, not a score."),
                       "clips": raw_metadata}
        if args.raw_out is not None:
            raw_text = bench.canonical_json(raw_payload)
            args.raw_out.parent.mkdir(parents=True, exist_ok=True)
            args.raw_out.write_text(raw_text, encoding="utf-8")
            print(f"candidate raw: {args.raw_out} sha256 "
                  f"{bench.sha256_text(raw_text)[:16]}..")
        else:
            print("candidate raw: not written (pass --raw-out to keep the per-frame "
                  "records)")

    results = bench.evaluate(
        dataset, validated_predictions, suppressions=suppressions,
        dataset_path=str(args.dataset), dataset_sha256=dataset_sha,
        predictions_path=str(args.out), predictions_sha256=predictions_sha,
        runtime=_runtime_record(args.variant))
    results["media_checks"] = media
    results["scope_binding"] = scope
    if raw_metadata:
        results["candidate_raw_summary"] = {
            clip_id: {target: {"summary": block["summary"],
                               "limitations": block["limitations"]}
                      for target, block in targets.items()}
            for clip_id, targets in raw_metadata.items()}
        results["candidate_raw_summary_note"] = (
            "summaries of the experimental foreground candidate's own per-frame "
            "records (schema " + RAW_SCHEMA + "). They explain acceptance, refusal and "
            "re-acquisition; the scored numbers are the tables above, and the FULL "
            "per-frame records are written only with --raw-out.")
    report_json = args.report_json or args.out.with_suffix(args.out.suffix + ".report.json")
    report_md = args.report_md or args.out.with_suffix(args.out.suffix + ".report.md")
    report_text = bench.canonical_json(results)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(report_text, encoding="utf-8")
    report_md.write_text(bench.render_markdown(results), encoding="utf-8")
    print(f"report: {report_json} sha256 {bench.sha256_text(report_text)[:16]}..")
    print(f"report: {report_md}")
    print(_summarise(results))
    return 0


def _entry() -> int:
    try:
        return main()
    except bench.TargetBenchmarkError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except RunnerError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(_entry())
