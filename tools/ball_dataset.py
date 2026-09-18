"""Ball dataset v2 tooling: manifest, split, validation, statistics, QA.

Every subcommand is read-only except ``manifest``, ``split`` and the label
mutators (``set``, ``propose``, ``resolve-proposal``), and none of them may touch
canonical perception: ball tracking stays UNAVAILABLE in the canonical workflow
regardless of what is in this dataset.

Development commands default to train/validation. Reaching final test requires an
explicit ``--final-test``.

Usage:
    python3 tools/ball_dataset.py manifest
    python3 tools/ball_dataset.py split --freeze
    python3 tools/ball_dataset.py validate
    python3 tools/ball_dataset.py stats
    python3 tools/ball_dataset.py status
    python3 tools/ball_dataset.py review
    python3 tools/ball_dataset.py qa --clip-id pexels_6573644 --out out/ball_dataset_v2/qa
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghostcaddie.video import ball_dataset as bd  # noqa: E402
from ghostcaddie.video.errors import VideoContractError  # noqa: E402

SOURCES = ROOT / "data" / "ball_dataset_v2" / "sources.json"
SPLIT = ROOT / "data" / "ball_dataset_v2" / "split.json"
PLAN = ROOT / "data" / "ball_dataset_v2" / "acquisition_plan.json"
ACQUIRED = ROOT / "out" / "ball_dataset_v2" / "acquired.jsonl"
ANNOTATIONS = ROOT / "data" / "ball_dataset_v2" / "annotations"
LEGACY_SPLIT = ROOT / "data" / "ball_split.json"
DATASET_ID = "fairwayos-ball-v2-2026-09"
DEFAULT_SEED = 20260918

SPLIT_TARGETS = {bd.SPLIT_TRAIN: 0.60, bd.SPLIT_VALIDATION: 0.20,
                 bd.SPLIT_FINAL_TEST: 0.20}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_sources_or_die() -> dict:
    if not SOURCES.exists():
        raise SystemExit(f"no sources manifest at {SOURCES}; run `manifest` first")
    return bd.load_sources(SOURCES)


def load_split_or_none() -> dict | None:
    if not SPLIT.exists():
        return None
    return bd.load_split(SPLIT, sources=load_sources_or_die(),
                         legacy_groups=legacy_groups())


def legacy_groups() -> set[str]:
    if not LEGACY_SPLIT.exists():
        return set()
    return bd.legacy_groups_from_split(LEGACY_SPLIT)


def annotation_files() -> list[Path]:
    if not ANNOTATIONS.exists():
        return []
    return sorted(ANNOTATIONS.glob("*.json"))


def reviewer_identity(args) -> dict:
    return {"kind": args.reviewer_kind, "id": args.reviewer, "date": args.date,
            "model": args.model}


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_manifest(args) -> int:
    if not PLAN.exists():
        raise SystemExit(f"no acquisition plan at {PLAN}")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if plan.get("schema_version") != bd.SCHEMA_ACQUISITION_PLAN:
        raise SystemExit(f"unexpected plan schema {plan.get('schema_version')!r}")
    planned = {entry["clip_id"]: entry for entry in plan["entries"]}
    acquired = {}
    if ACQUIRED.exists():
        for line in ACQUIRED.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                acquired[record["clip_id"]] = record

    clips = []
    for clip_id, entry in sorted(planned.items()):
        record = acquired.get(clip_id)
        if record is None:
            print(f"skip {clip_id}: not acquired yet")
            continue
        probe = record.get("probe") or {}
        if not probe.get("width"):
            print(f"skip {clip_id}: probe failed ({probe.get('error')})")
            continue
        clips.append({
            "clip_id": clip_id,
            "provider": entry["provider"],
            "source_identifier": entry["source_identifier"],
            "source_url": entry["source_url"],
            "download_url": entry["download_url"],
            "title": entry.get("title_from_search") or None,
            "acquisition_date": record["acquisition_date"],
            "license": plan["license"],
            "license_url": plan["license_url"],
            "license_text_reference": plan.get("license_text_reference"),
            "rights_class": record.get("rights_class") or "stock_license_reusable",
            "original_filename": record["original_filename"],
            "media_path": record["media_path"],
            "bytes": record["bytes"],
            "sha256": record["sha256"],
            "width": probe["width"],
            "height": probe["height"],
            "fps": probe.get("fps"),
            "frames": probe.get("frames"),
            "duration_s": probe.get("duration_s"),
            "codec": probe.get("codec"),
            "container": probe.get("container"),
            "source_group": entry["source_group"],
            "intended_tags": entry.get("intended_tags", []),
            "tags": [],
            "notes": entry.get("notes", ""),
        })

    payload = {
        "schema_version": bd.SCHEMA_SOURCES,
        "dataset_id": DATASET_ID,
        "created": plan.get("created"),
        "license_summary": {
            "provider": plan["provider"],
            "license": plan["license"],
            "license_url": plan["license_url"],
            "license_text_reference": plan.get("license_text_reference"),
            "media_committed": False,
            "note": "Media is rights-cleared stock footage held locally only. It is "
                    "never committed and never redistributed; the manifest plus "
                    "tools/ball_dataset_acquire.py are what another machine needs.",
        },
        "clips": clips,
    }
    bd.validate_sources(payload)
    SOURCES.parent.mkdir(parents=True, exist_ok=True)
    SOURCES.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {SOURCES.relative_to(ROOT)} with {len(clips)} clips")
    return 0


def cmd_split(args) -> int:
    sources = load_sources_or_die()
    if SPLIT.exists() and not args.force:
        existing = bd.load_split(SPLIT, sources=sources, legacy_groups=legacy_groups())
        counts = collections.Counter(clip["split"] for clip in existing["clips"])
        print(f"{SPLIT.relative_to(ROOT)} already exists and is frozen: "
              f"{dict(counts)}")
        print("Refusing to rewrite a frozen split without --force. Moving clips "
              "between splits after seeing results is how a benchmark becomes a "
              "development set.")
        return 1
    if not args.freeze:
        print("refusing to write a split without --freeze")
        return 1

    legacy = legacy_groups()
    by_group: dict[str, list[dict]] = collections.defaultdict(list)
    for clip in sources["clips"]:
        by_group[clip["source_group"]].append(clip)

    total = sum(len(v) for v in by_group.values())
    wanted = {name: fraction * total for name, fraction in SPLIT_TARGETS.items()}
    order = sorted(by_group, key=lambda group: hashlib.sha256(
        f"{args.seed}:{group}".encode()).hexdigest())

    assignment, placed = {}, collections.Counter()
    for group in order:
        clips = by_group[group]
        scores = {}
        for name, fraction in SPLIT_TARGETS.items():
            if name == bd.SPLIT_FINAL_TEST and group in legacy:
                continue
            deficit = wanted[name] - placed[name]
            scores[name] = deficit
        if not scores:
            raise SystemExit(f"cannot place group {group}: no eligible split")
        chosen = max(scores.items(), key=lambda item: (item[1], item[0]))[0]
        assignment[group] = chosen
        placed[chosen] += len(clips)

    clips_out = []
    for clip in sorted(sources["clips"], key=lambda c: c["clip_id"]):
        split = assignment[clip["source_group"]]
        clips_out.append({
            "clip_id": clip["clip_id"],
            "split": split,
            "source_group": clip["source_group"],
            "sha256": clip["sha256"],
            "legacy_clip": False,
            "legacy_family": clip["source_group"] in legacy,
            "sequence_id": clip["clip_id"],
            "subject_id": clip["source_group"],
            "reason": ("deterministic group assignment: groups ordered by "
                       f"sha256('{args.seed}:<group>'), placed into the most "
                       "under-filled split by clip count; final test excludes "
                       "groups shared with the consumed v1 corpus"),
        })

    payload = {
        "schema_version": bd.SCHEMA_SPLIT,
        "dataset_id": DATASET_ID,
        "status": "frozen",
        "frozen_on": args.date,
        "seed": args.seed,
        "targets": SPLIT_TARGETS,
        "legacy_corpus": {
            "path": "data/ball_split.json",
            "status": "development-consumed",
            "groups": sorted(legacy),
            "rule": "no group shared with the consumed v1 corpus may enter final test",
        },
        "clips": clips_out,
        "warnings": [
            "Frozen before any tracker, baseline or threshold was run on v2.",
            "Clips are never moved between splits because results look bad.",
            "Final test is opt-in: tools require --final-test to read it.",
        ],
    }
    bd.validate_split(payload, sources=sources, legacy_groups=legacy)
    SPLIT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    counts = collections.Counter(clip["split"] for clip in clips_out)
    groups = collections.Counter(clip["source_group"] for clip in clips_out
                                 if clip["split"] == bd.SPLIT_FINAL_TEST)
    print(f"froze {SPLIT.relative_to(ROOT)}: {dict(counts)}")
    print(f"final test groups ({len(groups)}): {sorted(groups)}")
    return 0


def cmd_list(args) -> int:
    sources = load_sources_or_die()
    split = load_split_or_none()
    split_of = {clip["clip_id"]: clip["split"] for clip in split["clips"]} if split else {}
    annotated = {path.stem for path in annotation_files()}
    print(f"{'clip_id':24s} {'split':11s} {'labelled':>8s}  tags")
    for clip in sorted(sources["clips"], key=lambda c: c["clip_id"]):
        clip_id = clip["clip_id"]
        label = split_of.get(clip_id, "-")
        count = "-"
        if clip_id in annotated:
            annotation = bd.load_annotation(
                bd.annotation_path_for(ANNOTATIONS, clip_id), sources=sources, bind=False)
            count = str(len(annotation["frames"]))
        print(f"{clip_id:24s} {label:11s} {count:>8s}  {','.join(clip.get('tags') or [])}")
    print(f"\nclips={len(sources['clips'])} annotated={len(annotated)}")
    return 0


def cmd_validate(args) -> int:
    sources = load_sources_or_die()
    split = load_split_or_none()
    if split is None:
        print("no frozen split yet; validating manifest and annotations only")
        split = {"schema_version": bd.SCHEMA_SPLIT, "status": "frozen", "clips": []}
    try:
        result = bd.validate_dataset(
            sources=sources, split=split, annotation_paths=annotation_files(),
            project_root=ROOT, legacy_groups=legacy_groups(),
            check_media=not args.no_media)
    except VideoContractError as exc:
        print(f"DATASET VALIDATION FAILED\n{exc}")
        return 1
    flags = bd.dataset_review_flags(sources=sources,
                                    annotation_paths=annotation_files(),
                                    project_root=ROOT, bind=False)
    labelled = sum(item["labelled_frames"] for item in result["summaries"])
    print(f"OK: {result['clips']} clips, {result['annotated_clips']} annotated, "
          f"{labelled} labelled frames, {len(flags)} review flags")
    if flags and args.show_flags:
        for flag in flags:
            print(f"  flag {flag['clip_id']} frame {flag['frame']}: {flag['kind']} "
                  f"- {flag['detail']}")
    return 0


def cmd_stats(args) -> int:
    sources = load_sources_or_die()
    split = load_split_or_none() or {"schema_version": bd.SCHEMA_SPLIT,
                                     "status": "frozen", "clips": []}
    report = bd.dataset_statistics(sources=sources, split=split,
                                   annotation_paths=annotation_files(),
                                   project_root=ROOT, bind=False)
    if args.camera_motion:
        for name, bucket in report.items():
            bucket["camera_motion"] = camera_motion(
                sources=sources, split=split, split_name=name)
    if args.json:
        print(json.dumps(report, indent=1, sort_keys=True))
        return 0
    for name in list(bd.SPLITS) + ["unassigned"]:
        bucket = report[name]
        print(f"=== {name}")
        print(f"  clips={bucket['clips']} annotated={bucket['annotated_clips']} "
              f"labelled_frames={bucket['labelled_frames']}")
        print(f"  states={bucket['states']}")
        print(f"  origin={bucket['origin']}")
        print(f"  runs: visible={bucket['visible_runs']} "
              f"absent={bucket['absent_runs']} "
              f"longest_visible={bucket['longest_visible_run']} "
              f"longest_absent={bucket['longest_absent_run']} "
              f"isolated_visible={bucket['isolated_visible']}")
        print(f"  resolutions={bucket['resolutions']}")
        print(f"  fps={bucket['fps']} groups={len(bucket['groups'])}")
        print(f"  ball_diameter_px={bucket['ball_diameter_px']} "
              f"(radius on {bucket['radii_with_radius']} frames)")
        print(f"  hard_negative_tags={bucket['tags']}")
        print(f"  diversity(intended)={bucket.get('intended_tags')}")
        if bucket.get("camera_motion"):
            print(f"  camera_motion={bucket['camera_motion']}")
    return 0


def camera_motion(*, sources: dict, split: dict, split_name: str,
                  per_clip_frames: int = 24) -> dict:
    """Median per-frame global translation on the labelled windows, measured.

    This is a real decode-based measurement, not an inference from labels. It is
    reported per split because camera motion is one of the axes a tracker has to
    survive."""
    try:
        import cv2  # noqa: F401
        import numpy as np
        from ghostcaddie.video.ball_pipeline import estimate_camera_motion
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"measured": False, "reason": f"{type(exc).__name__}: {exc}"}
    split_of = {clip["clip_id"]: clip["split"] for clip in split.get("clips", [])}
    by_id = bd.sources_by_id(sources)
    values, per_clip = [], {}
    for path in annotation_files():
        annotation = bd.load_annotation(path, sources=sources, bind=False)
        clip_id = annotation["clip_id"]
        if split_of.get(clip_id, "unassigned") != split_name:
            continue
        media = ROOT / by_id[clip_id]["media_path"]
        if not media.exists():
            continue
        indices = sorted(int(k) for k in annotation["frames"])[:per_clip_frames]
        if len(indices) < 2:
            continue
        capture = cv2.VideoCapture(str(media))
        if not capture.isOpened():
            continue
        magnitude = []
        previous = None
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous is not None:
                dx, dy = estimate_camera_motion(previous, gray)
                magnitude.append(float(np.hypot(dx, dy)))
            previous = gray
        capture.release()
        if magnitude:
            median = statistics.median(magnitude)
            width = by_id[clip_id]["width"]
            per_clip[clip_id] = {"median_px": round(median, 2),
                                 "fraction_of_width": round(median / width, 5),
                                 "frames_measured": len(magnitude)}
            values.append(median / width)
    if not values:
        return {"measured": False, "reason": "no decodable labelled clips in this split"}
    median_fraction = statistics.median(values)
    bucket = ("static" if median_fraction < 0.002 else
              "low" if median_fraction < 0.006 else
              "moderate" if median_fraction < 0.02 else "high_jitter")
    return {"measured": True, "clips": len(values),
            "median_fraction_of_width": round(median_fraction, 5),
            "bucket": bucket, "per_clip": per_clip}


def cmd_status(args) -> int:
    sources = load_sources_or_die()
    split = load_split_or_none()
    result = bd.media_status(sources=sources, split=split, project_root=ROOT)
    if args.json:
        print(json.dumps(result, indent=1, sort_keys=True))
        return 0
    exit_code = 0
    for name in list(bd.SPLITS) + ["unassigned"]:
        bucket = result.get(name)
        if bucket is None or bucket["total"] == 0:
            continue
        print(f"{name.upper()}: {bucket['present']}/{bucket['total']} clips present")
        for missing in bucket["missing"]:
            exit_code = 1
            print(f"  MISSING {missing['clip_id']}")
            print(f"    expected sha256 {missing['expected_sha256']}")
            print(f"    source {missing['source']}")
            if missing.get("reason"):
                print(f"    {missing['reason']}")
    if exit_code:
        print("\nMedia is not committed. Re-acquire with "
              "tools/ball_dataset_acquire.py, then re-run status.")
    return exit_code


def cmd_review(args) -> int:
    sources = load_sources_or_die()
    paths = ([bd.annotation_path_for(ANNOTATIONS, args.clip_id)] if args.clip_id
             else annotation_files())
    flags = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"no annotation at {path}")
        annotation = bd.load_annotation(path, sources=sources, bind=False)
        flags.extend(bd.review_flags(annotation, project_root=ROOT))
    for flag in flags:
        print(f"{flag['clip_id']:24s} frame {flag['frame']:>5d} {flag['kind']:18s} "
              f"{flag['detail']}")
    print(f"\n{len(flags)} flags for a human to adjudicate. Nothing was changed: "
          "the tooling never rewrites a label.")
    return 0


def cmd_qa(args) -> int:
    """Write overlay sheets: label on its own frame, magnified crop, state, ids."""
    import cv2

    sources = load_sources_or_die()
    split = load_split_or_none()
    split_of = {clip["clip_id"]: clip["split"] for clip in split["clips"]} if split else {}
    paths = ([bd.annotation_path_for(ANNOTATIONS, args.clip_id)] if args.clip_id
             else annotation_files())
    if args.split:
        if split is None:
            raise SystemExit("no frozen split yet")
        if args.split == bd.SPLIT_FINAL_TEST and not args.final_test:
            raise SystemExit("final test is opt-in: pass --final-test")
        wanted = bd.clips_for_split(split, args.split,
                                    allow_final_test=args.final_test)
        paths = [p for p in paths if p.stem in wanted]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in paths:
        if not path.exists():
            continue
        annotation = bd.load_annotation(path, sources=sources, bind=False)
        clip_id = annotation["clip_id"]
        media = ROOT / annotation["video"]
        if not media.exists():
            print(f"skip {clip_id}: media missing")
            continue
        indices = sorted(int(k) for k in annotation["frames"])
        if args.limit:
            indices = indices[:args.limit]
        capture = cv2.VideoCapture(str(media))
        width, height = annotation["width"], annotation["height"]
        scale = min(1.0, args.width / width)
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            entry = annotation["frames"][str(index)]
            canvas = cv2.resize(frame, (int(width * scale), int(height * scale)))
            if entry["state"] == bd.STATE_VISIBLE:
                cx, cy = int(entry["x"] * scale), int(entry["y"] * scale)
                radius = entry.get("radius")
                radius_px = int((radius or 0.02 * width) * scale)
                cv2.drawMarker(canvas, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS,
                               radius_px * 2, 2)
                cv2.circle(canvas, (cx, cy), max(radius_px, 6), (0, 200, 0), 2)
            header = (f"{clip_id} f{index} {entry['state']} "
                      f"origin={entry['origin']}")
            cv2.rectangle(canvas, (0, 0), (int(width * scale), 34), (0, 0, 0), -1)
            cv2.putText(canvas, header, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
            target = out_dir / f"{clip_id}_f{index:06d}.jpg"
            cv2.imwrite(str(target), canvas)
            written += 1
        capture.release()
    print(f"wrote {written} QA sheets to {out_dir} "
          f"(gitignored; QA media is never committed)")
    return 0


def cmd_set(args) -> int:
    sources = load_sources_or_die()
    annotation = bd.load_or_new_annotation(
        annotations_dir=ANNOTATIONS, sources=sources, clip_id=args.clip_id,
        project_root=ROOT, annotator=args.annotator,
        reviewer=reviewer_identity(args), bind=not args.no_bind)
    if args.state == bd.STATE_VISIBLE:
        if args.x is None or args.y is None:
            raise SystemExit("VISIBLE requires --x and --y; a visible frame without "
                             "a coordinate is not a label")
        updated = bd.set_label(annotation, args.frame, state=args.state,
                               origin=args.origin, reviewer=args.reviewer,
                               x=args.x, y=args.y, radius=args.radius,
                               tags=args.tag, confidence=args.confidence,
                               note=args.note)
    else:
        if args.x is not None or args.y is not None:
            raise SystemExit(f"{args.state} forbids coordinates")
        if args.state == bd.STATE_NOT_VISIBLE and not args.reason:
            raise SystemExit("NOT_VISIBLE requires --reason")
        if args.state == bd.STATE_OCCLUDED and not (args.reason or args.explanation):
            raise SystemExit("OCCLUDED requires --reason or --explanation")
        updated = bd.set_label(annotation, args.frame, state=args.state,
                               origin=args.origin, reviewer=args.reviewer,
                               reason=args.reason, explanation=args.explanation,
                               candidate_region=args.candidate_region,
                               tags=args.tag, confidence=args.confidence,
                               note=args.note)
    bd.save_annotation(bd.annotation_path_for(ANNOTATIONS, args.clip_id), updated,
                       sources=sources, project_root=None if args.no_bind else ROOT)
    summary = bd.annotation_summary(updated)
    print(f"{args.clip_id} frame {args.frame} -> {args.state}; "
          f"labelled={summary['labelled_frames']} states={summary['states']}")
    return 0


def cmd_propose(args) -> int:
    sources = load_sources_or_die()
    annotation = bd.load_or_new_annotation(
        annotations_dir=ANNOTATIONS, sources=sources, clip_id=args.clip_id,
        project_root=ROOT, annotator=args.annotator, bind=not args.no_bind)
    updated = bd.set_proposal(annotation, args.frame, x=args.x, y=args.y,
                              method=args.method, status=args.status,
                              confidence=args.confidence)
    bd.save_annotation(bd.annotation_path_for(ANNOTATIONS, args.clip_id), updated,
                       sources=sources, project_root=None if args.no_bind else ROOT)
    print(f"{args.clip_id} frame {args.frame}: proposal recorded as "
          f"{bd.PROPOSAL_ORIGIN}/{args.status}. A proposal is not a label.")
    return 0


def cmd_resolve(args) -> int:
    sources = load_sources_or_die()
    path = bd.annotation_path_for(ANNOTATIONS, args.clip_id)
    annotation = bd.load_annotation(path, sources=sources, bind=not args.no_bind)
    updated = bd.resolve_proposal(annotation, args.frame, status=args.status)
    bd.save_annotation(path, updated, sources=sources,
                       project_root=None if args.no_bind else ROOT)
    print(f"{args.clip_id} frame {args.frame}: proposal {args.status}")
    return 0


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("manifest", help="build sources.json from the plan + downloads")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("split", help="freeze a group-disjoint split")
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="rewrite an existing frozen split (discouraged)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--date", default="2026-09-18")
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("list", help="clips, split and annotation progress")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("validate", help="validate manifest, annotations and split")
    p.add_argument("--final-test", action="store_true")
    p.add_argument("--no-media", action="store_true",
                   help="skip re-hashing media bytes")
    p.add_argument("--show-flags", action="store_true")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("stats", help="dataset statistics per split")
    p.add_argument("--json", action="store_true")
    p.add_argument("--final-test", action="store_true")
    p.add_argument("--camera-motion", action="store_true",
                   help="decode labelled windows and measure camera motion")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("status", help="which media is present or missing")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("review", help="flag suspicious labels for adjudication")
    p.add_argument("--clip-id")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("qa", help="write overlay QA sheets (gitignored)")
    p.add_argument("--clip-id")
    p.add_argument("--split", choices=list(bd.SPLITS))
    p.add_argument("--final-test", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--out", default=str(ROOT / "out" / "ball_dataset_v2" / "qa"))
    p.set_defaults(func=cmd_qa)

    p = sub.add_parser("set", help="record one reviewed frame")
    p.add_argument("--clip-id", required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--state", required=True, choices=list(bd.STATES))
    p.add_argument("--origin", choices=list(bd.LABEL_ORIGINS), required=True)
    p.add_argument("--x", type=float)
    p.add_argument("--y", type=float)
    p.add_argument("--radius", type=float)
    p.add_argument("--reason")
    p.add_argument("--explanation")
    p.add_argument("--candidate-region", type=float, nargs=4)
    p.add_argument("--tag", action="append", choices=list(bd.HARD_NEGATIVE_TAGS))
    p.add_argument("--confidence", choices=["high", "medium", "low"])
    p.add_argument("--note")
    p.add_argument("--annotator", default="unattributed")
    p.add_argument("--reviewer", default=None)
    p.add_argument("--reviewer-kind", choices=["human", "agent", "mixed"],
                   default="agent")
    p.add_argument("--model", default=None)
    p.add_argument("--date", default="2026-09-18")
    p.add_argument("--no-bind", action="store_true")
    p.set_defaults(func=cmd_set)

    p = sub.add_parser("propose", help="record a machine proposal (never a label)")
    p.add_argument("--clip-id", required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--x", type=float)
    p.add_argument("--y", type=float)
    p.add_argument("--status", choices=["pending", "accepted", "rejected"],
                   default="pending")
    p.add_argument("--confidence", choices=["high", "medium", "low"])
    p.add_argument("--annotator", default="unattributed")
    p.add_argument("--no-bind", action="store_true")
    p.set_defaults(func=cmd_propose)

    p = sub.add_parser("resolve-proposal", help="accept or reject a proposal")
    p.add_argument("--clip-id", required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--status", required=True, choices=["accepted", "rejected"])
    p.add_argument("--no-bind", action="store_true")
    p.set_defaults(func=cmd_resolve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except VideoContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
