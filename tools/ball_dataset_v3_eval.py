"""Build and freeze the ball tracker v3 EVALUATION corpus manifests.

This corpus is evaluation-only. Nothing here is ever trained on, so it carries no
``train`` split: clips are assigned to ``validation`` (used for architecture,
thresholds and abstention rules) or ``final_test`` (scored exactly once, after
everything is frozen).

Two invariants are enforced rather than assumed, because the whole point of a new
corpus is that the old one was consumed:

* **no shoot family may appear in both splits** -- enforced by the existing
  ``ball_dataset.validate_split`` contract, which refuses a group that straddles
  two splits;
* **no clip or family may come from the v2 dataset or the legacy human-labelled
  corpus** -- checked here against both manifests, because a v2 *train* clip
  reused as v3 evaluation is training leakage and would invalidate the result.

Assignment is deterministic: families are ordered by ``sha256(seed + group)`` and
dealt alternately, so the split is reproducible and was not chosen after seeing a
number. Balance is checked, not engineered: both splits must contain wide swing
footage and at least one ball-free sequence, or the tool refuses to freeze.

Usage:
    .venv-video-ai/bin/python3 tools/ball_dataset_v3_eval.py --freeze
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghostcaddie.video import ball_dataset as bds            # noqa: E402

DATA = ROOT / "data" / "ball_dataset_v3_eval"
ACQUIRED = ROOT / "out" / "ball_dataset_v3_eval" / "acquired.jsonl"
PLAN = DATA / "acquisition_plan.json"
V2_SPLIT = ROOT / "data" / "ball_dataset_v2" / "split.json"
LEGACY_SPLIT = ROOT / "data" / "ball_split.json"
DATASET_ID = "fairwayos-ball-v3-eval-2026-09"
SEED = "20260918-v3-eval"

#: Clips whose content class means they hold no playable ball. They are kept on
#: purpose: a tracker that stays silent on them is demonstrating the behaviour the
#: product needs, and a tracker that claims a ball on them is fabricating one.
BALL_FREE_TAGS = {"scenery"}


def load_acquired() -> list[dict]:
    records = {}
    for line in ACQUIRED.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            record = json.loads(line)
            records[record["clip_id"]] = record
    return [records[key] for key in sorted(records)]


def forbidden_identity() -> tuple[set[str], set[str]]:
    """Clip ids and shoot groups that this corpus may not touch."""
    clips: set[str] = set()
    groups: set[str] = set()
    for path in (V2_SPLIT, LEGACY_SPLIT):
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for clip in payload.get("clips", []):
            clips.add(clip["clip_id"])
            group = clip.get("source_group") or clip.get("source_id")
            if group:
                groups.add(group)
            # the legacy manifest names families only via source_id; derive the
            # same 4-digit family key this corpus uses so the check is comparable
            groups.add("pexels_" + str(clip["clip_id"]).removeprefix("pexels_")[:4])
    return clips, groups


def assign(groups: list[str]) -> dict[str, str]:
    """Deal families alternately in hash order. Deterministic, and fixed before use."""
    ordered = sorted(groups, key=lambda g: hashlib.sha256(
        (SEED + "|" + g).encode("utf-8")).hexdigest())
    return {group: ("validation" if index % 2 == 0 else "final_test")
            for index, group in enumerate(ordered)}


def build() -> tuple[dict, dict]:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    tags = {entry["clip_id"]: (entry.get("intended_tags") or ["unclassified"])[0]
            for entry in plan["entries"]}
    groups_by_clip = {entry["clip_id"]: entry["source_group"]
                      for entry in plan["entries"]}
    acquired = load_acquired()

    bad_clips, bad_groups = forbidden_identity()
    for record in acquired:
        clip_id = record["clip_id"]
        group = groups_by_clip[clip_id]
        if clip_id in bad_clips:
            raise SystemExit(f"{clip_id} is already in the v2 or legacy corpus; "
                             "it cannot be a v3 evaluation clip")
        if group in bad_groups:
            raise SystemExit(f"{clip_id} belongs to shoot family {group}, which is "
                             "already in the v2 or legacy corpus; that is leakage")

    placement = assign(sorted({groups_by_clip[r["clip_id"]] for r in acquired}))

    sources = {
        "schema_version": bds.SCHEMA_SOURCES,
        "dataset_id": DATASET_ID,
        "created": "2026-09-18",
        "purpose": ("Ball tracker v3 evaluation corpus. Evaluation only: no clip "
                    "here is ever used to train or to mine negatives."),
        "license_summary": {
            "provider": "pexels",
            "license": "Pexels License",
            "license_url": "https://www.pexels.com/license/",
            "license_text_reference": plan.get("license_text_reference"),
            "media_committed": False,
            "note": ("Media is rights-cleared stock footage held locally only. It is "
                     "never committed and never redistributed; this manifest plus "
                     "tools/ball_dataset_acquire.py are what another machine needs."),
        },
        "clips": [],
    }
    split = {
        "schema_version": bds.SCHEMA_SPLIT,
        "dataset_id": DATASET_ID,
        "status": "frozen",
        "frozen_on": "2026-09-18",
        "seed": SEED,
        "purpose": "evaluation only; there is deliberately no train split",
        "clips": [],
        "warnings": [
            "Frozen before any v3 model was trained, selected or thresholded.",
            "Clips are never moved between splits because results look bad.",
            "Final test is opt-in: tools require --final-test to read it.",
            ("Every shoot family here is disjoint from data/ball_dataset_v2 and "
             "data/ball_split.json. pexels_38025678 was dropped from the candidate "
             "pool for exactly this reason: it is a v2 train clip."),
            ("Clips tagged 'scenery' hold no playable ball and are scored as "
             "negative sequences: silence is the correct output."),
        ],
    }

    for record in acquired:
        clip_id = record["clip_id"]
        group = groups_by_clip[clip_id]
        probe = record.get("probe") or {}
        sources["clips"].append({
            "clip_id": clip_id,
            "provider": record["provider"],
            "source_identifier": record["source_identifier"],
            "source_url": record["source_url"],
            "download_url": record.get("download_url"),
            "title": record.get("title"),
            "acquisition_date": record["acquisition_date"],
            "license": record["license"],
            "license_url": record["license_url"],
            "license_text_reference": record.get("license_text_reference"),
            "rights_class": record["rights_class"],
            "source_group": group,
            "original_filename": record["original_filename"],
            "media_path": record["media_path"],
            "bytes": record["bytes"],
            "sha256": record["sha256"],
            "width": probe.get("width"),
            "height": probe.get("height"),
            "fps": probe.get("fps"),
            "frames": probe.get("frames"),
            "duration_s": probe.get("duration_s"),
            "content_class": tags.get(clip_id, "unclassified"),
        })
        split["clips"].append({
            "clip_id": clip_id,
            "split": placement[group],
            "source_group": group,
            "sha256": record["sha256"],
            "legacy_clip": False,
            "legacy_family": False,
            "sequence_id": clip_id,
            "content_class": tags.get(clip_id, "unclassified"),
            "holds_playable_ball": tags.get(clip_id) not in BALL_FREE_TAGS,
            "reason": ("deterministic family assignment: families ordered by "
                       f"sha256({SEED!r} + group) and dealt alternately"),
        })
    return sources, split


def check_balance(split: dict) -> list[str]:
    """Both splits must be scorable. A split with no ball-free sequence cannot
    measure fabrication, and one with no swing footage cannot measure tracking."""
    problems = []
    by_split = collections.defaultdict(list)
    for clip in split["clips"]:
        by_split[clip["split"]].append(clip)
    for name in ("validation", "final_test"):
        clips = by_split.get(name, [])
        if len(clips) < 5:
            problems.append(f"{name} has only {len(clips)} clips")
        if not any(c["content_class"] in ("swing", "putting") for c in clips):
            problems.append(f"{name} contains no swing or putting footage")
        if not any(not c["holds_playable_ball"] for c in clips):
            problems.append(f"{name} contains no ball-free negative sequence")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true",
                        help="write the manifests (otherwise just report)")
    args = parser.parse_args(argv)

    sources, split = build()
    bds.validate_sources(sources)
    legacy = bds.legacy_groups_from_split(LEGACY_SPLIT) if LEGACY_SPLIT.exists() else set()
    bds.validate_split(split, sources=sources, legacy_groups=legacy)

    problems = check_balance(split)
    if problems:
        for problem in problems:
            print(f"UNBALANCED: {problem}", file=sys.stderr)
        return 1

    counts = collections.Counter(c["split"] for c in split["clips"])
    print(f"clips={len(split['clips'])} families="
          f"{len({c['source_group'] for c in split['clips']})} {dict(counts)}")
    for name in ("validation", "final_test"):
        rows = [c for c in split["clips"] if c["split"] == name]
        kinds = collections.Counter(c["content_class"] for c in rows)
        print(f"  {name:11s} {len(rows):2d} clips  {dict(kinds)}")
        for clip in sorted(rows, key=lambda c: c["clip_id"]):
            print(f"      {clip['clip_id']:20s} {clip['source_group']:12s} "
                  f"{clip['content_class']:16s} "
                  f"ball={'yes' if clip['holds_playable_ball'] else 'no'}")

    if args.freeze:
        (DATA / "sources.json").write_text(json.dumps(sources, indent=1) + "\n")
        (DATA / "split.json").write_text(json.dumps(split, indent=1) + "\n")
        print(f"froze {DATA/'sources.json'} and {DATA/'split.json'}")
    else:
        print("(dry run; pass --freeze to write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
