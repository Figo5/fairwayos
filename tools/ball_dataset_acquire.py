"""Acquire rights-cleared research media for ball dataset v2.

Media is NEVER committed. This tool exists so that the committed manifest can be
re-acquired elsewhere: it downloads each planned clip from its recorded provider
URL, verifies the bytes, probes them, and appends a provenance record that the
dataset manifest is built from.

Design rules:

* the acquisition plan is committed and is the only source of clip identity;
* a clip already on disk is never re-downloaded, and never overwritten;
* a file whose SHA-256 does not match the recorded one is reported as a mismatch
  and left alone -- a differently hashed file is a different clip, not a
  substitute;
* nothing here writes a label, and nothing here may be treated as ground truth.

Usage:
    python3 tools/ball_dataset_acquire.py --plan data/ball_dataset_v2/acquisition_plan.json
    python3 tools/ball_dataset_acquire.py --only pexels_6573644 --only pexels_5200681
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "data" / "ball_dataset_v2" / "acquisition_plan.json"
DEFAULT_MEDIA_ROOT = ROOT / "out" / "ball_dataset_v2" / "media"
DEFAULT_OUT = ROOT / "out" / "ball_dataset_v2" / "acquired.jsonl"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict:
    command = ["ffprobe", "-v", "error", "-print_format", "json",
               "-show_entries",
               "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,"
               "duration:format=duration,size,format_name",
               str(path)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": f"{type(exc).__name__}"}
    if result.returncode != 0:
        return {"error": (result.stderr or "").strip()[:300]}
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "ffprobe returned non-JSON"}
    stream = next((s for s in payload.get("streams", []) if s.get("width")), {})
    fmt = payload.get("format", {})

    def fps(value):
        if not value or value == "0/0":
            return None
        if "/" in value:
            num, den = value.split("/", 1)
            den = float(den)
            return round(float(num) / den, 6) if den else None
        return round(float(value), 6)

    duration = fmt.get("duration") or stream.get("duration")
    fps_value = fps(stream.get("avg_frame_rate")) or fps(stream.get("r_frame_rate"))
    raw_frames = stream.get("nb_frames")
    frames = int(raw_frames) if isinstance(raw_frames, str) and raw_frames.isdigit() else (
        int(float(duration) * fps_value) if duration and fps_value else None)
    return {
        "codec": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": fps_value,
        "duration_s": round(float(duration), 6) if duration else None,
        "frames": frames,
        "container": fmt.get("format_name"),
        "probe_bytes": int(fmt["size"]) if str(fmt.get("size", "")).isdigit() else None,
    }


def download(url: str, destination: Path) -> tuple[bool, str]:
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "video/*,*/*;q=0.8",
        "Referer": "https://www.pexels.com/",
    })
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            if response.status != 200:
                return False, f"http {response.status}"
            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith(("video/", "application/octet-stream")):
                return False, f"content-type {content_type!r}"
            with temporary.open("wb") as handle:
                while True:
                    block = response.read(CHUNK)
                    if not block:
                        break
                    handle.write(block)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        temporary.unlink(missing_ok=True)
        return False, f"{type(exc).__name__}: {exc}"
    if temporary.stat().st_size < 4096:
        temporary.unlink(missing_ok=True)
        return False, "response too small to be a video"
    with temporary.open("rb") as handle:
        header = handle.read(12)
    if b"ftyp" not in header and b"webm" not in header:
        temporary.unlink(missing_ok=True)
        return False, "response is not an ISO media / webm file"
    temporary.replace(destination)
    return True, "downloaded"


def load_records(path: Path) -> dict:
    records = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        records[record.get("clip_id")] = record
    return records


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", action="append", default=None,
                        help="restrict to these clip ids (repeatable)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pause", type=float, default=0.5,
                        help="seconds between downloads (be polite)")
    parser.add_argument("--refresh", action="store_true",
                        help="re-probe and re-hash files already on disk")
    args = parser.parse_args(argv)
    # A relative --media-root is resolved against the repository, not the caller's
    # working directory: the provenance record stores a repo-relative media path,
    # and relative_to() raises rather than guessing when the two disagree.
    args.media_root = (args.media_root if args.media_root.is_absolute()
                       else (ROOT / args.media_root)).resolve()
    args.out = (args.out if args.out.is_absolute() else (ROOT / args.out)).resolve()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    entries = plan["entries"]
    if args.only:
        wanted = set(args.only)
        entries = [e for e in entries if e["clip_id"] in wanted]
        missing = wanted - {e["clip_id"] for e in entries}
        if missing:
            print(f"not in plan: {sorted(missing)}", file=sys.stderr)
    if args.limit:
        entries = entries[:args.limit]

    known = load_records(args.out)
    acquired = skipped = failed = mismatched = 0
    for index, entry in enumerate(entries, start=1):
        clip_id = entry["clip_id"]
        media = args.media_root / f"{clip_id}.mp4"
        previous = known.get(clip_id)
        if media.exists() and previous and not args.refresh:
            digest = sha256_of(media)
            if digest != previous.get("sha256"):
                mismatched += 1
                print(f"[{index}/{len(entries)}] {clip_id}: SHA MISMATCH on disk "
                      f"(recorded {previous.get('sha256', '')[:12]}..., "
                      f"found {digest[:12]}...) -- left alone")
                continue
            skipped += 1
            continue
        if media.exists() and not previous:
            failed += 1
            print(f"[{index}/{len(entries)}] {clip_id}: file exists but is not in the "
                  "provenance record -- refusing to adopt unrecorded bytes")
            continue
        ok, detail = download(entry["download_url"], media)
        if not ok:
            failed += 1
            print(f"[{index}/{len(entries)}] {clip_id}: FAILED ({detail})")
            continue
        probe_result = probe(media)
        record = {
            "clip_id": clip_id,
            "provider": entry["provider"],
            "source_identifier": entry["source_identifier"],
            "source_url": entry["source_url"],
            "download_url": entry["download_url"],
            "source_group": entry["source_group"],
            "title": entry.get("title_from_search") or None,
            "acquisition_date": time.strftime("%Y-%m-%d"),
            "license": plan["license"],
            "license_url": plan["license_url"],
            "license_text_reference": plan.get("license_text_reference"),
            "rights_class": "stock_license_reusable",
            "original_filename": f"{clip_id}.mp4",
            "media_path": str(media.relative_to(ROOT)),
            "bytes": media.stat().st_size,
            "sha256": sha256_of(media),
            "intended_tags": entry.get("intended_tags", []),
            "probe": probe_result,
        }
        append_record(args.out, record)
        known[clip_id] = record
        acquired += 1
        print(f"[{index}/{len(entries)}] {clip_id}: {record['bytes'] / 1e6:.1f} MB "
              f"{probe_result.get('width')}x{probe_result.get('height')} "
              f"@{probe_result.get('fps')} fps "
              f"{probe_result.get('frames')} frames sha={record['sha256'][:12]}")
        time.sleep(args.pause)

    print(f"\nacquired={acquired} already-present={skipped} failed={failed} "
          f"sha-mismatch={mismatched} records={len(known)}")
    print(f"provenance: {args.out}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
