"""Synthetic tests for the ClubheadDB assembler.

Everything here is generated in a temp directory: no real video, no real
annotation file, no Tiger labels, no sealed data. The JPEGs are tiny solid
frames encoded at test time. Each ``(video_id, frame index)`` fixture frame is
given a distinct solid colour so that byte-identical content only ever appears
when a test deliberately forces a duplicate.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

from ghostcaddie.demo.clubhead_data import (
    ClubheadDataError,
    assemble,
    golfer_identity,
    public_identity,
)


def jpeg_bytes(width: int, height: int, color=(12, 34, 56)) -> bytes:
    try:
        from PIL import Image  # type: ignore

        buffer = io.BytesIO()
        Image.new("RGB", (width, height), color).save(
            buffer, format="JPEG", quality=90
        )
        return buffer.getvalue()
    except ImportError:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        array = np.zeros((height, width, 3), dtype=np.uint8)
        array[:, :] = tuple(reversed(color))
        ok, encoded = cv2.imencode(".jpg", array)
        if not ok:
            raise RuntimeError("cv2 could not encode the fixture JPEG")
        return encoded.tobytes()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha(obj) -> str:
    """Independent reproduction of the assembler's canonical JSON digest."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(text.encode("utf-8"))


def fixture_color(video_id: str, index: int):
    """A deterministic solid colour distinct per (video, frame index)."""
    digest = hashlib.sha256(f"{video_id}:{index}".encode("utf-8")).digest()
    return (digest[0], digest[1], digest[2])


def make_media(root: Path, video_id: str, count: int, size=(1080, 1920)):
    """Create ``count`` real JPEGs plus a fake source file, and its receipt."""
    images_dir = Path(root) / "source" / video_id / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(1, count + 1):
        data = jpeg_bytes(size[0], size[1], fixture_color(video_id, index))
        path = images_dir / f"frame_{index:04d}.jpg"
        path.write_bytes(data)
        frames.append({
            "index": index,
            "path": str(path),
            "sha256": sha256(data),
            "width": size[0],
            "height": size[1],
        })
    source = Path(root) / "source" / f"{video_id}.mp4"
    source.write_bytes(f"source-bytes-{video_id}".encode())
    return {
        "state": "acquired",
        "images_dir": str(images_dir),
        "frames": frames,
        "source_sha256": sha256(source.read_bytes()),
        "video_path": str(source),
    }


def frames_binding(entry) -> str:
    """Canonical hash of the ordered ``[[index, frame_sha256], ...]`` list."""
    pairs = [[f["index"], f["sha256"]] for f in sorted(entry["frames"], key=lambda f: f["index"])]
    return canonical_sha(pairs)


def rebind_frames(entry) -> None:
    entry["alignment_review"]["frames_sha256"] = frames_binding(entry)


def overwrite_source(entry, data: bytes) -> None:
    """Replace a clip's source bytes and re-bind both source hashes."""
    Path(entry["video_path"]).write_bytes(data)
    entry["source_sha256"] = sha256(data)
    entry["alignment_review"]["source_sha256"] = sha256(data)


def overwrite_frame(entry, index: int, data: bytes) -> None:
    """Replace one frame's bytes and re-bind its hash and the frames binding."""
    frame = next(f for f in entry["frames"] if f["index"] == index)
    Path(frame["path"]).write_bytes(data)
    frame["sha256"] = sha256(data)
    rebind_frames(entry)


def author_rows(video_id: str, count: int, null_indexes=(), bbox=(0.4, 0.5, 0.02, 0.03)):
    rows = []
    for index in range(1, count + 1):
        path = f"{video_id}/images/frame_{index:04d}.jpg"
        if index in null_indexes:
            rows.append({
                "video_id": video_id,
                "image_path": path,
                "class_id": None,
                "x_center": None,
                "y_center": None,
                "width": None,
                "height": None,
            })
        else:
            x_center, y_center, width, height = bbox
            rows.append({
                "video_id": video_id,
                "image_path": path,
                "class_id": 0.0,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
            })
    return rows


class Fixture:
    """Accumulates a plan, its receipt and its author rows.

    Every clip gets a receipt ``alignment_review`` record bound to its *actual*
    source hash, the canonical hash of its own ordered author rows and the
    canonical hash of its ordered ``[[index, frame_sha256], ...]`` receipt list,
    exactly as a real operator-approved receipt would be expected to.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.clips = []
        self.annotations = []
        self.receipt = {"clips": {}}

    def add(self, video_id, split, note, url, count=2, null_indexes=(),
            annotated=True, size=(1080, 1920), review_state: Optional[str] = "accepted",
            source="yt"):
        entry = make_media(self.root, video_id, count, size)
        self.receipt["clips"][video_id] = entry
        self.clips.append({
            "video_id": video_id,
            "split": split,
            "youtube_id": url,
            "notes": note,
            "source": source,
        })
        rows = []
        if annotated:
            rows = author_rows(video_id, count, null_indexes)
            self.annotations.extend(rows)

        if review_state is not None:
            entry["alignment_review"] = {
                "state": review_state,
                "source_sha256": entry["source_sha256"],
                "annotations_sha256": canonical_sha(rows),
                "frames_sha256": frames_binding(entry),
            }
        return entry

    @property
    def plan(self):
        return {
            "clips": self.clips,
            "annotations_json_sha256": canonical_sha(self.annotations),
        }


class TestClubheadDataAssemble(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.dest = self.root / "out"

    def _staging_leftovers(self):
        return sorted(p.name for p in self.root.glob(f".{self.dest.name}.staging-*"))

    # ---- happy paths ---------------------------------------------------- #

    def test_positive_null_vs_missing_rows(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=2, null_indexes={2})
        fx.add("swing_002", "val", "amateur_002", "u2", count=2, annotated=False)

        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)

        positive = self.dest / "labels" / "train" / "swing_001__frame_0001.txt"
        negative = self.dest / "labels" / "train" / "swing_001__frame_0002.txt"
        self.assertEqual(positive.read_text(encoding="utf-8"), "0 0.4 0.5 0.02 0.03")
        self.assertEqual(negative.read_text(encoding="utf-8"), "")

        # unannotated clip: no output at all, never an empty negative
        self.assertFalse((self.dest / "labels" / "val").exists())
        self.assertEqual([r["video_id"] for r in manifest["rejected"]], ["swing_002"])
        self.assertEqual(manifest["rejected"][0]["reason"], "unannotated")
        self.assertEqual(manifest["counts"]["accepted"], 1)
        self.assertEqual(manifest["counts"]["unannotated"], 1)

    def test_images_copied_byte_identical_with_hashes(self):
        fx = Fixture(self.root)
        entry = fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        source_bytes = Path(entry["frames"][0]["path"]).read_bytes()

        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)

        copied = self.dest / "images" / "train" / "swing_001__frame_0001.jpg"
        self.assertEqual(copied.read_bytes(), source_bytes)
        accepted = manifest["accepted"][0]
        self.assertEqual(accepted["frames"][0]["image_sha256"], sha256(source_bytes))
        self.assertEqual(accepted["source_sha256"], entry["source_sha256"])
        self.assertEqual(accepted["receipt_source_sha256"], entry["source_sha256"])
        self.assertEqual(
            manifest["plan_annotations_json_sha256"],
            fx.plan["annotations_json_sha256"],
        )
        self.assertEqual(
            manifest["annotations_rows_sha256"], canonical_sha(fx.annotations)
        )

    def test_dataset_yaml_absolute_root_and_names(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        assemble(fx.plan, fx.annotations, fx.receipt, self.dest)

        text = (self.dest / "dataset.yaml").read_text(encoding="utf-8")
        self.assertIn(f"path: {self.dest.resolve().as_posix()}", text)
        self.assertIn("train: images/train", text)
        self.assertIn("val: images/val", text)
        self.assertIn("test: images/test", text)
        self.assertIn("names:\n  0: clubhead", text)
        self.assertNotIn("download", text.lower())

    def test_manifest_disclaims_heldout_and_production(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertFalse(manifest["claims"]["statistical_heldout"])
        self.assertFalse(manifest["claims"]["production"])
        self.assertFalse(manifest["ground_truth"]["independently_confirmed"])
        # manifest.json mirrors the returned mapping
        on_disk = json.loads((self.dest / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk, manifest)

    # ---- fatal validation, before any output ---------------------------- #

    def test_duplicate_author_row_is_fatal_with_no_output(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        fx.annotations.append(dict(fx.annotations[0]))

        with self.assertRaises(ClubheadDataError):
            assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertFalse(self.dest.exists())

    def test_path_traversal_and_absolute_refused(self):
        for bad in ("../evil.jpg", "/abs/evil.jpg",
                    "swing_001/../../evil.jpg", "swing_001\\images\\frame_0001.jpg"):
            with self.subTest(path=bad):
                fx = Fixture(self.root)
                fx.add("swing_001", "train", "amateur_001", "u1", count=1)
                fx.annotations[0]["image_path"] = bad
                with self.assertRaises(ClubheadDataError):
                    assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
                self.assertFalse(self.dest.exists())

    def test_malformed_bbox_refused(self):
        cases = {
            "center_oob": {"x_center": 1.2},
            "zero_width": {"width": 0.0},
            "nan": {"y_center": float("nan")},
            "edge_overshoot": {"x_center": 0.995, "width": 0.02},
            "negative": {"height": -0.03},
        }
        for name, patch in cases.items():
            with self.subTest(case=name):
                fx = Fixture(self.root)
                fx.add("swing_001", "train", "amateur_001", "u1", count=1)
                fx.annotations[0].update(patch)
                with self.assertRaises(ClubheadDataError):
                    assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
                self.assertFalse(self.dest.exists())

    def test_non_zero_class_refused(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        fx.annotations[0]["class_id"] = 1.0
        with self.assertRaises(ClubheadDataError):
            assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertFalse(self.dest.exists())

    def test_partial_null_row_refused(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        fx.annotations[0]["class_id"] = None  # but box still present
        with self.assertRaises(ClubheadDataError):
            assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertFalse(self.dest.exists())

    def test_unknown_video_row_refused(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        stray = author_rows("swing_009", 1)[0]
        fx.annotations.append(stray)
        with self.assertRaises(ClubheadDataError):
            assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertFalse(self.dest.exists())

    # ---- cross-split leakage ------------------------------------------- #

    def test_url_leakage_across_splits_refused(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "same-url", count=1)
        fx.add("swing_002", "val", "amateur_002", "same-url", count=1)
        with self.assertRaises(ClubheadDataError):
            assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertFalse(self.dest.exists())

    def test_pro_numeric_suffixes_group_and_refuse_leakage(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "pro_001", "u1", count=1)
        fx.add("swing_002", "val", "pro_002", "u2", count=1)
        with self.assertRaises(ClubheadDataError):
            assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertFalse(self.dest.exists())

    def test_amateur_numeric_suffixes_are_preserved(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        fx.add("swing_002", "val", "amateur_002", "u2", count=1)
        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertEqual(manifest["counts"]["accepted"], 2)
        self.assertEqual(manifest["splits"]["train"], ["swing_001"])
        self.assertEqual(manifest["splits"]["val"], ["swing_002"])

    def test_golfer_identity_normalizes_case_and_separators(self):
        self.assertEqual(golfer_identity("  Pro_001 "), golfer_identity("pro-001"))
        self.assertEqual(golfer_identity("PRO   001"), golfer_identity("pro__1"))
        self.assertEqual(golfer_identity("Pro_001"), "pro")

    def test_golfer_identity_keeps_distinct_amateur_roster_ids(self):
        self.assertNotEqual(
            golfer_identity("amateur_001"), golfer_identity("amateur_002")
        )
        self.assertEqual(golfer_identity("Amateur-007"), "amateur 007")

    def test_public_identity_collapses_youtube_forms(self):
        target = public_identity("dQw4w9WgXcQ", "yt")
        for url in (
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=5s",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ?si=abc",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        ):
            with self.subTest(url=url):
                self.assertEqual(public_identity(url, "yt_shorts"), target)

    def test_public_identity_reddit_ignores_host_query_fragment(self):
        a = public_identity(
            "https://www.reddit.com/r/golf/comments/abc123/title/?utm=1#frag", "reddit"
        )
        b = public_identity(
            "https://old.reddit.com/r/golf/comments/abc123/title", "reddit"
        )
        self.assertEqual(a, b)
        self.assertNotEqual(
            a, public_identity("https://reddit.com/r/golf/comments/zzz", "reddit")
        )

    def test_youtube_url_forms_leak_across_splits(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001",
               "https://www.youtube.com/watch?v=AbC123xyz", count=1)
        fx.add("swing_002", "val", "amateur_002", "AbC123xyz", count=1)
        with self.assertRaises(ClubheadDataError):
            assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertFalse(self.dest.exists())

    def test_reddit_host_variants_leak_across_splits(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001",
               "https://www.reddit.com/r/golf/comments/abc123/x?u=1", count=1,
               source="reddit")
        fx.add("swing_002", "val", "amateur_002",
               "https://old.reddit.com/r/golf/comments/abc123/x#f", count=1,
               source="reddit")
        with self.assertRaises(ClubheadDataError):
            assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertFalse(self.dest.exists())

    def test_duplicate_plan_video_id_refused(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        fx.clips.append(dict(fx.clips[0], youtube_id="u2"))
        with self.assertRaises(ClubheadDataError):
            assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertFalse(self.dest.exists())

    # ---- plan annotation hash binding ----------------------------------- #

    def test_plan_annotations_hash_mismatch_is_fatal(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        bad = fx.plan
        bad["annotations_json_sha256"] = "a" * 64
        with self.assertRaises(ClubheadDataError):
            assemble(bad, fx.annotations, fx.receipt, self.dest)
        self.assertFalse(self.dest.exists())

    def test_plan_annotations_hash_binds_supplied_rows(self):
        # rows mutated after the plan was bound must not be silently accepted
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        plan = fx.plan
        fx.annotations[0]["x_center"] = 0.6
        with self.assertRaises(ClubheadDataError):
            assemble(plan, fx.annotations, fx.receipt, self.dest)
        self.assertFalse(self.dest.exists())

    def test_plan_annotations_hash_uppercase_is_fatal(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        bad = fx.plan
        bad["annotations_json_sha256"] = canonical_sha(fx.annotations).upper()
        with self.assertRaises(ClubheadDataError):
            assemble(bad, fx.annotations, fx.receipt, self.dest)
        self.assertFalse(self.dest.exists())

    # ---- required source/frame hashes ----------------------------------- #

    def test_missing_source_hash_is_incomplete_receipt(self):
        fx = Fixture(self.root)
        entry = fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        del entry["source_sha256"]
        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertEqual(manifest["rejection_counts"]["incomplete_receipt"], 1)
        self.assertEqual(manifest["splits"]["train"], [])

    def test_malformed_source_hash_is_incomplete_receipt(self):
        for i, bad in enumerate(("", "abc", "g" * 64, "a" * 63, "A" * 64)):
            with self.subTest(bad=bad):
                fx = Fixture(self.root)
                entry = fx.add("swing_001", "train", "amateur_001", "u1", count=1)
                entry["source_sha256"] = bad
                dest = self.root / f"out_{i}"
                manifest = assemble(fx.plan, fx.annotations, fx.receipt, dest)
                self.assertEqual(
                    manifest["rejection_counts"]["incomplete_receipt"], 1
                )
                self.assertEqual(manifest["splits"]["train"], [])

    def test_missing_frame_hash_is_incomplete_receipt(self):
        fx = Fixture(self.root)
        entry = fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        del entry["frames"][0]["sha256"]
        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertEqual(manifest["rejection_counts"]["incomplete_receipt"], 1)

    def test_malformed_frame_hash_is_incomplete_receipt(self):
        for i, bad in enumerate(("", "Z" * 64, "z" * 64)):
            with self.subTest(bad=bad):
                fx = Fixture(self.root)
                entry = fx.add("swing_001", "train", "amateur_001", "u1", count=1)
                entry["frames"][0]["sha256"] = bad
                dest = self.root / f"out_{i}"
                manifest = assemble(fx.plan, fx.annotations, fx.receipt, dest)
                self.assertEqual(
                    manifest["rejection_counts"]["incomplete_receipt"], 1
                )
                self.assertEqual(manifest["splits"]["train"], [])

    def test_wellformed_source_hash_mismatch_still_excludes(self):
        fx = Fixture(self.root)
        entry = fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        entry["source_sha256"] = "0" * 64
        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertEqual(manifest["rejection_counts"]["source_sha_mismatch"], 1)

    # ---- alignment review (mandatory operator assertion) ---------------- #

    def test_alignment_review_preserved_in_manifest(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)

        accepted = manifest["accepted"][0]
        self.assertEqual(accepted["alignment_review"]["state"], "accepted")
        self.assertEqual(
            accepted["alignment_review"]["source_sha256"],
            accepted["source_sha256"],
        )
        self.assertEqual(
            accepted["alignment_review"]["annotations_sha256"],
            accepted["annotations_sha256"],
        )
        self.assertEqual(
            accepted["annotations_sha256"], canonical_sha(author_rows("swing_001", 1))
        )
        self.assertFalse(manifest["alignment_review"]["automatic_proof"])
        self.assertFalse(manifest["alignment_review"]["independently_confirmed"])

    def test_alignment_frames_binding_present_and_exact(self):
        fx = Fixture(self.root)
        entry = fx.add("swing_001", "train", "amateur_001", "u1", count=2)
        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        accepted = manifest["accepted"][0]
        self.assertEqual(
            accepted["alignment_review"]["frames_sha256"],
            canonical_sha([[f["index"], f["sha256"]] for f in entry["frames"]]),
        )

    def test_missing_alignment_frames_binding_is_binding_mismatch(self):
        fx = Fixture(self.root)
        entry = fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        del entry["alignment_review"]["frames_sha256"]
        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertEqual(
            manifest["rejection_counts"]["alignment_binding_mismatch"], 1
        )
        self.assertEqual(manifest["splits"]["train"], [])

    def test_tampered_alignment_frames_binding_excluded(self):
        fx = Fixture(self.root)
        entry = fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        entry["alignment_review"]["frames_sha256"] = "e" * 64
        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertEqual(
            manifest["rejection_counts"]["alignment_binding_mismatch"], 1
        )

    def test_alignment_frames_binding_rejects_reversed_order(self):
        fx = Fixture(self.root)
        entry = fx.add("swing_001", "train", "amateur_001", "u1", count=2)
        pairs = [[f["index"], f["sha256"]] for f in entry["frames"]]
        entry["alignment_review"]["frames_sha256"] = canonical_sha(list(reversed(pairs)))
        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertEqual(
            manifest["rejection_counts"]["alignment_binding_mismatch"], 1
        )

    def test_missing_alignment_review_is_unreviewed(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1,
               review_state=None)
        fx.add("swing_002", "val", "amateur_002", "u2", count=1)
        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)

        self.assertEqual(manifest["splits"]["train"], [])
        self.assertEqual(manifest["splits"]["val"], ["swing_002"])
        self.assertEqual(manifest["rejection_counts"]["alignment_unreviewed"], 1)

    def test_pending_alignment_review_is_unreviewed(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1,
               review_state="pending")
        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertEqual(manifest["rejection_counts"]["alignment_unreviewed"], 1)
        self.assertEqual(manifest["splits"]["train"], [])

    def test_unknown_alignment_state_is_unreviewed(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1,
               review_state="maybe")
        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertEqual(manifest["rejection_counts"]["alignment_unreviewed"], 1)

    def test_rejected_alignment_review_is_rejected(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1,
               review_state="rejected")
        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertEqual(manifest["rejection_counts"]["alignment_rejected"], 1)
        self.assertEqual(manifest["splits"]["train"], [])

    def test_tampered_alignment_source_binding_excluded(self):
        fx = Fixture(self.root)
        entry = fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        entry["alignment_review"]["source_sha256"] = "c" * 64
        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertEqual(
            manifest["rejection_counts"]["alignment_binding_mismatch"], 1
        )

    def test_tampered_alignment_annotations_binding_excluded(self):
        fx = Fixture(self.root)
        entry = fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        entry["alignment_review"]["annotations_sha256"] = "d" * 64
        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertEqual(
            manifest["rejection_counts"]["alignment_binding_mismatch"], 1
        )

    def test_alignment_annotations_binding_rejects_tuple_form(self):
        # binding must be over the row *dicts*, not (index, row) pairs
        fx = Fixture(self.root)
        entry = fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        rows = author_rows("swing_001", 1)
        entry["alignment_review"]["annotations_sha256"] = canonical_sha(
            list(enumerate(rows, start=1))
        )
        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertEqual(
            manifest["rejection_counts"]["alignment_binding_mismatch"], 1
        )

    # ---- per-clip media exclusions (non-fatal) -------------------------- #

    def test_media_tamper_excludes_whole_video(self):
        fx = Fixture(self.root)
        tampered = fx.add("swing_001", "train", "amateur_001", "u1", count=2)
        fx.add("swing_002", "val", "amateur_002", "u2", count=1)
        # rewrite one frame's bytes after the receipt hash was taken
        Path(tampered["frames"][1]["path"]).write_bytes(
            jpeg_bytes(1080, 1920, (1, 2, 3))
        )

        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)

        self.assertEqual(manifest["splits"]["train"], [])
        self.assertEqual(manifest["splits"]["val"], ["swing_002"])
        self.assertEqual(manifest["rejection_counts"]["image_sha_mismatch"], 1)
        self.assertFalse(
            (self.dest / "images" / "train" / "swing_001__frame_0001.jpg").exists()
        )
        self.assertTrue(
            (self.dest / "images" / "val" / "swing_002__frame_0001.jpg").exists()
        )

    def test_count_mismatch_excludes_whole_video(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=2, annotated=False)
        # only one author row for a two-frame clip
        fx.annotations.extend(author_rows("swing_001", 1))
        fx.add("swing_002", "val", "amateur_002", "u2", count=1)

        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)

        self.assertEqual(manifest["rejection_counts"]["annotation_frame_mismatch"], 1)
        self.assertEqual(manifest["splits"]["val"], ["swing_002"])
        self.assertFalse((self.dest / "images" / "train").exists())

    # ---- decoder validation --------------------------------------------- #

    def test_truncated_jpeg_is_image_unreadable(self):
        fx = Fixture(self.root)
        entry = fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        full = Path(entry["frames"][0]["path"]).read_bytes()
        truncated = full[: max(2, len(full) // 2)]
        overwrite_frame(entry, 1, truncated)

        try:
            from PIL import Image  # type: ignore
        except ImportError:
            Image = None  # type: ignore
        if Image is not None:
            # the header still parses (size is known) but the body will not
            # fully decode - proving the check goes past the header.
            with Image.open(io.BytesIO(truncated)) as image:
                self.assertEqual(image.size, (1080, 1920))
                with self.assertRaises(Exception):
                    image.load()

        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertEqual(manifest["rejection_counts"]["image_unreadable"], 1)
        self.assertEqual(manifest["splits"]["train"], [])

    def test_missing_decoder_is_fatal_before_writes(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        with mock.patch.dict(sys.modules, {"PIL": None, "cv2": None}):
            with self.assertRaises(ClubheadDataError):
                assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertFalse(self.dest.exists())

    # ---- duplicate-content detection (exact bytes only) ----------------- #

    def test_cross_split_source_duplicate_is_fatal(self):
        fx = Fixture(self.root)
        a = fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        b = fx.add("swing_002", "val", "amateur_002", "u2", count=1)
        overwrite_source(b, Path(a["video_path"]).read_bytes())
        with self.assertRaises(ClubheadDataError):
            assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertFalse(self.dest.exists())

    def test_cross_split_frame_duplicate_is_fatal(self):
        fx = Fixture(self.root)
        a = fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        b = fx.add("swing_002", "val", "amateur_002", "u2", count=1)
        overwrite_frame(b, 1, Path(a["frames"][0]["path"]).read_bytes())
        with self.assertRaises(ClubheadDataError):
            assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertFalse(self.dest.exists())

    def test_same_split_repeated_frames_are_allowed(self):
        fx = Fixture(self.root)
        entry = fx.add("swing_001", "train", "amateur_001", "u1", count=2)
        duplicate = Path(entry["frames"][0]["path"]).read_bytes()
        overwrite_frame(entry, 2, duplicate)
        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertEqual(manifest["counts"]["accepted"], 1)
        frames = manifest["accepted"][0]["frames"]
        self.assertEqual(frames[0]["image_sha256"], frames[1]["image_sha256"])

    def test_manifest_duplicate_and_alignment_nonclaims(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertFalse(manifest["duplicate_content"]["near_duplicate_identification"])
        self.assertFalse(manifest["duplicate_content"]["repost_identification"])
        self.assertFalse(manifest["claims"]["near_duplicate_identification"])
        self.assertFalse(manifest["claims"]["independent_human_alignment_verification"])
        self.assertFalse(manifest["alignment_review"]["automatic_proof"])
        self.assertFalse(manifest["alignment_review"]["human_proof_claimed"])
        self.assertEqual(
            manifest["alignment_review"]["bindings"],
            ["source_sha256", "annotations_sha256", "frames_sha256"],
        )

    # ---- split isolation ------------------------------------------------ #

    def test_split_directories_are_isolated(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        fx.add("swing_002", "val", "amateur_002", "u2", count=1)
        fx.add("swing_003", "test", "amateur_003", "u3", count=1)
        assemble(fx.plan, fx.annotations, fx.receipt, self.dest)

        for split, stem in (("train", "swing_001"), ("val", "swing_002"),
                            ("test", "swing_003")):
            names = sorted(p.name for p in (self.dest / "images" / split).iterdir())
            self.assertEqual(names, [f"{stem}__frame_0001.jpg"])
            labels = sorted(p.name for p in (self.dest / "labels" / split).iterdir())
            self.assertEqual(labels, [f"{stem}__frame_0001.txt"])

    # ---- destination handling ------------------------------------------ #

    def test_destination_must_be_fresh(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        self.dest.mkdir()
        (self.dest / "leftover.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(ClubheadDataError):
            assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertEqual((self.dest / "leftover.txt").read_text(encoding="utf-8"), "x")

    def test_destination_that_is_a_file_refused(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        self.dest.write_bytes(b"not a directory")
        with self.assertRaises(ClubheadDataError):
            assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertEqual(self.dest.read_bytes(), b"not a directory")

    # ---- staged atomic publish ----------------------------------------- #

    def test_success_leaves_no_staging_directory(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertEqual(self._staging_leftovers(), [])

    def test_yaml_and_manifest_use_final_destination_not_staging(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        final = self.dest.resolve().as_posix()
        self.assertEqual(manifest["destination"], final)
        self.assertNotIn("staging", manifest["destination"])
        text = (self.dest / "dataset.yaml").read_text(encoding="utf-8")
        self.assertIn(f"path: {final}", text)
        self.assertNotIn("staging", text)

    def test_failure_during_write_preserves_inputs_and_destination(self):
        fx = Fixture(self.root)
        entry = fx.add("swing_001", "train", "amateur_001", "u1", count=2)
        frame_path = Path(entry["frames"][0]["path"])
        source_path = Path(entry["video_path"])
        frame_before = frame_path.read_bytes()
        source_before = source_path.read_bytes()
        # an existing *empty* destination is acceptable, and must stay untouched
        self.dest.mkdir()

        with mock.patch.object(Path, "write_bytes", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                assemble(fx.plan, fx.annotations, fx.receipt, self.dest)

        self.assertEqual(frame_path.read_bytes(), frame_before)
        self.assertEqual(source_path.read_bytes(), source_before)
        self.assertEqual(list(self.dest.iterdir()), [])
        self.assertEqual(self._staging_leftovers(), [])

    def test_failure_during_publish_leaves_destination_absent(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        with mock.patch("ghostcaddie.demo.clubhead_data._publish",
                        side_effect=OSError("rename failed")):
            with self.assertRaises(OSError):
                assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertFalse(self.dest.exists())
        self.assertEqual(self._staging_leftovers(), [])

    def test_existing_empty_destination_is_replaced(self):
        fx = Fixture(self.root)
        fx.add("swing_001", "train", "amateur_001", "u1", count=1)
        self.dest.mkdir()
        manifest = assemble(fx.plan, fx.annotations, fx.receipt, self.dest)
        self.assertTrue((self.dest / "dataset.yaml").is_file())
        self.assertEqual(manifest["counts"]["accepted"], 1)
        self.assertEqual(self._staging_leftovers(), [])


if __name__ == "__main__":
    unittest.main()

class YouTubeIdentityCaseTests(unittest.TestCase):
    def test_video_ids_are_case_sensitive_while_hosts_are_not(self):
        from ghostcaddie.demo.clubhead_data import public_identity
        self.assertEqual(public_identity('AbC','yt'),public_identity('https://WWW.YouTube.com/watch?v=AbC&t=8','yt'))
        self.assertNotEqual(public_identity('AbC','yt'),public_identity('abc','yt'))
