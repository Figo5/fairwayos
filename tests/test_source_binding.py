import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ghostcaddie.video.source_binding import (
    SourceBindingError,
    bind_source,
    sha256_file,
)


def write_bytes(root: Path, name: str, payload: bytes):
    path = root / name
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def write_seed(root: Path, **payload) -> Path:
    path = root / "seeds.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class SourceBindingTests(unittest.TestCase):
    def test_missing_video_and_expected_hash_fail_before_any_model_import(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video, digest = write_bytes(root, "clip.mp4", b"clip bytes")
            seeds = write_seed(root, source_sha256=digest, seeds=[])
            with self.assertRaises(SourceBindingError):
                bind_source(None, digest, seeds)
            with self.assertRaises(SourceBindingError):
                bind_source(video, None, seeds)
            import ghostcaddie.video.source_binding as source_binding
            self.assertNotIn("torch", source_binding._IMPORTS_AT_MODULE_LEVEL)

    def test_video_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video, _ = write_bytes(root, "lipsky_bytes.mp4", b"lipsky")
            _, expected = write_bytes(root, "morikawa_bytes.mp4", b"morikawa")
            seeds = write_seed(root, source_sha256=expected, clip="morikawa_f350_400", seeds=[])
            with self.assertRaisesRegex(SourceBindingError, "sha256"):
                bind_source(video, expected, seeds)

    def test_seed_without_source_hash_is_rejected_even_with_clip_name(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video, digest = write_bytes(root, "clip.mp4", b"bytes")
            seeds = write_seed(root, clip="gotterup_f340_410", seeds=[])
            with self.assertRaisesRegex(SourceBindingError, "source_sha256"):
                bind_source(video, digest, seeds)

    def test_seed_source_hash_must_match_video_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video, digest = write_bytes(root, "clipA.mp4", b"A")
            _, other_digest = write_bytes(root, "clipB.mp4", b"B")
            seeds = write_seed(root, source_sha256=other_digest, seeds=[])
            with self.assertRaisesRegex(SourceBindingError, "transplant"):
                bind_source(video, digest, seeds)

    def test_fully_bound_source_returns_research_only_record(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video, digest = write_bytes(root, "clip.mp4", b"same bytes")
            seeds = write_seed(root, source_sha256=digest, clip="clip", seeds=[{"frame": 1}])
            bound = bind_source(video, digest, seeds)
            self.assertTrue(bound["hash_bound"])
            self.assertEqual(bound["source_sha256"], digest)
            self.assertEqual(bound["seed_count"], 1)
            self.assertTrue(bound["research_only"])
            self.assertFalse(bound["ground_truth"])
            self.assertFalse(bound["production_eligible"])

    def test_sha256_file_matches_hashlib(self):
        with tempfile.TemporaryDirectory() as td:
            video, digest = write_bytes(Path(td), "clip.mp4", b"payload")
            self.assertEqual(sha256_file(video), digest)


if __name__ == "__main__":
    unittest.main()
