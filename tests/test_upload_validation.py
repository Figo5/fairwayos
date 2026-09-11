"""TDD for upload validation and the three-target job workflow.

All fixtures here are EXPLICITLY SYNTHETIC. Synthetic media is allowed in tests
and is never permitted to appear in a user-facing result.
"""
import os, tempfile, unittest
from ghostcaddie.upload.validation import (
    UploadRejected, VideoLimits, validate_upload, safe_join,
)


def make_video(path, seconds=1.0, w=320, h=240, fps=30):
    """Synthetic test video. Marked synthetic; never a user result."""
    import cv2, numpy as np
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(int(seconds * fps)):
        vw.write(np.full((h, w, 3), i % 255, np.uint8))
    vw.release()
    return path


class PathSafetyTests(unittest.TestCase):
    def test_safe_join_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            for bad in ("../x", "a/../../x", "/etc/passwd"):
                with self.assertRaises(UploadRejected):
                    safe_join(d, bad)

    def test_safe_join_allows_plain_name(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(safe_join(d, "clip.mp4").startswith(os.path.realpath(d)))

    def test_remote_urls_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            for u in ("http://x/a.mp4", "https://x/a.mp4", "file:///etc/passwd",
                      "ftp://x/a.mp4"):
                with self.assertRaises(UploadRejected):
                    validate_upload(u, VideoLimits(), workdir=d)


class LimitTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_missing_file_is_rejected(self):
        with self.assertRaises(UploadRejected):
            validate_upload(os.path.join(self.d, "nope.mp4"), VideoLimits(), workdir=self.d)

    def test_oversize_file_is_rejected_before_decode(self):
        p = os.path.join(self.d, "big.mp4")
        with open(p, "wb") as f:
            f.write(b"\0" * 2048)
        with self.assertRaises(UploadRejected) as cm:
            validate_upload(p, VideoLimits(max_bytes=1024), workdir=self.d)
        self.assertIn("size", str(cm.exception).lower())

    def test_non_video_bytes_are_rejected(self):
        p = os.path.join(self.d, "notvideo.mp4")
        with open(p, "wb") as f:
            f.write(b"this is not a video")
        with self.assertRaises(UploadRejected):
            validate_upload(p, VideoLimits(), workdir=self.d)

    def test_too_long_duration_is_rejected(self):
        p = make_video(os.path.join(self.d, "long.mp4"), seconds=2.0)
        with self.assertRaises(UploadRejected) as cm:
            validate_upload(p, VideoLimits(max_seconds=1.0), workdir=self.d)
        self.assertIn("duration", str(cm.exception).lower())

    def test_too_small_dimensions_are_rejected(self):
        p = make_video(os.path.join(self.d, "tiny.mp4"), w=64, h=48)
        with self.assertRaises(UploadRejected) as cm:
            validate_upload(p, VideoLimits(min_width=320, min_height=240), workdir=self.d)
        self.assertIn("dimension", str(cm.exception).lower())

    def test_valid_video_reports_decoded_facts_and_hash(self):
        p = make_video(os.path.join(self.d, "ok.mp4"), seconds=1.0, w=320, h=240)
        r = validate_upload(p, VideoLimits(), workdir=self.d)
        self.assertEqual((r.width, r.height), (320, 240))
        self.assertGreater(r.frames, 0)
        self.assertEqual(len(r.sha256), 64)
        # decoded facts come from the file, never from the caller
        self.assertGreater(r.duration_seconds, 0)


if __name__ == "__main__":
    unittest.main()
