import unittest

try:
    import cv2  # noqa: F401
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from ghostcaddie.video.ai_demo import should_infer_pose_for_frame
from ghostcaddie.video.clubhead_methods import (
    CandidateState,
    SUPPORTED_METHODS,
    track_candidate,
)


class TestResearchSharedPath(unittest.TestCase):
    def test_pose_schedule_fills_missing_native_frames_without_reusing_stale_pose(self):
        cache = {10: ("pose-10", None)}
        self.assertTrue(should_infer_pose_for_frame(11, cache, native_roi=True))
        self.assertFalse(should_infer_pose_for_frame(10, cache, native_roi=True))

    @unittest.skipUnless(_HAS_CV2, "requires optional OpenCV research dependency")
    def test_clubhead_track_loss_is_explicit_for_every_remaining_frame(self):
        import numpy as np
        frames = [np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(4)]
        result = track_candidate("lk_point", frames, 0, (10, 10))
        self.assertEqual(len(result), len(frames))
        self.assertEqual(result[0].state, CandidateState.OBSERVED)
        self.assertTrue(all(item.state is CandidateState.UNAVAILABLE for item in result[1:]))

    def test_supported_methods_are_distinct_algorithms(self):
        self.assertEqual(SUPPORTED_METHODS, ("lk_point", "region_template"))

    def test_rejects_alias_method_names_that_are_not_distinct_algorithms(self):
        import numpy as np
        frames = [np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(2)]
        for method in ("flow", "lk", "lucas_kanade", "optical_flow", "csrt", ""):
            with self.assertRaises(ValueError, msg=method):
                track_candidate(method, frames, 0, (10, 10))

    @unittest.skipUnless(_HAS_CV2, "requires optional OpenCV research dependency")
    def test_backward_flow_limit_is_explicit(self):
        import numpy as np
        frames = [np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(3)]
        with self.assertRaises(ValueError):
            track_candidate("lk_point", frames, 0, (10, 10), max_backward_error_pixels=0.0)
        with self.assertRaises(ValueError):
            track_candidate("lk_point", frames, 0, (10, 10), max_backward_error_pixels=float("nan"))
        result = track_candidate("lk_point", frames, 0, (10, 10), max_backward_error_pixels=0.9)
        self.assertEqual(result[0].state, CandidateState.OBSERVED)

    @unittest.skipUnless(_HAS_CV2, "requires optional OpenCV research dependency")
    def test_stricter_backward_flow_limit_reports_unavailable_sooner(self):
        import numpy as np
        # Constant frames give degenerate flow; shift them so LK has a gradient.
        frames = []
        for i in range(6):
            frame = np.zeros((40, 40, 3), dtype=np.uint8)
            frame[5 + i:15 + i, 5 + i:15 + i] = 255
            frames.append(frame)
        loose = track_candidate("lk_point", frames, 0, (10, 10))
        strict = track_candidate("lk_point", frames, 0, (10, 10), max_backward_error_pixels=0.999)
        self.assertTrue(
            sum(c.state is CandidateState.UNAVAILABLE for c in strict)
            >= sum(c.state is CandidateState.UNAVAILABLE for c in loose)
        )


if __name__ == "__main__":
    unittest.main()
