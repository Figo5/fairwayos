import unittest

from ghostcaddie.video.ai_demo import should_infer_pose_for_frame
from ghostcaddie.video.clubhead_methods import CandidateState, track_candidate


class TestResearchSharedPath(unittest.TestCase):
    def test_pose_schedule_fills_missing_native_frames_without_reusing_stale_pose(self):
        cache = {10: ("pose-10", None)}
        self.assertTrue(should_infer_pose_for_frame(11, cache, native_roi=True))
        self.assertFalse(should_infer_pose_for_frame(10, cache, native_roi=True))

    def test_clubhead_track_loss_is_explicit_for_every_remaining_frame(self):
        import numpy as np
        frames = [np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(4)]
        result = track_candidate("flow", frames, 0, (10, 10))
        self.assertEqual(len(result), len(frames))
        self.assertEqual(result[0].state, CandidateState.OBSERVED)
        self.assertTrue(all(item.state is CandidateState.UNAVAILABLE for item in result[1:]))


if __name__ == "__main__":
    unittest.main()
