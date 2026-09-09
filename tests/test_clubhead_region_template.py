import unittest

try:
    import cv2  # noqa: F401
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from ghostcaddie.video.clubhead_methods import (
    CandidateState,
    SUPPORTED_METHODS,
    track_candidate,
)


def _background(w=160, h=120, seed=7):
    import numpy as np
    rng = np.random.default_rng(seed)
    return np.repeat(rng.integers(0, 60, size=(h, w), dtype=np.uint8)[:, :, None], 3, axis=2)


def _patch(frame, x, y, size=28):
    # Textured appearance: solid block with a distinct inner pattern, so the
    # template has nonzero variance (required for meaningful NCC).
    half = size // 2
    frame[y - half:y + half, x - half:x + half] = 230
    frame[y - half + 4:y, x - half + 4:x] = 30
    frame[y:y + half - 4, x:x + half - 4] = 120


class TestRegionTemplateRedetection(unittest.TestCase):
    def test_region_template_is_a_registered_distinct_method(self):
        self.assertEqual(SUPPORTED_METHODS, ("lk_point", "region_template", "reacquire_color"))

    @unittest.skipUnless(_HAS_CV2, "requires optional OpenCV research dependency")
    def test_region_template_retracks_moving_patch_from_seed(self):
        import numpy as np
        frames = []
        for i in range(4):
            frame = _background().copy()
            _patch(frame, 40 + 10 * i, 60)
            frames.append(frame)
        result = track_candidate("region_template", frames, 0, (40, 60))
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0].state, CandidateState.OBSERVED)
        for i, cand in enumerate(result[1:], start=1):
            self.assertEqual(cand.state, CandidateState.OBSERVED, i)
            self.assertAlmostEqual(cand.point[0], 40 + 10 * i, delta=3)
            self.assertAlmostEqual(cand.point[1], 60, delta=3)
            self.assertGreater(cand.confidence, 0.0)

    @unittest.skipUnless(_HAS_CV2, "requires optional OpenCV research dependency")
    def test_region_template_returns_unavailable_on_ambiguous_match(self):
        # Two identical patches side by side within the search radius:
        # the best match is ambiguous, so every later frame is unavailable.
        import numpy as np
        frame0 = _background().copy()
        _patch(frame0, 40, 60)
        frame1 = _background(seed=8).copy()
        _patch(frame1, 50, 60)
        _patch(frame1, 90, 60)  # duplicate appearance at similar offset
        result = track_candidate(
            "region_template", [frame0, frame1], 0, (40, 60),
            max_search_radius_px=40, ambiguity_margin=0.05,
        )
        self.assertEqual(result[0].state, CandidateState.OBSERVED)
        self.assertEqual(result[1].state, CandidateState.UNAVAILABLE)
        self.assertEqual(result[1].warning, "appearance_match_ambiguous")

    @unittest.skipUnless(_HAS_CV2, "requires optional OpenCV research dependency")
    def test_region_template_returns_unavailable_when_appearance_disappears(self):
        frame0 = _background().copy()
        _patch(frame0, 40, 60)
        frame1 = _background(seed=9).copy()  # patch gone entirely
        result = track_candidate("region_template", [frame0, frame1], 0, (40, 60))
        self.assertEqual(result[1].state, CandidateState.UNAVAILABLE)
        self.assertEqual(result[1].warning, "low_appearance_match")

    def test_region_template_validates_parameters(self):
        import numpy as np
        frames = [np.zeros((20, 20, 3), dtype=np.uint8)]
        with self.assertRaises(ValueError):
            track_candidate("region_template", frames, 0, (10, 10), match_threshold=0.0)
        with self.assertRaises(ValueError):
            track_candidate("region_template", frames, 0, (10, 10), max_search_radius_px=-1)
        with self.assertRaises(ValueError):
            track_candidate("region_template", frames, 0, (10, 10), template_size_px=0)
        with self.assertRaises(ValueError):
            track_candidate("region_template", frames, 0, (10, 10), ambiguity_margin=float("nan"))
        # aliasing an LK name onto the template method must not be possible:
        with self.assertRaises(ValueError):
            track_candidate("lk_point", frames, 0, (10, 10), match_threshold=0.5)


if __name__ == "__main__":
    unittest.main()