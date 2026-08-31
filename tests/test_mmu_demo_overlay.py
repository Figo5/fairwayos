from pathlib import Path
import unittest

from ghostcaddie.video.research_overlay import (
    artifact_source_label,
    is_clipped,
    overlay_lines,
)


class MmuDemoOverlayTests(unittest.TestCase):
    def test_artifact_source_label_is_relative_and_stable(self):
        self.assertEqual(
            artifact_source_label(Path("/private/work/out/clip/source.mp4"),
                                  Path("/private/work/out")),
            "clip/source.mp4",
        )

    def test_overlay_includes_one_based_frame_and_timestamp(self):
        lines = overlay_lines(
            frame_index=7,
            fps=25.0,
            state="tracked",
            confidence=0.91,
            uncertainty_px=4.2,
        )

        self.assertIn("FRAME 8", lines[0])
        self.assertIn("TIME 0.280s", lines[0])

    def test_boundary_helper_and_overlay_mark_clipped_target(self):
        self.assertTrue(is_clipped(20, 2, 8, 100, 100))
        self.assertFalse(is_clipped(20, 20, 8, 100, 100))
        lines = overlay_lines(frame_index=0, fps=25.0, state="tracked",
                              confidence=0.8, uncertainty_px=3.0, clipped=True)
        self.assertIn("CLIPPED TARGET", lines[3])


if __name__ == "__main__":
    unittest.main()
