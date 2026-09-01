from pathlib import Path
import unittest

from ghostcaddie.video.research_overlay import (
    artifact_source_label,
    build_research_ffmpeg_filter,
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

    def test_overlay_explicitly_disclaims_validated_ball_identity(self):
        lines = overlay_lines(frame_index=0, fps=25.0, state="tracked",
                              confidence=0.8, uncertainty_px=3.0)

        self.assertIn("VALIDATED BALL IDENTITY: UNAVAILABLE", lines)

    def test_boundary_helper_and_overlay_mark_clipped_target(self):
        self.assertTrue(is_clipped(20, 2, 8, 100, 100))
        self.assertFalse(is_clipped(20, 20, 8, 100, 100))
        lines = overlay_lines(frame_index=0, fps=25.0, state="tracked",
                              confidence=0.8, uncertainty_px=3.0, clipped=True)
        self.assertIn("CLIPPED TARGET", lines[3])

    def test_research_ffmpeg_filter_draws_candidate_uncertainty_and_trail(self):
        graph = build_research_ffmpeg_filter(
            [{"frame": 0, "x": 100, "y": 120, "radius": 8, "uncertainty_px": 4.0},
             {"frame": 1, "x": 110, "y": 130, "radius": 9, "uncertainty_px": 5.0}],
            fps=25.0,
            width=600,
            height=480,
        )
        self.assertGreaterEqual(graph.count("drawbox"), 5)
        self.assertIn("color=yellow", graph)
        self.assertIn("color=orange", graph)
        self.assertIn("enable='eq(n\\,1)'", graph)
        self.assertIn("enable='gte(n\\,1)'", graph)

    def test_research_ffmpeg_filter_suppresses_rejected_alignment(self):
        graph = build_research_ffmpeg_filter(
            [{"frame": 0, "x": 100, "y": 120, "radius": 8, "uncertainty_px": 4.0}],
            fps=25.0,
            width=600,
            height=480,
            visually_aligned=False,
        )
        self.assertNotIn("enable='eq(n\\,0)'", graph)
        self.assertNotIn("color=yellow:t=2", graph)
        self.assertIn("color=red", graph)


if __name__ == "__main__":
    unittest.main()
