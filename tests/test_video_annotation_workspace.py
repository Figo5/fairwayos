import json
import unittest

from ghostcaddie.video.annotation_workspace import build_annotation_workspace


class TestAnnotationWorkspace(unittest.TestCase):
    def frames(self):
        return [
            {"frame_index": 0, "timestamp_seconds": 0.0, "filename": "frame_000001.jpg"},
            {"frame_index": 1, "timestamp_seconds": 0.5, "filename": "frame_000002.jpg"},
        ]

    def test_generates_deterministic_offline_html_with_frame_timestamps(self):
        first = build_annotation_workspace(
            self.frames(), video={"width": 640, "height": 360, "frame_count": 2, "duration_seconds": 0.5},
            contact_sheet_href="contact_sheet.jpg", title="Shot <Review>", context="fairway & dry"
        )
        second = build_annotation_workspace(
            self.frames(), video={"width": 640, "height": 360, "frame_count": 2, "duration_seconds": 0.5},
            contact_sheet_href="contact_sheet.jpg", title="Shot <Review>", context="fairway & dry"
        )
        self.assertEqual(first, second)
        self.assertIn("frame_000001.jpg", first)
        self.assertIn("0.000 s", first)
        self.assertIn("contact_sheet.jpg", first)
        self.assertIn("Shot &lt;Review&gt;", first)
        self.assertIn("fairway &amp; dry", first)
        self.assertNotIn("<script src=", first.lower())
        self.assertNotRegex(first.lower(), r'(?:src|href)=["\'](?:https?:|//)')

    def test_has_all_annotation_modes_and_explicit_action_boundary(self):
        html = build_annotation_workspace(self.frames(), video={"width": 640, "height": 360, "frame_count": 2, "duration_seconds": 0.5})
        for label in ("Calibration source 1", "Calibration source 2", "Calibration source 3", "Calibration source 4",
                      "Golfer anchor", "Ball", "Clubhead", "Contact", "Intended target / direction", "Landing"):
            self.assertIn(label, html)
        for mode in ("calibration_0", "calibration_1", "calibration_2", "calibration_3", "golfer_anchor", "ball", "clubhead", "contact", "target_intended_direction", "landing"):
            self.assertIn(mode, html)
        self.assertIn("Save Draft", html)
        self.assertIn("Submit Annotations", html)
        self.assertIn("explicit_submit", html)
        self.assertIn("Unavailable", html)
        for control in ("confidence", "provenance", "phase", "warnings", "club selection", "context"):
            self.assertIn(control.lower(), html.lower())

    def test_export_controls_are_explicit_and_deterministic(self):
        html = build_annotation_workspace(self.frames(), video={"width": 640, "height": 360, "frame_count": 2, "duration_seconds": 0.5})
        for marker in (
            "video-human-annotations.v1", "buildExportPayload", "deterministicJson",
            "validateExportPayload", "exportDraft", "submitAnnotations",
            "copyExport", "downloadExport", "explicit_submit", "Blob",
            "submitted_without_explicit_submit", "calibration_points.length", "engine_points.length",
            "Export validation error",
        ):
            self.assertIn(marker, html)
        self.assertIn('id="copy-json"', html)
        self.assertIn('id="download-json"', html)
        self.assertIn('id="export-json"', html)
        self.assertNotRegex(html.lower(), r'''(?:src|href)=["'](?:https?:|//)''')
        self.assertNotIn("fetch(", html)
        self.assertNotIn("XMLHttpRequest", html)

    def test_export_status_is_gated_by_explicit_actions(self):
        html = build_annotation_workspace(self.frames(), video={"width": 640, "height": 360, "frame_count": 2, "duration_seconds": 0.5})
        self.assertIn('addEventListener("click",()=>exportDraft())', html)
        self.assertIn('addEventListener("click",()=>submitAnnotations())', html)
        self.assertIn('state.status="submitted"', html)
        self.assertIn('state.explicit_submit=true', html)
        self.assertIn('state.status="draft"', html)
        self.assertIn('state.explicit_submit=false', html)

    def test_rejects_remote_assets_and_invalid_frames(self):
        with self.assertRaises(ValueError):
            build_annotation_workspace([{"frame_index": 0, "timestamp_seconds": 0, "filename": "https://bad.test/x.jpg"}], video={"width": 1, "height": 1, "frame_count": 1, "duration_seconds": 0})
        with self.assertRaises(ValueError):
            build_annotation_workspace([{ "frame_index": 0, "timestamp_seconds": 0, "filename": "x.jpg"}], video={"width": 1, "height": 1, "frame_count": 2, "duration_seconds": 0})


if __name__ == "__main__":
    unittest.main()
