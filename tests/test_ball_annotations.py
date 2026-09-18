import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ghostcaddie.video.ball_annotations import (
    SCHEMA_VERSION,
    annotation_summary,
    bind_annotation,
    drop_frame,
    load_annotation,
    new_annotation,
    save_annotation,
    serialize_annotation,
    set_frame,
    validate_annotation,
)
from ghostcaddie.video.errors import VideoContractError
from ghostcaddie.video.source_binding import SourceBindingError

PAYLOAD = b"not really an mp4, but it has stable bytes"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def base_document(**overrides):
    document = {
        "schema_version": SCHEMA_VERSION,
        "clip_id": "pexels_test",
        "video": "research_stock/pexels_test.mp4",
        "source_sha256": DIGEST,
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "annotator": "tester",
        "notes": "",
        "frames": {},
    }
    document.update(overrides)
    return document


class AnnotationFormatTests(unittest.TestCase):
    def test_a_visible_label_requires_coordinates_inside_the_frame(self):
        with self.assertRaises(VideoContractError):
            validate_annotation(base_document(frames={"10": {"visible": True}}))
        with self.assertRaises(VideoContractError):
            validate_annotation(base_document(frames={"10": {"visible": True, "x": 1920.0, "y": 5.0}}))
        with self.assertRaises(VideoContractError):
            validate_annotation(base_document(frames={"10": {"visible": True, "x": 5.0, "y": -1.0}}))
        document = validate_annotation(base_document(
            frames={"10": {"visible": True, "x": 5.0, "y": 6.0, "radius": 4.0}}))
        self.assertEqual(document["frames"]["10"]["x"], 5.0)

    def test_a_not_visible_label_requires_a_reason_and_forbids_coordinates(self):
        with self.assertRaises(VideoContractError):
            validate_annotation(base_document(frames={"10": {"visible": False}}))
        with self.assertRaises(VideoContractError):
            validate_annotation(base_document(
                frames={"10": {"visible": False, "reason": "gone", "x": 1.0}}))
        document = validate_annotation(base_document(
            frames={"10": {"visible": False, "reason": "left frame"}}))
        self.assertNotIn("x", document["frames"]["10"])

    def test_ambiguity_is_an_orthogonal_flag_on_either_state(self):
        document = validate_annotation(base_document(frames={
            "1": {"visible": True, "x": 1.0, "y": 2.0, "ambiguous": True, "reason": "blurred"},
            "2": {"visible": False, "reason": "occluded", "ambiguous": True},
        }))
        self.assertTrue(document["frames"]["1"]["ambiguous"])
        self.assertTrue(document["frames"]["2"]["ambiguous"])

    def test_unknown_fields_and_bad_frame_keys_are_refused(self):
        with self.assertRaises(VideoContractError):
            validate_annotation(base_document(frames={"10": {"visible": True, "x": 1.0,
                                                             "y": 2.0, "certainty": 0.9}}))
        with self.assertRaises(VideoContractError):
            validate_annotation(base_document(frames={"010": {"visible": False, "reason": "x"}}))
        with self.assertRaises(VideoContractError):
            validate_annotation(base_document(frames={"-1": {"visible": False, "reason": "x"}}))
        with self.assertRaises(VideoContractError):
            validate_annotation(base_document(extra="fabricated"))

    def test_video_must_be_repository_relative_and_hash_must_be_lowercase_hex(self):
        with self.assertRaises(VideoContractError):
            validate_annotation(base_document(video="/absolute/clip.mp4"))
        with self.assertRaises(VideoContractError):
            validate_annotation(base_document(source_sha256=DIGEST.upper()))
        with self.assertRaises(VideoContractError):
            validate_annotation(base_document(source_sha256="abc"))

    def test_serialization_is_deterministic_and_frame_ordered_numerically(self):
        document = base_document(frames={
            "100": {"visible": False, "reason": "gone"},
            "9": {"visible": True, "x": 1.0, "y": 2.0},
        })
        text = serialize_annotation(document)
        self.assertEqual(text, serialize_annotation(json.loads(text)))
        self.assertLess(text.index('"9"'), text.index('"100"'))

    def test_set_and_drop_round_trip_through_validation(self):
        document = new_annotation(clip_id="c", video="a/b.mp4", source_sha256=DIGEST,
                                  width=100, height=100, fps=25.0, annotator="t")
        document = set_frame(document, 4, visible=True, x=10.0, y=20.0, radius=3.0)
        self.assertEqual(document["frames"]["4"]["y"], 20.0)
        document = set_frame(document, 4, visible=False, reason="occluded")
        self.assertFalse(document["frames"]["4"]["visible"])
        self.assertNotIn("x", document["frames"]["4"])
        self.assertEqual(drop_frame(document, 4)["frames"], {})
        with self.assertRaises(VideoContractError):
            set_frame(document, -1, visible=False, reason="x")

    def test_summary_reports_runs_of_consecutive_labels(self):
        document = base_document(frames={
            str(index): {"visible": False, "reason": "gone"} for index in (1, 2, 3, 9, 10)})
        summary = annotation_summary(document)
        self.assertEqual(summary["labeled_frames"], 5)
        self.assertEqual(summary["not_visible_frames"], 5)
        self.assertEqual([run["length"] for run in summary["runs"]], [3, 2])


class AnnotationSourceBindingTests(unittest.TestCase):
    def test_labels_are_refused_when_the_media_bytes_no_longer_match(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "research_stock").mkdir()
            video = root / "research_stock" / "pexels_test.mp4"
            video.write_bytes(PAYLOAD)
            document = base_document()
            self.assertTrue(bind_annotation(document, project_root=root)["hash_bound"])

            video.write_bytes(PAYLOAD + b" re-encoded")
            with self.assertRaises(SourceBindingError):
                bind_annotation(document, project_root=root)
            with self.assertRaises(SourceBindingError):
                save_annotation(root / "labels.json", document, project_root=root)

    def test_load_and_save_round_trip_against_bound_media(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "research_stock").mkdir()
            (root / "research_stock" / "pexels_test.mp4").write_bytes(PAYLOAD)
            document = set_frame(base_document(), 7, visible=True, x=1.0, y=2.0)
            path = root / "data" / "labels.json"
            save_annotation(path, document, project_root=root)
            self.assertEqual(load_annotation(path, project_root=root)["frames"]["7"]["x"], 1.0)

    def test_corrupt_json_is_a_contract_error_not_a_crash(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "labels.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(VideoContractError):
                load_annotation(path)


if __name__ == "__main__":
    unittest.main()
