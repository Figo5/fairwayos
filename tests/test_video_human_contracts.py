import json
import math
import unittest

from ghostcaddie.video.human_contracts import (
    HumanAnnotationDocument,
    SCHEMA_VERSION,
    deserialize_human_annotations,
    serialize_human_annotations,
    validate_human_annotations,
)
from ghostcaddie.video.errors import VideoContractError


class TestHumanAnnotationContract(unittest.TestCase):
    def valid_payload(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "draft",
            "explicit_submit": False,
            "video": {"width": 1920, "height": 1080, "frame_count": 120, "duration_seconds": 4.0},
            "calibration_points": [
                {"x": 100.0, "y": 200.0, "frame_index": 0, "timestamp_seconds": 0.0, "confidence": 1.0, "source": "user_supplied"},
                {"x": 1700.0, "y": 200.0, "frame_index": 0, "timestamp_seconds": 0.0, "confidence": 1.0, "source": "user_supplied"},
                {"x": 1700.0, "y": 900.0, "frame_index": 0, "timestamp_seconds": 0.0, "confidence": 1.0, "source": "user_supplied"},
                {"x": 100.0, "y": 900.0, "frame_index": 0, "timestamp_seconds": 0.0, "confidence": 1.0, "source": "user_supplied"},
            ],
            "engine_points": [{"x": 0.0, "y": 0.0}, {"x": 100.0, "y": 0.0}, {"x": 100.0, "y": 100.0}, {"x": 0.0, "y": 100.0}],
            "golfer_anchor": {"value": {"x": 500.0, "y": 800.0, "frame_index": 10, "timestamp_seconds": 0.333, "confidence": 0.9}, "source": "user_confirmed"},
            "ball": {"value": None, "source": "unavailable"},
            "clubhead": {"value": {"x": 550.0, "y": 700.0, "frame_index": 11, "timestamp_seconds": 0.367, "confidence": 0.8}, "source": "observed"},
            "contact": {"value": {"frame_index": 20, "timestamp_seconds": 0.667, "confidence": 0.7, "phase": "contact"}, "source": "inferred"},
            "target_intended_direction": {"value": {"x": 1.0, "y": 0.0, "confidence": 0.6}, "source": "user_supplied"},
            "landing": {"value": None, "source": "unavailable"},
            "club_selection": {"value": "7-iron", "source": "user_confirmed"},
            "context": {"value": {"lie": "fairway"}, "source": "user_supplied"},
            "warnings": [],
        }

    def test_valid_draft_and_submitted_requires_explicit_submit(self):
        draft = HumanAnnotationDocument(self.valid_payload())
        self.assertEqual(draft.status, "draft")
        submitted = dict(self.valid_payload(), status="submitted", explicit_submit=True)
        self.assertEqual(HumanAnnotationDocument(submitted).status, "submitted")
        with self.assertRaisesRegex(VideoContractError, "submitted_without_explicit_submit"):
            HumanAnnotationDocument(dict(submitted, explicit_submit=False))
        self.assertEqual(draft.submit(explicit_submit=True).status, "submitted")
        with self.assertRaisesRegex(VideoContractError, "submitted_without_explicit_submit"):
            draft.submit()

    def test_deterministic_round_trip(self):
        document = HumanAnnotationDocument(self.valid_payload())
        encoded = serialize_human_annotations(document)
        self.assertEqual(encoded, serialize_human_annotations(deserialize_human_annotations(encoded)))
        self.assertEqual(encoded, json.dumps(document.to_dict(), allow_nan=False, sort_keys=True, separators=(",", ":")))

    def test_each_source_and_unavailable_are_supported(self):
        for source in ("user_supplied", "user_confirmed", "observed", "inferred"):
            payload = self.valid_payload()
            payload["ball"] = {"value": {"x": 1, "y": 2, "frame_index": 1, "timestamp_seconds": 0.1, "confidence": .5}, "source": source}
            validate_human_annotations(payload)
        validate_human_annotations(self.valid_payload())

    def test_rejects_malformed_strict_payloads(self):
        cases = []
        cases.append(("missing required", lambda p: p.pop("ball")))
        cases.append(("source", lambda p: p["ball"].update(source="guess")))
        cases.append(("fabricated field", lambda p: p.update(extra="nope")))
        cases.append(("absolute path", lambda p: p.update(note_path="/tmp/video.mp4")))
        cases.append(("nonfinite", lambda p: p["clubhead"]["value"].update(x=math.nan)))
        cases.append(("out of bounds", lambda p: p["clubhead"]["value"].update(x=2000)))
        cases.append(("bad frame", lambda p: p["clubhead"]["value"].update(frame_index=120)))
        cases.append(("bad timestamp", lambda p: p["clubhead"]["value"].update(timestamp_seconds=5)))
        cases.append(("bad confidence", lambda p: p["clubhead"]["value"].update(confidence=1.1)))
        cases.append(("ambiguous phase", lambda p: p["contact"]["value"].update(phase="maybe_contact")))
        cases.append(("ambiguous warning", lambda p: p.update(warnings=["unclear ball position"])))
        for name, mutate in cases:
            with self.subTest(name=name):
                payload = self.valid_payload()
                mutate(payload)
                with self.assertRaises(VideoContractError):
                    validate_human_annotations(payload)

    def test_calibration_requires_exactly_four_points(self):
        payload = self.valid_payload()
        point = payload["calibration_points"][0]
        payload["calibration_points"] = [dict(point, x=100.0 + i * 100.0, y=200.0 + i * 50.0) for i in range(4)]
        for count in (0, 1, 3, 5):
            candidate = json.loads(json.dumps(payload))
            candidate["calibration_points"] = candidate["calibration_points"][:count]
            if count == 5:
                candidate["calibration_points"].append(dict(point, x=600.0, y=450.0))
            with self.subTest(count=count), self.assertRaises(VideoContractError):
                validate_human_annotations(candidate)

        with self.assertRaises(VideoContractError):
            deserialize_human_annotations("not-json")
        payload = self.valid_payload()
        payload["schema_version"] = "video-human-annotations.v0"
        with self.assertRaises(VideoContractError):
            validate_human_annotations(payload)

        payload = self.valid_payload()
        payload["ball"] = {"source": "unavailable"}
        with self.assertRaises(VideoContractError):
            validate_human_annotations(payload)
        payload = self.valid_payload()
        payload["ball"] = {"value": {"x": 1, "y": 2}, "source": "unavailable"}
        with self.assertRaises(VideoContractError):
            validate_human_annotations(payload)


if __name__ == "__main__":
    unittest.main()
