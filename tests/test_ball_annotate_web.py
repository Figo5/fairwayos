"""Tests for the local annotation UI.

The UI is the tool a careful pass actually uses, so these tests drive it over
HTTP rather than trusting the page: identity and hash are shown, a click is
stored in *source* pixels, the magnifier cannot alter label geometry, a
non-visible state can never carry a coordinate, undo and skip behave, a machine
guess stays a proposal, and labels survive a restart (resume).
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from ghostcaddie.video import ball_dataset as bd  # noqa: E402
import ball_annotate_web as web  # noqa: E402


def make_clip(tmp: Path, name: str, *, frames: int = 12, width: int = 64,
              height: int = 48, fill: int = 40) -> tuple[str, str]:
    import cv2
    import numpy as np

    media_dir = tmp / "out" / "ball_dataset_v2" / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    path = media_dir / f"{name}.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                             (width, height))
    for index in range(frames):
        frame = np.full((height, width, 3), (fill + index * 5) % 255, dtype="uint8")
        cv2.circle(frame, (20 + index, 24), 3, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return f"out/ball_dataset_v2/media/{name}.mp4", hashlib.sha256(
        path.read_bytes()).hexdigest()


class AnnotatorHttpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        media_path, sha = make_clip(self.tmp, "pexels_1000000")
        self.clip_id = "pexels_1000000"
        self.sources_path = self.tmp / "sources.json"
        self.sources_path.write_text(json.dumps({
            "schema_version": bd.SCHEMA_SOURCES, "dataset_id": "fixture",
            "clips": [{
                "clip_id": self.clip_id, "provider": "pexels",
                "source_identifier": "1000000",
                "source_url": "https://www.pexels.com/video/1000000/",
                "download_url": "https://www.pexels.com/download/video/1000000/",
                "title": "fixture", "acquisition_date": "2026-09-18",
                "license": "Pexels License",
                "license_url": "https://www.pexels.com/license/",
                "license_text_reference": None,
                "rights_class": "stock_license_reusable",
                "original_filename": f"{self.clip_id}.mp4", "media_path": media_path,
                "bytes": 4096, "sha256": sha, "width": 64, "height": 48, "fps": 10.0,
                "frames": 12, "duration_s": 1.2, "codec": "h264", "container": "mp4",
                "source_group": "pexels_1000", "intended_tags": [], "tags": [],
                "notes": ""}]}))
        self.annotations = self.tmp / "annotations"
        self.sha = sha
        self._servers = []

    def start(self):
        server, state = web.build_server(
            self.clip_id, host="127.0.0.1", port=0, sources_path=self.sources_path,
            annotations_dir=self.annotations, project_root=self.tmp)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._servers.append(server)
        self.addCleanup(server.shutdown)
        return server, state

    def call(self, base: str, path: str, body: dict | None = None,
             method: str | None = None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(base + path, data=data,
                                         method=method or ("POST" if data else "GET"))
        if data:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, response.read(), response.headers
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), exc.headers

    def stored(self) -> dict:
        path = bd.annotation_path_for(self.annotations, self.clip_id)
        if not path.exists():
            # Nothing was written at all, which is the correct outcome when the
            # store refuses a label.
            return {"frames": {}, "proposals": {}}
        return json.loads(path.read_text())


class AnnotatorUiTests(AnnotatorHttpCase):
    def test_the_page_loads_and_names_the_magnifier_rule(self):
        server, _ = self.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        status, body, headers = self.call(base, "/")
        self.assertEqual(status, 200)
        text = body.decode()
        self.assertIn("ball annotate", text)
        self.assertIn("never changes label geometry", text)
        self.assertIn("AMBIGUOUS", text)

    def test_the_clip_identity_and_source_hash_are_shown(self):
        server, _ = self.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        status, body, _ = self.call(base, "/api/clip")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["clip_id"], self.clip_id)
        self.assertEqual(payload["sha256"], self.sha)
        self.assertEqual((payload["width"], payload["height"]), (64, 48))
        self.assertTrue(payload["media_present"])
        self.assertEqual(payload["license"], "Pexels License")

    def test_a_frame_can_be_fetched_and_is_decodable(self):
        server, _ = self.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        status, body, headers = self.call(base, "/api/frame?index=3")
        self.assertEqual(status, 200)
        self.assertTrue(body.startswith(b"\xff\xd8"), "expected JPEG bytes")

    def test_the_magnifier_returns_a_crop_without_touching_labels(self):
        server, state = self.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        status, body, _ = self.call(base, "/api/crop?index=3&x=20&y=24&w=16&scale=8")
        self.assertEqual(status, 200)
        self.assertTrue(body.startswith(b"\x89PNG"))
        self.assertEqual(state.annotation["frames"], {})


class LabelWriteTests(AnnotatorHttpCase):
    def test_a_click_is_stored_in_source_pixels(self):
        server, _ = self.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        status, body, _ = self.call(base, "/api/label", {
            "frame": 5, "state": "VISIBLE", "x": 21, "y": 23, "origin": "AGENT_REVIEWED",
            "reviewer": "agent-visual", "tags": ["glare"], "confidence": "medium"})
        self.assertEqual(status, 200, body)
        entry = self.stored()["frames"]["5"]
        self.assertEqual((entry["x"], entry["y"]), (21.0, 23.0),
                         "the stored coordinate must be the source pixel, not the "
                         "display pixel")
        self.assertEqual(entry["state"], "VISIBLE")
        self.assertEqual(entry["origin"], "AGENT_REVIEWED")
        self.assertEqual(entry["tags"], ["glare"])

    def test_visible_without_coordinates_is_refused(self):
        server, _ = self.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        status, body, _ = self.call(base, "/api/label",
                                    {"frame": 5, "state": "VISIBLE", "x": 10})
        self.assertEqual(status, 400)
        self.assertIn("requires x and y", json.loads(body)["error"])
        self.assertEqual(self.stored().get("frames", {}), {})

    def test_a_non_visible_state_may_not_carry_coordinates(self):
        server, _ = self.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        for state in ("NOT_VISIBLE", "OCCLUDED", "AMBIGUOUS"):
            status, body, _ = self.call(base, "/api/label", {
                "frame": 5, "state": state, "x": 10, "y": 10,
                "reason": "fixture", "explanation": "fixture"})
            self.assertEqual(status, 400, state)
            self.assertIn("forbids coordinates", json.loads(body)["error"])
        self.assertEqual(self.stored().get("frames", {}), {})

    def test_absent_occluded_and_ambiguous_are_recorded_with_their_reasons(self):
        server, _ = self.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        cases = [("NOT_VISIBLE", 1), ("OCCLUDED", 2), ("AMBIGUOUS", 3)]
        for state, frame in cases:
            status, body, _ = self.call(base, "/api/label", {
                "frame": frame, "state": state, "reason": f"{state} reason",
                "explanation": f"{state} explanation"})
            self.assertEqual(status, 200, body)
        frames = self.stored()["frames"]
        self.assertEqual(frames["1"]["reason"], "NOT_VISIBLE reason")
        self.assertEqual(frames["2"]["reason"], "OCCLUDED reason")
        self.assertEqual(frames["3"]["explanation"], "AMBIGUOUS explanation")
        for key in ("1", "2", "3"):
            self.assertNotIn("x", frames[key])
            self.assertNotIn("y", frames[key])

    def test_a_coordinate_outside_the_frame_is_refused(self):
        server, _ = self.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        status, body, _ = self.call(base, "/api/label",
                                    {"frame": 5, "state": "VISIBLE", "x": 500, "y": 10})
        self.assertEqual(status, 400)
        self.assertIn("outside the frame", json.loads(body)["error"])

    def test_undo_removes_a_label(self):
        server, _ = self.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        self.call(base, "/api/label", {"frame": 5, "state": "VISIBLE", "x": 10, "y": 10})
        self.assertIn("5", self.stored()["frames"])
        status, body, _ = self.call(base, "/api/undo", {"frame": 5})
        self.assertEqual(status, 200, body)
        self.assertNotIn("5", self.stored()["frames"])

    def test_a_skipped_frame_is_not_a_label(self):
        server, _ = self.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        status, body, _ = self.call(base, "/api/skip", {"frame": 7})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["skipped"], [7])
        self.assertEqual(self.stored().get("frames", {}), {})

    def test_progress_reports_states_and_next_work(self):
        server, _ = self.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        self.call(base, "/api/label", {"frame": 1, "state": "VISIBLE", "x": 5, "y": 5})
        self.call(base, "/api/label", {"frame": 2, "state": "NOT_VISIBLE",
                                       "reason": "left the frame"})
        status, body, _ = self.call(base, "/api/progress")
        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["labelled"], 2)
        self.assertEqual(payload["states"]["VISIBLE"], 1)
        self.assertEqual(payload["states"]["NOT_VISIBLE"], 1)
        self.assertAlmostEqual(payload["progress"], 2 / 12, places=4)


class ProposalTests(AnnotatorHttpCase):
    def test_a_carry_over_guess_stays_a_proposal(self):
        server, state = self.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        self.call(base, "/api/label", {"frame": 4, "state": "VISIBLE", "x": 30, "y": 24})
        status, body, _ = self.call(base, "/api/propose", {"frame": 5})
        self.assertEqual(status, 200)
        proposal = json.loads(body)["proposal"]
        self.assertEqual(proposal["status"], "pending")
        self.assertEqual((proposal["x"], proposal["y"]), (30.0, 24.0))
        self.assertNotIn("5", self.stored()["frames"],
                         "a proposal must never appear as a label")
        self.assertEqual(self.stored()["proposals"]["5"]["origin"], "AGENT_PROPOSED")

    def test_no_proposal_is_offered_without_a_previous_visible_label(self):
        server, _ = self.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        status, body, _ = self.call(base, "/api/propose", {"frame": 5})
        self.assertEqual(status, 200)
        self.assertIsNone(json.loads(body)["proposal"])


class ResumeTests(AnnotatorHttpCase):
    def test_labels_survive_a_restart(self):
        server, _ = self.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        self.call(base, "/api/label", {"frame": 2, "state": "VISIBLE", "x": 12, "y": 9,
                                       "radius": 3})
        self.call(base, "/api/label", {"frame": 3, "state": "OCCLUDED",
                                       "reason": "club face"})
        server.shutdown()

        server2, state2 = self.start()
        base2 = f"http://127.0.0.1:{server2.server_address[1]}"
        status, body, _ = self.call(base2, "/api/progress?frame=2")
        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["labelled"], 2)
        self.assertEqual(payload["label"]["x"], 12.0)
        self.assertEqual(payload["label"]["radius"], 3.0)
        self.assertEqual(state2.annotation["frames"]["3"]["reason"], "club face")

    def test_the_annotation_stays_bound_to_the_source_bytes(self):
        server, _ = self.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        self.call(base, "/api/label", {"frame": 2, "state": "VISIBLE", "x": 12, "y": 9})
        stored = self.stored()
        self.assertEqual(stored["source_sha256"], self.sha)
        annotation = bd.load_annotation(
            bd.annotation_path_for(self.annotations, self.clip_id),
            project_root=self.tmp)
        self.assertEqual(annotation["clip_id"], self.clip_id)


class MissingMediaTests(AnnotatorHttpCase):
    def test_missing_media_is_reported_rather_than_silently_blank(self):
        server, state = self.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        state.media_path.unlink()
        status, body, _ = self.call(base, "/api/frame?index=1")
        self.assertEqual(status, 409)
        self.assertIn("Media is not committed", json.loads(body)["error"])


if __name__ == "__main__":
    unittest.main()
