import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ghostcaddie.demo import target_benchmark as bench
from ghostcaddie.demo import saved_tracks
from ghostcaddie.demo.errors import DemoConfigError
from ghostcaddie.demo.saved_tracks import build_saved_track, load_saved_tracks


def point(frame, status, origin, x=None, y=None, method="", score=None):
    return {"source_frame": frame, "status": status, "origin": origin,
            "x": x, "y": y, "score": score, "method": method}


class SavedTracksBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "predictions.json"
        self.record = SimpleNamespace(sha256="a" * 64, frame_count=10,
                                      width=100, height=80)
        self.payload = {
            "schema_version": "fairwayos-target-predictions.v1",
            "variant": "research",
            "clips": [{"id": "source", "sha256": self.record.sha256, "targets": {
                "clubhead": {"seed": {"frame": 2, "x": 50, "y": 60}, "points": [
                    point(2, "manual_anchor", "operator", 50, 60, "operator_placement"),
                    point(3, "observed_template", "automatic", 51, 60,
                          "sam2_head_region"),
                    point(4, "missing", "derived"),
                    point(6, "interpolated", "derived", 55, 60),
                    point(7, "observed_template", "automatic", 56, 60,
                          "sam2_head_region_reacquired"),
                    point(8, "lost", "derived")]}}}]}

    def write(self, payload=None):
        self.path.write_text(json.dumps(self.payload if payload is None else payload))
        return self.path

    def load(self, payload=None, strict=True):
        return load_saved_tracks(self.write(payload), [self.record],
                                 require_all_registered=strict)

    def block(self, payload=None):
        result = self.load(payload)
        return result["clips_by_sha"][self.record.sha256]["targets"]["clubhead"]


class LoadSavedTracksTests(SavedTracksBase):
    def test_returns_hash_variant_and_clips_by_sha(self):
        result = self.load()
        self.assertEqual(set(result),
                         {"prediction_sha256", "variant", "clips_by_sha"})
        self.assertEqual(result["variant"], "research")
        self.assertRegex(result["prediction_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(result["prediction_sha256"],
                         bench.sha256_text(self.path.read_text()))
        clip = result["clips_by_sha"][self.record.sha256]
        self.assertEqual(clip["id"], "source")
        self.assertEqual((clip["width"], clip["height"], clip["frame_count"]),
                         (100, 80, 10))
        self.assertEqual(set(clip["targets"]), {"clubhead"})

    def test_target_block_carries_validated_seed_and_points(self):
        block = self.block()
        self.assertEqual(block["seed"], {"frame": 2, "x": 50.0, "y": 60.0})
        frames = [p["source_frame"] for p in block["points"]]
        self.assertEqual(frames, [2, 3, 4, 6, 7, 8])
        for p in block["points"]:
            if p["status"] in ("missing", "lost"):
                self.assertIsNone(p["x"])
                self.assertIsNone(p["y"])
                self.assertIsNone(p["score"])

    def test_unknown_hash_refused_strict_but_permitted_when_not_strict(self):
        other = {"id": "other", "sha256": "b" * 64, "targets": {
            "ball": {"seed": {"frame": 1, "x": 10, "y": 10},
                     "points": [point(1, "manual_anchor", "operator", 10, 10,
                                      "operator_placement")]}}}
        payload = json.loads(json.dumps(self.payload))
        payload["clips"].append(other)
        with self.assertRaises(DemoConfigError):
            self.load(payload, strict=True)
        result = self.load(payload, strict=False)
        self.assertEqual(list(result["clips_by_sha"]), [self.record.sha256])

    def test_duplicate_ids_and_hashes_are_refused_everywhere(self):
        dup_id = json.loads(json.dumps(self.payload))
        dup_id["clips"].append({"id": "source", "sha256": "b" * 64, "targets": {
            "ball": {"seed": None, "points": []}}})
        with self.assertRaises(DemoConfigError):
            self.load(dup_id, strict=False)

        dup_sha = json.loads(json.dumps(self.payload))
        dup_sha["clips"].append({"id": "other", "sha256": self.record.sha256,
                                 "targets": {
                                     "ball": {"seed": None, "points": []}}})
        with self.assertRaises(DemoConfigError):
            self.load(dup_sha, strict=False)

    def test_unknown_target_is_refused(self):
        payload = json.loads(json.dumps(self.payload))
        payload["clips"][0]["targets"]["shaft"] = {"seed": None, "points": []}
        with self.assertRaises(DemoConfigError):
            self.load(payload)

    def test_bogus_method_status_and_origin_are_refused(self):
        payload = json.loads(json.dumps(self.payload))
        payload["clips"][0]["targets"]["clubhead"]["points"][1]["method"] = "manual_path"
        with self.assertRaises(DemoConfigError):
            self.load(payload)

        payload = json.loads(json.dumps(self.payload))
        payload["clips"][0]["targets"]["clubhead"]["points"][1]["origin"] = "operator"
        with self.assertRaises(DemoConfigError):
            self.load(payload)

        payload = json.loads(json.dumps(self.payload))
        payload["clips"][0]["targets"]["clubhead"]["points"][1]["status"] = "teleported"
        with self.assertRaises(DemoConfigError):
            self.load(payload)

    def test_automatic_at_seed_frame_is_refused(self):
        payload = json.loads(json.dumps(self.payload))
        payload["clips"][0]["targets"]["clubhead"]["points"] = [
            point(2, "observed_template", "automatic", 50, 60, "sam2_head_region")]
        with self.assertRaises(DemoConfigError):
            self.load(payload)

    def test_automatic_before_the_seed_is_refused(self):
        payload = json.loads(json.dumps(self.payload))
        payload["clips"][0]["targets"]["clubhead"]["points"] = [
            point(1, "observed_template", "automatic", 50, 60, "sam2_head_region")]
        with self.assertRaises(DemoConfigError):
            self.load(payload)

    def test_seed_bounds_and_non_integer_frames_are_refused(self):
        for field, bad in [("frame", True), ("frame", 10), ("x", 101),
                           ("y", 80.0001), ("x", float("nan"))]:
            payload = json.loads(json.dumps(self.payload))
            payload["clips"][0]["targets"]["clubhead"]["seed"][field] = bad
            with self.assertRaises(DemoConfigError):
                self.load(payload)

    def test_source_hash_mismatch_is_refused(self):
        payload = json.loads(json.dumps(self.payload))
        payload["clips"][0]["sha256"] = "b" * 64
        with self.assertRaises(DemoConfigError):
            self.load(payload, strict=True)

    def test_file_over_16mib_is_refused(self):
        self.write()
        with mock.patch.object(saved_tracks, "MAX_SAVED_TRACK_BYTES", 10):
            with self.assertRaises(DemoConfigError):
                load_saved_tracks(self.path, [self.record])


class BuildSavedTrackTests(SavedTracksBase):
    def test_gap_breaks_the_trail_and_reacquisition_is_preserved(self):
        track = build_saved_track("clubhead", self.block(), range(0, 10))
        by_frame = {p.source_frame: p for p in track.points}
        self.assertEqual(by_frame[2].status, "manual_anchor")
        self.assertEqual(by_frame[2].origin, "operator")
        self.assertEqual((by_frame[2].x, by_frame[2].y), (50.0, 60.0))
        self.assertEqual(by_frame[3].status, "observed_template")
        self.assertEqual(by_frame[3].method, "sam2_head_region")
        self.assertEqual(by_frame[7].status, "observed_template")
        self.assertEqual(by_frame[7].method, "sam2_head_region_reacquired")
        for frame in (0, 1, 4, 5, 6, 9):
            self.assertEqual(by_frame[frame].status, "missing")
            self.assertIsNone(by_frame[frame].x)
        # the trail is split: the re-acquired point cannot join the first leg
        self.assertNotEqual(by_frame[3].segment, by_frame[7].segment)
        self.assertEqual(len(track.segments), 2)
        self.assertEqual(track.coverage["interpolated_frames"], 0)
        self.assertEqual(track.coverage["automatic_observed_frames"], 2)
        self.assertEqual(track.coverage["operator_anchor_frames"], 1)

    def test_interpolated_is_excluded_and_emitted_missing(self):
        track = build_saved_track("clubhead", self.block(), range(2, 9))
        by_frame = {p.source_frame: p for p in track.points}
        self.assertEqual(by_frame[6].status, "missing")
        self.assertIsNone(by_frame[6].x)
        self.assertEqual(track.coverage["interpolated_frames"], 0)
        self.assertTrue(any("excluded from rendering" in n
                            for n in track.limitations))

    def test_seed_origin_and_score_are_preserved(self):
        payload = json.loads(json.dumps(self.payload))
        payload["clips"][0]["targets"]["clubhead"]["points"][1]["method"] = "template"
        payload["clips"][0]["targets"]["clubhead"]["points"][1]["score"] = 0.5
        block = self.block(payload)
        track = build_saved_track("clubhead", block, range(2, 5))
        by_frame = {p.source_frame: p for p in track.points}
        self.assertEqual(track.seed["source"], "seed")
        self.assertEqual(by_frame[3].score, 0.5)       # classic score preserved
        self.assertEqual(by_frame[3].origin, "automatic")
        self.assertIsNone(by_frame[2].score)           # seed carries no score
        # a SAM2 observation keeps its null score: nothing is invented
        sam2 = self.block()["points"][1]
        self.assertIsNone(sam2["score"])

    def test_configured_seed_must_agree_and_supplies_when_absent(self):
        block = self.block()
        matching = build_saved_track("clubhead", block, range(2, 4),
                                     expected_seed={"frame": 2, "x": 50, "y": 60})
        self.assertEqual(matching.seed["frame"], 2)
        with self.assertRaises(DemoConfigError):
            build_saved_track("clubhead", block, range(2, 4),
                              expected_seed={"frame": 3, "x": 50, "y": 60})
        with self.assertRaises(DemoConfigError):
            build_saved_track("clubhead", block, range(2, 4),
                              expected_seed={"frame": 2, "x": 51, "y": 60})
        # saved seed supplies the seed when the caller has none
        supplied = build_saved_track("clubhead", block, range(2, 4))
        self.assertEqual((supplied.seed["x"], supplied.seed["y"]), (50.0, 60.0))

    def test_no_automatic_crosses_a_source_cut(self):
        payload = {"schema_version": "fairwayos-target-predictions.v1",
                   "variant": "research",
                   "clips": [{"id": "source", "sha256": self.record.sha256,
                              "targets": {"clubhead": {
                                  "seed": {"frame": 2, "x": 50, "y": 60},
                                  "points": [
                                      point(2, "manual_anchor", "operator", 50, 60,
                                            "operator_placement"),
                                      point(3, "observed_template", "automatic", 51,
                                            60, "sam2_head_region"),
                                      point(4, "observed_template", "automatic", 52,
                                            60, "sam2_head_region"),
                                      point(5, "observed_template", "automatic", 53,
                                            60, "sam2_head_region")]}}}]}
        block = self.block(payload)
        track = build_saved_track("clubhead", block, range(2, 6), cut_frames=(4,))
        by_frame = {p.source_frame: p for p in track.points}
        self.assertEqual(by_frame[3].status, "observed_template")
        self.assertEqual(by_frame[4].status, "missing")
        self.assertEqual(by_frame[4].x, None)
        self.assertEqual(by_frame[5].status, "missing")
        self.assertTrue(any("structural cut" in n for n in track.limitations))
        # with no cut the same observations survive
        uncut = build_saved_track("clubhead", block, range(2, 6))
        kept = {p.source_frame: p for p in uncut.points}
        self.assertEqual(kept[4].status, "observed_template")
        self.assertEqual(kept[5].status, "observed_template")

    def test_manual_anchor_other_than_the_seed_is_refused(self):
        block = {"seed": {"frame": 2, "x": 50, "y": 60},
                 "points": [point(4, "manual_anchor", "operator", 44, 44,
                                  "operator_placement")]}
        with self.assertRaises(DemoConfigError):
            build_saved_track("clubhead", block, range(2, 6))

    def test_unknown_target_name_is_refused(self):
        with self.assertRaises(DemoConfigError):
            build_saved_track("shaft", self.block(), range(2, 4))

    def test_unrepresented_frames_are_missing(self):
        track = build_saved_track("clubhead", self.block(), [2, 50, 3])
        self.assertEqual([p.source_frame for p in track.points], [2, 50, 3])
        by_frame = {p.source_frame: p for p in track.points}
        self.assertEqual(by_frame[50].status, "missing")
        self.assertEqual(track.coverage["missing_frames"], 1)
        self.assertEqual(track.accounting["space"], "2d_image")

    def test_unseeded_block_renders_nothing(self):
        block = {"seed": None, "points": []}
        track = build_saved_track("clubhead", block, range(0, 3))
        self.assertFalse(track.available)
        self.assertEqual(track.seed, None)
        self.assertTrue(all(p.status == "missing" for p in track.points))


if __name__ == "__main__":
    unittest.main()
