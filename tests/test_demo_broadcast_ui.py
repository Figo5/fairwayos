"""Tests for the broadcast review UI: the page's own behaviour, and the server's
clip binding, config filtering, import and save round-trip.

Why these exist: primary reviewed the running UI in a browser and found that loading a
config left the previously selected clip active (so a Hole 12 config's scenes were
edited against the Hole 7 clip), that a save silently dropped ``manual_segments``,
``roi_keyframes``, ``presentation`` and ``cuts``, and that dragging a region never
reached the scene that gets saved. Every case below is one of those, exercised against
the code that ships.

The page is driven through node with a small DOM stub (``tests/fixtures/
broadcast_ui_dom.js``) so the assertions run against the real script, not a
re-implementation of it; the suite skips cleanly where node is absent.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ghostcaddie.demo import broadcast as broadcast_module
from ghostcaddie.demo import broadcast_config as bc
from ghostcaddie.demo.broadcast import _parent_source_record, verify_broadcast_render
from ghostcaddie.demo.broadcast_ui import Registry, make_handler
from ghostcaddie.demo.broadcast_ui_page import PAGE
from ghostcaddie.demo.errors import DemoConfigError
from tests.demo_fixtures import TempDirTestCase, synthetic_probe

NODE = shutil.which("node")
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def base_scene(**overrides) -> dict:
    scene = {
        "id": "s1", "label": "scene one", "kind": "golfer_stroke",
        "stroke_type": "full_swing",
        "window": {"start_frame": 10, "end_frame": 40},
        "keyframes": {"address": 10, "top": 22, "impact": 30, "finish": 38},
        "roi": [40, 20, 80, 90],
        "seeds": {"ball": {"frame": 12, "x": 50.0, "y": 60.0}},
        "targets": {"ball": {"mode": "template", "template_size": 21,
                             "search_radius": 30, "min_score": 0.5}},
    }
    scene.update(overrides)
    return scene


def base_config(**overrides) -> dict:
    payload = {
        "schema_version": bc.BROADCAST_CONFIG_SCHEMA,
        "clip_path_hint": "clip.mp4",
        "shot": {"id": "shot1", "label": "shot one", "stroke_type": "full_swing"},
        "interval": {"start_seconds": 0.0, "end_seconds": None, "max_frames": 400},
        "scenes": [base_scene()],
        "render": {"max_width": 320, "crf": 30, "preset": "ultrafast"},
    }
    payload.update(overrides)
    return payload


# ------------------------------------------------------------ the page itself

class UiPageBehaviourTests(unittest.TestCase):
    """The shipped page's script, run under node with a DOM stub."""

    def test_the_page_preserves_documents_and_commits_edits(self):
        if not NODE:
            self.skipTest("node is not on PATH")
        prelude = (REPO / "tests" / "fixtures" / "broadcast_ui_dom.js").read_text(
            encoding="utf-8")
        behaviour = (REPO / "tests" / "fixtures" / "broadcast_ui_behaviour.js").read_text(
            encoding="utf-8")
        script = PAGE.split("<script>", 1)[1].split("</script>", 1)[0]
        program = "\n".join([prelude, script, behaviour])
        import tempfile

        with tempfile.TemporaryDirectory(prefix="bcui-page-") as tmp:
            path = Path(tmp) / "page_behaviour.mjs"
            path.write_text(program, encoding="utf-8")
            proc = subprocess.run([NODE, str(path)], capture_output=True, text=True,
                                  timeout=300)
        detail = (proc.stdout or "")[-4000:] + (proc.stderr or "")[-2000:]
        self.assertTrue(proc.stdout.strip(), f"the page produced no result: {detail}")
        line = [row for row in proc.stdout.splitlines() if row.startswith("{")][-1]
        report = json.loads(line)
        failures = [row for row in report["results"] if not row["ok"]]
        self.assertEqual(
            failures, [],
            "the review page failed its own cases: "
            + json.dumps([{row["name"]: row["detail"]} for row in failures], indent=2)
            + f"\nnode said: {detail}")
        self.assertEqual(proc.returncode, 0, detail)
        self.assertGreater(len(report["results"]), 25)

    def test_the_page_loads_a_config_only_onto_its_own_clip(self):
        """The refusal path must be in the page, not only in the operator's head."""
        self.assertIn("resolveClipForConfig", PAGE)
        self.assertIn("not loaded: ", PAGE)
        self.assertIn("Import that clip above", PAGE)
        # The clip is selected (which refreshes the frame bounds) before any config
        # field is written to the screen.
        self.assertLess(PAGE.index("selectClip(resolved.clip.id)"),
                        PAGE.index("applyConfig(config, name, resolved.how)"))

    def test_the_page_edits_the_loaded_document_rather_than_rebuilding_it(self):
        self.assertIn("state.config = clone(config)", PAGE)
        self.assertIn("function commitEditors", PAGE)
        self.assertIn("function addScene", PAGE)
        self.assertIn("function addCorrection", PAGE)
        self.assertIn("function addAnchorToSegment", PAGE)
        self.assertIn("function addRoiKeyframe", PAGE)
        # A save must not be the place presentation/timing/cuts are re-created.
        self.assertNotIn("presentation:{", PAGE)
        self.assertNotIn("cuts:[", PAGE)


# ------------------------------------------------------------ the server

class UiConfigBindingTests(TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.clips = self.tmp / "clips"
        self.clips.mkdir()
        self.configs = self.tmp / "configs"
        self.configs.mkdir()
        self.other = self.tmp / "elsewhere"
        self.other.mkdir()

        def make(path: Path, flag: str) -> None:
            path.write_bytes(flag.encode() * 32)

        self.clip_a = self.clips / "01_hole07_approach.mp4"
        make(self.clip_a, "a")
        self.clip_c = self.clips / "03_hole12_tee.mp4"
        make(self.clip_c, "c")
        nested = self.clips / "new-clips"
        nested.mkdir()
        self.clip_new = nested / "09_hole03_tee.mp4"
        make(self.clip_new, "n")
        self.clip_outside = self.other / "stray.mp4"
        make(self.clip_outside, "s")

        self.registry = Registry([self.clips])
        handler = make_handler(self.registry, "tok123", self.configs,
                               {"python": sys.executable, "out_root": str(self.tmp)})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        super().tearDown()

    def clip_id(self, name: str) -> str:
        for record in self.registry.clips.values():
            if record.name == name:
                return record.clip_id
        raise AssertionError(f"{name} was not registered")

    def sha(self, name: str) -> str:
        for record in self.registry.clips.values():
            if record.name == name:
                return record.sha256
        raise AssertionError(f"{name} was not registered")

    def call(self, path, *, method="GET", body=None, token=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if token is not None:
            req.add_header("X-Broadcast-Token", token)
        try:
            with urllib.request.urlopen(req) as response:
                raw = response.read().decode()
                return response.status, (json.loads(raw) if raw.strip().startswith(
                    ("{", "[")) else raw)
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode()
            try:
                return exc.code, json.loads(payload or "{}")
            except json.JSONDecodeError:
                return exc.code, {"raw": payload}

    def write_config(self, name: str, payload: dict) -> Path:
        path = self.configs / name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def test_each_render_pins_its_config_and_has_independent_output(self):
        from ghostcaddie.demo.broadcast_ui import _start_render
        cid = self.clip_id('01_hole07_approach.mp4')
        config = base_config(clip_path_hint=str(self.clip_a))
        config['shot']['expected_source_sha256'] = self.sha('01_hole07_approach.mp4')
        config['shot']['parent_source_manifest'] = 'manifest.json'
        payload = {'clip': cid, 'config': config, 'name': 'review.json', 'save_config': True}
        settings = {'python': sys.executable, 'out_root': str(self.tmp / 'renders')}
        from dataclasses import replace
        probe = replace(synthetic_probe(width=320,height=180,frame_count=120),path=str(self.clip_a))
        with patch('ghostcaddie.demo.probe.probe_source',return_value=probe), \
             patch('ghostcaddie.demo.broadcast_ui.threading.Thread') as thread:
            first = _start_render(self.registry,self.configs,settings,payload)
            config['scenes'][0]['label'] = 'second document'
            second = _start_render(self.registry,self.configs,settings,payload)
        jobs = [self.registry.jobs[r['job']['id']] for r in [first,second]]
        self.assertNotEqual(jobs[0]['out_dir'],jobs[1]['out_dir'])
        self.assertNotEqual(jobs[0]['config'],jobs[1]['config'])
        self.assertEqual(json.loads(Path(jobs[0]['config']).read_text())['scenes'][0]['label'],'scene one')
        self.assertEqual(json.loads(Path(jobs[1]['config']).read_text())['scenes'][0]['label'],'second document')
        self.assertEqual(thread.call_args_list[0].kwargs['args'][3],jobs[0]['config'])
        self.write_config('review.json',base_config())
        code, pinned = self.call('/api/config?id=' + jobs[0]['id'])
        self.assertEqual(code,200)
        self.assertEqual(pinned['scenes'][0]['label'],'scene one')
        self.assertEqual(pinned['shot']['parent_source_manifest'],str((self.configs / 'manifest.json').resolve()))

    def test_saved_points_api_is_source_bound_and_read_only(self):
        cid = self.clip_id('01_hole07_approach.mp4')
        sha = self.sha('01_hole07_approach.mp4')
        target = {'seed': {'frame': 2, 'x': 50, 'y': 60}, 'points': []}
        self.registry.saved_tracks = {'prediction_sha256': 'c'*64,
            'clips_by_sha': {sha: {'targets': {'clubhead': target}}}}
        code, result = self.call(f'/api/tracking/points?clip={cid}&target=clubhead')
        self.assertEqual(code, 200); self.assertEqual(result['result'], target)
        self.assertEqual(result['source_sha256'], sha); self.assertTrue(result['available'])
        code, _ = self.call('/api/tracking/points?clip=../../clip.mp4')
        self.assertEqual(code, 400)
        code, _ = self.call(f'/api/tracking/points?clip={cid}&target=shaft')
        self.assertEqual(code, 400)
        code, _ = self.call('/api/tracking/points',method='POST',body={},token='tok123')
        self.assertEqual(code, 404)

    def test_tracking_review_uses_registered_clip_ids_and_has_no_write_endpoint(self):
        cid = self.clip_id('01_hole07_approach.mp4')
        sha = self.sha('01_hole07_approach.mp4')
        summary = {'target': 'clubhead', 'source_sha256': sha, 'gaps': [
            {'start_frame': 3, 'end_frame': 5}], 'accuracy_claim': False}
        self.registry.tracking_review[sha] = {'clubhead': summary}
        code, result = self.call(f'/api/tracking?clip={cid}&target=clubhead')
        self.assertEqual(code, 200); self.assertEqual(result['tracking'], summary)
        _, result = self.call(f'/api/tracking?clip={cid}&target=ball')
        self.assertIsNone(result['tracking'])
        code, _ = self.call('/api/tracking?clip=../../outside.mp4')
        self.assertEqual(code, 400)
        code, _ = self.call(f'/api/tracking?clip={cid}&target=shaft')
        self.assertEqual(code, 400)
        code, _ = self.call('/api/tracking', method='POST', body={}, token='tok123')
        self.assertEqual(code, 404)

    def test_the_config_list_holds_only_loadable_broadcast_configs(self):
        self.write_config("03_hole12_tee.broadcast.json", base_config(
            clip_path_hint=str(self.clip_c),
            shot={"id": "03_hole12_tee", "stroke_type": "full_swing",
                  "expected_source_sha256": self.sha("03_hole12_tee.mp4")}))
        self.write_config("cuts-03.json", {"clip": "03_hole12_tee.mp4",
                                           "candidates": [{"frame": 96}]})
        self.write_config("batch-smoke.json", {"entries": []})
        self.write_config("demo.json", dict(base_config(),
                                            schema_version="fairwayos-demo-config.v1"))
        self.write_config("broken.broadcast.json", {"schema_version":
                                                    bc.BROADCAST_CONFIG_SCHEMA,
                                                    "scenes": []})
        _, payload = self.call("/api/configs")
        names = [entry["name"] for entry in payload["configs"]]
        self.assertIn("03_hole12_tee.broadcast.json", names)
        for absent in ("cuts-03.json", "batch-smoke.json", "demo.json"):
            self.assertNotIn(absent, names)
        entry = next(row for row in payload["configs"]
                     if row["name"] == "03_hole12_tee.broadcast.json")
        self.assertTrue(entry["loadable"])
        self.assertEqual(entry["clip_id"], self.clip_id("03_hole12_tee.mp4"))
        self.assertEqual(entry["clip_resolved_by"], "expected_source_sha256")
        # A config that the loader refuses is listed as refused, not as loadable.
        broken = next(row for row in payload["configs"]
                      if row["name"] == "broken.broadcast.json")
        self.assertFalse(broken["loadable"])
        self.assertIn("error", broken)

    def test_a_config_for_another_clip_reports_the_other_clip(self):
        """This is the bug: the config's own clip, not whichever is selected."""
        self.write_config("01_hole07_approach.broadcast.json", base_config(
            clip_path_hint=str(self.clip_a),
            shot={"id": "01_hole07_approach", "stroke_type": "full_swing",
                  "expected_source_sha256": self.sha("01_hole07_approach.mp4")}))
        self.write_config("03_hole12_tee.broadcast.json", base_config(
            clip_path_hint=str(self.clip_c),
            shot={"id": "03_hole12_tee", "stroke_type": "full_swing",
                  "expected_source_sha256": self.sha("03_hole12_tee.mp4")}))
        _, payload = self.call("/api/configs")
        resolved = {row["name"]: row["clip_id"] for row in payload["configs"]}
        self.assertEqual(resolved["03_hole12_tee.broadcast.json"],
                         self.clip_id("03_hole12_tee.mp4"))
        self.assertNotEqual(resolved["03_hole12_tee.broadcast.json"],
                            resolved["01_hole07_approach.broadcast.json"])

    def test_a_config_whose_clip_is_missing_resolves_to_nothing(self):
        self.write_config("gone.broadcast.json", base_config(
            clip_path_hint=str(self.tmp / "nowhere" / "99_gone.mp4"),
            shot={"id": "gone", "stroke_type": "putt"}))
        _, payload = self.call("/api/configs")
        entry = payload["configs"][0]
        self.assertIsNone(entry["clip_id"])
        self.assertIn("not in the library", entry["clip_resolved_by"])

    def test_loading_a_non_broadcast_file_is_refused(self):
        self.write_config("cuts-03.json", {"clip": "x", "candidates": []})
        status, payload = self.call("/api/config?name=cuts-03.json")
        self.assertEqual(status, 400)
        self.assertIn("not a broadcast config", payload["error"])

    def test_loading_a_broadcast_config_returns_it_unchanged(self):
        payload = base_config(clip_path_hint=str(self.clip_c))
        payload["notes"] = "an operator comment"
        self.write_config("c.json", payload)
        status, body = self.call("/api/config?name=c.json")
        self.assertEqual(status, 200)
        self.assertEqual(body, payload)

    def test_the_library_scans_nested_folders_and_confines_imports(self):
        _, payload = self.call("/api/library")
        root = payload["roots"][0]
        names = [item["name"] for item in root["files"]]
        self.assertIn("new-clips/09_hole03_tee.mp4", names)
        self.assertIn("01_hole07_approach.mp4", names)

        status, payload = self.call(
            "/api/import", method="POST", token="tok123",
            body={"root": str(self.clips), "name": "new-clips/09_hole03_tee.mp4"})
        self.assertEqual(status, 200)
        self.assertEqual(payload["clip"]["name"], "09_hole03_tee.mp4")

        status, payload = self.call(
            "/api/import", method="POST", token="tok123",
            body={"path": str(self.clip_outside)})
        self.assertEqual(status, 400)
        self.assertIn("outside the allow-listed roots", payload["error"])

        status, payload = self.call(
            "/api/import", method="POST", token="tok123",
            body={"root": str(self.clips), "name": "../elsewhere/stray.mp4"})
        self.assertEqual(status, 400)
        self.assertIn("outside the allow-listed roots", payload["error"])

        status, payload = self.call(
            "/api/import", method="POST", token="tok123",
            body={"path": str(self.clip_a)})
        self.assertEqual(status, 200)
        self.assertEqual(payload["clip"]["name"], "01_hole07_approach.mp4")

    def test_a_save_writes_the_operators_document_back_whole(self):
        """No save may remove an assistance mark the operator loaded."""
        rich = base_config(
            clip_path_hint=str(self.clip_c),
            shot={"id": "03_hole12_tee", "label": "Hole 12 tee",
                  "stroke_type": "full_swing",
                  "expected_source_sha256": self.sha("03_hole12_tee.mp4"),
                  "source_notes": "operator's own words",
                  "parent_source_manifest": "manifest.json"},
            presentation={"brand": "FAIRWAY OS", "brand_sub": "SWING ANALYSIS",
                          "disclosure": "OPERATOR-ASSISTED",
                          "metrics": ["shoulder_tilt", "hip_tilt", "tempo"],
                          "card_debounce_frames": 1, "card_position": "auto",
                          "metrics_required": False},
            cuts=[{"frame": 41, "kind": "hard_cut", "confidence": 1.0,
                   "decision": "accepted", "decided_by": "primary",
                   "decided_at": "2026-10-01", "note": "cut"}],
            overrides=[{"frame": 41, "decision": "accept", "note": "kept"}],
            timing={"source_cadence_caveat": "nominal", "nominal_fps_trusted": False,
                    "accept_variable_cadence": False},
            notes="top level note",
            source_description="Tiger Woods, 2019 Masters",
            _resolved={"note": "a run's own comment"},
            scenes=[base_scene(
                roi_keyframes=[{"frame": 10, "roi": [40, 20, 80, 90],
                                "note": "first crop"}],
                corrections={"clubhead": [{"frame": 14, "x": 1.0, "y": 2.0,
                                           "note": "operator correction"}]},
                manual_segments={"clubhead": {
                    "author": "primary", "replace_automatic": True,
                    "hide_frames": [], "trail": "segments_only",
                    "trail_from_frame": 12,
                    "note": "operator note on the block",
                    "segments": [{"name": "downswing",
                                  "max_anchor_spacing_frames": 12,
                                  "anchors": [{"frame": 12, "x": 1.5, "y": 2.5,
                                               "note": "first"},
                                              {"frame": 16, "x": 3.5, "y": 4.5,
                                               "note": "second"}],
                                  "note": "segment note"}]}},
                interpolation={"enabled": True, "max_gap_frames": 2},
                metric_hide_frames=[[30, 36]])])
        status, saved = self.call("/api/save", method="POST", token="tok123",
                                  body={"name": "kept.json", "config": rich})
        self.assertEqual(status, 200, saved)
        written = json.loads((self.configs / "kept.json").read_text(encoding="utf-8"))
        for key in ("presentation", "cuts", "overrides", "timing", "notes",
                    "source_description", "_resolved"):
            self.assertEqual(written.get(key), rich[key], key)
        self.assertEqual(written["shot"], rich["shot"])
        scene = written["scenes"][0]
        for key in ("manual_segments", "roi_keyframes", "corrections", "seeds",
                    "targets", "interpolation", "metric_hide_frames"):
            self.assertEqual(scene.get(key), rich["scenes"][0][key], key)
        # And what was written is a broadcast config that loads again, with its derived
        # keys (start_frame/anchor_count/...) not smuggled into the saved file.
        again = bc.load_broadcast(str(self.configs / "kept.json"))
        self.assertEqual(again.scenes[0].demo.manual_segments["clubhead"].segments[0]
                         .frames(), [12, 16])

    def test_validate_and_render_refuse_a_config_for_another_clip(self):
        config = base_config(clip_path_hint=str(self.clip_a),
                             shot={"id": "01", "stroke_type": "full_swing",
                                   "expected_source_sha256": self.sha(
                                       "01_hole07_approach.mp4")})
        status, payload = self.call("/api/validate", method="POST", token="tok123",
                                    body={"clip": self.clip_id("03_hole12_tee.mp4"),
                                          "config": config})
        self.assertEqual(status, 400)
        self.assertIn("was written for", payload["error"])
        status, payload = self.call("/api/render", method="POST", token="tok123",
                                    body={"clip": self.clip_id("03_hole12_tee.mp4"),
                                          "config": config, "name": "x.json"})
        self.assertEqual(status, 400)
        self.assertIn("was written for", payload["error"])
        self.assertEqual(self.registry.jobs, {})
        self.assertFalse((self.configs / "x.json").exists())


# ------------------------------------------------------------ parent source

class ParentSourceManifestTests(TempDirTestCase):
    def _clip(self, name: str = "03_hole12_tee.mp4", *, folder: str = "clips") -> Path:
        directory = self.tmp / folder
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        path.write_bytes(b"clip" * 16)
        return path

    @staticmethod
    def _sha(path: Path) -> str:
        import hashlib

        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_no_user_specific_manifest_path_is_baked_into_the_module(self):
        self.assertFalse(hasattr(broadcast_module, "BASELINE_MANIFEST"))
        source = Path(broadcast_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("tiger-masters-2019-2026-10-01/manifest.json", source)
        self.assertNotIn("/Users/giofiore", source.split("PROTECTED_OUTPUT_PREFIXES")[1]
                         .split(")", 1)[1])
        self.assertIn("shot.parent_source_manifest", source)

    def test_an_explicit_manifest_is_used_and_matched_by_clip_sha(self):
        clip = self._clip()
        manifest = self.tmp / "records" / "manifest.json"
        manifest.parent.mkdir()
        manifest.write_text(json.dumps({
            "source": "/parent/source.mp4", "source_sha256": "1" * 64,
            "trim_method": "exact native trims", "body_model_sha256": "2" * 64,
            "clips": [{"id": "03_hole12_tee", "label": "Hole 12 tee",
                       "start_frame": 89, "end_frame_exclusive": 556,
                       "body_state": "supported",
                       "clip_sha256": self._sha(clip)}]}), encoding="utf-8")
        record = _parent_source_record(self._sha(clip),
                                       manifest_hint=str(manifest),
                                       clip_path=str(clip))
        self.assertTrue(record["found"])
        self.assertEqual(record["resolved_from"], "shot.parent_source_manifest")
        self.assertEqual(record["manifest"], str(manifest))
        self.assertEqual(record["native_trim_frames"],
                         {"start_frame": 89, "end_frame_exclusive": 556,
                          "start_key": "start_frame",
                          "end_key": "end_frame_exclusive"})
        self.assertEqual(record["baseline_body_state"], "supported")

    def test_a_manifest_beside_the_clip_is_found_and_its_own_spelling_read(self):
        clip = self._clip("09_hole03_tee.mp4", folder="new-clips")
        manifest = self.tmp / "new-clips_manifest.json"
        manifest.write_text(json.dumps({
            "source": "/parent/birdies.mp4", "source_sha256": "3" * 64,
            "clips": [{"id": "09_hole03_tee",
                       "source_start_frame": 15198,
                       "source_end_frame_exclusive": 15464,
                       "golfer_window_end": 99,
                       "clip_sha256": self._sha(clip)}]}), encoding="utf-8")
        record = _parent_source_record(self._sha(clip), clip_path=str(clip))
        self.assertTrue(record["found"])
        self.assertEqual(record["resolved_from"], "beside the clip")
        self.assertEqual(record["manifest"], str(manifest))
        self.assertEqual(record["native_trim_frames"]["start_frame"], 15198)
        self.assertEqual(record["native_trim_frames"]["end_frame_exclusive"], 15464)
        self.assertEqual(record["native_trim_frames"]["start_key"],
                         "source_start_frame")
        self.assertEqual(record["native_trim_frames"]["end_key"],
                         "source_end_frame_exclusive")
        self.assertEqual(record["golfer_window_end"], 99)

    def test_a_manifest_that_does_not_list_the_clip_is_reported_not_claimed(self):
        clip = self._clip()
        manifest = self.tmp / "clips" / "manifest.json"
        manifest.write_text(json.dumps({"source": "/parent.mp4",
                                        "clips": [{"id": "other",
                                                   "clip_sha256": "9" * 64}]}),
                            encoding="utf-8")
        record = _parent_source_record(self._sha(clip), clip_path=str(clip))
        self.assertFalse(record["found"])
        self.assertIn("not among the", record["note"])
        # A parent digest beside a clip the manifest does not list would read as THIS
        # clip's parent source, so nothing about the parent is claimed.
        self.assertNotIn("parent_source_sha256", record)
        self.assertNotIn("parent_source", record)
        self.assertNotIn("native_trim_frames", record)
        self.assertIn("manifest", record)

    def test_a_missing_manifest_is_an_absence_with_a_reason(self):
        clip = self._clip("lonely.mp4", folder="alone")
        record = _parent_source_record(self._sha(clip), clip_path=str(clip))
        self.assertFalse(record["found"])
        self.assertEqual(record["manifest"], "")
        self.assertIn("none was found beside the clip", record["note"])
        self.assertTrue(record["tried"])

    def test_an_unreadable_named_manifest_is_reported(self):
        clip = self._clip()
        record = _parent_source_record(self._sha(clip),
                                       manifest_hint=str(self.tmp / "nope.json"),
                                       clip_path=str(clip))
        self.assertFalse(record["found"])
        self.assertIn("none was found beside the clip", record["note"])

    def test_a_relative_hint_is_resolved_against_the_config_directory(self):
        clip = self._clip("04_hole13_putt.mp4")
        manifest = self.tmp / "configs" / "manifest.json"
        manifest.parent.mkdir()
        manifest.write_text(json.dumps({
            "source": "/parent.mp4",
            "clips": [{"id": "04", "start_frame": 10, "end_frame_exclusive": 300,
                       "clip_sha256": self._sha(clip)}]}), encoding="utf-8")
        record = _parent_source_record(self._sha(clip), manifest_hint="manifest.json",
                                       clip_path=str(clip),
                                       config_dir=self.tmp / "configs")
        self.assertTrue(record["found"])
        self.assertEqual(record["native_trim_frames"]["start_frame"], 10)


# ------------------------------------------------------------ verification

class RenderTimelineVerificationTests(TempDirTestCase):
    """Every decoded frame's timestamp and the duration, not just the first five."""

    def _clip(self, name="source.mp4", *, frames=48, size="160x120") -> Path:
        path = self.tmp / name
        cmd = [FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i",
               f"testsrc=size={size}:rate=30000/1001:duration=10",
               "-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(path)]
        subprocess.run(cmd, capture_output=True, timeout=180, check=True)
        return path

    def test_a_frame_identical_copy_passes_on_every_frame_and_the_duration(self):
        if not (FFMPEG and FFPROBE):
            self.skipTest("ffmpeg/ffprobe are not on PATH")
        source = self._clip()
        identical = self.tmp / "identical.mp4"
        subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(source), "-c", "copy",
                        "-movflags", "+faststart", str(identical)],
                       capture_output=True, timeout=180, check=True)
        result = verify_broadcast_render(
            str(identical), expected_frames=48, expected_rational="30000/1001",
            source_path=str(source), source_audio={"present": False})
        self.assertTrue(result["ok"], result["problems"])
        self.assertTrue(result["timestamps"]["checked"])
        self.assertEqual(result["timestamps"]["compared_frames"], 48)
        self.assertEqual(result["timestamps"]["max_delta_seconds"], 0.0)
        self.assertEqual(result["timestamps"]["frames_within_half_a_frame"], 48)
        self.assertTrue(result["timestamps"]["output_strictly_increasing"])
        self.assertTrue(result["duration"]["checked"])
        self.assertAlmostEqual(result["duration"]["measured_seconds"],
                               result["duration"]["expected_seconds_from_frames_and_cadence"],
                               places=3)

    def test_a_tail_truncation_is_caught_even_though_the_first_frames_agree(self):
        """The regression: a five-frame prefix check passes a truncated output."""
        if not (FFMPEG and FFPROBE):
            self.skipTest("ffmpeg/ffprobe are not on PATH")
        source = self._clip()
        short = self.tmp / "short.mp4"
        subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(source),
                        "-frames:v", "30", "-c", "copy",
                        "-movflags", "+faststart", str(short)],
                       capture_output=True, timeout=180, check=True)
        result = verify_broadcast_render(
            str(short), expected_frames=48, expected_rational="30000/1001",
            source_path=str(source), source_audio={"present": False})
        self.assertFalse(result["ok"])
        joined = " ".join(result["problems"])
        self.assertIn("the output holds 30 frames, the source has 48", joined)
        self.assertIn("different number of timestamped frames", joined)
        self.assertIn("frames at 30000/1001", joined)
        self.assertLessEqual(result["timestamps"]["max_delta_seconds"],
                             result["timestamps"]["half_frame_interval"],
                             "the first frames agree, which is why a prefix check "
                             "would have passed this output")

    def test_the_tolerance_separates_mux_rounding_from_a_missing_frame(self):
        if not (FFMPEG and FFPROBE):
            self.skipTest("ffmpeg/ffprobe are not on PATH")
        source = self._clip(frames=1)
        result = verify_broadcast_render(
            str(source), expected_frames=1, expected_rational="30000/1001",
            source_path=str(source), source_audio={"present": False})
        self.assertTrue(result["duration"]["checked"])
        self.assertLess(result["duration"]["tolerance_seconds"], 0.0334,
                        "half a frame must be the bound, so a single dropped frame is "
                        "never inside the tolerance")


# ------------------------------------------------------------ contract round-trip

class BroadcastConfigRoundTripTests(unittest.TestCase):
    def test_a_config_with_manual_segments_survives_its_own_to_dict(self):
        """A run's config copy must be re-runnable, derived keys and all."""
        scene = base_scene(
            roi_keyframes=[{"frame": 10, "roi": [40, 20, 80, 90], "note": "crop"}],
            corrections={"clubhead": [{"frame": 14, "x": 1.0, "y": 2.0}]},
            manual_segments={"clubhead": {
                "author": "primary", "replace_automatic": True,
                "hide_frames": [], "trail": "segments_only",
                "trail_from_frame": 12, "note": "block note",
                "segments": [{"name": "downswing",
                              "max_anchor_spacing_frames": 12,
                              "anchors": [{"frame": 12, "x": 1.5, "y": 2.5},
                                          {"frame": 16, "x": 3.5, "y": 4.5}],
                              "note": "segment note"}]}},
            interpolation={"enabled": True, "max_gap_frames": 2},
            metric_hide_frames=[[30, 36]])
        payload = base_config(
            scenes=[scene],
            cuts=[{"frame": 41, "kind": "hard_cut", "confidence": 1.0,
                   "decision": "accepted"}],
            presentation={"brand": "F", "brand_sub": "S", "disclosure": "D",
                          "metrics": ["shoulder_tilt", "hip_tilt", "tempo"],
                          "card_position": "top"},
            shot={"id": "s", "label": "l", "stroke_type": "full_swing",
                  "source_notes": "operator words",
                  "parent_source_manifest": "manifest.json",
                  "expected_fps": 29.97, "expected_frames": 120})
        first = bc.parse_broadcast(payload)
        rendered = first.to_dict()
        second = bc.parse_broadcast(rendered)
        self.assertEqual(json.dumps(rendered, sort_keys=True),
                         json.dumps(second.to_dict(), sort_keys=True))
        scene = rendered["scenes"][0]
        self.assertEqual(scene["manual_segments"], payload["scenes"][0]["manual_segments"])
        self.assertEqual(scene["roi_keyframes"], payload["scenes"][0]["roi_keyframes"])
        self.assertEqual(rendered["shot"]["source_notes"], "operator words")
        self.assertEqual(rendered["shot"]["parent_source_manifest"], "manifest.json")
        self.assertEqual(rendered["shot"]["expected_fps"], 29.97)
        self.assertEqual(rendered["shot"]["expected_frames"], 120)
        # No derived key may appear where the parser reads operator input.
        for name, block in scene["manual_segments"].items():
            self.assertNotIn("segment_count", block)
            self.assertNotIn("rule", block)
            for segment in block["segments"]:
                self.assertNotIn("start_frame", segment)
                self.assertNotIn("anchor_count", segment)
                self.assertNotIn("interpolated_rule", segment)

    def test_manual_segments_written_by_code_still_load(self):
        """A scene built in code (not parsed) must serialize to something loadable."""
        from ghostcaddie.demo.config import ManualAnchor, ManualAnchorSegment, ManualSegments

        config = bc.parse_broadcast(base_config())
        scene = config.scenes[0]
        scene.demo.manual_segments["clubhead"] = ManualSegments(
            author="code", replace_automatic=True, trail="all",
            segments=[ManualAnchorSegment(
                name="s", max_anchor_spacing_frames=12,
                anchors=[ManualAnchor(frame=12, x=1.0, y=1.0),
                         ManualAnchor(frame=16, x=2.0, y=2.0)])])
        payload = config.to_dict()
        again = bc.parse_broadcast(payload)
        self.assertEqual(again.scenes[0].demo.manual_segments["clubhead"].segments[0]
                         .frames(), [12, 16])

    def test_a_malformed_parent_source_manifest_is_refused(self):
        payload = base_config()
        payload["shot"]["parent_source_manifest"] = 7
        with self.assertRaises(DemoConfigError) as caught:
            bc.parse_broadcast(payload)
        self.assertIn("shot.parent_source_manifest", str(caught.exception))
        payload = base_config()
        payload["shot"]["expected_fps"] = 0
        with self.assertRaises(DemoConfigError) as caught:
            bc.parse_broadcast(payload)
        self.assertIn("shot.expected_fps", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
