"""Focused tests for the broadcast showcase workflow.

Groups:

* **contract** — the config schema and every refusal that keeps a scene honest
  (overlap, out-of-window frames, putt vocabulary, non-stroke scenes, anchor spacing,
  accepted-cut crossing, `to_dict` being reloadable).
* **discovery** — the cut/dissolve detector, on synthetic scores and on a real
  generated clip.
* **batch** — every manifest entry appears exactly once and a failure does not stop
  the others.
* **ui** — the loopback guards (token, Host, body bounds, path confinement) and the
  shared coordinate-mapping vectors.
* **end to end** — a real broadcast render on a generated clip: output frame count,
  exact rational cadence, audio present and copied.

No source footage, no research media, no model, no network.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ghostcaddie.demo import broadcast_config as bc
from ghostcaddie.demo.batch import load_manifest, run_batch
from ghostcaddie.demo.broadcast import (PROTECTED_OUTPUT_PREFIXES, guard_output_dir,
                                        verify_broadcast_render)
from ghostcaddie.demo.broadcast_ui import viewport_to_native
from ghostcaddie.demo.discovery import (CutCandidate, SceneScores, detect_cuts,
                                        merge_decisions)
from ghostcaddie.demo.errors import DemoConfigError
from ghostcaddie.demo.pipeline import verify_render
from tests.demo_fixtures import TempDirTestCase, synthetic_probe

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def require_ffmpeg(case: unittest.TestCase) -> None:
    if not (FFMPEG and FFPROBE):
        case.skipTest("ffmpeg/ffprobe are not on PATH")


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


# ---------------------------------------------------------------- contract

class BroadcastTempoTests(unittest.TestCase):
    """Tempo must be the stroke's own numbers, with known frames.

    Pinned because the first implementation got both branches wrong: a putt used
    ``contact - backstroke`` (which counts the whole stroke, not the backswing), and
    the full-swing branch returned "unavailable" while the pilot published 2.0.
    """

    def setUp(self):
        try:
            from ghostcaddie.demo import worker
        except Exception:  # pragma: no cover - bare interpreter
            self.skipTest("this interpreter has no cv2")
        self.worker = worker

    def test_a_putt_tempo_is_forward_minus_backstroke_over_contact_minus_forward(self):
        tempo = self.worker._broadcast_tempo(
            "putt", ("address", "backstroke", "forward_stroke", "contact", "finish"),
            {"address": 10, "backstroke": 10, "forward_stroke": 30, "contact": 40,
             "finish": 50})
        self.assertEqual(tempo["state"], "supported")
        self.assertEqual(tempo["backswing_frames"], 20)
        self.assertEqual(tempo["downswing_frames"], 10)
        self.assertEqual(tempo["backswing_downswing_ratio"], 2.0)

    def test_a_putt_without_all_three_marks_is_unavailable_not_estimated(self):
        for missing in ("backstroke", "forward_stroke", "contact"):
            with self.subTest(missing=missing):
                marks = {"address": 10, "backstroke": 10, "forward_stroke": 30,
                         "contact": 40}
                marks.pop(missing)
                tempo = self.worker._broadcast_tempo("putt", tuple(marks), marks)
                self.assertEqual(tempo["state"], "unavailable")

    def test_a_full_swing_publishes_the_canonical_tempo_record(self):
        canonical = {"state": "supported", "backswing_frames": 14,
                     "downswing_frames": 7, "backswing_downswing_ratio": 2.0,
                     "label": "TEMPO"}
        tempo = self.worker._broadcast_tempo(
            "full_swing",
            ("address", "takeaway", "top", "downswing", "impact", "follow_through",
             "finish"),
            {"address": 0, "takeaway": 20, "top": 34, "downswing": 41, "impact": 48,
             "finish": 90},
            canonical=canonical)
        self.assertEqual(tempo, canonical)

    def test_a_full_swing_with_no_canonical_record_says_so(self):
        tempo = self.worker._broadcast_tempo("full_swing", ("address",), {})
        self.assertEqual(tempo["state"], "unavailable")
        self.assertIn("phase resolver", tempo["reason"])


class CardPositionTests(unittest.TestCase):
    """The card row must not cover the golfer, and the choice must be per scene."""

    def setUp(self):
        try:
            from ghostcaddie.demo import worker
        except Exception:  # pragma: no cover - bare interpreter
            self.skipTest("this interpreter has no cv2")
        self.worker = worker

    def test_auto_moves_the_row_to_the_top_when_content_reaches_the_bottom_band(self):
        from ghostcaddie.demo import showcase as sc

        layout = sc.plan_layout(1920, 1092, ("shoulder_tilt", "hip_tilt"))
        inside_band = float(layout.strip_top + 20)
        body = [{"available": True,
                 "keypoints": [{"name": "ank_l", "y": inside_band, "visible": True,
                                "x": 1.0}]}]
        position, why = self.worker._broadcast_card_position(
            "auto", body=body, tracks={}, scale=1.0, out_w=1920, out_h=1092,
            metrics=("shoulder_tilt", "hip_tilt"), max_row_width=None)
        self.assertEqual(position, "top")
        self.assertIn("card band", why)

    def test_auto_keeps_the_row_at_the_bottom_when_the_golfer_is_clear(self):
        body = [{"available": True,
                 "keypoints": [{"name": "ank_l", "y": 100.0, "visible": True, "x": 1.0}]}]
        position, _why = self.worker._broadcast_card_position(
            "auto", body=body, tracks={}, scale=1.0, out_w=1920, out_h=1092,
            metrics=("shoulder_tilt",), max_row_width=None)
        self.assertEqual(position, "bottom")

    def test_an_operator_choice_is_never_overridden(self):
        position, why = self.worker._broadcast_card_position(
            "top", body=[{"available": True,
                          "keypoints": [{"name": "ank_l", "y": 10.0, "visible": True,
                                         "x": 1.0}]}],
            tracks={}, scale=1.0, out_w=1920, out_h=1092, metrics=("hip_tilt",),
            max_row_width=None)
        self.assertEqual(position, "top")
        self.assertIn("operator", why)

    def test_the_top_row_stays_inside_the_picture(self):
        from ghostcaddie.demo import showcase as sc

        layout = sc.plan_layout(1920, 1092, ("shoulder_tilt", "hip_tilt"),
                                card_position="top")
        self.assertEqual(sc.layout_problems(layout), [])
        for rect in layout.cards:
            self.assertGreaterEqual(rect.y, 0)
            self.assertLessEqual(rect.y + rect.h, 1092)

    def test_the_default_position_is_unchanged(self):
        """Behaviour-neutral: the existing showcase layout must not move."""
        from ghostcaddie.demo import showcase as sc

        default = sc.plan_layout(1920, 1092, ("shoulder_tilt", "hip_tilt"))
        explicit = sc.plan_layout(1920, 1092, ("shoulder_tilt", "hip_tilt"),
                                  card_position="bottom")
        self.assertEqual(default.cards, explicit.cards)
        self.assertEqual(default.to_dict(), explicit.to_dict())


class BroadcastContractTests(TempDirTestCase):
    def test_the_shipped_templates_parse_as_they_ship(self):
        for name in ("broadcast_config.template.json",
                     "broadcast_config_putt.template.json"):
            with self.subTest(template=name):
                path = REPO / "examples" / name
                config = bc.load_broadcast(str(path))
                self.assertTrue(config.scenes)
                self.assertTrue(config.presentation.disclosure)

    def test_a_config_written_by_the_run_loads_again(self):
        """The run's own ``broadcast_config.json`` must be re-runnable.

        This is the regression test for a real defect: the first version emitted
        derived keys (``phase_vocabulary``, ``contact_frame``, ``window.frame_count``)
        that the parser refuses, so the copy the run wrote could not be passed back.
        """
        first = bc.parse_broadcast(base_config())
        payload = first.to_dict()
        second = bc.parse_broadcast(payload)
        self.assertEqual(json.dumps(payload, sort_keys=True),
                         json.dumps(second.to_dict(), sort_keys=True))

    def test_unknown_keys_are_refused_not_ignored(self):
        payload = base_config()
        payload["scenes"][0]["phse"] = {"address": 10}
        with self.assertRaises(DemoConfigError) as caught:
            bc.parse_broadcast(payload)
        self.assertIn("unknown field", str(caught.exception))

    def test_overlapping_windows_are_refused(self):
        quiet = dict(seeds={}, targets={}, roi=None)
        payload = base_config(scenes=[
            base_scene(**quiet),
            base_scene(id="s2", window={"start_frame": 30, "end_frame": 60},
                       keyframes={"address": 30, "finish": 58}, **quiet)])
        with self.assertRaises(DemoConfigError) as caught:
            bc.parse_broadcast(payload)
        self.assertIn("overlap", str(caught.exception))

    def test_out_of_order_scenes_are_refused_rather_than_sorted(self):
        quiet = dict(seeds={}, targets={}, roi=None)
        payload = base_config(scenes=[
            base_scene(id="late", window={"start_frame": 60, "end_frame": 90},
                       keyframes={"address": 60, "finish": 88}, **quiet),
            base_scene(**quiet)])
        with self.assertRaises(DemoConfigError) as caught:
            bc.parse_broadcast(payload)
        self.assertIn("ascending", str(caught.exception))

    def test_a_scene_frame_outside_its_window_is_refused(self):
        payload = base_config()
        payload["scenes"][0]["seeds"]["ball"]["frame"] = 400
        with self.assertRaises(DemoConfigError) as caught:
            bc.parse_broadcast(payload)
        self.assertIn("outside the scene window", str(caught.exception))

    def test_a_putt_scene_refuses_a_full_swing_phase(self):
        payload = base_config(scenes=[base_scene(
            stroke_type="putt",
            keyframes={"address": 10, "backstroke": 20, "top": 25})])
        with self.assertRaises(DemoConfigError) as caught:
            bc.parse_broadcast(payload)
        self.assertIn("not a phase of a 'putt' scene", str(caught.exception))

    def test_a_putt_scene_maps_its_marks_and_disables_approximation(self):
        config = bc.parse_broadcast(base_config(scenes=[base_scene(
            stroke_type="putt",
            keyframes={"address": 10, "backstroke": 20, "forward_stroke": 26,
                       "contact": 30, "finish": 38},
            allow_approx_phases=True,
            targets={}, seeds={})]))
        scene = config.scenes[0]
        self.assertEqual(scene.contact_frame, 30)
        self.assertEqual(scene.canonical_keyframes["impact"], 30)
        self.assertEqual(scene.canonical_keyframes["takeaway"], 20)
        self.assertFalse(scene.demo.allow_approx_phases,
                         "a putt must not inherit the minimum-wrist TOP derivation")

    def test_a_non_stroke_window_may_not_carry_tracking_inputs(self):
        payload = base_config(scenes=[
            base_scene(),
            base_scene(id="reaction", kind="reaction",
                       window={"start_frame": 60, "end_frame": 90},
                       keyframes={}, roi=[1, 2, 3, 4], seeds={}, targets={})])
        with self.assertRaises(DemoConfigError) as caught:
            bc.parse_broadcast(payload)
        self.assertIn("no golfer to track", str(caught.exception))

    def test_anchor_spacing_is_capped_harder_near_contact(self):
        payload = base_config()
        payload["scenes"][0]["manual_segments"] = {"clubhead": {
            "author": "operator", "replace_automatic": True,
            "segments": [{"name": "downswing", "max_anchor_spacing_frames": 12,
                          "anchors": [[28, 10.0, 10.0], [36, 20.0, 20.0]]}]}}
        with self.assertRaises(DemoConfigError) as caught:
            bc.parse_broadcast(payload)
        self.assertIn("broadcast cap", str(caught.exception))

    def test_anchor_spacing_far_from_contact_allows_the_wider_cap(self):
        config = bc.parse_broadcast(base_config())
        self.assertEqual(config.spacing_records, [])

    def test_a_dissolve_needs_a_span_and_a_hard_cut_refuses_one(self):
        payload = base_config(cuts=[{"frame": 200, "kind": "dissolve",
                                     "confidence": 0.5}])
        with self.assertRaises(DemoConfigError) as caught:
            bc.parse_broadcast(payload)
        self.assertIn("no span", str(caught.exception))
        payload = base_config(cuts=[{"frame": 200, "kind": "hard_cut",
                                     "span": [199, 201]}])
        with self.assertRaises(DemoConfigError) as caught:
            bc.parse_broadcast(payload)
        self.assertIn("hard cut with a span", str(caught.exception))

    def test_an_empty_scene_list_is_refused(self):
        with self.assertRaises(DemoConfigError):
            bc.parse_broadcast(base_config(scenes=[]))

    def test_outside_windows_is_a_closed_set(self):
        with self.assertRaises(DemoConfigError):
            bc.parse_broadcast(base_config(outside_windows="nothing"))


class BroadcastBindingTests(unittest.TestCase):
    def test_windows_are_integer_frames_and_stay_integral(self):
        """A window is never round-tripped through seconds.

        At 9832/331 fps, ``floor(seconds*fps)`` is off by one for many frames, so the
        contract carries integers and the binder must not reinterpret them.
        """
        config = bc.parse_broadcast(base_config())
        probe = synthetic_probe(width=640, height=480, fps=9832 / 331,
                                frame_count=400)
        resolved = bc.validate_broadcast(config, probe)
        self.assertEqual(resolved.start_frame, 0)
        self.assertEqual(resolved.end_frame, 399)
        self.assertEqual(resolved.scenes[0].start_frame, 10)
        self.assertEqual(resolved.scenes[0].end_frame, 40)

    def test_a_window_crossing_an_accepted_cut_is_refused(self):
        payload = base_config(
            cuts=[{"frame": 25, "kind": "hard_cut", "confidence": 0.9,
                   "decision": "accepted"}])
        config = bc.parse_broadcast(payload)
        probe = synthetic_probe(frame_count=400)
        with self.assertRaises(DemoConfigError) as caught:
            bc.validate_broadcast(config, probe)
        self.assertIn("crosses the accepted cut", str(caught.exception))

    def test_a_proposed_cut_does_not_bind_anything(self):
        payload = base_config(
            cuts=[{"frame": 25, "kind": "hard_cut", "confidence": 0.9,
                   "decision": "proposed"}])
        resolved = bc.validate_broadcast(bc.parse_broadcast(payload),
                                         synthetic_probe(frame_count=400))
        self.assertEqual(len(resolved.scenes), 1)

    def test_a_window_past_the_last_frame_is_refused(self):
        config = bc.parse_broadcast(base_config())
        with self.assertRaises(DemoConfigError):
            bc.validate_broadcast(config, synthetic_probe(frame_count=30))


# ---------------------------------------------------------------- discovery

class DiscoveryTests(unittest.TestCase):
    def scores(self, values, start=0):
        return SceneScores(frames=list(range(start, start + len(values))),
                           score=[float(v) for v in values],
                           mafd=[float(v) for v in values])

    def test_a_single_large_spike_is_a_hard_cut(self):
        values = [0.01] * 40
        values[20] = 12.0
        found = detect_cuts(self.scores(values))
        cuts = [c for c in found if c.kind == "hard_cut"]
        self.assertEqual([c.frame for c in cuts], [20])
        self.assertGreater(cuts[0].confidence, 0.5)

    def test_sustained_motion_alone_is_not_a_cut(self):
        """A fast swing must not be reported as a camera cut: the score stays high
        for several frames, which is motion, not a new camera.

        Neither the leading nor the trailing edge of the motion qualifies, because a
        cut has a quiet frame on *both* sides.
        """
        values = [0.01] * 10 + [3.0] * 12 + [0.01] * 10
        cuts = [c for c in detect_cuts(self.scores(values)) if c.kind == "hard_cut"]
        self.assertEqual(cuts, [])

    def test_a_run_of_modest_change_is_a_dissolve_with_a_span(self):
        values = [0.005] * 20 + [0.09] * 8 + [0.005] * 20
        dissolves = [c for c in detect_cuts(self.scores(values)) if c.kind == "dissolve"]
        self.assertEqual(len(dissolves), 1)
        self.assertEqual(dissolves[0].span, [20, 27])

    def test_candidates_are_deterministic(self):
        values = [0.01] * 30
        values[7] = 20.0
        first = [c.to_dict() for c in detect_cuts(self.scores(values))]
        second = [c.to_dict() for c in detect_cuts(self.scores(values))]
        self.assertEqual(first, second)

    def test_a_saved_decision_survives_a_rescan(self):
        values = [0.01] * 30
        values[12] = 15.0
        found = detect_cuts(self.scores(values))
        merged = merge_decisions(found, [{"frame": 12, "decision": "reject",
                                          "decided_by": "PRIMARY"}])
        self.assertEqual(merged[0].decision, "reject")
        self.assertEqual(merged[0].decided_by, "PRIMARY")

    def test_a_new_detection_near_a_saved_decision_inherits_it(self):
        merged = merge_decisions(
            [CutCandidate(frame=13, kind="hard_cut", confidence=0.5)],
            [{"frame": 12, "decision": "accept", "decided_by": "operator"}])
        self.assertEqual(merged[0].decision, "accept")


class DiscoveryOnRealClipTests(TempDirTestCase):
    def _cut_clip(self, path: Path) -> Path:
        cmd = [FFMPEG, "-y", "-v", "error",
               "-f", "lavfi", "-i", "color=c=red:s=160x120:d=1:r=25",
               "-f", "lavfi", "-i", "color=c=blue:s=160x120:d=1:r=25",
               "-filter_complex", "[0][1]concat=n=2:v=1:a=0",
               "-pix_fmt", "yuv420p", str(path)]
        subprocess.run(cmd, capture_output=True, timeout=180, check=True)
        return path

    def test_a_real_hard_cut_is_found_at_the_right_frame(self):
        require_ffmpeg(self)
        from ghostcaddie.demo.discovery import discover

        clip = self._cut_clip(self.tmp / "cut.mp4")
        record = discover(str(clip))
        frames = [c["frame"] for c in record["hard_cuts"]]
        self.assertTrue(any(abs(frame - 25) <= 1 for frame in frames),
                        f"the cut at frame 25 was not proposed: {frames}")

    def test_discovery_is_repeatable_on_the_same_clip(self):
        require_ffmpeg(self)
        from ghostcaddie.demo.discovery import discover

        clip = self._cut_clip(self.tmp / "cut.mp4")
        first = discover(str(clip))["candidates"]
        second = discover(str(clip))["candidates"]
        self.assertEqual(first, second)


# ---------------------------------------------------------------- output guard

class OutputGuardTests(TempDirTestCase):
    def test_the_preserved_trees_are_refused(self):
        for protected in PROTECTED_OUTPUT_PREFIXES:
            with self.subTest(protected=protected):
                with self.assertRaises(DemoConfigError):
                    guard_output_dir(protected + "/nope")

    def test_a_normal_directory_is_allowed(self):
        target = self.tmp / "render"
        self.assertEqual(guard_output_dir(str(target)), target)

    def test_a_finished_render_is_not_overwritten_without_force(self):
        target = self.tmp / "render"
        target.mkdir(parents=True)
        (target / "broadcast_annotated.mp4").write_bytes(b"x")
        with self.assertRaises(DemoConfigError) as caught:
            guard_output_dir(str(target))
        self.assertIn("already exists", str(caught.exception))
        self.assertEqual(guard_output_dir(str(target), force=True), target)


# ---------------------------------------------------------------- batch

class BatchTests(TempDirTestCase):
    def test_every_entry_appears_exactly_once_and_failures_are_reported(self):
        good = self.tmp / "good.mp4"
        good.write_bytes(b"\x00")
        for name in ("a.json", "b.json", "c.json"):
            (self.tmp / name).write_text("{}", encoding="utf-8")
        manifest = self.tmp / "manifest.json"
        manifest.write_text(json.dumps({"entries": [
            {"id": "a_good", "clip": str(good), "config": str(self.tmp / "a.json")},
            {"id": "b_corrupt", "clip": str(self.tmp / "missing.mp4"),
             "config": str(self.tmp / "b.json")},
            {"id": "c_sha", "clip": str(good), "config": str(self.tmp / "c.json"),
             "clip_sha256": "0" * 64},
        ]}), encoding="utf-8")

        def runner(cmd, timeout, cwd):
            return 0, "fine"

        record = run_batch(str(manifest), str(self.tmp / "out"),
                           command_runner=runner)
        self.assertEqual(len(record["results"]), 3)
        statuses = {item["id"]: item["status"] for item in record["results"]}
        self.assertEqual(statuses["a_good"], "succeeded")
        self.assertEqual(statuses["b_corrupt"], "refused")
        self.assertEqual(statuses["c_sha"], "refused")
        self.assertIn("does not exist", record["results"][1]["reason"])
        self.assertIn("sha256", record["results"][2]["reason"])
        report = (self.tmp / "out" / "batch_report.md").read_text(encoding="utf-8")
        for entry_id in ("a_good", "b_corrupt", "c_sha"):
            self.assertIn(entry_id, report)
        self.assertEqual(record["counts"]["succeeded"], 1)

    def test_a_failing_render_is_a_failure_with_its_reason(self):
        good = self.tmp / "good.mp4"
        good.write_bytes(b"\x00")
        (self.tmp / "a.json").write_text("{}", encoding="utf-8")
        manifest = self.tmp / "manifest.json"
        manifest.write_text(json.dumps([
            {"id": "boom", "clip": str(good), "config": str(self.tmp / "a.json")}]),
            encoding="utf-8")
        record = run_batch(str(manifest), str(self.tmp / "out"),
                           command_runner=lambda cmd, t, cwd: (1, "ffmpeg said no"))
        self.assertEqual(record["counts"]["failed"], 1)
        self.assertEqual(record["results"][0]["exit_code"], 1)

    def test_the_baseline_manifest_shape_is_accepted(self):
        manifest = self.tmp / "baseline.json"
        manifest.write_text(json.dumps({"clips": [
            {"id": "01x", "clip": "/tmp/clip.mp4"}]}), encoding="utf-8")
        entries = load_manifest(str(manifest))
        self.assertEqual(entries[0]["id"], "01x")

    def test_an_entry_without_a_config_is_refused_not_guessed(self):
        clip = self.tmp / "c.mp4"
        clip.write_bytes(b"\x00")
        manifest = self.tmp / "m.json"
        manifest.write_text(json.dumps([{"id": "x", "clip": str(clip)}]),
                            encoding="utf-8")
        record = run_batch(str(manifest), str(self.tmp / "out"),
                           command_runner=lambda c, t, w: (0, ""))
        self.assertEqual(record["results"][0]["status"], "refused")
        self.assertIn("no broadcast config", record["results"][0]["reason"])


# ---------------------------------------------------------------- ui guards

class ViewportMappingTests(unittest.TestCase):
    def test_the_shared_vectors_agree_with_the_python_reference(self):
        vectors = json.loads(
            (REPO / "tests" / "fixtures" / "viewport_vectors.json").read_text())
        for vector in vectors["vectors"]:
            with self.subTest(vector=vector["name"]):
                got = viewport_to_native(vector["rect"], vector["natural"],
                                         *vector["click"])
                if vector["expect"] is None:
                    self.assertIsNone(got)
                else:
                    self.assertIsNotNone(got)
                    self.assertEqual(got["x"], vector["expect"]["x"])
                    self.assertEqual(got["y"], vector["expect"]["y"])

    def test_the_page_runs_the_same_mapping(self):
        """The JS mirror is checked for the same two decisions: fit-and-centre, and
        refuse a click outside the picture."""
        from ghostcaddie.demo.broadcast_ui import PAGE

        self.assertIn("function viewportToNative", PAGE)
        self.assertIn("Math.min(rect.width", PAGE)
        self.assertIn("if (x < 0 || y < 0 || x > natural.w || y > natural.h)",
                      PAGE)


class UiServerTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        from ghostcaddie.demo.broadcast_ui import Registry, make_handler

        self.clip = self.tmp / "clip.mp4"
        self.clip.write_bytes(b"\x00")
        self.registry = Registry([self.tmp])
        handler = make_handler(self.registry, "tok123", self.tmp / "configs",
                               {"python": sys.executable, "out_root": str(self.tmp)})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        super().tearDown()

    def call(self, path, *, method="GET", body=None, token=None, host=None):
        url = f"http://127.0.0.1:{self.port}{path}"
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if token is not None:
            req.add_header("X-Broadcast-Token", token)
        if host is not None:
            req.add_header("Host", host)
        try:
            with urllib.request.urlopen(req) as response:
                return response.status, json.loads(response.read().decode() or "{}")
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode()
            try:
                return exc.code, json.loads(payload or "{}")
            except json.JSONDecodeError:
                return exc.code, {"raw": payload}

    def test_a_mutation_without_the_token_is_refused(self):
        status, payload = self.call("/api/validate", method="POST",
                                    body={"clip": "clip01", "config": {}})
        self.assertEqual(status, 403)
        self.assertIn("Token", payload["error"])

    def test_a_mutation_with_the_wrong_token_is_refused(self):
        status, _ = self.call("/api/validate", method="POST",
                              body={"clip": "clip01", "config": {}}, token="wrong")
        self.assertEqual(status, 403)

    def test_a_non_loopback_host_is_refused(self):
        status, payload = self.call("/api/clips", host="evil.example")
        self.assertEqual(status, 403)
        self.assertIn("loopback", payload["error"])

    def test_an_unknown_clip_id_is_refused(self):
        status, payload = self.call("/api/validate", method="POST",
                                    body={"clip": "../etc/passwd", "config": {}},
                                    token="tok123")
        self.assertEqual(status, 400)
        self.assertIn("unknown clip id", payload["error"])

    def test_a_non_numeric_frame_is_refused(self):
        status, payload = self.call("/api/frame?clip=clip01&frame=../../etc")
        self.assertEqual(status, 400)
        self.assertIn("whole number", payload["error"])

    def test_an_oversized_body_is_refused(self):
        big = {"clip": "clip01", "config": {"x": "y" * (70 * 1024)}}
        status, payload = self.call("/api/validate", method="POST", body=big,
                                    token="tok123")
        self.assertEqual(status, 400)
        self.assertIn("at most", payload["error"])

    def test_an_invalid_config_is_not_written(self):
        bad = {"schema_version": bc.BROADCAST_CONFIG_SCHEMA, "scenes": []}
        status, payload = self.call("/api/save", method="POST",
                                    body={"name": "bad.json", "config": bad},
                                    token="tok123")
        self.assertEqual(status, 400)
        self.assertFalse((self.tmp / "configs" / "bad.json").exists())

    def test_an_artifact_name_cannot_escape_the_job_directory(self):
        self.registry.jobs["job1"] = {"id": "job1", "out_dir": str(self.tmp / "out")}
        status, payload = self.call("/api/artifact?id=job1&name=../../../etc/passwd")
        self.assertEqual(status, 400)
        self.assertIn("escapes", payload["error"])


# ---------------------------------------------------------------- cli

class BroadcastCliIsolationTests(unittest.TestCase):
    def test_help_does_not_import_cv2_or_research_modules(self):
        code = (
            "import sys\n"
            "from ghostcaddie.cli import main\n"
            "try:\n"
            "    main(['broadcast-showcase', '--help'])\n"
            "except SystemExit as exc:\n"
            "    assert exc.code == 0\n"
            "for name in ('cv2', 'numpy', 'torch', 'ghostcaddie.video.ai_demo'):\n"
            "    assert name not in sys.modules, name\n")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                text=True, timeout=60, cwd=str(REPO))
        self.assertEqual(result.returncode, 0, result.stderr)


# ---------------------------------------------------------------- end to end

class BroadcastEndToEndTests(TempDirTestCase):
    def _clip(self, name="clip.mp4", *, seconds=2.0, size="160x120") -> Path:
        path = self.tmp / name
        cmd = [FFMPEG, "-y", "-v", "error",
               "-f", "lavfi", "-i",
               f"testsrc=size={size}:rate=30000/1001:duration={seconds}",
               "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
               "-shortest", "-movflags", "+faststart", str(path)]
        subprocess.run(cmd, capture_output=True, timeout=180, check=True)
        return path

    def test_a_real_render_keeps_frames_cadence_and_audio(self):
        require_ffmpeg(self)
        from ghostcaddie.demo.broadcast import run_broadcast

        clip = self._clip()
        config = base_config(
            scenes=[base_scene(window={"start_frame": 5, "end_frame": 40},
                               keyframes={"address": 6, "top": 20, "impact": 28,
                                          "finish": 38},
                               roi=[20, 10, 60, 80], targets={}, seeds={})],
            render={"max_width": 160, "crf": 30, "preset": "ultrafast",
                    "render": True})
        config_path = self.tmp / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        out = self.tmp / "out"
        result = run_broadcast(str(clip), str(config_path), str(out),
                               cv_python=sys.executable, guard_outputs=True)
        from ghostcaddie.demo.probe import probe_source

        probe = probe_source(str(clip))
        self.assertEqual(result.render["frames"], probe.frame_count)
        self.assertTrue((out / "broadcast_annotated.mp4").is_file())
        check = verify_render(str(out / "broadcast_annotated.mp4"),
                              expected_frames=probe.frame_count,
                              expected_fps=probe.fps,
                              expected_wh=result.render["wh"])
        self.assertTrue(check["ok"], check["problems"])
        audio = verify_broadcast_render(
            str(out / "broadcast_annotated.mp4"),
            expected_frames=probe.frame_count,
            expected_rational=result.source["fps_rational"],
            source_path=str(clip),
            source_audio=result.source["audio"])
        self.assertTrue(audio["ok"], audio["problems"])
        self.assertTrue(audio["audio"]["bit_exact"])
        # A window of 36 frames in a 60-frame clip leaves the rest bare.
        self.assertGreater(result.render["empty_frames"], 0)
        # The window's own phases produced stills, named per scene.
        stills = (out / "keyframes")
        self.assertTrue(any(p.name.startswith("scene-s1_") for p in stills.iterdir()))

    def test_a_render_outside_the_windows_is_the_source_picture(self):
        """On a frame in no window the output must be the source frame itself."""
        require_ffmpeg(self)
        import numpy as np

        try:
            import cv2
        except Exception:
            self.skipTest("no cv2 in this interpreter")
        from ghostcaddie.demo.broadcast import run_broadcast

        clip = self._clip(seconds=2.0)
        config = base_config(
            scenes=[base_scene(window={"start_frame": 4, "end_frame": 10},
                               keyframes={"address": 5, "finish": 9},
                               roi=[10, 10, 80, 90], targets={}, seeds={})],
            render={"max_width": 160, "crf": 18, "preset": "ultrafast",
                    "render": True})
        config_path = self.tmp / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        out = self.tmp / "out"
        run_broadcast(str(clip), str(config_path), str(out),
                      cv_python=sys.executable)

        def frame_at(path, index, width, height):
            cap = cv2.VideoCapture(str(path))
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                self.assertTrue(ok)
                return cv2.resize(frame, (width, height))
            finally:
                cap.release()

        src = frame_at(clip, 45, 160, 120)
        dst = frame_at(out / "broadcast_annotated.mp4", 45, 160, 120)
        delta = np.abs(src.astype(int) - dst.astype(int))
        # Only the tiny persistent brand mark may differ on a bare frame; the
        # overlay layer must be empty, so the mean difference stays very small.
        self.assertLess(float(delta.mean()), 6.0,
                        "a frame in no window is not the source picture")


if __name__ == "__main__":
    unittest.main()
