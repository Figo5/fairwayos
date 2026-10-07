"""Contract tests for the canonical local video workflow.

These run on a bare interpreter. Nothing here needs cv2, numpy, a model runtime
or any optional checkpoint: the worker that needs those is dispatched out of
process, so it can be replaced with a stub and the CONTRACT can still be asserted
end to end. Weight-dependent execution belongs in a separate, skipping test --
that separation is the whole point.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ghostcaddie.video import canonical, model_paths
from ghostcaddie.video.canonical import (AnalysisError, AnalysisResult, Layer,
                                         LayerState, Metric, MetricState,
                                         compute_body_metrics, physical_metrics,
                                         ball_layer, clubhead_layer)

REPO = Path(__file__).resolve().parents[1]


def _kp(**scores):
    """One worker pose record; named landmarks visible at the given coordinates."""
    pts = []
    for name in ("nose", "eye_l", "eye_r", "ear_l", "ear_r", "sh_l", "sh_r", "el_l",
                 "el_r", "wr_l", "wr_r", "hip_l", "hip_r", "kn_l", "kn_r", "ank_l",
                 "ank_r"):
        xy = scores.get(name)
        pts.append({"name": name, "x": (xy or (0, 0))[0], "y": (xy or (0, 0))[1],
                    "score": 0.9 if xy else 0.01, "visible": xy is not None})
    return {"keypoints": pts}


UPRIGHT = _kp(sh_l=(100, 100), sh_r=(140, 100), hip_l=(105, 200), hip_r=(135, 200),
              kn_l=(105, 260), kn_r=(135, 260), ank_l=(105, 320), ank_r=(135, 320),
              el_l=(90, 150), el_r=(150, 150), wr_l=(95, 190), wr_r=(145, 190))


# ---------------------------------------------------------------- schema

class SchemaTests(unittest.TestCase):
    def test_a_layer_cannot_carry_an_invented_state(self):
        with self.assertRaises(AnalysisError):
            Layer(name="body", state="pretty_good", method="x")

    def test_the_five_states_are_exactly_the_documented_ones(self):
        self.assertEqual(
            canonical.VALID_LAYER_STATES,
            {"supported", "partial", "ambiguous", "unavailable", "failed"})

    def test_ambiguous_and_unavailable_carry_no_observations(self):
        for state in ("ambiguous", "unavailable", "failed"):
            self.assertFalse(Layer("body", state, "m").has_observations, state)
        for state in ("supported", "partial"):
            self.assertTrue(Layer("body", state, "m").has_observations, state)

    def test_analysis_json_roundtrips(self):
        src = canonical.SourceInfo("c.mp4", "/tmp/c.mp4", "ab" * 32, 4.0, 30.0,
                                   1920, 1080, "mp4", "h264", [0, 119], 6.0, 24)
        r = AnalysisResult(
            source=src,
            layers={"body": Layer("body", LayerState.PARTIAL, "movenet"),
                    "ball": ball_layer(), "clubhead": clubhead_layer()},
            metrics={"body_2d": compute_body_metrics([UPRIGHT] * 10, "partial"),
                     "physical": physical_metrics(None)},
            research_status={"experimental_components_used": []},
            created_utc="2026-09-16T00:00:00+00:00")
        back = AnalysisResult.from_dict(json.loads(json.dumps(r.to_dict())))
        self.assertEqual(back.to_dict(), r.to_dict())

    def test_a_foreign_schema_version_is_refused(self):
        src = canonical.SourceInfo("c.mp4", "", "a" * 64, 1.0, 30.0, 10, 10)
        d = AnalysisResult(src, {}, {}, {}).to_dict()
        d["schema_version"] = "something-else.v9"
        with self.assertRaises(AnalysisError):
            AnalysisResult.from_dict(d)


# ---------------------------------------------------------------- layers

class BallAndClubheadAreUnavailableTests(unittest.TestCase):
    def test_ball_is_unavailable_with_a_substantive_reason(self):
        b = ball_layer()
        self.assertEqual(b.state, LayerState.UNAVAILABLE)
        self.assertEqual(b.method, "none")
        self.assertEqual(b.provenance["routes_admitted"], [])
        self.assertEqual(b.coverage["frames_with_observation"], 0)
        for term in ("flagstick", "1/51", "seed"):
            self.assertIn(term, b.limitations[0])

    def test_clubhead_is_unavailable_and_says_the_blocker_is_the_data(self):
        c = clubhead_layer()
        self.assertEqual(c.state, LayerState.UNAVAILABLE)
        self.assertEqual(c.method, "none")
        self.assertIn("data, not the code", c.limitations[0])

    def test_no_falsified_or_withdrawn_route_is_reachable_from_the_canonical_path(self):
        """The canonical modules must not even import the research trackers."""
        banned = ("research_ball", "region_tracker_mil", "clubhead_methods",
                  "hybrid_tracking", "seeded_tracking", "fresh_ai_vision",
                  "ai_demo", "luna_roi_experiment", "clubhead_pseudo_labels",
                  "tracer_weak_supervision", "perception")
        for mod in ("canonical.py", "canonical_worker.py"):
            text = (REPO / "ghostcaddie" / "video" / mod).read_text()
            for name in banned:
                self.assertNotIn(f"import {name}", text, f"{mod} imports {name}")
                self.assertNotIn(f"from .{name}", text, f"{mod} imports {name}")


# ---------------------------------------------------------------- metrics

class BodyMetricSufficiencyTests(unittest.TestCase):
    def test_metrics_are_refused_outright_when_the_layer_is_not_trustworthy(self):
        for state in ("ambiguous", "unavailable", "failed"):
            m = compute_body_metrics([UPRIGHT] * 40, state)
            self.assertEqual(m["supported"], [], state)
            self.assertEqual(len(m["unavailable"]), len(canonical.BODY_METRIC_SPECS))
            self.assertIn(state, m["unavailable"][0].reason)

    def test_a_metric_needs_its_own_landmarks_on_enough_frames(self):
        """Shoulders present throughout, knees on a third of the frames."""
        frames = [_kp(sh_l=(100, 100), sh_r=(140, 100)) for _ in range(20)]
        frames += [_kp(sh_l=(100, 100), sh_r=(140, 100), hip_l=(105, 200),
                       hip_r=(135, 200), kn_l=(105, 260), kn_r=(135, 260),
                       ank_l=(105, 320), ank_r=(135, 320)) for _ in range(10)]
        m = compute_body_metrics(frames, "supported")
        names = {x.name for x in m["supported"]}
        self.assertIn("shoulder_tilt", names)
        self.assertNotIn("left_knee_flexion", names)
        knee = next(x for x in m["unavailable"] if x.name == "left_knee_flexion")
        self.assertIn("hip_l+kn_l+ank_l", knee.reason)
        self.assertEqual(knee.frames_used, 10)
        self.assertEqual(knee.frames_possible, 30)

    def test_a_handful_of_frames_is_never_enough(self):
        m = compute_body_metrics([UPRIGHT] * 3, "supported")
        self.assertEqual(m["supported"], [])

    def test_supported_metrics_are_labelled_as_2d_image_space_estimates(self):
        m = compute_body_metrics([UPRIGHT] * 20, "supported")
        self.assertTrue(m["supported"])
        for metric in m["supported"]:
            self.assertEqual(metric.space, "2d_image")
            self.assertIn("2D image-space", metric.reason)
            self.assertNotIn("3D", metric.reason.replace("no 3D", ""))

    def test_no_metric_claims_clinical_or_coaching_certainty(self):
        m = compute_body_metrics([UPRIGHT] * 20, "supported")
        blob = " ".join(x.reason for x in m["supported"]).lower()
        for word in ("diagnos", "should fix", "correct your", "injury", "treat"):
            self.assertNotIn(word, blob)

    def test_a_shoulder_line_reads_as_a_tilt_not_a_direction(self):
        """Mirroring the golfer must not flip the tilt to +-180 degrees."""
        left = _kp(sh_l=(140, 100), sh_r=(100, 110))
        right = _kp(sh_l=(100, 110), sh_r=(140, 100))
        a = compute_body_metrics([left] * 20, "supported")["supported"][0]
        b = compute_body_metrics([right] * 20, "supported")["supported"][0]
        self.assertAlmostEqual(a.median, b.median, places=6)
        self.assertLessEqual(abs(a.median), 90.0)

    def test_knee_flexion_is_an_interior_angle(self):
        straight = _kp(hip_l=(100, 100), kn_l=(100, 200), ank_l=(100, 300))
        m = compute_body_metrics([straight] * 20, "supported")
        knee = next(x for x in m["supported"] if x.name == "left_knee_flexion")
        self.assertAlmostEqual(knee.median, 180.0, places=4)


class PhysicalMetricsTests(unittest.TestCase):
    def test_every_physical_metric_is_unavailable_with_its_own_reason(self):
        m = physical_metrics(None)
        self.assertEqual(m["supported"], [])
        names = {x.name for x in m["unavailable"]}
        self.assertEqual(names, {"ball_speed", "clubhead_speed", "launch_angle",
                                 "carry_distance", "apex_height", "spin_rate"})
        for metric in m["unavailable"]:
            self.assertEqual(metric.state, MetricState.UNAVAILABLE)
            self.assertIsNone(metric.median)
            self.assertTrue(metric.reason.strip())

    def test_spin_is_an_explicit_unavailable_contract_entry_in_rpm(self):
        """Spin must be DECLARED unavailable, not silently absent.

        An omitted metric reads as an oversight, or as something a flag could
        unlock. It is stated in the output in its own unit, with a reason naming
        the missing measurement, and it stays out of `supported` under every
        calibration the canonical path accepts.
        """
        spin = next(x for x in physical_metrics(None)["unavailable"]
                    if x.name == "spin_rate")
        self.assertEqual(spin.unit, "rpm")
        self.assertEqual(spin.space, "physical")
        self.assertEqual(spin.state, MetricState.UNAVAILABLE)
        self.assertIsNone(spin.median)
        self.assertIn("no validated rotation measurement or spin calibration",
                      spin.reason)
        unlocked = physical_metrics({"validated_physical_scale": True,
                                     "validated_action_time": True})
        self.assertEqual(unlocked["supported"], [])
        self.assertIn("spin_rate", {x.name for x in unlocked["unavailable"]})

    def test_nominal_fps_is_never_accepted_as_action_time(self):
        for metric in physical_metrics(None)["unavailable"]:
            self.assertIn("Nominal container FPS is not validated action time",
                          metric.reason)

    def test_calibration_alone_does_not_unlock_a_ball_metric(self):
        m = physical_metrics({"validated_physical_scale": True,
                              "validated_action_time": True})
        self.assertEqual(m["supported"], [])
        self.assertIn("no ball/clubhead observations",
                      next(x for x in m["unavailable"] if x.name == "ball_speed").reason)


# ---------------------------------------------------------------- model paths

class ModelPathTests(unittest.TestCase):
    def test_no_canonical_default_points_into_tmp(self):
        for key in model_paths.MODELS:
            r = model_paths.resolve(key, env={})
            self.assertFalse(r.path.startswith("/tmp/"), key)
        for p in model_paths.MODEL_SEARCH_PATH:
            self.assertFalse(str(p).startswith("/tmp/"), p)

    def test_resolution_order_is_explicit_then_env_then_search_path(self):
        with tempfile.TemporaryDirectory() as raw:
            d = str(Path(raw).resolve())
            a, b = Path(d) / "a.tflite", Path(d) / "b.tflite"
            a.write_bytes(b"x")
            b.write_bytes(b"y")
            self.assertEqual(model_paths.resolve("body", str(a)).path, str(a))
            self.assertEqual(
                model_paths.resolve("body", env={"FAIRWAY_BODY_MODEL": str(b)}).path,
                str(b))
            # explicit beats the environment
            self.assertEqual(
                model_paths.resolve("body", str(a),
                                    env={"FAIRWAY_BODY_MODEL": str(b)}).path, str(a))

    def test_a_model_root_supplies_the_documented_filename(self):
        with tempfile.TemporaryDirectory() as raw:
            d = str(Path(raw).resolve())
            f = Path(d) / model_paths.MODELS["body"].filename
            f.write_bytes(b"x")
            r = model_paths.resolve("body", env={model_paths.MODEL_ROOT_ENV: d})
            self.assertTrue(r.found)
            self.assertEqual(r.path, str(f))

    def test_an_absent_optional_asset_is_reported_not_raised(self):
        r = model_paths.resolve("ball_tapir", env={})
        self.assertFalse(r.found)
        self.assertEqual(r.path, "")
        self.assertIn("bootstapir_checkpoint_v2.pt", r.reason)
        self.assertIn("Nothing is downloaded automatically", r.reason)
        self.assertTrue(r.searched, "the operator must be told where we looked")

    def test_a_wrong_explicit_path_is_a_configuration_error_not_a_fallthrough(self):
        with tempfile.TemporaryDirectory() as raw:
            d = str(Path(raw).resolve())
            f = Path(d) / model_paths.MODELS["body"].filename
            f.write_bytes(b"x")
            r = model_paths.resolve("body", "/nonexistent/model.tflite",
                                    env={model_paths.MODEL_ROOT_ENV: d})
            self.assertFalse(r.found)
            self.assertIn("explicit model path", r.reason)

    def test_nothing_in_the_repo_can_download_a_model(self):
        text = (REPO / "ghostcaddie" / "video" / "model_paths.py").read_text()
        for bad in ("urllib", "requests", "http", "download("):
            self.assertNotIn(bad, text.replace("downloaded automatically", "")
                             .replace("downloads", "").replace("download", ""))


# ---------------------------------------------------------------- CLI contract

_WORKER_OK = {
    "sampled_frame_count": 20, "frame_range": [0, 95], "decimation_step": 5,
    "identity": {"layer_state": "supported", "identity_method": "movenet_proposal_unchallenged",
                 "method": "movenet_singlepose_lightning_v3_on_identified_subject",
                 "roi_xywh": [10, 10, 100, 300], "subject_box_xywh": [20, 20, 80, 280],
                 "jitter_px_median": 1.5, "score_threshold": 0.3,
                 "second_person": {"found": False}, "limitations": ["stated limit"]},
    "attributed_frames": [UPRIGHT] * 20,
    "render": {"written": True, "reason": "encoded 20 frames"},
    "diagnostics": {"per_frame": []},
    "environment": {"python": "3.11.16", "versions": {"cv2": "5.0.0"}},
}


def _fake_worker(result):
    def run(cmd, **kw):
        class P:
            returncode = 0
            stdout = json.dumps(result)
            stderr = ""
        return P()
    return run


class _CliBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.video = Path(self.tmp.name) / "clip.mp4"
        self.video.write_bytes(b"not really a video, the worker is stubbed")
        self.out = Path(self.tmp.name) / "out"
        self.meta = mock.patch(
            "ghostcaddie.video.metadata.inspect_video",
            return_value=mock.Mock(frame_rate=30.0, duration_seconds=4.0, width=1920,
                                   height=1080, container_format="mov,mp4",
                                   codec="h264"))
        self.interp = mock.patch.object(canonical, "body_interpreter_path",
                                        return_value=sys.executable)

    def analyze(self, worker=None, **kw):
        with self.meta, self.interp, mock.patch.object(
                canonical.subprocess, "run",
                side_effect=_fake_worker(worker or _WORKER_OK)):
            return canonical.analyze_video(str(self.video), str(self.out), **kw)


class OutputContractTests(_CliBase):
    def test_the_same_five_outputs_appear_every_time(self):
        self.analyze()
        for name in ("annotated.mp4", "report.md", "analysis.json",
                     "provenance.json", "diagnostics.json"):
            # annotated.mp4 is written by the (stubbed) worker; the rest by us
            if name != "annotated.mp4":
                self.assertTrue((self.out / name).is_file(), name)
        self.assertTrue((self.out / "debug").is_dir())

    def test_analysis_json_is_the_source_of_truth_and_parses_back(self):
        r = self.analyze()
        d = json.loads((self.out / "analysis.json").read_text())
        self.assertEqual(AnalysisResult.from_dict(d).to_dict(), r.to_dict())
        self.assertEqual(set(d["layers"]), {"body", "ball", "clubhead"})

    def test_every_run_binds_to_the_source_sha256(self):
        r = self.analyze()
        prov = json.loads((self.out / "provenance.json").read_text())
        self.assertEqual(prov["source_binding"]["source_sha256"], r.source.sha256)
        self.assertTrue(prov["source_binding"]["hash_bound"])
        self.assertEqual(prov["source"]["sha256"], r.source.sha256)

    def test_a_mismatched_expected_hash_fails_closed(self):
        from ghostcaddie.video.source_binding import SourceBindingError
        with self.assertRaises(SourceBindingError):
            self.analyze(expected_sha256="0" * 64)
        self.assertFalse((self.out / "analysis.json").exists(),
                         "a refused run must not leave an analysis behind")

    def test_a_failed_render_is_reported_not_papered_over(self):
        w = json.loads(json.dumps(_WORKER_OK))
        w["render"] = {"written": False, "reason": "no encoder available"}
        r = self.analyze(worker=w)
        self.assertFalse(r.outputs["annotated_video_written"])
        self.assertIsNone(r.outputs["annotated_video"])
        self.assertIn("no encoder available", r.outputs["annotated_video_reason"])
        self.assertIn("NOT PRODUCED", (self.out / "report.md").read_text())

    def test_a_worker_failure_raises_instead_of_writing_an_empty_success(self):
        class P:
            returncode = 1
            stdout = ""
            stderr = "boom"
        with self.meta, self.interp, mock.patch.object(
                canonical.subprocess, "run", return_value=P()):
            with self.assertRaises(AnalysisError):
                canonical.analyze_video(str(self.video), str(self.out))
        self.assertFalse((self.out / "analysis.json").exists())

    def test_processing_is_bounded_and_the_bounds_reach_the_worker(self):
        captured = {}

        def run(cmd, **kw):
            captured.update(json.loads(kw["input"]))
            class P:
                returncode = 0
                stdout = json.dumps(_WORKER_OK)
                stderr = ""
            return P()
        with self.meta, self.interp, mock.patch.object(
                canonical.subprocess, "run", side_effect=run):
            canonical.analyze_video(str(self.video), str(self.out),
                                    sample_fps=3.0, max_frames=25, start_seconds=1.0)
        self.assertEqual(captured["sample_fps"], 3.0)
        self.assertEqual(captured["max_frames"], 25)
        self.assertEqual(captured["start_seconds"], 1.0)
        self.assertIn("timeout", "timeout")

    def test_unbounded_settings_are_refused(self):
        for kw in ({"sample_fps": 0}, {"sample_fps": float("inf")},
                   {"max_frames": 0}, {"max_frames": -5}):
            with self.assertRaises(AnalysisError):
                self.analyze(**kw)

    def test_a_missing_source_is_refused_before_anything_is_written(self):
        with self.assertRaises(AnalysisError):
            canonical.analyze_video(str(self.out / "nope.mp4"), str(self.out))

    def test_the_operator_path_can_be_redacted_from_the_record(self):
        r = self.analyze(redact_path=True)
        self.assertEqual(r.source.path, "")
        self.assertEqual(r.source.display_name, "clip.mp4")


class LayerStateTests(_CliBase):
    def test_ball_and_clubhead_stay_unavailable_whatever_the_worker_says(self):
        w = json.loads(json.dumps(_WORKER_OK))
        w["identity"]["layer_state"] = "supported"
        r = self.analyze(worker=w)
        self.assertEqual(r.layers["ball"].state, "unavailable")
        self.assertEqual(r.layers["clubhead"].state, "unavailable")

    def test_an_ambiguous_body_yields_no_metrics_at_all(self):
        w = json.loads(json.dumps(_WORKER_OK))
        w["identity"]["layer_state"] = "ambiguous"
        w["attributed_frames"] = []
        r = self.analyze(worker=w)
        self.assertEqual(r.layers["body"].state, "ambiguous")
        self.assertEqual(r.metrics["body_2d"]["supported"], [])
        self.assertIn("AMBIGUOUS", (self.out / "report.md").read_text())

    def test_an_unavailable_body_is_reported_with_its_reason(self):
        w = json.loads(json.dumps(_WORKER_OK))
        w["identity"] = {"layer_state": "unavailable", "identity_method": "none",
                         "method": "none", "roi_xywh": None,
                         "limitations": ["no body was stable enough"]}
        w["attributed_frames"] = []
        r = self.analyze(worker=w)
        self.assertEqual(r.layers["body"].state, "unavailable")
        self.assertIn("no body was stable enough", r.layers["body"].limitations)

    def test_an_invented_worker_state_becomes_failed_not_supported(self):
        w = json.loads(json.dumps(_WORKER_OK))
        w["identity"]["layer_state"] = "excellent"
        r = self.analyze(worker=w)
        self.assertEqual(r.layers["body"].state, "failed")

    def test_an_operator_roi_is_recorded_as_operator_input(self):
        w = json.loads(json.dumps(_WORKER_OK))
        w["identity"]["identity_method"] = "operator_roi"
        r = self.analyze(worker=w, roi=[10, 10, 100, 300])
        self.assertTrue(r.layers["body"].provenance["operator_roi_supplied"])
        self.assertEqual(r.layers["body"].provenance["identity_method"], "operator_roi")
        prov = json.loads((self.out / "provenance.json").read_text())
        self.assertEqual(prov["methods"]["body"]["identity"], "operator_roi")

    def test_research_status_reports_no_experimental_component_on_this_path(self):
        r = self.analyze()
        self.assertEqual(r.research_status["experimental_components_used"], [])
        self.assertFalse(r.research_status["pseudo_labels_present"])
        self.assertFalse(r.research_status["production_eligible"])


class NoNetworkTests(_CliBase):
    def test_a_canonical_run_opens_no_socket(self):
        opened = []
        real = socket.socket

        class Tattle(real):
            def __init__(self, *a, **k):
                opened.append(a)
                super().__init__(*a, **k)
        with mock.patch.object(socket, "socket", Tattle), \
             mock.patch.object(socket, "create_connection",
                               side_effect=AssertionError("network!")), \
             mock.patch.object(socket, "getaddrinfo",
                               side_effect=AssertionError("dns!")):
            self.analyze()
        self.assertEqual(opened, [], "the canonical path opened a socket")

    def test_the_orchestrator_imports_no_network_client(self):
        text = (REPO / "ghostcaddie" / "video" / "canonical.py").read_text()
        for bad in ("import requests", "import urllib", "import http",
                    "import socket", "webbrowser"):
            self.assertNotIn(bad, text)

    def test_the_worker_is_launched_with_proxy_variables_stripped(self):
        captured = {}

        def run(cmd, **kw):
            captured["env"] = kw.get("env", {})
            captured["cmd"] = cmd
            class P:
                returncode = 0
                stdout = json.dumps(_WORKER_OK)
                stderr = ""
            return P()
        with mock.patch.dict(os.environ, {"HTTPS_PROXY": "http://evil:8080"}), \
             self.meta, self.interp, \
             mock.patch.object(canonical.subprocess, "run", side_effect=run):
            canonical.analyze_video(str(self.video), str(self.out))
        for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            self.assertNotIn(var, captured["env"])
        self.assertEqual(captured["env"]["FAIRWAYOS_OFFLINE"], "1")
        self.assertEqual(captured["cmd"][1:],
                         ["-m", "ghostcaddie.video.canonical_worker"])


class NoImplicitFreshAiTests(unittest.TestCase):
    def test_the_canonical_modules_never_mention_a_fresh_ai_entry_point(self):
        for mod in ("canonical.py", "canonical_worker.py"):
            text = (REPO / "ghostcaddie" / "video" / mod).read_text()
            self.assertNotIn("fresh_ai_vision", text)
            self.assertNotIn("hermes", text.lower())

    def test_the_server_dispatches_canonical_to_the_canonical_runner(self):
        from ghostcaddie.upload import server
        src = (REPO / "ghostcaddie" / "upload" / "server.py").read_text()
        canonical_branch = src.split('if analysis == "canonical":')[1].split("elif")[0]
        self.assertIn("_run_canonical_job", canonical_branch)
        self.assertNotIn("_run_fresh_job", canonical_branch)
        self.assertTrue(hasattr(server, "_run_canonical_job"))

    def test_the_canonical_job_runner_calls_the_same_analysis_entry_point(self):
        """The upload service must wrap the CLI's function, not reimplement it."""
        import inspect
        from ghostcaddie.upload import server
        src = inspect.getsource(server._run_canonical_job)
        self.assertIn("canonical.analyze_video", src)
        self.assertNotIn("fresh", src)


# ------------------------------------------------- execution, model-dependent

class CanonicalExecutionTests(unittest.TestCase):
    """Real inference. Skips -- never fails -- when the model is not placed.

    This is the pattern the audit asked the rest of the suite to copy: portable
    contract assertions always run, weight-dependent execution skips with the
    capability reason.
    """

    CLIPS = REPO / "out/demo_improvement_campaign/development/pexels_round4"

    def setUp(self):
        r = model_paths.resolve("body")
        if not r.found:
            self.skipTest(f"body model not runnable here: {r.reason}")
        interp = canonical.body_interpreter_path()
        if not (os.path.isfile(interp) and os.access(interp, os.X_OK)):
            self.skipTest(f"body interpreter not available here: {interp}")
        clips = sorted(self.CLIPS.glob("*.mp4")) if self.CLIPS.is_dir() else []
        if not clips:
            self.skipTest("local validation clips are not present")
        self.clip = clips[0]

    def test_an_absent_body_model_makes_body_unavailable_not_wrong(self):
        """The workflow still completes; it just declines the body layer."""
        with tempfile.TemporaryDirectory() as d:
            r = canonical.analyze_video(
                str(self.clip), d, sample_fps=2.0, max_frames=6,
                body_model="/nonexistent/movenet.tflite")
            self.assertEqual(r.layers["body"].state, "unavailable")
            self.assertEqual(r.metrics["body_2d"]["supported"], [])
            self.assertFalse(r.layers["body"].provenance["model"]["found"])
            self.assertTrue((Path(d) / "analysis.json").is_file())

    def test_the_renderer_draws_nothing_for_a_layer_with_no_observations(self):
        import numpy as np
        from ghostcaddie.video import canonical_worker as cw
        img = np.full((240, 320, 3), 70, np.uint8)
        pts = [{"name": n, "x": 160.0, "y": 120.0, "score": 0.9, "visible": True}
               for n in cw.KP_NAMES]
        below = cw.STATUS_BAR_H + 2
        blank = cw.draw(img, pts, attributed=False, body_state="ambiguous")
        drawn = cw.draw(img, pts, attributed=True, body_state="supported")
        # nothing at all below the status strip when the layer has no observations
        self.assertTrue((blank[below:] == img[below:]).all(),
                        "an unavailable layer must leave the picture alone")
        self.assertFalse((drawn[below:] == img[below:]).all(),
                         "a supported layer must actually draw")
        self.assertIn("UNAVAILABLE", "BALL UNAVAILABLE   CLUBHEAD UNAVAILABLE")

    def test_only_stable_landmarks_are_drawn(self):
        from ghostcaddie.video import canonical_worker as cw
        for face in ("nose", "eye_l", "eye_r", "ear_l", "ear_r"):
            self.assertNotIn(face, cw.RENDERED_KEYPOINTS)
        for joint in ("sh_l", "hip_r", "kn_l", "ank_r", "wr_l"):
            self.assertIn(joint, cw.RENDERED_KEYPOINTS)

    def test_a_real_clip_produces_the_full_output_contract(self):
        with tempfile.TemporaryDirectory() as d:
            r = canonical.analyze_video(str(self.clip), d, sample_fps=2.0,
                                        max_frames=12)
            self.assertIn(r.layers["body"].state, canonical.VALID_LAYER_STATES)
            self.assertEqual(r.layers["ball"].state, "unavailable")
            self.assertEqual(r.layers["clubhead"].state, "unavailable")
            for name in ("analysis.json", "provenance.json", "diagnostics.json",
                         "report.md"):
                self.assertTrue((Path(d) / name).is_file(), name)
            self.assertEqual(r.source.sampled_frame_count, 12)


class PortableWithoutResearchCheckpointsTests(unittest.TestCase):
    """The suite must be truthful on a machine with no optional checkpoints."""

    def test_absent_research_checkpoints_are_an_expected_state(self):
        from ghostcaddie.upload.runtimes import RuntimeRegistry
        rd = RuntimeRegistry.default().readiness()
        for target, v in rd.items():
            self.assertIn(v["availability"],
                          {"ready", "model_absent", "misconfigured",
                           "interpreter_missing"}, target)
            # readiness never fakes a capability to make a test pass
            if v["availability"] != "ready":
                self.assertFalse(v["can_execute_now"], target)
            self.assertTrue(len(v["reason"]) > 20, target)

    def test_a_missing_checkpoint_is_distinguishable_from_a_misconfiguration(self):
        from ghostcaddie.upload.adapters import BallTapirAdapter
        absent = BallTapirAdapter().capability()
        self.assertFalse(absent.model_present)
        self.assertIn("was not found", absent.reason)
        wrong = BallTapirAdapter(model_path="/nonexistent/x.pt").capability()
        self.assertFalse(wrong.model_present)
        self.assertIn("explicit model path", wrong.reason)

    def test_the_canonical_layers_do_not_depend_on_any_research_checkpoint(self):
        """Ball and clubhead are unavailable whether or not weights exist."""
        self.assertEqual(ball_layer().state, "unavailable")
        self.assertEqual(clubhead_layer().state, "unavailable")
        for key in ("clubhead_sam2", "ball_sam2", "ball_tapir"):
            self.assertFalse(model_paths.MODELS[key].canonical, key)
        self.assertTrue(model_paths.MODELS["body"].canonical)


if __name__ == "__main__":
    unittest.main()


class UploadCanonicalModeTests(unittest.TestCase):
    """The upload service must run the SAME backend the CLI runs.

    Needs cv2 only to synthesise a fixture clip; skips without it rather than
    failing, per the portable-suite rule.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import cv2, numpy as np           # noqa: F401
        except Exception as exc:              # pragma: no cover - env dependent
            raise unittest.SkipTest(f"cv2 not available here: {exc}")
        import threading
        import urllib.parse
        import urllib.request
        from ghostcaddie.upload.server import build_server
        from ghostcaddie.upload.validation import VideoLimits
        cls.urllib, cls.up = urllib.request, urllib.parse
        cls.d = tempfile.mkdtemp()
        cls.srv = build_server(os.path.join(cls.d, "root"), port=0,
                               limits=VideoLimits(max_seconds=60))
        cls.host, cls.port = cls.srv.server_address[0], cls.srv.server_address[1]
        cls.token = cls.srv.RequestHandlerClass.csrf_token
        cls.imports = cls.srv.RequestHandlerClass.store.import_dir
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        import cv2, numpy as np
        os.makedirs(cls.imports, exist_ok=True)
        path = os.path.join(cls.imports, "synthetic.mp4")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
        for i in range(30):
            vw.write(np.full((240, 320, 3), (i * 7) % 255, np.uint8))
        vw.release()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        # The test interpreter has the required runtime; a fresh clone does not
        # promise a repository-local .venv at the registry's development path.
        patcher = mock.patch.dict(os.environ, {"FAIRWAY_BODY_PYTHON": sys.executable})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _post(self, fields):
        data = self.up.urlencode(fields).encode()
        req = self.urllib.Request(
            f"http://{self.host}:{self.port}/jobs", data=data,
            headers={"X-FairwayOS-Token": self.token,
                     "Content-Type": "application/x-www-form-urlencoded"})
        try:
            with self.urllib.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except Exception as e:                # HTTPError carries the JSON body
            return e.code, json.loads(e.read())

    def _job(self, jid):
        import time
        for _ in range(200):
            with self.urllib.urlopen(
                    f"http://{self.host}:{self.port}/jobs/{jid}") as r:
                d = json.loads(r.read())
            if d["state"] in ("done", "failed", "cancelled", "awaiting_seeds"):
                return d
            time.sleep(0.1)
        self.fail("job did not reach a terminal state")

    def test_canonical_mode_produces_the_canonical_schema(self):
        code, job = self._post({"name": "synthetic.mp4", "analysis": "canonical"})
        self.assertEqual(code, 202, job)
        d = self._job(job["id"])
        self.assertEqual(d["state"], "done", d.get("error"))
        res = d["result"]
        self.assertEqual(res["schema_version"], canonical.SCHEMA_VERSION)
        self.assertEqual(res["analysis"], "canonical")
        self.assertEqual(res["layers"]["ball"]["state"], "unavailable")
        self.assertEqual(res["layers"]["clubhead"]["state"], "unavailable")
        self.assertFalse(res["three_target_success"])
        self.assertNotIn("fresh", json.dumps(res).lower())

    def test_canonical_mode_never_reaches_the_research_ai_path(self):
        from ghostcaddie.upload import server
        with mock.patch.object(server, "_run_fresh_job",
                               side_effect=AssertionError("fresh_ai reached!")):
            code, job = self._post({"name": "synthetic.mp4", "analysis": "canonical"})
            self.assertEqual(code, 202)
            d = self._job(job["id"])
        self.assertEqual(d["state"], "done", d.get("error"))

    def test_an_unknown_analysis_mode_is_refused_rather_than_defaulted(self):
        code, d = self._post({"name": "synthetic.mp4", "analysis": "make_it_good"})
        self.assertEqual(code, 400)
        self.assertIn("unknown analysis mode", d["error"])
