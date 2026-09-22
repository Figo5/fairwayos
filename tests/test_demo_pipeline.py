"""End-to-end demo tests: isolation, provenance, wording, render contract.

These run the real pipeline on a generated synthetic clip in a temporary directory.
Nothing here touches source footage, research media, the sealed evaluation corpus or
the canonical workflow's outputs; the isolation tests exist to prove that.
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ghostcaddie.demo import (DEMO_ANALYSIS_SCHEMA, DEMO_CONFIG_SCHEMA,
                              DEMO_PROVENANCE_SCHEMA)
from ghostcaddie.demo.config import parse_config, validate_config
from ghostcaddie.demo.errors import DemoConfigError, DemoError, DemoRenderError
from ghostcaddie.demo.pipeline import run_demo, verify_render
from ghostcaddie.demo.probe import probe_source
from ghostcaddie.demo.report import render_report, render_summary
from ghostcaddie.demo.result import DemoResult
from tests.demo_fixtures import (TempDirTestCase, base_config, make_synthetic_clip,
                                 require_cv2, synthetic_probe, write_config)

DEMO_MODULES = ("ghostcaddie.demo", "ghostcaddie.demo.config",
                "ghostcaddie.demo.tracking", "ghostcaddie.demo.analytics",
                "ghostcaddie.demo.geometry", "ghostcaddie.demo.phases",
                "ghostcaddie.demo.report", "ghostcaddie.demo.result",
                "ghostcaddie.demo.pipeline", "ghostcaddie.demo.paths",
                "ghostcaddie.demo.picker", "ghostcaddie.demo.probe",
                "ghostcaddie.demo.cli", "ghostcaddie.demo.errors",
                "ghostcaddie.demo.showcase", "ghostcaddie.demo.showcase_report")

#: The half of the package that may import cv2, because it never runs in the
#: process that owns the output contract.
CV2_MODULES = ("ghostcaddie.demo.worker", "ghostcaddie.demo.render",
               "ghostcaddie.demo.render_showcase", "ghostcaddie.demo.setup_gui")


class IsolationTests(unittest.TestCase):
    """Demo mode must not be able to change canonical behaviour."""

    def test_canonical_modules_do_not_mention_the_demo_namespace(self):
        for name in ("canonical.py", "canonical_worker.py"):
            text = (REPO / "ghostcaddie" / "video" / name).read_text(encoding="utf-8")
            self.assertNotIn("ghostcaddie.demo", text)
            self.assertNotIn("from .demo", text)

    def test_importing_canonical_does_not_import_demo(self):
        code = ("import sys; import ghostcaddie.video.canonical as c; "
                "print('demo' if any(m.startswith('ghostcaddie.demo') "
                "for m in sys.modules) else 'clean'); "
                "print(c.SCHEMA_VERSION)")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, cwd=str(REPO), timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        lines = out.stdout.strip().splitlines()
        self.assertEqual(lines[0], "clean")
        self.assertEqual(lines[1], "fairwayos-analysis.v1")

    def test_the_demo_schemas_are_not_the_canonical_analysis_schema(self):
        from ghostcaddie.video import canonical
        self.assertEqual(canonical.SCHEMA_VERSION, "fairwayos-analysis.v1")
        for schema in (DEMO_ANALYSIS_SCHEMA, DEMO_CONFIG_SCHEMA,
                       DEMO_PROVENANCE_SCHEMA):
            self.assertNotEqual(schema, canonical.SCHEMA_VERSION)
            self.assertIn("demo", schema)

    def test_the_stdlib_side_of_demo_imports_without_cv2(self):
        """The orchestrator half must import on an interpreter with no cv2."""
        blocked = ("cv2", "numpy", "ai_edge_litert", "torch")

        class Blocker:
            def find_module(self, name, path=None):  # pragma: no cover - py<3.12
                return self if name.split(".")[0] in blocked else None

            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in blocked:
                    raise ImportError(f"{name} is blocked for this test")
                return None

        for module in list(sys.modules):
            if module.startswith("ghostcaddie.demo"):
                del sys.modules[module]
        blocker = Blocker()
        sys.meta_path.insert(0, blocker)
        try:
            for name in DEMO_MODULES:
                importlib.import_module(name)
        finally:
            sys.meta_path.remove(blocker)
            for module in list(sys.modules):
                if module.startswith("ghostcaddie.demo"):
                    del sys.modules[module]
            for name in ("ghostcaddie.demo", "ghostcaddie.demo.config"):
                importlib.import_module(name)

    def test_the_worker_is_the_only_demo_module_that_needs_cv2(self):
        for name in DEMO_MODULES + CV2_MODULES:
            path = REPO / (name.replace(".", "/") + ".py")
            if not path.is_file():
                path = REPO / name.replace(".", "/") / "__init__.py"
            self.assertTrue(path.is_file(), name)
            text = path.read_text(encoding="utf-8")
            needs_cv2 = any(line.strip().startswith(("import cv2", "from cv2"))
                            for line in text.splitlines())
            if name in CV2_MODULES:
                self.assertTrue(needs_cv2, name)
            else:
                self.assertFalse(needs_cv2, name)

    def test_the_demo_output_contract_does_not_collide_with_canonical_names(self):
        from ghostcaddie.demo.paths import out_paths
        from ghostcaddie.video.canonical import _out_paths
        # files only: both contracts have a debug directory, and that is fine
        demo = {p.name for p in out_paths("/tmp/x").values() if p.suffix}
        canonical = {p.name for p in _out_paths(Path("/tmp/x")).values() if p.suffix}
        self.assertEqual(canonical & demo, set())
        self.assertIn("demo_annotated.mp4", demo)
        self.assertIn("annotated.mp4", canonical)

    def test_the_cli_registers_the_demo_commands(self):
        from ghostcaddie.cli import _build_parser
        parser = _build_parser()
        actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
        choices = set()
        for action in actions:
            choices |= set(action.choices or ())
        self.assertIn("demo-video", choices)
        self.assertIn("demo-setup", choices)
        self.assertIn("analyze-video", choices)

    def test_demo_setup_refuses_a_broken_interpreter_rather_than_falling_back(self):
        from ghostcaddie.demo import cli as demo_cli
        args = mock.Mock(command="demo-setup", video=Path("/tmp/none.mp4"),
                         config=None, out=None, start=None,
                         cv_python="/nonexistent/python3")
        with self.assertRaises(SystemExit) as caught:
            demo_cli.dispatch(args, argv=None)
        self.assertIn("demo-setup failed", str(caught.exception))


class PipelineTestCase(TempDirTestCase):
    """Shared fixture: one synthetic clip, one or two real pipeline runs."""

    @classmethod
    def setUpClass(cls):
        cls._class_tmp = None

    def run_fixture(self, *, frames=40, ball_motion="slide", config_overrides=None,
                    start_seconds=0.0, render=True, summary=True, keyframes=None,
                    extra=None):
        require_cv2(self)
        video = make_synthetic_clip(self.tmp, frames=frames, ball_motion=ball_motion,
                                    width=320, height=240, fps=20.0)
        probe = probe_source(str(video))
        first = int(start_seconds * probe.fps)
        last = frames - 1
        # every operator input below is expressed relative to the interval, so the
        # same fixture works for short and long clips
        seed_frame = first + 1
        correction_frame = first + max(2, (last - first) // 2 + 1)
        if correction_frame >= last:
            correction_frame = last - 1
        defaults_keyframes = {
            "address": first,
            "top": first + max(1, (last - first) // 3),
            "impact": first + max(2, 2 * (last - first) // 3),
            "finish": last,
        }
        payload = {
            "schema_version": DEMO_CONFIG_SCHEMA,
            "interval": {"start_seconds": start_seconds,
                         "end_seconds": round(frames / probe.fps, 6),
                         "max_frames": 400},
            "roi": [80, 40, 160, 180],
            "seeds": {"ball": {"frame": seed_frame,
                               "x": 61.0 + seed_frame - 1, "y": 200.0,
                               "note": "synthetic operator seed"}},
            "keyframes": defaults_keyframes if keyframes is None else keyframes,
            "targets": {"ball": {"mode": "template", "template_size": 15,
                                 "search_radius": 20, "min_score": 0.5,
                                 "max_jump_px": 20.0, "loss_frames": 3}},
            "corrections": {"ball": [{"frame": correction_frame,
                                      "x": 61.0 + correction_frame - 1, "y": 200.0,
                                      "note": "synthetic correction"}]},
            "interpolation": {"enabled": True, "max_gap_frames": 2},
            "render": {"max_width": 320, "crf": 28, "preset": "ultrafast"},
            "notes": "synthetic test fixture",
        }
        if config_overrides:
            payload.update(config_overrides)
        config = write_config(self.tmp, payload)
        out = Path(self.tmp) / "out"
        result = run_demo(str(video), str(config), str(out),
                          cv_python=sys.executable, render=render, summary=summary)
        return result, out, video, config


class RunContractTests(PipelineTestCase):
    def test_the_five_documented_outputs_and_their_directories_are_written(self):
        result, out, _video, _config = self.run_fixture()
        for name in ("demo_annotated.mp4", "demo_report.md", "demo_analysis.json",
                     "demo_provenance.json", "demo_summary.md"):
            self.assertTrue((out / name).is_file(), name)
        self.assertTrue((out / "keyframes").is_dir())
        self.assertTrue((out / "debug").is_dir())
        self.assertTrue(list((out / "keyframes").glob("*.png")))

    def test_the_analysis_document_declares_the_demo_mode_and_its_limits(self):
        result, out, _video, _config = self.run_fixture()
        analysis = json.loads((out / "demo_analysis.json").read_text())
        self.assertEqual(analysis["schema_version"], DEMO_ANALYSIS_SCHEMA)
        mode = analysis["mode"]
        self.assertTrue(mode["operator_assisted"])
        self.assertTrue(mode["experimental"])
        self.assertTrue(mode["image_space"])
        self.assertFalse(mode["physical_metrics"])
        self.assertFalse(mode["automatic_detection_claims"])
        self.assertEqual(mode["network_calls"], "none")
        self.assertEqual(mode["external_ai_service"], "none")
        self.assertIn("operator", mode["summary"])

    def test_the_provenance_binds_source_and_config_hashes_and_manual_records(self):
        result, out, video, config = self.run_fixture()
        from ghostcaddie.video.source_binding import sha256_file
        provenance = json.loads((out / "demo_provenance.json").read_text())
        self.assertEqual(provenance["schema_version"], DEMO_PROVENANCE_SCHEMA)
        self.assertEqual(provenance["source"]["sha256"], sha256_file(video))
        self.assertEqual(provenance["config"]["sha256"], sha256_file(config))
        self.assertEqual(provenance["source_binding"]["source_sha256"],
                         sha256_file(video))
        manual = provenance["manual_input_records"]
        self.assertEqual(manual["origin"], "operator")
        self.assertTrue(manual["seeds"])
        self.assertTrue(manual["corrections"])
        self.assertTrue(manual["keyframes"])
        self.assertIn("never", provenance["reuse"]["note"].lower().replace(
            "untouched", "never"))
        self.assertFalse(provenance["reuse"]["canonical_files_modified"])
        self.assertEqual(provenance["network"]["calls"], "none")
        effective = provenance["config"]["effective_resolved_config"]
        self.assertIn("source_frame_range", effective)
        self.assertEqual(effective["keyframes"]["top"],
                         result.config["keyframes"]["top"])

    def test_every_recorded_output_is_hashed_and_matches_the_file_on_disk(self):
        result, out, _video, _config = self.run_fixture()
        from ghostcaddie.video.source_binding import sha256_file
        provenance = json.loads((out / "demo_provenance.json").read_text())
        roles = {entry["role"]: entry for entry in provenance["outputs"]}
        self.assertIn("annotated", roles)
        self.assertIn("analysis", roles)
        self.assertIn("keyframe:address", roles)
        for entry in provenance["outputs"]:
            self.assertTrue(Path(entry["path"]).is_file(), entry["path"])
            self.assertEqual(entry["sha256"], sha256_file(entry["path"]), entry["role"])

    def test_a_nonzero_interval_start_keeps_every_record_in_source_frames(self):
        result, out, _video, _config = self.run_fixture(start_seconds=0.5)
        start = result.interval["start_frame"]
        self.assertGreater(start, 0)
        for point in result.targets["ball"]["points"]:
            self.assertGreaterEqual(point["source_frame"], start)
        self.assertEqual(result.interval["frame_count"],
                         result.interval["end_frame"] - start + 1)
        self.assertLess(result.interval["frame_count"], result.source["frame_count"])
        report = (out / "demo_report.md").read_text()
        self.assertIn(f"source frames {start}–{result.interval['end_frame']}", report)

    def test_the_render_is_verified_by_ffprobe_not_merely_claimed(self):
        result, out, _video, _config = self.run_fixture()
        self.assertTrue(result.render["written"])
        check = json.loads((out / "debug" / "verification.json").read_text())["render"]
        inner = check["render_ffprobe"]
        self.assertTrue(inner["checked"])
        self.assertTrue(inner["ok"], inner.get("problems"))
        self.assertEqual(inner["codec"], "h264")
        self.assertEqual(inner["pix_fmt"], "yuv420p")
        self.assertTrue(inner["faststart"])
        self.assertEqual(inner["frames"], result.interval["frame_count"])

    def test_the_body_is_unavailable_rather_than_invented_on_a_pose_free_clip(self):
        result, out, _video, _config = self.run_fixture()
        self.assertIn(result.body["state"], ("unavailable", "ambiguous"))
        self.assertTrue(all(not record["available"]
                            for record in result.body["per_frame"]))
        self.assertFalse(result.analytics["body_2d"]["supported"])
        report = (out / "demo_report.md").read_text()
        self.assertIn("**UNAVAILABLE**", report)

    def test_no_roi_means_no_body_measurement_at_all(self):
        result, _out, _video, _config = self.run_fixture(
            config_overrides={"roi": None})
        self.assertEqual(result.body["state"], "ambiguous")
        self.assertFalse(result.analytics["body_2d"]["supported"])
        self.assertTrue(any("No operator ROI" in limit
                            for limit in result.limitations))

    def test_the_ball_track_records_observations_a_correction_and_no_stale_points(self):
        result, _out, _video, _config = self.run_fixture()
        track = result.targets["ball"]
        statuses = [p["status"] for p in track["points"]]
        self.assertIn("manual_anchor", statuses)
        self.assertIn("observed_template", statuses)
        self.assertEqual(track["coverage"]["operator_anchor_frames"], 2)
        metrics = result.analytics["targets"]["ball"]
        self.assertGreater(metrics["observed_frames"], 5)
        self.assertEqual(metrics["unit"], "px")
        self.assertEqual(metrics["space"], "2d_image")


class ReportWordingTests(PipelineTestCase):
    def test_the_report_has_the_documented_sections(self):
        result, out, _video, _config = self.run_fixture()
        report = (out / "demo_report.md").read_text()
        for heading in ("## VIDEO", "## SETUP", "## BODY ANALYSIS", "## SWING PHASES",
                        "## TEMPO", "## BALL / CLUB", "## LIMITATIONS", "## OUTPUTS"):
            self.assertIn(heading, report)

    def test_the_report_discloses_operator_input_and_says_it_is_not_detection(self):
        result, out, _video, _config = self.run_fixture()
        report = (out / "demo_report.md").read_text()
        self.assertIn("operator-placed", report)
        self.assertIn("not automatic detection", report)
        self.assertIn("OPERATOR-MARKED", report)
        self.assertIn("2D image space", report)

    def test_the_report_never_claims_a_physical_quantity(self):
        result, out, _video, _config = self.run_fixture()
        text = (out / "demo_report.md").read_text().lower()
        for forbidden in ("mph", "m/s", "rpm", "carry distance", "spin rate",
                          "detected automatically", "ground truth",
                          "clubhead speed of", "ball speed of"):
            self.assertNotIn(forbidden, text, forbidden)
        # the physical quantities appear only inside a negated statement
        for phrase in ("ball speed", "clubhead speed", "launch angle", "spin"):
            for line in text.splitlines():
                if phrase not in line:
                    continue
                self.assertTrue(any(marker in line for marker in
                                    ("no ", "not ", "never", "unavailable")),
                                f"{phrase} claimed without a negation: {line!r}")

    def test_an_unmarked_phase_is_reported_unavailable_not_invented(self):
        result, out, _video, _config = self.run_fixture(
            keyframes={"address": 0, "finish": 30})
        report = (out / "demo_report.md").read_text()
        self.assertIn("- **impact** — UNAVAILABLE", report)
        self.assertIn("no operator keyframe was supplied", report)
        tempo = result.phases["tempo"]
        self.assertEqual(tempo["state"], "unavailable")

    def test_the_phase_lines_carry_the_angles_the_json_sibling_carries(self):
        """A phase line that says 'no 2D angles' while the JSON has them is a bug."""
        result, out, _video, _config = self.run_fixture()
        report = (out / "demo_report.md").read_text()
        phases = json.loads((out / "demo_analysis.json").read_text())["phases"]["phases"]
        for name, info in phases.items():
            if info.get("frame") is None:
                continue
            line = next(ln for ln in report.splitlines()
                        if ln.startswith(f"- **{name}** —"))
            has_angle = any(info.get("angles_2d", {}).get(key) is not None
                            for key in ("shoulder_tilt_deg", "hip_tilt_deg"))
            if has_angle:
                self.assertNotIn("no 2D angle was measurable", line, line)
            else:
                self.assertIn("no 2D angle was measurable", line, line)
        self.assertNotIn("no 2D angles available", report)

    def test_the_missingness_counts_are_not_nested_and_the_reasons_are_grouped(self):
        """'55 missing (91 lost)' was a nested-count contradiction in the report."""
        result, out, _video, _config = self.run_fixture()
        report = (out / "demo_report.md").read_text()
        targets = result.analytics["missingness"]["targets"]
        self.assertTrue(targets)
        for name, info in targets.items():
            line = next(ln for ln in report.splitlines()
                        if ln.startswith(f"- {name}: "))
            self.assertIn(f"{info['unobserved_frames']} unobserved", line)
            self.assertIn(f"{info['missing_frames']} with no bounded match", line)
            self.assertIn(f"{info['lost_frames']} declared lost", line)
            self.assertIn("disjoint, not nested", line)
        reasons = result.analytics["missingness"]["body_missing_reasons"]
        self.assertTrue(reasons, "the fixture should have some missing body frames")
        examples = result.analytics["missingness"]["body_missing_reason_examples"]
        for key, count in reasons.items():
            self.assertEqual(len(examples[key]) > 0, True)
        # the grouped key masks the measured numbers, so per-frame percentages cannot
        # split one kind of reason into several histogram buckets
        self.assertFalse(any(any(ch.isdigit() for ch in key) for key in reasons),
                         reasons)
        # ...and the report shows the measured text, not a Python dict
        self.assertNotIn("{'", report.split("### Missingness")[1])

    def test_the_summary_is_short_and_carries_the_disclosure(self):
        result, out, _video, _config = self.run_fixture()
        summary = (out / "demo_summary.md").read_text()
        self.assertIn("Operator-assisted", summary)
        self.assertIn("the seeds, keyframes and corrections are operator input",
                      summary)
        # a two-screen summary, not a report: the project paragraph, the clip, the
        # limitations and the adaptable post paragraph
        self.assertLess(len(summary.splitlines()), 40)
        self.assertIn("## What it does not claim", summary)
        self.assertIn("## If you post this", summary)

    def test_the_summary_can_be_skipped(self):
        result, out, _video, _config = self.run_fixture(summary=False)
        self.assertFalse((out / "demo_summary.md").exists())

    def test_report_and_summary_render_from_the_same_result_object(self):
        result = DemoResult(
            source={"display_name": "c.mp4", "path": "", "sha256": "a" * 64,
                    "width": 640, "height": 480, "fps": 25.0, "nominal_fps": 25.0,
                    "duration_seconds": 4.0, "frame_count": 100,
                    "pixel_aspect_ratio": 1.0, "container_format": "mp4",
                    "codec": "h264", "variable_cadence": False,
                    "cadence_note": "container cadence 25 fps"},
            interval={"start_frame": 0, "end_frame": 24, "frame_count": 25,
                      "endpoint_semantics": "half-open"},
            config={"sha256": "b" * 64, "path": "/tmp/c.json", "out_dir": "/tmp/out",
                    "roi_xywh": [1, 2, 3, 4], "seeds": {}, "corrections": {},
                    "keyframes": {}, "interpolation": {"enabled": True,
                                                       "max_gap_frames": 2},
                    "timing": {}, "render": {}, "handedness": "right",
                    "allow_approx_phases": False, "effective": {}},
            body={"state": "unavailable", "method": "none",
                  "coverage": {"frames_with_pose": 0, "frames_total": 25},
                  "limitations": ["no ROI was supplied"]},
            analytics={"body_2d": {"supported": [], "unavailable": [],
                                   "conventions": {}},
                       "targets": {}, "missingness": {}},
            phases={"phases": {}, "tempo": {"state": "unavailable",
                                            "reason": "no markers"},
                    "cards": [], "notes": []},
            targets={}, render={"written": False, "reason": "not requested"},
            keyframes={"cards": [], "unavailable": []},
            limitations=["no ROI was supplied, so the body layer is ambiguous"],
            created_utc="2026-09-21T00:00:00+00:00")
        report = render_report(result)
        summary = render_summary(result)
        self.assertIn("no ROI was supplied", report)
        self.assertIn("NOT PRODUCED", report)
        self.assertIn("no markers", report)
        self.assertIn("unavailable", summary)


class RenderFailureTests(PipelineTestCase):
    def test_a_failed_render_raises_and_is_still_documented(self):
        require_cv2(self)
        video = make_synthetic_clip(self.tmp, frames=12)
        config = write_config(self.tmp, {
            "schema_version": DEMO_CONFIG_SCHEMA,
            "interval": {"start_seconds": 0.0, "end_seconds": None,
                         "max_frames": 40},
            "roi": [80, 40, 160, 180],
            "seeds": {"ball": {"frame": 1, "x": 61, "y": 200}},
            "keyframes": {"address": 0, "top": 4, "impact": 8, "finish": 11},
            "render": {"max_width": 320, "crf": 30, "preset": "ultrafast"}})
        out = Path(self.tmp) / "out-noflac"
        original = shutil.which("ffmpeg")
        if not original:
            self.skipTest("ffmpeg is not installed, so its absence cannot be simulated")
        bin_dir = Path(self.tmp) / "bin"
        bin_dir.mkdir(exist_ok=True)
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            self.skipTest("ffprobe is not installed, so the probe could not run")
        os.symlink(str(ffprobe), str(bin_dir / "ffprobe"))
        with mock.patch.dict(os.environ, {"PATH": str(bin_dir)}, clear=False):
            with self.assertRaises(DemoRenderError) as caught:
                run_demo(str(video), str(config), str(out), cv_python=sys.executable)
        self.assertIn("ffmpeg", str(caught.exception))
        report = (out / "demo_report.md").read_text()
        self.assertIn("NOT PRODUCED", report)
        provenance = json.loads((out / "demo_provenance.json").read_text())
        self.assertFalse(provenance["render"]["written"])
        self.assertIn("ffmpeg", provenance["render"]["reason"])
        self.assertFalse(provenance["verification"]["render_ffprobe"]["checked"])

    def test_a_worker_that_reports_a_missing_video_fails_the_run(self):
        require_cv2(self)
        video = make_synthetic_clip(self.tmp, frames=12)
        config = write_config(self.tmp, {
            "schema_version": DEMO_CONFIG_SCHEMA,
            "interval": {"start_seconds": 0.0, "end_seconds": None,
                         "max_frames": 40},
            "roi": [80, 40, 160, 180],
            "seeds": {"ball": {"frame": 1, "x": 61, "y": 200}},
            "render": {"max_width": 320, "crf": 30, "preset": "ultrafast"}})
        out = Path(self.tmp) / "out-fake"

        from ghostcaddie.demo import paths as demo_paths
        real_runner = demo_paths.run_demo_worker

        def fake_runner(spec, interpreter):
            payload = real_runner(spec, interpreter)
            payload["render"] = {"written": False, "frames": 0,
                                 "reason": "encoder refused the stream"}
            return payload

        with self.assertRaises(DemoRenderError) as caught:
            run_demo(str(video), str(config), str(out), cv_python=sys.executable,
                     worker_runner=fake_runner)
        self.assertIn("encoder refused the stream", str(caught.exception))
        self.assertIn("NOT PRODUCED", (out / "demo_report.md").read_text())

    def test_verify_render_reports_problems_instead_of_passing_a_bad_file(self):
        require_cv2(self)
        result, out, _video, _config = self.run_fixture(frames=20)
        # the output may be WIDER than the source: the panel can sit in an external
        # gutter, so the contract is checked against the size the run reported
        canvas_wh = result.render["wh"]
        picture_wh = result.render["picture_wh"]
        self.assertEqual(picture_wh, [320, 240])
        self.assertGreaterEqual(canvas_wh[0], picture_wh[0])
        self.assertEqual(canvas_wh[1], picture_wh[1])
        good = verify_render(str(out / "demo_annotated.mp4"), expected_frames=20,
                             expected_fps=20.0, expected_wh=canvas_wh)
        self.assertTrue(good["ok"], good["problems"])
        good_picture = verify_render(str(out / "demo_annotated.mp4"), expected_frames=20,
                                     expected_fps=20.0, expected_wh=[320, 240])
        self.assertFalse(good_picture["ok"])       # the picture alone is not the file
        wrong = verify_render(str(out / "demo_annotated.mp4"), expected_frames=999,
                              expected_fps=20.0, expected_wh=canvas_wh)
        self.assertFalse(wrong["ok"])
        self.assertTrue(any("frames" in problem for problem in wrong["problems"]))
        missing = verify_render(str(out / "nope.mp4"), expected_frames=1,
                                expected_fps=20.0, expected_wh=canvas_wh)
        self.assertFalse(missing["ok"])

    def test_a_missing_source_or_config_is_an_error_before_any_work(self):
        with self.assertRaises(DemoError):
            run_demo(str(self.tmp / "nope.mp4"), str(self.tmp / "nope.json"),
                     str(self.tmp / "out"), cv_python=sys.executable)


class PickerTests(TempDirTestCase):
    def _state(self, **kwargs):
        from ghostcaddie.demo.picker import PickerState
        return PickerState("/tmp/synthetic.mp4", synthetic_probe(width=640, height=480,
                                                                fps=25.0,
                                                                frame_count=200),
                           **kwargs)

    def test_the_picker_saves_a_config_the_pipeline_accepts(self):
        state = self._state()
        state.goto(20)
        state.set_interval_start()
        state.goto(120)
        state.set_interval_end()
        self.assertTrue(state.set_roi(200, 80, 240, 320))
        state.goto(30)
        self.assertTrue(state.place_seed("ball", 320, 300))
        state.goto(90)
        self.assertTrue(state.place_correction("ball", 330, 302))
        state.goto(40)
        state.mark_phase("takeaway")
        state.goto(70)
        state.mark_phase("top")
        path = state.save(str(Path(self.tmp) / "picked.json"))
        parsed = parse_config(json.loads(Path(path).read_text()))
        resolved = validate_config(parsed, synthetic_probe(width=640, height=480,
                                                           fps=25.0, frame_count=200))
        self.assertEqual((resolved.start_frame, resolved.end_frame), (20, 120))
        self.assertEqual(parsed.seeds["ball"].frame, 30)
        self.assertEqual([p.frame for p in parsed.corrections["ball"]], [90])
        self.assertEqual(parsed.keyframes, {"takeaway": 40, "top": 70})
        self.assertEqual(parsed.roi, [200, 80, 240, 320])

    def test_the_picker_marks_operator_input_as_operator_input(self):
        state = self._state()
        state.goto(10)
        state.place_seed("clubhead", 100.0, 200.0)
        payload = state.to_config_dict()
        self.assertIn("operator", payload["seeds"]["clubhead"]["note"])
        self.assertEqual(payload["schema_version"], DEMO_CONFIG_SCHEMA)

    def test_an_out_of_frame_point_is_refused_and_said_so(self):
        state = self._state()
        state.goto(10)
        self.assertFalse(state.place_seed("ball", 5000.0, 10.0))
        self.assertNotIn("ball", state.seeds)
        self.assertTrue(any("outside the" in message for message in state.messages))

    def test_an_roi_that_does_not_fit_is_refused(self):
        state = self._state()
        self.assertFalse(state.set_roi(600, 100, 200, 200))
        self.assertIsNone(state.roi)

    def test_a_seed_cannot_collide_with_a_correction_at_the_same_frame(self):
        state = self._state()
        state.goto(25)
        state.place_seed("ball", 100.0, 100.0)
        self.assertTrue(state.place_correction("ball", 120.0, 120.0))
        self.assertNotIn("ball", state.seeds)
        self.assertEqual(len(state.corrections["ball"]), 1)

    def test_the_picker_writes_endpoint_semantics_it_can_be_held_to(self):
        state = self._state()
        state.goto(50)
        state.set_interval_end()
        payload = state.to_config_dict()
        self.assertIn("EXCLUSIVE", payload["notes"])
        # the written end_seconds is the frame AFTER the marked one, because the
        # interval is half-open: frame 50 inclusive is 51/25s = 2.04s
        self.assertIn("51", payload["notes"])
        self.assertAlmostEqual(payload["interval"]["end_seconds"], 51 / 25.0, places=6)

    def test_an_existing_config_round_trips_through_the_picker(self):
        payload = base_config()
        state = self._state(existing=parse_config(payload))
        self.assertEqual(state.roi, payload["roi"])
        self.assertEqual(state.seeds["ball"][0], payload["seeds"]["ball"]["frame"])
        self.assertEqual(state.keyframes["top"], payload["keyframes"]["top"])
        saved = state.to_config_dict()
        self.assertEqual(saved["keyframes"]["top"], payload["keyframes"]["top"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
