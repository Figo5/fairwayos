"""Bounded batching of fresh vision jobs over a contiguous native interval."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from ghostcaddie.video.fresh_ai_vision import (
    MAX_BATCH_FRAMES,
    detect_repeated_decisions,
    merge_batch_documents,
    parse_interval,
    partition_interval,
    reusable_batch_document,
    run_batched_pipeline,
    validate_inspection_evidence,
)

SHA = "a" * 64
OTHER_SHA = "b" * 64


def decision(frame, sha=SHA):
    return {"source_frame": frame, "source_sha256": sha,
            "ball": {"visible": False, "point_xy": None, "bbox_xyxy": None, "confidence": 0.0},
            "clubhead": {"visible": False, "point_xy": None, "bbox_xyxy": None, "confidence": 0.0}}


def batch_doc(frames, sha=SHA):
    return {"source": {"sha256": sha}, "requested_frames": list(frames),
            "decisions": [decision(f, sha) for f in frames]}


class PartitionTests(unittest.TestCase):
    def test_contiguous_interval_splits_into_bounded_batches(self):
        parts = partition_interval(3063, 3114, batch_size=32)
        self.assertEqual(parts[0], list(range(3063, 3095)))
        self.assertEqual(parts[-1], list(range(3095, 3115)))
        self.assertEqual([f for p in parts for f in p], list(range(3063, 3115)))
        self.assertTrue(all(len(p) <= MAX_BATCH_FRAMES for p in parts))

    def test_single_frame_and_exact_multiple_have_no_empty_tail(self):
        self.assertEqual(partition_interval(7, 7, batch_size=32), [[7]])
        parts = partition_interval(0, 63, batch_size=32)
        self.assertEqual(len(parts), 2)
        self.assertTrue(all(len(p) == 32 for p in parts))

    def test_partition_rejects_unbounded_or_invalid_intervals(self):
        for kwargs in ({"batch_size": 33}, {"batch_size": 0}, {"max_total_frames": 10}):
            with self.assertRaises(ValueError):
                partition_interval(0, 51, **kwargs)
        with self.assertRaises(ValueError):
            partition_interval(10, 9)
        with self.assertRaises(ValueError):
            partition_interval(-1, 5)
        with self.assertRaises(ValueError):
            partition_interval(True, 5)

    def test_parse_interval_accepts_colon_and_dash_forms(self):
        self.assertEqual(parse_interval("3063:3114"), (3063, 3114))
        self.assertEqual(parse_interval("3063-3114"), (3063, 3114))
        for bad in ("3063", "a:b", "3063:", "1:2:3", ""):
            with self.assertRaises(ValueError):
                parse_interval(bad)


class MergeTests(unittest.TestCase):
    def test_merge_preserves_native_order_and_continuity(self):
        frames = list(range(10, 20))
        merged = merge_batch_documents(
            [batch_doc(range(10, 15)), batch_doc(range(15, 20))],
            requested_frames=frames, source_sha256=SHA)
        self.assertEqual([d["source_frame"] for d in merged], frames)

    def test_merge_rejects_missing_duplicate_extra_or_foreign_source(self):
        with self.assertRaises(ValueError):  # missing 14
            merge_batch_documents([batch_doc([10, 11, 12, 13])],
                                  requested_frames=list(range(10, 15)), source_sha256=SHA)
        with self.assertRaises(ValueError):  # duplicate 12 across batches
            merge_batch_documents([batch_doc([10, 11, 12]), batch_doc([12, 13])],
                                  requested_frames=list(range(10, 14)), source_sha256=SHA)
        with self.assertRaises(ValueError):  # extra 99
            merge_batch_documents([batch_doc([10, 11, 99])],
                                  requested_frames=[10, 11], source_sha256=SHA)
        with self.assertRaises(ValueError):  # batch decoded from another source
            merge_batch_documents([batch_doc([10, 11], sha=OTHER_SHA)],
                                  requested_frames=[10, 11], source_sha256=SHA)

    def test_merge_rejects_decision_whose_own_sha_drifts(self):
        doc = batch_doc([10, 11])
        doc["decisions"][1]["source_sha256"] = OTHER_SHA
        with self.assertRaises(ValueError):
            merge_batch_documents([doc], requested_frames=[10, 11], source_sha256=SHA)


class ResumeReuseTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.p = self.d / "fresh_ai_vision_results.json"

    def test_reuses_only_content_hash_and_frame_matched_document(self):
        self.p.write_text(json.dumps(batch_doc([10, 11, 12])))
        self.assertIsNotNone(reusable_batch_document(self.p, frames=[10, 11, 12], source_sha256=SHA))
        self.assertIsNone(reusable_batch_document(self.p, frames=[10, 11, 12], source_sha256=OTHER_SHA))
        self.assertIsNone(reusable_batch_document(self.p, frames=[10, 11], source_sha256=SHA))
        self.assertIsNone(reusable_batch_document(self.d / "missing.json", frames=[10], source_sha256=SHA))

    def test_corrupt_or_partial_document_is_not_reused(self):
        self.p.write_text("{not json")
        self.assertIsNone(reusable_batch_document(self.p, frames=[10], source_sha256=SHA))
        doc = batch_doc([10, 11])
        doc["decisions"] = doc["decisions"][:1]
        self.p.write_text(json.dumps(doc))
        self.assertIsNone(reusable_batch_document(self.p, frames=[10, 11], source_sha256=SHA))


class BatchedFixture:
    """Drives run_batched_pipeline with the per-batch job faked; no child inference."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.video = self.d / "src.mp4"
        self.video.write_bytes(b"x")
        self.meta = {"width": 1280, "height": 720, "r_frame_rate": "30000/1001",
                     "avg_frame_rate": "30000/1001", "nb_frames": 4000,
                     "duration": 133.0, "codec_name": "h264", "pix_fmt": "yuv420p"}

    def patched(self, smoke):
        return mock.patch.multiple(
            "ghostcaddie.video.fresh_ai_vision",
            sha256_file=mock.Mock(return_value=SHA),
            ffprobe_video=mock.Mock(return_value=self.meta),
            run_smoke_pipeline=mock.Mock(side_effect=smoke),
        )

    def _fake_smoke(self, fail_frames=()):
        calls = []

        def smoke(video, frames, outdir, *, render=True, max_turns=8, budget_s=None):
            calls.append(list(frames))
            if list(frames) == list(fail_frames):
                raise RuntimeError("child failed")
            self.assertFalse(render, "per-batch jobs must not render")
            Path(outdir).mkdir(parents=True, exist_ok=True)
            (Path(outdir) / "fresh_ai_vision_results.json").write_text(json.dumps(batch_doc(frames)))
            return {"ok": True, "outputs": {}, "routing": {"returncode": 0}}

        return smoke, calls


class BatchedPipelineTests(BatchedFixture, unittest.TestCase):
    def test_batches_sequentially_and_merges_one_document(self):
        smoke, calls = self._fake_smoke()
        with self.patched(smoke):
            res = run_batched_pipeline(self.video, 100, 103, self.d / "out",
                                       batch_size=2, render=False)
        self.assertEqual(calls, [[100, 101], [102, 103]])
        doc = json.loads(Path(res["outputs"]["decisions_json"]).read_text())
        self.assertEqual(doc["requested_frames"], [100, 101, 102, 103])
        self.assertEqual([d["source_frame"] for d in doc["decisions"]], [100, 101, 102, 103])
        self.assertEqual(doc["interval_native_frames"], [100, 103])
        self.assertEqual(doc["source"]["sha256"], SHA)
        self.assertFalse(doc["metric_speed"]["available"])
        self.assertEqual(len(doc["batching"]["batches"]), 2)

    def test_partial_batch_failure_fails_closed_and_keeps_completed_work(self):
        smoke, calls = self._fake_smoke(fail_frames=[102, 103])
        with self.patched(smoke):
            with self.assertRaises(RuntimeError):
                run_batched_pipeline(self.video, 100, 105, self.d / "out",
                                     batch_size=2, render=False)
        out = self.d / "out"
        self.assertFalse((out / "fresh_ai_vision_results.json").exists(),
                         "no merged document may be written from a partial run")
        status = json.loads((out / "batch_status.json").read_text())
        self.assertEqual([b["status"] for b in status["batches"]][:2], ["fresh", "failed"])
        self.assertEqual(calls, [[100, 101], [102, 103]], "must stop after the failing batch")

    def test_resume_reuses_source_bound_batches_and_reruns_the_rest(self):
        smoke, calls = self._fake_smoke(fail_frames=[102, 103])
        with self.patched(smoke):
            with self.assertRaises(RuntimeError):
                run_batched_pipeline(self.video, 100, 103, self.d / "out", batch_size=2, render=False)
        smoke2, calls2 = self._fake_smoke()
        with self.patched(smoke2):
            res = run_batched_pipeline(self.video, 100, 103, self.d / "out",
                                       batch_size=2, render=False, resume=True)
        self.assertEqual(calls2, [[102, 103]], "the completed batch must not be re-inferred")
        doc = json.loads(Path(res["outputs"]["decisions_json"]).read_text())
        self.assertEqual([b["status"] for b in doc["batching"]["batches"]], ["reused", "fresh"])

    def test_default_is_fresh_inference_not_reuse(self):
        smoke, _ = self._fake_smoke()
        with self.patched(smoke):
            run_batched_pipeline(self.video, 100, 101, self.d / "out", batch_size=2, render=False)
        smoke2, calls2 = self._fake_smoke()
        with self.patched(smoke2):
            run_batched_pipeline(self.video, 100, 101, self.d / "out", batch_size=2, render=False)
        self.assertEqual(calls2, [[100, 101]], "without --resume every batch must re-infer")

    def test_interval_outside_source_frame_count_is_rejected(self):
        smoke, _ = self._fake_smoke()
        with self.patched(smoke):
            with self.assertRaises(ValueError):
                run_batched_pipeline(self.video, 3990, 4010, self.d / "out", render=False)


class StaleOverlayTests(unittest.TestCase):
    def test_render_clears_previous_marked_frames(self):
        from ghostcaddie.video.fresh_ai_vision import render_video

        d = Path(tempfile.mkdtemp())
        stale = d / "render_frames_marked"
        stale.mkdir(parents=True)
        for i in range(5):
            (stale / f"seq_{i:06d}.jpg").write_bytes(b"stale")
        meta = {"width": 8, "height": 8, "r_frame_rate": "30/1"}
        img = mock.MagicMock()
        cv2 = mock.MagicMock()
        cv2.imread.return_value = img
        with mock.patch.dict("sys.modules", {"cv2": cv2}), \
                mock.patch("ghostcaddie.video.fresh_ai_vision.decode_frames",
                           return_value={0: d / "a.jpg", 1: d / "b.jpg"}), \
                mock.patch("ghostcaddie.video.fresh_ai_vision._run_checked"):
            render_video(d / "src.mp4", [
                {"source_frame": 0, "ball": {"visible": False}, "clubhead": {"visible": False}},
                {"source_frame": 1, "ball": {"visible": False}, "clubhead": {"visible": False}},
            ], meta, d)
        written = {c.args[0] for c in cv2.imwrite.call_args_list}
        self.assertEqual(len(written), 2)
        leftovers = [p.name for p in stale.glob("seq_*.jpg") if p.read_bytes() == b"stale"]
        self.assertEqual(leftovers, [], "stale overlay frames must not survive into the render")


if __name__ == "__main__":
    unittest.main()


class DeadlineTests(BatchedFixture, unittest.TestCase):
    """A bounded run must never report success after its own wall-clock bound."""

    def _slow_smoke(self, seconds):
        seen = []

        def smoke(video, frames, outdir, *, render=True, max_turns=8, budget_s=None):
            seen.append({"frames": list(frames), "budget_s": budget_s})
            time.sleep(seconds)
            Path(outdir).mkdir(parents=True, exist_ok=True)
            (Path(outdir) / "fresh_ai_vision_results.json").write_text(json.dumps(batch_doc(frames)))
            return {"ok": True, "outputs": {}}

        return smoke, seen

    def test_successful_batch_overrunning_total_deadline_fails_closed(self):
        smoke, seen = self._slow_smoke(0.05)
        out = self.d / "out"
        with self.patched(smoke):
            with self.assertRaises(TimeoutError) as ctx:
                run_batched_pipeline(self.video, 0, 1, out, batch_size=2,
                                     deadline_seconds=0.001, render=False)
        self.assertIn("deadline_seconds", str(ctx.exception))
        self.assertFalse((out / "fresh_ai_vision_results.json").exists(),
                         "an overrunning run must not publish a merged document")
        status = json.loads((out / "batch_status.json").read_text())
        self.assertTrue(status["deadline_exceeded"])
        self.assertEqual(status["batches"][0]["status"], "fresh")
        self.assertTrue(Path(status["batches"][0]["outdir"], "fresh_ai_vision_results.json").exists(),
                        "the completed batch must be preserved for resume")

    def test_remaining_budget_shrinks_for_each_later_batch(self):
        smoke, seen = self._slow_smoke(0.05)
        with self.patched(smoke):
            run_batched_pipeline(self.video, 0, 3, self.d / "out", batch_size=2,
                                 deadline_seconds=5.0, render=False)
        self.assertEqual(len(seen), 2)
        self.assertLess(seen[1]["budget_s"], seen[0]["budget_s"],
                        "each batch must be handed only the time actually left")
        self.assertLessEqual(seen[0]["budget_s"], 5.0)

    def test_exhausted_budget_stops_before_starting_the_next_batch(self):
        smoke, seen = self._slow_smoke(0.08)
        with self.patched(smoke):
            with self.assertRaises((TimeoutError, RuntimeError)):
                run_batched_pipeline(self.video, 0, 5, self.d / "out", batch_size=2,
                                     deadline_seconds=0.1, render=False)
        self.assertLess(len(seen), 3, "batches must not start with no budget left")


class ParallelAbortTests(BatchedFixture, unittest.TestCase):
    def test_known_failure_does_not_wait_for_slow_sibling_batches(self):
        started_at = []

        def smoke(video, frames, outdir, *, render=True, max_turns=8, budget_s=None):
            started_at.append(list(frames))
            if frames[0] == 0:
                time.sleep(1.5)
                Path(outdir).mkdir(parents=True, exist_ok=True)
                (Path(outdir) / "fresh_ai_vision_results.json").write_text(json.dumps(batch_doc(frames)))
                return {"ok": True, "outputs": {}}
            raise RuntimeError("synthetic fast failure")

        t0 = time.perf_counter()
        with self.patched(smoke):
            with self.assertRaises(RuntimeError) as ctx:
                run_batched_pipeline(self.video, 0, 3, self.d / "out", batch_size=2,
                                     concurrency=2, deadline_seconds=30, render=False)
        elapsed = time.perf_counter() - t0
        self.assertIn("synthetic fast failure", str(ctx.exception))
        self.assertLess(elapsed, 1.0,
                        f"aborted after {elapsed:.3f}s; a known failure must not wait on slow siblings")
        self.assertFalse((self.d / "out" / "fresh_ai_vision_results.json").exists())

    def test_abort_terminates_in_flight_child_process_groups(self):
        from ghostcaddie.video import fresh_ai_vision as fav

        result = {}

        def run_child():
            try:
                fav._run_bounded_process(["sleep", "60"], timeout_s=60)
                result["outcome"] = "completed"
            except Exception as exc:
                result["outcome"] = type(exc).__name__

        thread = threading.Thread(target=run_child, daemon=True)
        thread.start()
        for _ in range(200):  # wait for the child to register its process group
            if fav._LIVE_PGIDS:
                break
            time.sleep(0.01)
        self.assertTrue(fav._LIVE_PGIDS, "a bounded child must register its process group")
        self.assertGreaterEqual(fav.terminate_live_children(), 1)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive(), "terminate_live_children must actually free the blocked thread")
        self.assertEqual(fav._LIVE_PGIDS, set(), "the registry must not leak process groups")


class JobBudgetTests(unittest.TestCase):
    def test_job_budget_caps_child_and_body_timeouts(self):
        from ghostcaddie.video import fresh_ai_vision as fav

        d = Path(tempfile.mkdtemp())
        video = d / "src.mp4"
        video.write_bytes(b"x")
        seen = {}

        def fake_child(query_file, outdir, max_turns, *, job_nonce=None, timeout_s=900):
            seen["child"] = timeout_s
            return ({"job_nonce": job_nonce,
                     "frames": [{"source_frame": 0,
                                 "inspected_images": [str(d / "f.jpg")],
                                 "ball": {"visible": False, "confidence": 0.0},
                                 "clubhead": {"visible": False, "confidence": 0.0}}]}, {"returncode": 0})

        def fake_body(video, frames, sha, outdir, *, registry=None, timeout_s=300):
            seen["body"] = timeout_s
            return {"state": "unavailable", "records": []}

        with mock.patch.multiple(
            "ghostcaddie.video.fresh_ai_vision",
            sha256_file=mock.Mock(return_value=SHA),
            ffprobe_video=mock.Mock(return_value={"width": 640, "height": 360, "nb_frames": 100, "r_frame_rate": "30/1"}),
            decode_frames=mock.Mock(return_value={0: d / "f.jpg"}),
            run_child_inference=mock.Mock(side_effect=fake_child),
            run_body_pose_inference=mock.Mock(side_effect=fake_body),
        ):
            fav.run_smoke_pipeline(video, [0], d / "job", render=False, budget_s=12.0)
        self.assertLessEqual(seen["child"], 12.0)
        self.assertLessEqual(seen["body"], 12.0)

    def test_unbudgeted_job_keeps_its_default_bounds(self):
        from ghostcaddie.video import fresh_ai_vision as fav

        d = Path(tempfile.mkdtemp())
        video = d / "src.mp4"
        video.write_bytes(b"x")
        seen = {}

        def fake_child(query_file, outdir, max_turns, *, job_nonce=None, timeout_s=900):
            seen["child"] = timeout_s
            return ({"job_nonce": job_nonce,
                     "frames": [{"source_frame": 0,
                                 "inspected_images": [str(d / "f.jpg")],
                                 "ball": {"visible": False, "confidence": 0.0},
                                 "clubhead": {"visible": False, "confidence": 0.0}}]}, {"returncode": 0})

        with mock.patch.multiple(
            "ghostcaddie.video.fresh_ai_vision",
            sha256_file=mock.Mock(return_value=SHA),
            ffprobe_video=mock.Mock(return_value={"width": 640, "height": 360, "nb_frames": 100, "r_frame_rate": "30/1"}),
            decode_frames=mock.Mock(return_value={0: d / "f.jpg"}),
            run_child_inference=mock.Mock(side_effect=fake_child),
            run_body_pose_inference=mock.Mock(return_value={"state": "unavailable", "records": []}),
        ):
            fav.run_smoke_pipeline(video, [0], d / "job", render=False)
        self.assertEqual(seen["child"], 900.0)


class InspectionEvidenceTests(unittest.TestCase):
    """Self-reported attribution only: an unattributed decision fails closed, never gets a coordinate.

    These cover what the contract can check - that an answer names its own frame and not a sibling's.
    None of them establish that a vision call was made; a claimed path is accepted on the child's word.
    """

    def setUp(self):
        self.paths = {10: Path("/tmp/j/native_000010.jpg"), 11: Path("/tmp/j/native_000011.jpg")}

    def test_accepts_a_decision_citing_its_own_frame_image(self):
        got = validate_inspection_evidence(
            {"source_frame": 10, "inspected_images": [str(self.paths[10]), "/tmp/j/crop_a.jpg"]},
            frame_paths=self.paths)
        self.assertIn(str(self.paths[10]), got)

    def test_rejects_decision_that_never_cites_its_own_frame(self):
        with self.assertRaises(ValueError) as ctx:
            validate_inspection_evidence(
                {"source_frame": 10, "inspected_images": ["/tmp/j/crop_a.jpg"]}, frame_paths=self.paths)
        self.assertIn("own image", str(ctx.exception))

    def test_rejects_decision_borrowing_another_requested_frames_image(self):
        with self.assertRaises(ValueError) as ctx:
            validate_inspection_evidence(
                {"source_frame": 10, "inspected_images": [str(self.paths[10]), str(self.paths[11])]},
                frame_paths=self.paths)
        self.assertIn("another requested frame", str(ctx.exception))

    def test_rejects_missing_empty_or_non_string_evidence(self):
        for bad in (None, [], "a string", [123], {"a": 1}):
            with self.assertRaises(ValueError):
                validate_inspection_evidence({"source_frame": 10, "inspected_images": bad},
                                             frame_paths=self.paths)


class RepetitionWarningTests(unittest.TestCase):
    """Repetition is surfaced as a warning to inspect, never treated as proof of copying."""

    def _dec(self, frame, point, note):
        return {"source_frame": frame,
                "ball": {"point_xy": point, "bbox_xyxy": None, "uncertainty": note},
                "clubhead": {"point_xy": None, "bbox_xyxy": None, "uncertainty": note}}

    def test_warns_when_every_frame_of_a_job_repeats_verbatim(self):
        decisions = [self._dec(f, [834.0, 563.0], "same prose") for f in range(3008, 3012)]
        flags = detect_repeated_decisions(decisions)["repeated_across_all_frames"]
        self.assertTrue(flags["ball"])
        self.assertTrue(flags["clubhead"])

    def test_does_not_flag_genuinely_per_frame_decisions(self):
        decisions = [self._dec(3008, [834.0, 563.0], "a"), self._dec(3009, [836.0, 561.0], "b")]
        self.assertFalse(detect_repeated_decisions(decisions)["repeated_across_all_frames"]["ball"])

    def test_single_frame_job_cannot_repeat_so_its_clear_flag_is_uninformative(self):
        out = detect_repeated_decisions([self._dec(3008, [1.0, 2.0], "x")])
        self.assertFalse(out["repeated_across_all_frames"]["ball"])
        self.assertFalse(out["repeated_across_all_frames"]["clubhead"])
        self.assertFalse(out["informative"], "one frame cannot repeat; absence of the flag proves nothing")

    def test_frozen_run_raises_the_warning(self):
        """The frozen four-frame jobs repeat verbatim. The warning fires; the cause stays undetermined."""
        frozen = Path("/tmp/fairway-parent-real-batched-smoke/batch_001_3008_3011/fresh_ai_vision_results.json")
        if not frozen.exists():
            self.skipTest("frozen reference run not present on this machine")
        decisions = json.loads(frozen.read_text())["decisions"]
        flags = detect_repeated_decisions(decisions)["repeated_across_all_frames"]
        self.assertTrue(flags["clubhead"], "the frozen four-frame job repeats verbatim and must be flagged for inspection")
