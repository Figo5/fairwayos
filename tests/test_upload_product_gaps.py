"""Behavioural tests for the four product gaps found in the release recheck.

None of these is a security regression: the original security fixes stay in
force and are asserted where they touch the same code.

  1. HTTP seeds were accepted, stored, and never reached the worker. The job
     thread started at upload time and ran to completion before a seed could
     possibly be posted, so the seeded targets could only ever fail. Running the
     models from the CLI does NOT make the HTTP path work, and must not be
     reported as if it did.
  2. Uploaded media was deleted the moment the job reached a terminal state,
     which is exactly when frame review and playback need it. Retention must
     still be explicit and BOUNDED -- kept for the review window, dropped on
     request, on cancel, or when the window expires.
  3. The UI read a /ready schema that no longer exists (`runtimes`/`ready`),
     so the readiness card silently failed.
  4. /runtime reported safe_to_run true for targets /ready reported as not
     executable. Load-path safety and executability are different claims and
     must not contradict each other.
"""
import json, os, tempfile, threading, time, unittest, urllib.error, urllib.parse, urllib.request

from ghostcaddie.upload.server import build_server
from ghostcaddie.upload.validation import VideoLimits


def synth(path, seconds=0.4, w=320, h=240, fps=30):
    """EXPLICITLY SYNTHETIC fixture. Never a user-facing result."""
    import cv2, numpy as np
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(int(seconds * fps)):
        vw.write(np.full((h, w, 3), (i * 7) % 255, np.uint8))
    vw.release()
    return path


class _Base(unittest.TestCase):
    RETENTION_SECONDS = 3600

    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        cls.srv = build_server(os.path.join(cls.d, "root"), port=0,
                               limits=VideoLimits(max_seconds=60),
                               media_retention_seconds=cls.RETENTION_SECONDS)
        cls.host, cls.port = cls.srv.server_address[0], cls.srv.server_address[1]
        cls.H = cls.srv.RequestHandlerClass
        cls.token = cls.H.csrf_token
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def url(self, p):
        return f"http://{self.host}:{self.port}{p}"

    def get(self, p, raw=False):
        try:
            with urllib.request.urlopen(self.url(p)) as r:
                body = r.read()
                return r.status, (body if raw else json.loads(body or b"null"))
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def post(self, p, payload=None, form=None):
        if form is not None:
            data, ctype = urllib.parse.urlencode(form).encode(), \
                          "application/x-www-form-urlencoded"
        else:
            data, ctype = json.dumps(payload or {}).encode(), "application/json"
        req = urllib.request.Request(
            self.url(p), data=data,
            headers={"X-FairwayOS-Token": self.token, "Content-Type": ctype})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"null")

    def upload(self, name="clip.mp4"):
        """Real upload BYTES through multipart, so the media is ours to delete.

        The import-directory route analyses the operator's own file in place and
        that file must never be deleted; media deletion is therefore exercised
        on genuinely uploaded bytes.
        """
        import tempfile, uuid
        src = os.path.join(tempfile.mkdtemp(), name)
        synth(src)
        boundary = "----ghostcaddie" + uuid.uuid4().hex
        with open(src, "rb") as fh:
            payload = fh.read()
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"video\"; "
                f"filename=\"{name}\"\r\nContent-Type: video/mp4\r\n\r\n"
                ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            self.url("/jobs"), data=body,
            headers={"X-FairwayOS-Token": self.token,
                     "Content-Type": f"multipart/form-data; boundary={boundary}",
                     "Content-Length": str(len(body))})
        with urllib.request.urlopen(req) as r:
            job = json.loads(r.read())
            self.assertEqual(r.status, 202, job)
        for _ in range(200):
            s, d = self.get(f"/jobs/{job['id']}")
            if d["state"] not in ("queued", "running"):
                return d
            time.sleep(0.1)
        self.fail("job never left running")


class SeedBindingTests(_Base):
    def test_a_seed_bound_to_another_source_is_still_refused(self):
        """Unchanged security rule: assistance may not be transplanted."""
        d = self.upload("binding.mp4")
        st, body = self.post(f"/jobs/{d['id']}/seeds",
                             {"source_sha256": "0" * 64,
                              "seeds": [{"target": "clubhead", "frame": 1,
                                         "box_xyxy": [1.0, 1.0, 5.0, 5.0]}]})
        self.assertEqual(st, 400)
        self.assertIn("different source", body["error"])

    def test_a_seed_post_without_the_csrf_token_is_refused(self):
        d = self.upload("csrf.mp4")
        req = urllib.request.Request(
            self.url(f"/jobs/{d['id']}/seeds"), data=b"{}",
            headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        self.assertEqual(e.exception.code, 403)


class SeedConsumptionTests(_Base):
    BOX = [10.0, 10.0, 30.0, 30.0]

    def seed_payload(self, job):
        """Seeds stay BOUND to the uploaded source hash. That security rule is
        unchanged: a bundle naming a different source is still refused."""
        return {"source_sha256": job["result"]["source"]["sha256"],
                "seeds": [{"target": "clubhead", "frame": 3, "box_xyxy": self.BOX}]}

    def test_upload_waits_for_seeds_instead_of_finishing_without_them(self):
        """The old path finished the whole job before a seed could be posted."""
        d = self.upload("waits.mp4")
        self.assertEqual(d["state"], "awaiting_seeds", d)
        self.assertIn("clubhead", d["awaiting"])
        self.assertIn("ball", d["awaiting"])

    def test_unseeded_targets_still_run_in_the_first_phase(self):
        d = self.upload("phase1.mp4")
        self.assertIn("body", d["result"]["targets"])
        self.assertNotEqual(d["result"]["targets"]["body"]["outcome"], "not_run")

    def test_a_posted_seed_actually_reaches_the_worker_request(self):
        """Wiring, not model success: the seed must appear in what the worker
        was asked to run. Whether the model then finds anything is a separate,
        environment-dependent question this test does not claim."""
        d = self.upload("seeded.mp4")
        jid = d["id"]
        st, body = self.post(f"/jobs/{jid}/seeds", self.seed_payload(d))
        self.assertEqual(st, 200, body)
        self.assertIn("ASSISTANCE", body["note"])
        for _ in range(600):
            s, j = self.get(f"/jobs/{jid}")
            if j["state"] in ("done", "failed", "cancelled"):
                break
            time.sleep(0.1)
        else:
            self.fail("seeded phase never finished")
        club = j["result"]["targets"]["clubhead"]
        self.assertEqual(club["seed_used"]["box_xyxy"], self.BOX)
        self.assertIn("not human-verified", club["assisted_disclosure"].lower())
        self.assertEqual(club["initialization"], "assisted")

    def test_a_target_with_no_seed_says_so_rather_than_claiming_cli_success(self):
        d = self.upload("partial.mp4")
        jid = d["id"]
        self.post(f"/jobs/{jid}/seeds", self.seed_payload(d))   # clubhead only
        for _ in range(600):
            s, j = self.get(f"/jobs/{jid}")
            if j["state"] in ("done", "failed", "cancelled"):
                break
            time.sleep(0.1)
        ball = j["result"]["targets"]["ball"]
        self.assertIsNone(ball["seed_used"])
        self.assertNotEqual(ball["outcome"], "observed")
        self.assertIn("seed", ball["reason"].lower())


class MediaRetentionTests(_Base):
    def test_media_survives_the_first_phase_so_review_and_playback_work(self):
        d = self.upload("review.mp4")
        st, _ = self.get(f"/video?job={d['id']}", raw=True)
        self.assertEqual(st, 200, "media was deleted before it could be reviewed")
        st, _ = self.get(f"/frame?job={d['id']}&n=2", raw=True)
        self.assertEqual(st, 200)

    def test_retention_is_reported_with_an_explicit_deadline(self):
        d = self.upload("deadline.mp4")
        r = d["media_retention"]
        self.assertTrue(r["retained"])
        self.assertEqual(r["retention_seconds"], self.RETENTION_SECONDS)
        self.assertGreater(r["expires_at"], time.time())

    def test_media_can_be_dropped_explicitly_and_is_then_really_gone(self):
        d = self.upload("drop.mp4")
        jid = d["id"]
        st, body = self.post(f"/jobs/{jid}/media/delete")
        self.assertEqual(st, 200)
        self.assertFalse(body["media_retention"]["retained"])
        st, _ = self.get(f"/video?job={jid}", raw=True)
        self.assertEqual(st, 410)

    def test_cancel_still_removes_the_media(self):
        d = self.upload("cancelled.mp4")
        jid = d["id"]
        self.post(f"/jobs/{jid}/cancel")
        st, _ = self.get(f"/video?job={jid}", raw=True)
        self.assertIn(st, (404, 410))


class ExpiredRetentionTests(_Base):
    RETENTION_SECONDS = 0            # everything is already past its window

    def test_media_past_its_retention_window_is_dropped_on_access(self):
        """Retention stays BOUNDED: waiting for a seed cannot keep media forever."""
        d = self.upload("expired.mp4")
        st, _ = self.get(f"/video?job={d['id']}", raw=True)
        self.assertEqual(st, 410)
        src = self.H.store.source(d["id"])
        self.assertFalse(os.path.exists(src.path),
                         "uploaded bytes past their window must be deleted")


class ReadinessContractTests(_Base):
    def test_the_ui_reads_the_schema_ready_actually_serves(self):
        st, page = self.get("/", raw=True)
        page = page.decode()
        self.assertIn("dependency_readiness", page)
        self.assertIn("can_execute_now", page)
        self.assertNotIn("d.runtimes", page, "UI still reads the removed schema")

    def test_ready_serves_every_field_the_ui_consumes(self):
        st, d = self.get("/ready")
        for t, v in d["dependency_readiness"].items():
            for k in ("interpreter_ready", "can_execute_now", "reason",
                      "requires_reviewed_seed"):
                self.assertIn(k, v, f"{t} missing {k}")

    def test_runtime_and_ready_never_contradict_each_other(self):
        _, rt = self.get("/runtime")
        _, rd = self.get("/ready")
        for t, v in rd["dependency_readiness"].items():
            if t in rt:
                self.assertEqual(rt[t]["can_execute_unattended"], v["can_execute_now"],
                                 f"{t}: /runtime and /ready disagree")

    def test_runtime_separates_load_path_safety_from_executability(self):
        """safe_to_run answers 'is the load path safe', not 'can this run now'."""
        _, rt = self.get("/runtime")
        for t, v in rt.items():
            self.assertIn("safe_to_run", v)
            self.assertIn("can_execute_unattended", v)
            if v.get("requires_reviewed_seed"):
                self.assertFalse(v["can_execute_unattended"], t)


if __name__ == "__main__":
    unittest.main()


class RoutingAndParameterTests(_Base):
    """Defects a static UI read suggested, reproduced as BEHAVIOUR.

    Each test below fails against the old code for a real reason, not because a
    source string changed.
    """

    def test_a_frame_request_without_n_is_refused_not_silently_frame_zero(self):
        """A caller asking for ?frame=2 used to receive frame 0 with HTTP 200.

        Silently serving a different frame than the one asked for is the worst
        possible failure for a review UI: the operator seeds a coordinate while
        looking at the wrong picture.
        """
        d = self.upload("frameparam.mp4")
        st, body = self.get(f"/frame?job={d['id']}&frame=2", raw=True)
        self.assertEqual(st, 400, "unknown frame parameter must be refused")

    def test_the_served_frame_is_the_frame_that_was_asked_for(self):
        d = self.upload("framecheck.mp4")
        req = urllib.request.Request(self.url(f"/frame?job={d['id']}&n=5"))
        with urllib.request.urlopen(req) as r:
            self.assertEqual(r.status, 200)
            self.assertEqual(r.headers["X-Requested-Frame"], "5")
            self.assertEqual(r.headers["X-Decoded-Frame"], "5")

    def test_mutating_routes_still_work_with_a_query_string(self):
        """Routing matched on the whole path, so /jobs/<id>/seeds?x=1 missed
        every handler and fell through."""
        d = self.upload("queryroute.mp4")
        jid = d["id"]
        st, body = self.post(f"/jobs/{jid}/seeds?ts=1", self.seed_payload(d)
                             if hasattr(self, "seed_payload") else
                             {"source_sha256": d["result"]["source"]["sha256"],
                              "seeds": [{"target": "clubhead", "frame": 3,
                                         "box_xyxy": [10.0, 10.0, 30.0, 30.0]}]})
        self.assertEqual(st, 200, body)

    def test_media_delete_route_works_with_a_query_string(self):
        d = self.upload("querydel.mp4")
        st, body = self.post(f"/jobs/{d['id']}/media/delete?ts=1")
        self.assertEqual(st, 200, body)
        self.assertFalse(body["media_retention"]["retained"])

    def test_cancel_route_works_with_a_query_string(self):
        d = self.upload("querycancel.mp4")
        st, body = self.post(f"/jobs/{d['id']}/cancel?ts=1")
        self.assertEqual(st, 200, body)
        self.assertEqual(body["state"], "cancelled")

    def test_the_seed_button_names_the_clubhead_target(self):
        """The seed the button posts has target "clubhead"; calling it "Head"
        in the UI names a target that does not exist."""
        st, page = self.get("/", raw=True)
        page = page.decode()
        self.assertIn("Clubhead box", page)
        self.assertNotIn(">Head box", page)


class SeedEndpointBoundsTests(_Base):
    """Two narrow gaps on the seed endpoint, from the security recheck."""
    RETENTION_SECONDS = 3600

    def _raw_post(self, path, body: bytes, declared=None, ctype="application/json"):
        req = urllib.request.Request(
            self.url(path), data=body,
            headers={"X-FairwayOS-Token": self.token, "Content-Type": ctype,
                     "Content-Length": str(len(body) if declared is None else declared)})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"null")

    def test_a_seed_is_refused_once_the_media_has_been_deleted(self):
        """Retention must be enforced BEFORE a seed is accepted or scheduled.

        Otherwise an operator can seed coordinates against media that is gone,
        and the worker is scheduled on a source nobody can review any more.
        """
        d = self.upload("seedafterdelete.mp4")
        jid = d["id"]
        sha = d["result"]["source"]["sha256"]
        self.post(f"/jobs/{jid}/media/delete")
        st, body = self.post(f"/jobs/{jid}/seeds",
                             {"source_sha256": sha,
                              "seeds": [{"target": "clubhead", "frame": 3,
                                         "box_xyxy": [10.0, 10.0, 30.0, 30.0]}]})
        self.assertEqual(st, 410, body)
        self.assertIn("media", json.dumps(body).lower())

    def test_an_oversized_seed_body_is_refused_before_it_is_read(self):
        d = self.upload("bigseed.mp4")
        big = b"{" + b'"x":"' + b"a" * (128 * 1024) + b'"}'
        st, body = self._raw_post(f"/jobs/{d['id']}/seeds", big)
        self.assertEqual(st, 413, body)

    def test_a_seed_post_with_no_content_length_is_refused(self):
        d = self.upload("noclen.mp4")
        st, body = self._raw_post(f"/jobs/{d['id']}/seeds", b"", declared=0)
        self.assertIn(st, (400, 411), body)

    def test_a_normal_sized_seed_still_works(self):
        d = self.upload("normalseed.mp4")
        st, body = self.post(f"/jobs/{d['id']}/seeds",
                             {"source_sha256": d["result"]["source"]["sha256"],
                              "seeds": [{"target": "ball", "frame": 2,
                                         "point_xy": [12.0, 14.0]}]})
        self.assertEqual(st, 200, body)


class SeedFrameFidelityTests(_Base):
    def test_the_worker_window_contains_the_operator_chosen_frame(self):
        """A strided window could skip the seeded frame entirely. The seeded
        rerun must decode a window that starts at the chosen frame."""
        d = self.upload("window.mp4")
        jid = d["id"]
        st, _ = self.post(f"/jobs/{jid}/seeds",
                          {"source_sha256": d["result"]["source"]["sha256"],
                           "seeds": [{"target": "clubhead", "frame": 4,
                                      "box_xyxy": [10.0, 10.0, 30.0, 30.0]}]})
        self.assertEqual(st, 200)
        for _ in range(1200):
            s, j = self.get(f"/jobs/{jid}")
            if j["state"] in ("done", "failed", "cancelled"):
                break
            time.sleep(0.1)
        w = j["result"]["targets"]["clubhead"]["seed_window_native_frames"]
        self.assertIsNotNone(w)
        self.assertLessEqual(w[0], 4)
        self.assertGreaterEqual(w[1], 4)
        self.assertEqual(w[0], 4, "window must start at the chosen frame")

    def test_a_seed_frame_beyond_the_source_is_refused(self):
        d = self.upload("beyond.mp4")
        st, body = self.post(f"/jobs/{d['id']}/seeds",
                             {"source_sha256": d["result"]["source"]["sha256"],
                              "seeds": [{"target": "ball", "frame": 99999,
                                         "point_xy": [1.0, 2.0]}]})
        self.assertEqual(st, 400, body)
        self.assertIn("outside this source", body["error"])


class LayerMergeTests(_Base):
    def test_the_seeded_rerun_preserves_earlier_layers_and_leaves_unseeded_alone(self):
        """Independent layers: seeding the clubhead must not wipe the body
        result from phase 1, and must not invent a ball result nobody seeded."""
        d = self.upload("merge.mp4")
        jid = d["id"]
        body_before = d["result"]["targets"]["body"]
        self.assertNotEqual(body_before["outcome"], "not_run")
        st, _ = self.post(f"/jobs/{jid}/seeds",
                          {"source_sha256": d["result"]["source"]["sha256"],
                           "seeds": [{"target": "clubhead", "frame": 2,
                                      "box_xyxy": [10.0, 10.0, 30.0, 30.0]}]})
        self.assertEqual(st, 200)
        for _ in range(1200):
            s, j = self.get(f"/jobs/{jid}")
            if j["state"] in ("done", "failed", "cancelled"):
                break
            time.sleep(0.1)
        t = j["result"]["targets"]
        self.assertEqual(t["body"]["outcome"], body_before["outcome"],
                         "phase 1 body result must survive the seeded rerun")
        self.assertEqual(t["body"]["seed_used"], None, "body takes no seed")
        self.assertIsNotNone(t["clubhead"]["seed_used"])
        self.assertIsNone(t["ball"]["seed_used"])
        self.assertNotEqual(t["ball"]["outcome"], "observed")
        self.assertFalse(j["result"]["three_target_success"],
                         "a partially seeded run is never a three-target success")


class ReadinessTruthfulnessTests(_Base):
    def test_readiness_does_not_claim_seeds_are_unwired(self):
        """Seeds DO reach the workers now; the readiness reason said otherwise."""
        _, d = self.get("/ready")
        for t, v in d["dependency_readiness"].items():
            self.assertNotIn("not yet wired", v["reason"], t)
        self.assertNotIn("do NOT yet execute", d["note"])

    def test_readiness_tells_the_operator_how_to_make_a_seeded_target_run(self):
        _, d = self.get("/ready")
        seeded = [v for v in d["dependency_readiness"].values()
                  if v["requires_reviewed_seed"] and v["interpreter_ready"]]
        self.assertTrue(seeded, "expected at least one seeded target")
        for v in seeded:
            self.assertIn("/jobs/<id>/seeds", v["reason"])


class PlanUsesIsolatedInterpretersTests(_Base):
    def test_a_target_is_not_blocked_for_a_module_its_own_interpreter_has(self):
        """The serving process deliberately imports no torch. Probing torch
        in-process marked clubhead/ball "blocked", contradicting /ready, which
        correctly probes each target's own isolated interpreter.
        """
        d = self.upload("planprobe.mp4")
        _, rd = self.get("/ready")
        for name, t in d["result"]["targets"].items():
            ready = rd["dependency_readiness"].get(name)
            if ready and ready["interpreter_ready"]:
                self.assertNotIn("not importable in this interpreter", t["reason"],
                                 f"{name} blocked on an in-process probe")
                self.assertNotEqual(t["outcome"], "blocked", name)

    def test_job_targets_and_ready_agree_on_what_is_blocked(self):
        d = self.upload("planagree.mp4")
        _, rd = self.get("/ready")
        for name, t in d["result"]["targets"].items():
            ready = rd["dependency_readiness"].get(name)
            if ready is None:
                continue
            if t["outcome"] == "blocked":
                self.assertFalse(ready["interpreter_ready"] and ready["model_file_present"],
                                 f"{name}: /jobs says blocked, /ready says runnable")


class ResultsDownloadTests(_Base):
    def test_the_result_json_is_complete_enough_to_download(self):
        """The download button serialises the job result. It must actually
        contain the records, not just a summary."""
        d = self.upload("download.mp4")
        r = d["result"]
        for key in ("source", "targets", "three_target_success", "media_retention",
                    "research_only", "pseudo_label", "ground_truth"):
            self.assertIn(key, r)
        body = r["targets"]["body"]
        if body["outcome"] == "observed":
            self.assertTrue(body["result"]["records"], "records must be downloadable")
            self.assertEqual(len(body["result"]["records"][0]["source_sha256"]), 64)

    def test_the_page_offers_a_results_download_and_an_explicit_media_delete(self):
        st, page = self.get("/", raw=True)
        page = page.decode()
        self.assertIn("Download results", page)
        self.assertIn("Delete media now", page)


class DisclosureAndRetentionMessageTests(_Base):
    def test_the_seed_disclosure_does_not_claim_a_review_that_never_happened(self):
        """A click in the UI is assistance. The response used to call it
        "AI/human reviewed", which claims verification nobody performed."""
        d = self.upload("disclosure.mp4")
        st, body = self.post(f"/jobs/{d['id']}/seeds",
                             {"source_sha256": d["result"]["source"]["sha256"],
                              "seeds": [{"target": "ball", "frame": 2,
                                         "point_xy": [10.0, 12.0]}]})
        self.assertEqual(st, 200, body)
        disc = body["accepted"]["seeds"][0]["disclosure"]
        self.assertNotIn("AI/human reviewed", disc)
        self.assertIn("not human-verified", disc.lower())
        self.assertIn("not ai-verified", disc.lower())
        self.assertFalse(body["accepted"]["seeds"][0]["ground_truth"])

    def test_retention_state_is_durable_so_a_refresh_cannot_erase_it(self):
        """The delete message was written by the click handler and wiped by the
        next poll. It now comes from the job's own state, which must therefore
        stay readable and truthful across repeated fetches."""
        d = self.upload("retmsg.mp4")
        jid = d["id"]
        st, body = self.post(f"/jobs/{jid}/media/delete")
        self.assertEqual(st, 200)
        for _ in range(3):
            _, j = self.get(f"/jobs/{jid}")
            mr = j["media_retention"]
            self.assertFalse(mr["retained"])
            self.assertTrue(mr["reason"], "a reason must survive every refresh")
            self.assertIn("request", mr["reason"])
