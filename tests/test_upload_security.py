"""Regression tests for the seven release-blocking security findings."""
import json, os, tempfile, threading, time, unittest, urllib.request, urllib.error, urllib.parse
from ghostcaddie.upload.server import build_server
from ghostcaddie.upload.validation import (UploadRejected, VideoLimits,
                                           validate_upload, resolve_import_path)


def synth(path, seconds=0.7, w=640, h=480, fps=30):
    """EXPLICITLY SYNTHETIC fixture."""
    import cv2, numpy as np
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(int(seconds*fps)):
        f = np.full((h, w, 3), 30, np.uint8); f[10+i:20+i, 10+i:20+i] = 255
        vw.write(f)
    vw.release(); return path


class F1_PathContainment(unittest.TestCase):
    """Finding 1: arbitrary absolute path read."""
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.imports = os.path.join(self.root, "import"); os.makedirs(self.imports)

    def test_absolute_path_outside_import_dir_is_refused(self):
        outside = os.path.join(tempfile.mkdtemp(), "v.mp4"); synth(outside)
        with self.assertRaises(UploadRejected) as cm:
            resolve_import_path(outside, self.imports)
        self.assertIn("import directory", str(cm.exception))

    def test_a_readable_video_outside_root_is_refused_even_though_decodable(self):
        """The old test only passed because /etc/passwd is not a video."""
        outside = os.path.join(tempfile.mkdtemp(), "real.mp4"); synth(outside)
        self.assertTrue(os.path.isfile(outside))
        with self.assertRaises(UploadRejected):
            resolve_import_path(outside, self.imports)

    def test_symlink_escape_is_refused(self):
        outside = os.path.join(tempfile.mkdtemp(), "v.mp4"); synth(outside)
        link = os.path.join(self.imports, "link.mp4"); os.symlink(outside, link)
        with self.assertRaises(UploadRejected):
            resolve_import_path(link, self.imports)

    def test_traversal_inside_import_dir_is_refused(self):
        with self.assertRaises(UploadRejected):
            resolve_import_path("../../etc/passwd", self.imports)

    def test_file_inside_import_dir_is_accepted(self):
        p = synth(os.path.join(self.imports, "ok.mp4"))
        self.assertEqual(resolve_import_path("ok.mp4", self.imports),
                         os.path.realpath(p))


class F3_ProbeTimeout(unittest.TestCase):
    """Finding 3: decode timeout declared but not enforced."""
    def test_probe_runs_out_of_process_with_a_hard_timeout(self):
        from ghostcaddie.upload.validation import probe_video
        d = tempfile.mkdtemp(); p = synth(os.path.join(d, "v.mp4"))
        info = probe_video(p, timeout=30.0)
        self.assertEqual(info["width"], 640)

    def test_timeout_is_enforced_not_merely_declared(self):
        from ghostcaddie.upload.validation import probe_video
        d = tempfile.mkdtemp(); p = synth(os.path.join(d, "v.mp4"))
        with self.assertRaises(UploadRejected) as cm:
            probe_video(p, timeout=0.001)
        self.assertIn("timed out", str(cm.exception).lower())


class ServerSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        cls.srv = build_server(os.path.join(cls.d, "root"), port=0,
                               limits=VideoLimits(max_bytes=200_000, max_seconds=60))
        cls.host, cls.port = cls.srv.server_address[0], cls.srv.server_address[1]
        cls.token = cls.srv.RequestHandlerClass.csrf_token
        cls.imports = cls.srv.RequestHandlerClass.store.import_dir
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls): cls.srv.shutdown()

    def u(self, p): return f"http://{self.host}:{self.port}{p}"

    def _post(self, path, data, headers=None, host=None):
        h = {"Content-Type": "application/x-www-form-urlencoded"}
        h.update(headers or {})
        req = urllib.request.Request(self.u(path), data=data, headers=h, method="POST")
        if host: req.add_header("Host", host)
        return urllib.request.urlopen(req)

    def test_f1_foreign_host_header_is_rejected(self):
        req = urllib.request.Request(self.u("/ready"))
        req.add_header("Host", "evil.example.com")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req)
        self.assertEqual(cm.exception.code, 403)

    def test_f1_cross_origin_post_is_rejected(self):
        body = urllib.parse.urlencode({"name": "x.mp4", "token": self.token}).encode()
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._post("/jobs", body, {"Origin": "http://evil.example.com"})
        self.assertEqual(cm.exception.code, 403)

    def test_f1_post_without_csrf_token_is_rejected(self):
        body = urllib.parse.urlencode({"name": "x.mp4"}).encode()
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._post("/jobs", body)
        self.assertEqual(cm.exception.code, 403)

    def test_f1_no_absolute_path_field_is_accepted(self):
        outside = os.path.join(tempfile.mkdtemp(), "v.mp4"); synth(outside)
        body = urllib.parse.urlencode({"path": outside, "token": self.token}).encode()
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._post("/jobs", body)
        self.assertIn(cm.exception.code, (400, 403))

    def test_f2_missing_content_length_is_rejected(self):
        import http.client
        c = http.client.HTTPConnection(self.host, self.port)
        c.putrequest("POST", "/jobs"); c.putheader("Content-Type", "multipart/form-data; boundary=x")
        c.putheader("Transfer-Encoding", "chunked"); c.endheaders(); c.send(b"0\r\n\r\n")
        self.assertIn(c.getresponse().status, (400, 411, 403))

    def test_f2_oversized_declared_length_is_rejected_before_parsing(self):
        import http.client
        c = http.client.HTTPConnection(self.host, self.port)
        c.putrequest("POST", "/jobs")
        c.putheader("Content-Type", "multipart/form-data; boundary=x")
        c.putheader("Content-Length", str(50_000_000))
        c.putheader("X-FairwayOS-Token", self.token)
        c.endheaders()
        r = c.getresponse()
        self.assertEqual(r.status, 413)

    def test_f6_readiness_is_dependency_only_and_says_so(self):
        with urllib.request.urlopen(self.u("/ready")) as r:
            d = json.loads(r.read())
        self.assertIn("dependency_readiness", d)
        self.assertNotIn("all_runtimes_ready", d)
        for t, v in d["dependency_readiness"].items():
            self.assertIn("interpreter_ready", v)
            self.assertIn("model_file_present", v)
            self.assertIn("can_execute_now", v)
        self.assertFalse(d["dependency_readiness"]["clubhead"]["can_execute_now"])
        self.assertFalse(d["dependency_readiness"]["ball"]["can_execute_now"])

    def test_f7_no_load_safety_claim_without_execution(self):
        with urllib.request.urlopen(self.u("/ready")) as r:
            d = json.loads(r.read())
        for t in ("clubhead", "ball"):
            v = d["dependency_readiness"][t]
            if not v["can_execute_now"]:
                self.assertNotIn("weights_only=True", v.get("reason", ""))


if __name__ == "__main__":
    unittest.main()


class F4_F5_CancelAndRetention(unittest.TestCase):
    """Findings 4 and 5: cancel must stop children; media must not linger."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        from ghostcaddie.upload.jobs import JobStore
        self.store = JobStore(os.path.join(self.d, "root"))

    def test_cancel_terminates_the_tracked_child(self):
        import subprocess, sys
        j = self.store.create("c.mp4")
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        self.store.set_proc(j.id, p)
        self.assertIsNone(p.poll(), "child should still be running")
        self.assertTrue(self.store.kill_proc(j.id))
        for _ in range(40):
            if p.poll() is not None:
                break
            time.sleep(0.1)
        self.assertIsNotNone(p.poll(), "cancel did not stop the child process")

    def test_kill_proc_is_safe_when_no_child(self):
        j = self.store.create("c.mp4")
        self.assertFalse(self.store.kill_proc(j.id))

    def test_uploaded_media_under_job_dir_is_deleted(self):
        j = self.store.create("v.mp4")
        wd = self.store.workdir(j.id); os.makedirs(wd, exist_ok=True)
        media = os.path.join(wd, "v.mp4"); synth(media)

        class V:  # minimal stand-in for the validated record
            path = media
        self.store.attach_source(j.id, V())
        self.assertTrue(self.store.drop_source_media(j.id))
        self.assertFalse(os.path.exists(media))

    def test_a_file_outside_the_job_dir_is_never_deleted(self):
        """An analysed-in-place import file belongs to the user, not to us."""
        outside = os.path.join(self.d, "users_own.mp4"); synth(outside)
        j = self.store.create("users_own.mp4")

        class V:
            path = outside
        self.store.attach_source(j.id, V())
        self.assertFalse(self.store.drop_source_media(j.id))
        self.assertTrue(os.path.exists(outside), "deleted a file outside the job dir")
