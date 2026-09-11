"""End-to-end tests against a real running server (localhost, ephemeral port)."""
import json, os, tempfile, threading, time, unittest, urllib.request, urllib.parse
from ghostcaddie.upload.server import build_server
from ghostcaddie.upload.validation import VideoLimits


def synth(path, seconds=1.0, w=640, h=480, fps=30):
    """EXPLICITLY SYNTHETIC fixture. Never a user-facing result."""
    import cv2, numpy as np
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(int(seconds*fps)):
        vw.write(np.full((h, w, 3), (i*7) % 255, np.uint8))
    vw.release(); return path


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        cls.srv = build_server(os.path.join(cls.d, "root"), port=0,
                               limits=VideoLimits(max_seconds=60))
        cls.port = cls.srv.server_address[1]
        cls.host = cls.srv.server_address[0]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def url(self, p): return f"http://{self.host}:{self.port}{p}"

    def get(self, p):
        with urllib.request.urlopen(self.url(p)) as r:
            return r.status, json.loads(r.read() or b"null")

    def post_path(self, path):
        data = urllib.parse.urlencode({"path": path}).encode()
        try:
            with urllib.request.urlopen(self.url("/jobs"), data=data) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_binds_localhost_only(self):
        self.assertIn(self.host, ("127.0.0.1", "::1"))

    def test_runtime_reports_every_target_with_a_reason(self):
        """Updated: body is no longer blocked (MoveNet TFLite route).

        The previous assertion (body.safe_to_run is False) encoded a fact that
        has since changed. The durable invariant is that every target reports a
        boolean plus a specific reason.
        """
        s, b = self.get("/runtime")
        self.assertEqual(s, 200)
        for t in ("body", "clubhead", "ball"):
            self.assertIn(t, b)
            self.assertIsInstance(b[t]["safe_to_run"], bool)
            self.assertTrue(len(b[t]["reason"]) > 20)

    def test_remote_url_is_refused(self):
        s, b = self.post_path("https://example.com/a.mp4")
        self.assertEqual(s, 400)
        self.assertIn("remote", b["error"].lower())

    def test_traversal_path_is_refused(self):
        s, b = self.post_path("/etc/passwd")
        self.assertEqual(s, 400)

    def test_real_video_runs_a_job_to_completion(self):
        p = synth(os.path.join(self.d, "ok.mp4"))
        s, job = self.post_path(p)
        self.assertEqual(s, 202)
        for _ in range(100):
            _, j = self.get(f"/jobs/{job['id']}")
            if j["state"] in ("done", "failed", "cancelled"): break
            time.sleep(0.05)
        self.assertEqual(j["state"], "done", j.get("error"))
        r = j["result"]
        self.assertEqual(r["source"]["width"], 640)
        self.assertEqual(len(r["source"]["sha256"]), 64)
        return r

    def test_incomplete_layers_are_not_three_target_success(self):
        """Synthetic fixture: no target should claim a golf observation."""
        r = self.test_real_video_runs_a_job_to_completion()
        self.assertFalse(r["three_target_success"])
        observed = [n for n, t in r["targets"].items() if t["outcome"] == "observed"]
        self.assertLess(len(observed), 3, f"unexpected full coverage: {observed}")

    def test_no_target_returns_a_synthetic_result(self):
        r = self.test_real_video_runs_a_job_to_completion()
        for t in r["targets"].values():
            if t["outcome"] != "observed":
                self.assertIsNone(t["result"])

    def test_unknown_job_is_404(self):
        try:
            self.get("/jobs/deadbeef")
            self.fail("expected 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)


if __name__ == "__main__":
    unittest.main()
