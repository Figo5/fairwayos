"""End-to-end tests for the review UI endpoints (real server, localhost)."""
import json, os, tempfile, threading, time, unittest, urllib.request, urllib.error, urllib.parse
from ghostcaddie.upload.server import build_server
from ghostcaddie.upload.validation import VideoLimits


def synth(path, seconds=1.0, w=640, h=480, fps=30):
    """EXPLICITLY SYNTHETIC fixture; never a user-facing result."""
    import cv2, numpy as np
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(int(seconds*fps)):
        f = np.full((h, w, 3), 30, np.uint8)
        f[10+i:20+i, 10+i:20+i] = 255          # a moving marker per frame
        vw.write(f)
    vw.release(); return path


class UIEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        cls.srv = build_server(os.path.join(cls.d, "root"), port=0,
                               limits=VideoLimits(max_seconds=60))
        cls.host, cls.port = cls.srv.server_address[0], cls.srv.server_address[1]
        cls.token = cls.srv.RequestHandlerClass.csrf_token
        imports = cls.srv.RequestHandlerClass.store.import_dir
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        synth(os.path.join(imports, "s.mp4"))
        data = urllib.parse.urlencode({"name": "s.mp4"}).encode()
        req = urllib.request.Request(cls.u("/jobs"), data=data,
            headers={"X-FairwayOS-Token": cls.token,
                     "Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req) as r:
            cls.job = json.loads(r.read())["id"]
        for _ in range(120):
            with urllib.request.urlopen(cls.u(f"/jobs/{cls.job}")) as r:
                j = json.loads(r.read())
            if j["state"] in ("done", "failed", "cancelled"):
                break
            time.sleep(0.05)
        cls.final = j

    @classmethod
    def tearDownClass(cls): cls.srv.shutdown()

    @classmethod
    def u(cls, p): return f"http://{cls.host}:{cls.port}{p}"

    def test_ui_page_is_served(self):
        with urllib.request.urlopen(self.u("/")) as r:
            body = r.read().decode()
        self.assertEqual(r.status, 200)
        self.assertIn("upload", body.lower())
        self.assertIn("assisted initialisation", body)

    def test_the_served_copy_does_not_claim_canonical_is_network_free_by_accident(self):
        """The header must not make one blanket claim across two different modes.

        It previously said, for the whole page, that "video stays on this machine,
        nothing is uploaded anywhere" and that results are "pseudo-labels". Both were
        false for `fresh_ai`, which sends selected frames to an external service, and
        the pseudo-label wording was misleading for a canonical body report, which is a
        measurement rather than a pseudo-label.
        """
        with urllib.request.urlopen(self.u("/")) as r:
            body = r.read().decode()
        flat = " ".join(body.split())
        self.assertNotIn("Results are pseudo-labels, not ground truth", flat)
        self.assertIn("Canonical", flat)
        self.assertIn("no network request is made", flat)
        self.assertIn("fresh AI sends selected frames to an external AI service", flat)
        self.assertIn("research results are pseudo-labels", flat)
        # the page title must not frame the canonical mode as research
        self.assertNotIn("<title>FairwayOS upload &amp; review (research)</title>", flat)

    def test_the_readiness_panel_does_not_imply_canonical_ball_can_be_unlocked(self):
        with urllib.request.urlopen(self.u("/")) as r:
            body = r.read().decode()
        flat = " ".join(body.split())
        self.assertIn("research dependency readiness", flat)
        self.assertIn("does <b>not</b> enable ball or clubhead in canonical mode", flat)
        self.assertNotIn("runtime readiness", flat)

    def test_frame_endpoint_binds_source_and_exact_frame(self):
        with urllib.request.urlopen(self.u(f"/frame?job={self.job}&n=7")) as r:
            self.assertEqual(r.headers["Content-Type"], "image/png")
            self.assertEqual(r.headers["X-Requested-Frame"], "7")
            self.assertEqual(r.headers["X-Decoded-Frame"], "7")
            self.assertEqual(len(r.headers["X-Source-Sha256"]), 64)
            self.assertEqual(r.headers["X-Native-Width"], "640")

    def test_frame_out_of_range_is_refused(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(self.u(f"/frame?job={self.job}&n=99999"))
        self.assertEqual(cm.exception.code, 400)

    def test_frames_differ_so_decoding_is_real(self):
        a = urllib.request.urlopen(self.u(f"/frame?job={self.job}&n=2")).read()
        b = urllib.request.urlopen(self.u(f"/frame?job={self.job}&n=20")).read()
        self.assertNotEqual(a, b, "different frames returned identical bytes")

    def test_seed_bound_to_another_source_is_refused(self):
        body = json.dumps({"source_sha256": "a"*64,
                           "seeds": [{"target": "ball", "frame": 1,
                                      "point_xy": [1, 2]}]}).encode()
        req = urllib.request.Request(self.u(f"/jobs/{self.job}/seeds"), data=body,
                                     headers={"Content-Type": "application/json",
                                              "X-FairwayOS-Token": self.token})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req)
        self.assertIn("different source", json.loads(cm.exception.read())["error"])

    def test_seed_bound_to_this_source_is_accepted_as_assisted(self):
        sha = self.final["result"]["source"]["sha256"]
        body = json.dumps({"source_sha256": sha,
                           "seeds": [{"target": "clubhead", "frame": 5,
                                      "box_xyxy": [10, 10, 40, 40]}]}).encode()
        req = urllib.request.Request(self.u(f"/jobs/{self.job}/seeds"), data=body,
                                     headers={"Content-Type": "application/json",
                                              "X-FairwayOS-Token": self.token})
        with urllib.request.urlopen(req) as r:
            d = json.loads(r.read())
        s = d["accepted"]["seeds"][0]
        self.assertEqual(s["initialization"], "assisted")
        self.assertFalse(s["ground_truth"])

    def test_synthetic_fixture_never_reports_three_target_success(self):
        self.assertFalse(self.final["result"]["three_target_success"])

    def test_no_target_has_a_result_without_being_observed(self):
        for name, t in self.final["result"]["targets"].items():
            if t["outcome"] != "observed":
                self.assertIsNone(t["result"], f"{name} carried a result while {t['outcome']}")


if __name__ == "__main__":
    unittest.main()
