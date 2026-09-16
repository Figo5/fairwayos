"""Fresh no-seed upload jobs through the existing local service.

EXPERIMENTAL / research-only. The fresh CLI runner is mocked here: the real one
calls an external AI-vision service, so it is network-dependent and
non-deterministic and is never exercised in portable CI.
"""
import json, os, tempfile, threading, time, unittest, urllib.error, urllib.parse, urllib.request
from unittest import mock

from ghostcaddie.upload.server import build_server
from ghostcaddie.upload.validation import VideoLimits


def synth(path, seconds=0.4, w=320, h=240, fps=30):
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi",
        "-i", f"testsrc2=size={w}x{h}:rate={fps}:duration={seconds}",
        "-pix_fmt", "yuv420p", path,
    ], check=True)
    return path


class FreshUploadJobTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.srv = build_server(os.path.join(self.d, "root"), port=0,
                                limits=VideoLimits(max_seconds=60))
        self.host, self.port = self.srv.server_address[0], self.srv.server_address[1]
        self.token = self.srv.RequestHandlerClass.csrf_token
        self.imports = self.srv.RequestHandlerClass.store.import_dir
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        synth(os.path.join(self.imports, "fresh.mp4"))

    def tearDown(self):
        self.srv.shutdown()

    def url(self, p):
        return f"http://{self.host}:{self.port}{p}"

    def post_form(self, form):
        data = urllib.parse.urlencode(form).encode()
        req = urllib.request.Request(self.url("/jobs"), data=data,
            headers={"X-FairwayOS-Token": self.token,
                     "Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"null")

    def get_json(self, p):
        with urllib.request.urlopen(self.url(p)) as r:
            return r.status, json.loads(r.read() or b"null")

    def wait_terminal(self, jid):
        for _ in range(100):
            _, j = self.get_json(f"/jobs/{jid}")
            if j["state"] in ("done", "failed", "cancelled"):
                return j
            time.sleep(0.05)
        self.fail("fresh job did not finish")

    @mock.patch("ghostcaddie.upload.server._run_fresh_cli")
    def test_fresh_upload_launches_cli_and_returns_rendered_result_without_seeds(self, run_cli):
        video = os.path.join(self.d, "fake-result.mp4")
        open(video, "wb").write(b"mp4")
        result_json = os.path.join(self.d, "fresh.json")
        open(result_json, "w").write("{}")
        run_cli.return_value = {"ok": True, "outputs": {"video": video, "decisions_json": result_json}, "routing": {"command": ["fresh"]}}

        st, job = self.post_form({"name": "fresh.mp4", "analysis": "fresh_ai", "frame_start": "3", "frame_end": "5"})
        self.assertEqual(st, 202, job)
        final = self.wait_terminal(job["id"])

        self.assertEqual(final["state"], "done", final)
        self.assertEqual(final["result"]["analysis"], "fresh_ai_vision")
        self.assertEqual(final["result"]["requested_frames"], [3, 4, 5])
        self.assertEqual(final["result"]["outputs"]["video"], video)
        self.assertTrue(final["result"]["automatic_no_seed"])
        self.assertIn("prompt-only", final["result"]["limitations"]["isolation"])
        args = run_cli.call_args.args
        self.assertEqual(args[1], [3, 4, 5])

    def test_invalid_fresh_interval_is_refused_before_job_creation(self):
        st, body = self.post_form({"name": "fresh.mp4", "analysis": "fresh_ai", "frame_start": "8", "frame_end": "3"})
        self.assertEqual(st, 400, body)
        self.assertIn("interval", body["error"])

    @mock.patch("ghostcaddie.upload.server._run_fresh_cli")
    def test_fresh_runtime_failure_finishes_as_failed_without_media_result(self, run_cli):
        run_cli.side_effect = RuntimeError("fresh runtime unavailable")
        st, job = self.post_form({"name": "fresh.mp4", "analysis": "fresh_ai", "frame_start": "1", "frame_end": "1"})
        self.assertEqual(st, 202, job)
        final = self.wait_terminal(job["id"])
        self.assertEqual(final["state"], "failed")
        self.assertIn("fresh runtime unavailable", final["error"])

    @mock.patch("ghostcaddie.upload.server._run_fresh_cli")
    def test_result_video_endpoint_serves_fresh_render_and_revokes_with_media(self, run_cli):
        video = os.path.join(self.d, "fake-result.mp4")
        open(video, "wb").write(b"mp4")
        run_cli.return_value = {"ok": True, "outputs": {"video": video}, "routing": {}}
        st, job = self.post_form({"name": "fresh.mp4", "analysis": "fresh_ai", "frame_start": "2", "frame_end": "2"})
        final = self.wait_terminal(job["id"])
        self.assertEqual(final["state"], "done")
        with urllib.request.urlopen(self.url(f"/result-video?job={job['id']}")) as r:
            self.assertEqual(r.status, 200)
            self.assertEqual(r.read(), b"mp4")
        req = urllib.request.Request(self.url(f"/jobs/{job['id']}/media/delete"), data=b"{}",
            headers={"X-FairwayOS-Token": self.token, "Content-Type": "application/json"})
        urllib.request.urlopen(req).read()
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(self.url(f"/result-video?job={job['id']}"))
        self.assertEqual(e.exception.code, 410)


if __name__ == "__main__":
    unittest.main()
