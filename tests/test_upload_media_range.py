"""HTTP byte-range serving for /video and /result-video.

Both endpoints previously answered every request with `200` and
`Accept-Ranges: none`. A media element is entitled to read that as "this
entity cannot be seeked", and the primary's Safari QA showed a black video
area for BOTH the source clip (moov before mdat) and the rendered clip (moov
after mdat) -- identical symptoms with opposite atom order, which is what a
missing Range implementation looks like and is not explained by faststart.

These tests pin the serving contract only. They do NOT claim the black-frame
capture is cured; that needs a real browser retest.
"""
import http.client, json, os, socket, tempfile, threading, time, unittest
import urllib.error, urllib.parse, urllib.request

from ghostcaddie.upload.server import build_server, parse_byte_range
from ghostcaddie.upload.validation import VideoLimits


def synth(path, seconds=1.0, w=320, h=240, fps=30):
    """EXPLICITLY SYNTHETIC fixture; never a user-facing result."""
    import cv2, numpy as np
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(int(seconds * fps)):
        f = np.full((h, w, 3), 30, np.uint8)
        f[10 + i:20 + i, 10 + i:20 + i] = 255
        vw.write(f)
    vw.release()
    return path


class ParseByteRangeTests(unittest.TestCase):
    """The parser, at the edges, without a socket."""

    def test_absent_or_unparsable_headers_serve_the_whole_entity(self):
        for header in (None, "", "   ", "items=0-1", "bytes", "bytes=",
                       "bytes=abc-10", "bytes=0-abc", "bytes=-abc",
                       "bytes=0-9,20-29",          # multi-range: MAY be ignored
                       "bytes=1_0-20",             # int() would accept the underscore
                       "bytes=+5-10"):
            with self.subTest(header=header):
                self.assertIsNone(parse_byte_range(header, 1000))

    def test_explicit_open_ended_and_suffix_forms(self):
        self.assertEqual(parse_byte_range("bytes=0-99", 1000), (0, 99))
        self.assertEqual(parse_byte_range("bytes=500-", 1000), (500, 999))
        self.assertEqual(parse_byte_range("bytes=-100", 1000), (900, 999))
        self.assertEqual(parse_byte_range("bytes=0-0", 1000), (0, 0))
        self.assertEqual(parse_byte_range("bytes=999-999", 1000), (999, 999))
        self.assertEqual(parse_byte_range(" bytes=0-1 ", 1000), (0, 1))

    def test_an_end_past_eof_is_clamped_not_refused(self):
        self.assertEqual(parse_byte_range("bytes=900-999999", 1000), (900, 999))
        self.assertEqual(parse_byte_range("bytes=0-999999", 1000), (0, 999))

    def test_a_suffix_longer_than_the_file_is_the_whole_file(self):
        self.assertEqual(parse_byte_range("bytes=-999999", 1000), (0, 999))

    def test_unsatisfiable_ranges_are_reported_as_such(self):
        self.assertEqual(parse_byte_range("bytes=1000-", 1000), "unsatisfiable")
        self.assertEqual(parse_byte_range("bytes=1000-1010", 1000), "unsatisfiable")
        self.assertEqual(parse_byte_range("bytes=-0", 1000), "unsatisfiable")
        self.assertEqual(parse_byte_range("bytes=10-5", 1000), "unsatisfiable")

    def test_an_empty_entity_can_satisfy_no_range(self):
        for header in ("bytes=0-", "bytes=0-0", "bytes=-1"):
            with self.subTest(header=header):
                self.assertEqual(parse_byte_range(header, 0), "unsatisfiable")
        self.assertIsNone(parse_byte_range(None, 0))


class MediaRangeEndpointTests(unittest.TestCase):
    """The real server, on both media endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        cls.srv = build_server(os.path.join(cls.d, "root"), port=0,
                               limits=VideoLimits(max_seconds=60))
        cls.host, cls.port = cls.srv.server_address[0], cls.srv.server_address[1]
        cls.token = cls.srv.RequestHandlerClass.csrf_token
        cls.store = cls.srv.RequestHandlerClass.store
        imports = cls.store.import_dir
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        synth(os.path.join(imports, "s.mp4"))
        data = urllib.parse.urlencode({"name": "s.mp4"}).encode()
        req = urllib.request.Request(
            cls.u("/jobs"), data=data,
            headers={"X-FairwayOS-Token": cls.token,
                     "Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req) as r:
            cls.job = json.loads(r.read())["id"]
        for _ in range(120):
            with urllib.request.urlopen(cls.u(f"/jobs/{cls.job}")) as r:
                if json.loads(r.read())["state"] in ("done", "failed", "cancelled"):
                    break
            time.sleep(0.05)
        # a stand-in rendered result, so /result-video exercises the same path
        cls.rendered = synth(os.path.join(cls.d, "rendered.mp4"))
        cls.store.attach_result_video(cls.job, cls.rendered)
        cls.paths = {"/video": cls.store.source(cls.job).path,
                     "/result-video": cls.rendered}

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    @classmethod
    def u(cls, p):
        return f"http://{cls.host}:{cls.port}{p}"

    def get(self, route, rng=None):
        req = urllib.request.Request(self.u(f"{route}?job={self.job}"))
        if rng is not None:
            req.add_header("Range", rng)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def full(self, route):
        with open(self.paths[route], "rb") as fh:
            return fh.read()

    def test_a_plain_request_still_returns_the_whole_file_and_advertises_ranges(self):
        for route in self.paths:
            with self.subTest(route=route):
                status, headers, body = self.get(route)
                whole = self.full(route)
                self.assertEqual(status, 200)
                self.assertEqual(headers["Accept-Ranges"], "bytes")
                self.assertEqual(headers["Content-Type"], "video/mp4")
                self.assertEqual(int(headers["Content-Length"]), len(whole))
                self.assertNotIn("Content-Range", headers)
                self.assertEqual(body, whole)

    def test_a_probe_range_is_answered_206_with_exactly_those_bytes(self):
        """This is the request Safari opens a media element with."""
        for route in self.paths:
            with self.subTest(route=route):
                whole = self.full(route)
                status, headers, body = self.get(route, "bytes=0-1")
                self.assertEqual(status, 206)
                self.assertEqual(headers["Content-Range"], f"bytes 0-1/{len(whole)}")
                self.assertEqual(headers["Content-Length"], "2")
                self.assertEqual(headers["Accept-Ranges"], "bytes")
                self.assertEqual(body, whole[:2])

    def test_mid_file_open_ended_and_suffix_ranges_return_the_right_slice(self):
        for route in self.paths:
            whole = self.full(route)
            n = len(whole)
            cases = {
                f"bytes=100-199": (100, 199),
                f"bytes={n - 50}-": (n - 50, n - 1),
                "bytes=-64": (n - 64, n - 1),
                f"bytes=0-{n * 2}": (0, n - 1),          # end past EOF, clamped
            }
            for header, (start, end) in cases.items():
                with self.subTest(route=route, header=header):
                    status, headers, body = self.get(route, header)
                    self.assertEqual(status, 206)
                    self.assertEqual(headers["Content-Range"],
                                     f"bytes {start}-{end}/{n}")
                    self.assertEqual(int(headers["Content-Length"]), end - start + 1)
                    self.assertEqual(body, whole[start:end + 1])

    def test_an_unparsable_range_degrades_to_a_normal_200(self):
        for route in self.paths:
            for header in ("bytes=oops", "kilobytes=0-1", "bytes=0-9,20-29"):
                with self.subTest(route=route, header=header):
                    status, headers, body = self.get(route, header)
                    self.assertEqual(status, 200)
                    self.assertNotIn("Content-Range", headers)
                    self.assertEqual(body, self.full(route))

    def test_an_unsatisfiable_range_is_416_with_the_entity_length(self):
        for route in self.paths:
            n = len(self.full(route))
            for header in (f"bytes={n}-", f"bytes={n + 10}-{n + 20}", "bytes=-0"):
                with self.subTest(route=route, header=header):
                    status, headers, body = self.get(route, header)
                    self.assertEqual(status, 416)
                    self.assertEqual(headers["Content-Range"], f"bytes */{n}")
                    self.assertEqual(body, b"")

    def test_ranged_reads_never_bypass_the_retention_check(self):
        """A revoked job must 410 for a ranged request exactly as for a plain one."""
        data = urllib.parse.urlencode({"name": "s.mp4"}).encode()
        req = urllib.request.Request(
            self.u("/jobs"), data=data,
            headers={"X-FairwayOS-Token": self.token,
                     "Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req) as r:
            jid = json.loads(r.read())["id"]
        for _ in range(120):
            with urllib.request.urlopen(self.u(f"/jobs/{jid}")) as r:
                if json.loads(r.read())["state"] in ("done", "failed", "cancelled"):
                    break
            time.sleep(0.05)
        self.store.attach_result_video(jid, self.rendered)
        drop = urllib.request.Request(
            self.u(f"/jobs/{jid}/media/delete"), data=b"",
            headers={"X-FairwayOS-Token": self.token})
        with urllib.request.urlopen(drop) as r:
            self.assertFalse(json.loads(r.read())["media_retention"]["retained"])
        for route in ("/video", "/result-video"):
            for header in (None, "bytes=0-1"):
                with self.subTest(route=route, header=header):
                    r = urllib.request.Request(self.u(f"{route}?job={jid}"))
                    if header:
                        r.add_header("Range", header)
                    with self.assertRaises(urllib.error.HTTPError) as cm:
                        urllib.request.urlopen(r)
                    self.assertEqual(cm.exception.code, 410)
                    self.assertIn("media_retention", json.loads(cm.exception.read()))

    def test_an_unknown_job_is_still_refused_when_a_range_is_asked_for(self):
        for route in ("/video", "/result-video"):
            with self.subTest(route=route):
                r = urllib.request.Request(self.u(f"{route}?job=nope"))
                r.add_header("Range", "bytes=0-1")
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    urllib.request.urlopen(r)
                self.assertEqual(cm.exception.code, 404)

    def test_a_client_that_disconnects_mid_stream_does_not_break_the_server(self):
        """A player that seeks away closes the socket; the server must survive."""
        c = http.client.HTTPConnection(self.host, self.port, timeout=5)
        c.request("GET", f"/video?job={self.job}", headers={"Range": "bytes=0-"})
        resp = c.getresponse()
        self.assertEqual(resp.status, 206)
        resp.read(16)
        c.close()
        time.sleep(0.2)
        status, _, body = self.get("/video", "bytes=0-1")
        self.assertEqual(status, 206)
        self.assertEqual(body, self.full("/video")[:2])


if __name__ == "__main__":
    unittest.main()
