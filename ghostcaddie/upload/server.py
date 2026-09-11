"""Local upload-to-analysis HTTP service (stdlib only, localhost-bound).

Deliberate choices:
  * binds 127.0.0.1 by default; there is no remote-fetch endpoint at all, so
    there is no SSRF surface to filter;
  * accepts a LOCAL FILE PATH or a multipart upload, validates it against real
    decoded facts, then runs a bounded job;
  * reports each target's honest outcome. Infrastructure that reports
    "unavailable"/"blocked" is NOT a three-target success and the API says so
    explicitly via `three_target_success: false`.

Run:  python -m ghostcaddie.upload.server --root /tmp/fairwayos-uploads
"""
from __future__ import annotations

import argparse, cgi, json, os, threading, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ghostcaddie.upload.validation import (UploadRejected, VideoLimits,
                                           resolve_import_path, safe_join,
                                           validate_upload)
from ghostcaddie.upload.jobs import JobStore, JobState
from ghostcaddie.upload.ui import PAGE
from ghostcaddie.upload.targets import (describe_runtime, plan_targets,
                                        is_three_target_success, TargetOutcome)

MAX_ANALYSED_FRAMES = 40        # bounded work per job
WORKER_TIMEOUT_SECONDS = 300    # bounded wall time per target
MULTIPART_OVERHEAD_ALLOWANCE = 1 << 20   # headers/boundaries above the media cap
RETAIN_UPLOADED_MEDIA = False   # finding 5: delete uploaded media at terminal state
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INDEX = """<!doctype html><meta charset=utf-8><title>FairwayOS upload (research)</title>
<style>body{font:14px system-ui;margin:2rem;max-width:46rem}
code{background:#eee;padding:.1rem .3rem}.b{color:#b00}.u{color:#a60}.o{color:#070}</style>
<h1>FairwayOS upload &rarr; analysis (research only)</h1>
<p>Local tool. Videos stay on this machine; nothing is uploaded anywhere.</p>
<form method=post action=/jobs enctype=multipart/form-data>
<input type=file name=video accept=video/*> <button>Analyse</button></form>
<p>Or analyse a local path:</p>
<form method=post action=/jobs><input name=path size=60 placeholder=/path/to/clip.mp4>
<button>Analyse</button></form>
<p><a href=/runtime>runtime status</a> &middot; <a href=/jobs>jobs</a></p>
"""


def _run_job(store: JobStore, job_id: str, video):
    try:
        store.start(job_id)
        store.progress(job_id, 0.1)
        runtime = describe_runtime()
        if store.cancelled(job_id):
            return
        plan = plan_targets(runtime)

        # Run each target in ITS OWN interpreter as a subprocess. The serving
        # process never imports torch or any model runtime.
        import json as _json, subprocess, tempfile
        from ghostcaddie.upload.runtimes import RuntimeRegistry, InterpreterUnavailable
        reg = RuntimeRegistry.default()
        step = max(1, video.frames // max(1, MAX_ANALYSED_FRAMES))
        for name in ("body", "clubhead", "ball"):
            if store.cancelled(job_id):
                return
            try:
                spec = reg.require(name)
            except InterpreterUnavailable as e:
                plan[name].outcome = TargetOutcome.UNAVAILABLE
                plan[name].reason = str(e)
                continue
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
                _json.dump({"video": video.path, "source_sha256": video.sha256,
                            "sampling_step": step,
                            "max_frames": MAX_ANALYSED_FRAMES}, fh)
                req = fh.name
            try:
                proc = subprocess.Popen(
                    [spec.interpreter, "-m", "ghostcaddie.upload.worker", name, req],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    cwd=REPO_ROOT)
                store.set_proc(job_id, proc)
                try:
                    stdout, _ = proc.communicate(timeout=WORKER_TIMEOUT_SECONDS)
                finally:
                    store.clear_proc(job_id)
                if store.cancelled(job_id):
                    return
                out = _json.loads((stdout or "").strip().splitlines()[-1])
            except subprocess.TimeoutExpired:
                plan[name].outcome = TargetOutcome.UNAVAILABLE
                plan[name].reason = f"worker exceeded {WORKER_TIMEOUT_SECONDS}s and was stopped"
                continue
            except Exception as e:
                plan[name].outcome = TargetOutcome.UNAVAILABLE
                plan[name].reason = f"worker error: {type(e).__name__}: {e}"
                continue
            finally:
                os.unlink(req)
            if not out.get("ok"):
                plan[name].outcome = TargetOutcome.UNAVAILABLE
                plan[name].reason = out.get("error", "worker reported failure")
                continue
            recs = out["records"]
            obs = [r for r in recs if r.get("visible_keypoint_count", 0) > 0]
            plan[name].outcome = (TargetOutcome.OBSERVED if obs
                                  else TargetOutcome.UNAVAILABLE)
            plan[name].reason = (f"{len(obs)}/{len(recs)} analysed frames produced "
                                 f"observations (interpreter {spec.interpreter})"
                                 if obs else "adapter ran but produced no observation")
            plan[name].result = ({"frames_analysed": len(recs),
                                  "frames_with_observation": len(obs),
                                  "sampling_step": out.get("sampling_step"),
                                  "interpreter": spec.interpreter,
                                  "records": recs} if obs else None)
        store.progress(job_id, 0.8)
        result = {
            "source": {"path": video.path, "sha256": video.sha256,
                       "width": video.width, "height": video.height,
                       "frames": video.frames, "fps": round(video.fps, 4),
                       "duration_seconds": round(video.duration_seconds, 3)},
            "targets": {n: {"outcome": p.outcome, "reason": p.reason,
                            "initialization": p.initialization,
                            "result": p.result} for n, p in plan.items()},
            "three_target_success": is_three_target_success(plan),
            "note": ("Targets reporting blocked/unavailable are infrastructure "
                     "states, NOT results. No detection, coordinate or "
                     "probability is synthesised anywhere in this response."),
            "research_only": True, "pseudo_label": True,
            "ground_truth": False, "production_eligible": False,
        }
        if store.cancelled(job_id):
            return
        if not RETAIN_UPLOADED_MEDIA:
            result["media_retention"] = (
                "uploaded media deleted at job completion; only sanitised result "
                "metadata is kept" if store.drop_source_media(job_id)
                else "no uploaded media to delete (analysed in place from the "
                     "import directory)")
        store.finish(job_id, result)
    except Exception as e:              # never leak a traceback to the client
        traceback.print_exc()
        store.fail(job_id, f"{type(e).__name__}: {e}")


ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


class Handler(BaseHTTPRequestHandler):
    store: JobStore = None
    limits: VideoLimits = None
    csrf_token: str = ""
    server_version = "FairwayOSUpload/0.1"

    def log_message(self, *a): pass

    def _send(self, code, payload, ctype="application/json"):
        body = (json.dumps(payload, indent=1).encode() if ctype == "application/json"
                else payload.encode())
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    # ---- finding 1: DNS-rebinding / cross-origin defence
    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0].strip().lower()
        return host in ALLOWED_HOSTS or host == ""

    def _origin_ok(self) -> bool:
        import urllib.parse as up
        for h in ("Origin", "Referer"):
            v = self.headers.get(h)
            if not v:
                continue
            host = (up.urlparse(v).hostname or "").lower()
            if host not in ALLOWED_HOSTS:
                return False
        return True

    def _guard(self, mutating: bool) -> bool:
        if not self._host_ok():
            self._send(403, {"error": "Host header not allowed; this service only "
                                      "answers to localhost"})
            return False
        if not self._origin_ok():
            self._send(403, {"error": "cross-origin request refused"})
            return False
        if mutating:
            tok = (self.headers.get("X-FairwayOS-Token") or "").strip()
            if not tok:
                import urllib.parse as up
                tok = up.parse_qs(up.urlparse(self.path).query).get("token", [""])[0]
            if not self.csrf_token or tok != self.csrf_token:
                self._send(403, {"error": "missing or invalid request token"})
                return False
        return True

    def do_GET(self):
        if not self._guard(mutating=False):
            return
        if self.path == "/":
            page = PAGE.replace("__FAIRWAYOS_TOKEN__", self.csrf_token)
            return self._send(200, page, "text/html; charset=utf-8")
        if self.path == "/ready":
            from ghostcaddie.upload.runtimes import RuntimeRegistry
            rd = RuntimeRegistry.default().readiness()
            return self._send(200, {
                "service": "ok",
                "dependency_readiness": rd,
                "executable_now": [t for t, v in rd.items() if v["can_execute_now"]],
                "note": "This is DEPENDENCY readiness, not tracking success and not "
                        "a three-target claim. A target is only executable when its "
                        "interpreter and model file are present AND it needs no "
                        "reviewed seed. clubhead and ball currently accept and store "
                        "seeds but do NOT yet execute their models."})
        if self.path == "/runtime":
            return self._send(200, {n: vars(r) for n, r in describe_runtime().items()})
        if self.path == "/jobs":
            return self._send(200, [j.to_dict() for j in self.store.list()])
        if self.path.startswith("/frame"):
            return self._serve_frame()
        if self.path.startswith("/video"):
            return self._serve_video()
        if self.path.startswith("/jobs/"):
            jid = self.path.split("/")[2].split("?")[0]
            try:
                return self._send(200, self.store.get(jid).to_dict())
            except KeyError:
                return self._send(404, {"error": "unknown job"})
        return self._send(404, {"error": "not found"})

    def _qs(self):
        import urllib.parse as up
        return dict(up.parse_qsl(up.urlparse(self.path).query))

    def _serve_frame(self):
        """Decode ONE exact native frame. Not a time seek."""
        q = self._qs()
        try:
            job = self.store.get(q.get("job", ""))
        except KeyError:
            return self._send(404, {"error": "unknown job"})
        src = self.store.source(job.id)
        if not src:
            return self._send(409, {"error": "job has no validated source yet"})
        try:
            n = int(q.get("n", "0"))
        except ValueError:
            return self._send(400, {"error": "n must be an integer"})
        if n < 0 or n >= src.frames:
            return self._send(400, {"error": f"frame {n} outside 0..{src.frames-1}"})
        import cv2
        cap = cv2.VideoCapture(src.path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            ok, img = cap.read()
            got = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        finally:
            cap.release()
        if not ok:
            return self._send(409, {"error": f"frame {n} could not be decoded"})
        ok2, buf = cv2.imencode(".png", img)
        if not ok2:
            return self._send(500, {"error": "frame encode failed"})
        body = buf.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Source-Sha256", src.sha256)
        self.send_header("X-Requested-Frame", str(n))
        self.send_header("X-Decoded-Frame", str(got))   # honest: what was decoded
        self.send_header("X-Native-Width", str(src.width))
        self.send_header("X-Native-Height", str(src.height))
        self.end_headers()
        self.wfile.write(body)

    def _serve_video(self):
        q = self._qs()
        try:
            job = self.store.get(q.get("job", ""))
        except KeyError:
            return self._send(404, {"error": "unknown job"})
        src = self.store.source(job.id)
        if not src:
            return self._send(409, {"error": "no source"})
        size = os.path.getsize(src.path)
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "none")
        self.end_headers()
        with open(src.path, "rb") as fh:
            while True:
                b = fh.read(1 << 20)
                if not b: break
                self.wfile.write(b)

    def do_POST(self):
        if not self._guard(mutating=True):
            return
        if self.path.startswith("/jobs/") and self.path.endswith("/cancel"):
            jid = self.path.split("/")[2]
            try:
                self.store.cancel(jid)
                killed = self.store.kill_proc(jid)
                self.store.cleanup(jid)
                d = self.store.get(jid).to_dict()
                d["active_child_terminated"] = killed
                return self._send(200, d)
            except KeyError:
                return self._send(404, {"error": "unknown job"})
        if self.path.startswith("/jobs/") and self.path.endswith("/seeds"):
            jid = self.path.split("/")[2]
            try:
                job = self.store.get(jid)
            except KeyError:
                return self._send(404, {"error": "unknown job"})
            src = self.store.source(jid)
            n = int(self.headers.get("Content-Length") or 0)
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
                from ghostcaddie.upload.seeds import SeedBundle, SeedRejected
                bundle = SeedBundle.from_dict(data, source_sha256=src.sha256)
            except SeedRejected as e:
                return self._send(400, {"error": str(e)})
            except Exception as e:
                return self._send(400, {"error": f"{type(e).__name__}: {e}"})
            self.store.attach_seeds(jid, bundle)
            return self._send(200, {"accepted": bundle.to_dict(),
                                    "note": "assisted initialisation recorded and "
                                            "bound to this source hash"})
        if self.path != "/jobs":
            return self._send(404, {"error": "not found"})

        ctype = self.headers.get("Content-Type", "")
        # finding 2: bound the request BEFORE parsing or writing anything
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            return self._send(411, {"error": "Content-Length is required"})
        try:
            declared = int(raw_len)
        except ValueError:
            return self._send(400, {"error": "invalid Content-Length"})
        if declared <= 0:
            return self._send(400, {"error": "empty request"})
        if declared > self.limits.max_bytes + MULTIPART_OVERHEAD_ALLOWANCE:
            return self._send(413, {"error": f"request of {declared} bytes exceeds "
                                             f"the limit {self.limits.max_bytes}"})

        path = None
        job = None
        try:
            if ctype.startswith("multipart/form-data"):
                fs = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                                      environ={"REQUEST_METHOD": "POST",
                                               "CONTENT_TYPE": ctype},
                                      limit=self.limits.max_bytes + MULTIPART_OVERHEAD_ALLOWANCE)
                item = fs["video"] if "video" in fs else None
                if item is None or not getattr(item, "filename", ""):
                    return self._send(400, {"error": "no video field"})
                job = self.store.create(item.filename)
                wd = self.store.workdir(job.id); os.makedirs(wd, exist_ok=True)
                path = safe_join(wd, os.path.basename(item.filename))
                written = 0
                with open(path, "wb") as out:
                    while True:
                        chunk = item.file.read(1 << 20)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > self.limits.max_bytes:   # cumulative cap
                            out.close()
                            self.store.fail(job.id, "upload exceeded size limit")
                            self.store.cleanup(job.id)        # drop partial file
                            return self._send(413, {"error": "upload exceeded the "
                                                             "size limit and was discarded"})
                        out.write(chunk)
            else:
                n = min(declared, 64 * 1024)
                raw = self.rfile.read(n).decode("utf-8", "replace")
                import urllib.parse as up
                form = up.parse_qs(raw)
                name = (form.get("name") or form.get("path") or [""])[0]
                if not name:
                    return self._send(400, {"error": "name is required"})
                # finding 1: only files inside the import directory, never an
                # arbitrary absolute path
                path = resolve_import_path(name, self.store.import_dir)
                job = self.store.create(os.path.basename(path))
        except UploadRejected as e:
            if job: self.store.cleanup(job.id)
            return self._send(400, {"error": str(e)})
        except Exception as e:
            if job: self.store.cleanup(job.id)
            return self._send(400, {"error": f"{type(e).__name__}: {e}"})

        try:
            video = validate_upload(path, self.limits, workdir=self.store.root)
        except UploadRejected as e:
            self.store.fail(job.id, str(e)); self.store.cleanup(job.id)
            return self._send(400, {"error": str(e), "job": job.to_dict()})

        self.store.attach_source(job.id, video)
        threading.Thread(target=_run_job, args=(self.store, job.id, video),
                         daemon=True).start()
        return self._send(202, job.to_dict())


def build_server(root: str, host="127.0.0.1", port=0, limits=None):
    import secrets
    Handler.store = JobStore(root)
    Handler.limits = limits or VideoLimits()
    # A per-launch token. Must never be empty: an empty token would compare equal
    # to an absent one and silently disable the CSRF check entirely.
    Handler.csrf_token = secrets.token_urlsafe(32)
    assert Handler.csrf_token, "csrf token must not be empty"
    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/fairwayos-uploads")
    ap.add_argument("--host", default="127.0.0.1")   # localhost only
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    srv = build_server(a.root, a.host, a.port)
    print(f"FairwayOS upload service on http://{a.host}:{a.port}  root={a.root}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
