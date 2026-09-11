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
                                           safe_join, validate_upload)
from ghostcaddie.upload.jobs import JobStore, JobState
from ghostcaddie.upload.targets import (describe_runtime, plan_targets,
                                        is_three_target_success, TargetOutcome)

MAX_ANALYSED_FRAMES = 40   # bounded work per job

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

        # Actually run every target whose adapter reports real capability.
        # Adapters that cannot run raise AdapterUnavailable and keep their
        # honest outcome; nothing is substituted for a missing layer.
        from ghostcaddie.upload.adapters import all_adapters, AdapterUnavailable
        import cv2
        for name, ad in all_adapters().items():
            if not runtime.get(name) or not runtime[name].safe_to_run:
                continue
            try:
                cap = cv2.VideoCapture(video.path)
                frames, i = [], 0
                step = max(1, video.frames // max(1, MAX_ANALYSED_FRAMES))
                while True:
                    ok, fr = cap.read()
                    if not ok or store.cancelled(job_id):
                        break
                    if i % step == 0:
                        frames.append((i, fr))
                        if len(frames) >= MAX_ANALYSED_FRAMES:
                            break
                    i += 1
                cap.release()
                recs = ad.run_frames(frames, source_sha256=video.sha256)
                obs = [r for r in recs if r.get("visible_keypoint_count", 0) > 0]
                plan[name].outcome = (TargetOutcome.OBSERVED if obs
                                      else TargetOutcome.UNAVAILABLE)
                plan[name].reason = (
                    f"{len(obs)}/{len(recs)} analysed frames produced observations"
                    if obs else "adapter ran but produced no observation on this source")
                plan[name].result = ({"frames_analysed": len(recs),
                                      "frames_with_observation": len(obs),
                                      "sampling_step": step,
                                      "records": recs} if obs else None)
            except AdapterUnavailable as e:
                plan[name].outcome = TargetOutcome.UNAVAILABLE
                plan[name].reason = str(e)
            except Exception as e:
                plan[name].outcome = TargetOutcome.UNAVAILABLE
                plan[name].reason = f"adapter error: {type(e).__name__}: {e}"
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
        store.finish(job_id, result)
    except Exception as e:              # never leak a traceback to the client
        traceback.print_exc()
        store.fail(job_id, f"{type(e).__name__}: {e}")


class Handler(BaseHTTPRequestHandler):
    store: JobStore = None
    limits: VideoLimits = None
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

    def do_GET(self):
        if self.path == "/":
            return self._send(200, INDEX, "text/html; charset=utf-8")
        if self.path == "/runtime":
            return self._send(200, {n: vars(r) for n, r in describe_runtime().items()})
        if self.path == "/jobs":
            return self._send(200, [j.to_dict() for j in self.store.list()])
        if self.path.startswith("/jobs/"):
            jid = self.path.split("/")[2].split("?")[0]
            try:
                return self._send(200, self.store.get(jid).to_dict())
            except KeyError:
                return self._send(404, {"error": "unknown job"})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.startswith("/jobs/") and self.path.endswith("/cancel"):
            jid = self.path.split("/")[2]
            try:
                self.store.cancel(jid); self.store.cleanup(jid)
                return self._send(200, self.store.get(jid).to_dict())
            except KeyError:
                return self._send(404, {"error": "unknown job"})
        if self.path != "/jobs":
            return self._send(404, {"error": "not found"})

        ctype = self.headers.get("Content-Type", "")
        path = None
        try:
            if ctype.startswith("multipart/form-data"):
                fs = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                                      environ={"REQUEST_METHOD": "POST",
                                               "CONTENT_TYPE": ctype})
                item = fs["video"] if "video" in fs else None
                if item is None or not getattr(item, "filename", ""):
                    return self._send(400, {"error": "no video field"})
                job = self.store.create(item.filename)
                wd = self.store.workdir(job.id); os.makedirs(wd, exist_ok=True)
                path = safe_join(wd, os.path.basename(item.filename))
                with open(path, "wb") as out:
                    while True:
                        chunk = item.file.read(1 << 20)
                        if not chunk: break
                        out.write(chunk)
            else:
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n).decode("utf-8", "replace")
                import urllib.parse as up
                path = up.parse_qs(raw).get("path", [""])[0]
                if not path:
                    return self._send(400, {"error": "path is required"})
                job = self.store.create(path)
        except UploadRejected as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            return self._send(400, {"error": f"{type(e).__name__}: {e}"})

        try:
            video = validate_upload(path, self.limits, workdir=self.store.root)
        except UploadRejected as e:
            self.store.fail(job.id, str(e)); self.store.cleanup(job.id)
            return self._send(400, {"error": str(e), "job": job.to_dict()})

        threading.Thread(target=_run_job, args=(self.store, job.id, video),
                         daemon=True).start()
        return self._send(202, job.to_dict())


def build_server(root: str, host="127.0.0.1", port=0, limits=None):
    Handler.store = JobStore(root)
    Handler.limits = limits or VideoLimits()
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
