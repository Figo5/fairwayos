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
MAX_SEED_BODY_BYTES = 64 * 1024          # a seed bundle is small; bound it separately
# A seeded rerun is bounded to a window that STARTS at the operator's chosen
# frame, so the selected frame is always inside the decoded order handed to the
# worker and the work stays bounded.
SEEDED_WINDOW_FRAMES = 24
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


UNSEEDED_TARGETS = ("body",)              # can run with no operator assistance
SEEDED_TARGETS = ("clubhead", "ball")     # refuse to run without a reviewed seed


def _run_one_target(store, job_id, video, name, plan, reg, step, seed):
    """Run ONE target in its own interpreter. Never imports torch here.

    The seed, when present, is passed THROUGH to the worker request. That is the
    whole point of the seeded phase: a seed that is stored but not sent is not
    assistance, it is decoration.
    """
    import json as _json, subprocess, tempfile
    from ghostcaddie.upload.runtimes import InterpreterUnavailable
    p = plan[name]
    p.seed_used = seed.to_dict() if seed is not None else None
    if name in SEEDED_TARGETS and seed is None:
        p.outcome = TargetOutcome.UNAVAILABLE
        p.reason = ("this target needs a reviewed source-specific seed and none "
                    "was supplied for it; it is not an unattended detector")
        return
    if seed is not None:
        p.initialization = "assisted"
        p.assisted_disclosure = (
            "operator-supplied seed for this exact source hash. This is "
            "USER-PROVIDED ASSISTANCE: it is not human-verified and not "
            "AI-verified truth, and nothing derived from it is ground truth.")
        # Record the bounded accepted-seed window before runtime resolution so
        # unavailable worker infrastructure cannot erase review metadata.
        lo = max(0, int(seed.frame))
        hi = min(video.frames - 1, lo + SEEDED_WINDOW_FRAMES - 1)
        p.seed_window = [lo, hi]
    try:
        spec = reg.require(name)
    except InterpreterUnavailable as e:
        p.outcome, p.reason = TargetOutcome.UNAVAILABLE, str(e)
        return
    req = {"video": video.path, "source_sha256": video.sha256,
           "sampling_step": step, "max_frames": MAX_ANALYSED_FRAMES}
    if seed is not None:
        # Window STARTS at the chosen frame and steps by 1, so the seeded frame
        # is guaranteed to be in the decoded order the worker receives. A
        # sampled/strided window could skip it entirely.
        lo, hi = p.seed_window
        req.update({"seed": {"frame": seed.frame, "box_xyxy": seed.box_xyxy,
                             "point_xy": seed.point_xy},
                    "frame_start": lo, "frame_end": hi,
                    "sampling_step": 1,
                    "max_frames": hi - lo + 1})
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        _json.dump(req, fh)
        req_path = fh.name
    try:
        proc = subprocess.Popen(
            [spec.interpreter, "-m", "ghostcaddie.upload.worker", name, req_path],
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
        store.kill_proc(job_id)
        p.outcome = TargetOutcome.UNAVAILABLE
        p.reason = f"worker exceeded {WORKER_TIMEOUT_SECONDS}s and was stopped"
        return
    except Exception as e:
        p.outcome = TargetOutcome.UNAVAILABLE
        p.reason = f"worker error: {type(e).__name__}: {e}"
        return
    finally:
        try:
            os.unlink(req_path)
        except OSError:
            pass
    if not out.get("ok"):
        p.outcome = TargetOutcome.UNAVAILABLE
        p.reason = out.get("error", "worker reported failure")
        return
    recs = out["records"]
    obs = [r for r in recs
           if r.get("visible") or r.get("visible_keypoint_count", 0) > 0]
    p.outcome = TargetOutcome.OBSERVED if obs else TargetOutcome.UNAVAILABLE
    p.reason = (f"{len(obs)}/{len(recs)} analysed frames produced observations "
                f"(interpreter {spec.interpreter})" if obs
                else "adapter ran but produced no observation")
    p.result = ({"frames_analysed": len(recs), "frames_with_observation": len(obs),
                 "sampling_step": out.get("sampling_step"),
                 "interpreter": spec.interpreter, "records": recs} if obs else None)


def _result_doc(store, job_id, video, plan):
    return {
        "source": {"path": video.path, "sha256": video.sha256,
                   "width": video.width, "height": video.height,
                   "frames": video.frames, "fps": round(video.fps, 4),
                   "duration_seconds": round(video.duration_seconds, 3)},
        "targets": {n: {"outcome": p.outcome, "reason": p.reason,
                        "initialization": p.initialization,
                        "assisted_disclosure": p.assisted_disclosure,
                        "seed_used": getattr(p, "seed_used", None),
                        "seed_window_native_frames": getattr(p, "seed_window", None),
                        "result": p.result} for n, p in plan.items()},
        "three_target_success": is_three_target_success(plan),
        "media_retention": store.media_state(job_id),
        "note": ("Targets reporting blocked/unavailable are infrastructure "
                 "states, NOT results. No detection, coordinate or probability "
                 "is synthesised anywhere in this response. Seeds supplied "
                 "through the UI are USER-PROVIDED ASSISTANCE, not verified "
                 "truth."),
        "research_only": True, "pseudo_label": True,
        "ground_truth": False, "production_eligible": False,
    }


def _plan_from_isolated_interpreters():
    """Plan each target from ITS OWN interpreter, not from this process.

    The serving process deliberately never imports torch, so an in-process
    capability probe marked every torch target "blocked: torch is not importable
    in this interpreter" -- which contradicted /ready, which probes the real
    isolated interpreters. Blocked must mean "unsafe to run", never "the web
    server happens not to have this module".
    """
    from ghostcaddie.upload.runtimes import RuntimeRegistry
    from ghostcaddie.upload.targets import TargetPlan
    rd = RuntimeRegistry.default().readiness()
    plan = {}
    for name, r in rd.items():
        if not r["model_file_present"]:
            plan[name] = TargetPlan(name, TargetOutcome.UNAVAILABLE, r["reason"])
        elif not r["interpreter_ready"]:
            plan[name] = TargetPlan(name, TargetOutcome.UNAVAILABLE, r["reason"])
        else:
            plan[name] = TargetPlan(name, TargetOutcome.NOT_RUN, r["reason"])
    # the legacy pickle route stays BLOCKED on safety grounds, not on imports
    for name, rs in describe_runtime().items():
        if name not in plan and not rs.safe_to_run and rs.model_present:
            plan[name] = TargetPlan(name, TargetOutcome.BLOCKED, rs.reason)
    return plan


def _run_job(store: JobStore, job_id: str, video):
    """Phase 1: everything that needs no assistance. Then HOLD for seeds.

    The previous version ran every target at upload time and finished the job,
    so a seed posted afterwards could never reach a worker and the assisted
    targets could only ever fail. Phase 1 now stops at awaiting_seeds and the
    media stays readable, which is exactly what frame review needs.
    """
    try:
        store.start(job_id)
        store.progress(job_id, 0.1)
        plan = _plan_from_isolated_interpreters()
        if store.cancelled(job_id):
            return
        from ghostcaddie.upload.runtimes import RuntimeRegistry
        reg = RuntimeRegistry.default()
        step = max(1, video.frames // max(1, MAX_ANALYSED_FRAMES))
        for name in UNSEEDED_TARGETS:
            if store.cancelled(job_id):
                return
            _run_one_target(store, job_id, video, name, plan, reg, step, None)
        if store.cancelled(job_id):
            return
        pending = [t for t in SEEDED_TARGETS if t in plan]
        doc = _result_doc(store, job_id, video, plan)
        doc["awaiting_seeds"] = pending
        doc["next_step"] = ("POST /jobs/<id>/seeds with a bundle bound to this "
                            "source sha256 to run the assisted targets")
        store._plan = getattr(store, "_plan", {})
        store._plan[job_id] = plan
        if pending:
            store.await_seeds(job_id, pending, doc)
        else:
            store.finish(job_id, doc)
    except Exception as e:                # never leak a traceback to the client
        traceback.print_exc()
        store.fail(job_id, f"{type(e).__name__}: {e}")


def _run_seeded(store: JobStore, job_id: str, video, bundle):
    """Phase 2: run the assisted targets with the seeds actually applied."""
    try:
        store.start(job_id)
        plan = (getattr(store, "_plan", {}).get(job_id)
                or _plan_from_isolated_interpreters())
        from ghostcaddie.upload.runtimes import RuntimeRegistry
        reg = RuntimeRegistry.default()
        step = max(1, video.frames // max(1, MAX_ANALYSED_FRAMES))
        for name in SEEDED_TARGETS:
            if name not in plan or store.cancelled(job_id):
                continue
            _run_one_target(store, job_id, video, name, plan, reg, step,
                            bundle.for_target(name) if bundle else None)
        if store.cancelled(job_id):
            return
        doc = _result_doc(store, job_id, video, plan)
        doc["seeds_applied"] = bundle.to_dict() if bundle else None
        if not RETAIN_UPLOADED_MEDIA:
            doc["media_retention"] = store.media_state(job_id)
        store.finish(job_id, doc)
    except Exception as e:
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

    @property
    def route(self):
        """Path with any query string removed.

        Routing used to match on self.path, so /jobs/<id>/seeds?ts=1 matched no
        handler at all and fell through. Every route decision uses this.
        """
        return self.path.split("?", 1)[0].rstrip("/") or "/"

    def do_GET(self):
        if not self._guard(mutating=False):
            return
        if self.route == "/":
            page = PAGE.replace("__FAIRWAYOS_TOKEN__", self.csrf_token)
            return self._send(200, page, "text/html; charset=utf-8")
        if self.route == "/ready":
            from ghostcaddie.upload.runtimes import RuntimeRegistry
            rd = RuntimeRegistry.default().readiness()
            return self._send(200, {
                "service": "ok",
                "dependency_readiness": rd,
                "executable_now": [t for t, v in rd.items() if v["can_execute_now"]],
                "note": "This is DEPENDENCY readiness, not tracking success and not "
                        "a three-target claim. A target is only executable when its "
                        "interpreter and model file are present AND it needs no "
                        "reviewed seed. clubhead and ball DO execute once a seed "
                        "bound to the uploaded source hash is posted to "
                        "/jobs/<id>/seeds; they never run unattended."})
        if self.route == "/runtime":
            from ghostcaddie.upload.runtimes import RuntimeRegistry
            rd = RuntimeRegistry.default().readiness()
            out = {}
            for n, r in describe_runtime().items():
                v = vars(r).copy()
                ready = rd.get(n, {})
                # safe_to_run answers "is the load path safe to execute",
                # NOT "can this produce a result right now". Conflating the two
                # is what made /runtime contradict /ready.
                v["requires_reviewed_seed"] = bool(ready.get("requires_reviewed_seed"))
                v["can_execute_unattended"] = bool(ready.get("can_execute_now"))
                v["safe_to_run_means"] = ("the load path is safe to execute; it "
                                          "does NOT mean this target can produce "
                                          "a result without a reviewed seed")
                out[n] = v
            return self._send(200, out)
        if self.route == "/jobs":
            return self._send(200, [j.to_dict() for j in self.store.list()])
        if self.route == "/frame":
            return self._serve_frame()
        if self.route == "/video":
            return self._serve_video()
        if self.route.startswith("/jobs/"):
            jid = self.route.split("/")[2]
            try:
                d = self.store.get(jid).to_dict()
            except KeyError:
                return self._send(404, {"error": "unknown job"})
            d["media_retention"] = self.store.enforce_retention(jid)
            return self._send(200, d)
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
        st = self.store.enforce_retention(job.id)
        if not st["retained"]:
            return self._send(410, {"error": "media is no longer available",
                                    "media_retention": st})
        # A missing or misspelled frame parameter used to default to 0, so a
        # client asking for ?frame=2 got frame 0 with HTTP 200 and no warning.
        # For a seeding UI that means picking coordinates on the wrong picture.
        unknown = set(q) - {"job", "n"}
        if unknown:
            return self._send(400, {
                "error": f"unknown query parameter(s): {sorted(unknown)}; the "
                         f"frame number parameter is 'n'"})
        if "n" not in q:
            return self._send(400, {"error": "n is required: the frame number to "
                                             "decode. It is never defaulted."})
        try:
            n = int(q["n"])
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
        st = self.store.enforce_retention(job.id)
        if not st["retained"]:
            return self._send(410, {"error": "media is no longer available",
                                    "media_retention": st})
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
        if self.route.startswith("/jobs/") and self.route.endswith("/cancel"):
            jid = self.route.split("/")[2]
            try:
                self.store.cancel(jid)
                killed = self.store.kill_proc(jid)
                self.store.cleanup(jid)
                d = self.store.get(jid).to_dict()
                d["active_child_terminated"] = killed
                return self._send(200, d)
            except KeyError:
                return self._send(404, {"error": "unknown job"})
        if self.route.startswith("/jobs/") and self.route.endswith("/media/delete"):
            jid = self.route.split("/")[2]
            try:
                self.store.get(jid)
            except KeyError:
                return self._send(404, {"error": "unknown job"})
            st = self.store.revoke_media(jid, reason="deleted on operator request")
            return self._send(200, {"job": jid, "media_retention": st})
        if self.route.startswith("/jobs/") and self.route.endswith("/seeds"):
            jid = self.route.split("/")[2]
            try:
                job = self.store.get(jid)
            except KeyError:
                return self._send(404, {"error": "unknown job"})
            src = self.store.source(jid)
            if not src:
                return self._send(409, {"error": "job has no validated source yet"})
            # Retention is enforced BEFORE the seed is accepted or scheduled.
            # Seeding coordinates onto media that is gone -- or scheduling a
            # worker on it -- is refused even when the operator's own import
            # file still happens to exist on disk.
            st = self.store.enforce_retention(jid)
            if not st["retained"]:
                return self._send(410, {
                    "error": "media is no longer available for review; a seed "
                             "cannot be accepted or scheduled against it",
                    "media_retention": st})
            # A seed bundle is small. Bound it separately from the media cap,
            # and decide before reading a single byte of the body.
            raw_len = self.headers.get("Content-Length")
            if raw_len is None:
                return self._send(411, {"error": "Content-Length is required"})
            try:
                declared = int(raw_len)
            except ValueError:
                return self._send(400, {"error": "invalid Content-Length"})
            if declared <= 0:
                return self._send(400, {"error": "empty seed bundle"})
            if declared > MAX_SEED_BODY_BYTES:
                return self._send(413, {
                    "error": f"seed bundle of {declared} bytes exceeds the "
                             f"{MAX_SEED_BODY_BYTES} byte limit"})
            try:
                data = json.loads(self.rfile.read(declared) or b"{}")
                from ghostcaddie.upload.seeds import SeedBundle, SeedRejected
                bundle = SeedBundle.from_dict(data, source_sha256=src.sha256)
            except SeedRejected as e:
                return self._send(400, {"error": str(e)})
            except Exception as e:
                return self._send(400, {"error": f"{type(e).__name__}: {e}"})
            for sd in bundle.seeds:
                if sd.frame >= src.frames:
                    return self._send(400, {
                        "error": f"seed frame {sd.frame} is outside this source "
                                 f"(0..{src.frames - 1})"})
            self.store.attach_seeds(jid, bundle)
            # the whole point: a posted seed RUNS the assisted targets
            threading.Thread(target=_run_seeded,
                             args=(self.store, jid, src, bundle),
                             daemon=True).start()
            return self._send(200, {
                "accepted": bundle.to_dict(), "job": jid,
                "rerunning": list(SEEDED_TARGETS),
                "note": "assisted initialisation recorded, bound to this source "
                        "hash, and handed to the workers. This is USER-PROVIDED "
                        "ASSISTANCE: not human-verified, not AI-verified truth."})
        if self.route != "/jobs":
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


def build_server(root: str, host="127.0.0.1", port=0, limits=None,
                 media_retention_seconds=None):
    import secrets
    from ghostcaddie.upload.jobs import DEFAULT_MEDIA_RETENTION_SECONDS
    Handler.store = JobStore(root, media_retention_seconds=(
        DEFAULT_MEDIA_RETENTION_SECONDS if media_retention_seconds is None
        else media_retention_seconds))
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
