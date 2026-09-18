"""Local annotation UI for ball dataset v2.

A small single-clip server plus a browser page. It exists because annotating a
sequence by hand is the only way this dataset becomes ground truth, and the
controls that matter are the ones that make a careful pass fast: step, jump,
magnify, click a centre, mark absent/occluded/ambiguous, undo, resume, and see
progress.

Two invariants are enforced here rather than trusted to the page:

* **Source coordinates only.** A click is converted to native frame pixels before
  it reaches the store. The magnifier is a separate view and can never change
  label geometry.
* **States are not interchangeable.** ``VISIBLE`` without a coordinate is
  refused, and any non-visible state carrying a coordinate is refused. The UI
  cannot promote an ambiguous frame to a visible one by accident.

Machine guesses, when requested, are stored as proposals with their own origin
and never as labels.

Usage:
    python3 tools/ball_annotate_web.py --clip-id pexels_6573644
    python3 tools/ball_annotate_web.py --clip-id pexels_6573644 --port 8791 --open
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghostcaddie.video import ball_dataset as bd  # noqa: E402
from ghostcaddie.video.errors import VideoContractError  # noqa: E402

SOURCES = ROOT / "data" / "ball_dataset_v2" / "sources.json"
ANNOTATIONS = ROOT / "data" / "ball_dataset_v2" / "annotations"
CACHE = ROOT / "out" / "ball_dataset_v2" / "frame_cache"

PAGE = """<!doctype html>
<meta charset="utf-8"><title>ball annotate</title>
<style>
 body{background:#111;color:#eee;font:13px ui-monospace,Menlo,monospace;margin:0;display:flex}
 #left{padding:10px} #side{width:330px;padding:10px;border-left:1px solid #333}
 canvas{image-rendering:pixelated;border:1px solid #444;cursor:crosshair}
 #mag{width:260px;height:260px;border:1px solid #444;image-rendering:pixelated}
 .k{color:#8cf} .v{color:#fff} .warn{color:#f88}
 table{border-collapse:collapse;width:100%} td{padding:2px 4px}
 #bar{height:8px;background:#333;margin-top:6px} #barfill{height:8px;background:#4c9}
</style>
<div id="left">
  <div><span class="k">clip</span> <span class="v" id="clip">…</span></div>
  <div><span class="k">sha256</span> <span class="v" id="sha">…</span></div>
  <div><span class="k">frame</span> <span class="v" id="frame">0</span>
       / <span class="v" id="total">?</span>
       <span class="k">state</span> <span class="v" id="state">-</span></div>
  <canvas id="view" width="960" height="540"></canvas>
  <div id="bar"><div id="barfill" style="width:0"></div></div>
  <div class="k" id="hint"></div>
</div>
<div id="side">
  <div><b>magnifier</b> (never changes label geometry)</div>
  <canvas id="mag" width="260" height="260"></canvas>
  <div style="margin-top:8px">
    1 VISIBLE &nbsp; 2 NOT_VISIBLE &nbsp; 3 OCCLUDED &nbsp; 4 AMBIGUOUS<br>
    click = centre &nbsp; u = undo &nbsp; space = skip<br>
    &larr;/&rarr; frame &nbsp; shift+&larr;/&rarr; 10 frames &nbsp; n/p next/prev unlabelled<br>
    g = propose from previous label (machine guess, stays a proposal)
  </div>
  <div style="margin-top:8px">tag (hold for the next label):
    <select id="tag" multiple size="4">
      <option value="body">body</option><option value="clothing">clothing</option>
      <option value="shoe">shoe</option><option value="club">club</option>
      <option value="tee">tee</option><option value="flagstick">flagstick</option>
      <option value="glare">glare</option><option value="background_ball">background_ball</option>
      <option value="unknown_bright_object">unknown_bright_object</option>
    </select>
  </div>
  <div style="margin-top:8px"><textarea id="reason" rows="3" cols="38"
    placeholder="reason / explanation (required for NOT_VISIBLE and OCCLUDED)"></textarea></div>
  <table id="stats"></table>
  <div id="msg" class="warn"></div>
</div>
<script>
const q = new URLSearchParams(location.search);
let info = null, frame = 0, scale = 1, lastX = null, lastY = null, magAt = null;
const view = document.getElementById('view'), ctx = view.getContext('2d');
const mag = document.getElementById('mag'), mctx = mag.getContext('2d');

async function api(path, body) {
  const r = await fetch(path, body ? {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)} : {});
  const text = await r.text();
  let data; try { data = JSON.parse(text) } catch (e) { data = {error:text} }
  if (!r.ok) { document.getElementById('msg').textContent = 'refused: ' + (data.error||r.status); }
  else { document.getElementById('msg').textContent = data.note || ''; }
  return data;
}
function drawFrame() {
  const img = new Image();
  img.onload = () => {
    scale = img.width / info.width;
    view.width = img.width; view.height = img.height;
    ctx.drawImage(img, 0, 0);
    if (lastX !== null) {
      ctx.strokeStyle = '#0f0'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(lastX*scale, lastY*scale, Math.max(6, 3*scale), 0, 6.3); ctx.stroke();
    }
  };
  img.src = '/api/frame?index=' + frame + '&t=' + Date.now();
}
function drawMag() {
  if (!magAt) return;
  const img = new Image();
  img.onload = () => { mctx.drawImage(img, 0, 0, img.width, img.height); };
  img.src = `/api/crop?index=${frame}&x=${Math.round(magAt.x)}&y=${Math.round(magAt.y)}`
          + `&w=180&scale=6&t=${Date.now()}`;
}
async function refresh() {
  const p = await api('/api/progress');
  if (p.error) return;
  document.getElementById('total').textContent = info.frames_total ?? '?';
  const s = p.states || {};
  document.getElementById('stats').innerHTML =
    Object.entries(s).map(([k,v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('')
    + `<tr><td>labelled</td><td>${p.labelled}</td></tr>`
    + `<tr><td>skipped</td><td>${p.skipped.length}</td></tr>`
    + `<tr><td>proposals</td><td>${p.proposals}</td></tr>`;
  document.getElementById('barfill').style.width = (p.progress*100).toFixed(1) + '%';
  document.getElementById('hint').textContent =
    p.has_label ? ('frame ' + frame + ' label: ' + JSON.stringify(p.label))
                : ('frame ' + frame + ' unlabelled');
  return p;
}
async function label(state, x, y) {
  const tags = Array.from(document.getElementById('tag').selectedOptions).map(o => o.value);
  const reason = document.getElementById('reason').value.trim();
  const payload = {frame, state, origin:'AGENT_REVIEWED', reviewer:'agent-visual',
                   tags, reason, explanation: reason};
  if (state === 'VISIBLE') { payload.x = x; payload.y = y; }
  const r = await api('/api/label', payload);
  if (r.state) { lastX = r.x ?? null; lastY = r.y ?? null;
    document.getElementById('state').textContent = r.state; }
  drawFrame(); refresh();
}
async function step(delta) {
  frame = Math.max(0, Math.min((info.frames_total||1)-1, frame + delta));
  document.getElementById('frame').textContent = frame;
  lastX = lastY = null;
  const p = await api('/api/progress?frame=' + frame);
  if (p.label && p.label.state === 'VISIBLE') { lastX = p.label.x; lastY = p.label.y; }
  document.getElementById('state').textContent = p.label ? p.label.state : '-';
  drawFrame(); drawMag(); refresh();
}
view.addEventListener('mousemove', e => {
  const r = view.getBoundingClientRect();
  magAt = {x: (e.clientX - r.left) / scale, y: (e.clientY - r.top) / scale};
  drawMag();
});
view.addEventListener('click', e => {
  const r = view.getBoundingClientRect();
  label('VISIBLE', Math.round((e.clientX - r.left) / scale),
                    Math.round((e.clientY - r.top) / scale));
});
document.addEventListener('keydown', async e => {
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
  const codes = {'Digit1':'VISIBLE','Digit2':'NOT_VISIBLE','Digit3':'OCCLUDED','Digit4':'AMBIGUOUS'};
  if (codes[e.code]) { e.preventDefault(); label(codes[e.code]); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); step(e.shiftKey ? 10 : 1); }
  else if (e.key === 'ArrowLeft')  { e.preventDefault(); step(e.shiftKey ? -10 : -1); }
  else if (e.key === 'u') { await api('/api/undo', {frame}); lastX = lastY = null;
                            document.getElementById('state').textContent = '-';
                            drawFrame(); refresh(); }
  else if (e.key === ' ') { await api('/api/skip', {frame}); step(1); }
  else if (e.key === 'g') { const r = await api('/api/propose', {frame});
                            if (r.proposal) { magAt = r.proposal; drawMag(); } }
});
(async () => {
  info = await api('/api/clip');
  document.getElementById('clip').textContent = info.clip_id;
  document.getElementById('sha').textContent = (info.sha256||'').slice(0, 24) + '…';
  await step(0);
})();
</script>
"""


class AnnotatorState:
    """One clip under review, with the label store behind it."""

    def __init__(self, clip_id: str, *, sources_path: Path = SOURCES,
                 annotations_dir: Path = ANNOTATIONS, cache_dir: Path = CACHE,
                 project_root: Path = ROOT):
        self.sources_path = Path(sources_path)
        self.annotations_dir = Path(annotations_dir)
        self.cache_dir = Path(cache_dir)
        self.project_root = Path(project_root)
        self.sources = bd.load_sources(self.sources_path)
        self.by_id = bd.sources_by_id(self.sources)
        if clip_id not in self.by_id:
            raise SystemExit(f"{clip_id} is not in {self.sources_path}")
        self.clip_id = clip_id
        self.clip = self.by_id[clip_id]
        self.media_path = self.project_root / self.clip["media_path"]
        self.lock = threading.Lock()
        self.skipped: set[int] = set()
        self.annotation = bd.load_or_new_annotation(
            annotations_dir=self.annotations_dir, sources=self.sources,
            clip_id=clip_id, project_root=self.project_root, annotator="agent-visual",
            reviewer={"kind": "agent", "id": "agent-visual"}, bind=True)

    # -- media ------------------------------------------------------------
    def frame_bytes(self, index: int, *, scale_to: int = 960) -> bytes:
        import cv2
        if not self.media_path.exists():
            raise SystemExit(
                f"media missing: {self.media_path}. Media is not committed; "
                "re-acquire it with tools/ball_dataset_acquire.py.")
        capture = cv2.VideoCapture(str(self.media_path))
        if not capture.isOpened():
            raise SystemExit(f"cannot decode {self.media_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise SystemExit(f"frame {index} is not decodable")
        height, width = frame.shape[:2]
        if width > scale_to:
            factor = scale_to / width
            frame = cv2.resize(frame, (scale_to, int(height * factor)))
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise SystemExit("JPEG encode failed")
        return buffer.tobytes()

    def crop_bytes(self, index: int, *, x: float, y: float, w: int, scale: int) -> bytes:
        import cv2
        capture = cv2.VideoCapture(str(self.media_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise SystemExit(f"frame {index} is not decodable")
        height, width = frame.shape[:2]
        half = w // 2
        x0 = max(0, min(width - 1, int(x) - half))
        y0 = max(0, min(height - 1, int(y) - half))
        x1 = min(width, x0 + w)
        y1 = min(height, y0 + w)
        patch = frame[y0:y1, x0:x1]
        patch = cv2.resize(patch, (patch.shape[1] * scale, patch.shape[0] * scale),
                           interpolation=cv2.INTER_NEAREST)
        cv2.drawMarker(patch, ((int(x) - x0) * scale, (int(y) - y0) * scale),
                       (0, 255, 255), cv2.MARKER_CROSS, 18, 1)
        ok, buffer = cv2.imencode(".png", patch)
        return buffer.tobytes()

    # -- labels -----------------------------------------------------------
    def snapshot(self) -> dict:
        return dict(self.annotation)

    def set_label(self, *, frame: int, state: str, x=None, y=None, radius=None,
                  tags=None, reason=None, explanation=None, confidence=None,
                  origin=bd.ORIGIN_AGENT_REVIEWED, reviewer="agent-visual",
                  candidate_region=None) -> dict:
        with self.lock:
            updated = bd.set_label(
                self.annotation, frame, state=state, origin=origin, reviewer=reviewer,
                x=x, y=y, radius=radius, tags=tags or None, reason=reason,
                explanation=explanation, confidence=confidence,
                candidate_region=candidate_region)
            self.annotation = updated
            bd.save_annotation(bd.annotation_path_for(self.annotations_dir, self.clip_id),
                               updated, sources=self.sources,
                               project_root=self.project_root)
            return updated["frames"][str(int(frame))]

    def undo(self, frame: int) -> dict:
        with self.lock:
            updated = bd.drop_label(self.annotation, frame)
            self.annotation = updated
            bd.save_annotation(bd.annotation_path_for(self.annotations_dir, self.clip_id),
                               updated, sources=self.sources,
                               project_root=self.project_root)
            return updated

    def propose(self, frame: int, method: str = "carry_over") -> dict | None:
        """A machine guess from the previous reviewed frame. Stored as a proposal."""
        previous = [int(k) for k in self.annotation["frames"] if int(k) < frame]
        if not previous:
            return None
        source = self.annotation["frames"][str(max(previous))]
        if source["state"] != bd.STATE_VISIBLE:
            return None
        with self.lock:
            self.annotation = bd.set_proposal(self.annotation, frame, x=source["x"],
                                              y=source["y"], method=method)
            bd.save_annotation(bd.annotation_path_for(self.annotations_dir, self.clip_id),
                               self.annotation, sources=self.sources,
                               project_root=self.project_root)
            return {"x": source["x"], "y": source["y"], "method": method,
                    "status": "pending"}

    def progress(self, frame: int | None = None) -> dict:
        summary = bd.annotation_summary(self.annotation)
        total = self.clip.get("frames") or summary["labelled_frames"]
        label = (self.annotation["frames"].get(str(int(frame)))
                 if frame is not None else None)
        return {
            "clip_id": self.clip_id,
            "labelled": summary["labelled_frames"],
            "states": summary["states"],
            "proposals": summary["proposals"],
            "skipped": sorted(self.skipped),
            "progress": min(1.0, summary["labelled_frames"] / max(total, 1)),
            "has_label": label is not None,
            "label": label,
        }


def make_handler(state: AnnotatorState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # quiet by default
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict, code: int = 200) -> None:
            self._send(json.dumps(payload).encode(), "application/json", code)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode() or "{}")
            except json.JSONDecodeError:
                return {}

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    return self._send(PAGE.encode(), "text/html; charset=utf-8")
                if parsed.path == "/api/clip":
                    return self._json({
                        "clip_id": state.clip_id,
                        "sha256": state.clip["sha256"],
                        "width": state.clip["width"],
                        "height": state.clip["height"],
                        "fps": state.clip.get("fps"),
                        "frames_total": state.clip.get("frames"),
                        "media_path": state.clip["media_path"],
                        "media_present": state.media_path.exists(),
                        "license": state.clip["license"],
                        "source_url": state.clip["source_url"],
                    })
                if parsed.path == "/api/frame":
                    index = int(query.get("index", ["0"])[0])
                    return self._send(state.frame_bytes(index), "image/jpeg")
                if parsed.path == "/api/crop":
                    index = int(query.get("index", ["0"])[0])
                    x = float(query.get("x", ["0"])[0])
                    y = float(query.get("y", ["0"])[0])
                    w = int(query.get("w", ["180"])[0])
                    scale = max(1, min(12, int(query.get("scale", ["6"])[0])))
                    return self._send(state.crop_bytes(index, x=x, y=y, w=w, scale=scale),
                                      "image/png")
                if parsed.path == "/api/progress":
                    frame = query.get("frame")
                    return self._json(state.progress(int(frame[0]) if frame else None))
                return self._json({"error": f"unknown path {parsed.path}"}, 404)
            except SystemExit as exc:
                return self._json({"error": str(exc)}, 409)
            except VideoContractError as exc:
                return self._json({"error": str(exc)}, 400)

        def do_POST(self):  # noqa: N802
            parsed = urlparse(self.path)
            body = self._body()
            try:
                if parsed.path == "/api/label":
                    frame = int(body["frame"])
                    state_name = body["state"]
                    if state_name == bd.STATE_VISIBLE:
                        if body.get("x") is None or body.get("y") is None:
                            return self._json(
                                {"error": "VISIBLE requires x and y (source pixels)"},
                                400)
                        stored = state.set_label(
                            frame=frame, state=state_name, x=float(body["x"]),
                            y=float(body["y"]),
                            radius=(float(body["radius"]) if body.get("radius") else None),
                            tags=body.get("tags"), reason=None,
                            confidence=body.get("confidence"),
                            origin=body.get("origin", bd.ORIGIN_AGENT_REVIEWED),
                            reviewer=body.get("reviewer", "agent-visual"))
                    else:
                        if body.get("x") is not None or body.get("y") is not None:
                            return self._json(
                                {"error": f"{state_name} forbids coordinates; the ball "
                                          "has no defended position on that frame"}, 400)
                        stored = state.set_label(
                            frame=frame, state=state_name,
                            reason=body.get("reason"), explanation=body.get("explanation"),
                            tags=body.get("tags"), confidence=body.get("confidence"),
                            candidate_region=body.get("candidate_region"),
                            origin=body.get("origin", bd.ORIGIN_AGENT_REVIEWED),
                            reviewer=body.get("reviewer", "agent-visual"))
                    return self._json({"state": stored["state"], "x": stored.get("x"),
                                       "y": stored.get("y"), "frame": frame,
                                       "note": "saved"})
                if parsed.path == "/api/undo":
                    state.undo(int(body["frame"]))
                    return self._json({"note": "removed"})
                if parsed.path == "/api/skip":
                    state.skipped.add(int(body["frame"]))
                    return self._json({"skipped": sorted(state.skipped)})
                if parsed.path == "/api/propose":
                    proposal = state.propose(int(body["frame"]),
                                            method=body.get("method", "carry_over"))
                    return self._json({"proposal": proposal,
                                       "note": "machine guess recorded as a PROPOSAL; "
                                               "it is not a label"})
                return self._json({"error": f"unknown path {parsed.path}"}, 404)
            except VideoContractError as exc:
                return self._json({"error": str(exc)}, 400)

    return Handler


def build_server(clip_id: str, *, host: str = "127.0.0.1", port: int = 8791,
                 sources_path: Path = SOURCES, annotations_dir: Path = ANNOTATIONS,
                 project_root: Path = ROOT) -> tuple[ThreadingHTTPServer, AnnotatorState]:
    state = AnnotatorState(clip_id, sources_path=sources_path,
                           annotations_dir=annotations_dir,
                           project_root=project_root)
    server = ThreadingHTTPServer((host, port), make_handler(state))
    return server, state


def serve(clip_id: str, *, host: str = "127.0.0.1", port: int = 8791,
          sources_path: Path = SOURCES, annotations_dir: Path = ANNOTATIONS,
          project_root: Path = ROOT) -> None:
    server, state = build_server(clip_id, host=host, port=port,
                                 sources_path=sources_path,
                                 annotations_dir=annotations_dir,
                                 project_root=project_root)
    print(f"annotating {clip_id} at http://{host}:{port}/  "
          f"(media {'present' if state.media_path.exists() else 'MISSING'})")
    print("labels are written to "
          f"{bd.annotation_path_for(annotations_dir, clip_id)}")
    server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--sources", type=Path, default=SOURCES)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--open", action="store_true", help="print the URL only")
    args = parser.parse_args(argv)
    if args.open:
        print(f"http://{args.host}:{args.port}/")
    serve(args.clip_id, host=args.host, port=args.port, sources_path=args.sources,
          annotations_dir=args.annotations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
