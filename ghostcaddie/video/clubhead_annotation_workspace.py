"""Offline sequence labeler for research-only golf clubhead annotations."""

from __future__ import annotations

import html
import json
import math
import posixpath
import re
from typing import Any, Mapping, Sequence


def _asset(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("frame filename must be non-empty")
    value = value.strip()
    if value.startswith(("/", "\\", "//")) or re.match(r"^[A-Za-z]:[\\/]", value) or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value):
        raise ValueError("frame filename must be local relative asset")
    normalized = posixpath.normpath(value.replace("\\", "/"))
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError("frame filename must not escape workspace")
    return normalized


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _frames_payload(frames: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(frames) != count or not frames:
        raise ValueError("frames length must equal positive video.frame_count")
    result = []
    for expected, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or set(frame) != {"frame_index", "source_frame_index", "timestamp_seconds", "filename"}:
            raise ValueError("frames have invalid fields")
        if frame["frame_index"] != expected:
            raise ValueError("frame_index values must be consecutive and zero-based")
        if not isinstance(frame["source_frame_index"], int) or frame["source_frame_index"] < 0:
            raise ValueError("source_frame_index must be a non-negative integer")
        if expected and frame["source_frame_index"] <= result[-1]["source_frame_index"]:
            raise ValueError("source_frame_index values must be strictly increasing")
        result.append({"frame_index": expected, "source_frame_index": frame["source_frame_index"], "timestamp_seconds": _number(frame["timestamp_seconds"], "timestamp_seconds"), "filename": _asset(frame["filename"])})
    return result


def build_clubhead_annotation_workspace(frames: Sequence[Mapping[str, Any]], *, video: Mapping[str, Any], clip_id: str, rights: str = "research_only_local") -> str:
    """Return deterministic local HTML for per-frame shaft/clubhead labels."""
    required = {"width", "height", "frame_count", "frame_rate"}
    if not isinstance(video, Mapping) or set(video) != required:
        raise ValueError("video must contain width, height, frame_count, frame_rate")
    width, height, count = video["width"], video["height"], video["frame_count"]
    if not all(isinstance(v, int) and v > 0 for v in (width, height, count)) or _number(video["frame_rate"], "frame_rate") <= 0:
        raise ValueError("video dimensions, count, and rate must be positive")
    if not isinstance(clip_id, str) or not clip_id.strip() or not isinstance(rights, str) or not rights.strip():
        raise ValueError("clip_id and rights must be non-empty")
    frame_data = _frames_payload(frames, count)
    frames_json = json.dumps(frame_data, sort_keys=True, separators=(",", ":"))
    initial = [{"frame_index": f["frame_index"], "source_frame_index": f["source_frame_index"], "timestamp_seconds": f["timestamp_seconds"], "clubhead": {"value": None, "visibility": "unavailable", "source": "unavailable"}, "shaft": {"value": None, "visibility": "unavailable", "source": "unavailable"}, "notes": []} for f in frame_data]
    state_json = json.dumps({"schema_version": "golf-research-clubhead-annotations.v1", "status": "draft", "explicit_submit": False, "video": dict(video, clip_id=clip_id), "split": "unassigned", "rights": rights, "provenance": {"label_type": "human", "pseudo_label": False, "ground_truth": True, "research_only": True, "production_eligible": False}, "frames": initial, "warnings": []}, sort_keys=True, separators=(",", ":"))
    items = "".join('<button type="button" class="frame" data-index="%d">Frame %d · source %d · %.3f s</button>' % (f["frame_index"], f["frame_index"], f["source_frame_index"], f["timestamp_seconds"]) for f in frame_data)
    title = html.escape("Manual clubhead/shaft labels — " + clip_id)
    return f'''<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>body{{font:14px system-ui;background:#10151c;color:#e8eef5;margin:0}}main{{display:grid;grid-template-columns:280px 1fr 260px;gap:12px;padding:12px}}section,aside{{border:1px solid #334252;padding:12px;border-radius:8px}}.frames{{display:grid;gap:5px;max-height:90vh;overflow:auto}}button,select{{background:#22303d;color:#e8eef5;border:1px solid #435466;border-radius:5px;padding:7px}}button.active{{border-color:#70d6a1}}#stage{{width:100%;aspect-ratio:{width}/{height};background:#18212c;cursor:crosshair}}#export{{width:100%;height:260px;background:#0d1319;color:#e8eef5}}label{{display:block;margin-top:10px}}</style>
<main><section><h2>Frames</h2><div class="frames">{items}</div></section><section><h2 id="heading">Frame 0</h2><svg id="stage" viewBox="0 0 {width} {height}" role="img" aria-label="Selected golf frame"><rect width="{width}" height="{height}" fill="#26313b"/><text x="{width/2}" y="{height/2}" text-anchor="middle" fill="#98a8b8">Frame image loads from local workspace</text></svg><p>Click the visible clubhead or shaft point only. Use unavailable for occluded/ambiguous evidence.</p></section><aside><label>Active label<select id="mode"><option value="clubhead">clubhead</option><option value="shaft_grip">shaft_grip</option><option value="shaft_neck">shaft_neck</option></select></label><label>Visibility<select id="visibility"><option>visible</option><option>occluded</option><option>ambiguous</option><option>unavailable</option></select></label><label>Source<select id="source"><option>human_ground_truth</option><option>human_confirmed</option><option>unavailable</option></select></label><button id="export" type="button">Export dataset</button><textarea id="json" aria-label="Exported dataset JSON" readonly></textarea><pre id="state"></pre><p>pseudo_label: false<br>ground_truth: true<br>research_only: true<br>production_eligible: false</p></aside></main>
<script>(function(){{"use strict";const frames={frames_json};let state={state_json};let selected=0;let mode="clubhead";const svg=document.getElementById("stage"),out=document.getElementById("json"),stateBox=document.getElementById("state");function render(){{const f=frames[selected];document.getElementById("heading").textContent="Frame "+f.frame_index+" · source "+f.source_frame_index+" · "+Number(f.timestamp_seconds).toFixed(3)+" s";svg.querySelectorAll("image,.marker").forEach(n=>n.remove());const image=document.createElementNS("http://www.w3.org/2000/svg","image");image.setAttribute("href",f.filename);image.setAttribute("width","{width}");image.setAttribute("height","{height}");svg.prepend(image);stateBox.textContent=JSON.stringify(state,null,2)}}document.querySelectorAll(".frame").forEach((b,i)=>b.addEventListener("click",()=>{{selected=i;document.querySelectorAll(".frame").forEach(x=>x.classList.remove("active"));b.classList.add("active");render()}}));document.getElementById("mode").addEventListener("change",e=>mode=e.target.value);svg.addEventListener("click",e=>{{const r=svg.getBoundingClientRect(),f=state.frames[selected],visibility=document.getElementById("visibility").value,source=document.getElementById("source").value;const point={{x:Number(((e.clientX-r.left)/r.width*{width}).toFixed(3)),y:Number(((e.clientY-r.top)/r.height*{height}).toFixed(3))}};if(visibility!=="visible"){{if(mode==="clubhead")f.clubhead={{value:null,visibility,source:"unavailable"}};else if(mode.startsWith("shaft_")){{f.shaft={{value:null,visibility,source:"unavailable"}}}}}}else if(mode==="clubhead")f.clubhead={{value:point,visibility,source}};else{{const current=f.shaft.value||{{grip:null,neck:null}};current[mode.slice(6)] = point;f.shaft={{value:current,visibility,source}}}}render()}});document.getElementById("export").addEventListener("click",()=>{{state.status="draft";state.explicit_submit=false;out.value=JSON.stringify(state,null,2)}});document.querySelector(".frame").classList.add("active");render()}})();</script>'''
