"""Pure, deterministic offline HTML/SVG workspace for human video annotations.

The generator returns HTML and never reads or writes files.  Frame references are
intentionally relative local asset names so the result can be opened offline
next to an extracted frame directory/contact sheet.
"""

from __future__ import annotations

import html
import json
import math
import posixpath
import re
from collections.abc import Mapping, Sequence
from typing import Any


_POINT_MODES = (
    ("calibration_0", "Calibration source 1"),
    ("calibration_1", "Calibration source 2"),
    ("calibration_2", "Calibration source 3"),
    ("calibration_3", "Calibration source 4"),
    ("golfer_anchor", "Golfer anchor"),
    ("ball", "Ball"),
    ("clubhead", "Clubhead"),
    ("contact", "Contact"),
    ("target_intended_direction", "Intended target / direction"),
    ("landing", "Landing"),
)


def _safe_asset(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty relative asset name")
    value = value.strip()
    if (value.startswith(("/", "\\", "//")) or re.match(r"^[A-Za-z]:[\\/]", value)
            or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value)):
        raise ValueError(f"{field} must be a local relative asset name")
    normalized = posixpath.normpath(value.replace("\\", "/"))
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError(f"{field} must not escape the workspace")
    return normalized


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def _frames_payload(frames: Sequence[Any], frame_count: int) -> list[dict[str, Any]]:
    result = []
    first_index = None
    for expected, frame in enumerate(frames):
        if isinstance(frame, Mapping):
            index, timestamp, filename = frame.get("frame_index"), frame.get("timestamp_seconds"), frame.get("filename")
        else:
            index, timestamp, filename = getattr(frame, "frame_index", None), getattr(frame, "timestamp_seconds", None), getattr(frame, "filename", None)
        if first_index is None:
            first_index = index
        if index != first_index + expected or first_index not in (0, 1):
            raise ValueError("frames must have deterministic consecutive 0- or 1-based frame_index values")
        if timestamp is not None:
            _number(timestamp, f"frames[{expected}].timestamp_seconds")
        result.append({"frame_index": index, "timestamp_seconds": timestamp, "filename": _safe_asset(filename, f"frames[{expected}].filename")})
    if len(result) != frame_count:
        raise ValueError("frames length must equal video.frame_count")
    return result


def _initial_state(video: Mapping[str, Any], *, blank_draft: bool = False,
                   artifact_references: Sequence[str] = ()) -> dict[str, Any]:
    unavailable = lambda: {"value": None, "source": "unavailable"}
    return {
        "schema_version": "video-human-annotations.v1", "status": "draft", "explicit_submit": False,
        "video": dict(video),
        "calibration_points": [unavailable() for _ in range(4)],
        "engine_points": [None for _ in range(4)] if blank_draft else [{"x": 0.0, "y": 0.0}, {"x": 100.0, "y": 0.0}, {"x": 100.0, "y": 100.0}, {"x": 0.0, "y": 100.0}],
        "golfer_anchor": unavailable(), "ball": unavailable(), "clubhead": unavailable(),
        "contact": unavailable(), "target_intended_direction": unavailable(), "landing": unavailable(),
        "club_selection": {"value": "", "source": "unavailable"}, "context": {"value": "", "source": "unavailable"},
        "warnings": [],
    }


def build_annotation_draft(frames: Sequence[Any], *, video: Mapping[str, Any],
                           artifact_references: Sequence[str] = ()) -> dict[str, Any]:
    """Return the blank, explicitly incomplete v1 document for first-run prep."""
    if not isinstance(video, Mapping):
        raise ValueError("video must be a mapping")
    required = {"width", "height", "frame_count", "duration_seconds"}
    if set(video) != required:
        raise ValueError("video must contain exactly width, height, frame_count, duration_seconds")
    frame_data = _frames_payload(frames, int(video["frame_count"]))
    for reference in artifact_references:
        _safe_asset(reference, "artifact_references")
    return _initial_state(video, blank_draft=True)


def build_annotation_workspace(frames: Sequence[Any], *, video: Mapping[str, Any], contact_sheet_href: str | None = None,
                               title: str = "Offline shot annotation workspace", context: str = "",
                               blank_draft: bool = False) -> str:
    """Return a self-contained offline annotation workspace as deterministic HTML."""
    if not isinstance(video, Mapping):
        raise ValueError("video must be a mapping")
    required = {"width", "height", "frame_count", "duration_seconds"}
    if set(video) != required:
        raise ValueError("video must contain exactly width, height, frame_count, duration_seconds")
    width, height, count = int(video["width"]), int(video["height"]), int(video["frame_count"])
    if width <= 0 or height <= 0 or count <= 0:
        raise ValueError("video dimensions and frame_count must be positive")
    duration = _number(video["duration_seconds"], "video.duration_seconds")
    if duration < 0:
        raise ValueError("video.duration_seconds must not be negative")
    frame_data = _frames_payload(frames, count)
    assets_json = json.dumps(frame_data, sort_keys=True, separators=(",", ":"))
    state_json = json.dumps(_initial_state(video, blank_draft=blank_draft), sort_keys=True, separators=(",", ":"))
    title_text, context_text = html.escape(str(title)), html.escape(str(context))
    sheet = html.escape(_safe_asset(contact_sheet_href, "contact_sheet_href"), quote=True) if contact_sheet_href else ""
    modes = "".join(f'<button type="button" class="mode" data-mode="{html.escape(key, quote=True)}">{html.escape(label)}</button>' for key, label in _POINT_MODES)
    def frame_item(item: dict[str, Any]) -> str:
        timestamp = "Unavailable" if item["timestamp_seconds"] is None else f'{float(item["timestamp_seconds"]):.3f} s'
        index = item["frame_index"]
        return f'<button type="button" class="frame" data-frame="{index}"><span>Frame {index}</span><small>{html.escape(timestamp)}</small></button>'
    frame_items = "".join(frame_item(item) for item in frame_data)
    sheet_markup = f'<img class="contact-sheet" src="{sheet}" alt="Deterministic contact sheet">' if sheet else '<div class="unavailable">Contact sheet unavailable</div>'
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title_text}</title>
<style>
:root{{color-scheme:dark;--bg:#10151c;--panel:#18212c;--line:#334252;--ink:#e8eef5;--muted:#98a8b8;--accent:#70d6a1;--warn:#f0b35b}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px system-ui,sans-serif}}header{{padding:18px 24px;border-bottom:1px solid var(--line)}}h1{{margin:0 0 5px;font-size:22px}}.context{{color:var(--muted)}}main{{display:grid;grid-template-columns:270px minmax(420px,1fr) 330px;gap:16px;padding:16px;max-width:1600px;margin:auto}}section,aside{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}}h2{{font-size:14px;text-transform:uppercase;letter-spacing:.08em;margin:0 0 12px;color:var(--muted)}}.frames{{display:grid;gap:7px;max-height:72vh;overflow:auto}}button,select,input,textarea{{font:inherit}}button{{background:#22303d;color:var(--ink);border:1px solid #435466;border-radius:6px;padding:8px;cursor:pointer;text-align:left}}button:hover,button.active{{border-color:var(--accent);background:#254235}}.frame{{display:flex;justify-content:space-between;gap:8px}}small,.hint,.unavailable{{color:var(--muted)}}.stage-wrap{{display:grid;gap:10px}}.stage{{width:100%;aspect-ratio:{width}/{height};background:#0b0e12;border:1px solid var(--line);border-radius:8px;cursor:crosshair}}.stage image{{opacity:.8}}.contact-sheet{{max-width:100%;height:auto;border-radius:5px}}.mode-grid{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}label{{display:block;color:var(--muted);margin-top:11px}}input,select,textarea{{width:100%;background:#0d1319;color:var(--ink);border:1px solid var(--line);border-radius:5px;padding:8px}}textarea{{min-height:58px;resize:vertical}}.actions{{display:flex;gap:8px;margin-top:14px}}.actions button{{flex:1;text-align:center;font-weight:700}}.submit{{background:#7b4c2a}}.state{{white-space:pre-wrap;overflow:auto;max-height:250px;font:11px ui-monospace,monospace;background:#0d1319;padding:8px;border-radius:5px}}.badge{{display:inline-block;padding:3px 7px;border-radius:10px;background:#314052;color:var(--muted);font-size:12px}}.export-json{{min-height:170px;font:11px ui-monospace,monospace}}.export-actions{{display:flex;gap:8px;margin-top:8px}}.export-actions button{{flex:1;text-align:center}}.export-status{{min-height:20px;color:var(--accent)}}.export-error{{min-height:20px;color:var(--warn);white-space:pre-wrap}}@media(max-width:1050px){{main{{grid-template-columns:220px 1fr}}aside{{grid-column:1/-1}}}}@media(max-width:700px){{main{{display:block}}section,aside{{margin-bottom:12px}}}}
</style></head><body><header><h1>{title_text}</h1><div class="context">{context_text or "No context supplied"} · <span id="status" class="badge">DRAFT</span></div></header>
<main><section><h2>Extracted frames</h2><div class="frames">{frame_items}</div><p class="hint">Select a frame, then choose an annotation mode and click the SVG.</p></section>
<section class="stage-wrap"><h2 id="selection">Frame 1 · 0.000 s</h2><svg id="stage" class="stage" viewBox="0 0 {width} {height}" role="img" aria-label="Selected extracted frame"><rect width="{width}" height="{height}" fill="#26313b"/><text x="{width/2}" y="{height/2}" text-anchor="middle" fill="#98a8b8">Frame unavailable</text></svg><div class="hint" id="mode-help">Choose a point mode to place a coordinate.</div>{sheet_markup}</section>
<aside><h2>Point selection</h2><div class="mode-grid">{modes}</div><label for="confidence">Confidence <output id="confidence-value">0.80</output></label><input id="confidence" type="range" min="0" max="1" step="0.01" value="0.80"><label for="provenance">Provenance</label><select id="provenance"><option>user_supplied</option><option>user_confirmed</option><option>observed</option><option>inferred</option><option>unavailable</option></select><label for="phase">Phase</label><select id="phase"><option>address</option><option>backswing</option><option>top</option><option>downswing</option><option>contact</option><option>follow_through</option><option>landing</option></select><label for="club">Club selection</label><input id="club" placeholder="e.g. 7-iron"><label for="context-field">Context</label><textarea id="context-field" placeholder="lie, camera notes, or other context"></textarea><label for="warnings">Warnings (one per line)</label><textarea id="warnings" placeholder="Optional, explicit warnings"></textarea><div class="actions"><button id="save" type="button">Save Draft</button><button id="submit" class="submit" type="button">Submit Annotations</button></div><p class="hint">These buttons only prepare an export after your explicit click. Nothing is written silently.</p><h2>Export JSON</h2><textarea id="export-json" class="export-json" readonly aria-label="Copyable annotation JSON"></textarea><div class="export-actions"><button id="copy-json" type="button">Copy JSON</button><button id="download-json" type="button">Download JSON</button></div><p id="export-status" class="export-status" role="status"></p><p id="export-error" class="export-error" role="alert"></p><h2>Current state</h2><div id="state" class="state"></div></aside></main>
<script>
(function(){{"use strict";
const frames={assets_json};let state={state_json};let selected=0;let mode="calibration_0";const svg=document.getElementById("stage"),selection=document.getElementById("selection"),stateBox=document.getElementById("state");
const esc=s=>String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function frame(){{return frames[selected]}}function render(){{const f=frame();selection.textContent="Frame "+f.frame_index+" · "+(f.timestamp_seconds===null?"Unavailable":Number(f.timestamp_seconds).toFixed(3)+" s");svg.querySelectorAll(".frame-image,.marker").forEach(n=>n.remove());if(f.filename){{const image=document.createElementNS("http://www.w3.org/2000/svg","image");image.setAttribute("class","frame-image");image.setAttribute("href",f.filename);image.setAttribute("width","{width}");image.setAttribute("height","{height}");image.setAttribute("preserveAspectRatio","none");svg.prepend(image)}}stateBox.textContent=JSON.stringify(state,null,2);document.getElementById("status").textContent=state.status.toUpperCase()}}
function setMode(next){{mode=next;document.querySelectorAll(".mode").forEach(b=>b.classList.toggle("active",b.dataset.mode===mode));document.getElementById("mode-help").textContent="Active: "+mode+" · click the frame to place a coordinate."}}document.querySelectorAll(".frame").forEach((b,i)=>b.addEventListener("click",()=>{{selected=i;document.querySelectorAll(".frame").forEach(x=>x.classList.remove("active"));b.classList.add("active");render()}}));document.querySelectorAll(".mode").forEach(b=>b.addEventListener("click",()=>setMode(b.dataset.mode)));document.querySelector(".frame").classList.add("active");setMode(mode);
svg.addEventListener("click",e=>{{const r=svg.getBoundingClientRect(),x=Math.max(0,Math.min({width},(e.clientX-r.left)/r.width*{width})),y=Math.max(0,Math.min({height},(e.clientY-r.top)/r.height*{height}));const f=frame(),point={{x:Number(x.toFixed(3)),y:Number(y.toFixed(3)),frame_index:f.frame_index-(frames[0].frame_index===1?1:0),timestamp_seconds:f.timestamp_seconds,confidence:Number(document.getElementById("confidence").value),phase:document.getElementById("phase").value}};if(mode.startsWith("calibration_"))state.calibration_points[Number(mode.slice(-1))]={{...point,source:document.getElementById("provenance").value}};else if(mode==="contact")state.contact={{value:{{x:point.x,y:point.y,frame_index:point.frame_index,timestamp_seconds:point.timestamp_seconds,confidence:point.confidence,phase:point.phase}},source:document.getElementById("provenance").value}};else state[mode]={{value:point,source:document.getElementById("provenance").value}};render()}});
document.getElementById("confidence").addEventListener("input",e=>document.getElementById("confidence-value").textContent=e.target.value);document.getElementById("club").addEventListener("input",e=>state.club_selection={{value:e.target.value,source:e.target.value?"user_supplied":"unavailable"}});document.getElementById("context-field").addEventListener("input",e=>state.context={{value:e.target.value,source:e.target.value?"user_supplied":"unavailable"}});document.getElementById("warnings").addEventListener("input",e=>state.warnings=e.target.value.split("\\n").map(x=>x.trim()).filter(Boolean));
function sortedClone(value){{if(Array.isArray(value))return value.map(sortedClone);if(value&&typeof value==="object")return Object.keys(value).sort().reduce((out,key)=>(out[key]=sortedClone(value[key]),out),{{}});return value}}
function deterministicJson(payload){{return JSON.stringify(sortedClone(payload),null,2)}}
function buildExportPayload(status,explicitSubmit){{const payload=sortedClone(state);payload.status=status;payload.explicit_submit=explicitSubmit;return payload}}
function validateExportPayload(payload){{if(payload.schema_version!=="video-human-annotations.v1")throw new Error("invalid schema_version");if(!Array.isArray(payload.calibration_points)||payload.calibration_points.length!==4)throw new Error("calibration_points.length must be 4");if(!Array.isArray(payload.engine_points)||payload.engine_points.length!==4)throw new Error("engine_points.length must be 4");if(payload.status==="submitted"){{if(payload.calibration_points.some(point=>!point||point.source==="unavailable"||!point.value||!Number.isFinite(point.value.x)||!Number.isFinite(point.value.y)))throw new Error("four available finite calibration points are required");if(payload.engine_points.some(point=>!point||!Number.isFinite(point.x)||!Number.isFinite(point.y)))throw new Error("four finite engine points are required")}}if(payload.status==="submitted"&&!payload.explicit_submit)throw new Error("submitted_without_explicit_submit");if(payload.status==="draft"&&payload.explicit_submit)throw new Error("draft cannot be explicitly submitted");if(payload.status!=="draft"&&payload.status!=="submitted")throw new Error("status must be draft or submitted");return payload}}
function showExport(payload,action){{const text=deterministicJson(validateExportPayload(payload));document.getElementById("export-json").value=text;document.getElementById("export-error").textContent="";document.getElementById("export-status").textContent=action+" exported deterministic JSON; use Copy JSON or Download JSON.";render();return text}}
function exportDraft(){{try{{state.status="draft";state.explicit_submit=false;showExport(buildExportPayload("draft",false),"Draft")}}catch(error){{document.getElementById("export-error").textContent="Export validation error: "+esc(error.message)}}}}
function submitAnnotations(){{try{{const payload=buildExportPayload("submitted",true);validateExportPayload(payload);state.status="submitted";state.explicit_submit=true;showExport(payload,"Submitted")}}catch(error){{document.getElementById("export-error").textContent="Export validation error: "+esc(error.message)}}}}
function copyExport(){{const box=document.getElementById("export-json");if(!box.value){{document.getElementById("export-error").textContent="Export validation error: export JSON first";return}};if(navigator.clipboard&&window.isSecureContext)navigator.clipboard.writeText(box.value).then(()=>document.getElementById("export-status").textContent="JSON copied to clipboard.").catch(()=>{{box.select();document.execCommand("copy");document.getElementById("export-status").textContent="JSON selected for copying."}});else{{box.select();document.execCommand("copy");document.getElementById("export-status").textContent="JSON selected for copying."}}}}
function downloadExport(){{const text=document.getElementById("export-json").value;if(!text){{document.getElementById("export-error").textContent="Export validation error: export JSON first";return}}const blob=new Blob([text],{{type:"application/json"}}),url=URL.createObjectURL(blob),link=document.createElement("a");link.href=url;link.download=state.status+"-video-human-annotations.v1.json";link.click();URL.revokeObjectURL(url);document.getElementById("export-status").textContent="JSON download started."}}
document.getElementById("save").addEventListener("click",()=>exportDraft());document.getElementById("submit").addEventListener("click",()=>submitAnnotations());document.getElementById("copy-json").addEventListener("click",()=>copyExport());document.getElementById("download-json").addEventListener("click",()=>downloadExport());render();
}})();</script></body></html>
'''


# Descriptive compatibility alias for callers using the milestone terminology.
generate_annotation_workspace = build_annotation_workspace
