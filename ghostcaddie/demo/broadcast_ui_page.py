"""The broadcast review UI's single page (markup, styles and script).

Kept apart from the server so the page's own behaviour is readable in one place: it is
one ``str`` with no Python in it, and the parts of it that must agree with the backend
(the display->native coordinate mapping, the vocabulary tables) are the parts the tests
read back out of it.

What the page is careful about
------------------------------
* **The document is the state.** Loading a config deep-clones the operator's own JSON
  and edits THAT document. Nothing is rebuilt from a list of known fields, so
  ``roi_keyframes``, ``manual_segments``, ``presentation``, ``timing``, ``cuts``,
  ``overrides``, ``notes`` and any key the schema merely tolerates survive a save
  byte for byte. A save can therefore never quietly drop an assistance mark.
* **A config loads onto its own clip.** ``clip_path_hint`` / ``expected_source_sha256``
  are resolved against the registry and the matching clip is selected (with its frame
  bounds) BEFORE any scene field is loaded; when nothing matches, the load is refused
  and the operator is told to import that clip.
* **Edits commit to the selected scene.** Phase marks, the ROI, seeds, the window and
  the stroke type are written into ``scenes[selected]`` as they change, and selecting a
  scene loads its fields back (``apply to selected scene`` re-applies them explicitly).
* **Assistance marks are editable.** Corrections, ROI keyframes and manual-segment
  anchors are listed per target and can be added to or removed; the page never rewrites
  a block it is not editing.
"""

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fairway OS · Broadcast review</title>
<style>
  :root{
    --ink:#0d1113; --panel:#141a1c; --panel-2:#1b2325; --line:#2a3538;
    --teal:#70d4b0; --teal-soft:#3f7f6c; --amber:#ffb030; --text:#f0f4f3;
    --muted:#9fb0b2; --ok:#70d4b0; --bad:#ff7a6b; --gap:8px;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--ink);color:var(--text);
    font:14px/1.45 Arial,Helvetica,sans-serif}
  header{display:flex;align-items:center;gap:var(--gap);padding:8px 12px;
    border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
  header h1{font-size:15px;margin:0;letter-spacing:.02em}
  header .sub{color:var(--muted);font-size:12px;min-width:0}
  main{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:var(--gap);
    padding:var(--gap)}
  @media (max-width:1000px){main{grid-template-columns:minmax(0,1fr)}}
  .stage{background:#000;border:1px solid var(--line);border-radius:4px;
    padding:var(--gap);display:flex;flex-direction:column;gap:var(--gap);
    min-width:0}
  .canvas-wrap{position:relative;display:inline-block;max-width:100%}
  canvas{display:block;max-width:100%;height:auto;cursor:crosshair}
  .timeline{display:flex;align-items:center;gap:var(--gap);flex-wrap:wrap}
  input[type=range]{flex:1 1 160px;min-width:120px;accent-color:var(--teal)}
  .timeline .chip{margin-left:0}
  @media (max-width:460px){
    .timeline input[type=range]{flex:1 1 100%;order:-1}
    .timeline button{flex:1 1 auto}
    .panel{max-height:none}
  }
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:4px;
    padding:12px;display:flex;flex-direction:column;gap:var(--gap);
    max-height:calc(100vh - 90px);overflow:auto;min-width:0}
  .panel h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;
    color:var(--muted);margin:0}
  label{display:block;font-size:12px;color:var(--muted);margin-bottom:2px}
  select,input[type=text],input[type=number]{width:100%;background:var(--panel-2);
    color:var(--text);border:1px solid var(--line);border-radius:3px;padding:6px}
  button{background:var(--panel-2);color:var(--text);border:1px solid var(--line);
    border-radius:3px;padding:7px 10px;font:inherit;cursor:pointer}
  button:hover{border-color:var(--teal-soft)}
  button.primary{background:var(--teal-soft);border-color:var(--teal)}
  button.accent{background:#7a4d00;border-color:var(--amber)}
  button.on{background:var(--teal-soft);border-color:var(--teal)}
  button:focus-visible,select:focus-visible,input:focus-visible{
    outline:2px solid var(--amber);outline-offset:1px}
  .row{display:flex;gap:var(--gap);flex-wrap:wrap}
  .row>button{flex:1 1 auto}
  .phases button{flex:1 1 46%}
  .chip{display:inline-block;background:var(--panel-2);border:1px solid var(--line);
    border-radius:999px;padding:2px 8px;font-size:12px;color:var(--muted);
    max-width:100%;overflow-wrap:anywhere}
  .chip.on{border-color:var(--teal);color:var(--teal)}
  .scene{border:1px solid var(--line);border-radius:4px;padding:8px;
    background:var(--panel-2);cursor:pointer}
  .scene[aria-current=true]{border-color:var(--teal)}
  .scene h3{margin:0 0 4px;font-size:13px}
  table{width:100%;border-collapse:collapse;font-size:12px}
  td,th{text-align:left;padding:3px 4px;border-bottom:1px solid var(--line)}
  .muted{color:var(--muted)}
  .status{padding:6px 8px;border-radius:3px;border:1px solid var(--line);
    font-size:12px;background:var(--panel-2)}
  .status.ok{border-color:var(--ok);color:var(--ok)}
  .status.bad{border-color:var(--bad);color:var(--bad)}
  code{font-size:12px;color:var(--muted);word-break:break-all}
  .anchors{font-size:12px}
  .anchors div{display:flex;gap:6px;align-items:center;justify-content:space-between;
    border-bottom:1px solid var(--line);padding:2px 0}
  .anchors span{overflow-wrap:anywhere}
  .anchors button{padding:2px 6px;font-size:11px}
  @media (prefers-reduced-motion:reduce){*{transition:none!important;
    animation:none!important}}
</style>
</head>
<body>
<header>
  <h1>FAIRWAY OS</h1>
  <span class="sub">broadcast review · whole shot, overlays only inside scene windows</span>
</header>
<main>
  <section class="stage">
    <div class="canvas-wrap"><canvas id="view" width="1266" height="720"></canvas></div>
    <div class="timeline">
      <button id="prev" title="previous frame">◀</button>
      <input type="range" id="scrub" min="0" max="0" value="0" step="1"
             aria-label="source frame">
      <button id="next" title="next frame">▶</button>
      <span class="chip" id="frameChip">frame 0</span>
      <span class="chip" id="coordChip">x – y –</span>
    </div>
    <div class="status" id="status">loading…</div>
    <div class="status" aria-label="tracking gap review">
      <strong>Review tracking gaps</strong>
      <div class="row">
        <select id="reviewTarget" aria-label="target to review">
          <option value="clubhead">clubhead</option><option value="ball">ball</option>
        </select>
        <select id="reviewGap" aria-label="saved tracking gap"></select>
        <button id="previousGap">previous gap</button>
        <button id="nextGap">next gap</button>
      </div>
      <label><input type="checkbox" id="previewAutomatic" checked> Show saved automatic positions (amber)</label>
      <div class="muted" id="automaticInfo">No saved automatic positions loaded.</div>
      <div class="muted" id="reviewInfo">No saved tracking results loaded.</div>
    </div>
  </section>
  <aside class="panel">
    <h2>Export</h2>
    <label for="trackingMode">Target positions</label>
    <select id="trackingMode">
      <option value="reviewed">Reviewed showcase (assisted)</option>
      <option value="saved_automatic" disabled>Saved automatic targets (research)</option>
    </select>
    <div class="muted">Automatic export uses saved observations and the original seed. Gaps stay empty; current seed edits and assisted paths are excluded.</div>
    <h2>Clip</h2>
    <select id="clip" aria-label="clip"></select>
    <div class="muted" id="clipInfo"></div>
    <div class="row">
      <div style="flex:2 1 140px"><label for="importFile">import from an allowed
        folder</label><select id="importFile"></select></div>
      <button id="doImport" style="flex:0 0 auto;align-self:flex-end">import</button>
    </div>
    <div class="row">
      <div style="flex:2 1 140px"><label for="importPath">…or an explicit path inside
        an allowed folder</label>
        <input type="text" id="importPath" placeholder="/path/to/clip.mp4"></div>
      <button id="doImportPath" style="flex:0 0 auto;align-self:flex-end">import
        path</button>
    </div>

    <h2>Or load an existing config</h2>
    <div class="row">
      <div style="flex:2 1 140px"><select id="configFile"></select></div>
      <button id="loadConfig" style="flex:0 0 auto">load</button>
    </div>
    <div class="muted" id="configInfo">no config loaded — the fields below are a new
      document for the selected clip</div>

    <h2>Stroke (the shot's own stroke type)</h2>
    <select id="strokeType">
      <option value="full_swing">full swing</option>
      <option value="putt">putt</option>
      <option value="chip">chip</option>
      <option value="pitch">pitch</option>
      <option value="bunker">bunker</option>
      <option value="unknown">unknown</option>
    </select>

    <h2>Scene window (inclusive source frames)</h2>
    <div class="row">
      <div style="flex:1"><label for="wStart">start</label>
        <input type="number" id="wStart" min="0" value="0"></div>
      <div style="flex:1"><label for="wEnd">end</label>
        <input type="number" id="wEnd" min="0" value="0"></div>
    </div>
    <div class="row">
      <button id="setStart">set start here</button>
      <button id="setEnd">set end here</button>
    </div>
    <div class="muted" id="windowInfo"></div>

    <h2>Golfer region (drag on the picture)</h2>
    <div class="row">
      <button id="clearRoi">clear region</button>
      <button id="saveRoiKey">save as ROI keyframe</button>
    </div>
    <div class="muted" id="roiInfo">no region</div>

    <h2>Phases</h2>
    <div class="phases row" id="phaseButtons"></div>
    <div class="muted" id="phaseInfo"></div>

    <h2>Place on the picture</h2>
    <div class="row">
      <button id="modeRegion" class="on">region (drag)</button>
      <button id="modeSeed">seed (click)</button>
      <button id="modeCorrection">correction (click)</button>
    </div>
    <div class="row">
      <div style="flex:1"><label for="target">target</label>
        <select id="target"><option value="ball">ball</option>
          <option value="clubhead">clubhead</option></select></div>
      <button id="placeSeed" style="flex:0 0 auto;align-self:flex-end">set the seed
        from the fields</button>
    </div>
    <div class="muted" id="seedInfo">region mode: drag a box. seed/correction mode:
      click a point. A seed asks you to confirm the selected target is visible; a correction does
      not claim anything.</div>

    <h2>Assistance marks on the selected scene</h2>
    <div class="anchors" id="anchors"></div>
    <div class="row">
      <div style="flex:2 1 140px"><label for="anchorSegment">add the last clicked
        point into a manual segment</label><select id="anchorSegment"></select></div>
      <button id="addAnchor" style="flex:0 0 auto">add anchor</button>
    </div>
    <div class="row">
      <div style="flex:2 1 140px"><label for="anchorTarget">new segment target</label>
        <select id="anchorTarget"><option value="ball">ball</option>
          <option value="clubhead">clubhead</option></select></div>
      <button id="newSegment" style="flex:0 0 auto">new segment here</button>
    </div>

    <h2>Scenes</h2>
    <div id="scenes"></div>
    <div class="muted" id="sceneHint">click a scene to select and load its fields</div>
    <div class="row">
      <button id="applyScene" class="accent">apply fields to selected scene</button>
      <button id="addScene" class="primary">add as a new scene</button>
    </div>
    <div class="row"><button id="removeScene">remove selected scene</button></div>

    <h2>Cut candidates (suggestions only)</h2>
    <div class="row"><button id="discover">scan this clip</button></div>
    <div id="cuts" class="muted">not scanned</div>

    <h2>Finish</h2>
    <input type="text" id="configName" placeholder="config file name"
           aria-label="config file name">
    <div class="row">
      <button id="validate">validate</button>
      <button id="save" class="primary">save config</button>
    </div>
    <div class="row"><button id="render" class="primary">render</button></div>
    <div id="result"></div>
  </aside>
</main>
<script>
const TOKEN = "__TOKEN__";
const VOCAB = {
  full_swing:["address","takeaway","top","downswing","impact","follow_through","finish"],
  chip:["address","takeaway","top","downswing","impact","follow_through","finish"],
  pitch:["address","takeaway","top","downswing","impact","follow_through","finish"],
  bunker:["address","takeaway","top","downswing","impact","follow_through","finish"],
  unknown:["address","takeaway","top","downswing","impact","follow_through","finish"],
  putt:["address","backstroke","forward_stroke","contact","finish"]
};
const state = {
  clips: [], clip: null, frame: 0,
  // The operator's own document. Everything is edited IN it; nothing is rebuilt from a
  // list of known fields, which is what used to lose manual segments and ROI keyframes.
  config: null, loadedName: null, dirty: false,
  scenes: [], current: -1,
  phases: {}, roi: null, seeds: {},
  mode: "region", target: "ball", lastPoint: null,
  dragging: null, job: null, tracking: null, savedPoints: null, savedAvailable: false,
  trackingPreference: null, reviewRequest: 0
};
const $ = (id) => document.getElementById(id);
function clone(value){
  if (value === undefined) return undefined;
  if (typeof structuredClone === "function") {
    try { return structuredClone(value); } catch (e) { /* fall through */ }
  }
  return JSON.parse(JSON.stringify(value));
}

// Native-coordinate mapping. The image is displayed scaled and may be letterboxed by
// object-fit; this converts a pointer position to NATIVE SOURCE pixels and returns
// null for a click outside the picture, exactly like the desktop picker.
function viewportToNative(rect, natural, clientX, clientY){
  const scale = Math.min(rect.width / natural.w, rect.height / natural.h);
  const drawnW = natural.w * scale, drawnH = natural.h * scale;
  const offX = rect.left + (rect.width - drawnW) / 2;
  const offY = rect.top + (rect.height - drawnH) / 2;
  const x = (clientX - offX) / scale, y = (clientY - offY) / scale;
  if (x < 0 || y < 0 || x > natural.w || y > natural.h) return null;
  return {x: Math.round(x), y: Math.round(y), scale};
}

async function api(path, options){
  const res = await fetch(path, options);
  const text = await res.text();
  let payload = {};
  try { payload = JSON.parse(text); } catch(e) { payload = {error:text}; }
  if (!res.ok || payload.ok === false)
    throw new Error(payload.error || res.statusText);
  return payload;
}
function post(path, body){
  return api(path, {method:"POST", headers:{"Content-Type":"application/json",
    "X-Broadcast-Token":TOKEN}, body:JSON.stringify(body)});
}
function say(message, kind){
  const el = $("status");
  el.textContent = message;
  el.className = "status" + (kind ? " " + kind : "");
}

// ---- loading ------------------------------------------------------------
async function loadClips(){
  const data = await api("/api/clips");
  state.clips = data.clips;
  $("clip").innerHTML = data.clips.map(c =>
    `<option value="${c.id}">${c.name}</option>`).join("");
  if (data.clips.length){ selectClip(data.clips[0].id); }
  else say("no clips found in the allow-listed roots", "bad");
  await loadLibrary();
  await loadConfigList();
}
async function loadLibrary(){
  try{
    const data = await api("/api/library");
    const options = [];
    for (const root of data.roots){
      const files = root.files || (root.media || []).map(n => ({name:n}));
      for (const file of files)
        options.push(`<option value="${root.root}|${file.name}">${file.name}` +
                     ` <span class="muted">(${root.root})</span></option>`);
    }
    $("importFile").innerHTML = options.join("") ||
      "<option value=''>nothing to import here</option>";
  }catch(e){ /* the clip list already failed and said so */ }
}
async function loadConfigList(){
  try{
    const data = await api("/api/configs");
    $("configFile").innerHTML = (data.configs || []).map(c =>
      `<option value="${c.name}">${c.name} (${c.scenes} scene(s)` +
      (c.clip_name ? `, ${c.clip_name}` : ", clip not in the library") +
      (c.loadable ? "" : ", NOT LOADABLE") + ")</option>").join("") ||
      "<option value=''>no broadcast configs in this folder</option>";
  }catch(e){ /* ignore */ }
}
async function doImport(){
  const value = $("importFile").value;
  if (!value) return;
  const cut = value.indexOf("|");
  const root = value.slice(0, cut), name = value.slice(cut + 1);
  try{
    const data = await post("/api/import", {root, name});
    await loadClips();
    $("clip").value = data.clip.id;
    selectClip(data.clip.id);
    say("imported " + data.clip.name, "ok");
  }catch(e){ say("import failed: " + e.message, "bad"); }
}
async function doImportPath(){
  const path = $("importPath").value.trim();
  if (!path) return;
  try{
    const data = await post("/api/import", {path});
    await loadClips();
    $("clip").value = data.clip.id;
    selectClip(data.clip.id);
    say("imported " + data.clip.name, "ok");
  }catch(e){ say("import failed: " + e.message, "bad"); }
}
function resolveClipForConfig(config){
  // Which registered clip this document was written for: the digest first (it is
  // authoritative), then the path hint (exact path, then file name).
  const shot = (config && config.shot) || {};
  const expected = String(shot.expected_source_sha256 || "");
  if (expected){
    const bySha = state.clips.find(c => c.sha256 === expected);
    if (bySha) return {clip: bySha, how: "expected_source_sha256"};
    return {clip: null, how: "no clip in the library has this config's sha256"};
  }
  const hint = String((config && config.clip_path_hint) || "");
  if (hint){
    const base = hint.split("/").pop();
    const byPath = state.clips.find(c => c.path === hint);
    if (byPath) return {clip: byPath, how: "clip_path_hint"};
    const byName = state.clips.find(c => c.name === base);
    if (byName) return {clip: byName, how: "clip_path_hint (by file name)"};
    return {clip: null, how: "the clip this config names is not in the library"};
  }
  return {clip: null, how: "the config names neither a clip path nor a digest"};
}
async function loadConfigFile(){
  const name = $("configFile").value;
  if (!name) return;
  try{
    const listing = await api("/api/configs");
    const entry = (listing.configs || []).find(c => c.name === name);
    if (entry && entry.loadable === false){
      say("that config would not load: " + (entry.error || "refused"), "bad");
      return;
    }
    const config = await api("/api/config?name=" + encodeURIComponent(name));
    let resolved = {clip: null, how: "no match"};
    if (entry && entry.clip_id){
      const clip = state.clips.find(c => c.id === entry.clip_id);
      if (clip) resolved = {clip, how: entry.clip_resolved_by || "the config listing"};
    }
    if (!resolved.clip) resolved = resolveClipForConfig(config);
    if (!resolved.clip){
      // Refused, and nothing is loaded: importing a config's scenes onto whichever clip
      // happened to be selected is exactly the defect this replaces.
      say("not loaded: " + resolved.how + ". Import that clip above, then load the " +
          "config again — the fields below are still the previous document.", "bad");
      return;
    }
    // The clip is selected (which refreshes the frame bounds) BEFORE any config field
    // is put on screen, so a window is always read against its own footage.
    $("clip").value = resolved.clip.id;
    selectClip(resolved.clip.id);
    applyConfig(config, name, resolved.how);
    say("loaded " + name + " onto " + resolved.clip.name + " (matched by " +
        resolved.how + ") · " + state.scenes.length + " scene(s) · save writes this " +
        "document back with your edits", "ok");
  }catch(e){ say("could not load that config: " + e.message, "bad"); }
}
function defaultPresentation(){
  const shotType = $("strokeType").value;
  return {
    brand:"FAIRWAY OS",
    brand_sub: shotType === "putt" ? "STROKE ANALYSIS" : "SWING ANALYSIS",
    disclosure:"OPERATOR-ASSISTED · BROADCAST FOOTAGE · 2D IMAGE-SPACE",
    metrics: shotType === "putt"
      ? ["shoulder_tilt","hip_tilt","sway","tempo"]
      : ["shoulder_tilt","hip_tilt","knee_flexion","tempo"],
    card_debounce_frames:1, impact_accent:true, impact_accent_span_frames:12
  };
}
function ensureConfig(){
  if (state.config) return state.config;
  state.config = {
    schema_version:"fairwayos-broadcast-config.v1",
    clip_path_hint: state.clip ? state.clip.path : "",
    shot:{id:"", label:"", stroke_type: $("strokeType").value,
          expected_source_sha256: state.clip ? state.clip.sha256 : ""},
    interval:{start_seconds:0.0, end_seconds:null, max_frames:20000},
    outside_windows:"brand_only",
    presentation: defaultPresentation(),
    render:{max_width:1920, crf:18, preset:"medium", render:true},
    scenes:[]
  };
  state.scenes = state.config.scenes;
  return state.config;
}
function applyConfig(config, name, how){
  state.config = clone(config);
  if (!Array.isArray(state.config.scenes)) state.config.scenes = [];
  state.scenes = state.config.scenes;
  const shot = state.config.shot || {};
  const stroke = shot.stroke_type || state.scenes[0]?.stroke_type || "full_swing";
  $("strokeType").value = stroke;
  state.loadedName = name || null;
  state.dirty = false;
  $("configName").value = name || $("configName").value;
  $("configInfo").textContent =
    `${name} · ${state.scenes.length} scene(s) · loaded from ${how || "the library"} · ` +
    `presentation/cuts/timing/overrides are carried through on save`;
  selectScene(state.scenes.length ? 0 : -1);
  renderPhases(); renderScenes(); renderAnchors(); updateWindowInfo();
}
function selectClip(id){
  const changed = state.clip && state.clip.id !== id;
  state.clip = state.clips.find(c => c.id === id);
  if (!state.clip) return;
  $("wEnd").value = Math.max(0, (state.clip.frame_count || 1) - 1);
  $("scrub").max = Math.max(0, (state.clip.frame_count || 1) - 1);
  $("clipInfo").textContent =
    `${state.clip.width}×${state.clip.height} · ${state.clip.fps_rational} fps · ` +
    `${state.clip.frame_count} frames` + (state.clip.has_audio ? " · audio" : " · no audio");
  if (changed){
    // A different source starts a different document. Loading a saved config calls
    // applyConfig after this reset; a direct clip change must not inherit its marks.
    state.config = null; state.scenes = []; state.current = -1;
    state.roi = null; state.phases = {}; state.seeds = {};
    state.dirty = false; state.loadedName = null;
    $("wStart").value = 0;
    $("configName").value = state.clip.name.replace(/\.[^.]+$/, "") + "_ui.json";
    $("configInfo").textContent = "new document for " + state.clip.name + "; load its saved config to restore marks";
    $("roiInfo").textContent = "no region";
    renderScenes(); renderAnchors(); renderPhases();
  }
  state.lastPoint = null;
  $("coordChip").textContent = "x – y –";
  state.frame = 0;
  setFrame(0);
  loadTrackingReview();
}

// Read-only navigation: result gaps never become seeds, corrections or detections.
function renderTrackingReview(message){
  const review = state.tracking, selector = $("reviewGap");
  selector.innerHTML = "";
  const gaps = review?.gaps || [];
  gaps.forEach((gap, index) => {
    const option = document.createElement("option"); option.value = index;
    option.textContent = `frames ${gap.start_frame}–${gap.end_frame}`;
    selector.appendChild(option);
  });
  selector.value = gaps.length ? "0" : "";
  selector.disabled = !gaps.length;
  $("previousGap").disabled = $("nextGap").disabled = !gaps.length;
  $("reviewInfo").textContent = message || (!review
    ? "No saved tracking results loaded for this target."
    : `${review.automatic_frames} frames with automatic positions · ${gaps.length} gaps ` +
      (review.start_frame === null ? "· no seeded result range" :
       `· saved range ${review.start_frame}–${review.end_frame}`) +
      ". Gaps may be occlusion or tracking loss. Position coverage does not measure accuracy.");
}
async function loadTrackingReview(){
  const generation = ++state.reviewRequest, clip = state.clip;
  const target = $("reviewTarget").value || "clubhead";
  state.tracking = null; state.savedPoints = null; state.savedAvailable = false;
  $("trackingMode").value = state.trackingPreference || "reviewed";
  $("render").disabled = true;
  const savedChoice = $("trackingMode").querySelector('option[value="saved_automatic"]');
  if (savedChoice) savedChoice.disabled = true;
  renderTrackingReview("Loading saved tracking results…");
  $("automaticInfo").textContent = "Loading saved positions…";
  if (clip) setFrame(state.frame);
  if (!clip) { $("render").disabled = false; return; }
  try {
    const data = await api(`/api/tracking?clip=${encodeURIComponent(clip.id)}&target=${encodeURIComponent(target)}`);
    if (generation !== state.reviewRequest || state.clip?.id !== clip.id) return;
    const review = data.tracking;
    if (review && (review.source_sha256 !== clip.sha256 || review.target !== target))
      throw new Error("saved results belong to another source or target");
    state.tracking = review || null; renderTrackingReview();
    const positions = await api(`/api/tracking/points?clip=${encodeURIComponent(clip.id)}&target=${encodeURIComponent(target)}`);
    if (generation !== state.reviewRequest || state.clip?.id !== clip.id) return;
    if ((positions.available || positions.result) && (positions.source_sha256 !== clip.sha256 || positions.target !== target))
      throw new Error("saved positions belong to another source or target");
    state.savedPoints = positions.result || null;
    state.savedAvailable = !!positions.available;
    $("trackingMode").value = state.trackingPreference ||
      (state.savedAvailable ? "saved_automatic" : "reviewed");
    const choice = $("trackingMode").querySelector('option[value="saved_automatic"]');
    if (choice) choice.disabled = !state.savedAvailable;
    updateAutomaticInfo();
    setFrame(state.frame);
  } catch (error) {
    if (generation === state.reviewRequest) {
      state.savedPoints = null; state.savedAvailable = false;
      $("automaticInfo").textContent = "Saved positions unavailable.";
      renderTrackingReview("Tracking review unavailable: " + error.message);
    }
  } finally {
    if (generation === state.reviewRequest) $("render").disabled = false;
  }
}
function automaticPoint(){
  if (!$("previewAutomatic").checked || !state.savedPoints) return null;
  // Export only uses the scene containing the original saved operator seed.
  const seedFrame = state.savedPoints.seed?.frame;
  const scene = state.scenes.find(s => (s.kind || "golfer_stroke") === "golfer_stroke" && s.window && seedFrame >= s.window.start_frame && seedFrame <= s.window.end_frame);
  if (!scene || state.frame < scene.window.start_frame || state.frame > scene.window.end_frame) return null;
  const p = (state.savedPoints.points || []).find(p => p.source_frame === state.frame);
  return p && p.origin === "automatic" &&
    ["observed_template", "observed_template_lk"].includes(p.status) &&
    Number.isFinite(p.x) && Number.isFinite(p.y) ? p : null;
}
function updateAutomaticInfo(){
  const p = automaticPoint();
  $("automaticInfo").textContent = p
    ? `Saved automatic ${$("reviewTarget").value || "clubhead"} · source frame ${state.frame} · ${p.method}`
    : state.savedPoints ? `No automatic position exported at source frame ${state.frame}; gaps and frames outside the original seed’s scene stay empty.`
                        : "No saved automatic positions loaded.";
}
function drawAutomatic(ctx, canvas){
  const p = automaticPoint();
  if (!p) return;
  ctx.save(); ctx.fillStyle = "#ffb030"; ctx.beginPath();
  ctx.arc(p.x * canvas.width / state.clip.width, p.y * canvas.height / state.clip.height,
          Math.max(4, canvas.width/200), 0, Math.PI*2);
  ctx.fill(); ctx.restore();
}
function showTrackingGap(index){
  const review = state.tracking, gap = review?.gaps[index];
  if (!gap || review.source_sha256 !== state.clip?.sha256) return;
  const scene = state.scenes.findIndex(s => s.window && gap.start_frame >= s.window.start_frame && gap.start_frame <= s.window.end_frame);
  // Outside scene windows, deselect so an operator cannot correct the wrong scene.
  if (scene !== state.current) selectScene(scene);
  state.lastPoint = null; $("coordChip").textContent = "x – y –";
  $("reviewGap").value = String(index);
  setFrame(gap.start_frame);
}
function nextTrackingGap(direction){
  const gaps = state.tracking?.gaps || [];
  const candidates = gaps.map((gap, index) => ({gap, index}));
  const next = direction > 0
    ? candidates.find(row => row.gap.start_frame > state.frame)
    : candidates.reverse().find(row => row.gap.start_frame < state.frame);
  if (next) showTrackingGap(next.index);
}

// ---- the frame on screen ------------------------------------------------
function apiFrameUrl(frame){
  return `/api/frame?clip=${encodeURIComponent(state.clip.id)}&frame=${frame}&t=${Date.now()}`;
}
function setFrame(frame){
  if (!state.clip) return;
  state.frame = Math.max(0, Math.min(frame, (state.clip.frame_count||1)-1));
  updateAutomaticInfo();
  $("scrub").value = state.frame;
  $("frameChip").textContent = "frame " + state.frame;
  const requestedFrame = state.frame, requestedClip = state.clip.id;
  say(`Loading ${state.clip.name} · source frame ${requestedFrame}…`);
  const img = new Image();
  img.onload = () => {
    if (state.frame !== requestedFrame || state.clip?.id !== requestedClip) return;
    const c = $("view");
    c.width = img.naturalWidth; c.height = img.naturalHeight;
    const ctx = c.getContext("2d");
    ctx.drawImage(img, 0, 0);
    drawRoi(ctx, c);
    drawMarks(ctx, c);
    drawAutomatic(ctx, c);
    updateWindowInfo();
    const inWindow = isInAnyWindow(state.frame);
    say(`${state.clip.name} · frame ${state.frame} of ` +
        `${(state.clip.frame_count||1)-1} loaded · ` +
        (inWindow ? "inside a scene window, so overlays render here"
                  : "outside every scene window, so this frame renders bare"), "ok");
  };
  img.onerror = () => {
    if (state.frame === requestedFrame && state.clip?.id === requestedClip)
      say("frame " + frame + " could not be decoded", "bad");
  };
  img.src = apiFrameUrl(state.frame);
}
function isInAnyWindow(frame){
  return state.scenes.some(s => s.window && frame >= s.window.start_frame &&
                                frame <= s.window.end_frame);
}
function drawRoi(ctx, canvas){
  if(!state.roi) return;
  ctx.save();
  ctx.strokeStyle = "#70d4b0"; ctx.lineWidth = Math.max(2, canvas.width/500);
  ctx.setLineDash([8, 6]);
  const sx = canvas.width / (state.clip?.width || canvas.width);
  const sy = canvas.height / (state.clip?.height || canvas.height);
  ctx.strokeRect(state.roi[0]*sx, state.roi[1]*sy, state.roi[2]*sx, state.roi[3]*sy);
  ctx.restore();
}
function drawMarks(ctx, canvas){
  if ($("trackingMode").value === "saved_automatic") return;
  const scene = currentScene();
  const marks = [];
  if (state.lastPoint) marks.push([state.lastPoint, "#ffb030"]);
  if (scene){
    for (const name of Object.keys(scene.corrections || {}))
      for (const point of (scene.corrections[name] || []))
        if (point && point.frame === state.frame) marks.push([point, "#ff7a6b"]);
    for (const name of Object.keys(scene.manual_segments || {}))
      for (const segment of ((scene.manual_segments[name] || {}).segments || []))
        for (const anchor of (segment.anchors || []))
          if (anchor && anchorPoint(anchor).frame === state.frame)
            marks.push([anchorPoint(anchor), "#70d4b0"]);
  }
  for (const [point, colour] of marks){
    ctx.save();
    ctx.fillStyle = colour;
    ctx.beginPath();
    ctx.arc(point.x * canvas.width / (state.clip?.width || canvas.width),
            point.y * canvas.height / (state.clip?.height || canvas.height),
            Math.max(4, canvas.width/200), 0, Math.PI*2);
    ctx.fill();
    ctx.restore();
  }
}
function updateWindowInfo(){
  const s = +$("wStart").value, e = +$("wEnd").value;
  $("windowInfo").textContent =
    `${Math.max(0, e - s + 1)} frames · end inclusive · ` +
    (e < s ? "END IS BEFORE START" : "in window: " +
      (state.frame >= s && state.frame <= e ? "yes (overlays drawn)" : "no (bare frame)"));
}

// ---- the document being built -------------------------------------------
function currentScene(){
  return (state.current >= 0 && state.current < state.scenes.length)
    ? state.scenes[state.current] : null;
}
function sceneWindowOrderNote(){
  const starts = state.scenes.map(s => (s.window || {}).start_frame);
  for (let i = 1; i < starts.length; i++)
    if (!(starts[i] > starts[i-1]))
      return "scene windows are not in ascending order — save will be refused until " +
             "they are";
  return "";
}
function sortScenes(){
  const before = state.scenes.map(s => s.id).join(",");
  const current = currentScene();
  state.scenes.sort((a, b) => (a.window?.start_frame ?? 0) - (b.window?.start_frame ?? 0));
  state.current = current ? state.scenes.indexOf(current) : -1;
  state.config.scenes = state.scenes;
  return state.scenes.map(s => s.id).join(",") !== before;
}
function markDirty(){
  state.dirty = true;
  const info = $("configInfo");
  if (info && !/unsaved/.test(info.textContent || ""))
    info.textContent = (info.textContent || "") + " · unsaved edits";
}
function commitEditors(){
  // Write the editors into the SELECTED scene. Every editor change calls this, so what
  // the panel shows is what a save will contain.
  const scene = currentScene();
  if (!scene) return false;
  const names = VOCAB[$("strokeType").value] || VOCAB.full_swing;
  const keyframes = {};
  for (const name of Object.keys(state.phases))
    if (names.includes(name)) keyframes[name] = +state.phases[name];
  scene.stroke_type = $("strokeType").value;
  scene.window = Object.assign({}, scene.window, {
    start_frame: +$("wStart").value, end_frame: +$("wEnd").value});
  scene.keyframes = keyframes;
  if (state.roi) scene.roi = state.roi.slice(); else delete scene.roi;
  if (Object.keys(state.seeds).length) scene.seeds = clone(state.seeds);
  else delete scene.seeds;
  if (!scene.kind) scene.kind = "golfer_stroke";
  if (!scene.label) scene.label = scene.id;
  markDirty();
  return true;
}
function syncEditorsFromScene(index){
  const scene = state.scenes[index];
  state.current = index;
  if (!scene){
    state.phases = {}; state.roi = null; state.seeds = {};
    renderPhases(); renderAnchors(); return;
  }
  $("strokeType").value = scene.stroke_type || $("strokeType").value;
  $("wStart").value = (scene.window || {}).start_frame ?? 0;
  $("wEnd").value = (scene.window || {}).end_frame ?? 0;
  state.phases = Object.assign({}, scene.keyframes || {});
  state.roi = scene.roi ? scene.roi.slice() : null;
  state.seeds = clone(scene.seeds || {});
  $("roiInfo").textContent = state.roi
    ? `native [${state.roi.join(", ")}]` : "no region";
  renderPhases(); renderAnchors(); updateWindowInfo();
}
function selectScene(index){
  syncEditorsFromScene(index);
  renderScenes();
  setFrame(state.frame);
}
function addScene(){
  const s = +$("wStart").value, e = +$("wEnd").value;
  if (!(e >= s)){ say("the window end is before its start", "bad"); return; }
  const used = new Set(state.scenes.map(x => x.id));
  let n = state.scenes.length + 1;
  while (used.has(`scene${n}`)) n++;
  // Only the marks that fall inside the new window: a phase mark outside it is refused
  // by the parser, so carrying one over would hand the operator a refusal for something
  // the tool did, not something they did.
  const inside = {}, dropped = [];
  for (const [name, frame] of Object.entries(state.phases))
    if (+frame >= s && +frame <= e) inside[name] = +frame; else dropped.push(name);
  const scene = {
    id:`scene${n}`,
    label:`${$("strokeType").value} window ${s}..${e}`,
    kind:"golfer_stroke",
    stroke_type:$("strokeType").value,
    window:{start_frame:s, end_frame:e},
    keyframes:inside
  };
  if (state.roi) scene.roi = state.roi.slice();
  if (Object.keys(state.seeds).length) scene.seeds = clone(state.seeds);
  state.scenes.push(scene);
  state.config.scenes = state.scenes;
  const moved = sortScenes();
  state.current = state.scenes.indexOf(scene);
  markDirty();
  renderScenes(); renderAnchors();
  say(`added ${scene.id} for frames ${s}..${e}` +
      (dropped.length ? ` (phase marks outside the window were not carried over: ` +
                        dropped.join(", ") + ")" : "") +
      (moved ? " (the list was re-sorted into ascending window order)" : ""), "ok");
}
function removeScene(){
  const scene = currentScene();
  if (!scene) return;
  state.scenes.splice(state.current, 1);
  state.config.scenes = state.scenes;
  markDirty();
  selectScene(Math.min(state.current, state.scenes.length - 1));
  say("removed " + scene.id, "ok");
}
function renderScenes(){
  const note = sceneWindowOrderNote();
  $("sceneHint").textContent = note ||
    "click a scene to select and load its fields; edits apply to the selected scene";
  $("sceneHint").className = note ? "status bad" : "muted";
  $("scenes").innerHTML = state.scenes.map((s,i) => {
    const w = s.window || {};
    const roi = s.roi ? `roi [${s.roi.join(", ")}]` : "no roi";
    const seeds = Object.keys(s.seeds||{}).join(", ") || "no seeds";
    const phases = Object.keys(s.keyframes||{}).length + " phases";
    const marks = [Object.keys(s.corrections||{}).length ? "corrections" : null,
                   Object.keys(s.manual_segments||{}).length ? "manual segments" : null,
                   (s.roi_keyframes||[]).length ? (s.roi_keyframes.length + " roi keyframes") : null,
                   (s.metric_hide_frames||[]).length
                     ? ("cards hidden " + JSON.stringify(s.metric_hide_frames)) : null]
                  .filter(Boolean).join(" · ");
    return `<div class="scene" aria-current="${i===state.current}" data-i="${i}">
      <h3>${s.id} <span class="chip">${s.stroke_type || "?"}</span></h3>
      <div class="muted">frames ${w.start_frame}..${w.end_frame} inclusive · ${roi}</div>
      <div class="muted">${seeds} · ${phases}${marks ? " · " + marks : ""}</div></div>`;
  }).join("") || "<div class='muted'>no scenes yet</div>";
  $("scenes").querySelectorAll(".scene").forEach(el => {
    el.onclick = () => selectScene(+el.dataset.i);
  });
}

// ---- assistance marks ---------------------------------------------------
function anchorPoint(anchor){
  return Array.isArray(anchor) ? {frame:anchor[0], x:anchor[1], y:anchor[2]} : anchor;
}
function targetNames(){
  return ["ball", "clubhead"];
}
function renderAnchors(){
  const scene = currentScene();
  const box = $("anchors");
  if (!scene){ box.innerHTML = "<div class='muted'>select a scene</div>";
    $("anchorSegment").innerHTML = ""; return; }
  const rows = [];
  for (const name of Object.keys(scene.seeds || {})){
    const point = scene.seeds[name] || {};
    rows.push(["seed", name, `frame ${point.frame} (${point.x}, ${point.y})`,
               () => { delete scene.seeds[name]; markDirty(); renderAnchors();
                       renderScenes(); setFrame(state.frame); }]);
  }
  for (const name of Object.keys(scene.corrections || {})){
    (scene.corrections[name] || []).forEach((point, index) => {
      rows.push(["correction", name, `frame ${point.frame} (${point.x}, ${point.y})`,
                 () => { scene.corrections[name].splice(index, 1);
                         if (!scene.corrections[name].length) delete scene.corrections[name];
                         markDirty(); renderAnchors(); setFrame(state.frame); }]);
    });
  }
  (scene.roi_keyframes || []).forEach((key, index) => {
    rows.push(["ROI keyframe", `frame ${key.frame}`,
               `[${(key.roi || []).join(", ")}]`,
               () => { scene.roi_keyframes.splice(index, 1);
                       if (!scene.roi_keyframes.length) delete scene.roi_keyframes;
                       markDirty(); renderAnchors(); renderScenes(); }]);
  });
  for (const name of Object.keys(scene.manual_segments || {})){
    const block = scene.manual_segments[name] || {};
    (block.segments || []).forEach((segment, segmentIndex) => {
      (segment.anchors || []).forEach((anchor, anchorIndex) => {
        rows.push([`anchor (${block.replace_automatic ? "replaces automatic" : "added"})`,
                   `${name}/${segment.name}`,
                   `frame ${anchorPoint(anchor).frame} (${anchorPoint(anchor).x}, ${anchorPoint(anchor).y})`,
                   () => { segment.anchors.splice(anchorIndex, 1);
                           if (!segment.anchors.length) block.segments.splice(segmentIndex, 1);
                           if (!(block.segments || []).length)
                             delete scene.manual_segments[name];
                           markDirty(); renderAnchors(); renderScenes();
                           setFrame(state.frame); }]);
      });
    });
  }
  box.innerHTML = rows.length
    ? rows.map((row, i) => `<div><span>${row[0]} · ${row[1]} · ${row[2]}</span>
        <button data-i="${i}">remove</button></div>`).join("")
    : "<div class='muted'>no seeds, corrections, ROI keyframes or manual anchors on " +
      "this scene</div>";
  // The rows carry the action each remove button performs, so the button and any other
  // caller (a test driving this script without a browser) run exactly the same code.
  state.anchorRows = rows;
  box.querySelectorAll("button").forEach(button => {
    button.onclick = () => state.anchorRows[+button.dataset.i][3]();
  });
  const options = [];
  for (const name of Object.keys(scene.manual_segments || {}))
    for (const segment of ((scene.manual_segments[name] || {}).segments || []))
      options.push(`<option value="${name}|${segment.name}">${name} / ${segment.name}` +
                   ` (spacing ${segment.max_anchor_spacing_frames})</option>`);
  $("anchorSegment").innerHTML = options.join("") ||
    "<option value=''>no manual segment on this scene</option>";
  $("anchorTarget").value = state.target;
}
function placeSeedAt(point){
  const target = state.target;
  const scene = currentScene();
  if (!scene){ say("select a scene before placing a point", "bad"); return false; }
  if (!scene.seeds) scene.seeds = {};
  scene.seeds[target] = {frame: state.frame, x: point.x, y: point.y,
    note: "operator: physical " + target + " visible (placed in the review UI)"};
  state.seeds = clone(scene.seeds);
  state.lastPoint = {frame: state.frame, x: point.x, y: point.y};
  markDirty(); renderAnchors(); renderScenes(); setFrame(state.frame);
  return true;
}
function addCorrection(point){
  const target = state.target;
  const scene = currentScene();
  if (!scene){ say("select a scene before placing a point", "bad"); return false; }
  if (!scene.corrections) scene.corrections = {};
  if (!Array.isArray(scene.corrections[target])) scene.corrections[target] = [];
  scene.corrections[target].push({frame: state.frame, x: point.x, y: point.y,
    note: "operator correction placed in the review UI"});
  scene.corrections[target].sort((a, b) => a.frame - b.frame);
  state.lastPoint = {frame: state.frame, x: point.x, y: point.y};
  markDirty(); renderAnchors(); renderScenes(); setFrame(state.frame);
  return true;
}
function addAnchorToSegment(value, point){
  const scene = currentScene();
  if (!scene || !value){ say("add a manual segment first", "bad"); return false; }
  const [target, segmentName] = value.split("|");
  const block = (scene.manual_segments || {})[target];
  if (!block){ say("that segment is not on this scene any more", "bad"); return false; }
  const segment = (block.segments || []).find(s => s.name === segmentName);
  if (!segment){ say("that segment is not on this scene any more", "bad"); return false; }
  if (!Array.isArray(segment.anchors)) segment.anchors = [];
  const at = point || state.lastPoint;
  if (!at){ say("click a point on the picture first, then add it as an anchor", "bad");
    return false; }
  segment.anchors.push({frame: at.frame, x: at.x, y: at.y,
    note: "operator anchor placed in the review UI"});
  segment.anchors.sort((a, b) => anchorPoint(a).frame - anchorPoint(b).frame);
  markDirty(); renderAnchors(); renderScenes(); setFrame(state.frame);
  say(`anchor added to ${target}/${segmentName} at frame ${at.frame}; validate to ` +
      `have the parser check the spacing bound`, "ok");
  return true;
}
function newSegment(target, frame){
  const scene = currentScene();
  if (!scene){ say("select a scene first", "bad"); return false; }
  if (!scene.manual_segments) scene.manual_segments = {};
  if (!scene.manual_segments[target])
    scene.manual_segments[target] = {author:"operator (review UI)",
      replace_automatic:false, hide_frames:[], trail:"all", trail_from_frame:null,
      segments:[]};
  const block = scene.manual_segments[target];
  if (!Array.isArray(block.segments)) block.segments = [];
  let name = `segment_${block.segments.length + 1}`;
  while (block.segments.some(s => s.name === name)) name += "b";
  block.segments.push({name, max_anchor_spacing_frames:2, anchors:[],
    note:"created in the review UI; the spacing bound is the broadcast cap"});
  block.segments.sort((a, b) =>
    (anchorPoint(a.anchors[0] || {}).frame ?? 0) - (anchorPoint(b.anchors[0] || {}).frame ?? 0));
  markDirty(); renderAnchors(); renderScenes();
  say(`created ${target}/${name}; now click points in correction mode and use ` +
      `"add anchor" to fill it (anchors must be within ` +
      `max_anchor_spacing_frames of each other)`, "ok");
  return true;
}
function addRoiKeyframe(roi, frame){
  const scene = currentScene();
  if (!scene){ say("select a scene first", "bad"); return false; }
  if (!roi){ say("draw a region first", "bad"); return false; }
  if (!Array.isArray(scene.roi_keyframes)) scene.roi_keyframes = [];
  scene.roi_keyframes = scene.roi_keyframes.filter(k => k.frame !== frame);
  scene.roi_keyframes.push({frame, roi: roi.slice(),
    note:"operator ROI keyframe placed in the review UI"});
  scene.roi_keyframes.sort((a, b) => a.frame - b.frame);
  markDirty(); renderAnchors(); renderScenes();
  say(`ROI keyframe saved at frame ${frame}`, "ok");
  return true;
}

// ---- pointer handling ---------------------------------------------------
$("view").addEventListener("pointerdown", (ev) => {
  const pt = localPoint(ev); if(!pt) return;
  if (state.mode === "seed"){
    if (window.confirm("Is the " + state.target + " physically visible at this point in frame " +
                       state.frame + "?")) placeSeedAt(pt);
    else say("seed not placed", "ok");
    return;
  }
  if (state.mode === "correction"){ addCorrection(pt); return; }
  state.dragging = {x0:pt.x, y0:pt.y};
  $("view").setPointerCapture(ev.pointerId);
});
$("view").addEventListener("pointermove", (ev) => {
  const pt = localPoint(ev);
  if (pt) $("coordChip").textContent = `x ${pt.x} y ${pt.y}`;
  if (!state.dragging || !pt) return;
  const d = state.dragging;
  state.roi = [Math.min(d.x0,pt.x), Math.min(d.y0,pt.y),
               Math.abs(pt.x-d.x0), Math.abs(pt.y-d.y0)];
  const c = $("view"), ctx = c.getContext("2d");
  const img = new Image();
  img.onload = () => { ctx.drawImage(img,0,0); drawRoi(ctx,c); drawMarks(ctx,c); drawAutomatic(ctx,c); };
  img.src = apiFrameUrl(state.frame);
});
$("view").addEventListener("pointerup", () => {
  if (state.dragging){
    state.dragging = null;
    if (state.roi && (state.roi[2] < 4 || state.roi[3] < 4)) state.roi = null;
    $("roiInfo").textContent = state.roi
      ? `native [${state.roi.join(", ")}]` : "no region";
    if (commitEditors()) renderScenes();
  }
});
function localPoint(ev){
  const c = $("view"), rect = c.getBoundingClientRect();
  return viewportToNative(rect, {w:state.clip?.width || c.width,
                                h:state.clip?.height || c.height},
                          ev.clientX, ev.clientY);
}

// ---- phase buttons follow the stroke type -------------------------------
function renderPhases(){
  const names = VOCAB[$("strokeType").value] || VOCAB.full_swing;
  $("phaseButtons").innerHTML = names.map(n =>
    `<button data-phase="${n}" class="${state.phases[n]!==undefined?"on":""}">${n.replace(/_/g," ")}</button>`
  ).join("");
  $("phaseButtons").querySelectorAll("button").forEach(b => {
    b.onclick = (event) => {
      const name = b.dataset.phase;
      if (event && event.shiftKey) delete state.phases[name];
      else state.phases[name] = state.frame;
      if (commitEditors()) renderScenes(); else renderPhases();
    };
  });
  $("phaseInfo").textContent = names.map(n =>
    `${n} ${state.phases[n]!==undefined?state.phases[n]:"–"}`).join(" · ") +
    " (shift-click clears)";
}
$("strokeType").onchange = () => {
  const names = VOCAB[$("strokeType").value] || [];
  Object.keys(state.phases).forEach(k => { if(!names.includes(k)) delete state.phases[k]; });
  if (commitEditors()) renderScenes(); else renderPhases();
};

// ---- discovery ----------------------------------------------------------
$("discover").onclick = async () => {
  try{
    say("scanning for cuts…");
    const data = await post("/api/scenes", {clip: state.clip.id});
    const cuts = data.candidates.filter(c => c.kind === "hard_cut").slice(0,8);
    $("cuts").innerHTML = cuts.map(c =>
      `<div><button data-frame="${c.frame}">go to ${c.frame}</button>
       <span class="muted">confidence ${c.confidence} · ${c.kind}</span></div>`).join("")
      || "no candidates";
    $("cuts").querySelectorAll("button").forEach(b => {
      b.onclick = () => setFrame(+b.dataset.frame);
    });
    say("scan complete: " + data.candidates.length + " candidate(s)", "ok");
  }catch(e){ say("scan failed: " + e.message, "bad"); }
};

// ---- config assembly ----------------------------------------------------
function buildConfig(){
  const cfg = clone(ensureConfig());
  const clip = state.clip;
  cfg.schema_version = cfg.schema_version || "fairwayos-broadcast-config.v1";
  if (typeof cfg.shot !== "object" || cfg.shot === null) cfg.shot = {};
  if (clip){
    cfg.clip_path_hint = clip.path;
    cfg.shot.expected_source_sha256 = clip.sha256 || cfg.shot.expected_source_sha256 || "";
    cfg.shot.label = cfg.shot.label || clip.name;
    cfg.shot.id = cfg.shot.id || clip.name.replace(/\.[^.]+$/, "");
  }
  cfg.shot.stroke_type = $("strokeType").value;
  cfg.interval = Object.assign({start_seconds:0.0, end_seconds:null, max_frames:20000},
                               cfg.interval || {});
  cfg.scenes = clone(state.scenes);
  if (!cfg.presentation) cfg.presentation = defaultPresentation();
  if (!cfg.render) cfg.render = {max_width:1920, crf:18, preset:"medium", render:true};
  if (!cfg.outside_windows) cfg.outside_windows = "brand_only";
  if (Array.isArray(cfg.cuts) && !cfg.cuts.length) delete cfg.cuts;
  if (Array.isArray(cfg.overrides) && !cfg.overrides.length) delete cfg.overrides;
  return cfg;
}
$("validate").onclick = async () => {
  try{
    if (!state.scenes.length) throw new Error("add at least one scene first");
    const data = await post("/api/validate",
      {clip: state.clip.id, config: buildConfig()});
    say("valid · matched the clip by " + data.clip_resolved_by + " · " +
        data.scenes.length + " scene(s) bound: " +
        data.scenes.map(s => `${s.id}[${s.start_frame}..${s.end_frame}]`).join(", "), "ok");
  }catch(e){ say("invalid: " + e.message, "bad"); }
};
$("save").onclick = async () => {
  try{
    if (!state.scenes.length) throw new Error("add at least one scene first");
    const name = $("configName").value ||
      `${state.clip.name.replace(/\.[^.]+$/,"")}_ui.json`;
    const data = await post("/api/save", {name, config: buildConfig()});
    state.dirty = false;
    state.loadedName = data.name;
    $("configInfo").textContent = data.name + " · saved · " + state.scenes.length + " scene(s) · existing assistance preserved";
    say("saved " + data.path, "ok");
    await loadConfigList();
    $("configFile").value = data.name;
  }catch(e){ say("not saved: " + e.message, "bad"); }
};
$("previewAutomatic").onchange = () => setFrame(state.frame);
$("trackingMode").onchange = () => {
  state.trackingPreference = $("trackingMode").value;
  setFrame(state.frame);
};
$("render").onclick = async () => {
  try{
    if (!state.scenes.length) throw new Error("add at least one scene first");
    const name = $("configName").value ||
      `${state.clip.name.replace(/\.[^.]+$/,"")}_ui.json`;
    const data = await post("/api/render",
      {clip: state.clip.id, config: buildConfig(), name, save_config:true,
       tracking_mode: $("trackingMode").value || "reviewed"});
    state.job = data.job.id;
    say(`rendering ${data.frames} frames…`);
    pollJob(data.job.id);
  }catch(e){ say("render not started: " + e.message, "bad"); }
};
async function pollJob(id){
  try{
    const data = await api("/api/job?id=" + encodeURIComponent(id));
    const job = data.job;
    if (job.state === "running"){
      say("rendering…");
      setTimeout(() => pollJob(id), 1500);
      return;
    }
    if (job.state !== "succeeded"){
      say("render failed: " + (job.log || "").slice(-300), "bad");
      return;
    }
    const r = job.result || {};
    const verify = (r.verification || {}).render_ffprobe || {};
    const stamps = verify.timestamps || {};
    const durations = verify.duration || {};
    const rows = (r.scenes || []).map(s =>
      `<tr><td>${s.id}</td><td>${s.window.start_frame}..${s.window.end_frame}</td>
       <td>${s.body_state || "not measured"}</td>
       <td>${(s.phases||[]).join(", ") || "none"}</td></tr>`).join("");
    $("result").innerHTML = `
      <div class="status ${verify.ok ? "ok" : "bad"}">
        ${verify.ok ? "render verified" : "render NOT verified"}</div>
      <div class="muted">frame timeline: ${stamps.checked
        ? (stamps.compared_frames + " of " + stamps.output_frames +
           " decoded frames compared, max delta " + stamps.max_delta_seconds + "s")
        : "not compared"} · duration: ${durations.checked
        ? (durations.measured_seconds + "s, expected " +
           durations.expected_seconds_from_frames_and_cadence + "s")
        : "not checked"}</div>
      <div class="muted">${r.tracking_export
        ? "Saved automatic target observations (research) · prediction " + r.tracking_export.prediction_sha256
        : "Reviewed showcase · operator-assisted target positions"}</div>
      <video controls preload="metadata" style="width:100%;border-radius:8px" src="/api/artifact?id=${id}&name=broadcast_annotated.mp4"></video>
      <table>${rows}</table>
      <div class="row">
        <button onclick="location='/api/artifact?id=${id}&name=broadcast_annotated.mp4'">download video</button>
        <button onclick="location='/api/artifact?id=${id}&name=broadcast_report.md'">report</button>
        <button onclick="location='/api/artifact?id=${id}&name=broadcast_analysis.json'">analysis</button>
        <button onclick="location='/api/config?id=${id}'">config</button>
      </div>
      <div class="muted">${r.out_dir || ""}</div>`;
    say("render complete: " + (verify.ok ? "verification PASS" : "verification FAILED"),
        verify.ok ? "ok" : "bad");
  }catch(e){ say("lost the job: " + e.message, "bad"); }
}

// ---- wiring -------------------------------------------------------------
$("clip").onchange = (e) => selectClip(e.target.value);
$("reviewTarget").onchange = loadTrackingReview;
$("reviewGap").onchange = (e) => showTrackingGap(+e.target.value);
$("previousGap").onclick = () => nextTrackingGap(-1);
$("nextGap").onclick = () => nextTrackingGap(1);
$("doImport").onclick = doImport;
$("doImportPath").onclick = doImportPath;
$("loadConfig").onclick = loadConfigFile;
$("scrub").oninput = (e) => setFrame(+e.target.value);
$("prev").onclick = () => setFrame(state.frame - 1);
$("next").onclick = () => setFrame(state.frame + 1);
$("setStart").onclick = () => { $("wStart").value = state.frame;
  if (commitEditors()) renderScenes(); updateWindowInfo(); };
$("setEnd").onclick = () => { $("wEnd").value = state.frame;
  if (commitEditors()) renderScenes(); updateWindowInfo(); };
$("wStart").oninput = () => { if (commitEditors()) renderScenes(); updateWindowInfo(); };
$("wEnd").oninput = () => { if (commitEditors()) renderScenes(); updateWindowInfo(); };
$("clearRoi").onclick = () => { state.roi = null;
  $("roiInfo").textContent = "no region";
  if (commitEditors()) renderScenes();
  setFrame(state.frame); };
$("saveRoiKey").onclick = () => { addRoiKeyframe(state.roi, state.frame); };
$("modeRegion").onclick = () => setMode("region");
$("modeSeed").onclick = () => setMode("seed");
$("modeCorrection").onclick = () => setMode("correction");
$("target").onchange = (e) => { state.target = e.target.value;
  $("anchorTarget").value = state.target; };
$("anchorTarget").onchange = (e) => { state.target = e.target.value;
  $("target").value = state.target; };
$("placeSeed").onclick = () => {
  const scene = currentScene();
  if (!scene){ say("select a scene first", "bad"); return; }
  const roi = state.roi || [0, 0, 0, 0];
  const point = {x: Math.round(roi[0] + roi[2]/2), y: Math.round(roi[1] + roi[3]/2)};
  if (window.confirm("Is the " + state.target + " physically visible at frame " + state.frame + "?"))
    placeSeedAt(point);
};
$("addAnchor").onclick = () => addAnchorToSegment($("anchorSegment").value,
                                                  state.lastPoint);
$("newSegment").onclick = () => newSegment($("anchorTarget").value, state.frame);
$("applyScene").onclick = () => {
  if (!currentScene()){ say("select a scene first", "bad"); return; }
  if (commitEditors()){ renderScenes(); renderAnchors();
    say(`fields applied to ${currentScene().id}`, "ok"); }
};
$("addScene").onclick = addScene;
$("removeScene").onclick = removeScene;
function setMode(mode){
  state.mode = mode;
  $("modeRegion").className = mode === "region" ? "on" : "";
  $("modeSeed").className = mode === "seed" ? "on" : "";
  $("modeCorrection").className = mode === "correction" ? "on" : "";
  $("seedInfo").textContent = mode === "region"
    ? "region mode: drag a box on the picture"
    : (mode === "seed"
       ? "seed mode: click the " + state.target + ", and confirm it is visible"
       : "correction mode: click a point to record the operator's own position for " +
         "the " + state.target + " at this frame (no claim of a detection)");
}
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "ArrowLeft") setFrame(state.frame - 1);
  if (e.key === "ArrowRight") setFrame(state.frame + 1);
});

// A hook for the tests that drive this script without a browser. Everything on it is a
// function of the state above, so a test can load a document, mutate a field the way a
// click would, and read back exactly what a save would send.
if (typeof window !== "undefined"){
  window.__bcui = {
    state, clone, applyConfig, buildConfig, commitEditors, selectScene, currentScene,
    loadTrackingReview, showTrackingGap, nextTrackingGap, automaticPoint, drawAutomatic,
    addScene, removeScene, sortScenes, sceneWindowOrderNote, renderScenes, renderAnchors,
    renderPhases, placeSeedAt, addCorrection, addAnchorToSegment, newSegment,
    addRoiKeyframe, resolveClipForConfig, selectClip, ensureConfig, defaultPresentation,
    setTarget:(value) => { state.target = value; $("target").value = value; },
    setMode, setStrokeType:(value) => { $("strokeType").value = value; },
    setWindow:(s, e) => { $("wStart").value = s; $("wEnd").value = e; },
    setPhase:(name, frame) => { state.phases[name] = frame; },
    dropPhase:(name) => { delete state.phases[name]; },
    setRoi:(roi) => { state.roi = roi; },
    setFrame:(frame) => { state.frame = frame; }
  };
}
if (typeof document !== "undefined" && document.getElementById("clip")){
  loadClips().then(renderPhases).catch(e => say("startup failed: " + e.message, "bad"));
}
</script>
</body>
</html>
"""
