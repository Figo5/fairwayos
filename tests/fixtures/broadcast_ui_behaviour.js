// Real behaviour tests for the broadcast review page, run against the SHIPPED script
// (concatenated after this file by tests/test_demo_broadcast_ui.py).
//
// The cases are the ones the operator hit in a browser: a config that loads onto the
// wrong clip, a save that silently drops manual segments / ROI keyframes / cuts /
// presentation, and edits that never reach the selected scene.

const results = [];
function check(name, ok, detail) {
  results.push({name, ok: Boolean(ok), detail: detail === undefined ? "" : String(detail)});
}
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

await sleep(30);  // let the page's own bootstrap (loadClips) finish

const ui = globalThis.__bcui;
check("the page exposes its test hook", Boolean(ui), typeof ui);
if (!ui) {
  ui.selectClip("clip01");
check("changing source directly cannot carry another clip's assistance",
      ui.state.scenes.length === 0 && ui.state.config === null &&
      ui.state.roi === null && Object.keys(ui.state.phases).length === 0 &&
      ui.state.lastPoint === null &&
      document.getElementById("configName").value === "01_hole07_approach_ui.json");
check("a new source resets its pointer readout",
      document.getElementById("coordChip").textContent === "x – y –");
setFrame(0);
const staleImage = imageRequests[imageRequests.length - 1];
setFrame(1);
const activeImage = imageRequests[imageRequests.length - 1];
let draws = 0;
document.getElementById("view").getContext = () => ({...context2dStub(),drawImage(){draws++;}});
staleImage.onload();
check("out-of-order frame responses cannot repaint a newer selection", draws === 0);
activeImage.onload();
check("the selected native frame response paints normally", draws === 1);
console.log(JSON.stringify({results}));
  process.exit(1);
}

// --- a config in the shape the actual operator configs have -----------------
function scene(id, start, end, extras) {
  const payload = {
    id, label: `${id} golfer stroke`, kind: "golfer_stroke",
    stroke_type: "full_swing",
    window: {start_frame: start, end_frame: end, note: "operator window"},
    keyframes: {address: start, takeaway: start + 5, top: start + 10,
                downswing: start + 12, impact: start + 14,
                follow_through: start + 20, finish: end - 5},
    roi: [275, 215, 250, 430],
    seeds: {ball: {frame: start, x: 503, y: 604, note: "primary visible seed"}},
    targets: {ball: {mode: "template", template_size: 21, search_radius: 55,
                     min_score: 0.65, max_jump_px: 70, loss_frames: 2}},
    corrections: {clubhead: [{frame: start + 3, x: 11, y: 12,
                              note: "operator correction"}]},
    roi_keyframes: [{frame: start, roi: [275, 215, 250, 430], note: "primary crop"},
                    {frame: start + 10, roi: [260, 150, 285, 500],
                     note: "primary crop mid"}],
    manual_segments: {clubhead: {
      author: "primary", replace_automatic: true, hide_frames: [],
      trail: "segments_only", trail_from_frame: start + 2,
      note: "operator note on the block",
      segments: [{name: "downswing", max_anchor_spacing_frames: 12,
                  anchors: [{frame: start + 2, x: 1.5, y: 2.5, note: "first"},
                            {frame: start + 8, x: 3.5, y: 4.5, note: "second"}],
                  note: "segment note"}]}},
    interpolation: {enabled: true, max_gap_frames: 2},
    handedness: "right",
    notes: "scene note",
    metric_hide_frames: [[start + 12, start + 20]]
  };
  return Object.assign(payload, extras || {});
}

const rich = {
  schema_version: "fairwayos-broadcast-config.v1",
  clip_path_hint: "/clips/03_hole12_tee.mp4",
  shot: {id: "03_hole12_tee", label: "Hole 12 tee shot",
         stroke_type: "full_swing", expected_source_sha256: "b".repeat(64),
         source_notes: "Tiger Woods, 2019 Masters final round",
         parent_source_manifest: "/manifests/baseline/manifest.json"},
  interval: {start_seconds: 0.0, end_seconds: null, max_frames: 20000},
  outside_windows: "brand_only",
  presentation: {brand: "FAIRWAY OS", brand_sub: "SWING ANALYSIS",
                 disclosure: "OPERATOR-ASSISTED", metrics: ["shoulder_tilt",
                 "hip_tilt", "knee_flexion", "tempo"], card_debounce_frames: 1,
                 card_row_max_width: 1450, ball_trail_points: 10,
                 club_trail_points: 12, impact_accent: true,
                 impact_accent_span_frames: 8, metrics_required: false,
                 card_position: "auto"},
  cuts: [{frame: 96, kind: "hard_cut", confidence: 1.0, span: null,
          method: "primary_native_frame_inspection", decision: "accepted",
          decided_by: "Codex primary visual review", decided_at: "2026-10-01",
          note: "cut to the result camera"}],
  overrides: [{frame: 96, decision: "accept", note: "kept",
               decided_by: "primary", decided_at: "2026-10-01"}],
  timing: {source_cadence_caveat: "nominal 29.704 fps; not trusted",
           nominal_fps_trusted: false, accept_variable_cadence: false},
  notes: "top level operator note",
  source_description: "Tiger Woods, 2019 Masters; supplied highlights.",
  _resolved: {note: "a run's own underscore comment must survive a save"},
  scenes: [scene("scene_a", 0, 95), scene("scene_b", 120, 180)]
};

// --- the config must land on ITS clip ---------------------------------------
check("the library starts on the first clip", ui.state.clip && ui.state.clip.id === "clip01",
      ui.state.clip && ui.state.clip.id);
const wrong = ui.resolveClipForConfig(rich);
check("a config is resolved to its own clip, not the selected one",
      wrong.clip && wrong.clip.id === "clip03", wrong.clip && wrong.clip.id);
ui.selectClip("clip03");
check("selecting the clip refreshes the frame bounds",
      Number(document.getElementById("scrub").max) === 388 &&
      ui.state.clip.frame_count === 389,
      document.getElementById("scrub").max);
check("a config whose clip is absent is NOT resolved",
      ui.resolveClipForConfig({schema_version: "fairwayos-broadcast-config.v1",
                               clip_path_hint: "/clips/99_nope.mp4"}).clip === null);

// --- load it, and prove the document is preserved ---------------------------
const before = JSON.stringify(rich);
ui.applyConfig(rich, "03_hole12_tee.broadcast.json", "expected_source_sha256");
check("loading does not mutate the input document", JSON.stringify(rich) === before);
check("the loaded document is a deep copy, not the object that was passed",
      ui.state.config !== rich && ui.state.scenes === ui.state.config.scenes);
check("the first scene's fields are loaded into the editors",
      document.getElementById("wStart").value === 0 &&
      document.getElementById("wEnd").value === 95 &&
      ui.state.phases.impact === 14 &&
      same(ui.state.roi, [275, 215, 250, 430]),
      [document.getElementById("wStart").value, document.getElementById("wEnd").value,
       ui.state.phases.impact].join(","));

let built = ui.buildConfig();
check("presentation survives a save", same(built.presentation, rich.presentation));
check("cuts survive a save", same(built.cuts, rich.cuts));
check("overrides survive a save", same(built.overrides, rich.overrides));
check("timing survives a save", same(built.timing, rich.timing));
check("notes survive a save", built.notes === rich.notes &&
      built.scenes[0].notes === rich.scenes[0].notes);
check("the shot's own source notes survive a save",
      built.shot.source_notes === rich.shot.source_notes &&
      built.shot.parent_source_manifest === rich.shot.parent_source_manifest);
check("a run's underscore comment survives a save", same(built._resolved, rich._resolved));
check("manual segments survive a save",
      same(built.scenes[0].manual_segments, rich.scenes[0].manual_segments) &&
      same(built.scenes[1].manual_segments, rich.scenes[1].manual_segments));
check("ROI keyframes survive a save",
      same(built.scenes[0].roi_keyframes, rich.scenes[0].roi_keyframes));
check("corrections survive a save",
      same(built.scenes[0].corrections, rich.scenes[0].corrections));
check("targets and seeds survive a save",
      same(built.scenes[0].targets, rich.scenes[0].targets) &&
      same(built.scenes[0].seeds, rich.scenes[0].seeds));
check("a scene's metric_hide_frames survive a save",
      same(built.scenes[0].metric_hide_frames, rich.scenes[0].metric_hide_frames));
check("the clip identity follows the selected clip",
      built.clip_path_hint === "/clips/03_hole12_tee.mp4" &&
      built.shot.expected_source_sha256 === "b".repeat(64));
check("a built config is a copy (editing it cannot corrupt the page state)",
      (built.scenes[0].keyframes.impact = 999) === 999 &&
      ui.state.scenes[0].keyframes.impact === 14);

// --- selecting a scene loads ITS fields, and edits commit to it -------------
ui.selectScene(1);
check("selecting a scene loads that scene's window and phases",
      document.getElementById("wStart").value === 120 &&
      document.getElementById("wEnd").value === 180 &&
      ui.state.phases.takeaway === 125,
      document.getElementById("wStart").value);
ui.setPhase("impact", 137);
ui.commitEditors();
built = ui.buildConfig();
check("a phase mark lands on the SELECTED scene",
      built.scenes[1].keyframes.impact === 137, built.scenes[1].keyframes.impact);
check("the other scene's phases are untouched",
      built.scenes[0].keyframes.impact === 14, built.scenes[0].keyframes.impact);
ui.setRoi([10, 20, 30, 40]);
ui.commitEditors();
ui.setWindow(120, 175);
ui.commitEditors();
built = ui.buildConfig();
check("a dragged region lands on the selected scene",
      same(built.scenes[1].roi, [10, 20, 30, 40]), JSON.stringify(built.scenes[1].roi));
check("the other scene's region is untouched",
      same(built.scenes[0].roi, [275, 215, 250, 430]));
check("a window edit lands on the selected scene",
      built.scenes[1].window.end_frame === 175);

// --- adding and removing scenes --------------------------------------------
ui.setWindow(200, 240);
ui.setPhase("address", 201);
ui.addScene();
built = ui.buildConfig();
check("adding a scene appends it with the editor values",
      built.scenes.length === 3 &&
      built.scenes[2].window.start_frame === 200 &&
      built.scenes[2].keyframes.address === 201,
      JSON.stringify(built.scenes.map((s) => s.window)));
check("a new scene keeps only the phase marks inside its own window",
      Object.keys(built.scenes[2].keyframes).length === 1,
      JSON.stringify(built.scenes[2].keyframes));
check("the operator is told which marks were not carried over",
      /outside the window were not carried over/.test(
        document.getElementById("status").textContent),
      document.getElementById("status").textContent);
check("the scene list stays in ascending window order",
      ui.sceneWindowOrderNote() === "", ui.sceneWindowOrderNote());
check("adding a scene keeps the presentation and the cuts",
      same(built.presentation, rich.presentation) && same(built.cuts, rich.cuts));
ui.removeScene();
check("removing the selected scene leaves the document valid",
      ui.buildConfig().scenes.length === 2 && ui.sceneWindowOrderNote() === "");

// --- assistance marks: place, list, remove ---------------------------------
ui.selectScene(0);
ui.setFrame(7);
ui.setTarget("clubhead");
ui.addCorrection({x: 41, y: 42});
built = ui.buildConfig();
const corrections = built.scenes[0].corrections.clubhead;
check("a correction is recorded at the current frame for the chosen target",
      corrections.length === 2 && corrections.some((p) => p.frame === 7 && p.x === 41),
      JSON.stringify(corrections));
check("the correction is listed with the other marks",
      /correction · clubhead · frame 7/.test(document.getElementById("anchors").innerHTML),
      document.getElementById("anchors").innerHTML.slice(0, 120));
ui.addRoiKeyframe([1, 2, 3, 4], 7);
check("a ROI keyframe is added at the current frame",
      same(ui.buildConfig().scenes[0].roi_keyframes.map((k) => k.frame), [0, 7, 10]),
      JSON.stringify(ui.buildConfig().scenes[0].roi_keyframes.map((k) => k.frame)));
check("adding an anchor into a manual segment appends it",
      ui.addAnchorToSegment("clubhead|downswing", {frame: 9, x: 6, y: 7}) &&
      same(ui.buildConfig().scenes[0].manual_segments.clubhead.segments[0]
             .anchors.map((a) => a.frame), [2, 8, 9]));
// Every listed mark can be removed through the row action the remove button runs.
ui.renderAnchors();
const rows = ui.state.anchorRows;
check("one row per mark is offered for removal", rows.length >= 6, rows.length);
const anchorRow = rows.findIndex((row) => row[0].startsWith("anchor") &&
                                           row[2] === "frame 9 (6, 7)");
check("the new anchor has its own row", anchorRow >= 0, JSON.stringify(rows.map(r => r[2])));
rows[anchorRow][3]();
check("removing an anchor removes exactly that anchor",
      same(ui.buildConfig().scenes[0].manual_segments.clubhead.segments[0]
             .anchors.map((a) => a.frame), [2, 8]));
const seg = ui.buildConfig().scenes[0].manual_segments.clubhead;
check("removing an anchor leaves the segment, its note and its bound",
      seg.segments.length === 1 && seg.segments[0].note === "segment note" &&
      seg.segments[0].max_anchor_spacing_frames === 12 &&
      seg.note === "operator note on the block" &&
      seg.trail === "segments_only" && seg.trail_from_frame === 2);
ui.renderAnchors();
const roiRow = ui.state.anchorRows.findIndex((row) => row[0] === "ROI keyframe" &&
                                                     row[1] === "frame 7");
ui.state.anchorRows[roiRow][3]();
check("a ROI keyframe can be removed and the block survives",
      same(ui.buildConfig().scenes[0].roi_keyframes.map((k) => k.frame), [0, 10]));
ui.renderAnchors();
const corrRow = ui.state.anchorRows.findIndex((row) => row[0] === "correction" &&
                                                      row[2] === "frame 7 (41, 42)");
ui.state.anchorRows[corrRow][3]();
check("a correction can be removed",
      ui.buildConfig().scenes[0].corrections.clubhead.length === 1,
      JSON.stringify(ui.buildConfig().scenes[0].corrections));

// --- a scene with no assistance marks can still be built and given one ------
ui.setWindow(300, 330);
ui.setPhase("address", 301);
ui.addScene();
const fresh = ui.buildConfig().scenes[2];
check("a new scene carries no invented assistance marks",
      fresh.manual_segments === undefined && fresh.corrections === undefined &&
      fresh.roi_keyframes === undefined);
ui.selectScene(2);
ui.setTarget("ball");
ui.newSegment("ball", 302);
check("a new manual segment starts empty and awaits anchors",
      same(ui.buildConfig().scenes[2].manual_segments.ball.segments[0].anchors, []));
check("the new segment is listed for anchoring",
      /ball \/ segment_1/.test(document.getElementById("anchorSegment").innerHTML),
      document.getElementById("anchorSegment").innerHTML);


// The actual delivery uses compact [frame,x,y] anchors, and 1080p previews are capped
// at 1280 pixels. These cases protect saved edits against both real-browser defects.
check("new segments fit the strict near-contact cap",
      ui.currentScene().manual_segments.ball.segments[0].max_anchor_spacing_frames <= 6);
const mixed = ui.currentScene().manual_segments.ball.segments[0];
mixed.anchors = [[302, 960, 540], [306, 990, 540]];
ui.renderAnchors();
check("compact anchors show real frame and coordinates",
      ui.state.anchorRows.some(row => row[2] === "frame 302 (960, 540)"));
ui.addAnchorToSegment("ball|segment_1", {frame:304,x:975,y:540});
check("mixed compact/object anchors retain format and chronological order",
      same(mixed.anchors, [[302,960,540],
        {frame:304,x:975,y:540,note:"operator anchor placed in the review UI"},
        [306,990,540]]));
const preview = document.getElementById("view");
preview.width = 1280; preview.height = 720;
preview.getBoundingClientRect = () => ({left:17,top:56,width:1280,height:720});
ui.state.clip.width = 1920; ui.state.clip.height = 1080;
check("pointer coordinates map to source pixels rather than preview pixels",
      same(localPoint({clientX:397,clientY:396}), {x:570,y:510,scale:2/3}));
ui.setRoi([570,510,240,450]);
let rectangle;
drawRoi({save(){},restore(){},setLineDash(){},strokeRect(...args){rectangle=args;}},preview);
check("native region maps back onto the scaled picture",
      rectangle.every((value,i)=>Math.abs(value-[380,340,160,300][i])<1e-6));
ui.state.frame = 302; ui.state.lastPoint = null;
let circles = [];
drawMarks({save(){},restore(){},beginPath(){},fill(){},arc(...args){circles.push(args);}},preview);
check("compact target marks paint at scaled native positions",
      circles.some(args=>args[0]===640 && args[1]===360));

ui.selectClip("clip01");
check("changing source directly cannot carry another clip's assistance",
      ui.state.scenes.length === 0 && ui.state.config === null &&
      ui.state.roi === null && Object.keys(ui.state.phases).length === 0 &&
      ui.state.lastPoint === null &&
      document.getElementById("configName").value === "01_hole07_approach_ui.json");
check("a new source resets its pointer readout",
      document.getElementById("coordChip").textContent === "x – y –");
setFrame(0);
const staleImage = imageRequests[imageRequests.length - 1];
setFrame(1);
const activeImage = imageRequests[imageRequests.length - 1];
let draws = 0;
document.getElementById("view").getContext = () => ({...context2dStub(),drawImage(){draws++;}});
staleImage.onload();
check("out-of-order frame responses cannot repaint a newer selection", draws === 0);
activeImage.onload();
check("the selected native frame response paints normally", draws === 1);
console.log(JSON.stringify({results}));
process.exit(results.every((r) => r.ok) ? 0 : 1);
