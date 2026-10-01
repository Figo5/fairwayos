// Just enough of a browser for the broadcast review page's own script to run under
// node, so its data handling can be tested for real instead of described.
//
// This file is prepended to the <script> body of the page (see
// tests/test_demo_broadcast_ui.py). It defines every global the page touches:
// a document whose elements are created on first request, an Image that never fires
// its onload, a fetch that answers the page's read-only endpoints and a confirm() the
// test controls.

const elements = new Map();

function context2dStub() {
  return {
    save() {}, restore() {}, beginPath() {}, closePath() {}, arc() {}, fill() {},
    drawImage() {}, strokeRect() {}, setLineDash() {}, clearRect() {},
    fillRect() {}, strokeText() {}, fillText() {}, measureText() { return {width: 0}; }
  };
}

function elementFor(id) {
  if (elements.has(id)) return elements.get(id);
  const listeners = {};
  const el = {
    id, value: "", textContent: "", className: "", innerHTML: "",
    dataset: {}, style: {}, width: 1266, height: 720, min: 0, max: 0, checked: false,
    listeners,
    addEventListener(type, fn) { (listeners[type] = listeners[type] || []).push(fn); },
    removeEventListener() {},
    querySelectorAll() { return []; },
    querySelector() { return null; },
    closest() { return null; },
    appendChild() {},
    setPointerCapture() {},
    getBoundingClientRect() { return {left: 0, top: 0, width: 1266, height: 720}; },
    getContext() { return context2dStub(); }
  };
  elements.set(id, el);
  return el;
}

const documentStub = {
  getElementById: elementFor,
  addEventListener() {},
  querySelectorAll() { return []; },
  createElement() { return elementFor(`__detached${elements.size}`); }
};

const imageRequests = [];
class ImageStub {
  constructor() {
    imageRequests.push(this);
    this.onload = null;
    this.onerror = null;
    this.naturalWidth = 1266;
    this.naturalHeight = 720;
    this._src = "";
  }
  set src(value) { this._src = value; }
  get src() { return this._src; }
}

// Two registered clips, deliberately including the pair the operator hit in the real
// UI: a Hole 7 clip and the Hole 12 clip a config names.
const CLIP_LIST = [
  {id: "clip01", name: "01_hole07_approach.mp4", path: "/clips/01_hole07_approach.mp4",
   sha256: "a".repeat(64), width: 1266, height: 720, fps_rational: "9832/331",
   frame_count: 467, has_audio: true},
  {id: "clip03", name: "03_hole12_tee.mp4", path: "/clips/03_hole12_tee.mp4",
   sha256: "b".repeat(64), width: 1266, height: 720, fps_rational: "9832/331",
   frame_count: 389, has_audio: true}
];

let confirmAnswer = true;
const requested = [];

async function fetchStub(url) {
  requested.push(String(url));
  let payload = {ok: true, configs: []};
  if (String(url).startsWith("/api/clips")) payload = {ok: true, clips: CLIP_LIST};
  else if (String(url).startsWith("/api/library")) payload = {ok: true, roots: []};
  return {
    ok: true, status: 200, statusText: "OK",
    async text() { return JSON.stringify(payload); },
    async json() { return payload; }
  };
}

globalThis.window = globalThis;
globalThis.document = documentStub;
globalThis.Image = ImageStub;
globalThis.fetch = fetchStub;
globalThis.confirm = () => confirmAnswer;
globalThis.performance = globalThis.performance || {now: () => Date.now()};
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
