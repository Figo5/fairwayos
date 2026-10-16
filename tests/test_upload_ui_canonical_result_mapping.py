"""A canonical job must render its OWN result shape in the review UI.

Runs the REAL page JavaScript out of ghostcaddie/upload/ui.py under node against
a stubbed DOM, so this pins the shipped mapping instead of a Python copy of it.

The defect this pins: the canonical result contract is
``{schema_version, source, layers, metrics, outputs: {annotated_video, …}}`` while
the research targets contract is ``{targets: {…}}``. ``renderTargets`` read
``res.targets`` unconditionally, so on a SUCCESSFUL canonical job it threw
``TypeError: Cannot convert undefined or null to object`` -- straight into the
red error line -- and ``poll()`` aborted before it set the raw panel, enabled the
download button, or pointed the rendered-result playback at ``/result-video``.
The job was ``done`` on the server and broken in the browser.

Also pinned: the annotated video is reached through ``outputs.annotated_video``,
not ``outputs.video``, so the playback mapping has to accept both.
"""
import json, os, re, shutil, subprocess, tempfile, unittest

UI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "ghostcaddie", "upload", "ui.py")

_FALLBACK = "~/.local/bin/node"
NODE = (os.environ.get("FAIRWAY_NODE")
        or (_FALLBACK if os.path.exists(_FALLBACK) else "")
        or (shutil.which("node") or ""))


def js_source():
    from ghostcaddie.upload.ui import PAGE
    i = PAGE.index("<script>")
    return PAGE[i + len("<script>"):PAGE.rindex("</script>")]


def extract(name, src):
    """Pull one top-level `function name(...){...}` out of the page script."""
    m = re.search(r"(?:async\s+)?function\s+%s\s*\(" % re.escape(name), src)
    if not m:
        raise AssertionError(f"function {name}() is not defined in ui.py")
    i = src.index("{", m.end() - 1)
    depth, j = 0, i
    while j < len(src):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[m.start():j + 1]
        j += 1
    raise AssertionError(f"unbalanced braces in {name}()")


#: The real shape a canonical job finishes with (trimmed to what the UI reads).
CANONICAL_RESULT = {
    "schema_version": "fairwayos-analysis.v1",
    "analysis": "canonical",
    "three_target_success": False,
    "note": "Canonical local analysis. Body is reported only where the golfer "
            "could be identified; ball and clubhead are unavailable by design.",
    "layers": {
        "body": {"state": "supported",
                 "method": "movenet_singlepose_lightning_v3_on_identified_subject",
                 "limitations": ["identity comes from a single-pose model"]},
        "ball": {"state": "unavailable", "method": "none",
                 "limitations": ["No ball route qualifies."]},
        "clubhead": {"state": "unavailable", "method": "none",
                     "limitations": ["No clubhead route qualifies."]},
    },
    "metrics": {"body_2d": {"supported": [], "unavailable": []},
                "physical": {"supported": [], "unavailable": []}},
    "outputs": {"annotated_video": "/tmp/qa/canonical/annotated.mp4",
                "annotated_video_written": True,
                "analysis": "/tmp/qa/canonical/analysis.json",
                "report": "/tmp/qa/canonical/report.md",
                "provenance": "/tmp/qa/canonical/provenance.json",
                "diagnostics": "/tmp/qa/canonical/diagnostics.json"},
}

RESEARCH_RESULT = {
    "targets": {"body": {"outcome": "not_run", "reason": "needs a reviewed seed"},
                "clubhead": {"outcome": "unavailable", "reason": "model absent"}},
    "three_target_success": False,
    "note": "research path",
}


@unittest.skipUnless(NODE and os.path.exists(NODE), "node is required to run the page JS")
class CanonicalResultUiMappingTests(unittest.TestCase):

    def _run(self, result, *, job_id="job1", state="done"):
        src = js_source()
        fns = "\n".join(extract(n, src) for n in ("renderTargets", "jget", "poll"))
        harness = """
const JOB_ID = %s, STATE = %s, RESULT = %s;
let JOB = null, POLL = null;
const initialDisabled = {cancel: true, refresh: true, dl: true, delMedia: true};
const nodes = {};
function el(id) {
  return {id, innerHTML: "", textContent: "", src: "", disabled: !!initialDisabled[id],
          style: {}, classList: {add(){}, remove(){}},
          getContext(){ return {}; }, getBoundingClientRect(){ return {}; }};
}
globalThis.$ = (s) => {
  const id = s.replace('#','');
  if (!nodes[id]) nodes[id] = el(id);
  return nodes[id];
};
globalThis.document = {querySelector: globalThis.$, createElement(){ return {}; }, body:{appendChild(){}}};
globalThis.setInterval = () => 1;
globalThis.clearInterval = () => {};
globalThis.fetch = async () => ({ok: true, json: async () => ({
  id: JOB_ID, state: STATE, progress: 1.0, error: "", result: RESULT,
  media_retention: {retained: true, reason: "retained for review until the deadline"}})});
%s
JOB = JOB_ID;
const out = {errors: []};
try { renderTargets(RESULT); } catch (e) { out.errors.push("renderTargets: " + e.message); }
out.targets_html = nodes["targets"] ? nodes["targets"].innerHTML : null;
(async () => {
  try { await poll(); } catch (e) { out.errors.push("poll: " + e.message); }
  out.jobErr = nodes["jobErr"] ? nodes["jobErr"].textContent : null;
  out.raw_len = nodes["raw"] && nodes["raw"].textContent ? nodes["raw"].textContent.length : 0;
  out.dl_disabled = nodes["dl"] ? nodes["dl"].disabled : null;
  out.result_video_src = nodes["resultVid"] ? nodes["resultVid"].src : null;
  console.log(JSON.stringify(out));
})();
""" % (json.dumps(job_id), json.dumps(state), json.dumps(result), fns)
        with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as fh:
            fh.write(harness)
            path = fh.name
        try:
            p = subprocess.run([NODE, path], capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 0, p.stderr)
            return json.loads(p.stdout.strip().splitlines()[-1])
        finally:
            os.unlink(path)

    # --- the regression -----------------------------------------------------

    def test_a_canonical_result_does_not_throw_in_renderTargets(self):
        out = self._run(CANONICAL_RESULT)
        self.assertEqual(out["errors"], [],
                         f"the served UI threw on a canonical result: {out['errors']}")

    def test_a_canonical_job_shows_no_error_and_no_empty_result(self):
        out = self._run(CANONICAL_RESULT)
        self.assertEqual(out["jobErr"], "",
                         "a successful canonical job must not render an error")
        self.assertGreater(out["raw_len"], 0,
                           "the raw result panel must be populated after poll()")

    def test_a_canonical_job_enables_the_result_and_playback(self):
        out = self._run(CANONICAL_RESULT)
        self.assertFalse(out["dl_disabled"],
                         "Download results must be enabled once a result exists")
        self.assertEqual(out["result_video_src"], "/result-video?job=job1",
                         "the rendered annotated video must be reachable through "
                         "outputs.annotated_video, not only outputs.video")

    def test_the_layer_states_and_reasons_are_rendered(self):
        html = self._run(CANONICAL_RESULT)["targets_html"] or ""
        for state in ("supported", "unavailable"):
            self.assertIn(state, html)
        self.assertIn("body", html)
        self.assertIn("ball", html)
        self.assertIn("clubhead", html)
        # the reason a layer is unavailable is shown, never only its state
        self.assertIn("No ball route qualifies.", html)
        self.assertIn("No clubhead route qualifies.", html)
        self.assertIn("physical", html)

    def test_no_invented_positive_state_is_rendered_for_canonical(self):
        """Canonical carries no research target outcomes at all."""
        out = self._run(CANONICAL_RESULT)
        self.assertEqual(out["errors"], [])
        self.assertNotIn("three target success", out["targets_html"] or "")

    # --- the other shape must keep working ---------------------------------

    def test_a_research_targets_result_still_renders_its_target_table(self):
        out = self._run(RESEARCH_RESULT)
        self.assertEqual(out["errors"], [])
        html = out["targets_html"] or ""
        self.assertIn("not_run", html)
        self.assertIn("unavailable", html)
        self.assertIn("needs a reviewed seed", html)
        self.assertIn("three target success", html)

    def test_a_result_with_neither_shape_is_stated_not_thrown(self):
        out = self._run({"unexpected": True})
        self.assertEqual(out["errors"], [])
        self.assertIn("no target or layer states", out["targets_html"] or "")


if __name__ == "__main__":
    unittest.main()
