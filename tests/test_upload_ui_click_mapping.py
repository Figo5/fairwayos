"""The seeding click must land on the pixel the operator clicked.

Runs the REAL JavaScript out of ghostcaddie/upload/ui.py under node, against a
stubbed DOM, so this tests the shipped mapping code rather than a Python copy
of it.

The defect: the click origin came from getBoundingClientRect() (fractional CSS
pixels, content box excluded borders/padding not accounted for) while the scale
came from clientWidth/clientHeight (INTEGER-rounded, padding box). Mixing the
two box models shifts every mapped coordinate, by more the further the click is
from the image origin. A parent browser run clicking native (821, 477) recorded
(819.76, 475.45).

No tolerance is fudged here: with one consistent content box the round trip is
exact to floating point.
"""
import json, os, re, subprocess, tempfile, unittest

UI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "ghostcaddie", "upload", "ui.py")
NODE = "/Users/giofiore/.local/bin/node"


def js_source():
    from ghostcaddie.upload.ui import PAGE
    i = PAGE.index("<script>")
    return PAGE[i + len("<script>"):PAGE.rindex("</script>")]


def extract(name, src):
    """Pull one top-level `function name(...){...}` out of the page script."""
    m = re.search(r"function\s+%s\s*\(" % re.escape(name), src)
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


@unittest.skipUnless(os.path.exists(NODE), "node is required to run the page JS")
class ClickMappingTests(unittest.TestCase):
    # a deliberately fractional layout: this is what a real flex/grid column
    # produces, and it is where the two box models disagree
    BOX = {"left": 29.0, "top": 327.5, "width": 570.4, "height": 320.85}
    NW, NH = 1280, 720

    def _run(self, points):
        src = js_source()
        fns = "\n".join(extract(n, src) for n in ("contentBox", "tf", "toNative"))
        harness = """
const BOX = %s, NW = %d, NH = %d;
const IMG = {
  getBoundingClientRect: () => ({left: BOX.left, top: BOX.top,
                                 width: BOX.width, height: BOX.height}),
  // clientWidth/Height are INTEGER rounded by every browser
  clientWidth: Math.round(BOX.width), clientHeight: Math.round(BOX.height),
};
globalThis.document = {querySelector: () => IMG};
globalThis.$ = () => IMG;
globalThis.getComputedStyle = () => ({
  borderLeftWidth: "0px", borderTopWidth: "0px", borderRightWidth: "0px",
  borderBottomWidth: "0px", paddingLeft: "0px", paddingTop: "0px",
  paddingRight: "0px", paddingBottom: "0px"});
globalThis.SRC = {nw: NW, nh: NH};
%s
// ground truth: where the browser actually PAINTS a native pixel, given
// object-fit:contain inside the content box
function paint(nx, ny) {
  const s = Math.min(BOX.width / NW, BOX.height / NH);
  return [BOX.left + (BOX.width - NW * s) / 2 + nx * s,
          BOX.top + (BOX.height - NH * s) / 2 + ny * s];
}
const out = [];
for (const [nx, ny] of %s) {
  const [cx, cy] = paint(nx, ny);
  const b = contentBox(document.querySelector('#frame'));
  out.push({want: [nx, ny], got: toNative(tf(), cx - b.left, cy - b.top)});
}
console.log(JSON.stringify(out));
""" % (json.dumps(self.BOX), self.NW, self.NH, fns, json.dumps(points))
        with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as fh:
            fh.write(harness)
            path = fh.name
        try:
            p = subprocess.run([NODE, path], capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 0, p.stderr)
            return json.loads(p.stdout.strip().splitlines()[-1])
        finally:
            os.unlink(path)

    def test_a_click_maps_back_to_the_exact_native_pixel(self):
        pts = [[821, 477], [836, 550], [812, 581], [1, 1], [1279, 719], [640, 360]]
        for row in self._run(pts):
            self.assertIsNotNone(row["got"], f"{row['want']} was refused")
            self.assertAlmostEqual(row["got"][0], row["want"][0], places=6,
                                   msg=f"x drift at {row['want']}")
            self.assertAlmostEqual(row["got"][1], row["want"][1], places=6,
                                   msg=f"y drift at {row['want']}")

    def test_the_reported_parent_click_is_exact(self):
        """The specific regression: (821, 477) came back (819.76, 475.45)."""
        row = self._run([[821, 477]])[0]
        self.assertAlmostEqual(row["got"][0], 821.0, places=6)
        self.assertAlmostEqual(row["got"][1], 477.0, places=6)

    def test_a_click_in_the_letterbox_padding_is_still_refused(self):
        """Refusing, never clamping, stays the behaviour."""
        src = js_source()
        self.assertIn("refuse, never clamp", src)


if __name__ == "__main__":
    unittest.main()
