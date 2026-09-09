"""Regression tests for truthful render/reconciliation semantics.

Covers exactly the defects Astra identified:
1. Trails clear across unavailable gaps, rejected/unclear points, new segments.
2. Rejected/off-clubhead observations must NOT render as accepted green.
3. Source/subclip frame-index mapping is exact.
4. Counts reconcile seed/predictive/supported/rejected/unclear/unavailable
   correctly and never claim unreviewed or rejected frames as supported.
"""
import unittest
from ghostcaddie.video.render_reconcile import (
    build_confirmed_segments,
    classify_display,
    reconcile_counts,
    source_index_for_frame,
)


class FrameMappingTests(unittest.TestCase):
    def test_local_to_source_mapping(self):
        self.assertEqual(source_index_for_frame(145, 5), 150)
        self.assertEqual(source_index_for_frame(145, 0), 145)
        self.assertEqual(source_index_for_frame(0, 200), 200)

    def test_rejects_negative_frame(self):
        with self.assertRaises(ValueError):
            source_index_for_frame(145, -1)


class DisplayClassificationTests(unittest.TestCase):
    def test_unavailable_stays_unavailable(self):
        self.assertEqual(classify_display("unavailable", "on_clubhead"), "unavailable")

    def test_on_clubhead_is_green(self):
        self.assertEqual(classify_display("observed", "on_clubhead"), "on_clubhead")

    def test_rejected_body_or_shaft(self):
        self.assertEqual(classify_display("observed", "on_body"), "rejected")
        self.assertEqual(classify_display("observed", "on_shaft"), "rejected")

    def test_unclear(self):
        self.assertEqual(classify_display("observed", "unclear"), "unclear")

    def test_unreviewed_never_green(self):
        # observed but no AI review -> unreviewed, never shown as on_clubhead
        self.assertEqual(classify_display("observed", None), "unreviewed")


class TrailGapTests(unittest.TestCase):
    def _states(self, mapping):
        return lambda f: mapping.get(f, "unavailable")

    def test_gap_clears_segment(self):
        # on at 150-152, gap 153, on again 154-155 -> two segments
        st = {150:"on_clubhead",151:"on_clubhead",152:"on_clubhead",
              154:"on_clubhead",155:"on_clubhead"}
        segs = build_confirmed_segments(range(145,157), self._states(st))
        self.assertEqual(segs, [(150,152),(154,155)])

    def test_rejected_point_breaks_trail(self):
        st = {150:"on_clubhead",151:"rejected",152:"on_clubhead"}
        segs = build_confirmed_segments(range(150,153), self._states(st))
        self.assertEqual(segs, [(150,150),(152,152)])

    def test_no_bridging_unclear(self):
        st = {150:"on_clubhead",151:"unclear",152:"on_clubhead"}
        segs = build_confirmed_segments(range(150,153), self._states(st))
        self.assertEqual(segs, [(150,150),(152,152)])


class ReconcileCountsTests(unittest.TestCase):
    def test_counts_are_truthful(self):
        # interval 150-152
        interval = [150,151,152]
        # 150 seed observed+on_clubhead, 151 observed but on_body (rejected),
        # 152 observed but unreviewed
        raw = {150:"observed",151:"observed",152:"observed"}
        verdicts = {150:"on_clubhead",151:"on_body",152:None}
        c = reconcile_counts(interval, lambda f: raw.get(f,"unavailable"),
                             lambda f: verdicts.get(f), seed_source=150)
        self.assertEqual(c["seed"], 1)
        self.assertEqual(c["total_observed"], 3)
        self.assertEqual(c["predictive"], 2)          # 151,152
        self.assertEqual(c["visually_supported"], 1)  # 150
        self.assertEqual(c["rejected"], 1)            # 151
        self.assertEqual(c["unreviewed"], 1)          # 152
        self.assertEqual(c["unclear"], 0)
        self.assertEqual(c["unavailable"], 0)
        # critical: 151 (rejected) and 152 (unreviewed) are NOT counted supported
        self.assertLess(c["visually_supported"], c["total_observed"])

    def test_unavailable_and_unclear_separated(self):
        interval=[150,151,152]
        raw={150:"observed",151:"observed",152:"unavailable"}
        verdicts={150:"on_clubhead",151:"unclear"}
        c=reconcile_counts(interval, lambda f: raw.get(f,"unavailable"),
                           lambda f: verdicts.get(f), seed_source=150)
        self.assertEqual(c["visually_supported"],1)
        self.assertEqual(c["unclear"],1)
        self.assertEqual(c["unavailable"],1)
        self.assertEqual(c["rejected"],0)


if __name__ == "__main__":
    unittest.main()
