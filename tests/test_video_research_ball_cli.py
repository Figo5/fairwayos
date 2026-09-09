"""Focused TDD tests for the research-only ``research-ball-track`` CLI."""
import json
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from ghostcaddie.cli import main

RESEARCH_STOCK = Path(__file__).resolve().parent.parent / "research_stock" / "pexels_6573644.mp4"


def _make_synthetic_video(path: Path, frames: int = 12) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=10:duration={frames / 10}",
        "-pix_fmt", "yuv420p", str(path),
    ], check=True)


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,pix_fmt,avg_frame_rate",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True)
    return json.loads(out.stdout)["streams"][0]


class ResearchBallTrackValidationTests(unittest.TestCase):
    def run_cli(self, *extra):
        with redirect_stdout(StringIO()) as stdout, redirect_stderr(StringIO()) as stderr:
            code = 0
            try:
                main(["research-ball-track", *extra])
            except SystemExit as exc:
                code = exc.code or 0
        return code, stdout.getvalue(), stderr.getvalue()

    def test_unknown_subcommand_is_rejected(self):
        code, _, stderr = self.run_cli("--video", str(RESEARCH_STOCK))
        self.assertNotEqual(code, 0)
        self.assertIn("research-ball-track", stderr)

    def test_required_arguments_are_enforced(self):
        code, _, _ = self.run_cli("--video", str(RESEARCH_STOCK))
        self.assertEqual(code, 2)

    def test_seed_outside_roi_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "src.mp4"
            _make_synthetic_video(video)
            code, _, _ = self.run_cli(
                "--video", str(video), "--out", str(Path(tmp) / "out"),
                "--start-frame", "0", "--end-frame", "6",
                "--seed-frame", "2", "--seed-x", "10", "--seed-y", "10",
                "--roi", "100", "60", "240", "200")
            self.assertNotEqual(code, 0)

    def test_inverted_frame_range_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "src.mp4"
            _make_synthetic_video(video)
            code, _, _ = self.run_cli(
                "--video", str(video), "--out", str(Path(tmp) / "out"),
                "--start-frame", "4", "--end-frame", "2",
                "--seed-frame", "2", "--seed-x", "160", "--seed-y", "120",
                "--roi", "100", "60", "240", "200")
            self.assertNotEqual(code, 0)

    def test_seed_frame_outside_range_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "src.mp4"
            _make_synthetic_video(video)
            code, _, _ = self.run_cli(
                "--video", str(video), "--out", str(Path(tmp) / "out"),
                "--start-frame", "0", "--end-frame", "6",
                "--seed-frame", "9", "--seed-x", "160", "--seed-y", "120",
                "--roi", "100", "60", "240", "200")
            self.assertNotEqual(code, 0)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                     "ffmpeg/ffprobe required")
class ResearchBallTrackEndToEndTests(unittest.TestCase):
    START, END, SEED_FRAME = 1, 8, 3

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.video = self.root / "src.mp4"
        _make_synthetic_video(self.video, frames=12)
        self.out = self.root / "out"
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            main([
                "research-ball-track",
                "--video", str(self.video), "--out", str(self.out),
                "--start-frame", str(self.START), "--end-frame", str(self.END),
                "--seed-frame", str(self.SEED_FRAME),
                "--seed-x", "160", "--seed-y", "120",
                "--roi", "100", "60", "240", "200",
            ])

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_research_artifacts(self):
        for name in ("diagnostics.json", "provenance.json", "README.md",
                     "annotated_source_rate.mp4", "annotated_slow_view.mp4",
                     "seed_crop.mp4", "contact_sheet.jpg"):
            self.assertTrue((self.out / name).is_file(), name)

    def test_videos_are_h264_yuv420p(self):
        for name in ("annotated_source_rate.mp4", "annotated_slow_view.mp4", "seed_crop.mp4"):
            stream = _probe(self.out / name)
            self.assertEqual(stream["codec_name"], "h264", name)
            self.assertEqual(stream["pix_fmt"], "yuv420p", name)

    def test_slow_view_is_slower_than_source_rate(self):
        source = _probe(self.out / "annotated_source_rate.mp4")["avg_frame_rate"]
        slow = _probe(self.out / "annotated_slow_view.mp4")["avg_frame_rate"]

        def rate(text):
            num, den = text.split("/")
            return float(num) / float(den)
        self.assertLess(rate(slow), rate(source))

    def test_diagnostics_preserve_source_frame_indices(self):
        payload = json.loads((self.out / "diagnostics.json").read_text())
        self.assertTrue(payload["research_only"])
        self.assertFalse(payload["ground_truth"])
        self.assertFalse(payload["production_eligible"])
        self.assertEqual(payload["seed_frame_index"], self.SEED_FRAME)
        self.assertEqual(payload["source_frame_start"], self.START)
        self.assertEqual(payload["source_frame_end"], self.END)
        items = payload["track"]["items"]
        self.assertEqual([item["frame_index"] for item in items],
                         list(range(self.START, self.END + 1)))
        seeded = [item for item in items if item["provenance"] == "seeded"]
        self.assertEqual(len(seeded), 1)
        self.assertEqual(seeded[0]["frame_index"], self.SEED_FRAME)

    def test_provenance_flags_and_source_label(self):
        payload = json.loads((self.out / "provenance.json").read_text())
        self.assertTrue(payload["research_only"])
        self.assertFalse(payload["ground_truth"])
        self.assertFalse(payload["production_eligible"])
        self.assertEqual(payload["seed_frame_index"], self.SEED_FRAME)
        self.assertEqual(payload["source_frame_range"], [self.START, self.END])
        self.assertNotIn(str(self.video), json.dumps(payload))
        self.assertEqual(payload["source_file"], self.video.name)
        self.assertEqual(payload["tracker"], "SeededBallTracker")

    def test_readme_marks_research_only(self):
        text = (self.out / "README.md").read_text()
        self.assertIn("RESEARCH ONLY", text)
        self.assertIn("NOT", text.upper())


if __name__ == "__main__":
    unittest.main()