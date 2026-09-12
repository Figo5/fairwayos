"""Whether a detector can be trained here, and if not, exactly what is missing.

"No golf detector checkpoint exists" is true but useless: it is the consequence,
not the cause. The cause is an ordered question --

    1. rights     may we use this asset at all?
    2. split      can we split it so a golfer/source cannot leak across splits?
    3. labels     do we have annotations?
    4. pixels     do we have the images those annotations describe?
    5. runtime    can we actually run training?

-- and the first unmet one is the blocker worth reporting. Reporting a later one
sends someone to solve the wrong problem.

Measured state, verified on disk rather than assumed:

  ClubheadDB (CC BY-NC 4.0) ships 10,180 YOLO-format clubhead boxes and a
  67-swing metadata table, and ZERO pixels: its image_path column points at the
  original author's desktop, and the package contains no image or video file at
  all. The pixels live on YouTube and Reddit and their reconstruction is gated --
  source-level permission is not documented and the Reddit URLs carry 2025
  expiry parameters.

  torch and torchvision ARE installed. The runtime is not the blocker and must
  never be reported as one.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import List, Optional

CLUBHEADDB_REL = "out/clubheaddb_recon/clubhead_db-1.0.1"
PIXEL_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".mov", ".mkv", ".webm")


@dataclass(frozen=True)
class Blocker:
    kind: str          # rights | split | labels | pixels | runtime
    detail: str
    remedy: str


@dataclass
class DatasetAsset:
    name: str
    license_declared: str
    rights_cleared: bool
    label_count: int
    label_path: str
    pixel_count: int
    pixel_root: str
    runtime_available: bool
    disjoint_split_field: str
    notes: str = ""


@dataclass
class TrainingFeasibility:
    can_train_now: bool
    runtime_present: bool
    runtime_detail: str
    assets: List[DatasetAsset] = field(default_factory=list)
    blockers: List[Blocker] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "can_train_now": self.can_train_now,
            "runtime_present": self.runtime_present,
            "runtime_detail": self.runtime_detail,
            "assets": [{"name": a.name, "license_declared": a.license_declared,
                        "rights_cleared": a.rights_cleared,
                        "label_count": a.label_count, "pixel_count": a.pixel_count,
                        "disjoint_split_field": a.disjoint_split_field,
                        "notes": a.notes} for a in self.assets],
            "blockers": [{"asset": b.kind, "kind": b.kind, "detail": b.detail,
                          "remedy": b.remedy} for b in self.blockers],
            "note": "A missing CHECKPOINT is a consequence, not a cause. The "
                    "blockers above are the causes, in the order they must be "
                    "resolved.",
            "research_only": True, "production_eligible": False,
        }


def first_blocker(a: DatasetAsset) -> Optional[Blocker]:
    if not a.rights_cleared:
        return Blocker("rights",
                       f"{a.name}: declared {a.license_declared}, but the "
                       f"underlying dataset/media rights are not cleared here. "
                       f"{a.notes}".strip(),
                       "obtain and record source-level permission before any "
                       "acquisition")
    if not a.disjoint_split_field:
        return Blocker("split",
                       f"{a.name}: no field identifies the golfer/source, so a "
                       f"split cannot be proven disjoint",
                       "add a per-record golfer/source id before splitting")
    if a.label_count <= 0:
        return Blocker("labels", f"{a.name}: no annotations are present locally",
                       "acquire or create annotations")
    if a.pixel_count <= 0:
        return Blocker("pixels",
                       f"{a.name}: {a.label_count} labels are present locally at "
                       f"{a.label_path}, but no image or video file ships with "
                       f"them, so there is nothing to train on",
                       "clear the rights for the referenced source media, then "
                       "acquire it with hashes recorded")
    if not a.runtime_available:
        return Blocker("runtime", f"{a.name}: no training runtime is importable",
                       "install a training runtime in an isolated env")
    return None


def _runtime() -> tuple:
    """Probe the isolated training interpreter, never this process."""
    import subprocess, sys
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for interp in (os.path.join(repo, ".venv-video-ai", "bin", "python3"),
                   sys.executable):
        if not os.path.exists(interp):
            continue
        p = subprocess.run(
            [interp, "-c", "import torch,torchvision;"
                           "print(torch.__version__,torchvision.__version__)"],
            capture_output=True, text=True)
        if p.returncode == 0:
            t, tv = p.stdout.split()
            return True, f"torch {t} + torchvision {tv} in {interp}"
    return False, "no interpreter with torch+torchvision was found"


def CLUBHEADDB(repo: str) -> DatasetAsset:
    root = os.path.join(repo, CLUBHEADDB_REL)
    labels = os.path.join(root, "src", "clubhead_db", "data", "annotations.parquet")
    meta = os.path.join(root, "src", "clubhead_db", "data", "metadata.csv")
    # the published frame count is recorded in the recon status; the parquet
    # itself needs a reader we deliberately have not added for a blocked path
    label_count = 0
    status = os.path.join(repo, "out", "clubheaddb_recon", "recon_status.json")
    if os.path.exists(status):
        import json
        label_count = int(json.load(open(status)).get("bundled_assets", {})
                          .get("published_annotation_frame_count", 0))
    pixels = 0
    if os.path.isdir(root):
        for _, _, files in os.walk(root):
            pixels += sum(1 for f in files
                          if os.path.splitext(f)[1].lower() in PIXEL_EXTS)
    golfers = 0
    if os.path.exists(meta):
        with open(meta, encoding="utf-8-sig") as fh:
            golfers = len({r["notes"] for r in csv.DictReader(fh, delimiter=";")
                           if r.get("notes")})
    ok, detail = _runtime()
    return DatasetAsset(
        name="ClubheadDB", license_declared="CC-BY-NC-4.0",
        # research terms are declared, but the underlying YouTube/Reddit media
        # is third-party and its reuse is NOT documented
        rights_cleared=False,
        label_count=label_count, label_path=labels,
        pixel_count=pixels, pixel_root=root,
        runtime_available=ok, disjoint_split_field="notes (golfer)",
        notes=(f"{golfers} distinct golfers over 67 down-the-line swings; source "
               f"media is third-party YouTube/Reddit and the Reddit URLs carry "
               f"2025 expiry parameters. Labels ship; pixels do not."))


# Independently verified against clean native crops by the coordinator: of the
# eight proximity hits the ONNX golf-ball detector produced over f3055-f3105,
# these two are the REAL moving ball in flight.
BALL_VERIFIED_MOVING_FRAMES = (3075, 3079)
# These six are the stationary ball at address, before it is struck. They are
# real detections of a real ball, but a ball that is not going anywhere; they
# say nothing about tracking a shot.
BALL_STATIONARY_HIT_FRAMES = (3055, 3056, 3057, 3058, 3059, 3060)


def GOLFBALL_ONNX(repo: str) -> DatasetAsset:
    """The golf-ball ONNX detector that was actually run on the Rory interval.

    Correcting two earlier overstatements of mine:
      * it is NOT license-undeclared -- the ONNX metadata declares AGPL-3.0 and
        names={0: golf_ball}, with output shape [1,5,8400];
      * it did NOT produce zero ball detections.

    What it did produce, over 51 frames: 6 hits on the stationary ball at
    address, 2 visually confirmed hits on the moving ball (f3075, f3079), and
    218 predictions that were not the ball. That is a weak detector on this
    footage, not a tracker, and it is not reported as one.
    """
    ok, _ = _runtime()
    return DatasetAsset(
        name="golf-ball ONNX detector (notjulietxd/golf-ball-tracker best.onnx)",
        license_declared="AGPL-3.0",
        # the WEIGHTS declare a licence; the data they were trained on does not
        rights_cleared=False,
        label_count=0, label_path="", pixel_count=0, pixel_root="",
        runtime_available=ok, disjoint_split_field="n/a (pretrained weights)",
        notes=("ONNX metadata declares AGPL-3.0 with names={0: golf_ball} and "
               "output [1,5,8400]; runs under onnxruntime with no torch/pickle. "
               "On Rory f3055-f3105 with 640px tiles it produced 8 proximity "
               f"hits: {len(BALL_STATIONARY_HIT_FRAMES)} on the STATIONARY ball "
               f"at address (f3055-f3060) and 2 visually confirmed on the MOVING "
               f"ball (f{BALL_VERIFIED_MOVING_FRAMES[0]}, "
               f"f{BALL_VERIFIED_MOVING_FRAMES[1]}), against 218 predictions "
               "that were not the ball. This is a weak detector on this footage "
               "and is not continuous tracking. Its training-dataset rights are "
               "unresolved, and AGPL-3.0 carries obligations of its own. "
               "PRESERVED DIAGNOSTIC (not acted on here): at the existing 0.01 "
               "raw floor, 46 of 49 reviewed-visible frames -- including 40 "
               "moving frames -- have a post-NMS candidate whose centre is "
               "within 15 px of the reviewed ball, while only 8 pass 0.25. "
               "Proximity is NOT identity and dense false candidates may explain "
               "it, so no recall is claimed and no threshold was moved; "
               "candidate identity and the ONNX decoder are under independent "
               "audit."))


def GOLFBALL_ROBOFLOW(repo: str) -> DatasetAsset:
    ok, _ = _runtime()
    return DatasetAsset(
        name="JMS-GOLF-BALL-TRACKING (Roboflow Universe)",
        license_declared="CC-BY-4.0", rights_cleared=True,
        label_count=0, label_path="", pixel_count=0, pixel_root="",
        runtime_available=ok, disjoint_split_field="source image",
        notes="6,392 single-class golfball IMAGES advertised. Nothing is present "
              "locally: neither labels nor pixels. Retrieval needs network "
              "access and an account/API key, which this environment does not "
              "have.")


def assess(repo: str) -> TrainingFeasibility:
    ok, detail = _runtime()
    assets = [CLUBHEADDB(repo), GOLFBALL_ONNX(repo), GOLFBALL_ROBOFLOW(repo)]
    blockers = [b for b in (first_blocker(a) for a in assets) if b]
    return TrainingFeasibility(can_train_now=not blockers, runtime_present=ok,
                               runtime_detail=detail, assets=assets,
                               blockers=blockers)
