"""Ball dataset v2 contract tests: provenance, states, splits, leakage guards.

These tests are the reason the dataset can be trusted as an evaluation base. They
assert the things that are expensive to notice later: a licence field that was
never filled in, a label that carries a coordinate on a frame where the ball is
absent, a shoot group that leaked across splits, final-test data reachable from a
development helper, and a machine guess promoted to a reviewed label.

Nothing here touches canonical perception, and the canonical ball layer is
asserted to remain UNAVAILABLE.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghostcaddie.video import ball_dataset as bd  # noqa: E402
from ghostcaddie.video import canonical  # noqa: E402
from ghostcaddie.video.errors import VideoContractError  # noqa: E402

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def make_clip(tmp: Path, name: str, *, frames: int = 8, width: int = 64,
              height: int = 48, fill: int = 30) -> tuple[str, str, str]:
    """Write a tiny real video and return (clip_id, media_path, sha256)."""
    import cv2
    import numpy as np

    media_dir = tmp / "out" / "ball_dataset_v2" / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    path = media_dir / f"{name}.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                             (width, height))
    for index in range(frames):
        frame = np.full((height, width, 3), (fill + index) % 255, dtype="uint8")
        cv2.circle(frame, (width // 2, height // 2), 4, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return name, f"out/ball_dataset_v2/media/{name}.mp4", digest


def source_record(clip_id: str, media_path: str, sha: str, *, group: str | None = None,
                  width: int = 64, height: int = 48, frames: int = 8,
                  tags: list | None = None, license_name: str = "Pexels License",
                  license_url: str | None = "https://www.pexels.com/license/") -> dict:
    return {
        "clip_id": clip_id,
        "provider": "pexels",
        "source_identifier": clip_id.split("_")[-1],
        "source_url": f"https://www.pexels.com/video/{clip_id.split('_')[-1]}/",
        "download_url": f"https://www.pexels.com/download/video/{clip_id.split('_')[-1]}/",
        "title": f"fixture {clip_id}",
        "acquisition_date": "2026-09-18",
        "license": license_name,
        "license_url": license_url,
        "license_text_reference": None,
        "rights_class": "stock_license_reusable",
        "original_filename": f"{clip_id}.mp4",
        "media_path": media_path,
        "bytes": 4096,
        "sha256": sha,
        "width": width,
        "height": height,
        "fps": 10.0,
        "frames": frames,
        "duration_s": 0.8,
        "codec": "h264",
        "container": "mov,mp4,m4a,3gp,3g2,mj2",
        "source_group": group or f"pexels_{clip_id.split('_')[-1][:4]}",
        "intended_tags": ["tee"],
        "tags": list(tags or []),
        "notes": "",
    }


def manifest(*records: dict, dataset_id: str = "fixture") -> dict:
    return {"schema_version": bd.SCHEMA_SOURCES, "dataset_id": dataset_id,
            "clips": list(records)}


class DatasetCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def fixture(self, count: int = 3) -> tuple[dict, dict]:
        records = []
        for index in range(count):
            clip_id, media_path, sha = make_clip(
                self.tmp, f"pexels_10000{index}", fill=20 + index * 60)
            records.append(source_record(clip_id, media_path, sha,
                                         group=f"pexels_100{index}"))
        return manifest(*records), {r["clip_id"]: r for r in records}


class ProvenanceTests(DatasetCase):
    def test_a_missing_licence_fails_the_manifest(self):
        sources, _ = self.fixture(1)
        del sources["clips"][0]["license"]
        with self.assertRaises(VideoContractError) as ctx:
            bd.validate_sources(sources)
        self.assertIn("license", str(ctx.exception))

    def test_an_empty_licence_url_fails_the_manifest(self):
        sources, _ = self.fixture(1)
        sources["clips"][0]["license_url"] = ""
        with self.assertRaises(VideoContractError):
            bd.validate_sources(sources)

    def test_a_malformed_source_hash_is_refused(self):
        sources, _ = self.fixture(1)
        sources["clips"][0]["sha256"] = "not-a-hash"
        with self.assertRaises(VideoContractError):
            bd.validate_sources(sources)

    def test_two_clips_may_not_share_one_source_hash(self):
        sources, records = self.fixture(2)
        first = list(records.values())[0]
        sources["clips"][1]["sha256"] = first["sha256"]
        with self.assertRaises(VideoContractError) as ctx:
            bd.validate_sources(sources)
        self.assertIn("duplicate source sha256", str(ctx.exception))

    def test_media_may_not_live_outside_the_gitignored_out_tree(self):
        sources, _ = self.fixture(1)
        sources["clips"][0]["media_path"] = "data/media/clip.mp4"
        with self.assertRaises(VideoContractError) as ctx:
            bd.validate_sources(sources)
        self.assertIn("never committed", str(ctx.exception))

    def test_media_is_required_to_be_recorded(self):
        sources, _ = self.fixture(1)
        del sources["clips"][0]["media_path"]
        with self.assertRaises(VideoContractError):
            bd.validate_sources(sources)

    def test_unknown_hard_negative_tag_is_refused(self):
        sources, _ = self.fixture(1)
        sources["clips"][0]["tags"] = ["definitely_not_a_tag"]
        with self.assertRaises(VideoContractError):
            bd.validate_sources(sources)

    def test_known_hard_negative_tags_are_accepted(self):
        sources, _ = self.fixture(1)
        sources["clips"][0]["tags"] = ["body", "clothing", "tee", "glare"]
        validated = bd.validate_sources(sources)
        self.assertEqual(validated["clips"][0]["tags"],
                         ["body", "clothing", "tee", "glare"])


class AnnotationStateTests(DatasetCase):
    def annotation(self, sources: dict, clip_id: str) -> dict:
        clip = bd.sources_by_id(sources)[clip_id]
        return bd.new_annotation(clip_id=clip_id, video=clip["media_path"],
                                 source_sha256=clip["sha256"], width=clip["width"],
                                 height=clip["height"], fps=clip["fps"],
                                 frames_total=clip["frames"], annotator="fixture")

    def test_a_frame_without_a_state_is_refused(self):
        sources, records = self.fixture(1)
        clip_id = list(records)[0]
        payload = self.annotation(sources, clip_id)
        payload["frames"]["3"] = {"origin": bd.ORIGIN_AGENT_REVIEWED}
        with self.assertRaises(VideoContractError) as ctx:
            bd.validate_annotation(payload)
        self.assertIn("no state", str(ctx.exception))

    def test_an_unknown_state_is_refused(self):
        sources, records = self.fixture(1)
        payload = self.annotation(sources, list(records)[0])
        payload["frames"]["3"] = {"state": "PROBABLY", "origin": bd.ORIGIN_AGENT_REVIEWED}
        with self.assertRaises(VideoContractError):
            bd.validate_annotation(payload)

    def test_visible_requires_coordinates(self):
        sources, records = self.fixture(1)
        payload = self.annotation(sources, list(records)[0])
        payload["frames"]["3"] = {"state": bd.STATE_VISIBLE,
                                  "origin": bd.ORIGIN_AGENT_REVIEWED, "x": 10.0}
        with self.assertRaises(VideoContractError) as ctx:
            bd.validate_annotation(payload)
        self.assertIn("must carry y", str(ctx.exception))

    def test_coordinates_outside_the_frame_are_refused(self):
        sources, records = self.fixture(1)
        payload = self.annotation(sources, list(records)[0])
        payload["frames"]["3"] = {"state": bd.STATE_VISIBLE,
                                  "origin": bd.ORIGIN_AGENT_REVIEWED,
                                  "x": 999.0, "y": 12.0}
        with self.assertRaises(VideoContractError) as ctx:
            bd.validate_annotation(payload)
        self.assertIn("outside the frame", str(ctx.exception))

    def test_absent_frames_may_not_carry_coordinates(self):
        sources, records = self.fixture(1)
        payload = self.annotation(sources, list(records)[0])
        for state in (bd.STATE_NOT_VISIBLE, bd.STATE_OCCLUDED, bd.STATE_AMBIGUOUS):
            payload["frames"]["3"] = {"state": state, "origin": bd.ORIGIN_AGENT_REVIEWED,
                                      "reason": "fixture", "x": 5.0, "y": 5.0}
            with self.assertRaises(VideoContractError) as ctx:
                bd.validate_annotation(payload)
            self.assertIn("fabricated position", str(ctx.exception))

    def test_not_visible_requires_a_reason(self):
        sources, records = self.fixture(1)
        payload = self.annotation(sources, list(records)[0])
        payload["frames"]["3"] = {"state": bd.STATE_NOT_VISIBLE,
                                  "origin": bd.ORIGIN_AGENT_REVIEWED}
        with self.assertRaises(VideoContractError):
            bd.validate_annotation(payload)

    def test_an_impossible_radius_is_refused(self):
        sources, records = self.fixture(1)
        payload = self.annotation(sources, list(records)[0])
        payload["frames"]["3"] = {"state": bd.STATE_VISIBLE,
                                  "origin": bd.ORIGIN_AGENT_REVIEWED,
                                  "x": 10.0, "y": 10.0, "radius": 5000.0}
        with self.assertRaises(VideoContractError):
            bd.validate_annotation(payload)

    def test_a_frame_beyond_the_clip_length_is_refused(self):
        sources, records = self.fixture(1)
        payload = self.annotation(sources, list(records)[0])
        payload["frames"]["900"] = {"state": bd.STATE_NOT_VISIBLE,
                                    "origin": bd.ORIGIN_AGENT_REVIEWED,
                                    "reason": "fixture"}
        with self.assertRaises(VideoContractError) as ctx:
            bd.validate_annotation(payload)
        self.assertIn("beyond the clip length", str(ctx.exception))

    def test_human_ground_truth_must_name_a_reviewer(self):
        sources, records = self.fixture(1)
        payload = self.annotation(sources, list(records)[0])
        payload["frames"]["3"] = {"state": bd.STATE_VISIBLE,
                                  "origin": bd.ORIGIN_HUMAN, "x": 10.0, "y": 10.0,
                                  "reviewer": ""}
        with self.assertRaises(VideoContractError) as ctx:
            bd.validate_annotation(payload)
        self.assertIn("without naming the reviewer", str(ctx.exception))

    def test_duplicate_frame_keys_in_the_file_are_refused(self):
        path = self.tmp / "dupe.json"
        path.write_text(json.dumps({
            "schema_version": bd.SCHEMA_ANNOTATION, "clip_id": "x",
            "video": "out/x.mp4", "source_sha256": SHA_A, "width": 10, "height": 10,
            "frames": {}, "proposals": {}}).replace(
                '"frames": {}', '"frames": {"1": {"state": "AMBIGUOUS", "origin": '
                '"AGENT_REVIEWED", "reason": "a"}, "1": {"state": "AMBIGUOUS", '
                '"origin": "AGENT_REVIEWED", "reason": "b"}}'))
        with self.assertRaises(VideoContractError) as ctx:
            bd.load_annotation(path)
        self.assertIn("duplicate key", str(ctx.exception))

    def test_an_annotation_bound_to_a_different_hash_is_refused(self):
        sources, records = self.fixture(1)
        clip_id = list(records)[0]
        payload = self.annotation(sources, clip_id)
        payload["source_sha256"] = SHA_C
        path = bd.annotation_path_for(self.tmp / "ann", clip_id)
        bd.save_annotation(path, payload)
        with self.assertRaises(VideoContractError) as ctx:
            bd.load_annotation(path, sources=sources, project_root=self.tmp)
        self.assertIn("different source hash", str(ctx.exception))
        with self.assertRaises(VideoContractError):
            bd.load_annotation(path, project_root=self.tmp)


class ProposalVersusLabelTests(DatasetCase):
    def annotation(self, sources: dict, clip_id: str) -> dict:
        clip = bd.sources_by_id(sources)[clip_id]
        return bd.new_annotation(clip_id=clip_id, video=clip["media_path"],
                                 source_sha256=clip["sha256"], width=clip["width"],
                                 height=clip["height"], frames_total=clip["frames"],
                                 annotator="fixture")

    def test_a_proposal_may_not_be_recorded_as_a_frame_label(self):
        sources, records = self.fixture(1)
        payload = self.annotation(sources, list(records)[0])
        payload["frames"]["2"] = {"state": bd.STATE_VISIBLE,
                                  "origin": bd.PROPOSAL_ORIGIN, "x": 5.0, "y": 5.0}
        with self.assertRaises(VideoContractError) as ctx:
            bd.validate_annotation(payload)
        self.assertIn("proposals belong in the proposals namespace",
                      str(ctx.exception))

    def test_a_pending_proposal_does_not_become_a_label(self):
        sources, records = self.fixture(1)
        payload = self.annotation(sources, list(records)[0])
        payload = bd.set_proposal(payload, 2, x=5.0, y=5.0, method="carry_over")
        self.assertNotIn("2", payload["frames"])
        self.assertEqual(payload["proposals"]["2"]["status"], "pending")
        self.assertEqual(payload["proposals"]["2"]["origin"], bd.PROPOSAL_ORIGIN)

    def test_a_pending_proposal_cannot_sit_on_an_existing_label(self):
        sources, records = self.fixture(1)
        payload = self.annotation(sources, list(records)[0])
        payload = bd.set_label(payload, 2, state=bd.STATE_VISIBLE,
                               origin=bd.ORIGIN_AGENT_REVIEWED, x=5.0, y=5.0)
        payload["proposals"]["2"] = {"origin": bd.PROPOSAL_ORIGIN, "status": "pending",
                                     "method": "flow", "x": 6.0, "y": 6.0}
        with self.assertRaises(VideoContractError) as ctx:
            bd.validate_annotation(payload)
        self.assertIn("a proposal is not a label", str(ctx.exception))

    def test_accepting_a_proposal_still_requires_an_explicit_label(self):
        sources, records = self.fixture(1)
        clip_id = list(records)[0]
        payload = self.annotation(sources, clip_id)
        payload = bd.set_proposal(payload, 2, x=5.0, y=5.0, method="flow")
        resolved = bd.resolve_proposal(payload, 2, status="accepted")
        self.assertNotIn("2", resolved["frames"],
                         "accepting a proposal must not create a label by itself")
        self.assertEqual(resolved["proposals"]["2"]["status"], "accepted")

    def test_a_rejected_proposal_is_kept_as_negative_evidence(self):
        sources, records = self.fixture(1)
        payload = self.annotation(sources, list(records)[0])
        payload = bd.set_proposal(payload, 2, x=5.0, y=5.0, method="flow")
        rejected = bd.resolve_proposal(payload, 2, status="rejected")
        self.assertEqual(rejected["proposals"]["2"]["status"], "rejected")
        self.assertEqual(rejected["proposals"]["2"]["x"], 5.0)


class SaveResumeTests(DatasetCase):
    def test_a_label_survives_save_and_reload(self):
        sources, records = self.fixture(1)
        clip_id = list(records)[0]
        path = bd.annotation_path_for(self.tmp / "ann", clip_id)
        payload = bd.load_or_new_annotation(annotations_dir=self.tmp / "ann",
                                            sources=sources, clip_id=clip_id,
                                            project_root=self.tmp,
                                            annotator="fixture")
        payload = bd.set_label(payload, 1, state=bd.STATE_VISIBLE,
                               origin=bd.ORIGIN_AGENT_REVIEWED, x=12.0, y=10.0,
                               radius=4.0, tags=["glare"], confidence="medium")
        payload = bd.set_label(payload, 2, state=bd.STATE_NOT_VISIBLE,
                               origin=bd.ORIGIN_AGENT_REVIEWED,
                               reason="club face covers the ball")
        bd.save_annotation(path, payload, sources=sources, project_root=self.tmp)

        resumed = bd.load_or_new_annotation(annotations_dir=self.tmp / "ann",
                                            sources=sources, clip_id=clip_id,
                                            project_root=self.tmp,
                                            annotator="fixture")
        self.assertEqual(len(resumed["frames"]), 2)
        self.assertEqual(resumed["frames"]["1"]["x"], 12.0)
        self.assertEqual(resumed["frames"]["1"]["tags"], ["glare"])
        self.assertEqual(resumed["frames"]["2"]["reason"],
                         "club face covers the ball")

    def test_relabelling_a_frame_replaces_it_and_undo_removes_it(self):
        sources, records = self.fixture(1)
        clip_id = list(records)[0]
        payload = bd.load_or_new_annotation(annotations_dir=self.tmp / "ann",
                                            sources=sources, clip_id=clip_id,
                                            project_root=self.tmp, annotator="fixture")
        payload = bd.set_label(payload, 4, state=bd.STATE_VISIBLE,
                               origin=bd.ORIGIN_AGENT_REVIEWED, x=1.0, y=1.0)
        payload = bd.set_label(payload, 4, state=bd.STATE_OCCLUDED,
                               origin=bd.ORIGIN_AGENT_REVIEWED, reason="turf")
        self.assertEqual(len(payload["frames"]), 1)
        self.assertEqual(payload["frames"]["4"]["state"], bd.STATE_OCCLUDED)
        self.assertNotIn("x", payload["frames"]["4"])
        payload = bd.drop_label(payload, 4)
        self.assertEqual(payload["frames"], {})

    def test_a_visible_label_without_a_coordinate_is_refused_at_write_time(self):
        sources, records = self.fixture(1)
        payload = bd.load_or_new_annotation(annotations_dir=self.tmp / "ann",
                                            sources=sources,
                                            clip_id=list(records)[0],
                                            project_root=self.tmp, annotator="fixture")
        with self.assertRaises(VideoContractError):
            bd.set_label(payload, 4, state=bd.STATE_VISIBLE,
                         origin=bd.ORIGIN_AGENT_REVIEWED, x=3.0)


class SplitTests(DatasetCase):
    def split_payload(self, sources: dict, assignment: dict[str, str]) -> dict:
        clips = []
        for clip in sources["clips"]:
            clips.append({"clip_id": clip["clip_id"],
                          "split": assignment[clip["clip_id"]],
                          "source_group": clip["source_group"],
                          "sha256": clip["sha256"], "legacy_clip": False})
        return {"schema_version": bd.SCHEMA_SPLIT, "dataset_id": "fixture",
                "status": "frozen", "clips": clips}

    def test_the_same_clip_cannot_be_in_two_splits(self):
        sources, records = self.fixture(2)
        ids = sorted(records)
        payload = self.split_payload(sources, {ids[0]: bd.SPLIT_TRAIN,
                                               ids[1]: bd.SPLIT_TRAIN})
        payload["clips"][1]["clip_id"] = ids[0]
        with self.assertRaises(VideoContractError):
            bd.validate_split(payload, sources=sources)

    def test_one_source_hash_cannot_be_in_two_splits(self):
        sources, records = self.fixture(2)
        ids = sorted(records)
        payload = self.split_payload(sources, {ids[0]: bd.SPLIT_TRAIN,
                                               ids[1]: bd.SPLIT_VALIDATION})
        payload["clips"][1]["sha256"] = payload["clips"][0]["sha256"]
        with self.assertRaises(VideoContractError) as ctx:
            bd.validate_split(payload, sources=sources)
        self.assertIn("cannot be in two splits", str(ctx.exception))

    def test_a_shoot_group_may_not_leak_across_splits(self):
        sources, records = self.fixture(2)
        ids = sorted(records)
        for clip in sources["clips"]:
            clip["source_group"] = "pexels_shared"
        payload = self.split_payload(sources, {ids[0]: bd.SPLIT_TRAIN,
                                               ids[1]: bd.SPLIT_FINAL_TEST})
        with self.assertRaises(VideoContractError) as ctx:
            bd.validate_split(payload, sources=sources)
        self.assertIn("may not leak", str(ctx.exception))

    def test_a_consumed_v1_clip_may_not_enter_the_v2_final_test(self):
        sources, records = self.fixture(2)
        ids = sorted(records)
        payload = self.split_payload(sources, {ids[0]: bd.SPLIT_TRAIN,
                                               ids[1]: bd.SPLIT_FINAL_TEST})
        payload["clips"][1]["legacy_clip"] = True
        with self.assertRaises(VideoContractError) as ctx:
            bd.validate_split(payload, sources=sources)
        self.assertIn("consumed v1 clip", str(ctx.exception))

    def test_a_group_shared_with_the_consumed_v1_corpus_may_not_be_in_final_test(self):
        sources, records = self.fixture(2)
        ids = sorted(records)
        sources["clips"][0]["source_group"] = "pexels_1004"
        sources["clips"][1]["source_group"] = "pexels_6573"
        payload = self.split_payload(sources, {ids[0]: bd.SPLIT_TRAIN,
                                               ids[1]: bd.SPLIT_FINAL_TEST})
        with self.assertRaises(VideoContractError) as ctx:
            bd.validate_split(payload, sources=sources, legacy_groups={"pexels_6573"})
        self.assertIn("with the consumed v1 corpus", str(ctx.exception))
        # the same assignment is fine once the group is not a consumed family
        bd.validate_split(payload, sources=sources, legacy_groups={"pexels_9999"})

    def test_an_unfrozen_split_is_refused(self):
        sources, records = self.fixture(1)
        payload = self.split_payload(sources, {sorted(records)[0]: bd.SPLIT_TRAIN})
        payload["status"] = "draft"
        with self.assertRaises(VideoContractError):
            bd.validate_split(payload, sources=sources)

    def test_a_split_clip_must_exist_in_the_sources_manifest(self):
        sources, records = self.fixture(1)
        payload = self.split_payload(sources, {sorted(records)[0]: bd.SPLIT_TRAIN})
        payload["clips"][0]["clip_id"] = "pexels_9999999"
        with self.assertRaises(VideoContractError):
            bd.validate_split(payload, sources=sources)

    def test_the_recorded_hash_must_match_the_manifest(self):
        sources, records = self.fixture(1)
        payload = self.split_payload(sources, {sorted(records)[0]: bd.SPLIT_TRAIN})
        payload["clips"][0]["sha256"] = SHA_B
        with self.assertRaises(VideoContractError):
            bd.validate_split(payload, sources=sources)


class LeakageGuardTests(DatasetCase):
    def build(self) -> tuple[dict, dict, Path]:
        sources, records = self.fixture(2)
        ids = sorted(records)
        for clip in sources["clips"]:
            clip["source_group"] = f"group_{clip['clip_id']}"
        clips = [{"clip_id": ids[0], "split": bd.SPLIT_TRAIN,
                  "source_group": sources["clips"][0]["source_group"],
                  "sha256": sources["clips"][0]["sha256"]},
                 {"clip_id": ids[1], "split": bd.SPLIT_FINAL_TEST,
                  "source_group": sources["clips"][1]["source_group"],
                  "sha256": sources["clips"][1]["sha256"]}]
        split = {"schema_version": bd.SCHEMA_SPLIT, "dataset_id": "fixture",
                 "status": "frozen", "clips": clips}
        ann_dir = self.tmp / "annotations"
        for clip_id in ids:
            clip = bd.sources_by_id(sources)[clip_id]
            payload = bd.new_annotation(clip_id=clip_id, video=clip["media_path"],
                                        source_sha256=clip["sha256"],
                                        width=clip["width"], height=clip["height"],
                                        frames_total=clip["frames"],
                                        annotator="fixture")
            payload = bd.set_label(payload, 1, state=bd.STATE_VISIBLE,
                                   origin=bd.ORIGIN_AGENT_REVIEWED, x=5.0, y=5.0)
            bd.save_annotation(bd.annotation_path_for(ann_dir, clip_id), payload,
                               sources=sources)
        return sources, split, ann_dir

    def test_final_test_cannot_be_listed_without_opting_in(self):
        sources, split, _ = self.build()
        with self.assertRaises(bd.LeakageError):
            bd.clips_for_split(split, bd.SPLIT_FINAL_TEST)
        allowed = bd.clips_for_split(split, bd.SPLIT_FINAL_TEST, allow_final_test=True)
        self.assertEqual(len(allowed), 1)

    def test_final_test_labels_cannot_be_loaded_without_opting_in(self):
        sources, split, ann_dir = self.build()
        with self.assertRaises(bd.LeakageError):
            bd.load_split_labels(split_manifest=split, annotations_dir=ann_dir,
                                 sources=sources, split=bd.SPLIT_FINAL_TEST,
                                 project_root=self.tmp)
        loaded = bd.load_split_labels(split_manifest=split, annotations_dir=ann_dir,
                                      sources=sources, split=bd.SPLIT_FINAL_TEST,
                                      project_root=self.tmp, allow_final_test=True)
        self.assertEqual(len(loaded), 1)

    def test_development_helpers_reach_only_train_and_validation(self):
        sources, split, _ = self.build()
        development = bd.development_clips(split)
        self.assertEqual(len(development), 1)
        self.assertNotIn(bd.clips_for_split(split, bd.SPLIT_FINAL_TEST,
                                            allow_final_test=True)[0], development)

    def test_the_legacy_group_extractor_matches_the_family_scheme(self):
        legacy = self.tmp / "ball_split.json"
        legacy.write_text(json.dumps({"clips": [
            {"clip_id": "pexels_6573644", "source_id": "pexels_6573_yellow_range"}]}))
        groups = bd.legacy_groups_from_split(legacy)
        self.assertIn("pexels_6573", groups)


class StatisticsTests(DatasetCase):
    def build(self) -> tuple[dict, dict, Path]:
        sources, records = self.fixture(2)
        ids = sorted(records)
        for clip in sources["clips"]:
            clip["source_group"] = f"group_{clip['clip_id']}"
            clip["tags"] = ["body"] if clip["clip_id"] == ids[0] else ["glare"]
        clips = [{"clip_id": ids[0], "split": bd.SPLIT_TRAIN,
                  "source_group": sources["clips"][0]["source_group"],
                  "sha256": sources["clips"][0]["sha256"]},
                 {"clip_id": ids[1], "split": bd.SPLIT_VALIDATION,
                  "source_group": sources["clips"][1]["source_group"],
                  "sha256": sources["clips"][1]["sha256"]}]
        split = {"schema_version": bd.SCHEMA_SPLIT, "dataset_id": "fixture",
                 "status": "frozen", "clips": clips}
        ann_dir = self.tmp / "annotations"
        clip = bd.sources_by_id(sources)[ids[0]]
        payload = bd.new_annotation(clip_id=ids[0], video=clip["media_path"],
                                    source_sha256=clip["sha256"], width=clip["width"],
                                    height=clip["height"], frames_total=clip["frames"],
                                    annotator="agent-fixture")
        for index, x in ((1, 10.0), (2, 11.0), (3, 12.0)):
            payload = bd.set_label(payload, index, state=bd.STATE_VISIBLE,
                                   origin=bd.ORIGIN_AGENT_REVIEWED, x=x, y=10.0,
                                   radius=3.0, tags=["body"])
        payload = bd.set_label(payload, 4, state=bd.STATE_NOT_VISIBLE,
                               origin=bd.ORIGIN_AGENT_REVIEWED, reason="left frame")
        payload = bd.set_label(payload, 5, state=bd.STATE_AMBIGUOUS,
                               origin=bd.ORIGIN_AGENT_REVIEWED,
                               explanation="partially hidden behind the club")
        bd.save_annotation(bd.annotation_path_for(ann_dir, ids[0]), payload,
                           sources=sources)
        return sources, split, ann_dir

    def test_statistics_count_states_runs_and_provenance(self):
        sources, split, ann_dir = self.build()
        report = bd.dataset_statistics(sources=sources, split=split,
                                       annotation_paths=sorted(ann_dir.glob("*.json")),
                                       project_root=self.tmp, bind=False)
        train = report[bd.SPLIT_TRAIN]
        self.assertEqual(train["clips"], 1)
        self.assertEqual(train["annotated_clips"], 1)
        self.assertEqual(train["labelled_frames"], 5)
        self.assertEqual(train["states"][bd.STATE_VISIBLE], 3)
        self.assertEqual(train["states"][bd.STATE_NOT_VISIBLE], 1)
        self.assertEqual(train["states"][bd.STATE_AMBIGUOUS], 1)
        self.assertEqual(train["origin"], {bd.ORIGIN_AGENT_REVIEWED: 5})
        self.assertEqual(train["longest_visible_run"], 3)
        self.assertEqual(train["visible_runs"], 1)
        self.assertEqual(train["tags"]["body"], 3)
        self.assertEqual(train["tags"]["clip:body"], 1)
        self.assertEqual(train["ball_diameter_px"]["median"], 6.0)
        self.assertEqual(report[bd.SPLIT_VALIDATION]["annotated_clips"], 0)
        self.assertNotIn("final_test", [k for k, v in report.items()
                                        if v["labelled_frames"]])

    def test_isolated_visible_frames_are_counted_not_averaged_away(self):
        sources, records = self.fixture(1)
        clip_id = sorted(records)[0]
        clip = bd.sources_by_id(sources)[clip_id]
        payload = bd.new_annotation(clip_id=clip_id, video=clip["media_path"],
                                    source_sha256=clip["sha256"], width=clip["width"],
                                    height=clip["height"], frames_total=clip["frames"],
                                    annotator="fixture")
        payload = bd.set_label(payload, 1, state=bd.STATE_VISIBLE,
                               origin=bd.ORIGIN_AGENT_REVIEWED, x=5.0, y=5.0)
        payload = bd.set_label(payload, 3, state=bd.STATE_VISIBLE,
                               origin=bd.ORIGIN_AGENT_REVIEWED, x=5.0, y=5.0)
        summary = bd.annotation_summary(payload)
        self.assertEqual(summary["isolated_visible"], 2)


class ReviewFlagTests(DatasetCase):
    def build(self) -> tuple[dict, dict]:
        sources, records = self.fixture(1)
        clip_id = sorted(records)[0]
        clip = bd.sources_by_id(sources)[clip_id]
        payload = bd.new_annotation(clip_id=clip_id, video=clip["media_path"],
                                    source_sha256=clip["sha256"], width=clip["width"],
                                    height=clip["height"], frames_total=clip["frames"],
                                    annotator="fixture")
        payload = bd.set_label(payload, 1, state=bd.STATE_VISIBLE,
                               origin=bd.ORIGIN_AGENT_REVIEWED, x=1.0, y=1.0)
        payload = bd.set_label(payload, 2, state=bd.STATE_VISIBLE,
                               origin=bd.ORIGIN_AGENT_REVIEWED, x=60.0, y=1.0)
        payload = bd.set_label(payload, 3, state=bd.STATE_NOT_VISIBLE,
                               origin=bd.ORIGIN_AGENT_REVIEWED,
                               reason="ball out of frame")
        payload = bd.set_label(payload, 4, state=bd.STATE_VISIBLE,
                               origin=bd.ORIGIN_AGENT_REVIEWED, x=5.0, y=5.0)
        payload = bd.set_label(payload, 5, state=bd.STATE_NOT_VISIBLE,
                               origin=bd.ORIGIN_AGENT_REVIEWED,
                               reason="ball out of frame")
        payload = bd.set_label(payload, 6, state=bd.STATE_AMBIGUOUS,
                               origin=bd.ORIGIN_AGENT_REVIEWED,
                               explanation="bright grass")
        payload = bd.set_proposal(payload, 7, x=5.0, y=5.0, method="carry_over")
        return sources, payload

    def test_flags_are_raised_and_nothing_is_rewritten(self):
        sources, payload = self.build()
        before = json.dumps(payload, sort_keys=True)
        flags = bd.review_flags(payload, project_root=self.tmp)
        kinds = {flag["kind"] for flag in flags}
        self.assertIn("large_jump", kinds)
        self.assertIn("ambiguous", kinds)
        self.assertIn("pending_proposal", kinds)
        self.assertEqual(before, json.dumps(payload, sort_keys=True),
                         "review must not modify the labels")

    def test_an_isolated_visible_frame_is_flagged(self):
        sources, payload = self.build()
        flags = bd.review_flags(payload, project_root=self.tmp)
        self.assertIn("isolated_visible", {flag["kind"] for flag in flags})


class MediaStatusTests(DatasetCase):
    def test_present_missing_and_mismatched_media_are_reported(self):
        sources, records = self.fixture(2)
        ids = sorted(records)
        for clip in sources["clips"]:
            clip["source_group"] = f"group_{clip['clip_id']}"
        clips = [{"clip_id": ids[0], "split": bd.SPLIT_TRAIN,
                  "source_group": sources["clips"][0]["source_group"],
                  "sha256": sources["clips"][0]["sha256"]},
                 {"clip_id": ids[1], "split": bd.SPLIT_VALIDATION,
                  "source_group": sources["clips"][1]["source_group"],
                  "sha256": sources["clips"][1]["sha256"]}]
        split = {"schema_version": bd.SCHEMA_SPLIT, "dataset_id": "fixture",
                 "status": "frozen", "clips": clips}
        # remove one file, and corrupt-fill the other with different bytes
        (self.tmp / sources["clips"][0]["media_path"]).unlink()
        (self.tmp / sources["clips"][1]["media_path"]).write_bytes(b"not the clip")
        result = bd.media_status(sources=sources, split=split, project_root=self.tmp)
        self.assertEqual(result[bd.SPLIT_TRAIN]["present"], 0)
        self.assertEqual(result[bd.SPLIT_TRAIN]["missing"][0]["clip_id"], ids[0])
        self.assertIn("expected_sha256", result[bd.SPLIT_TRAIN]["missing"][0])
        self.assertEqual(result[bd.SPLIT_VALIDATION]["present"], 0)
        self.assertIn("different hash",
                      result[bd.SPLIT_VALIDATION]["missing"][0]["reason"])


class CanonicalIsolationTests(unittest.TestCase):
    def test_canonical_ball_is_still_unavailable(self):
        layer = canonical.ball_layer()
        self.assertEqual(layer.state, canonical.LayerState.UNAVAILABLE)
        self.assertEqual(layer.method, "none")
        self.assertEqual(layer.coverage["frames_with_observation"], 0)

    def test_the_dataset_module_is_not_imported_by_the_canonical_path(self):
        import ast
        tree = ast.parse((ROOT / "ghostcaddie" / "video" / "canonical.py")
                         .read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        self.assertFalse({name for name in imported if "ball_dataset" in name},
                         "the canonical path must not import the v2 dataset tooling")


class ShippedDatasetTests(unittest.TestCase):
    """The committed manifest, split and annotations must satisfy the contract."""

    def test_the_committed_manifest_validates(self):
        path = ROOT / "data" / "ball_dataset_v2" / "sources.json"
        if not path.exists():
            self.skipTest("dataset manifest not built yet")
        manifest = bd.load_sources(path)
        self.assertGreater(len(manifest["clips"]), 0)
        for clip in manifest["clips"]:
            self.assertTrue(clip["license"])
            self.assertTrue(clip["license_url"])
            self.assertTrue(clip["media_path"].startswith("out/"))

    def test_the_committed_split_is_frozen_and_leak_free(self):
        path = ROOT / "data" / "ball_dataset_v2" / "split.json"
        if not path.exists():
            self.skipTest("split not frozen yet")
        sources = bd.load_sources(ROOT / "data" / "ball_dataset_v2" / "sources.json")
        split = bd.load_split(path, sources=sources,
                              legacy_groups=bd.legacy_groups_from_split(
                                  ROOT / "data" / "ball_split.json"))
        self.assertEqual(split["status"], "frozen")
        group_splits: dict[str, set] = {}
        for clip in split["clips"]:
            group_splits.setdefault(clip["source_group"], set()).add(clip["split"])
        for group, splits in group_splits.items():
            self.assertEqual(len(splits), 1,
                             f"shoot group {group} appears in {sorted(splits)}; a "
                             "shared golfer, wardrobe or location may not leak "
                             "across splits")
        self.assertTrue(any(clip["split"] == bd.SPLIT_FINAL_TEST
                            for clip in split["clips"]))
        legacy = bd.legacy_groups_from_split(ROOT / "data" / "ball_split.json")
        for clip in split["clips"]:
            if clip["split"] == bd.SPLIT_FINAL_TEST:
                self.assertNotIn(clip["source_group"], legacy)

    def test_the_committed_annotations_validate_against_the_manifest(self):
        ann_dir = ROOT / "data" / "ball_dataset_v2" / "annotations"
        if not ann_dir.exists():
            self.skipTest("no v2 annotations yet")
        sources = bd.load_sources(ROOT / "data" / "ball_dataset_v2" / "sources.json")
        paths = sorted(ann_dir.glob("*.json"))
        self.assertTrue(paths)
        for path in paths:
            annotation = bd.load_annotation(path, sources=sources, bind=False)
            for key, entry in annotation["frames"].items():
                if entry["state"] == bd.STATE_VISIBLE:
                    self.assertIn("x", entry)
                    self.assertIn("y", entry)
                else:
                    self.assertNotIn("x", entry)
                    self.assertNotIn("y", entry)


if __name__ == "__main__":
    unittest.main()
