"""ClubheadDB dataset assembler.

Builds a YOLO-format clubhead-detection dataset from three explicit,
operator-supplied inputs and nothing else:

* ``plan``        - the clip roster the primary CLI has already approved,
                    including the SHA-256 it bound to the author annotation
                    file (``annotations_json_sha256``).
* ``annotations`` - the parsed author annotation rows, verbatim.
* ``receipt``     - the acquisition receipt: the declared SHA-256 of the source
                    media and of every downloaded frame, plus native geometry.

``assemble(plan, annotations, receipt, destination) -> manifest``

The rules this module enforces, all of them stated failures rather than guesses:

* Nothing is interpolated, inferred, smoothed or median-filtered. There are no
  "future" or "target" label fields and no synthetic negatives. A label is
  written only where the author wrote a row; a row with a null class id and
  null box is the author's *explicit negative*, and is the only way an empty
  label is ever produced.
* A frame with no author row is UNANNOTATED, which is not a negative and is
  never written. If the author rows do not cover exactly ``1..N`` of the
  clip's frames, the whole clip is rejected (with a recorded reason), not
  silently negative-labelled.
* Media provenance is checked against the receipt before anything is written:
  the source file bytes, every frame's bytes, native width/height and the
  contiguous ``frame_0001.jpg`` index. The source and every frame hash must be
  a *declared* lower-case 64-hex SHA-256 - a missing or malformed hash is an
  ``incomplete_receipt`` and a mismatch rejects the whole clip.
* Every frame is decoded, not merely header-inspected: ``_image_size`` forces a
  full Pillow ``load()`` (or an OpenCV ``imdecode``) so a truncated or corrupt
  body is ``image_unreadable``. A missing decoder is a fatal
  ``ClubheadDataError`` raised before any output exists, never a silent
  per-clip rejection.
* ``plan.annotations_json_sha256`` is verified as the exact SHA-256 of the
  canonical author-annotation payload
  (``json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=True)``)
  before ``destination`` is touched; a mismatch is fatal.
* Every clip's receipt must carry a mandatory ``alignment_review`` record - an
  operator visual-review assertion that the clip/image alignment was checked by
  eye. Its state must be ``accepted`` and it must bind, before the clip is
  accepted: the clip's actual ``source_sha256``, the canonical hash of its
  ordered author rows (``annotations_sha256``) and the canonical hash of its
  ordered ``[[frame_index, frame_sha256], ...]`` receipt list
  (``frames_sha256``). A missing/pending record is ``alignment_unreviewed``, an
  explicit rejection is ``alignment_rejected`` and any mismatched or missing
  binding is ``alignment_binding_mismatch``. Review coverage is the declared
  sampled frames only and may be partial - this is a declared operator
  assertion, not automatic proof, not an independent human annotation and not
  a claim of human verification.
* Cross-split leakage is fatal, normalised before the check: golfer identities
  are casefolded with runs of whitespace/hyphen/underscore collapsed and a
  trailing numeric suffix dropped for pros (amateur roster ids stay distinct),
  and public media identities collapse the YouTube raw-id / ``watch?v=`` /
  ``shorts/`` / ``youtu.be`` forms and the Reddit ``old.``/``www.`` host, query
  and fragment variants. Malformed annotation rows, duplicate author rows and
  plan defects are likewise fatal and raised *before* any output exists.
* After every accepted payload is collected and still before any write, an
  accepted source or image whose exact SHA-256 is already accepted in a
  different split is fatal (equal frames repeating within one split are
  allowed). This is exact byte overlap only - no near-duplicate, perceptual or
  repost identification is performed or claimed.
* The whole dataset is staged in a temporary sibling directory and published
  with a single atomic ``os.replace``. ``dataset.yaml`` and the manifest
  ``destination`` name the *final* destination, never the temporary path. On
  any failure the staging directory is removed and inputs and any existing
  (empty) destination are left untouched, so a failed run never leaves a
  partial destination. ``destination`` must be fresh; source media is only ever
  read.

The produced manifest records source/image/annotation provenance hashes, the
accepted and rejected clip ids with reason counts, and explicit non-claims
(this is author-declared ground truth, not an independently confirmed
measurement, and it is neither a statistical held-out split nor a production
release).

Only the standard library plus a lazily imported JPEG decoder (Pillow, or
OpenCV) are required. No network access, no training, no auto-download.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlsplit

from .errors import DemoConfigError

SCHEMA = "clubheaddb.assemble.manifest.v1"

CLASS_NAME = "clubhead"
CLASS_ID = 0
SPLITS = ("train", "val", "test")
SOURCES = ("yt", "yt_shorts", "reddit")

VIDEO_ID_RE = re.compile(r"^swing_\d{3}$")
FRAME_PATH_RE = re.compile(r"^(swing_\d{3})/images/frame_(\d{4})\.jpg$")
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")

#: Runs of whitespace, hyphen or underscore collapse to a single space.
_SEPARATOR_RE = re.compile(r"[\s_\-]+")
#: URL host prefixes that do not change a public media identity.
_HOST_PREFIX_RE = re.compile(r"^(?:www|m|old|new|np|mobile)\.")

#: Author-row keys that must be all-``None`` together for an explicit negative.
_BOX_KEYS = ("x_center", "y_center", "width", "height")

#: Non-fatal per-clip exclusion reasons. Anything else is fatal.
_REASON_ORDER = (
    "missing_receipt",
    "failed_acquisition",
    "incomplete_receipt",
    "receipt_frame_index_invalid",
    "receipt_duplicate_frame",
    "receipt_frame_path_invalid",
    "receipt_geometry_invalid",
    "receipt_frame_gap",
    "unannotated",
    "annotation_frame_mismatch",
    "missing_source_media",
    "missing_image_media",
    "source_sha_mismatch",
    "image_sha_mismatch",
    "image_unreadable",
    "image_geometry_mismatch",
    "geometry_inconsistent",
    "alignment_unreviewed",
    "alignment_rejected",
    "alignment_binding_mismatch",
)


class ClubheadDataError(DemoConfigError):
    """A fatal input defect. Raised before any output is written."""


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, default=str)


def _annotations_canonical(obj: Any) -> str:
    """Canonical JSON for hashing annotation payloads.

    Deliberately exact: ``sort_keys=True``, tight separators and
    ``ensure_ascii=True``, with no ``default=`` fallback, so the digest of the
    author rows can be independently reproduced by whoever bound
    ``plan.annotations_json_sha256``.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _annotations_sha256(obj: Any) -> str:
    try:
        text = _annotations_canonical(obj)
    except (TypeError, ValueError) as exc:
        raise ClubheadDataError(
            "annotations are not canonically JSON-serializable"
        ) from exc
    return _sha256_bytes(text.encode("utf-8"))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_zero_class(value: Any) -> bool:
    return _is_number(value) and float(value) == 0.0


def golfer_identity(note: str) -> str:
    """Normalized golfer identity shared with the evaluation tools.

    * casefolded, with runs of whitespace/hyphen/underscore collapsed to one
      space and surrounding space stripped, so ``Pro_001``, ``pro-001`` and
      ``PRO 001`` all share one identity;
    * a trailing numeric suffix is dropped for a pro (``pro 001`` -> ``pro``)
      so one pro can never straddle two splits;
    * an amateur roster id is kept verbatim (``amateur 001`` stays distinct
      from ``amateur 002``), so distinct amateur clips are not collapsed.
    """
    text = _SEPARATOR_RE.sub(" ", note.strip()).casefold()
    if not text:
        return text
    if text.startswith("amateur"):
        return text
    match = re.match(r"^(?P<base>.*?)\s*\d+$", text)
    if match and match.group("base").strip():
        return match.group("base").strip()
    return text


def _url_host(text: str) -> str:
    parts = urlsplit(text if "://" in text else "//" + text)
    host = (parts.hostname or "").casefold()
    return _HOST_PREFIX_RE.sub("", host)


def public_identity(url: str, source: Optional[str] = None) -> str:
    """Normalized public-media identity shared with the evaluation tools.

    YouTube: a raw video id, a ``watch?v=…`` URL, a ``shorts/…`` URL and a
    ``youtu.be/…`` URL collapse to one ``yt:<id>`` identity, and irrelevant
    query parameters are ignored. Reddit: ``old.``/``www.`` host prefixes, the
    query string and the fragment are ignored, leaving ``reddit:<host><path>``.
    Anything else falls back to a lower-cased ``host/path`` with the query and
    fragment dropped. Query parameters that carry identity (YouTube's ``v``)
    are the only ones consulted.
    """
    text = url.strip() if isinstance(url, str) else ""
    if not text:
        return ""

    if source == "reddit" or _url_host(text).endswith("reddit.com"):
        parts = urlsplit(text if "://" in text else "//" + text)
        host = _HOST_PREFIX_RE.sub("", (parts.hostname or "reddit.com").casefold())
        return "reddit:" + host + parts.path.rstrip("/").casefold()

    if "://" not in text and "/" not in text:
        return "yt:" + text

    parts = urlsplit(text if "://" in text else "//" + text)
    host = _HOST_PREFIX_RE.sub("", (parts.hostname or "").casefold())
    path = parts.path
    if host == "youtu.be":
        video = path.strip("/").split("/")[0]
    elif host.endswith("youtube.com") and path.startswith("/watch"):
        video = (parse_qs(parts.query).get("v") or [""])[0]
    elif host.endswith("youtube.com") and path.startswith("/shorts/"):
        video = path[len("/shorts/"):].strip("/").split("/")[0]
    elif host.endswith("youtube.com") and path.startswith("/embed/"):
        video = path[len("/embed/"):].strip("/").split("/")[0]
    else:
        video = text
    return "yt:" + video


def _image_size(data: bytes) -> Tuple[int, int]:
    """Native (width, height) of a JPEG byte string, via Pillow or OpenCV.

    Pillow's ``load()`` is forced so the *body* is decoded, not just the header
    parsed: a truncated or corrupt JPEG raises here rather than passing as a
    valid image. If no decoder is importable a fatal ``ClubheadDataError`` is
    raised, so an unusable environment is never mistaken for an undecodable
    clip.
    """
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        Image = None  # type: ignore
    if Image is not None:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            image.load()
        return int(width), int(height)

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ClubheadDataError(
            "no JPEG decoder available: install Pillow or opencv-python"
        ) from exc
    buffer = np.frombuffer(data, dtype=np.uint8)
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("undecodable JPEG")
    height, width = decoded.shape[:2]
    return int(width), int(height)


# --------------------------------------------------------------------------- #
# validation (all fatal checks happen before any write)
# --------------------------------------------------------------------------- #

def _validate_plan(plan: Any) -> Tuple[List[dict], Dict[str, dict], str]:
    if not isinstance(plan, dict):
        raise ClubheadDataError("plan must be a mapping")
    clips = plan.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ClubheadDataError("plan.clips must be a non-empty list")
    bound = plan.get("annotations_json_sha256")
    if not isinstance(bound, str) or not HEX_SHA256_RE.match(bound):
        raise ClubheadDataError(
            "plan.annotations_json_sha256 must be a bound sha256 hex digest"
        )

    ordered: List[dict] = []
    by_id: Dict[str, dict] = {}
    seen_urls: Dict[str, str] = {}
    golfer_split: Dict[str, str] = {}

    for clip in clips:
        if not isinstance(clip, dict):
            raise ClubheadDataError("every plan clip must be a mapping")
        video_id = clip.get("video_id")
        split = clip.get("split")
        url = clip.get("youtube_id")
        notes = clip.get("notes")
        source = clip.get("source")

        if not isinstance(video_id, str) or not VIDEO_ID_RE.match(video_id):
            raise ClubheadDataError(
                f"plan clip video_id must match swing_###: {video_id!r}"
            )
        if video_id in by_id:
            raise ClubheadDataError(f"duplicate plan video_id: {video_id}")
        if split not in SPLITS:
            raise ClubheadDataError(
                f"{video_id}: split must be one of {SPLITS}, got {split!r}"
            )
        if not isinstance(url, str) or not url.strip():
            raise ClubheadDataError(f"{video_id}: youtube_id must be non-empty")
        if not isinstance(notes, str) or not notes.strip():
            raise ClubheadDataError(f"{video_id}: notes must declare a golfer")
        if source not in SOURCES:
            raise ClubheadDataError(
                f"{video_id}: source must be one of {SOURCES}, got {source!r}"
            )

        # Normalized public-media identity: raw id / watch / shorts / youtu.be
        # forms and Reddit host/query/fragment variants collapse together, so a
        # reused source cannot slip across splits behind cosmetic URL spelling.
        identity_url = public_identity(url, source)
        prior_url = seen_urls.get(identity_url)
        if prior_url is not None:
            raise ClubheadDataError(
                f"public media identity {identity_url!r} reused by {prior_url} "
                f"and {video_id} (raw URLs {url!r}); URLs must be disjoint "
                "across splits"
            )
        seen_urls[identity_url] = video_id

        # Normalized golfer identity: separator/case variants and pro numeric
        # suffixes collapse, so a golfer can never straddle two splits. This is
        # fatal and raised here, before any output exists.
        identity = golfer_identity(notes)
        prior = golfer_split.get(identity)
        if prior is not None and prior != split:
            raise ClubheadDataError(
                f"golfer identity {identity!r} spans splits {prior} and {split}; "
                "refusing leakage across splits"
            )
        golfer_split[identity] = split

        by_id[video_id] = clip
        ordered.append(clip)

    return ordered, by_id, bound


def _reject_path(path: str, video_id: str) -> re.Match:
    if not path:
        raise ClubheadDataError(f"{video_id}: empty image_path")
    if "\\" in path:
        raise ClubheadDataError(
            f"{video_id}: backslashes are not allowed in image_path: {path!r}"
        )
    if path.startswith("/") or _WINDOWS_DRIVE_RE.match(path):
        raise ClubheadDataError(
            f"{video_id}: absolute image_path rejected: {path!r}"
        )
    segments = path.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        raise ClubheadDataError(
            f"{video_id}: traversal/empty segment in image_path: {path!r}"
        )
    match = FRAME_PATH_RE.match(path)
    if match is None:
        raise ClubheadDataError(
            f"{video_id}: image_path must be swing_###/images/frame_####.jpg: "
            f"{path!r}"
        )
    return match


def _check_box(video_id: str, frame_index: int, values: Sequence[Any]) -> None:
    for value in values:
        if not _is_number(value) or not math.isfinite(float(value)):
            raise ClubheadDataError(
                f"{video_id}#{frame_index}: bbox values must be finite numbers"
            )
    x_center, y_center, width, height = (float(v) for v in values)
    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
        raise ClubheadDataError(
            f"{video_id}#{frame_index}: center must lie within the unit frame"
        )
    if width <= 0.0 or height <= 0.0:
        raise ClubheadDataError(
            f"{video_id}#{frame_index}: width/height must be positive"
        )
    if (x_center - width / 2.0) < 0.0 or (x_center + width / 2.0) > 1.0:
        raise ClubheadDataError(
            f"{video_id}#{frame_index}: box edges fall outside [0,1]; "
            "refusing to clip"
        )
    if (y_center - height / 2.0) < 0.0 or (y_center + height / 2.0) > 1.0:
        raise ClubheadDataError(
            f"{video_id}#{frame_index}: box edges fall outside [0,1]; "
            "refusing to clip"
        )


def _validate_annotations(annotations: Any, by_id: Dict[str, dict]
                          ) -> Dict[str, List[Tuple[int, dict]]]:
    if not isinstance(annotations, list):
        raise ClubheadDataError("annotations must be a list of author rows")

    rows_by_video: Dict[str, List[Tuple[int, dict]]] = {}
    seen: set = set()

    for row in annotations:
        if not isinstance(row, dict):
            raise ClubheadDataError("every annotation row must be a mapping")
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or video_id not in by_id:
            raise ClubheadDataError(
                f"annotation row references unknown video_id: {video_id!r}"
            )
        image_path = row.get("image_path")
        if not isinstance(image_path, str):
            raise ClubheadDataError(f"{video_id}: image_path must be a string")
        match = _reject_path(image_path, video_id)
        if match.group(1) != video_id:
            raise ClubheadDataError(
                f"{video_id}: image_path belongs to {match.group(1)}"
            )
        frame_index = int(match.group(2))
        if frame_index < 1:
            raise ClubheadDataError(
                f"{video_id}: frame index must be >= 1 in {image_path!r}"
            )

        class_id = row.get("class_id")
        values = [row.get(key) for key in _BOX_KEYS]
        if class_id is None:
            if any(value is not None for value in values):
                raise ClubheadDataError(
                    f"{video_id}#{frame_index}: null class_id requires all "
                    "null box fields"
                )
        else:
            if not _is_zero_class(class_id):
                raise ClubheadDataError(
                    f"{video_id}#{frame_index}: only class id 0 is allowed, "
                    f"got {class_id!r}"
                )
            if any(value is None for value in values):
                raise ClubheadDataError(
                    f"{video_id}#{frame_index}: class 0 requires all box fields"
                )
            _check_box(video_id, frame_index, values)

        key = (video_id, frame_index)
        if key in seen:
            raise ClubheadDataError(
                f"duplicate author row for {video_id}#{frame_index}"
            )
        seen.add(key)
        rows_by_video.setdefault(video_id, []).append((frame_index, row))

    for video_id in rows_by_video:
        rows_by_video[video_id].sort(key=lambda item: item[0])
    return rows_by_video


def _assert_no_cross_split_duplicates(accepted: List[dict]) -> None:
    """Fatal cross-split exact-byte overlap check, run before any write.

    An accepted source or image hash may repeat within a single split, but the
    same exact SHA-256 appearing in two different splits means one physical
    artifact straddles splits - fatal. Only exact byte overlap is considered;
    no near-duplicate, perceptual or repost identification is attempted.
    """
    source_split: Dict[str, str] = {}
    frame_split: Dict[str, str] = {}
    for entry in accepted:
        video_id = entry["video_id"]
        split = entry["split"]

        source_sha = entry["source_sha256"]
        prior = source_split.get(source_sha)
        if prior is not None and prior != split:
            raise ClubheadDataError(
                f"source sha256 {source_sha} shared across splits {prior} and "
                f"{split} (video {video_id}); refusing cross-split media leakage"
            )
        source_split.setdefault(source_sha, split)

        for frame in entry["frames"]:
            frame_sha = frame["image_sha256"]
            prior = frame_split.get(frame_sha)
            if prior is not None and prior != split:
                raise ClubheadDataError(
                    f"frame sha256 {frame_sha} shared across splits {prior} and "
                    f"{split} (video {video_id}); refusing exact cross-split "
                    "duplicate frames"
                )
            frame_split.setdefault(frame_sha, split)


# --------------------------------------------------------------------------- #
# per-clip media alignment
# --------------------------------------------------------------------------- #

def _rejected(video_id: str, reason: str) -> Tuple[str, str, Optional[dict]]:
    return "rejected", reason, None


def _label_text(row: dict) -> Tuple[str, str]:
    """Return (label_kind, label_text). Text is the author's values, verbatim."""
    if row.get("class_id") is None:
        return "explicit_negative", ""
    values = [float(row[key]) for key in _BOX_KEYS]
    return "positive", f"{CLASS_ID} " + " ".join(repr(v) for v in values)


def _review_state(receipt_clip: dict, video_id: str
                  ) -> Optional[Tuple[str, str, Optional[dict]]]:
    """Gate on the mandatory operator alignment-review record.

    ``state == "accepted"`` passes; an explicit ``"rejected"`` is
    ``alignment_rejected`` and anything else (missing record, missing/``pending``
    state, unrecognised value) is ``alignment_unreviewed`` - never silently
    treated as reviewed.
    """
    review = receipt_clip.get("alignment_review")
    if not isinstance(review, dict):
        return _rejected(video_id, "alignment_unreviewed")
    state = review.get("state")
    if state == "rejected":
        return _rejected(video_id, "alignment_rejected")
    if state != "accepted":
        return _rejected(video_id, "alignment_unreviewed")
    return None


def _align_clip(video_id: str, rows: Optional[List[Tuple[int, dict]]],
                receipt_clips: Any) -> Tuple[str, str, Optional[dict]]:
    receipt_clip = receipt_clips.get(video_id) if isinstance(receipt_clips, dict) else None
    if receipt_clip is None:
        return _rejected(video_id, "missing_receipt")
    if not isinstance(receipt_clip, dict):
        return _rejected(video_id, "incomplete_receipt")
    if receipt_clip.get("state") != "acquired":
        return _rejected(video_id, "failed_acquisition")

    review_reject = _review_state(receipt_clip, video_id)
    if review_reject is not None:
        return review_reject

    frames = receipt_clip.get("frames")
    if not isinstance(frames, list) or not frames:
        return _rejected(video_id, "incomplete_receipt")
    video_path = receipt_clip.get("video_path")
    if not isinstance(video_path, str) or not video_path:
        return _rejected(video_id, "incomplete_receipt")

    declared_source_sha = receipt_clip.get("source_sha256")
    if (not isinstance(declared_source_sha, str)
            or HEX_SHA256_RE.match(declared_source_sha) is None):
        return _rejected(video_id, "incomplete_receipt")

    by_index: Dict[int, dict] = {}
    for frame in frames:
        if not isinstance(frame, dict):
            return _rejected(video_id, "incomplete_receipt")
        index = frame.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 1:
            return _rejected(video_id, "receipt_frame_index_invalid")
        if index in by_index:
            return _rejected(video_id, "receipt_duplicate_frame")
        path = frame.get("path")
        if not isinstance(path, str) or Path(path).name != f"frame_{index:04d}.jpg":
            return _rejected(video_id, "receipt_frame_path_invalid")
        width = frame.get("width")
        height = frame.get("height")
        if (not isinstance(width, int) or isinstance(width, bool) or width <= 0
                or not isinstance(height, int) or isinstance(height, bool)
                or height <= 0):
            return _rejected(video_id, "receipt_geometry_invalid")
        declared_image_sha = frame.get("sha256")
        if (not isinstance(declared_image_sha, str)
                or HEX_SHA256_RE.match(declared_image_sha) is None):
            return _rejected(video_id, "incomplete_receipt")
        by_index[index] = frame

    frame_count = len(frames)
    if sorted(by_index) != list(range(1, frame_count + 1)):
        return _rejected(video_id, "receipt_frame_gap")

    if rows is None:
        return _rejected(video_id, "unannotated")
    if [index for index, _ in rows] != list(range(1, frame_count + 1)):
        return _rejected(video_id, "annotation_frame_mismatch")

    source_path = Path(video_path)
    if not source_path.is_file():
        return _rejected(video_id, "missing_source_media")
    actual_source_sha = _sha256_file(source_path)
    if declared_source_sha != actual_source_sha:
        return _rejected(video_id, "source_sha_mismatch")

    # Operator alignment-review binding, checked before the clip is accepted.
    # The record must assert this clip's actual source hash, the canonical hash
    # of its ordered author rows (row *dicts*, not the (index, row) pairs held
    # in memory) and the canonical hash of its ordered receipt frame list
    # ``[[index, frame_sha256], ...]`` over the sorted receipt indices.
    rows_sha = _annotations_sha256([row for _, row in rows])
    frames_sha = _annotations_sha256(
        [[index, by_index[index]["sha256"]] for index in sorted(by_index)]
    )
    review = receipt_clip["alignment_review"]
    if (review.get("source_sha256") != actual_source_sha
            or review.get("annotations_sha256") != rows_sha
            or review.get("frames_sha256") != frames_sha):
        return _rejected(video_id, "alignment_binding_mismatch")

    out_frames: List[dict] = []
    geometry: Optional[Tuple[int, int]] = None
    for index, row in rows:
        frame = by_index[index]
        frame_path = Path(frame["path"])
        if not frame_path.is_file():
            return _rejected(video_id, "missing_image_media")
        data = frame_path.read_bytes()
        image_sha = _sha256_bytes(data)
        if frame["sha256"] != image_sha:
            return _rejected(video_id, "image_sha_mismatch")
        try:
            width, height = _image_size(data)
        except ClubheadDataError:
            # An unusable decoding environment is fatal, not a per-clip defect.
            raise
        except Exception:
            return _rejected(video_id, "image_unreadable")
        if (width, height) != (frame["width"], frame["height"]):
            return _rejected(video_id, "image_geometry_mismatch")
        if geometry is None:
            geometry = (width, height)
        elif geometry != (width, height):
            return _rejected(video_id, "geometry_inconsistent")

        kind, text = _label_text(row)
        out_frames.append({
            "index": index,
            "image_name": f"{video_id}__frame_{index:04d}.jpg",
            "label_name": f"{video_id}__frame_{index:04d}.txt",
            "label_kind": kind,
            "label_text": text,
            "image_sha256": image_sha,
            "receipt_image_sha256": frame["sha256"],
            "width": width,
            "height": height,
            "data": data,
        })

    return "accepted", "", {
        "frames": out_frames,
        "source_sha256": actual_source_sha,
        "receipt_source_sha256": declared_source_sha,
        "rows_sha256": rows_sha,
        "frames_sha256": frames_sha,
        "alignment_review": review,
    }


# --------------------------------------------------------------------------- #
# writing (staged, then atomically published)
# --------------------------------------------------------------------------- #

def _ensure_fresh_destination(destination: Path) -> None:
    if destination.exists():
        if destination.is_file():
            raise ClubheadDataError(
                f"destination exists and is a file: {destination}"
            )
        if any(destination.iterdir()):
            raise ClubheadDataError(
                f"destination is not fresh (non-empty): {destination}"
            )


def _staging_dir(destination: Path) -> Path:
    """Create a fresh temporary sibling directory for this assembly."""
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{destination.name}.staging-{uuid.uuid4().hex[:12]}"
    staging.mkdir()
    return staging


def _discard_staging(staging: Path) -> None:
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)


def _publish(staging: Path, destination: Path) -> None:
    """Atomically move the complete staging directory onto ``destination``.

    ``destination`` must be absent or an existing *empty* directory; a
    non-empty destination is refused here as well, so a late race cannot
    overwrite unrelated files. Because the whole tree is renamed in one
    ``os.replace``, a failure anywhere earlier leaves no partial destination.
    """
    if destination.exists() and any(destination.iterdir()):
        raise ClubheadDataError(
            f"refusing to overwrite non-empty destination: {destination}"
        )
    os.replace(staging, destination)


def _write_dataset_yaml(destination: Path) -> str:
    root = destination.resolve().as_posix()
    return (
        "# YOLO dataset descriptor for the ClubheadDB author annotation set.\n"
        "# Absolute root; splits are the standard images/<split> layout.\n"
        "# Nothing is fetched and no training is triggered by this file.\n"
        f"path: {root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"  {CLASS_ID}: {CLASS_NAME}\n"
    )


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def assemble(plan: Any, annotations: Any, receipt: Any,
             destination: Any) -> Dict[str, Any]:
    """Assemble the clubhead dataset and return its manifest.

    All fatal validation runs before ``destination`` is touched. Per-clip
    media-alignment failures are non-fatal and recorded in the manifest. The
    dataset is built in a temporary sibling directory and published with one
    atomic ``os.replace`` naming the *final* destination.
    """
    ordered_clips, by_id, bound = _validate_plan(plan)
    rows_by_video = _validate_annotations(annotations, by_id)

    # The plan binds the *exact* author-annotation payload it approved. Recompute
    # the canonical digest and refuse to proceed on any mismatch, before a
    # single byte is written.
    supplied_annotations_sha = _annotations_sha256(annotations)
    if supplied_annotations_sha != bound:
        raise ClubheadDataError(
            "plan.annotations_json_sha256 does not match the supplied "
            f"annotations: plan={bound} supplied={supplied_annotations_sha}"
        )

    dest = Path(destination).expanduser()
    _ensure_fresh_destination(dest)

    receipt_clips = receipt.get("clips") if isinstance(receipt, dict) else None

    accepted: List[dict] = []
    rejected: List[dict] = []
    unannotated: List[str] = []

    for clip in ordered_clips:
        video_id = clip["video_id"]
        rows = rows_by_video.get(video_id)
        status, reason, payload = _align_clip(video_id, rows, receipt_clips)
        if status == "accepted":
            assert payload is not None
            accepted.append({
                "video_id": video_id,
                "split": clip["split"],
                "source": clip["source"],
                "youtube_id": clip["youtube_id"],
                "notes": clip["notes"],
                "golfer_identity": golfer_identity(clip["notes"]),
                "public_identity": public_identity(clip["youtube_id"], clip["source"]),
                "annotations_sha256": payload["rows_sha256"],
                "frames": payload["frames"],
                "source_sha256": payload["source_sha256"],
                "receipt_source_sha256": payload["receipt_source_sha256"],
                "alignment_review": payload["alignment_review"],
            })
        else:
            rejected.append({
                "video_id": video_id,
                "split": clip["split"],
                "reason": reason,
            })
            if reason == "unannotated":
                unannotated.append(video_id)

    # Fatal exact-byte cross-split overlap check, still before any write and
    # after every accepted payload is known.
    _assert_no_cross_split_duplicates(accepted)

    # ---- write (only after every fatal check has passed) ----------------- #
    # Everything is built in a temporary sibling directory and published with a
    # single os.replace, so inputs and any existing empty destination are never
    # mutated by a failed run.
    staging = _staging_dir(dest)
    try:
        written: Dict[str, int] = {split: 0 for split in SPLITS}
        for entry in accepted:
            video_id = entry["video_id"]
            split = entry["split"]
            image_dir = staging / "images" / split
            label_dir = staging / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            public_frames: List[dict] = []
            for frame in entry["frames"]:
                (image_dir / frame["image_name"]).write_bytes(frame["data"])
                (label_dir / frame["label_name"]).write_text(
                    frame["label_text"], encoding="utf-8"
                )
                written[split] += 1
                public_frames.append({
                    key: value for key, value in frame.items() if key != "data"
                })
            entry["frames"] = public_frames

        # Both name the *final* destination, never the staging path.
        (staging / "dataset.yaml").write_text(
            _write_dataset_yaml(dest), encoding="utf-8"
        )

        rejection_counts = {reason: 0 for reason in _REASON_ORDER}
        for item in rejected:
            rejection_counts[item["reason"]] = (
                rejection_counts.get(item["reason"], 0) + 1
            )

        manifest: Dict[str, Any] = {
            "schema": SCHEMA,
            "destination": dest.resolve().as_posix(),
            "class_names": {str(CLASS_ID): CLASS_NAME},
            "plan_annotations_json_sha256": bound,
            "plan_sha256": _sha256_bytes(_canonical(plan).encode("utf-8")),
            "annotations_rows_sha256": supplied_annotations_sha,
            "splits": {
                split: [e["video_id"] for e in accepted if e["split"] == split]
                for split in SPLITS
            },
            "counts": {
                "clips_planned": len(ordered_clips),
                "accepted": len(accepted),
                "rejected": len(rejected),
                "unannotated": len(unannotated),
                "images_written": sum(written.values()),
            },
            "accepted": accepted,
            "rejected": rejected,
            "rejection_counts": rejection_counts,
            "ground_truth": {
                "source": "author_declared_annotations",
                "independently_confirmed": False,
                "bbox_note": (
                    "box conveys the author-marked clubhead center, not a "
                    "physical exact core"
                ),
            },
            "alignment_review": {
                "kind": "operator_visual_assertion",
                "automatic_proof": False,
                "independently_confirmed": False,
                "human_proof_claimed": False,
                "coverage": "declared_sampled_frames_only",
                "coverage_may_be_partial": True,
                "bindings": [
                    "source_sha256",
                    "annotations_sha256",
                    "frames_sha256",
                ],
                "note": (
                    "each accepted clip carries the receipt's alignment_review "
                    "record: a declared operator assertion that the sampled "
                    "clip / image alignment was visually checked, bound before "
                    "accept to the clip's actual source hash, the canonical hash "
                    "of its ordered author rows and the canonical hash of its "
                    "ordered [[frame_index, frame_sha256], ...] receipt list. "
                    "Review coverage is the declared sampled frames only and may "
                    "be partial; this is not automatic proof, not an independent "
                    "human annotation and not a claim of human verification."
                ),
            },
            "duplicate_content": {
                "check": "exact_byte_overlap_across_splits",
                "scope": "accepted_clips",
                "near_duplicate_identification": False,
                "repost_identification": False,
                "note": (
                    "an accepted source or image whose exact SHA-256 is already "
                    "accepted in a different split is fatal; equal frames "
                    "repeating within one split are allowed. Only exact byte "
                    "overlap is checked - no near-duplicate, perceptual or repost "
                    "identification is performed."
                ),
            },
            "claims": {
                "statistical_heldout": False,
                "production": False,
                "training_performed": False,
                "near_duplicate_identification": False,
                "independent_human_alignment_verification": False,
            },
        }

        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        _publish(staging, dest)
    except BaseException:
        _discard_staging(staging)
        raise
    return manifest
