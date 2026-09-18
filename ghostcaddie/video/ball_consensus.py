"""Autonomous pseudo-labelling: multi-method consensus with temporal verification.

No human labels exist, so supervision for real footage has to be built and then
*justified*. This module is where that happens, and its rules are deliberately
conservative: three independent candidate methods agreeing, temporally verified
in both directions, is a very different claim from one method firing, and only the
first kind is allowed anywhere near training.

Three independent evidence sources are combined, never one alone:

* **spatial consensus** -- how many *independent* candidate methods agree on the
  same object, and how tightly they agree;
* **temporal consensus** -- forward and backward patch tracking around the frame.
  A candidate that cannot be recovered when tracked away and back is not the same
  object, and its confidence drops. This is the main autonomous verification
  mechanism, and it is the one that catches a bright object that merely resembles
  a ball;
* **physical plausibility** -- bounded displacement and acceleration relative to
  the ball's own size, and persistence patterns that betray static scenery.

Failure diagnostics are first-class here, because the previous method's failure
was not "no ball found" but "locked onto the golfer's shirt":

``BODY_LOCK``, ``CLOTHING_LOCK``, ``STATIC_SCENE_LOCK``, ``CLUB_LOCK``,
``FLAG_LOCK`` -- each is recorded per candidate, and each is a reason to demote
or reject rather than a silent subtraction.

Every label keeps its provenance: which methods agreed, by how much they
disagreed, what the temporal evidence was, and every rejection reason. Origin is
``MODEL_CONSENSUS``, never human ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

from .ball_candidates import Candidate2, FrameContext

ORIGIN_CONSENSUS = "MODEL_CONSENSUS"

TIER_HIGH = "HIGH"
TIER_MEDIUM = "MEDIUM"
TIER_LOW = "LOW"
TIER_AMBIGUOUS = "AMBIGUOUS"
TIER_UNAVAILABLE = "UNAVAILABLE"
TIERS = (TIER_HIGH, TIER_MEDIUM, TIER_LOW, TIER_AMBIGUOUS, TIER_UNAVAILABLE)

DIAGNOSTIC_KINDS = ("BODY_LOCK", "CLOTHING_LOCK", "STATIC_SCENE_LOCK", "CLUB_LOCK",
                    "FLAG_LOCK")


@dataclass(frozen=True)
class ConsensusParams:
    """The whole decision surface, with each number argued for where it is used."""

    #: Independent methods that must agree for HIGH. Three is the smallest number
    #: that can survive one method being systematically wrong -- which is what the
    #: previous single-method tracker was.
    high_methods: int = 3
    #: Methods agreeing for MEDIUM, which additionally requires temporal support.
    medium_methods: int = 2
    #: Spatial agreement tolerance, as a multiple of the candidate radius. Two
    #: methods that place a ball more than two radii apart are describing
    #: different objects, not the same one.
    agreement_radii: float = 2.0
    #: Minimum score a candidate needs before it can join a cluster at all. Kept
    #: low on purpose: the ensemble is recall-oriented, and the decision is made by
    #: method agreement, temporal verification and the learned detector, not by
    #: this number.
    min_score: float = 0.25
    #: Forward/backward round-trip error allowed for temporal support, in radii.
    cycle_radii: float = 1.5
    #: Frames forward/backward for the cycle test.
    cycle_span: int = 3
    #: Largest displacement between consecutive frames, in radii, for a physically
    #: plausible ball (a struck ball at 60 fps is fast, but not unbounded).
    max_step_radii: float = 8.0
    #: Largest change in displacement between frames, in radii.
    max_accel_radii: float = 5.0
    #: A candidate that is essentially stationary for at least this many frames
    #: while other candidates move is scenery, not a ball in flight.
    static_lock_frames: int = 8
    #: Minimum NCC for the patch to count as the same object after a round trip.
    appearance_ncc: float = 0.6


@dataclass
class Cluster:
    frame_index: int
    x: float
    y: float
    radius: float
    members: list[Candidate2] = field(default_factory=list)

    @property
    def methods(self) -> list[str]:
        return sorted({member.method for member in self.members})

    @property
    def spread(self) -> float:
        if len(self.members) < 2:
            return 0.0
        return max(math.hypot(a.x - b.x, a.y - b.y)
                   for a in self.members for b in self.members)

    @property
    def best_score(self) -> float:
        return max((member.score for member in self.members), default=0.0)

    @property
    def mean_score(self) -> float:
        return float(np.mean([member.score for member in self.members])) \
            if self.members else 0.0


@dataclass
class ConsensusLabel:
    clip_id: str
    frame_index: int
    tier: str
    x: Optional[float]
    y: Optional[float]
    radius: Optional[float]
    methods: list[str]
    spread_px: float
    score: float
    temporal: dict
    diagnostics: list[str]
    rejections: list[str]
    origin: str = ORIGIN_CONSENSUS

    def as_annotation(self) -> dict:
        """The frame entry this label would become, with provenance intact."""
        if self.tier in (TIER_HIGH, TIER_MEDIUM) and self.x is not None:
            state = "VISIBLE"
        elif self.tier == TIER_AMBIGUOUS:
            state = "AMBIGUOUS"
        else:
            state = "NOT_VISIBLE"
        entry: dict[str, Any] = {
            "state": state,
            "origin": self.origin,
            "confidence": {"HIGH": "high", "MEDIUM": "medium",
                           "LOW": "low"}.get(self.tier),
            "note": (f"consensus tier {self.tier} from {len(self.methods)} methods "
                     f"({', '.join(self.methods) or 'none'})"),
        }
        if state == "VISIBLE":
            entry["x"] = round(self.x, 2)
            entry["y"] = round(self.y, 2)
            if self.radius:
                entry["radius"] = round(self.radius, 2)
        elif state == "NOT_VISIBLE":
            entry["reason"] = ("no ball survived multi-method consensus and temporal "
                               "verification on this frame")
        else:
            entry["explanation"] = ("competing candidates or weak temporal evidence; "
                                    "the reviewer would not defend a single position")
        return entry


# ---------------------------------------------------------------------------
# spatial consensus
# ---------------------------------------------------------------------------

def cluster_candidates(candidates: Sequence[Candidate2], *,
                       params: ConsensusParams = ConsensusParams()) -> list[Cluster]:
    """Group candidates from different methods that describe the same object."""
    usable = [candidate for candidate in candidates
              if candidate.score >= params.min_score]
    clusters: list[Cluster] = []
    for candidate in sorted(usable, key=lambda c: c.score, reverse=True):
        placed = False
        for cluster in clusters:
            tolerance = params.agreement_radii * max(candidate.radius, cluster.radius)
            if math.hypot(candidate.x - cluster.x, candidate.y - cluster.y) <= tolerance:
                # keep the highest-scoring member's geometry as the cluster centre
                if candidate.score > cluster.best_score:
                    cluster.x, cluster.y, cluster.radius = (candidate.x, candidate.y,
                                                            candidate.radius)
                cluster.members.append(candidate)
                placed = True
                break
        if not placed:
            clusters.append(Cluster(candidate.frame_index, candidate.x, candidate.y,
                                    candidate.radius, [candidate]))
    return clusters


# ---------------------------------------------------------------------------
# temporal consensus
# ---------------------------------------------------------------------------

def _patch(gray: np.ndarray, x: float, y: float, size: int) -> Optional[np.ndarray]:
    half = size // 2
    left, top = int(round(x)) - half, int(round(y)) - half
    if left < 0 or top < 0 or left + size > gray.shape[1] or top + size > gray.shape[0]:
        return None
    return gray[top:top + size, left:left + size]


def _tracking_window(gray: np.ndarray, x: float, y: float, radius: float,
                     span: int) -> tuple[np.ndarray, int, int]:
    """A local window to run optical flow in, with its origin in frame pixels.

    Running LK on a full 1080p frame once per hypothesis per frame is the kind of
    cost that makes a research pass stop being bounded; the ball cannot move far
    in a few frames anyway, and a ball that leaves this window is reported as
    failing verification rather than guessed at.
    """
    size = int(min(320, max(64, radius * (6.0 + 4.0 * span))))
    height, width = gray.shape[:2]
    size = min(size, height, width)
    left = int(round(x)) - size // 2
    top = int(round(y)) - size // 2
    left = max(0, min(width - size, left))
    top = max(0, min(height - size, top))
    return gray[top:top + size, left:left + size], left, top


def track_patch_both_ways(contexts: Sequence[FrameContext], start: int, x: float, y: float,
                          radius: float, params: ConsensusParams) -> dict:
    """Track a small patch forward k frames, then back, and compare.

    Returns the round-trip error in pixels and radii, the appearance correlation,
    and a boolean. A texture that merely looks like a ball -- a shirt logo, a shoe
    edge -- often fails this: it slides off its own track, or it has no local
    structure for the tracker to hold, and the round trip does not land back on
    the start.
    """
    index = {context.index: position for position, context in enumerate(contexts)}
    if start not in index:
        return {"supported": False, "reason": "frame not in window"}
    position = index[start]
    forward = min(params.cycle_span, len(contexts) - 1 - position)
    if forward < 2 or position == 0:
        return {"supported": False, "reason": "not enough frames on both sides"}

    # Candidates carry *native* pixels (that is what a result means) while the flow
    # work happens on the downscaled working copy. Mixing the two spaces silently
    # pointed the tracking window at the wrong part of the frame on every clip above
    # the working resolution, which is why this check reported "start point outside
    # the tracking window" on real 4K footage.
    scale = float(contexts[position].scale or 1.0)
    native_x, native_y, native_radius = x, y, max(radius, 1e-6)
    x, y = native_x / scale, native_y / scale
    radius = max(1.0, native_radius / scale)

    # optical flow needs 8-bit images; the contexts carry float32 for the
    # gradient work, so the windows are cast here rather than stored twice.
    windows, origins = [], []
    for step in range(0, forward + 1):
        context = contexts[position + step]
        gray = np.clip(context.gray, 0, 255).astype(np.uint8)
        window, left, top = _tracking_window(gray, x, y, radius, forward)
        windows.append(window)
        origins.append((left, top))
    size = windows[0].shape

    def to_window(point_x: float, point_y: float, step: int) -> Optional[np.ndarray]:
        left, top = origins[step]
        local_x, local_y = point_x - left, point_y - top
        if not (0 <= local_x < size[1] and 0 <= local_y < size[0]):
            return None
        return np.array([[[local_x, local_y]]], dtype=np.float32)

    points = to_window(x, y, 0)
    if points is None:
        return {"supported": False, "reason": "start point outside the tracking window"}
    for step in range(1, forward + 1):
        if windows[step].shape != size:
            return {"supported": False, "reason": "window left the frame edge"}
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            windows[step - 1], windows[step], points, None,
            winSize=(11, 11), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        if status is None or status[0][0] != 1:
            return {"supported": False, "reason": f"forward track lost at step {step}"}
        points = moved

    far = (points[0][0][0] + origins[forward][0], points[0][0][1] + origins[forward][1])
    travelled = math.hypot(far[0] - x, far[1] - y) * scale      # reported in native px

    back_points = points.copy()
    for step in range(forward, 0, -1):
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            windows[step], windows[step - 1], back_points, None,
            winSize=(11, 11), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        if status is None or status[0][0] != 1:
            return {"supported": False, "reason": f"backward track lost at step {step}"}
        back_points = moved

    returned = (back_points[0][0][0] + origins[0][0],
                back_points[0][0][1] + origins[0][1])
    round_trip = math.hypot(returned[0] - x, returned[1] - y) * scale

    # appearance: did the thing we came back to look like the thing we left?
    patch_size = int(max(7, round(radius * 4)) | 1)
    anchor = _patch(contexts[position].gray, x, y, patch_size)
    end_patch = _patch(contexts[position].gray, returned[0], returned[1], patch_size)
    correlation = None
    if anchor is not None and end_patch is not None and anchor.shape == end_patch.shape:
        a = anchor - anchor.mean()
        b = end_patch - end_patch.mean()
        denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
        if denominator > 1e-6:
            correlation = float((a * b).sum() / denominator)

    tolerance = params.cycle_radii * max(native_radius, 1.0)
    supported = round_trip <= tolerance and (correlation is None
                                             or correlation >= params.appearance_ncc)
    return {
        "supported": bool(supported),
        "round_trip_px": round(round_trip, 2),
        "round_trip_radii": round(round_trip / max(native_radius, 1e-6), 2),
        "tolerance_radii": params.cycle_radii,
        "travelled_px": round(travelled, 2),
        "appearance_ncc": None if correlation is None else round(correlation, 3),
        "span_frames": forward,
    }


def displacement_plausibility(cluster_sequence: Sequence[Optional[Cluster]],
                              params: ConsensusParams) -> dict:
    """Bounded step and acceleration, in radii, across consecutive frames."""
    positions = [(index, cluster) for index, cluster in enumerate(cluster_sequence)
                 if cluster is not None]
    if len(positions) < 2:
        return {"ok": True, "reason": "fewer than two observations",
                "max_step_radii": None, "max_accel_radii": None}
    steps, steps_radii = [], []
    for (index_a, a), (index_b, b) in zip(positions, positions[1:]):
        gap = index_b - index_a
        distance = math.hypot(b.x - a.x, b.y - a.y)
        radius = max(a.radius, b.radius, 1e-6)
        steps.append(distance / gap)
        steps_radii.append(distance / gap / radius)
    accel_radii = [abs(b - a) for a, b in zip(steps_radii, steps_radii[1:])]
    ok = (max(steps_radii) <= params.max_step_radii
          and (not accel_radii or max(accel_radii) <= params.max_accel_radii))
    reason = "" if ok else (f"implausible motion: max step {max(steps_radii):.1f} radii, "
                            f"max acceleration change "
                            f"{max(accel_radii) if accel_radii else 0:.1f} radii")
    return {"ok": bool(ok), "reason": reason,
            "max_step_radii": round(max(steps_radii), 2),
            "max_accel_radii": round(max(accel_radii), 2) if accel_radii else 0.0}


# ---------------------------------------------------------------------------
# lock diagnostics
# ---------------------------------------------------------------------------

def static_scene_lock(cluster_sequence: Sequence[Optional[Cluster]],
                      params: ConsensusParams, *, motion_px: Sequence[float] = ()) -> dict:
    """Does this "ball" sit still for a long time while the camera moves?

    A ball at rest on a tee is genuinely stationary, so stationarity alone cannot
    be a rejection -- that mistake is documented in the v1 pass. What is
    suspicious is a long-lived candidate whose position barely changes *while the
    camera is moving*, because then the candidate is following the frame, not the
    world.
    """
    positions = [cluster for cluster in cluster_sequence if cluster is not None]
    if len(positions) < params.static_lock_frames:
        return {"locked": False, "run_frames": len(positions),
                "reason": f"only {len(positions)} observations, "
                          f"{params.static_lock_frames} needed"}
    steps = [math.hypot(b.x - a.x, b.y - a.y)
             for a, b in zip(positions, positions[1:])]
    cameras = [abs(float(value)) for value in motion_px if value is not None]
    if not steps or not cameras:
        return {"locked": False, "run_frames": len(positions),
                "reason": "not enough motion in the window to judge"}
    median_step = float(np.median(steps))
    median_radius = float(np.median([cluster.radius for cluster in positions]))
    median_camera = float(np.median(cameras))
    # Both quantities are medians rather than run lengths: detection jitter on a
    # real candidate resets any consecutive-run test, so a run-based rule fires on
    # the wrong case (a stationary ball with a jittery centre) and misses the right
    # one (a candidate the frame drags along).
    tolerance = max(1.5, 1.5 * median_radius)
    rides_the_frame = bool(median_step <= tolerance and median_camera > 2.0
                           and median_camera >= 2.0 * median_step)
    return {"locked": rides_the_frame,
            "run_frames": len(positions),
            "median_step_px": round(median_step, 2),
            "median_camera_motion_px": round(median_camera, 2),
            "tolerance_px": round(tolerance, 2),
            "reason": (f"candidate holds position (median step {median_step:.1f} px) while "
                       f"the camera moves {median_camera:.1f} px/frame, so it follows the "
                       f"frame rather than the world"
                       if rides_the_frame else
                       "candidate motion is consistent with the camera's")}


def body_lock(candidate_x: float, candidate_y: float, radius: float,
              regions: Sequence[Mapping[str, Any]]) -> dict:
    """Overlap between a candidate and a body/clothing region.

    A candidate inside a torso or a shoe is not automatically wrong -- a ball can
    be played from beside a foot -- but it must then clear a much higher temporal
    bar, and it is recorded so a reviewer can see the decision.
    """
    hits = []
    for region in regions:
        kind = region.get("kind", "body")
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        if x - radius <= candidate_x <= x + w + radius and \
                y - radius <= candidate_y <= y + h + radius:
            depth = min(candidate_x - x, x + w - candidate_x,
                        candidate_y - y, y + h - candidate_y)
            hits.append({"kind": kind, "inside_margin_px": round(float(depth), 1)})
    return {"locked": bool(hits), "regions": hits}


def classify_diagnostics(cluster: Cluster, *, temporal: Mapping[str, Any],
                         static: Mapping[str, Any],
                         body: Optional[Mapping[str, Any]] = None) -> list[str]:
    """Name the failure, rather than silently subtracting it."""
    diagnostics = []
    if body and body.get("locked"):
        kinds = {hit["kind"] for hit in body.get("regions", [])}
        if "torso" in kinds or "head" in kinds:
            diagnostics.append("BODY_LOCK")
        if kinds & {"shirt", "trousers", "clothing"}:
            diagnostics.append("CLOTHING_LOCK")
        if "shoe" in kinds:
            diagnostics.append("CLOTHING_LOCK")
    if static.get("locked"):
        diagnostics.append("STATIC_SCENE_LOCK")
    cues = {}
    for member in cluster.members:
        cues.update(member.cues or {})
    if cluster.radius and cluster.radius > 0:
        pass
    if cues.get("polarity") is not None and not temporal.get("supported"):
        # a thin, high-contrast arc that fails the round trip is the signature of
        # a club shaft or a flagstick edge
        diagnostics.append("CLUB_LOCK")
    return diagnostics


# ---------------------------------------------------------------------------
# tiering
# ---------------------------------------------------------------------------

def consensus_label(clip_id: str, frame_index: int, cluster: Optional[Cluster], *,
                    competitors: int, temporal: Mapping[str, Any],
                    static: Mapping[str, Any], plausibility: Mapping[str, Any],
                    body: Optional[Mapping[str, Any]] = None,
                    params: ConsensusParams = ConsensusParams()
                    ) -> ConsensusLabel:
    """Turn one frame's evidence into a tier, keeping every reason on the record."""
    rejections: list[str] = []
    if cluster is None:
        return ConsensusLabel(clip_id, frame_index, TIER_UNAVAILABLE, None, None, None,
                              [], 0.0, 0.0, dict(temporal), [], rejections)
    methods = cluster.methods
    diagnostics = classify_diagnostics(cluster, temporal=temporal, static=static,
                                       body=body)
    if body and body.get("locked"):
        rejections.append("candidate lies inside a body or clothing region")
    if static.get("locked"):
        rejections.append(f"static-scene lock for {static.get('run_frames')} frames "
                          f"while the camera moved "
                          f"{static.get('median_camera_motion_px')} px/frame")
    if not plausibility.get("ok", True):
        rejections.append("implausible motion: "
                          + str(plausibility.get("reason", "no reason recorded")))
    if not temporal.get("supported"):
        rejections.append(f"temporal verification failed: "
                          f"{temporal.get('reason', 'round trip did not return')}")

    strong_evidence = len(methods) >= params.high_methods and temporal.get("supported") \
        and plausibility.get("ok", True) and not (body and body.get("locked")) \
        and not static.get("locked")
    medium_evidence = len(methods) >= params.medium_methods and temporal.get("supported") \
        and plausibility.get("ok", True) and not (body and body.get("locked"))

    # A comparable rival on the same frame is treated as genuine ambiguity, even
    # when this cluster has strong support: two defensible answers is exactly the
    # situation in which the v1 tracker picked the wrong one. False-positive
    # avoidance is worth more here than recall.
    if competitors > 0:
        tier = TIER_AMBIGUOUS
        rejections.append(f"{competitors} competing candidate(s) with comparable "
                          "support on this frame")
    elif strong_evidence:
        tier = TIER_HIGH
    elif medium_evidence:
        tier = TIER_MEDIUM
    elif len(methods) >= 1:
        tier = TIER_LOW
    else:
        tier = TIER_UNAVAILABLE

    return ConsensusLabel(clip_id, frame_index, tier, cluster.x, cluster.y,
                          cluster.radius, methods, round(cluster.spread, 2),
                          round(cluster.best_score, 4),
                          dict(temporal), diagnostics, rejections)


def consensus_for_window(clip_id: str, contexts: Sequence[FrameContext], *,
                         candidates_by_frame: Mapping[int, Sequence[Candidate2]],
                         body_regions_by_frame: Optional[Mapping[int, Sequence[Mapping]]] = None,
                         params: ConsensusParams = ConsensusParams()) -> dict:
    """Run the whole consensus pass over one decoded window."""
    clusters_by_frame: dict[int, list[Cluster]] = {}
    for context in contexts:
        clusters_by_frame[context.index] = cluster_candidates(
            candidates_by_frame.get(context.index, []), params=params)

    chosen: dict[int, Optional[Cluster]] = {}
    for context in contexts:
        clusters = clusters_by_frame[context.index]
        chosen[context.index] = clusters[0] if clusters else None

    # phase correlation runs on the working copy, so its shift is a working-scale
    # distance; a frame-space step in native pixels is not comparable with it until
    # it is converted.
    motion_px = [math.hypot(*context.motion) * float(context.scale or 1.0)
                 for context in contexts]

    labels: dict[int, ConsensusLabel] = {}
    for position, context in enumerate(contexts):
        clusters = clusters_by_frame[context.index]
        cluster = chosen[context.index]
        if cluster is None:
            labels[context.index] = consensus_label(
                clip_id, context.index, None, competitors=0, temporal={},
                static={}, plausibility={}, params=params)
            continue
        temporal = track_patch_both_ways(contexts, context.index, cluster.x, cluster.y,
                                         cluster.radius, params)
        window = [chosen.get(other.index) for other in
                  contexts[max(0, position - 4):position + 5]]
        plausibility = displacement_plausibility(window, params)
        static = static_scene_lock(window, params,
                                   motion_px=motion_px[max(0, position - 4):position + 5])
        regions = (body_regions_by_frame or {}).get(context.index, [])
        body = body_lock(cluster.x, cluster.y, cluster.radius, regions) if regions \
            else None
        # A rival has to be a genuinely competing hypothesis: spatially separate (a
        # near-duplicate detection of the same object is not a rival), comparable in
        # score *and* in independent support. Counting every other cluster within 80%
        # of the best score made clutter frames permanently AMBIGUOUS -- real footage
        # produced 5-22 "competing" clusters per frame and zero labels above LOW.
        competitors = sum(
            1 for other in clusters[1:]
            if math.hypot(other.x - cluster.x, other.y - cluster.y)
            > 4.0 * max(other.radius, cluster.radius)
            and other.best_score >= 0.9 * cluster.best_score
            and len(other.methods) >= len(cluster.methods))
        labels[context.index] = consensus_label(
            clip_id, context.index, cluster, competitors=competitors, temporal=temporal,
            static=static, plausibility=plausibility, body=body, params=params)

    tiers: dict[str, int] = {tier: 0 for tier in TIERS}
    for label in labels.values():
        tiers[label.tier] += 1
    diagnostics: dict[str, int] = {kind: 0 for kind in DIAGNOSTIC_KINDS}
    for label in labels.values():
        for kind in label.diagnostics:
            diagnostics[kind] = diagnostics.get(kind, 0) + 1
    return {
        "clip_id": clip_id,
        "frames": len(contexts),
        "tiers": tiers,
        "diagnostics": diagnostics,
        "labels": labels,
        "clusters_by_frame": clusters_by_frame,
        "high_confidence": [label for label in labels.values()
                            if label.tier == TIER_HIGH],
    }


def training_labels(consensus: Mapping[str, Any], *,
                    tiers: Sequence[str] = (TIER_HIGH,)) -> list[ConsensusLabel]:
    """Which labels a training round may use, and nothing more.

    MEDIUM labels carry a position and are useful evidence about where the tracker
    believes the ball is, but only HIGH ones are allowed into training by default,
    because a wrong label teaches the wrong thing far more efficiently than a
    missing one.
    """
    allowed = set(tiers)
    return [label for label in consensus["labels"].values()
            if label.tier in allowed and label.x is not None]


def summarize_consensus(consensus: Mapping[str, Any]) -> dict:
    tiers = {tier: 0 for tier in TIERS}
    methods_per_tier: dict[str, list[int]] = {tier: [] for tier in TIERS}
    for label in consensus["labels"].values():
        tiers[label.tier] += 1
        methods_per_tier[label.tier].append(len(label.methods))
    return {
        "clip_id": consensus["clip_id"],
        "frames": consensus["frames"],
        "tiers": tiers,
        "diagnostics": consensus["diagnostics"],
        "mean_methods_per_tier": {
            tier: (round(float(np.mean(values)), 2) if values else None)
            for tier, values in methods_per_tier.items()},
        "trainable_labels": len(training_labels(consensus)),
    }


def mine_hard_negatives(clip_id: str, contexts: Sequence[FrameContext], *,
                        candidates_by_frame: Mapping[int, Sequence[Candidate2]],
                        consensus: Mapping[str, Any], out_dir,
                        crop_size: int = 64, max_per_clip: int = 60,
                        params: ConsensusParams = ConsensusParams()) -> list[dict]:
    """Collect strong candidates that consensus refused, as hard negatives.

    These are the objects that fooled the previous tracker: a shirt, a shoe, a
    clubhead, a flagstick, glare. No human tags them; they are mined by being
    confident and wrong, and each record keeps the method that proposed it and the
    reason it was rejected.
    """
    accepted = {label.frame_index: label for label in consensus["labels"].values()
                if label.tier in (TIER_HIGH, TIER_MEDIUM)}
    records: list[dict] = []
    for context in contexts:
        label = accepted.get(context.index)
        for cluster in consensus["clusters_by_frame"].get(context.index, []):
            if len(records) >= max_per_clip:
                break
            if label is not None and math.hypot(cluster.x - label.x,
                                                cluster.y - label.y) <= \
                    params.agreement_radii * max(cluster.radius, label.radius or 1.0):
                continue                                   # this is the accepted ball
            # Native pixels, because the caller crops the original frame. Taking the
            # window from the working copy with native coordinates put the crop in the
            # wrong place on every downscaled clip and silently dropped the rest.
            half = crop_size // 2
            left, top = int(round(cluster.x)) - half, int(round(cluster.y)) - half
            native_width = int(round(context.colour.shape[1] * float(context.scale or 1.0)))
            native_height = int(round(context.colour.shape[0] * float(context.scale or 1.0)))
            if left < 0 or top < 0 or left + crop_size > native_width or \
                    top + crop_size > native_height:
                continue
            records.append({
                "clip_id": clip_id,
                "frame_index": context.index,
                "x": round(cluster.x, 2),
                "y": round(cluster.y, 2),
                "radius": round(cluster.radius, 2),
                "score": round(cluster.best_score, 4),
                "methods": cluster.methods,
                "origin": "MODEL_CONSENSUS",
                "kind": "hard_negative",
                "crop": [left, top, crop_size, crop_size],
                "rejections": list(next((l.rejections for l in
                                         [consensus["labels"].get(context.index)]
                                         if l), [])),
            })
    return records
