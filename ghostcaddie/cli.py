"""argparse surface + composition root: wire adapters and strategies together."""

import argparse
import json
import tempfile
import shutil
from dataclasses import asdict, replace
from pathlib import Path

from .adapters.json_file import (
    JsonCourseDataSource,
    JsonPlayerProfileSource,
    JsonShotDataSource,
)
from .config import Config
from .pipeline import run_pipeline
from .session import (InMemoryCourseSource, InMemoryPlayerSource, InMemoryShotSource,
                      parse_session, run_session, serialize_session_report)
from .adapters.provider_session import run_provider_session
from .geometry import Point2D
from .video.annotations import annotate_frame, render_annotated_video
from .video.calibration import load_video_calibration
from .video.diagnostics import build_video_diagnostics, serialize_video_diagnostics
from .video.errors import VideoContractError, VideoReconstructionUnavailable
from .video.extraction import extract_frames, generate_contact_sheet, validate_video_source
from .video.observations import load_fixture_observations
from .video.automatic_perception import evaluate_sequence_gates, reconstruct_automatic_shot
from .video.automatic_render import (build_automatic_report, build_evaluation_report,
                                     render_automatic_frame,
                                     serialize_evaluation_report)
from .video.orchestration import run_video_pipeline, run_human_video_pipeline
from .video.paths import ProjectBoundary
from .video.reconstruction import ShotContext
from .video.human_import import load_human_annotations, import_human_annotations
from .video.metadata import inspect_video
from .video.annotation_workspace import build_annotation_workspace
from .video.prepare import prepare_video
from .video.human_import import observations_from_human_annotations
from .video.youtube import DownloadError, DownloadLimits, YtDlpDownloader, parse_youtube_url
from .video.youtube_auto_try import AUTO_FORMAT, AutoTryConfig, DEFAULT_YTDLP, auto_try
from .video.fairwayos_research import sidecar_from_mapping, write_fairwayos_sidecar
from .video.ai_demo import run_local_demo


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ghostcaddie",
        description="FairwayOS — golf shot-analytics engine (synthetic data only).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="Run the full analytics pipeline on one shot.")
    run_p.add_argument("--shot", required=True, type=Path, help="Path to shot JSON")
    run_p.add_argument("--course", required=True, type=Path, help="Path to course JSON")
    run_p.add_argument("--player", required=True, type=Path, help="Path to player JSON")
    run_p.add_argument("--out", type=Path, default=Path("out"), help="Output directory (default: out/)")
    run_p.add_argument("--seed", type=int, default=None, help="Override random seed")
    run_p.add_argument("--samples", type=int, default=None, help="Override Monte Carlo sample count")
    session_p = sub.add_parser("session", help="Run the session pipeline over a multi-shot envelope.")
    session_p.add_argument("--input", required=True, type=Path, help="Path to session envelope JSON")
    session_p.add_argument("--out", type=Path, default=Path("out"), help="Output directory (default: out/)")
    session_p.add_argument("--seed", type=int, default=None, help="Override session seed")
    session_p.add_argument("--samples", type=int, default=None, help="Override Monte Carlo sample count")
    provider_p = sub.add_parser("provider-session", help="Run a provider-aware session envelope.")
    provider_p.add_argument("--input", required=True, type=Path, help="Path to provider-session JSON")
    provider_p.add_argument("--out", type=Path, default=Path("out"), help="Output directory (default: out/)")
    provider_p.add_argument("--seed", type=int, default=None, help="Override session seed")
    provider_p.add_argument("--samples", type=int, default=None, help="Override Monte Carlo sample count")
    provider_p.add_argument("--permissive", action="store_true", help="Preserve unknown fields as diagnostics")
    prep_p = sub.add_parser("video-prepare", help="Prepare deterministic offline artifacts and a blank human annotation draft.")
    prep_p.add_argument("--video", required=True, type=Path, help="Local video source; never serialized.")
    prep_p.add_argument("--out", required=True, type=Path, help="Output directory for the prepared package.")
    prep_p.add_argument("--sample-fps", type=float, default=2.0)
    prep_p.add_argument("--max-frames", type=int, default=None)
    video_p = sub.add_parser("video-analyze", help="Analyze video with deterministic fixture observations (no model perception).")
    video_p.add_argument("--video", required=True, type=Path, help="Local video; absolute paths are allowed and never serialized.")
    video_p.add_argument("--project-root", type=Path, default=Path.cwd(), help="Root for project-relative resources.")
    for name in ("calibration", "course", "player", "observations"):
        video_p.add_argument("--" + name, required=True, help="Project-relative JSON (absolute/traversal/symlink escapes rejected).")
    video_p.add_argument("--out", type=Path, default=Path("out"))
    video_p.add_argument("--sample-fps", type=float, default=2.0)
    video_p.add_argument("--max-frames", type=int, default=None)
    video_p.add_argument("--render-video", action="store_true", help="Export sampled annotated frames as annotated_video.mp4.")
    video_p.add_argument("--event-id", default="VIDEO-EVENT-0001")
    video_p.add_argument("--tournament-id", default="VIDEO-FIXTURE")
    video_p.add_argument("--hole", type=int, default=1)
    video_p.add_argument("--shot-number", type=int, default=1)
    video_p.add_argument("--lie", default="fairway")
    video_p.add_argument("--club", default="7i")
    video_p.add_argument("--distance-to-pin", type=float, default=150.0)
    video_p.add_argument("--wind-speed", type=float, default=0.0)
    video_p.add_argument("--wind-direction", type=float, default=0.0)
    video_p.add_argument("--timestamp", default="video-fixture")
    video_p.add_argument("--target-x", type=float, default=None)
    video_p.add_argument("--target-y", type=float, default=None)
    automatic_p = sub.add_parser("video-automatic-analyze", aliases=["video-auto-analyze"], help="Analyze approved video-observations.v1 evidence with provisional automatic gates; no model perception is run.")
    automatic_p.add_argument("--video", required=True, type=Path, help="Local video; absolute paths are allowed and never serialized.")
    automatic_p.add_argument("--observations", required=True, help="Project-relative video-observations.v1 from an approved automatic adapter.")
    automatic_p.add_argument("--calibration", required=True, help="Project-relative calibration JSON.")
    automatic_p.add_argument("--course", required=True, help="Project-relative course JSON.")
    automatic_p.add_argument("--player", required=True, help="Project-relative player JSON.")
    automatic_p.add_argument("--project-root", required=True, type=Path, help="Existing project root for relative resources.")
    automatic_p.add_argument("--out", required=True, type=Path, help="Output directory for deterministic reports and artifacts.")
    automatic_p.add_argument("--render-video", action="store_true", help="Export sampled annotated frames as annotated_video.mp4.")
    automatic_p.add_argument("--fallback-human", action="store_true", help="Explicitly prepare a blank human annotation workspace if automatic evidence is blocked.")
    automatic_p.add_argument("--sample-fps", type=float, default=2.0)
    automatic_p.add_argument("--max-frames", type=int, default=None)
    automatic_p.add_argument("--event-id", default="VIDEO-EVENT-0001")
    automatic_p.add_argument("--tournament-id", default="VIDEO-AUTOMATIC")
    automatic_p.add_argument("--hole", type=int, default=1)
    automatic_p.add_argument("--shot-number", type=int, default=1)
    automatic_p.add_argument("--lie", default="fairway")
    automatic_p.add_argument("--club", default="7i")
    automatic_p.add_argument("--distance-to-pin", type=float, default=150.0)
    automatic_p.add_argument("--wind-speed", type=float, default=0.0)
    automatic_p.add_argument("--wind-direction", type=float, default=0.0)
    automatic_p.add_argument("--timestamp", default="video-automatic")
    automatic_p.add_argument("--target-x", type=float, default=None)
    automatic_p.add_argument("--target-y", type=float, default=None)
    import_p = sub.add_parser("video-import", help="Import one submitted video-human-annotations.v1 document.")
    import_p.add_argument("--annotations", required=True, help="Project-relative submitted annotation JSON.")
    import_p.add_argument("--calibration", required=True, help="Project-relative calibration JSON.")
    import_p.add_argument("--course", required=True, help="Project-relative course JSON.")
    import_p.add_argument("--player", required=True, help="Project-relative player JSON.")
    import_p.add_argument("--project-root", type=Path, default=Path.cwd())
    import_p.add_argument("--video", type=Path, default=None, help="Optional local source video; never serialized.")
    import_p.add_argument("--out", type=Path, default=Path("out"))
    import_p.add_argument("--event-id", default="VIDEO-EVENT-0001")
    import_p.add_argument("--tournament-id", default="VIDEO-HUMAN")
    import_p.add_argument("--hole", type=int, default=1)
    import_p.add_argument("--shot-number", type=int, default=1)
    import_p.add_argument("--distance-to-pin", type=float, default=150.0)
    import_p.add_argument("--wind-speed", type=float, default=0.0)
    import_p.add_argument("--wind-direction", type=float, default=0.0)
    import_p.add_argument("--timestamp", default="video-human-annotation")
    import_p.add_argument("--target-x", type=float, required=True)
    import_p.add_argument("--target-y", type=float, required=True)
    human_p = sub.add_parser("video-human-analyze", help="Analyze submitted video-human-annotations.v1 with existing analytics.")
    human_p.add_argument("--annotations", required=True, help="Project-relative submitted annotation JSON.")
    human_p.add_argument("--calibration", required=True, help="Project-relative calibration JSON.")
    human_p.add_argument("--course", required=True, help="Project-relative course JSON.")
    human_p.add_argument("--player", required=True, help="Project-relative player JSON.")
    human_p.add_argument("--project-root", type=Path, default=Path.cwd())
    human_p.add_argument("--video", required=True, type=Path, help="Local source video; never serialized.")
    human_p.add_argument("--out", type=Path, default=Path("out"))
    human_p.add_argument("--event-id", default="VIDEO-EVENT-0001")
    human_p.add_argument("--tournament-id", default="VIDEO-HUMAN")
    human_p.add_argument("--hole", type=int, default=1)
    human_p.add_argument("--shot-number", type=int, default=1)
    human_p.add_argument("--distance-to-pin", type=float, default=150.0)
    human_p.add_argument("--wind-speed", type=float, default=0.0)
    human_p.add_argument("--wind-direction", type=float, default=0.0)
    human_p.add_argument("--timestamp", default="video-human-annotation")
    human_p.add_argument("--target-x", type=float, required=True)
    human_p.add_argument("--target-y", type=float, required=True)
    human_p.add_argument("--sample-fps", type=float, default=2.0)
    human_p.add_argument("--max-frames", type=int, default=None)
    human_p.add_argument("--render-video", action="store_true", help="Export sampled annotated frames as annotated_video.mp4.")
    youtube_p = sub.add_parser("youtube-analyze", help="Ingest one YouTube video; automatic perception is gated and unavailable by default.")
    youtube_p.add_argument("--url", required=True, help="Explicit public HTTPS YouTube video URL.")
    youtube_p.add_argument("--calibration", required=True, help="Project-relative calibration JSON.")
    youtube_p.add_argument("--course", required=True, help="Project-relative course JSON.")
    youtube_p.add_argument("--player", required=True, help="Project-relative player JSON.")
    youtube_p.add_argument("--project-root", required=True, type=Path, help="Existing project root for relative resources.")
    youtube_p.add_argument("--out", required=True, type=Path, help="Output directory for sanitized diagnostics/artifacts.")
    youtube_p.add_argument("--yt-dlp", "--downloader", dest="downloader", required=True, help="Explicit executable path for yt-dlp.")
    youtube_p.add_argument("--fallback-human", action="store_true", help="Explicitly prepare a blank human annotation workspace after download.")
    youtube_p.add_argument("--sample-fps", type=float, default=2.0)
    youtube_p.add_argument("--max-frames", type=int, default=None)
    auto_p = sub.add_parser("youtube-auto-try", help="Best-effort bounded YouTube perception; incomplete evidence remains blocked.")
    auto_p.add_argument("--url", required=True)
    auto_p.add_argument("--out", required=True, type=Path)
    auto_p.add_argument("--calibration")
    auto_p.add_argument("--course")
    auto_p.add_argument("--player")
    auto_p.add_argument("--project-root", type=Path, default=Path.cwd())
    auto_p.add_argument("--segment-start", type=float, default=0.0)
    auto_p.add_argument("--segment-duration", type=float, default=None)
    auto_p.add_argument("--render-video", action="store_true")
    auto_p.add_argument("--fallback-human", action="store_true")
    auto_p.add_argument("--yt-dlp", default=DEFAULT_YTDLP,
                        help="Explicit executable path for bounded yt-dlp download.")
    demo_p = sub.add_parser("ai-demo", help="Run bounded research-only AI demo and render H.264.",
                             description="Bounded research-only AI Demo Mode; H.264 output with visible uncertainty and warnings.")
    source_group = demo_p.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--url", help="Explicit public HTTPS YouTube video URL.")
    source_group.add_argument("--video", type=Path, help="Existing local video for offline demo validation.")
    demo_p.add_argument("--out", required=True, type=Path, help="Local output directory; media remains ignored.")
    demo_p.add_argument("--yt-dlp", default=DEFAULT_YTDLP, help="Explicit yt-dlp executable for --url.")
    demo_p.add_argument("--segment-start", type=float, default=0.0)
    demo_p.add_argument("--segment-duration", type=float, default=None)
    demo_p.add_argument("--max-duration", type=float, default=8.0)
    demo_p.add_argument("--sample-fps", type=float, default=4.0)
    demo_p.add_argument("--max-frames", type=int, default=None)
    demo_p.add_argument("--pose-model", default=None, help="Optional local pose checkpoint path/adapter hint.")
    demo_p.add_argument("--ball-model", default=None, help="Optional local ball checkpoint path/adapter hint.")
    sidecar_p = sub.add_parser(
        "fairwayos-ball-sidecar",
        help="Serialize shared research ball candidates for FairwayOS diagnostics; never runs analytics.",
    )
    sidecar_p.add_argument("--input", required=True, type=Path,
                           help="JSON track emitted by a research-only ball runner.")
    sidecar_p.add_argument("--out", required=True, type=Path,
                           help="Output FairwayOS research sidecar JSON.")
    sidecar_p.add_argument("--source", default=None,
                           help="Optional human-readable source label; no video bytes are copied.")
    return parser


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        raise SystemExit(1)

    if args.command == "provider-session":
        _run_provider_session_command(args)
        return

    if args.command == "session":
        _run_session_command(args)
        return

    if args.command == "video-prepare":
        _run_video_prepare_command(args)
        return

    if args.command == "video-analyze":
        _run_video_analyze_command(args)
        return

    if args.command in {"video-automatic-analyze", "video-auto-analyze"}:
        _run_video_automatic_analyze_command(args)
        return

    if args.command == "video-import":
        _run_video_import_command(args)
        return

    if args.command == "video-human-analyze":
        _run_video_human_analyze_command(args)
        return

    if args.command == "youtube-analyze":
        _run_youtube_analyze_command(args)
        return

    if args.command == "youtube-auto-try":
        _run_youtube_auto_try_command(args)
        return

    if args.command == "ai-demo":
        _run_ai_demo_command(args)
        return

    if args.command == "fairwayos-ball-sidecar":
        _run_fairwayos_ball_sidecar_command(args)
        return

    config = Config.default()
    if args.seed is not None:
        config = replace(config, simulation=replace(config.simulation, random_seed=args.seed))
    if args.samples is not None:
        config = replace(
            config, simulation=replace(config.simulation, monte_carlo_samples=args.samples)
        )

    course_source = JsonCourseDataSource(args.course)
    course = course_source.load_course()
    shot_source = JsonShotDataSource(args.shot, course.coordinate_system)
    player_source = JsonPlayerProfileSource(args.player)

    result = run_pipeline(shot_source, course_source, player_source, config)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "recommendation.json", "w") as fh:
        json.dump(asdict(result.recommendation), fh, indent=2, default=str)
    with open(out_dir / "overlay.svg", "w") as fh:
        fh.write(result.svg)

    _print_terminal_summary(result)


def _run_fairwayos_ball_sidecar_command(args) -> None:
    with args.input.open() as fh:
        payload = json.load(fh)
    sidecar = sidecar_from_mapping(payload, source=args.source)
    write_fairwayos_sidecar(args.out, sidecar)
    print(f"Wrote research-only FairwayOS ball sidecar to {args.out}")


def _run_provider_session_command(args) -> None:
    with open(args.input) as fh:
        raw = json.load(fh)
    if args.seed is not None:
        raw.setdefault("session", {})["seed"] = args.seed
    config = Config.default()
    if args.samples is not None:
        config = replace(config, simulation=replace(config.simulation, monte_carlo_samples=args.samples))
    report = run_provider_session(raw, args.input, config, strict=not args.permissive)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "session_report.json", "w") as fh:
        fh.write(serialize_session_report(report))
    _print_session_summary(report)


def _run_session_command(args) -> None:
    with open(args.input) as fh:
        raw = json.load(fh)
    session = parse_session(raw)
    config = Config.default()
    if args.seed is not None:
        session.seed = args.seed
    if args.samples is not None:
        config = replace(
            config, simulation=replace(config.simulation, monte_carlo_samples=args.samples)
        )
    report = run_session(session, config)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "session_report.json", "w") as fh:
        fh.write(serialize_session_report(report))
    _print_session_summary(report)


def _run_video_prepare_command(args) -> None:
    artifacts = prepare_video(str(args.video), str(args.out), sample_fps=args.sample_fps,
                              max_frames=args.max_frames)
    print("Prepared blank video-human-annotations.v1 draft in " + artifacts["output_directory"])


def _run_video_import_command(args) -> None:
    project = ProjectBoundary(args.project_root)
    document = load_human_annotations(args.annotations, project)
    calibration = load_video_calibration(args.calibration, project)
    player_path = project.resolve_player(args.player)
    course_path = project.resolve_course(args.course)
    player = JsonPlayerProfileSource(player_path).load_player()
    # Load course through the boundary as an import prerequisite; analytics is
    # intentionally not run in this milestone.
    JsonCourseDataSource(course_path).load_course()
    if args.video is not None:
        metadata = inspect_video(str(validate_video_source(str(args.video))))
        if (metadata.width, metadata.height) != (calibration.width, calibration.height):
            raise VideoContractError("video dimensions do not match calibration and annotations")
    result = import_human_annotations(document, calibration, event_id=args.event_id,
        player_id=player.player_id, tournament_id=args.tournament_id, hole_number=args.hole,
        shot_number=args.shot_number, distance_to_pin=args.distance_to_pin,
        wind={"speed_mph": args.wind_speed, "direction_deg": args.wind_direction},
        timestamp=args.timestamp, target_pixel=Point2D(args.target_x, args.target_y))
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    payload = {"import_result": {"status": "complete", "schema_version": "video-human-annotations.v1",
        "reconstructed": True, "analytics": "not_run"}, "event": asdict(result.shot_event),
        "metadata": result.metadata}
    (out / "normalized_shot.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _run_video_human_analyze_command(args) -> None:
    project = ProjectBoundary(args.project_root)
    document = load_human_annotations(args.annotations, project)
    calibration = load_video_calibration(args.calibration, project)
    player_source = JsonPlayerProfileSource(project.resolve_player(args.player))
    course_source = JsonCourseDataSource(project.resolve_course(args.course))
    course = course_source.load_course()
    video_path = validate_video_source(str(args.video))
    metadata = inspect_video(str(video_path))
    if (metadata.width, metadata.height) != (calibration.width, calibration.height):
        raise VideoContractError("video dimensions do not match calibration and annotations")
    if (metadata.width, metadata.height) != (document.payload["video"]["width"], document.payload["video"]["height"]):
        raise VideoContractError("video dimensions do not match calibration and annotations")
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    frames = extract_frames(str(video_path), str(out / "frames"), sample_fps=args.sample_fps, max_frames=args.max_frames)
    generate_contact_sheet(frames.output_directory, str(out / "contact_sheet.jpg"), columns=min(4, len(frames.frames)))
    result = run_human_video_pipeline(
        document, calibration, course_source, player_source, Config.default(),
        event_id=args.event_id, tournament_id=args.tournament_id, hole_number=args.hole,
        shot_number=args.shot_number, distance_to_pin=args.distance_to_pin,
        wind={"speed_mph": args.wind_speed, "direction_deg": args.wind_direction},
        timestamp=args.timestamp, target_pixel=Point2D(args.target_x, args.target_y),
    )
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    recommendation = asdict(result.recommendation)
    recommendation["provenance"].pop("generated_at", None)
    recommendation["analytics_status"] = result.analytics_status
    recommendation["human_annotation_status"] = document.status
    (out / "recommendation.json").write_text(json.dumps(recommendation, indent=2, default=str) + "\n")
    (out / "overlay.svg").write_text(result.svg)
    normalized = {"event": asdict(result.shot_event), "metadata": result.metadata,
                  "analytics_status": result.analytics_status}
    (out / "normalized_shot.json").write_text(json.dumps(normalized, indent=2, default=str) + "\n")
    human_observations = observations_from_human_annotations(document)
    annotated_dir = out / "annotated_frames"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames.frames:
        selected = min(human_observations.items,
                       key=lambda item: abs(item.frame_index - (frame.frame_index - 1)))
        annotate_frame(Path(frames.output_directory) / frame.filename,
                       annotated_dir / frame.filename, selected, calibration)
    workspace_video = {"width": metadata.width, "height": metadata.height,
                       "frame_count": len(frames.frames), "duration_seconds": metadata.duration_seconds}
    workspace_frames = []
    for frame in frames.frames:
        frame_data = frame.to_dict()
        frame_data["filename"] = "frames/" + frame_data["filename"]
        workspace_frames.append(frame_data)
    workspace = build_annotation_workspace(
        workspace_frames, video=workspace_video,
        contact_sheet_href="contact_sheet.jpg", title="Offline human annotation workspace",
        context="Submitted video-human-annotations.v1; deterministic local extraction")
    (out / "annotation_workspace.html").write_text(workspace)
    references = ["frames/frame_manifest.json", "contact_sheet.jpg", "annotation_workspace.html",
                  "recommendation.json", "overlay.svg", "normalized_shot.json"]
    references.extend(f"annotated_frames/{frame.filename}" for frame in frames.frames)
    diagnostics = build_video_diagnostics(
        human_observations, metadata, artifact_references=references,
        reconstruction=result.reconstruction, analytics_result=recommendation,
        calibration=calibration, warnings=["human-submitted coordinates; no model perception"],
        model_provider_provenance={"model": "none", "provider": "none", "mode": "human-submitted",
                                   "source": "video-human-annotations.v1"}, status="complete")
    if args.render_video:
        render_annotated_video(annotated_dir, out / "annotated_video.mp4", frame_rate=args.sample_fps)
        diagnostics.artifact_references.append("annotated_video.mp4")
    (out / "diagnostics.json").write_text(serialize_video_diagnostics(diagnostics) + "\n")


def _run_video_automatic_analyze_command(args) -> None:
    """Run only validated, approved observation evidence through guarded analytics."""
    project = ProjectBoundary(args.project_root)
    video = validate_video_source(str(args.video))
    calibration = load_video_calibration(args.calibration, project)
    observations = load_fixture_observations(args.observations, project)
    metadata = inspect_video(str(video))
    if (metadata.width, metadata.height) != (calibration.width, calibration.height) or \
            (metadata.width, metadata.height) != (observations.image_width, observations.image_height):
        raise VideoContractError("video dimensions do not match calibration and observations")

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    # A blocked rerun must not leave stale analytics artifacts looking valid.
    for name in ("recommendation.json", "normalized_shot.json", "overlay.svg"):
        (out / name).unlink(missing_ok=True)
    frames = extract_frames(str(video), str(out / "frames"), sample_fps=args.sample_fps,
                            max_frames=args.max_frames)
    generate_contact_sheet(frames.output_directory, str(out / "contact_sheet.jpg"),
                           columns=min(4, len(frames.frames)))
    annotated_dir = out / "annotated_frames"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames.frames):
        observation = observations.items[min(index, len(observations.items) - 1)]
        render_automatic_frame(Path(frames.output_directory) / frame.filename,
                               annotated_dir / frame.filename, observation, calibration)

    frame_count = max(item.frame_index for item in observations.items) + 1
    observed = [item.frame_index for item in observations.items]
    anchor_frames = [item.frame_index for item in observations.items
                     if item.golfer.anchor is not None]
    gate = evaluate_sequence_gates(
        camera_motion_displacements=(), cut_frames=(),
        track_coverage=len(set(observed)) / frame_count,
        longest_gap=max((b - a - 1 for a, b in zip(sorted(set(observed)), sorted(set(observed))[1:])), default=0),
        anchor_coverage=len(set(anchor_frames)) / frame_count,
    )
    reasons = list(gate.blocking_reasons)
    required = {
        "ball": any(item.ball is not None for item in observations.items),
        "clubhead": any(item.clubhead is not None for item in observations.items),
        "contact": any(item.contact is not None for item in observations.items),
        "landing": any(item.landing is not None for item in observations.items),
        "calibration": (metadata.width, metadata.height) == (calibration.width, calibration.height),
    }
    reasons.extend(name + "_evidence_unavailable" for name, present in required.items() if not present)
    if reasons:
        from .video.automatic_perception import GateDecision, Thresholds, ContinuityMetrics
        gate = GateDecision("blocked", False, tuple(sorted(set(reasons))), True)
    else:
        from .video.automatic_perception import Thresholds, ContinuityMetrics
    continuity = ContinuityMetrics(len(set(observed)) / frame_count,
                                   max((b - a - 1 for a, b in zip(sorted(set(observed)), sorted(set(observed))[1:])), default=0),
                                   len(set(observed)))
    confidence_values = [item.golfer.confidence for item in observations.items]
    confidence_values.extend(item.ball["confidence"] for item in observations.items if item.ball is not None)
    confidence_values.extend(item.clubhead["confidence"] for item in observations.items if item.clubhead is not None)
    from .video.automatic_perception import ConfidenceMetrics, Thresholds
    automatic_report = build_automatic_report(
        [item.to_dict() for item in observations.items], gate_decision=gate,
        thresholds=Thresholds(), confidence_metrics=ConfidenceMetrics.from_values(confidence_values),
        continuity_metrics=continuity,
        artifact_references=["frames/frame_manifest.json", "contact_sheet.jpg"],
        visual_references=[f"annotated_frames/{frame.filename}" for frame in frames.frames])
    evaluation = build_evaluation_report(
        track_continuity=continuity, unavailable_reasons={
            "anchor_error": "no ground truth supplied", "impact_error": "no ground truth supplied",
            "ball_precision_recall": "no ground truth supplied", "clubhead_precision_recall": "no ground truth supplied",
            "landing_error": "no ground truth supplied", "false_positives": "no ground truth supplied",
            "runtime": "not measured by this deterministic boundary"})
    (out / "evaluation.json").write_text(serialize_evaluation_report(evaluation) + "\n")
    references = ["frames/frame_manifest.json", "contact_sheet.jpg", "evaluation.json"] + [
        f"annotated_frames/{frame.filename}" for frame in frames.frames]
    diagnostics = build_video_diagnostics(
        observations, metadata, artifact_references=references, calibration=calibration,
        warnings=["provisional automatic gate; approved adapter evidence only"],
        model_provider_provenance={"model": "none", "provider": "none",
                                   "mode": "approved-automatic-adapter",
                                   "source": "video-observations.v1"},
        status="failed" if not gate.passed else "pending")
    diagnostics_payload = json.loads(serialize_video_diagnostics(diagnostics))
    diagnostics_payload.update({"status": "blocked" if not gate.passed else "pending",
                                "gate": automatic_report["gate"],
                                "automatic_report": automatic_report})
    if args.render_video:
        render_annotated_video(annotated_dir, out / "annotated_video.mp4", frame_rate=args.sample_fps)
        references.append("annotated_video.mp4")
        diagnostics_payload["artifact_references"] = sorted(set(references))
    (out / "diagnostics.json").write_text(json.dumps(diagnostics_payload, sort_keys=True, separators=(",", ":")) + "\n")
    if not gate.passed:
        if args.fallback_human:
            prepare_video(str(video), str(out), sample_fps=args.sample_fps, max_frames=args.max_frames)
        raise SystemExit(2)

    player_source = JsonPlayerProfileSource(project.resolve_player(args.player))
    course_source = JsonCourseDataSource(project.resolve_course(args.course))
    player = player_source.load_player()
    course = course_source.load_course()
    first_anchor = observations.items[0].golfer.anchor
    target_pixel = calibration.from_engine(course.pin_position) if args.target_x is None and args.target_y is None else Point2D(
        args.target_x if args.target_x is not None else first_anchor["x"],
        args.target_y if args.target_y is not None else first_anchor["y"])
    context = ShotContext(
        event_id=args.event_id, player_id=player.player_id, tournament_id=args.tournament_id,
        hole_number=args.hole, shot_number=args.shot_number, lie=args.lie, club=args.club,
        distance_to_pin=args.distance_to_pin, wind={"speed_mph": args.wind_speed, "direction_deg": args.wind_direction},
        timestamp=args.timestamp, target_pixel=target_pixel)
    try:
        reconstruction = reconstruct_automatic_shot(observations, calibration, context)
        event = reconstruction.shot_event
        event.provenance = {"source": "video-automatic", "video": dict(reconstruction.metadata)}
        pipeline_result = run_pipeline(InMemoryShotSource(event, "video:automatic:inline:shot"),
            InMemoryCourseSource(course, "video:automatic:inline:course"),
            InMemoryPlayerSource(player, "video:automatic:inline:player"), Config.default())
    except VideoReconstructionUnavailable as exc:
        diagnostics_payload["status"] = "blocked"
        diagnostics_payload["warnings"] = sorted(set(diagnostics_payload.get("warnings", []) + [str(exc)]))
        (out / "diagnostics.json").write_text(json.dumps(diagnostics_payload, sort_keys=True, separators=(",", ":")) + "\n")
        raise SystemExit(2) from exc
    recommendation = asdict(pipeline_result.recommendation)
    recommendation["provenance"].pop("generated_at", None)
    (out / "recommendation.json").write_text(json.dumps(recommendation, indent=2, default=str) + "\n")
    (out / "overlay.svg").write_text(pipeline_result.svg)
    (out / "normalized_shot.json").write_text(json.dumps({"event": asdict(event), "metadata": reconstruction.metadata}, indent=2, default=str) + "\n")
    diagnostics_payload["status"] = "complete"
    diagnostics_payload["artifact_references"] = sorted(set(references + ["recommendation.json", "overlay.svg", "normalized_shot.json"]))
    diagnostics_payload["normalized_shot"] = {"event": asdict(event), "metadata": reconstruction.metadata}
    (out / "diagnostics.json").write_text(json.dumps(diagnostics_payload, sort_keys=True, separators=(",", ":")) + "\n")


def _run_video_analyze_command(args) -> None:
    project = ProjectBoundary(args.project_root)
    video = validate_video_source(str(args.video))
    calibration = load_video_calibration(args.calibration, project)
    observations = load_fixture_observations(args.observations, project)
    metadata = inspect_video(str(video))
    dimensions = (metadata.width, metadata.height)
    if dimensions != (calibration.width, calibration.height) or dimensions != (observations.image_width, observations.image_height):
        raise VideoContractError("video dimensions do not match calibration and observations")
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    frames = extract_frames(str(video), str(out / "frames"), sample_fps=args.sample_fps, max_frames=args.max_frames)
    generate_contact_sheet(frames.output_directory, str(out / "contact_sheet.jpg"), columns=min(4, len(frames.frames)))
    annotated_dir = out / "annotated_frames"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames.frames):
        observation = observations.items[min(index, len(observations.items) - 1)]
        annotate_frame(Path(frames.output_directory) / frame.filename, annotated_dir / frame.filename, observation, calibration)
    references = ["frames/frame_manifest.json", "contact_sheet.jpg"] + [f"annotated_frames/{f.filename}" for f in frames.frames]
    player_path = project.resolve_player(args.player)
    player_source = JsonPlayerProfileSource(player_path)
    player_id = player_source.load_player().player_id
    course_source = JsonCourseDataSource(project.resolve_course(args.course))
    course = course_source.load_course()
    first_anchor = observations.items[0].golfer.anchor
    target_pixel = calibration.from_engine(course.pin_position) if args.target_x is None and args.target_y is None else Point2D(
        args.target_x if args.target_x is not None else first_anchor["x"],
        args.target_y if args.target_y is not None else first_anchor["y"])
    context = ShotContext(
        event_id=args.event_id, player_id=player_id, tournament_id=args.tournament_id,
        hole_number=args.hole, shot_number=args.shot_number, lie=args.lie, club=args.club,
        distance_to_pin=args.distance_to_pin, wind={"speed_mph": args.wind_speed, "direction_deg": args.wind_direction},
        timestamp=args.timestamp, target_pixel=target_pixel,
    )
    try:
        result = run_video_pipeline(observations, calibration, context, course_source, player_source, Config.default())
        recommendation_payload = asdict(result.recommendation)
        recommendation_payload["provenance"].pop("generated_at", None)
        (out / "recommendation.json").write_text(json.dumps(recommendation_payload, indent=2, default=str) + "\n")
        (out / "overlay.svg").write_text(result.svg)
        (out / "normalized_shot.json").write_text(json.dumps({"event": asdict(result.shot_event), "metadata": result.reconstruction.metadata}, indent=2, default=str) + "\n")
        references.extend(["recommendation.json", "overlay.svg", "normalized_shot.json"])
        diagnostics = build_video_diagnostics(observations, metadata, artifact_references=references,
            reconstruction=result.reconstruction, analytics_result=recommendation_payload, calibration=calibration,
            warnings=["fixture perception; no model-backed perception"])
    except VideoReconstructionUnavailable as exc:
        diagnostics = build_video_diagnostics(observations, metadata, artifact_references=references,
            calibration=calibration, status="failed", warnings=[str(exc), "analytics unavailable"])
    if args.render_video:
        render_annotated_video(annotated_dir, out / "annotated_video.mp4", frame_rate=args.sample_fps)
        references.append("annotated_video.mp4")
        diagnostics.artifact_references = sorted(set(references))
    (out / "diagnostics.json").write_text(serialize_video_diagnostics(diagnostics) + "\n")


def _run_ai_demo_command(args) -> None:
    """Run only the research demo path; never invoke validated analytics."""
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for stale_name in ("recommendation.json", "normalized_shot.json", "overlay.svg"):
        stale = out / stale_name
        if stale.is_file() or stale.is_symlink():
            stale.unlink()
    ingest_dir = None
    source = None
    video = args.video
    try:
        if args.url:
            source = parse_youtube_url(args.url)
            ingest_dir = Path(tempfile.mkdtemp(prefix=".ai-demo-ingest-", dir=str(out.parent)))
            downloader = YtDlpDownloader(
                args.yt_dlp,
                limits=DownloadLimits(max_segment_seconds=20.0),
                format_selector=AUTO_FORMAT,
            )
            result = downloader.download(args.url, str(ingest_dir),
                                         segment_start=args.segment_start,
                                         segment_duration=args.segment_duration)
            video = Path(result.path)
            source = source.to_dict()
        elif video is not None:
            source = {"platform": "local", "video_id": video.stem}
        report = run_local_demo(
            str(video), str(out), sample_fps=args.sample_fps,
            max_duration_seconds=args.max_duration, max_frames=args.max_frames,
            source=source, pose_model=args.pose_model, ball_model=args.ball_model,
        )
        print(json.dumps({"status": report["status"], "output": str(args.out)}))
    except (DownloadError, OSError, RuntimeError, ValueError) as exc:
        (out / "diagnostics.json").write_text(json.dumps({
            "schema_version": "fairwayos-ai-demo.v1", "status": "blocked",
            "research_only": True, "ground_truth": False,
            "production_eligible": False, "coordinate_space": "pixels",
            "analytics": None, "shot_event": None, "warnings": [str(exc)],
        }, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "blocked", "output": str(args.out)}))
        raise SystemExit(2) from exc
    finally:
        if ingest_dir is not None:
            shutil.rmtree(str(ingest_dir), ignore_errors=True)


def _write_youtube_diagnostics(out, payload):
    """Write only sanitized, status-oriented YouTube orchestration evidence."""
    out = Path(out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnostics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run_youtube_analyze_command(args) -> None:
    """Download YouTube evidence, but never pretend download is perception."""
    ProjectBoundary(args.project_root)
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    ingest_dir = Path(tempfile.mkdtemp(prefix=".youtube-ingest-", dir=str(out.parent)))
    base = {
        "schema_version": "youtube-analysis.v1",
        "ingestion_status": "pending",
        "perception_status": "not_run",
        "calibration_status": "not_run",
        "reconstruction_status": "not_run",
        "analytics_status": "not_run",
        "overall_status": "failed",
        "status": "failed",
        "artifact_references": [],
    }
    try:
        downloader = YtDlpDownloader(args.downloader)
        result = downloader.download(args.url, str(ingest_dir))
    except DownloadError as exc:
        base.update({"ingestion_status": "failed", "warnings": ["YouTube ingestion failed"]})
        _write_youtube_diagnostics(out, base)
        raise SystemExit(2) from exc

    base["ingestion_status"] = "complete"
    base["source"] = result.source.to_dict()
    base["downloader"] = "yt-dlp"
    if args.fallback_human:
        artifacts = prepare_video(result.path, str(out), sample_fps=args.sample_fps, max_frames=args.max_frames)
        shutil.rmtree(str(ingest_dir), ignore_errors=True)
        refs = [artifacts[key] for key in ("manifest", "contact_sheet", "workspace", "draft")]
        base.update({
            "overall_status": "fallback-human",
            "status": "fallback-human",
            "fallback_mode": "human-annotation-preparation",
            "warnings": ["explicit fallback-human selected; blank draft only; no automatic perception"],
            "artifact_references": refs,
        })
        _write_youtube_diagnostics(out, base)
        print("Downloaded YouTube source; prepared explicit fallback-human blank annotation workspace")
        return

    base.update({
        "perception_status": "unavailable",
        "reconstruction_status": "blocked",
        "analytics_status": "blocked",
        "warnings": ["no genuinely available and evaluated automatic detector is configured; use --fallback-human"],
    })
    shutil.rmtree(str(ingest_dir), ignore_errors=True)
    _write_youtube_diagnostics(out, base)
    raise SystemExit(2)


def _run_youtube_auto_try_command(args) -> None:
    calibration = None
    course = player = None
    analytics_runner = None
    if args.calibration:
        try:
            project = ProjectBoundary(args.project_root)
            calibration = load_video_calibration(args.calibration, project)
            if args.course and args.player:
                course = JsonCourseDataSource(project.resolve_course(args.course)).load_course()
                player = JsonPlayerProfileSource(project.resolve_player(args.player)).load_player()
                analytics_runner = _run_youtube_auto_try_analytics
        except Exception:
            calibration = None
    payload = auto_try(AutoTryConfig(
        url=args.url, out=args.out, calibration=calibration, course=course,
        player=player, project_root=args.project_root,
        segment_start=args.segment_start, segment_duration=args.segment_duration,
        render_video=args.render_video, fallback_human=args.fallback_human,
        yt_dlp=args.yt_dlp,
    ), analytics_runner=analytics_runner)
    print(json.dumps({"status": payload.get("status"), "output": str(args.out)}))
    if payload.get("status") == "blocked":
        raise SystemExit(2)


def _run_youtube_auto_try_analytics(observations, calibration, course, player):
    """Run unchanged analytics only after validated evidence reaches the gate."""
    target_pixel = calibration.from_engine(course.pin_position)
    context = ShotContext(
        event_id="YOUTUBE-AUTO-TRY-0001", player_id=player.player_id,
        tournament_id="YOUTUBE-AUTO-TRY", hole_number=1, shot_number=1,
        lie="unknown", club="7i", distance_to_pin=150.0,
        wind={"speed_mph": 0.0, "direction_deg": 0.0},
        timestamp="youtube-auto-try", target_pixel=target_pixel,
    )
    reconstruction = reconstruct_automatic_shot(observations, calibration, context)
    event = reconstruction.shot_event
    event.provenance = {"source": "youtube-auto-try", "video": dict(reconstruction.metadata)}
    pipeline_result = run_pipeline(
        InMemoryShotSource(event, "video:youtube-auto-try:inline:shot"),
        InMemoryCourseSource(course, "video:youtube-auto-try:inline:course"),
        InMemoryPlayerSource(player, "video:youtube-auto-try:inline:player"),
        Config.default(),
    )
    recommendation = asdict(pipeline_result.recommendation)
    recommendation["provenance"].pop("generated_at", None)
    return {
        "analytics": recommendation,
        "normalized_shot": {"event": asdict(event), "metadata": reconstruction.metadata},
    }


def _print_session_summary(report) -> None:
    s = report["session"]
    summary = report["summary"]
    print("=" * 72)
    print(f"SESSION {s['session_id']} — round {s['round_number']}")
    print(f"  {s['shot_count']} shots across {s['hole_count']} holes")
    print(f"  sum_local_decision_cost: {summary['sum_local_decision_cost']:+.2f} "
          "(local diagnostic, NOT official Strokes Gained)")
    print("-" * 72)
    for hole in report["holes"]:
        print(
            f"  Hole {hole['hole_number']}: {hole['shot_count']} shot(s) "
            f"[{', '.join(hole['shot_ids'])}]  local cost {hole['sum_local_decision_cost']:+.2f}"
        )
    print("=" * 72)


def _print_terminal_summary(result) -> None:
    print("=" * 72)
    print(f"CANDIDATES for {result.shot.player_id} — hole {result.shot.hole_number}, shot {result.shot.shot_number}")
    print(f"Start ({result.shot.start_position.x:.0f}, {result.shot.start_position.y:.0f})  "
          f"pin ({result.course.pin_position.x:.0f}, {result.course.pin_position.y:.0f})  "
          f"lie={result.shot.lie}  distance_to_pin={result.shot.distance_to_pin:.1f}yd")
    print("-" * 72)
    print(f"{'CLUB':<6}{'AIM':<18}{'EXPECTED':>9}  {'TOP HAZARD':<22}")
    for cr in sorted(result.candidate_results, key=lambda r: r.expected_strokes):
        top = sorted(cr.hazard_probabilities.items(), key=lambda kv: kv[1], reverse=True)
        top_hazard = f"{top[0][0]}: {top[0][1]:.0%}" if top else "—"
        print(
            f"{cr.candidate.club:<6}{cr.candidate.label:<18}{cr.expected_strokes:>9.2f}  {top_hazard:>9}"
        )

    best = result.recommendation
    print("-" * 72)
    print("RECOMMENDATION vs ACTUAL")
    print(
        f"  Recommended: {best.recommended_club} aimed at "
        f"({best.recommended_target.x:.0f}, {best.recommended_target.y:.0f}) — "
        f"expected {best.expected_strokes:.2f} strokes"
    )
    print(
        f"  Actual:      {result.shot.club} at target — "
        f"expected {best.actual_expected_strokes:.2f} strokes"
    )
    print(f"  Decision cost: {best.decision_cost:+.2f} strokes")
    print("-" * 72)
    print(best.explanation)
    print("=" * 72)
