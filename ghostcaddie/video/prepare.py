"""First-run preparation of deterministic local video annotation artifacts."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .annotation_workspace import build_annotation_draft, build_annotation_workspace
from .extraction import extract_frames, generate_contact_sheet, validate_video_source
from .metadata import inspect_video


def prepare_video(source: str, output_directory: str, *, sample_fps: Optional[float] = 2.0,
                  max_frames: Optional[int] = None) -> Dict[str, str]:
    """Validate, inspect, extract, and write a blank offline annotation package."""
    source_path = validate_video_source(source)
    output = Path(output_directory).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise ValueError("output path must be a directory")
    if output == source_path:
        raise ValueError("output path must be a directory separate from the source video")
    try:
        source_path.relative_to(output)
    except ValueError:
        pass
    else:
        raise ValueError("output directory cannot contain the source video")
    output.mkdir(parents=True, exist_ok=True)

    metadata = inspect_video(str(source_path))
    frames = extract_frames(str(source_path), str(output / "frames"),
                            sample_fps=sample_fps, max_frames=max_frames)
    generate_contact_sheet(frames.output_directory, str(output / "contact_sheet.jpg"),
                           columns=min(4, len(frames.frames)))
    video = {
        "width": metadata.width,
        "height": metadata.height,
        "frame_count": len(frames.frames),
        "duration_seconds": metadata.duration_seconds,
    }
    frame_dicts = []
    for frame in frames.frames:
        item = frame.to_dict()
        item["filename"] = "frames/" + item["filename"]
        frame_dicts.append(item)
    html_text = build_annotation_workspace(
        frame_dicts, video=video, contact_sheet_href="contact_sheet.jpg",
        title="Offline video annotation workspace — BLANK DRAFT",
        context="First-run local preparation; add human observations before Submit.",
        blank_draft=True,
    )
    (output / "annotation_workspace.html").write_text(html_text)
    draft = build_annotation_draft(
        frame_dicts, video=video,
        artifact_references=["frames/frame_manifest.json", "contact_sheet.jpg",
                             "annotation_workspace.html"],
    )
    (output / "video-human-annotations.v1.json").write_text(
        json.dumps(draft, sort_keys=True, indent=2) + "\n"
    )
    return {
        "output_directory": str(output),
        "manifest": "frames/frame_manifest.json",
        "contact_sheet": "contact_sheet.jpg",
        "workspace": "annotation_workspace.html",
        "draft": "video-human-annotations.v1.json",
    }
