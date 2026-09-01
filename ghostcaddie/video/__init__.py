"""Isolated video diagnostics: contracts, metadata, and frame artifacts."""

from .contracts import SCHEMA_VERSION, VideoDiagnostics, VideoMetadata
from .human_contracts import (HumanAnnotationDocument, SCHEMA_VERSION as HUMAN_ANNOTATIONS_SCHEMA_VERSION,
                              deserialize_human_annotations, serialize_human_annotations,
                              validate_human_annotations)
from .observations import (OBSERVATIONS_SCHEMA_VERSION, CANONICAL_PHASES, PixelBBox, PixelObservation, VideoObservations)
from .perception import (FixturePerception, DeterministicFixturePerception,
                         OllamaPerceptionAdapter, ModelPerceptionAdapter,
                         PerceptionResult, load_fixture_observations)
from .errors import (VideoCalibrationError, VideoContractError, VideoExtractionError,
                     VideoMetadataError, VideoPathError, VideoProbeError,
                     VideoReconstructionError, VideoReconstructionUnavailable)
from .calibration import VideoCalibration, load_video_calibration
from .paths import ProjectBoundary
from .extraction import (ContactSheetResult, FrameExtractionResult, FrameRecord,
                         extract_frames, generate_contact_sheet, validate_video_source)
from .metadata import inspect_video, parse_ffprobe_metadata
from .reconstruction import ReconstructionResult, ShotContext, reconstruct_shot
from .human_import import import_human_annotations, load_human_annotations
from .annotations import annotate_frame, annotate_video_frame, build_annotation_filter
from .annotation_workspace import build_annotation_workspace, generate_annotation_workspace, build_annotation_draft
from .prepare import prepare_video
from .diagnostics import (build_diagnostics, build_video_diagnostics,
                          serialize_diagnostics, serialize_video_diagnostics)
from .orchestration import (VideoPipelineResult, run_fixture_video_pipeline,
                            run_video_fixture_pipeline, run_video_pipeline)
from .youtube import (DownloadError, DownloadLimits, DownloaderUnavailable,
                      YouTubeDownloadResult, YouTubeSource, YtDlpDownloader,
                      configured_yt_dlp, parse_youtube_url)
from .automatic_perception import (AUTOMATIC_PERCEPTION_SCHEMA_VERSION, AnchorValidation,
                                   BodyAnchor, ConfidenceMetrics, ContinuityMetrics,
                                   Detection, Detector, EvaluationMetrics, GateDecision,
                                   ImpactCandidateInterval, OpticalFlowPolicy, Provenance,
                                   SwingPhase, SwingPhaseObservation, SwingPhaseStateMachine,
                                   Thresholds, Track, Tracker, continuity_metrics,
                                   evaluate_sequence_gates, precision_recall,
                                   reconstruct_automatic_shot, select_single_golfer_track,
                                   validate_body_anchor)
from .automatic_render import (build_automatic_report, build_evaluation_report,
                               serialize_automatic_report, serialize_evaluation_report,
                               render_automatic_frame)
from .research_split import (RESEARCH_SPLIT_SCHEMA_VERSION, serialize_split_manifest,
                             validate_split_manifest)

__all__ = [
    "SCHEMA_VERSION", "VideoDiagnostics", "VideoMetadata", "HUMAN_ANNOTATIONS_SCHEMA_VERSION", "HumanAnnotationDocument",
    "validate_human_annotations", "serialize_human_annotations", "deserialize_human_annotations",
    "import_human_annotations", "load_human_annotations",
    "OBSERVATIONS_SCHEMA_VERSION", "CANONICAL_PHASES",
    "PixelBBox", "PixelObservation", "VideoObservations", "FixturePerception",
    "DeterministicFixturePerception", "OllamaPerceptionAdapter", "ModelPerceptionAdapter",
    "PerceptionResult", "load_fixture_observations", "VideoContractError",
    "VideoMetadataError", "VideoProbeError", "VideoExtractionError", "VideoPathError",
    "VideoCalibrationError", "VideoCalibration", "load_video_calibration", "ProjectBoundary", "inspect_video",
    "parse_ffprobe_metadata", "FrameRecord", "FrameExtractionResult", "ContactSheetResult",
    "extract_frames", "generate_contact_sheet", "validate_video_source",
    "ShotContext", "ReconstructionResult", "reconstruct_shot", "VideoReconstructionError",
    "VideoReconstructionUnavailable", "VideoPipelineResult", "run_video_pipeline",
    "run_video_fixture_pipeline", "run_fixture_video_pipeline", "build_annotation_filter",
    "annotate_frame", "annotate_video_frame", "build_annotation_workspace", "generate_annotation_workspace", "build_video_diagnostics", "build_diagnostics",
    "serialize_video_diagnostics", "serialize_diagnostics",
    "DownloadError", "DownloadLimits", "DownloaderUnavailable", "YouTubeDownloadResult",
    "YouTubeSource", "YtDlpDownloader", "configured_yt_dlp", "parse_youtube_url",
    "AUTOMATIC_PERCEPTION_SCHEMA_VERSION", "Provenance", "Detection", "Detector",
    "Track", "Tracker", "BodyAnchor", "AnchorValidation", "ContinuityMetrics",
    "ConfidenceMetrics", "EvaluationMetrics", "OpticalFlowPolicy", "Thresholds",
    "GateDecision", "SwingPhase", "SwingPhaseObservation", "SwingPhaseStateMachine",
    "ImpactCandidateInterval", "continuity_metrics", "select_single_golfer_track",
    "validate_body_anchor", "evaluate_sequence_gates", "precision_recall",
    "reconstruct_automatic_shot", "build_automatic_report", "build_evaluation_report",
    "serialize_automatic_report", "serialize_evaluation_report", "render_automatic_frame",
    "RESEARCH_SPLIT_SCHEMA_VERSION", "validate_split_manifest", "serialize_split_manifest",
]
