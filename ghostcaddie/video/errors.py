"""Errors raised by the isolated video Milestone 2 boundary."""


class VideoContractError(ValueError):
    """The video contract contains invalid or unsafe data."""


class VideoMetadataError(VideoContractError):
    """ffprobe metadata is missing, malformed, or invalid."""


class VideoProbeError(VideoContractError):
    """ffprobe could not inspect the source."""


class VideoExtractionError(VideoContractError):
    """Frames or a contact sheet could not be generated."""


class VideoPathError(VideoExtractionError):
    """A source or project-bound resource path is invalid or unsafe."""


class VideoCalibrationError(VideoContractError):
    """A video calibration contract is invalid."""


class VideoReconstructionError(VideoContractError):
    """Shot context or reconstruction inputs are invalid."""


class VideoReconstructionUnavailable(VideoReconstructionError):
    """Validated video data does not contain the required shot evidence."""
