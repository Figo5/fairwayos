"""Explicit, safe filesystem boundaries for project-owned resources."""

import os
import re
from pathlib import Path

from .errors import VideoPathError


class ProjectBoundary:
    """Resolve project-bound resources without permitting path escapes."""

    def __init__(self, root):
        root_path = Path(root).expanduser()
        if not root_path.is_dir():
            raise VideoPathError("project boundary must be an existing directory")
        self.root = root_path.resolve()

    def resolve_resource(self, resource):
        if not isinstance(resource, (str, os.PathLike)):
            raise VideoPathError("project resource path must be a path")
        text = os.fspath(resource)
        candidate = Path(text)
        if candidate.is_absolute() or re.match(r"^[A-Za-z]:[\\\\/]", text) or text.startswith(("~/", "~\\")):
            raise VideoPathError("project resources must use relative paths")
        if ".." in candidate.parts:
            raise VideoPathError("project resources must not contain traversal")
        resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise VideoPathError("project resource escapes project boundary") from exc
        if not resolved.is_file():
            raise VideoPathError("project resource must be an existing regular file")
        if not os.access(resolved, os.R_OK):
            raise VideoPathError("project resource is not readable")
        return resolved

    def resolve_json(self, resource):
        resolved = self.resolve_resource(resource)
        if resolved.suffix.lower() != ".json":
            raise VideoPathError("project resource must be a JSON file")
        return resolved

    # Named entry points make the boundary explicit at each ingestion seam.
    def resolve_calibration(self, resource):
        return self.resolve_json(resource)

    def resolve_course(self, resource):
        return self.resolve_json(resource)

    def resolve_player(self, resource):
        return self.resolve_json(resource)

    def resolve_observation(self, resource):
        return self.resolve_json(resource)

    def resolve_annotation(self, resource):
        return self.resolve_json(resource)
