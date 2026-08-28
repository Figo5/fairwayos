"""Video perception adapters.

Fixture perception is deterministic and remains the default.  The Ollama adapter
is deliberately isolated here and must be explicitly enabled by its caller.
"""

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .errors import VideoContractError, VideoPathError
from .observations import VideoObservations
from .paths import ProjectBoundary


def load_fixture_observations(resource, project_boundary):
    """Load and validate project-owned observation JSON without model inference."""
    if not isinstance(project_boundary, ProjectBoundary):
        raise VideoPathError("a ProjectBoundary is required for observation resources")
    path = project_boundary.resolve_observation(resource)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise VideoContractError("unable to load observation JSON: " + str(exc)) from exc
    return VideoObservations.from_dict(payload)


class FixturePerception:
    """A deterministic perception implementation for checked-in test fixtures."""

    def __init__(self, project_boundary, resource):
        if not isinstance(project_boundary, ProjectBoundary):
            raise VideoPathError("a ProjectBoundary is required for fixture perception")
        self._boundary = project_boundary
        self._resource = resource

    def perceive(self, *args, **kwargs):
        return load_fixture_observations(self._resource, self._boundary)


@dataclass(frozen=True)
class PerceptionResult:
    """Validated observations or an explicit, non-fatal unavailable result."""

    observations: Optional[VideoObservations]
    status: str
    confidence: Dict[str, float]
    warnings: List[str]
    provenance: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "observations": self.observations.to_dict() if self.observations else None,
            "confidence": dict(self.confidence),
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
        }


class OllamaPerceptionAdapter:
    """Opt-in Ollama vision adapter using bounded, real image inputs.

    The default model is local ``gemma4:e2b``.  No CLI or fixture path invokes
    this adapter unless the caller constructs it with ``enabled=True``.
    """

    provider = "ollama"
    DEFAULT_ENDPOINT = "http://127.0.0.1:11434/api/chat"
    SYSTEM_INSTRUCTION = (
        "Return ONLY a JSON object, never markdown, reasoning, or detector output. "
        "It MUST exactly match video-observations.v1: top-level keys are schema_version, image, observations; "
        "schema_version is video-observations.v1; image has width and height; observations is a non-empty list. "
        "Each observation MUST contain frame_index, timestamp_seconds, golfer, club, clubhead, ball, phase, "
        "contact, intended_direction, landing, warnings. Use null for unknown quantities, valid confidence values, "
        "and only the defined phase and warning codes. Canonical phases are unknown, address, backswing, top, "
        "downswing, contact, follow_through, ball_flight, landing, rolling, finish."
    )

    def __init__(self, *, model: str = "gemma4:e2b", endpoint: str = DEFAULT_ENDPOINT,
                 enabled: bool = False, max_frames: int = 8,
                 max_image_bytes: int = 2 * 1024 * 1024,
                 max_payload_bytes: int = 8 * 1024 * 1024,
                 timeout_seconds: float = 30.0,
                 transport: Optional[Callable[[Dict[str, Any], float], Dict[str, Any]]] = None):
        if not isinstance(model, str) or not model.strip() or ":cloud" in model.lower():
            raise ValueError("model must be a non-empty local model name")
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or not 1 <= max_frames <= 32:
            raise ValueError("max_frames must be between 1 and 32")
        if max_image_bytes <= 0 or max_payload_bytes <= 0 or timeout_seconds <= 0:
            raise ValueError("image, payload, and timeout bounds must be positive")
        self.model = model
        self.endpoint = endpoint
        self.enabled = enabled
        self.max_frames = max_frames
        self.max_image_bytes = max_image_bytes
        self.max_payload_bytes = max_payload_bytes
        self.timeout_seconds = timeout_seconds
        self._transport = transport or self._post_json

    @property
    def provenance(self) -> Dict[str, str]:
        return {"model": self.model, "provider": self.provider, "mode": "model"}

    def _unavailable(self, warning: str) -> PerceptionResult:
        return PerceptionResult(None, "unavailable", {}, [warning], self.provenance)

    def perceive(self, frame_paths: Iterable[os.PathLike], **_: Any) -> PerceptionResult:
        if not self.enabled:
            return self._unavailable("model perception is opt-in; adapter is disabled")
        paths = list(frame_paths)
        if not paths:
            return self._unavailable("model perception unavailable: no sampled frames")
        if len(paths) > self.max_frames:
            raise VideoContractError("model perception frame limit exceeded")
        images = []
        total = 0
        for raw_path in paths:
            path = Path(raw_path).expanduser()
            if not path.is_file() or not os.access(path, os.R_OK):
                return self._unavailable("model perception unavailable: sampled frame is unreadable")
            size = path.stat().st_size
            if size <= 0 or size > self.max_image_bytes:
                raise VideoContractError("sampled frame exceeds image size bound")
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            total += len(encoded)
            if total > self.max_payload_bytes:
                raise VideoContractError("encoded image payload exceeds bound")
            images.append(encoded)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": self.SYSTEM_INSTRUCTION, "images": images}],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        }
        try:
            response = self._transport(payload, self.timeout_seconds)
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            return self._unavailable("model perception unavailable: Ollama request failed (" + type(exc).__name__ + ")")
        content = self._response_content(response)
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError) as exc:
            raise VideoContractError("model returned non-JSON observations") from exc
        observations = VideoObservations.from_dict(parsed)
        confidence = {}
        quantities = [item.golfer.confidence for item in observations.items]
        confidence["overall"] = min(quantities)
        warnings = sorted({warning for item in observations.items for warning in item.warnings})
        return PerceptionResult(observations, "complete", confidence, warnings, self.provenance)

    @staticmethod
    def _response_content(response: Dict[str, Any]) -> str:
        if not isinstance(response, dict) or set(response) - {
            "created_at", "done", "done_reason", "eval_count", "eval_duration", "load_duration",
            "message", "model", "prompt_eval_count", "prompt_eval_duration", "total_duration"
        }:
            raise VideoContractError("model response has unexpected fields")
        message = response.get("message")
        if not isinstance(message, dict) or set(message) - {"content", "role", "thinking"} or "content" not in message or not isinstance(message["content"], str):
            raise VideoContractError("model response has no strict content field")
        content = message["content"].strip()
        if not content or not content.startswith("{") or not content.endswith("}"):
            raise VideoContractError("model returned instruction-like or non-object output")
        return content

    def _post_json(self, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        request = urllib.request.Request(
            self.endpoint, data=json.dumps(payload, separators=(",", ":")).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())


# Clear aliases for integrations that use generic adapter naming.
ModelPerceptionAdapter = OllamaPerceptionAdapter
DeterministicFixturePerception = FixturePerception
load_observation_fixture = load_fixture_observations
