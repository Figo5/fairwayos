"""Errors for the operator-assisted demo workflow.

Every failure here is a *stated* failure. The demo path never renders a missing
artifact as success, never silently drops an out-of-bounds operator input, and
never substitutes a guess for an observation.
"""
from __future__ import annotations


class DemoError(RuntimeError):
    """The demo run could not be completed."""


class DemoConfigError(DemoError):
    """The config or the operator inputs are invalid. Never silently repaired."""


class DemoSourceError(DemoError):
    """The source media cannot be probed or bound."""


class DemoWorkerError(DemoError):
    """The out-of-process CV worker failed. Its stderr says why."""


class DemoRenderError(DemoError):
    """The annotated video or a rendered card was not produced as promised."""


class DemoUnavailable(DemoError):
    """A requested capability is genuinely unavailable (no model, no interpreter)."""
