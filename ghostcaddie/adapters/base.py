"""Structural typing (Protocol) seams for future data sources.

A future ShotLink/TrackMan/TOURCAST adapter only needs to implement these
three protocols — the analytics engine core never knows or cares where the
data came from.
"""

from typing import Protocol, runtime_checkable

from ..models import CourseModel, PlayerProfile, ShotEvent


@runtime_checkable
class ShotDataSource(Protocol):
    def load_shot(self) -> ShotEvent: ...


@runtime_checkable
class CourseDataSource(Protocol):
    def load_course(self) -> CourseModel: ...


@runtime_checkable
class PlayerProfileSource(Protocol):
    def load_player(self) -> PlayerProfile: ...
