"""Landing-region classification: where did the shot end up on the course?"""

from typing import List

from .geometry import Point2D, point_in_polygon
from .models import CourseModel, RegionType


def _first_containing(point: Point2D, polygons: List[List[Point2D]]) -> bool:
    return any(point_in_polygon(point, poly) for poly in polygons)


def classify_landing(point: Point2D, course: CourseModel) -> RegionType:
    """Classify a landing point against the course's regions.

    Priority order is deliberate: OB > water > bunker > green > fairway, so
    overlapping regions resolve deterministically (first check to hit wins).
    ROUGH has no polygon list of its own — it is the implicit "everywhere
    else" region, which matches how rough is modeled on real courses (all the
    space not covered by a marked feature).
    """
    if _first_containing(point, course.out_of_bounds):
        return RegionType.OUT_OF_BOUNDS
    if _first_containing(point, course.water_hazards):
        return RegionType.WATER
    if _first_containing(point, course.bunkers):
        return RegionType.BUNKER
    if _first_containing(point, course.green):
        return RegionType.GREEN
    if _first_containing(point, course.fairway):
        return RegionType.FAIRWAY
    return RegionType.ROUGH
