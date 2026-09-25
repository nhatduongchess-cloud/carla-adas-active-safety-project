"""Clearance between two actors, surface to surface.

Pure Python, no CARLA import: it reads only `get_location`, `get_transform`
and `bounding_box`, so it can be tested with plain stand-ins.

WHY THIS EXISTS
---------------
The scenario acceptance criterion is "minimum clearance >= 0.25 m". It was fed
the distance between actor *centres*. Two cars touching bumper to bumper have
centres roughly 4.8 m apart, so the criterion could not fail for a vehicle
target - it was satisfied by construction, in the same way the reaction-delay
criterion once was. The 2026-09-25 evidence review reproduced it: centres 3.0 m
apart, a 2 m half-length, reported "clearance" 3.0 m.

Centre distance is still the right quantity for *triggering* a scenario, and
RunningScenario keeps using it for that; only the acceptance measurement moves.

THREE BASES, FROM EXACT TO CONSERVATIVE
---------------------------------------
``oriented_boxes``   both actors expose a transform and a bounding box: the
                     true gap between two oriented rectangles in the ground
                     plane, 0 if they overlap.
``radius``           a box without orientation: each box is replaced by a
                     circle of its largest half-extent. This under-states
                     clearance for a side approach, which errs toward failing
                     the criterion rather than passing it.
``center``           neither actor has a box: centre distance, flagged so a
                     report cannot mistake it for surface clearance.

The basis actually used is returned alongside the number.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
BASES = ("oriented_boxes", "radius", "center")


def _shape(actor: Any) -> dict:
    """Read what geometry an actor offers, tolerating anything missing."""
    loc = actor.get_location()
    center = (float(loc.x), float(loc.y))
    box = getattr(actor, "bounding_box", None)
    extent = getattr(box, "extent", None) if box is not None else None
    half = None
    if extent is not None:
        half = (abs(float(extent.x)), abs(float(extent.y)))

    corners = None
    get_transform = getattr(actor, "get_transform", None)
    if half is not None and callable(get_transform):
        try:
            tf = get_transform()
            yaw = math.radians(float(tf.rotation.yaw))
            offset = getattr(box, "location", None)
            ox = float(getattr(offset, "x", 0.0)) if offset is not None else 0.0
            oy = float(getattr(offset, "y", 0.0)) if offset is not None else 0.0
            c, s = math.cos(yaw), math.sin(yaw)
            base = (float(tf.location.x), float(tf.location.y))
            cx = base[0] + c * ox - s * oy
            cy = base[1] + s * ox + c * oy
            corners = [(cx + c * dx - s * dy, cy + s * dx + c * dy)
                       for dx, dy in ((half[0], half[1]), (half[0], -half[1]),
                                      (-half[0], -half[1]), (-half[0], half[1]))]
        except Exception:
            corners = None
    return {"center": center, "half": half, "corners": corners}


def _axes(poly: Sequence[Point]) -> List[Point]:
    axes = []
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        ex, ey = x2 - x1, y2 - y1
        n = math.hypot(ex, ey)
        if n > 1e-12:
            axes.append((-ey / n, ex / n))
    return axes


def _overlap(a: Sequence[Point], b: Sequence[Point]) -> bool:
    for ax, ay in _axes(a) + _axes(b):
        pa = [x * ax + y * ay for x, y in a]
        pb = [x * ax + y * ay for x, y in b]
        if max(pa) < min(pb) or max(pb) < min(pa):
            return False
    return True


def _point_segment(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 < 1e-12 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def polygon_gap(a: Sequence[Point], b: Sequence[Point]) -> float:
    """Distance between two convex polygons; 0 if they touch or overlap."""
    if _overlap(a, b):
        return 0.0
    best = math.inf
    for poly, other in ((a, b), (b, a)):
        for p in poly:
            for i in range(len(other)):
                best = min(best, _point_segment(p, other[i], other[(i + 1) % len(other)]))
    return best


def clearance(ego: Any, actor: Any) -> Tuple[float, str]:
    """Surface-to-surface clearance in metres, and the basis it was computed on."""
    a, b = _shape(ego), _shape(actor)
    if a["corners"] is not None and b["corners"] is not None:
        return polygon_gap(a["corners"], b["corners"]), "oriented_boxes"
    centre = math.hypot(a["center"][0] - b["center"][0], a["center"][1] - b["center"][1])
    ra = max(a["half"]) if a["half"] is not None else None
    rb = max(b["half"]) if b["half"] is not None else None
    if ra is None and rb is None:
        return centre, "center"
    return max(0.0, centre - (ra or 0.0) - (rb or 0.0)), "radius"


def weakest_basis(bases: Sequence[str]) -> Optional[str]:
    """The least exact basis among several measurements, for reporting."""
    present = [b for b in bases if b in BASES]
    return max(present, key=BASES.index) if present else None
