"""Hình học quỹ đạo/làn dùng chung cho safety, không phụ thuộc CARLA ở import time."""

import math


def straight_path(length_m=40.0):
    return [(0.0, 0.0), (float(length_m), 0.0)]


def project_to_path(x, y, path_points):
    """Chiếu điểm ego-frame lên polyline.

    Trả ``(s, lateral, euclidean_distance)``; ``s`` là chiều dài dọc path,
    lateral dương về bên phải theo hệ tọa độ CARLA (x tiến, y phải).
    """
    return _project_path(x, y, path_points, safety=False)


def project_to_safety_path(x, y, path_points):
    """Safety distance without clipping an obstacle to the finite path ends.

    Select the nearest FINITE segment as before; retain longitudinal overflow
    only at the first/last nonzero segment. Beyond the horizon this assumes the
    terminal tangent continues, not that the map/planner has verified that road.
    Interior projections and the signed lateral corridor are unchanged. The
    third result remains distance to the finite polyline, NOT extended corridor.
    Missing/degenerate geometry uses the existing ego-straight fallback.
    """
    return _project_path(x, y, path_points, safety=True)


def _project_path(x, y, path_points, *, safety):
    if safety and not (math.isfinite(x) and math.isfinite(y)):
        return math.inf, math.inf, math.inf
    path = path_points if path_points and len(path_points) >= 2 else straight_path()
    # Safety distances start at the ego origin even for a truncated waypoint
    # list; a near obstacle before its first waypoint must not disappear.
    if safety and (float(path[0][0]) != 0.0 or float(path[0][1]) != 0.0):
        path = [(0.0, 0.0), *path]
    best = None
    accumulated = 0.0
    first_valid = last_valid = None

    for i in range(len(path) - 1):
        ax, ay = float(path[i][0]), float(path[i][1])
        bx, by = float(path[i + 1][0]), float(path[i + 1][1])
        if safety and not all(math.isfinite(v) for v in (ax, ay, bx, by)):
            return float(x), float(y), abs(float(y))
        dx, dy = bx - ax, by - ay
        seg_len2 = dx * dx + dy * dy
        if seg_len2 <= 1e-9:
            continue
        if first_valid is None:
            first_valid = i
        last_valid = i
        seg_len = math.sqrt(seg_len2)
        raw_u = ((x - ax) * dx + (y - ay) * dy) / seg_len2
        u = max(0.0, min(1.0, raw_u))
        qx, qy = ax + u * dx, ay + u * dy
        ex, ey = x - qx, y - qy
        dist = math.hypot(ex, ey)
        # Right-normal của tangent trong hệ CARLA: (-ty, tx).
        lateral = ex * (-dy / seg_len) + ey * (dx / seg_len)
        candidate = (dist, accumulated + u * seg_len, lateral, i, raw_u, seg_len)
        if best is None or candidate[0] < best[0]:
            best = candidate
        accumulated += seg_len

    if best is None:
        return float(x), float(y), abs(float(y))
    if safety:
        distance, along, lateral, index, raw_u, seg_len = best
        if index == first_valid and raw_u < 0.0:
            # A reversed initial tangent must not hide an ego-forward hazard.
            if x > 0.0:
                return float(x), float(y), distance
            along += raw_u * seg_len
        elif index == last_valid and raw_u > 1.0:
            along += (raw_u - 1.0) * seg_len
        return along, lateral, distance
    return best[1], best[2], best[0]


def _select_next_waypoint(candidates, previous_yaw, turn_intent=None):
    """Chọn nhánh route liên tục; ở junction ưu tiên ý định trái/phải nếu có."""
    def signed_delta(candidate):
        delta = candidate.transform.rotation.yaw - previous_yaw
        return (delta + 180.0) % 360.0 - 180.0

    intent = str(turn_intent or "").lower()
    if intent == "left":
        left = [candidate for candidate in candidates if signed_delta(candidate) < -5.0]
        if left:
            return min(left, key=signed_delta)
    elif intent == "right":
        right = [candidate for candidate in candidates if signed_delta(candidate) > 5.0]
        if right:
            return max(right, key=signed_delta)
    return min(candidates, key=lambda candidate: abs(signed_delta(candidate)))


def _heading_aligned(candidate, transform):
    yaw = transform.rotation.yaw
    return (candidate is not None
            and str(candidate.lane_type).split('.')[-1] == 'Driving'
            and math.isfinite(yaw)
            and abs((candidate.transform.rotation.yaw-yaw+180.) % 360.-180.) < 90.)


def _footprint_overlaps(candidate, ego, transform):
    """Planar body/lane overlap, not permission to enter another lane."""
    yaw = transform.rotation.yaw
    box = getattr(ego, 'bounding_box', None)
    extent = getattr(box, 'extent', None)
    if extent is None or not all(math.isfinite(v) and v > 0. for v in (extent.x, extent.y)):
        return False
    center = getattr(box, 'location', None)
    offset_x, offset_y = getattr(center, 'x', 0.), getattr(center, 'y', 0.)
    box_yaw = getattr(getattr(box, 'rotation', None), 'yaw', 0.)
    if not all(math.isfinite(v) for v in (yaw, offset_x, offset_y, box_yaw)):
        return False
    heading = math.radians(yaw)
    center_x = transform.location.x+math.cos(heading)*offset_x-math.sin(heading)*offset_y
    center_y = transform.location.y+math.sin(heading)*offset_x+math.cos(heading)*offset_y

    width = candidate.lane_width
    if not math.isfinite(width) or width <= 0.:
        return False
    angle = math.radians(candidate.transform.rotation.yaw)
    delta = angle-heading-math.radians(box_yaw)
    dx = center_x-candidate.transform.location.x
    dy = center_y-candidate.transform.location.y
    lateral = abs(-math.sin(angle)*dx+math.cos(angle)*dy)
    footprint = abs(math.sin(delta))*extent.x+abs(math.cos(delta))*extent.y
    return lateral <= width/2.+footprint


def _ego_waypoint(world, ego, transform):
    """Resolve only immediate same-road Driving-lane footprint ambiguities."""
    waypoint = world.get_map().get_waypoint(transform.location)
    if waypoint is None or _heading_aligned(waypoint, transform):
        return waypoint
    candidates = [side for side in (waypoint.get_left_lane(), waypoint.get_right_lane())
                  if (_heading_aligned(side, transform) and side.road_id == waypoint.road_id
                      and side.section_id == waypoint.section_id
                      and _footprint_overlaps(side, ego, transform))]
    return min(candidates, key=lambda w: math.hypot(
        w.transform.location.x-transform.location.x,
        w.transform.location.y-transform.location.y), default=None)


def _path_points(waypoints, pose, origin_offset_x=0.0, step_m=2.0):
    x, y, z, heading = pose
    if not all(math.isfinite(v) for v in (*pose, origin_offset_x)):
        return []
    yaw = math.radians(heading)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    origin_x = x + cos_yaw * origin_offset_x
    origin_y = y + sin_yaw * origin_offset_x

    def to_local(world_location):
        dx = world_location.x - origin_x
        dy = world_location.y - origin_y
        return (cos_yaw * dx + sin_yaw * dy,
                -sin_yaw * dx + cos_yaw * dy)

    points = [(0.0, 0.0)]
    for waypoint in waypoints[1:]:
        point = to_local(waypoint.transform.location)
        if not all(math.isfinite(v) for v in point):
            return []
        if point[0] > 0.0 and point[0] > points[-1][0] - step_m:
            points.append(point)
    return points if len(points) >= 2 else []


def _trace_waypoints(waypoint, lookahead_m, step_m, turn_intent):
    if waypoint is None:
        return []
    waypoints = [waypoint]
    travelled = 0.0
    current = waypoint
    previous_yaw = current.transform.rotation.yaw
    while travelled < lookahead_m:
        candidates = current.next(step_m)
        if not candidates:
            break

        current = _select_next_waypoint(candidates, previous_yaw, turn_intent)
        previous_yaw = current.transform.rotation.yaw
        waypoints.append(current)
        travelled += step_m
    return waypoints


def build_ego_path(world, ego, lookahead_m=40.0, step_m=2.0,
                   origin_offset_x=0.0, turn_intent=None):
    """Legacy stateless path; runtime uses its owned EgoRoute for validity."""
    transform = ego.get_transform()
    location = transform.location
    waypoints = _trace_waypoints(_ego_waypoint(world, ego, transform),
                                 lookahead_m, step_m, turn_intent)
    return _path_points(waypoints, (location.x, location.y, location.z,
                                   transform.rotation.yaw), origin_offset_x, step_m) or straight_path(lookahead_m)


def get_lane_context(world, ego):
    """Trạng thái topology + quyền chuyển làn tại waypoint hiện tại."""
    transform = ego.get_transform()
    waypoint = _ego_waypoint(world, ego, transform)
    nearest = world.get_map().get_waypoint(transform.location) if waypoint is None else waypoint
    return _lane_context(waypoint, bool(getattr(nearest, 'is_junction', False)))


def _lane_context(waypoint, at_junction=False):
    import carla

    if waypoint is None:
        # An unresolved route must not erase junction map priority for lanes.
        return {
            "current_lane_id": None,
            "is_junction": at_junction,
            "left_exists": False, "right_exists": False,
            "left_change_allowed": False, "right_change_allowed": False,
        }

    def valid_side(side):
        return (side is not None
                and side.lane_type == carla.LaneType.Driving
                and side.lane_id * waypoint.lane_id > 0)

    left = waypoint.get_left_lane()
    right = waypoint.get_right_lane()
    lane_change = waypoint.lane_change
    try:
        left_allowed = bool(lane_change & carla.LaneChange.Left)
        right_allowed = bool(lane_change & carla.LaneChange.Right)
    except TypeError:
        text = str(lane_change).lower()
        left_allowed = "left" in text or "both" in text
        right_allowed = "right" in text or "both" in text

    return {
        "current_lane_id": waypoint.lane_id,
        "road_id": waypoint.road_id,
        "section_id": waypoint.section_id,
        "is_junction": at_junction or bool(getattr(waypoint, "is_junction", False)),
        "left_exists": valid_side(left),
        "right_exists": valid_side(right),
        "left_change_allowed": valid_side(left) and left_allowed,
        "right_change_allowed": valid_side(right) and right_allowed,
    }


class EgoRoute:
    """One ego-owned, bounded map reference shared by safety and controller.

    Not a global planner: preserve a previously chosen junction branch only
    while its nearby, heading-aligned lane still overlaps ego. No global cache.
    """
    def __init__(self, world, ego):
        self.world, self.ego = world, ego
        self.episode = getattr(world, 'id', None)
        self.waypoints = []
        self.pose = (0., 0., 0., 0.)
        self.context = {'route_valid': False, 'route_reason': 'route not prepared'}

    def update(self, lookahead_m=45., turn_intent=None):
        transform = self.ego.get_transform()
        location = transform.location
        self.pose = (location.x, location.y, location.z, transform.rotation.yaw)
        episode = getattr(self.world, 'id', None)
        if episode != self.episode:
            self.waypoints = []
        self.episode = episode
        nearest = self.world.get_map().get_waypoint(location)
        at_junction = bool(getattr(nearest, 'is_junction', False))
        waypoint = _ego_waypoint(self.world, self.ego, transform)
        if at_junction and self.waypoints and all(math.isfinite(v) for v in self.pose):
            previous = min(self.waypoints, key=lambda w: math.hypot(
                w.transform.location.x-location.x, w.transform.location.y-location.y))
            p = previous.transform.location
            # ponytail: planar lane footprint; reject another elevation, qualify slopes separately.
            if (_heading_aligned(previous, transform)
                    and math.hypot(p.x-location.x, p.y-location.y) <= 2.+previous.lane_width/2.
                    and abs(p.z-location.z) <= 1.
                    and _footprint_overlaps(previous, self.ego, transform)):
                waypoint = previous
        self.waypoints = _trace_waypoints(waypoint, lookahead_m, 2., turn_intent)
        valid = bool(self.path())
        self.context = {**_lane_context(waypoint, at_junction), 'route_valid': valid,
                        'route_reason': None if valid else 'unresolved map route'}
        if not valid:
            self.waypoints = []
            self.context.update(left_change_allowed=False, right_change_allowed=False)

    def path(self, origin_offset_x=0., lookahead_m=None):
        """Empty means unresolved; never pretend a straight fallback is valid map."""
        waypoints = self.waypoints
        if lookahead_m is not None:
            waypoints = waypoints[:max(0, math.ceil(lookahead_m/2.))+1]
        return _path_points(waypoints, self.pose, origin_offset_x)
