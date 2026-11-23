"""Short radar position histories for lateral conflicts outside the current path."""

import math
from collections import deque

if __package__:
    from .road_geometry import project_to_safety_path
else:
    from road_geometry import project_to_safety_path


def radar_reference_pose(transform, reference_x_m, offset_y_m=0.0):
    """World pose of the radar processor's tracking reference (metres, radians)."""
    yaw = math.radians(transform.rotation.yaw)
    return (
        transform.location.x + reference_x_m * math.cos(yaw) - offset_y_m * math.sin(yaw),
        transform.location.y + reference_x_m * math.sin(yaw) + offset_y_m * math.cos(yaw),
        yaw,
    )


def crossing_conflict(x, y, ux, uy, path, speed, lane_half, horizon, velocity_covariance=None):
    """World-relative object velocity in ego axes, constant speed along the ego path."""
    along, lat, _ = project_to_safety_path(x, y, path)
    ahead, side, _ = project_to_safety_path(x + 0.25 * ux, y + 0.25 * uy, path)
    inward = -math.copysign(1.0, lat) * (side - lat) / 0.25
    closing = speed - (ahead - along) / 0.25
    if abs(lat) <= lane_half or inward < 0.65 or closing <= 0.3 or along <= 0:
        return math.inf, math.inf
    if velocity_covariance is not None:
        # An immature track can have a large inward velocity estimate with
        # either sign still plausible. Project its covariance onto the path normal.
        nx = (project_to_safety_path(x + 0.1, y, path)[1] - lat) / 0.1
        ny = (project_to_safety_path(x, y + 0.1, path)[1] - lat) / 0.1
        cov = velocity_covariance
        variance = nx * nx * cov[0, 0] + 2 * nx * ny * cov[0, 1] + ny * ny * cov[1, 1]
        if not math.isfinite(variance) or inward <= 1.645 * math.sqrt(max(0.0, variance)):
            return math.inf, math.inf
    arrival = along / closing
    if arrival <= horizon:
        final_s, final_lat, _ = project_to_safety_path(x + arrival * ux, y + arrival * uy, path)
        if abs(final_lat) <= lane_half and abs(final_s - speed * arrival) <= 1.0:
            return along, arrival
    return math.inf, math.inf


class RadarCrossingPredictor:
    def __init__(self):
        self.time = 0.0
        self.histories = []

    def threat(self, targets, dt, pose, speed, path, lane_half, horizon):
        # Work in an inertial frame so a turning ego cannot make a static
        # roadside reflector look like it is walking into the lane.
        if pose is None or not 0 < dt <= 0.15 or not all(map(math.isfinite, pose)):
            self.histories.clear()
            return math.inf, math.inf
        self.time += dt
        ox, oy, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        points = []
        for target in targets or ():
            x, y = target.get("distance_m", math.inf), target.get("lateral_m", math.inf)
            if 0 < x < 40 and abs(y) < 8:
                points.append((float(ox + x * c - y * s), float(oy + x * s + y * c)))

        nearest = ttc = math.inf
        for history in self._associate(points):
            motion = self._fit_motion(history)
            if motion is None:
                continue
            wx, wy, vx, vy = motion
            dx, dy = wx - ox, wy - oy
            x, y = dx * c + dy * s, -dx * s + dy * c
            ux, uy = vx * c + vy * s, -vx * s + vy * c
            distance, arrival = crossing_conflict(x, y, ux, uy, path, speed, lane_half, horizon)
            nearest, ttc = min(nearest, distance), min(ttc, arrival)
        return nearest, ttc

    def _associate(self, points):
        self.histories = [h for h in self.histories if self.time - h[-1][0] <= 0.15]
        # One return per history per tick. Gate by the distance a pedestrian
        # can move between scans plus the radar's surface-return scatter.
        pairs = []
        for i, history in enumerate(self.histories):
            t, x, y = history[-1]
            for j, (px, py) in enumerate(points):
                distance = math.hypot(px - x, py - y)
                if distance <= 0.45 + 3.0 * (self.time - t):
                    pairs.append((distance, i, j))
        used, matched = set(), set()
        for _, i, j in sorted(pairs):
            if i not in used and j not in matched:
                self.histories[i].append((self.time, *points[j]))
                used.add(i)
                matched.add(j)
        for j, point in enumerate(points):
            if j not in matched:
                self.histories.append(deque([(self.time, *point)]))

        return (self.histories[i] for i in used)

    def _fit_motion(self, history):
        """Least-squares position and velocity from a consistent sequence of scans."""
        while self.time - history[0][0] > 0.6:
            history.popleft()
        if len(history) < 7 or self.time - history[0][0] < 0.4:
            return None
        mt, mx, my = (sum(values) / len(history) for values in zip(*history))
        denom = sum((t - mt) ** 2 for t, _, _ in history)
        vx = sum((t - mt) * (x - mx) for t, x, _ in history) / denom
        vy = sum((t - mt) * (y - my) for t, _, y in history) / denom
        error = sum((x - mx - vx * (t - mt)) ** 2 + (y - my - vy * (t - mt)) ** 2 for t, x, y in history)
        if error / len(history) > 0.22**2 or not 0.65 <= math.hypot(vx, vy) <= 3.5:
            return None
        return mx + vx * (self.time - mt), my + vy * (self.time - mt), vx, vy
