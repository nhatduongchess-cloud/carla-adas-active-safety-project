"""Ước lượng delta pose ego giữa hai CARLA frame, không import CARLA."""

import math


class EgoMotionEstimator:
    def __init__(self):
        self._previous = None

    def reset(self):
        self._previous = None

    def update(self, transform):
        location = transform.location
        yaw = math.radians(transform.rotation.yaw)
        current = (float(location.x), float(location.y), yaw)
        if self._previous is None:
            self._previous = current
            return {"dx": 0.0, "dy": 0.0, "dyaw": 0.0, "valid": False}

        px, py, pyaw = self._previous
        wx, wy = current[0] - px, current[1] - py
        dx = math.cos(pyaw) * wx + math.sin(pyaw) * wy
        dy = -math.sin(pyaw) * wx + math.cos(pyaw) * wy
        dyaw = (yaw - pyaw + math.pi) % (2.0 * math.pi) - math.pi
        self._previous = current
        return {"dx": dx, "dy": dy, "dyaw": dyaw, "valid": True}
