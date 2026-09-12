"""Local trajectory generation for lane following and smooth lane changes."""

from typing import List, Tuple
import math


class LocalPlanner:
    """Biến centerline local-frame thành quỹ đạo điều khiển liên tục."""

    @staticmethod
    def _smoothstep5(t: float) -> float:
        t = max(0.0, min(1.0, t))
        return 10.0 * t ** 3 - 15.0 * t ** 4 + 6.0 * t ** 5

    @classmethod
    def lane_change_trajectory(
        cls,
        base_path: List[Tuple[float, float]],
        lateral_offset_m: float,
        transition_m: float = 22.0,
    ) -> List[Tuple[float, float]]:
        """Dịch path bằng quintic smoothstep: zero velocity/acceleration ở hai đầu."""
        if not base_path:
            return []
        output = []
        travelled = 0.0
        previous = base_path[0]
        for i, point in enumerate(base_path):
            x, y = float(point[0]), float(point[1])
            if i:
                travelled += math.hypot(x - previous[0], y - previous[1])
            blend = cls._smoothstep5(travelled / max(transition_m, 1.0))
            output.append((x, y + lateral_offset_m * blend))
            previous = (x, y)
        return output

    @staticmethod
    def generate_trajectory(current_location: Tuple[float, float], target_speed_kmh: float) -> List[Tuple[float, float]]:
        """Generates a sequence of target local waypoints ahead of the ego vehicle.

        Args:
            current_location: Tuple of (x, y) coordinates of the ego vehicle.
            target_speed_kmh: Current operating speed in km/h.

        Returns:
            List of 2D coordinates representing the local trajectory path.
        """
        base_x, base_y = current_location
        spacing = max(2.0, min(6.0, target_speed_kmh / 10.0))
        return [(float(base_x + step * spacing), float(base_y))
                for step in range(1, 11)]
