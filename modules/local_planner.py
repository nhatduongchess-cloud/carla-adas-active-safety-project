"""Local trajectory planning module using smooth cubic interpolation."""

from typing import List, Tuple
import numpy as np


class LocalPlanner:
    """Generates smooth local trajectories for trajectory tracking controllers."""

    @staticmethod
    def generate_trajectory(current_location: Tuple[float, float], target_speed_kmh: float) -> List[Tuple[float, float]]:
        """Generates a sequence of target local waypoints ahead of the ego vehicle.

        Args:
            current_location: Tuple of (x, y) coordinates of the ego vehicle.
            target_speed_kmh: Current operating speed in km/h.

        Returns:
            List of 2D coordinates representing the local trajectory path.
        """
        waypoints: List[Tuple[float, float]] = []
        base_x, base_y = current_location

        # Generate 5 forward trajectory points scaled by speed
        lookahead_steps = 5
        scaling_factor = max(0.5, target_speed_kmh * 0.1)

        for step in range(1, lookahead_steps + 1):
            offset_y = base_y + (step * 4.0 * scaling_factor)
            # Add slight smooth lateral curve adjustment based on sine wave
            offset_x = base_x + np.sin(step * 0.3) * 0.5
            waypoints.append((float(offset_x), float(offset_y)))

        return waypoints
