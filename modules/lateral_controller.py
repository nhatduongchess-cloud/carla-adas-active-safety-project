"""Lateral vehicle controller using the Stanley control algorithm."""

import numpy as np


class LateralController:
    """Implements Stanley steering control for robust path tracking.

    Trade-off Analysis (Stanley vs. Pure Pursuit):
        - Stanley Controller: Directly accounts for both heading error and cross-track error.
          Highly responsive and precise at medium-to-high speeds. Requires a softening constant
          to prevent division-by-zero oscillations at near-zero speeds.
        - Pure Pursuit: Relies entirely on a geometric lookahead point. Simple to tune, but suffers
          from trajectory cutting on sharp curves and instability if lookahead distance is poorly selected.
    """

    def __init__(self, k_gain: float = 2.5, softening_constant: float = 1.0) -> None:
        """Initializes controller gains.

        Args:
            k_gain: Gain controlling cross-track error correction intensity.
            softening_constant: Prevents mathematical singularities when vehicle velocity is near zero.
        """
        self.k_gain: float = k_gain
        self.softening_constant: float = softening_constant

    def compute_steering_angle(self, vehicle_heading: float, cross_track_error: float, forward_speed: float) -> float:
        """Computes optimal steering angle command in radians.

        Args:
            vehicle_heading: Current orientation angle of the vehicle in radians.
            cross_track_error: Lateral distance from the front axle to the path centerline.
            forward_speed: Current longitudinal velocity in m/s.

        Returns:
            Steering control command bounded between -1.0 and 1.0.
        """
        # Heading error component (assume path heading is zero for local frame alignment)
        heading_error = 0.0 - vehicle_heading

        # Cross-track error correction term using Stanley formula
        denominator = max(self.softening_constant, forward_speed)
        cross_track_steering = np.arctan2(
            self.k_gain * cross_track_error, denominator)

        total_steering = heading_error + cross_track_steering

        # Normalize steering output to CARLA control bounds [-1.0, 1.0]
        normalized_steering = float(
            np.clip(total_steering / np.radians(30.0), -1.0, 1.0))
        return normalized_steering

    def compute_path_steering(self, path_points, forward_speed: float,
                              wheelbase_m: float = 2.85,
                              max_steer_deg: float = 35.0) -> float:
        """Pure-pursuit trên path đã đổi sang ego local frame (x tiến, y phải)."""
        if not path_points:
            return 0.0
        lookahead = float(np.clip(4.0 + 0.55 * max(0.0, forward_speed), 5.0, 15.0))
        target = path_points[-1]
        for point in path_points:
            if point[0] > 0.0 and np.hypot(point[0], point[1]) >= lookahead:
                target = point
                break
        tx, ty = float(target[0]), float(target[1])
        ld = max(1.0, float(np.hypot(tx, ty)))
        alpha = float(np.arctan2(ty, tx))
        steer_rad = float(np.arctan2(2.0 * wheelbase_m * np.sin(alpha), ld))
        return float(np.clip(steer_rad / np.radians(max_steer_deg), -1.0, 1.0))
